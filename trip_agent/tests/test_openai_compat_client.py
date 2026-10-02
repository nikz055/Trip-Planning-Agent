"""The OpenAI-compatible (OpenRouter) client, against a fake transport.

No request leaves the machine. The client has not been run against a live endpoint.
"""

import json

import httpx
import pytest

from trip_agent.config import Settings
from trip_agent.llm.base import LLMError, LLMToolCall, LLMToolResult, Message, ToolSpec
from trip_agent.llm.factory import make_llm
from trip_agent.llm.openai_compat_client import OpenAICompatClient, from_response, to_tools, to_wire
from trip_agent.llm.standin import StandInPlanner

TOOL = ToolSpec(name="search_places", description="Search places", input_schema={"type": "object", "properties": {}})
KEY = "test-key-not-real"


def reply(**message) -> dict:
    return {
        "choices": [{"message": message, "finish_reason": message.pop("finish", "tool_calls")}],
        "usage": {"prompt_tokens": 1200, "completion_tokens": 80, "cost": 0.0031},
    }


def client_with(handler, retry_delays=()) -> OpenAICompatClient:
    client = OpenAICompatClient("some/model", KEY, "https://llm.test/api/v1", transport=httpx.MockTransport(handler))
    client.retry_delays = retry_delays
    return client


async def test_transient_errors_are_retried_then_succeed():
    replies = [
        httpx.Response(429, json={"error": {"message": "slow down"}}),
        httpx.Response(200, json={"error": {"message": "Upstream error: Service temporarily overloaded"}}),
        httpx.Response(200, json=reply(content="ok", finish="stop")),
    ]
    client = client_with(lambda request: replies.pop(0), retry_delays=(0, 0, 0))
    turn = await client.complete(system="s", messages=[Message(role="user", text="hi")], tools=[TOOL])
    assert turn.text == "ok" and replies == []


async def test_a_bad_key_is_not_retried():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(401, json={"error": {"message": "bad key"}})

    with pytest.raises(LLMError, match="rejected the API key"):
        await client_with(handler, retry_delays=(0, 0)).complete(system="s", messages=[Message(role="user", text="hi")], tools=[TOOL])
    assert len(calls) == 1


def test_to_wire_orders_tool_results_before_text():
    messages = [
        Message(role="user", text="<context>{}</context>\nPlan this trip."),
        Message(role="assistant", text="Searching.", tool_calls=[LLMToolCall(id="c1", name="search_places", args={"city": "Vienna"})]),
        Message(role="user", text="<context>{}</context>", tool_results=[LLMToolResult(tool_call_id="c1", content="[]")]),
    ]
    wire = to_wire("SYSTEM", messages)
    assert [m["role"] for m in wire] == ["system", "user", "assistant", "tool", "user"]
    assert wire[0]["content"] == "SYSTEM"
    assert wire[2]["tool_calls"] == [
        {"id": "c1", "type": "function", "function": {"name": "search_places", "arguments": '{"city": "Vienna"}'}}
    ]
    assert wire[3] == {"role": "tool", "tool_call_id": "c1", "content": "[]"}


def test_to_tools_uses_function_format():
    assert to_tools([TOOL]) == [
        {"type": "function", "function": {"name": "search_places", "description": "Search places", "parameters": TOOL.input_schema}}
    ]


def test_from_response_parses_calls_usage_and_cost():
    body = reply(content=None, tool_calls=[{"id": "c9", "function": {"name": "search_places", "arguments": '{"city": "Vienna"}'}}])
    turn = from_response(body)
    assert turn.tool_calls == [LLMToolCall(id="c9", name="search_places", args={"city": "Vienna"})]
    assert (turn.input_tokens, turn.output_tokens, turn.cost_usd) == (1200, 80, 0.0031)
    assert turn.text is None and turn.stop_reason == "tool_calls"


def test_unparseable_arguments_are_kept_for_the_schema_check_to_reject():
    body = reply(tool_calls=[{"id": "c1", "function": {"name": "search_places", "arguments": '{"city": "Vien'}}])
    [call] = from_response(body).tool_calls
    assert list(call.args) == ["unparseable_arguments"]  # no silent guess at what was meant


def test_truncated_output_is_flagged():
    assert from_response(reply(content="cut off", finish="length")).stop_reason == "max_tokens"


def test_error_body_raises():
    with pytest.raises(LLMError, match="No endpoints found"):
        from_response({"error": {"message": "No endpoints found that support tool use"}})


async def test_complete_sends_the_request_and_keeps_the_key_in_the_header_only():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"], seen["auth"] = str(request.url), request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=reply(tool_calls=[{"id": "c1", "function": {"name": "search_places", "arguments": "{}"}}]))

    client = client_with(handler)
    turn = await client.complete(system="SYS", messages=[Message(role="user", text="hi")], tools=[TOOL])
    assert seen["url"] == "https://llm.test/api/v1/chat/completions" and seen["auth"] == f"Bearer {KEY}"
    assert seen["body"]["model"] == "some/model" and seen["body"]["tool_choice"] == "auto"
    assert KEY not in json.dumps(seen["body"])
    assert turn.tool_calls[0].name == "search_places"
    assert client.name == "some/model via llm.test" and KEY not in client.name


@pytest.mark.parametrize(
    "status, body, expected",
    [
        (401, {"error": {"message": "bad key"}}, "rejected the API key"),
        (429, {"error": {"message": "slow down"}}, "rate-limiting"),
        (500, {"error": {"message": "upstream exploded"}}, "returned an error: upstream exploded"),
        (400, {"error": {"message": "bad request body"}}, "status 400: bad request body"),
    ],
)
async def test_http_errors_become_llm_errors_without_the_key(status, body, expected):
    client = client_with(lambda request: httpx.Response(status, json=body))
    with pytest.raises(LLMError, match=expected) as exc:
        await client.complete(system="s", messages=[Message(role="user", text="hi")], tools=[TOOL])
    assert KEY not in str(exc.value)


async def test_network_failure_becomes_an_llm_error():
    def handler(request):
        raise httpx.ConnectError("boom")

    with pytest.raises(LLMError, match="could not reach"):
        await client_with(handler).complete(system="s", messages=[Message(role="user", text="hi")], tools=[TOOL])


def test_factory_selection(monkeypatch):
    for name in ("LLM_API_KEY", "LLM_MODEL", "LLM_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    assert isinstance(make_llm(Settings(llm_provider="auto")), StandInPlanner)
    with pytest.raises(ValueError, match="LLM_API_KEY and LLM_MODEL"):
        make_llm(Settings(llm_provider="openrouter"))
    monkeypatch.setenv("LLM_API_KEY", KEY)
    monkeypatch.setenv("LLM_MODEL", "vendor/model:free")
    llm = make_llm(Settings(llm_provider="auto"))
    assert llm.name == "vendor/model:free via openrouter.ai"  # default base URL
    assert isinstance(make_llm(Settings(llm_provider="standin")), StandInPlanner)  # explicit choice still wins
