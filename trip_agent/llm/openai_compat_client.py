"""Planner LLM over an OpenAI-compatible chat-completions API (OpenRouter and others).

Configured with LLM_BASE_URL, LLM_API_KEY and LLM_MODEL. The key is read from
the environment, sent only in the Authorization header, and never logged or
put in an error message. Tool credentials never pass through here (BR-03).

Not yet exercised against a live endpoint. The unit tests cover the message
conversion and response parsing against a fake transport.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx

from trip_agent.llm.base import AssistantTurn, LLMError, LLMToolCall, Message, ToolSpec

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
REQUEST_TIMEOUT_S = 120.0


def to_wire(system: str, messages: list[Message]) -> list[dict[str, Any]]:
    """Neutral messages -> chat-completions format.

    Each tool result becomes its own `tool` message, directly after the
    assistant turn that asked for it; any text follows as a `user` message.
    """
    wire: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for message in messages:
        if message.role == "assistant":
            entry: dict[str, Any] = {"role": "assistant", "content": message.text or None}
            if message.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": json.dumps(call.args)},
                    }
                    for call in message.tool_calls
                ]
            wire.append(entry)
            continue
        for result in message.tool_results:
            wire.append({"role": "tool", "tool_call_id": result.tool_call_id, "content": result.content})
        if message.text:
            wire.append({"role": "user", "content": message.text})
    return wire


def to_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    return [
        {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.input_schema}}
        for t in tools
    ]


def from_response(body: dict[str, Any]) -> AssistantTurn:
    if "choices" not in body or not body["choices"]:
        detail = (body.get("error") or {}).get("message", "no choices in the response")
        raise LLMError(f"the model endpoint returned an error: {detail}")
    choice = body["choices"][0]
    message = choice.get("message") or {}
    calls = []
    for index, raw in enumerate(message.get("tool_calls") or []):
        function = raw.get("function") or {}
        try:
            args = json.loads(function.get("arguments") or "{}")
            if not isinstance(args, dict):
                raise ValueError("arguments are not an object")
        except ValueError:
            # Leave something the schema check will reject, so the model is told and can retry.
            args = {"unparseable_arguments": str(function.get("arguments"))[:200]}
        calls.append(LLMToolCall(id=raw.get("id") or f"call_{index}", name=function.get("name", ""), args=args))
    usage = body.get("usage") or {}
    finish = choice.get("finish_reason")
    return AssistantTurn(
        text=message.get("content") or None,
        tool_calls=calls,
        input_tokens=int(usage.get("prompt_tokens") or 0),
        output_tokens=int(usage.get("completion_tokens") or 0),
        cost_usd=float(usage.get("cost") or 0.0),  # OpenRouter reports it; other endpoints may not
        stop_reason="max_tokens" if finish == "length" else finish,
    )


class _Transient(Exception):
    """A failure worth retrying: rate limit, overload, network."""


class OpenAICompatClient:
    retry_delays: tuple[float, ...] = (3.0, 8.0, 15.0)

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        max_tokens: int = 16000,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.model, self.max_tokens = model, max_tokens
        self.name = f"{model} via {httpx.URL(base_url).host}"
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}", "X-Title": "trip-agent"},
            timeout=REQUEST_TIMEOUT_S,
            transport=transport,
        )

    async def complete(
        self, *, system: str, messages: list[Message], tools: list[ToolSpec]
    ) -> AssistantTurn:
        payload = {
            "model": self.model,
            "messages": to_wire(system, messages),
            "tools": to_tools(tools),
            "tool_choice": "auto",
            "max_tokens": self.max_tokens,
        }
        # Rate limits and upstream overloads are usually brief: try a few times before giving up.
        for attempt in range(len(self.retry_delays) + 1):
            try:
                return await self._once(payload)
            except _Transient as exc:
                if attempt == len(self.retry_delays):
                    raise LLMError(str(exc)) from exc
                await asyncio.sleep(self.retry_delays[attempt])
        raise AssertionError("unreachable")

    async def _once(self, payload: dict[str, Any]) -> AssistantTurn:
        try:
            response = await self._client.post("/chat/completions", json=payload)
        except httpx.HTTPError as exc:
            raise _Transient(f"could not reach the model endpoint ({type(exc).__name__})") from exc
        if response.status_code == 401:
            raise LLMError("the model endpoint rejected the API key; check LLM_API_KEY")
        if response.status_code == 429:
            raise _Transient("the model endpoint is rate-limiting requests")
        try:
            body = response.json()
        except ValueError as exc:
            raise _Transient(f"the model endpoint returned status {response.status_code} with no JSON") from exc
        error = (body.get("error") or {}) if isinstance(body, dict) else {}
        if response.status_code >= 500 or (response.status_code == 200 and error):
            raise _Transient(f"the model endpoint returned an error: {error.get('message', response.status_code)}"[:300])
        if response.status_code >= 400:
            raise LLMError(f"the model endpoint returned status {response.status_code}: {error.get('message', '')}"[:300])
        if any((choice.get("finish_reason") == "error") for choice in body.get("choices") or []):
            raise _Transient("the model provider failed while generating its answer")
        return from_response(body)
