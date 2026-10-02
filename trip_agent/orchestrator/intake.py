"""Intake helper step: free text -> TripRequest (state NEW).

A separate, single-tool LLM call. Its output is schema-validated like any
other tool call, and its tokens and cost are reported so they count toward
the trip's limits.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import ValidationError

from trip_agent.config import Clock
from trip_agent.llm.base import LLMClient, LLMToolResult, Message
from trip_agent.llm.prompts import INTAKE_SYSTEM_PROMPT
from trip_agent.models import ASKABLE_FIELDS, TripRequest
from trip_agent.tools.schemas import RECORD_REQUEST_SPEC, RecordRequestArgs, validation_message

MAX_ATTEMPTS = 2


@dataclass
class IntakeResult:
    request: TripRequest
    guessed_fields: list[str] = field(default_factory=list)
    llm_calls: int = 0
    tokens: int = 0
    cost_usd: float = 0.0


async def parse_trip_request(llm: LLMClient, raw_text: str, clock: Clock) -> IntakeResult:
    """Ask the LLM what the request states. Falls back to an empty request,
    which sends every required field to the clarification form."""
    result = IntakeResult(request=TripRequest(raw_text=raw_text))
    messages = [
        Message(
            role="user",
            text=f"Today is {clock.today().isoformat()}.\n<request>\n{raw_text}\n</request>",
        )
    ]
    for _ in range(MAX_ATTEMPTS):
        turn = await llm.complete(
            system=INTAKE_SYSTEM_PROMPT, messages=messages, tools=[RECORD_REQUEST_SPEC]
        )
        result.llm_calls += 1
        result.tokens += turn.input_tokens + turn.output_tokens
        result.cost_usd += turn.cost_usd
        call = next((c for c in turn.tool_calls if c.name == "record_request"), None)
        if call is None:
            messages += [turn.as_message(), Message(role="user", text="Call record_request.")]
            continue
        try:
            args = RecordRequestArgs.model_validate(call.args)
        except ValidationError as exc:
            error = LLMToolResult(
                tool_call_id=call.id,
                content=f"invalid arguments: {validation_message(exc)}",
                is_error=True,
            )
            messages += [turn.as_message(), Message(role="user", tool_results=[error])]
            continue
        fields = args.model_dump(exclude_none=True, exclude={"guessed_fields"})
        result.request = TripRequest(raw_text=raw_text, **fields)
        result.guessed_fields = [f for f in args.guessed_fields if f in ASKABLE_FIELDS]
        break
    return result
