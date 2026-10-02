"""Builds what the planner sees: the per-call context block and tool results.

The context is appended to the newest user-side message on every call instead
of being edited into earlier turns, so the conversation stays append-only.
Tool results are passed as JSON data and never interpreted (BR-11).
"""

from __future__ import annotations

import json
from typing import Any

from trip_agent.config import Clock, Settings
from trip_agent.llm.base import LLMToolResult, Message
from trip_agent.models import Session, ToolCall


def context_block(
    session: Session, settings: Settings, clock: Clock, *, final_call: bool = False
) -> str:
    """Current TripRequest, locked items, open violations and limits, as JSON."""
    request = session.request
    under_repair = session.version(session.repairing_version)
    payload: dict[str, Any] = {
        "today": clock.today().isoformat(),
        "trip_request": request.model_dump(mode="json"),
        "travellers": request.travellers,
        "budget": {
            "amount": request.total_budget,
            "currency": request.currency,
            "limit_in_plan_currency": settings.to_plan_currency(
                request.total_budget, request.currency
            ),
            "plan_currency": settings.plan_currency,
        },
        "locked_item_ids": session.locked_item_ids,
        "violations": [v.model_dump() for v in under_repair.plan.violations] if under_repair else [],
        "unavailable_tools": session.unavailable_tools,
        "clarifications": session.clarifications,
        "clarifications_left": max(
            0, settings.max_clarification_rounds - session.stats.clarification_rounds
        ),
        "iterations_used": session.iteration_count,
        "iterations_max": settings.max_iterations,
        "final_call": final_call,
        "rules": {
            "buffer_minutes": settings.buffer_minutes,
            "arrival_buffer_minutes": settings.arrival_buffer_minutes,
            "departure_buffer_minutes": settings.departure_buffer_minutes,
            "arrival_day_max_activities": settings.arrival_day_max_activities,
            "max_activities_per_day": settings.pace_caps[request.pace.value],
            "freshness_minutes": settings.freshness_minutes,
            "flex_days": settings.flex_days,
        },
    }
    if session.revision is not None and session.plan is not None:
        payload["revision"] = {
            "instruction": session.revision.instruction,
            "affected_days": session.revision.affected_days,
            "previous_plan": session.plan.model_dump(mode="json", exclude={"gaps"}),
        }
    return "<context>\n" + json.dumps(payload, ensure_ascii=False) + "\n</context>"


def _result_body(call: ToolCall, session: Session) -> dict[str, Any]:
    body: dict[str, Any] = {"call_id": call.id, "tool": call.tool, "args": call.args}
    result = session.tool_results.get(call.id)
    if call.status == "ok" and result is not None:
        body.update(
            status="ok",
            source=result.source,
            fetched_at=result.fetched_at.isoformat(),
            data=result.payload,
        )
    else:
        body.update(status="error", error=call.error or "call failed")
    return body


def tool_result_message(call: ToolCall, session: Session) -> LLMToolResult:
    """One data-tool call rendered for the planner."""
    assert call.llm_call_id is not None
    return LLMToolResult(
        tool_call_id=call.llm_call_id,
        content=json.dumps(_result_body(call, session), ensure_ascii=False),
        is_error=call.status != "ok",
    )


def refreshed_block(session: Session) -> str:
    """Results that were re-fetched since the planner last saw them (BR-10)."""
    calls = [c for c in session.tool_calls if c.id in session.refreshed_call_ids]
    if not calls:
        return ""
    bodies = [_result_body(call, session) for call in calls]
    return (
        "These tool results were re-fetched; they replace the earlier results with the same "
        "call_id.\n<refreshed_results>\n" + json.dumps(bodies, ensure_ascii=False) + "\n</refreshed_results>"
    )


def control_result(llm_call_id: str, body: dict[str, Any], *, is_error: bool = False) -> LLMToolResult:
    return LLMToolResult(
        tool_call_id=llm_call_id, content=json.dumps(body, ensure_ascii=False), is_error=is_error
    )


def user_message(text: str, context: str, results: list[LLMToolResult] | None = None) -> Message:
    return Message(role="user", text=f"{context}\n{text}".strip(), tool_results=results or [])
