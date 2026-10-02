"""Grounding (BR-08): map a plan item back to the tool result it claims to come from."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from trip_agent.models import (
    ALLOWANCE_TYPES,
    ItemType,
    Plan,
    PlanItem,
    ToolCall,
    ToolResult,
    Violation,
)

EXPECTED_TOOL = {
    ItemType.flight: "search_flights",
    ItemType.hotel: "search_hotels",
    ItemType.activity: "search_places",
    ItemType.transfer: "get_travel_times",
}
RECORD_ID_KEY = {
    ItemType.flight: "offer_id",
    ItemType.hotel: "hotel_id",
    ItemType.activity: "place_id",
}


@dataclass
class Grounded:
    """The stored facts behind one plan item."""

    call: ToolCall
    result: ToolResult
    record: dict[str, Any] | None = None  # the offer, hotel or place
    minutes: int | None = None  # transfers: matrix time for the leg
    unit_cost: float | None = None  # transfers: matrix cost for the leg
    from_id: str | None = None
    to_id: str | None = None


def resolve(
    item: PlanItem, calls: dict[str, ToolCall], results: dict[str, ToolResult]
) -> Grounded | str:
    """Return the grounding for an item, or a message saying why it has none."""
    if item.ref is None:
        return "has no ref; every item except allowances must reference a tool result"
    call = calls.get(item.ref.result_id)
    result = results.get(item.ref.result_id)
    if call is None or result is None or call.status != "ok" or result.error:
        return f"references tool result '{item.ref.result_id}', which does not exist"
    expected = EXPECTED_TOOL[item.type]
    if call.tool != expected:
        return (
            f"references a {call.tool} result, but a {item.type.value} "
            f"must come from {expected}"
        )
    if item.type is ItemType.transfer:
        return _resolve_leg(item, call, result)
    key = RECORD_ID_KEY[item.type]
    record = next(
        (r for r in result.payload or [] if isinstance(r, dict) and r.get(key) == item.ref.item_id),
        None,
    )
    if record is None:
        return f"'{item.ref.item_id}' is not in tool result '{call.id}'"
    return Grounded(call=call, result=result, record=record)


def _resolve_leg(item: PlanItem, call: ToolCall, result: ToolResult) -> Grounded | str:
    assert item.ref is not None
    from_id, sep, to_id = item.ref.item_id.partition("->")
    payload = result.payload or {}
    ids = payload.get("point_ids") or []
    if not sep or from_id not in ids or to_id not in ids:
        return (
            f"leg '{item.ref.item_id}' is not in tool result '{call.id}'; "
            "use '<from point id>-><to point id>'"
        )
    i, j = ids.index(from_id), ids.index(to_id)
    cost_matrix = payload.get("estimated_cost")
    return Grounded(
        call=call,
        result=result,
        minutes=payload["matrix_minutes"][i][j],
        unit_cost=cost_matrix[i][j] if cost_matrix else 0.0,
        from_id=from_id,
        to_id=to_id,
    )


def check_grounding(
    plan: Plan, calls: list[ToolCall], results: dict[str, ToolResult]
) -> list[Violation]:
    """Verifier check 2: every non-allowance item references an existing ToolResult."""
    by_id = {c.id: c for c in calls}
    violations = []
    for item in plan.items():
        if item.type in ALLOWANCE_TYPES:
            if item.ref is not None:
                violations.append(
                    Violation(rule="BR-08", check=2, target=item.id, message="allowances carry no ref")
                )
            continue
        outcome = resolve(item, by_id, results)
        if isinstance(outcome, str):
            violations.append(
                Violation(
                    rule="BR-08",
                    check=2,
                    target=item.id,
                    message=f"{item.type.value} '{item.title or item.id}' {outcome}",
                )
            )
    return violations
