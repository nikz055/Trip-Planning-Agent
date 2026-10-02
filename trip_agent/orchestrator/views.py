"""The JSON view of a trip: what the API returns and what an action records as its result."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from trip_agent.config import Clock, Settings
from trip_agent.models import ALLOWANCE_TYPES, TERMINAL_STATES, Plan, Session
from trip_agent.orchestrator.budget import check_budget


def _plan_view(plan: Plan, session: Session, settings: Settings, clock: Clock) -> dict[str, Any]:
    freshness = timedelta(minutes=settings.freshness_minutes)
    days = []
    for day in plan.days:
        items = []
        for item in day.items:
            data = item.model_dump(mode="json")
            data["locked"] = item.id in session.locked_item_ids
            result = session.tool_results.get(item.ref.result_id) if item.ref else None
            if item.type in ALLOWANCE_TYPES or result is None:
                data.update(source=None, fetched_at=None, stale=False)
            else:  # BR-10: every priced item shows its source and fetch time
                data.update(
                    source=result.source,
                    fetched_at=result.fetched_at.isoformat(),
                    stale=clock.now() - result.fetched_at > freshness,
                )
            items.append(data)
        days.append({"day": day.day, "date": day.date.isoformat(), "summary": day.summary, "items": items})
    budget = plan.budget
    check = check_budget(budget.total if budget else 0.0, session.request, settings)
    return {
        "days": days,
        "budget": budget.model_dump() if budget else None,
        "budget_check": {**check.model_dump(), "label": check.label},
        "assumptions": plan.assumptions,
        "violations": [v.model_dump() for v in plan.violations],
        "gaps": plan.gaps,
    }


def build_view(session: Session, settings: Settings, clock: Clock, planner_name: str) -> dict[str, Any]:
    plan = session.plan
    return {
        "trip_id": session.trip_id,
        "state": session.state.value,
        "terminal": session.state in TERMINAL_STATES,
        "planner": planner_name,
        "request": session.request.model_dump(mode="json"),
        "form": session.pause.model_dump(mode="json") if session.pause else None,
        "version": session.current_version,
        "versions": [
            {
                "version": v.version,
                "reason": v.reason,
                "parent_version": v.parent_version,
                "created_at": v.created_at.isoformat(),
            }
            for v in session.versions
        ],
        "plan": _plan_view(plan, session, settings, clock) if plan else None,
        "locked_item_ids": session.locked_item_ids,
        "notices": session.notices,
        "give_up": session.give_up.model_dump() if session.give_up else None,
        "limits": {
            "iterations": session.iteration_count,
            "tokens_used": session.tokens_used,
            "estimated_cost_usd": round(session.estimated_cost_usd, 4),
            "stopped_by": session.guard_hit,
        },
        "share_path": f"/share/{session.trip_id}",
    }
