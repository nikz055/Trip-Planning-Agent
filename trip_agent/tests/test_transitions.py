"""The transitions added to the original spec, and the planner actions the
state machine refuses. See docs/state-machine.md."""

from __future__ import annotations

import pytest

from trip_agent.llm.base import LLMToolCall
from trip_agent.llm.standin import StandInPlanner
from trip_agent.models import State
from trip_agent.orchestrator.orchestrator import InvalidState, NotFound, TripError
from trip_agent.tests.conftest import VIENNA, MutatingPlanner, find_item, items_of

FORM = {"origin": "London", "start_date": "2026-11-07", "end_date": "2026-11-11", "adults": "2"}


def path(session) -> list[tuple[str, str, str]]:
    return [(t.from_state.value, t.event, t.to_state.value) for t in session.history]


def _invent_hotel(plan: dict) -> None:
    find_item(plan, "hotel")["ref"]["item_id"] = "VIE-H99"


class RepairBehaviour(MutatingPlanner):
    """Proposes an invented hotel, then does `in_repair` when asked to repair it."""

    def __init__(self, in_repair: str):
        super().__init__(_invent_hotel)
        self._in_repair = in_repair
        self.repair_turns = 0

    async def complete(self, *, system, messages, tools):
        turn = await super().complete(system=system, messages=messages, tools=tools)
        repairing = len(self.proposals) >= 1 and any(t.name == "propose_plan" for t in tools) and not any(
            t.name == "ask_user" for t in tools
        )
        if not repairing:
            return turn
        self.repair_turns += 1
        call_id = turn.tool_calls[0].id
        if self._in_repair == "give_up":
            turn.tool_calls = [LLMToolCall(id=call_id, name="give_up", args={"reason": "cannot find another hotel"})]
        elif self._in_repair == "ask_user":
            turn.tool_calls = [LLMToolCall(id=call_id, name="ask_user", args={"fields": ["total_budget"], "reason": "more money?"})]
            self._in_repair = "propose"  # after the refusal, repair normally
        elif self._in_repair == "keep_searching":
            turn.tool_calls = [LLMToolCall(id=call_id, name="search_places", args={"city": "Vienna", "limit": 3})]
        elif self._in_repair == "search_then_propose" and self.repair_turns == 1:
            turn.tool_calls = [LLMToolCall(id=call_id, name="search_places", args={"city": "Vienna", "limit": 3})]
        return turn


# -- VALIDATING -> PARTIAL (tool_gaps) ----------------------------------------


async def test_validating_to_partial_when_a_tool_was_unavailable(orchestrator, failures, store):
    failures.set("search_places", {"mode": "error"})
    view = await orchestrator.create_trip(VIENNA)
    assert view["state"] == "PARTIAL" and view["plan"]["violations"] == []
    assert view["plan"]["gaps"] == ["Places could not be searched, so the plan has no activities."]
    session = await store.load(view["trip_id"])
    assert path(session)[-1] == ("VALIDATING", "tool_gaps", "PARTIAL")
    assert len(items_of(view, "flight")) == 2 and items_of(view, "activity") == []


# -- REPAIRING -> PARTIAL (repair_abandoned) ----------------------------------


async def test_repairing_to_partial_when_the_planner_gives_up(make_orchestrator, store):
    view = await make_orchestrator(RepairBehaviour("give_up")).create_trip(VIENNA)
    assert view["state"] == "PARTIAL"
    assert any(v["rule"] == "BR-08" for v in view["plan"]["violations"])  # shown with the violations listed
    assert any("could not be repaired" in n and "cannot find another hotel" in n for n in view["notices"])
    session = await store.load(view["trip_id"])
    assert path(session)[-2:] == [("VALIDATING", "violations", "REPAIRING"), ("REPAIRING", "repair_abandoned", "PARTIAL")]
    assert session.current_version == 1 and session.repairing_version is None


async def test_repairing_to_partial_when_the_repair_hits_its_turn_limit(make_orchestrator, store):
    planner = RepairBehaviour("keep_searching")
    view = await make_orchestrator(planner, repair_max_turns=3).create_trip(VIENNA)
    assert view["state"] == "PARTIAL" and planner.repair_turns == 3
    assert any("ran out of turns" in n for n in view["notices"])
    session = await store.load(view["trip_id"])
    assert path(session)[-4:] == [
        ("REPAIRING", "data_tools", "REPAIRING"),
        ("REPAIRING", "data_tools", "REPAIRING"),
        ("REPAIRING", "data_tools", "REPAIRING"),
        ("REPAIRING", "repair_abandoned", "PARTIAL"),
    ]


async def test_repairing_to_partial_when_the_wall_clock_runs_out(make_orchestrator, clock, store):
    class SlowRepair(RepairBehaviour):
        async def complete(self, *, system, messages, tools):
            turn = await super().complete(system=system, messages=messages, tools=tools)
            if self.repair_turns:
                clock.advance(minutes=3)
            return turn

    view = await make_orchestrator(SlowRepair("keep_searching"), wall_clock_s=120).create_trip(VIENNA)
    assert view["state"] == "PARTIAL"
    assert any("wall-clock limit of 120 s was reached" in n for n in view["notices"])
    assert path(await store.load(view["trip_id"]))[-1] == ("REPAIRING", "repair_abandoned", "PARTIAL")


async def test_a_repair_may_look_something_up_before_proposing_again(make_orchestrator, store):
    view = await make_orchestrator(RepairBehaviour("search_then_propose")).create_trip(VIENNA)
    assert view["state"] == "PRESENTED" and view["version"] == 2
    events = path(await store.load(view["trip_id"]))
    assert events[-3:] == [
        ("REPAIRING", "data_tools", "REPAIRING"),
        ("REPAIRING", "repaired_plan", "VALIDATING"),
        ("VALIDATING", "no_violations", "PRESENTED"),
    ]


async def test_only_one_repair_attempt_per_proposal(make_orchestrator, store):
    planner = MutatingPlanner(_invent_hotel, count=5)
    view = await make_orchestrator(planner).create_trip(VIENNA)
    assert view["state"] == "PARTIAL" and len(planner.proposals) == 2  # the proposal and one repair, no more
    session = await store.load(view["trip_id"])
    assert [p.phase for p in session.stats.proposals] == ["initial", "repair"]
    assert path(session)[-1] == ("VALIDATING", "violations_after_repair", "PARTIAL")


# -- PARTIAL -> REVISING ------------------------------------------------------


async def test_a_partial_plan_can_be_revised(orchestrator, failures, store):
    failures.set("search_flights", {"mode": "timeout"})
    partial = await orchestrator.create_trip(VIENNA)
    assert partial["state"] == "PARTIAL"
    revised = await orchestrator.revise(partial["trip_id"], "Less walking on day 2")
    assert revised["state"] == "PARTIAL"  # the flights are still missing, so it is still partial
    assert revised["version"] == 2 and revised["versions"][-1]["reason"] == "revision"
    events = path(await store.load(partial["trip_id"]))
    assert ("PARTIAL", "request_change", "REVISING") in events
    assert events[-1] == ("VALIDATING", "tool_gaps", "PARTIAL")
    assert revised["plan"]["days"][0] == partial["plan"]["days"][0]  # day 1 untouched


async def test_a_partial_plan_cannot_be_accepted_but_can_be_cancelled(orchestrator, failures):
    failures.set("search_flights", {"mode": "timeout"})
    partial = await orchestrator.create_trip(VIENNA)
    with pytest.raises(InvalidState, match="Only a presented plan can be accepted"):
        await orchestrator.accept(partial["trip_id"])
    cancelled = await orchestrator.cancel(partial["trip_id"])
    assert cancelled["state"] == "CANCELLED" and cancelled["terminal"] is True


# -- REVISING -> PRESENTED / PARTIAL (change_rejected) ------------------------


async def test_a_change_that_hits_a_locked_item_is_refused_and_asks_to_unlock(orchestrator, store):
    plan = await orchestrator.create_trip(VIENNA)
    await orchestrator.set_lock(plan["trip_id"], "hotel", True)
    refused = await orchestrator.revise(plan["trip_id"], "Please change the hotel to a cheaper one")
    assert refused["state"] == "PRESENTED" and refused["version"] == 1  # the plan is untouched
    hotel_name = items_of(plan, "hotel")[0]["title"]
    assert refused["notices"] == [f"That change affects a locked item ('{hotel_name}'). Unlock it and ask again."]
    assert refused["plan"]["days"] == (await orchestrator.get(plan["trip_id"]))["plan"]["days"]
    session = await store.load(plan["trip_id"])
    assert path(session)[-2:] == [("PRESENTED", "request_change", "REVISING"), ("REVISING", "change_rejected", "PRESENTED")]
    assert session.revision is None and len(session.versions) == 1

    await orchestrator.set_lock(plan["trip_id"], "hotel", False)  # after unlocking, a revision goes through
    assert (await orchestrator.revise(plan["trip_id"], "Less walking on day 2"))["version"] == 2


async def test_a_refused_change_on_a_partial_plan_returns_to_partial(orchestrator, failures, store):
    failures.set("search_flights", {"mode": "timeout"})
    partial = await orchestrator.create_trip(VIENNA)
    await orchestrator.set_lock(partial["trip_id"], "hotel", True)
    refused = await orchestrator.revise(partial["trip_id"], "switch to a different hotel")
    assert refused["state"] == "PARTIAL" and refused["version"] == 1
    assert path(await store.load(partial["trip_id"]))[-1] == ("REVISING", "change_rejected", "PARTIAL")


async def test_a_change_that_cannot_be_scoped_is_refused(make_orchestrator, store):
    class BadScope(StandInPlanner):
        async def complete(self, *, system, messages, tools):
            turn = await super().complete(system=system, messages=messages, tools=tools)
            if any(t.name == "scope_revision" for t in tools):
                turn.tool_calls[0].args["affected_days"] = [42]  # not a day of the plan
            return turn

    orchestrator = make_orchestrator(BadScope())
    plan = await orchestrator.create_trip(VIENNA)
    refused = await orchestrator.revise(plan["trip_id"], "make it nicer")
    assert refused["state"] == "PRESENTED" and refused["version"] == 1
    assert "could not work out which days" in refused["notices"][0]
    session = await store.load(plan["trip_id"])
    assert path(session)[-1] == ("REVISING", "change_rejected", "PRESENTED")
    assert session.stats.llm_calls == 5 + 2  # the scoping step was tried twice, and both calls were counted


# -- planner actions the state machine refuses --------------------------------


async def test_ask_user_is_not_available_while_repairing(make_orchestrator, store):
    planner = RepairBehaviour("ask_user")
    view = await make_orchestrator(planner).create_trip(VIENNA)
    assert view["state"] == "PRESENTED"  # the refusal was returned to the planner, which then repaired
    session = await store.load(view["trip_id"])
    refusals = [r.content for m in session.messages for r in m.tool_results if "cannot be called now" in r.content]
    assert len(refusals) == 1 and "ask_user cannot be called now (state REPAIRING)" in refusals[0]
    assert session.stats.clarification_rounds == 0 and "ask_user" not in [e for _, e, _ in path(session)]


async def test_planner_cannot_act_once_a_plan_is_presented(orchestrator, store):
    view = await orchestrator.create_trip(VIENNA)
    session = await store.load(view["trip_id"])
    assert session.state is State.PRESENTED
    assert session.messages[-1].role == "user"  # the conversation rests on the system's side
    with pytest.raises(InvalidState):
        await orchestrator.submit_form(view["trip_id"], FORM)  # not waiting for details


# -- user actions in the wrong state ------------------------------------------


async def test_user_actions_are_checked_against_the_state(orchestrator):
    waiting = await orchestrator.create_trip("Plan a trip to Austria")
    trip = waiting["trip_id"]
    for action in (orchestrator.accept(trip), orchestrator.revise(trip, "less walking"), orchestrator.set_lock(trip, "hotel", True)):
        with pytest.raises(InvalidState):
            await action
    presented = await orchestrator.submit_form(trip, FORM)
    assert presented["state"] == "PRESENTED"
    with pytest.raises(NotFound):
        await orchestrator.set_lock(trip, "no-such-item", True)
    with pytest.raises(TripError, match="Say what you would like to change"):
        await orchestrator.revise(trip, "   ")
    accepted = await orchestrator.accept(trip)
    assert accepted["state"] == "ACCEPTED" and accepted["terminal"] is True
    for action in (orchestrator.revise(trip, "less walking"), orchestrator.cancel(trip), orchestrator.accept(trip)):
        with pytest.raises(InvalidState):
            await action


async def test_cancel_from_waiting_and_presented(orchestrator):
    waiting = await orchestrator.create_trip("Plan a trip to Austria")
    assert (await orchestrator.cancel(waiting["trip_id"]))["state"] == "CANCELLED"
    with pytest.raises(InvalidState, match="already CANCELLED"):
        await orchestrator.cancel(waiting["trip_id"])
    presented = await orchestrator.create_trip(VIENNA)
    cancelled = await orchestrator.cancel(presented["trip_id"])
    assert cancelled["state"] == "CANCELLED" and cancelled["form"] is None


async def test_unknown_trip(orchestrator):
    with pytest.raises(NotFound):
        await orchestrator.get("00000000-0000-0000-0000-000000000000")
    with pytest.raises(NotFound):
        await orchestrator.accept("not-even-a-uuid")
