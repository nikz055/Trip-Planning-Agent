"""The transition table in code must be the one documented in docs/state-machine.md."""

import re
from pathlib import Path

import pytest

from trip_agent.models import TERMINAL_STATES, State
from trip_agent.orchestrator.state import (
    PLANNER_TOOLS,
    TRANSITIONS,
    Event,
    IllegalTransition,
    legal_tools,
    transition,
)
from trip_agent.tests.conftest import new_session, trip_request

DOC = Path(__file__).resolve().parents[2] / "docs" / "state-machine.md"


def documented_transitions() -> dict[tuple[State, Event], frozenset[State]]:
    table: dict[tuple[State, Event], frozenset[State]] = {}
    for line in DOC.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 3 or cells[1] not in Event.__members__:
            continue
        targets = frozenset(State(t.strip()) for t in cells[2].split(","))
        sources = [s for s in State if s not in TERMINAL_STATES] if cells[0] == "any non-terminal" else [State(cells[0])]
        for source in sources:
            table[(source, Event(cells[1]))] = targets
    return table


def test_code_matches_the_documented_table():
    documented = documented_transitions()
    assert len(documented) >= 20, "the table in docs/state-machine.md was not parsed"
    assert TRANSITIONS == documented


def test_partial_is_not_terminal_and_can_be_revised_or_cancelled():
    assert TERMINAL_STATES == {State.ACCEPTED, State.FAILED, State.CANCELLED}
    assert TRANSITIONS[(State.PARTIAL, Event.request_change)] == {State.REVISING}
    assert TRANSITIONS[(State.PARTIAL, Event.cancel)] == {State.CANCELLED}
    assert (State.PARTIAL, Event.accept) not in TRANSITIONS  # a partial plan cannot be accepted


def test_terminal_states_have_no_way_out():
    for (source, _), _targets in TRANSITIONS.items():
        assert source not in TERMINAL_STATES


def test_every_state_is_reachable():
    reachable = {State.NEW} | {target for targets in TRANSITIONS.values() for target in targets}
    assert reachable == set(State)


def test_transition_moves_the_session_and_records_history(clock):
    session = new_session(trip_request(), clock)
    transition(session, Event.request_complete, State.PLANNING, clock.now())
    transition(session, Event.data_tools, State.PLANNING, clock.now())
    assert session.state is State.PLANNING
    assert [(t.from_state, t.event, t.to_state) for t in session.history] == [
        (State.NEW, "request_complete", State.PLANNING),
        (State.PLANNING, "data_tools", State.PLANNING),
    ]


@pytest.mark.parametrize(
    "state, event, to",
    [
        (State.NEW, Event.accept, State.ACCEPTED),  # no such row
        (State.PLANNING, Event.propose_plan, State.PRESENTED),  # must go through VALIDATING
        (State.PRESENTED, Event.propose_plan, State.VALIDATING),  # planner cannot act here
        (State.ACCEPTED, Event.request_change, State.REVISING),  # terminal
        (State.CANCELLED, Event.cancel, State.CANCELLED),
        (State.PARTIAL, Event.accept, State.ACCEPTED),
        (State.WAITING_FOR_DETAILS, Event.form_valid, State.PRESENTED),  # wrong target
    ],
)
def test_illegal_transitions_are_rejected(clock, state, event, to):
    session = new_session(trip_request(), clock)
    session.state = state
    with pytest.raises(IllegalTransition):
        transition(session, event, to, clock.now())
    assert session.state is state and session.history == []


def test_planner_tools_per_state():
    assert set(PLANNER_TOOLS) == {State.PLANNING, State.REPAIRING}
    assert len(legal_tools(State.PLANNING)) == 7
    assert legal_tools(State.PLANNING, wrap_up=True) == {"propose_plan", "give_up"}
    assert "ask_user" not in legal_tools(State.REPAIRING)
    assert {"search_flights", "propose_plan", "give_up"} <= legal_tools(State.REPAIRING)
    for state in State:
        if state not in (State.PLANNING, State.REPAIRING):
            assert legal_tools(state) == frozenset()
