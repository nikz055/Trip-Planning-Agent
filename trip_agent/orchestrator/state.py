"""The workflow state machine. Mirrors docs/state-machine.md row for row."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from trip_agent.models import TERMINAL_STATES, Session, State, Transition
from trip_agent.tools.schemas import DATA_TOOLS

S = State
TERMINAL = TERMINAL_STATES


class Event(str, Enum):
    request_incomplete = "request_incomplete"
    request_complete = "request_complete"
    form_valid = "form_valid"
    form_invalid = "form_invalid"
    data_tools = "data_tools"
    ask_user = "ask_user"
    propose_plan = "propose_plan"
    give_up = "give_up"
    loop_guard = "loop_guard"
    no_violations = "no_violations"
    violations = "violations"
    violations_after_repair = "violations_after_repair"
    tool_gaps = "tool_gaps"
    repaired_plan = "repaired_plan"
    repair_abandoned = "repair_abandoned"
    accept = "accept"
    request_change = "request_change"
    change_scoped = "change_scoped"
    change_rejected = "change_rejected"
    cancel = "cancel"


E = Event

TRANSITIONS: dict[tuple[State, Event], frozenset[State]] = {
    (S.NEW, E.request_incomplete): frozenset({S.WAITING_FOR_DETAILS}),
    (S.NEW, E.request_complete): frozenset({S.PLANNING}),
    (S.WAITING_FOR_DETAILS, E.form_valid): frozenset({S.PLANNING}),
    (S.WAITING_FOR_DETAILS, E.form_invalid): frozenset({S.WAITING_FOR_DETAILS}),
    (S.PLANNING, E.data_tools): frozenset({S.PLANNING}),
    (S.PLANNING, E.ask_user): frozenset({S.WAITING_FOR_DETAILS}),
    (S.PLANNING, E.propose_plan): frozenset({S.VALIDATING}),
    (S.PLANNING, E.give_up): frozenset({S.FAILED}),
    (S.PLANNING, E.loop_guard): frozenset({S.PARTIAL, S.FAILED}),
    (S.VALIDATING, E.no_violations): frozenset({S.PRESENTED}),
    (S.VALIDATING, E.violations): frozenset({S.REPAIRING}),
    (S.VALIDATING, E.violations_after_repair): frozenset({S.PARTIAL}),
    (S.VALIDATING, E.tool_gaps): frozenset({S.PARTIAL}),
    (S.REPAIRING, E.data_tools): frozenset({S.REPAIRING}),
    (S.REPAIRING, E.repaired_plan): frozenset({S.VALIDATING}),
    (S.REPAIRING, E.repair_abandoned): frozenset({S.PARTIAL}),
    (S.PRESENTED, E.accept): frozenset({S.ACCEPTED}),
    (S.PRESENTED, E.request_change): frozenset({S.REVISING}),
    (S.PARTIAL, E.request_change): frozenset({S.REVISING}),
    (S.REVISING, E.change_scoped): frozenset({S.PLANNING}),
    (S.REVISING, E.change_rejected): frozenset({S.PRESENTED, S.PARTIAL}),
    **{(state, E.cancel): frozenset({S.CANCELLED}) for state in State if state not in TERMINAL},
}

# Planner actions that are legal in each state; anything else is rejected.
_CONTROL_ALL = frozenset({"ask_user", "propose_plan", "give_up"})
_WRAP_UP = frozenset({"propose_plan", "give_up"})
PLANNER_TOOLS: dict[State, frozenset[str]] = {
    S.PLANNING: DATA_TOOLS | _CONTROL_ALL,
    S.REPAIRING: DATA_TOOLS | _WRAP_UP,
}
WRAP_UP_TOOLS = _WRAP_UP

# States in which a planning run is in flight and should be driven forward.
RUNNING: frozenset[State] = frozenset({S.PLANNING, S.VALIDATING, S.REPAIRING})


class IllegalTransition(Exception):
    def __init__(self, state: State, event: Event, to: State | None = None):
        target = f" to {to.value}" if to else ""
        super().__init__(f"illegal transition: {state.value} --{event.value}-->{target}")


def legal_tools(state: State, *, wrap_up: bool = False) -> frozenset[str]:
    tools = PLANNER_TOOLS.get(state, frozenset())
    return tools & WRAP_UP_TOOLS if wrap_up else tools


def transition(session: Session, event: Event, to: State, now: datetime) -> None:
    """Move the session to `to`, or raise if the table does not allow it."""
    allowed = TRANSITIONS.get((session.state, event))
    if allowed is None or to not in allowed:
        raise IllegalTransition(session.state, event, to)
    session.history.append(Transition(at=now, from_state=session.state, to_state=to, event=event.value))
    session.state = to
