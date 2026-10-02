"""Core data model (spec section 4) plus the persisted session."""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any, Iterator, Literal

from pydantic import BaseModel, Field, PrivateAttr, field_validator

from trip_agent.llm.base import Message

# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------


class State(str, Enum):
    NEW = "NEW"
    WAITING_FOR_DETAILS = "WAITING_FOR_DETAILS"
    PLANNING = "PLANNING"
    VALIDATING = "VALIDATING"
    REPAIRING = "REPAIRING"
    PRESENTED = "PRESENTED"
    REVISING = "REVISING"
    ACCEPTED = "ACCEPTED"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


# PARTIAL is not terminal: a partial plan can still be revised (docs/state-machine.md).
TERMINAL_STATES = frozenset({State.ACCEPTED, State.FAILED, State.CANCELLED})

# --------------------------------------------------------------------------
# Trip request
# --------------------------------------------------------------------------


class Pace(str, Enum):
    relaxed = "relaxed"
    balanced = "balanced"
    packed = "packed"


class TripRequest(BaseModel):
    raw_text: str = ""
    origin: str | None = None
    destination_city: str | None = None
    start_date: date | None = None
    end_date: date | None = None
    dates_flexible: bool = False
    adults: int | None = Field(default=None, ge=1, le=9)
    children: int = Field(default=0, ge=0, le=8)
    total_budget: float | None = Field(default=None, gt=0)
    currency: str = "EUR"
    pace: Pace = Pace.relaxed
    interests: list[str] = Field(default_factory=list)
    # Extension to the spec: a stated hotel class ("5-star") has to live
    # somewhere for the preference-conflict rule (edge case 13) to be askable.
    min_hotel_rating: float | None = Field(default=None, ge=1, le=5)

    @field_validator("origin", "destination_city", mode="before")
    @classmethod
    def _blank_to_none(cls, value: Any) -> Any:
        if isinstance(value, str) and not value.strip():
            return None
        return value.strip() if isinstance(value, str) else value

    @field_validator("currency", mode="before")
    @classmethod
    def _upper_currency(cls, value: Any) -> Any:
        return value.strip().upper() if isinstance(value, str) and value.strip() else "EUR"

    @property
    def travellers(self) -> int:
        return (self.adults or 0) + self.children

    @property
    def trip_days(self) -> int | None:
        if self.start_date is None or self.end_date is None:
            return None
        return (self.end_date - self.start_date).days + 1


# BR-01: what a priced, date-specific search needs.
REQUIRED_FIELDS: tuple[str, ...] = ("origin", "destination_city", "start_date", "end_date", "adults")
ASKABLE_FIELDS: tuple[str, ...] = tuple(f for f in TripRequest.model_fields if f != "raw_text")


def missing_required_fields(request: TripRequest) -> list[str]:
    return [name for name in REQUIRED_FIELDS if getattr(request, name) is None]


# --------------------------------------------------------------------------
# Tool calls and results
# --------------------------------------------------------------------------

ToolCallStatus = Literal["pending", "running", "ok", "failed"]


class ToolCall(BaseModel):
    id: str
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    status: ToolCallStatus = "pending"
    attempts: int = 0
    # bookkeeping beyond the spec
    label: str | None = None
    llm_call_id: str | None = None
    error: str | None = None
    iteration: int = 0
    started_at: datetime | None = None
    finished_at: datetime | None = None


class ToolResult(BaseModel):
    call_id: str
    payload: Any = None
    fetched_at: datetime
    source: str
    error: str | None = None


# --------------------------------------------------------------------------
# Plan
# --------------------------------------------------------------------------


class ItemType(str, Enum):
    flight = "flight"
    hotel = "hotel"
    activity = "activity"
    transfer = "transfer"
    meal_allowance = "meal_allowance"
    other_allowance = "other_allowance"


ALLOWANCE_TYPES = frozenset({ItemType.meal_allowance, ItemType.other_allowance})


class CostStatus(str, Enum):
    confirmed = "confirmed"
    estimated = "estimated"


class ItemRef(BaseModel):
    result_id: str = Field(description="call_id of the tool result this item came from")
    item_id: str = Field(
        description="offer_id, hotel_id or place_id inside that result; "
        "for a transfer use '<from point id>-><to point id>'"
    )


class PlanItem(BaseModel):
    id: str
    type: ItemType
    title: str = ""
    day: int = Field(ge=1)
    start: datetime | None = Field(default=None, description="local time, YYYY-MM-DDTHH:MM")
    end: datetime | None = Field(default=None, description="local time, YYYY-MM-DDTHH:MM")
    ref: ItemRef | None = Field(default=None, description="null only for allowances")
    cost: float = Field(default=0, ge=0)
    cost_status: CostStatus = CostStatus.estimated
    locked: bool = False
    link: str | None = None
    notes: str | None = None

    @field_validator("start", "end")
    @classmethod
    def _naive_local(cls, value: datetime | None) -> datetime | None:
        # All plan times are local wall-clock times; drop any offset the LLM adds.
        if value is not None and value.tzinfo is not None:
            return value.replace(tzinfo=None)
        return value


class PlanDay(BaseModel):
    day: int = Field(ge=1)
    date: date
    summary: str | None = None
    items: list[PlanItem] = Field(default_factory=list)


class Budget(BaseModel):
    flights: float = 0
    lodging: float = 0
    local_transport: float = 0
    activities: float = 0
    meals_allowance: float = 0
    other_allowance: float = 0
    total: float = 0
    confirmed_total: float = 0
    estimated_total: float = 0


class Violation(BaseModel):
    rule: str  # business rule id, e.g. "BR-13"
    check: int | None = None  # verifier check number (spec section 7)
    target: str  # item id, "day N" or "plan"
    message: str


class Plan(BaseModel):
    days: list[PlanDay] = Field(default_factory=list)
    budget: Budget | None = Field(
        default=None, description="omit to let the system compute it; if given it must add up"
    )
    assumptions: list[str] = Field(default_factory=list)
    violations: list[Violation] = Field(default_factory=list, description="set by the system")
    gaps: list[str] = Field(default_factory=list, description="set by the system")

    def items(self) -> Iterator[PlanItem]:
        for day in self.days:
            yield from day.items

    def item(self, item_id: str) -> PlanItem | None:
        return next((i for i in self.items() if i.id == item_id), None)


# --------------------------------------------------------------------------
# Session (everything persisted after each transition)
# --------------------------------------------------------------------------


class FormField(BaseModel):
    name: str
    label: str
    kind: Literal["text", "date", "int", "number", "select", "bool"] = "text"
    required: bool = True
    value: Any = None
    options: list[str] = Field(default_factory=list)
    error: str | None = None


class Pause(BaseModel):
    """Why the session is waiting on the user (the 'pause reason')."""

    kind: Literal["missing_details", "ask_user"]
    reason: str
    fields: list[FormField] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    tool_call_id: str | None = None  # the planner's pending ask_user call, if any


class Transition(BaseModel):
    at: datetime
    from_state: State
    to_state: State
    event: str


class ProposalRecord(BaseModel):
    phase: Literal["initial", "repair"]
    violations: list[Violation] = Field(default_factory=list)
    ungrounded_items: int = 0


class SessionStats(BaseModel):
    llm_calls: int = 0
    clarification_rounds: int = 0
    proposals: list[ProposalRecord] = Field(default_factory=list)
    latency_s: float = 0.0


class GiveUp(BaseModel):
    reason: str
    suggestions: list[str] = Field(default_factory=list)


class Revision(BaseModel):
    instruction: str
    affected_days: list[int] = Field(default_factory=list)  # empty until the change is scoped


VersionReason = Literal["initial", "repair", "revision", "refresh"]


class PlanVersion(BaseModel):
    version: int = Field(ge=1)
    reason: VersionReason
    parent_version: int | None = None
    created_at: datetime
    plan: Plan


class Session(BaseModel):
    trip_id: str  # unguessable UUID; doubles as the share link
    state: State = State.NEW
    request: TripRequest
    messages: list[Message] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_results: dict[str, ToolResult] = Field(default_factory=dict)
    versions: list[PlanVersion] = Field(default_factory=list)  # never overwritten
    current_version: int | None = None  # the version last shown to the user
    repairing_version: int | None = None  # a flawed proposal that is under repair
    candidate: Plan | None = None  # proposed, not yet verified
    revision: Revision | None = None
    revision_origin: State | None = None  # PRESENTED or PARTIAL: where a rejected change returns to
    locked_item_ids: list[str] = Field(default_factory=list)
    iteration_count: int = 0  # per planning run
    tokens_used: int = 0  # per trip
    estimated_cost_usd: float = 0.0  # per trip
    repair_used: bool = False
    repair_turns: int = 0
    guard_hit: str | None = None  # which BR-06 limit ended planning, if any
    pause: Pause | None = None
    pending_call_id: str | None = None  # the planner's propose_plan call awaiting its result
    unavailable_tools: list[str] = Field(default_factory=list)
    refreshed_call_ids: list[str] = Field(default_factory=list)
    clarifications: list[dict[str, Any]] = Field(default_factory=list)
    guessed_fields: list[str] = Field(default_factory=list)
    notices: list[str] = Field(default_factory=list)
    give_up: GiveUp | None = None
    history: list[Transition] = Field(default_factory=list)
    stats: SessionStats = Field(default_factory=SessionStats)
    created_at: datetime
    updated_at: datetime

    # what the store has already written, so a save only sends what changed
    _persisted: dict[str, Any] = PrivateAttr(default_factory=dict)

    def call(self, call_id: str) -> ToolCall | None:
        return next((c for c in self.tool_calls if c.id == call_id), None)

    def version(self, number: int | None) -> PlanVersion | None:
        return next((v for v in self.versions if v.version == number), None)

    @property
    def plan(self) -> Plan | None:
        """The plan last shown to the user."""
        current = self.version(self.current_version)
        return current.plan if current else None

    def add_version(
        self, plan: Plan, reason: VersionReason, parent: int | None, now: datetime
    ) -> PlanVersion:
        version = PlanVersion(
            version=len(self.versions) + 1,
            reason=reason,
            parent_version=parent,
            created_at=now,
            plan=plan,
        )
        self.versions.append(version)
        return version
