"""Argument schemas for every tool the planner can call (spec section 5).

Each call's arguments are validated with these models before anything runs
(BR-04). `today` is supplied through the validation context so "today or
later" can be checked without the model reaching for the wall clock.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

from trip_agent.llm.base import ToolSpec
from trip_agent.models import ASKABLE_FIELDS, Plan

IATA_PATTERN = r"^[A-Z]{3}$"


class DataToolArgs(BaseModel):
    """Fields shared by all data tools. They steer execution order (BR-05)."""

    model_config = ConfigDict(extra="forbid")

    label: str | None = Field(
        default=None, description="Optional name for this call so another call can depend on it."
    )
    depends_on: list[str] = Field(
        default_factory=list,
        description="Labels or call_ids this call must wait for. Leave empty to run in parallel.",
    )


class SearchFlightsArgs(DataToolArgs):
    origin: str = Field(pattern=IATA_PATTERN, description="IATA airport code, e.g. DEL")
    destination: str = Field(pattern=IATA_PATTERN, description="IATA airport code, e.g. VIE")
    depart_date: date = Field(description="YYYY-MM-DD, today or later")
    return_date: date | None = Field(default=None, description="YYYY-MM-DD, >= depart_date")
    adults: int = Field(ge=1, le=9)
    children: int = Field(default=0, ge=0, le=8)
    cabin: Literal["economy", "premium", "business", "first"] = "economy"
    max_price: float | None = Field(default=None, gt=0)

    @field_validator("depart_date")
    @classmethod
    def _not_in_past(cls, value: date, info: ValidationInfo) -> date:
        today = (info.context or {}).get("today")
        if today is not None and value < today:
            raise ValueError(f"depart_date must be today ({today}) or later")
        return value

    @model_validator(mode="after")
    def _return_after_depart(self) -> "SearchFlightsArgs":
        if self.return_date is not None and self.return_date < self.depart_date:
            raise ValueError("return_date must be on or after depart_date")
        return self


class SearchHotelsArgs(DataToolArgs):
    city: str = Field(min_length=1)
    check_in: date | None = Field(
        default=None,
        description="YYYY-MM-DD. Omit it when this call depends_on a search_flights call: "
        "the system then derives check-in from that flight's local arrival date.",
    )
    check_out: date = Field(description="YYYY-MM-DD, after check_in")
    guests: int = Field(ge=1, le=10)
    rooms: int = Field(default=1, ge=1, le=5)
    max_price_per_night: float | None = Field(default=None, gt=0)
    min_rating: float | None = Field(default=None, ge=1, le=5)
    flight_offer_id: str | None = Field(
        default=None,
        description="With depends_on: the offer whose arrival date sets check_in "
        "(defaults to the cheapest offer of the flight search).",
    )

    @model_validator(mode="after")
    def _check_out_after_check_in(self) -> "SearchHotelsArgs":
        if self.check_in is not None and self.check_out <= self.check_in:
            raise ValueError("check_out must be after check_in")
        return self


class SearchPlacesArgs(DataToolArgs):
    city: str = Field(min_length=1)
    interests: list[str] | None = None
    limit: int = Field(default=20, ge=1, le=30)


class TravelPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)


class GetTravelTimesArgs(DataToolArgs):
    points: list[TravelPoint] = Field(min_length=2, max_length=25)
    mode: Literal["walk", "transit", "taxi"]

    @field_validator("points")
    @classmethod
    def _unique_ids(cls, value: list[TravelPoint]) -> list[TravelPoint]:
        ids = [p.id for p in value]
        if len(ids) != len(set(ids)):
            raise ValueError("point ids must be unique")
        return value


class AskUserArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fields: list[str] = Field(min_length=1, description="TripRequest fields to ask for, all at once")
    reason: str = Field(min_length=1)
    options: dict[str, list[str]] | None = Field(
        default=None, description="Optional suggested values per field, e.g. nearby dates that work."
    )

    @field_validator("fields")
    @classmethod
    def _known_fields(cls, value: list[str]) -> list[str]:
        unknown = [f for f in value if f not in ASKABLE_FIELDS]
        if unknown:
            raise ValueError(
                f"unknown TripRequest fields {unknown}; allowed: {list(ASKABLE_FIELDS)}"
            )
        return list(dict.fromkeys(value))


class ProposePlanArgs(BaseModel):
    plan: Plan


class GiveUpArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str = Field(min_length=1)
    suggestions: list[str] = Field(default_factory=list)


class RecordRequestArgs(BaseModel):
    """Output of the intake helper step: what the free-text request states.

    Not a planner tool. It is the only tool offered while a trip is in NEW.
    """

    model_config = ConfigDict(extra="forbid")

    origin: str | None = Field(default=None, description="Departure city as the user wrote it")
    destination_city: str | None = None
    start_date: date | None = Field(default=None, description="YYYY-MM-DD")
    end_date: date | None = Field(default=None, description="YYYY-MM-DD")
    dates_flexible: bool | None = None
    adults: int | None = Field(default=None, ge=1, le=9)
    children: int | None = Field(default=None, ge=0, le=8)
    total_budget: float | None = Field(default=None, gt=0)
    currency: str | None = Field(default=None, description="ISO code, e.g. EUR")
    pace: Literal["relaxed", "balanced", "packed"] | None = None
    interests: list[str] | None = None
    min_hotel_rating: float | None = Field(default=None, ge=1, le=5)
    guessed_fields: list[str] = Field(
        default_factory=list,
        description="Fields you inferred rather than read (e.g. a city guessed from a "
        "country). The user is asked to confirm these.",
    )


RECORD_REQUEST_SPEC = ToolSpec(
    name="record_request",
    description=(
        "Record the trip details stated in the user's request. Leave a field out when the "
        "request does not state it; never invent a value."
    ),
    input_schema=RecordRequestArgs.model_json_schema(),
)


class ScopeRevisionArgs(BaseModel):
    """Output of the revision-scoping helper step (state REVISING). Not a planner tool."""

    model_config = ConfigDict(extra="forbid")

    affected_days: list[int] = Field(min_length=1, description="Day numbers the change touches")
    target_item_ids: list[str] = Field(
        default_factory=list,
        description="Ids of existing plan items the user wants changed, moved or removed",
    )
    summary: str = Field(default="", description="One sentence: what will change")

    @field_validator("affected_days")
    @classmethod
    def _positive_unique(cls, value: list[int]) -> list[int]:
        if any(day < 1 for day in value):
            raise ValueError("day numbers start at 1")
        return sorted(set(value))


SCOPE_REVISION_SPEC = ToolSpec(
    name="scope_revision",
    description=(
        "Record which days of the current plan a change request affects, and which existing "
        "items it targets. Only those days will be replanned."
    ),
    input_schema=ScopeRevisionArgs.model_json_schema(),
)


DATA_TOOL_ARGS: dict[str, type[DataToolArgs]] = {
    "search_flights": SearchFlightsArgs,
    "search_hotels": SearchHotelsArgs,
    "search_places": SearchPlacesArgs,
    "get_travel_times": GetTravelTimesArgs,
}
CONTROL_TOOL_ARGS: dict[str, type[BaseModel]] = {
    "ask_user": AskUserArgs,
    "propose_plan": ProposePlanArgs,
    "give_up": GiveUpArgs,
}
DATA_TOOLS = frozenset(DATA_TOOL_ARGS)
CONTROL_TOOLS = frozenset(CONTROL_TOOL_ARGS)
# BR-01: priced, date-specific searches need the full set of required fields.
PRICED_TOOLS = frozenset({"search_flights", "search_hotels"})

_DESCRIPTIONS = {
    "search_flights": (
        "Search flight offers for one route and date. Returns offers with offer_id, airline, "
        "local depart_time and arrive_time, stops, price (total for all travellers), currency "
        "and booking_link. Search each direction as its own one-way call."
    ),
    "search_hotels": (
        "Search hotels in a city for a stay. Returns hotel_id, name, lat, lon, rating, "
        "price_per_night (per room), currency, check_in_time, taxes_included and booking_link."
    ),
    "search_places": (
        "Search attractions and activities in a city. Returns place_id, name, lat, lon, "
        "category, typical_duration_min, opening_hours per weekday, closed_dates, entry_price "
        "(per person) and currency. Needs only a destination."
    ),
    "get_travel_times": (
        "Travel time between up to 25 points. Returns matrix_minutes[from][to] and, for transit "
        "and taxi, estimated_cost[from][to], indexed in the order of `points`."
    ),
    "ask_user": (
        "Ask the user for TripRequest fields through one form. Include every field you need "
        "in a single call and say why."
    ),
    "propose_plan": (
        "Submit the day-wise plan for verification. Every item except allowances must carry "
        "a ref to the tool result (call_id) and the offer/place id it came from."
    ),
    "give_up": "Stop when no useful plan is possible. Say what blocked it and what could change.",
}


def tool_specs(names: list[str] | None = None) -> list[ToolSpec]:
    """Tool definitions for native tool calling, in a stable order."""
    ordered = list(DATA_TOOL_ARGS) + list(CONTROL_TOOL_ARGS)
    models: dict[str, type[BaseModel]] = {**DATA_TOOL_ARGS, **CONTROL_TOOL_ARGS}
    return [
        ToolSpec(
            name=name,
            description=_DESCRIPTIONS[name],
            input_schema=models[name].model_json_schema(),
        )
        for name in ordered
        if names is None or name in names
    ]


def validation_message(exc: Any) -> str:
    """Compact, LLM-actionable rendering of a pydantic ValidationError."""
    parts = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "arguments"
        parts.append(f"{loc}: {err['msg']}")
    return "; ".join(parts)
