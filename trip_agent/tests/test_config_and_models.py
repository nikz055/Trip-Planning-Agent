from datetime import date, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from trip_agent.config import FakeClock, Settings
from trip_agent.llm.base import AssistantTurn, LLMToolCall, estimate_tokens
from trip_agent.models import (
    ASKABLE_FIELDS,
    REQUIRED_FIELDS,
    CostStatus,
    ItemType,
    Pace,
    Plan,
    PlanDay,
    PlanItem,
    Session,
    State,
    TERMINAL_STATES,
    TripRequest,
    missing_required_fields,
)
from trip_agent.orchestrator.rules import request_errors
from trip_agent.tests.conftest import new_session, trip_request

# -- settings and clock ------------------------------------------------------


def test_defaults_match_the_business_rules():
    s = Settings()
    assert s.max_iterations == 8  # BR-06
    assert s.wall_clock_s == 120  # BR-06
    assert s.tool_retries == 2  # BR-07
    assert s.freshness_minutes == 60  # BR-10
    assert (s.min_trip_days, s.max_trip_days) == (1, 14)  # BR-02
    assert s.buffer_minutes == 30  # BR-13
    assert s.pace_caps == {"relaxed": 3, "balanced": 4, "packed": 6}  # BR-13


def test_settings_read_overrides_from_environment(monkeypatch):
    monkeypatch.setenv("TRIP_AGENT_MAX_ITERATIONS", "5")
    monkeypatch.setenv("TRIP_AGENT_COST_CAP_USD", "0.75")
    s = Settings.from_env()
    assert s.max_iterations == 5 and s.cost_cap_usd == 0.75
    assert Settings.from_env(max_iterations=3).max_iterations == 3  # explicit beats env


def test_budget_conversion_to_plan_currency():
    s = Settings()
    assert s.to_plan_currency(1000, "EUR") == 1000
    assert s.to_plan_currency(1000, "usd") == 920
    assert s.to_plan_currency(None, "EUR") is None
    assert s.to_plan_currency(1000, "XYZ") is None  # unknown currency is not guessed


def test_fake_clock_is_frozen_until_advanced():
    clock = FakeClock()
    start = clock.now()
    assert clock.now() == start and clock.today() == date(2026, 10, 1)
    clock.advance(minutes=61)
    assert clock.now() - start == timedelta(minutes=61)


# -- trip request ------------------------------------------------------------


def test_trip_request_defaults():
    r = TripRequest(raw_text="x")
    assert r.dates_flexible is False
    assert r.pace is Pace.relaxed
    assert r.children == 0
    assert r.currency == "EUR"
    assert r.interests == []


def test_missing_required_fields_lists_every_gap():
    assert missing_required_fields(TripRequest(raw_text="x")) == list(REQUIRED_FIELDS)
    assert missing_required_fields(trip_request()) == []
    assert missing_required_fields(trip_request(origin=None, adults=None)) == ["origin", "adults"]


def test_trip_request_normalises_text_fields():
    r = trip_request(origin="  London ", destination_city="   ", currency=" usd ")
    assert r.origin == "London"
    assert r.destination_city is None  # blank is treated as missing
    assert r.currency == "USD"


@pytest.mark.parametrize(
    "field, value",
    [("adults", 0), ("adults", 10), ("children", 9), ("total_budget", 0), ("min_hotel_rating", 6)],
)
def test_trip_request_rejects_out_of_range_values(field, value):
    with pytest.raises(ValidationError):
        trip_request(**{field: value})


def test_trip_request_derived_values():
    r = trip_request(children=1)
    assert r.travellers == 3
    assert r.trip_days == 5
    assert TripRequest(raw_text="x").trip_days is None


def test_askable_fields_exclude_raw_text():
    assert "raw_text" not in ASKABLE_FIELDS
    assert set(REQUIRED_FIELDS) <= set(ASKABLE_FIELDS)


# -- BR-02 request rules -----------------------------------------------------


def test_request_errors_accepts_a_valid_request():
    assert request_errors(trip_request(), date(2026, 10, 1), Settings()) == {}


def test_request_errors_flags_missing_fields():
    errors = request_errors(TripRequest(raw_text="x"), date(2026, 10, 1), Settings())
    assert set(errors) == set(REQUIRED_FIELDS)


def test_request_errors_flags_past_and_reversed_dates():
    today = date(2026, 10, 1)
    past = request_errors(trip_request(start_date=date(2026, 9, 1), end_date=date(2026, 9, 5)), today, Settings())
    assert "start_date" in past
    reversed_ = request_errors(trip_request(start_date=date(2026, 11, 7), end_date=date(2026, 11, 5)), today, Settings())
    assert "end_date" in reversed_


def test_request_errors_flags_trips_longer_than_14_days():
    errors = request_errors(trip_request(end_date=date(2026, 11, 25)), date(2026, 10, 1), Settings())
    assert "1 to 14 days" in errors["end_date"]
    one_day = trip_request(end_date=date(2026, 11, 7))
    assert request_errors(one_day, date(2026, 10, 1), Settings()) == {}


def test_request_errors_flags_unsupported_budget_currency():
    errors = request_errors(trip_request(currency="JPY"), date(2026, 10, 1), Settings())
    assert "currency" in errors


# -- plan --------------------------------------------------------------------


def _item(**overrides) -> PlanItem:
    fields = dict(id="a", type=ItemType.activity, title="A", day=1, cost=10, cost_status=CostStatus.confirmed)
    fields.update(overrides)
    return PlanItem(**fields)


def test_plan_item_times_are_local_and_drop_offsets():
    aware = datetime(2026, 11, 7, 9, 0, tzinfo=timezone.utc)
    item = _item(start=aware, end="2026-11-07T10:30:00+01:00")
    assert item.start == datetime(2026, 11, 7, 9, 0) and item.start.tzinfo is None
    assert item.end == datetime(2026, 11, 7, 10, 30)


def test_plan_item_rejects_negative_cost_and_day_zero():
    with pytest.raises(ValidationError):
        _item(cost=-1)
    with pytest.raises(ValidationError):
        _item(day=0)


def test_plan_iterates_and_finds_items():
    plan = Plan(
        days=[
            PlanDay(day=1, date=date(2026, 11, 7), items=[_item(id="a"), _item(id="b")]),
            PlanDay(day=2, date=date(2026, 11, 8), items=[_item(id="c", day=2)]),
        ]
    )
    assert [i.id for i in plan.items()] == ["a", "b", "c"]
    assert plan.item("c").day == 2
    assert plan.item("zzz") is None


def test_terminal_states():
    assert TERMINAL_STATES == {State.ACCEPTED, State.FAILED, State.CANCELLED}  # PARTIAL can be revised


def test_session_survives_a_json_round_trip(clock):
    session = new_session(trip_request(), clock)
    session.add_version(Plan(days=[PlanDay(day=1, date=date(2026, 11, 7), items=[_item()])]), "initial", None, clock.now())
    session.current_version = 1
    restored = Session.model_validate_json(session.model_dump_json())
    assert restored == session
    assert restored.plan.item("a").cost == 10


def test_versions_are_appended_and_the_pointer_picks_the_shown_plan(clock):
    session = new_session(trip_request(), clock)
    assert session.plan is None
    first = session.add_version(Plan(assumptions=["one"]), "initial", None, clock.now())
    second = session.add_version(Plan(assumptions=["two"]), "repair", first.version, clock.now())
    assert (first.version, second.version, second.parent_version) == (1, 2, 1)
    assert session.plan is None  # nothing is shown until the pointer moves
    session.current_version = 2
    assert session.plan.assumptions == ["two"]
    assert session.version(1).plan.assumptions == ["one"]  # the earlier version is still there


# -- llm base ----------------------------------------------------------------


def test_assistant_turn_becomes_a_message():
    turn = AssistantTurn(text="hi", tool_calls=[LLMToolCall(id="1", name="give_up", args={})], raw=[{"x": 1}])
    message = turn.as_message()
    assert message.role == "assistant" and message.raw == [{"x": 1}]
    assert message.tool_calls[0].name == "give_up"


def test_estimate_tokens_ignores_missing_parts():
    assert estimate_tokens("a" * 40, None, "b" * 40) == 20
