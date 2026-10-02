"""BR-04: every tool call is validated against these schemas before it runs."""

from datetime import date

import pytest
from pydantic import ValidationError

from trip_agent.tools.schemas import (
    CONTROL_TOOLS,
    DATA_TOOLS,
    PRICED_TOOLS,
    AskUserArgs,
    GetTravelTimesArgs,
    GiveUpArgs,
    ProposePlanArgs,
    RecordRequestArgs,
    SearchFlightsArgs,
    SearchHotelsArgs,
    SearchPlacesArgs,
    tool_specs,
    validation_message,
)

TODAY = {"today": date(2026, 10, 1)}
FLIGHT = dict(origin="LHR", destination="VIE", depart_date="2026-11-07", adults=2)
HOTEL = dict(city="Vienna", check_in="2026-11-07", check_out="2026-11-11", guests=2)
POINTS = [{"id": "a", "lat": 48.2, "lon": 16.37}, {"id": "b", "lat": 48.21, "lon": 16.36}]


def test_flight_defaults():
    args = SearchFlightsArgs.model_validate(FLIGHT, context=TODAY)
    assert args.children == 0 and args.cabin == "economy"
    assert args.return_date is None and args.depends_on == []


@pytest.mark.parametrize("code", ["Austria", "vie", "VI", "VIEN", ""])
def test_flight_rejects_non_iata_codes(code):
    with pytest.raises(ValidationError) as exc:
        SearchFlightsArgs.model_validate({**FLIGHT, "destination": code}, context=TODAY)
    assert "destination" in validation_message(exc.value)


def test_flight_rejects_departure_in_the_past():
    with pytest.raises(ValidationError) as exc:
        SearchFlightsArgs.model_validate({**FLIGHT, "depart_date": "2026-09-30"}, context=TODAY)
    assert "today (2026-10-01) or later" in validation_message(exc.value)
    SearchFlightsArgs.model_validate({**FLIGHT, "depart_date": "2026-10-01"}, context=TODAY)  # today is fine


def test_flight_rejects_return_before_departure():
    with pytest.raises(ValidationError):
        SearchFlightsArgs.model_validate({**FLIGHT, "return_date": "2026-11-06"}, context=TODAY)
    SearchFlightsArgs.model_validate({**FLIGHT, "return_date": "2026-11-07"}, context=TODAY)  # same day ok


@pytest.mark.parametrize(
    "field, value",
    [("adults", 0), ("adults", 10), ("children", 9), ("children", -1), ("cabin", "luxury"), ("max_price", 0)],
)
def test_flight_rejects_out_of_range_arguments(field, value):
    with pytest.raises(ValidationError):
        SearchFlightsArgs.model_validate({**FLIGHT, field: value}, context=TODAY)


def test_data_tools_reject_unknown_arguments():
    with pytest.raises(ValidationError):
        SearchFlightsArgs.model_validate({**FLIGHT, "api_key": "x"}, context=TODAY)


def test_hotel_defaults_and_date_order():
    assert SearchHotelsArgs.model_validate(HOTEL).rooms == 1
    with pytest.raises(ValidationError):
        SearchHotelsArgs.model_validate({**HOTEL, "check_out": "2026-11-07"})  # must be after check_in


@pytest.mark.parametrize(
    "field, value",
    [("guests", 0), ("guests", 11), ("rooms", 6), ("min_rating", 0.5), ("min_rating", 5.5), ("max_price_per_night", -5)],
)
def test_hotel_rejects_out_of_range_arguments(field, value):
    with pytest.raises(ValidationError):
        SearchHotelsArgs.model_validate({**HOTEL, field: value})


def test_hotel_check_in_may_be_left_for_the_system_to_derive():
    args = SearchHotelsArgs.model_validate(
        {"city": "Vienna", "check_out": "2026-11-11", "guests": 2, "depends_on": ["outbound"]}
    )
    assert args.check_in is None


def test_places_limit():
    assert SearchPlacesArgs.model_validate({"city": "Vienna"}).limit == 20
    SearchPlacesArgs.model_validate({"city": "Vienna", "limit": 30})
    with pytest.raises(ValidationError):
        SearchPlacesArgs.model_validate({"city": "Vienna", "limit": 31})
    with pytest.raises(ValidationError):
        SearchPlacesArgs.model_validate({"city": ""})


def test_travel_times_point_limits():
    GetTravelTimesArgs.model_validate({"points": POINTS, "mode": "walk"})
    many = [{"id": str(i), "lat": 48.0, "lon": 16.0} for i in range(26)]
    with pytest.raises(ValidationError):
        GetTravelTimesArgs.model_validate({"points": many, "mode": "walk"})
    with pytest.raises(ValidationError):
        GetTravelTimesArgs.model_validate({"points": POINTS[:1], "mode": "walk"})
    with pytest.raises(ValidationError):
        GetTravelTimesArgs.model_validate({"points": POINTS, "mode": "bike"})
    with pytest.raises(ValidationError):
        GetTravelTimesArgs.model_validate({"points": [POINTS[0], POINTS[0]], "mode": "walk"})  # duplicate id


def test_ask_user_only_accepts_trip_request_fields():
    args = AskUserArgs.model_validate({"fields": ["origin", "adults", "origin"], "reason": "need them"})
    assert args.fields == ["origin", "adults"]  # de-duplicated, order kept
    with pytest.raises(ValidationError) as exc:
        AskUserArgs.model_validate({"fields": ["passport_number"], "reason": "x"})
    assert "passport_number" in validation_message(exc.value)
    with pytest.raises(ValidationError):
        AskUserArgs.model_validate({"fields": [], "reason": "x"})
    with pytest.raises(ValidationError):
        AskUserArgs.model_validate({"fields": ["raw_text"], "reason": "x"})


def test_propose_plan_must_match_the_plan_schema():
    ProposePlanArgs.model_validate({"plan": {"days": [{"day": 1, "date": "2026-11-07", "items": []}]}})
    with pytest.raises(ValidationError):
        ProposePlanArgs.model_validate({"plan": {"days": [{"day": 1, "items": []}]}})  # no date
    bad_item = {"id": "x", "type": "spaceship", "day": 1}
    with pytest.raises(ValidationError):
        ProposePlanArgs.model_validate({"plan": {"days": [{"day": 1, "date": "2026-11-07", "items": [bad_item]}]}})


def test_give_up_needs_a_reason():
    assert GiveUpArgs.model_validate({"reason": "no flights"}).suggestions == []
    with pytest.raises(ValidationError):
        GiveUpArgs.model_validate({"reason": ""})


def test_record_request_is_validated_like_any_tool():
    args = RecordRequestArgs.model_validate({"destination_city": "Vienna", "adults": 2})
    assert args.origin is None and args.guessed_fields == []
    with pytest.raises(ValidationError):
        RecordRequestArgs.model_validate({"adults": 40})
    with pytest.raises(ValidationError):
        RecordRequestArgs.model_validate({"passport": "X123"})


def test_tool_registry():
    assert DATA_TOOLS == {"search_flights", "search_hotels", "search_places", "get_travel_times"}
    assert CONTROL_TOOLS == {"ask_user", "propose_plan", "give_up"}
    assert PRICED_TOOLS == {"search_flights", "search_hotels"}


def test_tool_specs_expose_seven_tools_with_json_schemas():
    specs = tool_specs()
    assert [s.name for s in specs] == [
        "search_flights", "search_hotels", "search_places", "get_travel_times",
        "ask_user", "propose_plan", "give_up",
    ]  # fmt: skip
    for spec in specs:
        assert spec.description
        assert spec.input_schema["type"] == "object"
    flights = specs[0].input_schema
    assert {"origin", "destination", "depart_date", "adults"} <= set(flights["required"])
    assert [s.name for s in tool_specs(["propose_plan", "give_up"])] == ["propose_plan", "give_up"]


def test_no_tool_schema_asks_for_credentials():
    """BR-03: nothing in what the LLM sees has a slot for a key or token."""
    for spec in tool_specs():
        text = str(spec.input_schema).lower()
        assert not any(word in text for word in ("api_key", "token", "password", "secret"))
