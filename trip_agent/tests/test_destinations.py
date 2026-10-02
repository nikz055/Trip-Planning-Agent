"""Trips to every destination in the sample catalogue, from airports that only
have generated flights, and the message for a destination with no data."""

import pytest

from trip_agent.tests.conftest import items_of


@pytest.mark.parametrize(
    "request_text, origin, destination",
    [
        ("4 days in Jaipur from Bengaluru, 12-15 Nov 2026, 2 adults, budget 1500 EUR", "BLR", "Jaipur"),
        ("Goa from Chennai, 20-24 Nov 2026, 2 adults and 1 child", "MAA", "Goa"),
        ("Lisbon from New York, 3-8 Nov 2026, solo", "JFK", "Lisbon"),
        ("5 days in Vienna from Singapore starting 16 November 2026, 2 adults", "SIN", "Vienna"),
        ("Jaipur from Delhi, 2026-12-04 to 2026-12-06, the two of us, packed", "DEL", "Jaipur"),
    ],
)
async def test_a_clean_plan_for_each_destination(orchestrator, provider, request_text, origin, destination):
    view = await orchestrator.create_trip(request_text)
    assert view["state"] == "PRESENTED", (view["form"], view["give_up"], view["plan"] and view["plan"]["violations"])
    assert view["plan"]["violations"] == [] and view["plan"]["gaps"] == []
    assert view["request"]["destination_city"] == destination
    flight_searches = [args for tool, args in provider.call_log if tool == "search_flights"]
    assert flight_searches[0]["origin"] == origin
    assert len(items_of(view, "flight")) == 2 and len(items_of(view, "hotel")) == 1
    assert len(items_of(view, "activity")) >= 3


async def test_destination_without_sample_data_says_so(orchestrator, provider):
    view = await orchestrator.create_trip("5 days in Paris from Delhi, 7-11 Nov 2026, 2 adults")
    assert view["state"] == "FAILED" and view["plan"] is None
    assert "no hotel or attraction data for Paris" in view["give_up"]["reason"]
    assert view["give_up"]["suggestions"] == ["Destinations with sample data: Vienna, Lisbon, Jaipur, Goa"]
    assert provider.call_log == []  # nothing was searched for a trip that cannot be planned


async def test_unknown_departure_city_says_so(orchestrator):
    view = await orchestrator.create_trip("Vienna from Atlantis, 7-11 Nov 2026, 2 adults")
    assert view["state"] == "FAILED" and "don't know an airport for 'Atlantis'" in view["give_up"]["reason"]
