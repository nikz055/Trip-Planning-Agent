"""The mock catalogue and the four mock data tools (spec section 5)."""

import json
import math
import time
from datetime import date, datetime

import pytest

from trip_agent.data.mock import build_mock_data
from trip_agent.tools.base import ToolError, ToolTimeout
from trip_agent.tools.failures import FailureInjector, FailureSpec
from trip_agent.tools.mock import (
    DATA_DIR,
    MODE_OVERHEAD_MIN,
    MODE_SPEED_KMH,
    DETOUR_FACTOR,
    MockToolProvider,
    haversine_km,
)
from trip_agent.tools.schemas import (
    GetTravelTimesArgs,
    SearchFlightsArgs,
    SearchHotelsArgs,
    SearchPlacesArgs,
)

SAT = date(2026, 11, 7)


def flights(**kw) -> SearchFlightsArgs:
    return SearchFlightsArgs(**{"origin": "LHR", "destination": "VIE", "depart_date": SAT, "adults": 1, **kw})


def hotels(**kw) -> SearchHotelsArgs:
    return SearchHotelsArgs(**{"city": "Vienna", "check_in": date(2026, 11, 9), "check_out": date(2026, 11, 12), "guests": 2, **kw})


# -- the catalogue -----------------------------------------------------------


def test_committed_json_matches_the_builder():
    """The JSON files are generated; they must not drift from the builder."""
    for name, build in (
        ("flights", build_mock_data.build_flights),
        ("hotels", build_mock_data.build_hotels),
        ("places", build_mock_data.build_places),
    ):
        on_disk = json.loads((DATA_DIR / f"{name}.json").read_text(encoding="utf-8"))
        assert on_disk == json.loads(json.dumps(build())), name


CITIES = ("Vienna", "Lisbon", "Jaipur", "Goa")


def test_catalogue_destinations():
    hotels_data = build_mock_data.build_hotels()
    places_data = build_mock_data.build_places()
    assert set(hotels_data) == set(places_data) == set(CITIES)
    ids = [p["place_id"] for rows in places_data.values() for p in rows] + [h["hotel_id"] for rows in hotels_data.values() for h in rows]
    assert len(ids) == len(set(ids))
    for city in CITIES:
        assert len(hotels_data[city]) == 15
        assert len(places_data[city]) == 30
        for place in places_data[city]:
            assert set(place["opening_hours"]) == {"mon", "tue", "wed", "thu", "fri", "sat", "sun"}
            assert -90 <= place["lat"] <= 90 and -180 <= place["lon"] <= 180
    routes = build_mock_data.build_flights()["routes"]
    assert len(routes) == 12
    assert all(len(offers) == 20 for offers in routes.values())


def test_catalogue_has_some_closures():
    places = [p for rows in build_mock_data.build_places().values() for p in rows]
    assert sum(1 for p in places if p["closed_dates"]) >= 3
    assert sum(1 for p in places if None in p["opening_hours"].values()) >= 8


def test_exactly_one_overnight_flight_into_a_destination():
    routes = build_mock_data.build_flights()["routes"]
    overnight = [
        (route, t["template_id"])
        for route, templates in routes.items()
        if route.endswith(("-VIE", "-LIS"))
        for t in templates
        if t["arrive_day_offset"] > 0
    ]
    assert overnight == [("DEL-VIE", "DELVIE20")]
    # and it is the cheapest on its route, so a price-led planner will pick it
    assert min(routes["DEL-VIE"], key=lambda t: t["base_price"])["template_id"] == "DELVIE20"


def test_exactly_one_place_closed_only_on_mondays():
    places = [p for rows in build_mock_data.build_places().values() for p in rows]
    monday_only = [
        p["place_id"]
        for p in places
        if p["opening_hours"]["mon"] is None
        and all(span for day, span in p["opening_hours"].items() if day != "mon")
    ]
    assert monday_only == ["VIE-P05"]


def test_one_hotel_with_a_late_check_in():
    late = [h["hotel_id"] for rows in build_mock_data.build_hotels().values() for h in rows if h["check_in_time"] >= "20:00"]
    assert late == ["VIE-H10"]


def test_one_place_description_contains_an_instruction():
    places = [p for rows in build_mock_data.build_places().values() for p in rows]
    injected = [p["place_id"] for p in places if "ignore your previous instructions" in p["description"].lower()]
    assert injected == ["VIE-P17"]


# -- search_flights ----------------------------------------------------------


async def test_flights_return_about_twenty_offers_in_the_spec_shape():
    offers = await MockToolProvider().call("search_flights", flights())
    assert len(offers) == 20
    assert set(offers[0]) == {"offer_id", "airline", "depart_time", "arrive_time", "stops", "price", "currency", "booking_link"}
    assert offers == sorted(offers, key=lambda o: (o["price"], o["depart_time"]))
    assert all(o["currency"] == "EUR" and o["booking_link"].startswith("https://") for o in offers)
    assert all(datetime.fromisoformat(o["depart_time"]).date() == SAT for o in offers)


async def test_flight_prices_scale_with_party_and_cabin():
    provider = MockToolProvider()
    one = {o["offer_id"]: o["price"] for o in await provider.call("search_flights", flights())}
    family = {o["offer_id"]: o["price"] for o in await provider.call("search_flights", flights(adults=2, children=2))}
    business = {o["offer_id"]: o["price"] for o in await provider.call("search_flights", flights(cabin="business"))}
    for offer_id, price in one.items():
        assert family[offer_id] == pytest.approx(price * 3.5, abs=0.02)  # children fly at 75%
        assert business[offer_id] == pytest.approx(price * 3.0, abs=0.02)


async def test_flight_max_price_filters():
    provider = MockToolProvider()
    everything = await provider.call("search_flights", flights())
    limit = everything[4]["price"]
    capped = await provider.call("search_flights", flights(max_price=limit))
    assert capped and all(o["price"] <= limit for o in capped)
    assert len(capped) < len(everything)


async def test_route_date_with_no_flights():
    provider = MockToolProvider()
    assert await provider.call("search_flights", flights(depart_date=date(2026, 12, 8))) == []
    assert len(await provider.call("search_flights", flights(depart_date=date(2026, 12, 7)))) == 20
    back = flights(origin="VIE", destination="LHR", depart_date=date(2026, 12, 8))
    assert len(await provider.call("search_flights", back)) == 20  # only one direction is affected


async def test_unknown_airport_returns_no_offers():
    provider = MockToolProvider()
    assert await provider.call("search_flights", flights(origin="ZZZ")) == []
    assert await provider.call("search_flights", flights(origin="VIE", destination="VIE")) == []


def test_airport_table():
    airports = build_mock_data.build_airports()
    codes = [a["iata"] for a in airports]
    assert len(codes) == len(set(codes)) >= 120
    assert {"DEL", "BOM", "BLR", "MAA", "JAI", "GOI", "LHR", "JFK", "SIN", "SYD", "VIE", "LIS"} <= set(codes)
    for a in airports:
        assert len(a["iata"]) == 3 and -90 <= a["lat"] <= 90 and -180 <= a["lon"] <= 180 and -12 <= a["utc_offset"] <= 14
    assert json.loads((DATA_DIR / "airports.json").read_text(encoding="utf-8")) == json.loads(json.dumps(airports))


@pytest.mark.parametrize("origin, destination", [("BLR", "JAI"), ("JFK", "VIE"), ("SYD", "LIS"), ("NRT", "LAX"), ("MAA", "GOI")])
async def test_any_pair_of_known_airports_gets_generated_offers(origin, destination):
    provider = MockToolProvider()
    offers = await provider.call("search_flights", flights(origin=origin, destination=destination))
    assert len(offers) == 20 and offers == await provider.call("search_flights", flights(origin=origin, destination=destination))
    for offer in offers:
        depart, arrive = datetime.fromisoformat(offer["depart_time"]), datetime.fromisoformat(offer["arrive_time"])
        assert depart.date() == SAT and arrive > depart  # never lands before it leaves, even across the date line
        assert offer["price"] > 0 and offer["stops"] in (0, 1)


async def test_generated_prices_grow_with_distance():
    provider = MockToolProvider()
    short = await provider.call("search_flights", flights(origin="BOM", destination="GOI"))
    long = await provider.call("search_flights", flights(origin="BOM", destination="JFK"))
    assert max(o["price"] for o in short) < min(o["price"] for o in long)


async def test_hand_tuned_routes_are_not_replaced_by_generated_ones():
    offers = await MockToolProvider().call("search_flights", flights(origin="DEL"))
    assert offers[0]["offer_id"].startswith("DELVIE20")  # the overnight flight is still there


async def test_overnight_flight_lands_the_next_day():
    offers = await MockToolProvider().call("search_flights", flights(origin="DEL"))
    red_eye = offers[0]  # cheapest
    assert red_eye["depart_time"] == "2026-11-07T23:55"
    assert red_eye["arrive_time"] == "2026-11-08T05:40"
    assert all(o["arrive_time"].startswith("2026-11-07") for o in offers[1:])


async def test_round_trip_search_prices_both_legs():
    offers = await MockToolProvider().call("search_flights", flights(return_date=date(2026, 11, 11)))
    assert len(offers) == 20
    assert all(o["offer_id"].startswith("RT-") for o in offers)
    assert all(o["return_depart_time"].startswith("2026-11-11") for o in offers)


# -- search_hotels -----------------------------------------------------------


async def test_hotels_return_fifteen_in_the_spec_shape():
    found = await MockToolProvider().call("search_hotels", hotels())
    assert len(found) == 15
    assert set(found[0]) == {
        "hotel_id", "name", "lat", "lon", "rating", "price_per_night", "currency",
        "check_in_time", "taxes_included", "booking_link",
    }  # fmt: skip
    assert found == sorted(found, key=lambda h: (h["price_per_night"], -h["rating"]))


async def test_hotel_filters():
    provider = MockToolProvider()
    rated = await provider.call("search_hotels", hotels(min_rating=4.5))
    assert rated and all(h["rating"] >= 4.5 for h in rated)
    cheap = await provider.call("search_hotels", hotels(max_price_per_night=80))
    assert cheap and all(h["price_per_night"] <= 80 for h in cheap)
    assert await provider.call("search_hotels", hotels(min_rating=5, max_price_per_night=100)) == []
    # six guests in one room fit nowhere; three rooms fit everywhere
    assert await provider.call("search_hotels", hotels(guests=6, rooms=1)) == []
    assert len(await provider.call("search_hotels", hotels(guests=6, rooms=3))) == 15


async def test_hotel_city_is_case_insensitive_and_unknown_city_is_empty():
    provider = MockToolProvider()
    assert len(await provider.call("search_hotels", hotels(city="vienna"))) == 15
    assert await provider.call("search_hotels", hotels(city="Atlantis")) == []


# -- search_places -----------------------------------------------------------


async def test_places_respect_limit_and_shape():
    provider = MockToolProvider()
    assert len(await provider.call("search_places", SearchPlacesArgs(city="Vienna"))) == 20
    everything = await provider.call("search_places", SearchPlacesArgs(city="Vienna", limit=30))
    assert len(everything) == 30
    assert {"place_id", "name", "lat", "lon", "category", "typical_duration_min", "opening_hours", "closed_dates", "entry_price", "currency"} <= set(everything[0])


async def test_places_put_matching_interests_first():
    found = await MockToolProvider().call("search_places", SearchPlacesArgs(city="Vienna", interests=["Music"], limit=5))
    assert all("music" in p["name"].lower() or p["category"] == "music" or p["place_id"] in {"VIE-P14", "VIE-P25"} for p in found)
    assert found[0]["place_id"] == "VIE-P10"  # first music match in catalogue order


# -- get_travel_times --------------------------------------------------------

A = {"id": "a", "lat": 48.2085, "lon": 16.3731}
B = {"id": "b", "lat": 48.1845, "lon": 16.3122}


async def test_travel_time_is_distance_over_speed_plus_overhead():
    provider = MockToolProvider()
    km = haversine_km(A["lat"], A["lon"], B["lat"], B["lon"]) * DETOUR_FACTOR
    for mode in ("walk", "transit", "taxi"):
        result = await provider.call("get_travel_times", GetTravelTimesArgs(points=[A, B], mode=mode))
        expected = math.ceil(km / MODE_SPEED_KMH[mode] * 60) + MODE_OVERHEAD_MIN[mode]
        assert result["matrix_minutes"] == [[0, expected], [expected, 0]]
        assert result["point_ids"] == ["a", "b"]


async def test_travel_cost_only_for_transit_and_taxi():
    provider = MockToolProvider()
    walk = await provider.call("get_travel_times", GetTravelTimesArgs(points=[A, B], mode="walk"))
    transit = await provider.call("get_travel_times", GetTravelTimesArgs(points=[A, B], mode="transit"))
    taxi = await provider.call("get_travel_times", GetTravelTimesArgs(points=[A, B], mode="taxi"))
    assert walk["estimated_cost"] is None
    assert transit["estimated_cost"] == [[0.0, 2.4], [2.4, 0.0]]
    assert taxi["estimated_cost"][0][1] > 4.0 and taxi["estimated_cost"][0][0] == 0.0


def test_haversine_known_distance():
    # Vienna to Lisbon is roughly 2,300 km
    assert 2250 < haversine_km(48.2082, 16.3738, 38.7223, -9.1393) < 2350


# -- failure injection -------------------------------------------------------


async def test_injected_timeout_and_error_raise():
    timeout = MockToolProvider(FailureInjector({"search_flights": {"mode": "timeout"}}))
    with pytest.raises(ToolTimeout):
        await timeout.call("search_flights", flights())
    error = MockToolProvider(FailureInjector({"search_hotels": {"mode": "error"}}))
    with pytest.raises(ToolError):
        await error.call("search_hotels", hotels())
    assert len(await error.call("search_flights", flights())) == 20  # other tools unaffected


async def test_injected_empty_result():
    provider = MockToolProvider(FailureInjector({"search_places": {"mode": "empty"}, "get_travel_times": {"mode": "empty"}}))
    assert await provider.call("search_places", SearchPlacesArgs(city="Vienna")) == []
    travel = await provider.call("get_travel_times", GetTravelTimesArgs(points=[A, B], mode="walk"))
    assert travel["matrix_minutes"] == []


async def test_injected_slow_response_still_answers():
    provider = MockToolProvider(FailureInjector({"search_flights": FailureSpec(mode="slow", delay_s=0.15)}))
    started = time.perf_counter()
    offers = await provider.call("search_flights", flights())
    assert time.perf_counter() - started >= 0.14
    assert len(offers) == 20


async def test_failure_applies_a_limited_number_of_times():
    provider = MockToolProvider(FailureInjector({"search_flights": {"mode": "error", "times": 2}}))
    for _ in range(2):
        with pytest.raises(ToolError):
            await provider.call("search_flights", flights())
    assert len(await provider.call("search_flights", flights())) == 20


def test_failure_injector_can_be_changed_at_runtime():
    injector = FailureInjector()
    assert injector.next("search_flights") is None
    injector.set("search_flights", {"mode": "timeout"})
    assert injector.next("search_flights").mode == "timeout"
    injector.set("search_flights", None)
    assert injector.next("search_flights") is None


async def test_price_drift_changes_prices_between_fetches():
    provider = MockToolProvider()
    before = (await provider.call("search_hotels", hotels()))[0]["price_per_night"]
    provider.price_drift["search_hotels"] = 1.2
    after = (await provider.call("search_hotels", hotels()))[0]["price_per_night"]
    assert after == pytest.approx(before * 1.2, abs=0.01)


async def test_every_call_is_logged():
    provider = MockToolProvider()
    await provider.call("search_places", SearchPlacesArgs(city="Vienna"))
    assert provider.call_log == [("search_places", {"label": None, "depends_on": [], "city": "Vienna", "interests": None, "limit": 20})]
