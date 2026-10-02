"""The intake helper step and the stand-in's request parser."""

from datetime import date

import pytest

from trip_agent.llm.scripted import ScriptedLLM, call
from trip_agent.llm.standin import StandInPlanner
from trip_agent.llm.standin_intake import parse_request
from trip_agent.orchestrator.intake import parse_trip_request

TODAY = date(2026, 10, 1)


def test_parses_a_complete_request():
    got = parse_request("5 days in Vienna from London, 7-11 Nov 2026, 2 adults, budget 2500 EUR, we like art and music", TODAY)
    assert got == {
        "origin": "London",
        "destination_city": "Vienna",
        "start_date": "2026-11-07",
        "end_date": "2026-11-11",
        "adults": 2,
        "total_budget": 2500.0,
        "currency": "EUR",
        "interests": ["art", "music"],
    }


def test_country_becomes_a_guessed_city_and_nothing_else_is_invented():
    got = parse_request("Plan a trip to Austria", TODAY)
    assert got == {"destination_city": "Vienna", "guessed_fields": ["destination_city"]}


@pytest.mark.parametrize(
    "text, origin, destination",
    [
        ("A week in Lisbon from Mumbai", "Mumbai", "Lisbon"),
        ("Vienna from Delhi, 2 adults", "Delhi", "Vienna"),  # destination with no preposition
        ("Flying from London to Vienna in November", "London", "Vienna"),
        ("From Delhi, visiting Lisbon", "Delhi", "Lisbon"),
        ("I want to go somewhere nice", None, None),
    ],
)
def test_origin_and_destination_phrasings(text, origin, destination):
    got = parse_request(text, TODAY)
    assert (got.get("origin"), got.get("destination_city")) == (origin, destination)
    assert "guessed_fields" not in got


@pytest.mark.parametrize(
    "text, start, end",
    [
        ("Vienna 2026-11-07 to 2026-11-11", "2026-11-07", "2026-11-11"),
        ("Vienna from 7 Nov to 11 Nov 2026", "2026-11-07", "2026-11-11"),
        ("Vienna, Nov 7-11", "2026-11-07", "2026-11-11"),
        ("Vienna November 7th to November 11th, 2026", "2026-11-07", "2026-11-11"),
        ("Vienna for 5 days starting 7 November", "2026-11-07", "2026-11-11"),
        ("Vienna, 4 nights from 2026-11-07", "2026-11-07", "2026-11-11"),
        ("Lisbon 3 to 6 March", "2027-03-03", "2027-03-06"),  # next occurrence after today
        ("Vienna 28 Dec to 2 Jan", "2026-12-28", "2027-01-02"),  # range crossing new year
    ],
)
def test_date_phrasings(text, start, end):
    got = parse_request(text, TODAY)
    assert (got.get("start_date"), got.get("end_date")) == (start, end)


def test_a_single_date_without_a_duration_leaves_the_end_date_open():
    got = parse_request("Trip to Vienna on 7 November", TODAY)
    assert got["start_date"] == "2026-11-07" and "end_date" not in got


@pytest.mark.parametrize(
    "text, adults, children",
    [
        ("two adults and 2 kids to Vienna", 2, 2),
        ("Vienna with my wife", 2, None),
        ("solo trip to Lisbon", 1, None),
        ("the three of us are going to Vienna", 3, None),
        ("Vienna for 5 days", None, None),  # "5 days" is not a traveller count
    ],
)
def test_traveller_phrasings(text, adults, children):
    got = parse_request(text, TODAY)
    assert got.get("adults") == adults and got.get("children") == children


@pytest.mark.parametrize(
    "text, amount, currency",
    [
        ("budget €1,800", 1800.0, "EUR"),
        ("around $2k total", 2000.0, "USD"),
        ("we have 150000 rupees", 150000.0, "INR"),
        ("budget of 900", 900.0, None),
        ("£750 for everything", 750.0, "GBP"),
    ],
)
def test_budget_phrasings(text, amount, currency):
    got = parse_request(f"Vienna trip, {text}", TODAY)
    assert got.get("total_budget") == amount and got.get("currency") == currency


def test_preferences():
    got = parse_request("A packed trip to Vienna, 5-star hotel, my dates are flexible, into museums and coffee", TODAY)
    assert got["pace"] == "packed" and got["min_hotel_rating"] == 5.0 and got["dates_flexible"] is True
    assert got["interests"] == ["museums", "coffee"]
    assert "dates_flexible" not in parse_request("Vienna, dates are not flexible", TODAY)
    assert "art" not in parse_request("We start in Vienna", TODAY).get("interests", [])  # no match inside "start"


# -- the helper step ---------------------------------------------------------


async def test_intake_with_the_stand_in(clock):
    result = await parse_trip_request(StandInPlanner(), "Vienna from Delhi, 7-11 Nov 2026, 2 adults", clock)
    r = result.request
    assert (r.origin, r.destination_city, r.adults) == ("Delhi", "Vienna", 2)
    assert (r.start_date, r.end_date) == (date(2026, 11, 7), date(2026, 11, 11))
    assert r.raw_text == "Vienna from Delhi, 7-11 Nov 2026, 2 adults"
    assert result.llm_calls == 1 and result.tokens > 0  # counted toward the trip's limits


async def test_intake_reports_guessed_fields(clock):
    result = await parse_trip_request(StandInPlanner(), "Plan a trip to Austria", clock)
    assert result.request.destination_city == "Vienna"
    assert result.guessed_fields == ["destination_city"]


async def test_intake_output_is_schema_validated_and_retried_once(clock):
    llm = ScriptedLLM(
        [
            [call("record_request", destination_city="Vienna", adults=40)],  # out of range
            [call("record_request", destination_city="Vienna", adults=4)],
        ],
        cost_per_turn=0.01,
    )
    result = await parse_trip_request(llm, "Vienna, 40 of us", clock)
    assert result.request.adults == 4
    assert result.llm_calls == 2 and result.cost_usd == pytest.approx(0.02)
    retry_messages, _ = llm.seen[1]
    assert retry_messages[-1].tool_results[0].is_error
    assert "adults" in retry_messages[-1].tool_results[0].content


async def test_intake_falls_back_to_an_empty_request(clock):
    llm = ScriptedLLM([[], [call("record_request", adults="many")]])
    result = await parse_trip_request(llm, "something unclear", clock)
    assert result.request.destination_city is None and result.request.adults is None
    assert result.request.raw_text == "something unclear"
    assert result.llm_calls == 2


async def test_intake_only_offers_the_record_request_tool(clock):
    llm = ScriptedLLM([[call("record_request", destination_city="Vienna")]])
    await parse_trip_request(llm, "Vienna", clock)
    messages, offered = llm.seen[0]
    assert offered == ["record_request"]
    assert "Today is 2026-10-01." in messages[0].text and "<request>\nVienna\n</request>" in messages[0].text
