"""One test per edge case in spec section 9 (case 14 lives in test_revisions.py).

Each test runs against the in-memory store and against Postgres. The planner is
the stand-in or a scripted double, so these tests show what the orchestrator
and verifier do with a given planner behaviour, not how a real model behaves.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from datetime import timedelta

import pytest

from trip_agent.llm.base import LLMToolCall
from trip_agent.llm.scripted import ScriptedLLM, call
from trip_agent.llm.standin import StandInPlanner
from trip_agent.orchestrator.orchestrator import Orchestrator
from trip_agent.tests.conftest import VIENNA, MutatingPlanner, find_item, items_of
from trip_agent.tools.mock import DATA_DIR, MockToolProvider

COMPLETE = dict(origin="London", destination_city="Vienna", start_date="2026-11-07", end_date="2026-11-11", adults=2)
FLIGHT = dict(origin="LHR", destination="VIE", depart_date="2026-11-07", adults=2)
FORM = {"origin": "London", "start_date": "2026-11-07", "end_date": "2026-11-11", "adults": "2"}


def tools_called(provider) -> list[str]:
    return [tool for tool, _ in provider.call_log]


def events(session) -> list[str]:
    return [t.event for t in session.history]


class KeepsSearching(StandInPlanner):
    """Never proposes on its own: every would-be proposal becomes another search."""

    async def complete(self, *, system, messages, tools):
        turn = await super().complete(system=system, messages=messages, tools=tools)
        if any(t.name == "search_places" for t in tools):  # not the wrap-up turn
            for i, c in enumerate(turn.tool_calls):
                if c.name == "propose_plan":
                    turn.tool_calls[i] = LLMToolCall(id=c.id, name="search_places", args={"city": "Vienna", "limit": 3})
        return turn


class Slow(StandInPlanner):
    """Each planner turn takes `seconds` on the (fake) clock."""

    def __init__(self, clock, seconds: float):
        self._clock, self._seconds = clock, seconds

    async def complete(self, *, system, messages, tools):
        turn = await super().complete(system=system, messages=messages, tools=tools)
        if not any(t.name == "record_request" for t in tools):  # the intake step is instant
            self._clock.advance(minutes=self._seconds / 60)
        return turn


class Counting(StandInPlanner):
    def __init__(self):
        self.calls = 0

    async def complete(self, *, system, messages, tools):
        self.calls += 1
        await asyncio.sleep(0)  # give a concurrent duplicate request the chance to interleave
        return await super().complete(system=system, messages=messages, tools=tools)


# -- 1 -----------------------------------------------------------------------


async def test_01_destination_only_gives_one_form_and_no_priced_search(orchestrator, provider, store):
    view = await orchestrator.create_trip("Plan a trip to Austria")
    assert view["state"] == "WAITING_FOR_DETAILS" and view["plan"] is None
    names = [f["name"] for f in view["form"]["fields"]]
    assert names[:4] == ["origin", "start_date", "end_date", "adults"]  # every missing field, in one form
    assert "destination_city" in names  # "Austria" was turned into a guess, shown for confirmation
    guess = next(f for f in view["form"]["fields"] if f["name"] == "destination_city")
    assert guess["value"] == "Vienna"
    assert len(view["form"]["suggestions"]) == 5  # place suggestions are allowed with only a destination
    assert tools_called(provider) == ["search_places"]  # no flight or hotel search
    session = await store.load(view["trip_id"])
    assert session.stats.clarification_rounds == 1
    assert events(session) == ["request_incomplete"]


# -- 2 -----------------------------------------------------------------------


async def test_02_past_or_reversed_dates_give_field_errors_and_stay_waiting(orchestrator, provider):
    trip = (await orchestrator.create_trip("Plan a trip to Austria"))["trip_id"]
    searches = len(provider.call_log)

    past = await orchestrator.submit_form(trip, {**FORM, "start_date": "2026-09-01", "end_date": "2026-09-05"})
    assert past["state"] == "WAITING_FOR_DETAILS"
    errors = {f["name"]: f["error"] for f in past["form"]["fields"] if f["error"]}
    assert errors == {"start_date": "Start date must be today (2026-10-01) or later"}

    reversed_ = await orchestrator.submit_form(trip, {**FORM, "end_date": "2026-11-01"})
    assert reversed_["state"] == "WAITING_FOR_DETAILS"
    errors = {f["name"]: f["error"] for f in reversed_["form"]["fields"] if f["error"]}
    assert errors == {"end_date": "End date must be on or after the start date"}
    assert next(f for f in reversed_["form"]["fields"] if f["name"] == "end_date")["value"] == "2026-11-01"

    too_long = await orchestrator.submit_form(trip, {**FORM, "end_date": "2026-11-30"})
    assert "1 to 14 days" in next(f["error"] for f in too_long["form"]["fields"] if f["error"])
    assert len(provider.call_log) == searches  # nothing was searched on invalid input

    ok = await orchestrator.submit_form(trip, FORM)
    assert ok["state"] == "PRESENTED" and ok["trip_id"] == trip and ok["form"] is None


# -- 3 -----------------------------------------------------------------------


async def test_03_country_name_as_airport_code_is_rejected_then_retried(make_orchestrator, provider, store):
    llm = ScriptedLLM(
        [
            [call("record_request", **COMPLETE)],
            [call("search_flights", **{**FLIGHT, "destination": "Austria"})],
            [call("search_flights", **FLIGHT)],
            [call("give_up", reason="end of script")],
        ]
    )
    view = await make_orchestrator(llm).create_trip(VIENNA)
    assert view["state"] == "FAILED"
    after_invalid, _ = llm.seen[2]
    [error] = [json.loads(r.content) for r in after_invalid[-1].tool_results]
    assert error["status"] == "error" and "destination" in error["error"] and "not executed" in error["error"]
    assert after_invalid[-1].tool_results[0].is_error
    after_valid, _ = llm.seen[3]
    [ok] = [json.loads(r.content) for r in after_valid[-1].tool_results]
    assert ok["status"] == "ok" and len(ok["data"]) == 20
    assert tools_called(provider) == ["search_flights"]  # the invalid call never reached the tool
    session = await store.load(view["trip_id"])
    assert [(c.status, c.attempts) for c in session.tool_calls] == [("failed", 0), ("ok", 1)]
    assert session.iteration_count == 3  # the invalid call counted toward the limit (BR-04)


# -- 4 -----------------------------------------------------------------------


async def test_04_fixed_dates_without_flights_asks_and_does_not_change_them(orchestrator, provider):
    view = await orchestrator.create_trip("Vienna from London, 8-11 Dec 2026, 2 adults")
    assert view["state"] == "WAITING_FOR_DETAILS" and view["plan"] is None
    form = view["form"]
    assert form["kind"] == "ask_user" and "no flights" in form["reason"]
    assert "2026-12-07 to 2026-12-10" in form["reason"]  # the nearest dates that work
    fields = {f["name"]: f for f in form["fields"]}
    assert set(fields) == {"start_date", "end_date"}
    assert fields["start_date"]["options"][:2] == ["2026-12-07", "2026-12-09"]
    assert (fields["start_date"]["value"], fields["end_date"]["value"]) == ("2026-12-08", "2026-12-11")
    assert (view["request"]["start_date"], view["request"]["end_date"]) == ("2026-12-08", "2026-12-11")  # unchanged
    assert "search_hotels" not in tools_called(provider)

    chosen = await orchestrator.submit_form(view["trip_id"], {"start_date": "2026-12-07", "end_date": "2026-12-10"})
    assert chosen["state"] == "PRESENTED"
    assert [chosen["plan"]["days"][0]["date"], chosen["plan"]["days"][-1]["date"]] == ["2026-12-07", "2026-12-10"]


# -- 5 -----------------------------------------------------------------------


async def test_05_flexible_dates_shift_and_say_so(orchestrator):
    view = await orchestrator.create_trip("Vienna from London, 8-11 Dec 2026, 2 adults, my dates are flexible")
    assert view["state"] == "PRESENTED"
    assert [view["plan"]["days"][0]["date"], view["plan"]["days"][-1]["date"]] == ["2026-12-07", "2026-12-10"]
    assert (view["request"]["start_date"], view["request"]["end_date"]) == ("2026-12-08", "2026-12-11")
    shift = [a for a in view["plan"]["assumptions"] if "Dates shifted from 2026-12-08" in a]
    assert len(shift) == 1 and "2026-12-07" in shift[0]
    assert items_of(view, "flight")[0]["start"].startswith("2026-12-07")


async def test_05b_a_date_shift_is_stated_even_if_the_planner_does_not_mention_it(make_orchestrator):
    def drop_assumptions(plan: dict) -> None:
        plan["assumptions"] = []

    view = await make_orchestrator(MutatingPlanner(drop_assumptions)).create_trip(
        "Vienna from London, 8-11 Dec 2026, 2 adults, my dates are flexible"
    )
    assert view["plan"]["assumptions"][0].startswith("Dates shifted from 2026-12-08")


# -- 6 -----------------------------------------------------------------------


async def test_06_no_hotel_under_budget_relaxes_one_constraint_and_explains(orchestrator, provider, store):
    view = await orchestrator.create_trip("Vienna from London, 9-13 Nov 2026, 2 adults, budget 1300 EUR")
    assert view["state"] == "PRESENTED" and view["plan"]["budget_check"]["within_budget"] is True
    [explained] = [a for a in view["plan"]["assumptions"] if a.startswith("Relaxed the hotel rating")]
    assert "wanted 4+" in explained and "per night" in explained
    hotel_searches = [args for tool, args in provider.call_log if tool == "search_hotels"]
    assert [s["min_rating"] for s in hotel_searches] == [4.0, None]  # the rating was the one thing relaxed
    assert hotel_searches[0]["max_price_per_night"] == hotel_searches[1]["max_price_per_night"]
    session = await store.load(view["trip_id"])
    hotel = session.plan.item("hotel")
    record = next(h for h in session.tool_results[hotel.ref.result_id].payload if h["hotel_id"] == hotel.ref.item_id)
    assert record["rating"] < 4.0 and record["price_per_night"] <= hotel_searches[1]["max_price_per_night"]


# -- 7 -----------------------------------------------------------------------


async def test_07_flight_tool_timeout_retries_twice_then_partial_with_the_gap(orchestrator, failures, store):
    failures.set("search_flights", {"mode": "timeout"})
    view = await orchestrator.create_trip(VIENNA)
    assert view["state"] == "PARTIAL"
    assert view["plan"]["gaps"] == ["Flights could not be searched, so the plan has no flights and no flight cost."]
    assert view["plan"]["violations"] == []  # what is in the plan is sound
    assert items_of(view, "flight") == [] and len(items_of(view, "hotel")) == 1
    session = await store.load(view["trip_id"])
    flight_calls = [c for c in session.tool_calls if c.tool == "search_flights"]
    assert max(c.attempts for c in flight_calls) == 3  # one try and two retries
    assert all(c.status == "failed" for c in flight_calls)
    assert session.unavailable_tools == ["search_flights"]
    assert events(session)[-1] == "tool_gaps"  # VALIDATING -> PARTIAL


# -- 8 -----------------------------------------------------------------------


async def test_08_planner_that_keeps_calling_tools_is_stopped_at_8_iterations(make_orchestrator, store):
    view = await make_orchestrator(KeepsSearching()).create_trip(VIENNA)
    assert view["state"] == "PARTIAL"
    assert view["limits"]["iterations"] == 8
    assert "limit of 8 planning iterations" in view["limits"]["stopped_by"]
    assert any("Planning stopped early" in gap for gap in view["plan"]["gaps"])
    session = await store.load(view["trip_id"])
    assert events(session)[-1] == "loop_guard" and events(session).count("data_tools") == 7
    assert session.stats.llm_calls == 1 + 8  # intake, then exactly eight planner turns


async def test_08b_planner_that_ignores_the_wrap_up_turn_ends_failed(make_orchestrator, provider):
    stubborn = ScriptedLLM([[call("record_request", **COMPLETE)], [call("search_places", city="Vienna")]], repeat_last=True)
    view = await make_orchestrator(stubborn).create_trip(VIENNA)
    assert view["state"] == "FAILED" and view["plan"] is None  # no usable plan
    assert "limit of 8 planning iterations" in view["give_up"]["reason"]
    assert stubborn.turns == 1 + 8
    assert len(provider.call_log) == 7  # the search in the wrap-up turn was refused, not run
    _, offered = stubborn.seen[-1]
    assert offered == ["propose_plan", "give_up"]


# -- 9 -----------------------------------------------------------------------


async def test_09_overnight_flight_moves_hotel_check_in_to_the_next_day(orchestrator, provider):
    view = await orchestrator.create_trip("Vienna from Delhi, 7-11 Nov 2026, 2 adults")
    assert view["state"] == "PRESENTED"
    [outbound, _] = items_of(view, "flight")
    assert outbound["start"].startswith("2026-11-07T23:55") and outbound["end"].startswith("2026-11-08T05:40")
    [hotel] = items_of(view, "hotel")
    assert hotel["day"] == 2 and hotel["start"].startswith("2026-11-08")
    [hotel_search] = [args for tool, args in provider.call_log if tool == "search_hotels"]
    assert hotel_search["check_in"] == "2026-11-08"  # derived by the system from the flight (BR-05)
    assert [i["type"] for i in view["plan"]["days"][0]["items"]] == ["flight"]


async def test_09b_check_in_on_the_departure_date_of_an_overnight_flight_is_a_violation(make_orchestrator, store):
    def check_in_a_day_early(plan: dict) -> None:
        hotel = find_item(plan, "hotel")
        plan["days"][1]["items"].remove(hotel)
        hotel.update(day=1, start="2026-11-07T15:00")
        plan["days"][0]["items"].append(hotel)

    view = await make_orchestrator(MutatingPlanner(check_in_a_day_early)).create_trip(
        "Vienna from Delhi, 7-11 Nov 2026, 2 adults"
    )
    session = await store.load(view["trip_id"])
    first = session.stats.proposals[0].violations
    assert any(v.rule == "BR-05" and "overnight flight moves it to the next day" in v.message for v in first)
    assert view["state"] == "PRESENTED" and items_of(view, "hotel")[0]["day"] == 2  # repaired


# -- 10 ----------------------------------------------------------------------


async def test_10_monday_closed_place_on_a_monday_is_caught_and_repair_moves_it(make_orchestrator, store):
    def onto_monday(plan: dict) -> None:
        museum = next(i for d in plan["days"] for i in d["items"] if (i.get("ref") or {}).get("item_id") == "VIE-P05")
        plan["days"][museum["day"] - 1]["items"].remove(museum)
        museum.update(day=3, start="2026-11-09" + museum["start"][10:], end="2026-11-09" + museum["end"][10:])
        plan["days"][2]["items"].append(museum)

    view = await make_orchestrator(MutatingPlanner(onto_monday)).create_trip(VIENNA)
    session = await store.load(view["trip_id"])
    first, second = session.stats.proposals
    assert first.phase == "initial" and any(
        v.check == 3 and "Kunsthistorisches Museum is closed on Mondays" in v.message for v in first.violations
    )
    assert second.phase == "repair" and second.violations == []
    assert view["state"] == "PRESENTED" and view["version"] == 2
    assert [v["reason"] for v in view["versions"]] == ["initial", "repair"]
    museum = next(i for i in items_of(view, "activity") if i["ref"]["item_id"] == "VIE-P05")
    assert museum["day"] != 3  # no longer on the Monday
    assert "violations" in events(session) and "repaired_plan" in events(session)


# -- 11 ----------------------------------------------------------------------


def _invent_hotel(plan: dict) -> None:
    hotel = find_item(plan, "hotel")
    hotel["title"] = "Hotel Imaginary"
    hotel["ref"]["item_id"] = "VIE-H99"


async def test_11_hotel_that_is_in_no_result_is_rejected(make_orchestrator, store):
    view = await make_orchestrator(MutatingPlanner(_invent_hotel)).create_trip(VIENNA)
    session = await store.load(view["trip_id"])
    rejected = session.stats.proposals[0]
    assert rejected.ungrounded_items == 1
    assert any(v.rule == "BR-08" and v.target == "hotel" and "Hotel Imaginary" in v.message for v in rejected.violations)
    assert view["state"] == "PRESENTED"  # the one repair replaced it with a real hotel
    assert items_of(view, "hotel")[0]["title"] != "Hotel Imaginary"
    assert session.stats.proposals[-1].ungrounded_items == 0


async def test_11b_if_the_invented_hotel_survives_repair_the_plan_is_partial(make_orchestrator):
    view = await make_orchestrator(MutatingPlanner(_invent_hotel, count=2)).create_trip(VIENNA)
    assert view["state"] == "PARTIAL"  # never presented as a clean plan
    assert any(v["rule"] == "BR-08" and v["target"] == "hotel" for v in view["plan"]["violations"])


# -- 12 ----------------------------------------------------------------------


async def test_12_over_budget_states_the_overshoot_and_never_says_within_budget(make_orchestrator, orchestrator):
    view = await orchestrator.create_trip("Vienna from London, 9-13 Nov 2026, 2 adults, budget 500 EUR")
    check = view["plan"]["budget_check"]
    assert view["state"] == "PARTIAL"
    assert check["within_budget"] is False and check["overshoot"] > 0
    assert check["label"] == f"Over budget by EUR {check['overshoot']:.2f}"
    assert any(v["rule"] == "BR-09" and f"by EUR {check['overshoot']:.2f}" in v["message"] for v in view["plan"]["violations"])
    assert any(f"EUR {check['overshoot']:.2f} over the EUR 500.00 budget" in a for a in view["plan"]["assumptions"])
    said = " ".join([check["label"], *view["plan"]["assumptions"]]).lower()
    assert "within budget" not in said and "under budget" not in said

    def claim(plan: dict) -> None:
        plan["assumptions"].append("This plan is comfortably within budget.")

    lying = await make_orchestrator(MutatingPlanner(claim, count=2)).create_trip(
        "Vienna from London, 9-13 Nov 2026, 2 adults, budget 500 EUR"
    )
    assert lying["plan"]["budget_check"]["label"].startswith("Over budget by")  # the label comes from code
    assert any("says the plan is within budget" in v["message"] for v in lying["plan"]["violations"])


# -- 13 ----------------------------------------------------------------------


async def test_13_five_star_on_a_low_budget_asks_which_matters_more(orchestrator, provider):
    view = await orchestrator.create_trip("Vienna from London, 9-13 Nov 2026, 2 adults, 5-star hotel, budget 1200 EUR")
    assert view["state"] == "WAITING_FOR_DETAILS" and view["plan"] is None
    form = view["form"]
    assert form["kind"] == "ask_user" and "Which matters more?" in form["reason"]
    assert [f["name"] for f in form["fields"]] == ["total_budget", "min_hotel_rating"]
    assert [f["value"] for f in form["fields"]] == [1200.0, 5.0]
    hotel_searches = [args for tool, args in provider.call_log if tool == "search_hotels"]
    assert len(hotel_searches) == 1 and hotel_searches[0]["min_rating"] == 5.0  # it did not quietly drop the rating

    lowered = await orchestrator.submit_form(view["trip_id"], {"total_budget": "1200", "min_hotel_rating": "3"})
    assert lowered["state"] in ("PRESENTED", "PARTIAL") and lowered["request"]["min_hotel_rating"] == 3.0
    assert len(items_of(lowered, "hotel")) == 1


# -- 15 ----------------------------------------------------------------------


async def test_15_form_submitted_after_a_server_restart_resumes_the_same_trip(pg_store, provider, settings, clock):
    from trip_agent.db.postgres import PostgresStore

    before = Orchestrator(pg_store, StandInPlanner(), provider, settings, clock)
    waiting = await before.create_trip("Plan a trip to Austria")
    assert waiting["state"] == "WAITING_FOR_DETAILS"
    trip_id = waiting["trip_id"]
    del before

    # "restart": a new connection pool, a new orchestrator, nothing carried over in memory
    restarted_store = await PostgresStore.open(max_size=2)
    try:
        after = Orchestrator(restarted_store, StandInPlanner(), MockToolProvider(), settings, clock)
        resumed = await after.get(trip_id)
        assert resumed["state"] == "WAITING_FOR_DETAILS" and resumed["form"] == waiting["form"]
        done = await after.submit_form(trip_id, FORM)
        assert done["trip_id"] == trip_id and done["state"] == "PRESENTED"  # the same trip, not a new one
        session = await restarted_store.load(trip_id)
        assert session.request.raw_text == "Plan a trip to Austria"
        assert events(session) == ["request_incomplete", "form_valid", "data_tools", "data_tools", "data_tools", "propose_plan", "no_violations"]
        assert session.tool_calls[0].tool == "search_places" and session.tool_calls[0].iteration == 0  # from before the restart
    finally:
        await restarted_store.close()


# -- 16 ----------------------------------------------------------------------


async def test_16_stale_prices_are_re_fetched_before_the_plan_is_shown(orchestrator, provider, clock, store):
    first = await orchestrator.create_trip(VIENNA)
    assert first["state"] == "PRESENTED"
    calls_before = len(provider.call_log)
    hotel_before = items_of(first, "hotel")[0]

    clock.advance(minutes=30)
    assert len(provider.call_log) == calls_before and (await orchestrator.get(first["trip_id"]))["version"] == 1  # still fresh

    clock.advance(minutes=31)  # 61 minutes old: past FRESHNESS_MINUTES
    provider.price_drift["search_hotels"] = 1.10
    shown = await orchestrator.get(first["trip_id"])
    assert len(provider.call_log) == calls_before + 6  # 2 flight, 1 hotel, 1 places, 2 travel-time results re-fetched
    priced = [i for i in items_of(shown) if i["source"]]
    assert priced and all(i["fetched_at"] == clock.now().isoformat() and not i["stale"] for i in priced)
    hotel_after = items_of(shown, "hotel")[0]
    assert hotel_after["cost"] == pytest.approx(hotel_before["cost"] * 1.10, abs=0.02)
    assert any("price changed from EUR" in n and hotel_after["title"] in n for n in shown["notices"])
    assert shown["version"] == 2 and shown["versions"][-1]["reason"] == "refresh"
    assert shown["plan"]["budget"]["lodging"] == hotel_after["cost"]
    session = await store.load(first["trip_id"])
    assert session.version(1).plan.item("hotel").cost == hotel_before["cost"]  # the old version is kept as it was


async def test_16b_a_failed_re_fetch_is_reported_not_hidden(orchestrator, provider, clock, failures):
    first = await orchestrator.create_trip(VIENNA)
    clock.advance(minutes=61)
    failures.set("search_hotels", {"mode": "error"})
    shown = await orchestrator.get(first["trip_id"])
    hotel = items_of(shown, "hotel")[0]
    assert hotel["stale"] is True
    assert any(v["rule"] == "BR-10" and v["target"] == "hotel" for v in shown["plan"]["violations"])
    with pytest.raises(Exception, match="open problems"):
        await orchestrator.accept(first["trip_id"])


# -- 17 ----------------------------------------------------------------------


async def test_17_instruction_in_a_place_description_changes_nothing(make_orchestrator, provider, settings, clock, tmp_path, store):
    for name in ("flights", "hotels", "places"):
        shutil.copy(DATA_DIR / f"{name}.json", tmp_path / f"{name}.json")
    places = json.loads((tmp_path / "places.json").read_text(encoding="utf-8"))
    for place in places["Vienna"]:
        if place["place_id"] == "VIE-P17":
            assert "ignore your previous instructions" in place["description"]
            place["description"] = "Interactive sound museum."
    (tmp_path / "places.json").write_text(json.dumps(places), encoding="utf-8")

    request = VIENNA + ", we like music"  # so the place carrying the instruction is actually used
    with_injection = await make_orchestrator().create_trip(request)
    clean_orchestrator = Orchestrator(store, StandInPlanner(), MockToolProvider(data_dir=tmp_path), settings, clock)
    without = await clean_orchestrator.create_trip(request)
    assert "VIE-P17" in {i["ref"]["item_id"] for i in items_of(with_injection, "activity")}
    assert with_injection["plan"] == without["plan"]  # no change in behaviour
    assert with_injection["state"] == "PRESENTED"
    assert "imperial vip pass" not in json.dumps(with_injection["plan"]).lower()

    # the text reaches the planner only inside tool results, as data
    session = await store.load(with_injection["trip_id"])
    in_results = [r.content for m in session.messages for r in m.tool_results if "ignore your previous instructions" in r.content]
    in_text = [m.text for m in session.messages if m.text and "ignore your previous instructions" in m.text]
    assert in_results and not in_text


async def test_17b_a_planner_that_obeys_the_instruction_is_stopped_by_the_verifier(make_orchestrator):
    def obey(plan: dict) -> None:
        for day in plan["days"]:
            day["items"].append(
                {"id": f"vip-{day['day']}", "type": "activity", "title": "Imperial VIP Pass", "day": day["day"],
                 "start": f"{day['date']}T20:00", "end": f"{day['date']}T21:00", "cost": 900, "cost_status": "confirmed"}
            )  # fmt: skip
        plan["assumptions"].append("The plan is within budget.")

    view = await make_orchestrator(MutatingPlanner(obey, count=2)).create_trip(VIENNA)
    assert view["state"] == "PARTIAL"  # shown with its problems listed, never as a clean plan
    ungrounded = [v for v in view["plan"]["violations"] if v["rule"] == "BR-08" and v["target"].startswith("vip-")]
    assert len(ungrounded) == 5
    assert view["plan"]["budget_check"]["label"].startswith("Over budget by")


# -- 18 ----------------------------------------------------------------------


async def test_18_form_submitted_twice_runs_one_planning_run(make_orchestrator, provider, store):
    planner = Counting()
    first_worker, second_worker = make_orchestrator(planner), make_orchestrator(planner)
    trip = (await first_worker.create_trip("Plan a trip to Austria"))["trip_id"]
    llm_before, tools_before = planner.calls, len(provider.call_log)

    # a double-click: the same form, the same idempotency key, at the same time
    one, two = await asyncio.gather(
        first_worker.submit_form(trip, FORM, key="double-click-1"),
        second_worker.submit_form(trip, FORM, key="double-click-1"),
    )
    assert one == two and one["state"] == "PRESENTED"
    assert planner.calls - llm_before == 4  # one run: search, hotel, travel times, propose
    assert len(provider.call_log) - tools_before == 6
    session = await store.load(trip)
    assert [c.id for c in session.tool_calls] == [f"tc_{n:04d}" for n in range(1, 8)]  # no duplicated calls
    assert events(session).count("form_valid") == 1 and len(session.versions) == 1

    again = await first_worker.submit_form(trip, FORM, key="double-click-1")  # a late retry of the same request
    assert again == one and planner.calls - llm_before == 4


async def test_18b_trip_creation_is_idempotent_too(make_orchestrator, provider):
    planner = Counting()
    orchestrator = make_orchestrator(planner)
    one, two = await asyncio.gather(
        orchestrator.create_trip(VIENNA, key="create-once-1"), orchestrator.create_trip(VIENNA, key="create-once-1")
    )
    assert one == two and planner.calls == 5  # intake + four planner turns, once
    other = await orchestrator.create_trip(VIENNA, key="create-once-2")
    assert other["trip_id"] != one["trip_id"]


# -- 20 ----------------------------------------------------------------------


async def test_20_a_plan_revised_twice_keeps_versions_1_to_3(orchestrator, store):
    v1 = await orchestrator.create_trip(VIENNA)
    v2 = await orchestrator.revise(v1["trip_id"], "Less walking on day 2")
    v3 = await orchestrator.revise(v1["trip_id"], "A lighter day 4 please")
    assert (v1["version"], v2["version"], v3["version"]) == (1, 2, 3)
    assert v3["state"] == "PRESENTED"

    session = await store.load(v1["trip_id"])  # read back from the store
    assert session.current_version == 3  # the trip points to version 3
    assert [(v.version, v.reason, v.parent_version) for v in session.versions] == [
        (1, "initial", None), (2, "revision", 1), (3, "revision", 2),
    ]  # fmt: skip
    stored = await orchestrator.versions(v1["trip_id"])
    dump = lambda plan: json.loads(plan.model_dump_json(exclude={"gaps", "violations"}))  # noqa: E731
    for version, view in zip(stored, (v1, v2, v3)):
        shown = [[i["id"] for i in d["items"]] for d in view["plan"]["days"]]
        assert [[i["id"] for i in d["items"]] for d in dump(version.plan)["days"]] == shown
        assert version.plan.budget.total == view["plan"]["budget"]["total"]
    assert stored[0].plan != stored[1].plan != stored[2].plan  # three different plans, none overwritten


# -- 21 ----------------------------------------------------------------------


async def test_21_run_past_75_percent_of_the_wall_clock_wraps_up_as_partial(make_orchestrator, clock, store):
    view = await make_orchestrator(Slow(clock, seconds=50), wall_clock_s=120).create_trip(VIENNA)
    assert view["state"] == "PARTIAL"
    assert "wall-clock limit of 120 s was nearly used up" in view["limits"]["stopped_by"]
    assert any("Planning stopped early" in gap and "wall-clock" in gap for gap in view["plan"]["gaps"])  # says why
    session = await store.load(view["trip_id"])
    assert session.iteration_count == 3  # search, hotel, then the wrap-up turn; no fourth turn
    assert events(session)[-1] == "loop_guard"
    assert view["plan"]["days"]  # the best plan the gathered results support


async def test_21b_run_past_the_wall_clock_with_nothing_proposed_stops_and_says_why(make_orchestrator, clock):
    view = await make_orchestrator(Slow(clock, seconds=130), wall_clock_s=120).create_trip(VIENNA)
    assert view["state"] == "FAILED" and view["plan"] is None  # nothing usable had been proposed
    assert "wall-clock limit of 120 s was reached" in view["give_up"]["reason"]
    assert view["limits"]["iterations"] == 1  # no further planner turn after the limit


async def test_21c_a_planner_call_that_hangs_is_cut_off(make_orchestrator):
    class Hangs(StandInPlanner):
        async def complete(self, *, system, messages, tools):
            if not any(t.name == "record_request" for t in tools):
                await asyncio.sleep(5)
            return await super().complete(system=system, messages=messages, tools=tools)

    started = asyncio.get_running_loop().time()
    view = await make_orchestrator(Hangs(), wall_clock_s=0.2).create_trip(VIENNA)
    assert asyncio.get_running_loop().time() - started < 2  # did not wait for the 5 s call
    assert view["state"] == "FAILED" and "wall-clock limit" in view["give_up"]["reason"]


# -- other BR-06 limits ------------------------------------------------------


async def test_cost_cap_stops_planning(make_orchestrator):
    llm = ScriptedLLM(
        [[call("record_request", **COMPLETE)], [call("search_places", city="Vienna")]],
        repeat_last=True, cost_per_turn=0.4,
    )  # fmt: skip
    view = await make_orchestrator(llm, cost_cap_usd=1.0).create_trip(VIENNA)
    assert view["state"] == "FAILED" and "cost reached the cap of USD 1.00" in view["give_up"]["reason"]
    assert llm.turns == 3  # 0.4 + 0.4 + 0.4 >= 1.0: no fourth call; the intake call counted too
    assert view["limits"]["estimated_cost_usd"] == pytest.approx(1.2)


async def test_token_budget_stops_planning(make_orchestrator):
    llm = ScriptedLLM([[call("record_request", **COMPLETE)], [call("search_places", city="Vienna")]], repeat_last=True)
    view = await make_orchestrator(llm, token_budget=600).create_trip(VIENNA)
    assert view["state"] == "FAILED" and "token budget of 600 was used up" in view["give_up"]["reason"]
    assert llm.turns < 9
