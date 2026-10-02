"""The nine verifier checks (spec section 7), each exercised on a plan that is
valid except for one deliberate fault."""

from datetime import date, datetime, timedelta

import pytest

from trip_agent.db.memory import MemoryStore
from trip_agent.llm.base import LLMToolCall
from trip_agent.llm.standin import StandInPlanner
from trip_agent.models import Budget, CostStatus, ItemRef, ItemType, Pace, Plan, PlanItem, Session
from trip_agent.orchestrator.orchestrator import Orchestrator
from trip_agent.orchestrator.verifier import VerifyContext, verify
from trip_agent.tests.conftest import VIENNA

MONDAY = date(2026, 11, 9)


@pytest.fixture
async def planned(provider, settings, clock) -> tuple[Session, Plan]:
    """A presented Vienna plan and the session (tool results) behind it."""
    store = MemoryStore()
    view = await Orchestrator(store, StandInPlanner(), provider, settings, clock).create_trip(VIENNA)
    assert view["state"] == "PRESENTED"
    session = await store.load(view["trip_id"])
    return session, session.plan.model_copy(deep=True)


@pytest.fixture
def check(planned, settings, clock):
    """Run the verifier on a plan against the planned session; returns the violations."""
    session, _ = planned

    def run(plan: Plan, **overrides):
        ctx = VerifyContext(
            request=overrides.pop("request", session.request),
            calls=session.tool_calls,
            results=session.tool_results,
            settings=overrides.pop("settings", settings),
            now=overrides.pop("now", clock.now()),
            **overrides,
        )
        return verify(plan, ctx)

    return run


def activities(plan: Plan, day: int) -> list[PlanItem]:
    return sorted((i for i in plan.days[day - 1].items if i.type is ItemType.activity), key=lambda i: i.start)


def by_place(plan: Plan, place_id: str) -> PlanItem:
    return next(i for i in plan.items() if i.type is ItemType.activity and i.ref.item_id == place_id)


def shift(item: PlanItem, **delta) -> None:
    item.start += timedelta(**delta)
    item.end += timedelta(**delta)


def found(violations, check_no: int, target: str | None = None, text: str = "") -> bool:
    return any(
        v.check == check_no and (target is None or v.target == target) and text in v.message for v in violations
    )


def test_the_stand_ins_plan_passes_every_check(planned, check):
    _, plan = planned
    assert check(plan) == []


# -- check 1: schema-level structure and dates --------------------------------


def test_check1_duplicate_item_ids(planned, check):
    _, plan = planned
    first, second = activities(plan, 2)[:2]
    second.id = first.id
    assert found(check(plan), 1, first.id, "used more than once")


def test_check1_day_numbers_and_dates_must_run_in_order(planned, check):
    _, plan = planned
    plan.days[2].date = date(2026, 11, 20)
    assert found(check(plan), 1, "day 3", "consecutive")
    plan2 = planned[0].plan.model_copy(deep=True)  # a fresh copy of the valid plan
    plan2.days[1].day = 7
    assert found(check(plan2), 1, text="numbered 1..5")


def test_check1_item_must_sit_on_its_own_day(planned, check):
    _, plan = planned
    item = activities(plan, 2)[0]
    shift(item, days=1)  # now dated day 3 but still listed under day 2
    assert found(check(plan), 1, item.id, "is listed under day 2")


def test_check1_times_must_be_present_and_ordered(planned, check):
    _, plan = planned
    first, second = activities(plan, 2)[:2]
    first.start = None
    second.end = second.start
    violations = check(plan)
    assert found(violations, 1, first.id, "needs a start and an end")
    assert found(violations, 1, second.id, "ends at or before it starts")


def test_check1_fixed_dates_cannot_be_moved(planned, check):
    session, plan = planned
    for day in plan.days:
        day.date += timedelta(days=1)
        for item in day.items:
            if item.start:
                shift(item, days=1)
    violations = check(plan)
    assert found(violations, 1, "plan", "trip dates are fixed") and violations[0].rule == "BR-12"
    flexible = session.request.model_copy(update={"dates_flexible": True})
    assert not any(v.rule == "BR-12" for v in check(plan, request=flexible))


def test_check1_flexible_dates_have_a_limit(planned, check):
    session, plan = planned
    for day in plan.days:
        day.date += timedelta(days=10)
    flexible = session.request.model_copy(update={"dates_flexible": True})
    assert found(check(plan, request=flexible), 1, "plan", "at most 3 days")


# -- check 2: grounding -------------------------------------------------------


def test_check2_hotel_not_in_any_result(planned, check):
    _, plan = planned
    plan.item("hotel").ref.item_id = "VIE-H99"
    violations = check(plan)
    assert found(violations, 2, "hotel", "'VIE-H99' is not in tool result")
    assert all(v.rule == "BR-08" for v in violations if v.check == 2)


def test_check2_item_without_a_ref_and_with_a_made_up_result(planned, check):
    _, plan = planned
    first, second = activities(plan, 2)[:2]
    first.ref = None
    second.ref = ItemRef(result_id="tc_9999", item_id=second.ref.item_id)
    violations = check(plan)
    assert found(violations, 2, first.id, "has no ref")
    assert found(violations, 2, second.id, "does not exist")


def test_check2_flight_times_must_be_the_offers(planned, check):
    _, plan = planned
    shift(plan.item("flight-out"), minutes=30)
    assert found(check(plan), 2, "flight-out", "flight times differ from the offer")


async def test_check2_return_flight_must_fly_the_route_back(planned, check, executor):
    session, plan = planned
    # a real offer for the last day, but flying the outbound direction again
    [wrong_way] = await executor.run_batch(
        session,
        [LLMToolCall(id="x", name="search_flights", args=dict(origin="LHR", destination="VIE", depart_date="2026-11-11", adults=2))],
    )
    offer = session.tool_results[wrong_way.id].payload[0]
    back = plan.item("flight-back")
    back.ref = ItemRef(result_id=wrong_way.id, item_id=offer["offer_id"])
    back.start, back.end = datetime.fromisoformat(offer["depart_time"]), datetime.fromisoformat(offer["arrive_time"])
    back.cost = offer["price"]
    assert found(check(plan), 2, "flight-back", "the return flight must go VIE to LHR")


def test_check2_plan_needs_flights_both_ways(planned, check):
    _, plan = planned
    plan.days[4].items = [i for i in plan.days[4].items if i.type is not ItemType.flight]
    assert found(check(plan), 7, "plan", "no return flight departing on 2026-11-11")
    # when the flight tool is unavailable this is reported as a gap, not a violation
    assert not found(check(plan, unavailable_tools=frozenset({"search_flights"})), 7, "plan", "no return flight")


# -- check 3: opening hours and closures --------------------------------------


def test_check3_place_closed_on_mondays(planned, check):
    _, plan = planned
    museum = by_place(plan, "VIE-P05")  # Kunsthistorisches Museum, closed on Mondays
    plan.days[museum.day - 1].items.remove(museum)
    museum.day = 3
    museum.start = datetime.combine(MONDAY, museum.start.time())
    museum.end = datetime.combine(MONDAY, museum.end.time())
    plan.days[2].items.append(museum)
    violations = check(plan)
    assert found(violations, 3, museum.id, "closed on Mondays")
    assert next(v for v in violations if v.check == 3).rule == "BR-13"


def test_check3_listed_closure_date(planned, check):
    session, plan = planned
    item = activities(plan, 2)[0]
    places_call = item.ref.result_id
    place = next(p for p in session.tool_results[places_call].payload if p["place_id"] == item.ref.item_id)
    place["closed_dates"].append(plan.days[1].date.isoformat())
    assert found(check(plan), 3, item.id, f"is closed on {plan.days[1].date}")


def test_check3_outside_opening_hours(planned, check):
    _, plan = planned
    item = activities(plan, 2)[0]
    item.start = item.start.replace(hour=5, minute=0)
    item.end = item.start + timedelta(hours=1)
    assert found(check(plan), 3, item.id, "but is scheduled 05:00-06:00")


# -- check 4: travel time and buffers ------------------------------------------


def test_check4_buffer_between_activities(planned, check):
    _, plan = planned
    before, after = activities(plan, 2)[:2]
    wanted = before.end + timedelta(minutes=10)
    shift(after, seconds=(wanted - after.start).total_seconds())
    violations = check(plan)
    assert found(violations, 4, after.id, "plus a 30 min buffer")


def test_check4_missing_travel_time(planned, check):
    _, plan = planned
    before, after = activities(plan, 2)[:2]
    leg = f"{before.ref.item_id}->{after.ref.item_id}"
    day = plan.days[1]
    day.items = [i for i in day.items if not (i.type is ItemType.transfer and i.ref.item_id == leg)]
    violations = check(plan)
    assert found(violations, 4, after.id, "no travel time from")
    assert found(violations, 4, after.id, f"'{leg}'")


def test_check4_overlapping_activities(planned, check):
    _, plan = planned
    before, after = activities(plan, 2)[:2]
    shift(after, seconds=(before.start - after.start).total_seconds() + 600)
    assert found(check(plan), 4, after.id, "overlaps")


def test_check4_transfer_shorter_than_the_travel_time(planned, check):
    _, plan = planned
    transfer = next(i for i in plan.days[1].items if i.type is ItemType.transfer)
    transfer.end = transfer.start + timedelta(minutes=1)
    assert found(check(plan), 4, transfer.id, "but only 1 min are scheduled")


def test_check4_no_sightseeing_before_landing_or_close_to_departure(planned, check):
    _, plan = planned
    arrival = plan.item("flight-out").end
    early = activities(plan, 1)[0]
    shift(early, seconds=(arrival - early.start).total_seconds() + 600)  # 10 minutes after landing
    assert found(check(plan), 4, early.id, "allow 90 min after landing")

    plan = planned[0].plan.model_copy(deep=True)
    departure = plan.item("flight-back").start
    late = activities(plan, 4)[-1].model_copy(deep=True)
    late.id, late.day = "late", 5
    late.start = departure - timedelta(minutes=100)
    late.end = departure - timedelta(minutes=40)
    plan.days[4].items.append(late)
    assert found(check(plan), 4, "late", "finish 180 min before departure")


# -- check 5: pace and a light arrival day -------------------------------------


def test_check5_pace_cap(planned, check, settings):
    session, plan = planned
    extra = activities(plan, 2)[0].model_copy(deep=True)
    extra.id = "one-too-many"
    plan.days[1].items.append(extra)
    violations = check(plan)
    assert found(violations, 5, "day 2", "the relaxed pace allows at most 3")
    packed = session.request.model_copy(update={"pace": Pace.packed})
    assert not found(check(plan, request=packed), 5)


def test_check5_arrival_day_stays_light(planned, check):
    _, plan = planned
    day1 = plan.days[0]
    for n, source in enumerate(activities(plan, 3)[:2]):
        extra = source.model_copy(deep=True)
        extra.id, extra.day = f"extra-{n}", 1
        day1.items.append(extra)
    assert found(check(plan), 5, "day 1", "arrival day")


# -- check 6: hotel -----------------------------------------------------------


def test_check6_a_hotel_must_cover_every_night(planned, check):
    _, plan = planned
    plan.days[0].items = [i for i in plan.days[0].items if i.type is not ItemType.hotel]
    violations = check(plan)
    assert found(violations, 6, "plan", "no hotel covers the night(s) of 2026-11-07, 2026-11-08, 2026-11-09, 2026-11-10")

    plan = planned[0].plan.model_copy(deep=True)
    hotel = plan.item("hotel")
    hotel.end -= timedelta(days=2)
    assert found(check(plan), 6, "plan", "2026-11-09, 2026-11-10")


def test_check6_check_in_follows_the_flights_arrival(planned, check):
    _, plan = planned
    hotel = plan.item("hotel")
    hotel.start = plan.item("flight-out").end - timedelta(hours=1)
    violations = check(plan)
    assert found(violations, 6, "hotel", "before the flight lands")
    assert any(v.rule == "BR-05" for v in violations)


def test_check6_late_check_in_hotel(planned, check):
    session, plan = planned
    hotel = plan.item("hotel")
    unfiltered = next(c for c in session.tool_calls if c.tool == "search_hotels")
    results = session.tool_results[unfiltered.id].payload
    results.append(
        {**results[0], "hotel_id": "VIE-H10", "name": "Donau Night Owl Hotel", "check_in_time": "22:00"}
    )
    hotel.ref.item_id = "VIE-H10"
    assert hotel.start.hour < 22
    assert found(check(plan), 6, "hotel", "allows check-in from 22:00")


def test_check6_stay_must_be_inside_the_searched_dates(planned, check):
    _, plan = planned
    plan.item("hotel").end += timedelta(days=3)
    assert found(check(plan), 6, "hotel", "outside the searched stay")


# -- check 7: costs and budget -------------------------------------------------


@pytest.mark.parametrize("item_id", ["flight-out", "hotel"])
def test_check7_cost_must_match_the_tool_result(planned, check, item_id):
    _, plan = planned
    plan.item(item_id).cost += 25
    violations = check(plan)
    assert found(violations, 7, item_id, "in the plan, but")
    assert all(v.rule == "BR-09" for v in violations if v.check == 7)


def test_check7_activity_and_transfer_costs(planned, check):
    _, plan = planned
    paid = next(i for i in plan.items() if i.type is ItemType.activity and i.cost > 0)
    paid.cost = paid.cost / 2  # priced for one traveller instead of two
    ride = next(i for i in plan.items() if i.type is ItemType.transfer and i.cost > 0)
    ride.cost_status = CostStatus.confirmed
    violations = check(plan)
    assert found(violations, 7, paid.id, "x 2 traveller(s)")
    assert found(violations, 7, ride.id, "travel costs are estimates")


def test_check7_allowances_are_always_estimates(planned, check):
    _, plan = planned
    meals = next(i for i in plan.items() if i.type is ItemType.meal_allowance)
    meals.cost_status = CostStatus.confirmed
    assert found(check(plan), 7, meals.id, "must be marked estimated")


def test_check7_budget_arithmetic(planned, check):
    _, plan = planned
    plan.budget = plan.budget.model_copy(update={"total": plan.budget.total - 100, "flights": 1.0})
    violations = check(plan)
    assert found(violations, 7, "plan", "budget.total is")
    assert found(violations, 7, "plan", "budget.flights is 1.00 but the items add up to")
    plan.budget = None  # leaving it out is allowed: the system computes it
    assert not found(check(plan), 7)


def test_check7_meals_allowance_is_required(planned, check):
    _, plan = planned
    for day in plan.days:
        day.items = [i for i in day.items if i.type is not ItemType.meal_allowance]
    plan.budget = None
    assert found(check(plan), 7, "plan", "meals allowance")


def test_check7_over_budget_and_false_claims(planned, check):
    session, plan = planned
    tight = session.request.model_copy(update={"total_budget": 1000.0})
    plan.assumptions.append("Good news: this plan is within your budget.")
    violations = check(plan, request=tight)
    assert found(violations, 7, "plan", "exceeds the budget EUR 1000.00 by EUR")
    assert found(violations, 7, "plan", "says the plan is within budget")
    assert not found(check(plan), 7)  # with the real budget the same sentence is true


# -- check 8: freshness --------------------------------------------------------


def test_check8_prices_older_than_the_limit(planned, check, clock):
    _, plan = planned
    assert not found(check(plan, now=clock.now() + timedelta(minutes=59)), 8)
    violations = check(plan, now=clock.now() + timedelta(minutes=61))
    stale = {v.target for v in violations if v.check == 8}
    assert all(v.rule == "BR-10" for v in violations if v.check == 8)
    assert stale == {i.id for i in plan.items() if i.ref is not None}  # every priced item, no allowance


# -- check 9: locked items and the scope of a revision --------------------------


def test_check9_locked_items_cannot_change_or_disappear(planned, check):
    _, baseline = planned
    plan = baseline.model_copy(deep=True)
    locked = activities(plan, 2)[0]
    locked.title = "Renamed"
    assert found(check(plan, baseline=baseline, locked_ids=frozenset({locked.id})), 9, locked.id, "was changed")
    plan.days[1].items.remove(locked)
    violations = check(plan, baseline=baseline, locked_ids=frozenset({locked.id}))
    assert found(violations, 9, locked.id, "was removed") and violations[-1].rule == "BR-15"
    assert not found(check(plan, baseline=baseline), 9)  # not locked: free to change


def test_check9_days_outside_a_revision_must_be_identical(planned, check):
    _, baseline = planned
    plan = baseline.model_copy(deep=True)
    shift(activities(plan, 3)[-1], minutes=5)
    changed_id = activities(plan, 3)[-1].id
    violations = check(plan, baseline=baseline, affected_days=frozenset({2}))
    assert found(violations, 9, "day 3", changed_id)
    assert not found(check(plan, baseline=baseline, affected_days=frozenset({3})), 9)
    assert not found(check(baseline, baseline=baseline, affected_days=frozenset({2})), 9)


def test_violations_name_a_rule_a_check_and_a_target(planned, check):
    _, plan = planned
    plan.item("hotel").ref.item_id = "VIE-H99"
    plan.item("flight-out").cost += 1
    for violation in check(plan):
        assert violation.rule.startswith("BR-") and 1 <= violation.check <= 9
        assert violation.target and len(violation.message) > 20


def test_budget_model_fields_are_all_checked():
    assert set(Budget.model_fields) == {
        "flights", "lodging", "local_transport", "activities", "meals_allowance",
        "other_allowance", "total", "confirmed_total", "estimated_total",
    }  # fmt: skip
