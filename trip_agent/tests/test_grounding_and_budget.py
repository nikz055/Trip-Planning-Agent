"""Grounding (BR-08) and budget arithmetic (BR-09)."""

from datetime import date

import pytest

from trip_agent.config import Settings
from trip_agent.llm.base import LLMToolCall
from trip_agent.models import CostStatus, ItemRef, ItemType, Plan, PlanDay, PlanItem
from trip_agent.orchestrator.budget import check_budget, compute_budget
from trip_agent.orchestrator.grounding import check_grounding, resolve
from trip_agent.tests.conftest import trip_request

POINTS = [{"id": "VIE-H07", "lat": 48.218, "lon": 16.392}, {"id": "VIE-P02", "lat": 48.2085, "lon": 16.3731}]


@pytest.fixture
async def searched(executor, session):
    """A session with one result of each data tool. Returns the call ids."""
    calls = await executor.run_batch(
        session,
        [
            LLMToolCall(id="f", name="search_flights", args=dict(origin="LHR", destination="VIE", depart_date="2026-11-07", adults=2)),
            LLMToolCall(id="h", name="search_hotels", args=dict(city="Vienna", check_in="2026-11-07", check_out="2026-11-11", guests=2)),
            LLMToolCall(id="p", name="search_places", args=dict(city="Vienna")),
            LLMToolCall(id="t", name="get_travel_times", args=dict(points=POINTS, mode="transit")),
        ],
    )
    assert all(c.status == "ok" for c in calls)
    return {c.tool: c.id for c in calls}


def item(kind: ItemType, ref: tuple[str, str] | None, **kw) -> PlanItem:
    return PlanItem(
        id=kw.pop("id", kind.value), type=kind, title=kw.pop("title", kind.value), day=1,
        ref=ItemRef(result_id=ref[0], item_id=ref[1]) if ref else None, **kw,
    )  # fmt: skip


def plan_of(*items: PlanItem) -> Plan:
    return Plan(days=[PlanDay(day=1, date=date(2026, 11, 7), items=list(items))])


def grounding(session, plan):
    return check_grounding(plan, session.tool_calls, session.tool_results)


async def test_items_that_reference_real_results_are_grounded(session, searched):
    offer = session.tool_results[searched["search_flights"]].payload[0]
    plan = plan_of(
        item(ItemType.flight, (searched["search_flights"], offer["offer_id"])),
        item(ItemType.hotel, (searched["search_hotels"], "VIE-H07")),
        item(ItemType.activity, (searched["search_places"], "VIE-P02")),
        item(ItemType.transfer, (searched["get_travel_times"], "VIE-H07->VIE-P02")),
        item(ItemType.meal_allowance, None),
    )
    assert grounding(session, plan) == []


async def test_resolve_returns_the_stored_record(session, searched):
    calls = {c.id: c for c in session.tool_calls}
    hotel = resolve(item(ItemType.hotel, (searched["search_hotels"], "VIE-H07")), calls, session.tool_results)
    assert hotel.record["name"] == "Praterstern City Hotel"
    leg = resolve(item(ItemType.transfer, (searched["get_travel_times"], "VIE-H07->VIE-P02")), calls, session.tool_results)
    assert leg.minutes > 0 and leg.unit_cost == 2.4 and (leg.from_id, leg.to_id) == ("VIE-H07", "VIE-P02")


async def test_item_without_a_ref_is_rejected(session, searched):
    [violation] = grounding(session, plan_of(item(ItemType.hotel, None, title="Hotel Imaginary")))
    assert violation.rule == "BR-08" and violation.check == 2 and violation.target == "hotel"
    assert "Hotel Imaginary" in violation.message and "no ref" in violation.message


async def test_item_referencing_a_missing_result_is_rejected(session, searched):
    [violation] = grounding(session, plan_of(item(ItemType.hotel, ("tc_9999", "VIE-H07"))))
    assert "tc_9999" in violation.message and "does not exist" in violation.message


async def test_hotel_not_in_any_result_is_rejected(session, searched):
    [violation] = grounding(session, plan_of(item(ItemType.hotel, (searched["search_hotels"], "VIE-H99"))))
    assert "'VIE-H99' is not in tool result" in violation.message


async def test_item_referencing_the_wrong_kind_of_result_is_rejected(session, searched):
    [violation] = grounding(session, plan_of(item(ItemType.hotel, (searched["search_places"], "VIE-P02"))))
    assert "must come from search_hotels" in violation.message


async def test_transfer_must_name_a_leg_in_the_matrix(session, searched):
    travel = searched["get_travel_times"]
    bad = [item(ItemType.transfer, (travel, leg), id=f"t{n}") for n, leg in enumerate(["VIE-H07", "VIE-H07->VIE-P30", "nowhere->VIE-P02"])]
    assert len(grounding(session, plan_of(*bad))) == 3


async def test_failed_call_cannot_ground_anything(session, executor, failures):
    failures.set("search_hotels", {"mode": "error"})
    [call] = await executor.run_batch(session, [LLMToolCall(id="h", name="search_hotels", args=dict(city="Vienna", check_in="2026-11-07", check_out="2026-11-11", guests=2))])
    [violation] = grounding(session, plan_of(item(ItemType.hotel, (call.id, "VIE-H07"))))
    assert "does not exist" in violation.message


def test_allowance_must_not_carry_a_ref(session):
    [violation] = grounding(session, plan_of(item(ItemType.meal_allowance, ("tc_0001", "x"))))
    assert violation.message == "allowances carry no ref"


# -- budget ------------------------------------------------------------------


def priced(kind: ItemType, cost: float, status: CostStatus, n: int = 0) -> PlanItem:
    return PlanItem(id=f"{kind.value}{n}", type=kind, day=1, cost=cost, cost_status=status)


def test_budget_sums_every_category_and_splits_confirmed_from_estimated():
    C, E = CostStatus.confirmed, CostStatus.estimated
    plan = plan_of(
        priced(ItemType.flight, 200.10, C), priced(ItemType.flight, 150.15, C, 1),
        priced(ItemType.hotel, 480, C), priced(ItemType.transfer, 4.8, E),
        priced(ItemType.activity, 35, C), priced(ItemType.meal_allowance, 90, E),
        priced(ItemType.other_allowance, 70, E),
    )  # fmt: skip
    b = compute_budget(plan)
    assert (b.flights, b.lodging, b.local_transport) == (350.25, 480, 4.8)
    assert (b.activities, b.meals_allowance, b.other_allowance) == (35, 90, 70)
    assert b.total == 1030.05
    assert b.confirmed_total == 865.25 and b.estimated_total == 164.8
    assert b.confirmed_total + b.estimated_total == pytest.approx(b.total)


def test_empty_plan_has_a_zero_budget():
    assert compute_budget(Plan()).total == 0


def test_budget_check_within_and_over():
    s = Settings()
    within = check_budget(1692.36, trip_request(total_budget=2500), s)
    assert within.within_budget is True and within.overshoot == 0
    assert within.label == "Within budget (EUR 807.64 to spare)"
    over = check_budget(1692.36, trip_request(total_budget=1500), s)
    assert over.within_budget is False and over.overshoot == 192.36
    assert over.label == "Over budget by EUR 192.36"
    assert "within" not in over.label.lower()  # never called within budget when it isn't
    exact = check_budget(1500, trip_request(total_budget=1500), s)
    assert exact.within_budget is True


def test_budget_check_without_a_budget_makes_no_claim():
    check = check_budget(1000, trip_request(total_budget=None), Settings())
    assert check.within_budget is None and check.label == "No budget given"


def test_budget_check_converts_other_currencies_and_says_so():
    check = check_budget(1000, trip_request(total_budget=1000, currency="USD"), Settings())
    assert check.limit == 920 and check.within_budget is False and check.overshoot == 80
    assert "USD 1,000.00 converted to EUR 920.00" in check.note


def test_budget_check_with_an_unknown_currency_makes_no_claim():
    check = check_budget(1000, trip_request(total_budget=1000, currency="JPY"), Settings())
    assert check.within_budget is None and "not checked" in check.note
