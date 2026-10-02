"""Locking and scoped revisions (BR-15), including edge case 14."""

from __future__ import annotations

from datetime import datetime

from trip_agent.tests.conftest import VIENNA, MutatingPlanner, find_item, items_of


def walking_minutes(view: dict, day: int) -> float:
    return sum(
        (datetime.fromisoformat(i["end"]) - datetime.fromisoformat(i["start"])).total_seconds() / 60
        for i in view["plan"]["days"][day - 1]["items"]
        if i["type"] == "transfer" and i["title"].startswith("Walk")
    )


async def test_14_less_walking_on_day_2_with_the_hotel_locked(orchestrator, store):
    before = await orchestrator.create_trip(VIENNA)
    locked = await orchestrator.set_lock(before["trip_id"], "hotel", True)
    assert items_of(locked, "hotel")[0]["locked"] is True and locked["version"] == 1
    assert walking_minutes(before, 2) > 0

    after = await orchestrator.revise(before["trip_id"], "Less walking on day 2")
    assert after["state"] == "PRESENTED" and after["version"] == 2
    assert walking_minutes(after, 2) < walking_minutes(before, 2)  # the change that was asked for
    for n in (0, 2, 3, 4):  # only day 2 changes
        assert after["plan"]["days"][n] == locked["plan"]["days"][n]
    assert after["plan"]["days"][1] != locked["plan"]["days"][1]
    assert items_of(after, "hotel") == items_of(locked, "hotel")  # hotel unchanged, still locked
    assert after["plan"]["violations"] == []
    session = await store.load(before["trip_id"])
    assert session.versions[-1].reason == "revision" and session.revision is None


async def test_a_revision_that_touches_another_day_is_caught(make_orchestrator, store):
    def also_change_day_4(plan: dict) -> None:
        if plan["days"][3]["items"][0]["type"] == "transfer":  # only alter the revision proposals
            plan["days"][3]["items"][0]["title"] = "A different way there"

    planner = MutatingPlanner(also_change_day_4, count=0)
    orchestrator = make_orchestrator(planner)
    before = await orchestrator.create_trip(VIENNA)
    planner._left = 1
    after = await orchestrator.revise(before["trip_id"], "Less walking on day 2")
    session = await store.load(before["trip_id"])
    caught = session.stats.proposals[-2].violations
    assert any(v.rule == "BR-15" and v.target == "day 4" for v in caught)
    assert after["state"] == "PRESENTED" and after["plan"]["days"][3] == before["plan"]["days"][3]  # repaired
    assert [v.reason for v in session.versions] == ["initial", "revision", "repair"]


async def test_a_locked_item_cannot_be_changed_by_a_revision(make_orchestrator, store):
    target: dict = {}

    def change_locked_activity(plan: dict) -> None:
        find_item(plan, target["id"])["title"] = "Quietly renamed"

    planner = MutatingPlanner(change_locked_activity, count=0)
    orchestrator = make_orchestrator(planner)
    before = await orchestrator.create_trip(VIENNA)
    activity = items_of(before, "activity")[1]  # the first activity of day 2
    assert activity["day"] == 2
    target["id"] = activity["id"]
    await orchestrator.set_lock(before["trip_id"], activity["id"], True)
    planner._left = 2  # the revision and its repair both tamper with it
    after = await orchestrator.revise(before["trip_id"], "Less walking on day 2")
    assert after["state"] == "PARTIAL"
    assert any(v["rule"] == "BR-15" and v["target"] == activity["id"] and "locked item" in v["message"] for v in after["plan"]["violations"])


async def test_locked_activity_stays_put_while_the_rest_of_its_day_is_replanned(orchestrator):
    before = await orchestrator.create_trip(VIENNA)
    first_stop = next(i for i in before["plan"]["days"][1]["items"] if i["type"] == "activity")
    await orchestrator.set_lock(before["trip_id"], first_stop["id"], True)
    after = await orchestrator.revise(before["trip_id"], "Something different on day 2")
    assert after["state"] == "PRESENTED"
    kept = find_item(after["plan"], first_stop["id"])
    assert {k: v for k, v in kept.items() if k != "locked"} == {k: v for k, v in first_stop.items() if k != "locked"}
    stops_before = {i["ref"]["item_id"] for i in before["plan"]["days"][1]["items"] if i["type"] == "activity"}
    stops_after = {i["ref"]["item_id"] for i in after["plan"]["days"][1]["items"] if i["type"] == "activity"}
    assert stops_after != stops_before and first_stop["ref"]["item_id"] in stops_after


async def test_lock_and_unlock(orchestrator):
    plan = await orchestrator.create_trip(VIENNA)
    locked = await orchestrator.set_lock(plan["trip_id"], "hotel", True)
    assert locked["locked_item_ids"] == ["hotel"] and locked["version"] == 1  # locking does not create a version
    again = await orchestrator.set_lock(plan["trip_id"], "hotel", True)
    assert again["locked_item_ids"] == ["hotel"]
    unlocked = await orchestrator.set_lock(plan["trip_id"], "hotel", False)
    assert unlocked["locked_item_ids"] == [] and items_of(unlocked, "hotel")[0]["locked"] is False
