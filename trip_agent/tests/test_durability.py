"""The run lease, fencing, idempotent actions and resume after a crash (BR-16).

Tests that take `store` run on both stores. Tests that take `pg_store` need
the database and are skipped, visibly, without DATABASE_URL.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from datetime import timedelta

import pytest

from trip_agent.db.base import LeaseLost
from trip_agent.llm.standin import StandInPlanner
from trip_agent.orchestrator.orchestrator import Busy, InvalidState, Orchestrator, TripError
from trip_agent.tests.conftest import VIENNA, new_session, trip_request
from trip_agent.tools.mock import MockToolProvider

FORM = {"origin": "London", "start_date": "2026-11-07", "end_date": "2026-11-11", "adults": "2"}
APP_TABLES = ("trips", "plan_versions", "tool_calls", "tool_results", "trip_messages", "user_actions")


def lease_end(clock, seconds=150):
    return clock.now() + timedelta(seconds=seconds)


async def stored_trip(store, clock, run_id="run-a"):
    session = new_session(trip_request(), clock)
    assert await store.insert_trip(session, f"create-{uuid.uuid4().hex}", run_id, clock.now(), lease_end(clock))
    return session


class Crashes(StandInPlanner):
    """Dies on its Nth planner turn, as a killed server process would."""

    def __init__(self, on_turn: int):
        self._on_turn, self.turns = on_turn, 0

    async def complete(self, *, system, messages, tools):
        if not any(t.name in ("record_request", "scope_revision") for t in tools):
            self.turns += 1
            if self.turns == self._on_turn:
                raise RuntimeError("simulated crash")
        return await super().complete(system=system, messages=messages, tools=tools)


# -- the lease -----------------------------------------------------------------


async def test_only_one_holder_at_a_time(store, clock):
    session = await stored_trip(store, clock, "run-a")  # created holding the lease
    assert not await store.claim(session.trip_id, "run-b", clock.now(), lease_end(clock))
    await store.release(session.trip_id, "run-b")  # releasing someone else's lease does nothing
    assert not await store.claim(session.trip_id, "run-b", clock.now(), lease_end(clock))
    await store.release(session.trip_id, "run-a")
    assert await store.claim(session.trip_id, "run-b", clock.now(), lease_end(clock))


async def test_an_expired_lease_can_be_reclaimed(store, clock):
    session = await stored_trip(store, clock, "crashed-run")  # never released
    clock.advance(minutes=2)
    assert not await store.claim(session.trip_id, "run-b", clock.now(), lease_end(clock))  # 120 s < 150 s
    clock.advance(minutes=1)
    assert await store.claim(session.trip_id, "run-b", clock.now(), lease_end(clock))  # expired: taken over


async def test_a_run_that_lost_its_lease_cannot_write(store, clock):
    session = await stored_trip(store, clock, "run-a")
    clock.advance(minutes=3)
    assert await store.claim(session.trip_id, "run-b", clock.now(), lease_end(clock))
    session.notices.append("written by the run that lost the lease")
    with pytest.raises(LeaseLost):
        await store.save(session, "run-a", clock.now())
    assert (await store.load(session.trip_id)).notices == []  # nothing was written


async def test_concurrent_claims_have_exactly_one_winner(pg_store, clock):
    session = await stored_trip(pg_store, clock, "run-a")
    await pg_store.release(session.trip_id, "run-a")
    outcomes = await asyncio.gather(
        *(pg_store.claim(session.trip_id, f"run-{n}", clock.now(), lease_end(clock)) for n in range(8))
    )
    assert sorted(outcomes) == [False] * 7 + [True]  # enforced by the database, not by the application


async def test_action_is_refused_while_another_run_holds_the_trip(orchestrator, store, clock):
    plan = await orchestrator.create_trip(VIENNA)
    assert await store.claim(plan["trip_id"], "someone-else", clock.now(), lease_end(clock))
    with pytest.raises(Busy):
        await orchestrator.accept(plan["trip_id"], key="accept-while-busy")
    assert (await orchestrator.get(plan["trip_id"]))["state"] == "PRESENTED"  # reads still work
    clock.advance(minutes=3)  # the other run died; its lease expires
    assert (await orchestrator.accept(plan["trip_id"], key="accept-while-busy"))["state"] == "ACCEPTED"


async def test_cancel_stops_a_run_in_flight(orchestrator, store, clock):
    plan = await orchestrator.create_trip(VIENNA)
    assert await store.claim(plan["trip_id"], "planning-run", clock.now(), lease_end(clock))
    running = await store.load(plan["trip_id"])
    assert (await orchestrator.cancel(plan["trip_id"]))["state"] == "CANCELLED"
    with pytest.raises(LeaseLost):  # the run finds out at its next save and writes nothing
        await store.save(running, "planning-run", clock.now())
    assert (await store.load(plan["trip_id"])).state.value == "CANCELLED"


# -- idempotent actions --------------------------------------------------------


async def test_repeated_key_returns_the_first_result(orchestrator, provider):
    plan = await orchestrator.create_trip(VIENNA, key="create-vienna-1")
    calls = len(provider.call_log)
    assert await orchestrator.create_trip(VIENNA, key="create-vienna-1") == plan
    first = await orchestrator.revise(plan["trip_id"], "Less walking on day 2", key="revise-0001")
    again = await orchestrator.revise(plan["trip_id"], "Less walking on day 2", key="revise-0001")
    assert again == first and first["version"] == 2  # not revised twice
    assert (await orchestrator.get(plan["trip_id"]))["version"] == 2
    assert len(provider.call_log) == calls
    assert (await orchestrator.accept(plan["trip_id"], key="accept-0001")) == (
        await orchestrator.accept(plan["trip_id"], key="accept-0001")
    )


async def test_a_refusal_is_replayed_too(orchestrator):
    waiting = await orchestrator.create_trip("Plan a trip to Austria")
    for _ in range(2):
        with pytest.raises(InvalidState, match="Only a presented plan"):
            await orchestrator.accept(waiting["trip_id"], key="accept-too-early")
    presented = await orchestrator.submit_form(waiting["trip_id"], FORM)
    assert presented["state"] == "PRESENTED"
    with pytest.raises(InvalidState):  # the key still answers with its first outcome
        await orchestrator.accept(waiting["trip_id"], key="accept-too-early")
    assert (await orchestrator.accept(waiting["trip_id"], key="accept-for-real"))["state"] == "ACCEPTED"


async def test_key_rules(orchestrator):
    plan = await orchestrator.create_trip(VIENNA)
    await orchestrator.set_lock(plan["trip_id"], "hotel", True, key="shared-key-1")
    with pytest.raises(TripError, match="different action"):
        await orchestrator.accept(plan["trip_id"], key="shared-key-1")
    with pytest.raises(TripError, match="8 to 200 characters"):
        await orchestrator.accept(plan["trip_id"], key="short")
    with pytest.raises(TripError, match="Describe the trip"):
        await orchestrator.create_trip("   ")


# -- resume after a crash ------------------------------------------------------


async def test_a_run_killed_mid_planning_is_resumed_without_repeating_searches(make_orchestrator, provider, store):
    with pytest.raises(RuntimeError, match="simulated crash"):
        await make_orchestrator(Crashes(on_turn=3)).create_trip(VIENNA, key="create-crash-1")
    trip_id = await store.find_created("create-crash-1")
    left = await store.load(trip_id)
    assert left.state.value == "PLANNING" and len(left.tool_calls) == 4  # two turns of searches were saved
    searched = len(provider.call_log)

    view = await make_orchestrator().get(trip_id)  # a new server process, reading the trip
    assert view["state"] == "PRESENTED" and view["trip_id"] == trip_id
    resumed = await store.load(trip_id)
    assert [c.id for c in resumed.tool_calls[:4]] == [c.id for c in left.tool_calls]
    assert len(provider.call_log) == searched + 2  # only the travel-time lookups that were still missing
    assert sum(c.tool == "search_flights" for c in resumed.tool_calls) == 2

    # the client's retry of the interrupted request gets this same trip, not a new one
    retried = await make_orchestrator().create_trip(VIENNA, key="create-crash-1")
    assert retried["trip_id"] == trip_id and retried["state"] == "PRESENTED"
    assert len(provider.call_log) == searched + 2


async def test_retrying_the_interrupted_request_resumes_it(make_orchestrator, provider, store):
    waiting = await make_orchestrator().create_trip("Plan a trip to Austria")
    with pytest.raises(RuntimeError):
        await make_orchestrator(Crashes(on_turn=2)).submit_form(waiting["trip_id"], FORM, key="form-crash-1")
    assert (await store.load(waiting["trip_id"])).state.value == "PLANNING"
    done = await make_orchestrator().submit_form(waiting["trip_id"], FORM, key="form-crash-1")
    assert done["state"] == "PRESENTED" and done["trip_id"] == waiting["trip_id"]
    session = await store.load(waiting["trip_id"])
    assert [t.event for t in session.history].count("form_valid") == 1  # the form was applied once
    assert sum(c.tool == "search_flights" for c in session.tool_calls) == 2


async def test_a_run_killed_while_scoping_a_revision_is_resumed(make_orchestrator, store):
    class DiesWhileScoping(StandInPlanner):
        async def complete(self, *, system, messages, tools):
            if any(t.name == "scope_revision" for t in tools):
                raise RuntimeError("simulated crash")
            return await super().complete(system=system, messages=messages, tools=tools)

    plan = await make_orchestrator().create_trip(VIENNA)
    with pytest.raises(RuntimeError):
        await make_orchestrator(DiesWhileScoping()).revise(plan["trip_id"], "Less walking on day 2", key="revise-crash")
    left = await store.load(plan["trip_id"])
    assert left.state.value == "REVISING" and left.revision.instruction == "Less walking on day 2"
    view = await make_orchestrator().get(plan["trip_id"])
    assert view["state"] == "PRESENTED" and view["version"] == 2


# -- Postgres only -------------------------------------------------------------


async def test_everything_survives_a_new_connection(pg_store, provider, settings, clock):
    from trip_agent.db.postgres import PostgresStore

    orchestrator = Orchestrator(pg_store, StandInPlanner(), provider, settings, clock)
    plan = await orchestrator.create_trip(VIENNA)
    await orchestrator.set_lock(plan["trip_id"], "hotel", True)
    revised = await orchestrator.revise(plan["trip_id"], "Less walking on day 2")
    in_memory = await pg_store.load(plan["trip_id"])

    other = await PostgresStore.open(max_size=1)
    try:
        loaded = await other.load(plan["trip_id"])
        assert loaded == in_memory  # request, messages, tool calls and results, versions, locks, counters
        assert loaded.current_version == 2 and loaded.locked_item_ids == ["hotel"]
        assert len(loaded.messages) > 8 and len(loaded.tool_calls) == 6 and len(loaded.versions) == 2
        assert all(r.fetched_at == clock.now() and r.source == "mock" for r in loaded.tool_results.values())
        shown = await Orchestrator(other, StandInPlanner(), MockToolProvider(), settings, clock).get(plan["trip_id"])
        assert shown["plan"] == revised["plan"]
    finally:
        await other.close()


async def test_re_fetch_keeps_the_old_result_as_history(pg_store, provider, settings, clock):
    orchestrator = Orchestrator(pg_store, StandInPlanner(), provider, settings, clock)
    plan = await orchestrator.create_trip(VIENNA)
    clock.advance(minutes=61)
    await orchestrator.get(plan["trip_id"])
    rows = await pg_store._pool.fetch(
        "select call_ref, superseded_at is null as current from public.tool_results where trip_id = $1", plan["trip_id"]
    )
    assert sum(r["current"] for r in rows) == 6 and sum(not r["current"] for r in rows) == 6


async def test_deleting_a_trip_removes_all_its_rows(pg_store, provider, settings, clock):
    orchestrator = Orchestrator(pg_store, StandInPlanner(), provider, settings, clock)
    plan = await orchestrator.create_trip(VIENNA)
    await pg_store.delete_trip(plan["trip_id"])
    for table in APP_TABLES:
        column = "id" if table == "trips" else "trip_id"
        assert await pg_store._pool.fetchval(f"select count(*) from public.{table} where {column} = $1", plan["trip_id"]) == 0


async def test_database_rejects_invalid_rows(pg_store, clock):
    import asyncpg

    session = await stored_trip(pg_store, clock)
    pool = pg_store._pool
    with pytest.raises(asyncpg.CheckViolationError):
        await pool.execute("update public.trips set state = 'DONE' where id = $1", session.trip_id)
    with pytest.raises(asyncpg.CheckViolationError):  # a lease always has an expiry
        await pool.execute("update public.trips set run_deadline = null where id = $1", session.trip_id)
    with pytest.raises(asyncpg.ForeignKeyViolationError):  # the pointer must name a stored version
        await pool.execute("update public.trips set current_version = 7 where id = $1", session.trip_id)
    with pytest.raises(asyncpg.UniqueViolationError):
        for _ in range(2):
            await pool.execute(
                "insert into public.user_actions (trip_id, idempotency_key, kind) values ($1, 'same-key-1', 'accept')",
                session.trip_id,
            )


async def test_rls_is_on_for_every_table_with_no_policies(pg_store):
    from trip_agent.db.migrate import status

    async with pg_store._pool.acquire() as conn:
        rows = {r["table_name"]: r for r in await status(conn)}
    assert set(APP_TABLES) <= set(rows)
    for table in APP_TABLES:
        assert rows[table]["rls_enabled"] is True and rows[table]["policies"] == 0
        assert rows[table]["anon_can_select"] is False and rows[table]["authenticated_can_select"] is False


async def test_19_the_public_key_can_read_nothing(pg_store, provider, settings, clock):
    """Edge case 19: a query using only the public (anon / publishable) key returns no rows."""
    import httpx

    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_PUBLISHABLE_KEY") or os.environ.get("SUPABASE_ANON_KEY") or ""
    if not url or not key or "<" in url + key:
        pytest.skip("SUPABASE_URL and the public key are not set: the anon-key test did not run")
    # make sure there is something to find
    plan = await Orchestrator(pg_store, StandInPlanner(), provider, settings, clock).create_trip(VIENNA)
    assert await pg_store._pool.fetchval("select count(*) from public.trips where id = $1", plan["trip_id"]) == 1

    headers = {"apikey": key}
    if not key.startswith("sb_"):  # legacy anon keys are JWTs and also go in Authorization
        headers["Authorization"] = f"Bearer {key}"
    async with httpx.AsyncClient(base_url=f"{url.rstrip('/')}/rest/v1", headers=headers, timeout=20) as client:
        for table in APP_TABLES:
            response = await client.get(f"/{table}", params={"select": "*", "limit": "5"})
            assert response.status_code < 500, f"{table}: unexpected server error"
            rows = response.json() if response.status_code == 200 else []
            assert rows == [], f"{table} is readable with the public key"
        write = await client.post("/trips", json={"request": {"raw_text": "from the public key"}})
        assert write.status_code >= 400  # and it cannot write either
    assert await pg_store._pool.fetchval("select count(*) from public.trips where request->>'raw_text' = 'from the public key'") == 0
