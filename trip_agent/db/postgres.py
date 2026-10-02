"""Supabase / PostgreSQL store.

Connects with a plain Postgres driver (asyncpg) using DATABASE_URL, so nothing
here is Supabase-specific. The connection string is a secret: it is read from
the environment and never logged or put in an error message.

Prepared-statement caching is switched off (`statement_cache_size=0`), so the
store works through Supabase's session pooler, its transaction pooler and a
direct connection alike.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from decimal import Decimal
from typing import Any

import asyncpg

from trip_agent.db.base import ActionRecord, LeaseLost
from trip_agent.models import Session

# trips columns that hold a Session field directly; everything else small goes in `runtime`
_COLUMNS = ("state", "request", "current_version", "locked_item_ids", "iteration_count",
            "tokens_used", "estimated_cost_usd", "repair_used", "pause")  # fmt: skip
_CHILD_TABLES = ("messages", "tool_calls", "tool_results", "versions")
_NOT_IN_RUNTIME = {"trip_id", "created_at", "updated_at", *_COLUMNS, *_CHILD_TABLES}

_LOAD = """
select t.id, t.state, t.request, t.current_version, t.locked_item_ids, t.iteration_count,
       t.tokens_used, t.estimated_cost_usd, t.repair_used, t.pause, t.runtime,
       t.created_at, t.updated_at,
       (select coalesce(jsonb_agg(m.message order by m.seq), '[]'::jsonb)
          from public.trip_messages m where m.trip_id = t.id) as messages,
       (select coalesce(jsonb_agg(to_jsonb(c) order by c.call_ref), '[]'::jsonb)
          from public.tool_calls c where c.trip_id = t.id) as calls,
       (select coalesce(jsonb_agg(to_jsonb(r)), '[]'::jsonb)
          from public.tool_results r where r.trip_id = t.id and r.superseded_at is null) as results,
       (select coalesce(jsonb_agg(to_jsonb(v) order by v.version), '[]'::jsonb)
          from public.plan_versions v where v.trip_id = t.id) as versions
  from public.trips t
 where t.id = $1
"""

_UPDATE_TRIP = """
update public.trips
   set state = $3, request = $4, current_version = $5, locked_item_ids = $6,
       iteration_count = $7, tokens_used = $8, estimated_cost_usd = $9, repair_used = $10,
       pause = $11, runtime = $12, updated_at = $13
 where id = $1 and run_id = $2
returning id
"""

_INSERT_MESSAGES = """
insert into public.trip_messages (trip_id, seq, message)
select $1, $2 + (e.ord - 1)::int, e.value
  from jsonb_array_elements($3::jsonb) with ordinality as e(value, ord)
"""

_UPSERT_CALLS = """
insert into public.tool_calls
       (trip_id, call_ref, tool, args, depends_on, status, attempts, label, llm_call_id,
        error, iteration, started_at, finished_at)
select $1, x.id, x.tool, x.args,
       array(select jsonb_array_elements_text(coalesce(x.depends_on, '[]'::jsonb))),
       x.status, x.attempts, x.label, x.llm_call_id, x.error, x.iteration, x.started_at, x.finished_at
  from jsonb_to_recordset($2::jsonb) as x(
       id text, tool text, args jsonb, depends_on jsonb, status text, attempts int, label text,
       llm_call_id text, error text, iteration int, started_at timestamptz, finished_at timestamptz)
on conflict (trip_id, call_ref) do update
   set args = excluded.args, depends_on = excluded.depends_on, status = excluded.status,
       attempts = excluded.attempts, label = excluded.label, error = excluded.error,
       started_at = excluded.started_at, finished_at = excluded.finished_at
"""

_SUPERSEDE_RESULTS = """
update public.tool_results set superseded_at = $3
 where trip_id = $1 and call_ref = any($2::text[]) and superseded_at is null
"""

_INSERT_RESULTS = """
insert into public.tool_results (trip_id, call_ref, payload, fetched_at, source, error)
select $1, x.call_id, x.payload, x.fetched_at, x.source, x.error
  from jsonb_to_recordset($2::jsonb) as x(
       call_id text, payload jsonb, fetched_at timestamptz, source text, error text)
"""

_INSERT_VERSIONS = """
insert into public.plan_versions (trip_id, version, reason, parent_version, plan, created_at)
select $1, x.version, x.reason, x.parent_version, x.plan, x.created_at
  from jsonb_to_recordset($2::jsonb) as x(
       version int, reason text, parent_version int, plan jsonb, created_at timestamptz)
"""


async def _init_connection(conn: asyncpg.Connection) -> None:
    await conn.set_type_codec("jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog")
    await conn.set_type_codec("json", encoder=json.dumps, decoder=json.loads, schema="pg_catalog")


async def connect(dsn: str | None = None) -> asyncpg.Connection:
    """One connection (used by the migration runner)."""
    conn = await asyncpg.connect(dsn or _dsn(), statement_cache_size=0, timeout=20)
    await _init_connection(conn)
    return conn


def _dsn() -> str:
    dsn = os.environ.get("DATABASE_URL")
    if not dsn or "<" in dsn or "YOUR_PASSWORD" in dsn:
        raise RuntimeError("DATABASE_URL is not set (or still holds a placeholder). Fill it in .env.")
    return dsn


class PostgresStore:
    def __init__(self, pool: asyncpg.Pool):
        self._pool = pool

    @classmethod
    async def open(cls, dsn: str | None = None, *, max_size: int = 5) -> "PostgresStore":
        pool = await asyncpg.create_pool(
            dsn or _dsn(), min_size=1, max_size=max_size, statement_cache_size=0,
            init=_init_connection, timeout=20, command_timeout=30,
        )  # fmt: skip
        return cls(pool)

    async def close(self) -> None:
        await self._pool.close()

    # -- mapping -----------------------------------------------------------

    @staticmethod
    def _trip_params(session: Session, now: datetime) -> list[Any]:
        data = session.model_dump(mode="json")
        runtime = {k: v for k, v in data.items() if k not in _NOT_IN_RUNTIME}
        return [
            data["state"], data["request"], data["current_version"], data["locked_item_ids"],
            data["iteration_count"], data["tokens_used"],
            Decimal(str(round(session.estimated_cost_usd, 4))), data["repair_used"],
            data["pause"], runtime, now,
        ]  # fmt: skip

    @staticmethod
    def _to_session(row: asyncpg.Record) -> Session:
        calls = [
            {**c, "id": c["call_ref"]}
            for c in row["calls"]
        ]
        results = {
            r["call_ref"]: {**r, "call_id": r["call_ref"]}
            for r in row["results"]
        }
        session = Session.model_validate(
            {
                **row["runtime"],
                "trip_id": str(row["id"]),
                "state": row["state"],
                "request": row["request"],
                "current_version": row["current_version"],
                "locked_item_ids": row["locked_item_ids"],
                "iteration_count": row["iteration_count"],
                "tokens_used": row["tokens_used"],
                "estimated_cost_usd": float(row["estimated_cost_usd"]),
                "repair_used": row["repair_used"],
                "pause": row["pause"],
                "messages": row["messages"],
                "tool_calls": calls,
                "tool_results": results,
                "versions": row["versions"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
        )
        session._persisted = PostgresStore._snapshot(session)
        return session

    @staticmethod
    def _snapshot(session: Session) -> dict[str, Any]:
        return {
            "messages": len(session.messages),
            "calls": {c.id: c.model_dump_json() for c in session.tool_calls},
            "results": {cid: r.fetched_at for cid, r in session.tool_results.items()},
            "versions": len(session.versions),
        }

    # -- trips -------------------------------------------------------------

    async def insert_trip(
        self, session: Session, key: str, run_id: str, now: datetime, deadline: datetime
    ) -> bool:
        params = self._trip_params(session, now)
        async with self._pool.acquire() as conn:
            try:
                async with conn.transaction():
                    await conn.execute(
                        """insert into public.trips
                               (id, run_id, state, request, current_version, locked_item_ids,
                                iteration_count, tokens_used, estimated_cost_usd, repair_used,
                                pause, runtime, updated_at, created_at, run_deadline)
                           values ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$13,$14)""",
                        session.trip_id, run_id, *params, deadline,
                    )  # fmt: skip
                    await conn.execute(
                        """insert into public.user_actions (trip_id, idempotency_key, kind, received_at)
                           values ($1, $2, 'create', $3)""",
                        session.trip_id, key, now,
                    )  # fmt: skip
            except asyncpg.UniqueViolationError:
                return False  # the key was already used; nothing was written
        session._persisted = self._snapshot(session)
        return True

    async def find_created(self, key: str) -> str | None:
        trip_id = await self._pool.fetchval(
            "select trip_id from public.user_actions where idempotency_key = $1 and kind = 'create'", key
        )
        return str(trip_id) if trip_id else None

    async def load(self, trip_id: str) -> Session | None:
        try:
            row = await self._pool.fetchrow(_LOAD, trip_id)
        except asyncpg.DataError:
            return None  # not a UUID
        return self._to_session(row) if row else None

    async def save(
        self,
        session: Session,
        run_id: str,
        now: datetime,
        action: tuple[str, dict[str, Any]] | None = None,
    ) -> None:
        seen = session._persisted or {"messages": 0, "calls": {}, "results": {}, "versions": 0}
        new_messages = [m.model_dump(mode="json") for m in session.messages[seen["messages"] :]]
        changed_calls = [
            json.loads(dumped)
            for call in session.tool_calls
            if (dumped := call.model_dump_json()) != seen["calls"].get(call.id)
        ]
        changed_results = [
            result.model_dump(mode="json")
            for call_id, result in session.tool_results.items()
            if seen["results"].get(call_id) != result.fetched_at
        ]
        new_versions = [v.model_dump(mode="json") for v in session.versions[seen["versions"] :]]

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                updated = await conn.fetchval(_UPDATE_TRIP, session.trip_id, run_id, *self._trip_params(session, now))
                if updated is None:
                    raise LeaseLost(session.trip_id)  # rolls the transaction back
                if new_messages:
                    await conn.execute(_INSERT_MESSAGES, session.trip_id, seen["messages"], new_messages)
                if changed_calls:
                    await conn.execute(_UPSERT_CALLS, session.trip_id, changed_calls)
                if changed_results:
                    refs = [r["call_id"] for r in changed_results]
                    await conn.execute(_SUPERSEDE_RESULTS, session.trip_id, refs, now)
                    await conn.execute(_INSERT_RESULTS, session.trip_id, changed_results)
                if new_versions:
                    await conn.execute(_INSERT_VERSIONS, session.trip_id, new_versions)
                if action is not None:
                    await conn.execute(
                        "update public.user_actions set result = $3 where trip_id = $1 and idempotency_key = $2",
                        session.trip_id, action[0], action[1],
                    )  # fmt: skip
        session.updated_at = now
        session._persisted = self._snapshot(session)

    async def delete_trip(self, trip_id: str) -> None:
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                # the pointer to a version would block deleting the versions
                await conn.execute("update public.trips set current_version = null where id = $1", trip_id)
                await conn.execute("delete from public.trips where id = $1", trip_id)

    # -- the run lease -----------------------------------------------------

    async def claim(self, trip_id: str, run_id: str, now: datetime, deadline: datetime) -> bool:
        """One atomic statement: the row lock decides which of two callers wins."""
        claimed = await self._pool.fetchval(
            """update public.trips set run_id = $2, run_deadline = $4
                where id = $1 and (run_id is null or run_deadline < $3)
            returning id""",
            trip_id, run_id, now, deadline,
        )  # fmt: skip
        return claimed is not None

    async def steal(self, trip_id: str, run_id: str, now: datetime, deadline: datetime) -> None:
        await self._pool.execute(
            "update public.trips set run_id = $2, run_deadline = $3 where id = $1", trip_id, run_id, deadline
        )

    async def release(self, trip_id: str, run_id: str) -> None:
        await self._pool.execute(
            "update public.trips set run_id = null, run_deadline = null where id = $1 and run_id = $2",
            trip_id, run_id,
        )  # fmt: skip

    # -- the action ledger (BR-16) -----------------------------------------

    async def begin_action(self, trip_id: str, key: str, kind: str, now: datetime) -> ActionRecord:
        inserted = await self._pool.fetchval(
            """insert into public.user_actions (trip_id, idempotency_key, kind, received_at)
               values ($1, $2, $3, $4)
               on conflict (trip_id, idempotency_key) do nothing
               returning id""",
            trip_id, key, kind, now,
        )  # fmt: skip
        if inserted is not None:
            return ActionRecord(kind=kind, result=None, created=True)
        existing = await self.get_action(trip_id, key)
        if existing is None:  # removed between the two statements: record it again
            return await self.begin_action(trip_id, key, kind, now)
        return existing

    async def get_action(self, trip_id: str, key: str) -> ActionRecord | None:
        row = await self._pool.fetchrow(
            "select kind, result from public.user_actions where trip_id = $1 and idempotency_key = $2",
            trip_id, key,
        )  # fmt: skip
        return ActionRecord(kind=row["kind"], result=row["result"], created=False) if row else None

    async def finish_action(self, trip_id: str, key: str, result: dict[str, Any]) -> None:
        await self._pool.execute(
            "update public.user_actions set result = $3 where trip_id = $1 and idempotency_key = $2",
            trip_id, key, result,
        )  # fmt: skip

    async def drop_action(self, trip_id: str, key: str) -> None:
        await self._pool.execute(
            "delete from public.user_actions where trip_id = $1 and idempotency_key = $2 and result is null",
            trip_id, key,
        )  # fmt: skip
