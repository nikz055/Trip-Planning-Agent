"""Applies supabase/migrations/*.sql and reports tables and RLS status.

    python -m trip_agent.db.migrate            apply pending migrations, then report
    python -m trip_agent.db.migrate --status   report only

The SQL files are the source of truth. Applied versions are recorded in
`supabase_migrations.schema_migrations`, the same table the Supabase CLI uses,
so `supabase db push` and the local Docker stack see them as already applied.

Reads DATABASE_URL from the environment or .env. It never prints the
connection string or any part of it.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import asyncpg

from trip_agent.db.postgres import connect
from trip_agent.env import load_env

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "supabase" / "migrations"
APP_TABLES = ("trips", "plan_versions", "tool_calls", "tool_results", "trip_messages", "user_actions")

_LEDGER = """
create schema if not exists supabase_migrations;
create table if not exists supabase_migrations.schema_migrations (
  version text not null primary key, statements text[], name text
);
"""

_STATUS = """
select c.relname as table_name,
       c.relrowsecurity as rls_enabled,
       (select count(*) from pg_policy p where p.polrelid = c.oid) as policies,
       has_table_privilege('anon', c.oid, 'select') as anon_can_select,
       has_table_privilege('authenticated', c.oid, 'select') as authenticated_can_select
  from pg_class c join pg_namespace n on n.oid = c.relnamespace
 where n.nspname = 'public' and c.relkind = 'r'
 order by c.relname
"""


async def apply_pending(conn: asyncpg.Connection) -> list[str]:
    await conn.execute(_LEDGER)
    done = {r["version"] for r in await conn.fetch("select version from supabase_migrations.schema_migrations")}
    applied = []
    for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
        version, _, name = path.stem.partition("_")
        if version in done:
            continue
        sql = path.read_text(encoding="utf-8")
        async with conn.transaction():  # a migration applies completely or not at all
            await conn.execute(sql)
            await conn.execute(
                "insert into supabase_migrations.schema_migrations (version, statements, name) values ($1, $2, $3)",
                version, [sql], name,
            )  # fmt: skip
        applied.append(path.name)
    return applied


async def status(conn: asyncpg.Connection) -> list[dict]:
    return [dict(row) for row in await conn.fetch(_STATUS)]


def report(rows: list[dict]) -> str:
    lines = [f"{'table':<16} {'RLS':<5} {'policies':<9} {'anon select':<12} authenticated select"]
    for row in rows:
        lines.append(
            f"{row['table_name']:<16} {'on' if row['rls_enabled'] else 'OFF':<5} {row['policies']:<9} "
            f"{'YES' if row['anon_can_select'] else 'no':<12} {'YES' if row['authenticated_can_select'] else 'no'}"
        )
    missing = sorted(set(APP_TABLES) - {r["table_name"] for r in rows})
    if missing:
        lines.append(f"MISSING TABLES: {', '.join(missing)}")
    unprotected = [r["table_name"] for r in rows if not r["rls_enabled"]]
    lines.append(
        "RLS is on for every table in the public schema."
        if rows and not unprotected
        else f"RLS IS OFF FOR: {', '.join(unprotected) or '(no tables found)'}"
    )
    return "\n".join(lines)


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="trip_agent.db.migrate")
    parser.add_argument("--status", action="store_true", help="report only; apply nothing")
    args = parser.parse_args(argv)
    load_env()
    try:
        conn = await connect()
    except RuntimeError as exc:
        print(exc)
        return 2
    except (OSError, asyncpg.PostgresError, asyncio.TimeoutError) as exc:
        # deliberately no details that could echo the connection string
        print(f"Could not connect to the database ({type(exc).__name__}). Check DATABASE_URL in .env.")
        return 1
    try:
        if not args.status:
            applied = await apply_pending(conn)
            print("Applied: " + (", ".join(applied) if applied else "nothing (already up to date)"))
        print(report(await status(conn)))
    finally:
        await conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
