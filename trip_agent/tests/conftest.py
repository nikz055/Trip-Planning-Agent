"""Shared fixtures. The clock is frozen at 2026-10-01 09:00 UTC (a Thursday).

Orchestrator tests take the `store` fixture and so run twice: against the
in-memory store and against Postgres (the Supabase project in DATABASE_URL).
The Postgres run is skipped, visibly, when DATABASE_URL is not set. Every trip
a test creates in the database is deleted when the test ends.
"""

from __future__ import annotations

import copy
import os
import uuid
from datetime import date
from typing import Any, Callable

import pytest
import pytest_asyncio

from trip_agent.config import FakeClock, Settings
from trip_agent.db.memory import MemoryStore
from trip_agent.env import load_env
from trip_agent.llm.base import AssistantTurn, Message, ToolSpec
from trip_agent.llm.standin import StandInPlanner
from trip_agent.models import Session, TripRequest
from trip_agent.orchestrator.executor import ToolExecutor
from trip_agent.orchestrator.orchestrator import Orchestrator
from trip_agent.tools.failures import FailureInjector
from trip_agent.tools.mock import MockToolProvider

load_env(".env.test")  # the local Supabase stack, if scripts/use_local_supabase.py wrote it
load_env()

# Saturday to Wednesday; Monday 2026-11-09 is day 3.
TRIP_START, TRIP_END = date(2026, 11, 7), date(2026, 11, 11)
VIENNA = "5 days in Vienna from London, 7-11 Nov 2026, 2 adults, budget 2500 EUR"


def trip_request(**overrides) -> TripRequest:
    """A complete Vienna request; override or blank fields as needed."""
    fields = dict(
        raw_text="5 days in Vienna from London",
        origin="London",
        destination_city="Vienna",
        start_date=TRIP_START,
        end_date=TRIP_END,
        adults=2,
        total_budget=2500,
        currency="EUR",
    )
    fields.update(overrides)
    return TripRequest(**fields)


def new_session(request: TripRequest, clock: FakeClock) -> Session:
    now = clock.now()
    return Session(trip_id=str(uuid.uuid4()), request=request, created_at=now, updated_at=now)


def database_configured() -> bool:
    dsn = os.environ.get("DATABASE_URL", "")
    return bool(dsn) and "<" not in dsn and "YOUR_PASSWORD" not in dsn


requires_database = pytest.mark.skipif(
    not database_configured(), reason="DATABASE_URL is not set: Postgres tests did not run"
)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def settings() -> Settings:
    return Settings(retry_backoff_s=0.0, llm_provider="standin")


@pytest.fixture
def failures() -> FailureInjector:
    return FailureInjector()


@pytest.fixture
def provider(failures) -> MockToolProvider:
    return MockToolProvider(failures)


@pytest.fixture
def executor(provider, settings, clock) -> ToolExecutor:
    return ToolExecutor(provider, settings, clock)


@pytest.fixture
def session(clock) -> Session:
    return new_session(trip_request(), clock)


# -- stores ------------------------------------------------------------------


class _Pool:
    """One connection pool for the whole test session, opened on first use."""

    def __init__(self):
        self.store = None

    async def get(self):
        if self.store is None:
            from trip_agent.db.postgres import PostgresStore

            self.store = await PostgresStore.open(max_size=4)
        return self.store


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def pg_pool():
    pool = _Pool()
    yield pool
    if pool.store is not None:
        await pool.store.close()


class _Tracking:
    """Wraps a store and remembers the trips created through it, for cleanup."""

    def __init__(self, inner):
        self._inner = inner
        self.trip_ids: list[str] = []

    def __getattr__(self, name):
        return getattr(self._inner, name)

    async def insert_trip(self, session, *args, **kwargs):
        created = await self._inner.insert_trip(session, *args, **kwargs)
        if created:
            self.trip_ids.append(session.trip_id)
        return created


@pytest_asyncio.fixture(loop_scope="session")
async def pg_store(pg_pool):
    """The Postgres store, with cleanup. Skips when no database is configured."""
    if not database_configured():
        pytest.skip("DATABASE_URL is not set: Postgres tests did not run")
    tracking = _Tracking(await pg_pool.get())
    yield tracking
    for trip_id in tracking.trip_ids:
        await tracking.delete_trip(trip_id)


@pytest_asyncio.fixture(loop_scope="session", params=["memory", "postgres"])
async def store(request, pg_pool):
    if request.param == "memory":
        yield MemoryStore()
        return
    if not database_configured():
        pytest.skip("DATABASE_URL is not set: Postgres tests did not run")
    tracking = _Tracking(await pg_pool.get())
    yield tracking
    for trip_id in tracking.trip_ids:
        await tracking.delete_trip(trip_id)


@pytest.fixture
def make_orchestrator(store, provider, settings, clock) -> Callable[..., Orchestrator]:
    """Build an orchestrator on the test store. Pass `llm` or setting overrides."""

    def build(llm=None, store_=None, **setting_overrides: Any) -> Orchestrator:
        chosen = settings.model_copy(update=setting_overrides)
        orch = Orchestrator(store_ or store, llm or StandInPlanner(), provider, chosen, clock)
        orch.poll_s = 0.01
        return orch

    return build


@pytest.fixture
def orchestrator(make_orchestrator) -> Orchestrator:
    return make_orchestrator()


# -- planner doubles ---------------------------------------------------------


class MutatingPlanner(StandInPlanner):
    """The stand-in, except that its first `count` proposals are altered by `mutate`.

    Lets a test make the planner propose something specific and wrong, then see
    what the verifier and the repair step do with it.
    """

    name = "stand-in with injected mistakes (test double)"

    def __init__(self, mutate: Callable[[dict], None], count: int = 1):
        self._mutate, self._left = mutate, count
        self.proposals: list[dict] = []

    async def complete(self, *, system: str, messages: list[Message], tools: list[ToolSpec]) -> AssistantTurn:
        turn = await super().complete(system=system, messages=messages, tools=tools)
        for call in turn.tool_calls:
            if call.name == "propose_plan":
                if self._left > 0:
                    self._left -= 1
                    self._mutate(call.args["plan"])
                self.proposals.append(copy.deepcopy(call.args["plan"]))
        return turn


def find_item(plan: dict, item_id: str) -> dict:
    return next(i for d in plan["days"] for i in d["items"] if i["id"] == item_id)


def items_of(view: dict, kind: str | None = None) -> list[dict]:
    return [i for d in view["plan"]["days"] for i in d["items"] if kind is None or i["type"] == kind]
