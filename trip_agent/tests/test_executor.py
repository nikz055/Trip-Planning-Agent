"""The tool executor: validation before execution, ordering, retries (BR-01, 03, 04, 05, 07)."""

import asyncio
import time
from datetime import timedelta

from trip_agent.config import Settings
from trip_agent.llm.base import LLMToolCall
from trip_agent.orchestrator.executor import ToolExecutor
from trip_agent.tests.conftest import new_session, trip_request
from trip_agent.tools.failures import FailureSpec
from trip_agent.tools.mock import MockToolProvider

FLIGHT = dict(origin="LHR", destination="VIE", depart_date="2026-11-07", adults=2)
PLACES = dict(city="Vienna")


def req(name: str, n: int = 0, **args) -> LLMToolCall:
    return LLMToolCall(id=f"llm_{name}_{n}", name=name, args=args)


class TimedProvider:
    """Wraps the mock provider and records when each call started and finished."""

    source = "mock"

    def __init__(self, inner: MockToolProvider, delay_s: float = 0.0):
        self.inner, self.delay_s = inner, delay_s
        self.spans: dict[str, tuple[float, float]] = {}

    async def call(self, tool, args):
        started = time.perf_counter()
        await asyncio.sleep(self.delay_s)
        payload = await self.inner.call(tool, args)
        self.spans[tool] = (started, time.perf_counter())
        return payload


# -- stamping and bookkeeping ------------------------------------------------


async def test_result_is_stamped_with_fetched_at_and_source(executor, session, clock):
    [call] = await executor.run_batch(session, [req("search_flights", **FLIGHT)], iteration=1)
    result = session.tool_results[call.id]
    assert call.id == "tc_0001" and call.status == "ok" and call.attempts == 1 and call.iteration == 1
    assert result.fetched_at == clock.now() and result.source == "mock" and result.error is None
    assert len(result.payload) == 20
    assert call.args["cabin"] == "economy"  # stored args are the validated ones, defaults included


async def test_call_ids_run_on_across_batches(executor, session):
    await executor.run_batch(session, [req("search_places", **PLACES)])
    second = await executor.run_batch(session, [req("search_places", 1, **PLACES), req("search_places", 2, **PLACES)])
    assert [c.id for c in session.tool_calls] == ["tc_0001", "tc_0002", "tc_0003"]
    assert [c.llm_call_id for c in second] == ["llm_search_places_1", "llm_search_places_2"]


# -- BR-04: validation before execution --------------------------------------


async def test_invalid_call_is_returned_as_an_error_and_not_executed(executor, session, provider):
    [call] = await executor.run_batch(session, [req("search_flights", **{**FLIGHT, "destination": "Austria"})])
    assert call.status == "failed" and call.attempts == 0
    assert "destination" in call.error and "not executed" in call.error
    assert provider.call_log == []
    assert call.id not in session.tool_results
    assert session.unavailable_tools == []  # a bad call does not take the tool down


async def test_unknown_tool_is_rejected(executor, session, provider):
    [call] = await executor.run_batch(session, [req("book_flight", offer_id="x")])
    assert call.status == "failed" and "not a data tool" in call.error
    assert provider.call_log == []


async def test_valid_and_invalid_calls_in_one_turn(executor, session, provider):
    good, bad = await executor.run_batch(
        session, [req("search_places", **PLACES), req("search_places", 1, city="Vienna", limit=99)]
    )
    assert (good.status, bad.status) == ("ok", "failed")
    assert len(provider.call_log) == 1


# -- BR-01: priced searches need the whole request ---------------------------


async def test_priced_search_is_blocked_until_details_are_complete(executor, provider, clock):
    session = new_session(trip_request(origin=None, start_date=None, end_date=None, adults=None), clock)
    flight, hotel, places = await executor.run_batch(
        session,
        [
            req("search_flights", **FLIGHT),
            req("search_hotels", city="Vienna", check_in="2026-11-07", check_out="2026-11-11", guests=2),
            req("search_places", **PLACES),
        ],
    )
    assert flight.status == hotel.status == "failed"
    assert "BR-01" in flight.error and "origin, start_date, end_date, adults" in flight.error
    assert places.status == "ok"  # allowed with only a destination
    assert [tool for tool, _ in provider.call_log] == ["search_places"]


# -- BR-05: parallel and dependent execution ---------------------------------


async def test_independent_calls_run_in_parallel(session, settings, clock):
    timed = TimedProvider(MockToolProvider(), delay_s=0.2)
    executor = ToolExecutor(timed, settings, clock)
    started = time.perf_counter()
    calls = await executor.run_batch(
        session,
        [req("search_flights", **FLIGHT), req("search_places", **PLACES)],
    )
    elapsed = time.perf_counter() - started
    assert all(c.status == "ok" for c in calls)
    assert elapsed < 0.35, f"two 0.2 s calls took {elapsed:.2f} s, so they did not overlap"


async def test_dependent_call_waits_for_its_dependency(session, settings, clock):
    timed = TimedProvider(MockToolProvider(), delay_s=0.1)
    executor = ToolExecutor(timed, settings, clock)
    hotel, flight = await executor.run_batch(
        session,
        [
            req("search_hotels", city="Vienna", check_out="2026-11-11", guests=2, depends_on=["outbound"]),
            req("search_flights", label="outbound", **FLIGHT),
        ],
    )
    assert hotel.status == flight.status == "ok"
    assert hotel.depends_on == [flight.id]  # label resolved to the call id
    assert timed.spans["search_hotels"][0] >= timed.spans["search_flights"][1]


async def test_hotel_check_in_is_derived_from_the_flights_arrival_date(executor, clock):
    session = new_session(trip_request(origin="Delhi"), clock)
    flight, hotel = await executor.run_batch(
        session,
        [
            req("search_flights", label="out", origin="DEL", destination="VIE", depart_date="2026-11-07", adults=2),
            req("search_hotels", city="Vienna", check_out="2026-11-11", guests=2, depends_on=["out"]),
        ],
    )
    # the cheapest DEL-VIE offer is the overnight flight, which lands on the 8th
    assert session.tool_results[flight.id].payload[0]["arrive_time"] == "2026-11-08T05:40"
    assert hotel.status == "ok" and hotel.args["check_in"] == "2026-11-08"


async def test_hotel_check_in_follows_the_named_flight_offer(executor, clock):
    session = new_session(trip_request(origin="Delhi"), clock)
    [flight] = await executor.run_batch(
        session, [req("search_flights", origin="DEL", destination="VIE", depart_date="2026-11-07", adults=2)]
    )
    same_day = session.tool_results[flight.id].payload[1]
    assert same_day["arrive_time"].startswith("2026-11-07")
    [hotel] = await executor.run_batch(  # dependency on a call from an earlier turn, by call id
        session,
        [req("search_hotels", city="Vienna", check_out="2026-11-11", guests=2, depends_on=[flight.id], flight_offer_id=same_day["offer_id"])],
    )
    assert hotel.status == "ok" and hotel.args["check_in"] == "2026-11-07"


async def test_hotel_without_check_in_or_flight_dependency_is_rejected(executor, session, provider):
    [hotel] = await executor.run_batch(session, [req("search_hotels", city="Vienna", check_out="2026-11-11", guests=2)])
    assert hotel.status == "failed" and "check_in is missing" in hotel.error
    assert provider.call_log == []


async def test_dependent_call_is_skipped_when_its_dependency_fails(executor, session, provider, failures):
    failures.set("search_flights", {"mode": "error"})
    flight, hotel = await executor.run_batch(
        session,
        [
            req("search_flights", label="out", **FLIGHT),
            req("search_hotels", city="Vienna", check_out="2026-11-11", guests=2, depends_on=["out"]),
        ],
    )
    assert flight.status == "failed" and hotel.status == "failed"
    assert flight.id in hotel.error and "not run" in hotel.error
    assert "search_hotels" not in [tool for tool, _ in provider.call_log]
    assert session.unavailable_tools == ["search_flights"]  # the hotel tool itself is fine


async def test_unknown_and_circular_dependencies_are_rejected(executor, session, provider):
    [orphan] = await executor.run_batch(session, [req("search_places", depends_on=["nope"], **PLACES)])
    assert orphan.status == "failed" and "unknown call 'nope'" in orphan.error
    a, b = await executor.run_batch(
        session,
        [
            req("search_places", 1, label="a", depends_on=["b"], **PLACES),
            req("search_places", 2, label="b", depends_on=["a"], **PLACES),
        ],
    )
    assert a.status == b.status == "failed" and "cycle" in a.error
    assert provider.call_log == []


# -- BR-07: retries, backoff, unavailability ---------------------------------


async def test_failed_call_is_retried_twice_with_exponential_backoff(provider, failures, session, clock):
    sleeps: list[float] = []

    async def record(seconds: float) -> None:
        sleeps.append(seconds)

    executor = ToolExecutor(provider, Settings(retry_backoff_s=0.5), clock, sleep=record)
    failures.set("search_flights", {"mode": "timeout", "times": 2})
    [call] = await executor.run_batch(session, [req("search_flights", **FLIGHT)])
    assert call.status == "ok" and call.attempts == 3
    assert sleeps == [0.5, 1.0]
    assert session.unavailable_tools == []


async def test_tool_is_marked_unavailable_after_the_retries_run_out(executor, session, provider, failures):
    failures.set("search_flights", {"mode": "timeout"})
    [call] = await executor.run_batch(session, [req("search_flights", **FLIGHT)])
    assert call.status == "failed" and call.attempts == 3  # one try plus two retries
    assert "timed out" in call.error and "3 attempt(s)" in call.error
    assert session.unavailable_tools == ["search_flights"]
    assert session.tool_results[call.id].error == call.error
    assert len(provider.call_log) == 3

    [later] = await executor.run_batch(session, [req("search_flights", 1, **FLIGHT)])
    assert later.status == "failed" and "unavailable" in later.error and "BR-07" in later.error
    assert len(provider.call_log) == 3  # not called again


async def test_a_real_timeout_counts_as_a_failure(provider, failures, session, clock):
    executor = ToolExecutor(provider, Settings(retry_backoff_s=0, tool_timeout_s=0.05, tool_retries=1), clock)
    failures.set("search_places", FailureSpec(mode="slow", delay_s=0.5))
    [call] = await executor.run_batch(session, [req("search_places", **PLACES)])
    assert call.status == "failed" and call.attempts == 2 and "timed out" in call.error


async def test_empty_result_is_not_a_failure(executor, session, failures):
    failures.set("search_flights", {"mode": "empty"})
    [call] = await executor.run_batch(session, [req("search_flights", **FLIGHT)])
    assert call.status == "ok" and call.attempts == 1
    assert session.tool_results[call.id].payload == []


# -- BR-10: re-fetching ------------------------------------------------------


async def test_refetch_runs_the_same_call_again_and_restamps_it(executor, session, provider, clock):
    [call] = await executor.run_batch(session, [req("search_flights", **FLIGHT)])
    first = session.tool_results[call.id]
    clock.advance(minutes=90)
    provider.price_drift["search_flights"] = 1.5
    again = await executor.refetch(session, call)
    assert again.fetched_at - first.fetched_at == timedelta(minutes=90)
    assert again.payload[0]["price"] > first.payload[0]["price"]
    assert provider.call_log[0] == provider.call_log[1]  # identical arguments
    assert [c.id for c in session.tool_calls] == ["tc_0001"]  # no new call record


async def test_failed_refetch_keeps_the_last_good_result(executor, session, failures, clock):
    [call] = await executor.run_batch(session, [req("search_flights", **FLIGHT)])
    first = session.tool_results[call.id]
    failures.set("search_flights", {"mode": "error"})
    assert await executor.refetch(session, call) is None
    assert session.tool_results[call.id] == first
    assert call.status == "ok" and call.error  # still usable, but flagged
