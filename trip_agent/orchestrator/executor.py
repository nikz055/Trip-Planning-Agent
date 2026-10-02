"""Runs the data-tool calls the planner asks for.

Nothing reaches a provider without passing through here (BR-03):
  * arguments are schema-validated first; an invalid call is recorded, returned
    to the planner as an error and never executed (BR-04)
  * priced searches need a complete trip request (BR-01)
  * calls with no dependencies run concurrently, dependent calls wait (BR-05)
  * a hotel search that depends on a flight search gets its check-in date from
    that flight's local arrival date (BR-05)
  * failures are retried with exponential backoff; after the last attempt the
    tool is marked unavailable for the session (BR-07)
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime
from typing import Awaitable, Callable

from pydantic import ValidationError

from trip_agent.config import Clock, Settings
from trip_agent.llm.base import LLMToolCall
from trip_agent.models import Session, ToolCall, ToolResult, missing_required_fields
from trip_agent.tools.base import ToolError, ToolProvider
from trip_agent.tools.schemas import (
    DATA_TOOL_ARGS,
    PRICED_TOOLS,
    DataToolArgs,
    SearchHotelsArgs,
    validation_message,
)

Sleep = Callable[[float], Awaitable[None]]


class ToolExecutor:
    def __init__(
        self,
        provider: ToolProvider,
        settings: Settings,
        clock: Clock,
        sleep: Sleep = asyncio.sleep,
    ):
        self.provider = provider
        self.settings = settings
        self.clock = clock
        self._sleep = sleep

    async def run_batch(
        self, session: Session, requested: list[LLMToolCall], iteration: int = 0
    ) -> list[ToolCall]:
        """Validate and execute one turn's data-tool calls; returns them in request order."""
        earlier_ids = {c.id for c in session.tool_calls}
        earlier_labels = {c.label: c.id for c in session.tool_calls if c.label}
        batch: list[ToolCall] = []
        runnable: list[tuple[ToolCall, DataToolArgs]] = []

        for req in requested:
            call = ToolCall(
                id=f"tc_{len(session.tool_calls) + 1:04d}",
                tool=req.name,
                args=req.args,
                llm_call_id=req.id,
                iteration=iteration,
            )
            session.tool_calls.append(call)
            batch.append(call)
            args = self._validate(session, call)
            if args is not None:
                call.label = args.label
                call.args = args.model_dump(mode="json", exclude_none=True)
                runnable.append((call, args))

        # Resolve depends_on (labels from this turn, or labels / call ids from earlier turns).
        batch_labels = {args.label: call.id for call, args in runnable if args.label}
        batch_ids = {call.id for call, _ in runnable}
        for call, args in runnable:
            resolved = []
            for dep in args.depends_on:
                dep_id = batch_labels.get(dep) or earlier_labels.get(dep)
                if dep_id is None and (dep in earlier_ids or dep in batch_ids):
                    dep_id = dep
                if dep_id is None or dep_id == call.id:
                    self._reject(call, f"depends_on refers to unknown call '{dep}'")
                    break
                resolved.append(dep_id)
            else:
                call.depends_on = resolved

        # Run in waves: everything whose dependencies are finished goes together.
        waiting = [(c, a) for c, a in runnable if c.status == "pending"]
        while waiting:
            unfinished = {c.id for c, _ in waiting}
            ready = [(c, a) for c, a in waiting if not unfinished & set(c.depends_on)]
            if not ready:
                for call, _ in waiting:
                    self._reject(call, "depends_on forms a cycle")
                break
            await asyncio.gather(*(self._run(session, call, args) for call, args in ready))
            waiting = [(c, a) for c, a in waiting if c.id in unfinished - {r.id for r, _ in ready}]
        return batch

    async def refetch(self, session: Session, call: ToolCall) -> ToolResult | None:
        """Run a stored call again with the same arguments (BR-10). None if it failed."""
        try:
            args = DATA_TOOL_ARGS[call.tool].model_validate(
                call.args, context={"today": self.clock.today()}
            )
        except ValidationError as exc:
            call.error = f"cannot re-fetch: {validation_message(exc)}"
            return None
        previous = session.tool_results.get(call.id)
        await self._attempt(session, call, args)
        if call.status == "ok":
            return session.tool_results[call.id]
        # keep the last good payload so the plan can still be shown, flagged stale
        call.status = "ok"
        if previous is not None:
            session.tool_results[call.id] = previous
        return None

    # -- validation --------------------------------------------------------

    def _validate(self, session: Session, call: ToolCall) -> DataToolArgs | None:
        model = DATA_TOOL_ARGS.get(call.tool)
        if model is None:
            self._reject(call, f"'{call.tool}' is not a data tool")
            return None
        if call.tool in session.unavailable_tools:
            self._reject(
                call,
                f"{call.tool} is unavailable for this session after repeated failures (BR-07); "
                "plan without it and state the gap",
            )
            return None
        if call.tool in PRICED_TOOLS:
            missing = missing_required_fields(session.request)
            if missing:
                self._reject(
                    call,
                    f"BR-01: {call.tool} needs the trip's {', '.join(missing)} first; "
                    "collect every missing field with one ask_user call",
                )
                return None
        try:
            return model.model_validate(call.args, context={"today": self.clock.today()})
        except ValidationError as exc:
            self._reject(call, f"invalid arguments, call not executed: {validation_message(exc)}")
            return None

    def _reject(self, call: ToolCall, message: str) -> None:
        call.status = "failed"
        call.error = message
        call.finished_at = self.clock.now()

    # -- execution ---------------------------------------------------------

    async def _run(self, session: Session, call: ToolCall, args: DataToolArgs) -> None:
        for dep_id in call.depends_on:
            dep = session.call(dep_id)
            if dep is None or dep.status != "ok":
                self._reject(call, f"dependency {dep_id} failed, so this call was not run")
                return
        if isinstance(args, SearchHotelsArgs) and args.check_in is None:
            derived = self._derive_check_in(session, call, args)
            if isinstance(derived, str):
                self._reject(call, derived)
                return
            args = args.model_copy(update={"check_in": derived})
            call.args = args.model_dump(mode="json", exclude_none=True)
        await self._attempt(session, call, args)

    def _derive_check_in(
        self, session: Session, call: ToolCall, args: SearchHotelsArgs
    ) -> date | str:
        """BR-05: hotel check-in is the flight's local arrival date."""
        flight_call = next(
            (
                dep
                for dep_id in call.depends_on
                if (dep := session.call(dep_id)) is not None and dep.tool == "search_flights"
            ),
            None,
        )
        if flight_call is None:
            return "check_in is missing: give a date, or depend on a search_flights call"
        offers = session.tool_results[flight_call.id].payload or []
        if args.flight_offer_id is not None:
            offers = [o for o in offers if o["offer_id"] == args.flight_offer_id]
        if not offers:
            return f"cannot derive check_in: no matching flight offer in {flight_call.id}"
        chosen = min(offers, key=lambda o: o["price"])
        check_in = datetime.fromisoformat(chosen["arrive_time"]).date()
        if args.check_out <= check_in:
            return (
                f"check_out {args.check_out} must be after the derived check_in {check_in} "
                f"(flight {chosen['offer_id']} lands {chosen['arrive_time']})"
            )
        return check_in

    async def _attempt(self, session: Session, call: ToolCall, args: DataToolArgs) -> None:
        call.started_at = self.clock.now()
        attempts = 1 + self.settings.tool_retries
        error = "not attempted"
        for attempt in range(attempts):
            if call.tool in session.unavailable_tools:
                error = f"{call.tool} became unavailable"
                break
            call.status = "running"
            call.attempts += 1
            try:
                payload = await asyncio.wait_for(
                    self.provider.call(call.tool, args), timeout=self.settings.tool_timeout_s
                )
            except asyncio.TimeoutError:
                error = f"{call.tool} timed out"
            except ToolError as exc:
                error = str(exc)
            else:
                call.status = "ok"
                call.error = None
                call.finished_at = self.clock.now()
                session.tool_results[call.id] = ToolResult(
                    call_id=call.id,
                    payload=payload,
                    fetched_at=call.finished_at,
                    source=self.provider.source,
                )
                return
            if attempt < attempts - 1:
                await self._sleep(self.settings.retry_backoff_s * 2**attempt)

        call.status = "failed"
        call.error = f"{error}; gave up after {call.attempts} attempt(s)"
        call.finished_at = self.clock.now()
        session.tool_results[call.id] = ToolResult(
            call_id=call.id,
            payload=None,
            fetched_at=call.finished_at,
            source=self.provider.source,
            error=call.error,
        )
        if call.tool not in session.unavailable_tools:
            session.unavailable_tools.append(call.tool)
