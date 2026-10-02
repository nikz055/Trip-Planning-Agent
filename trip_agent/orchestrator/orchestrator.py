"""The orchestrator: owns the session, checks every planner action against the
state machine and the tool schemas, runs tools, verifies plans and decides
when the loop ends. The planner proposes; this code disposes.

Durability, in three parts (see db/base.py):
  * every user action records its idempotency key first; a repeat is answered
    with the first one's result and never starts a second run (BR-16)
  * a planning run holds the trip's lease; saves are fenced on it
  * a trip left mid-run by a crash is driven forward again by whoever next
    claims the lease (`get`, or a retry of the interrupted action)
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import timedelta
from typing import Any, Awaitable, Callable

from pydantic import ValidationError

from trip_agent.config import Clock, Settings
from trip_agent.db.base import LeaseLost, TripStore
from trip_agent.llm.base import AssistantTurn, LLMClient, LLMError, LLMToolCall, Message, ToolSpec
from trip_agent.llm.prompts import PLANNER_SYSTEM_PROMPT, REVISION_SCOPE_SYSTEM_PROMPT
from trip_agent.models import (
    ALLOWANCE_TYPES,
    REQUIRED_FIELDS,
    FormField,
    GiveUp,
    Pause,
    Plan,
    PlanVersion,
    ProposalRecord,
    Revision,
    Session,
    State,
    TripRequest,
)
from trip_agent.orchestrator.budget import compute_budget
from trip_agent.orchestrator.context import (
    context_block,
    control_result,
    refreshed_block,
    tool_result_message,
    user_message,
)
from trip_agent.orchestrator.executor import ToolExecutor
from trip_agent.orchestrator.gaps import tool_gaps
from trip_agent.orchestrator.grounding import Grounded, resolve
from trip_agent.orchestrator.intake import parse_trip_request
from trip_agent.orchestrator.rules import request_errors
from trip_agent.orchestrator.state import RUNNING, TERMINAL, Event, legal_tools, transition
from trip_agent.orchestrator.verifier import VerifyContext, expected_cost, hydrate, verify
from trip_agent.orchestrator.views import build_view
from trip_agent.tools.base import ToolProvider
from trip_agent.tools.schemas import (
    CONTROL_TOOLS,
    SCOPE_REVISION_SPEC,
    AskUserArgs,
    GiveUpArgs,
    ProposePlanArgs,
    ScopeRevisionArgs,
    tool_specs,
    validation_message,
)

S, E = State, Event
View = dict[str, Any]
# States a crashed run can be left in; whoever next holds the lease drives them forward.
RESUMABLE = RUNNING | {S.NEW, S.REVISING}

FIELD_META: dict[str, tuple[str, str]] = {
    "origin": ("Departure city", "text"),
    "destination_city": ("Destination city", "text"),
    "start_date": ("Start date", "date"),
    "end_date": ("End date", "date"),
    "dates_flexible": ("My dates are flexible", "bool"),
    "adults": ("Adults", "int"),
    "children": ("Children", "int"),
    "total_budget": ("Total budget for the trip", "number"),
    "currency": ("Budget currency", "select"),
    "pace": ("Pace", "select"),
    "interests": ("Interests (comma separated)", "text"),
    "min_hotel_rating": ("Minimum hotel rating (1-5)", "number"),
}


class TripError(Exception):
    status, code = 400, "bad_request"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class NotFound(TripError):
    status, code = 404, "not_found"


class InvalidState(TripError):
    status, code = 409, "invalid_state"


class Busy(TripError):
    status, code = 409, "busy"


class PlannerUnavailable(TripError):
    status, code = 503, "planner_unavailable"


_ERRORS = {cls.code: cls for cls in (TripError, NotFound, InvalidState, Busy, PlannerUnavailable)}


class _WallClock(Exception):
    """The run's wall-clock limit ran out during an LLM call."""


class _Run:
    """One holder of a trip's lease: a planning run or a short user action."""

    def __init__(self, clock: Clock, settings: Settings):
        self.id = uuid.uuid4().hex
        self._clock, self._limit = clock, settings.wall_clock_s
        self._started = clock.monotonic()

    def elapsed(self) -> float:
        return self._clock.monotonic() - self._started

    def remaining(self) -> float:
        return self._limit - self.elapsed()


Handler = Callable[[Session, _Run], Awaitable[None]]


class Orchestrator:
    def __init__(
        self,
        store: TripStore,
        llm: LLMClient,
        provider: ToolProvider,
        settings: Settings | None = None,
        clock: Clock | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ):
        self.store, self.llm = store, llm
        self.settings = settings or Settings()
        self.clock = clock or Clock()
        self.executor = ToolExecutor(provider, self.settings, self.clock, sleep)
        self._sleep = sleep
        self.poll_s = 0.1

    # ======================================================================
    # User actions. Each one is idempotent on `key`.
    # ======================================================================

    async def create_trip(self, text: str, key: str | None = None) -> View:
        if not text or not text.strip():
            raise TripError("Describe the trip you want to plan.")
        key = self._key(key)
        existing = await self.store.find_created(key)
        if existing is None:
            now = self.clock.now()
            session = Session(
                trip_id=str(uuid.uuid4()),
                request=TripRequest(raw_text=text.strip()),
                created_at=now,
                updated_at=now,
            )
            run = _Run(self.clock, self.settings)
            if await self.store.insert_trip(session, key, run.id, now, self._lease_until(now)):
                return await self._execute(session.trip_id, key, run, self._start, kind="create")
            existing = await self.store.find_created(key)  # lost a race on the same key
        return await self._await_result(existing, key, "create", self._start)

    async def submit_form(self, trip_id: str, answers: dict[str, Any], key: str | None = None) -> View:
        return await self._action(trip_id, key, "submit_form", lambda s, r: self._submit_form(s, r, answers))

    async def revise(self, trip_id: str, instruction: str, key: str | None = None) -> View:
        return await self._action(trip_id, key, "revise", lambda s, r: self._revise(s, r, instruction))

    async def set_lock(self, trip_id: str, item_id: str, locked: bool, key: str | None = None) -> View:
        return await self._action(trip_id, key, "lock", lambda s, r: self._set_lock(s, r, item_id, locked))

    async def accept(self, trip_id: str, key: str | None = None) -> View:
        return await self._action(trip_id, key, "accept", self._accept)

    async def cancel(self, trip_id: str, key: str | None = None) -> View:
        return await self._action(trip_id, key, "cancel", self._cancel)

    async def get(self, trip_id: str) -> View:
        """Read a trip. Also the resume point: an interrupted run is driven forward,
        and stale prices are re-fetched before the plan is shown (BR-10)."""
        session = await self._load(trip_id)
        needs_work = session.state in RESUMABLE or (
            session.state in (S.PRESENTED, S.PARTIAL) and self._stale_calls(session, session.plan)
        )
        if needs_work:
            run = _Run(self.clock, self.settings)
            now = self.clock.now()
            if await self.store.claim(trip_id, run.id, now, self._lease_until(now)):
                try:
                    session = await self._load(trip_id)
                    if session.state in RESUMABLE:
                        await self._resume(session, run)
                    elif session.state in (S.PRESENTED, S.PARTIAL):
                        await self._refresh_presented(session)
                    await self._save(session, run)
                except LLMError as exc:
                    raise PlannerUnavailable(f"The planner is unavailable: {exc}") from exc
                except LeaseLost:
                    session = await self._load(trip_id)
                finally:
                    await self.store.release(trip_id, run.id)
        return self.view(session)

    async def versions(self, trip_id: str) -> list[PlanVersion]:
        return (await self._load(trip_id)).versions

    def view(self, session: Session) -> View:
        return build_view(session, self.settings, self.clock, self.llm.name)

    # ======================================================================
    # Idempotency and the lease
    # ======================================================================

    def _lease_until(self, now):
        return now + timedelta(seconds=self.settings.wall_clock_s + self.settings.lease_margin_s)

    @staticmethod
    def _key(key: str | None) -> str:
        """An idempotency key. A caller that sends none gets a fresh one (no de-duplication)."""
        if key is None:
            return uuid.uuid4().hex
        if not 8 <= len(key) <= 200:
            raise TripError("The idempotency key must be 8 to 200 characters long.")
        return key

    async def _load(self, trip_id: str) -> Session:
        session = await self.store.load(trip_id)
        if session is None:
            raise NotFound(f"No trip with id {trip_id}.")
        return session

    async def _save(self, session: Session, run: _Run) -> None:
        await self.store.save(session, run.id, self.clock.now())

    async def _action(self, trip_id: str, key: str | None, kind: str, handler: Handler) -> View:
        await self._load(trip_id)  # 404 before anything is recorded
        key = self._key(key)
        record = await self.store.begin_action(trip_id, key, kind, self.clock.now())
        if not record.created:
            if record.kind != kind:
                raise TripError("This idempotency key was already used for a different action.")
            return await self._await_result(trip_id, key, kind, handler)
        run = _Run(self.clock, self.settings)
        now = self.clock.now()
        if kind == "cancel":
            await self.store.steal(trip_id, run.id, now, self._lease_until(now))
        elif not await self.store.claim(trip_id, run.id, now, self._lease_until(now)):
            await self.store.drop_action(trip_id, key)
            raise Busy("Another request for this trip is still running. Try again in a moment.")
        return await self._execute(trip_id, key, run, handler, kind=kind)

    async def _await_result(self, trip_id: str, key: str, kind: str, handler: Handler) -> View:
        """A repeated key: return the first request's result, never start a second run.

        If the first request died (no result and nobody holds the lease), take it
        over from wherever the persisted state says it stopped.
        """
        give_up_at = self.clock.monotonic() + self.settings.wall_clock_s + self.settings.lease_margin_s
        while True:
            record = await self.store.get_action(trip_id, key)
            if record is None:  # the first attempt was rolled back (busy or planner down): start over
                return await self._action(trip_id, key, kind, handler)
            if record.result is not None:
                return self._replay(record.result)
            run = _Run(self.clock, self.settings)
            now = self.clock.now()
            if await self.store.claim(trip_id, run.id, now, self._lease_until(now)):
                finished = await self.store.get_action(trip_id, key)
                if finished is not None and finished.result is not None:
                    await self.store.release(trip_id, run.id)
                    return self._replay(finished.result)
                return await self._execute(trip_id, key, run, handler, kind=kind, takeover=True)
            if self.clock.monotonic() > give_up_at:
                raise Busy("The first request with this key is still running.")
            await self._sleep(self.poll_s)

    @staticmethod
    def _replay(result: dict[str, Any]) -> View:
        error = result.get("error")
        if error:
            raise _ERRORS.get(error["code"], TripError)(error["message"])
        return result

    async def _execute(
        self, trip_id: str, key: str, run: _Run, handler: Handler, *, kind: str, takeover: bool = False
    ) -> View:
        """Run an action while holding the lease, then release it.

        The final state and the action's result are written in one transaction,
        so an action with no result has not finished: its trip is either
        untouched (run the handler) or mid-run (drive the run forward).
        """
        try:
            session = await self._load(trip_id)
            session.notices = []
            try:
                run_handler = not (kind == "create" and session.state is not S.NEW)
                if session.state in RESUMABLE and kind != "cancel":
                    await self._resume(session, run)
                    # a taken-over action already had its effect before the crash
                    run_handler = not takeover and kind != "create" and session.state not in TERMINAL
                if run_handler:
                    try:
                        await handler(session, run)
                    except InvalidState:
                        if not takeover:
                            raise
                        # the interrupted action was already carried through (for example by
                        # a resume on read); its result is the trip as it now stands
            except TripError as exc:
                # a refusal is this action's result; a retry with the same key gets the same answer
                await self.store.finish_action(trip_id, key, {"error": {"code": exc.code, "message": exc.message}})
                raise
            session.stats.latency_s += max(0.0, run.elapsed())
            result = self.view(session)
            await self.store.save(session, run.id, self.clock.now(), action=(key, result))
            return result
        except LLMError as exc:
            # the action stays open: a retry with the same key takes it over from the saved state
            raise PlannerUnavailable(f"The planner is unavailable: {exc}") from exc
        except LeaseLost as exc:
            error = InvalidState("This trip was cancelled while the request was running.")
            await self.store.finish_action(trip_id, key, {"error": {"code": error.code, "message": error.message}})
            raise error from exc
        finally:
            await self.store.release(trip_id, run.id)

    async def _resume(self, session: Session, run: _Run) -> None:
        if session.state is S.NEW:
            await self._start(session, run)
        elif session.state is S.REVISING:
            await self._scope_and_replan(session, run)
        else:
            await self._advance(session, run)

    # ======================================================================
    # LLM access: metered, and bounded by the wall clock
    # ======================================================================

    def _llm(self, session: Session, run: _Run) -> "_MeteredLLM":
        return _MeteredLLM(self.llm, session, run)

    def _context(self, session: Session, *, final_call: bool = False) -> str:
        return context_block(session, self.settings, self.clock, final_call=final_call)

    def _hard_limit(self, session: Session, run: _Run) -> str | None:
        """A BR-06 limit after which no further LLM call is allowed."""
        settings = self.settings
        if run.elapsed() >= settings.wall_clock_s:
            return f"the wall-clock limit of {settings.wall_clock_s:g} s was reached"
        if session.tokens_used >= settings.token_budget:
            return f"the token budget of {settings.token_budget} was used up"
        if session.estimated_cost_usd >= settings.cost_cap_usd:
            return f"the estimated LLM cost reached the cap of USD {settings.cost_cap_usd:.2f}"
        return None

    def _wrap_up_reason(self, session: Session, run: _Run) -> str | None:
        """Close enough to a limit that this turn may only propose a plan or give up."""
        settings = self.settings
        if session.iteration_count >= settings.max_iterations - 1:
            return f"the limit of {settings.max_iterations} planning iterations was reached"
        if run.elapsed() >= settings.wrap_up_fraction * settings.wall_clock_s:
            return f"the wall-clock limit of {settings.wall_clock_s:g} s was nearly used up"
        return None

    # ======================================================================
    # NEW: parse the request
    # ======================================================================

    async def _start(self, session: Session, run: _Run) -> None:
        try:
            intake = await parse_trip_request(self._llm(session, run), session.request.raw_text, self.clock)
            session.request, session.guessed_fields = intake.request, intake.guessed_fields
        except _WallClock:
            pass  # nothing parsed: the form asks for everything
        errors = request_errors(session.request, self.clock.today(), self.settings)
        if errors or session.guessed_fields:
            await self._pause_for_details(session, errors)
            transition(session, E.request_incomplete, S.WAITING_FOR_DETAILS, self.clock.now())
            return
        self._begin_planning_run(session)
        session.messages.append(user_message("Plan this trip.", self._context(session)))
        transition(session, E.request_complete, S.PLANNING, self.clock.now())
        await self._save(session, run)
        await self._advance(session, run)

    def _field(self, session: Session, name: str, *, error: str | None = None, options: list[str] | None = None) -> FormField:
        label, kind = FIELD_META[name]
        value = session.request.model_dump(mode="json").get(name)
        if name == "interests":
            value = ", ".join(value or [])
        if name == "currency":
            options = sorted(self.settings.fx_to_plan_currency)
        if name == "pace":
            options = ["relaxed", "balanced", "packed"]
        if options and kind == "text":
            kind = "select"
        return FormField(
            name=name, label=label, kind=kind, required=name in REQUIRED_FIELDS,
            value=value, options=options or [], error=error,
        )  # fmt: skip

    async def _pause_for_details(self, session: Session, errors: dict[str, str]) -> None:
        """One compact form: every missing or invalid field, guesses to confirm, optional budget."""
        names = list(dict.fromkeys([*errors, *session.guessed_fields]))
        if session.request.total_budget is None:
            names += [n for n in ("total_budget", "currency") if n not in names]
        fields = [
            self._field(session, name, error=None if errors.get(name) == "Required" else errors.get(name))
            for name in names
        ]
        suggestions: list[str] = []
        if session.request.destination_city:
            # BR-01: places may be searched with only a destination, to show what the trip could include
            [call] = await self.executor.run_batch(
                session, [LLMToolCall(id="system-suggestions", name="search_places", args={"city": session.request.destination_city, "limit": 5})]
            )
            if call.status == "ok":
                suggestions = [p["name"] for p in session.tool_results[call.id].payload]
        session.pause = Pause(
            kind="missing_details",
            reason="A few details are needed before flights and hotels can be searched.",
            fields=fields,
            suggestions=suggestions,
        )
        session.stats.clarification_rounds += 1

    # ======================================================================
    # WAITING_FOR_DETAILS: the form
    # ======================================================================

    async def _submit_form(self, session: Session, run: _Run, answers: dict[str, Any]) -> None:
        if session.state is not S.WAITING_FOR_DETAILS or session.pause is None:
            raise InvalidState(f"This trip is not waiting for details (state {session.state.value}).")
        pause = session.pause
        asked = {f.name for f in pause.fields}
        cleaned: dict[str, Any] = {}
        for name, value in answers.items():
            if name not in asked:
                continue
            if isinstance(value, str):
                value = value.strip() or None
            if name == "interests" and isinstance(value, str):
                value = [part.strip() for part in value.split(",") if part.strip()]
            cleaned[name] = [] if name == "interests" and value is None else value

        errors: dict[str, str] = {}
        merged = {**session.request.model_dump(), **cleaned}
        try:
            updated: TripRequest | None = TripRequest.model_validate(merged)
        except ValidationError as exc:
            updated = None
            for err in exc.errors():
                name = str(err["loc"][0]) if err["loc"] else next(iter(asked), "form")
                errors[name] = err["msg"].removeprefix("Value error, ")
        if updated is not None:
            errors = request_errors(updated, self.clock.today(), self.settings)

        if errors:  # BR-02: field errors, and the trip stays where it is
            for form_field in pause.fields:
                if form_field.name in answers:
                    form_field.value = answers[form_field.name]
                form_field.error = errors.get(form_field.name)
            for name in errors:
                if name not in asked:
                    pause.fields.append(self._field(session, name, error=errors[name]))
            transition(session, E.form_invalid, S.WAITING_FOR_DETAILS, self.clock.now())
            return

        session.request, session.guessed_fields, session.pause = updated, [], None
        self._begin_planning_run(session)
        if pause.tool_call_id:
            body = {
                "status": "answered",
                "answers": {k: (v if isinstance(v, (int, float, bool, list)) or v is None else str(v)) for k, v in cleaned.items()},
                "note": "trip_request in the context block reflects these answers.",
            }
            session.messages.append(
                Message(role="user", text=self._context(session), tool_results=[control_result(pause.tool_call_id, body)])
            )
        else:
            session.messages.append(user_message("Plan this trip.", self._context(session)))
        transition(session, E.form_valid, S.PLANNING, self.clock.now())
        await self._save(session, run)
        await self._advance(session, run)

    # ======================================================================
    # PLANNING / VALIDATING / REPAIRING
    # ======================================================================

    @staticmethod
    def _begin_planning_run(session: Session) -> None:
        session.iteration_count = 0
        session.repair_used = False
        session.repair_turns = 0
        session.repairing_version = None
        session.candidate = None
        session.guard_hit = None

    async def _advance(self, session: Session, run: _Run) -> None:
        """Drive a run until it rests (waiting, presented, partial) or ends."""
        while session.state in RUNNING:
            try:
                if session.state is S.PLANNING:
                    await self._planning_step(session, run)
                elif session.state is S.VALIDATING:
                    await self._validate(session)
                else:
                    await self._repair_step(session, run)
            except _WallClock:
                reason = f"the wall-clock limit of {self.settings.wall_clock_s:g} s was reached"
                if session.state is S.REPAIRING:
                    self._abandon_repair(session, reason)
                else:
                    await self._stop_by_guard(session, reason)
            await self._save(session, run)

    async def _planning_step(self, session: Session, run: _Run) -> None:
        stop = self._hard_limit(session, run)
        if stop:
            await self._stop_by_guard(session, stop)
            return
        wrap_up = self._wrap_up_reason(session, run)
        if wrap_up:
            session.messages.append(
                user_message(
                    f"Planning has to finish now: {wrap_up}. Call propose_plan with the best plan the "
                    "results so far support, or call give_up.",
                    self._context(session, final_call=True),
                )
            )
        await self._planner_turn(session, run, wrap_up=bool(wrap_up))
        if wrap_up and session.state is S.PLANNING:
            await self._stop_by_guard(session, wrap_up)

    async def _repair_step(self, session: Session, run: _Run) -> None:
        stop = self._hard_limit(session, run)
        if stop is None and session.repair_turns >= self.settings.repair_max_turns:
            stop = "the repair attempt ran out of turns"
        if stop:
            self._abandon_repair(session, stop)
            return
        await self._planner_turn(session, run)

    async def _planner_turn(self, session: Session, run: _Run, *, wrap_up: bool = False) -> None:
        """One planner turn: ask the LLM, then check and carry out what it asked for."""
        state = session.state
        allowed = legal_tools(state, wrap_up=wrap_up)
        turn = await self._llm(session, run).complete(
            system=PLANNER_SYSTEM_PROMPT, messages=session.messages, tools=tool_specs(list(allowed))
        )
        if state is S.PLANNING:
            session.iteration_count += 1
        else:
            session.repair_turns += 1
        session.messages.append(turn.as_message())

        calls = turn.tool_calls
        if not calls:
            session.messages.append(user_message("Take an action by calling one of the tools.", self._context(session)))
            return
        control = [c for c in calls if c.name in CONTROL_TOOLS]
        illegal = sorted({c.name for c in calls if c.name not in allowed})
        rejection = None
        if turn.stop_reason == "max_tokens":
            rejection = "Your output was cut off. Send the complete call again."
        elif illegal:
            rejection = (
                f"{', '.join(illegal)} cannot be called now (state {state.value}). "
                f"Available: {', '.join(sorted(allowed))}."
            )
        elif control and len(calls) > 1:
            rejection = "One kind of action per turn: data tools, or a single control tool."

        if rejection:
            results = [control_result(c.id, {"status": "error", "error": rejection}, is_error=True) for c in calls]
        elif not control:
            executed = await self.executor.run_batch(session, calls, iteration=session.iteration_count)
            results = [tool_result_message(call, session) for call in executed]
            transition(session, E.data_tools, state, self.clock.now())
        else:
            results = self._control(session, control[0], wrap_up=wrap_up)
        if results:
            session.messages.append(Message(role="user", text=self._context(session), tool_results=results))

    def _control(self, session: Session, call: LLMToolCall, *, wrap_up: bool) -> list | None:
        """Handle ask_user, give_up or propose_plan. Returns results only when the call was refused."""
        now = self.clock.now()
        try:
            if call.name == "ask_user":
                return self._ask_user(session, call, AskUserArgs.model_validate(call.args))
            if call.name == "give_up":
                args = GiveUpArgs.model_validate(call.args)
                if session.state is S.REPAIRING:
                    self._abandon_repair(session, f"the planner could not repair it ({args.reason})")
                else:
                    session.give_up = GiveUp(reason=args.reason, suggestions=args.suggestions)
                    session.revision = None
                    transition(session, E.give_up, S.FAILED, now)
                return None
            plan = ProposePlanArgs.model_validate(call.args).plan
        except ValidationError as exc:
            error = f"invalid arguments, call not executed: {validation_message(exc)}"
            return [control_result(call.id, {"status": "error", "error": error}, is_error=True)]
        session.candidate, session.pending_call_id = plan, call.id
        if session.state is S.REPAIRING:
            transition(session, E.repaired_plan, S.VALIDATING, now)
        elif not wrap_up:
            transition(session, E.propose_plan, S.VALIDATING, now)
        return None  # in the wrap-up turn the caller verifies it and ends the run

    def _ask_user(self, session: Session, call: LLMToolCall, args: AskUserArgs) -> list | None:
        if session.stats.clarification_rounds >= self.settings.max_clarification_rounds:
            error = "The user has already been asked the maximum number of times. Plan with what you have, or give_up."
            return [control_result(call.id, {"status": "error", "error": error}, is_error=True)]
        # BR-01: everything that is missing is asked for at once, whatever the planner listed
        missing = [f for f in REQUIRED_FIELDS if getattr(session.request, f) is None]
        names = list(dict.fromkeys([*args.fields, *missing]))
        options = args.options or {}
        session.pause = Pause(
            kind="ask_user",
            reason=args.reason,
            fields=[self._field(session, name, options=options.get(name)) for name in names],
            tool_call_id=call.id,
        )
        session.clarifications.append({"fields": names, "reason": args.reason})
        session.stats.clarification_rounds += 1
        transition(session, E.ask_user, S.WAITING_FOR_DETAILS, self.clock.now())
        return None

    # -- verification ------------------------------------------------------

    def _verify_context(self, session: Session, *, for_proposal: bool) -> VerifyContext:
        revising = for_proposal and session.revision is not None
        return VerifyContext(
            request=session.request,
            calls=session.tool_calls,
            results=session.tool_results,
            settings=self.settings,
            now=self.clock.now(),
            baseline=session.plan if for_proposal else None,
            locked_ids=frozenset(session.locked_item_ids),
            affected_days=frozenset(session.revision.affected_days) if revising else None,
            unavailable_tools=frozenset(session.unavailable_tools),
        )

    async def _check(self, session: Session, plan: Plan, *, phase: str) -> PlanVersion:
        """Verify a proposal and save it as a new version (not yet the current one)."""
        ctx = self._verify_context(session, for_proposal=True)
        hydrate(plan, ctx)
        await self._refetch(session, self._stale_calls(session, plan))  # BR-10 before check 8
        ctx.now = self.clock.now()
        plan.violations = verify(plan, ctx)
        plan.gaps = tool_gaps(session.unavailable_tools)
        plan.budget = compute_budget(plan)
        self._note_date_shift(session, plan)
        session.stats.proposals.append(
            ProposalRecord(
                phase=phase,
                violations=plan.violations,
                ungrounded_items=sum(v.rule == "BR-08" and v.check == 2 for v in plan.violations),
            )
        )
        if phase == "repair":
            reason, parent = "repair", session.repairing_version
        else:
            reason, parent = ("revision" if session.revision else "initial"), session.current_version
        session.candidate = None
        return session.add_version(plan, reason, parent, self.clock.now())

    @staticmethod
    def _note_date_shift(session: Session, plan: Plan) -> None:
        """BR-12: a date shift is always stated, whether or not the planner mentioned it."""
        request = session.request
        if not plan.days or (plan.days[0].date, plan.days[-1].date) == (request.start_date, request.end_date):
            return
        if not any(str(plan.days[0].date) in a and str(request.start_date) in a for a in plan.assumptions):
            plan.assumptions.insert(
                0,
                f"Dates shifted from {request.start_date}–{request.end_date} to "
                f"{plan.days[0].date}–{plan.days[-1].date}.",
            )

    def _present(self, session: Session, version: PlanVersion) -> None:
        session.current_version = version.version
        session.repairing_version = None
        session.revision = None
        session.revision_origin = None

    def _answer_pending(self, session: Session, body: dict[str, Any], *, with_context: bool = False, is_error: bool = False) -> None:
        if session.pending_call_id is None:
            return
        result = control_result(session.pending_call_id, body, is_error=is_error)
        text = self._context(session) if with_context else None
        session.messages.append(Message(role="user", text=text, tool_results=[result]))
        session.pending_call_id = None

    async def _validate(self, session: Session) -> None:
        now = self.clock.now()
        phase = "repair" if session.repair_used else "initial"
        version = await self._check(session, session.candidate, phase=phase)
        violations = version.plan.violations
        if not violations:
            self._present(session, version)
            self._answer_pending(session, {"status": "presented", "version": version.version})
            if version.plan.gaps:
                transition(session, E.tool_gaps, S.PARTIAL, now)
            else:
                transition(session, E.no_violations, S.PRESENTED, now)
        elif not session.repair_used:  # BR-14: one repair attempt per proposal
            session.repair_used = True
            session.repair_turns = 0
            session.repairing_version = version.version
            transition(session, E.violations, S.REPAIRING, now)
            self._answer_pending(
                session,
                {
                    "status": "violations",
                    "violations": [v.model_dump() for v in violations],
                    "note": "You get one chance to repair. Fix each violation without touching locked items, then call propose_plan again.",
                },
                with_context=True,
                is_error=True,
            )
        else:
            self._present(session, version)
            self._answer_pending(session, {"status": "presented_with_violations", "version": version.version})
            transition(session, E.violations_after_repair, S.PARTIAL, now)

    def _abandon_repair(self, session: Session, why: str) -> None:
        """REPAIRING -> PARTIAL: show the flawed proposal with its violations listed."""
        version = session.version(session.repairing_version)
        session.notices.append(f"The plan could not be repaired: {why}.")
        self._present(session, version)
        session.candidate = None
        transition(session, E.repair_abandoned, S.PARTIAL, self.clock.now())

    async def _stop_by_guard(self, session: Session, reason: str) -> None:
        """PLANNING -> PARTIAL or FAILED because a BR-06 limit was hit."""
        now = self.clock.now()
        session.guard_hit = reason
        note = f"Planning stopped early: {reason}."
        if session.candidate is not None:  # proposed in the wrap-up turn: verify, no repair
            version = await self._check(session, session.candidate, phase="initial")
            version.plan.gaps.append(note)
            self._present(session, version)
            self._answer_pending(session, {"status": "presented_as_partial", "version": version.version})
            transition(session, E.loop_guard, S.PARTIAL, now)
        elif session.current_version is not None:
            session.notices.append(f"{note} The previous version of the plan is kept.")
            session.revision = session.revision_origin = None
            transition(session, E.loop_guard, S.PARTIAL, now)
        else:
            session.give_up = GiveUp(reason=f"{note} No plan had been proposed yet.", suggestions=["Try again"])
            transition(session, E.loop_guard, S.FAILED, now)

    # ======================================================================
    # Freshness (BR-10)
    # ======================================================================

    def _stale_calls(self, session: Session, plan: Plan | None) -> list[str]:
        if plan is None:
            return []
        limit = timedelta(minutes=self.settings.freshness_minutes)
        now = self.clock.now()
        stale = []
        for item in plan.items():
            if item.ref is None or item.type in ALLOWANCE_TYPES or item.ref.result_id in stale:
                continue
            result = session.tool_results.get(item.ref.result_id)
            if result is not None and not result.error and now - result.fetched_at > limit:
                stale.append(item.ref.result_id)
        return stale

    async def _refetch(self, session: Session, call_ids: list[str]) -> None:
        calls = [c for c in (session.call(cid) for cid in call_ids) if c is not None]
        outcomes = await asyncio.gather(*(self.executor.refetch(session, call) for call in calls))
        for call, outcome in zip(calls, outcomes):
            if outcome is not None and call.id not in session.refreshed_call_ids:
                session.refreshed_call_ids.append(call.id)

    async def _refresh_presented(self, session: Session) -> bool:
        """Re-fetch stale prices behind the shown plan. A change becomes a new version."""
        plan = session.plan
        stale = self._stale_calls(session, plan)
        if not stale:
            return False
        await self._refetch(session, stale)
        fresh = plan.model_copy(deep=True)
        calls = {c.id: c for c in session.tool_calls}
        currency = self.settings.plan_currency
        changes: list[str] = []
        for item in fresh.items():
            if item.ref is None or item.ref.result_id not in stale:
                continue
            grounded = resolve(item, calls, session.tool_results)
            if not isinstance(grounded, Grounded):
                changes.append(f"'{item.title}' is no longer available at the provider.")
                continue
            expected = expected_cost(item, grounded, session.request.travellers)
            if expected is not None and abs(expected[0] - item.cost) > 0.011:
                changes.append(
                    f"'{item.title}': price changed from {currency} {item.cost:.2f} to {currency} {expected[0]:.2f}."
                )
                item.cost = round(expected[0], 2)
        ctx = self._verify_context(session, for_proposal=False)
        violations = verify(fresh, ctx)
        if not changes and violations == plan.violations:
            return False
        fresh.violations, fresh.budget = violations, compute_budget(fresh)
        version = session.add_version(fresh, "refresh", session.current_version, self.clock.now())
        session.current_version = version.version
        session.notices += changes or ["Prices were re-checked."]
        return True

    # ======================================================================
    # PRESENTED / PARTIAL: lock, accept, revise, cancel
    # ======================================================================

    async def _set_lock(self, session: Session, run: _Run, item_id: str, locked: bool) -> None:
        if session.state not in (S.PRESENTED, S.PARTIAL):
            raise InvalidState(f"Items can be locked once a plan is shown (state {session.state.value}).")
        if session.plan.item(item_id) is None:
            raise NotFound(f"The current plan has no item '{item_id}'.")
        ids = [i for i in session.locked_item_ids if i != item_id]
        session.locked_item_ids = ids + [item_id] if locked else ids

    async def _accept(self, session: Session, run: _Run) -> None:
        if session.state is not S.PRESENTED:
            raise InvalidState(f"Only a presented plan can be accepted (state {session.state.value}).")
        if await self._refresh_presented(session):
            session.notices.append("Prices were re-checked before accepting and something changed. Review the plan and accept again.")
            return
        if session.plan.violations:
            raise InvalidState("This plan has open problems. Ask for a change that fixes them before accepting.")
        transition(session, E.accept, S.ACCEPTED, self.clock.now())

    async def _cancel(self, session: Session, run: _Run) -> None:
        if session.state in TERMINAL:
            raise InvalidState(f"This trip is already {session.state.value}.")
        session.pause = None
        transition(session, E.cancel, S.CANCELLED, self.clock.now())

    async def _revise(self, session: Session, run: _Run, instruction: str) -> None:
        if session.state not in (S.PRESENTED, S.PARTIAL):
            raise InvalidState(f"Only a shown plan can be revised (state {session.state.value}).")
        if not instruction or not instruction.strip():
            raise TripError("Say what you would like to change.")
        await self._refresh_presented(session)
        session.revision_origin = session.state
        session.revision = Revision(instruction=instruction.strip())
        transition(session, E.request_change, S.REVISING, self.clock.now())
        await self._save(session, run)
        await self._scope_and_replan(session, run)

    def _reject_change(self, session: Session, message: str) -> None:
        """REVISING -> back to where it came from, with the plan untouched."""
        session.notices.append(message)
        origin = session.revision_origin or S.PRESENTED
        session.revision = session.revision_origin = None
        transition(session, E.change_rejected, origin, self.clock.now())

    async def _scope_revision(self, session: Session, run: _Run) -> ScopeRevisionArgs | str:
        """Helper LLM step: which days does the change touch? Schema-validated, metered."""
        plan = session.plan
        outline = [
            {
                "day": day.day,
                "date": day.date.isoformat(),
                "items": [
                    {"id": i.id, "type": i.type.value, "title": i.title, "locked": i.id in session.locked_item_ids}
                    for i in day.items
                ],
            }
            for day in plan.days
        ]
        messages = [
            Message(
                role="user",
                text=f"<plan>\n{json.dumps(outline, ensure_ascii=False)}\n</plan>\n"
                f"<change_request>\n{session.revision.instruction}\n</change_request>",
            )
        ]
        item_ids = {i.id for i in plan.items()}
        llm = self._llm(session, run)
        for _ in range(2):
            turn = await llm.complete(system=REVISION_SCOPE_SYSTEM_PROMPT, messages=messages, tools=[SCOPE_REVISION_SPEC])
            call = next((c for c in turn.tool_calls if c.name == "scope_revision"), None)
            if call is None:
                messages += [turn.as_message(), Message(role="user", text="Call scope_revision.")]
                continue
            try:
                args = ScopeRevisionArgs.model_validate(call.args)
                problems = [f"day {d} is not in the plan" for d in args.affected_days if d > len(plan.days)]
                problems += [f"item '{i}' is not in the plan" for i in args.target_item_ids if i not in item_ids]
                if not problems:
                    return args
                error = "; ".join(problems)
            except ValidationError as exc:
                error = validation_message(exc)
            messages += [
                turn.as_message(),
                Message(role="user", tool_results=[control_result(call.id, {"status": "error", "error": error}, is_error=True)]),
            ]
        return "I could not work out which days that change affects. Try naming the day, for example \"on day 2\"."

    async def _scope_and_replan(self, session: Session, run: _Run) -> None:
        try:
            scope = await self._scope_revision(session, run)
        except _WallClock:
            scope = "Working out the change took too long. Please try again."
        if isinstance(scope, str):
            self._reject_change(session, scope)
            return
        locked = [i for i in scope.target_item_ids if i in session.locked_item_ids]
        if locked:
            titles = ", ".join(f"'{session.plan.item(i).title}'" for i in locked)
            self._reject_change(session, f"That change affects a locked item ({titles}). Unlock it and ask again.")
            return
        session.revision.affected_days = scope.affected_days
        self._begin_planning_run(session)
        days = ", ".join(str(d) for d in scope.affected_days)
        text = (
            f"The user asked for a change: \"{session.revision.instruction}\"\n"
            f"Replan only day(s) {days}. Keep every other day exactly as it is in previous_plan, "
            "and do not change locked items. Then call propose_plan with the whole plan."
        )
        refreshed = refreshed_block(session)
        session.refreshed_call_ids = []
        session.messages.append(user_message(f"{refreshed}\n{text}".strip(), self._context(session)))
        transition(session, E.change_scoped, S.PLANNING, self.clock.now())
        await self._save(session, run)
        await self._advance(session, run)


class _MeteredLLM:
    """Wraps the planner so every call counts toward the trip's token, cost and time limits."""

    def __init__(self, llm: LLMClient, session: Session, run: _Run):
        self._llm, self._session, self._run = llm, session, run
        self.name = llm.name

    async def complete(self, *, system: str, messages: list[Message], tools: list[ToolSpec]) -> AssistantTurn:
        remaining = self._run.remaining()
        if remaining <= 0:
            raise _WallClock()
        try:
            turn = await asyncio.wait_for(
                self._llm.complete(system=system, messages=messages, tools=tools), timeout=remaining
            )
        except asyncio.TimeoutError as exc:
            raise _WallClock() from exc
        self._session.stats.llm_calls += 1
        self._session.tokens_used += turn.input_tokens + turn.output_tokens
        self._session.estimated_cost_usd += turn.cost_usd
        return turn
