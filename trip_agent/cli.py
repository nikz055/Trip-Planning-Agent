"""Command-line interface.

One-shot, nothing stored (state lives in memory for the length of the command):

    python -m trip_agent "5 days in Vienna from London, 7-11 Nov 2026, 2 adults, budget 2500 EUR"
    python -m trip_agent plan "Plan a trip to Austria" --set origin=London --set adults=2 \
        --set start_date=2026-11-07 --set end_date=2026-11-11

Stored in the database (needs DATABASE_URL); each command is a separate
process, so a trip carries on across restarts:

    python -m trip_agent new "Plan a trip to Austria"
    python -m trip_agent answer  <trip_id> --set origin=London --set adults=2 ...
    python -m trip_agent show    <trip_id>
    python -m trip_agent revise  <trip_id> "less walking on day 2"
    python -m trip_agent lock    <trip_id> <item_id>      (--off to unlock)
    python -m trip_agent accept  <trip_id>
    python -m trip_agent cancel  <trip_id>
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from trip_agent.config import Clock, Settings
from trip_agent.db.memory import MemoryStore
from trip_agent.env import load_env
from trip_agent.llm.factory import make_llm
from trip_agent.orchestrator.orchestrator import Orchestrator, TripError
from trip_agent.render import render_view
from trip_agent.tools.failures import FailureInjector
from trip_agent.tools.mock import MockToolProvider

STORED_COMMANDS = ("new", "answer", "show", "revise", "lock", "accept", "cancel")
COMMANDS = ("plan", *STORED_COMMANDS)


def _parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--llm", choices=["auto", "openrouter", "standin"], default=None)
    common.add_argument(
        "--fail", action="append", default=[], metavar="TOOL=MODE",
        help="inject a failure into a mock tool: MODE is timeout, empty, error or slow",
    )  # fmt: skip
    common.add_argument("--key", default=None, help="idempotency key for this action")
    parser = argparse.ArgumentParser(prog="trip_agent", description="Plan a 3-7 day city trip.")
    sub = parser.add_subparsers(dest="command", required=True)

    def command(name: str, text: str) -> argparse.ArgumentParser:
        return sub.add_parser(name, help=text, parents=[common])

    def with_answers(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--set", action="append", default=[], metavar="FIELD=VALUE",
            help="answer a form field, e.g. --set origin=London",
        )  # fmt: skip

    plan = command("plan", "plan once, in memory")
    plan.add_argument("request")
    with_answers(plan)
    new = command("new", "start a stored trip")
    new.add_argument("request")
    answer = command("answer", "submit the clarification form")
    answer.add_argument("trip_id")
    with_answers(answer)
    for name, text in (("show", "show a trip"), ("accept", "accept the plan"), ("cancel", "cancel the trip")):
        command(name, text).add_argument("trip_id")
    revise = command("revise", "ask for a change")
    revise.add_argument("trip_id")
    revise.add_argument("instruction")
    lock = command("lock", "lock or unlock a plan item")
    lock.add_argument("trip_id")
    lock.add_argument("item_id")
    lock.add_argument("--off", action="store_true", help="unlock instead")
    return parser


def _pairs(items: list[str], flag: str) -> dict[str, str]:
    pairs = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise SystemExit(f"{flag} expects NAME=VALUE, got '{item}'")
        pairs[key.strip()] = value.strip()
    return pairs


async def run(argv: list[str] | None = None, clock: Clock | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] not in COMMANDS and argv[0] not in ("-h", "--help"):
        argv.insert(0, "plan")  # a bare request means "plan"
    args = _parser().parse_args(argv)
    load_env()
    settings = Settings.from_env(**({"llm_provider": args.llm} if args.llm else {}))
    failures = FailureInjector({tool: {"mode": mode} for tool, mode in _pairs(args.fail, "--fail").items()})

    store = MemoryStore()
    if args.command in STORED_COMMANDS:
        from trip_agent.db.postgres import PostgresStore

        try:
            store = await PostgresStore.open()
        except RuntimeError as exc:
            print(exc)
            return 2
        except Exception as exc:  # no details: they could echo the connection string
            print(f"Could not connect to the database ({type(exc).__name__}). Check DATABASE_URL in .env.")
            return 1

    orchestrator = Orchestrator(store, make_llm(settings), MockToolProvider(failures), settings, clock or Clock())
    try:
        if args.command == "plan":
            view = await orchestrator.create_trip(args.request)
            answers = _pairs(args.set, "--set")
            if view["state"] == "WAITING_FOR_DETAILS" and answers:
                view = await orchestrator.submit_form(view["trip_id"], answers)
        elif args.command == "new":
            view = await orchestrator.create_trip(args.request, args.key)
        elif args.command == "answer":
            view = await orchestrator.submit_form(args.trip_id, _pairs(args.set, "--set"), args.key)
        elif args.command == "show":
            view = await orchestrator.get(args.trip_id)
        elif args.command == "revise":
            view = await orchestrator.revise(args.trip_id, args.instruction, args.key)
        elif args.command == "lock":
            view = await orchestrator.set_lock(args.trip_id, args.item_id, not args.off, args.key)
        elif args.command == "accept":
            view = await orchestrator.accept(args.trip_id, args.key)
        else:
            view = await orchestrator.cancel(args.trip_id, args.key)
    except TripError as exc:
        print(f"{exc.code}: {exc.message}")
        return 1
    finally:
        if hasattr(store, "close"):
            await store.close()

    print(render_view(view))
    if view["state"] == "WAITING_FOR_DETAILS":
        return 2
    return 1 if view["state"] in ("FAILED", "CANCELLED") else 0


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(asyncio.run(run()))


if __name__ == "__main__":
    main()
