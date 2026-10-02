"""Evaluation harness (spec section 11).

Runs the scripted requests in requests.json through the full orchestrator with
a chosen planner and writes a report.

    python -m trip_agent.eval.harness --llm standin
    python -m trip_agent.eval.harness --llm openrouter --only vienna-london,austria-only

What it measures depends on the planner. With the stand-in it measures the
guardrail pipeline (state machine, executor, verifier, repair), not planning
quality. Scenarios marked "injected planner mistake" wrap the planner so that
its first proposal contains a known fault; they show whether the verifier
catches it and whether one repair fixes it.

The date is fixed at 2026-10-01 so the sample data lines up; latency is real.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import statistics
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable

from trip_agent.config import FakeClock, Settings
from trip_agent.db.memory import MemoryStore
from trip_agent.env import load_env
from trip_agent.llm.base import AssistantTurn, LLMClient, LLMError, Message, ToolSpec
from trip_agent.llm.factory import make_llm
from trip_agent.orchestrator.orchestrator import Orchestrator, TripError
from trip_agent.tools.failures import FailureInjector
from trip_agent.tools.mock import MockToolProvider

HERE = Path(__file__).resolve().parent
EVAL_NOW = datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc)
MAX_FORM_ROUNDS = 4


class EvalClock(FakeClock):
    """A fixed calendar date with real elapsed time, so limits and latency are real."""

    def monotonic(self) -> float:
        return time.monotonic()


# -- injected planner mistakes -------------------------------------------------


def _items(plan: dict) -> list[dict]:
    return [i for d in plan["days"] for i in d["items"]]


def _monday_closed(plan: dict) -> None:
    museum = next((i for i in _items(plan) if (i.get("ref") or {}).get("item_id") == "VIE-P05"), None)
    monday = next((d for d in plan["days"] if date.fromisoformat(d["date"]).weekday() == 0), None)
    if museum and monday and museum["day"] != monday["day"]:
        plan["days"][museum["day"] - 1]["items"].remove(museum)
        museum.update(day=monday["day"], start=monday["date"] + museum["start"][10:], end=monday["date"] + museum["end"][10:])
        monday["items"].append(museum)


def _invented_hotel(plan: dict) -> None:
    for item in _items(plan):
        if item["type"] == "hotel":
            item["title"], item["ref"]["item_id"] = "Hotel Imaginary", "XX-H99"


def _bad_arithmetic(plan: dict) -> None:
    budget = plan.get("budget") or {"total": 0}
    budget["total"] = budget.get("total", 0) + 100
    plan["budget"] = budget


def _no_buffer(plan: dict) -> None:
    for day in plan["days"]:
        stops = sorted((i for i in day["items"] if i["type"] == "activity"), key=lambda i: i["start"])
        if len(stops) >= 2:
            first, second = stops[0], stops[1]
            length = datetime.fromisoformat(second["end"]) - datetime.fromisoformat(second["start"])
            start = datetime.fromisoformat(first["end"]).replace(second=0) + (datetime(1, 1, 1, 0, 5) - datetime(1, 1, 1))
            second["start"], second["end"] = start.isoformat(), (start + length).isoformat()
            return


FLAWS: dict[str, Callable[[dict], None]] = {
    "monday_closed": _monday_closed,
    "invented_hotel": _invented_hotel,
    "bad_arithmetic": _bad_arithmetic,
    "no_buffer": _no_buffer,
}


class Flawed:
    """Wraps a planner and alters its first `count` proposals with a known mistake."""

    def __init__(self, inner: LLMClient, mutate: Callable[[dict], None], count: int = 1):
        self._inner, self._mutate, self._left = inner, mutate, count
        self.name = inner.name

    async def complete(self, *, system: str, messages: list[Message], tools: list[ToolSpec]) -> AssistantTurn:
        turn = await self._inner.complete(system=system, messages=messages, tools=tools)
        for call in turn.tool_calls:
            if call.name == "propose_plan" and self._left > 0 and isinstance(call.args.get("plan"), dict):
                self._left -= 1
                try:
                    self._mutate(call.args["plan"])
                except (KeyError, TypeError, ValueError):
                    pass  # the proposal has an unexpected shape; leave it as the planner wrote it
                turn.raw = None  # the stored raw content no longer matches the altered call
        return turn


# -- one scenario --------------------------------------------------------------


async def run_scenario(scenario: dict[str, Any], llm: LLMClient, settings: Settings) -> dict[str, Any]:
    failures = FailureInjector({tool: {"mode": mode} for tool, mode in scenario.get("failures", {}).items()})
    provider = MockToolProvider(failures)
    planner: LLMClient = llm
    if scenario.get("flaw"):
        planner = Flawed(llm, FLAWS[scenario["flaw"]], scenario.get("flaw_count", 1))
    store = MemoryStore()
    orchestrator = Orchestrator(store, planner, provider, settings, EvalClock(EVAL_NOW))

    started = time.perf_counter()
    error = None
    view: dict[str, Any] | None = None
    try:
        view = await orchestrator.create_trip(scenario["request"])
        answers = list(scenario.get("answers", []))
        rounds = 0
        while view["state"] == "WAITING_FOR_DETAILS" and answers and rounds < MAX_FORM_ROUNDS:
            view = await orchestrator.submit_form(view["trip_id"], answers.pop(0))
            rounds += 1
        for item_id in scenario.get("lock", []):
            if view["state"] in ("PRESENTED", "PARTIAL") and any(i["id"] == item_id for d in view["plan"]["days"] for i in d["items"]):
                view = await orchestrator.set_lock(view["trip_id"], item_id, True)
        for instruction in scenario.get("revise", []):
            if view["state"] in ("PRESENTED", "PARTIAL"):
                view = await orchestrator.revise(view["trip_id"], instruction)
    except (TripError, LLMError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    latency = time.perf_counter() - started

    result: dict[str, Any] = {
        "id": scenario["id"],
        "category": scenario["category"],
        "edge_case": scenario.get("edge_case"),
        "expected": scenario["expect"],
        "state": view["state"] if view else "ERROR",
        "error": error,
        "latency_s": round(latency, 3),
    }
    session = await store.load(view["trip_id"]) if view else None
    if session is None:
        return {**result, "matched": False, "success": False}
    plan = view["plan"]
    proposals = session.stats.proposals
    first_tries = [p for p in proposals if p.phase == "initial"]
    result.update(
        success=view["state"] == "PRESENTED" and not plan["violations"] and not plan["gaps"],
        matched=view["state"] in scenario["expect"] and error is None,
        proposals=len(proposals),
        repairs=sum(p.phase == "repair" for p in proposals),
        violations_before_repair=[len(p.violations) for p in first_tries],
        violations_final=len(plan["violations"]) if plan else None,
        ungrounded_final=sum(v["rule"] == "BR-08" and v["check"] == 2 for v in plan["violations"]) if plan else None,
        ungrounded_proposed=sum(p.ungrounded_items for p in proposals),
        gaps=len(plan["gaps"]) if plan else None,
        tool_calls=len(session.tool_calls),
        tool_calls_rejected=sum(c.status == "failed" and c.attempts == 0 for c in session.tool_calls),
        llm_calls=session.stats.llm_calls,
        tokens=session.tokens_used,
        cost_usd=round(session.estimated_cost_usd, 4),
        clarification_rounds=session.stats.clarification_rounds,
        versions=len(session.versions),
        stopped_by=session.guard_hit,
        notices=session.notices,
    )
    forbidden = scenario.get("must_not_contain")
    if forbidden and plan and forbidden in json.dumps(plan).lower():
        result["matched"], result["error"] = False, f"the plan contains '{forbidden}'"
    return result


# -- the report ----------------------------------------------------------------


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def summarise(results: list[dict[str, Any]]) -> dict[str, Any]:
    with_plan = [r for r in results if r.get("violations_final") is not None]
    first_tries = [n for r in results for n in r.get("violations_before_repair", [])]
    repaired = [r for r in results if r.get("repairs")]
    return {
        "scenarios": len(results),
        "task_success": sum(r["success"] for r in results),
        "task_success_rate": _mean([r["success"] for r in results]),
        "matched_expectation": sum(r["matched"] for r in results),
        "plans": len(with_plan),
        "violations_per_proposal_before_repair": _mean(first_tries),
        "violations_per_plan_after_repair": _mean([r["violations_final"] for r in with_plan]),
        "repairs_attempted": len(repaired),
        "repairs_that_cleared_everything": sum(r["violations_final"] == 0 for r in repaired),
        "ungrounded_items_per_plan": _mean([r["ungrounded_final"] for r in with_plan]),
        "ungrounded_items_proposed_total": sum(r.get("ungrounded_proposed", 0) for r in results),
        "avg_tool_calls": _mean([r.get("tool_calls", 0) for r in results]),
        "avg_llm_calls": _mean([r.get("llm_calls", 0) for r in results]),
        "avg_latency_s": _mean([r["latency_s"] for r in results]),
        "max_latency_s": max((r["latency_s"] for r in results), default=0.0),
        "avg_tokens": _mean([r.get("tokens", 0) for r in results]),
        "total_cost_usd": sum(r.get("cost_usd", 0.0) for r in results),
        "avg_clarification_rounds": _mean([r.get("clarification_rounds", 0) for r in results]),
        "errors": sum(r["error"] is not None for r in results),
    }


def render_report(planner: str, is_stand_in: bool, summary: dict[str, Any], results: list[dict[str, Any]], run_at: str) -> str:
    s = summary
    lines = [
        "# Evaluation report",
        "",
        f"- **Planner:** {planner}",
        f"- **Run at:** {run_at} (scenario date fixed at {EVAL_NOW.date()})",
        f"- **Scenarios:** {s['scenarios']} scripted requests from `requests.json`, sample (mock) supplier data",
        "",
    ]
    if is_stand_in:
        lines += [
            "> **This run used the stand-in planner, which is rule-based and not a language model.**",
            "> The numbers below describe the guardrail pipeline: whether the orchestrator, executor,",
            "> verifier and repair step behave as specified. They say nothing about how well a real",
            "> model plans. Token counts are character-based estimates and cost is zero by construction.",
            "",
        ]
    lines += [
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Task success (presented, no violations, no gaps) | {s['task_success']} of {s['scenarios']} ({s['task_success_rate']:.0%}) |",
        f"| Ended in the state the scenario expects | {s['matched_expectation']} of {s['scenarios']} |",
        f"| Verifier violations per proposal, before repair | {s['violations_per_proposal_before_repair']:.2f} |",
        f"| Verifier violations per plan shown, after repair | {s['violations_per_plan_after_repair']:.2f} |",
        f"| Repairs attempted / cleared every violation | {s['repairs_attempted']} / {s['repairs_that_cleared_everything']} |",
        f"| Ungrounded items per plan shown (target 0) | {s['ungrounded_items_per_plan']:.2f} |",
        f"| Ungrounded items proposed and rejected | {s['ungrounded_items_proposed_total']} |",
        f"| Tool calls per scenario | {s['avg_tool_calls']:.1f} |",
        f"| Planner calls per scenario | {s['avg_llm_calls']:.1f} |",
        f"| Latency per scenario | {s['avg_latency_s']:.2f} s average, {s['max_latency_s']:.2f} s maximum |",
        f"| Tokens per scenario | {s['avg_tokens']:.0f} |",
        f"| Estimated LLM cost, whole run | USD {s['total_cost_usd']:.4f} |",
        f"| Clarification rounds per scenario | {s['avg_clarification_rounds']:.2f} |",
        f"| Scenarios that ended in an error | {s['errors']} |",
        "",
        "Task success is not expected to be 100%: the set deliberately includes requests that",
        "cannot end in a clean plan (supplier failures, impossible budgets, destinations with no data).",
        "\"Ended in the state the scenario expects\" is the measure of whether the system did the right thing.",
        "",
        "## By category",
        "",
        "| Category | Scenarios | Clean plan | As expected |",
        "|---|---|---|---|",
    ]
    for category in dict.fromkeys(r["category"] for r in results):
        group = [r for r in results if r["category"] == category]
        lines.append(f"| {category} | {len(group)} | {sum(r['success'] for r in group)} | {sum(r['matched'] for r in group)} |")
    lines += [
        "",
        "## Every scenario",
        "",
        "| Scenario | Edge case | Final state | Expected | OK | Violations first / final | Tool calls | Planner calls | Seconds |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        first = "+".join(str(n) for n in r.get("violations_before_repair", [])) or "-"
        final = r.get("violations_final")
        lines.append(
            f"| {r['id']} | {r['edge_case'] or ''} | {r['state']} | {' or '.join(r['expected'])} | "
            f"{'yes' if r['matched'] else '**NO**'} | {first} / {final if final is not None else '-'} | "
            f"{r.get('tool_calls', '-')} | {r.get('llm_calls', '-')} | {r['latency_s']:.2f} |"
        )
    problems = [r for r in results if not r["matched"]]
    if problems:
        lines += ["", "## Scenarios that did not end as expected", ""]
        lines += [f"- **{r['id']}**: ended {r['state']}, expected {' or '.join(r['expected'])}. {r['error'] or ''} {r.get('stopped_by') or ''}".rstrip() for r in problems]
    lines += [
        "",
        "## Not covered here",
        "",
        "The user test in spec section 11 (time to a usable plan, share of plans accepted after at most",
        "one revision, share of users who correctly tell confirmed from estimated costs) needs 5 to 10",
        "people and has not been run.",
        "",
    ]
    return "\n".join(lines)


# -- command line --------------------------------------------------------------


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="trip_agent.eval.harness")
    parser.add_argument("--llm", choices=["auto", "openrouter", "standin"], default="standin")
    parser.add_argument("--only", default="", help="comma-separated scenario ids")
    parser.add_argument("--limit", type=int, default=0, help="run only the first N scenarios")
    parser.add_argument("--out", default=None, help="report name; default report-<planner>")
    args = parser.parse_args(argv)
    load_env()
    settings = Settings.from_env(llm_provider=args.llm, retry_backoff_s=0.05)
    llm = make_llm(settings)
    is_stand_in = bool(getattr(llm, "is_stand_in", False))

    scenarios = json.loads((HERE / "requests.json").read_text(encoding="utf-8"))
    if args.only:
        wanted = {s.strip() for s in args.only.split(",")}
        scenarios = [s for s in scenarios if s["id"] in wanted]
    if args.limit:
        scenarios = scenarios[: args.limit]

    print(f"Planner: {llm.name}\nRunning {len(scenarios)} scenario(s)...")
    results = []
    for scenario in scenarios:
        result = await run_scenario(copy.deepcopy(scenario), llm, settings)
        results.append(result)
        mark = "ok " if result["matched"] else "NO "
        print(f"  {mark} {result['id']:<28} {result['state']:<20} {result['latency_s']:>7.2f}s  {result['error'] or ''}")

    summary = summarise(results)
    run_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    name = args.out or ("report-standin" if is_stand_in else "report-" + "".join(c if c.isalnum() else "-" for c in llm.name.lower()).strip("-")[:60])
    reports = HERE / "reports"
    reports.mkdir(exist_ok=True)
    (reports / f"{name}.md").write_text(render_report(llm.name, is_stand_in, summary, results, run_at), encoding="utf-8")
    (reports / f"{name}.json").write_text(
        json.dumps({"planner": llm.name, "run_at": run_at, "summary": summary, "results": results}, indent=1), encoding="utf-8"
    )
    print(
        f"\nTask success {summary['task_success']}/{summary['scenarios']}, as expected "
        f"{summary['matched_expectation']}/{summary['scenarios']}, errors {summary['errors']}"
    )
    print(f"Report: trip_agent/eval/reports/{name}.md")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
