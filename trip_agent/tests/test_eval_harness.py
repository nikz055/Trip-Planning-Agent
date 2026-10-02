"""The evaluation harness: scenario file, per-scenario metrics, and the report."""

import json

from trip_agent.config import Settings
from trip_agent.eval.harness import FLAWS, HERE, render_report, run_scenario, summarise
from trip_agent.llm.standin import StandInPlanner

SCENARIOS = {s["id"]: s for s in json.loads((HERE / "requests.json").read_text(encoding="utf-8"))}
SETTINGS = Settings(retry_backoff_s=0)


def test_scenario_file_is_well_formed():
    assert len(SCENARIOS) == 40  # "about 40 scripted requests"
    states = {"PRESENTED", "PARTIAL", "FAILED", "WAITING_FOR_DETAILS"}
    for scenario in SCENARIOS.values():
        assert scenario["request"] and scenario["category"] and set(scenario["expect"]) <= states
        assert scenario.get("flaw") in (None, *FLAWS)
    covered = {s["edge_case"] for s in SCENARIOS.values() if s.get("edge_case")}
    assert covered >= {1, 2, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 17, 20}  # the edge cases a request can express


async def test_clean_scenario_metrics():
    result = await run_scenario(SCENARIOS["vienna-london"], StandInPlanner(), SETTINGS)
    assert result["state"] == "PRESENTED" and result["success"] and result["matched"]
    assert result["violations_before_repair"] == [0] and result["violations_final"] == 0
    assert result["ungrounded_final"] == 0 and result["tool_calls"] == 6 and result["llm_calls"] == 5
    assert result["clarification_rounds"] == 0 and result["latency_s"] >= 0


async def test_injected_mistake_is_counted_before_and_after_repair():
    result = await run_scenario(SCENARIOS["flaw-invented-hotel"], StandInPlanner(), SETTINGS)
    assert result["violations_before_repair"][0] >= 1 and result["violations_final"] == 0
    assert result["repairs"] == 1 and result["ungrounded_proposed"] == 1 and result["ungrounded_final"] == 0
    twice = await run_scenario(SCENARIOS["flaw-invented-hotel-twice"], StandInPlanner(), SETTINGS)
    assert twice["state"] == "PARTIAL" and twice["ungrounded_final"] == 1 and not twice["success"] and twice["matched"]


async def test_form_and_failure_scenarios():
    form = await run_scenario(SCENARIOS["bad-dates-then-good"], StandInPlanner(), SETTINGS)
    assert form["state"] == "PRESENTED" and form["clarification_rounds"] == 1
    failed = await run_scenario(SCENARIOS["flights-timeout"], StandInPlanner(), SETTINGS)
    assert failed["state"] == "PARTIAL" and failed["gaps"] == 1 and failed["matched"] and not failed["success"]


async def test_report_says_plainly_when_the_stand_in_was_used():
    results = [await run_scenario(SCENARIOS[i], StandInPlanner(), SETTINGS) for i in ("vienna-london", "over-budget")]
    summary = summarise(results)
    assert summary["scenarios"] == 2 and summary["task_success"] == 1 and summary["matched_expectation"] == 2
    report = render_report(StandInPlanner.name, True, summary, results, "now")
    assert "not a language model" in report and "say nothing about how well a real" in report
    assert "| over-budget |" in report and "1 of 2 (50%)" in report
    assert "not a language model" not in render_report("Some model", False, summary, results, "now")
