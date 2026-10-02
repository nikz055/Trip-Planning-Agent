"""The CLI, the planner factory and the .env loader."""

import pytest

from trip_agent import cli
from trip_agent.config import Settings
from trip_agent.env import load_env
from trip_agent.llm.factory import make_llm
from trip_agent.llm.standin import StandInPlanner

REQUEST = "5 days in Vienna from London, 7-11 Nov 2026, 2 adults, budget 2500 EUR"


@pytest.fixture(autouse=True)
def no_credentials(monkeypatch, tmp_path):
    """Run each test in an empty directory with no API credentials set."""
    for name in ("LLM_API_KEY", "LLM_MODEL", "LLM_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)


# -- CLI ---------------------------------------------------------------------


async def test_cli_prints_a_plan_and_labels_the_stand_in(capsys, clock):
    assert await cli.run([REQUEST, "--llm", "standin"], clock=clock) == 0
    out = capsys.readouterr().out
    assert out.startswith("Planner: STAND-IN planner (rule-based, not an LLM)")
    assert "state PRESENTED  plan version 1" in out
    assert "Day 1  2026-11-07" in out and "Day 5  2026-11-11" in out
    assert "confirmed (mock, fetched 2026-10-01 09:00 UTC)" in out  # source and fetch time (BR-10)
    assert "estimated (estimate, no source)" in out
    assert "confirmed (from tool results)" in out and "Within budget" in out
    assert "https://flights.mock.example/book/" in out


async def test_cli_never_says_within_budget_when_over(capsys, clock):
    assert await cli.run([REQUEST.replace("2500", "900"), "--llm", "standin"], clock=clock) == 0
    out = capsys.readouterr().out
    assert "state PARTIAL" in out
    assert "Over budget by EUR" in out and "Within budget" not in out
    assert "over the EUR 900.00 budget" in out  # the overshoot is stated in the assumptions
    assert "[BR-09] plan: the total EUR" in out  # and listed as a violation


async def test_cli_says_partial_when_a_search_fails(capsys, clock, monkeypatch):
    monkeypatch.setenv("TRIP_AGENT_RETRY_BACKOFF_S", "0")
    assert await cli.run([REQUEST, "--llm", "standin", "--fail", "search_flights=timeout"], clock=clock) == 0
    out = capsys.readouterr().out
    assert "PARTIAL PLAN" in out and "Flights could not be searched" in out
    assert "  flight   " not in out  # no flight line in the schedule


async def test_cli_shows_the_form_instead_of_searching(capsys, clock):
    assert await cli.run(["Plan a trip to Austria", "--llm", "standin"], clock=clock) == 2
    out = capsys.readouterr().out
    assert "state WAITING_FOR_DETAILS" in out
    for line in ("origin: Departure city", "start_date: Start date", "end_date: End date", "adults: Adults"):
        assert line in out
    assert "destination_city: Destination city [Vienna]" in out  # the guess, to confirm
    assert "Ideas for this destination: Schoenbrunn Palace" in out
    assert "Day 1" not in out


async def test_cli_set_answers_the_form(capsys, clock):
    code = await cli.run(
        ["plan", "Plan a trip to Austria", "--llm", "standin", "--set", "origin=London", "--set", "adults=2",
         "--set", "start_date=2026-11-07", "--set", "end_date=2026-11-11"],
        clock=clock,
    )  # fmt: skip
    out = capsys.readouterr().out
    assert code == 0 and "state PRESENTED" in out and "London to Vienna, 2026-11-07 to 2026-11-11" in out


async def test_cli_shows_field_errors_for_bad_answers(capsys, clock):
    code = await cli.run(
        ["Plan a trip to Austria", "--llm", "standin", "--set", "origin=London", "--set", "adults=40",
         "--set", "start_date=2026-09-01", "--set", "end_date=2026-08-01"],
        clock=clock,
    )  # fmt: skip
    out = capsys.readouterr().out
    assert code == 2 and "state WAITING_FOR_DETAILS" in out
    assert "adults: Adults [40]  ERROR: Input should be less than or equal to 9" in out


async def test_cli_rejects_past_dates(capsys, clock):
    code = await cli.run(["Vienna from London, 2026-09-01 to 2026-09-05, 2 adults", "--llm", "standin"], clock=clock)
    assert code == 2 and "ERROR: Start date must be today (2026-10-01) or later" in capsys.readouterr().out


async def test_cli_stored_commands_need_a_database(capsys, clock, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert await cli.run(["show", "00000000-0000-0000-0000-000000000000", "--llm", "standin"], clock=clock) == 2
    assert "DATABASE_URL is not set" in capsys.readouterr().out


# -- factory and env ---------------------------------------------------------


def test_factory_uses_the_stand_in_without_credentials():
    assert isinstance(make_llm(Settings(llm_provider="auto")), StandInPlanner)
    assert isinstance(make_llm(Settings(llm_provider="standin")), StandInPlanner)
    with pytest.raises(ValueError):
        make_llm(Settings(llm_provider="gpt"))


def test_stand_in_name_says_what_it_is():
    assert "STAND-IN" in StandInPlanner.name and "not an LLM" in StandInPlanner.name


def test_load_env_returns_names_only_and_does_not_override(monkeypatch, tmp_path):
    monkeypatch.setenv("TRIP_AGENT_TEST_KEEP", "from-environment")
    monkeypatch.delenv("TRIP_AGENT_TEST_NEW", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text('# comment\nTRIP_AGENT_TEST_KEEP=from-file\nTRIP_AGENT_TEST_NEW="quoted value"\nnot a pair\n')
    import os

    assert load_env(env_file) == ["TRIP_AGENT_TEST_NEW"]
    assert os.environ["TRIP_AGENT_TEST_KEEP"] == "from-environment"
    assert os.environ["TRIP_AGENT_TEST_NEW"] == "quoted value"
    assert load_env(tmp_path / "missing.env") == []
    monkeypatch.delenv("TRIP_AGENT_TEST_NEW")
