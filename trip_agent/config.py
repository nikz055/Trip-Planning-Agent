"""Runtime settings and the clock abstraction.

Every limit a business rule refers to lives here so that tests and the eval
harness can change it without touching orchestrator code.
"""

from __future__ import annotations

import os
import time
from datetime import date, datetime, timedelta, timezone

from pydantic import BaseModel, Field


class Settings(BaseModel):
    # BR-06 limits per planning run (tokens and cost count per trip)
    max_iterations: int = 8
    token_budget: int = 400_000
    wall_clock_s: float = 120.0
    wrap_up_fraction: float = 0.75  # past this share of the wall clock, only a wrap-up turn is allowed
    lease_margin_s: float = 30.0  # a run's lease outlives its wall clock by this much
    cost_cap_usd: float = 2.0
    repair_max_turns: int = 3
    max_clarification_rounds: int = 3

    # BR-07 retries
    tool_retries: int = 2
    retry_backoff_s: float = 0.5
    tool_timeout_s: float = 10.0

    # BR-10 freshness
    freshness_minutes: int = 60

    # BR-02 trip length
    min_trip_days: int = 1
    max_trip_days: int = 14

    # BR-12 how far flexible dates may move
    flex_days: int = 3

    # BR-13 feasibility
    buffer_minutes: int = 30
    arrival_buffer_minutes: int = 90
    departure_buffer_minutes: int = 180
    arrival_day_max_activities: int = 2
    pace_caps: dict[str, int] = Field(
        default_factory=lambda: {"relaxed": 3, "balanced": 4, "packed": 6}
    )

    # money: v1 has no live FX, so budgets are converted with static sample rates
    plan_currency: str = "EUR"
    fx_to_plan_currency: dict[str, float] = Field(
        default_factory=lambda: {"EUR": 1.0, "USD": 0.92, "GBP": 1.17, "INR": 0.011}
    )

    # LLM
    llm_provider: str = "auto"  # auto | openrouter | standin; the model itself is set by LLM_MODEL
    llm_max_tokens: int = 16000

    @classmethod
    def from_env(cls, **overrides) -> "Settings":
        """Build settings from TRIP_AGENT_* environment variables."""
        values: dict = {}
        for name, field in cls.model_fields.items():
            raw = os.environ.get(f"TRIP_AGENT_{name.upper()}")
            if raw is not None and field.annotation in (int, float, str):
                values[name] = raw
        values.update(overrides)
        return cls(**values)

    def to_plan_currency(self, amount: float | None, currency: str) -> float | None:
        """Convert a budget into the currency prices are quoted in; None if unknown."""
        if amount is None:
            return None
        rate = self.fx_to_plan_currency.get(currency.upper())
        return None if rate is None else round(amount * rate, 2)


class Clock:
    """Wall clock. Tool results are stamped with `now()`; dates use `today()`."""

    def now(self) -> datetime:
        return datetime.now(timezone.utc)

    def today(self) -> date:
        return self.now().date()

    def monotonic(self) -> float:
        """Seconds on a clock that never goes backwards; used for the wall-clock limit."""
        return time.monotonic()


class FakeClock(Clock):
    """Controllable clock for tests and the eval harness."""

    def __init__(self, start: datetime | None = None):
        self._now = start or datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc)

    def now(self) -> datetime:
        return self._now

    def monotonic(self) -> float:
        return self._now.timestamp()

    def advance(self, *, minutes: float = 0, days: float = 0) -> None:
        self._now += timedelta(minutes=minutes, days=days)
