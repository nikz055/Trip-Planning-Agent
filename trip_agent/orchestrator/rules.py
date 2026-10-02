"""Request-level business rules that are checked in code, not in the prompt."""

from __future__ import annotations

from datetime import date

from trip_agent.config import Settings
from trip_agent.models import REQUIRED_FIELDS, TripRequest


def request_errors(request: TripRequest, today: date, settings: Settings) -> dict[str, str]:
    """Field errors for BR-01 (required fields) and BR-02 (dates and trip length)."""
    errors = {name: "Required" for name in REQUIRED_FIELDS if getattr(request, name) is None}
    start, end = request.start_date, request.end_date
    if start is not None and start < today:
        errors["start_date"] = f"Start date must be today ({today}) or later"
    if start is not None and end is not None:
        if end < start:
            errors["end_date"] = "End date must be on or after the start date"
        elif not settings.min_trip_days <= (end - start).days + 1 <= settings.max_trip_days:
            errors["end_date"] = (
                f"Trips must be {settings.min_trip_days} to {settings.max_trip_days} days long; "
                f"this one is {(end - start).days + 1}"
            )
    if request.total_budget is not None and request.currency not in settings.fx_to_plan_currency:
        errors["currency"] = "Unsupported currency; use one of " + ", ".join(
            sorted(settings.fx_to_plan_currency)
        )
    return errors
