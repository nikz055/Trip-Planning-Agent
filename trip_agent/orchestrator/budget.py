"""Budget arithmetic (BR-09). The numbers shown to the user always come from here."""

from __future__ import annotations

from pydantic import BaseModel

from trip_agent.config import Settings
from trip_agent.models import Budget, CostStatus, ItemType, Plan, TripRequest

_CATEGORY = {
    ItemType.flight: "flights",
    ItemType.hotel: "lodging",
    ItemType.transfer: "local_transport",
    ItemType.activity: "activities",
    ItemType.meal_allowance: "meals_allowance",
    ItemType.other_allowance: "other_allowance",
}


def compute_budget(plan: Plan) -> Budget:
    sums = {field: 0.0 for field in Budget.model_fields}
    for item in plan.items():
        sums[_CATEGORY[item.type]] += item.cost
        sums["total"] += item.cost
        key = "confirmed_total" if item.cost_status is CostStatus.confirmed else "estimated_total"
        sums[key] += item.cost
    return Budget(**{field: round(value, 2) for field, value in sums.items()})


class BudgetCheck(BaseModel):
    """Whether the plan fits the user's budget, decided by code and nothing else."""

    currency: str
    total: float
    limit: float | None  # the user's budget in the plan currency; None if not given
    within_budget: bool | None  # None when there is no budget to compare with
    overshoot: float = 0.0
    note: str | None = None

    @property
    def label(self) -> str:
        if self.within_budget is None:
            return "No budget given"
        if self.within_budget:
            return f"Within budget ({self.currency} {self.limit - self.total:.2f} to spare)"
        return f"Over budget by {self.currency} {self.overshoot:.2f}"


def check_budget(total: float, request: TripRequest, settings: Settings) -> BudgetCheck:
    limit = settings.to_plan_currency(request.total_budget, request.currency)
    currency = settings.plan_currency
    if request.total_budget is None:
        return BudgetCheck(currency=currency, total=total, limit=None, within_budget=None)
    if limit is None:
        return BudgetCheck(
            currency=currency,
            total=total,
            limit=None,
            within_budget=None,
            note=f"No conversion rate for {request.currency}; the budget was not checked.",
        )
    note = None
    if request.currency != currency:
        note = (
            f"Budget {request.currency} {request.total_budget:,.2f} converted to "
            f"{currency} {limit:,.2f} at a fixed sample rate."
        )
    overshoot = round(max(0.0, total - limit), 2)
    return BudgetCheck(
        currency=currency,
        total=total,
        limit=limit,
        within_budget=overshoot == 0,
        overshoot=overshoot,
        note=note,
    )
