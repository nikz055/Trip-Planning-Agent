"""The verifier: deterministic checks on every proposed plan (spec section 7).

It never calls the LLM or a tool. It reads the plan, the trip request and the
stored tool results, and returns violations the planner can act on. Each
violation names a business rule, the check number, and the item or day.

Cost rules it enforces (BR-09):
  flight    = the offer's price (already the total for the party)
  hotel     = price_per_night x nights x rooms
  activity  = entry_price x travellers
  transfer  = the travel-time tool's estimate x travellers (transit) or x cabs
              (taxi, 4 per cab); always marked "estimated"
  allowance = any amount, always marked "estimated"
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Any

from trip_agent.config import Settings
from trip_agent.models import (
    ALLOWANCE_TYPES,
    Budget,
    CostStatus,
    ItemType,
    Plan,
    PlanItem,
    ToolCall,
    ToolResult,
    TripRequest,
    Violation,
)
from trip_agent.orchestrator.budget import check_budget, compute_budget
from trip_agent.orchestrator.grounding import Grounded, resolve

WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
WEEKDAY_NAMES = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
TOLERANCE = 0.011
_WITHIN_BUDGET_CLAIM = re.compile(r"\b(?:within|under|inside)\s+(?:the\s+|your\s+)?budget\b", re.I)
TAXI_SEATS = 4


@dataclass
class VerifyContext:
    request: TripRequest
    calls: list[ToolCall]
    results: dict[str, ToolResult]
    settings: Settings
    now: datetime
    baseline: Plan | None = None  # the previous presented version
    locked_ids: frozenset[str] = frozenset()
    affected_days: frozenset[int] | None = None  # set on a revision
    unavailable_tools: frozenset[str] = frozenset()


def hydrate(plan: Plan, ctx: VerifyContext) -> None:
    """Fill what the system owns: lock flags, provider links, item order."""
    calls = {c.id: c for c in ctx.calls}
    for day in plan.days:
        for item in day.items:
            item.locked = item.id in ctx.locked_ids
            if item.type in (ItemType.flight, ItemType.hotel):
                grounded = resolve(item, calls, ctx.results)
                if isinstance(grounded, Grounded) and grounded.record:
                    item.link = grounded.record.get("booking_link") or item.link
        day.items.sort(key=lambda i: (i.start is None, i.start or datetime.min, i.id))


def verify(plan: Plan, ctx: VerifyContext) -> list[Violation]:
    return _Verifier(plan, ctx).run()


def expected_cost(item: PlanItem, grounded: Grounded, travellers: int) -> tuple[float, str] | None:
    """What an item costs according to its tool result, with the working shown.

    None for a round-trip flight offer, whose price is shared between two items.
    """
    travellers = max(1, travellers)
    if item.type is ItemType.flight:
        if "return_depart_time" in grounded.record:
            return None
        return grounded.record["price"], "the offer price, total for all travellers"
    if item.type is ItemType.hotel:
        nights = (item.end.date() - item.start.date()).days
        rooms = grounded.call.args.get("rooms", 1)
        rate = grounded.record["price_per_night"]
        return rate * nights * rooms, f"{rate:.2f} x {nights} night(s) x {rooms} room(s)"
    if item.type is ItemType.activity:
        price = grounded.record["entry_price"]
        return price * travellers, f"{price:.2f} x {travellers} traveller(s)"
    if item.type is ItemType.transfer:
        mode = grounded.call.args["mode"]
        units = {"walk": 0, "transit": travellers, "taxi": math.ceil(travellers / TAXI_SEATS)}[mode]
        return grounded.unit_cost * units, f"{grounded.unit_cost:.2f} x {units} ({mode})"
    return None


@dataclass
class _Verifier:
    plan: Plan
    ctx: VerifyContext
    violations: list[Violation] = field(default_factory=list)
    grounded: dict[str, Grounded] = field(default_factory=dict)
    timed: set[str] = field(default_factory=set)  # items with usable start and end
    arrival: datetime | None = None  # local landing time of the outbound flight
    departure: datetime | None = None  # local departure time of the return flight

    def run(self) -> list[Violation]:
        if self._structure():  # check 1
            self._grounding()  # check 2
            self._flights()
            self._opening_hours()  # check 3
            self._travel_and_buffers()  # check 4
            self._pace()  # check 5
            self._hotel()  # check 6
            self._costs_and_budget()  # check 7
            self._freshness()  # check 8
            self._locks_and_scope()  # check 9
        return self.violations

    def _add(self, rule: str, check: int, target: str, message: str) -> None:
        self.violations.append(Violation(rule=rule, check=check, target=target, message=message))

    def _items(self, kind: ItemType, *, usable: bool = True) -> list[PlanItem]:
        """Items of one type; by default only those that are grounded and have valid times."""
        return [
            i
            for i in self.plan.items()
            if i.type is kind and (not usable or (i.id in self.grounded and i.id in self.timed))
        ]

    @property
    def _first(self) -> date:
        return self.plan.days[0].date

    @property
    def _last(self) -> date:
        return self.plan.days[-1].date

    # -- check 1: schema-level structure and dates -------------------------

    def _structure(self) -> bool:
        plan, request = self.plan, self.ctx.request
        if not plan.days:
            self._add("BR-02", 1, "plan", "the plan has no days")
            return False
        seen: set[str] = set()
        for item in plan.items():
            if item.id in seen:
                self._add("BR-08", 1, item.id, f"item id '{item.id}' is used more than once; ids must be unique")
            seen.add(item.id)
        for index, day in enumerate(plan.days):
            if day.day != index + 1:
                self._add("BR-02", 1, f"day {day.day}", f"days must be numbered 1..{len(plan.days)} in order")
            if index and day.date != plan.days[index - 1].date + timedelta(days=1):
                self._add("BR-02", 1, f"day {day.day}", "day dates must be consecutive")

        if (self._first, self._last) != (request.start_date, request.end_date):
            wanted = f"{request.start_date} to {request.end_date}"
            got = f"{self._first} to {self._last}"
            if not request.dates_flexible:
                self._add(
                    "BR-12", 1, "plan",
                    f"the plan covers {got} but the trip dates are fixed at {wanted}; do not change "
                    "the dates, offer alternatives through ask_user",
                )  # fmt: skip
            else:
                shift = abs((self._first - request.start_date).days)
                same_length = self._last - self._first == request.end_date - request.start_date
                if not same_length or shift > self.ctx.settings.flex_days:
                    self._add(
                        "BR-12", 1, "plan",
                        f"the plan covers {got}; flexible dates may move by at most "
                        f"{self.ctx.settings.flex_days} days from {wanted} and must keep the trip length",
                    )  # fmt: skip
                if self._first < self.ctx.now.date():
                    self._add("BR-02", 1, "plan", f"the plan starts on {self._first}, which is in the past")

        for day in plan.days:
            for item in day.items:
                if item.day != day.day:
                    self._add("BR-02", 1, item.id, f"is listed under day {day.day} but says day {item.day}")
                if item.type in ALLOWANCE_TYPES:
                    continue
                if item.start is None or item.end is None:
                    self._add("BR-13", 1, item.id, f"{item.type.value} '{item.title}' needs a start and an end time")
                elif item.end <= item.start:
                    self._add("BR-13", 1, item.id, f"'{item.title}' ends at or before it starts")
                elif item.start.date() != day.date:
                    self._add(
                        "BR-02", 1, item.id,
                        f"'{item.title}' starts on {item.start.date()} but is listed under day {day.day} ({day.date})",
                    )  # fmt: skip
                elif item.type in (ItemType.activity, ItemType.transfer) and item.end.date() != day.date:
                    self._add("BR-13", 1, item.id, f"'{item.title}' runs past midnight")
                else:
                    self.timed.add(item.id)
        return True

    # -- check 2: grounding ------------------------------------------------

    def _grounding(self) -> None:
        calls = {c.id: c for c in self.ctx.calls}
        for item in self.plan.items():
            if item.type in ALLOWANCE_TYPES:
                if item.ref is not None:
                    self._add("BR-08", 2, item.id, "allowances carry no ref")
                continue
            outcome = resolve(item, calls, self.ctx.results)
            if isinstance(outcome, str):
                self._add("BR-08", 2, item.id, f"{item.type.value} '{item.title or item.id}' {outcome}")
            else:
                self.grounded[item.id] = outcome

    # -- flights: which is the way out, which is the way back ---------------

    @staticmethod
    def _legs(offer: dict[str, Any]) -> list[tuple[datetime, datetime]]:
        legs = [(datetime.fromisoformat(offer["depart_time"]), datetime.fromisoformat(offer["arrive_time"]))]
        if "return_depart_time" in offer:
            legs.append(
                (datetime.fromisoformat(offer["return_depart_time"]), datetime.fromisoformat(offer["return_arrive_time"]))
            )
        return legs

    def _flights(self) -> None:
        flights = sorted(self._items(ItemType.flight), key=lambda i: i.start)
        for item in flights:
            legs = self._legs(self.grounded[item.id].record)
            if (item.start, item.end) not in legs:
                depart, arrive = legs[0]
                self._add(
                    "BR-08", 2, item.id,
                    f"flight times differ from the offer, which departs {depart:%Y-%m-%d %H:%M} and "
                    f"arrives {arrive:%Y-%m-%d %H:%M} (local)",
                )  # fmt: skip
        outbound = next((f for f in flights if f.start.date() == self._first), None)
        inbound = next((f for f in reversed(flights) if f.start.date() == self._last and f is not outbound), None)
        if outbound is not None:
            self.arrival = outbound.end
        if inbound is not None:
            self.departure = inbound.start
        if "search_flights" in self.ctx.unavailable_tools:
            return  # reported as a gap instead
        if outbound is None:
            self._add("BR-09", 7, "plan", f"the plan has no outbound flight departing on {self._first}")
        if inbound is None:
            self._add("BR-09", 7, "plan", f"the plan has no return flight departing on {self._last}")
        if outbound is not None and inbound is not None:
            out_args, in_args = self.grounded[outbound.id].call.args, self.grounded[inbound.id].call.args
            round_trip = outbound.ref == inbound.ref
            reversed_route = (in_args["origin"], in_args["destination"]) == (out_args["destination"], out_args["origin"])
            if not round_trip and not reversed_route:
                self._add(
                    "BR-08", 2, inbound.id,
                    f"the return flight must go {out_args['destination']} to {out_args['origin']}, "
                    f"but this offer is {in_args['origin']} to {in_args['destination']}",
                )  # fmt: skip

    # -- check 3: opening hours and closures --------------------------------

    def _opening_hours(self) -> None:
        for item in self._items(ItemType.activity):
            place = self.grounded[item.id].record
            day = item.start.date()
            weekday = WEEKDAY_NAMES[day.weekday()]
            if day.isoformat() in place.get("closed_dates", []):
                self._add("BR-13", 3, item.id, f"{place['name']} is closed on {day}; move it to another day")
                continue
            hours = place["opening_hours"].get(WEEKDAYS[day.weekday()])
            if not hours:
                self._add(
                    "BR-13", 3, item.id,
                    f"{place['name']} is closed on {weekday}s ({day}); move it to a day it is open",
                )  # fmt: skip
                continue
            opens = datetime.combine(day, time.fromisoformat(hours[0]))
            closes = datetime.combine(day, time.fromisoformat(hours[1]))
            if item.start < opens or item.end > closes:
                self._add(
                    "BR-13", 3, item.id,
                    f"{place['name']} is open {hours[0]}-{hours[1]} on {weekday}s but is scheduled "
                    f"{item.start:%H:%M}-{item.end:%H:%M}",
                )  # fmt: skip

    # -- check 4: travel time and buffers ----------------------------------

    def _travel_and_buffers(self) -> None:
        settings = self.ctx.settings
        buffer = settings.buffer_minutes
        for item in self._items(ItemType.transfer):
            needed = self.grounded[item.id].minutes
            scheduled = (item.end - item.start).total_seconds() / 60
            if scheduled < needed:
                self._add(
                    "BR-13", 4, item.id,
                    f"'{item.title}' takes {needed} min according to the travel-time result, but only "
                    f"{scheduled:.0f} min are scheduled",
                )  # fmt: skip

        activities = self._items(ItemType.activity, usable=False)
        for item in activities:
            if item.id not in self.timed:
                continue
            if self.arrival is not None:
                earliest = self.arrival + timedelta(minutes=settings.arrival_buffer_minutes)
                if item.start < earliest:
                    self._add(
                        "BR-13", 4, item.id,
                        f"'{item.title}' starts {item.start:%Y-%m-%d %H:%M}, but the flight lands "
                        f"{self.arrival:%Y-%m-%d %H:%M}; allow {settings.arrival_buffer_minutes} min after landing",
                    )  # fmt: skip
            if self.departure is not None:
                latest = self.departure - timedelta(minutes=settings.departure_buffer_minutes)
                if item.end > latest:
                    self._add(
                        "BR-13", 4, item.id,
                        f"'{item.title}' ends {item.end:%H:%M}, but the return flight departs "
                        f"{self.departure:%H:%M}; finish {settings.departure_buffer_minutes} min before departure",
                    )  # fmt: skip

        transfers = self._items(ItemType.transfer)
        for day in self.plan.days:
            timeline = sorted(
                (i for i in day.items if i.type is ItemType.activity and i.id in self.timed),
                key=lambda i: i.start,
            )
            for before, after in zip(timeline, timeline[1:]):
                gap = (after.start - before.end).total_seconds() / 60
                if gap < 0:
                    self._add("BR-13", 4, after.id, f"'{after.title}' overlaps '{before.title}'")
                    continue
                if before.id not in self.grounded or after.id not in self.grounded:
                    continue
                origin, dest = before.ref.item_id, after.ref.item_id
                if origin == dest:
                    travel = 0
                elif "get_travel_times" in self.ctx.unavailable_tools:
                    continue  # reported as a gap instead
                else:
                    leg = next(
                        (
                            t for t in transfers
                            if t.day == day.day
                            and (self.grounded[t.id].from_id, self.grounded[t.id].to_id) == (origin, dest)
                        ),
                        None,
                    )  # fmt: skip
                    if leg is None:
                        self._add(
                            "BR-13", 4, after.id,
                            f"no travel time from '{before.title}' to '{after.title}': call get_travel_times "
                            f"and add a transfer item with ref item_id '{origin}->{dest}'",
                        )  # fmt: skip
                        continue
                    travel = self.grounded[leg.id].minutes
                    if leg.start < before.end or leg.end > after.start:
                        self._add(
                            "BR-13", 4, leg.id,
                            f"'{leg.title}' must sit between '{before.title}' and '{after.title}'",
                        )  # fmt: skip
                if gap < travel + buffer:
                    self._add(
                        "BR-13", 4, after.id,
                        f"only {gap:.0f} min between '{before.title}' and '{after.title}'; it needs "
                        f"{travel} min travel plus a {buffer} min buffer",
                    )  # fmt: skip

    # -- check 5: pace and a light arrival day ------------------------------

    def _pace(self) -> None:
        settings, request = self.ctx.settings, self.ctx.request
        cap = settings.pace_caps[request.pace.value]
        arrival_date = self.arrival.date() if self.arrival else self._first
        for day in self.plan.days:
            count = sum(i.type is ItemType.activity for i in day.items)
            if count > cap:
                self._add(
                    "BR-13", 5, f"day {day.day}",
                    f"day {day.day} has {count} activities; the {request.pace.value} pace allows at most {cap}",
                )  # fmt: skip
            elif day.date == arrival_date and count > settings.arrival_day_max_activities:
                self._add(
                    "BR-13", 5, f"day {day.day}",
                    f"day {day.day} is the arrival day and has {count} activities; keep it light, at most "
                    f"{settings.arrival_day_max_activities}",
                )  # fmt: skip

    # -- check 6: hotel covers every night; check-in follows arrival --------

    def _hotel(self) -> None:
        first_night = self.arrival.date() if self.arrival else self._first
        last_morning = self.departure.date() if self.departure else self._last
        nights = [first_night + timedelta(days=n) for n in range((last_morning - first_night).days)]
        covered: set[date] = set()
        for item in self._items(ItemType.hotel):
            grounded = self.grounded[item.id]
            hotel, args = grounded.record, grounded.call.args
            night = item.start.date()
            while night < item.end.date():
                covered.add(night)
                night += timedelta(days=1)
            searched_in, searched_out = date.fromisoformat(args["check_in"]), date.fromisoformat(args["check_out"])
            if item.start.date() < searched_in or item.end.date() > searched_out:
                self._add(
                    "BR-08", 6, item.id,
                    f"the stay {item.start.date()} to {item.end.date()} is outside the searched stay "
                    f"{searched_in} to {searched_out}; search hotels for the nights you need",
                )  # fmt: skip
            if self.arrival is not None and item.start < self.arrival:
                self._add(
                    "BR-05", 6, item.id,
                    f"check-in {item.start:%Y-%m-%d %H:%M} is before the flight lands "
                    f"({self.arrival:%Y-%m-%d %H:%M} local); check-in follows the flight's arrival, and an "
                    "overnight flight moves it to the next day",
                )  # fmt: skip
            opens = datetime.combine(item.start.date(), time.fromisoformat(hotel["check_in_time"]))
            if item.start < opens:
                self._add(
                    "BR-13", 6, item.id,
                    f"{hotel['name']} allows check-in from {hotel['check_in_time']}, but check-in is "
                    f"scheduled at {item.start:%H:%M}",
                )  # fmt: skip
        missing = [n for n in nights if n not in covered]
        if missing and "search_hotels" not in self.ctx.unavailable_tools:
            listed = ", ".join(n.isoformat() for n in missing)
            self._add("BR-13", 6, "plan", f"no hotel covers the night(s) of {listed}")

    # -- check 7: costs and budget -----------------------------------------

    def _expected_cost(self, item: PlanItem) -> tuple[float, str] | None:
        if item.type is ItemType.hotel and item.id not in self.timed:
            return None
        return expected_cost(item, self.grounded[item.id], self.ctx.request.travellers)

    def _costs_and_budget(self) -> None:
        estimated = CostStatus.estimated
        flight_groups: dict[tuple[str, str], list[PlanItem]] = {}
        for item in self.plan.items():
            if item.type in ALLOWANCE_TYPES:
                if item.cost_status is not estimated:
                    self._add("BR-09", 7, item.id, f"'{item.title}' is an allowance, so its cost must be marked estimated")
                continue
            if item.id not in self.grounded:
                continue
            if item.type is ItemType.flight:
                flight_groups.setdefault((item.ref.result_id, item.ref.item_id), []).append(item)
                continue
            if item.type is ItemType.transfer and item.cost_status is not estimated:
                self._add("BR-09", 7, item.id, f"'{item.title}': travel costs are estimates, mark it estimated")
            expected = self._expected_cost(item)
            if expected is not None and abs(item.cost - expected[0]) > TOLERANCE:
                self._add(
                    "BR-09", 7, item.id,
                    f"'{item.title}' costs {item.cost:.2f} in the plan, but the tool result gives "
                    f"{expected[0]:.2f} ({expected[1]})",
                )  # fmt: skip
        for legs in flight_groups.values():
            price = self.grounded[legs[0].id].record["price"]
            if abs(sum(leg.cost for leg in legs) - price) > TOLERANCE:
                self._add(
                    "BR-09", 7, legs[0].id,
                    f"'{legs[0].title}' costs {sum(leg.cost for leg in legs):.2f} in the plan, but the offer "
                    f"price is {price:.2f} (total for all travellers)",
                )  # fmt: skip

        if not any(i.type is ItemType.meal_allowance and i.cost > 0 for i in self.plan.items()):
            self._add("BR-09", 7, "plan", "the budget must include an estimated meals allowance")

        computed = compute_budget(self.plan)
        if self.plan.budget is not None:
            for name in Budget.model_fields:
                stated, actual = getattr(self.plan.budget, name), getattr(computed, name)
                if abs(stated - actual) > TOLERANCE:
                    self._add("BR-09", 7, "plan", f"budget.{name} is {stated:.2f} but the items add up to {actual:.2f}")

        check = check_budget(computed.total, self.ctx.request, self.ctx.settings)
        if check.within_budget is False:
            self._add(
                "BR-09", 7, "plan",
                f"the total {check.currency} {computed.total:.2f} exceeds the budget {check.currency} "
                f"{check.limit:.2f} by {check.currency} {check.overshoot:.2f}; reduce the cost if the "
                "results allow it",
            )  # fmt: skip
            for assumption in self.plan.assumptions:
                if _WITHIN_BUDGET_CLAIM.search(assumption):
                    self._add(
                        "BR-09", 7, "plan",
                        f"an assumption says the plan is within budget, but it is over by "
                        f"{check.currency} {check.overshoot:.2f}: \"{assumption[:80]}\"",
                    )  # fmt: skip

    # -- check 8: freshness -------------------------------------------------

    def _freshness(self) -> None:
        limit = timedelta(minutes=self.ctx.settings.freshness_minutes)
        for item in self.plan.items():
            grounded = self.grounded.get(item.id)
            if grounded is not None and self.ctx.now - grounded.result.fetched_at > limit:
                self._add(
                    "BR-10", 8, item.id,
                    f"'{item.title}' uses a result fetched at {grounded.result.fetched_at:%Y-%m-%d %H:%M} UTC, "
                    f"more than {self.ctx.settings.freshness_minutes} minutes ago; it must be re-fetched",
                )  # fmt: skip

    # -- check 9: locked items and the scope of a revision ------------------

    @staticmethod
    def _same(a: PlanItem, b: PlanItem) -> bool:
        return a.model_dump(exclude={"locked"}) == b.model_dump(exclude={"locked"})

    def _locks_and_scope(self) -> None:
        baseline = self.ctx.baseline
        if baseline is None:
            return
        before = {i.id: i for i in baseline.items()}
        after = {i.id: i for i in self.plan.items()}
        for item_id in sorted(self.ctx.locked_ids & set(before)):
            if item_id not in after:
                self._add("BR-15", 9, item_id, f"locked item '{before[item_id].title}' was removed; locked items cannot change")
            elif not self._same(after[item_id], before[item_id]):
                self._add("BR-15", 9, item_id, f"locked item '{before[item_id].title}' was changed; locked items cannot change")

        if self.ctx.affected_days is None:
            return
        if len(self.plan.days) != len(baseline.days):
            self._add("BR-15", 9, "plan", f"the previous version has {len(baseline.days)} days; this one has {len(self.plan.days)}")
        for old_day, new_day in zip(baseline.days, self.plan.days):
            if new_day.day in self.ctx.affected_days:
                continue
            old = {i.id: i for i in old_day.items}
            new = {i.id: i for i in new_day.items}
            changed = sorted(
                item_id
                for item_id in old.keys() | new.keys()
                if item_id not in old or item_id not in new or not self._same(old[item_id], new[item_id])
            )
            if changed or old_day.date != new_day.date:
                affected = ", ".join(str(d) for d in sorted(self.ctx.affected_days))
                self._add(
                    "BR-15", 9, f"day {new_day.day}",
                    f"day {new_day.day} is outside the requested change (day {affected}) but differs from the "
                    f"previous version in: {', '.join(changed) or 'its date'}; keep it exactly as it was",
                )  # fmt: skip
