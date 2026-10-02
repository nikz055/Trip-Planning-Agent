"""STAND-IN planner: a rule-based substitute for the planner LLM. Not a language model.

It implements the same `LLMClient` interface as the real client: it receives
the conversation, answers with tool calls, and is subject to the same
orchestrator, executor and verifier. It exists so that tests, the demo and the
guardrail pipeline can run with no API key. It says nothing about how well a
real model plans.

Policy, in the order the planner prompt describes it:
  1. missing required details       -> ask_user for all of them at once
  2. flights both ways + places     -> searched in parallel
  3. no flights on the dates        -> probe nearby dates; fixed dates: ask_user
                                       with the nearest ones; flexible: shift
  4. hotel                          -> depends on the chosen flight so check-in
                                       follows its local arrival date
  5. no hotel fits                  -> stated rating preference: ask which
                                       matters more; otherwise relax the rating
  6. travel times, then a day-by-day schedule built greedily
"""

from __future__ import annotations

import json
import math
import re
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Any

from trip_agent.llm.base import AssistantTurn, LLMToolCall, Message, ToolSpec, estimate_tokens
from trip_agent.llm.standin_data import city_airports, destinations_with_data
from trip_agent.llm.standin_intake import parse_request
from trip_agent.models import REQUIRED_FIELDS, Plan
from trip_agent.orchestrator.budget import compute_budget

Call = dict[str, Any]

DEFAULT_MIN_RATING = 4.0
MEALS_PER_PERSON_DAY = 45.0
INCIDENTALS_PER_PERSON = 35.0
ACTIVITY_RESERVE_PER_PERSON_DAY = 12.0
HOTEL_TAX_RATE = 0.10
MAX_WALK_MINUTES = 20
MAX_CANDIDATES = 24  # plus the hotel = 25 points, the get_travel_times limit
DAY_START, DAY_END = time(9, 0), time(20, 0)
LUNCH_AFTER, LUNCH_MINUTES = time(12, 0), 60
RANK_PENALTY_MINUTES = 3  # per position in the places list, so popular sights win ties
ENTRY_PRICE_CEILINGS = (20.0, 15.0, 10.0, 0.0)  # tried in turn when a plan is over budget
WEEKDAY_KEYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


def _call(name: str, **args: Any) -> Call:
    return {"name": name, "args": {k: v for k, v in args.items() if v is not None}}


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _iso(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M")


def airport_code(place: str | None) -> str | None:
    if not place:
        return None
    if re.fullmatch(r"[A-Z]{3}", place.strip()):
        return place.strip()
    return city_airports().get(place.strip().lower())


# --------------------------------------------------------------------------
# Reading the conversation
# --------------------------------------------------------------------------


@dataclass
class View:
    """What the stand-in knows: the newest context block and every tool result."""

    ctx: dict[str, Any] = field(default_factory=dict)
    entries: dict[str, dict[str, Any]] = field(default_factory=dict)  # by call_id

    @classmethod
    def read(cls, messages: list[Message]) -> "View":
        view = cls()
        for message in messages:
            for result in message.tool_results:
                view._add(result.content)
            if message.text:
                for block in re.findall(r"<context>\n(.*?)\n</context>", message.text, re.S):
                    view.ctx = json.loads(block)
                for block in re.findall(
                    r"<refreshed_results>\n(.*?)\n</refreshed_results>", message.text, re.S
                ):
                    for entry in json.loads(block):
                        view.entries[entry["call_id"]] = entry
        return view

    def _add(self, content: str) -> None:
        try:
            entry = json.loads(content)
        except ValueError:
            return
        if isinstance(entry, dict) and "call_id" in entry:
            self.entries[entry["call_id"]] = entry

    def find(self, tool: str, **want: Any) -> dict[str, Any] | None:
        """Newest successful result of `tool` whose args match (None = argument absent)."""
        for entry in reversed(list(self.entries.values())):
            if entry.get("tool") != tool or entry.get("status") != "ok":
                continue
            args = entry.get("args", {})
            if all(args.get(key) == value for key, value in want.items()):
                return entry
        return None

    def find_travel(self, mode: str, point_ids: list[str]) -> dict[str, Any] | None:
        for entry in reversed(list(self.entries.values())):
            if entry.get("tool") != "get_travel_times" or entry.get("status") != "ok":
                continue
            args = entry["args"]
            if args.get("mode") == mode and [p["id"] for p in args["points"]] == point_ids:
                return entry
        return None


# --------------------------------------------------------------------------
# One planning decision
# --------------------------------------------------------------------------


@dataclass
class Prefs:
    max_entry_price: float | None = None  # per person; 0 = free attractions only
    max_walk_minutes: int = MAX_WALK_MINUTES
    day_start: time = DAY_START
    cap_delta: int = 0  # negative = a lighter day
    excluded: frozenset[str] = frozenset()  # place ids not to use


def prefs_from(instruction: str, places: list[dict[str, Any]], previous_place_ids: set[str]) -> Prefs:
    """Keyword reading of a change request. A real model reads the sentence; this does not."""
    low = instruction.lower()
    prefs = Prefs()
    understood = False
    if re.search(r"less walking|walk less|no walking|too much walking|tired|easier on (?:the |my |our )?(?:feet|legs)", low):
        prefs.max_walk_minutes, understood = 5, True
    if re.search(r"\b(?:fewer|lighter|less busy|slower|easier|quieter|more relaxed)\b", low):
        prefs.cap_delta, understood = -1, True
    if re.search(r"\b(?:free|cheaper|cheap|save money|less expensive)\b", low):
        prefs.max_entry_price, understood = 0, True
    if re.search(r"\b(?:later|sleep in|late start|lie in)\b", low):
        prefs.day_start, understood = time(10, 30), True
    named = {p["place_id"] for p in places if p["name"].lower() in low}
    if named:
        prefs.excluded = frozenset(named)  # "skip X", "replace X", "not X"
    elif not understood:
        prefs.excluded = frozenset(previous_place_ids)  # anything else: offer different places
    return prefs


class _Decision:
    """Works out the next action from the current view. Holds no state between turns."""

    def __init__(self, view: View):
        self.v = view
        self.ctx = view.ctx
        self.req = view.ctx["trip_request"]
        self.rules = view.ctx["rules"]
        self.unavailable = set(view.ctx.get("unavailable_tools", []))
        self.travellers = max(1, int(view.ctx.get("travellers") or 1))
        self.limit = view.ctx["budget"]["limit_in_plan_currency"]
        self.assumptions: list[str] = []
        self.start: date
        self.end: date
        self.out_entry = self.back_entry = self.hotel_entry = self.places_entry = None
        self.out_offer = self.back_offer = self.hotel = None
        self.places: list[dict[str, Any]] = []
        self.walk = self.transit = None
        self._fixed_arrival: datetime | None = None  # set from the previous plan on a revision
        self._fixed_departure: datetime | None = None

    def next_calls(self) -> list[Call]:
        if self.ctx.get("revision"):
            return self._revise()
        for stage in (self._details, self._flights, self._hotel, self._places, self._travel_times):
            calls = stage()
            if calls:
                return calls
        return [self._propose()]

    # -- 1. details --------------------------------------------------------

    def _details(self) -> list[Call] | None:
        missing = [f for f in REQUIRED_FIELDS if self.req.get(f) in (None, "")]
        if missing:
            return [
                _call(
                    "ask_user",
                    fields=missing,
                    reason="I need these details before I can search flights and hotels.",
                )
            ]
        self.origin = airport_code(self.req["origin"])
        self.dest = airport_code(self.req["destination_city"])
        if not self.origin or not self.dest:
            unknown = self.req["origin"] if not self.origin else self.req["destination_city"]
            return [
                _call(
                    "give_up",
                    reason=f"I don't know an airport for '{unknown}'.",
                    suggestions=["Use a major city, or a three-letter airport code such as DEL."],
                )
            ]
        available = destinations_with_data()
        if self.req["destination_city"].strip().lower() not in {c.lower() for c in available}:
            return [
                _call(
                    "give_up",
                    reason=(
                        f"There is no hotel or attraction data for {self.req['destination_city']} yet, "
                        "so I can't build a plan there."
                    ),
                    suggestions=["Destinations with sample data: " + ", ".join(available)],
                )
            ]
        self.start = date.fromisoformat(self.req["start_date"])
        self.end = date.fromisoformat(self.req["end_date"])
        return None

    # -- 2/3. flights ------------------------------------------------------

    def _flight_search(self, origin: str, dest: str, day: date) -> Call:
        return _call(
            "search_flights",
            origin=origin,
            destination=dest,
            depart_date=day.isoformat(),
            adults=self.req["adults"],
            children=self.req.get("children") or 0,
        )

    def _flight_entries(self, start: date, end: date):
        out = self.v.find("search_flights", origin=self.origin, destination=self.dest, depart_date=start.isoformat())
        back = self.v.find("search_flights", origin=self.dest, destination=self.origin, depart_date=end.isoformat())
        return out, back

    def _flights(self) -> list[Call] | None:
        if "search_flights" in self.unavailable:
            return None
        out, back = self._flight_entries(self.start, self.end)
        calls = []
        if out is None:
            calls.append(self._flight_search(self.origin, self.dest, self.start))
        if back is None:
            calls.append(self._flight_search(self.dest, self.origin, self.end))
        if calls:
            if self._find_places() is None and "search_places" not in self.unavailable:
                calls.append(self._places_search())
            return calls
        if not out["data"] or not back["data"]:
            return self._no_flights()
        self._choose_flights(out, back)
        return None

    def _choose_flights(self, out: dict, back: dict) -> None:
        cheapest = lambda offers: min(offers, key=lambda o: (o["price"], o["depart_time"]))  # noqa: E731
        self.out_entry, self.back_entry = out, back
        self.out_offer, self.back_offer = cheapest(out["data"]), cheapest(back["data"])

    def _no_flights(self) -> list[Call] | None:
        today = date.fromisoformat(self.ctx["today"])
        flex = int(self.rules.get("flex_days", 3))
        probes: list[Call] = []
        viable: list[int] = []
        for shift in (s for k in range(1, flex + 1) for s in (-k, k)):
            start, end = self.start + timedelta(days=shift), self.end + timedelta(days=shift)
            if start < today:
                continue
            out, back = self._flight_entries(start, end)
            if out is None:
                probes.append(self._flight_search(self.origin, self.dest, start))
            if back is None:
                probes.append(self._flight_search(self.dest, self.origin, end))
            if out and back and out["data"] and back["data"]:
                viable.append(shift)
        if probes:
            return probes
        route = f"{self.origin}–{self.dest}"
        if not viable:
            return [
                _call(
                    "give_up",
                    reason=f"No flights on {route} for {self.start} to {self.end} or within {flex} days either side.",
                    suggestions=["Try dates further away", "Try a different departure city"],
                )
            ]
        if self.req.get("dates_flexible"):
            shift = viable[0]
            new_start, new_end = self.start + timedelta(days=shift), self.end + timedelta(days=shift)
            self.assumptions.append(
                f"Dates shifted from {self.start}–{self.end} to {new_start}–{new_end}: there are no "
                f"flights on {route} for the requested dates and you said your dates are flexible."
            )
            self.start, self.end = new_start, new_end
            self._choose_flights(*self._flight_entries(new_start, new_end))
            return None
        if self.ctx.get("clarifications_left", 1) <= 0:
            return [
                _call(
                    "give_up",
                    reason=f"No flights on {route} for the fixed dates {self.start} to {self.end}.",
                    suggestions=[f"Travel {self.start + timedelta(days=s)} to {self.end + timedelta(days=s)}" for s in viable[:3]],
                )
            ]
        nearest = viable[:3]
        options = "; ".join(f"{self.start + timedelta(days=s)} to {self.end + timedelta(days=s)}" for s in nearest)
        return [
            _call(
                "ask_user",
                fields=["start_date", "end_date"],
                reason=(
                    f"There are no flights on {route} for {self.start} to {self.end}. Your dates are "
                    f"fixed, so I have not changed them. The nearest dates with flights both ways: {options}."
                ),
                options={
                    "start_date": [(self.start + timedelta(days=s)).isoformat() for s in nearest],
                    "end_date": [(self.end + timedelta(days=s)).isoformat() for s in nearest],
                },
            )
        ]

    # -- 4/5. hotel --------------------------------------------------------

    @property
    def arrival(self) -> datetime | None:
        if self._fixed_arrival is not None:
            return self._fixed_arrival
        return _dt(self.out_offer["arrive_time"]) if self.out_offer else None

    @property
    def departure(self) -> datetime | None:
        if self._fixed_departure is not None:
            return self._fixed_departure
        return _dt(self.back_offer["depart_time"]) if self.back_offer else None

    @property
    def check_in_date(self) -> date:
        return self.arrival.date() if self.arrival else self.start

    @property
    def check_out_date(self) -> date:
        return self.departure.date() if self.departure else self.end

    @property
    def nights(self) -> int:
        return (self.check_out_date - self.check_in_date).days

    @property
    def rooms(self) -> int:
        return math.ceil(self.travellers / 2)

    def _hotel_cap(self) -> float | None:
        """Highest nightly price that still leaves room for everything else."""
        if self.limit is None:
            return None
        days = (self.end - self.start).days + 1
        flights = sum(o["price"] for o in (self.out_offer, self.back_offer) if o)
        reserve = (
            flights
            + (MEALS_PER_PERSON_DAY + ACTIVITY_RESERVE_PER_PERSON_DAY) * self.travellers * days
            + INCIDENTALS_PER_PERSON * self.travellers
        )
        cap = (self.limit - reserve) / (self.nights * self.rooms)
        return max(1.0, math.floor(cap * 100) / 100)

    def _hotel_search(self, min_rating: float | None, cap: float | None) -> Call:
        common = dict(
            city=self.req["destination_city"],
            check_out=self.check_out_date.isoformat(),
            guests=self.travellers,
            rooms=self.rooms,
            min_rating=min_rating,
            max_price_per_night=cap,
        )
        if self.out_entry is not None:
            # BR-05: no check_in; the system derives it from the chosen flight's arrival.
            return _call(
                "search_hotels",
                depends_on=[self.out_entry["call_id"]],
                flight_offer_id=self.out_offer["offer_id"],
                **common,
            )
        return _call("search_hotels", check_in=self.check_in_date.isoformat(), **common)

    def _find_hotels(self, min_rating: float | None, cap: float | None) -> dict | None:
        return self.v.find(
            "search_hotels",
            city=self.req["destination_city"],
            check_in=self.check_in_date.isoformat(),
            check_out=self.check_out_date.isoformat(),
            rooms=self.rooms,
            min_rating=min_rating,
            max_price_per_night=cap,
        )

    def _hotel(self) -> list[Call] | None:
        if self.nights <= 0 or "search_hotels" in self.unavailable:
            return None
        cap = self._hotel_cap()
        stated = self.req.get("min_hotel_rating")
        rating = stated or DEFAULT_MIN_RATING
        money = f"EUR {cap:.2f} per night" if cap else "any price"

        def pick(entry: dict, *, best_rated: bool = False) -> None:
            self.hotel_entry = entry
            if best_rated:
                self.hotel = max(entry["data"], key=lambda h: (h["rating"], -h["price_per_night"]))
            else:
                self.hotel = min(entry["data"], key=lambda h: (h["price_per_night"], -h["rating"]))

        preferred = self._find_hotels(rating, cap)
        if preferred is None:
            return [self._hotel_search(rating, cap)]
        if preferred["data"]:
            pick(preferred)
            return None

        if stated:
            # A stated preference conflicts with the budget: ask which matters more.
            already_asked = any("min_hotel_rating" in c.get("fields", []) for c in self.ctx.get("clarifications", []))
            if cap is not None and not already_asked and self.ctx.get("clarifications_left", 1) > 0:
                return [
                    _call(
                        "ask_user",
                        fields=["total_budget", "min_hotel_rating"],
                        reason=(
                            f"A {stated:g}-star hotel does not fit this budget: after flights, meals and "
                            f"activities there is about {money} for the room, and no {stated:g}-star hotel "
                            "costs that little. Which matters more? Raise the budget or lower the hotel rating."
                        ),
                    )
                ]
            uncapped = self._find_hotels(rating, None)
            if uncapped is None:
                return [self._hotel_search(rating, None)]
            if uncapped["data"]:
                pick(uncapped)
                self.assumptions.append(
                    f"Kept the {stated:g}-star hotel preference. No such hotel costs {money} or less, "
                    "so lodging takes the plan over budget."
                )
                return None

        relaxed = self._find_hotels(None, cap)
        if relaxed is None:
            return [self._hotel_search(None, cap)]
        if relaxed["data"]:
            pick(relaxed, best_rated=True)
            self.assumptions.append(
                f"Relaxed the hotel rating (wanted {rating:g}+): no hotel at that rating costs {money} "
                f"or less. Chose {self.hotel['name']} (rating {self.hotel['rating']:g})."
            )
            return None

        cheapest = self._find_hotels(None, None)
        if cheapest is None:
            return [self._hotel_search(None, None)]
        if cheapest["data"]:
            pick(cheapest)
            self.assumptions.append(
                f"No hotel costs {money} or less, even with the rating relaxed. Chose the cheapest "
                f"available, {self.hotel['name']}."
            )
        else:
            self.assumptions.append("No hotel was available for these dates; lodging is not included.")
        return None

    # -- 6. places and travel times ---------------------------------------

    def _places_search(self) -> Call:
        return _call(
            "search_places",
            city=self.req["destination_city"],
            interests=self.req.get("interests") or None,
            limit=30,
        )

    def _find_places(self) -> dict | None:
        return self.v.find("search_places", city=self.req["destination_city"])

    def _places(self) -> list[Call] | None:
        if "search_places" in self.unavailable:
            return None
        entry = self._find_places()
        if entry is None:
            return [self._places_search()]
        self.places_entry = entry
        self.places = entry["data"][:MAX_CANDIDATES]
        return None

    def _points(self) -> list[dict[str, Any]]:
        points = [{"id": p["place_id"], "lat": p["lat"], "lon": p["lon"]} for p in self.places]
        if self.hotel:
            points.insert(0, {"id": self.hotel["hotel_id"], "lat": self.hotel["lat"], "lon": self.hotel["lon"]})
        return points

    def _travel_times(self) -> list[Call] | None:
        points = self._points()
        if len(points) < 2 or "get_travel_times" in self.unavailable:
            return None
        ids = [p["id"] for p in points]
        self.walk = self.v.find_travel("walk", ids)
        self.transit = self.v.find_travel("transit", ids)
        calls = [
            _call("get_travel_times", points=points, mode=mode)
            for mode, entry in (("walk", self.walk), ("transit", self.transit))
            if entry is None
        ]
        return calls or None

    # -- 7. the plan -------------------------------------------------------

    def _propose(self) -> Call:
        plan = self._build(Prefs())
        total = compute_budget(Plan.model_validate(plan)).total
        if self.limit is not None and total > self.limit:
            # over budget: lower the entry-price ceiling step by step until the plan fits
            ceiling = None
            for price_cap in ENTRY_PRICE_CEILINGS:
                cheaper = self._build(Prefs(max_entry_price=price_cap))
                cheaper_total = compute_budget(Plan.model_validate(cheaper)).total
                if cheaper_total < total:
                    plan, total, ceiling = cheaper, cheaper_total, price_cap
                if total <= self.limit:
                    break
            if ceiling is not None:
                plan["assumptions"].append(
                    "Only free attractions are included, to keep the cost down."
                    if ceiling == 0
                    else f"Attractions costing more than EUR {ceiling:g} per person were left out, to keep the cost down."
                )
            if total > self.limit:
                plan["assumptions"].append(
                    f"The plan costs EUR {total:.2f}, which is EUR {total - self.limit:.2f} over the "
                    f"EUR {self.limit:.2f} budget, even with the cheapest flights."
                )
        plan["budget"] = compute_budget(Plan.model_validate(plan)).model_dump()
        return _call("propose_plan", plan=plan)

    def _build(self, prefs: Prefs) -> dict[str, Any]:
        n_days = (self.end - self.start).days + 1
        day_no = lambda d: (d - self.start).days + 1  # noqa: E731
        days = [
            {"day": i + 1, "date": (self.start + timedelta(days=i)).isoformat(), "items": []}
            for i in range(n_days)
        ]
        add = lambda day, item: days[day - 1]["items"].append(item)  # noqa: E731
        arrival_day = min(day_no(self.check_in_date), n_days)
        assumptions = list(self.assumptions)

        if self.out_offer:
            add(1, self._flight_item("flight-out", 1, self.out_offer, self.out_entry, self.origin, self.dest))
        if self.back_offer:
            add(n_days, self._flight_item("flight-back", n_days, self.back_offer, self.back_entry, self.dest, self.origin))
        if self.hotel:
            hotel_item = self._hotel_item(arrival_day)
            add(arrival_day, hotel_item)
            if not self.hotel["taxes_included"]:
                add(
                    arrival_day,
                    self._allowance(
                        "hotel-taxes", "other_allowance", arrival_day,
                        "Hotel taxes (estimate; not included in the room rate)",
                        round(hotel_item["cost"] * HOTEL_TAX_RATE, 2),
                    ),
                )

        used: set[str] = set()
        for day in range(arrival_day, n_days + 1):
            day_date = self.start + timedelta(days=day - 1)
            start_at, end_at, day_cap = self._window(day, day_date, n_days, arrival_day, prefs)
            for item in self._day_items(day, day_date, start_at, end_at, day_cap, used, prefs):
                add(day, item)
            add(
                day,
                self._allowance(
                    f"meals-d{day}", "meal_allowance", day, "Meals (estimate)",
                    MEALS_PER_PERSON_DAY * self.travellers,
                ),
            )
        add(
            arrival_day,
            self._allowance(
                "incidentals", "other_allowance", arrival_day,
                "Airport transfers and incidentals (estimate)", INCIDENTALS_PER_PERSON * self.travellers,
            ),
        )

        assumptions.append(f"Prices are in EUR for {self.travellers} traveller(s).")
        assumptions.append(f"Meals are estimated at EUR {MEALS_PER_PERSON_DAY:.0f} per person per day.")
        budget = self.ctx["budget"]
        if budget["amount"] and budget["currency"] != budget["plan_currency"] and self.limit is not None:
            assumptions.append(
                f"Budget {budget['currency']} {budget['amount']:,.0f} converted to EUR {self.limit:,.2f} "
                "at a fixed sample rate."
            )
        return {"days": days, "assumptions": assumptions}

    def _window(self, day: int, day_date: date, n_days: int, arrival_day: int, prefs: Prefs):
        """When sightseeing can start and must end on a day, and how many activities fit the pace."""
        start_at = datetime.combine(day_date, prefs.day_start)
        end_at = datetime.combine(day_date, DAY_END)
        cap = int(self.rules["max_activities_per_day"])
        if day == arrival_day:
            cap = min(cap, int(self.rules["arrival_day_max_activities"]))
            if self.arrival:
                settle = self.arrival + timedelta(minutes=self.rules["arrival_buffer_minutes"] + 30)
                start_at = max(start_at, settle)
        if day == n_days and self.departure:
            leave = self.departure - timedelta(minutes=self.rules["departure_buffer_minutes"] + 30)
            end_at = min(end_at, leave)
        return start_at, end_at, max(1, cap + prefs.cap_delta) if prefs.cap_delta < 0 else cap

    def _flight_item(self, item_id, day, offer, entry, origin, dest) -> dict[str, Any]:
        stops = "nonstop" if offer["stops"] == 0 else f"{offer['stops']} stop"
        return {
            "id": item_id,
            "type": "flight",
            "title": f"{offer['airline']} {origin} to {dest} ({stops})",
            "day": day,
            "start": offer["depart_time"],
            "end": offer["arrive_time"],
            "ref": {"result_id": entry["call_id"], "item_id": offer["offer_id"]},
            "cost": offer["price"],
            "cost_status": "confirmed",
            "link": offer["booking_link"],
        }

    def _hotel_item(self, day: int) -> dict[str, Any]:
        hotel = self.hotel
        opens = datetime.combine(self.check_in_date, time.fromisoformat(hotel["check_in_time"]))
        check_in = opens
        if self.arrival:
            reach = self.arrival + timedelta(minutes=self.rules["arrival_buffer_minutes"])
            latest = datetime.combine(self.check_in_date, time(23, 59))
            check_in = min(max(opens, reach), latest)
        check_out = datetime.combine(self.check_out_date, time(11, 0))
        if self.departure:
            check_out = min(check_out, self.departure - timedelta(minutes=self.rules["departure_buffer_minutes"]))
        return {
            "id": "hotel",
            "type": "hotel",
            "title": hotel["name"],
            "day": day,
            "start": _iso(check_in),
            "end": _iso(check_out),
            "ref": {"result_id": self.hotel_entry["call_id"], "item_id": hotel["hotel_id"]},
            "cost": round(hotel["price_per_night"] * self.nights * self.rooms, 2),
            "cost_status": "confirmed",
            "link": hotel["booking_link"],
            "notes": f"{self.nights} night(s), {self.rooms} room(s); check-in from {hotel['check_in_time']}",
        }

    @staticmethod
    def _allowance(item_id: str, kind: str, day: int, title: str, cost: float) -> dict[str, Any]:
        return {
            "id": item_id, "type": kind, "title": title, "day": day,
            "cost": round(cost, 2), "cost_status": "estimated",
        }  # fmt: skip

    # -- day schedule ------------------------------------------------------

    def _leg(self, origin: str | None, dest: str, prefs: Prefs) -> dict[str, Any] | None:
        """Travel from one point to another: walk when short, otherwise transit."""
        if origin is None or self.walk is None:
            return None
        ids = [p["id"] for p in self.walk["args"]["points"]]
        i, j = ids.index(origin), ids.index(dest)
        walk_minutes = self.walk["data"]["matrix_minutes"][i][j]
        if walk_minutes <= prefs.max_walk_minutes or self.transit is None:
            return {"mode": "walk", "minutes": walk_minutes, "cost": 0.0, "call_id": self.walk["call_id"]}
        fare = self.transit["data"]["estimated_cost"][i][j]
        return {
            "mode": "transit",
            "minutes": self.transit["data"]["matrix_minutes"][i][j],
            "cost": round(fare * self.travellers, 2),
            "call_id": self.transit["call_id"],
        }

    @staticmethod
    def _open_hours(place: dict[str, Any], day: date) -> tuple[datetime, datetime] | None:
        if day.isoformat() in place["closed_dates"]:
            return None
        span = place["opening_hours"].get(WEEKDAY_KEYS[day.weekday()])
        if not span:
            return None
        return (
            datetime.combine(day, time.fromisoformat(span[0])),
            datetime.combine(day, time.fromisoformat(span[1])),
        )

    def _transfer_item(self, item_id, day, leg, origin, dest, title, start) -> dict[str, Any]:
        return {
            "id": item_id,
            "type": "transfer",
            "title": title,
            "day": day,
            "start": _iso(start),
            "end": _iso(start + timedelta(minutes=leg["minutes"])),
            "ref": {"result_id": leg["call_id"], "item_id": f"{origin}->{dest}"},
            "cost": leg["cost"],
            "cost_status": "estimated",
        }

    def _day_items(
        self, day, day_date, start_at, end_at, cap, used, prefs, *, after: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Greedy schedule for one day. `after` is a kept activity to continue from."""
        items: list[dict[str, Any]] = []
        buffer = timedelta(minutes=self.rules["buffer_minutes"])
        here = self.hotel["hotel_id"] if self.hotel else None
        free_from = start_at  # when the travellers can next set off
        last_end: datetime | None = None
        if after is not None:
            here, last_end = after["ref"]["item_id"], _dt(after["end"])
            free_from = last_end
        had_lunch = free_from.time() >= time(13, 30)
        count = 0
        while count < cap:
            best = None
            for rank, place in enumerate(self.places):
                if place["place_id"] in used or place["place_id"] in prefs.excluded:
                    continue
                if prefs.max_entry_price is not None and place["entry_price"] > prefs.max_entry_price:
                    continue
                hours = self._open_hours(place, day_date)
                if hours is None:
                    continue
                leg = self._leg(here, place["place_id"], prefs)
                travel = timedelta(minutes=leg["minutes"]) if leg else timedelta(minutes=45 if last_end else 0)
                earliest = free_from + travel + (buffer if last_end else timedelta())
                begin = max(earliest, hours[0])
                finish = begin + timedelta(minutes=place["typical_duration_min"])
                if finish > min(hours[1], end_at):
                    continue
                wait = (begin - earliest).total_seconds() / 60
                score = travel.total_seconds() / 60 + wait + RANK_PENALTY_MINUTES * rank
                if best is None or score < best[0]:
                    best = (score, place, leg, begin, finish)
            if best is None:
                break
            _, place, leg, begin, finish = best
            if leg and leg["minutes"] > 0:
                # after an activity leave straight away; from the hotel leave just in time
                depart = last_end if last_end else begin - timedelta(minutes=leg["minutes"])
                items.append(
                    self._transfer_item(
                        f"d{day}-to-{place['place_id']}", day, leg, here, place["place_id"],
                        f"{leg['mode'].title()} to {place['name']}", depart,
                    )
                )
            items.append(
                {
                    "id": f"d{day}-{place['place_id']}",
                    "type": "activity",
                    "title": place["name"],
                    "day": day,
                    "start": _iso(begin),
                    "end": _iso(finish),
                    "ref": {"result_id": self.places_entry["call_id"], "item_id": place["place_id"]},
                    "cost": round(place["entry_price"] * self.travellers, 2),
                    "cost_status": "confirmed",
                }
            )
            used.add(place["place_id"])
            here, last_end, free_from = place["place_id"], finish, finish
            count += 1
            if not had_lunch and finish.time() >= LUNCH_AFTER:
                free_from = finish + timedelta(minutes=LUNCH_MINUTES)
                had_lunch = True
        if self.hotel and last_end and here != self.hotel["hotel_id"]:
            back = self._leg(here, self.hotel["hotel_id"], prefs)
            if back and back["minutes"] > 0:
                items.append(
                    self._transfer_item(
                        f"d{day}-to-hotel", day, back, here, self.hotel["hotel_id"],
                        f"{back['mode'].title()} back to the hotel", free_from,
                    )
                )
        return items

    # -- revisions ---------------------------------------------------------

    def _revise(self) -> list[Call]:
        """Replan only the affected days of the previous plan; everything else is passed through."""
        revision = self.ctx["revision"]
        previous = revision["previous_plan"]
        affected = set(revision["affected_days"])
        locked = set(self.ctx.get("locked_item_ids", []))
        days = [dict(day, items=list(day["items"])) for day in previous["days"]]
        items = [item for day in days for item in day["items"]]
        self.start, self.end = date.fromisoformat(days[0]["date"]), date.fromisoformat(days[-1]["date"])
        n_days = len(days)

        flights = sorted((i for i in items if i["type"] == "flight"), key=lambda i: i["start"])
        if flights and flights[0]["day"] == 1:
            self._fixed_arrival = _dt(flights[0]["end"])
        if len(flights) > 1 and flights[-1]["day"] == n_days:
            self._fixed_departure = _dt(flights[-1]["start"])
        hotel_item = next((i for i in items if i["type"] == "hotel"), None)
        if hotel_item:
            self.hotel_entry = self.v.entries.get(hotel_item["ref"]["result_id"])
            self.hotel = next(
                (h for h in (self.hotel_entry or {}).get("data", []) if h["hotel_id"] == hotel_item["ref"]["item_id"]),
                None,
            )
        entry = self._find_places()
        if entry:
            self.places_entry, self.places = entry, entry["data"][:MAX_CANDIDATES]
        calls = self._travel_times()
        if calls:
            return calls

        used = {
            item["ref"]["item_id"]
            for day in days
            for item in day["items"]
            if item["type"] == "activity" and (day["day"] not in affected or item["id"] in locked)
        }
        arrival_day = min((self.check_in_date - self.start).days + 1, n_days)
        for day in days:
            if day["day"] not in affected:
                continue
            day_date = date.fromisoformat(day["date"])
            previous_places = {i["ref"]["item_id"] for i in day["items"] if i["type"] == "activity"}
            prefs = prefs_from(revision["instruction"], self.places, previous_places)
            start_at, end_at, cap = self._window(day["day"], day_date, n_days, arrival_day, prefs)
            # Everything up to the last locked activity stays; the rest of the day is rebuilt.
            locked_stops = [i for i in day["items"] if i["type"] == "activity" and i["id"] in locked]
            last_locked = max(locked_stops, key=lambda i: i["end"]) if locked_stops else None
            kept = [
                item for item in day["items"]
                if item["type"] not in ("activity", "transfer")
                or item["id"] in locked
                or (last_locked is not None and item["end"] <= last_locked["end"])
            ]  # fmt: skip
            kept_stops = sum(i["type"] == "activity" for i in kept)
            rebuilt = self._day_items(
                day["day"], day_date, start_at, end_at, max(0, cap - kept_stops), used, prefs, after=last_locked
            )
            kept_ids = {i["id"] for i in kept}
            for item in rebuilt:  # an id must not collide with a kept item
                while item["id"] in kept_ids:
                    item["id"] += "-b"
            day["items"] = kept + rebuilt

        plan = {"days": days, "assumptions": list(previous.get("assumptions", []))}
        plan["budget"] = compute_budget(Plan.model_validate(plan)).model_dump()
        return [_call("propose_plan", plan=plan)]

    # -- loop guard --------------------------------------------------------

    def wrap_up(self) -> list[Call]:
        """Final call: only propose_plan and give_up are on offer, so use what is already known."""
        if self.ctx.get("revision"):
            calls = self._revise()
            if calls[0]["name"] == "propose_plan":
                return calls
            return [_call("give_up", reason="The change needs more travel-time data than there was time to fetch.", suggestions=[])]
        if self._details():
            return [_call("give_up", reason="The trip details are incomplete.", suggestions=[])]
        out, back = self._flight_entries(self.start, self.end)
        if out and back and out["data"] and back["data"]:
            self._choose_flights(out, back)
        if self.nights > 0:
            stays = [
                e for e in self.v.entries.values()
                if e.get("tool") == "search_hotels" and e.get("status") == "ok" and e["data"]
                and e["args"].get("check_in") == self.check_in_date.isoformat()
                and e["args"].get("check_out") == self.check_out_date.isoformat()
            ]  # fmt: skip
            if stays:
                self.hotel_entry = stays[-1]
                self.hotel = min(stays[-1]["data"], key=lambda h: h["price_per_night"])
        entry = self._find_places()
        if entry:
            self.places_entry, self.places = entry, entry["data"][:MAX_CANDIDATES]
            ids = [p["id"] for p in self._points()]
            self.walk, self.transit = self.v.find_travel("walk", ids), self.v.find_travel("transit", ids)
        if not (self.out_offer or self.hotel or self.places):
            return [
                _call(
                    "give_up",
                    reason="Planning stopped before any usable search results came back.",
                    suggestions=["Try again"],
                )
            ]
        self.assumptions.append("Planning was cut short, so this plan uses only the results gathered so far.")
        return [self._propose()]


# --------------------------------------------------------------------------
# The client
# --------------------------------------------------------------------------


class StandInPlanner:
    name = "STAND-IN planner (rule-based, not an LLM)"
    is_stand_in = True

    async def complete(
        self, *, system: str, messages: list[Message], tools: list[ToolSpec]
    ) -> AssistantTurn:
        offered = {tool.name for tool in tools}
        if "record_request" in offered:
            calls = [self._intake(messages)]
        elif "scope_revision" in offered:
            calls = [self._scope(messages)]
        else:
            decision = _Decision(View.read(messages))
            wrapping_up = decision.ctx.get("final_call") or "search_flights" not in offered
            calls = decision.wrap_up() if wrapping_up else decision.next_calls()
        return AssistantTurn(
            tool_calls=[
                LLMToolCall(id=f"standin_{uuid.uuid4().hex[:12]}", name=c["name"], args=c["args"])
                for c in calls
            ],
            input_tokens=estimate_tokens(system, *(m.text for m in messages)),
            output_tokens=estimate_tokens(json.dumps(calls)),
            stop_reason="tool_use",
        )

    @staticmethod
    def _scope(messages: list[Message]) -> Call:
        """Which days does a change request touch? Looks for day numbers and named items."""
        text = messages[0].text or ""
        outline = json.loads(re.search(r"<plan>\n(.*?)\n</plan>", text, re.S).group(1))
        request = re.search(r"<change_request>\n(.*?)\n</change_request>", text, re.S).group(1)
        low = request.lower()
        all_days = [day["day"] for day in outline]
        days = {int(n) for n in re.findall(r"\bday\s+(\d+)", low)}
        if re.search(r"\b(?:first|arrival) day\b", low):
            days.add(all_days[0])
        if re.search(r"\b(?:last|final|departure) day\b", low):
            days.add(all_days[-1])
        change_words = r"(?:change|different|another|cheaper|better|switch|swap|replace|upgrade|move|cancel|remove|drop)"
        targets, target_days = [], set()
        for day in outline:
            for item in day["items"]:
                named = item["type"] in ("activity", "hotel") and item["title"] and item["title"].lower() in low
                by_kind = item["type"] in ("hotel", "flight") and (
                    re.search(rf"{change_words}\b.*\b{item['type']}", low) or re.search(rf"\b{item['type']}\b.*{change_words}", low)
                )
                if named or by_kind:
                    targets.append(item["id"])
                    target_days.add(day["day"])
        affected = sorted(days & set(all_days)) or sorted(target_days) or all_days
        return _call("scope_revision", affected_days=affected, target_item_ids=targets, summary=request[:120])

    @staticmethod
    def _intake(messages: list[Message]) -> Call:
        text = messages[-1].text or ""
        today = re.search(r"Today is (\d{4}-\d{2}-\d{2})", text)
        request = re.search(r"<request>\n(.*?)\n</request>", text, re.S)
        fields = parse_request(
            request.group(1) if request else text,
            date.fromisoformat(today.group(1)) if today else date.today(),
        )
        return {"name": "record_request", "args": fields}
