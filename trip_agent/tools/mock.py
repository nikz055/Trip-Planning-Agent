"""Mock implementations of the four data tools, backed by data/mock/*.json.

Deterministic: the same arguments on the same catalogue return the same
payload, apart from `price_drift`, which tests use to simulate prices moving
between two fetches.
"""

from __future__ import annotations

import asyncio
import json
import math
import random
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from trip_agent.tools.base import ToolError, ToolTimeout
from trip_agent.tools.failures import FailureInjector
from trip_agent.tools.schemas import (
    GetTravelTimesArgs,
    SearchFlightsArgs,
    SearchHotelsArgs,
    SearchPlacesArgs,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "mock"
CURRENCY = "EUR"

CABIN_FACTOR = {"economy": 1.0, "premium": 1.6, "business": 3.0, "first": 5.0}
CHILD_FARE = 0.75
WEEKEND_FLIGHT_FACTOR = 1.12  # Friday to Sunday departures
WEEKEND_HOTEL_FACTOR = 1.10  # Friday and Saturday check-ins
ROUND_TRIP_DISCOUNT = 0.95

# Travel model: haversine distance (with a detour factor) / speed + fixed overhead per leg.
DETOUR_FACTOR = 1.3
MODE_SPEED_KMH = {"walk": 4.8, "transit": 18.0, "taxi": 24.0}
MODE_OVERHEAD_MIN = {"walk": 2, "transit": 8, "taxi": 4}
TRANSIT_FARE = 2.4  # flat, per person
TAXI_BASE = 4.0
TAXI_PER_KM = 1.8


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


CRUISE_KMH = 800.0
GENERATED_OFFERS = 20
INDIA_DOMESTIC_AIRLINES = ("IndiGo", "Air India", "Akasa Air", "SpiceJet", "Air India Express")
GENERATED_AIRLINES = ("IndiGo", "Air India", "Emirates", "Qatar", "Lufthansa", "Turkish", "Singapore Airlines", "British Airways")


def _hhmm(hours: float) -> str:
    minutes = int(round(hours * 60 / 5.0)) * 5
    return f"{(minutes // 60) % 24:02d}:{minutes % 60:02d}"


def _at(day: date, hhmm: str) -> datetime:
    return datetime.combine(day, time.fromisoformat(hhmm))


def _iso(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M")


class MockToolProvider:
    source = "mock"

    def __init__(self, failures: FailureInjector | None = None, data_dir: Path | None = None):
        root = data_dir or DATA_DIR
        self._flights = json.loads((root / "flights.json").read_text(encoding="utf-8"))
        self._hotels = json.loads((root / "hotels.json").read_text(encoding="utf-8"))
        self._places = json.loads((root / "places.json").read_text(encoding="utf-8"))
        airports_file = root / "airports.json" if (root / "airports.json").is_file() else DATA_DIR / "airports.json"
        self._airports = {a["iata"]: a for a in json.loads(airports_file.read_text(encoding="utf-8"))}
        self._generated_routes: dict[str, list[dict[str, Any]]] = {}
        self.failures = failures or FailureInjector()
        self.price_drift: dict[str, float] = {}  # tool -> multiplier on returned prices
        self.call_log: list[tuple[str, dict[str, Any]]] = []

    async def call(self, tool: str, args: BaseModel) -> Any:
        self.call_log.append((tool, args.model_dump(mode="json")))
        failure = self.failures.next(tool)
        if failure is not None:
            if failure.mode == "timeout":
                raise ToolTimeout(f"{tool} timed out")
            if failure.mode == "error":
                raise ToolError(f"{tool} failed upstream")
            if failure.mode == "empty":
                return self._empty(tool)
            await asyncio.sleep(failure.delay_s)  # slow, then answer normally
        handler = getattr(self, f"_{tool}", None)
        if handler is None:
            raise ToolError(f"unknown tool {tool}")
        return handler(args)

    @staticmethod
    def _empty(tool: str) -> Any:
        if tool == "get_travel_times":
            return {"point_ids": [], "mode": None, "matrix_minutes": [], "estimated_cost": None}
        return []

    # -- flights -----------------------------------------------------------

    def _has_service(self, origin: str, destination: str, day: date) -> bool:
        return not any(
            rule["origin"] == origin
            and rule["destination"] == destination
            and rule["date"] == day.isoformat()
            for rule in self._flights["no_service"]
        )

    def _generated(self, origin: str, destination: str) -> list[dict[str, Any]]:
        """Sample offers for a pair of known airports that has no hand-tuned route.

        Flight time comes from great-circle distance, local arrival time from the
        two UTC offsets, price from distance. Deterministic for a given pair.
        """
        key = f"{origin}-{destination}"
        if key not in self._generated_routes:
            a, b = self._airports.get(origin), self._airports.get(destination)
            templates: list[dict[str, Any]] = []
            if a and b and origin != destination:
                rng = random.Random(key)
                km = haversine_km(a["lat"], a["lon"], b["lat"], b["lon"])
                nonstop_hours = km / CRUISE_KMH + 0.7
                shift = b["utc_offset"] - a["utc_offset"]
                domestic_india = a["country"] == b["country"] == "India"
                airlines = INDIA_DOMESTIC_AIRLINES if domestic_india else GENERATED_AIRLINES
                for i in range(GENERATED_OFFERS):
                    is_nonstop = km < 9000 and i % 3 == 0
                    hours = nonstop_hours if is_nonstop else nonstop_hours * 1.25 + rng.uniform(1.5, 4.0)
                    depart = 6.0 + 16.0 * i / (GENERATED_OFFERS - 1)
                    # local arrival; never before departure, even across the date line
                    arrive = depart + max(hours + shift, 0.5)
                    price = (35 + km * 0.065) * rng.uniform(0.85, 1.35) * (1.15 if is_nonstop else 1.0)
                    templates.append(
                        {
                            "template_id": f"{origin}{destination}{i + 1:02d}",
                            "airline": airlines[i % len(airlines)],
                            "depart": _hhmm(depart),
                            "arrive": _hhmm(arrive),
                            "arrive_day_offset": int(arrive // 24),
                            "stops": 0 if is_nonstop else 1,
                            "base_price": float(round(price)),
                        }
                    )
            self._generated_routes[key] = templates
        return self._generated_routes[key]

    def _leg_offers(self, origin: str, destination: str, day: date) -> list[dict[str, Any]]:
        if not self._has_service(origin, destination, day):
            return []
        factor = WEEKEND_FLIGHT_FACTOR if day.weekday() >= 4 else 1.0
        legs = []
        templates = self._flights["routes"].get(f"{origin}-{destination}") or self._generated(origin, destination)
        for tpl in templates:
            arrive_day = day + timedelta(days=tpl["arrive_day_offset"])
            legs.append(
                {
                    "template_id": tpl["template_id"],
                    "airline": tpl["airline"],
                    "depart_time": _iso(_at(day, tpl["depart"])),
                    "arrive_time": _iso(_at(arrive_day, tpl["arrive"])),
                    "stops": tpl["stops"],
                    "adult_fare": tpl["base_price"] * factor,
                }
            )
        return legs

    def _search_flights(self, args: SearchFlightsArgs) -> list[dict[str, Any]]:
        party = args.adults + CHILD_FARE * args.children
        multiplier = party * CABIN_FACTOR[args.cabin] * self.price_drift.get("search_flights", 1.0)
        outbound = self._leg_offers(args.origin, args.destination, args.depart_date)
        offers = []
        if args.return_date is None:
            for leg in outbound:
                offer_id = f"{leg['template_id']}-{args.depart_date:%Y%m%d}"
                offers.append(self._offer(offer_id, leg, leg["adult_fare"] * multiplier))
        else:
            inbound = self._leg_offers(args.destination, args.origin, args.return_date)
            for out_leg, in_leg in zip(outbound, inbound):
                offer_id = (
                    f"RT-{out_leg['template_id']}-{args.depart_date:%Y%m%d}"
                    f"-{in_leg['template_id']}-{args.return_date:%Y%m%d}"
                )
                fare = (out_leg["adult_fare"] + in_leg["adult_fare"]) * ROUND_TRIP_DISCOUNT
                offer = self._offer(offer_id, out_leg, fare * multiplier)
                offer["return_depart_time"] = in_leg["depart_time"]
                offer["return_arrive_time"] = in_leg["arrive_time"]
                offers.append(offer)
        if args.max_price is not None:
            offers = [o for o in offers if o["price"] <= args.max_price]
        return sorted(offers, key=lambda o: (o["price"], o["depart_time"]))

    @staticmethod
    def _offer(offer_id: str, leg: dict[str, Any], price: float) -> dict[str, Any]:
        return {
            "offer_id": offer_id,
            "airline": leg["airline"],
            "depart_time": leg["depart_time"],
            "arrive_time": leg["arrive_time"],
            "stops": leg["stops"],
            "price": round(price, 2),
            "currency": CURRENCY,
            "booking_link": f"https://flights.mock.example/book/{offer_id}",
        }

    # -- hotels ------------------------------------------------------------

    def _city(self, catalogue: dict[str, list], city: str) -> list[dict[str, Any]]:
        wanted = city.strip().lower()
        return next((rows for name, rows in catalogue.items() if name.lower() == wanted), [])

    def _search_hotels(self, args: SearchHotelsArgs) -> list[dict[str, Any]]:
        assert args.check_in is not None, "the executor fills check_in before calling"
        factor = WEEKEND_HOTEL_FACTOR if args.check_in.weekday() in (4, 5) else 1.0
        factor *= self.price_drift.get("search_hotels", 1.0)
        hotels = []
        for row in self._city(self._hotels, args.city):
            price = round(row["base_price"] * factor, 2)
            if row["max_guests_per_room"] * args.rooms < args.guests:
                continue
            if args.max_price_per_night is not None and price > args.max_price_per_night:
                continue
            if args.min_rating is not None and row["rating"] < args.min_rating:
                continue
            hotels.append(
                {
                    "hotel_id": row["hotel_id"],
                    "name": row["name"],
                    "lat": row["lat"],
                    "lon": row["lon"],
                    "rating": row["rating"],
                    "price_per_night": price,
                    "currency": CURRENCY,
                    "check_in_time": row["check_in_time"],
                    "taxes_included": row["taxes_included"],
                    "booking_link": (
                        f"https://hotels.mock.example/{row['hotel_id']}"
                        f"?in={args.check_in}&out={args.check_out}&rooms={args.rooms}"
                    ),
                }
            )
        return sorted(hotels, key=lambda h: (h["price_per_night"], -h["rating"]))

    # -- places ------------------------------------------------------------

    def _search_places(self, args: SearchPlacesArgs) -> list[dict[str, Any]]:
        rows = self._city(self._places, args.city)
        interests = {i.strip().lower() for i in args.interests or []}
        if interests:
            # Matching places first; the rest keep their catalogue (popularity) order.
            rows = sorted(
                rows,
                key=lambda r: not (interests & ({r["category"]} | set(r["tags"]))),
            )
        factor = self.price_drift.get("search_places", 1.0)
        return [
            {
                "place_id": row["place_id"],
                "name": row["name"],
                "lat": row["lat"],
                "lon": row["lon"],
                "category": row["category"],
                "typical_duration_min": row["typical_duration_min"],
                "opening_hours": row["opening_hours"],
                "closed_dates": row["closed_dates"],
                "entry_price": round(row["entry_price"] * factor, 2),
                "currency": row["currency"],
                "description": row["description"],
            }
            for row in rows[: args.limit]
        ]

    # -- travel times ------------------------------------------------------

    def _get_travel_times(self, args: GetTravelTimesArgs) -> dict[str, Any]:
        speed, overhead = MODE_SPEED_KMH[args.mode], MODE_OVERHEAD_MIN[args.mode]
        minutes: list[list[int]] = []
        cost: list[list[float]] = []
        for a in args.points:
            minute_row, cost_row = [], []
            for b in args.points:
                if a.id == b.id:
                    minute_row.append(0)
                    cost_row.append(0.0)
                    continue
                km = haversine_km(a.lat, a.lon, b.lat, b.lon) * DETOUR_FACTOR
                minute_row.append(math.ceil(km / speed * 60) + overhead)
                if args.mode == "transit":
                    cost_row.append(TRANSIT_FARE)
                elif args.mode == "taxi":
                    cost_row.append(round(TAXI_BASE + TAXI_PER_KM * km, 2))
                else:
                    cost_row.append(0.0)
            minutes.append(minute_row)
            cost.append(cost_row)
        return {
            "point_ids": [p.id for p in args.points],
            "mode": args.mode,
            "matrix_minutes": minutes,
            # per person for transit, per vehicle for taxi; walking has no cost
            "estimated_cost": None if args.mode == "walk" else cost,
            "currency": CURRENCY,
        }
