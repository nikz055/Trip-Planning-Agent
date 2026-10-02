"""What the stand-in "knows" about the world: airport codes and which cities have sample data.

A real model brings this knowledge itself; the stand-in reads it from the mock catalogue.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "mock"
ALIASES = {
    "new delhi": "DEL", "bombay": "BOM", "bangalore": "BLR", "madras": "MAA", "calcutta": "CCU",
    "cochin": "COK", "trivandrum": "TRV", "nyc": "JFK", "washington dc": "IAD", "saigon": "SGN",
    "panaji": "GOI", "panjim": "GOI",
}  # fmt: skip


@lru_cache(maxsize=1)
def city_airports() -> dict[str, str]:
    """Lower-case city name -> IATA code of its main airport."""
    airports = json.loads((DATA_DIR / "airports.json").read_text(encoding="utf-8"))
    table = {a["city"].lower(): a["iata"] for a in reversed(airports)}  # first listed wins
    return {**table, **ALIASES}


@lru_cache(maxsize=1)
def destinations_with_data() -> tuple[str, ...]:
    """Cities that have hotels and places in the sample catalogue."""
    return tuple(json.loads((DATA_DIR / "hotels.json").read_text(encoding="utf-8")))
