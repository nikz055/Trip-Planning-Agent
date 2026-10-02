"""Free-text request parsing for the stand-in planner.

Pattern matching, not language understanding: it covers the phrasings used in
the tests, the eval set and the demo, and leaves a field out when unsure so the
clarification form asks for it.
"""

from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Any

from trip_agent.llm.standin_data import city_airports

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}  # fmt: skip
_MONTH = (
    r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|"
    r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
)
_ORD = r"(?:st|nd|rd|th)?"
_SEP = r"\s*(?:-|–|—|to|until|through)\s*"
_YEAR = r"(?:,?\s+(\d{4}))?"
_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}  # fmt: skip
_NUM = r"(\d+|" + "|".join(_NUMBER_WORDS) + ")"

_COUNTRY_ENTRY_CITY = {"austria": "Vienna", "portugal": "Lisbon", "rajasthan": "Jaipur"}
_NOT_A_PLACE = set(_MONTHS) | {
    "january", "february", "march", "april", "june", "july", "august", "september",
    "october", "november", "december", "monday", "tuesday", "wednesday", "thursday",
    "friday", "saturday", "sunday", "for", "on", "with", "from", "in", "to", "and",
    "the", "my", "our", "i", "we", "plan", "europe",
}  # fmt: skip
_CURRENCY = {
    "€": "EUR", "eur": "EUR", "euro": "EUR", "euros": "EUR",
    "$": "USD", "usd": "USD", "dollar": "USD", "dollars": "USD",
    "£": "GBP", "gbp": "GBP", "pound": "GBP", "pounds": "GBP",
    "₹": "INR", "inr": "INR", "rupee": "INR", "rupees": "INR", "rs": "INR",
}  # fmt: skip
_INTERESTS = {
    "museums": r"museums?", "art": r"art|gallery|galleries", "history": r"history|historic\w*",
    "architecture": r"architecture", "music": r"music|opera|concerts?",
    "food": r"food|cuisine|eating", "coffee": r"coffee|caf[eé]s?",
    "parks": r"parks?|gardens?", "nature": r"nature|hiking", "views": r"views?|viewpoints?",
    "shopping": r"shopping|markets?", "nightlife": r"nightlife|bars",
    "family": r"family|kids|children",
}  # fmt: skip
_PACE = {
    "packed": r"packed|busy|as much as possible|action-packed",
    "balanced": r"balanced|moderate",
    "relaxed": r"relaxed|relaxing|slow|easy-?going|laid-?back",
}


def _number(token: str) -> int:
    return _NUMBER_WORDS[token] if token in _NUMBER_WORDS else int(token)


def _amount(digits: str, thousands: str | None) -> float:
    value = float(digits.replace(",", ""))
    return value * 1000 if thousands else value


def _place(raw: str) -> str | None:
    words = raw.split()
    while words and words[-1].lower().strip(".,") in _NOT_A_PLACE:
        words.pop()
    name = " ".join(words).strip(".,")
    return name if name and name.lower() not in _NOT_A_PLACE else None


def _day_in_year(day: int, month: int, year: int | None, not_before: date) -> date | None:
    try:
        if year is not None:
            return date(year, month, day)
        candidate = date(not_before.year, month, day)
        return candidate if candidate >= not_before else date(not_before.year + 1, month, day)
    except ValueError:
        return None


def _dates(text: str, today: date) -> list[date]:
    iso = [date.fromisoformat(m) for m in re.findall(r"\b\d{4}-\d{2}-\d{2}\b", text)]
    if iso:
        return iso[:2]
    low = text.lower()
    month = lambda token: _MONTHS[token[:3]]  # noqa: E731

    ranged = re.search(rf"(?<!\d)(\d{{1,2}}){_ORD}{_SEP}(\d{{1,2}}){_ORD}\s+(?:of\s+)?{_MONTH}{_YEAR}", low)
    if ranged:
        d1, d2, mon, year = ranged.groups()
        start = _day_in_year(int(d1), month(mon), int(year) if year else None, today)
        end = start and _day_in_year(int(d2), month(mon), start.year, start)
        return [d for d in (start, end) if d]
    ranged = re.search(rf"{_MONTH}\s+(\d{{1,2}}){_ORD}{_SEP}(\d{{1,2}})(?!\d){_ORD}{_YEAR}", low)
    if ranged:
        mon, d1, d2, year = ranged.groups()
        start = _day_in_year(int(d1), month(mon), int(year) if year else None, today)
        end = start and _day_in_year(int(d2), month(mon), start.year, start)
        return [d for d in (start, end) if d]

    found: list[tuple[int, int, int, int | None]] = []  # position, day, month, year
    spans: list[tuple[int, int]] = []
    for m in re.finditer(rf"(?<!\d)(\d{{1,2}}){_ORD}\s+(?:of\s+)?{_MONTH}{_YEAR}", low):
        found.append((m.start(), int(m.group(1)), month(m.group(2)), int(m.group(3)) if m.group(3) else None))
        spans.append(m.span())
    for m in re.finditer(rf"{_MONTH}\s+(\d{{1,2}})(?!\d){_ORD}{_YEAR}", low):
        if any(a <= m.start() < b for a, b in spans):
            continue
        found.append((m.start(), int(m.group(2)), month(m.group(1)), int(m.group(3)) if m.group(3) else None))
    found.sort()
    # a year stated once ("3 Nov to 7 Nov 2026") applies to every date
    stated_year = next((y for *_, y in found if y), None)
    dates: list[date] = []
    for _, day, mon, year in found[:2]:
        parsed = _day_in_year(day, mon, year or stated_year, dates[-1] if dates else today)
        if parsed:
            dates.append(parsed)
    return dates


def parse_request(text: str, today: date) -> dict[str, Any]:
    """Extract TripRequest fields from free text. Unstated fields are left out."""
    low = text.lower()
    out: dict[str, Any] = {}
    guessed: list[str] = []

    # origin and destination
    origin = re.search(r"\b(?i:from|leaving|departing(?:\s+from)?)\s+([A-Z][\w'.-]*(?:\s+[A-Z][\w'.-]*)?)", text)
    if origin and (name := _place(origin.group(1))):
        out["origin"] = name
    for m in re.finditer(r"\b(?i:to|in|visit|visiting|around|explore|exploring)\s+([A-Z][\w'.-]*(?:\s+[A-Z][\w'.-]*)?)", text):
        name = _place(m.group(1))
        if not name or name == out.get("origin"):
            continue
        if name.lower() in _COUNTRY_ENTRY_CITY:
            out["destination_city"] = _COUNTRY_ENTRY_CITY[name.lower()]
            guessed.append("destination_city")
        else:
            out["destination_city"] = name
        break
    if "destination_city" not in out:
        # no "to X" / "in X": fall back to the first place name it recognises
        origin_name = out.get("origin", "").lower()
        named = [
            (m.start(), city, guess)
            for known, city, guess in (
                [(c, c.title(), False) for c in city_airports()]
                + [(country, city, True) for country, city in _COUNTRY_ENTRY_CITY.items()]
            )
            if known != origin_name and (m := re.search(rf"\b{known}\b", low))
        ]
        if named:
            _, out["destination_city"], is_guess = min(named)
            if is_guess:
                guessed.append("destination_city")

    # dates
    dates = _dates(text, today)
    duration = re.search(rf"\b{_NUM}[\s-]*(day|night)s?\b", low)
    if dates:
        out["start_date"] = dates[0].isoformat()
        if len(dates) > 1:
            out["end_date"] = dates[1].isoformat()
        elif duration:
            span = _number(duration.group(1)) - (1 if duration.group(2) == "day" else 0)
            out["end_date"] = (dates[0] + timedelta(days=max(span, 0))).isoformat()
    if re.search(r"\bflexible\b", low) and not re.search(r"\b(?:not|aren't|isn't)\s+flexible\b", low):
        out["dates_flexible"] = True

    # travellers
    adults = re.search(rf"\b{_NUM}\s+adults?\b", low) or re.search(
        rf"\b{_NUM}\s+(?:people|persons|travell?ers|friends)\b", low
    )
    if adults:
        out["adults"] = _number(adults.group(1))
    elif m := re.search(rf"\b{_NUM}\s+of\s+us\b", low):
        out["adults"] = _number(m.group(1))
    elif re.search(r"\b(?:couple|my (?:wife|husband|partner|girlfriend|boyfriend|friend))\b", low):
        out["adults"] = 2
    elif re.search(r"\b(?:solo|alone|by myself|just me)\b", low):
        out["adults"] = 1
    if m := re.search(rf"\b{_NUM}\s+(?:kids?|children|child)\b", low):
        out["children"] = _number(m.group(1))

    # budget
    symbol = re.search(r"([€$£₹])\s?(\d[\d,]*(?:\.\d+)?)\s?(k\b)?", low)
    worded = re.search(r"(\d[\d,]*(?:\.\d+)?)\s?(k)?\s?(eur|euros?|usd|dollars?|gbp|pounds?|inr|rupees?|rs)\b", low)
    bare = re.search(r"budget\D{0,15}?(\d[\d,]*(?:\.\d+)?)\s?(k\b)?", low)
    if symbol:
        out["total_budget"] = _amount(symbol.group(2), symbol.group(3))
        out["currency"] = _CURRENCY[symbol.group(1)]
    elif worded:
        out["total_budget"] = _amount(worded.group(1), worded.group(2))
        out["currency"] = _CURRENCY[worded.group(3)]
    elif bare:
        out["total_budget"] = _amount(bare.group(1), bare.group(2))

    # preferences
    for pace, pattern in _PACE.items():
        if re.search(rf"\b(?:{pattern})\b", low):
            out["pace"] = pace
            break
    interests = [name for name, pattern in _INTERESTS.items() if re.search(rf"\b(?:{pattern})\b", low)]
    if interests:
        out["interests"] = interests
    if m := re.search(r"\b([1-5])[\s-]?star\b", low):
        out["min_hotel_rating"] = float(m.group(1))

    if guessed:
        out["guessed_fields"] = guessed
    return out
