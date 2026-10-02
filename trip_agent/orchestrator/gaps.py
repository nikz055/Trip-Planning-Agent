"""What a plan is missing because a tool became unavailable (BR-07)."""

from __future__ import annotations

_GAP = {
    "search_flights": "Flights could not be searched, so the plan has no flights and no flight cost.",
    "search_hotels": "Hotels could not be searched, so the plan has no lodging and no lodging cost.",
    "search_places": "Places could not be searched, so the plan has no activities.",
    "get_travel_times": "Travel times could not be fetched, so time between stops is not checked.",
}


def tool_gaps(unavailable_tools: list[str]) -> list[str]:
    return [_GAP.get(tool, f"{tool} was unavailable.") for tool in unavailable_tools]
