"""Demo: a trip survives a server restart.

Runs three separate Python processes against the database in DATABASE_URL.
Nothing is shared between them except what is stored there.

    python scripts/restart_demo.py           (add --keep to leave the trip in the database)
"""

from __future__ import annotations

import asyncio
import re
import subprocess
import sys

STEPS = [
    ("Process 1: start a trip with only a destination", ["new", "Plan a trip to Austria"]),
    ("Process 2 (after a 'restart'): read the same trip back", ["show", "{trip}"]),
    (
        "Process 3 (another 'restart'): submit the form; the same trip continues",
        ["answer", "{trip}", "--set", "origin=London", "--set", "adults=2",
         "--set", "start_date=2026-11-07", "--set", "end_date=2026-11-11"],
    ),  # fmt: skip
]


def run(args: list[str]) -> str:
    done = subprocess.run(
        [sys.executable, "-m", "trip_agent", *args, "--llm", "standin"],
        capture_output=True, text=True, encoding="utf-8",
    )  # fmt: skip
    return done.stdout + done.stderr


async def delete(trip_id: str) -> None:
    from trip_agent.db.postgres import PostgresStore
    from trip_agent.env import load_env

    load_env()
    store = await PostgresStore.open(max_size=1)
    await store.delete_trip(trip_id)
    await store.close()


def main() -> int:
    trip = ""
    for title, args in STEPS:
        output = run([a.format(trip=trip) for a in args])
        print(f"\n=== {title} ===")
        print("\n".join(output.splitlines()[:14]))
        found = re.search(r"^Trip ([0-9a-f-]{36})", output, re.M)
        if not found:
            print("\nNo trip id in the output; stopping.")
            return 1
        if trip and found.group(1) != trip:
            print("\nA DIFFERENT TRIP ID CAME BACK: the trip did not survive.")
            return 1
        trip = found.group(1)
    print(f"\nSame trip id in all three processes: {trip}")
    if "--keep" not in sys.argv:
        asyncio.run(delete(trip))
        print("Demo trip deleted from the database.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
