"""The storage interface the orchestrator depends on.

A database stores state; it does not make the workflow durable by itself.
Three things here do that work, and the orchestrator relies on all of them:

  * the run lease: `claim` succeeds for one caller at a time, and an expired
    lease can be reclaimed so a crashed run cannot hold a trip forever
  * fencing: `save` only writes while the caller still holds the lease, so a
    run that lost its lease cannot overwrite newer state
  * the action ledger: `begin_action` records an idempotency key once, so a
    repeated request can be answered with the first one's result (BR-16)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from trip_agent.models import Session


class LeaseLost(Exception):
    """This run no longer holds the trip's lease; nothing was written."""


@dataclass
class ActionRecord:
    kind: str
    result: dict[str, Any] | None  # None while the action is still running
    created: bool  # True if this call recorded the key for the first time


class TripStore(Protocol):
    async def insert_trip(
        self, session: Session, key: str, run_id: str, now: datetime, deadline: datetime
    ) -> bool:
        """Create the trip, its 'create' action and its first lease in one transaction.
        Returns False, writing nothing, if the idempotency key was already used."""
        ...

    async def find_created(self, key: str) -> str | None:
        """The id of the trip a 'create' key produced, if any."""
        ...

    async def load(self, trip_id: str) -> Session | None: ...

    async def save(
        self,
        session: Session,
        run_id: str,
        now: datetime,
        action: tuple[str, dict[str, Any]] | None = None,
    ) -> None:
        """Write the state change and everything that caused it in one transaction.

        `action` is (idempotency key, result): the result of the user action that
        ends here, written in the same transaction as the final state.
        Raises LeaseLost, writing nothing, if `run_id` no longer holds the lease."""
        ...

    async def claim(self, trip_id: str, run_id: str, now: datetime, deadline: datetime) -> bool:
        """Take the lease if it is free or expired. Atomic."""
        ...

    async def steal(self, trip_id: str, run_id: str, now: datetime, deadline: datetime) -> None:
        """Take the lease unconditionally (used by cancel to stop a run in flight)."""
        ...

    async def release(self, trip_id: str, run_id: str) -> None: ...

    async def begin_action(self, trip_id: str, key: str, kind: str, now: datetime) -> ActionRecord: ...

    async def get_action(self, trip_id: str, key: str) -> ActionRecord | None: ...

    async def finish_action(self, trip_id: str, key: str, result: dict[str, Any]) -> None: ...

    async def drop_action(self, trip_id: str, key: str) -> None: ...

    async def delete_trip(self, trip_id: str) -> None:
        """Remove a trip and everything that belongs to it (tests clean up with this)."""
        ...
