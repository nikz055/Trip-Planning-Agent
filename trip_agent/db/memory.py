"""In-process store with the same semantics as the Postgres one.

Used by the one-shot `plan` CLI command and to run the orchestrator's logic
tests without a database. It keeps serialized copies, so nothing is shared by
reference between a "saved" session and the one in use. It does not survive a
process restart; anything about durability is tested against Postgres.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from trip_agent.db.base import ActionRecord, LeaseLost
from trip_agent.models import Session


@dataclass
class _Row:
    data: str  # Session JSON
    run_id: str | None = None
    run_deadline: datetime | None = None
    actions: dict[str, ActionRecord] = field(default_factory=dict)


class MemoryStore:
    def __init__(self) -> None:
        self._trips: dict[str, _Row] = {}
        self._create_keys: dict[str, str] = {}

    async def insert_trip(
        self, session: Session, key: str, run_id: str, now: datetime, deadline: datetime
    ) -> bool:
        if key in self._create_keys:
            return False
        row = _Row(data=session.model_dump_json(), run_id=run_id, run_deadline=deadline)
        row.actions[key] = ActionRecord(kind="create", result=None, created=True)
        self._trips[session.trip_id] = row
        self._create_keys[key] = session.trip_id
        return True

    async def find_created(self, key: str) -> str | None:
        return self._create_keys.get(key)

    async def load(self, trip_id: str) -> Session | None:
        row = self._trips.get(trip_id)
        return Session.model_validate_json(row.data) if row else None

    async def save(
        self,
        session: Session,
        run_id: str,
        now: datetime,
        action: tuple[str, dict[str, Any]] | None = None,
    ) -> None:
        row = self._trips.get(session.trip_id)
        if row is None or row.run_id != run_id:
            raise LeaseLost(session.trip_id)
        session.updated_at = now
        row.data = session.model_dump_json()
        if action is not None:
            row.actions[action[0]].result = action[1]

    async def claim(self, trip_id: str, run_id: str, now: datetime, deadline: datetime) -> bool:
        row = self._trips.get(trip_id)
        if row is None:
            return False
        if row.run_id is not None and row.run_deadline is not None and row.run_deadline >= now:
            return False
        row.run_id, row.run_deadline = run_id, deadline
        return True

    async def steal(self, trip_id: str, run_id: str, now: datetime, deadline: datetime) -> None:
        row = self._trips[trip_id]
        row.run_id, row.run_deadline = run_id, deadline

    async def release(self, trip_id: str, run_id: str) -> None:
        row = self._trips.get(trip_id)
        if row is not None and row.run_id == run_id:
            row.run_id = row.run_deadline = None

    async def begin_action(self, trip_id: str, key: str, kind: str, now: datetime) -> ActionRecord:
        actions = self._trips[trip_id].actions
        if key in actions:
            existing = actions[key]
            return ActionRecord(kind=existing.kind, result=existing.result, created=False)
        actions[key] = ActionRecord(kind=kind, result=None, created=True)
        return actions[key]

    async def get_action(self, trip_id: str, key: str) -> ActionRecord | None:
        row = self._trips.get(trip_id)
        existing = row.actions.get(key) if row else None
        return ActionRecord(kind=existing.kind, result=existing.result, created=False) if existing else None

    async def finish_action(self, trip_id: str, key: str, result: dict[str, Any]) -> None:
        self._trips[trip_id].actions[key].result = result

    async def drop_action(self, trip_id: str, key: str) -> None:
        row = self._trips.get(trip_id)
        if row is not None:
            row.actions.pop(key, None)
            if self._create_keys.get(key) == trip_id:
                del self._create_keys[key]

    async def delete_trip(self, trip_id: str) -> None:
        self._trips.pop(trip_id, None)
        self._create_keys = {k: v for k, v in self._create_keys.items() if v != trip_id}
