"""Config-driven failure injection for the mock tools (spec section 5)."""

from __future__ import annotations

import threading
from typing import Literal

from pydantic import BaseModel, Field

FailureMode = Literal["timeout", "empty", "error", "slow"]


class FailureSpec(BaseModel):
    mode: FailureMode
    times: int | None = Field(
        default=None, description="How many calls are affected; None means every call."
    )
    delay_s: float = Field(default=0.2, description="Sleep for 'slow' responses.")


class FailureInjector:
    """Per-tool failure plan, e.g. {"search_flights": {"mode": "timeout"}}."""

    def __init__(self, config: dict[str, FailureSpec | dict] | None = None):
        self._specs = {
            tool: spec if isinstance(spec, FailureSpec) else FailureSpec(**spec)
            for tool, spec in (config or {}).items()
        }
        self._lock = threading.Lock()

    def set(self, tool: str, spec: FailureSpec | dict | None) -> None:
        with self._lock:
            if spec is None:
                self._specs.pop(tool, None)
            else:
                self._specs[tool] = spec if isinstance(spec, FailureSpec) else FailureSpec(**spec)

    def next(self, tool: str) -> FailureSpec | None:
        """Return the failure to apply to this call, consuming one use of it."""
        with self._lock:
            spec = self._specs.get(tool)
            if spec is None:
                return None
            if spec.times is not None:
                if spec.times <= 0:
                    return None
                spec.times -= 1
            return spec
