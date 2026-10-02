"""The tool provider interface.

Mock implementations sit behind this today; a real supplier integration only
has to implement `call` and set `source`. Providers never see the LLM and the
LLM never sees a provider (BR-03).
"""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel


class ToolError(Exception):
    """A tool call failed in a way that is worth retrying."""


class ToolTimeout(ToolError):
    """A tool call did not answer in time."""


class ToolProvider(Protocol):
    source: str

    async def call(self, tool: str, args: BaseModel) -> Any:
        """Run one validated data-tool call and return its JSON-serialisable payload."""
        ...
