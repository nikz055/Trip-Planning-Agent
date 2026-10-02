"""Provider-agnostic LLM interface.

The orchestrator only ever talks to `LLMClient.complete`. A provider turns the
neutral `Message` list into its own wire format and back. `raw` carries the
provider's untouched assistant content so it can be replayed verbatim on later
calls (some providers require their reasoning blocks to be echoed unchanged).
"""

from __future__ import annotations

from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field


class ToolSpec(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any]


class LLMToolCall(BaseModel):
    id: str
    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class LLMToolResult(BaseModel):
    tool_call_id: str
    content: str  # JSON text; tool results are data, never instructions (BR-11)
    is_error: bool = False


class Message(BaseModel):
    role: Literal["user", "assistant"]
    text: str | None = None
    tool_calls: list[LLMToolCall] = Field(default_factory=list)  # assistant only
    tool_results: list[LLMToolResult] = Field(default_factory=list)  # user only
    raw: Any | None = None  # provider-native assistant content, replayed as-is


class AssistantTurn(BaseModel):
    text: str | None = None
    tool_calls: list[LLMToolCall] = Field(default_factory=list)
    raw: Any | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0  # estimate from token counts and list prices
    stop_reason: str | None = None

    def as_message(self) -> Message:
        return Message(role="assistant", text=self.text, tool_calls=self.tool_calls, raw=self.raw)


class LLMError(Exception):
    """The provider could not produce a turn (network, auth, ...)."""


class LLMClient(Protocol):
    name: str  # shown to the user, so a stand-in must say that it is one

    async def complete(
        self, *, system: str, messages: list[Message], tools: list[ToolSpec]
    ) -> AssistantTurn: ...


def estimate_tokens(*parts: str | None) -> int:
    """Rough token estimate for offline planners (about 4 characters per token)."""
    return sum(len(p) for p in parts if p) // 4
