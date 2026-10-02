"""A scripted LLM for tests: plays back prepared turns, or computes them.

Each step is either a list of tool calls or a function that receives the
conversation and the offered tools and returns such a list. This lets a test
make the "model" do something specific and wrong (an invalid airport code, an
invented hotel) and assert on how the orchestrator reacts.
"""

from __future__ import annotations

from typing import Any, Callable, Union

from trip_agent.llm.base import AssistantTurn, LLMToolCall, Message, ToolSpec, estimate_tokens

Call = dict[str, Any]  # {"name": ..., "args": {...}}
Step = Union[list[Call], Callable[[list[Message], list[ToolSpec]], list[Call]]]


def call(name: str, **args: Any) -> Call:
    return {"name": name, "args": args}


class ScriptedLLM:
    name = "scripted (test double)"

    def __init__(self, steps: list[Step], *, repeat_last: bool = False, cost_per_turn: float = 0.0):
        self._steps = list(steps)
        self._repeat_last = repeat_last
        self._cost_per_turn = cost_per_turn
        self.turns = 0
        self.seen: list[tuple[list[Message], list[str]]] = []

    async def complete(
        self, *, system: str, messages: list[Message], tools: list[ToolSpec]
    ) -> AssistantTurn:
        self.seen.append((list(messages), [t.name for t in tools]))
        index = self.turns
        if index >= len(self._steps):
            if not self._repeat_last or not self._steps:
                raise AssertionError(f"ScriptedLLM ran out of steps at turn {index + 1}")
            index = len(self._steps) - 1
        step = self._steps[index]
        self.turns += 1
        calls = step(messages, tools) if callable(step) else step
        return AssistantTurn(
            tool_calls=[
                LLMToolCall(id=f"scripted_{self.turns}_{i}", name=c["name"], args=c["args"])
                for i, c in enumerate(calls)
            ],
            input_tokens=estimate_tokens(system, *(m.text for m in messages)),
            output_tokens=estimate_tokens(str(calls)),
            cost_usd=self._cost_per_turn,
            stop_reason="tool_use" if calls else "end_turn",
        )
