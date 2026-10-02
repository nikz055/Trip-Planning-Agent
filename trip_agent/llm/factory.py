"""Chooses the planner implementation from settings and the environment."""

from __future__ import annotations

import os

from trip_agent.config import Settings
from trip_agent.llm.base import LLMClient


def has_llm_credentials() -> bool:
    return bool(os.environ.get("LLM_API_KEY") and os.environ.get("LLM_MODEL"))


def make_llm(settings: Settings) -> LLMClient:
    """auto = the configured model endpoint if LLM_API_KEY and LLM_MODEL are set,
    otherwise the stand-in."""
    provider = settings.llm_provider
    if provider == "auto":
        provider = "openrouter" if has_llm_credentials() else "standin"
    if provider == "standin":
        from trip_agent.llm.standin import StandInPlanner

        return StandInPlanner()
    if provider == "openrouter":
        from trip_agent.llm.openai_compat_client import DEFAULT_BASE_URL, OpenAICompatClient

        if not has_llm_credentials():
            raise ValueError("LLM_API_KEY and LLM_MODEL must be set in .env to use the openrouter provider")
        return OpenAICompatClient(
            model=os.environ["LLM_MODEL"],
            api_key=os.environ["LLM_API_KEY"],
            base_url=os.environ.get("LLM_BASE_URL") or DEFAULT_BASE_URL,
            max_tokens=settings.llm_max_tokens,
        )
    raise ValueError(f"unknown llm_provider '{settings.llm_provider}' (auto | openrouter | standin)")
