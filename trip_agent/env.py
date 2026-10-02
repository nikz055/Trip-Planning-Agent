"""Minimal .env loader. Values are put into os.environ and never printed or logged."""

from __future__ import annotations

import os
from pathlib import Path


def load_env(path: str | Path = ".env") -> list[str]:
    """Load KEY=VALUE lines without overriding variables that are already set.

    Returns the names (never the values) that were loaded.
    """
    first = os.environ.pop("TRIP_AGENT_ENV_FILE", None)
    if first:
        # an earlier file wins, so e.g. .env.test (the local stack) takes priority over .env
        return load_env(first) + load_env(path)
    file = Path(path)
    if not file.is_file():
        return []
    loaded = []
    for line in file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded
