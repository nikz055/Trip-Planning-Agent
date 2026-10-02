"""Run the web app:  python -m trip_agent.api [--env-file .env.test] [--llm standin] [--port 8000]"""

from __future__ import annotations

import argparse
import os

import uvicorn

from trip_agent.api.app import create_app
from trip_agent.config import Settings
from trip_agent.env import load_env


def main() -> None:
    parser = argparse.ArgumentParser(prog="trip_agent.api")
    parser.add_argument("--env-file", default=None, help="load this file before .env (e.g. .env.test for the local stack)")
    parser.add_argument("--llm", choices=["auto", "openrouter", "standin"], default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.env_file:
        os.environ["TRIP_AGENT_ENV_FILE"] = args.env_file
    load_env()
    settings = Settings.from_env(**({"llm_provider": args.llm} if args.llm else {}))
    uvicorn.run(create_app(settings=settings), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
