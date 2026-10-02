# Trip-Planning Agent

A bounded workflow with one LLM inside it. A traveller describes a 3â€“7 day
city trip in plain language and gets one day-by-day plan with flights, a
hotel, activities and travel time between stops, with confirmed costs kept
apart from estimates. Plans can be revised in plain words without starting
over. Planning only: no booking, no payments, no accounts.

**What is and is not verified**

- The guardrails, revisions, API and web UI are tested (398 tests) on an
  in-memory store and on Postgres (the local Supabase stack).
- All supplier data is **sample data**. Place names are real; hours, prices,
  hotels and flights are illustrative.
- Tests and the committed evaluation report use a **rule-based stand-in
  planner, not a language model**. A real model has driven the pipeline
  through its tool calls but has not yet completed a full plan, so there is no
  real-model evaluation.
- A hosted Supabase project has not been migrated or tested.

## Architecture

| Component | Role |
|---|---|
| **Planner LLM** (`trip_agent/llm/`) | Interprets the request, decides the next action, writes the plan. Sees only tool definitions and tool results; never holds credentials, never calls an API. |
| **Orchestrator** (`trip_agent/orchestrator/`) | A plain Python state machine. Owns the session, validates every planner action against the state and the tool schemas, runs tools, enforces the business rules and limits, decides when the loop ends. |
| **Verifier** (`trip_agent/orchestrator/verifier.py`) | Deterministic code. Nine checks on every proposed plan: structure and dates, grounding, opening hours, travel time and buffers, pace, hotel coverage, costs and budget, freshness, locks and revision scope. One repair attempt per proposal. |
| **Tools** (`trip_agent/tools/`) | Four data tools behind a `ToolProvider` interface. Mock implementations today; real suppliers implement the same interface. |
| **Storage** (`trip_agent/db/`) | Postgres through `asyncpg`. A run lease with expiry, fenced saves, idempotent user actions and immutable plan versions. |
| **API and UI** (`trip_agent/api/`, `trip_agent/ui/`) | FastAPI routes and a no-build web UI. The browser talks only to the API. |

Hard rules live in code. The planner prompt (`trip_agent/llm/prompts.py`)
guides the model; the orchestrator and verifier enforce. The workflow, its
states and its limits are documented in
[docs/state-machine.md](docs/state-machine.md), and a test fails if the code
and that document disagree.

### The planner is swappable

`LLMClient` (`trip_agent/llm/base.py`) has three implementations:

- **`OpenAICompatClient`** â€” any OpenAI-compatible chat-completions endpoint
  with tool calling, such as OpenRouter. The model must be able to emit a
  whole plan as one tool call.
- **`StandInPlanner`** â€” a rule-based stand-in, *not a language model*. It
  goes through the same orchestrator, tools and checks, so the pipeline can be
  run and tested without an API key. Wherever it is used, output is labelled
  `STAND-IN`.
- **`ScriptedLLM`** â€” a test double that plays back prepared tool calls,
  including deliberately wrong ones.

`TRIP_AGENT_LLM_PROVIDER=auto` (the default) uses the configured endpoint if
`LLM_API_KEY` and `LLM_MODEL` are set and the stand-in otherwise.
`--llm openrouter` or `--llm standin` forces one.

## Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"     # Windows; use .venv/bin/python elsewhere
cp .env.example .env                                # then fill in; .env is git-ignored
```

Nothing in `.env` is needed to try the CLI with the stand-in planner.

## Command line

```bash
.venv/Scripts/python -m trip_agent "5 days in Vienna from London, 7-11 Nov 2026, 2 adults, budget 2500 EUR, we like art and music" --llm standin
```

- `--set FIELD=VALUE` answers a form field (`--set origin=London`).
- `--fail TOOL=MODE` injects a failure into a mock tool
  (`--fail search_flights=timeout`; modes: `timeout`, `empty`, `error`, `slow`).
- `--llm standin|openrouter|auto` picks the planner.

With a database configured, `new`, `answer`, `show`, `revise`, `lock`,
`accept` and `cancel` work on stored trips; each command is a separate
process, so a trip carries on across restarts.

## Database (Supabase Postgres)

Local stack (needs Docker Desktop):

```bash
npx supabase start -x studio,realtime,storage-api,imgproxy,edge-runtime,logflare,vector,postgres-meta,mailpit,supavisor
.venv/Scripts/python scripts/use_local_supabase.py       # writes .env.test (git-ignored)
```

Tests load `.env.test` before `.env`, so with that file present they use the
local stack. For the commands below, set `TRIP_AGENT_ENV_FILE=.env.test` to do
the same; without it they use `DATABASE_URL` from `.env`.

```bash
.venv/Scripts/python -m trip_agent.db.migrate            # apply supabase/migrations, report tables and RLS
.venv/Scripts/python -m trip_agent.db.migrate --status   # report only
.venv/Scripts/python scripts/restart_demo.py             # a trip surviving three separate processes
```

`supabase/migrations/` is the source of truth. The backend connects with
`DATABASE_URL`; the browser never talks to the database. Row Level Security is
on for every table with no policies, so the public key reads nothing.

## Web app

```bash
.venv/Scripts/python -m trip_agent.api --env-file .env.test --llm standin
```

Then open http://127.0.0.1:8000. Leave out `--env-file .env.test` to use
`DATABASE_URL` from `.env`; with no database configured the app keeps trips in
memory. Leave out `--llm standin` to use the model configured in `.env`. A
plan's share link is `/share/<trip id>` and is read-only.

## Tests

```bash
.venv/Scripts/python -m pytest
```

Every orchestrator test runs on the in-memory store and on Postgres. Tests
that need the database are skipped, and say so, when `DATABASE_URL` is not
set. Every trip a test creates is deleted when the test ends.

## Evaluation

```bash
.venv/Scripts/python -m trip_agent.eval.harness --llm standin      # all 40 scripted requests
.venv/Scripts/python -m trip_agent.eval.harness --llm openrouter   # the model configured in .env
```

Scenarios are in `trip_agent/eval/requests.json`; reports are written to
`trip_agent/eval/reports/`. The committed `report-standin.md` was produced with
the stand-in planner, so it measures the guardrails, not a model's planning.
A user test with real travellers has not been run.

## Sample data

`trip_agent/data/mock/` holds four destinations (Vienna, Lisbon, Jaipur, Goa)
with 15 hotels and 30 places each, hand-tuned flight routes, and a table of
about 120 airports from which flights for any other pair are generated. The
JSON is generated: edit `build_mock_data.py` and run
`python -m trip_agent.data.mock.build_mock_data`. A test fails if the JSON and
the builder drift apart. Prices are in EUR throughout.
