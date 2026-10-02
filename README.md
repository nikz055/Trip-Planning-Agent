# Trip-Planning Agent

**A trip plan you can actually follow.** Describe a city trip in one sentence
and get a day-by-day itinerary with flights, a hotel, activities and realistic
travel time between stops, with every cost marked as confirmed or estimated.

## The problem

AI trip planners are easy to demo and hard to trust. They invent hotels,
schedule museums on the day they are closed, forget the time it takes to cross
a city, and call a plan "within budget" when it is not. First-time
international travellers are the people least able to spot those mistakes.

This project treats the language model as one component inside a controlled
workflow rather than as the whole product. The model proposes; deterministic
code checks every proposal before the traveller sees it.

## What it does

- **Plain-language requests.** "5 days in Vienna from London, 7–11 Nov, 2
  adults, budget 2500 EUR, we like art and music" is enough to start.
- **One short form, at most.** If details are missing, a single form asks for
  all of them together, with ideas for the destination shown alongside.
- **A complete day-wise plan.** Flights both ways, a hotel for every night,
  activities that fit the pace you asked for, and transfers between stops.
- **Costs you can trust.** Each price shows its source and when it was
  fetched. Confirmed costs and estimates are labelled and totalled separately.
- **Honest budgets.** The whole trip is counted: flights, lodging, local
  transport, activities and meals. If the plan is over budget, it says by how
  much.
- **Revise without starting over.** Lock what you want to keep, then ask for a
  change such as "less walking on day 2". Only the affected days are replanned.
- **Version history.** Every revision is saved as a new version; earlier ones
  are never overwritten.
- **Shareable plans.** Each plan has a read-only link with links to providers.

## What makes it reliable

**Nothing is made up.** Every flight, hotel and attraction in a plan must
trace back to a stored search result. Anything that does not is rejected.

**Nine checks on every plan**, run by deterministic code before the plan is shown:

1. Structure and dates match the trip
2. Every item is grounded in a search result
3. Activities fall within opening hours and avoid closures
4. Travel time and a 30-minute buffer fit between stops
5. The day respects the chosen pace, and the arrival day stays light
6. A hotel covers every night, with check-in after the flight lands
7. Costs match the search results and the budget adds up
8. Prices are fresh; stale ones are re-fetched before the plan is shown
9. Locked items and untouched days are unchanged after a revision

**One chance to repair.** If a check fails, the planner is told exactly what is
wrong and gets one attempt to fix it. If problems remain, the plan is shown as
partial with the problems listed, never as a clean plan.

**Bounded, predictable runs.** Planning is limited by iterations, tokens,
estimated cost and wall-clock time. Near a limit, the planner is asked to wrap
up with the best plan it has, so the traveller gets a result instead of a
timeout.

**Graceful supplier failures.** Failed searches are retried with backoff. If a
supplier stays down, the plan is still produced and states plainly what is
missing.

**Resistant to prompt injection.** Search results are treated as data, never as
instructions. Text inside a result that tries to steer the planner has no
effect, and anything it might add fails the grounding check.

**Built to survive restarts.**

- Every state change is saved with the data that caused it, in one transaction.
- A trip left mid-run by a crash is picked up where it stopped, without
  repeating finished searches.
- A double-clicked button never starts a second planning run: every action is
  idempotent.
- A run lease with an expiry guarantees one active planning run per trip,
  enforced in the database.

**Safe by design.** The model never sees credentials and never calls an API
directly. The browser never talks to the database. Row Level Security is on
for every table, so the public key can read nothing.

## How it works

```
 request ──► intake ──► PLANNING ◄──► tools (flights, hotels, places, travel times)
                           │
                     propose plan
                           ▼
                      VALIDATING ──► violations ──► REPAIRING (once)
                           │
                           ▼
                       PRESENTED ──► lock / revise ──► PLANNING (affected days only)
                           │
                        accept
```

| Component | Role |
|---|---|
| **Planner LLM** (`trip_agent/llm/`) | Interprets the request, decides the next action, writes the plan. |
| **Orchestrator** (`trip_agent/orchestrator/`) | A plain Python state machine. Validates every planner action, runs tools, enforces the rules and limits. |
| **Verifier** (`trip_agent/orchestrator/verifier.py`) | The nine deterministic checks. |
| **Tools** (`trip_agent/tools/`) | Flight, hotel, place and travel-time searches behind one interface, ready for real suppliers. |
| **Storage** (`trip_agent/db/`) | Supabase Postgres: trips, plan versions, tool calls and results, and an action ledger. |
| **API and UI** (`trip_agent/api/`, `trip_agent/ui/`) | FastAPI routes and a no-build web interface. |

The full state machine is documented in
[docs/state-machine.md](docs/state-machine.md), and a test keeps the code and
that document in step.

**Any model, one interface.** The planner sits behind a small `LLMClient`
interface. It works with any OpenAI-compatible endpoint that supports tool
calling (OpenRouter by default), and ships with a rule-based stand-in so the
whole pipeline runs offline with no API key.

## Tech stack

Python 3.11+ · FastAPI · Pydantic · asyncpg · Supabase (PostgreSQL) · pytest ·
vanilla JavaScript

## Quick start

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"     # Windows; use .venv/bin/python elsewhere
```

Plan a trip from the command line, no keys or database needed:

```bash
.venv/Scripts/python -m trip_agent "5 days in Vienna from London, 7-11 Nov 2026, 2 adults, budget 2500 EUR" --llm standin
```

Run the web app, then open http://127.0.0.1:8000:

```bash
.venv/Scripts/python -m trip_agent.api --llm standin
```

### Use a real model

Copy `.env.example` to `.env` and set `LLM_API_KEY` and `LLM_MODEL`
(`LLM_BASE_URL` defaults to OpenRouter). Drop `--llm standin` to use it.

### Use a database

With Docker Desktop running:

```bash
npx supabase start -x studio,realtime,storage-api,imgproxy,edge-runtime,logflare,vector,postgres-meta,mailpit,supavisor
.venv/Scripts/python scripts/use_local_supabase.py           # writes .env.test
.venv/Scripts/python -m trip_agent.api --env-file .env.test --llm standin
```

For a hosted Supabase project, put its connection string in `.env` as
`DATABASE_URL` and apply the schema with
`python -m trip_agent.db.migrate`. The SQL in `supabase/migrations/` is the
source of truth.

## Sample data

The demo ships with four destinations (Vienna, Lisbon, Jaipur and Goa), each
with 15 hotels and 30 places, and about 120 departure airports worldwide. The
catalogue deliberately includes awkward cases, such as an overnight flight, a
museum closed on Mondays, a hotel with late check-in, a date with no flights
and a place description that tries to give the planner instructions, so the
safeguards have something real to catch.

## Testing and evaluation

```bash
.venv/Scripts/python -m pytest                                  # 398 tests
.venv/Scripts/python -m trip_agent.eval.harness --llm standin   # 40 scripted requests
.venv/Scripts/python scripts/restart_demo.py                    # one trip across three processes
```

- **398 automated tests**, including one for each of 21 specified edge cases.
  Orchestrator tests run on both an in-memory store and Postgres.
- **An evaluation harness** that runs 40 scripted requests end to end and
  reports task success, violations before and after repair, ungrounded items,
  tool calls, latency, tokens, cost and clarification rounds. On the stand-in
  planner all 40 end in the expected state; the report is in
  `trip_agent/eval/reports/`.
