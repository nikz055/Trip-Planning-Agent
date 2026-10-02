# Evaluation report

- **Planner:** STAND-IN planner (rule-based, not an LLM)
- **Run at:** 2026-10-01 17:25 UTC (scenario date fixed at 2026-10-01)
- **Scenarios:** 40 scripted requests from `requests.json`, sample (mock) supplier data

> **This run used the stand-in planner, which is rule-based and not a language model.**
> The numbers below describe the guardrail pipeline: whether the orchestrator, executor,
> verifier and repair step behave as specified. They say nothing about how well a real
> model plans. Token counts are character-based estimates and cost is zero by construction.

## Summary

| Metric | Value |
|---|---|
| Task success (presented, no violations, no gaps) | 31 of 40 (78%) |
| Ended in the state the scenario expects | 40 of 40 |
| Verifier violations per proposal, before repair | 0.34 |
| Verifier violations per plan shown, after repair | 0.11 |
| Repairs attempted / cleared every violation | 7 / 4 |
| Ungrounded items per plan shown (target 0) | 0.03 |
| Ungrounded items proposed and rejected | 3 |
| Tool calls per scenario | 6.5 |
| Planner calls per scenario | 5.5 |
| Latency per scenario | 0.08 s average, 0.26 s maximum |
| Tokens per scenario | 9630 |
| Estimated LLM cost, whole run | USD 0.0000 |
| Clarification rounds per scenario | 0.17 |
| Scenarios that ended in an error | 0 |

Task success is not expected to be 100%: the set deliberately includes requests that
cannot end in a clean plan (supplier failures, impossible budgets, destinations with no data).
"Ended in the state the scenario expects" is the measure of whether the system did the right thing.

## By category

| Category | Scenarios | Clean plan | As expected |
|---|---|---|---|
| complete | 12 | 12 | 12 |
| missing details | 5 | 5 | 5 |
| dates | 3 | 3 | 3 |
| budget | 4 | 2 | 4 |
| supplier failure | 5 | 1 | 5 |
| cannot plan | 2 | 0 | 2 |
| injected planner mistake | 5 | 4 | 5 |
| revision | 3 | 3 | 3 |
| prompt injection | 1 | 1 | 1 |

## Every scenario

| Scenario | Edge case | Final state | Expected | OK | Violations first / final | Tool calls | Planner calls | Seconds |
|---|---|---|---|---|---|---|---|---|
| vienna-london |  | PRESENTED | PRESENTED | yes | 0 / 0 | 6 | 5 | 0.06 |
| vienna-art-music |  | PRESENTED | PRESENTED | yes | 0 / 0 | 6 | 5 | 0.06 |
| vienna-mumbai-balanced |  | PRESENTED | PRESENTED | yes | 0 / 0 | 6 | 5 | 0.05 |
| lisbon-london-solo |  | PRESENTED | PRESENTED | yes | 0 / 0 | 6 | 5 | 0.05 |
| lisbon-mumbai |  | PRESENTED | PRESENTED | yes | 0 / 0 | 6 | 5 | 0.06 |
| lisbon-newyork |  | PRESENTED | PRESENTED | yes | 0 / 0 | 6 | 5 | 0.05 |
| jaipur-bengaluru |  | PRESENTED | PRESENTED | yes | 0 / 0 | 6 | 5 | 0.06 |
| jaipur-delhi-packed |  | PRESENTED | PRESENTED | yes | 0 / 0 | 6 | 5 | 0.06 |
| goa-chennai-family |  | PRESENTED | PRESENTED | yes | 0 / 0 | 6 | 5 | 0.05 |
| goa-hyderabad |  | PRESENTED | PRESENTED | yes | 0 / 0 | 6 | 5 | 0.05 |
| vienna-singapore |  | PRESENTED | PRESENTED | yes | 0 / 0 | 6 | 5 | 0.05 |
| vienna-usd-budget |  | PRESENTED | PRESENTED | yes | 0 / 0 | 6 | 5 | 0.05 |
| austria-only | 1 | PRESENTED | PRESENTED | yes | 0 / 0 | 7 | 5 | 0.05 |
| no-dates |  | PRESENTED | PRESENTED | yes | 0 / 0 | 7 | 5 | 0.05 |
| no-origin |  | PRESENTED | PRESENTED | yes | 0 / 0 | 7 | 5 | 0.05 |
| no-travellers |  | PRESENTED | PRESENTED | yes | 0 / 0 | 7 | 5 | 0.06 |
| bad-dates-then-good | 2 | PRESENTED | PRESENTED | yes | 0 / 0 | 7 | 5 | 0.05 |
| fixed-dates-no-flights | 4 | PRESENTED | PRESENTED | yes | 0 / 0 | 18 | 7 | 0.08 |
| flexible-dates-no-flights | 5 | PRESENTED | PRESENTED | yes | 0 / 0 | 18 | 6 | 0.07 |
| overnight-flight | 9 | PRESENTED | PRESENTED | yes | 0 / 0 | 6 | 5 | 0.06 |
| hotel-relax-rating | 6 | PRESENTED | PRESENTED | yes | 0 / 0 | 7 | 6 | 0.07 |
| five-star-low-budget | 13 | PRESENTED | PRESENTED or PARTIAL | yes | 0 / 0 | 8 | 8 | 0.09 |
| over-budget | 12 | PARTIAL | PARTIAL | yes | 1 / 1 | 8 | 8 | 0.09 |
| tight-jaipur |  | PARTIAL | PRESENTED or PARTIAL | yes | 1 / 1 | 8 | 8 | 0.09 |
| flights-timeout | 7 | PARTIAL | PARTIAL | yes | 0 / 0 | 6 | 5 | 0.21 |
| hotels-error |  | PARTIAL | PARTIAL | yes | 0 / 0 | 6 | 5 | 0.20 |
| places-error |  | PARTIAL | PARTIAL | yes | 0 / 0 | 4 | 4 | 0.20 |
| travel-times-error |  | PARTIAL | PARTIAL | yes | 0 / 0 | 6 | 5 | 0.22 |
| flights-slow |  | PRESENTED | PRESENTED | yes | 0 / 0 | 6 | 5 | 0.26 |
| no-data-destination |  | FAILED | FAILED | yes | - / - | 0 | 2 | 0.01 |
| unknown-origin |  | FAILED | FAILED | yes | - / - | 0 | 2 | 0.01 |
| flaw-monday-closed | 10 | PRESENTED | PRESENTED | yes | 5 / 0 | 6 | 6 | 0.06 |
| flaw-invented-hotel | 11 | PRESENTED | PRESENTED | yes | 2 / 0 | 6 | 6 | 0.06 |
| flaw-bad-arithmetic |  | PRESENTED | PRESENTED | yes | 1 / 0 | 6 | 6 | 0.10 |
| flaw-no-buffer |  | PRESENTED | PRESENTED | yes | 2 / 0 | 6 | 6 | 0.06 |
| flaw-invented-hotel-twice |  | PARTIAL | PARTIAL | yes | 2 / 2 | 6 | 6 | 0.07 |
| revise-less-walking | 14 | PRESENTED | PRESENTED | yes | 0+0 / 0 | 6 | 7 | 0.07 |
| revise-twice | 20 | PRESENTED | PRESENTED | yes | 0+0+0 / 0 | 6 | 9 | 0.08 |
| revise-locked-hotel |  | PRESENTED | PRESENTED | yes | 0 / 0 | 6 | 6 | 0.06 |
| injection-in-place-text | 17 | PRESENTED | PRESENTED | yes | 0 / 0 | 6 | 5 | 0.05 |

## Not covered here

The user test in spec section 11 (time to a usable plan, share of plans accepted after at most
one revision, share of users who correctly tell confirmed from estimated costs) needs 5 to 10
people and has not been run.
