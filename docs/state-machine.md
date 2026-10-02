# State machine

This file is the reference for the trip workflow. It replaces section 3 of the
build spec. `trip_agent/orchestrator/state.py` implements exactly this table,
and a test (`test_state_machine.py`) fails if the two drift apart.

## States

`NEW`, `WAITING_FOR_DETAILS`, `PLANNING`, `VALIDATING`, `REPAIRING`,
`PRESENTED`, `REVISING`, `PARTIAL`.

Terminal: `ACCEPTED`, `FAILED`, `CANCELLED`.

**Changed from the original spec:** `PARTIAL` is no longer terminal. A partial
plan can be revised or cancelled. It cannot be accepted.

## Transitions

Rows marked *added* are not in the original spec.

| From | Event | To | Note |
|---|---|---|---|
| NEW | request_incomplete | WAITING_FOR_DETAILS | Request parsed, required fields missing |
| NEW | request_complete | PLANNING | Request parsed, required fields complete |
| WAITING_FOR_DETAILS | form_valid | PLANNING | Form submitted and valid |
| WAITING_FOR_DETAILS | form_invalid | WAITING_FOR_DETAILS | Field errors shown |
| PLANNING | data_tools | PLANNING | Planner called data tools (loop) |
| PLANNING | ask_user | WAITING_FOR_DETAILS | Planner called `ask_user` |
| PLANNING | propose_plan | VALIDATING | Planner called `propose_plan` |
| PLANNING | give_up | FAILED | Planner called `give_up` |
| PLANNING | loop_guard | PARTIAL, FAILED | A BR-06 limit was hit; FAILED if there is no usable plan |
| VALIDATING | no_violations | PRESENTED | |
| VALIDATING | violations | REPAIRING | Repair not yet used |
| VALIDATING | violations_after_repair | PARTIAL | Presented with the violations listed |
| VALIDATING | tool_gaps | PARTIAL | *Added.* No violations, but a tool was unavailable, so the plan has a stated gap |
| REPAIRING | data_tools | REPAIRING | *Added.* A repair may need a lookup (loop) |
| REPAIRING | repaired_plan | VALIDATING | Planner returned a repaired plan |
| REPAIRING | repair_abandoned | PARTIAL | *Added.* Planner gave up or the repair hit a limit |
| PRESENTED | accept | ACCEPTED | |
| PRESENTED | request_change | REVISING | |
| PARTIAL | request_change | REVISING | *Added.* A partial plan can be revised |
| REVISING | change_scoped | PLANNING | Change scoped to the affected days |
| REVISING | change_rejected | PRESENTED, PARTIAL | *Added.* The request could not be scoped, or it targets a locked item. Returns to the state it came from and asks the user to unlock or rephrase |
| any non-terminal | cancel | CANCELLED | |

The orchestrator rejects any transition not in this table, and any planner
action that is not legal in the current state:

| State | Planner tools offered |
|---|---|
| PLANNING | all seven |
| PLANNING, wrap-up turn | `propose_plan`, `give_up` |
| REPAIRING | the four data tools, `propose_plan`, `give_up` |
| any other | none |

## Limits (BR-06)

Per planning run (a run starts with a new trip, a form submit or a revision):

- **8 iterations.** The 8th is a wrap-up turn: only `propose_plan` and
  `give_up` are offered.
- **Wall clock**, 120 s by default (`TRIP_AGENT_WALL_CLOCK_S`). Once 75% is
  used, the next turn is the wrap-up turn. At 100% the run stops.

Per trip: a **token budget** and an **estimated cost cap**.

A plan proposed in the wrap-up turn is verified and shown as `PARTIAL` with the
limit that was hit. If a hard limit is hit with no new proposal, the last
presented version is kept as `PARTIAL`; with no plan at all the trip is `FAILED`.

## Plan versions

Every verified proposal is saved as a new version; versions are never
overwritten. `reason` is `initial`, `repair`, `revision` or `refresh` (a
re-fetch changed a price). The trip points to the version last shown to the
user. A proposal that goes to repair is saved too, but the pointer only moves
when a plan is presented.
