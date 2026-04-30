# Pipeline Labels (`agent:*`)

This document defines all six `agent:*` GitHub labels used by the fourth-path-gemma agent pipeline.
These labels coordinate state transitions between the workflow, Claude, and SHIHCHENG.

---

## Label Reference

| Label | Color | Purpose | When to apply | Who applies it |
|---|---|---|---|---|
| `agent:plan` | `#ededed` | Signals that an issue is in the planning phase — scope and decomposition in progress | When a new issue needs decomposition before execution can begin | Workflow (`claude.yml`) or human at triage |
| `agent:run` | `#0E8A16` | Triggers the Claude workflow to process the issue | When scope is clear and Claude should execute the task | Human (SHIHCHENG) adds to kick off; not removed automatically by the current workflow |
| `agent:stuck` | `#B60205` | Pipeline has failed repeatedly; a refusal log entry has been written; awaiting human decision or future recovery handling | When Claude cannot make progress after multiple attempts and has recorded the failure in `docs/refusal-log.md` | Workflow currently adds this in the PR-review repeated-review path |
| `agent:needs-user-decision` | `#D93F0B` | Pipeline cannot infer the next safe step without human input | Planned recovery label for ambiguous, out-of-scope, or governance-blocked tasks | Not automated yet; reserved for future recovery flow or manual use |
| `agent:child` | `#FBCA04` | Marks an issue as a child task decomposed from a parent issue | When a parent issue is broken down into smaller executable sub-issues | Planning workflow / Claude during planning phase |
| `agent:parent` | `#0052CC` | Marks an issue that has spawned one or more child tasks | When an issue has been decomposed and child issues have been created | Planning workflow / Claude during planning phase |

---

## Current automation status

| Behavior | Current status |
|---|---|
| `agent:plan` triggers planning-only mode | Implemented in `.github/workflows/claude.yml` |
| `agent:run` triggers implementation mode | Implemented in `.github/workflows/claude.yml` |
| First child issue receives `agent:run` during planning | Implemented by the planning prompt, not by a separate deterministic workflow step |
| `agent:stuck` is added after repeated review detection | Implemented in the PR-review path |
| `agent:run` is removed automatically | Not implemented |
| `agent:needs-user-decision` is added automatically | Not implemented yet; planned for the recovery-flow child issue |

---

## State-transition rules

- `agent:plan` → child issue with `agent:run`: current planning flow creates child issues and labels only the first executable child with `agent:run`.
- `agent:run` → `agent:stuck`: currently automated only in the PR-review repeated-review path; `agent:run` is **not** removed automatically and may remain present.
- `agent:run` → `agent:needs-user-decision`: planned, not yet automated. Until the recovery-flow child issue is implemented, this transition is manual.
- `agent:parent` + `agent:child`: set during decomposition; neither implies execution state on its own.
- Operators should avoid keeping more than one execution-state label (`agent:plan`, `agent:run`, `agent:stuck`, `agent:needs-user-decision`) active on the same issue, but the current workflow does not enforce this automatically.

---

## Validation

```bash
test -f docs/pipeline-labels.md && grep -c "agent:" docs/pipeline-labels.md
```

Expected: exits 0, prints 6 or more.
