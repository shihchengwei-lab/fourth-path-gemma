# Pipeline Labels (`agent:*`)

This document defines all six `agent:*` GitHub labels used by the fourth-path-gemma agent pipeline.
These labels coordinate state transitions between the workflow, Claude, and SHIHCHENG.

---

## Label Reference

| Label | Color | Purpose | When to apply | Who applies it |
|---|---|---|---|---|
| `agent:plan` | `#ededed` | Signals that an issue is in the planning phase — scope and decomposition in progress | When a new issue needs decomposition before execution can begin | Workflow (`claude.yml`) or human at triage |
| `agent:run` | `#0E8A16` | Triggers the Claude workflow to process the issue | When scope is clear and Claude should execute the task | Human (SHIHCHENG) adds to kick off; not removed automatically by the current workflow |
| `agent:stuck` | `#B60205` | Pipeline has failed repeatedly; a refusal log entry has been written; awaiting human decision | When Claude cannot make progress after multiple attempts and has recorded the failure in `docs/refusal-log.md` | Claude (via workflow) |
| `agent:needs-user-decision` | `#D93F0B` | Pipeline cannot infer the next safe step without human input | When the issue is ambiguous, out of scope, or requires a governance call that Claude must not make unilaterally | Claude (via workflow) |
| `agent:child` | `#FBCA04` | Marks an issue as a child task decomposed from a parent issue | When a parent issue is broken down into smaller executable sub-issues | Workflow or Claude during planning phase |
| `agent:parent` | `#0052CC` | Marks an issue that has spawned one or more child tasks | When an issue has been decomposed and child issues have been created | Workflow or Claude during planning phase |

---

## State-transition rules

- `agent:plan` → `agent:run`: human approves decomposition and is ready to execute.
- `agent:run` → `agent:stuck`: workflow adds `agent:stuck` after repeated failure + refusal log entry; `agent:run` is **not** removed automatically and remains on the issue.
- `agent:run` → `agent:needs-user-decision`: not yet automated; SHIHCHENG manually removes `agent:run` and adds `agent:needs-user-decision` when Claude halts pending a human decision.
- `agent:parent` + `agent:child`: set together at decomposition time; neither implies execution state on its own.
- Only one execution-state label (`agent:plan`, `agent:run`, `agent:stuck`, `agent:needs-user-decision`) should be active on an issue at a time.

---

## Validation

```bash
test -f docs/pipeline-labels.md && grep -c "agent:" docs/pipeline-labels.md
```

Expected: exits 0, prints 6 or more.
