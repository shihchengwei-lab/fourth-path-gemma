# Pipeline Labels (`agent:*`)

This document defines all six `agent:*` GitHub labels used by the fourth-path-gemma agent pipeline.
These labels coordinate state transitions between the workflow, Claude, Codex, and SHIHCHENG.

---

## Label Reference

| Label | Color | Purpose | When to apply | Who applies it |
|---|---|---|---|---|
| `agent:plan` | `#ededed` | Signals that an issue is in the planning phase — scope and decomposition in progress | When a new issue needs decomposition before execution can begin | Workflow (`claude.yml`) or human at triage |
| `agent:run` | `#0E8A16` | Triggers the Claude workflow to process the issue | When scope is clear and Claude should execute the task | Workflow auto-adds it to new unlabelled issues; humans may add it manually for explicit reruns |
| `agent:stuck` | `#B60205` | Pipeline has failed repeatedly; a refusal log entry has been written; awaiting human decision or future recovery handling | When Claude cannot make progress after multiple attempts and has recorded the failure in `docs/refusal-log.md` | Workflow currently adds this in the PR-review repeated-review path |
| `agent:needs-user-decision` | `#D93F0B` | Pipeline cannot infer the next safe step without human input | Ambiguous, out-of-scope, or governance-blocked tasks | Recovery workflow can add it; humans may also add it manually |
| `agent:child` | `#FBCA04` | Marks an issue as a child task decomposed from a parent issue | When a parent issue is broken down into smaller executable sub-issues | Planning workflow / Claude during planning phase |
| `agent:parent` | `#0052CC` | Marks an issue that has spawned one or more child tasks | When an issue has been decomposed and child issues have been created | Planning workflow / Claude during planning phase |

---

## Current automation status

| Behavior | Current status |
|---|---|
| New unlabelled issue is queued for execution | Implemented in `.github/workflows/claude.yml` by adding `agent:run` automatically |
| `agent:plan` triggers planning-only mode | Implemented in `.github/workflows/claude.yml` |
| `agent:run` triggers implementation mode | Implemented in `.github/workflows/claude.yml` |
| First child issue receives `agent:run` during planning | Implemented by the planning prompt, not by a separate deterministic workflow step |
| Claude requests another Codex review after pushing a review fix | Implemented in the PR-review path by commenting `@codex review` after a pushed fix commit |
| Codex environment setup blocker is surfaced | Implemented in the issue-comment path by adding `agent:needs-user-decision` when Codex says the repo has no environment |
| `agent:stuck` is added after repeated review detection | Implemented in the PR-review path |
| `agent:run` is removed automatically | Implemented by the process prompt after the PR is open |
| `agent:needs-user-decision` is added automatically | Implemented in the repeated-review recovery path when a real human decision is required |

---

## State-transition rules

- New unlabelled issue → `agent:run`: bootstrap flow queues the issue automatically so SHIHCHENG does not need to add the execution label by hand.
- `agent:plan` → child issue with `agent:run`: current planning flow creates child issues and labels only the first executable child with `agent:run`.
- `agent:run` → ready PR: Claude implements the issue, opens a ready-for-review PR linked with a closing keyword, then removes `agent:run` from the source issue.
- Codex review → Claude fix → Codex review: Trigger B applies the requested fix, pushes it, then comments `@codex review` to request the next review round.
- Codex environment missing → `agent:needs-user-decision`: Trigger C labels the PR and leaves one setup comment if Codex says the repo needs an environment before review can run.
- Repeated identical Codex review → `agent:stuck`: automated in the PR-review repeated-review path to prevent an infinite loop.
- Ambiguous task → `agent:needs-user-decision`: recovery flow uses this only when it cannot infer a safe next step and must ask one narrow question.
- `agent:parent` + `agent:child`: set during decomposition; neither implies execution state on its own.
- Operators should avoid keeping more than one execution-state label (`agent:plan`, `agent:run`, `agent:stuck`, `agent:needs-user-decision`) active on the same issue, but the current workflow does not enforce this automatically.

---

## Validation

```bash
test -f docs/pipeline-labels.md && grep -c "agent:" docs/pipeline-labels.md
```

Expected: exits 0, prints 6 or more.
