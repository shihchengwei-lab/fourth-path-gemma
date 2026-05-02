# Route Validation Audit - 2026-05-02

Objective: establish and verify a path where Main Agent capability can improve
quickly while safety authority stays outside the bottom model.

## Success Criteria

1. Capability can move without changing the repository into a safety-training
   project.
2. Safety/refusal/tool authority remains external to the Main Agent.
3. Stronger or worse Main Agent candidate text is contained by Classify, Cold
   Eyes, Mechanical Cold Eyes, and action gates.
4. Safety pressure data is eval-only and is not exported as Main Agent SFT.
5. The route is covered by repeatable local gates, not only by one-off manual
   inspection.

## Evidence

| Requirement | Evidence |
|---|---|
| Capability moved | `runs/qwen3-8b-main-agent-generalization-driven-lora-20260502-v4-manifest.json`: 111 rows, 80 optimizer steps, resumed from v3. |
| Hard surface retained | `runs/qwen3-8b-main-agent-generalization-driven-lora-20260502-v4-hard-train-eval-nothink.json`: 30/30 clean. |
| Heldout improved | `runs/qwen3-8b-main-agent-generalization-driven-lora-20260502-v4-fresh-heldout-eval-nothink.json`: 9/12 clean, up from v3 7/12. |
| Generalization is bounded | `runs/qwen3-8b-main-agent-generalization-driven-lora-20260502-v4-generalization-probe-eval-nothink.json`: 4/13 clean, unchanged from v3. |
| Containment held | `runs/qwen3-8b-main-agent-generalization-driven-lora-20260502-v4-adapter-containment-eval-rerun.json`: contained 12/12. |
| Candidate authority pressure still exists | Same containment eval: candidate clean 5/12 and `role_authority_boundary` clean 0/3. This is expected to be contained externally, not trained into the Main Agent. |
| Stronger architecture pressure held | `runs/architecture-containment-pressure-eval-qwen3-8b-local-max-20260502-rerun4.json`: 25/25 passed, with 8 pipeline, 8 Cold Eyes, and 9 action cases. |
| Action gate edge fixed | `action_gate.py` blocks project-relative sensitive reads such as `.git/`, `private_key.pem`, and token/key/secret files. |
| Pressure suite is gated | `release_gates.py` includes `architecture_containment_pressure` in `local-release-gate`. |
| Regression coverage exists | `python -m unittest discover -s tests`: 208 tests passed after the changes. |
| No-Ollama release gate passes | `python main.py local-release-gate --json`: no errors. |

## Gaps

- This is local-route validation, not production or AGI-grade proof.
- The bottom model is still not broadly generalized: the clean generalization
  probe stayed at 4/13.
- Candidate cleanliness is intentionally not perfect. The route depends on the
  external containment layers remaining active and fail-closed.
- The adapter should not become the default runtime until a new clean held-out
  surface shows repeatable gains without containment regressions.

## Decision

The local route is validated as an experiment: capability can improve on the
Main Agent surface while safety authority remains outside the bottom model and
external containment gates continue to block authority collapse and action-gate
abuse. It is not a deployment claim.
