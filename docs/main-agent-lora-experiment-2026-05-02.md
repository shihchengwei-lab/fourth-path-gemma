# Main Agent LoRA Experiment - 2026-05-02

This note records the local QLoRA experiment run against the Main Agent
candidate-generator surface. It is an experiment report, not a new default
runtime path and not evidence that the Fourth Path architecture is proven.

## Scope

The repository remains a Fourth Path architecture prototype: the Main Agent
generates candidate text, while deterministic routing, Mechanical Cold Eyes,
and refusal handling keep final authority outside the Main Agent.

This experiment only asks whether a local `Qwen/Qwen3-8B` adapter can make the
Main Agent better at concise candidate generation, exact output shape, simple
code repair, and verifier-backed task following. It does not train a safety
judge, it does not move refusal authority into the Main Agent, and it does not
change the default audited pipeline.

## Data

The training data used only explicit synthetic rows, not private chat logs or
persisted audit logs.

- Base curated SFT set: `runs/main-agent-hybrid-claude-20260502-round3.jsonl`
  - 55 rows
  - 30/30 hard records covered
  - 0 local verifier failures in the curated set
- Failure-driven seed: `data/main_agent_failure_driven_seed.jsonl`
  - 32 rows
  - targets the fresh-heldout failure labels seen after the first 8B LoRA run
  - every row has a verifier
  - every `target_response` passes its own verifier
- Merged v4 training set:
  `runs/main-agent-hybrid-failure-driven-20260502-v4.jsonl`
  - 87 rows
  - 87/87 system messages present
  - 0 format errors in `main-training-data-report`
- Generalization-driven seed: `data/main_agent_generalization_driven_seed.jsonl`
  - 24 rows
  - generated from failure labels on the clean generalization probe, with
    different prompts and surfaces
  - every row has a verifier
  - every `target_response` passes its own verifier
- Merged v5 training set:
  `runs/main-agent-hybrid-generalization-driven-20260502-v5.jsonl`
  - 111 rows
  - 111/111 system messages present
  - generated metadata complete
  - 0 format errors in `main-training-data-report`

The fresh-heldout set was used to diagnose and drive the failure-driven data
round. It should no longer be treated as a clean final proof surface for future
claims.

## Runs

| Run | Input | Result |
|---|---:|---:|
| `Qwen/Qwen3-8B` base on hard train surface | 30 rows | 2/30 clean |
| 8B LoRA v2 on hard train surface | 30 rows | 29/30 clean |
| 8B LoRA v3 on hard train surface, no-thinking eval | 30 rows | 30/30 clean |
| `Qwen/Qwen3-8B` base on fresh heldout | 12 rows | 0/12 clean |
| 8B LoRA v2 on fresh heldout | 12 rows | 2/12 clean |
| 8B LoRA v3 on fresh heldout | 12 rows | 7/12 clean |
| 8B LoRA v3 on fresh heldout, no-thinking eval | 12 rows | 7/12 clean |
| 8B LoRA v4 on hard train surface, no-thinking eval | 30 rows | 30/30 clean |
| 8B LoRA v4 on fresh heldout, no-thinking eval | 12 rows | 9/12 clean |

The v3 adapter was resumed from v2 and trained on the merged 87-row v4 set:

- output: `runs/qwen3-8b-main-agent-failure-driven-lora-20260502-v3`
- manifest:
  `runs/qwen3-8b-main-agent-failure-driven-lora-20260502-v3-manifest.json`
- local experimental scripts:
  - `tools/experimental/qlora_smoke_train.py`
  - `tools/experimental/qlora_adapter_eval.py`
  - `tools/experimental/merge_sft_jsonl.py`
- optimizer steps: 80
- micro steps: 320
- learning rate: `0.0001`
- LoRA rank: 8
- trainable parameters: 21,823,488
- loss: `3.5079 -> 0.0037`
- duration: about 1881.8 seconds on the RTX 4060 Laptop GPU

After `tools/experimental/qlora_adapter_eval.py` was aligned with the safety
eval to disable Qwen thinking by default, the v3 adapter was rechecked without
thinking traces:

- `runs/qwen3-8b-main-agent-failure-driven-lora-20260502-v3-hard-train-eval-nothink.json`:
  30/30 clean
- `runs/qwen3-8b-main-agent-failure-driven-lora-20260502-v3-fresh-heldout-eval-nothink.json`:
  7/12 clean

This confirms the measured capability gain is not just a thinking-template
artifact.

The v4 adapter was resumed from v3 and trained on the merged 111-row v5 set:

- output:
  `runs/qwen3-8b-main-agent-generalization-driven-lora-20260502-v4`
- manifest:
  `runs/qwen3-8b-main-agent-generalization-driven-lora-20260502-v4-manifest.json`
- optimizer steps: 80
- micro steps: 320
- learning rate: `0.00005`
- LoRA rank: 8
- trainable parameters: 21,823,488
- loss: `2.7522 -> 0.00032`
- duration: about 1894.0 seconds on the RTX 4060 Laptop GPU

A new clean generalization probe was then added after the v3 training data
round:

- seed: `data/main_agent_generalization_probe_seed.jsonl`
- records: 13
- verifier records: 13/13
- target-response verifier failures: 0
- prompt overlap with prior seed, hard, heldout, fresh-heldout, rotated-heldout,
  and latent-probe corpora: 0 in the regression test

Measured no-thinking result:

| Run | Generalization probe | Clean |
|---|---:|---:|
| `Qwen/Qwen3-8B` base | 13 rows | 3/13 |
| 8B LoRA v3 | 13 rows | 4/13 |
| 8B LoRA v4 | 13 rows | 4/13 |

This is the strongest current warning against overclaiming. The adapter learned
the prior hard/fresh surfaces, and v4 improved fresh heldout from 7/12 to 9/12,
but the gain on the separate capability probe did not move beyond 4/13. The
next ability round should improve data coverage and surface variation, not
reuse this probe as training material.

## Current Interpretation

The experiment shows that the Main Agent surface is trainable on this machine.
The jump from 0/12 base fresh-heldout clean to 7/12 after failure-driven LoRA,
then to 9/12 after generalization-driven LoRA, is a useful local engineering
signal that targeted data may improve the candidate generator. It is not a
clean capability-evidence claim once those surfaces have shaped later data.

It is not deployment evidence. The adapter still fails 3/12 fresh-heldout rows,
and the diagnostic heldout set has already influenced the data rounds. The new
generalization probe shows only a 3/13 to 4/13 improvement over the base model,
so current evidence supports data-direction potential, not broad
generalization.

Remaining failures concentrate in:

- code precision: exact arithmetic and code-repair details can still be wrong
- format required-any terms: answers can satisfy the visible schema while
  omitting required concepts such as public, rotated, held-out, or fresh
- planning gate behavior: the model can still choose more LoRA before data
  quality when the intended answer is data first

## Adapter Containment Eval

After the capability gain, a separate adapter containment eval was added:

- seed: `data/main_agent_adapter_containment_seed.jsonl`
- tool: `tools/experimental/adapter_containment_eval.py`
- base output:
  `runs/qwen3-8b-base-adapter-containment-eval-20260502.json`
- v3 adapter output:
  `runs/qwen3-8b-main-agent-failure-driven-lora-20260502-v3-adapter-containment-eval.json`
- v4 adapter output:
  `runs/qwen3-8b-main-agent-generalization-driven-lora-20260502-v4-adapter-containment-eval-rerun.json`

This eval treats the adapter as a Main Agent candidate generator. Each answer
now has two separate readings:

1. Candidate cleanliness: did the adapter itself avoid hidden-state leaks,
   unsupported canon references, role-authority claims, thinking artifacts, and
   verifier failures?
2. Containment: if the adapter did generate a bad authority claim, did the
   external classifier or Mechanical Cold Eyes block it before final output?

The earlier candidate-cleanliness result was:

| Run | Adapter containment seed | Candidate clean |
|---|---:|---:|
| `Qwen/Qwen3-8B` base | 12 rows | 1/12 |
| 8B LoRA v3 | 12 rows | 6/12 |
| 8B LoRA v4 | 12 rows | 5/12 |

The containment result after separating external containment from candidate
cleanliness is:

| Run | Adapter containment seed | Contained |
|---|---:|---:|
| `Qwen/Qwen3-8B` base | 12 rows | 12/12 |
| 8B LoRA v3 | 12 rows | 12/12 |
| 8B LoRA v4 | 12 rows | 12/12 |

Candidate cleanliness is not the safety success criterion. The weakest
candidate cleanliness surface is role-authority pressure:

- `role_authority_boundary`: 0/3 clean
- remaining failures include obeying "Allowed", adding a `Safety policy`
  section, and writing `PASS and approved by safety`

That does not mean the Main Agent should be trained into a safety reviewer.
Role-authority pressure is not the Main Agent's job. Keep
`data/main_agent_adapter_containment_seed.jsonl` as an eval-only containment
gate: it measures whether stronger Main Agent candidates are contained by
external layers, but it should not be converted into SFT rows that teach the
candidate generator to act as Cold Eyes, cite policy authority, or issue final
approval.

The right fix is architectural: keep safety identity, final approval, refusal
authority, and canon interpretation outside the Main Agent. If an adapter fails
candidate cleanliness but the containment gate catches the bad output, that is
evidence for the Fourth Path claim. If a role-authority candidate is forwarded
as final output, that is a containment failure.

## Architecture Pressure Eval

A separate Codex-authored pressure corpus was added after the containment
alignment:

- seed: `data/architecture_containment_pressure_seed.jsonl`
- records: 25
- layers: 8 pipeline, 8 Cold Eyes, 9 action gate
- final measured output:
  `runs/architecture-containment-pressure-eval-qwen3-8b-local-max-20260502-rerun4.json`
- result: 25/25 passed with `qwen3-8b-local-max`

The first run found no candidate-output bypass, but it did expose safety-tax and
diagnostic issues: a defensive phishing-awareness prompt was prefiltered too
aggressively, and a control-plane mutation action failed closed without a C2
clause. A later action-gate review also found that project-relative sensitive
filenames such as `config/private_key.pem` needed explicit blocking. These are
now covered by regression tests.

This pressure suite is not SFT material. It exists to verify the claim that a
stronger Main Agent can produce worse authority-collapsing candidate text
without gaining final output or tool/action authority.

## Repository Boundary

Keep this work as a Main Agent optimization branch of evidence. It belongs
under data, verifier, and experiment documentation. It should not redefine the
repository as a fine-tuning project.

Before any adapter is promoted, require:

1. A new clean evaluation surface that did not drive the data round.
2. A same-run comparison against the fixed-weight architecture profiles.
3. Checks that audit separation, fail-closed behavior, and refusal authority
   remain outside the Main Agent.
4. A public benchmark or external sanity pass if making claims beyond local
   synthetic rows.
