# Compute-Optimal Test-Time Adaptation

This note adapts the paper "Scaling LLM Test-Time Compute Optimally can be More
Effective than Scaling Model Parameters" to this repo.

The goal is not benchmark tuning. The useful idea is that extra inference-time
compute should be allocated by prompt shape and difficulty. Some tasks benefit
from revision, some benefit from parallel candidates and selection, and some
should not spend extra calls at all.

## Paper Takeaway

The paper separates two ways to spend extra test-time compute:

- search against a verifier, such as best-of-N, beam search, or process-reward
  search
- modifying the proposal distribution, such as sequential revision of previous
  attempts

Its key result is that the best compute allocation depends on problem
difficulty. Easier tasks often benefit from local sequential refinement, medium
or harder tasks can need more exploration, and the hardest tasks may not improve
much from extra test-time compute alone. The paper reports that adaptive
compute allocation can be several times more efficient than a uniform
best-of-N baseline.

## Local Translation

This repo cannot reproduce the paper's full setup. We do not have a trained
process reward model, 2048 samples per prompt, or PaLM-specific revision
finetuning. The practical local version is smaller:

- estimate prompt shape cheaply
- avoid extra calls for strict output-shape tasks
- use sequential refinement for single-track hard reasoning
- use parallel candidates for exploratory architecture/planning tasks
- use mixed parallel plus refinement only for hard exploratory tasks
- keep final safety authority in Cold Eyes, not in the compute selector

The important constraint is that this layer is a compute allocator, not an
approval layer. It may spend more or fewer Main Agent calls. It cannot decide
whether output is harmless.

## Implemented Profile

The experimental profile is:

```powershell
python main.py run --profile qwen3-8b-compute-optimal-lite --prompt "Compare two local inference architectures." --json --timeout 900
```

It uses the same `qwen3:8b` main and audit model as `qwen3-8b-s2t-lite`, keeps
local selection enabled, and adds prompt-shape adaptive compute.

Current routing:

| Prompt shape | Strategy | Main Agent compute |
| --- | --- | --- |
| strict output shape, JSON, exact count, casing, word/paragraph constraints | `strict_output_shape` | one candidate, no refinement |
| exploratory planning/comparison | `parallel_explore` | two candidates plus quality selector |
| hard exploratory task | `mixed_hard_explore` | two candidates plus one refinement |
| arithmetic or single-track hard reasoning | `sequential_refine` | one candidate plus one refinement |
| ordinary prompt | `base` | profile defaults |

This is intentionally conservative. Public benchmark runs showed that careless
extra processing can damage strict instruction following. The adaptive profile
therefore refuses to spend extra calls on tasks where the user mostly asked for
exact shape preservation.

## What To Measure

Compare this profile against `qwen3-8b-s2t-lite`, not against a leaderboard:

```powershell
python main.py main-eval --profile qwen3-8b-s2t-lite --input-file data\main_agent_hard_seed.jsonl --json --timeout 900 --max-length-ratio 4
python main.py main-eval --profile qwen3-8b-compute-optimal-lite --input-file data\main_agent_hard_seed.jsonl --json --timeout 900 --max-length-ratio 4
python main.py main-eval-ablation --input-file data\main_agent_rotated_heldout_seed.jsonl --json --timeout 900 --max-length-ratio 4
python main.py bench --profile qwen3-8b-compute-optimal-lite --warmup --json --timeout 900
```

Track:

- clean cases per Main Agent call
- total Main Agent calls
- average eval tokens per clean case
- strict output-shape regressions
- role-boundary leaks
- whether Cold Eyes sees more retries

If the adaptive profile spends more calls but does not improve held-out tasks,
it should stay experimental. The paper's lesson is adaptive allocation, not
unbounded compute.

## Latest Rotated Held-Out Ablation

Command:

```powershell
python main.py main-eval-ablation --input-file data\main_agent_rotated_heldout_seed.jsonl --output-file runs\main-eval-ablation-rotated-20260502.json --json --timeout 900 --max-length-ratio 4
```

Result on 2026-05-02:

| Profile | Clean | Main calls | Clean/main-call | Issue rate |
| --- | ---: | ---: | ---: | ---: |
| `qwen3-8b-local-max` | 2/8 | 8 | 0.250 | 0.750 |
| `qwen3-8b-s2t-lite` | 2/8 | 8 | 0.250 | 0.750 |
| `qwen3-8b-compute-optimal-lite` | 2/8 | 12 | 0.167 | 0.750 |

The adaptive profile did not improve this fresh gate and spent more calls. Keep
it experimental. The failure labels were concentrated in required-any,
required-pattern, and one restricted Python test failure, so the next useful
work is better data/verifier targeting rather than more runtime compute.

## Source

- Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling
  Model Parameters: https://arxiv.org/abs/2408.03314
