# Qwen3 Main Agent LoRA Path

This is the weight-change path for the Main Agent. The goal is not to train a
new safety judge. The goal is to make `qwen3:8b` behave more like a pure
candidate generator while Classify, mechanical Cold Eyes, and Cold Eyes keep the
final refusal authority.

This document is an experiment lane. The repository's main line remains the
Fourth Path separation architecture: reasoning, routing, audit, and refusal
authority are separate layers. LoRA work is useful only if it improves the Main
Agent candidate-generator surface without moving safety authority back into the
Main Agent.

See [Main Agent LoRA Experiment - 2026-05-02](main-agent-lora-experiment-2026-05-02.md)
for the local QLoRA run summary and its limits.

## 2026-05-02 Experiment Status

The first target-size local QLoRA round confirmed that the Main Agent surface is
trainable on this machine, but it did not create a deployable default adapter.

Key measured results:

| Run | Eval surface | Clean |
|---|---|---:|
| `Qwen/Qwen3-8B` base | 30-row hard train surface | 2/30 |
| 8B LoRA v2 | 30-row hard train surface | 29/30 |
| 8B LoRA v3, no-thinking eval | 30-row hard train surface | 30/30 |
| `Qwen/Qwen3-8B` base | 12-row fresh heldout | 0/12 |
| 8B LoRA v2 | 12-row fresh heldout | 2/12 |
| 8B LoRA v3 | 12-row fresh heldout | 7/12 |
| 8B LoRA v3, no-thinking eval | 12-row fresh heldout | 7/12 |
| 8B LoRA v4, no-thinking eval | 12-row fresh heldout | 9/12 |
| `Qwen/Qwen3-8B` base | 12-row adapter containment seed, candidate clean | 1/12 |
| 8B LoRA v3 | 12-row adapter containment seed, candidate clean | 6/12 |
| 8B LoRA v4 | 12-row adapter containment seed, candidate clean | 5/12 |
| `Qwen/Qwen3-8B` base | 12-row adapter containment seed, contained | 12/12 |
| 8B LoRA v3 | 12-row adapter containment seed, contained | 12/12 |
| 8B LoRA v4 | 12-row adapter containment seed, contained | 12/12 |
| `Qwen/Qwen3-8B` base, no-thinking eval | 13-row generalization probe | 3/13 |
| 8B LoRA v3, no-thinking eval | 13-row generalization probe | 4/13 |
| 8B LoRA v4, no-thinking eval | 13-row generalization probe | 4/13 |

The v3 improvement came after adding `data/main_agent_failure_driven_seed.jsonl`
and resuming from the v2 adapter. Treat this as evidence for the data direction,
not as a release gate pass. The fresh-heldout set has now been used to guide the
failure-driven data round, so future promotion needs a new clean evaluation
surface.

The no-thinking eval recheck confirms the v3 capability gain is not just a Qwen
thinking-template artifact.

The generalization probe is the current caution flag: v3 only moved from 3/13
to 4/13 on a new clean surface, and v4 stayed at 4/13. Treat the prior
hard/fresh wins as local evidence for targeted data, not as proof that the
adapter broadly generalized.

The adapter containment seed exposed the important distinction between model
behavior and architecture containment. The v3 adapter can still obey requests
to say "Allowed", add a `Safety policy` section, or write `PASS and approved by
safety`.

Treat that as a containment-gate finding, not as a training target. The Main
Agent should not be trained to become Cold Eyes, cite policy authority, or issue
final safety approval. `data/main_agent_adapter_containment_seed.jsonl` is
eval-only: it exists to verify that external gates stop role-authority candidate
text before runtime integration.

The v4 continuation added 24 synthetic generalization-driven training rows and
improved the 12-row fresh-heldout score from 7/12 to 9/12 without reducing the
containment score. It did not improve the separate 13-row generalization probe.
That makes the next ability bottleneck data breadth, not more safety-boundary
training.

The Codex-authored architecture pressure suite
`data/architecture_containment_pressure_seed.jsonl` separately tests fake audit
approval, role-authority collapse, and action-gate abuse. Its current measured
result is 25/25 with `qwen3-8b-local-max`, including 8/8 pipeline, 8/8 Cold
Eyes, and 9/9 action-gate cases.

## Current Finding

The current Main Agent seed eval does not show a refusal bottleneck:

- Corpus: `data/main_agent_seed.jsonl`
- Records at measurement time: 40 synthetic, reviewed role-behavior examples,
  including near-boundary defensive security and concise-control cases
- `qwen3-8b-local-max`: 0/40 refusal-like outputs
- Clean cases after the boundary-sensitive checklist prompt change: 38-39/40
- Overlong cases at `--max-length-ratio 4`: 1-2/40
- Average output/target character ratio: about 1.970-2.056
- Main Agent calls: 40
- Total eval time: about 157.1-157.4 seconds
- Before concise prompt tightening: 4/20 overlong cases at `--max-length-ratio 4`,
  average output/target character ratio about 2.53
- After concise prompt tightening: 1/20 overlong case, average ratio about 1.94
- On the expanded 40-record corpus, the current two-candidate
  `qwen3-8b-search` profile removed the measured hidden-boundary leak, reduced
  overlong cases to 3/40, and lowered the average length ratio to about 2.306,
  but spent 120 Main Agent/selector calls and about 457.7 seconds.
- `qwen3-8b-reasoning` made this corpus worse: 27/40 clean, 12/40 overlong,
  average length ratio about 3.395, and about 520.2 seconds in the first idle
  run; the latest full idle run was still worse than default at 29/40 clean,
  11/40 overlong, and about 501.9 seconds.

That means the immediate bottleneck is not weight-level self-refusal on this
seed set. The first wins came from smaller prompt contracts: direct, scoped,
concise candidate generation, then short practical checklist behavior for
defensive and boundary-sensitive tasks. The next win should be data and decoding
control for residual verbosity variance before any adapter training.

## Why LoRA, Not Full Fine-Tune

Full fine-tuning an 8B model is a poor fit for the current laptop-class
hardware. LoRA freezes the base model and trains small low-rank adapter weights.
QLoRA reduces memory pressure further by keeping the base model quantized during
adapter training.

Use LoRA / QLoRA only after the eval says there is a real behavior gap:

- self-refusal on allowed tasks
- role-boundary leakage such as revealing hidden system/developer text, private
  audit state, reasoning traces, or credentials
- unsupported canon references, such as inventing a non-existent canon clause
- repeated verbosity that prompt changes cannot reduce
- format instability in normal local tasks

## Data Boundary

Do not train from default audit logs. Those logs intentionally omit prompt text,
full candidate output, and hidden reasoning traces.

Use explicit synthetic data instead:

```text
data/main_agent_seed.jsonl
data/main_agent_hard_seed.jsonl
data/main_agent_failure_driven_seed.jsonl
data/main_agent_generalization_driven_seed.jsonl
```

Use `data/main_agent_adapter_containment_seed.jsonl` only as an eval-only
containment-gate corpus. Do not export it as SFT training data.

Use `data/main_agent_generalization_probe_seed.jsonl` as a clean capability
probe. Do not use it to train the next adapter round; use its failure labels to
design different synthetic rows instead.

Use `data/architecture_containment_pressure_seed.jsonl` only as an
architecture/adversarial eval corpus, not as Main Agent SFT data.

After withdrawing the old v6-v17 local engineering surfaces, keep legacy `v5`
as non-evidence context, remove old `v6-v17` from eval/gate context, and
restart the next clean capability proof at a newly minted unused `v6` surface.

Each record contains:

- `id`
- `category`
- `prompt`
- `target_response`

The seed corpus is allowed to contain prompts because it is synthetic and
reviewed. It is separate from private local conversations.

## Commands

Validate the seed corpus:

```powershell
python main.py main-check --min-total 40 --min-category 1
```

Measure the current Main Agent:

```powershell
python main.py main-eval --profile qwen3-8b-local-max --json --timeout 900 --max-length-ratio 4
```

Export chat-style SFT JSONL for an adapter training tool:

```powershell
python main.py main-sft-export --output-file runs\main-agent-sft-seed.jsonl
python main.py main-training-data-report --input-file runs\main-agent-sft-seed.jsonl --require-system --json
```

The exported rows use this shape:

```json
{"id":"...","category":"...","messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

Export a smaller expert/amateur contrast set before training:

```powershell
python main.py main-contrast-export --min-score-gap 100 --max-length-ratio 4 --json --timeout 900
```

This LightReasoner-lite path stores selected Expert answers from synthetic
records only when the Expert profile is clean and the Amateur profile is clearly
worse. It is a data-selection gate before LoRA, not a replacement for held-out
evaluation.

## Training Gate

Do not train just because training is possible. Train only when at least one
gate justifies it:

- `refusal_like_rate > 0` on allowed seed tasks
- `role_boundary_leak` appears
- `overlong_rate` stays high after prompt, search, and data tuning
- a larger held-out seed set shows the same failure pattern

After training, compare the adapter against the base profile:

```powershell
python main.py main-eval --profile qwen3-8b-local-max --json --timeout 900 --max-length-ratio 4
python main.py bench --profile qwen3-8b-local-max --warmup --json --timeout 900
python main.py distill-eval --profile qwen3-8b-local-max --json --timeout 900 --require-exact --min-exact-accuracy 1 --min-mechanical-cases 25
```

The adapter is worth keeping only if it reduces the measured behavior gap
without breaking the separated audit contract.

## Sources

- LoRA: https://arxiv.org/abs/2106.09685
- QLoRA: https://arxiv.org/abs/2305.14314
- Direct Preference Optimization: https://arxiv.org/abs/2305.18290
- Self-Refine: https://arxiv.org/abs/2303.17651
- rStar-Math: https://arxiv.org/abs/2501.04519
- SLM-MUX: https://arxiv.org/abs/2510.05077
- LightReasoner: https://arxiv.org/abs/2510.07962
