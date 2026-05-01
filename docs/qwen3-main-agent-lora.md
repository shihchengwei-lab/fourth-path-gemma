# Qwen3 Main Agent LoRA Path

This is the weight-change path for the Main Agent. The goal is not to train a
new safety judge. The goal is to make `qwen3:8b` behave more like a pure
candidate generator while Classify, mechanical Cold Eyes, and Cold Eyes keep the
final refusal authority.

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
```

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
