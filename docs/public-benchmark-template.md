# Public Benchmark Template

This repo's internal Main Agent evals are useful for development, but they are
not enough evidence for public model comparisons. Use this template when the
question is whether `qwen3:8b` or a Fourth Path profile improves on common,
reproducible benchmarks.

## Benchmark Stack

Primary runner: EleutherAI `lm-evaluation-harness`.

Reasons:

- It is widely used for open-model comparison.
- It includes standard tasks such as `ifeval`, `gsm8k`, `mmlu`, and
  `humaneval_instruct`.
- It can call local OpenAI-compatible servers through `local-chat-completions`.
- Ollama exposes an OpenAI-compatible `/v1/chat/completions` endpoint, so raw
  local models and this repo's wrapper can use the same runner.

Secondary runner: LiveBench.

Use LiveBench after the first harness pass when contamination resistance matters
more than setup cost. LiveBench is designed around objective scoring and
frequently refreshed questions, but it is heavier and should not be the first
smoke test on this hardware.

## Targets

Run at least these targets:

- `raw`: direct `qwen3:8b` through Ollama's OpenAI-compatible endpoint.
- `main`: Fourth Path Main Agent profile only. This tests dynamic thinking,
  distilled hints, and local selection without Cold Eyes audit cost.
- `pipeline`: full separated reasoning and audit path. This is useful for
  end-to-end product behavior, not for isolating raw capability.

The `main` target is the fairest comparison for capability claims. The
`pipeline` target answers whether the full product path preserves benchmark
answers after audit.

## Task Tiers

Start small, then expand.

| Tier | Tasks | Purpose |
| --- | --- | --- |
| P0 smoke | `ifeval,gsm8k` with `--limit 50` | Prove the harness and wrapper work. |
| P1 aligned | `ifeval,gsm8k,humaneval_instruct` | Tests instruction following, math, and code. |
| P2 broad | `mmlu,arc_challenge,hellaswag` through a logprob-capable backend | General open-model comparison. |
| P3 fresh | LiveBench public release | Contamination-resistant objective scoring. |

This script uses `local-chat-completions`, so it is meant for generation tasks.
Multiple-choice or log-likelihood-heavy tasks need a backend that exposes
completion logprobs. Treat those as a separate backend test unless the server,
quantization, prompt format, few-shot setting, and task version match the
leaderboard method.

## Setup

Use a separate environment so benchmark dependencies do not affect this repo:

```powershell
python -m venv .venv-bench
.\.venv-bench\Scripts\python -m pip install -U pip
.\.venv-bench\Scripts\python -m pip install "lm-eval[api,ifeval]"
```

Some tasks may need extra dependencies. Install task-specific extras only when
the harness reports a missing package.

## Smoke Run

Raw `qwen3:8b` through Ollama:

```powershell
.\tools\run-public-bench.ps1 -Target raw -Tasks ifeval,gsm8k -Limit 50 -Python .\.venv-bench\Scripts\python
```

Main Agent effective capability:

```powershell
.\tools\run-public-bench.ps1 -Target main -Tasks ifeval,gsm8k -Limit 50 -Python .\.venv-bench\Scripts\python
```

Full end-to-end pipeline:

```powershell
.\tools\run-public-bench.ps1 -Target pipeline -Tasks ifeval,gsm8k -Limit 50 -Python .\.venv-bench\Scripts\python
```

All three targets:

```powershell
.\tools\run-public-bench.ps1 -Target all -Tasks ifeval,gsm8k -Limit 50 -Python .\.venv-bench\Scripts\python
```

Results are written under `runs\public-bench`.

The template was smoke-tested on 2026-05-01 with `ifeval --limit 1` for all
three targets: raw `qwen3:8b`, `qwen3-8b-s2t-lite` Main Agent, and the full
pipeline. All three calls completed through the same `lm-evaluation-harness`
runner. The `--limit 1` result is only a wiring check, not a capability score.

## Current Smoke Results

The 2026-05-02 GSM8K debugging pass found a local wrapper bug: the S2T-lite
local selector treated non-ASCII GSM8K prompts as concise-output requests and
truncated reasoning before the final answer. After disabling length capping for
math reasoning prompts and adding conditional math-state hints, the same
`qwen3-8b-s2t-lite` Main Agent target reached:

```text
tasks: gsm8k
limit: 50
target: main
strict-match exact_match: 1.00
flexible-extract exact_match: 1.00
output_path: runs\public-bench-loop\qwen3-8b-s2t-lite-main-20260502-012447
```

The same pass rechecked IFEval:

```text
tasks: ifeval
limit: 50
target: main
prompt_level_strict_acc: 0.7600
inst_level_strict_acc: 0.8421
output_path: runs\public-bench-loop\qwen3-8b-s2t-lite-main-20260502-012938
```

After the held-out prompt-shape pass, the same P0 public smoke was rerun:

```text
tasks: ifeval,gsm8k
limit: 50
target: main
GSM8K strict-match exact_match: 1.00
GSM8K flexible-extract exact_match: 1.00
IFEval prompt_level_strict_acc: 0.7600
IFEval inst_level_strict_acc: 0.8421
output_path: runs\public-bench-post-heldout\qwen3-8b-s2t-lite-main-20260502-022218
```

A later release-gate rerun after strategy-layer refactoring stayed in the same
smoke band but did not reproduce the exact GSM8K 1.00:

```text
tasks: ifeval,gsm8k
limit: 50
target: main
GSM8K strict-match exact_match: 0.98
GSM8K flexible-extract exact_match: 0.98
IFEval prompt_level_strict_acc: 0.7800
IFEval inst_level_strict_acc: 0.8553
output_path: runs\public-bench-release-gate\qwen3-8b-s2t-lite-main-20260502-032026
```

These are limited smoke runs, not full benchmark claims. The useful conclusion
is narrower: the previous GSM8K `flexible-extract=0.18` was mostly an
architecture/wrapper tax, not a proof that the local 8B model could not solve
grade-school arithmetic.

## Manual Wrapper

To inspect or debug the wrapper directly:

```powershell
python .\tools\public_bench_server.py --profile qwen3-8b-s2t-lite --mode main --port 8008
```

Then point an OpenAI-compatible client at:

```text
http://127.0.0.1:8008/v1/chat/completions
```

The wrapper also exposes:

```text
GET /health
GET /v1/models
```

## Reporting Template

Record each run with this shape:

```text
runner: lm-evaluation-harness
tasks: ifeval,gsm8k
limit: 50
target: raw | main | pipeline
model: qwen3:8b | qwen3-8b-s2t-lite-main | qwen3-8b-s2t-lite-pipeline
output_path: runs\public-bench\<run>
score_summary:
runtime:
notes:
```

Do not compare against public leaderboards as a hard claim unless the backend,
quantization, prompt format, few-shot setting, and task version match. Use
leaderboards as context; use same-machine A/B runs as evidence.

## Sources

- EleutherAI lm-evaluation-harness: https://github.com/EleutherAI/lm-evaluation-harness
- lm-eval interface docs: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md
- Ollama OpenAI compatibility: https://docs.ollama.com/api/openai-compatibility
- IFEval paper: https://arxiv.org/abs/2311.07911
- GSM8K repository: https://github.com/openai/grade-school-math
- HumanEval task docs: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/humaneval/README.md
- LiveBench: https://github.com/livebench/livebench
