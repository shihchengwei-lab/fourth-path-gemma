# Research Next Steps, 2026-05-02

This note records a bounded paper pass for the next engineering round. It does
not replace the existing distillation-first roadmap. It narrows the next work to
changes that can improve local evidence before touching backend or KV-cache
surgery.

## Decision

Continue, but shift the next round from broad refactor to capability evidence.

The priority remains:

1. distillation data quality
2. distillation format
3. verifier and tool-use
4. inference-time compute
5. KV cache changes

The paper pass strengthens that order. The most repeated signal is that extra
compute, search, or training helps only when the data and verifier loop are
clean enough to distinguish better attempts from worse attempts.

## What Changed From This Pass

### 1. Data Quality Is The Main Lever

LIMO and s1 both support a small, curated-data direction rather than a large
low-quality dump. LIMO frames high-value examples as cognitive templates. s1
uses a 1,000-example set selected for difficulty, diversity, and quality before
adding test-time budget control.

Local consequence:

- grow hard and rotated held-out rows before generating more training rows;
- score rows by difficulty, diversity, and verifier clarity;
- keep rejected rows for taxonomy, not training;
- require per-row source, split, verifier labels, and known failure surface;
- stop treating "more accepted JSONL" as improvement until held-out eval moves.

Next implementation:

1. Add a `main-data-quality-report` mode that emits row metadata only:
   category distribution, split, prompt hash duplication, verifier-label
   coverage, difficulty bucket, and failure-surface coverage.
2. Add fresh rotated held-out records before any LoRA or adapter claim.
3. Add rejection taxonomy counts to the R1-lite export summary.

### 2. Distillation Format Should Encode Traceability, Not Hidden Reasoning

SCOTT and related distillation work show that rationale supervision can help
only if the rationale is aligned with the answer. For this repo, the safer
translation is not to store private chains of thought. The useful artifact is a
chat row plus short, auditable metadata showing why the row is teachable.

Local consequence:

- keep `messages` as the training row body;
- keep long/private reasoning out of Main Agent training targets;
- add short metadata fields outside `messages`, such as `source`,
  `split`, `verifier_labels`, `difficulty`, and `template_tags`;
- keep pairwise better/worse data separate from SFT rows.

Next implementation:

1. Add a stricter schema validator for generated SFT and pairwise rows.
2. Require `source`, `split`, and non-empty verifier metadata on generated
   rows.
3. Add a format report that fails when training rows mix held-out examples into
   train artifacts.

### 3. Verifier-First Search Is Higher ROI Than Raw More Tokens

Training Verifiers, Process Supervision, rStar-Math, ReAct, and Toolformer all
point to the same principle: extra attempts or tools need a checker. Without a
checker, search only generates more plausible text.

Local consequence:

- expand deterministic verifiers before expanding candidate search;
- use best-of-N only for tasks with exact, regex, unit-test, schema, or
  mechanical issue checks;
- make tool calls auditable action candidates before execution;
- record verifier issue labels without printing private prompts or outputs.

Next implementation:

1. Add exact-answer and regex verifiers for synthetic math, code, and format
   rows.
2. Add code-repair rows with tiny local unit tests as verifier-backed cases.
3. Extend `verifier-tool-gate` to report verifier type coverage, not only pass
   balance.

### 4. Test-Time Compute Should Be A Measured Policy

The compute-optimal test-time paper, s1, and rStar-Math support spending
inference compute adaptively. They do not justify unconditional long thinking.
GSM-Symbolic and ConStat also warn that benchmark-looking gains can be brittle
or contaminated unless the held-out surface is rotated and variant-based.

Local consequence:

- keep `qwen3-8b-compute-optimal-lite` experimental;
- add variant tests that change numbers, clauses, and wording;
- measure clean cases per Main Agent call, not only clean cases;
- do not use budget forcing as a default until it beats one-shot on fresh
  held-out rows.

Next implementation:

1. Add a rotated-variant generator for synthetic numeric and planning rows.
2. Add an eval summary field for `clean_cases_per_main_call`.
3. Add an ablation command that compares base, local selection, and adaptive
   compute on the same fresh held-out set.

### 5. KV Cache Is Real, But Still Not Next

PagedAttention and DiffKV confirm that KV cache memory is a real systems
bottleneck, especially for long context and batched serving. That does not make
KV cache the next repo change. This project still lacks the data/verifier gates
needed to prove that a backend change is solving the limiting problem.

Local consequence:

- keep `llama-cpp-turboquant` reference-only;
- do not push, PR, or burden that repo;
- revisit backend/KV only after data, format, verifier, and compute gates show
  a measured runtime bottleneck.

Next implementation:

1. No KV implementation this round.
2. Keep `next-token-headroom`, `kv-cache-estimate`, and
   `llama-cpp-turboquant-backend.md` as deferred evidence.

## Immediate Backlog

Do these before another architecture refactor round:

1. Implement `main-data-quality-report`.
2. Add fresh rotated held-out data.
3. Add generated-row schema enforcement for `source`, `split`, and verifier
   metadata.
4. Add verifier coverage reporting.
5. Add exact/regex/unit-test backed rows for math, code repair, and strict
   format cases.
6. Add `clean_cases_per_main_call` to eval summaries.
7. Run an ablation: base profile vs local selector vs adaptive compute on the
   fresh held-out set.

Local implementation hooks now exist for this backlog:

- `main-data-quality-report`
- `data/main_agent_rotated_heldout_seed.jsonl`
- generated-row `source`, `split`, and `verifier_labels` metadata checks
- verifier coverage totals, including restricted `python_tests`
- `clean_cases_per_main_call`
- `main-eval-ablation`
- `main-eval-failure-report`

The newest no-Ollama bridge is `main-eval-failure-report`. It reads a saved
`main-eval` or `main-eval-ablation` JSON file and turns it into issue counts,
category/issue failure targets, profile efficiency ranking, and local-selection
reason counts without printing prompt text, target text, or model output. Use it
before adding more hard rows so the next data batch follows measured failures
rather than guesses.

The next targeted hard-data batch follows that report directly: code-repair rows
with regex plus restricted Python tests, strict-format rows with one-line or JSON
shape checks, and planning rows that require explicit data/verifier/evaluation
terms without copying held-out prompts.

The expanded hard corpus is now a tuned regression surface after prompt-side
distillation hints lifted `qwen3-8b-s2t-lite` to 30/30 clean on
`runs\main-eval-qwen3-8b-s2t-lite-hard-expanded-post-hints-v3-20260502.json`.
Use a fresh rotated or public surface for broader capability claims.

## Do Not Do Yet

- Do not start KV-cache integration.
- Do not train LoRA just because data can be exported.
- Do not let held-out rows become training rows.
- Do not use broad web data or private chat logs as distillation data.
- Do not treat larger candidate count as improvement without verifier-backed
  held-out evidence.

## Sources Reviewed

Distillation and data curation:

- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- LIMO: Less is More for Reasoning: https://arxiv.org/abs/2502.03387
- s1: Simple test-time scaling: https://arxiv.org/abs/2501.19393
- SCOTT: Self-Consistent Chain-of-Thought Distillation:
  https://arxiv.org/abs/2305.01879
- Democratizing Reasoning Ability: Tailored Learning from Large Language Model:
  https://arxiv.org/abs/2310.13332

Verifier, tool-use, and process supervision:

- Training Verifiers to Solve Math Word Problems:
  https://arxiv.org/abs/2110.14168
- Let's Verify Step by Step: https://arxiv.org/abs/2305.20050
- ReAct: Synergizing Reasoning and Acting in Language Models:
  https://arxiv.org/abs/2210.03629
- Toolformer: Language Models Can Teach Themselves to Use Tools:
  https://arxiv.org/abs/2302.04761
- rStar-Math: https://arxiv.org/abs/2501.04519

Evaluation robustness and contamination:

- GSM-Symbolic: https://arxiv.org/abs/2410.05229
- ConStat: https://arxiv.org/abs/2405.16281

Inference-time compute and backend deferral:

- Scaling LLM Test-Time Compute Optimally:
  https://arxiv.org/abs/2408.03314
- PagedAttention / vLLM: https://arxiv.org/abs/2309.06180
- DiffKV: https://arxiv.org/abs/2412.03131
