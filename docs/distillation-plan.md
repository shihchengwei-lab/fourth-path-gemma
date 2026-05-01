# Cold Eyes Distillation Plan

This is the practical distillation path for this repository.

The goal is not to train from local user conversations. The default logs still
omit prompts and model outputs. The first usable corpus is a synthetic,
human-written set of Cold Eyes examples that can be reviewed, versioned, and
validated before any training work begins.

## Why This Shape

Public distillation work points to two useful constraints:

- Distilling step-by-step uses intermediate rationales as extra supervision for
  smaller models, but this repository should keep rationales short and
  non-sensitive.
- LLM-as-a-judge work shows that smaller judges can be useful, but judge
  distillation can overfit or fail to generalize. The corpus should stay
  explicit, balanced, and easy to audit before it becomes training data.

## Current Artifact

Seed corpus:

```text
data/cold_eyes_seed.jsonl
```

Each JSONL record has:

- `id`
- `candidate`
- `verdict`: `pass` or `fail`
- `canon_clause`: `null`, `C1`, `C2`, or `C3`
- `reason`

The corpus intentionally does not contain `prompt`, chat history, system prompt,
model output logs, or hidden reasoning traces.

Validate it with:

```powershell
python main.py distill-check
python main.py distill-check --json
python main.py distill-check --min-pass 19 --min-fail 25 --min-clause 8
```

Measure current audit-model agreement with:

```powershell
python main.py distill-eval --profile qwen3-8b-local-max --json --timeout 900 --require-exact --min-exact-accuracy 1 --min-mechanical-cases 25
python main.py distill-eval --profile qwen3-8b-split-audit --json --timeout 900
```

The evaluation summary reports two scores:

- `verdict_accuracy`: pass/fail agreement.
- `exact_accuracy`: pass/fail plus exact canon-clause agreement.
- `mechanical_cases` and `llm_cases`: how many cases were handled by the
  fail-only mechanical gate versus the LLM judge.
- `estimated_llm_audit_calls_saved`: how many LLM audit calls the mechanical
  gate avoided during this corpus run.
- `source_counts_by_expected_clause`: mechanical, LLM, and cache counts split
  by expected pass/C1/C2/C3 label.
- `mismatches`: record ids where the verdict or clause differs, without
  including candidate text.

On the current 44-record seed corpus, `qwen3:8b` with structured JSON output
and audit `num_predict=64` got 44/44 pass/fail verdicts and 44/44 exact clause
matches in about 59.5 seconds in the 2026-05-01 full idle run. The run used 25
mechanical cases and 19 LLM judge cases, with no mismatches and no gate errors.

The same strict corpus also passed on `gemma3:12b` as the LLM judge, with
44/44 exact matches and no gate errors. It took about 106.4 seconds, so it is a
useful quality comparison point rather than the current compute-first judge.

That means the current pipeline is reliable enough for this first-stage seed
set, but the corpus is still small. It needs more held-out boundary examples
before a small judge can replace clause-level Cold Eyes adjudication.

## Next Training Path

1. Grow the synthetic corpus until each canon clause has diverse positive and
   negative examples.
2. Add contrast pairs: one candidate that passes and one near-miss that fails.
3. Fine-tune or LoRA a small local judge to emit the strict Cold Eyes JSON.
4. Compare the small judge against `qwen3-8b-local-max` on the fixed benchmark
   suite before using it as an audit profile.
5. Keep the existing Cold Eyes layer as the reference judge until the small
   judge matches pass/refusal behavior on a held-out set.

## Sources

- Distilling Step-by-Step: https://arxiv.org/abs/2305.02301
- Training an LLM-as-a-Judge Model: https://arxiv.org/abs/2502.02988
- Empirical Study of LLM-as-a-Judge: https://arxiv.org/abs/2403.02839
