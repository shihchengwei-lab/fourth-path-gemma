# Closed-Loop Evaluation Path

This repo now separates three things that should not be confused:

1. training-data generation;
2. held-out evaluation;
3. public benchmark comparison.

The goal is to make Main Agent improvements falsifiable. A curated file is not
evidence of capability. A held-out or public eval improvement is evidence.

## Held-Out Gate

The current fresh held-out corpus is:

```powershell
python main.py main-check --input-file data\main_agent_fresh_heldout_seed.jsonl --min-total 12 --min-category 2 --json
python main.py main-eval --profile qwen3-8b-s2t-lite --input-file data\main_agent_fresh_heldout_seed.jsonl --json --timeout 900 --max-length-ratio 4
```

Do not use this file as R1/LIMO/Mix training input. It exists to catch
overfitting to `main_agent_seed.jsonl`, `main_agent_hard_seed.jsonl`, and the
older held-out corpora that already informed prompt-shape fixes.

Latest fresh held-out evidence:

- Baseline path:
  `runs\main-eval-qwen3-8b-s2t-lite-fresh-heldout-20260502.json`
- Baseline result: 2/12 clean, 0 refusal-like, 0 overlong. Failures were
  concentrated in required-any checks, code-repair verifier checks, one numeric
  rounding case, planning, and defensive concise-helpfulness.
- Current tuned-regression path:
  `runs\main-eval-qwen3-8b-s2t-lite-fresh-heldout-final-20260502.json`
- Current tuned-regression result: 12/12 clean, 0 refusal-like, 0 overlong,
  12 Main Agent calls, 1.0 clean cases per Main Agent call.
- Latest repeat ablation path:
  `runs\main-eval-ablation-fresh-heldout-final-v3-20260502.json`
- Latest repeat ablation result: `qwen3-8b-local-max`, `qwen3-8b-s2t-lite`,
  and `qwen3-8b-compute-optimal-lite` each reached 11/12 clean. The adaptive
  profile used 16 Main Agent calls versus 12 for the other two, so it did not
  justify extra inference-time compute on this tuned surface.
- Failure-report path:
  `runs\main-eval-failure-report-ablation-fresh-heldout-final-v3-20260502.json`
- Remaining repeated target: `fresh-heldout-plan-001` with
  `missing_required_any`, which points to planning-data/verifier targeting
  rather than KV-cache or broader runtime changes.
- Caveat: this fresh file has now informed prompt hints and verifier repairs.
  Use another fresh held-out file or public benchmark before claiming broader
  capability improvement.

Latest local evidence:

- Baseline smoke path:
  `runs\main-eval-qwen3-8b-s2t-lite-heldout-smoke.json`
- Baseline smoke result: 8/12 clean, with failures concentrated in planning
  and safe near-boundary concise-helpfulness tasks.
- Current path after prompt-shape hint and local selector budget fixes:
  `runs\main-eval-qwen3-8b-s2t-lite-heldout-v3.json`
- Current result: 12/12 clean, 0 refusal-like, 0 overlong, 12 Main Agent calls,
  average 95.17 eval tokens per clean case.
- Caveat: this older held-out set has now informed general prompt-shape fixes.
  Use `data\main_agent_fresh_heldout_seed.jsonl` or a public benchmark before
  claiming broader capability improvement.

## Data Pipeline

The one-command data pipeline is:

```powershell
python main.py main-distill-pipeline `
  --profile qwen3-8b-s2t-lite `
  --input-file data\main_agent_hard_seed.jsonl `
  --samples-per-record 4 `
  --max-length-ratio 4 `
  --json `
  --timeout 900
```

It writes:

- R1-lite verifier-accepted samples;
- LIMO-style cognitive-template curation;
- Mix Distillation short/long ratio curation;
- a manifest tying the artifacts and parameters together.
The manifest points the next evaluation step at the fresh held-out file rather
than the older saturated held-out set.

## Data Report

For any generated SFT JSONL file:

```powershell
python main.py main-training-data-report --input-file runs\main-agent-mix-distill.jsonl --require-system --json
```

The report includes only metadata such as category counts, source counts,
short/long bucket counts, row length statistics, duplicate ids, and whether
system messages are present. With `--require-system`, it fails the release gate
when any row lacks a system message or duplicate row ids are present. It does
not print prompts, targets, or assistant outputs.

## Acceptance Rule

Do not train or claim improvement unless all of these are true:

- generated data passes the report without accidental duplication or collapse;
- held-out `main-eval` improves against the same runner and profile baseline;
- public benchmark smoke or full run does not regress;
- Main Agent refusal-like behavior does not increase;
- Cold Eyes remains outside the Main Agent capability path.
