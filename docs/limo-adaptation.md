# LIMO Adaptation Note

Source: LIMO: Less is More for Reasoning, arXiv:2502.03387,
https://arxiv.org/abs/2502.03387

## What Transfers

LIMO argues that a knowledge-rich base model may need a small number of
carefully chosen reasoning demonstrations, not a massive SFT dump. The useful
local takeaway is not "train on everything." It is:

1. start from verifier-backed rows;
2. keep rows that are challenging enough to require reasoning;
3. prefer assistant answers that show useful cognitive templates;
4. preserve diversity by category or skill area;
5. stop when extra rows are no longer improving held-out evaluation.

For this repo, LIMO is a curation layer after R1-lite rejection sampling. R1-lite
answers "which generated samples are correct enough?" LIMO-style curation asks
"which correct samples are worth teaching back to the model?"

## Local Command

```powershell
python main.py main-r1-sample-export `
  --profile qwen3-8b-s2t-lite `
  --samples-per-record 4 `
  --max-length-ratio 4 `
  --json `
  --timeout 900

python main.py main-limo-curate `
  --input-file runs\main-agent-r1-samples.jsonl `
  --output-file runs\main-agent-limo-curated.jsonl `
  --max-records 800 `
  --json
```

The first command spends inference compute. The second command is local scoring
only.

## Scoring

The local scorer follows the paper's curation direction, not its exact training
setup. It favors rows with:

- enough answer length to expose reasoning structure;
- verification markers such as checking or validating;
- exploratory markers such as cases, alternatives, or assumptions;
- connective markers such as because, therefore, since, then;
- explicit structure or final-answer markers.

The scorer is intentionally simple and inspectable. It is not a neural reward
model and it does not judge safety. It only ranks already accepted training rows
as possible cognitive templates.

## Fit With Fourth Path

This does not change the 3H split:

- Main Agent: generate candidate answers and future training rows.
- Verifier/R1-lite: check task correctness and output shape for data quality.
- LIMO curation: select compact, high-value cognitive templates.
- Cold Eyes: final Harmless audit of user-facing output.

LIMO strengthens the Main Agent through better examples. It does not make the
Main Agent responsible for final safety review.

## Limits

The LIMO paper used Qwen2.5-32B-Instruct and competition math. This repo's
current target is `qwen3:8b`, so the expected ceiling is lower. The lesson that
does transfer is sample efficiency: before collecting tens of thousands of
mixed-quality rows, try a few hundred high-quality, verifier-backed rows and
measure held-out improvement.

Do not count a curated file as a capability gain. It becomes evidence only after
training and held-out evaluation improve without benchmark-specific overfitting.

If the curated file contains long reasoning traces, run
`main-mix-distill-curate` before LoRA/SFT. That keeps the less-is-more set from
becoming a long-CoT dump that a smaller local model may learn poorly.
