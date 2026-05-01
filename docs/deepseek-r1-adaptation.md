# DeepSeek-R1 Adaptation Note

Source: DeepSeek-R1, arXiv:2501.12948, https://arxiv.org/abs/2501.12948

## What Transfers

DeepSeek-R1 is useful here because its core signal is simple: generate multiple
reasoning attempts for verifiable tasks, reward answers with rule-based checks,
and use the successful trajectories to improve smaller models.

The full paper-scale path is not local-hardware realistic. GRPO training,
large rollout groups, and long context RL are outside this repo's current
budget. The transferable local path is:

1. choose prompts with deterministic verifiers;
2. sample multiple Main Agent answers;
3. score each answer with content-free verifier issue labels;
4. export only passing samples as LoRA/SFT candidate rows;
5. run held-out public or synthetic evaluation before accepting a weight change.

This is R1-lite rejection sampling, not full reinforcement learning.

## Why This Fits Fourth Path

The Main Agent still has one job: generate useful candidate answers. It does
not decide permission and it does not perform final safety review.

The verifier used by this path checks task correctness and output shape for
training data quality. It is not Cold Eyes and it is not a safety authority.
Final harmlessness remains outside the Main Agent.

## Command

```powershell
python main.py main-r1-sample-export `
  --profile qwen3-8b-s2t-lite `
  --samples-per-record 4 `
  --max-length-ratio 4 `
  --json `
  --timeout 900
```

The default input is `data/main_agent_hard_seed.jsonl`. The default output is
`runs/main-agent-r1-samples.jsonl`.

The JSON summary intentionally omits prompts, targets, and model outputs. The
JSONL export contains the accepted chat rows because that is the training
artifact.

## Relationship To Existing Paths

LightReasoner-style contrast export asks: where does a weaker model fail
relative to a stronger one?

R1-lite sample export asks: among several attempts from the current model, which
ones pass deterministic verification?

They are complementary. Contrast export is good at finding high-divergence
failure surfaces. R1-lite is good at turning additional inference-time compute
into cleaner supervised data for the same model family.

LIMO-style curation is the next pass after R1-lite. R1-lite keeps correct
samples; LIMO-style curation keeps the small subset most likely to act as useful
cognitive templates.

## Acceptance Gate

Do not train or publish a LoRA claim just because rows were exported. A useful
weight change needs at least:

- accepted sample count and acceptance rate;
- issue-label distribution for rejected samples;
- held-out synthetic evaluation that was not used by prompt hints;
- public benchmark smoke or full run where applicable;
- regression check that refusal-like behavior did not increase in Main Agent
  role tasks.

The target is better reasoning per local watt, not benchmark overfitting.
