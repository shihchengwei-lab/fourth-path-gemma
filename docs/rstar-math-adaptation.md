# rStar-Math Adaptation Note

This note records what this repository can use from rStar-Math without pretending
that a laptop-class local assistant can reproduce the full training system.

Source:

- Paper: https://arxiv.org/abs/2501.04519
- Code branch: https://github.com/microsoft/rStar/tree/rStar-math

## First-Principles Reading

rStar-Math is useful here because it treats small-model capability as a compute
allocation problem, not only a model-size problem.

The core loop is:

1. A policy small language model proposes the next reasoning step.
2. A process reward or preference model scores partial trajectories.
3. Search keeps branches that are more likely to lead to a verifiable correct
   answer.
4. The best traces become training data for the next round.

That matches this repository's question: can an 8B local model recover more
useful capability if the architecture gives raw reasoning more budget while
separate layers handle selection, verification, and final audit?

## What Transfers

The transferable parts are structural:

- **Test-time search beats one-shot generation when verification exists.**
  rStar-Math uses MCTS because a hard problem can be decomposed into smaller
  step choices. For this repo, the equivalent is not every chat turn. It is
  coding, math, format-constrained, and seed-eval tasks where a verifier can
  cheaply score progress.
- **A selector is more powerful when it sees process signals, not only final
  text.** The current `qwen3-8b-s2t-lite` selector is final-output only. The
  rStar direction says the next selector should score partial drafts or steps
  when the task has a checkable structure.
- **Do not train from noisy scalar scores when preferences are enough.**
  rStar-Math trains a process preference model from positive/negative pairs
  instead of trusting exact per-step Q-values. For this repo, that suggests a
  small pairwise dataset: better/worse candidate, same prompt, same public
  verifier labels.
- **Self-evolution needs a closed measurement loop.** Each round should produce
  new candidate traces, filter them by local verifiers, export SFT/preference
  data, then compare against the previous profile. No teacher model is required
  for the basic loop.

## What Does Not Transfer Directly

The full rStar-Math recipe is too heavy for the current hardware and too
domain-specific for general chat:

- It relies on large-scale MCTS rollouts and millions of synthesized math
  solutions.
- The reported training setup assumes high-end multi-GPU resources, not a
  16 GB RAM / RTX 4060 Laptop 8 GB VRAM machine.
- Code execution is a strong verifier for math and programming, but not for
  open-ended writing or advice.
- MCTS over hidden chain-of-thought is not needed for the public chat path.
  If traces are generated for training, they should be synthetic, explicit, and
  separate from private audit logs.

## Local Adaptation Path

The right local version is a small, opt-in reasoning lab:

1. **Start with step beam search, not full MCTS.**
   For a selected task, generate 2-4 short partial candidates per step for
   2-3 steps. Keep the branching factor low enough that it stays usable on
   `qwen3:8b`.
2. **Use mechanical verifiers first.**
   Score candidates with existing local checks: `main_candidate_issues`,
   requested format fit, length ratio, exact/regex answer checks, Python unit
   tests for code tasks, and arithmetic checks for math tasks.
3. **Keep safety out of the search reward.**
   Search may optimize helpfulness, honesty, correctness, and format fit. Final
   permission still belongs to Classify, mechanical Cold Eyes, and Cold Eyes.
4. **Export pairwise preference data.**
   For each prompt, store only reviewed synthetic rows:
   `prompt`, `better_candidate`, `worse_candidate`, `verifier_labels`, and
   `reason`. Do not use default audit logs.
5. **Train only after the eval proves a gap.**
   The current `qwen3-8b-s2t-lite` run already reaches 40/40 clean on the seed
   eval. Weight changes should wait for a harder held-out set where search
   clearly finds better candidates than one-shot generation.

## Candidate Profiles

The next experimental profiles should be narrow:

- `qwen3-8b-step-beam-lite`: generate a few short candidates per step, select
  with local verifiers, then send one final answer through the existing audit
  path.
- `qwen3-8b-verify-code-lite`: for coding/math tasks only, let the candidate
  include a small executable check in a sandboxed verifier, then discard the
  check and output the answer.
- `qwen3-8b-preference-data`: offline command only; generate better/worse pairs
  for LoRA or a small selector model.

These profiles should not become defaults until they beat `qwen3-8b-s2t-lite`
on a harder corpus without unacceptable latency.

## Next Implementation Gate

Before writing code, add a harder eval set where one-shot `qwen3-8b-s2t-lite`
is not already saturated:

- multi-step math with exact answer checks
- small code repair tasks with local unit tests
- format-constrained summaries where both brevity and coverage are measurable
- planning tasks with required bullets or fields

Initial gate now exists:

```powershell
python main.py main-check --input-file data\main_agent_hard_seed.jsonl --min-total 24 --min-category 2 --json
python main.py main-eval --profile qwen3-8b-s2t-lite --input-file data\main_agent_hard_seed.jsonl --json --timeout 900 --max-length-ratio 4
```

Only after that should the repo implement step search. Otherwise the search
will look impressive while measuring an already-solved seed set.
