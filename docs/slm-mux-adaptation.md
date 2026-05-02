# SLM-MUX Adaptation Note

This note records what this repository can use from SLM-MUX without turning the
local pipeline into a discussion-based multi-agent system.

Sources:

- Paper: https://arxiv.org/abs/2510.05077
- Project page: https://slm-mux.github.io
- Code: https://github.com/slm-mux/SLM-MUX

## First-Principles Reading

SLM-MUX is useful here because it asks the same hardware-level question as this
repository: if one small model is not enough, should capability come from a
larger single model or from multiple smaller models coordinated by a simple
architecture?

The paper's answer is specific:

1. Let each small model generate independently.
2. Do not make small models debate each other in natural language.
3. Estimate each model's confidence from self-consistency across its own
   samples.
4. Pick the output from the most confident model.
5. Select model groups by complementarity, not by standalone leaderboard rank.

That aligns with the Fourth Path idea: raw reasoning work can be split from
selection and final audit. It also directly warns against a tempting but bad
local design: several weak models talking to each other until they amplify the
same error.

## What Transfers

The transferable parts are practical:

- **Independent generation beats discussion for SLMs.** The repo should avoid
  "small-agent debate" as a default capability strategy. The local equivalent
  should be isolated samples plus deterministic selection.
- **Confidence can come from consistency.** For exact-answer tasks, a model that
  repeatedly gives the same extracted answer across samples can be treated as
  more confident than a model with scattered answers.
- **Complementarity matters more than count.** Adding another model is useful
  only if it solves tasks the current model misses without adding too many
  confident contradictions.
- **Model selection should be measured offline.** Before creating a live
  multi-model profile, collect outputs on a fixed corpus and compute:
  standalone clean rate, union clean rate, contradiction rate, final selected
  clean rate, latency, load time, and main calls.
- **Final safety stays separate.** SLM-MUX selects a candidate answer. It does
  not approve safety. The selected answer must still go through Classify,
  mechanical Cold Eyes, and Cold Eyes.

## What Does Not Transfer Directly

The full paper setup is not the immediate default for this repo:

- The strongest examples use multiple models and multiple samples per model.
  On the current machine, switching resident models can dominate runtime.
- Many SLM-MUX benchmarks use extractable answers. Open-ended chat needs a
  different confidence method, such as embedding similarity or local verifier
  labels, before selection is reliable.
- The current `qwen3-8b-s2t-lite` seed eval is already saturated at 40/40 clean,
  so a live multi-model mux would not prove much on the existing corpus.
- A mux profile that asks all models to answer every request would increase
  compute cost. It should be opt-in and task-routed.

## Local Adaptation Path

The right local version is an offline-first model complementarity lab:

1. **Create a harder mux eval set.**
   Include exact-answer math, small code repair, constrained formatting, and
   near-boundary helpfulness tasks where the current model still has variance.
2. **Collect independent samples.**
   Start with `qwen3:8b`, `llama3.1:8b`, and one optional larger/quality model
   such as `gemma3:12b`, using 2-3 samples per model.
3. **Extract task-level answers.**
   Use numeric extraction for math, test results for code, field coverage for
   structured output, and existing issue labels for role/canon leakage.
4. **Compute mux metrics offline.**
   Track:
   - best single-model score
   - union score: whether any model got the case right
   - contradiction penalty: confident wrong output competing with a correct one
   - selected score under consistency routing
   - total runtime and model load time
5. **Promote only if it beats simpler options.**
   A live `qwen3-8b-mux-lite` profile is worth adding only if offline results
   beat `qwen3-8b-s2t-lite` or `qwen3-8b-search` on a harder set by enough to
   justify the model-switch cost.

## Candidate Commands

The useful future commands are offline, not chat defaults:

```powershell
python main.py mux-collect --models qwen3:8b llama3.1:8b gemma3:12b --samples 3 --json
python main.py mux-eval --input-file runs\mux-samples.jsonl --json
python main.py mux-search --input-file runs\mux-samples.jsonl --max-models 2 --json
```

The live profile, if justified later, should stay narrow:

```text
qwen3-8b-mux-lite
```

It should route only exact-answer, code, or harder eval tasks at first. General
chat should keep the simpler `qwen3-8b-s2t-lite` path until measured otherwise.

## Relationship To Existing Profiles

- `qwen3-8b-s2t-lite`: cheapest current default candidate. Single model, local
  final-output selector, no extra model calls.
- `qwen3-8b-search`: single model, multiple candidates, model-based selector.
- `qwen3-8b-step-beam-lite`: future rStar-style intra-model step search.
- `qwen3-8b-mux-lite`: future SLM-MUX-style inter-model selection.

The distinction matters:

- rStar-style search asks one model to explore more deeply.
- SLM-MUX asks different models to expose complementary strengths.
- Separation-and-audit asks none of these selectors to own final safety.

## Implementation Gate

Do not implement live mux until these are true:

- a harder eval set exists
- at least two local models are installed and can be measured without unstable
  load failures
- offline union score is meaningfully higher than best single-model score
- contradiction penalty is low enough that consistency routing can select the
  correct candidate
- selected answers still pass architecture adversarial evaluation after the
  normal audit path

Initial hard eval gate:

```powershell
python main.py main-check --input-file data\main_agent_hard_seed.jsonl --min-total 24 --min-category 2 --json
python main.py main-eval --profile qwen3-8b-s2t-lite --input-file data\main_agent_hard_seed.jsonl --json --timeout 900 --max-length-ratio 4
```

The live mux path should be judged against this hard corpus, not only the
original 40-record seed eval.
