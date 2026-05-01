# R2R Adaptation Note

Source paper: [R2R: Efficiently Navigating Divergent Reasoning Paths with Small-Large Model Token Routing](https://arxiv.org/abs/2505.21600)

Code reference: [thu-nics/R2R](https://github.com/thu-nics/R2R)

## Useful Idea

R2R asks a sharper version of the local compute question:

> Can a small model generate most tokens, while a large model is used only when
> the next token would change the reasoning path?

The paper reports that small and large models often differ on only a minority of
tokens, and only a subset of those differences are truly path-divergent. A
router can therefore keep most decoding on the small model and route only
critical tokens to the large model.

This is different from query-level routing. It does not choose small or large
for the whole response. It routes inside the response.

## Why It Is Not An Ollama-Chat Feature

True R2R needs backend features that the current Ollama chat path does not
expose:

- token-level logits or top-k probabilities from the small model
- hidden states or equivalent router features
- token-by-token accept/replace control
- large-model prefill and KV-cache update from a mixed prefix
- small model, large model, and router resident at the same time
- a trained router checkpoint for the exact model pair

So this repo should not claim that `qwen3:8b` under Ollama has full R2R routing.
The useful immediate step is to measure the economics and keep a clear backend
gate.

## Local Command

Estimate the parameter budget for a Qwen3-1.7B plus Qwen3-8B style router:

```powershell
python main.py r2r-estimate --json
```

Defaults:

- small model: 1.7B
- large model: 8B
- router: 0.056B
- routed-to-large token rate: 0.13
- backend model: `ollama-chat`

The default estimate is:

```text
average activated params/token ~= 1.7 + 0.056 + 0.13 * 8 = 2.796B
```

That is about 35% of an 8B large-only decode by activated-parameter accounting.
It is not a wall-clock promise, because real speed depends on backend scheduling,
KV-cache reuse, batching, and whether both models stay resident.

Check the same economics under an R2R-capable backend model:

```powershell
python main.py r2r-estimate --backend sglang-r2r --json
```

## How This Fits The Repo

R2R is a backend replacement path, not a prompt tweak:

- `qwen3-8b-s2t-lite` improves final candidate selection without extra model
  calls.
- `qwen3-8b-compute-optimal-lite` changes how many calls are spent per prompt.
- `main-contrast-export` finds Expert/Amateur training rows.
- R2R would change the decoding engine itself so small/large models cooperate
  at token level.

If implemented later, the Main Agent can use R2R as its generation backend, but
final safety authority should remain unchanged: Classify, mechanical Cold Eyes,
and Cold Eyes still review the final candidate output.

## Practical Gate

Do not add a live R2R profile until these are true:

1. An R2R-capable backend is installed and can expose an OpenAI-compatible chat
   endpoint.
2. A router checkpoint exists for the exact local pair, preferably
   `Qwen3-1.7B + Qwen3-8B` or `Qwen3-0.6B + Qwen3-8B`.
3. The backend can keep both models resident without pushing this machine into
   unstable memory pressure.
4. Public benchmark and repo hard-corpus runs show either lower latency at equal
   quality or higher quality at similar latency.
5. The separated audit contract remains unchanged.
