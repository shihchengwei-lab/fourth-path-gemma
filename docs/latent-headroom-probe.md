# Main Agent Latent Headroom Probe

This probe asks a narrower question than `main-eval`:

Can the same fixed bottom model produce a verifier-passing answer for a task
after repeated prompt-shape attempts, even when the first pass fails?

That is a latent-capability question. It is not training data, not a public
benchmark, and not evidence that weights changed.

## Command

```powershell
python main.py main-latent-headroom --profile qwen3-8b-local-max --json --timeout 1200 --max-length-ratio 4
```

Default input:

```text
data\main_agent_latent_probe_seed.jsonl
```

Default prompt-shape variants:

- `baseline`: the original prompt through the current Main Agent prompt stack.
- `constraint_first`: front-load output-shape, unit, count, and allowed-content
  constraints.
- `self_check`: ask the model to silently check arithmetic, code behavior, and
  exact format before final output.

The JSON summary stores record ids, categories, verifier issue labels, counts,
timing, token counts, and variant names. It does not store prompt text, target
text, or model output.

## Metrics

- `first_pass_clean_count`: records that pass on the first baseline attempt.
- `any_clean_count`: records that pass at least once across variants and
  repeated attempts.
- `latent_rescue_count`: records that fail first pass but pass later.
- `never_clean_count`: records that never pass within the probe budget.
- `stable_clean_count`: records where every attempt passes.

Interpretation:

- `first_pass_clean` means the bottom model can produce the answer under the
  default route.
- `latent_rescued` means the answer is probably inside the model's reachable
  distribution, but the default route is unstable.
- `never_clean` means this probe did not find a usable route. That can mean a
  real bottom-model boundary, an unfair verifier, or a missing prompt/data
  pattern; inspect issue labels before changing architecture.

## 2026-05-02 Result

`qwen3-8b-local-max`, 8 records, 3 variants, 2 attempts per variant:

```text
runs\main-latent-headroom-qwen3-8b-local-max-20260502.json
```

- First-pass clean: 2/8.
- Any-clean after probe: 3/8.
- Latent rescued: 1/8.
- Stable clean: 2/8.
- Never clean: 5/8.
- Attempt clean rate: 14/48.
- Main calls: 48.
- Rescued category: code repair.

`qwen3-8b-s2t-lite`, same 48-call probe budget:

```text
runs\main-latent-headroom-qwen3-8b-s2t-lite-20260502.json
```

- First-pass clean: 2/8.
- Any-clean after probe: 3/8.
- Latent rescued: 1/8.
- Stable clean: 2/8.
- Never clean: 5/8.
- Attempt clean rate: 15/48.

`qwen3-8b-deliberate`, 3 variants, 1 attempt per variant, 48 total Main Agent
calls because each attempt includes one refinement pass:

```text
runs\main-latent-headroom-qwen3-8b-deliberate-lite-20260502.json
```

- First-pass clean: 2/8.
- Any-clean after probe: 2/8.
- Latent rescued: 0/8.
- Stable clean: 2/8.
- Never clean: 6/8.
- Attempt clean rate: 6/24.

## Current Reading

The bottom model still has some latent headroom, but this probe found it in a
narrow band. Prompt reshaping rescued one code-repair record. Local selection
slightly improved attempt-level cleanliness but not record-level headroom.
One-pass self-refinement did not help under the same approximate call budget.

The next high-value work is not KV cache and not generic extra runtime compute.
It is targeted prompt/data work for verifier-backed planning and strict-format
failures, followed by another fresh latent probe that has not already driven
fixes.
