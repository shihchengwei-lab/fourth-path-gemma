# Mix Distillation Adaptation Note

Source: Small Models Struggle to Learn from Strong Reasoners,
arXiv:2502.12143, https://arxiv.org/abs/2502.12143

## What Transfers

The paper identifies a small-model learnability gap: very small students can
perform worse when trained directly on long chain-of-thought or outputs from
much stronger teachers. Shorter reasoning chains, smaller-teacher traces, or a
mix of both can be easier for the student to internalize.

For this repo, the transferable rule is simple:

1. do not dump every long reasoning trace into a local adapter dataset;
2. keep a controlled mixture of short and long rows;
3. default to a small long-row budget, roughly the paper's 1:4 long/short mix;
4. treat stronger-teacher rows as useful but not automatically superior;
5. measure held-out eval before claiming the adapter helped.

## Local Command

Run this after R1-lite and LIMO-style curation:

```powershell
python main.py main-mix-distill-curate `
  --input-file runs\main-agent-limo-curated.jsonl `
  --output-file runs\main-agent-mix-distill.jsonl `
  --max-records 800 `
  --long-ratio 0.2 `
  --json
```

The command is local-only. It does not call Ollama. It reads SFT-style JSONL
rows, classifies assistant answers into short or long buckets by character
length, and writes a mixed dataset with a capped long-reasoning ratio.
`--max-records` is an upper bound; if there are not enough short rows, the
command will keep fewer rows instead of filling the dataset with extra long
traces.

## Fit With Qwen3:8B

The paper's strongest warning targets models at or below 3B parameters. Our
current target, `qwen3:8b`, is larger, and the paper's tables suggest 7B-8B
models can sometimes benefit from longer reasoning traces.

That does not mean unlimited long CoT is free. On this machine, long traces
increase latency and can teach verbosity. The practical default is therefore a
balanced mix: preserve some long templates for branching and verification, but
keep most rows short enough to match the local model's useful output budget.

## Pipeline Position

The intended order is:

1. `main-r1-sample-export`: spend inference compute and keep verifier-passing
   samples.
2. `main-limo-curate`: select high-quality cognitive templates.
3. `main-mix-distill-curate`: cap long-template ratio before LoRA/SFT.
4. held-out `main-eval` and public benchmark runs: accept or reject the adapter.

This is a data-quality control layer, not a safety layer. It keeps the Main
Agent focused on reasoning and leaves final Harmless review to Cold Eyes.
