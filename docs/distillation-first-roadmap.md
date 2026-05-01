# Distillation-First Roadmap

The next improvement order is:

1. distillation data quality
2. distillation format
3. verifier and tool-use
4. inference-time compute
5. KV cache changes

This order matters because better runtime plumbing cannot repair bad training
targets. A faster backend only makes the same mistakes cheaper if the data,
format, and verifier are weak.

One local no-Ollama gate now runs the publishable subset in this same order:

```powershell
python main.py local-release-gate --json
```

## 1. Distillation Data Quality

Data quality is the first gate.

A useful row must teach the Main Agent one clear behavior:

- answer directly without taking safety authority;
- follow exact output shape;
- avoid hidden-state or audit-role leakage;
- preserve concise helpfulness in defensive near-boundary cases;
- solve numeric, code, and planning tasks without benchmark-specific hints.

Near-term work:

- rotate in fresh held-out records before tuning more prompt hints;
- label failure types with verifier issue names, not private prompt/output text;
- keep rejected rows for error taxonomy, not for training;
- prefer small, high-confidence rows over broad low-quality corpora;
- split rows by capability target: format, code repair, numeric reasoning,
  planning, and defensive concise helpfulness.

Acceptance gate:

- held-out records improve without copying tuned examples;
- public smoke does not regress;
- summaries still omit private prompt and candidate text.

Local quality gate:

```powershell
python main.py main-data-quality-check --json
```

This checks the current seed, hard, and held-out corpora together for duplicate
ids, duplicate prompt hashes, and verifier coverage on hard/held-out files
without printing prompt or target text.

## 2. Distillation Format

The training format must preserve the architecture boundary.

Allowed training row shape:

```json
{"id":"row-id","category":"hard_format_constraints","messages":[{"role":"system","content":"generation role boundary"},{"role":"user","content":"synthetic prompt"},{"role":"assistant","content":"accepted Main Agent response"}],"source":"verifier_accepted"}
```

Format rules:

- use synthetic or explicitly opt-in prompts only;
- use chat-style `messages`, not top-level `prompt`, `response`, `candidate`,
  `target_response`, or `output` fields;
- never train the Main Agent to refuse or approve safety;
- do not include Cold Eyes reasoning as a Main Agent target;
- keep labels mechanical and short;
- keep train, tuned-dev, and held-out splits explicit.

Acceptance gate:

- JSONL validates deterministically;
- no row contains private chat logs;
- no row trains the Main Agent to cite hidden policy or audit internals.

Local format gates:

```powershell
python main.py main-check --input-file data\main_agent_seed.jsonl --min-total 40 --min-category 1 --json
python main.py main-check --input-file data\main_agent_hard_seed.jsonl --min-total 16 --min-category 2 --json
python main.py main-check --input-file data\main_agent_heldout_seed.jsonl --min-total 12 --min-category 2 --json
python main.py distill-check --min-pass 19 --min-fail 25 --min-clause 8 --json
python main.py main-training-data-report --input-file runs\main-agent-mix-distill.jsonl --require-system --json
```

## 3. Verifier And Tool-Use

Verifier quality comes before larger search.

Useful verifier targets:

- exact numeric answer extraction;
- required and forbidden terms;
- JSON/schema validity;
- code-only output shape;
- basic static checks for generated code snippets;
- future audited tool actions before side effects.

Tool-use belongs behind the side-effect boundary. A tool call should first
become an auditable action candidate with target, intent, args summary, and risk
surface. Text review after execution is not enough.

Acceptance gate:

- deterministic verifier labels match hand inspection on a small sample;
- tool actions fail closed before execution when unaudited;
- verifier summaries do not leak raw prompts or outputs.

Local verifier/tool-use gate:

```powershell
python main.py verifier-tool-gate --json
```

This combines the Cold Eyes distillation corpus balance check with the
mechanical fail-only and pre-execution action-audit boundary checks. It does not
call Ollama and does not print action targets, intents, or args.

## 4. Inference-Time Compute

Spend more inference only after the row quality and verifier loop are useful.

Useful directions:

- local selection for concise and exact-format failures;
- bounded candidate search for planning or comparison tasks;
- sequential refinement only where a verifier can catch improvements;
- prompt-shape compute allocation instead of uniform extra calls.

Acceptance gate:

- extra calls improve held-out quality enough to justify latency;
- Cold Eyes remains the final safety authority;
- search outputs become candidate data only after verifier acceptance.

Local inference-compute gate:

```powershell
python main.py inference-compute-gate --json
```

This refuses to treat compute as ready unless the default data-quality gate and
verifier/tool-use gate are clean, adaptive compute stays bounded by prompt
shape, and Ollama chat is still correctly represented as not token-level-ready.

## 5. KV Cache Changes

KV cache work is last for this repo.

`C:\Users\kk789\Desktop\llama-cpp-turboquant` is a reference implementation, not
an owned project. Do not push, open PRs, or add burden to that repository from
this work. Its useful role is to show what a logits/KV-capable backend might
look like after the higher-priority data and verifier work is ready.

Acceptance gate before revisiting KV:

- distillation data and format are stable;
- verifier/tool-use gates are useful;
- inference-time compute has a measured bottleneck that KV cache can address;
- any local backend test happens in this repo or a private scratch area, not by
  pushing to the reference repo.
