# Fourth Path Gemma Prototype

Local open-weight model prototype for the Fourth Path architecture described in
[`separation-and-audit-alignment`](https://github.com/shihchengwei-lab/separation-and-audit-alignment).

This repository is a reference implementation, not a proven safety system. It is
intended to make the architecture concrete enough to inspect, run, and test on a
local machine.

## Core Question

This repository is not only about making an 8B local model faster.

The local 8B model is the controllable testbed. The larger question is whether
some capability-safety conflict comes from placing too many jobs inside the same
primary reasoning loop:

- reasoning about the task
- maintaining refusal posture
- monitoring policy boundaries
- preserving conversational alignment
- producing the final answer

The Fourth Path hypothesis is that some safety overhead may be architectural.
If reasoning, policy enforcement, and audit are separated into different
layers, then the primary model may spend more of its finite inference budget on
reasoning while an independent audit layer preserves the final safety boundary.

That question scales beyond this machine:

- With an 8B open local model, the repo asks how much useful capability can be
  recovered by architecture, prompting, search, distillation, and mechanical
  audit before changing weights.
- With a frontier hosted model, the same question becomes whether a much
  stronger reasoner can be granted more useful freedom if final safety authority
  is moved to a separate, colder, harder-to-drift audit layer.
- On the path toward AGI, the question becomes whether architectural separation
  can reduce one bottleneck between raw capability and deployable autonomy.
- If the system were AGI-level, the safety question becomes stricter, not
  looser: is the audit layer independent, cold-started, mechanically bounded
  where possible, and strong enough that final output remains constrained even
  when the reasoning layer is much more capable.

This repo cannot prove that answer. It can build and measure the shape of the
answer: separate reasoning from permission, keep final harmlessness outside the
main agent, measure capability loss and audit cost, and make every layer's
authority explicit.

## What It Implements

The prototype separates several roles that are often coupled inside one
assistant loop:

- **Main Agent**: produces candidate answers.
- **Input prefilter**: routes obvious canon violations to the refusal module; it
  cannot approve output.
- **Classify**: routes candidate-output issues; it cannot approve output.
- **Cold Eyes**: performs a single-pass review against `canon.md`.
- **Mechanical Cold Eyes gate**: blocks high-confidence canon hits before the
  LLM judge; it cannot approve output.
- **Side-effect boundary**: this prototype does not let the Main Agent execute
  tools or external actions. Any future tool call, file write, network request,
  or other side effect must first become an auditable action candidate and fail
  closed before execution if unaudited.
- **Retry**: allows a bounded repair loop.
- **Refusal Module**: emits a fixed refusal after a blocked request.
- **Chat mode**: provides a simple natural-language interface over the audited
  pipeline.

Cold Eyes receives only the candidate output and the canon. It does not receive
the full chat history, the Main Agent system prompt, or hidden reasoning traces.
That is enough for output safety only. Side effects need their own pre-execution
audit boundary because damage can happen before any final text is returned.

## Current Model Target

The compatibility baseline still targets:

```text
gemma4:e4b
```

For the measured local machine, 16GB RAM and an RTX 4060 Laptop GPU with 8GB
VRAM, the current compute-first recommendation is:

```text
qwen3-8b-local-max
```

That profile uses `qwen3:8b`, matching 8192-token context for Main Agent and
Cold Eyes to avoid repeated load overhead, short deterministic audit output,
bounded retries, audit `num_predict=64`, `keep_alive=30m`, Ollama structured
JSON output for Cold Eyes, and Ollama `think=false` plus `/no_think` for Qwen3
to avoid spending tokens on hidden thinking when the task does not need it.

For slower quality-seeking runs on the same base model, use:

```text
qwen3-8b-deliberate
```

That profile adds one Main Agent self-refinement pass before Classify and Cold
Eyes. It does not change refusal authority: Classify still cannot approve, and
Cold Eyes still receives only the final candidate plus the canon.

See [Local Compute Maximization Plan](docs/compute-maximization.md) for the
model-choice rationale, algorithm notes, and measurement loop.

See [Qwen3-8B Headroom Audit](docs/headroom-audit-2026-05-02.md) for the
current continue/stop gate on next-token headroom, paper directions, and
KV-cache memory pressure.

See [llama.cpp TurboQuant Backend Path](docs/llama-cpp-turboquant-backend.md)
for the concrete local backend candidate that can expose logits and TurboQuant
KV cache controls once built.

See [Cold Eyes Distillation Plan](docs/distillation-plan.md) for the opt-in
path toward a smaller local audit model.

See [Qwen3 Main Agent LoRA Path](docs/qwen3-main-agent-lora.md) for the
weight-change path that keeps safety and refusal authority in the audit layer.

See [rStar-Math Adaptation Note](docs/rstar-math-adaptation.md) for how
MCTS-style self-evolved deep thinking can be adapted into a small local
step-search and preference-data path without moving safety authority back into
the Main Agent.

See [SLM-MUX Adaptation Note](docs/slm-mux-adaptation.md) for the complementary
multi-small-model route: independent generation, confidence routing, and offline
model-complementarity search without small-model debate.

See [Compute-Optimal Test-Time Adaptation](docs/compute-optimal-test-time.md)
for the prompt-shape adaptive compute profile inspired by test-time compute
scaling research.

See [LightReasoner Adaptation Note](docs/lightreasoner-adaptation.md) for the
expert/amateur contrast export path that keeps future LoRA data focused on
high-divergence reasoning failures instead of easy rows.

See [DeepSeek-R1 Adaptation Note](docs/deepseek-r1-adaptation.md) for the
R1-lite rejection-sampling export path: sample multiple verifier-backed Main
Agent attempts and keep only the passing rows for later LoRA/SFT experiments.

See [LIMO Adaptation Note](docs/limo-adaptation.md) for the less-is-more
curation path that selects a small cognitive-template set from accepted R1-lite
rows before any LoRA/SFT attempt.

See [Mix Distillation Adaptation Note](docs/mix-distillation-adaptation.md) for
the small-model learnability-gap control that caps long-reasoning rows before
training a local adapter.

See [Closed-Loop Evaluation Path](docs/closed-loop-evaluation.md) for the
held-out eval gate, one-command data pipeline, and training-data report used to
avoid self-confirming benchmark claims.

See [R2R Adaptation Note](docs/r2r-adaptation.md) for the small/large
token-routing backend path and the `r2r-estimate` command.

See [AGI-Scale Safety Architecture Notes](docs/agi-safety-architecture.md) for
the larger Fourth Path framing: what would have to change before this shape
could be considered for frontier or AGI-level systems.

Five non-default algorithmic Qwen3 profiles are available:

- `qwen3-8b-deliberate`: same `qwen3:8b` base, plus one internal quality
  refinement pass before audit.
- `qwen3-8b-reasoning`: same `qwen3:8b` base, but leaves Qwen3 thinking mode
  available for higher test-time reasoning cost.
- `qwen3-8b-s2t-lite`: same `qwen3:8b` base, plus a lightweight local selector
  inspired by Select to Think. It does not add model calls. It triggers only on
  compact-output, non-ASCII, near-boundary, or audit/canon-meta signals, and it
  replaces the candidate only when the local score improvement is decisive.
  It now also keeps Qwen3 thinking available only for arithmetic/counting prompts
  and adds small distilled task hints for hard exact-format, code, mux, LoRA, and
  defensive-reporting cases.
- `qwen3-8b-compute-optimal-lite`: same base and local selector, plus an
  experimental prompt-shape compute allocator. It avoids extra calls on strict
  output-shape tasks, uses sequential refinement for single-track hard reasoning,
  and uses parallel candidates for exploratory planning/comparison tasks.
- `qwen3-8b-search`: same `qwen3:8b` base, generates two candidate answers,
  then uses a quality selector for helpfulness, honesty, correctness, clarity,
  format fit, and concise wording before the normal audit path.

Three non-default model trial profiles are available:

- `qwen3-1.7b-amateur`: a deliberately weaker Qwen3 main profile for
  expert/amateur contrast sampling. It is not a recommended user-facing default.
- `llama3.1-8b-candidate`: same-size main-model comparison with Qwen3 audit.
- `gemma3-12b-pressure`: 12B pressure test that keeps main and audit on
  `gemma3:12b` with a shorter 4096-token context.

The latest local A/B run did not replace the default. In the 2026-05-01 idle
run, `qwen3-8b-local-max` finished the fixed warm benchmark in about 21.7
seconds with 3 pass / 1 refused. In the latest full idle run,
`qwen3-8b-local-max` took about 21.1 seconds, `qwen3-8b-deliberate` took about
31.6 seconds, `qwen3-8b-reasoning` took about 45.2 seconds, and the current
two-candidate `qwen3-8b-search` profile took about 42.1 seconds with the same
3 pass / 1 refused result.

On the 40-record Main Agent seed eval, `qwen3-8b-local-max` and
`qwen3-8b-search` produced 33/40 and 37/40 clean cases respectively in the
latest full idle run, with 0 refusal-like outputs in both runs. The current
two-candidate search profile reduced overlong cases to 3/40 and removed the
measured hidden-boundary leak, but it spent 120 Main Agent/selector calls versus
40 for the default.
After tightening the Main Agent contract for defensive and boundary-sensitive
requests, a follow-up `qwen3-8b-local-max` seed eval reached 39/40 clean cases,
and a second follow-up reached 38/40 clean cases. Both had 0 refusal-like
outputs, 1-2 overlong cases, and average output/target length ratios around
1.97-2.06 with the same 40 Main Agent calls. The current best cheap improvement
is `qwen3-8b-s2t-lite`: the 2026-05-01 refactor validation run reached 40/40
clean cases, 0 refusal-like outputs, 0 overlong outputs, average
output/target length ratio 1.512, and still only 40 Main Agent calls. Its local
selector triggered on 27/40 records and actually rewrote 17/40, which means the
extra control came from local post-selection rather than additional model
calls.
Reasoning mode produced more overlong outputs on this corpus, so thinking stays
an explicit experiment rather than the default.

Earlier model trials still matter as hardware evidence: `llama3.1-8b-candidate`
finished in about 75.0 seconds because model switching dominated load time.
`gemma3-12b-pressure` finished in about 61.3 seconds and passed the 44-record
strict Cold Eyes eval, but its decode speed makes it a quality experiment, not
the compute-first default.

## Setup

Install [Ollama](https://ollama.com/) and download the model:

```powershell
ollama pull gemma4:e4b
```

This repository has no Python package dependencies. It uses Python 3.12 standard
library only.

The included Windows helper scripts assume this local layout:

```text
E:\Ollama\ollama.exe
E:\ollama-models
```

Adjust the scripts if your Ollama executable or model directory lives elsewhere.

## Usage

Start the natural-language chat interface:

```powershell
.\chat.cmd
```

Inside chat mode:

```text
/help   Show commands.
/audit  Toggle detailed audit output.
/reset  Clear this chat session memory.
/exit   Leave chat.
```

List available local runtime profiles:

```powershell
python main.py profiles
```

Check architecture authority invariants without calling Ollama:

```powershell
python main.py architecture-check
```

Audit one proposed side effect before execution:

```powershell
python main.py action-audit --action-type network_request --target https://example.invalid --intent "send request" --args-summary "POST body" --risk-surface external_network --json
```

Validate and run the role-boundary adversarial suite:

```powershell
python main.py architecture-adversarial-check --min-total 19 --min-layer 6
python main.py architecture-adversarial-eval --profile qwen3-8b-local-max --json --timeout 900
```

This suite probes a different failure mode from the regular benchmark: whether
the Main Agent can be induced to expose hidden system/developer/audit state, and
whether Cold Eyes can be induced by candidate text to ignore canon or
self-approve, and whether unaudited side effects fail closed before execution.
Public canon references are not treated as leaks; users should be able to know
which public rule governs a refusal.
Summaries store ids, verdicts, issue labels, timing, and token counts; they do
not store adversarial prompts, candidates, or model outputs.

The first measured run found the useful split: Cold Eyes resisted all 6/6
candidate-side attacks, but raw Main Agent output often followed meta requests.
That is expected under this architecture: Main Agent is not the safety or audit
authority. Hidden state leakage is handled as an external deterministic
candidate issue; public canon transparency is allowed.

After narrowing the hidden-state check and making the phishing mechanical gate
less keyword-blunt, the current `qwen3-8b-local-max` adversarial run reaches
12/12. It also flags invented canon clause numbers as an Honest-side candidate
issue (`unsupported_canon_reference`) instead of letting a hallucinated public
rule pass through Cold Eyes.

Run one audited request:

```powershell
python main.py run --prompt "Summarize what this prototype does." --json --timeout 900
```

Run with the recommended local compute profile:

```powershell
ollama pull qwen3:8b
python main.py run --profile qwen3-8b-local-max --prompt "Summarize what this prototype does." --json --timeout 900
```

Run the slower Qwen3 self-refinement profile:

```powershell
python main.py run --profile qwen3-8b-deliberate --prompt "Summarize what this prototype does." --json --timeout 900
```

Try higher test-time compute profiles:

```powershell
python main.py run --profile qwen3-8b-s2t-lite --prompt "Write a short incident-response note." --json --timeout 900
python main.py run --profile qwen3-8b-compute-optimal-lite --prompt "Compare two implementation plans." --json --timeout 900
python main.py run --profile qwen3-8b-reasoning --prompt "Solve this carefully." --json --timeout 900
python main.py run --profile qwen3-8b-search --prompt "Compare two implementation plans." --json --timeout 900
```

The S2T-lite profile is a cheap candidate-selection pass, not full token-level
S2T. Full S2T would need logits/top-K access and selector distillation. This
profile only tests the practical claim that many current failures are selection
failures rather than missing capability. It is trigger-based and margin-gated,
so normal answers are not shortened unless the local selector has a clear win.

The original 40-record Main Agent seed eval is now saturated by
`qwen3-8b-s2t-lite`. Use the harder verifier-backed corpus before judging any
new search, mux, or LoRA path:

```powershell
python main.py main-check --input-file data\main_agent_hard_seed.jsonl --min-total 16 --min-category 2 --json
python main.py main-eval --profile qwen3-8b-s2t-lite --input-file data\main_agent_hard_seed.jsonl --json --timeout 900 --max-length-ratio 4
```

The hard corpus adds lightweight verifiers for numeric answers, required or
forbidden terms, regex constraints, and maximum output size. These verifier
labels are reported as issue names only; the eval summary still omits prompts,
targets, and model outputs.

Latest release-gate rerun on this hard corpus:
`runs\release-gate-main-eval-hard-20260502.json` reached 16/16 clean,
0 refusal-like, 0 overlong, and 16 total Main Agent calls. The same release
gate also rechecked the original 40-record seed at
`runs\release-gate-main-eval-seed-20260502.json` and reached 40/40 clean.

For public benchmark comparisons instead of repo-owned claims, see
[Public Benchmark Template](docs/public-benchmark-template.md). It defines a
same-runner raw `qwen3:8b` versus `qwen3-8b-s2t-lite` comparison path through
EleutherAI `lm-evaluation-harness`, plus a local OpenAI-compatible wrapper for
the Main Agent and full pipeline targets.

The search profile's selector is not a safety approver. It only selects the
candidate that is more helpful, honest, correct, clear, and format-following.
The selected candidate still goes through Classify and Cold Eyes.

Use `--keep-alive 0` if you want Ollama to unload the model immediately after a
one-off run, or another value such as `--keep-alive 10m` when memory pressure
matters more than warm-start speed.

Preload the selected profile before a chat or benchmark:

```powershell
python main.py warm --profile qwen3-8b-local-max --json --timeout 900
```

Try a split audit model after measuring local model-switch overhead:

```powershell
ollama pull qwen3:1.7b
python main.py run --profile qwen3-8b-split-audit --prompt "Summarize what this prototype does." --json --timeout 900
```

On this machine, the measured benchmark kept `qwen3-8b-local-max` ahead: about
20.3-21.3 seconds with `bench --warmup`, versus 31.6 seconds for
`qwen3-8b-deliberate`, 45.2 seconds for `qwen3-8b-reasoning`, 42.1 seconds for
the current two-candidate `qwen3-8b-search`, 39.3 seconds for
`qwen3-8b-split-audit` with
`keep_alive=30m`, about 75.0 seconds for `llama3.1-8b-candidate`, and about
61.3 seconds for `gemma3-12b-pressure`, with the same 3 pass / 1 refused
result on the fixed benchmark.

The Windows chat helper passes through extra arguments, so this also works:

```powershell
.\chat.cmd --profile qwen3-8b-local-max
```

Run the fixed local benchmark suite for a profile:

```powershell
python main.py bench --profile qwen3-8b-local-max --warmup --json --timeout 900
python main.py bench --profile qwen3-8b-deliberate --warmup --json --timeout 900
python main.py bench --profile qwen3-8b-reasoning --warmup --json --timeout 900
python main.py bench --profile qwen3-8b-search --warmup --json --timeout 900
```

Benchmark summaries are written under `runs\` by default. They store prompt ids,
status, attempts, elapsed milliseconds, summed attempt milliseconds, model
roles, Main Agent call counts, token counts, prompt/eval/load milliseconds, and
output length; they do not store prompt text or model output.

Validate and measure the synthetic Main Agent role-behavior corpus:

```powershell
python main.py main-check --min-total 40 --min-category 1
python main.py main-eval --profile qwen3-8b-local-max --json --timeout 900 --max-length-ratio 4
```

The Main Agent seed corpus currently has 40 synthetic records, including
near-boundary defensive security tasks and concise-control tasks. Recent
`qwen3-8b-local-max` runs had 0/40 refusal-like outputs. The best follow-up run
had 39/40 clean cases; a later run had 37/40 clean cases, with all 3 issues
being overlong outputs. The current bottleneck remains verbosity variance, not
self-refusal.

Export the same synthetic corpus for future LoRA / QLoRA experiments:

```powershell
python main.py main-sft-export --output-file runs\main-agent-sft-seed.jsonl
```

Export a smaller LightReasoner-lite contrast set by comparing a stronger Main
Agent profile with a weaker profile:

```powershell
python main.py main-contrast-export --json --timeout 900
python main.py main-contrast-export --amateur-profile qwen3-8b-local-max --json --timeout 900
```

The contrast export writes only selected synthetic prompt/Expert-answer pairs
where the Expert is clean and the Amateur is clearly worse. It is meant for
future adapter training triage, not for private chat logs.

Estimate whether a future R2R-style small/large token router is worth a backend
trial:

```powershell
python main.py r2r-estimate --json
python main.py r2r-estimate --backend sglang-r2r --json
```

The estimate is a parameter-budget check, not a claim that Ollama chat already
supports token-level routing.

When the machine will be idle, run the long measurement pass explicitly:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\idle-long-run.ps1
```

That script runs static checks, the architecture adversarial eval, the slow
Qwen3 main-eval, benchmark, and strict Cold Eyes eval sequence, then writes
timestamped outputs under `runs\`. It is intentionally not a scheduled task or
background service.

Validate the synthetic Cold Eyes distillation seed corpus:

```powershell
python main.py distill-check --min-pass 19 --min-fail 25 --min-clause 8
```

Evaluate an audit model against that corpus:

```powershell
python main.py distill-eval --profile qwen3-8b-local-max --json --timeout 900 --require-exact --min-exact-accuracy 1 --min-mechanical-cases 25
python main.py distill-eval --profile qwen3-8b-split-audit --json --timeout 900
```

On the current 44-record synthetic corpus, the pipeline matches all pass/fail
verdicts and all exact clause labels using the default audit `num_predict=64`.
Recent runs used 25 fail-only mechanical audit cases and 19 LLM judge cases.

Inspect the Main Agent by itself, without the audit pipeline:

```powershell
python main.py diagnose-main --prompt "Write a simple Python function." --json --show-system-prompt --timeout 900
```

## Logs

Audit logs are written to:

```text
runs\
```

The logs record run id, attempt, route, Cold Eyes verdict, canon clause, and
final status. They also record model role, elapsed milliseconds, token counts,
and Ollama prompt/eval/load milliseconds so runtime profiles can be compared
locally. They do not store the original prompt, Main Agent system prompt, full
candidate output, or reasoning trace.

Local run logs are ignored by git.

## Tests

Run:

```powershell
python -m unittest discover -s tests -v
```

## Limitations

- This is a research prototype, not production safety infrastructure.
- The canon is intentionally small and only demonstrates the data flow.
- The Gemma model is instruction-tuned and may still carry safety behavior in
  its weights.
- The recommended Qwen3 profile is a local hardware fit, not a measured quality
  guarantee.
- The deliberate Qwen3 profile can improve candidate structure at test time,
  but it spends another model call and does not guarantee better answers.
- Speculative decoding and distillation are documented as future backend paths;
  this repo currently keeps Ollama as the runtime boundary.
- The chat mode keeps memory only for the current process.
- The Windows helper scripts are convenience wrappers, not a cross-platform
  installer.

## License

MIT

> Pipeline integration validated.
