# Local Compute Maximization Plan

This note records one subproblem of the Fourth Path hypothesis: how much useful
reasoning can be recovered from the same local model and hardware before
changing weights or moving to a stronger backend. It is not the full repository
thesis and it is not a benchmark claim.

The broader thesis is about placement of work. The 8B local model is a
controllable testbed for asking whether reasoning, policy, and audit should be
separated before the same architecture is trusted with larger frontier models
or AGI-level systems.

## Hardware Target

Measured local target:

- CPU: 13th Gen Intel Core i5-13500HX, 14 cores / 20 threads
- RAM: 16 GB
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB VRAM

The practical envelope is a strong 4B-9B quantized model, short audit prompts,
and bounded retries. Larger 12B+ models may work in narrow cases, but the
8 GB VRAM / 16 GB RAM mix makes long-context runs and model switching fragile.

## Default Recommendation

Use `qwen3-8b-s2t-lite` first:

```powershell
ollama pull qwen3:8b
python main.py run --profile qwen3-8b-s2t-lite --prompt "Summarize this project." --json
```

This profile keeps Main Agent and Cold Eyes on `qwen3:8b` to avoid expensive
model swapping. Main Agent and Cold Eyes both use `num_ctx=8192`, because local
benchmarks showed that changing context size between calls can force repeated
load overhead. Cold Eyes still uses deterministic sampling and a short
`num_predict=64` limit. Cold Eyes also asks Ollama for structured JSON output,
so the runtime does not rely only on prompt wording for audit shape. The profile
sets Ollama `keep_alive=30m` so chat and benchmark loops are less likely to pay
cold-load cost between calls. It appends `/no_think` and sends Ollama's
`think=false` API parameter for profiles with thinking disabled. Compared with
`qwen3-8b-local-max`, it adds a local selector for concise and boundary-sensitive
answers without adding model calls.

Use `qwen3-8b-local-max` as the same-base ablation without the local selector:

```powershell
python main.py run --profile qwen3-8b-local-max --prompt "Summarize this project." --json
```

Use `qwen3-8b-deliberate` when the task is worth extra latency:

```powershell
python main.py run --profile qwen3-8b-deliberate --prompt "Summarize this project." --json
```

This keeps the same `qwen3:8b` base model but adds one Main Agent refinement
pass before Classify and Cold Eyes. It is an algorithmic stretch mode: the
weights are unchanged, but the model gets a second local pass to improve
correctness, completeness, structure, and concise wording. It is deliberately
not an approval path. Classify still cannot approve, and Cold Eyes still sees
only the final candidate plus the canon.

Use `qwen3-8b-reasoning` to test whether allowing Qwen3's thinking path improves
harder tasks:

```powershell
python main.py run --profile qwen3-8b-reasoning --prompt "Solve this carefully." --json
```

Use `qwen3-8b-search` to spend more test-time compute on candidate search:

```powershell
python main.py run --profile qwen3-8b-search --prompt "Compare two implementation plans." --json
```

Use `qwen3-8b-s2t-lite` to test a cheap Select-to-Think-inspired local selector
without extra model calls:

```powershell
python main.py run --profile qwen3-8b-s2t-lite --prompt "Write a short incident-response note." --json
```

This is not full token-level S2T. It is a small local selector for the current
failure mode: overlong near-boundary answers and accidental canon/audit meta
language. It triggers only on compact-output, non-ASCII, near-boundary, or
audit/canon-meta signals, and replaces the candidate only when the local score
improvement is decisive. Full S2T would require logits/top-K access and selector
distillation.

Use `qwen3-8b-compute-optimal-lite` to test adaptive test-time compute:

```powershell
python main.py run --profile qwen3-8b-compute-optimal-lite --prompt "Compare two implementation plans." --json
```

This profile keeps the same base model and local selector, but chooses the
extra Main Agent compute per prompt shape. It does not target benchmark ids.
Strict output-shape prompts get one candidate and no refinement. Exploratory
planning/comparison prompts get two candidates and a Helpful/Honest quality
selector. Single-track hard reasoning gets one sequential refinement. The
selector still cannot approve safety, and the final candidate still goes
through Classify and Cold Eyes.

The search profile generates two candidate answers, then asks a quality
selector to choose by Helpful/Honest criteria: usefulness, correctness, clarity,
format fit, and concise wording. The selector does not decide harmlessness and
does not approve output. The selected candidate still goes through Classify and
Cold Eyes.

Two trial profiles are available for bottom-model checks, but they are not the
default:

```powershell
ollama pull llama3.1:8b
ollama pull gemma3:12b
python main.py bench --profile llama3.1-8b-candidate --warmup --json --timeout 900
python main.py bench --profile gemma3-12b-pressure --warmup --json --timeout 900
```

`llama3.1-8b-candidate` tests an 8B peer model while keeping Qwen3 as the audit
judge. This isolates main-model behavior, but it also exposes the real cost of
alternating between two 8B models on an 8 GB laptop GPU. `gemma3-12b-pressure`
tests whether a 12B model can run at all under the local memory budget. It uses
one model for main and audit to avoid model-switch churn, and lowers context to
4096 tokens to keep the run inside the practical memory envelope.

Keep `qwen3-8b-split-audit` as an experiment, not the default:

```powershell
ollama pull qwen3:1.7b
python main.py run --profile qwen3-8b-split-audit --prompt "Summarize this project." --json
```

The split-audit profile makes Cold Eyes cheaper per token, but Ollama may pay a
reload cost when switching between the main and audit models. On an 8 GB laptop
GPU, that trade must be measured instead of assumed.

On the current machine, benchmark runs after downloading both Qwen models
showed:

- `qwen3-8b-local-max` with matching 8192 context: 3 pass / 1 refused, about
  27.8 seconds shortly after the context fix, and about 21.9 seconds once the
  model was warm.
- `qwen3-8b-local-max` after adding API `keep_alive=30m`: 3 pass / 1 refused,
  about 26.5 seconds with first-call load included.
- `qwen3-8b-local-max` with `bench --warmup`, structured JSON audit, and audit
  `num_predict=64`: 3 pass / 1 refused, about 22.4 seconds after the
  concise-output prompt change, with about 0.4 seconds of recorded load time
  inside benchmark cases.
- `qwen3-8b-deliberate` with `bench --warmup`: 3 pass / 1 refused, about 31.6
  seconds in the 2026-05-01 idle run. Each passing case made two Main Agent
  calls, so it is a quality mode rather than the fastest default.
- `qwen3-8b-reasoning` with `bench --warmup`: 3 pass / 1 refused, about 45.2
  seconds in the 2026-05-01 idle run. It did not improve the fixed benchmark
  result, and it made the Main Agent role-behavior eval slower and more
  verbose.
- `qwen3-8b-search` with `bench --warmup`: 3 pass / 1 refused, about 42.1
  seconds after reducing the search profile to two candidates. It is a
  high-cost optional quality mode, not the default.
- `llama3.1-8b-candidate`: 3 pass / 1 refused, about 75.0 seconds. About 56.0
  seconds were recorded load time from switching between `llama3.1:8b` and
  `qwen3:8b`, so it is not a good default on this hardware.
- `gemma3-12b-pressure`: 3 pass / 1 refused, about 61.3 seconds. Load time was
  low because main and audit used the same model, but decode was much slower.
- `qwen3-8b-local-max` before matching audit context: 3 pass / 1 refused, about
  53.1 seconds.
- `qwen3-8b-split-audit`: 3 pass / 1 refused, about 47.1 seconds before the
  keep-alive change, and about 39.3 seconds with `keep_alive=30m`.

Even with `keep_alive=30m`, the split-audit run still paid multi-second load
costs when alternating between the 8B main model and 1.7B audit model. The
2026-05-02 full idle run also added `qwen3-8b-s2t-lite` to the standard long
measurement pass. That run keeps the base model choice on `qwen3:8b`, but moves
the compute-first recommendation from raw `qwen3-8b-local-max` to
`qwen3-8b-s2t-lite`.

The 2026-05-02 full idle run gives the current Qwen3 comparison point on the
40-record Main Agent seed corpus:

- `qwen3-8b-s2t-lite`: 40/40 clean, 0 refusal-like outputs, 0 overlong cases,
  40 Main Agent calls, about 162.7 seconds.
- `qwen3-8b-local-max`: 36/40 clean, 0 refusal-like outputs, 3/40 overlong
  cases, 40 Main Agent calls, about 521.1 seconds in this run.
- `qwen3-8b-deliberate`: 37/40 clean, 0 refusal-like outputs, 2/40 overlong
  cases, 80 Main Agent calls, about 333.8 seconds.
- `qwen3-8b-reasoning`: 36/40 clean, 0 refusal-like outputs, 4/40 overlong
  cases, 40 Main Agent calls, about 511.4 seconds.
- `qwen3-8b-search`: 37/40 clean, 0 refusal-like outputs, 2/40 overlong cases,
  120 Main Agent/selector calls, about 480.8 seconds.

The earlier 2026-05-01 idle run remains useful historical context:

- `qwen3-8b-local-max`: 33/40 clean, 0 refusal-like outputs, 7/40 overlong
  cases, average output/target length ratio 2.631, 40 Main Agent calls, about
  176.7 seconds.
- `qwen3-8b-deliberate`: 32/40 clean, 0 refusal-like outputs, 7/40 overlong
  cases, average length ratio 2.475, 80 Main Agent calls, about 348.7 seconds.
  The extra pass did not buy enough quality on this corpus.
- `qwen3-8b-reasoning`: 29/40 clean, 0 refusal-like outputs, 11/40 overlong
  cases, average length ratio 3.017, 40 Main Agent calls, about 501.9 seconds.
  Leaving thinking mode available increased cost and verbosity here.
- `qwen3-8b-search`: 37/40 clean, 0 refusal-like outputs, 3/40 overlong cases,
  average length ratio 2.306, 120 Main Agent/selector calls, about 457.7
  seconds after reducing the search profile to two candidates. It removed the
  measured hidden-boundary leak and reduced overlong failures, but the cost is
  still too high for the default.
- A direct `--main-num-predict 192` cap was also tested against
  `qwen3-8b-local-max`. It lowered average length ratio to 2.396, but clean
  cases dropped to 34/40, overlong cases rose to 6/40, and total time stayed
  about the same. A blunt output cap is not the right next default.
- A narrow Main Agent contract change for defensive and boundary-sensitive
  requests was the best cheap improvement so far. A follow-up
  `qwen3-8b-local-max` seed eval reached 39/40 clean, 0 refusal-like outputs,
  1/40 overlong case, average length ratio 1.970, 40 Main Agent calls, and
  about 157.1 seconds. A second follow-up run was 38/40 clean, 2/40 overlong,
  average length ratio 2.056, 40 Main Agent calls, and about 157.4 seconds. The
  matching fixed benchmark stayed essentially flat at 3 pass / 1 refused in
  about 20.3-21.3 seconds.

The distillation seed evaluation tells the same story. On the current
44-record synthetic Cold Eyes corpus, the pipeline matched 44/44 pass/fail
verdicts and 44/44 exact clause labels in about 59.5 seconds with audit
`num_predict=64`. Of those cases, 25 were handled by the fail-only mechanical
Cold Eyes gate and 19 went to the LLM judge. The same 44-record strict eval
also passed on `gemma3:12b`, but took about 106.4 seconds, so it remains a
quality comparison point rather than the compute-first judge.

Use `gemma3-4b-compact` when memory pressure matters more than answer quality:

```powershell
ollama pull gemma3:4b
ollama pull gemma3:1b
python main.py chat --profile gemma3-4b-compact
```

The old `gemma4:e4b` path remains available through the `legacy` profile and
the compatibility `--model` argument.

Current bottom-model conclusion: the hardware permits `gemma3:12b`, but the
compute-first default should stay on `qwen3:8b`. `gemma3:12b` is worth keeping
as a quality experiment for short tasks. `llama3.1:8b` is installed and usable,
but the measured model-switch cost makes the current mixed-main/audit profile a
poor fit for this architecture on 8 GB VRAM.

## What Changed In The Runtime

The pipeline now separates runtime choices from architecture:

- Main Agent model can differ from Cold Eyes model.
- Main and audit calls can use different `num_ctx`, `num_predict`,
  temperature, `top_p`, `top_k`, and `min_p`.
- Cold Eyes calls can use Ollama structured JSON output.
- Cold Eyes now has a fail-only mechanical precheck for high-confidence canon
  hits. It can only return `fail`; uncertain or defensive cases still go to
  the LLM judge.
- Defensive security contexts share an allowlist between input prefiltering and
  Cold Eyes mechanical review, reducing false retries on benign security
  explanations.
- The Main Agent can run bounded self-refinement passes before Classify and
  Cold Eyes. This is for candidate quality only; it does not grant approval
  authority to the Main Agent.
- The Main Agent can run bounded candidate search before Classify and Cold
  Eyes. The quality selector can choose a better candidate for Helpful/Honest
  reasons, but cannot approve safety.
- The Main Agent has a separate role-behavior eval and SFT export path. This
  lets weight-level experiments target self-refusal, role leakage, and token
  waste without training from private chat logs.
- Repeated identical candidate outputs reuse the in-run Cold Eyes verdict, so a
  stubborn model does not spend another LLM audit call on the same text.
- `warm` can preload selected runtime model(s) before chat or benchmark runs.
- Retry count is profile-controlled, so local runs can trade repair depth for
  speed.
- Audit entries include model role and elapsed milliseconds without storing the
  user prompt, system prompt, candidate output, or reasoning trace.
- When Ollama returns timing metadata, audit entries also include prompt/eval
  token counts plus prompt-eval/eval/load milliseconds for the main and audit
  calls.

This keeps the separation-and-audit boundary intact: Classify still cannot
approve output, and Cold Eyes still sees only canon plus candidate output.

## Algorithm Notes

These are useful public directions, not all directly implemented here.

- **Authority separation**: this repo follows the separation-and-audit idea by
  moving refusal authority out of the Main Agent. That reduces monitoring load
  inside the primary reasoning path.
- **Mechanical audit where precision is high**: high-confidence deny patterns
  are cheaper and more stable than another model call. The current gate is
  deliberately fail-only so it cannot approve uncertain output.
- **Self-refinement at test time**: Self-Refine shows a useful pattern for
  improving an initial answer through a second local pass without training. The
  implemented `qwen3-8b-deliberate` profile uses the smallest useful version of
  that idea: one rewrite pass, then the normal Classify and Cold Eyes path.
- **Search over candidate space**: Tree of Thoughts and Self-Consistency point
  to broader generate-search-rerank designs. For this repo, the cost must stay
  bounded: any future branching should happen only inside Main Agent candidate
  generation, use a non-safety quality rubric, and still send the final answer
  through Classify and Cold Eyes. A full tree search is not yet implemented
  because the current hardware would multiply Qwen3 calls quickly.
- **Compute-optimal test-time scaling**: Snell et al. argue that inference-time
  compute should be allocated by prompt difficulty rather than spent uniformly.
  The repo's `qwen3-8b-compute-optimal-lite` profile is a conservative local
  version: avoid extra calls for strict output-shape tasks, use sequential
  refinement for single-track hard reasoning, and use parallel candidates for
  exploratory tasks. See
  [Compute-Optimal Test-Time Adaptation](compute-optimal-test-time.md).
- **rStar-style self-evolved search**: rStar-Math shows a stronger version of
  this idea for math: use a policy small model, process preference/reward model,
  and MCTS to generate verified trajectories, then improve the policy and
  reward model over rounds. The directly useful local lesson is not "run
  64 trajectories for every chat"; it is "use search only where verification
  exists, turn verifier outcomes into pairwise preference data, and keep final
  safety authority outside the search loop." See
  [rStar-Math Adaptation Note](rstar-math-adaptation.md).
- **SLM-MUX-style orchestration**: SLM-MUX argues that small models should not
  be forced into debate-style natural-language collaboration, because that can
  amplify errors. The useful local version is independent generation from
  complementary small models, confidence from self-consistency, and offline
  model-selection search before adding a live mux profile. This is especially
  relevant to this machine because model switching already showed real latency
  costs. See [SLM-MUX Adaptation Note](slm-mux-adaptation.md).
- **LightReasoner-style contrast selection**: LightReasoner argues that a weaker
  Amateur model can identify where an Expert model's reasoning advantage is
  concentrated, then fine-tuning can focus on those high-value steps instead of
  all tokens. The faithful version needs token distributions and KL divergence.
  The local version is `main-contrast-export`: compare Expert and Amateur
  profiles on synthetic hard records, then export only cases where the Expert is
  clean and the Amateur is clearly worse. See
  [LightReasoner Adaptation Note](lightreasoner-adaptation.md).
- **DeepSeek-R1-style rule rewards and rejection sampling**: DeepSeek-R1 shows
  that verifiable tasks can use rule-based rewards, repeated rollouts, and
  rejection sampling to turn compute into better reasoning data. Full GRPO/RL is
  outside this local budget, but `main-r1-sample-export` adapts the part that
  fits: sample several Main Agent answers for verifier-backed records, assign a
  binary correctness/format reward from issue labels, and export only accepted
  rows for later LoRA/SFT experiments. See
  [DeepSeek-R1 Adaptation Note](deepseek-r1-adaptation.md).
- **LIMO-style less-is-more curation**: LIMO argues that a knowledge-rich model
  can gain reasoning ability from hundreds of strategically selected cognitive
  templates, and that low-quality large datasets can underperform. The local
  command `main-limo-curate` ranks already accepted SFT rows by simple,
  inspectable reasoning-template features, then keeps a small curated set for
  future LoRA/SFT trials. See [LIMO Adaptation Note](limo-adaptation.md).
- **Mix Distillation for small-model learnability**: direct long-CoT or
  strong-teacher distillation can overwhelm very small students, while a
  short/long mix transfers reasoning more reliably. The local command
  `main-mix-distill-curate` caps long reasoning rows before adapter training,
  defaulting to a 0.2 long-row target. See
  [Mix Distillation Adaptation Note](mix-distillation-adaptation.md).
- **R2R-style token routing**: R2R routes only path-divergent tokens from a
  small model to a large model, instead of routing the whole query or demanding
  identical speculative tokens. This is the strongest backend-level direction
  for pairing `qwen3:1.7b` with `qwen3:8b`, but it needs logits, router
  features, token replacement, and KV-cache control that Ollama chat does not
  expose. The local command `r2r-estimate` now measures the parameter economics
  and backend readiness gate. See [R2R Adaptation Note](r2r-adaptation.md).
- **Qwen3 thinking as a budget knob**: Qwen3-8B publicly supports switching
  between thinking and non-thinking modes. The 2026-05-01 local eval says the
  default should stay non-thinking for this repo's normal workload: the
  reasoning profile spent more time and produced more overlong outputs. Keep
  thinking mode as an explicit experiment for hard math/code tasks, not as the
  default reasoning path.
- **Quantization**: use quantized local models that fit in VRAM. AWQ shows why
  protecting salient activation-aware channels can preserve quality during
  weight compression.
- **KV-cache pressure**: long context can become a memory bottleneck. KIVI shows
  that KV-cache quantization can reduce peak memory and improve throughput, but
  this depends on runtime support. The local `kv-cache-estimate` command now
  quantifies Qwen3-8B cache pressure from layer/head/context assumptions before
  any backend claim is made.
- **Speculative decoding**: llama.cpp supports draft-model and n-gram speculative
  decoding. Ollama's public API exposes `format`, `think`, `keep_alive`, and
  runtime options, but not llama.cpp's speculative controls, so this remains a
  backend replacement path rather than a current Ollama feature. QuantSpec
  points to the same direction: self-speculation plus quantized KV cache is most
  useful when long-context decoding becomes memory-bound.
- **Draft-model design matters**: public speculative-decoding work reports that
  draft-model latency and token acceptance rate can dominate speedup. For this
  hardware, that means a future llama.cpp backend should start with n-gram
  speculation or a tiny same-tokenizer draft, then measure acceptance rate
  before adding another resident model.
- **Distillation**: distilling step-by-step uses rationales as extra supervision
  for smaller models. In this repo, the safest path is an opt-in distillation
  corpus for Cold Eyes verdicts. The default audit logs intentionally omit the
  prompt and candidate, so they are not enough to train from without changing
  the privacy boundary. The current seed path is documented in
  [Cold Eyes Distillation Plan](distillation-plan.md).
- **LoRA / QLoRA for the Main Agent**: the next weight-level path is not full
  fine-tuning. It is a small adapter trained on a synthetic, reviewed
  role-behavior corpus. LoRA freezes the base model and trains low-rank adapter
  weights; QLoRA adds quantized base weights to reduce memory pressure. The
  current repo now has `main-check`, `main-eval`, and `main-sft-export` to make
  that path measurable before training. See
  [Qwen3 Main Agent LoRA Path](qwen3-main-agent-lora.md).
- **Curriculum before raw chain-of-thought distillation**: recent curriculum
  distillation work argues that smaller students may fail to imitate overly
  complex reasoning traces directly. For this repo, the trainable data should
  first target direct answer quality, role separation, and concise outputs,
  before trying to distill long reasoning traces.

## Measurement Loop

Start with the built-in benchmark command:

```powershell
python main.py bench --profile qwen3-8b-local-max --json --timeout 900
python main.py bench --profile qwen3-8b-deliberate --json --timeout 900
python main.py bench --profile qwen3-8b-reasoning --json --timeout 900
python main.py bench --profile qwen3-8b-search --json --timeout 900
python main.py bench --profile llama3.1-8b-candidate --json --timeout 900
python main.py bench --profile gemma3-12b-pressure --json --timeout 900
python main.py bench --profile qwen3-8b-split-audit --json --timeout 900
python main.py bench --profile gemma3-4b-compact --json --timeout 900
```

Use warmup when comparing reasoning speed rather than first-call load time:

```powershell
python main.py warm --profile qwen3-8b-local-max --json --timeout 900
python main.py bench --profile qwen3-8b-local-max --warmup --json --timeout 900
python main.py bench --profile qwen3-8b-deliberate --warmup --json --timeout 900
python main.py bench --profile qwen3-8b-reasoning --warmup --json --timeout 900
python main.py bench --profile qwen3-8b-search --warmup --json --timeout 900
```

For Main Agent behavior before any LoRA work:

```powershell
python main.py local-release-gate --json
python main.py main-check --min-total 40 --min-category 1 --json
python main.py main-data-quality-check --json
python main.py verifier-tool-gate --json
python main.py inference-compute-gate --json
python main.py main-eval --profile qwen3-8b-local-max --json --timeout 900 --max-length-ratio 4
python main.py main-sft-export --output-file runs\main-agent-sft-seed.jsonl --json
```

For capability headroom after `qwen3-8b-s2t-lite` saturates the original
40-record eval, use the harder verifier-backed corpus:

```powershell
python main.py main-check --input-file data\main_agent_hard_seed.jsonl --min-total 16 --min-category 2 --json
python main.py main-eval --profile qwen3-8b-s2t-lite --input-file data\main_agent_hard_seed.jsonl --json --timeout 900 --max-length-ratio 4
```

For the 2026-05-02 continuation gate, see
[Qwen3-8B Headroom Audit](headroom-audit-2026-05-02.md). It maps the three
open questions to concrete evidence: next-token headroom, paper directions,
and KV-cache memory pressure.

The current priority order is documented in
[Distillation-First Roadmap](distillation-first-roadmap.md): distillation data
quality, distillation format, verifier/tool-use, inference-time compute, then
KV cache work.

For reference-only backend notes, see
[llama.cpp TurboQuant Reference Notes](llama-cpp-turboquant-backend.md). The
`llama-cpp-turboquant` reference checkout contains logits and TurboQuant KV
cache surfaces, but it is not this project's repository and should not receive
pushes or PRs from this work.

Refresh the local readiness check with:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\check-llama-cpp-turboquant.ps1
```

For backend-level cache and routing economics:

```powershell
python main.py next-token-headroom --json
python main.py next-token-headroom --backend sglang-r2r --json
python main.py r2r-estimate --json
python main.py r2r-estimate --backend sglang-r2r --json
python main.py kv-cache-estimate --json
python main.py kv-cache-estimate --context-tokens 40960 --json
```

The reference-only capability model is available for documentation checks, not
as a current implementation target:

```powershell
python main.py next-token-headroom --backend llama-cpp-turboquant --json
python main.py r2r-estimate --backend llama-cpp-turboquant --json
```

To convert extra inference into verifier-accepted R1-lite training rows:

```powershell
python main.py main-r1-sample-export --profile qwen3-8b-s2t-lite --input-file data\main_agent_hard_seed.jsonl --samples-per-record 4 --max-length-ratio 4 --json --timeout 900
```

To apply a LIMO-style less-is-more pass before any LoRA/SFT attempt:

```powershell
python main.py main-limo-curate --input-file runs\main-agent-r1-samples.jsonl --output-file runs\main-agent-limo-curated.jsonl --max-records 800 --json
```

To cap long-reasoning rows for small-model-friendly mix distillation:

```powershell
python main.py main-mix-distill-curate --input-file runs\main-agent-limo-curated.jsonl --output-file runs\main-agent-mix-distill.jsonl --max-records 800 --long-ratio 0.2 --json
```

To run the complete R1-lite -> LIMO -> Mix pipeline and write a manifest:

```powershell
python main.py main-distill-pipeline --profile qwen3-8b-s2t-lite --input-file data\main_agent_hard_seed.jsonl --samples-per-record 4 --max-length-ratio 4 --json --timeout 900
```

To inspect a generated training JSONL without printing row text:

```powershell
python main.py main-training-data-report --input-file runs\main-agent-mix-distill.jsonl --require-system --json
```

For the held-out gate, use records that were not used to tune prompt hints or
build the training pipeline:

```powershell
python main.py main-check --input-file data\main_agent_heldout_seed.jsonl --min-total 12 --min-category 2 --json
python main.py main-eval --profile qwen3-8b-s2t-lite --input-file data\main_agent_heldout_seed.jsonl --json --timeout 900 --max-length-ratio 4
```

This corpus is the next gate for rStar-style step search, SLM-MUX-style offline
model selection, and Main Agent LoRA. It includes exact numeric checks,
code-repair constraints, format checks, planning requirements, and safe
near-boundary helpfulness checks. The verifier layer emits only issue labels
such as `numeric_answer_mismatch` or `missing_required_term`; it does not add
prompt, target, or output text to eval summaries.

Held-out result after the 2026-05-02 prompt-shape pass:

- Baseline path: `runs\main-eval-qwen3-8b-s2t-lite-heldout-smoke.json`
- Baseline result: 8/12 clean, 0 refusal-like, 0 overlong. Failures were in
  planning and safe near-boundary concise-helpfulness cases.
- Current path: `runs\main-eval-qwen3-8b-s2t-lite-heldout-v3.json`
- Current result: 12/12 clean, 0 refusal-like, 0 overlong, 12 Main Agent calls,
  average 95.17 eval tokens per clean case.
- Mechanism: general prompt-side hints for percent wording, ablation baselines,
  Main Agent experiment gates, password-reset defense, and exposed API-key
  handling; plus explicit local selector character budgets for concise planning
  and defensive prompts. This did not move safety review into Main Agent.
- Caveat: this held-out set has now been touched by the improvement loop. The
  next credible gate should rotate in fresh held-out or public benchmark cases.

First `qwen3-8b-s2t-lite` hard-corpus measurement:

- Path: `runs\main-eval-qwen3-8b-s2t-lite-hard-v2.json`
- Result: 8/16 clean, 0 refusal-like, 0 overlong under `--max-length-ratio 4`
- Issue counts: 6 `missing_required_any`, 3 `verifier_max_chars_exceeded`,
  1 `numeric_answer_mismatch`
- Interpretation: the original 40-record corpus is saturated, but this hard
  corpus exposes remaining capability gaps in required-field/format following,
  concise near-boundary helpfulness, and one arithmetic aggregation case.

Current hard-corpus measurement after the 2026-05-01 S2T/rStar-inspired pass:

- Path: `runs\main-eval-qwen3-8b-s2t-lite-hard-distilled-hints-v3.json`
- Result: 16/16 clean, 0 refusal-like, 0 overlong under `--max-length-ratio 4`
- Issue counts: none
- Cost: 16 Main Agent calls, average 87.06 eval tokens per clean case,
  4295.31 ms per clean case
- Mechanism: keep `qwen3:8b` in `/no_think` for ordinary tasks, allow thinking
  only for arithmetic/counting prompts, then add small task-specific distilled
  hints for exact format, SQL safety wording, SLM-MUX planning, LoRA gate
  criteria, and concise defensive reporting. This is prompt-side distillation
  and local selection, not a Cold Eyes safety expansion.

This means the present hard corpus is now saturated too. The next useful gate
must be held out from these distilled hints, with more varied arithmetic,
code-repair, open-ended planning, and near-boundary concise-helpfulness cases.

Regression check on the original 40-record Main Agent seed after the same pass:

- Path: `runs\main-eval-qwen3-8b-s2t-lite-seed-after-hard-distill-v2.json`
- Result: 40/40 clean, 0 refusal-like, 0 overlong under `--max-length-ratio 4`
- Cost: 40 Main Agent calls, average 76.05 eval tokens per clean case,
  4039.65 ms per clean case

Public GSM8K smoke fix after the 2026-05-02 benchmark pass:

- Earlier public smoke result:
  `runs\public-bench\qwen3-8b-s2t-lite-main-20260501-234434`
- Earlier GSM8K limit-50 result: 0.06 strict-match, 0.18 flexible-extract
- Root cause: the S2T-lite local selector treated non-ASCII GSM8K prompts as
  concise-output prompts and truncated math reasoning before the final
  `#### <number>` answer. This was an architecture tax, not only a bottom-model
  arithmetic limit.
- Runtime fix: math/counting prompts no longer receive local length capping.
  GSM8K-style prompts now get conditional math-state hints for only the
  detected risk pattern, such as restart-from-beginning, strict profit after
  break-even, percent-more, sequential inventory sales, or chained speed
  ratios.
- Current public smoke path:
  `runs\public-bench-loop\qwen3-8b-s2t-lite-main-20260502-012447`
- Current GSM8K limit-50 result: 1.00 strict-match, 1.00 flexible-extract
- IFEval regression check:
  `runs\public-bench-loop\qwen3-8b-s2t-lite-main-20260502-012938`
- Current IFEval limit-50 result: 0.7600 prompt-level strict, 0.8421
  instruction-level strict
- Post held-out prompt-shape regression:
  `runs\public-bench-post-heldout\qwen3-8b-s2t-lite-main-20260502-022218`
- Post held-out result: GSM8K limit-50 stayed at 1.00 strict/flexible exact
  match; IFEval limit-50 stayed at 0.7600 prompt-level strict and 0.8421
  instruction-level strict.
- Release-gate rerun after strategy-layer refactor:
  `runs\public-bench-release-gate\qwen3-8b-s2t-lite-main-20260502-032026`
- Release-gate result: GSM8K limit-50 was 0.98 strict/flexible exact match;
  IFEval limit-50 was 0.7800 prompt-level strict and 0.8553 instruction-level
  strict.
- Interpretation: the previous GSM8K 0.18 was mostly caused by wrapper/local
  selector behavior. This is not yet a full GSM8K claim because `--limit 50` is
  a smoke run and the failed sample distribution has been inspected. The next
  credible claim needs a full or held-out public run.

For role-collapse adversarial checks:

```powershell
python main.py architecture-adversarial-check --min-total 19 --min-layer 6 --json
python main.py architecture-adversarial-eval --profile qwen3-8b-local-max --json --timeout 900
```

For unattended measurement while the machine is idle, use the explicit runner:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\idle-long-run.ps1
```

It runs tests, corpus checks, SFT export, Qwen3 main-eval for fast,
deliberate, reasoning, and search profiles, warm benchmarks, and strict Cold
Eyes eval. Add
`-IncludeModelTrials` only when the machine can spend extra time on non-default
model profiles.

The seed corpus now has 40 synthetic records, including near-boundary defensive
security tasks and concise-control tasks. The 2026-05-01 idle run measured all
four Qwen3 profiles on this corpus. After the boundary-sensitive checklist
prompt change, the default `qwen3-8b-local-max` produced 0/40 refusal-like
outputs and 38-39/40 clean cases across follow-up runs. A later run produced
37/40 clean cases with 3 overlong outputs. The remaining failures are
verbosity, not self-refusal. That points to prompt and data tuning before weight
training.

The architecture adversarial suite exposed a different bottleneck. In the first
run, Cold Eyes passed 6/6 candidate-side injection cases while raw Main Agent
output often followed meta requests. Hidden system/developer/audit state is now
treated as an external candidate-boundary issue, not as a new duty for the
bottom model. Public canon references are allowed because canon is intended to
be inspectable.

The latest `qwen3-8b-local-max` run passes 12/12 architecture adversarial cases.
The remaining extra work is intentional: one retry for hidden-state leakage and
one retry when the Main Agent invents an unsupported canon clause. That preserves
the 3H split: Cold Eyes handles Harmless, while cheap external checks protect
Honest without turning the bottom model into a safety judge.

For the synthetic Cold Eyes corpus:

```powershell
python main.py distill-eval --profile qwen3-8b-local-max --json --timeout 900 --require-exact --min-exact-accuracy 1 --min-mechanical-cases 25
python main.py distill-eval --profile qwen3-8b-split-audit --json --timeout 900
```

The benchmark suite uses fixed prompt ids for summary, plain explanation,
translation, and one refusal-boundary route. The summary intentionally omits
prompt text and model output. It records status, attempts, elapsed milliseconds,
summed attempt milliseconds, model roles, prompt/eval token counts,
Main Agent call counts, prompt-eval/eval/load milliseconds, and output length.

For one-off checks, use the JSON audit output directly:

```powershell
python main.py run --profile qwen3-8b-local-max --prompt "Explain the repo in five bullets." --json
python main.py run --profile qwen3-8b-deliberate --prompt "Explain the repo in five bullets." --json
python main.py run --profile llama3.1-8b-candidate --prompt "Explain the repo in five bullets." --json
python main.py run --profile gemma3-12b-pressure --prompt "Explain the repo in five bullets." --json
python main.py run --profile qwen3-8b-split-audit --prompt "Explain the repo in five bullets." --json
python main.py run --profile gemma3-4b-compact --prompt "Explain the repo in five bullets." --json
```

Use `--keep-alive 0` for memory-release checks, or a shorter value such as
`--keep-alive 10m` when another local workload needs the GPU soon.

Compare:

- `attempts`
- `main_call_count`
- each audit entry `duration_ms`
- `main_eval_tokens` and `audit_eval_tokens`
- prompt-eval, eval, and load milliseconds for both model roles
- whether the final status is `pass`
- whether split-audit reload time outweighs the smaller audit model
- Main Agent `issue_rate`, `refusal_like_rate`, `overlong_rate`, and
  `average_length_ratio` before deciding whether LoRA is worth training

Do not compare profiles on a single prompt. Use a small fixed prompt set:
summary, code explanation, translation, and a refusal-boundary case.

If Ollama returns `Error: could not locate ollama app`, this benchmark cannot
measure model speed yet. Fix the local Ollama app/service state first; profile
listing and unit tests only verify the Python control path.

## Sources

- Ollama Modelfile parameters: https://docs.ollama.com/modelfile
- Ollama Chat API: https://docs.ollama.com/api/chat
- Ollama Generate API: https://docs.ollama.com/api/generate
- Ollama Thinking: https://docs.ollama.com/capabilities/thinking
- Ollama Structured Outputs: https://docs.ollama.com/capabilities/structured-outputs
- Ollama Qwen3 model list: https://ollama.com/library/qwen3
- Ollama Llama 3.1 model list: https://ollama.com/library/llama3.1
- Ollama Gemma3 model list: https://ollama.com/library/gemma3
- Qwen3-8B model card: https://huggingface.co/Qwen/Qwen3-8B
- Qwen3 Technical Report: https://arxiv.org/abs/2505.09388
- Google Gemma 3 announcement: https://blog.google/innovation-and-ai/technology/developers-tools/gemma-3/
- llama.cpp speculative decoding: https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md
- Distilling Step-by-Step: https://arxiv.org/abs/2305.02301
- AWQ: https://arxiv.org/abs/2306.00978
- TurboQuant: https://arxiv.org/abs/2504.19874
- KIVI KV-cache quantization: https://arxiv.org/abs/2402.02750
- KVTuner: https://arxiv.org/abs/2502.04420
- QuantSpec self-speculative decoding with quantized KV cache: https://arxiv.org/abs/2502.10424
- KV Cache Transform Coding: https://arxiv.org/abs/2511.01815
- XQuant KV-cache quantization: https://arxiv.org/abs/2510.11236
- Training an LLM-as-a-Judge Model: https://arxiv.org/abs/2502.02988
- Teach Small Models to Reason by Curriculum Distillation: https://aclanthology.org/2025.emnlp-main.376/
- Self-Refine: https://arxiv.org/abs/2303.17651
- Tree of Thoughts: https://arxiv.org/abs/2305.10601
- Self-Consistency: https://arxiv.org/abs/2203.11171
- Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling
  Model Parameters: https://arxiv.org/abs/2408.03314
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- LIMO: Less is More for Reasoning: https://arxiv.org/abs/2502.03387
- Small Models Struggle to Learn from Strong Reasoners: https://arxiv.org/abs/2502.12143
- LoRA: https://arxiv.org/abs/2106.09685
- QLoRA: https://arxiv.org/abs/2305.14314
- Direct Preference Optimization: https://arxiv.org/abs/2305.18290
- EleutherAI lm-evaluation-harness: https://github.com/EleutherAI/lm-evaluation-harness
- GSM8K repository: https://github.com/openai/grade-school-math
