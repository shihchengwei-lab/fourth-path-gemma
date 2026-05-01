# Qwen3-8B Headroom Audit, 2026-05-02

Question: under this local hardware budget, is there still a justified path to
make `qwen3:8b` more effective?

Answer: yes, but the first question must be read at the next-token level, not
as arithmetic. The base model's fixed weights are not made better by a prompt.
What can still improve is the conditioning, decoding, token-level backend, or
offline adapter that changes future next-token probabilities.

## 1. Can Next-Token Computation Improve?

Yes, with a boundary.

Current Ollama chat path:

- can shift the conditional distribution with prompt context, Qwen3 thinking
  mode, and decoding options;
- cannot expose true next-token logits or top-k probabilities;
- cannot replace one token and continue generation with an updated KV cache;
- therefore cannot run faithful token-level R2R, S2T, or KL-style selectors.

Local command:

```powershell
python main.py next-token-headroom --json
python main.py next-token-headroom --backend sglang-r2r --json
```

Interpretation:

- Under fixed `qwen3:8b` weights plus Ollama chat, true next-token computation
  is mostly a black box.
- A token-level backend opens real next-token experiments: inspect logits,
  compare candidate next tokens, route path-divergent tokens, and update the
  large model KV cache from mixed prefixes.
- Offline LoRA or adapter training is the direct way to change the learned
  next-token distribution, but only after held-out gates prove prompt/runtime
  changes are insufficient.

Local backend reality from this machine:

- `E:\Ollama\ollama.exe` is installed and has `qwen3:8b`, `qwen3:1.7b`,
  `llama3.1:8b`, `gemma3:12b`, and `gemma4:e4b`.
- `llama-server` and `llama-cli` were not found on PATH.
- `C:\Users\kk789\Desktop\llama-cpp-turboquant` exists on
  `feature/turboquant-kv-cache` and materially changes the next step. It exposes
  logits APIs, KV cache type controls, and TurboQuant cache types (`turbo2`,
  `turbo3`, `turbo4`), but it is not built locally yet.
- Python imports for `torch`, `transformers`, `sglang`, `llama_cpp`, and `vllm`
  were not installed in the active Python.
- `nvidia-smi` reported an RTX 4060 Laptop GPU with `8188 MiB` total VRAM at
  inspection time.
- The existing Ollama `qwen3:8b` blob is a GGUF file, so it can be used as the
  first llama.cpp test model after the backend is built.

So the immediate blocker for faithful next-token experiments is no longer
"unknown backend direction." It is build/runtime wiring for a concrete local
backend candidate. See [llama.cpp TurboQuant Backend Path](llama-cpp-turboquant-backend.md).

## 2. Are There More Papers Worth Using?

Yes.

Useful references:

- Qwen3 Technical Report: supports thinking and non-thinking modes in one model.
  This backs treating thinking as an opt-in next-token conditioning budget.
- LightReasoner: the faithful method needs token probability distributions and
  KL divergence. This confirms that the current repo's sample-level contrast is
  only a proxy until a logits-capable backend exists.
- R2R: token-level small/large routing targets path-divergent next tokens. This
  directly matches the corrected question, but it needs logits, token
  replacement, router features, and KV cache update.
- TurboQuant, KIVI, KVTuner, KVTC, and XQuant: KV-cache compression and
  self-speculative
  decoding are still active directions for making next-token generation cheaper
  or more scalable under long context.
- rStar-Math and DeepSeek-R1 remain useful only as verifier-backed search/data
  ideas; they are not the answer to "can the next-token calculation itself
  improve" unless their outputs are turned into adapter data or token-level
  search with backend support.

## 3. Does KV Cache Still Have Benefit?

Yes, especially for long context and repeated prefixes, but not through the
current Ollama chat controls.

Local estimate for Qwen3-8B assumptions from the model config
(`36` layers, `8` KV heads, `128` head dimension):

```powershell
python main.py kv-cache-estimate --json
python main.py kv-cache-estimate --context-tokens 40960 --json
```

Verified local estimates:

- `8192` context, fp16 KV: `1152.0 MiB`; 4-bit KV estimate: `288.0 MiB`.
- `40960` context, fp16 KV: `5760.0 MiB`; 4-bit KV estimate: `1440.0 MiB`.
- estimated KV memory reduction: `0.75`.

On an 8 GB laptop GPU, the long-context number is large enough that KV-cache
compression or shared-prefix reuse remains strategically relevant. This is a
backend-replacement path; the current Ollama chat API does not expose the
needed cache controls.

## Continue / Stop Decision

Continue.

At least one gate is clearly open:

1. `next-token-headroom` says Ollama chat is not token-level-ready, but a
   logits/token-routing backend would unlock real next-token experiments.
2. `kv-cache-estimate` shows long-context KV cache pressure is material on this
   hardware.
3. Current papers support backend and adapter directions that are not exhausted
   by the existing prompt/profile work.

The next concrete work should be backend evaluation, not another prompt-only
patch:

1. keep default chat on `qwen3-8b-s2t-lite`;
2. build or otherwise obtain `llama-cpp-turboquant` binaries, then test token
   probabilities, prefix cache, KV quantization, and speculative decoding;
3. accept it only if held-out repo checks and public smoke keep quality while
   latency or memory improves;
4. use LoRA/adapter only when held-out failures prove the fixed distribution
   needs a weight-level change.

## Sources

- Qwen3 Technical Report: https://arxiv.org/abs/2505.09388
- Qwen3-8B config: https://huggingface.co/Qwen/Qwen3-8B/blob/2069b3fae1114555f3c020c81410e51fa0f656f2/config.json
- R2R: https://arxiv.org/abs/2505.21600
- LightReasoner: https://arxiv.org/abs/2510.07962
- TurboQuant: https://arxiv.org/abs/2504.19874
- KIVI KV-cache quantization: https://arxiv.org/abs/2402.02750
- KVTuner: https://arxiv.org/abs/2502.04420
- QuantSpec: https://arxiv.org/abs/2502.10424
- KV Cache Transform Coding: https://arxiv.org/abs/2511.01815
- XQuant: https://arxiv.org/abs/2510.11236
