# llama.cpp TurboQuant Reference Notes

This is reference material only. `C:\Users\kk789\Desktop\llama-cpp-turboquant`
is not this project's repository. Do not push to it, open PRs from this work, or
create maintainer burden there.

The previous audit proved that Ollama chat is not enough for faithful
next-token experiments. It did not prove that no local backend path exists.
This fork is useful evidence for what a logits/KV-capable runtime can expose,
but it is not an owned dependency or the current implementation priority.

Candidate backend:

```text
C:\Users\kk789\Desktop\llama-cpp-turboquant
branch: feature/turboquant-kv-cache
observed head: e0954d1b9
```

## Reference Surfaces

The local reference checkout has surfaces that are relevant to a future
next-token and KV-cache trial:

- `include/llama.h` exposes `llama_batch.logits`,
  `llama_get_logits()`, `llama_get_logits_ith()`, and backend-sampled logits.
- `llama_context_params` includes `type_k` and `type_v` for KV cache data
  types.
- `common/arg.cpp` accepts `-ctk/--cache-type-k` and `-ctv/--cache-type-v`.
  The allowed KV cache types include `f32`, `f16`, `bf16`, `q8_0`, `q4_0`,
  `q4_1`, `iq4_nl`, `q5_0`, `q5_1`, `turbo2`, `turbo3`, and `turbo4`.
- `examples/debug` supports `--save-logits`, which is useful for output-side
  logit verification.
- `src/llama-kv-cache.cpp` has TurboQuant-specific behavior, including
  auto-asymmetric K handling for high-GQA models and layer-adaptive V modes.
- `src/llama-context.cpp` auto-enables flash attention when TurboQuant cache
  types are requested.
- `tests/test-turbo-quant.c` checks TurboQuant round trips.

The existing Ollama `qwen3:8b` blob at
`E:\ollama-models\blobs\sha256-a3de86cd1c132c822487ededd47a324c50491393e6565cd14bafa40d0b8e686f`
starts with `GGUF`, so it can be the first model file to try once the backend is
built.

To refresh this local evidence:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\check-llama-cpp-turboquant.ps1
```

## Current Local Status

The fork is not currently runnable on this Windows shell, and this repo should
not spend current effort on changing that:

- no `llama-cli.exe`, `llama-server.exe`, or `llama-bench.exe` found under the
  checkout;
- `cmake`, `ninja`, `nvcc`, and `cl` are not on the Windows PATH;
- WSL can see the NVIDIA GPU, but WSL did not expose the needed compiler chain
  in PATH, and WSL `git status` reports many false dirty files on this Windows
  checkout. Do not edit this checkout from WSL unless a clean WSL-native clone
  is made.

So the correct status is:

```text
KV cache path: reference implementation exists, deferred.
Next-token logits path: reference C API exists, not wired into this repo.
R2R path: still missing router, residency proof, and project-owned integration.
```

## Deferred Test Shape

Do not run this before the higher-priority distillation and verifier gates are
stable. If revisited later, build in a clean private scratch area or project-owned
workspace, not by pushing to the reference repo.

Possible future build shape:

```powershell
$source = "E:\scratch\llama-cpp-turboquant-reference"
$build = "E:\scratch\llama-cpp-turboquant-build"
cmake -S $source -B $build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build $build --config Release -j
```

Possible future benchmark shape:

```powershell
$model = "E:\ollama-models\blobs\sha256-a3de86cd1c132c822487ededd47a324c50491393e6565cd14bafa40d0b8e686f"
$bench = "E:\scratch\llama-cpp-turboquant-build\bin\llama-bench.exe"
& $bench -m $model -ngl 99 -p 512 -n 128 -r 3 -ctk f16 -ctv f16 -fa on
& $bench -m $model -ngl 99 -p 512 -n 128 -r 3 -ctk q8_0 -ctv q8_0 -fa on
& $bench -m $model -ngl 99 -p 512 -n 128 -r 3 -ctk q8_0 -ctv turbo4 -fa on
& $bench -m $model -ngl 99 -p 512 -n 128 -r 3 -ctk q8_0 -ctv turbo2 -fa on
```

Use `q8_0` for K first on Qwen-style high-GQA models, because the fork itself
warns that symmetric TurboQuant K can degrade quality when many Q heads share
few KV heads.

Deferred acceptance gate:

1. `llama-bench` runs on `qwen3:8b` without falling back to CPU-only speed.
2. Quantized KV reduces memory or enables longer context without a large decode
   regression.
3. A held-out Main Agent eval and at least one public smoke do not regress.
4. Logits can be extracted for the last token, then compared across candidate
   next-token choices.

## CLI Capability Model

This repo recognizes the reference model only to prevent overclaiming:

```powershell
python main.py next-token-headroom --backend llama-cpp-turboquant --json
python main.py r2r-estimate --backend llama-cpp-turboquant --json
```

Interpretation:

- `next-token-headroom` reports reference primitives, not current project
  readiness.
- `r2r-estimate` does not treat it as full R2R-ready, because the trained router,
  memory-residency proof, and project-owned integration are still external work.

Current project priority remains:

1. distillation data quality
2. distillation format
3. verifier and tool-use
4. inference-time compute
5. KV cache changes

## Paper Pointers

- TurboQuant: https://arxiv.org/abs/2504.19874
- KIVI: https://arxiv.org/abs/2402.02750
- KVTuner: https://arxiv.org/abs/2502.04420
- KV Cache Transform Coding: https://arxiv.org/abs/2511.01815
- R2R token routing: https://arxiv.org/abs/2505.21600
- LightReasoner: https://arxiv.org/abs/2510.07962
