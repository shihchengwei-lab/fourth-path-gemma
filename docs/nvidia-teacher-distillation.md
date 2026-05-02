# NVIDIA Teacher Distillation

This path uses NVIDIA-hosted OpenAI-compatible chat endpoints as external
teachers, then keeps only rows that pass the local Main Agent verifiers.

It is for urgent data generation while free or expiring endpoints are available.
It is not a runtime dependency for the local pipeline.

## Secret Handling

Set the API key only in the local shell environment. Do not paste it into chat,
write it into a repo file, or commit it.

```powershell
$env:NVIDIA_API_KEY = "<set locally>"
$env:NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
```

`NVIDIA_BASE_URL` is optional. The default is
`https://integrate.api.nvidia.com/v1`.

## Fast Distill Command

The default model order prioritizes the short-window and useful current
endpoints:

1. `deepseek-ai/deepseek-v3.2`
2. `minimaxai/minimax-m2.7`
3. `nvidia/nemotron-3-super-120b-a12b`
4. `openai/gpt-oss-120b`
5. `qwen/qwen3-next-80b-a3b-instruct`

Run a small urgent batch first:

```powershell
python main.py main-nvidia-teacher-export `
  --input-file data\main_agent_hard_seed.jsonl `
  --output-file runs\main-agent-nvidia-teacher.jsonl `
  --limit-records 3 `
  --samples-per-model 1 `
  --json `
  --timeout 1200
```

If one model endpoint is unavailable, the command records
`teacher_request_failed` and continues to the next model by default. Add
`--stop-on-error` when debugging a single endpoint.

Override the model list when the catalog changes:

```powershell
python main.py main-nvidia-teacher-export `
  --model minimaxai/minimax-m2.7 `
  --model nvidia/nemotron-3-super-120b-a12b `
  --limit-records 5 `
  --json
```

## Output Contract

The export writes accepted SFT-style rows to `runs/`, which is git-ignored.
Rows include:

- `messages`: Main Agent system prompt, user prompt, accepted assistant answer;
- `source`: `nvidia_teacher_synthetic`;
- `split`: `train_candidate`;
- `verifier_labels`: local verifier metadata plus the teacher provider/model;
- `teacher_provider`, `teacher_model`, `sample_index`, and `reward`.

The command summary reports counts, issue labels, token totals, and model
acceptance counts. It does not print prompts, targets, or generated answers.

Validate the generated file before using it for any training step:

```powershell
python main.py main-training-data-report `
  --input-file runs\main-agent-nvidia-teacher.jsonl `
  --require-system `
  --require-generated-metadata `
  --json
```

## NVIDIA API Basis

The client sends `POST /v1/chat/completions` requests to the configured base
URL with bearer auth. NVIDIA NIM LLM documentation describes these endpoints as
OpenAI-compatible, and the NVIDIA catalog currently lists
`minimaxai/minimax-m2.7` as the MiniMax M2.7 endpoint.

References:

- https://docs.api.nvidia.com/nim/reference/llm-apis
- https://docs.nvidia.com/nim/large-language-models/2.0.3/reference/api-reference.html
- https://build.nvidia.com/minimaxai/minimax-m2.7
