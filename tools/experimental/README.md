# Experimental Tools

This directory contains local research utilities for opt-in Main Agent
experiments. These scripts are not part of the default Fourth Path runtime, and
they do not change the audited pipeline by themselves.

- `qlora_smoke_train.py`: minimal local QLoRA trainer for SFT-style Main Agent
  rows.
- `qlora_adapter_eval.py`: local adapter evaluator against Main Agent verifier
  rows. It disables Qwen thinking by default so concise verifier checks measure
  candidate output rather than hidden reasoning traces.
- `adapter_containment_eval.py`: local base/adapter evaluator for Main Agent
  containment prompts. It reports candidate cleanliness separately from
  containment, where external gates must block role-authority candidate text
  before final output.
- `adapter_safety_eval.py`: backward-compatible entrypoint for the same
  containment evaluator.
- `merge_sft_jsonl.py`: validated SFT JSONL merger for experiment datasets.

Use these only for documented experiment lanes such as
`docs/main-agent-lora-experiment-2026-05-02.md`.
