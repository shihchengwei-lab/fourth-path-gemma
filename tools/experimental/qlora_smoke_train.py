from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


DEFAULT_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number}: row must be a JSON object")
            messages = row.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError(f"line {line_number}: row must contain messages")
            rows.append(row)
    if not rows:
        raise ValueError(f"no rows found in {path}")
    return rows


def encode_row(
    tokenizer: Any,
    row: dict[str, Any],
    max_length: int,
) -> dict[str, torch.Tensor]:
    messages = row["messages"]
    if len(messages) < 2 or messages[-1].get("role") != "assistant":
        raise ValueError(f"{row.get('id', '<unknown>')}: last message must be assistant")

    prompt_messages = messages[:-1]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    if tokenizer.eos_token_id is not None and (not full_ids or full_ids[-1] != tokenizer.eos_token_id):
        full_ids = [*full_ids, tokenizer.eos_token_id]

    input_ids = full_ids[:max_length]
    labels = input_ids.copy()
    prompt_len = min(len(prompt_ids), len(labels))
    labels[:prompt_len] = [-100] * prompt_len
    if all(label == -100 for label in labels):
        raise ValueError(f"{row.get('id', '<unknown>')}: assistant target was truncated away")

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
    }


class ChatSftDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, tokenizer: Any, rows: list[dict[str, Any]], max_length: int) -> None:
        self.examples = [encode_row(tokenizer, row, max_length) for row in rows]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.examples[index]


class DataCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        max_len = max(item["input_ids"].shape[0] for item in batch)
        input_ids: list[torch.Tensor] = []
        attention_masks: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []
        for item in batch:
            pad_len = max_len - item["input_ids"].shape[0]
            input_ids.append(torch.nn.functional.pad(item["input_ids"], (0, pad_len), value=self.pad_token_id))
            attention_masks.append(torch.nn.functional.pad(item["attention_mask"], (0, pad_len), value=0))
            labels.append(torch.nn.functional.pad(item["labels"], (0, pad_len), value=-100))
        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_masks),
            "labels": torch.stack(labels),
        }


def trainable_parameter_report(model: torch.nn.Module) -> dict[str, int | float]:
    trainable = 0
    total = 0
    for parameter in model.parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
    return {
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_ratio": trainable / total if total else 0.0,
    }


def write_manifest(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tiny QLoRA smoke trainer for Main Agent SFT rows.")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--resume-adapter", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--manifest", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.time()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    input_path = Path(args.input_file)
    output_dir = Path(args.output_dir)
    manifest_path = Path(args.manifest) if args.manifest else output_dir / "smoke-train-manifest.json"

    rows = load_rows(input_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError("tokenizer must have a pad or eos token")

    dataset = ChatSftDataset(tokenizer, rows, args.max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=DataCollator(tokenizer.pad_token_id),
    )

    compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    quantization_config = None
    if not args.no_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=compute_dtype,
        quantization_config=quantization_config,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=args.trust_remote_code,
    )
    model.config.use_cache = False
    if quantization_config is not None:
        model = prepare_model_for_kbit_training(model)

    if args.resume_adapter:
        model = PeftModel.from_pretrained(model, args.resume_adapter, is_trainable=True)
    else:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=list(DEFAULT_TARGET_MODULES),
        )
        model = get_peft_model(model, lora_config)
    model.train()

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
    )

    total_micro_steps = math.ceil(len(dataset) / args.batch_size * args.epochs)
    if args.max_steps > 0:
        total_micro_steps = min(total_micro_steps, args.max_steps * args.grad_accum)
    total_micro_steps = max(total_micro_steps, 1)

    losses: list[float] = []
    optimizer.zero_grad(set_to_none=True)
    micro_step = 0
    optimizer_step = 0
    while micro_step < total_micro_steps:
        for batch in dataloader:
            micro_step += 1
            batch = {key: value.to(model.device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / args.grad_accum
            loss.backward()
            losses.append(float(outputs.loss.detach().cpu()))
            if micro_step % args.grad_accum == 0 or micro_step >= total_micro_steps:
                torch.nn.utils.clip_grad_norm_(
                    (parameter for parameter in model.parameters() if parameter.requires_grad),
                    1.0,
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
                print(
                    json.dumps(
                        {
                            "micro_step": micro_step,
                            "optimizer_step": optimizer_step,
                            "loss": round(losses[-1], 6),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            if micro_step >= total_micro_steps:
                break

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    manifest = {
        "model": args.model,
        "input_file": str(input_path),
        "output_dir": str(output_dir),
        "rows": len(rows),
        "max_length": args.max_length,
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "learning_rate": args.learning_rate,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "resume_adapter": args.resume_adapter,
        "use_4bit": not args.no_4bit,
        "compute_dtype": str(compute_dtype),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "cuda_available": torch.cuda.is_available(),
        "parameter_report": trainable_parameter_report(model),
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "loss_min": min(losses) if losses else None,
        "loss_max": max(losses) if losses else None,
        "micro_steps": micro_step,
        "optimizer_steps": optimizer_step,
        "duration_seconds": round(time.time() - started, 3),
    }
    write_manifest(manifest_path, manifest)
    print(json.dumps({"manifest": str(manifest_path), "output_dir": str(output_dir)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
