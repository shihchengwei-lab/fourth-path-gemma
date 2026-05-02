from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import main


def strip_qwen_think(text: str) -> str:
    return re.sub(r"^\s*<think>.*?</think>\s*", "", text, flags=re.S).strip()


def category_max_new_tokens(category: str) -> int:
    if "code_repair" in category:
        return 180
    return 240


def chat_template_text(tokenizer: Any, messages: list[dict[str, str]], enable_thinking: bool) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def run_eval(
    model_name: str,
    input_file: Path,
    output_file: Path,
    adapter_dir: Path | None = None,
    max_new_tokens: int | None = None,
    load_4bit: bool = True,
    enable_thinking: bool = False,
) -> dict[str, Any]:
    records, errors, _ = main.load_main_agent_records(input_file)
    if errors:
        raise ValueError("; ".join(errors))

    compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    quantization_config = None
    if load_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

    tokenizer_source = adapter_dir if adapter_dir else model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        torch_dtype=compute_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if adapter_dir:
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()

    results: list[dict[str, Any]] = []
    for record in records:
        messages = [
            {"role": "system", "content": main.MAIN_AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": record.prompt},
        ]
        prompt_text = chat_template_text(tokenizer, messages, enable_thinking=enable_thinking)
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        generation_limit = max_new_tokens or category_max_new_tokens(record.category)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=generation_limit,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        raw_answer = tokenizer.decode(generated[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
        answer = strip_qwen_think(raw_answer)
        issues = main.main_verifier_issues(answer, record.verifier)
        results.append(
            {
                "id": record.record_id,
                "category": record.category,
                "clean": not issues,
                "issues": issues,
                "raw_answer": raw_answer,
                "answer": answer,
            }
        )

    summary = {
        "model": model_name,
        "adapter": str(adapter_dir) if adapter_dir else None,
        "enable_thinking": enable_thinking,
        "input_file": str(input_file),
        "total": len(results),
        "clean": sum(1 for result in results if result["clean"]),
        "issue_counts": dict(Counter(issue for result in results for issue in result["issues"])),
        "category_clean": {},
        "results": results,
    }
    for category in sorted({result["category"] for result in results}):
        category_results = [result for result in results if result["category"] == category]
        summary["category_clean"][category] = {
            "clean": sum(1 for result in category_results if result["clean"]),
            "total": len(category_results),
        }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a HF QLoRA adapter against Main Agent verifier rows.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--adapter-dir", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--no-4bit", action="store_true")
    return parser.parse_args()


def main_cli() -> int:
    args = parse_args()
    summary = run_eval(
        model_name=args.model,
        input_file=Path(args.input_file),
        output_file=Path(args.output_file),
        adapter_dir=Path(args.adapter_dir) if args.adapter_dir else None,
        max_new_tokens=args.max_new_tokens,
        load_4bit=not args.no_4bit,
        enable_thinking=args.enable_thinking,
    )
    print(
        json.dumps(
            {
                "output_file": args.output_file,
                "total": summary["total"],
                "clean": summary["clean"],
                "issue_counts": summary["issue_counts"],
                "category_clean": summary["category_clean"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
