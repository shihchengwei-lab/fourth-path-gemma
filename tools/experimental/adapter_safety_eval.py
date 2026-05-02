from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import main


@dataclass(frozen=True)
class AdapterContainmentRecord:
    record_id: str
    category: str
    prompt: str
    verifier: dict[str, Any]


def strip_qwen_think(text: str) -> str:
    return re.sub(r"^\s*<think>.*?</think>\s*", "", text, flags=re.S).strip()


def load_records(path: Path) -> tuple[list[AdapterContainmentRecord], list[str], int]:
    if not path.exists():
        return [], [f"file not found: {path}"], 0
    records: list[AdapterContainmentRecord] = []
    errors: list[str] = []
    total = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        total += 1
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {line_number}: invalid JSON: {exc.msg}")
            continue
        if not isinstance(row, dict):
            errors.append(f"line {line_number}: row must be an object")
            continue
        record_id = row.get("id")
        category = row.get("category")
        prompt = row.get("prompt")
        verifier = row.get("verifier")
        if not isinstance(record_id, str) or not record_id.strip():
            errors.append(f"line {line_number}: id must be a non-empty string")
        if not isinstance(category, str) or not category.strip():
            errors.append(f"line {line_number}: category must be a non-empty string")
        if not isinstance(prompt, str) or not prompt.strip():
            errors.append(f"line {line_number}: prompt must be a non-empty string")
        if not isinstance(verifier, dict) or not verifier:
            errors.append(f"line {line_number}: verifier must be a non-empty object")
        if errors and errors[-1].startswith(f"line {line_number}:"):
            continue
        records.append(
            AdapterContainmentRecord(
                record_id=record_id.strip(),
                category=category.strip(),
                prompt=prompt.strip(),
                verifier=dict(verifier),
            )
        )
    if total == 0:
        errors.append("corpus is empty")
    return records, errors, total


def adapter_containment_review(answer: str, verifier: dict[str, Any]) -> tuple[list[str], list[str], dict[str, Any]]:
    candidate_issues: list[str] = []
    containment_issues: list[str] = []
    classify = main.classify_candidate(answer)
    if classify.route != "forward_to_cold_eyes":
        candidate_issues.append(f"classify:{classify.reason or classify.route}")

    cold_eyes = main.mechanical_cold_eyes_review(main.DEFAULT_POLICY_PATH, answer)
    if cold_eyes is None:
        candidate_issues.append("mechanical_audit_unavailable")
    elif cold_eyes.verdict != "pass":
        candidate_issues.append(f"mechanical:{cold_eyes.canon_clause or 'fail'}")

    verifier_issues = main.main_verifier_issues(answer, verifier)
    candidate_issues.extend(f"verifier:{issue}" for issue in verifier_issues)

    if classify.route != "forward_to_cold_eyes":
        external_gate = "blocked_by_classifier"
    elif cold_eyes is None:
        external_gate = "blocked_by_missing_mechanical_audit"
        containment_issues.append("mechanical_audit_unavailable")
    elif cold_eyes.verdict != "pass":
        external_gate = "blocked_by_mechanical_audit"
    else:
        external_gate = "forwarded"

    return list(dict.fromkeys(candidate_issues)), list(dict.fromkeys(containment_issues)), {
        "classify_route": classify.route,
        "classify_reason": classify.reason,
        "classify_canon_clause": classify.canon_clause,
        "mechanical_verdict": cold_eyes.verdict if cold_eyes else None,
        "mechanical_canon_clause": cold_eyes.canon_clause if cold_eyes else None,
        "mechanical_reason": cold_eyes.reason if cold_eyes else None,
        "external_gate": external_gate,
    }


def adapter_safety_issues(answer: str, verifier: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    candidate_issues, _, audit = adapter_containment_review(answer, verifier)
    return candidate_issues, audit


def load_model(
    model_name: str,
    adapter_dir: Path | None,
    load_4bit: bool,
) -> tuple[Any, Any]:
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
    return tokenizer, model


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


def generate_answer(
    tokenizer: Any,
    model: Any,
    prompt: str,
    max_new_tokens: int,
    enable_thinking: bool,
) -> tuple[str, str]:
    messages = [
        {"role": "system", "content": main.MAIN_AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    prompt_text = chat_template_text(tokenizer, messages, enable_thinking=enable_thinking)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    raw_answer = tokenizer.decode(generated[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
    return raw_answer, strip_qwen_think(raw_answer)


def run_eval(
    model_name: str,
    input_file: Path,
    output_file: Path,
    adapter_dir: Path | None = None,
    max_new_tokens: int = 180,
    load_4bit: bool = True,
    enable_thinking: bool = False,
) -> dict[str, Any]:
    records, errors, total = load_records(input_file)
    if errors:
        raise ValueError("; ".join(errors))
    tokenizer, model = load_model(model_name, adapter_dir, load_4bit=load_4bit)

    results: list[dict[str, Any]] = []
    for record in records:
        raw_answer, answer = generate_answer(
            tokenizer,
            model,
            record.prompt,
            max_new_tokens,
            enable_thinking=enable_thinking,
        )
        candidate_issues, containment_issues, audit = adapter_containment_review(answer, record.verifier)
        results.append(
            {
                "id": record.record_id,
                "category": record.category,
                "clean": not candidate_issues,
                "contained": not containment_issues,
                "issues": candidate_issues,
                "candidate_issues": candidate_issues,
                "containment_issues": containment_issues,
                "audit": audit,
                "raw_answer": raw_answer,
                "answer": answer,
            }
        )

    issue_counts = Counter(issue for result in results for issue in result["candidate_issues"])
    containment_issue_counts = Counter(issue for result in results for issue in result["containment_issues"])
    category_clean: dict[str, dict[str, int]] = {}
    for category in sorted({result["category"] for result in results}):
        category_results = [result for result in results if result["category"] == category]
        category_clean[category] = {
            "clean": sum(1 for result in category_results if result["clean"]),
            "contained": sum(1 for result in category_results if result["contained"]),
            "total": len(category_results),
        }
    summary = {
        "model": model_name,
        "adapter": str(adapter_dir) if adapter_dir else None,
        "enable_thinking": enable_thinking,
        "input_file": str(input_file),
        "source_total": total,
        "total": len(results),
        "clean": sum(1 for result in results if result["clean"]),
        "contained": sum(1 for result in results if result["contained"]),
        "issue_counts": dict(issue_counts),
        "containment_issue_counts": dict(containment_issue_counts),
        "category_clean": category_clean,
        "results": results,
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a HF adapter against eval-only Main Agent containment "
            "rows. Candidate cleanliness measures model behavior; containment "
            "measures whether external gates stop unsafe authority claims."
        )
    )
    parser.add_argument("--input-file", default="data/main_agent_adapter_containment_seed.jsonl")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--adapter-dir", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=180)
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
                "contained": summary["contained"],
                "issue_counts": summary["issue_counts"],
                "containment_issue_counts": summary["containment_issue_counts"],
                "category_clean": summary["category_clean"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
