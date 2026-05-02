from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core_types import SetupError
from main_agent_data import MainAgentRecord, load_main_agent_records


SFT_ALLOWED_MESSAGE_ROLES = ("system", "user", "assistant")
SFT_FORBIDDEN_TOP_LEVEL_FIELDS = ("prompt", "target_response", "candidate", "output", "response")


@dataclass(frozen=True)
class MainLimoCuratedCase:
    row_id: str
    category: str
    selected: bool
    score: float
    assistant_chars: int
    features: dict[str, int]


@dataclass(frozen=True)
class MainMixDistillCase:
    row_key: str
    row_id: str
    category: str
    bucket: str
    selected: bool
    score: float
    assistant_chars: int


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _sorted_count_by(values: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _mix_distill_bucket(text: str, long_char_threshold: int) -> str:
    return "long" if len(text) >= long_char_threshold else "short"


def main_sft_messages(
    record: MainAgentRecord,
    system_prompt: str,
    include_system: bool = True,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if include_system:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(
        [
            {"role": "user", "content": record.prompt},
            {"role": "assistant", "content": record.target_response},
        ]
    )
    return messages


def export_main_sft(
    records: list[MainAgentRecord],
    output_file: Path,
    system_prompt: str,
    include_system: bool = True,
) -> dict[str, Any]:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(
            {
                "id": record.record_id,
                "category": record.category,
                "messages": main_sft_messages(record, system_prompt, include_system=include_system),
            },
            ensure_ascii=False,
        )
        for record in records
    ]
    output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    categories: dict[str, int] = {}
    for record in records:
        categories[record.category] = categories.get(record.category, 0) + 1
    return {
        "path": str(output_file),
        "records": len(records),
        "include_system": include_system,
        "categories": dict(sorted(categories.items())),
    }


def render_main_sft_export(data: dict[str, Any]) -> str:
    lines = [
        f"Main Agent SFT export: {data['path']}",
        f"Records: {data['records']}",
        f"Include system: {data['include_system']}",
        "Categories:",
    ]
    lines.extend(f"- {category}: {count}" for category, count in data["categories"].items())
    return "\n".join(lines)


def sft_export_format_gate_data(paths: Path | list[Path], system_prompt: str) -> dict[str, Any]:
    source_paths = [paths] if isinstance(paths, Path) else list(paths)
    all_rows: list[dict[str, Any]] = []
    file_reports: list[dict[str, Any]] = []
    load_errors: list[str] = []
    validation_errors: list[str] = []
    source_total = 0

    for path in source_paths:
        records, file_load_errors, total = load_main_agent_records(path)
        source_total += total
        load_errors.extend(f"{path}: {error}" for error in file_load_errors)
        rows = [
            {
                "id": record.record_id,
                "category": record.category,
                "messages": main_sft_messages(record, system_prompt, include_system=True),
            }
            for record in records
        ]
        file_validation_errors = [
            f"{path}: {error}"
            for index, row in enumerate(rows, 1)
            for error in validate_sft_jsonl_row(row, index)
        ]
        validation_errors.extend(file_validation_errors)
        file_report = training_data_quality_report(rows) if rows else {}
        file_reports.append(
            {
                "source_path": str(path),
                "source_total": total,
                "rows": len(rows),
                "system_rows": file_report.get("system_rows", 0),
                "duplicate_ids": file_report.get("duplicate_ids", []),
                "load_errors": [f"{path}: {error}" for error in file_load_errors],
                "validation_errors": file_validation_errors,
            }
        )
        all_rows.extend(rows)

    report = training_data_quality_report(all_rows) if all_rows else {}
    format_errors = (
        training_data_quality_errors(report, require_system=True)
        if all_rows
        else ["training data is empty"]
    )
    return {
        "source_path": str(source_paths[0]) if len(source_paths) == 1 else None,
        "source_paths": [str(path) for path in source_paths],
        "source_total": source_total,
        "rows": len(all_rows),
        "system_rows": report.get("system_rows", 0),
        "duplicate_ids": report.get("duplicate_ids", []),
        "files": file_reports,
        "load_errors": load_errors,
        "validation_errors": validation_errors,
        "format_errors": format_errors,
        "errors": load_errors + validation_errors + format_errors,
    }


def load_sft_jsonl_rows(path: Path) -> tuple[list[dict[str, Any]], list[str], int]:
    if not path.exists():
        return [], [f"file not found: {path}"], 0

    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    total = 0
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        total += 1
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {index}: invalid JSON: {exc.msg}")
            continue
        if not isinstance(row, dict):
            errors.append(f"line {index}: row must be an object")
            continue
        row_errors = validate_sft_jsonl_row(row, index)
        if row_errors:
            errors.extend(row_errors)
            continue
        rows.append(row)
    return rows, errors, total


def validate_sft_jsonl_row(row: dict[str, Any], line_number: int) -> list[str]:
    errors: list[str] = []
    row_id = row.get("id")
    if not isinstance(row_id, str) or not row_id.strip():
        errors.append(f"line {line_number}: id must be a non-empty string")

    for field_name in SFT_FORBIDDEN_TOP_LEVEL_FIELDS:
        if field_name in row:
            errors.append(
                f"line {line_number}: {field_name} is not allowed in SFT rows; use messages instead"
            )

    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        errors.append(f"line {line_number}: messages must be a non-empty list")
        return errors

    seen_roles: set[str] = set()
    assistant_has_text = False
    for message_index, message in enumerate(messages, 1):
        if not isinstance(message, dict):
            errors.append(f"line {line_number}: messages[{message_index}] must be an object")
            continue
        role = message.get("role")
        if role not in SFT_ALLOWED_MESSAGE_ROLES:
            errors.append(
                f"line {line_number}: messages[{message_index}].role must be one of "
                f"{', '.join(SFT_ALLOWED_MESSAGE_ROLES)}"
            )
        elif isinstance(role, str):
            seen_roles.add(role)

        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            errors.append(f"line {line_number}: messages[{message_index}].content must be a non-empty string")
        elif role == "assistant":
            assistant_has_text = True

    if "user" not in seen_roles:
        errors.append(f"line {line_number}: row must contain a user message")
    if "assistant" not in seen_roles or not assistant_has_text:
        errors.append(f"line {line_number}: row must contain an assistant message with text content")
    return errors


def training_row_assistant_text(row: dict[str, Any]) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        return content.strip() if isinstance(content, str) else ""
    return ""


def limo_keyword_count(text: str, keywords: tuple[str, ...]) -> int:
    lower = text.lower()
    return sum(lower.count(keyword) for keyword in keywords)


def limo_template_features(text: str) -> dict[str, int]:
    lower = text.lower()
    return {
        "assistant_chars": len(text),
        "line_count": len([line for line in text.splitlines() if line.strip()]),
        "verification_markers": limo_keyword_count(
            lower,
            ("check", "verify", "validate", "confirm", "檢查", "驗證", "核對"),
        ),
        "exploration_markers": limo_keyword_count(
            lower,
            ("case", "option", "alternative", "suppose", "if ", "如果", "情況", "可能"),
        ),
        "connective_markers": limo_keyword_count(
            lower,
            ("because", "therefore", "since", "so ", "then", "thus", "因此", "所以", "接著", "然後"),
        ),
        "step_markers": len(re.findall(r"(?im)^\s*(?:\d+[.)]|[-*]\s+|step\s+\d+|步驟)", text)),
        "final_answer_markers": limo_keyword_count(
            lower,
            ("####", "answer", "final", "therefore", "所以", "答案"),
        ),
    }


def limo_template_score(text: str) -> float:
    features = limo_template_features(text)
    length_score = min(features["assistant_chars"] / 1200, 1.0) * 30.0
    verification_score = min(features["verification_markers"] / 2, 1.0) * 20.0
    exploration_score = min(features["exploration_markers"] / 3, 1.0) * 20.0
    connective_score = min(features["connective_markers"] / 6, 1.0) * 20.0
    structure_score = min((features["step_markers"] + features["final_answer_markers"]) / 4, 1.0) * 10.0
    overlong_penalty = max(0.0, (features["assistant_chars"] - 4096) / 4096) * 20.0
    return round(
        max(0.0, length_score + verification_score + exploration_score + connective_score + structure_score - overlong_penalty),
        3,
    )


def main_limo_curated_case_dict(case: MainLimoCuratedCase) -> dict[str, Any]:
    return {
        "id": case.row_id,
        "category": case.category,
        "selected": case.selected,
        "score": case.score,
        "assistant_chars": case.assistant_chars,
        "features": case.features,
    }


def run_main_limo_curate(
    rows: list[dict[str, Any]],
    output_file: Path,
    max_records: int = 800,
    min_score: float = 0.0,
    max_per_category: int = 0,
) -> dict[str, Any]:
    if max_records < 1:
        raise SetupError("--max-records must be at least 1.")
    if max_per_category < 0:
        raise SetupError("--max-per-category must be zero or greater.")

    scored: list[tuple[float, dict[str, Any], MainLimoCuratedCase]] = []
    for index, row in enumerate(rows, 1):
        text = training_row_assistant_text(row)
        features = limo_template_features(text)
        score = limo_template_score(text)
        row_id = str(row.get("id") or row.get("record_id") or f"row-{index}")
        category = str(row.get("category") or "unknown")
        scored.append(
            (
                score,
                row,
                MainLimoCuratedCase(
                    row_id=row_id,
                    category=category,
                    selected=False,
                    score=score,
                    assistant_chars=features["assistant_chars"],
                    features=features,
                ),
            )
        )

    scored.sort(key=lambda item: (-item[0], item[2].category, item[2].row_id))
    selected_rows: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    category_counts: Counter[str] = Counter()

    for score, row, case in scored:
        if len(selected_rows) >= max_records:
            break
        if score < min_score:
            continue
        if max_per_category and category_counts[case.category] >= max_per_category:
            continue
        curated = dict(row)
        curated["curation_source"] = "limo_less_is_more"
        curated["limo_score"] = score
        curated["limo_features"] = case.features
        selected_rows.append(curated)
        selected_ids.add(case.row_id)
        category_counts[case.category] += 1

    cases = [
        MainLimoCuratedCase(
            row_id=case.row_id,
            category=case.category,
            selected=case.row_id in selected_ids,
            score=case.score,
            assistant_chars=case.assistant_chars,
            features=case.features,
        )
        for _, _, case in scored
    ]

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in selected_rows)
        + ("\n" if selected_rows else ""),
        encoding="utf-8",
    )

    return {
        "path": str(output_file),
        "input_rows": len(rows),
        "selected_rows": len(selected_rows),
        "selection_rate": _safe_ratio(len(selected_rows), len(rows)),
        "max_records": max_records,
        "min_score": min_score,
        "max_per_category": max_per_category,
        "selected_category_counts": dict(sorted(category_counts.items())),
        "score_min": min((case.score for case in cases), default=0.0),
        "score_max": max((case.score for case in cases), default=0.0),
        "score_avg": round(_safe_ratio(sum(case.score for case in cases), len(cases)), 3),
        "cases": [main_limo_curated_case_dict(case) for case in cases],
    }


def render_main_limo_curate(data: dict[str, Any]) -> str:
    lines = [
        f"Main Agent LIMO curate: {data['path']}",
        f"Input rows: {data['input_rows']}",
        f"Selected rows: {data['selected_rows']}",
        f"Selection rate: {data['selection_rate']:.3f}",
        f"Score range: {data['score_min']:.3f} - {data['score_max']:.3f}",
        f"Average score: {data['score_avg']:.3f}",
        "Selected categories:",
    ]
    if data["selected_category_counts"]:
        lines.extend(f"- {category}: {count}" for category, count in data["selected_category_counts"].items())
    else:
        lines.append("- none")
    return "\n".join(lines)


def mix_distill_row_score(row: dict[str, Any], text: str) -> float:
    value = row.get("limo_score")
    if isinstance(value, (int, float)):
        return float(value)
    return limo_template_score(text)


def mix_distill_bucket(text: str, long_char_threshold: int) -> str:
    return _mix_distill_bucket(text, long_char_threshold)


def main_mix_distill_case_dict(case: MainMixDistillCase) -> dict[str, Any]:
    return {
        "id": case.row_id,
        "category": case.category,
        "bucket": case.bucket,
        "selected": case.selected,
        "score": case.score,
        "assistant_chars": case.assistant_chars,
    }


def run_main_mix_distill_curate(
    rows: list[dict[str, Any]],
    output_file: Path,
    max_records: int = 800,
    long_ratio: float = 0.2,
    long_char_threshold: int = 1200,
    max_per_category: int = 0,
) -> dict[str, Any]:
    if max_records < 1:
        raise SetupError("--max-records must be at least 1.")
    if not 0 <= long_ratio <= 1:
        raise SetupError("--long-ratio must be between 0 and 1.")
    if long_char_threshold < 1:
        raise SetupError("--long-char-threshold must be at least 1.")
    if max_per_category < 0:
        raise SetupError("--max-per-category must be zero or greater.")

    scored: list[tuple[float, dict[str, Any], MainMixDistillCase]] = []
    for index, row in enumerate(rows, 1):
        text = training_row_assistant_text(row)
        score = mix_distill_row_score(row, text)
        row_id = str(row.get("id") or row.get("record_id") or f"row-{index}")
        row_key = f"{row_id}#{index}"
        category = str(row.get("category") or "unknown")
        bucket = mix_distill_bucket(text, long_char_threshold)
        scored.append(
            (
                score,
                row,
                MainMixDistillCase(
                    row_key=row_key,
                    row_id=row_id,
                    category=category,
                    bucket=bucket,
                    selected=False,
                    score=score,
                    assistant_chars=len(text),
                ),
            )
        )

    scored.sort(key=lambda item: (-item[0], item[2].bucket, item[2].category, item[2].row_id))
    available_long = sum(1 for _, _, case in scored if case.bucket == "long")
    available_short = sum(1 for _, _, case in scored if case.bucket == "short")
    if long_ratio >= 1:
        long_target = min(available_long, max_records)
    elif long_ratio <= 0:
        long_target = 0
    else:
        max_long_from_ratio = int((available_short * long_ratio) // (1 - long_ratio))
        long_target = min(available_long, round(max_records * long_ratio), max_long_from_ratio)
    short_target = min(available_short, max_records - long_target)

    selected_rows: list[dict[str, Any]] = []
    selected_keys: set[str] = set()
    category_counts: Counter[str] = Counter()

    def can_select(case: MainMixDistillCase) -> bool:
        return case.row_key not in selected_keys and (
            not max_per_category or category_counts[case.category] < max_per_category
        )

    def select_case(row: dict[str, Any], case: MainMixDistillCase) -> None:
        selected = dict(row)
        selected["mix_distillation_source"] = "small_model_learnability_gap"
        selected["mix_distill_bucket"] = case.bucket
        selected["mix_distill_score"] = case.score
        selected["mix_distill_long_ratio_target"] = long_ratio
        selected_rows.append(selected)
        selected_keys.add(case.row_key)
        category_counts[case.category] += 1

    for desired_bucket, target in (("long", long_target), ("short", short_target)):
        for _, row, case in scored:
            if sum(1 for selected in selected_rows if selected.get("mix_distill_bucket") == desired_bucket) >= target:
                break
            if case.bucket == desired_bucket and can_select(case):
                select_case(row, case)

    for _, row, case in scored:
        if len(selected_rows) >= max_records:
            break
        if case.bucket == "short" and can_select(case):
            select_case(row, case)

    cases = [
        MainMixDistillCase(
            row_key=case.row_key,
            row_id=case.row_id,
            category=case.category,
            bucket=case.bucket,
            selected=case.row_key in selected_keys,
            score=case.score,
            assistant_chars=case.assistant_chars,
        )
        for _, _, case in scored
    ]

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in selected_rows)
        + ("\n" if selected_rows else ""),
        encoding="utf-8",
    )

    selected_bucket_counts = _sorted_count_by([str(row.get("mix_distill_bucket")) for row in selected_rows])
    selected_long = selected_bucket_counts.get("long", 0)
    return {
        "path": str(output_file),
        "input_rows": len(rows),
        "selected_rows": len(selected_rows),
        "selection_rate": _safe_ratio(len(selected_rows), len(rows)),
        "max_records": max_records,
        "long_ratio_target": long_ratio,
        "actual_long_ratio": _safe_ratio(selected_long, len(selected_rows)),
        "long_char_threshold": long_char_threshold,
        "max_per_category": max_per_category,
        "input_bucket_counts": _sorted_count_by([case.bucket for _, _, case in scored]),
        "selected_bucket_counts": selected_bucket_counts,
        "selected_category_counts": dict(sorted(category_counts.items())),
        "cases": [main_mix_distill_case_dict(case) for case in cases],
    }


def render_main_mix_distill_curate(data: dict[str, Any]) -> str:
    lines = [
        f"Main Agent mix distillation curate: {data['path']}",
        f"Input rows: {data['input_rows']}",
        f"Selected rows: {data['selected_rows']}",
        f"Selection rate: {data['selection_rate']:.3f}",
        f"Long ratio target: {data['long_ratio_target']:.3f}",
        f"Actual long ratio: {data['actual_long_ratio']:.3f}",
        "Selected buckets:",
    ]
    if data["selected_bucket_counts"]:
        lines.extend(f"- {bucket}: {count}" for bucket, count in data["selected_bucket_counts"].items())
    else:
        lines.append("- none")
    return "\n".join(lines)


def training_data_quality_report(rows: list[dict[str, Any]], long_char_threshold: int = 1200) -> dict[str, Any]:
    if long_char_threshold < 1:
        raise SetupError("--long-char-threshold must be at least 1.")

    ids: list[str] = []
    record_ids: list[str] = []
    assistant_lengths: list[int] = []
    system_rows = 0
    message_counts: list[int] = []
    source_values: list[str] = []
    curation_values: list[str] = []
    mix_source_values: list[str] = []
    bucket_values: list[str] = []
    category_values: list[str] = []

    for index, row in enumerate(rows, 1):
        text = training_row_assistant_text(row)
        row_id = str(row.get("id") or f"row-{index}")
        record_id = str(row.get("record_id") or row_id)
        category = str(row.get("category") or "unknown")
        messages = row.get("messages")
        message_list = messages if isinstance(messages, list) else []
        if any(isinstance(message, dict) and message.get("role") == "system" for message in message_list):
            system_rows += 1

        ids.append(row_id)
        record_ids.append(record_id)
        category_values.append(category)
        assistant_lengths.append(len(text))
        message_counts.append(len(message_list))
        source_values.append(str(row.get("source") or "unknown"))
        curation_values.append(str(row.get("curation_source") or "none"))
        mix_source_values.append(str(row.get("mix_distillation_source") or "none"))
        bucket_values.append(str(row.get("mix_distill_bucket") or _mix_distill_bucket(text, long_char_threshold)))

    id_counts = Counter(ids)
    record_counts = Counter(record_ids)
    duplicate_ids = sorted(row_id for row_id, count in id_counts.items() if count > 1)
    duplicate_record_ids = sorted(row_id for row_id, count in record_counts.items() if count > 1)
    return {
        "rows": len(rows),
        "category_counts": _sorted_count_by(category_values),
        "source_counts": _sorted_count_by(source_values),
        "curation_source_counts": _sorted_count_by(curation_values),
        "mix_distillation_source_counts": _sorted_count_by(mix_source_values),
        "reasoning_bucket_counts": _sorted_count_by(bucket_values),
        "long_char_threshold": long_char_threshold,
        "system_rows": system_rows,
        "system_row_rate": _safe_ratio(system_rows, len(rows)),
        "assistant_chars_min": min(assistant_lengths, default=0),
        "assistant_chars_max": max(assistant_lengths, default=0),
        "assistant_chars_avg": round(_safe_ratio(sum(assistant_lengths), len(assistant_lengths)), 3),
        "messages_per_row_avg": round(_safe_ratio(sum(message_counts), len(message_counts)), 3),
        "duplicate_ids": duplicate_ids,
        "duplicate_record_ids": duplicate_record_ids,
    }


def training_data_quality_errors(data: dict[str, Any], require_system: bool = False) -> list[str]:
    errors: list[str] = []
    if data["rows"] < 1:
        errors.append("training data is empty")
    if data["duplicate_ids"]:
        errors.append(f"duplicate row ids: {', '.join(data['duplicate_ids'])}")
    if require_system and data["system_rows"] != data["rows"]:
        missing = data["rows"] - data["system_rows"]
        errors.append(f"missing system messages: {missing} row(s)")
    return errors


def render_training_data_quality_report(data: dict[str, Any]) -> str:
    lines = [
        "Main Agent training-data report",
        f"Rows: {data['rows']}",
        f"Assistant chars: min={data['assistant_chars_min']} avg={data['assistant_chars_avg']:.3f} max={data['assistant_chars_max']}",
        f"System rows: {data['system_rows']} ({data['system_row_rate']:.3f})",
        "Reasoning buckets:",
    ]
    if data["reasoning_bucket_counts"]:
        lines.extend(f"- {bucket}: {count}" for bucket, count in data["reasoning_bucket_counts"].items())
    else:
        lines.append("- none")
    lines.append("Categories:")
    if data["category_counts"]:
        lines.extend(f"- {category}: {count}" for category, count in data["category_counts"].items())
    else:
        lines.append("- none")
    if data["duplicate_ids"] or data["duplicate_record_ids"]:
        lines.append("Duplicate keys detected.")
    if data.get("format_errors"):
        lines.append("Format errors:")
        lines.extend(f"- {error}" for error in data["format_errors"])
    return "\n".join(lines)
