from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core_types import SetupError


@dataclass(frozen=True)
class MainAgentCheck:
    path: Path
    total: int
    categories: dict[str, int]
    errors: list[str]
    verifier_records: int = 0

    def public_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "total": self.total,
            "categories": self.categories,
            "verifier_records": self.verifier_records,
            "errors": self.errors,
        }


@dataclass(frozen=True)
class MainAgentRecord:
    record_id: str
    category: str
    prompt: str
    target_response: str
    verifier: dict[str, Any] = field(default_factory=dict)


def safe_ratio(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def sorted_count_by(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def validate_main_agent_record(record: Any, index: int) -> list[str]:
    prefix = f"line {index}"
    if not isinstance(record, dict):
        return [f"{prefix}: record must be an object"]

    errors: list[str] = []
    for field_name in ("id", "category", "prompt", "target_response"):
        if not isinstance(record.get(field_name), str) or not record[field_name].strip():
            errors.append(f"{prefix}: {field_name} must be a non-empty string")

    if "candidate" in record:
        errors.append(f"{prefix}: candidate is an evaluation output; use target_response for the seed answer")
    if "output" in record:
        errors.append(f"{prefix}: output is ambiguous; use target_response instead")
    if "verifier" in record:
        verifier = record.get("verifier")
        if not isinstance(verifier, dict):
            errors.append(f"{prefix}: verifier must be an object")
        else:
            errors.extend(validate_main_verifier(verifier, prefix))
    return errors


def validate_main_verifier(verifier: dict[str, Any], prefix: str) -> list[str]:
    errors: list[str] = []
    allowed = {
        "required_terms",
        "required_any",
        "forbidden_terms",
        "required_regex",
        "forbidden_regex",
        "numeric_answer",
        "max_chars",
    }
    for field_name in sorted(set(verifier) - allowed):
        errors.append(f"{prefix}: verifier.{field_name} is not supported")
    for field_name in ("required_terms", "forbidden_terms", "required_regex", "forbidden_regex"):
        value = verifier.get(field_name)
        if value is None:
            continue
        if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
            errors.append(f"{prefix}: verifier.{field_name} must be a list of non-empty strings")
            continue
        if field_name.endswith("_regex"):
            for pattern in value:
                try:
                    re.compile(pattern)
                except re.error as exc:
                    errors.append(f"{prefix}: verifier.{field_name} contains invalid regex: {exc}")
    required_any = verifier.get("required_any")
    if required_any is not None:
        if not isinstance(required_any, list) or not required_any:
            errors.append(f"{prefix}: verifier.required_any must be a non-empty list of term groups")
        else:
            for group in required_any:
                if not isinstance(group, list) or not all(
                    isinstance(item, str) and item.strip() for item in group
                ):
                    errors.append(
                        f"{prefix}: verifier.required_any must contain non-empty string groups"
                    )
                    break
    if "numeric_answer" in verifier and not isinstance(verifier.get("numeric_answer"), (int, float, str)):
        errors.append(f"{prefix}: verifier.numeric_answer must be a string or number")
    if "max_chars" in verifier:
        max_chars = verifier.get("max_chars")
        if not isinstance(max_chars, int) or max_chars < 1:
            errors.append(f"{prefix}: verifier.max_chars must be a positive integer")
    return errors


def load_main_agent_records(path: Path) -> tuple[list[MainAgentRecord], list[str], int]:
    if not path.exists():
        raise SetupError(f"Main Agent corpus not found: {path}")

    records: list[MainAgentRecord] = []
    errors: list[str] = []
    total = 0

    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        total += 1
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {index}: invalid JSON: {exc.msg}")
            continue

        record_errors = validate_main_agent_record(record, index)
        errors.extend(record_errors)
        if not record_errors and isinstance(record, dict):
            records.append(
                MainAgentRecord(
                    record_id=record["id"].strip(),
                    category=record["category"].strip(),
                    prompt=record["prompt"].strip(),
                    target_response=record["target_response"].strip(),
                    verifier=dict(record.get("verifier") or {}),
                )
            )

    if total == 0:
        errors.append("corpus is empty")
    return records, errors, total


def check_main_agent_corpus(path: Path) -> MainAgentCheck:
    records, errors, total = load_main_agent_records(path)
    categories: dict[str, int] = {}
    for record in records:
        categories[record.category] = categories.get(record.category, 0) + 1
    return MainAgentCheck(
        path,
        total,
        dict(sorted(categories.items())),
        errors,
        verifier_records=sum(bool(record.verifier) for record in records),
    )


def apply_main_agent_requirements(
    result: MainAgentCheck,
    min_total: int = 0,
    min_category: int = 0,
) -> MainAgentCheck:
    errors = list(result.errors)
    if result.total < min_total:
        errors.append(f"records below minimum: {result.total} < {min_total}")
    for category, count in result.categories.items():
        if count < min_category:
            errors.append(f"{category} records below minimum: {count} < {min_category}")
    return MainAgentCheck(
        result.path,
        result.total,
        result.categories,
        errors,
        result.verifier_records,
    )


def render_main_agent_check(result: MainAgentCheck) -> str:
    status = "ok" if not result.errors else "error"
    lines = [
        f"Main Agent corpus: {result.path}",
        f"Status: {status}",
        f"Records: {result.total}",
        f"Verifier records: {result.verifier_records}",
        "Categories:",
    ]
    lines.extend(f"- {category}: {count}" for category, count in result.categories.items())
    if result.errors:
        lines.extend(["", "Errors:"])
        lines.extend(f"- {error}" for error in result.errors)
    return "\n".join(lines)


def stable_text_hash(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.strip().casefold())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]


def main_data_quality_check_data(
    paths: list[Path],
    require_verifier_patterns: tuple[str, ...] = ("hard", "heldout"),
    max_category_share: float = 0.5,
    min_records_for_category_balance: int = 8,
    min_verifier_types: int = 3,
) -> dict[str, Any]:
    errors: list[str] = []
    files: list[dict[str, Any]] = []
    seen_ids: dict[str, str] = {}
    seen_prompts: dict[str, tuple[str, str]] = {}
    duplicate_ids: list[str] = []
    duplicate_prompt_hashes: list[str] = []
    total_records = 0
    total_verifier_records = 0

    if not 0 < max_category_share <= 1:
        errors.append("--max-category-share must be greater than 0 and at most 1")
    if min_records_for_category_balance < 1:
        errors.append("--min-records-for-category-balance must be at least 1")
    if min_verifier_types < 1:
        errors.append("--min-verifier-types must be at least 1")

    for path in paths:
        records, load_errors, total = load_main_agent_records(path)
        errors.extend(f"{path}: {error}" for error in load_errors)
        total_records += total
        verifier_records = sum(bool(record.verifier) for record in records)
        total_verifier_records += verifier_records
        categories = sorted_count_by(record.category for record in records)
        verifier_type_counts = Counter(
            verifier_name
            for record in records
            for verifier_name, verifier_value in record.verifier.items()
            if verifier_value
        )
        requires_verifier = any(pattern in path.name for pattern in require_verifier_patterns)
        all_missing_verifier_ids = [record.record_id for record in records if not record.verifier]
        missing_verifier_ids = all_missing_verifier_ids if requires_verifier else []
        dominant_category = None
        dominant_category_share = 0.0
        category_balance_checked = total >= min_records_for_category_balance and bool(categories)

        if categories:
            dominant_category, dominant_count = max(categories.items(), key=lambda item: (item[1], item[0]))
            dominant_category_share = round(safe_ratio(dominant_count, total), 3)
        if category_balance_checked and dominant_category_share > max_category_share:
            errors.append(
                f"{path}: dominant category {dominant_category} covers "
                f"{dominant_category_share:.3f} of records; limit is {max_category_share:.3f}"
            )

        if requires_verifier and missing_verifier_ids:
            errors.append(
                f"{path}: verifier required but missing for {len(missing_verifier_ids)} records"
            )
        if requires_verifier and verifier_records and len(verifier_type_counts) < min_verifier_types:
            errors.append(
                f"{path}: verifier diversity has {len(verifier_type_counts)} type(s); "
                f"minimum is {min_verifier_types}"
            )

        for record in records:
            previous_path = seen_ids.get(record.record_id)
            if previous_path is not None:
                duplicate_ids.append(record.record_id)
                errors.append(
                    f"duplicate id across corpora: {record.record_id} in {previous_path} and {path}"
                )
            else:
                seen_ids[record.record_id] = str(path)

            prompt_hash = stable_text_hash(record.prompt)
            previous_prompt = seen_prompts.get(prompt_hash)
            if previous_prompt is not None:
                duplicate_prompt_hashes.append(prompt_hash)
                previous_id, previous_prompt_path = previous_prompt
                errors.append(
                    "duplicate prompt across corpora: "
                    f"hash={prompt_hash} ids={previous_id},{record.record_id} "
                    f"paths={previous_prompt_path},{path}"
                )
            else:
                seen_prompts[prompt_hash] = (record.record_id, str(path))

        files.append(
            {
                "path": str(path),
                "total": total,
                "categories": categories,
                "dominant_category": dominant_category,
                "dominant_category_share": dominant_category_share,
                "category_balance_checked": category_balance_checked,
                "max_category_share": max_category_share,
                "verifier_records": verifier_records,
                "verifier_rate": round(safe_ratio(verifier_records, total), 3),
                "verifier_type_counts": dict(sorted(verifier_type_counts.items())),
                "verifier_type_count": len(verifier_type_counts),
                "min_verifier_types": min_verifier_types,
                "unverified_records": len(all_missing_verifier_ids),
                "requires_verifier": requires_verifier,
                "missing_verifier_ids": missing_verifier_ids,
            }
        )

    return {
        "files": files,
        "total_records": total_records,
        "total_verifier_records": total_verifier_records,
        "overall_verifier_rate": round(safe_ratio(total_verifier_records, total_records), 3),
        "duplicate_ids": sorted(set(duplicate_ids)),
        "duplicate_prompt_hashes": sorted(set(duplicate_prompt_hashes)),
        "require_verifier_patterns": list(require_verifier_patterns),
        "max_category_share": max_category_share,
        "min_records_for_category_balance": min_records_for_category_balance,
        "min_verifier_types": min_verifier_types,
        "errors": errors,
    }


def render_main_data_quality_check(data: dict[str, Any]) -> str:
    status = "ok" if not data["errors"] else "error"
    lines = [
        f"Main Agent data quality: {status}",
        f"Records: {data['total_records']}",
        f"Verifier records: {data['total_verifier_records']} ({data['overall_verifier_rate']:.3f})",
        "Files:",
    ]
    for file_data in data["files"]:
        lines.append(
            "- {path}: total={total}, verifier={verifier_records} ({verifier_rate:.3f}), "
            "types={verifier_type_count}, dominant={dominant_category} "
            "({dominant_category_share:.3f}), unverified={unverified_records}, "
            "requires_verifier={requires_verifier}".format(**file_data)
        )
    if data["duplicate_ids"] or data["duplicate_prompt_hashes"]:
        lines.append("Duplicates detected.")
    if data["errors"]:
        lines.extend(["", "Errors:"])
        lines.extend(f"- {error}" for error in data["errors"])
    return "\n".join(lines)


def normalize_numeric_token(value: str | int | float) -> str:
    text = str(value).strip()
    try:
        number = float(text)
    except ValueError:
        return text
    if number.is_integer():
        return str(int(number))
    return f"{number:.8g}"


def extract_numeric_tokens(text: str) -> set[str]:
    values: set[str] = set()
    for match in re.finditer(r"(?<![\w.])-?\d+(?:\.\d+)?(?!\w)", text):
        values.add(normalize_numeric_token(match.group(0)))
    return values


def main_verifier_issues(text: str, verifier: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    lower = text.lower()
    for term in verifier.get("required_terms", []):
        if term.lower() not in lower:
            issues.append("missing_required_term")
            break
    for group in verifier.get("required_any", []):
        if not any(term.lower() in lower for term in group):
            issues.append("missing_required_any")
            break
    for term in verifier.get("forbidden_terms", []):
        if term.lower() in lower:
            issues.append("forbidden_term_present")
            break
    for pattern in verifier.get("required_regex", []):
        if re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE) is None:
            issues.append("missing_required_pattern")
            break
    for pattern in verifier.get("forbidden_regex", []):
        if re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE) is not None:
            issues.append("forbidden_pattern_present")
            break
    if "numeric_answer" in verifier:
        expected = normalize_numeric_token(verifier["numeric_answer"])
        if expected not in extract_numeric_tokens(text):
            issues.append("numeric_answer_mismatch")
    max_chars = verifier.get("max_chars")
    if isinstance(max_chars, int) and len(text) > max_chars:
        issues.append("verifier_max_chars_exceeded")
    return list(dict.fromkeys(issues))
