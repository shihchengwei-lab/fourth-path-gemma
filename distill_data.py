from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core_types import SetupError


@dataclass(frozen=True)
class DistillCheck:
    path: Path
    total: int
    pass_count: int
    fail_count: int
    clauses: dict[str, int]
    errors: list[str]

    def public_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "total": self.total,
            "pass_count": self.pass_count,
            "fail_count": self.fail_count,
            "clauses": self.clauses,
            "errors": self.errors,
        }


@dataclass(frozen=True)
class DistillRecord:
    record_id: str
    candidate: str
    verdict: str
    canon_clause: str | None
    reason: str


def validate_distill_record(record: Any, index: int) -> list[str]:
    prefix = f"line {index}"
    if not isinstance(record, dict):
        return [f"{prefix}: record must be an object"]

    errors: list[str] = []
    for field_name in ("id", "candidate", "verdict", "reason"):
        if not isinstance(record.get(field_name), str) or not record[field_name].strip():
            errors.append(f"{prefix}: {field_name} must be a non-empty string")

    verdict = record.get("verdict")
    clause = record.get("canon_clause")
    if verdict not in {"pass", "fail"}:
        errors.append(f"{prefix}: verdict must be pass or fail")
    elif verdict == "pass" and clause is not None:
        errors.append(f"{prefix}: pass records must use canon_clause null")
    elif verdict == "fail" and clause not in {"C1", "C2", "C3"}:
        errors.append(f"{prefix}: fail records must use canon_clause C1, C2, or C3")

    if "prompt" in record:
        errors.append(f"{prefix}: prompt is not allowed in distillation seed records")
    if "output" in record:
        errors.append(f"{prefix}: output is ambiguous; use candidate instead")
    return errors


def load_distill_records(path: Path) -> tuple[list[DistillRecord], list[str], int]:
    if not path.exists():
        raise SetupError(f"Distillation corpus not found: {path}")

    records: list[DistillRecord] = []
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

        record_errors = validate_distill_record(record, index)
        errors.extend(record_errors)
        if not record_errors and isinstance(record, dict):
            records.append(
                DistillRecord(
                    record_id=record["id"].strip(),
                    candidate=record["candidate"].strip(),
                    verdict=record["verdict"].strip(),
                    canon_clause=record.get("canon_clause"),
                    reason=record["reason"].strip(),
                )
            )

    if total == 0:
        errors.append("corpus is empty")
    return records, errors, total


def check_distillation_corpus(path: Path) -> DistillCheck:
    records, errors, total = load_distill_records(path)
    clauses = {"C1": 0, "C2": 0, "C3": 0}
    for record in records:
        if record.canon_clause in clauses:
            clauses[record.canon_clause] += 1

    pass_count = sum(record.verdict == "pass" for record in records)
    fail_count = sum(record.verdict == "fail" for record in records)
    return DistillCheck(path, total, pass_count, fail_count, clauses, errors)


def apply_distill_balance_requirements(
    result: DistillCheck,
    min_pass: int = 0,
    min_fail: int = 0,
    min_clause: int = 0,
) -> DistillCheck:
    errors = list(result.errors)
    if result.pass_count < min_pass:
        errors.append(f"pass records below minimum: {result.pass_count} < {min_pass}")
    if result.fail_count < min_fail:
        errors.append(f"fail records below minimum: {result.fail_count} < {min_fail}")
    for clause, count in result.clauses.items():
        if count < min_clause:
            errors.append(f"{clause} records below minimum: {count} < {min_clause}")
    return DistillCheck(result.path, result.total, result.pass_count, result.fail_count, result.clauses, errors)


def render_distill_check(result: DistillCheck) -> str:
    status = "ok" if not result.errors else "error"
    lines = [
        f"Distillation corpus: {result.path}",
        f"Status: {status}",
        f"Records: {result.total}",
        f"Pass: {result.pass_count}",
        f"Fail: {result.fail_count}",
        f"Clauses: C1={result.clauses['C1']}, C2={result.clauses['C2']}, C3={result.clauses['C3']}",
    ]
    if result.errors:
        lines.extend(["", "Errors:"])
        lines.extend(f"- {error}" for error in result.errors)
    return "\n".join(lines)
