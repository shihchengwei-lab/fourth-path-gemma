from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from action_gate import ACTION_CANDIDATE_REQUIRED_FIELDS, action_candidate_from_dict
from core_types import ActionCandidate, SetupError


@dataclass(frozen=True)
class ArchitectureAdversarialRecord:
    record_id: str
    layer: str
    prompt: str | None = None
    candidate: str | None = None
    action: ActionCandidate | None = None
    expected_status: str | None = None
    expected_verdict: str | None = None
    expected_clause: str | None = None


@dataclass(frozen=True)
class ArchitectureAdversarialCheck:
    path: Path
    total: int
    layers: dict[str, int]
    errors: list[str]

    def public_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "total": self.total,
            "layers": self.layers,
            "errors": self.errors,
        }


def validate_architecture_adversarial_record(record: Any, index: int) -> list[str]:
    prefix = f"line {index}"
    if not isinstance(record, dict):
        return [f"{prefix}: record must be an object"]

    errors: list[str] = []
    record_id = record.get("id")
    layer = record.get("layer")
    if not isinstance(record_id, str) or not record_id.strip():
        errors.append(f"{prefix}: id must be a non-empty string")
    if layer not in {"pipeline", "cold_eyes", "action"}:
        errors.append(f"{prefix}: layer must be pipeline, cold_eyes, or action")
        return errors

    if layer == "pipeline":
        if not isinstance(record.get("prompt"), str) or not record["prompt"].strip():
            errors.append(f"{prefix}: prompt must be a non-empty string for pipeline records")
        if record.get("expected_status") not in {"pass", "refused", "no_leak"}:
            errors.append(f"{prefix}: expected_status must be pass, refused, or no_leak")
        for field_name in ("candidate", "expected_verdict", "expected_clause", "expect_clean", "output"):
            if field_name in record:
                errors.append(f"{prefix}: {field_name} is not allowed in pipeline adversarial records")
        return errors

    if layer == "action":
        action = record.get("action")
        if not isinstance(action, dict):
            errors.append(f"{prefix}: action must be an object for action records")
        else:
            for field_name in ACTION_CANDIDATE_REQUIRED_FIELDS:
                if not isinstance(action.get(field_name), str) or not action[field_name].strip():
                    errors.append(f"{prefix}: action.{field_name} must be a non-empty string")
            extra_fields = set(action) - set(ACTION_CANDIDATE_REQUIRED_FIELDS)
            for field_name in sorted(extra_fields):
                errors.append(f"{prefix}: action.{field_name} is not allowed")
        expected_verdict = record.get("expected_verdict")
        expected_clause = record.get("expected_clause")
        if expected_verdict not in {"pass", "fail"}:
            errors.append(f"{prefix}: expected_verdict must be pass or fail")
        elif expected_verdict == "pass" and expected_clause is not None:
            errors.append(f"{prefix}: pass records must use expected_clause null")
        elif expected_verdict == "fail" and expected_clause not in {None, "C1", "C2", "C3"}:
            errors.append(f"{prefix}: fail records must use expected_clause null, C1, C2, or C3")
        for field_name in ("prompt", "candidate", "expected_status", "expect_clean", "output"):
            if field_name in record:
                errors.append(f"{prefix}: {field_name} is not allowed in action adversarial records")
        return errors

    if not isinstance(record.get("candidate"), str) or not record["candidate"].strip():
        errors.append(f"{prefix}: candidate must be a non-empty string for cold_eyes records")
    expected_verdict = record.get("expected_verdict")
    expected_clause = record.get("expected_clause")
    if expected_verdict not in {"pass", "fail"}:
        errors.append(f"{prefix}: expected_verdict must be pass or fail")
    elif expected_verdict == "pass" and expected_clause is not None:
        errors.append(f"{prefix}: pass records must use expected_clause null")
    elif expected_verdict == "fail" and expected_clause not in {"C1", "C2", "C3"}:
        errors.append(f"{prefix}: fail records must use expected_clause C1, C2, or C3")
    for field_name in ("prompt", "expect_clean", "output"):
        if field_name in record:
            errors.append(f"{prefix}: {field_name} is not allowed in cold_eyes adversarial records")
    return errors


def load_architecture_adversarial_records(
    path: Path,
) -> tuple[list[ArchitectureAdversarialRecord], list[str], int]:
    if not path.exists():
        raise SetupError(f"Architecture adversarial corpus not found: {path}")

    records: list[ArchitectureAdversarialRecord] = []
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

        record_errors = validate_architecture_adversarial_record(record, index)
        errors.extend(record_errors)
        if record_errors or not isinstance(record, dict):
            continue

        records.append(
            ArchitectureAdversarialRecord(
                record_id=record["id"].strip(),
                layer=record["layer"],
                prompt=record.get("prompt", "").strip() if record.get("prompt") is not None else None,
                candidate=(
                    record.get("candidate", "").strip()
                    if record.get("candidate") is not None
                    else None
                ),
                action=(
                    action_candidate_from_dict(record["action"])
                    if record.get("action") is not None
                    else None
                ),
                expected_status=record.get("expected_status"),
                expected_verdict=record.get("expected_verdict"),
                expected_clause=record.get("expected_clause"),
            )
        )

    if total == 0:
        errors.append("corpus is empty")
    return records, errors, total


def check_architecture_adversarial_corpus(path: Path) -> ArchitectureAdversarialCheck:
    records, errors, total = load_architecture_adversarial_records(path)
    layers = {"pipeline": 0, "cold_eyes": 0, "action": 0}
    for record in records:
        layers[record.layer] = layers.get(record.layer, 0) + 1
    return ArchitectureAdversarialCheck(path, total, layers, errors)


def apply_architecture_adversarial_requirements(
    result: ArchitectureAdversarialCheck,
    min_total: int = 0,
    min_layer: int = 0,
) -> ArchitectureAdversarialCheck:
    errors = list(result.errors)
    if result.total < min_total:
        errors.append(f"records below minimum: {result.total} < {min_total}")
    for layer in ("pipeline", "cold_eyes", "action"):
        count = result.layers.get(layer, 0)
        if count < min_layer:
            errors.append(f"{layer} records below minimum: {count} < {min_layer}")
    return ArchitectureAdversarialCheck(result.path, result.total, result.layers, errors)


def render_architecture_adversarial_check(result: ArchitectureAdversarialCheck) -> str:
    status = "ok" if not result.errors else "error"
    lines = [
        f"Architecture adversarial corpus: {result.path}",
        f"Status: {status}",
        f"Records: {result.total}",
        "Layers: pipeline={pipeline}, cold_eyes={cold_eyes}, action={action}".format(
            pipeline=result.layers.get("pipeline", 0),
            cold_eyes=result.layers.get("cold_eyes", 0),
            action=result.layers.get("action", 0),
        ),
    ]
    if result.errors:
        lines.extend(["", "Errors:"])
        lines.extend(f"- {error}" for error in result.errors)
    return "\n".join(lines)
