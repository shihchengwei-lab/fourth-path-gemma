from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

VALID_SEVERITIES: frozenset[str] = frozenset({"critical", "major", "minor"})


@dataclass(frozen=True)
class PatternSpec:
    id: str
    regex: str
    flags: dict[str, bool]


@dataclass(frozen=True)
class RuleSpec:
    id: str
    severity: str
    reason: str
    patterns: tuple[PatternSpec, ...]


@dataclass(frozen=True)
class Policy:
    version: str
    rules: tuple[RuleSpec, ...]


class PolicyError(ValueError):
    pass


def load_policy(path: Path) -> Policy:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise PolicyError(f"cannot read policy file: {exc}") from exc

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PolicyError(f"invalid policy JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise PolicyError("policy must be a JSON object")

    version = data.get("version")
    if not isinstance(version, str) or not version.strip():
        raise PolicyError("policy missing required string field: version")

    raw_rules = data.get("rules")
    if not isinstance(raw_rules, list):
        raise PolicyError("policy missing required list field: rules")

    rules = [_parse_rule(raw, i) for i, raw in enumerate(raw_rules)]
    return Policy(version=version.strip(), rules=tuple(rules))


def _parse_rule(raw: Any, index: int) -> RuleSpec:
    if not isinstance(raw, dict):
        raise PolicyError(f"rule[{index}] is not an object")

    rule_id = raw.get("id")
    if not isinstance(rule_id, str) or not rule_id.strip():
        raise PolicyError(f"rule[{index}] missing required string field: id")
    rule_id = rule_id.strip()

    severity = raw.get("severity")
    if not isinstance(severity, str):
        raise PolicyError(f"rule {rule_id!r}: missing required string field: severity")
    if severity not in VALID_SEVERITIES:
        raise PolicyError(f"rule {rule_id!r}: unknown severity {severity!r}")

    reason = raw.get("reason", "")
    if not isinstance(reason, str):
        raise PolicyError(f"rule {rule_id!r}: reason must be a string")

    raw_patterns = raw.get("patterns")
    if not isinstance(raw_patterns, list):
        raise PolicyError(f"rule {rule_id!r}: missing required list field: patterns")

    patterns = [_parse_pattern(p, rule_id, j) for j, p in enumerate(raw_patterns)]
    return RuleSpec(id=rule_id, severity=severity, reason=reason, patterns=tuple(patterns))


def _parse_pattern(raw: Any, rule_id: str, index: int) -> PatternSpec:
    if not isinstance(raw, dict):
        raise PolicyError(f"rule {rule_id!r} pattern[{index}]: not an object")

    pat_id = raw.get("id")
    if not isinstance(pat_id, str) or not pat_id.strip():
        raise PolicyError(f"rule {rule_id!r} pattern[{index}]: missing required string field: id")
    pat_id = pat_id.strip()

    regex = raw.get("regex")
    if not isinstance(regex, str) or not regex:
        raise PolicyError(f"rule {rule_id!r} pattern {pat_id!r}: missing required string field: regex")

    raw_flags = raw.get("flags", {})
    if not isinstance(raw_flags, dict):
        raise PolicyError(f"rule {rule_id!r} pattern {pat_id!r}: flags must be an object")

    flags: dict[str, bool] = {
        "ignorecase": bool(raw_flags.get("ignorecase", False)),
        "dotall": bool(raw_flags.get("dotall", False)),
        "multiline": bool(raw_flags.get("multiline", False)),
    }
    return PatternSpec(id=pat_id, regex=regex, flags=flags)
