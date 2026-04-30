from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AuditMatch:
    rule_id: str
    pattern_id: str
    severity: str
    reason: str
    start: int | None = None
    end: int | None = None


@dataclass(frozen=True)
class AuditResult:
    verdict: str  # "pass" | "fail"
    canon_clause: str | None
    reason: str
    matches: tuple[AuditMatch, ...]
