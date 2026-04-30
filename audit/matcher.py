from __future__ import annotations

import re

from audit.policy import PatternSpec, RuleSpec
from audit.result import AuditMatch


class MatcherError(RuntimeError):
    pass


def compile_pattern(pat: PatternSpec) -> re.Pattern[str]:
    flags = 0
    if pat.flags.get("ignorecase"):
        flags |= re.IGNORECASE
    if pat.flags.get("dotall"):
        flags |= re.DOTALL
    if pat.flags.get("multiline"):
        flags |= re.MULTILINE
    try:
        return re.compile(pat.regex, flags)
    except re.error as exc:
        raise MatcherError(f"invalid regex in pattern {pat.id!r}: {exc}") from exc


def match_rule(rule: RuleSpec, text: str) -> list[AuditMatch]:
    hits: list[AuditMatch] = []
    for pat in rule.patterns:
        compiled = compile_pattern(pat)
        m = compiled.search(text)
        if m:
            hits.append(
                AuditMatch(
                    rule_id=rule.id,
                    pattern_id=pat.id,
                    severity=rule.severity,
                    reason=rule.reason,
                    start=m.start(),
                    end=m.end(),
                )
            )
    return hits
