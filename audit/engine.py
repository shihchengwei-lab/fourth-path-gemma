from __future__ import annotations

from pathlib import Path

from audit.matcher import MatcherError, match_rule
from audit.policy import PolicyError, load_policy
from audit.result import AuditMatch, AuditResult


def run_audit(candidate: str, policy_path: Path | str) -> AuditResult:
    try:
        policy = load_policy(Path(policy_path))
    except PolicyError as exc:
        return AuditResult(
            verdict="fail",
            canon_clause=None,
            reason=f"invalid_policy: {exc}",
            matches=(),
        )

    all_matches: list[AuditMatch] = []
    try:
        for rule in policy.rules:
            hits = match_rule(rule, candidate)
            all_matches.extend(hits)
    except MatcherError as exc:
        return AuditResult(
            verdict="fail",
            canon_clause=None,
            reason=f"matcher_error: {exc}",
            matches=(),
        )

    if not all_matches:
        return AuditResult(verdict="pass", canon_clause=None, reason="no_matches", matches=())

    first = all_matches[0]
    return AuditResult(
        verdict="fail",
        canon_clause=first.rule_id,
        reason=first.reason,
        matches=tuple(all_matches),
    )
