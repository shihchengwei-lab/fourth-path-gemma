from __future__ import annotations

from pathlib import Path

from audit.matcher import MatcherError, match_rule
from audit.policy import PolicyError, load_policy
from audit.result import AuditMatch, AuditResult

CLAUSE_PRECEDENCE: tuple[str, ...] = ("C2", "C3", "C1")


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

    selected = select_canon_match(all_matches)
    return AuditResult(
        verdict="fail",
        canon_clause=selected.rule_id,
        reason=selected.reason,
        matches=tuple(all_matches),
    )


def select_canon_match(matches: list[AuditMatch]) -> AuditMatch:
    priority = {clause: index for index, clause in enumerate(CLAUSE_PRECEDENCE)}
    return min(matches, key=lambda match: priority.get(match.rule_id, len(priority)))
