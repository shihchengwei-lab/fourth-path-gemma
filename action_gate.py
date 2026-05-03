from __future__ import annotations

import re
from typing import Any

from core_types import ActionCandidate, ColdEyesVerdict


SIDE_EFFECT_BOUNDARY_POLICY = """This prototype does not let the Main Agent execute tools, write files, send network requests, or perform other external side effects.
Any future tool call or external side effect must first be represented as an auditable action candidate.
Unaudited side effects must fail closed before execution.
Cold Eyes output review is not sufficient for side effects that happen before output.
"""

ACTION_CANDIDATE_REQUIRED_FIELDS: tuple[str, ...] = (
    "action_type",
    "target",
    "intent",
    "args_summary",
    "risk_surface",
)

SENSITIVE_READ_TARGET_PATTERNS: tuple[str, ...] = (
    r"(^|/)\.git(/|$)",
    r"(^|/)\.env(\.|$)",
    r"(^|/)auth\.json$",
    r"(^|/)id_(rsa|ed25519)$",
    r"(^|/)\.ssh(/|$)",
    r"(^|/)credentials?(\.(json|toml|ya?ml)|$)",
    r"(^|/)secrets?(\.(json|toml|ya?ml)|$)",
    r"(^|/)service[-_]?account(\.[^/]+)?$",
    r"(^|/)[^/]*(api[-_]?key|private[-_]?key|secret|credential|token)[^/]*\.(env|json|key|pem|txt|toml|ya?ml)$",
    r"\.(key|pem)$",
)


def _matches_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def action_candidate_from_dict(payload: dict[str, Any]) -> ActionCandidate:
    return ActionCandidate(
        action_type=str(payload["action_type"]).strip(),
        target=str(payload["target"]).strip(),
        intent=str(payload["intent"]).strip(),
        args_summary=str(payload["args_summary"]).strip(),
        risk_surface=str(payload["risk_surface"]).strip(),
    )


def action_candidate_text(action: ActionCandidate) -> str:
    return "\n".join(
        [
            f"action_type: {action.action_type}",
            f"target: {action.target}",
            f"intent: {action.intent}",
            f"args_summary: {action.args_summary}",
            f"risk_surface: {action.risk_surface}",
        ]
    )


def read_file_target_scope_issue(target: str) -> str | None:
    normalized = target.strip().replace("\\", "/")
    lowered = normalized.lower()
    if not lowered:
        return "read_target_empty"
    if any(part == ".." for part in lowered.split("/")):
        return "read_target_parent_traversal"
    if _matches_any(lowered, SENSITIVE_READ_TARGET_PATTERNS):
        return "read_target_sensitive_file"
    if re.match(r"^[a-z]:/", lowered) or lowered.startswith(("/", "//", "~")):
        return "read_target_not_project_relative"
    return None


def audit_action_candidate(action: ActionCandidate | None) -> ColdEyesVerdict:
    if action is None:
        return ColdEyesVerdict("fail", None, "action_candidate_missing", "", source="action_fail_closed")

    mechanical = mechanical_action_audit(action)
    if mechanical is not None:
        return mechanical

    return ColdEyesVerdict(
        "fail",
        None,
        "action_audit_no_pass_rule",
        action_candidate_text(action),
        source="action_fail_closed",
    )


def action_audit_data(action: ActionCandidate) -> dict[str, Any]:
    verdict = audit_action_candidate(action)
    return {
        "approved": verdict.verdict == "pass",
        "verdict": verdict.verdict,
        "canon_clause": verdict.canon_clause,
        "reason": verdict.reason,
        "source": verdict.source,
        "action_type": action.action_type,
        "risk_surface": action.risk_surface,
    }


def render_action_audit(data: dict[str, Any]) -> str:
    status = "approved" if data["approved"] else "blocked"
    clause = data["canon_clause"] if data["canon_clause"] is not None else "none"
    return "\n".join(
        [
            f"Action audit: {status}",
            f"Verdict: {data['verdict']}",
            f"Canon clause: {clause}",
            f"Source: {data['source']}",
            f"Reason: {data['reason']}",
        ]
    )


def mechanical_action_audit(action: ActionCandidate) -> ColdEyesVerdict | None:
    text = action_candidate_text(action).lower()
    action_type = action.action_type.strip().lower()
    risk_surface = action.risk_surface.strip().lower()

    if _matches_any(
        text,
        (
            r"\b(weapon|explosive|bomb|poison|physical trap)\b",
            r"\b(harm|injure|kill)\b.{0,80}\b(person|someone|target)\b",
        ),
    ):
        return ColdEyesVerdict("fail", "C1", "action_mechanical_c1_high_confidence", text, source="action_mechanical")

    if _matches_any(
        text,
        (
            r"\b(phishing|impersonat(e|ing|ion)|fake support|pretext)\b",
            r"\btrick\b.{0,80}\b(reveal|share|enter)\b.{0,40}\b(password|credential|code|otp|token)\b",
        ),
    ):
        return ColdEyesVerdict("fail", "C3", "action_mechanical_c3_high_confidence", text, source="action_mechanical")

    if _matches_any(
        text,
        (
            r"\b(secret|credential|token|api key|cookie|session)\b.{0,80}\b(exfiltrate|upload|post|send|leak)\b",
            r"\b(exfiltrate|upload|post|send|leak)\b.{0,80}\b(secret|credential|token|api key|cookie|session)\b",
            r"\b(copy|steal|extract)\b.{0,80}\b(cookie|session token|credential|token)\b",
            r"\b(remove-item|delete|del |erase|rm -rf)\b",
            r"\b(canon|audit|refusal|review criteria|audit log)\b.{0,80}\b(write|modify|delete|overwrite|patch)\b",
            r"\b(write|modify|delete|overwrite|patch)\b.{0,80}\b(canon|audit|refusal|review criteria|audit log)\b",
            r"\b(policy|safety layer|action gate|gate|control[- ]?plane)\b.{0,80}\b(write|modify|delete|overwrite|patch|disable|weaken)\b",
            r"\b(write|modify|delete|overwrite|patch|disable|weaken)\b.{0,80}\b(policy|safety layer|action gate|gate|control[- ]?plane)\b",
        ),
    ):
        return ColdEyesVerdict("fail", "C2", "action_mechanical_c2_high_confidence", text, source="action_mechanical")

    if action_type == "noop" and risk_surface in {"none", "no_external_effect"}:
        return ColdEyesVerdict("pass", None, "action_mechanical_noop", text, source="action_mechanical")

    if action_type == "read_file":
        read_issue = read_file_target_scope_issue(action.target)
        if read_issue == "read_target_sensitive_file":
            return ColdEyesVerdict("fail", "C2", read_issue, text, source="action_mechanical")
        if read_issue is not None:
            return ColdEyesVerdict("fail", None, read_issue, text, source="action_fail_closed")
        if risk_surface == "read_only_project_file":
            return ColdEyesVerdict("pass", None, "action_mechanical_read_only_project_file", text, source="action_mechanical")

    return None
