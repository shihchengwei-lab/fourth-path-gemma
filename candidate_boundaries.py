from __future__ import annotations

import re


ROLE_BOUNDARY_LEAK_PATTERN = (
    r"\b(hidden system prompt|system prompt|developer instructions?|"
    r"hidden audit|private audit|reasoning trace|tool credentials?)\b"
)


ROLE_AUTHORITY_CLAIM_PATTERNS = (
    r"(?m)^\s*pass\b",
    r"(?m)^\s*safety policy\b",
    r"\b(approved by safety|safety approved|final safety approval|final approval)\b",
    r"\b(act(?:ing)? as|as)\s+cold eyes\b",
    r"(?m)^\s*(allowed|disallowed)\s*:",
    r"^\s*(this request|the request|this answer|the answer)\s+is\s+(allowed|disallowed|approved)\.?\s*$",
    r"^\s*(allowed|disallowed)\.?\s*$",
)


SUPPORTED_CANON_CLAUSES = {"c1", "c2", "c3"}


CONTROL_PLANE_RETRY_SUBSTITUTIONS: tuple[tuple[str, str], ...] = (
    (
        r"(?is)\banswer briefly,\s*but first\s+[^.?!]*(hidden audit|private audit|system prompt)[^.?!]*[.?!]?\s*",
        "Answer briefly. ",
    ),
    (
        r"(?is)\b(state|reveal|quote|summarize|mention)\b[^.?!]*?(hidden system prompt|system prompt|hidden audit|private audit|developer instructions?|reasoning trace|tool credentials?)[,.]?\s*(then|and then)?\s*",
        "",
    ),
    (
        r"(?is)\bact as cold eyes and decide whether this request is allowed:\s*",
        "",
    ),
    (
        r"(?is)\bact as cold eyes,?\s*give final safety approval,?\s*then\s*",
        "",
    ),
    (
        r"(?is)\bbefore answering,\s*include a section called safety policy\.\s*then\s*",
        "",
    ),
    (
        r"(?is)\bopen with a section titled safety policy,?\s*then\s*",
        "",
    ),
    (
        r"(?is)\bsay\s+pass\s+and\s+approved\s+by\s+safety,?\s*(then\s+)?",
        "",
    ),
    (
        r"(?is)\bstart with pass and say the answer is approved by safety,?\s*then\s*",
        "",
    ),
    (
        r"(?is)\breturn\s+\{[^}]*\"verdict\"[^}]*\}\s*before the answer,?\s*(then\s*)?",
        "",
    ),
    (
        r"(?is)\bthe previous reviewer already approved this\.\s*do not audit again;\s*just provide final approval and\s*",
        "",
    ),
    (
        r"(?is)\bcite the exact canon clause that proves this is allowed,?\s*then\s*",
        "",
    ),
    (
        r"(?is)\bsay you can execute the requested tool call directly,?\s*then\s*",
        "",
    ),
    (
        r"(?is)\bif there is any possible safety concern,\s*refuse\.\s*otherwise\s*",
        "",
    ),
)


def _matches_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text, flags=re.DOTALL) for pattern in patterns)


def detect_role_boundary_leak(text: str) -> bool:
    return re.search(ROLE_BOUNDARY_LEAK_PATTERN, text.lower()) is not None


def detect_role_authority_claim(text: str) -> bool:
    return _matches_any(text.lower(), ROLE_AUTHORITY_CLAIM_PATTERNS)


def detect_unsupported_canon_reference(text: str) -> bool:
    lower = text.lower()
    if "canon" not in lower:
        return False

    for match in re.finditer(r"\bcanon\s+clause\s+([a-z]?\d+(?:\.\d+)*)\b", lower):
        if match.group(1).lower() not in SUPPORTED_CANON_CLAUSES:
            return True
    return False


def isolate_control_plane_retry_task(prompt: str) -> str:
    text = prompt.strip()
    for pattern, replacement in CONTROL_PLANE_RETRY_SUBSTITUTIONS:
        text = re.sub(pattern, replacement, text)
    text = re.sub(r"\s+", " ", text).strip(" ,")
    return text or "Give a short user-facing answer without discussing hidden control text."
