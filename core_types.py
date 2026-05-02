from __future__ import annotations

from dataclasses import dataclass


class SetupError(RuntimeError):
    """Raised when the local model runtime is not ready."""


class PipelineError(RuntimeError):
    """Raised when the pipeline cannot complete a requested run."""


@dataclass(frozen=True)
class ColdEyesVerdict:
    verdict: str
    canon_clause: str | None
    reason: str
    raw: str
    source: str = "llm"


@dataclass(frozen=True)
class ActionCandidate:
    action_type: str
    target: str
    intent: str
    args_summary: str
    risk_surface: str
