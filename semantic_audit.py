from __future__ import annotations

from typing import Any

from main import ColdEyesVerdict, cold_eyes_review


def semantic_audit_subagent_review(
    client: Any,
    model: str,
    canon: str,
    candidate: str,
) -> ColdEyesVerdict:
    """Legacy LLM semantic audit path.

    This wrapper intentionally preserves the old LLM-based review behavior while
    making its architectural role explicit: it is a semantic audit subagent path,
    not the default final Cold Eyes adjudicator.

    The default runtime final audit is mechanical and lives in
    main.mechanical_cold_eyes_review via run_pipeline().
    """
    return cold_eyes_review(
        client=client,
        model=model,
        canon=canon,
        candidate=candidate,
    )
