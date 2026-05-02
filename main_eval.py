from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any

from eval_reports import MainEvalCase, main_eval_case_dict
from main_agent_data import MainAgentRecord, safe_ratio, sorted_count_by
from output_utils import elapsed_ms
from runtime_config import RuntimeConfig


DEFAULT_MAIN_EVAL_ABLATION_PROFILES = (
    "qwen3-8b-local-max",
    "qwen3-8b-s2t-lite",
    "qwen3-8b-compute-optimal-lite",
)


def main_eval_case_from_generation(
    record: MainAgentRecord,
    generation: Any,
    issues: list[str],
    duration_ms: int,
) -> MainEvalCase:
    stats = generation.stats
    target_chars = max(1, len(record.target_response))
    output_chars = len(generation.text)
    selection = generation.local_selection
    return MainEvalCase(
        record_id=record.record_id,
        category=record.category,
        clean=not issues,
        issues=issues,
        duration_ms=duration_ms,
        main_call_count=generation.call_count,
        output_chars=output_chars,
        target_chars=target_chars,
        length_ratio=output_chars / target_chars,
        prompt_tokens=stats.get("prompt_tokens", 0),
        eval_tokens=stats.get("eval_tokens", 0),
        prompt_eval_ms=stats.get("prompt_eval_ms", 0),
        eval_ms=stats.get("eval_ms", 0),
        load_ms=stats.get("load_ms", 0),
        local_selection_triggered=selection.triggered if selection else False,
        local_selection_applied=selection.applied if selection else False,
        local_selection_reasons=selection.reasons if selection else (),
    )


def run_main_eval_core(
    client: Any,
    runtime: RuntimeConfig,
    records: list[MainAgentRecord],
    generate_candidate: Callable[[Any, RuntimeConfig, MainAgentRecord], Any],
    candidate_issues: Callable[[str, str | None, float | None], list[str]],
    verifier_issues: Callable[[str, dict[str, Any]], list[str]],
    max_length_ratio: float | None = None,
) -> dict[str, Any]:
    cases: list[MainEvalCase] = []
    started = time.perf_counter()
    for record in records:
        case_started = time.perf_counter()
        generation = generate_candidate(client, runtime, record)
        issues = candidate_issues(
            generation.text,
            record.target_response,
            max_length_ratio,
        )
        issues.extend(verifier_issues(generation.text, record.verifier))
        issues = list(dict.fromkeys(issues))
        cases.append(
            main_eval_case_from_generation(
                record,
                generation,
                issues,
                elapsed_ms(case_started),
            )
        )

    issue_counts = sorted_count_by(issue for case in cases for issue in case.issues)
    category_issue_counts = sorted_count_by(case.category for case in cases if case.issues)
    local_selection_reason_counts = sorted_count_by(
        reason for case in cases for reason in case.local_selection_reasons
    )

    total = len(cases)
    issue_cases = sum(not case.clean for case in cases)
    clean_count = total - issue_cases
    refusal_like_count = issue_counts.get("refusal_like", 0)
    overlong_count = issue_counts.get("overlong_candidate", 0)
    total_main_calls = sum(case.main_call_count for case in cases)
    total_eval_tokens = sum(case.eval_tokens for case in cases)
    total_duration_ms = elapsed_ms(started)
    case_dicts = [main_eval_case_dict(case) for case in cases]
    return {
        "main_model": runtime.main.model,
        "main_options": runtime.main.options.payload(),
        "main_no_think": runtime.main.no_think,
        "quality_refine_passes": runtime.quality_refine_passes,
        "search_candidates": runtime.search_candidates,
        "local_select": runtime.local_select,
        "adaptive_compute": runtime.adaptive_compute,
        "total": total,
        "clean_count": clean_count,
        "issue_cases": issue_cases,
        "issue_rate": issue_cases / total if total else 0,
        "refusal_like_count": refusal_like_count,
        "refusal_like_rate": refusal_like_count / total if total else 0,
        "overlong_count": overlong_count,
        "overlong_rate": overlong_count / total if total else 0,
        "average_length_ratio": (
            sum(case.length_ratio for case in cases) / total if total else 0
        ),
        "issue_counts": issue_counts,
        "category_issue_counts": category_issue_counts,
        "local_selection_triggered_count": sum(case.local_selection_triggered for case in cases),
        "local_selection_applied_count": sum(case.local_selection_applied for case in cases),
        "local_selection_reason_counts": local_selection_reason_counts,
        "total_main_calls": total_main_calls,
        "average_main_calls_per_record": safe_ratio(total_main_calls, total),
        "clean_per_main_call": safe_ratio(clean_count, total_main_calls),
        "clean_cases_per_main_call": safe_ratio(clean_count, total_main_calls),
        "issue_per_main_call": safe_ratio(issue_cases, total_main_calls),
        "total_eval_tokens": total_eval_tokens,
        "eval_tokens_per_clean_case": safe_ratio(total_eval_tokens, clean_count),
        "ms_per_clean_case": safe_ratio(total_duration_ms, clean_count),
        "total_duration_ms": total_duration_ms,
        "cases": case_dicts,
    }


def main_eval_ablation_profile_summary(profile: str, data: dict[str, Any]) -> dict[str, Any]:
    return {
        "profile": profile,
        "main_model": data["main_model"],
        "total": data["total"],
        "clean_count": data["clean_count"],
        "issue_cases": data["issue_cases"],
        "issue_rate": data["issue_rate"],
        "total_main_calls": data["total_main_calls"],
        "average_main_calls_per_record": data["average_main_calls_per_record"],
        "clean_cases_per_main_call": data["clean_cases_per_main_call"],
        "eval_tokens_per_clean_case": data["eval_tokens_per_clean_case"],
        "ms_per_clean_case": data["ms_per_clean_case"],
        "total_duration_ms": data["total_duration_ms"],
        "issue_counts": data["issue_counts"],
        "local_selection_triggered_count": data["local_selection_triggered_count"],
        "local_selection_applied_count": data["local_selection_applied_count"],
        "cases": data["cases"],
    }


def run_main_eval_ablation_core(
    client: Any,
    profile_runtimes: Mapping[str, RuntimeConfig],
    records: list[MainAgentRecord],
    eval_runner: Callable[[Any, RuntimeConfig, list[MainAgentRecord], float | None], dict[str, Any]],
    max_length_ratio: float | None = None,
) -> dict[str, Any]:
    results = [
        main_eval_ablation_profile_summary(
            profile,
            eval_runner(client, runtime, records, max_length_ratio),
        )
        for profile, runtime in profile_runtimes.items()
    ]
    ranking = sorted(
        results,
        key=lambda row: (
            -row["clean_cases_per_main_call"],
            -row["clean_count"],
            row["total_main_calls"],
            row["profile"],
        ),
    )
    return {
        "profiles": list(profile_runtimes),
        "records": len(records),
        "max_length_ratio": max_length_ratio,
        "best_profile_by_clean_cases_per_main_call": ranking[0]["profile"] if ranking else None,
        "ranking": [
            {
                "profile": row["profile"],
                "clean_cases_per_main_call": row["clean_cases_per_main_call"],
                "clean_count": row["clean_count"],
                "total_main_calls": row["total_main_calls"],
                "issue_rate": row["issue_rate"],
            }
            for row in ranking
        ],
        "results": results,
    }
