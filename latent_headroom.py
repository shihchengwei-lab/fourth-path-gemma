from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable, Iterable
from typing import Any

from main_agent_data import MainAgentRecord, safe_ratio, sorted_count_by
from output_utils import elapsed_ms
from runtime_config import RuntimeConfig


DEFAULT_LATENT_HEADROOM_VARIANTS = (
    "baseline",
    "constraint_first",
    "self_check",
)

LATENT_HEADROOM_VARIANT_DESCRIPTIONS = {
    "baseline": "Original prompt through the current Main Agent prompt stack.",
    "constraint_first": "Front-load format, unit, count, and allowed-content constraints.",
    "self_check": "Ask the model to silently check arithmetic, code behavior, and format before final output.",
}


def latent_prompt_variant(prompt: str, variant: str) -> str:
    text = prompt.strip()
    if variant == "baseline":
        return text
    if variant == "constraint_first":
        return (
            "Follow every explicit output constraint first: format, count, units, "
            "allowed content, and forbidden content. Return only the final answer.\n\n"
            f"{text}"
        )
    if variant == "self_check":
        return (
            "Before finalizing, silently check arithmetic, code behavior, and exact "
            "format constraints. Return only the final answer.\n\n"
            f"{text}"
        )
    raise ValueError(f"Unsupported latent headroom variant: {variant}")


def normalize_latent_headroom_variants(variants: Iterable[str] | None) -> tuple[str, ...]:
    selected = tuple(variants or DEFAULT_LATENT_HEADROOM_VARIANTS)
    unknown = sorted(set(selected) - set(LATENT_HEADROOM_VARIANT_DESCRIPTIONS))
    if unknown:
        raise ValueError(f"Unsupported latent headroom variant(s): {', '.join(unknown)}")
    if not selected:
        raise ValueError("At least one latent headroom variant is required.")
    return selected


def latent_headroom_attempt_dict(
    *,
    variant: str,
    attempt: int,
    clean: bool,
    issues: list[str],
    generation: Any,
) -> dict[str, Any]:
    stats = generation.stats
    selection = generation.local_selection
    return {
        "variant": variant,
        "attempt": attempt,
        "clean": clean,
        "issues": issues,
        "main_call_count": generation.call_count,
        "candidate_count": generation.candidate_count,
        "compute_strategy": generation.compute_strategy,
        "eval_tokens": stats.get("eval_tokens", 0),
        "eval_ms": stats.get("eval_ms", 0),
        "load_ms": stats.get("load_ms", 0),
        "local_selection_triggered": selection.triggered if selection else False,
        "local_selection_applied": selection.applied if selection else False,
        "local_selection_reasons": selection.reasons if selection else (),
    }


def latent_headroom_record_summary(
    record: MainAgentRecord,
    attempts: list[dict[str, Any]],
) -> dict[str, Any]:
    first = attempts[0]
    clean_attempts = [attempt for attempt in attempts if attempt["clean"]]
    any_clean = bool(clean_attempts)
    first_pass_clean = bool(first["clean"])
    issue_counts = sorted_count_by(issue for attempt in attempts for issue in attempt["issues"])
    first_clean_index = next(
        (index for index, attempt in enumerate(attempts, 1) if attempt["clean"]),
        None,
    )
    return {
        "id": record.record_id,
        "category": record.category,
        "first_pass_variant": first["variant"],
        "first_pass_clean": first_pass_clean,
        "any_clean": any_clean,
        "latent_rescued": (not first_pass_clean) and any_clean,
        "stable_clean": len(clean_attempts) == len(attempts),
        "attempt_clean_count": len(clean_attempts),
        "total_attempts": len(attempts),
        "attempts_to_first_clean": first_clean_index,
        "clean_variants": sorted({attempt["variant"] for attempt in clean_attempts}),
        "issue_counts": issue_counts,
        "attempts": attempts,
    }


def run_latent_headroom_probe(
    *,
    client: Any,
    runtime: RuntimeConfig,
    records: list[MainAgentRecord],
    generate_candidate: Callable[[Any, RuntimeConfig, MainAgentRecord], Any],
    candidate_issues: Callable[[str, str | None, float | None], list[str]],
    verifier_issues: Callable[[str, dict[str, Any]], list[str]],
    attempts_per_variant: int = 2,
    variants: Iterable[str] | None = None,
    max_length_ratio: float | None = None,
) -> dict[str, Any]:
    if attempts_per_variant < 1:
        raise ValueError("attempts_per_variant must be at least 1.")
    selected_variants = normalize_latent_headroom_variants(variants)
    started = time.perf_counter()
    record_summaries: list[dict[str, Any]] = []

    for record in records:
        attempts: list[dict[str, Any]] = []
        for variant in selected_variants:
            variant_record = MainAgentRecord(
                record_id=record.record_id,
                category=record.category,
                prompt=latent_prompt_variant(record.prompt, variant),
                target_response=record.target_response,
                verifier=record.verifier,
            )
            for attempt_number in range(1, attempts_per_variant + 1):
                generation = generate_candidate(client, runtime, variant_record)
                issues = candidate_issues(generation.text, record.target_response, max_length_ratio)
                issues.extend(verifier_issues(generation.text, record.verifier))
                issues = list(dict.fromkeys(issues))
                attempts.append(
                    latent_headroom_attempt_dict(
                        variant=variant,
                        attempt=attempt_number,
                        clean=not issues,
                        issues=issues,
                        generation=generation,
                    )
                )
        record_summaries.append(latent_headroom_record_summary(record, attempts))

    total_records = len(record_summaries)
    first_pass_clean_count = sum(record["first_pass_clean"] for record in record_summaries)
    any_clean_count = sum(record["any_clean"] for record in record_summaries)
    latent_rescue_count = sum(record["latent_rescued"] for record in record_summaries)
    never_clean_count = sum(not record["any_clean"] for record in record_summaries)
    stable_clean_count = sum(record["stable_clean"] for record in record_summaries)
    total_attempts = sum(record["total_attempts"] for record in record_summaries)
    total_main_calls = sum(
        attempt["main_call_count"]
        for record in record_summaries
        for attempt in record["attempts"]
    )
    total_eval_tokens = sum(
        attempt["eval_tokens"]
        for record in record_summaries
        for attempt in record["attempts"]
    )
    issue_counts = sorted_count_by(
        issue
        for record in record_summaries
        for attempt in record["attempts"]
        for issue in attempt["issues"]
    )
    category_counts = sorted_count_by(record["category"] for record in record_summaries)
    category_headroom_counts = sorted_count_by(
        record["category"] for record in record_summaries if record["latent_rescued"]
    )
    variant_clean_counts = Counter()
    variant_attempt_counts = Counter()
    for record in record_summaries:
        for attempt in record["attempts"]:
            variant_attempt_counts[attempt["variant"]] += 1
            if attempt["clean"]:
                variant_clean_counts[attempt["variant"]] += 1

    first_pass_misses = total_records - first_pass_clean_count
    return {
        "main_model": runtime.main.model,
        "main_options": runtime.main.options.payload(),
        "main_no_think": runtime.main.no_think,
        "local_select": runtime.local_select,
        "quality_refine_passes": runtime.quality_refine_passes,
        "search_candidates": runtime.search_candidates,
        "adaptive_compute": runtime.adaptive_compute,
        "variants": list(selected_variants),
        "variant_descriptions": {
            name: LATENT_HEADROOM_VARIANT_DESCRIPTIONS[name] for name in selected_variants
        },
        "attempts_per_variant": attempts_per_variant,
        "total_records": total_records,
        "total_attempts": total_attempts,
        "total_main_calls": total_main_calls,
        "total_eval_tokens": total_eval_tokens,
        "first_pass_clean_count": first_pass_clean_count,
        "first_pass_clean_rate": safe_ratio(first_pass_clean_count, total_records),
        "any_clean_count": any_clean_count,
        "any_clean_rate": safe_ratio(any_clean_count, total_records),
        "latent_rescue_count": latent_rescue_count,
        "latent_rescue_rate": safe_ratio(latent_rescue_count, total_records),
        "latent_rescue_rate_among_first_pass_misses": safe_ratio(latent_rescue_count, first_pass_misses),
        "never_clean_count": never_clean_count,
        "never_clean_rate": safe_ratio(never_clean_count, total_records),
        "stable_clean_count": stable_clean_count,
        "stable_clean_rate": safe_ratio(stable_clean_count, total_records),
        "attempt_clean_count": sum(record["attempt_clean_count"] for record in record_summaries),
        "attempt_clean_rate": safe_ratio(
            sum(record["attempt_clean_count"] for record in record_summaries),
            total_attempts,
        ),
        "records_per_main_call_any_clean": safe_ratio(any_clean_count, total_main_calls),
        "issue_counts": issue_counts,
        "category_counts": category_counts,
        "category_headroom_counts": category_headroom_counts,
        "variant_clean_counts": dict(sorted(variant_clean_counts.items())),
        "variant_attempt_counts": dict(sorted(variant_attempt_counts.items())),
        "records": record_summaries,
        "total_duration_ms": elapsed_ms(started),
    }


def render_latent_headroom_probe(data: dict[str, Any], path: Any) -> str:
    lines = [
        f"Main latent headroom probe: {path}",
        f"Model: {data['main_model']}",
        f"Records: {data['total_records']}",
        f"Variants: {', '.join(data['variants'])}",
        f"Attempts/variant: {data['attempts_per_variant']}",
        (
            "First-pass clean: {count}/{total} ({rate:.3f})"
        ).format(
            count=data["first_pass_clean_count"],
            total=data["total_records"],
            rate=data["first_pass_clean_rate"],
        ),
        (
            "Any-clean after probe: {count}/{total} ({rate:.3f})"
        ).format(
            count=data["any_clean_count"],
            total=data["total_records"],
            rate=data["any_clean_rate"],
        ),
        (
            "Latent rescued: {count}/{total} ({rate:.3f}; miss-rescue={miss_rate:.3f})"
        ).format(
            count=data["latent_rescue_count"],
            total=data["total_records"],
            rate=data["latent_rescue_rate"],
            miss_rate=data["latent_rescue_rate_among_first_pass_misses"],
        ),
        (
            "Stable clean: {count}/{total} ({rate:.3f})"
        ).format(
            count=data["stable_clean_count"],
            total=data["total_records"],
            rate=data["stable_clean_rate"],
        ),
        (
            "Never clean: {count}/{total} ({rate:.3f})"
        ).format(
            count=data["never_clean_count"],
            total=data["total_records"],
            rate=data["never_clean_rate"],
        ),
        f"Main calls: {data['total_main_calls']}",
        f"Attempt clean rate: {data['attempt_clean_rate']:.3f}",
    ]
    if data["issue_counts"]:
        lines.append("Attempt issue counts:")
        lines.extend(f"- {issue}: {count}" for issue, count in data["issue_counts"].items())
    if data["category_headroom_counts"]:
        lines.append("Latent-rescue categories:")
        lines.extend(f"- {category}: {count}" for category, count in data["category_headroom_counts"].items())
    return "\n".join(lines)
