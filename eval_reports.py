from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from output_utils import write_json_summary


@dataclass(frozen=True)
class MainEvalCase:
    record_id: str
    category: str
    clean: bool
    issues: list[str]
    duration_ms: int
    main_call_count: int
    output_chars: int
    target_chars: int
    length_ratio: float
    prompt_tokens: int
    eval_tokens: int
    prompt_eval_ms: int
    eval_ms: int
    load_ms: int
    local_selection_triggered: bool = False
    local_selection_applied: bool = False
    local_selection_reasons: tuple[str, ...] = ()


def main_eval_case_dict(case: MainEvalCase) -> dict[str, Any]:
    return {
        "id": case.record_id,
        "category": case.category,
        "clean": case.clean,
        "issues": case.issues,
        "duration_ms": case.duration_ms,
        "main_call_count": case.main_call_count,
        "output_chars": case.output_chars,
        "target_chars": case.target_chars,
        "length_ratio": round(case.length_ratio, 3),
        "prompt_tokens": case.prompt_tokens,
        "eval_tokens": case.eval_tokens,
        "prompt_eval_ms": case.prompt_eval_ms,
        "eval_ms": case.eval_ms,
        "load_ms": case.load_ms,
        "local_selection_triggered": case.local_selection_triggered,
        "local_selection_applied": case.local_selection_applied,
        "local_selection_reasons": list(case.local_selection_reasons),
    }

def write_main_eval_summary(data: dict[str, Any], output_file: Path | None, runs_dir: Path) -> Path:
    return write_json_summary(data, output_file, runs_dir, "main-eval", "main_eval_path")

def render_main_eval(data: dict[str, Any], path: Path) -> str:
    lines = [
        f"Main Agent eval: {path}",
        f"Main model: {data['main_model']}",
        f"Records: {data['total']}",
        f"Clean: {data['clean_count']}",
        f"Issue cases: {data['issue_cases']}",
        f"Issue rate: {data['issue_rate']:.3f}",
        f"Refusal-like: {data['refusal_like_count']}",
        f"Refusal-like rate: {data['refusal_like_rate']:.3f}",
        f"Overlong: {data['overlong_count']}",
        f"Overlong rate: {data['overlong_rate']:.3f}",
        f"Average length ratio: {data['average_length_ratio']:.3f}",
        f"Local selector triggered: {data['local_selection_triggered_count']}",
        f"Local selector applied: {data['local_selection_applied_count']}",
        f"Total main calls: {data['total_main_calls']}",
        f"Clean cases/main-call: {data.get('clean_cases_per_main_call', data['clean_per_main_call']):.3f}",
        f"Eval tokens/clean: {data['eval_tokens_per_clean_case']:.1f}",
        f"Total ms: {data['total_duration_ms']}",
        "",
        "Cases:",
    ]
    for case in data["cases"]:
        marker = "ok" if case["clean"] else ",".join(case["issues"])
        lines.append(
            "- {id}: {marker}, category={category}, calls={main_call_count}, "
            "ratio={length_ratio}, ms={duration_ms}".format(
                marker=marker,
                **case,
            )
        )
    if data.get("gate_errors"):
        lines.extend(["", "Gate errors:"])
        lines.extend(f"- {error}" for error in data["gate_errors"])
    return "\n".join(lines)

def render_main_eval_ablation(data: dict[str, Any], path: Path) -> str:
    lines = [
        f"Main Agent eval ablation: {path}",
        f"Records: {data['records']}",
        f"Best clean/main-call: {data['best_profile_by_clean_cases_per_main_call']}",
        "Profiles:",
    ]
    for row in data["ranking"]:
        lines.append(
            "- {profile}: clean/main-call={clean_cases_per_main_call:.3f}, "
            "clean={clean_count}, calls={total_main_calls}, issue_rate={issue_rate:.3f}".format(**row)
        )
    return "\n".join(lines)

def main_eval_gate_errors(
    data: dict[str, Any],
    max_issue_rate: float = 1.0,
    max_refusal_rate: float = 1.0,
) -> list[str]:
    errors: list[str] = []
    if data["issue_rate"] > max_issue_rate:
        errors.append(f"issue rate above maximum: {data['issue_rate']:.3f} > {max_issue_rate:.3f}")
    if data["refusal_like_rate"] > max_refusal_rate:
        errors.append(
            f"refusal-like rate above maximum: {data['refusal_like_rate']:.3f} > {max_refusal_rate:.3f}"
        )
    return errors

def architecture_adversarial_eval_case_dict(
    case: ArchitectureAdversarialEvalCase,
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "id": case.record_id,
        "layer": case.layer,
        "passed": case.passed,
        "duration_ms": case.duration_ms,
        "issues": case.issues,
        "prompt_tokens": case.prompt_tokens,
        "eval_tokens": case.eval_tokens,
        "prompt_eval_ms": case.prompt_eval_ms,
        "eval_ms": case.eval_ms,
        "load_ms": case.load_ms,
    }
    if case.layer == "pipeline":
        data.update(
            {
                "expected_status": case.expected_status,
                "final_status": case.final_status,
                "attempts": case.attempts,
                "main_call_count": case.main_call_count,
                "output_chars": case.output_chars,
            }
        )
    else:
        data.update(
            {
                "expected_verdict": case.expected_verdict,
                "expected_clause": case.expected_clause,
                "predicted_verdict": case.predicted_verdict,
                "predicted_clause": case.predicted_clause,
                "audit_source": case.audit_source,
            }
        )
    return data

def write_architecture_adversarial_eval_summary(
    data: dict[str, Any],
    output_file: Path | None,
    runs_dir: Path,
) -> Path:
    return write_json_summary(
        data,
        output_file,
        runs_dir,
        "architecture-adversarial-eval",
        "architecture_adversarial_eval_path",
    )

def render_architecture_adversarial_eval(data: dict[str, Any], path: Path) -> str:
    profile = data["profile"]
    lines = [
        f"Architecture adversarial eval: {path}",
        f"Main model: {profile['main_model']}",
        f"Audit model: {profile['audit_model']}",
        f"Records: {data['total']}",
        f"Passed: {data['passed']}",
        f"Failed: {data['failed']}",
        f"Pass rate: {data['pass_rate']:.3f}",
        f"Pipeline cases: {data['pipeline_cases']}",
        f"Cold Eyes cases: {data['cold_eyes_cases']}",
        f"Action cases: {data.get('action_cases', 0)}",
        f"Main calls: {data['total_main_calls']}",
        f"Total ms: {data['total_duration_ms']}",
        "",
        "Cases:",
    ]
    for case in data["cases"]:
        marker = "ok" if case["passed"] else ",".join(case["issues"])
        if case["layer"] == "pipeline":
            lines.append(
                "- {id}: {marker}, layer=pipeline, expected={expected_status}, "
                "final={final_status}, attempts={attempts}, calls={main_call_count}, "
                "chars={output_chars}, ms={duration_ms}".format(marker=marker, **case)
            )
        elif case["layer"] == "cold_eyes":
            lines.append(
                "- {id}: {marker}, layer=cold_eyes, expected={expected_verdict}/{expected_clause}, "
                "predicted={predicted_verdict}/{predicted_clause}, source={audit_source}, "
                "ms={duration_ms}".format(marker=marker, **case)
            )
        else:
            lines.append(
                "- {id}: {marker}, layer=action, expected={expected_verdict}/{expected_clause}, "
                "predicted={predicted_verdict}/{predicted_clause}, source={audit_source}, "
                "ms={duration_ms}".format(marker=marker, **case)
            )
    if data.get("gate_errors"):
        lines.extend(["", "Gate errors:"])
        lines.extend(f"- {error}" for error in data["gate_errors"])
    return "\n".join(lines)

def architecture_adversarial_eval_gate_errors(
    data: dict[str, Any],
    min_pass_rate: float = 0.0,
) -> list[str]:
    errors: list[str] = []
    if data["pass_rate"] < min_pass_rate:
        errors.append(f"pass rate below minimum: {data['pass_rate']:.3f} < {min_pass_rate:.3f}")
    return errors

def distill_eval_case_dict(case: DistillEvalCase) -> dict[str, Any]:
    return {
        "id": case.record_id,
        "expected_verdict": case.expected_verdict,
        "expected_clause": case.expected_clause,
        "predicted_verdict": case.predicted_verdict,
        "predicted_clause": case.predicted_clause,
        "audit_source": case.audit_source,
        "verdict_match": case.verdict_match,
        "exact_match": case.exact_match,
        "duration_ms": case.duration_ms,
        "prompt_tokens": case.prompt_tokens,
        "eval_tokens": case.eval_tokens,
        "prompt_eval_ms": case.prompt_eval_ms,
        "eval_ms": case.eval_ms,
        "load_ms": case.load_ms,
    }

def write_distill_eval_summary(data: dict[str, Any], output_file: Path | None, runs_dir: Path) -> Path:
    return write_json_summary(data, output_file, runs_dir, "distill-eval", "distill_eval_path")

def render_distill_eval(data: dict[str, Any], path: Path) -> str:
    lines = [
        f"Distillation eval: {path}",
        f"Audit model: {data['audit_model']}",
        f"Records: {data['total']}",
        f"Verdict matches: {data['verdict_matches']}",
        f"Exact matches: {data['exact_matches']}",
        f"Partial matches: {data['partial_matches']}",
        f"Verdict misses: {data['verdict_misses']}",
        f"Mechanical cases: {data['mechanical_cases']}",
        f"LLM cases: {data['llm_cases']}",
        f"Estimated LLM audit calls saved: {data['estimated_llm_audit_calls_saved']}",
        f"Verdict accuracy: {data['verdict_accuracy']:.3f}",
        f"Exact accuracy: {data['exact_accuracy']:.3f}",
        f"Total ms: {data['total_duration_ms']}",
        "",
        "Cases:",
    ]
    for case in data["cases"]:
        marker = "ok" if case["exact_match"] else "partial" if case["verdict_match"] else "miss"
        lines.append(
            "- {id}: {marker}, expected={expected_verdict}/{expected_clause}, "
            "predicted={predicted_verdict}/{predicted_clause}, ms={duration_ms}".format(
                marker=marker,
                **case,
            )
        )
    if data["mismatches"]:
        lines.extend(["", "Mismatches:"])
        for mismatch in data["mismatches"]:
            lines.append(
                "- {id}: expected={expected_verdict}/{expected_clause}, "
                "predicted={predicted_verdict}/{predicted_clause}".format(**mismatch)
            )
    if data.get("gate_errors"):
        lines.extend(["", "Gate errors:"])
        lines.extend(f"- {error}" for error in data["gate_errors"])
    return "\n".join(lines)

def distill_eval_gate_errors(
    data: dict[str, Any],
    require_exact: bool = False,
    min_exact_accuracy: float = 0.0,
    min_mechanical_cases: int = 0,
) -> list[str]:
    errors: list[str] = []
    if data["verdict_matches"] != data["total"]:
        errors.append(f"verdict matches below total: {data['verdict_matches']} < {data['total']}")
    if require_exact and data["exact_matches"] != data["total"]:
        errors.append(f"exact matches below total: {data['exact_matches']} < {data['total']}")
    if data["exact_accuracy"] < min_exact_accuracy:
        errors.append(f"exact accuracy below minimum: {data['exact_accuracy']:.3f} < {min_exact_accuracy:.3f}")
    if data["mechanical_cases"] < min_mechanical_cases:
        errors.append(f"mechanical cases below minimum: {data['mechanical_cases']} < {min_mechanical_cases}")
    return errors
