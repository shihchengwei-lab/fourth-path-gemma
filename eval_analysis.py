from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from main_agent_data import safe_ratio
from output_utils import write_json_summary


def _safe_int(value: Any) -> int:
    return value if isinstance(value, int) else 0


def _safe_float(value: Any) -> float:
    return value if isinstance(value, (int, float)) else 0.0


def _safe_str(value: Any, fallback: str) -> str:
    return value if isinstance(value, str) and value else fallback


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _profile_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    results = summary.get("results")
    if isinstance(results, list):
        return [row for row in results if isinstance(row, dict)]
    return [dict(summary)]


def _profile_name(row: Mapping[str, Any], index: int) -> str:
    return _safe_str(row.get("profile"), _safe_str(row.get("main_model"), f"profile-{index}"))


def _clean_cases_per_main_call(row: Mapping[str, Any]) -> float:
    if isinstance(row.get("clean_cases_per_main_call"), (int, float)):
        return float(row["clean_cases_per_main_call"])
    if isinstance(row.get("clean_per_main_call"), (int, float)):
        return float(row["clean_per_main_call"])
    return safe_ratio(_safe_int(row.get("clean_count")), _safe_int(row.get("total_main_calls")))


def _counter_from_mapping(value: Any) -> Counter[str]:
    counter: Counter[str] = Counter()
    if not isinstance(value, Mapping):
        return counter
    for key, count in value.items():
        if isinstance(key, str) and isinstance(count, int):
            counter[key] += count
    return counter


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def _case_issue_labels(case: Mapping[str, Any]) -> list[str]:
    return [issue for issue in _safe_list(case.get("issues")) if isinstance(issue, str) and issue]


def main_eval_failure_report_data(
    summary: Mapping[str, Any],
    source_path: str | None = None,
) -> dict[str, Any]:
    kind = "main_eval_ablation" if isinstance(summary.get("results"), list) else "main_eval"
    issue_counts: Counter[str] = Counter()
    category_issue_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    local_selection_reason_counts: Counter[str] = Counter()
    failure_target_profiles: dict[tuple[str, str], set[str]] = defaultdict(set)
    case_failures: list[dict[str, Any]] = []
    profile_summaries: list[dict[str, Any]] = []

    for index, row in enumerate(_profile_rows(summary), start=1):
        profile = _profile_name(row, index)
        cases = [case for case in _safe_list(row.get("cases")) if isinstance(case, dict)]
        row_issue_counts = Counter()
        row_local_selection_reason_counts: Counter[str] = Counter()

        for case in cases:
            issues = _case_issue_labels(case)
            category = _safe_str(case.get("category"), "unknown")
            row_local_selection_reason_counts.update(
                reason
                for reason in _safe_list(case.get("local_selection_reasons"))
                if isinstance(reason, str) and reason
            )
            if not issues:
                continue
            row_issue_counts.update(issues)
            category_counts[category] += 1
            for issue in issues:
                category_issue_counts[f"{category}|{issue}"] += 1
                failure_target_profiles[(category, issue)].add(profile)
            case_failures.append(
                {
                    "profile": profile,
                    "id": _safe_str(case.get("id"), "unknown"),
                    "category": category,
                    "issues": issues,
                    "main_call_count": _safe_int(case.get("main_call_count")),
                    "length_ratio": _safe_float(case.get("length_ratio")),
                }
            )

        if not row_issue_counts:
            row_issue_counts.update(_counter_from_mapping(row.get("issue_counts")))
        issue_counts.update(row_issue_counts)
        if not row_local_selection_reason_counts:
            row_local_selection_reason_counts.update(
                _counter_from_mapping(row.get("local_selection_reason_counts"))
            )
        local_selection_reason_counts.update(row_local_selection_reason_counts)

        total = _safe_int(row.get("total"))
        clean_count = _safe_int(row.get("clean_count"))
        issue_cases = _safe_int(row.get("issue_cases"))
        total_main_calls = _safe_int(row.get("total_main_calls"))
        profile_summaries.append(
            {
                "profile": profile,
                "main_model": _safe_str(row.get("main_model"), "unknown"),
                "total": total,
                "clean_count": clean_count,
                "issue_cases": issue_cases,
                "issue_rate": _safe_float(row.get("issue_rate")),
                "total_main_calls": total_main_calls,
                "clean_cases_per_main_call": _clean_cases_per_main_call(row),
                "issue_counts": _sorted_counter(row_issue_counts),
            }
        )

    failure_targets = [
        {
            "category": category,
            "issue": issue,
            "count": count,
            "profiles": sorted(failure_target_profiles[(category, issue)]),
        }
        for pair, count in category_issue_counts.items()
        for category, issue in [pair.split("|", 1)]
    ]
    failure_targets.sort(key=lambda item: (-item["count"], item["category"], item["issue"]))

    efficiency_ranking = sorted(
        profile_summaries,
        key=lambda row: (
            -row["clean_cases_per_main_call"],
            -row["clean_count"],
            row["total_main_calls"],
            row["profile"],
        ),
    )

    return {
        "source_path": source_path,
        "kind": kind,
        "profile_count": len(profile_summaries),
        "profiles": [row["profile"] for row in profile_summaries],
        "issue_counts": _sorted_counter(issue_counts),
        "category_counts": _sorted_counter(category_counts),
        "category_issue_counts": _sorted_counter(category_issue_counts),
        "local_selection_reason_counts": _sorted_counter(local_selection_reason_counts),
        "failure_targets": failure_targets,
        "case_failures": sorted(
            case_failures,
            key=lambda row: (row["profile"], row["category"], row["id"]),
        ),
        "profile_summaries": profile_summaries,
        "efficiency_ranking": [
            {
                "profile": row["profile"],
                "clean_cases_per_main_call": row["clean_cases_per_main_call"],
                "clean_count": row["clean_count"],
                "total_main_calls": row["total_main_calls"],
                "issue_rate": row["issue_rate"],
            }
            for row in efficiency_ranking
        ],
    }


def load_main_eval_failure_report(input_file: Path) -> dict[str, Any]:
    summary = json.loads(input_file.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise ValueError("main eval summary must be a JSON object")
    return main_eval_failure_report_data(summary, source_path=str(input_file))


def write_main_eval_failure_report(
    data: dict[str, Any],
    output_file: Path | None,
    runs_dir: Path,
) -> Path:
    return write_json_summary(
        data,
        output_file,
        runs_dir,
        "main-eval-failure-report",
        "main_eval_failure_report_path",
    )


def render_main_eval_failure_report(data: dict[str, Any], path: Path) -> str:
    lines = [
        f"Main Agent eval failure report: {path}",
        f"Source: {data.get('source_path')}",
        f"Kind: {data['kind']}",
        f"Profiles: {', '.join(data['profiles']) if data['profiles'] else 'none'}",
        "",
        "Efficiency ranking:",
    ]
    for row in data["efficiency_ranking"]:
        lines.append(
            "- {profile}: clean/main-call={clean_cases_per_main_call:.3f}, "
            "clean={clean_count}, calls={total_main_calls}, issue_rate={issue_rate:.3f}".format(**row)
        )
    if data["issue_counts"]:
        lines.extend(["", "Issue labels:"])
        lines.extend(f"- {issue}: {count}" for issue, count in data["issue_counts"].items())
    if data["failure_targets"]:
        lines.extend(["", "Failure targets:"])
        for target in data["failure_targets"]:
            profiles = ",".join(target["profiles"])
            lines.append(
                f"- {target['category']} / {target['issue']}: {target['count']} ({profiles})"
            )
    if data["local_selection_reason_counts"]:
        lines.extend(["", "Local selection reasons:"])
        lines.extend(f"- {reason}: {count}" for reason, count in data["local_selection_reason_counts"].items())
    return "\n".join(lines)
