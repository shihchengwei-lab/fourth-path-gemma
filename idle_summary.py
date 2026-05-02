from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


IDLE_LOG_RE = re.compile(r"^idle-long-run-(?P<stamp>\d{8}-\d{6})\.log$")
IDLE_STEP_START_RE = re.compile(r"^\[(?P<time>[^\]]+)\] START (?P<name>.+)$")
IDLE_STEP_END_RE = re.compile(
    r"^\[(?P<time>[^\]]+)\] END (?P<name>.+) exit=(?P<exit>-?\d+) seconds=(?P<seconds>\d+)$"
)


def idle_artifact_profile(path: Path, prefix: str, stamp: str) -> str:
    stem = path.stem
    full_prefix = f"{prefix}-"
    suffix = f"-idle-{stamp}"
    if stem.startswith(full_prefix) and stem.endswith(suffix):
        return stem[len(full_prefix) : -len(suffix)]
    return stem


def latest_idle_stamp(runs_dir: Path) -> str | None:
    candidates: list[tuple[float, str]] = []
    for path in runs_dir.glob("idle-long-run-*.log"):
        match = IDLE_LOG_RE.match(path.name)
        if match:
            candidates.append((path.stat().st_mtime, match.group("stamp")))
    if not candidates:
        return None
    return max(candidates)[1]


def summarize_idle_log(log_path: Path) -> dict[str, Any]:
    steps: list[dict[str, Any]] = []
    step_by_name: dict[str, dict[str, Any]] = {}
    started_at = ""
    completed_at = ""
    for line in read_text_with_bom(log_path).splitlines():
        if line.startswith("Idle long run started at "):
            started_at = line.removeprefix("Idle long run started at ").strip()
            continue
        if line.startswith("Idle long run completed at "):
            completed_at = line.removeprefix("Idle long run completed at ").strip()
            continue
        start_match = IDLE_STEP_START_RE.match(line)
        if start_match:
            step = {
                "name": start_match.group("name"),
                "started_at": start_match.group("time"),
                "ended_at": "",
                "exit_code": None,
                "seconds": None,
            }
            steps.append(step)
            step_by_name[step["name"]] = step
            continue
        end_match = IDLE_STEP_END_RE.match(line)
        if end_match:
            name = end_match.group("name")
            step = step_by_name.get(name)
            if step is None:
                step = {"name": name, "started_at": "", "ended_at": "", "exit_code": None, "seconds": None}
                steps.append(step)
                step_by_name[name] = step
            step["ended_at"] = end_match.group("time")
            step["exit_code"] = int(end_match.group("exit"))
            step["seconds"] = int(end_match.group("seconds"))

    failed_steps = [step for step in steps if step.get("exit_code") not in (0, None)]
    incomplete_steps = [step for step in steps if step.get("exit_code") is None]
    return {
        "path": str(log_path),
        "started_at": started_at,
        "completed_at": completed_at,
        "completed": bool(completed_at),
        "step_count": len(steps),
        "failed_steps": failed_steps,
        "incomplete_steps": incomplete_steps,
        "total_step_seconds": sum(step.get("seconds") or 0 for step in steps),
        "steps": steps,
    }


def read_text_with_bom(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16", errors="replace")
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig", errors="replace")
    return raw.decode("utf-8", errors="replace")


def load_idle_artifact(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        return None, f"{path}: read failed: {exc}"
    except json.JSONDecodeError as exc:
        return None, f"{path}: invalid JSON at line {exc.lineno}: {exc.msg}"
    if not isinstance(loaded, dict):
        return None, f"{path}: expected JSON object"
    return loaded, None


def summarize_bench_artifact(path: Path, stamp: str, data: dict[str, Any]) -> dict[str, Any]:
    status_counts = Counter(
        case.get("status", "unknown") for case in data.get("cases", []) if isinstance(case, dict)
    )
    return {
        "path": str(path),
        "profile": idle_artifact_profile(path, "bench", stamp),
        "total_cases": data.get("total_cases", 0),
        "pass_count": data.get("pass_count", 0),
        "refused_count": data.get("refused_count", 0),
        "status_counts": dict(status_counts),
        "total_main_calls": data.get("total_main_calls", 0),
        "average_main_calls_per_case": data.get("average_main_calls_per_case", 0),
        "total_duration_ms": data.get("total_duration_ms", 0),
    }


def summarize_main_eval_artifact(path: Path, stamp: str, data: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": str(path),
        "profile": idle_artifact_profile(path, "main-eval", stamp),
        "total": data.get("total", 0),
        "clean_count": data.get("clean_count", 0),
        "issue_cases": data.get("issue_cases", 0),
        "refusal_like_count": data.get("refusal_like_count", 0),
        "overlong_count": data.get("overlong_count", 0),
        "average_length_ratio": data.get("average_length_ratio", 0),
        "issue_counts": data.get("issue_counts", {}),
        "category_issue_counts": data.get("category_issue_counts", {}),
        "local_selection_triggered_count": data.get("local_selection_triggered_count", 0),
        "local_selection_applied_count": data.get("local_selection_applied_count", 0),
        "total_main_calls": data.get("total_main_calls", 0),
        "clean_per_main_call": data.get("clean_per_main_call", 0),
        "total_duration_ms": data.get("total_duration_ms", 0),
    }


def summarize_architecture_adversarial_artifact(path: Path, stamp: str, data: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": str(path),
        "profile": idle_artifact_profile(path, "architecture-adversarial-eval", stamp),
        "total": data.get("total", 0),
        "passed": data.get("passed", 0),
        "failed": data.get("failed", 0),
        "pass_rate": data.get("pass_rate", 0),
        "layer_counts": data.get("layer_counts", {}),
        "layer_passed": data.get("layer_passed", {}),
        "issue_counts": data.get("issue_counts", {}),
        "audit_source_counts": data.get("audit_source_counts", {}),
        "total_main_calls": data.get("total_main_calls", 0),
        "total_duration_ms": data.get("total_duration_ms", 0),
    }


def summarize_distill_eval_artifact(path: Path, stamp: str, data: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": str(path),
        "profile": idle_artifact_profile(path, "distill-eval", stamp),
        "audit_model": data.get("audit_model", ""),
        "total": data.get("total", 0),
        "verdict_matches": data.get("verdict_matches", 0),
        "exact_matches": data.get("exact_matches", 0),
        "partial_matches": data.get("partial_matches", 0),
        "verdict_misses": data.get("verdict_misses", 0),
        "mechanical_cases": data.get("mechanical_cases", 0),
        "llm_cases": data.get("llm_cases", 0),
        "mismatch_count": len(data.get("mismatches", [])),
        "mismatch_counts_by_expected_clause": data.get("mismatch_counts_by_expected_clause", {}),
        "exact_accuracy": data.get("exact_accuracy", 0),
        "total_duration_ms": data.get("total_duration_ms", 0),
    }


def idle_run_summary_data(runs_dir: Path, stamp: str | None = None) -> dict[str, Any]:
    stamp = stamp or latest_idle_stamp(runs_dir)
    if stamp is None:
        return {
            "runs_dir": str(runs_dir),
            "stamp": None,
            "completed": False,
            "errors": [f"no idle-long-run-*.log found under {runs_dir}"],
        }

    log_path = runs_dir / f"idle-long-run-{stamp}.log"
    errors: list[str] = []
    log_summary: dict[str, Any]
    if log_path.exists():
        log_summary = summarize_idle_log(log_path)
        if not log_summary["completed"]:
            errors.append(f"{log_path}: run did not record completion")
        for step in log_summary["failed_steps"]:
            errors.append(f"{log_path}: step failed: {step['name']} exit={step['exit_code']}")
        for step in log_summary["incomplete_steps"]:
            errors.append(f"{log_path}: step incomplete: {step['name']}")
    else:
        log_summary = {
            "path": str(log_path),
            "started_at": "",
            "completed_at": "",
            "completed": False,
            "step_count": 0,
            "failed_steps": [],
            "incomplete_steps": [],
            "total_step_seconds": 0,
            "steps": [],
        }
        errors.append(f"{log_path}: missing idle long-run log")

    artifacts: dict[str, Any] = {
        "architecture_adversarial": [],
        "main_eval": [],
        "bench": [],
        "distill_eval": [],
        "unknown": [],
    }
    for path in sorted(runs_dir.glob(f"*-idle-{stamp}.json")):
        loaded, error = load_idle_artifact(path)
        if error:
            errors.append(error)
            continue
        assert loaded is not None
        name = path.name
        if name.startswith("architecture-adversarial-eval-"):
            artifacts["architecture_adversarial"].append(
                summarize_architecture_adversarial_artifact(path, stamp, loaded)
            )
        elif name.startswith("main-eval-"):
            artifacts["main_eval"].append(summarize_main_eval_artifact(path, stamp, loaded))
        elif name.startswith("bench-"):
            artifacts["bench"].append(summarize_bench_artifact(path, stamp, loaded))
        elif name.startswith("distill-eval-"):
            artifacts["distill_eval"].append(summarize_distill_eval_artifact(path, stamp, loaded))
        else:
            artifacts["unknown"].append(str(path))

    return {
        "runs_dir": str(runs_dir),
        "stamp": stamp,
        "completed": bool(log_summary["completed"]) and not errors,
        "log": log_summary,
        "artifacts": artifacts,
        "errors": errors,
    }


def render_idle_run_summary(data: dict[str, Any]) -> str:
    lines = [
        f"Idle run summary: {data.get('stamp')}",
        f"Log: {data.get('log', {}).get('path', '')}",
        f"Completed: {'yes' if data.get('completed') else 'no'}",
    ]
    log = data.get("log", {})
    if log:
        lines.append(
            f"Steps: {log.get('step_count', 0)}, failed={len(log.get('failed_steps', []))}, "
            f"incomplete={len(log.get('incomplete_steps', []))}, seconds={log.get('total_step_seconds', 0)}"
        )

    artifacts = data.get("artifacts", {})
    if artifacts.get("architecture_adversarial"):
        lines.extend(["", "Architecture adversarial eval:"])
        for item in artifacts["architecture_adversarial"]:
            lines.append(
                "- {profile}: passed {passed}/{total}, failed={failed}, calls={total_main_calls}, ms={total_duration_ms}".format(
                    **item
                )
            )
    if artifacts.get("main_eval"):
        lines.extend(["", "Main eval:"])
        for item in artifacts["main_eval"]:
            lines.append(
                "- {profile}: clean {clean_count}/{total}, issues={issue_cases}, refusals={refusal_like_count}, "
                "overlong={overlong_count}, calls={total_main_calls}, ms={total_duration_ms}".format(**item)
            )
    if artifacts.get("bench"):
        lines.extend(["", "Bench:"])
        for item in artifacts["bench"]:
            lines.append(
                "- {profile}: pass {pass_count}/{total_cases}, refused={refused_count}, "
                "calls={total_main_calls}, ms={total_duration_ms}".format(**item)
            )
    if artifacts.get("distill_eval"):
        lines.extend(["", "Distill eval:"])
        for item in artifacts["distill_eval"]:
            lines.append(
                "- {profile}: exact {exact_matches}/{total}, mechanical={mechanical_cases}, "
                "llm={llm_cases}, mismatches={mismatch_count}, ms={total_duration_ms}".format(**item)
            )
    if data.get("errors"):
        lines.extend(["", "Errors:"])
        lines.extend(f"- {error}" for error in data["errors"])
    return "\n".join(lines)
