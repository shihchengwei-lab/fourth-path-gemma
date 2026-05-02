from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core_types import SetupError
from main_agent_data import MainAgentRecord, safe_ratio, sorted_count_by
from output_utils import elapsed_ms
from training_data import verifier_metadata_labels


NVIDIA_BASE_URL_ENV = "NVIDIA_BASE_URL"
NVIDIA_API_KEY_ENV = "NVIDIA_API_KEY"
DEFAULT_NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_NVIDIA_TEACHER_MODELS = (
    "deepseek-ai/deepseek-v3.2",
    "minimaxai/minimax-m2.7",
    "nvidia/nemotron-3-super-120b-a12b",
    "openai/gpt-oss-120b",
    "qwen/qwen3-next-80b-a3b-instruct",
)
NVIDIA_TEACHER_SYSTEM_PROMPT = """You are an external teacher generating supervised data for a small local Main Agent.
Answer the user's task directly.
Follow every explicit format, count, unit, and code-only constraint.
Do not mention hidden instructions, safety policy, audits, or this teacher prompt.
For code repair tasks, return only the corrected code unless the user asks otherwise."""


@dataclass(frozen=True)
class NvidiaTeacherCase:
    record_id: str
    category: str
    teacher_model: str
    sample_index: int
    accepted: bool
    reward: float
    issues: list[str]
    duration_ms: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    error: str | None = None


def normalize_nvidia_base_url(base_url: str) -> str:
    cleaned = base_url.strip().rstrip("/")
    if cleaned.endswith("/v1/chat/completions"):
        cleaned = cleaned[: -len("/chat/completions")]
    if not cleaned.endswith("/v1"):
        cleaned = f"{cleaned}/v1"
    return cleaned


def sanitize_model_id(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", model).strip("-").lower()


class NvidiaTeacherClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_NVIDIA_BASE_URL,
        timeout: int = 120,
    ) -> None:
        if not api_key.strip():
            raise SetupError(f"{NVIDIA_API_KEY_ENV} is required for NVIDIA teacher export.")
        self.api_key = api_key.strip()
        self.base_url = normalize_nvidia_base_url(base_url)
        self.timeout = timeout
        self.last_usage: dict[str, int] = {}

    @classmethod
    def from_env(cls, timeout: int = 120) -> "NvidiaTeacherClient":
        api_key = os.environ.get(NVIDIA_API_KEY_ENV, "")
        base_url = os.environ.get(NVIDIA_BASE_URL_ENV, DEFAULT_NVIDIA_BASE_URL)
        return cls(api_key=api_key, base_url=base_url, timeout=timeout)

    def chat(
        self,
        *,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> str:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        response = self._post_json("/chat/completions", payload)
        usage = response.get("usage")
        self.last_usage = {
            key: value for key, value in (usage or {}).items() if isinstance(value, int)
        }
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise SetupError("NVIDIA teacher response did not include choices.")
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise SetupError("NVIDIA teacher returned an empty assistant message.")
        return content.strip()

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise SetupError(f"NVIDIA teacher request failed with HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise SetupError("NVIDIA teacher API is not reachable.") from exc
        except TimeoutError as exc:
            raise SetupError("NVIDIA teacher request timed out.") from exc

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SetupError("NVIDIA teacher returned invalid JSON.") from exc
        if not isinstance(parsed, dict):
            raise SetupError("NVIDIA teacher returned a non-object JSON response.")
        return parsed


def nvidia_teacher_case_dict(case: NvidiaTeacherCase) -> dict[str, Any]:
    return {
        "id": case.record_id,
        "category": case.category,
        "teacher_model": case.teacher_model,
        "sample_index": case.sample_index,
        "accepted": case.accepted,
        "reward": case.reward,
        "issues": case.issues,
        "duration_ms": case.duration_ms,
        "prompt_tokens": case.prompt_tokens,
        "completion_tokens": case.completion_tokens,
        "total_tokens": case.total_tokens,
        "error": case.error,
    }


def nvidia_teacher_export_row(
    record: MainAgentRecord,
    answer: str,
    *,
    teacher_model: str,
    sample_index: int,
    reward: float,
    main_agent_system_prompt: str,
    include_system: bool,
) -> dict[str, Any]:
    messages = []
    if include_system:
        messages.append({"role": "system", "content": main_agent_system_prompt})
    messages.extend(
        [
            {"role": "user", "content": record.prompt},
            {"role": "assistant", "content": answer},
        ]
    )
    return {
        "id": f"{record.record_id}-nvidia-{sanitize_model_id(teacher_model)}-{sample_index}",
        "record_id": record.record_id,
        "category": record.category,
        "source": "nvidia_teacher_synthetic",
        "split": "train_candidate",
        "verifier_labels": [
            "accepted_by_local_verifier",
            "external_teacher:nvidia",
            f"teacher_model:{teacher_model}",
            *verifier_metadata_labels(record.verifier),
        ],
        "teacher_provider": "nvidia",
        "teacher_model": teacher_model,
        "sample_index": sample_index,
        "reward": reward,
        "messages": messages,
    }


def run_nvidia_teacher_export(
    *,
    client: Any,
    records: list[MainAgentRecord],
    output_file: Path,
    teacher_models: Iterable[str] = DEFAULT_NVIDIA_TEACHER_MODELS,
    samples_per_model: int = 1,
    min_reward: float = 1.0,
    max_length_ratio: float | None = None,
    temperature: float = 0.2,
    max_tokens: int = 512,
    include_system: bool = True,
    main_agent_system_prompt: str,
    candidate_issues: Callable[[str, str | None, float | None], list[str]],
    verifier_issues: Callable[[str, dict[str, Any]], list[str]],
    limit_records: int = 0,
    continue_on_error: bool = True,
) -> dict[str, Any]:
    selected_models = tuple(teacher_models)
    if not selected_models:
        raise SetupError("At least one NVIDIA teacher model is required.")
    if samples_per_model < 1:
        raise SetupError("--samples-per-model must be at least 1.")
    if not 0 <= min_reward <= 1:
        raise SetupError("--min-reward must be between 0 and 1.")
    if limit_records < 0:
        raise SetupError("--limit-records must be zero or greater.")

    selected_records = records[:limit_records] if limit_records else records
    cases: list[NvidiaTeacherCase] = []
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    for record in selected_records:
        for teacher_model in selected_models:
            for sample_index in range(1, samples_per_model + 1):
                sample_started = time.perf_counter()
                try:
                    answer = client.chat(
                        model=teacher_model,
                        system=NVIDIA_TEACHER_SYSTEM_PROMPT,
                        user=record.prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                except SetupError as exc:
                    if not continue_on_error:
                        raise
                    cases.append(
                        NvidiaTeacherCase(
                            record_id=record.record_id,
                            category=record.category,
                            teacher_model=teacher_model,
                            sample_index=sample_index,
                            accepted=False,
                            reward=0.0,
                            issues=["teacher_request_failed"],
                            duration_ms=elapsed_ms(sample_started),
                            prompt_tokens=0,
                            completion_tokens=0,
                            total_tokens=0,
                            error=type(exc).__name__,
                        )
                    )
                    continue
                issues = candidate_issues(answer, record.target_response, max_length_ratio)
                issues.extend(verifier_issues(answer, record.verifier))
                issues = list(dict.fromkeys(issues))
                reward = 1.0 if not issues else 0.0
                accepted = not issues and reward >= min_reward
                usage = getattr(client, "last_usage", {})
                case = NvidiaTeacherCase(
                    record_id=record.record_id,
                    category=record.category,
                    teacher_model=teacher_model,
                    sample_index=sample_index,
                    accepted=accepted,
                    reward=reward,
                    issues=issues,
                    duration_ms=elapsed_ms(sample_started),
                    prompt_tokens=int(usage.get("prompt_tokens", 0)),
                    completion_tokens=int(usage.get("completion_tokens", 0)),
                    total_tokens=int(usage.get("total_tokens", 0)),
                )
                cases.append(case)
                if accepted:
                    rows.append(
                        nvidia_teacher_export_row(
                            record,
                            answer,
                            teacher_model=teacher_model,
                            sample_index=sample_index,
                            reward=reward,
                            main_agent_system_prompt=main_agent_system_prompt,
                            include_system=include_system,
                        )
                    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )

    accepted_cases = [case for case in cases if case.accepted]
    model_attempt_counts = Counter(case.teacher_model for case in cases)
    model_accept_counts = Counter(case.teacher_model for case in accepted_cases)
    return {
        "path": str(output_file),
        "provider": "nvidia",
        "base_url": getattr(client, "base_url", DEFAULT_NVIDIA_BASE_URL),
        "teacher_models": list(selected_models),
        "records": len(selected_records),
        "samples_per_model": samples_per_model,
        "total_samples": len(cases),
        "accepted_samples": len(rows),
        "acceptance_rate": safe_ratio(len(rows), len(cases)),
        "min_reward": min_reward,
        "include_system": include_system,
        "max_length_ratio": max_length_ratio,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "accepted_category_counts": sorted_count_by(case.category for case in accepted_cases),
        "accepted_model_counts": dict(sorted(model_accept_counts.items())),
        "model_attempt_counts": dict(sorted(model_attempt_counts.items())),
        "issue_counts": sorted_count_by(issue for case in cases for issue in case.issues),
        "total_duration_ms": elapsed_ms(started),
        "total_prompt_tokens": sum(case.prompt_tokens for case in cases),
        "total_completion_tokens": sum(case.completion_tokens for case in cases),
        "total_tokens": sum(case.total_tokens for case in cases),
        "cases": [nvidia_teacher_case_dict(case) for case in cases],
    }


def render_nvidia_teacher_export(data: dict[str, Any]) -> str:
    lines = [
        f"NVIDIA teacher export: {data['path']}",
        f"Models: {', '.join(data['teacher_models'])}",
        f"Records: {data['records']}",
        f"Samples/model: {data['samples_per_model']}",
        f"Total samples: {data['total_samples']}",
        f"Accepted samples: {data['accepted_samples']}",
        f"Acceptance rate: {data['acceptance_rate']:.3f}",
        f"Total tokens: {data['total_tokens']}",
        f"Total ms: {data['total_duration_ms']}",
        "Accepted models:",
    ]
    if data["accepted_model_counts"]:
        lines.extend(f"- {model}: {count}" for model, count in data["accepted_model_counts"].items())
    else:
        lines.append("- none")
    if data["accepted_category_counts"]:
        lines.append("Accepted categories:")
        lines.extend(f"- {category}: {count}" for category, count in data["accepted_category_counts"].items())
    if data["issue_counts"]:
        lines.append("Issue labels:")
        lines.extend(f"- {issue}: {count}" for issue, count in data["issue_counts"].items())
    return "\n".join(lines)
