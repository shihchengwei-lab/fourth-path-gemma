from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


DEFAULT_MAX_ATTEMPTS = 3


@dataclass(frozen=True)
class ModelOptions:
    num_ctx: int | None = None
    num_predict: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None

    def payload(self) -> dict[str, int | float]:
        data: dict[str, int | float] = {}
        for key, value in {
            "num_ctx": self.num_ctx,
            "num_predict": self.num_predict,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
        }.items():
            if value is not None:
                data[key] = value
        return data


@dataclass(frozen=True)
class RoleRuntime:
    model: str
    options: ModelOptions = field(default_factory=ModelOptions)
    no_think: bool = False
    keep_alive: str | None = None
    response_format: str | dict[str, Any] | None = None

    def user_prompt(self, prompt: str) -> str:
        if not self.no_think or "/no_think" in prompt.lower():
            return prompt
        return f"{prompt}\n\n/no_think"


@dataclass(frozen=True)
class RuntimeConfig:
    main: RoleRuntime
    audit: RoleRuntime
    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    quality_refine_passes: int = 0
    search_candidates: int = 1
    local_select: bool = False
    adaptive_compute: bool = False


def build_runtime_profiles(
    default_model: str,
    cold_eyes_json_schema: dict[str, Any],
) -> dict[str, RuntimeConfig]:
    return {
        "legacy": RuntimeConfig(
            main=RoleRuntime(default_model),
            audit=RoleRuntime(default_model, response_format=cold_eyes_json_schema),
        ),
        "qwen3-8b-local-max": RuntimeConfig(
            main=RoleRuntime(
                "qwen3:8b",
                ModelOptions(num_ctx=8192, temperature=0.7, top_p=0.8, top_k=20),
                no_think=True,
                keep_alive="30m",
            ),
            audit=RoleRuntime(
                "qwen3:8b",
                ModelOptions(num_ctx=8192, num_predict=64, temperature=0.0, top_p=0.5, top_k=10),
                no_think=True,
                keep_alive="30m",
                response_format=cold_eyes_json_schema,
            ),
            max_attempts=2,
        ),
        "qwen3-8b-deliberate": RuntimeConfig(
            main=RoleRuntime(
                "qwen3:8b",
                ModelOptions(num_ctx=8192, temperature=0.7, top_p=0.8, top_k=20),
                no_think=True,
                keep_alive="30m",
            ),
            audit=RoleRuntime(
                "qwen3:8b",
                ModelOptions(num_ctx=8192, num_predict=64, temperature=0.0, top_p=0.5, top_k=10),
                no_think=True,
                keep_alive="30m",
                response_format=cold_eyes_json_schema,
            ),
            max_attempts=2,
            quality_refine_passes=1,
        ),
        "qwen3-8b-s2t-lite": RuntimeConfig(
            main=RoleRuntime(
                "qwen3:8b",
                ModelOptions(num_ctx=8192, temperature=0.7, top_p=0.8, top_k=20),
                no_think=True,
                keep_alive="30m",
            ),
            audit=RoleRuntime(
                "qwen3:8b",
                ModelOptions(num_ctx=8192, num_predict=64, temperature=0.0, top_p=0.5, top_k=10),
                no_think=True,
                keep_alive="30m",
                response_format=cold_eyes_json_schema,
            ),
            max_attempts=2,
            local_select=True,
        ),
        "qwen3-8b-compute-optimal-lite": RuntimeConfig(
            main=RoleRuntime(
                "qwen3:8b",
                ModelOptions(num_ctx=8192, temperature=0.7, top_p=0.8, top_k=20),
                no_think=True,
                keep_alive="30m",
            ),
            audit=RoleRuntime(
                "qwen3:8b",
                ModelOptions(num_ctx=8192, num_predict=64, temperature=0.0, top_p=0.5, top_k=10),
                no_think=True,
                keep_alive="30m",
                response_format=cold_eyes_json_schema,
            ),
            max_attempts=2,
            local_select=True,
            adaptive_compute=True,
        ),
        "qwen3-8b-reasoning": RuntimeConfig(
            main=RoleRuntime(
                "qwen3:8b",
                ModelOptions(num_ctx=8192, temperature=0.6, top_p=0.8, top_k=20),
                no_think=False,
                keep_alive="30m",
            ),
            audit=RoleRuntime(
                "qwen3:8b",
                ModelOptions(num_ctx=8192, num_predict=64, temperature=0.0, top_p=0.5, top_k=10),
                no_think=True,
                keep_alive="30m",
                response_format=cold_eyes_json_schema,
            ),
            max_attempts=2,
        ),
        "qwen3-8b-search": RuntimeConfig(
            main=RoleRuntime(
                "qwen3:8b",
                ModelOptions(num_ctx=8192, temperature=0.8, top_p=0.9, top_k=40),
                no_think=True,
                keep_alive="30m",
            ),
            audit=RoleRuntime(
                "qwen3:8b",
                ModelOptions(num_ctx=8192, num_predict=64, temperature=0.0, top_p=0.5, top_k=10),
                no_think=True,
                keep_alive="30m",
                response_format=cold_eyes_json_schema,
            ),
            max_attempts=2,
            search_candidates=2,
        ),
        "qwen3-8b-split-audit": RuntimeConfig(
            main=RoleRuntime(
                "qwen3:8b",
                ModelOptions(num_ctx=8192, temperature=0.7, top_p=0.8, top_k=20),
                no_think=True,
                keep_alive="30m",
            ),
            audit=RoleRuntime(
                "qwen3:1.7b",
                ModelOptions(num_ctx=2048, num_predict=64, temperature=0.0, top_p=0.5, top_k=10),
                no_think=True,
                keep_alive="30m",
                response_format=cold_eyes_json_schema,
            ),
            max_attempts=2,
        ),
        "qwen3-1.7b-amateur": RuntimeConfig(
            main=RoleRuntime(
                "qwen3:1.7b",
                ModelOptions(num_ctx=4096, temperature=0.7, top_p=0.8, top_k=20),
                no_think=True,
                keep_alive="30m",
            ),
            audit=RoleRuntime(
                "qwen3:8b",
                ModelOptions(num_ctx=8192, num_predict=64, temperature=0.0, top_p=0.5, top_k=10),
                no_think=True,
                keep_alive="30m",
                response_format=cold_eyes_json_schema,
            ),
            max_attempts=1,
        ),
        "llama3.1-8b-candidate": RuntimeConfig(
            main=RoleRuntime(
                "llama3.1:8b",
                ModelOptions(num_ctx=8192, temperature=0.7, top_p=0.9, top_k=40),
                keep_alive="30m",
            ),
            audit=RoleRuntime(
                "qwen3:8b",
                ModelOptions(num_ctx=8192, num_predict=64, temperature=0.0, top_p=0.5, top_k=10),
                no_think=True,
                keep_alive="30m",
                response_format=cold_eyes_json_schema,
            ),
            max_attempts=2,
        ),
        "gemma3-12b-pressure": RuntimeConfig(
            main=RoleRuntime(
                "gemma3:12b",
                ModelOptions(num_ctx=4096, temperature=0.6, top_p=0.9),
                keep_alive="10m",
            ),
            audit=RoleRuntime(
                "gemma3:12b",
                ModelOptions(num_ctx=4096, num_predict=96, temperature=0.0),
                keep_alive="10m",
                response_format=cold_eyes_json_schema,
            ),
            max_attempts=2,
        ),
        "gemma3-4b-compact": RuntimeConfig(
            main=RoleRuntime("gemma3:4b", ModelOptions(num_ctx=8192, temperature=0.6, top_p=0.9)),
            audit=RoleRuntime(
                "gemma3:1b",
                ModelOptions(num_ctx=2048, num_predict=160, temperature=0.0),
                response_format=cold_eyes_json_schema,
            ),
            max_attempts=2,
        ),
    }
