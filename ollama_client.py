from __future__ import annotations

import json
import os
import shutil
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from core_types import PipelineError, SetupError
from runtime_config import ModelOptions

DEFAULT_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
DEFAULT_TIMEOUT_SECONDS = 600
LOCAL_OLLAMA_EXE = Path("E:/Ollama/ollama.exe")


def ns_to_ms(value: Any) -> int | None:
    return int(value / 1_000_000) if isinstance(value, int) else None


def int_stat(value: Any) -> int | None:
    return value if isinstance(value, int) else None


def ollama_response_stats(response: dict[str, Any]) -> dict[str, int]:
    stats = {
        "prompt_tokens": int_stat(response.get("prompt_eval_count")),
        "eval_tokens": int_stat(response.get("eval_count")),
        "prompt_eval_ms": ns_to_ms(response.get("prompt_eval_duration")),
        "eval_ms": ns_to_ms(response.get("eval_duration")),
        "load_ms": ns_to_ms(response.get("load_duration")),
    }
    return {key: value for key, value in stats.items() if value is not None}


class OllamaClient:
    def __init__(self, host: str = DEFAULT_OLLAMA_HOST, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> None:
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.last_stats: dict[str, int] | None = None

    def ensure_ready(self, model: str) -> None:
        if (
            shutil.which("ollama.exe") is None
            and shutil.which("ollama") is None
            and not LOCAL_OLLAMA_EXE.exists()
        ):
            raise SetupError(
                "Ollama is not available in PATH. Install Ollama for Windows, then run: "
                f"ollama pull {model}"
            )

        tags = self._get_json("/api/tags", timeout=10)
        models = tags.get("models", [])
        available = {
            value
            for item in models
            for value in (item.get("name"), item.get("model"))
            if isinstance(value, str)
        }
        if model not in available:
            raise SetupError(
                f"Model {model!r} is not downloaded. Run: ollama pull {model}"
            )

    def chat(
        self,
        model: str,
        system: str,
        user: str,
        options: ModelOptions | None = None,
        think: bool | None = None,
        keep_alive: str | None = None,
        response_format: str | dict[str, Any] | None = None,
    ) -> str:
        payload = {
            "model": model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if options is not None:
            option_payload = options.payload()
            if option_payload:
                payload["options"] = option_payload
        if think is not None:
            payload["think"] = think
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
        if response_format is not None:
            payload["format"] = response_format
        response = self._post_json("/api/chat", payload, timeout=self.timeout)
        self.last_stats = ollama_response_stats(response)
        message = response.get("message", {})
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise PipelineError("Ollama returned an empty assistant message.")
        return content.strip()

    def keepalive(
        self,
        model: str,
        keep_alive: str,
        options: ModelOptions | None = None,
    ) -> dict[str, int]:
        payload: dict[str, Any] = {
            "model": model,
            "prompt": "",
            "stream": False,
            "keep_alive": keep_alive,
        }
        if options is not None:
            option_payload = options.payload()
            if option_payload:
                payload["options"] = option_payload
        response = self._post_json("/api/generate", payload, timeout=self.timeout)
        self.last_stats = ollama_response_stats(response)
        return self.last_stats

    def _get_json(self, path: str, timeout: int) -> dict[str, Any]:
        request = urllib.request.Request(f"{self.host}{path}", method="GET")
        return self._open_json(request, timeout)

    def _post_json(self, path: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.host}{path}",
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        return self._open_json(request, timeout)

    def _open_json(self, request: urllib.request.Request, timeout: int) -> dict[str, Any]:
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise SetupError(
                "Ollama service is not reachable. Open Ollama, then retry this command."
            ) from exc
        except TimeoutError as exc:
            raise SetupError("Ollama request timed out. Confirm the model is loaded and retry.") from exc

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise PipelineError(f"Ollama returned invalid JSON: {raw[:200]}") from exc
        if not isinstance(parsed, dict):
            raise PipelineError("Ollama returned a non-object JSON response.")
        return parsed
