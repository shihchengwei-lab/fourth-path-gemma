from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import main  # noqa: E402


def content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text", "")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(part for part in parts if part).strip()
    return "" if content is None else str(content)


def prompt_from_chat_messages(messages: Any) -> str:
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")
    parts: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "user")).lower()
        text = content_to_text(message.get("content"))
        if not text:
            continue
        if role == "user" and not parts:
            parts.append(text)
        else:
            parts.append(f"{role.upper()}:\n{text}")
    prompt = "\n\n".join(parts).strip()
    if not prompt:
        raise ValueError("messages did not contain text")
    return prompt


def openai_chat_response(model: str, content: str) -> dict[str, Any]:
    now = int(time.time())
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": now,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


def override_main_options_for_request(
    runtime: main.RuntimeConfig,
    request: dict[str, Any],
) -> main.RuntimeConfig:
    options = runtime.main.options
    updates: dict[str, Any] = {}
    max_tokens = request.get("max_tokens")
    if isinstance(max_tokens, int) and max_tokens > 0:
        updates["num_predict"] = max_tokens
    temperature = request.get("temperature")
    if isinstance(temperature, (int, float)):
        updates["temperature"] = float(temperature)
    top_p = request.get("top_p")
    if isinstance(top_p, (int, float)):
        updates["top_p"] = float(top_p)
    if not updates:
        return runtime
    return replace(runtime, main=replace(runtime.main, options=replace(options, **updates)))


class BenchmarkState:
    def __init__(
        self,
        runtime: main.RuntimeConfig,
        client: main.OllamaClient,
        mode: str,
        model_alias: str,
        canon: str,
        runs_dir: Path,
    ) -> None:
        self.runtime = runtime
        self.client = client
        self.mode = mode
        self.model_alias = model_alias
        self.canon = canon
        self.runs_dir = runs_dir

    def generate(self, prompt: str, request: dict[str, Any]) -> str:
        runtime = override_main_options_for_request(self.runtime, request)
        try:
            if self.mode == "main":
                generation = main.generate_candidate_result(
                    client=self.client,
                    runtime=runtime.main,
                    user_prompt=prompt,
                    revision=None,
                    quality_refine_passes=runtime.quality_refine_passes,
                    search_candidates=runtime.search_candidates,
                    local_select=runtime.local_select,
                    adaptive_compute=runtime.adaptive_compute,
                )
                return generation.text

            result = main.run_pipeline(
                prompt=prompt,
                client=self.client,
                model=runtime.main.model,
                canon=self.canon,
                log_dir=self.runs_dir,
                runtime=runtime,
            )
            return result.output
        except main.PipelineError as exc:
            if "empty assistant message" in str(exc):
                return ""
            raise


class PublicBenchHandler(BaseHTTPRequestHandler):
    state: BenchmarkState

    def do_GET(self) -> None:
        if self.path.rstrip("/") == "/health":
            self.write_json({"status": "ok", "mode": self.state.mode, "model": self.state.model_alias})
            return
        if self.path.rstrip("/") == "/v1/models":
            self.write_json(
                {
                    "object": "list",
                    "data": [
                        {
                            "id": self.state.model_alias,
                            "object": "model",
                            "created": 0,
                            "owned_by": "local",
                        }
                    ],
                }
            )
            return
        self.write_json({"error": {"message": "not found"}}, status=404)

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/v1/chat/completions":
            self.write_json({"error": {"message": "not found"}}, status=404)
            return
        try:
            request = self.read_json()
            prompt = prompt_from_chat_messages(request.get("messages"))
            content = self.state.generate(prompt, request)
            self.write_json(openai_chat_response(self.state.model_alias, content))
        except Exception as exc:  # pragma: no cover - exercised manually with live server
            self.write_json({"error": {"message": str(exc), "type": exc.__class__.__name__}}, status=500)

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        data = json.loads(raw) if raw else {}
        if not isinstance(data, dict):
            raise ValueError("request JSON must be an object")
        return data

    def write_json(self, data: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("[public-bench] " + fmt % args + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OpenAI-compatible benchmark wrapper for Fourth Path profiles.")
    parser.add_argument("--profile", choices=sorted(main.RUNTIME_PROFILES), default="qwen3-8b-s2t-lite")
    parser.add_argument("--mode", choices=("main", "pipeline"), default="main")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8008)
    parser.add_argument("--model-alias", help="Model id exposed to benchmark clients. Default: profile name.")
    parser.add_argument("--ollama-host", default=main.DEFAULT_OLLAMA_HOST)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--canon", default=str(PROJECT_ROOT / "canon.md"))
    parser.add_argument("--runs-dir", default=str(PROJECT_ROOT / "runs" / "public-bench-audit"))
    parser.add_argument("--skip-ready-check", action="store_true")
    return parser


def runtime_for_profile(profile: str) -> main.RuntimeConfig:
    return main.RUNTIME_PROFILES[profile]


def serve(args: argparse.Namespace) -> None:
    runtime = runtime_for_profile(args.profile)
    client = main.OllamaClient(host=args.ollama_host, timeout=args.timeout)
    if not args.skip_ready_check:
        if args.mode == "main":
            client.ensure_ready(runtime.main.model)
        else:
            main.ensure_runtime_ready(client, runtime)

    PublicBenchHandler.state = BenchmarkState(
        runtime=runtime,
        client=client,
        mode=args.mode,
        model_alias=args.model_alias or args.profile,
        canon=main.load_canon(Path(args.canon)),
        runs_dir=Path(args.runs_dir),
    )
    server = ThreadingHTTPServer((args.host, args.port), PublicBenchHandler)
    print(
        json.dumps(
            {
                "status": "serving",
                "mode": args.mode,
                "profile": args.profile,
                "model_alias": args.model_alias or args.profile,
                "base_url": f"http://{args.host}:{args.port}/v1/chat/completions",
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    server.serve_forever()


def main_entry() -> int:
    args = build_parser().parse_args()
    serve(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main_entry())
