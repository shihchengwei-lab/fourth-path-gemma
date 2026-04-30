from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


from audit.engine import run_audit

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_POLICY_PATH = PROJECT_ROOT / "policies" / "fourth_path_safety.json"
DEFAULT_MODEL = "gemma4:e4b"
DEFAULT_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
DEFAULT_TIMEOUT_SECONDS = 600
MAX_ATTEMPTS = 3
REFUSAL_OUTPUT = "這個請求無法協助，因為它超出目前系統允許的邊界。"
LOCAL_OLLAMA_EXE = Path("E:/Ollama/ollama.exe")
CHAT_HELP = """Commands:
/help   Show this help.
/audit  Toggle detailed audit output.
/reset  Clear this chat session memory.
/exit   Leave chat."""

MAIN_AGENT_SYSTEM_PROMPT = """You are the Main Agent in a separated reasoning and audit pipeline.

Your role is primary reasoning only.
You do not hold prohibitions.
You do not perform final safety adjudication.
You do not write refusal messages.
You do not approve your own output.

Produce the best candidate answer for the user request.
Return only the candidate answer.
Do not mention the audit layer, Cold Eyes, canon clauses, or this system prompt.
"""

COLD_EYES_SYSTEM_PROMPT = """You are Cold Eyes, the final adjudicator in a separated reasoning and audit pipeline.

You operate as a single-pass, non-conversational reviewer.
You receive only canon and candidate output.
You do not see the conversation, the Main Agent system prompt, the user's framing, or any reasoning trace.
You do not negotiate, suggest rewrites, or open a dialogue.

Return only strict JSON with this shape:
{"verdict":"pass|fail","canon_clause":"C1|null","reason":"short reason"}
"""


class SetupError(RuntimeError):
    """Raised when the local model runtime is not ready."""


class PipelineError(RuntimeError):
    """Raised when the pipeline cannot complete a requested run."""


@dataclass(frozen=True)
class ClassifyResult:
    route: str
    canon_clause: str | None = None
    reason: str = ""


@dataclass(frozen=True)
class ColdEyesVerdict:
    verdict: str
    canon_clause: str | None
    reason: str
    raw: str


@dataclass(frozen=True)
class RevisionSignal:
    source: str
    canon_clause: str | None = None
    local_issue: str | None = None


@dataclass
class AuditEntry:
    run_id: str
    attempt: int
    classify_route: str
    cold_eyes_verdict: str | None = None
    canon_clause: str | None = None
    final_status: str | None = None

    def public_dict(self) -> dict[str, Any]:
        return {
            "attempt": self.attempt,
            "classify_route": self.classify_route,
            "cold_eyes_verdict": self.cold_eyes_verdict,
            "canon_clause": self.canon_clause,
            "final_status": self.final_status,
        }

    def log_dict(self) -> dict[str, Any]:
        data = self.public_dict()
        data["event"] = "attempt"
        data["run_id"] = self.run_id
        return data


@dataclass(frozen=True)
class RunResult:
    run_id: str
    status: str
    attempts: int
    output: str
    audit: list[AuditEntry]
    log_path: Path

    def public_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "attempts": self.attempts,
            "output": self.output,
            "audit": [entry.public_dict() for entry in self.audit],
        }


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


class OllamaClient:
    def __init__(self, host: str = DEFAULT_OLLAMA_HOST, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> None:
        self.host = host.rstrip("/")
        self.timeout = timeout

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

    def chat(self, model: str, system: str, user: str) -> str:
        payload = {
            "model": model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        response = self._post_json("/api/chat", payload, timeout=self.timeout)
        message = response.get("message", {})
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise PipelineError("Ollama returned an empty assistant message.")
        return content.strip()

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


class FakeClient:
    """Small test double used by unit tests."""

    def __init__(self, main_outputs: list[str], cold_outputs: list[str]) -> None:
        self.main_outputs = list(main_outputs)
        self.cold_outputs = list(cold_outputs)
        self.calls: list[dict[str, str]] = []

    def chat(self, model: str, system: str, user: str) -> str:
        self.calls.append({"model": model, "system": system, "user": user})
        if system == COLD_EYES_SYSTEM_PROMPT:
            if not self.cold_outputs:
                raise PipelineError("No fake Cold Eyes output left.")
            return self.cold_outputs.pop(0)
        if not self.main_outputs:
            raise PipelineError("No fake Main Agent output left.")
        return self.main_outputs.pop(0)


def load_canon(path: Path) -> str:
    if not path.exists():
        raise SetupError(f"Canon file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def build_main_user_prompt(user_prompt: str, revision: RevisionSignal | None) -> str:
    if revision is None:
        return user_prompt

    if revision.source in {"cold_eyes_fail", "classify_canon_issue"}:
        clause = revision.canon_clause or "an unspecified canon clause"
        return (
            "Original user request:\n"
            f"{user_prompt}\n\n"
            "A previous candidate was rejected for violating "
            f"{clause}. Produce a revised candidate answer. Do not write a refusal. "
            "Return only the revised candidate answer."
        )

    issue = revision.local_issue or "a local formatting issue"
    return (
        "Original user request:\n"
        f"{user_prompt}\n\n"
        f"A previous candidate had {issue}. Produce a clean revised candidate answer. "
        "Do not write a refusal. Return only the revised candidate answer."
    )


def generate_candidate(
    client: Any,
    model: str,
    user_prompt: str,
    revision: RevisionSignal | None,
) -> str:
    return client.chat(
        model=model,
        system=MAIN_AGENT_SYSTEM_PROMPT,
        user=build_main_user_prompt(user_prompt, revision),
    )


def classify_candidate(candidate: str) -> ClassifyResult:
    text = candidate.strip()
    lower = text.lower()

    if not text:
        return ClassifyResult("local_issue", reason="empty_candidate")

    if "<|channel>thought" in lower or "<|think|>" in lower:
        return ClassifyResult("local_issue", reason="thinking_artifact")

    if _looks_like_refusal(text):
        return ClassifyResult("local_issue", reason="candidate_wrote_refusal")

    clause = _detect_obvious_canon_issue(lower)
    if clause is not None:
        return ClassifyResult("canon_issue", canon_clause=clause, reason="keyword_prefilter")

    return ClassifyResult("forward_to_cold_eyes")


def classify_input(prompt: str) -> ClassifyResult:
    clause = _detect_obvious_canon_issue(prompt.lower())
    if clause is not None:
        return ClassifyResult("input_canon_issue", canon_clause=clause, reason="input_prefilter")
    return ClassifyResult("forward_to_main_agent")


def _looks_like_refusal(text: str) -> bool:
    starts = text.strip().lower()
    return starts.startswith(
        (
            "i can't help",
            "i cannot help",
            "i'm unable to help",
            "sorry, i can't",
            "這個請求無法協助",
            "我無法協助",
            "我無法提供",
            "無法提供",
            "我不能提供",
            "不能提供",
            "抱歉，我無法",
            "抱歉，我不能",
        )
    )


def _detect_obvious_canon_issue(lower: str) -> str | None:
    c2_patterns = (
        r"\bbypass\b.{0,80}\b(password|login|auth|authentication|security|permission|monitoring)\b",
        r"\bdisable\b.{0,80}\b(windows defender|antivirus|edr|logging|monitoring)\b",
        r"\bsteal\b.{0,80}\b(cookie|token|credential|password|session)\b",
        r"\bextract\b.{0,80}\b(password|credential|token|cookie)\b",
        r"\bmimikatz\b",
        r"\bprivilege escalation\b",
        r"(繞過|規避|破解|跳過).{0,30}(登入|密碼|驗證|認證|安全|權限|監控|日誌)",
        r"(取得|竊取|偷取|擷取).{0,30}(密碼|憑證|權杖|token|cookie|session)",
        r"(關閉|停用|禁用).{0,30}(防毒|監控|日誌|紀錄|安全系統)",
    )
    c3_patterns = (
        r"\bphishing\b",
        r"\bfake login\b",
        r"\bimpersonat(e|ion)\b.{0,80}\b(account|user|employee|bank|support)\b",
        r"\bspoof\b.{0,80}\b(email|caller|identity|login)\b",
        r"(釣魚|假登入|偽造登入|冒充|冒名|仿冒)",
    )
    c1_patterns = (
        r"\bmake\b.{0,80}\b(explosive|bomb|poison)\b",
        r"\bweapon\b.{0,80}\b(instruction|build|assemble)\b",
        r"\bkill\b.{0,80}\b(step|method|guide|instructions)\b",
        r"(製作|組裝).{0,30}(爆裂物|炸彈|毒物|武器)",
    )

    for pattern in c2_patterns:
        if re.search(pattern, lower, flags=re.DOTALL):
            return "C2"
    for pattern in c3_patterns:
        if re.search(pattern, lower, flags=re.DOTALL):
            return "C3"
    for pattern in c1_patterns:
        if re.search(pattern, lower, flags=re.DOTALL):
            return "C1"
    return None


def cold_eyes_review(client: Any, model: str, canon: str, candidate: str) -> ColdEyesVerdict:
    user_content = (
        "Canon:\n"
        f"{canon}\n\n"
        "Candidate output:\n"
        f"{candidate}\n\n"
        "Return only strict JSON."
    )
    raw = client.chat(model=model, system=COLD_EYES_SYSTEM_PROMPT, user=user_content)
    return parse_cold_eyes_json(raw)


def mechanical_cold_eyes_review(policy_path: Path | str, candidate: str) -> ColdEyesVerdict:
    audit_result = run_audit(candidate, policy_path)
    raw = json.dumps(
        {
            "verdict": audit_result.verdict,
            "canon_clause": audit_result.canon_clause,
            "reason": audit_result.reason,
            "matches": [
                {
                    "rule_id": m.rule_id,
                    "pattern_id": m.pattern_id,
                    "severity": m.severity,
                    "reason": m.reason,
                }
                for m in audit_result.matches
            ],
        },
        ensure_ascii=False,
    )
    return ColdEyesVerdict(
        verdict=audit_result.verdict,
        canon_clause=audit_result.canon_clause,
        reason=audit_result.reason,
        raw=raw,
    )


def parse_cold_eyes_json(raw: str) -> ColdEyesVerdict:
    payload = _extract_json_object(raw)
    if payload is None:
        return ColdEyesVerdict("fail", None, "cold_eyes_invalid_json", raw)

    verdict = str(payload.get("verdict", "")).strip().lower()
    if verdict not in {"pass", "fail"}:
        verdict = "fail"

    clause_value = payload.get("canon_clause")
    canon_clause = None if clause_value in (None, "null", "") else str(clause_value).strip()
    if canon_clause not in {None, "C1", "C2", "C3"}:
        canon_clause = None

    reason = str(payload.get("reason", "")).strip()
    if verdict == "pass" and canon_clause is not None:
        verdict = "fail"
        reason = reason or "cold_eyes_inconsistent_pass_with_canon_clause"
    return ColdEyesVerdict(verdict, canon_clause, reason, raw)


def _extract_json_object(raw: str) -> dict[str, Any] | None:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

    candidates = [text]
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def run_pipeline(
    prompt: str,
    client: Any,
    model: str,
    canon: str,
    log_dir: Path,
) -> RunResult:
    run_id = new_run_id()
    audit: list[AuditEntry] = []
    revision: RevisionSignal | None = None
    final_status = "refused"
    final_output = REFUSAL_OUTPUT

    input_classify = classify_input(prompt)
    if input_classify.route == "input_canon_issue":
        entry = AuditEntry(
            run_id=run_id,
            attempt=1,
            classify_route=input_classify.route,
            canon_clause=input_classify.canon_clause,
            final_status="refused",
        )
        audit.append(entry)
        log_path = write_audit_log(log_dir, run_id, audit, final_status, 1, final_output)
        return RunResult(
            run_id=run_id,
            status=final_status,
            attempts=1,
            output=final_output,
            audit=audit,
            log_path=log_path,
        )

    for attempt in range(1, MAX_ATTEMPTS + 1):
        candidate = generate_candidate(client, model, prompt, revision)
        classify = classify_candidate(candidate)
        entry = AuditEntry(
            run_id=run_id,
            attempt=attempt,
            classify_route=classify.route,
            canon_clause=classify.canon_clause,
        )

        if classify.route == "local_issue":
            if attempt == MAX_ATTEMPTS:
                entry.final_status = "refused"
                audit.append(entry)
                break
            audit.append(entry)
            revision = RevisionSignal(
                source="local_issue",
                local_issue=classify.reason or "a local issue",
            )
            continue

        if classify.route == "canon_issue":
            if attempt == MAX_ATTEMPTS:
                entry.final_status = "refused"
                audit.append(entry)
                break
            audit.append(entry)
            revision = RevisionSignal(
                source="classify_canon_issue",
                canon_clause=classify.canon_clause,
            )
            continue

        verdict = cold_eyes_review(client, model, canon, candidate)
        entry.cold_eyes_verdict = verdict.verdict
        entry.canon_clause = verdict.canon_clause

        if verdict.verdict == "pass":
            entry.final_status = "pass"
            audit.append(entry)
            final_status = "pass"
            final_output = candidate
            break

        if attempt == MAX_ATTEMPTS:
            entry.final_status = "refused"
            audit.append(entry)
            break

        audit.append(entry)
        revision = RevisionSignal(
            source="cold_eyes_fail",
            canon_clause=verdict.canon_clause,
        )

    attempts = audit[-1].attempt if audit else 0
    log_path = write_audit_log(log_dir, run_id, audit, final_status, attempts, final_output)
    return RunResult(
        run_id=run_id,
        status=final_status,
        attempts=attempts,
        output=final_output,
        audit=audit,
        log_path=log_path,
    )


def new_run_id() -> str:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}-{uuid.uuid4().hex[:8]}"


def write_audit_log(
    log_dir: Path,
    run_id: str,
    audit: list[AuditEntry],
    final_status: str,
    attempts: int,
    final_output: str,
) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{run_id}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for entry in audit:
            handle.write(json.dumps(entry.log_dict(), ensure_ascii=False) + "\n")
        final_event = {
            "event": "final",
            "run_id": run_id,
            "status": final_status,
            "attempts": attempts,
            "output_chars": len(final_output),
        }
        handle.write(json.dumps(final_event, ensure_ascii=False) + "\n")
    return path


def read_input(args: argparse.Namespace) -> str:
    if args.prompt is not None:
        return args.prompt
    path = Path(args.input_file)
    return path.read_text(encoding="utf-8")


def add_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model name. Default: {DEFAULT_MODEL}")
    parser.add_argument(
        "--ollama-host",
        default=DEFAULT_OLLAMA_HOST,
        help=f"Ollama host. Default: {DEFAULT_OLLAMA_HOST}",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Per-request Ollama timeout in seconds. Default: {DEFAULT_TIMEOUT_SECONDS}",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fourth Path local CLI prototype using Gemma 4 through Ollama."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run the separated reasoning and audit pipeline.")
    input_group = run.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--prompt", help="User request to process.")
    input_group.add_argument("--input-file", help="UTF-8 file containing the user request.")
    run.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    add_runtime_args(run)
    run.add_argument(
        "--canon",
        default=str(PROJECT_ROOT / "canon.md"),
        help="Path to read-only canon markdown.",
    )
    run.add_argument(
        "--runs-dir",
        default=str(PROJECT_ROOT / "runs"),
        help="Directory for audit JSONL files.",
    )

    diagnose = subparsers.add_parser(
        "diagnose-main",
        help="Call only the Main Agent and print its raw candidate output.",
    )
    diagnose_input = diagnose.add_mutually_exclusive_group(required=True)
    diagnose_input.add_argument("--prompt", help="User request to process.")
    diagnose_input.add_argument("--input-file", help="UTF-8 file containing the user request.")
    diagnose.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    diagnose.add_argument(
        "--show-system-prompt",
        action="store_true",
        help="Include the Main Agent system prompt in the output.",
    )
    add_runtime_args(diagnose)

    chat = subparsers.add_parser("chat", help="Start an interactive audited chat session.")
    add_runtime_args(chat)
    chat.add_argument(
        "--canon",
        default=str(PROJECT_ROOT / "canon.md"),
        help="Path to read-only canon markdown.",
    )
    chat.add_argument(
        "--runs-dir",
        default=str(PROJECT_ROOT / "runs"),
        help="Directory for audit JSONL files.",
    )
    chat.add_argument(
        "--show-audit",
        action="store_true",
        help="Start with detailed audit output enabled.",
    )
    return parser


def render_human(result: RunResult) -> str:
    lines = [
        f"Status: {result.status}",
        f"Attempts: {result.attempts}",
        f"Audit log: {result.log_path}",
        "",
        "Output:",
        result.output,
        "",
        "Audit:",
    ]
    for entry in result.audit:
        lines.append(
            "- attempt {attempt}: classify={classify}, cold_eyes={cold}, canon={canon}, final={final}".format(
                attempt=entry.attempt,
                classify=entry.classify_route,
                cold=entry.cold_eyes_verdict or "-",
                canon=entry.canon_clause or "-",
                final=entry.final_status or "-",
            )
        )
    return "\n".join(lines)


def build_chat_prompt(history: list[ChatMessage], user_message: str) -> str:
    if not history:
        return user_message

    lines = [
        "This is an ongoing chat session. Use the visible conversation history for context.",
        "Conversation history:",
    ]
    for message in history:
        lines.append(f"{message.role}: {message.content}")
    lines.extend(["", "Current user message:", user_message])
    return "\n".join(lines)


def normalize_chat_input(raw: str) -> str:
    return raw.strip().lstrip("\ufeff")


def summarize_chat_audit(result: RunResult) -> str:
    last = result.audit[-1] if result.audit else None
    if last is None:
        return "[audit] status=unknown"
    cold = last.cold_eyes_verdict or "-"
    canon = last.canon_clause or "-"
    return (
        f"[audit] status={result.status}; attempts={result.attempts}; "
        f"route={last.classify_route}; cold_eyes={cold}; canon={canon}"
    )


def render_chat_turn(result: RunResult, show_detailed_audit: bool) -> str:
    lines = [result.output, summarize_chat_audit(result)]
    if show_detailed_audit:
        lines.append(json.dumps(result.public_dict()["audit"], ensure_ascii=False, indent=2))
    return "\n".join(lines)


def run_chat_loop(
    client: Any,
    model: str,
    canon: str,
    log_dir: Path,
    input_func: Any = input,
    output_func: Any = print,
    show_detailed_audit: bool = False,
) -> int:
    history: list[ChatMessage] = []
    output_func("Fourth Path chat mode. Type /help for commands.")

    while True:
        try:
            raw = input_func("你> ")
        except EOFError:
            output_func("")
            output_func("[chat ended]")
            return 0

        user_message = normalize_chat_input(raw)
        if not user_message:
            continue

        command = user_message.lower()
        if command == "/exit":
            output_func("[chat ended]")
            return 0
        if command == "/help":
            output_func(CHAT_HELP)
            continue
        if command == "/reset":
            history.clear()
            output_func("[memory reset]")
            continue
        if command == "/audit":
            show_detailed_audit = not show_detailed_audit
            state = "on" if show_detailed_audit else "off"
            output_func(f"[detailed audit: {state}]")
            continue

        prompt = build_chat_prompt(history, user_message)
        result = run_pipeline(
            prompt=prompt,
            client=client,
            model=model,
            canon=canon,
            log_dir=log_dir,
        )
        output_func(render_chat_turn(result, show_detailed_audit))
        history.append(ChatMessage("user", user_message))
        history.append(ChatMessage("assistant", result.output))


def run_command(args: argparse.Namespace) -> int:
    prompt = read_input(args).strip()
    if not prompt:
        raise SetupError("Input prompt is empty.")

    canon = load_canon(Path(args.canon))
    client = OllamaClient(host=args.ollama_host, timeout=args.timeout)
    client.ensure_ready(args.model)

    result = run_pipeline(
        prompt=prompt,
        client=client,
        model=args.model,
        canon=canon,
        log_dir=Path(args.runs_dir),
    )

    if args.json:
        print(json.dumps(result.public_dict(), ensure_ascii=False, indent=2))
    else:
        print(render_human(result))
    return 0


def diagnose_main(prompt: str, client: Any, model: str, show_system_prompt: bool) -> dict[str, Any]:
    candidate = generate_candidate(client, model, prompt, revision=None)
    return {
        "model": model,
        "system_prompt": MAIN_AGENT_SYSTEM_PROMPT if show_system_prompt else None,
        "prompt": prompt,
        "candidate": candidate,
    }


def diagnose_main_command(args: argparse.Namespace) -> int:
    prompt = read_input(args).strip()
    if not prompt:
        raise SetupError("Input prompt is empty.")

    client = OllamaClient(host=args.ollama_host, timeout=args.timeout)
    client.ensure_ready(args.model)
    result = diagnose_main(
        prompt=prompt,
        client=client,
        model=args.model,
        show_system_prompt=args.show_system_prompt,
    )

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"Model: {args.model}")
        if args.show_system_prompt:
            print("\nSystem prompt:")
            print(MAIN_AGENT_SYSTEM_PROMPT)
        print("\nCandidate:")
        print(result["candidate"])
    return 0


def chat_command(args: argparse.Namespace) -> int:
    canon = load_canon(Path(args.canon))
    client = OllamaClient(host=args.ollama_host, timeout=args.timeout)
    client.ensure_ready(args.model)
    return run_chat_loop(
        client=client,
        model=args.model,
        canon=canon,
        log_dir=Path(args.runs_dir),
        show_detailed_audit=args.show_audit,
    )


def main(argv: list[str] | None = None) -> int:
    configure_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            return run_command(args)
        if args.command == "diagnose-main":
            return diagnose_main_command(args)
        if args.command == "chat":
            return chat_command(args)
    except SetupError as exc:
        print(f"Setup error: {exc}", file=sys.stderr)
        return 2
    except PipelineError as exc:
        print(f"Pipeline error: {exc}", file=sys.stderr)
        return 1
    return 0


def configure_stdio() -> None:
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


if __name__ == "__main__":
    raise SystemExit(main())
