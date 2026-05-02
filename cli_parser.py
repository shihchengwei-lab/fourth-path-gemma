from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from compute_gates import (
    DEFAULT_KV_CACHE_BITS,
    DEFAULT_KV_CACHE_QUANT_BITS,
    DEFAULT_QWEN3_8B_CONTEXT,
    DEFAULT_QWEN3_8B_HEAD_DIM,
    DEFAULT_QWEN3_8B_KV_HEADS,
    DEFAULT_QWEN3_8B_LAYERS,
    DEFAULT_R2R_LARGE_PARAMS_B,
    DEFAULT_R2R_LARGE_TOKEN_RATE,
    DEFAULT_R2R_ROUTER_PARAMS_B,
    DEFAULT_R2R_SMALL_PARAMS_B,
    TOKEN_BACKEND_CHOICES,
)
from core_types import SetupError
from latent_headroom import DEFAULT_LATENT_HEADROOM_VARIANTS
from runtime_config import ModelOptions, RoleRuntime, RuntimeConfig


@dataclass(frozen=True)
class CliParserConfig:
    project_root: Path
    runtime_profiles: Mapping[str, RuntimeConfig]
    default_ollama_host: str
    default_timeout_seconds: int
    default_contrast_expert_profile: str
    default_contrast_amateur_profile: str


def add_role_option_args(parser: argparse.ArgumentParser, role: str) -> None:
    label = role.replace("_", "-")
    parser.add_argument(f"--{label}-num-ctx", type=int, help=f"{role} context window.")
    parser.add_argument(f"--{label}-num-predict", type=int, help=f"{role} maximum generated tokens.")
    parser.add_argument(f"--{label}-temperature", type=float, help=f"{role} sampling temperature.")
    parser.add_argument(f"--{label}-top-p", type=float, help=f"{role} top-p sampling value.")
    parser.add_argument(f"--{label}-top-k", type=int, help=f"{role} top-k sampling value.")
    parser.add_argument(f"--{label}-min-p", type=float, help=f"{role} min-p sampling value.")
    parser.add_argument(
        f"--{label}-no-think",
        action="store_true",
        help=f"Append /no_think for models that support it, such as Qwen3.",
    )


def add_runtime_args(parser: argparse.ArgumentParser, config: CliParserConfig) -> None:
    parser.add_argument(
        "--profile",
        choices=sorted(config.runtime_profiles),
        default="legacy",
        help="Runtime profile. Default: legacy.",
    )
    parser.add_argument(
        "--model",
        help="Compatibility shortcut: use one model for both Main Agent and Cold Eyes.",
    )
    parser.add_argument("--main-model", help="Ollama model for the Main Agent.")
    parser.add_argument("--audit-model", help="Ollama model for Cold Eyes.")
    parser.add_argument("--max-attempts", type=int, help="Bounded repair attempts. Must be at least 1.")
    parser.add_argument(
        "--quality-refine-passes",
        type=int,
        help="Main Agent self-refinement passes before Classify and Cold Eyes. Default: profile value.",
    )
    parser.add_argument(
        "--search-candidates",
        type=int,
        help="Candidate answers to generate before quality selection. Default: profile value.",
    )
    parser.add_argument(
        "--local-select",
        action="store_true",
        help="Apply lightweight local candidate selection after Main Agent generation.",
    )
    parser.add_argument(
        "--adaptive-compute",
        action="store_true",
        help="Choose extra Main Agent compute per prompt shape instead of using a fixed refine/search setting.",
    )
    parser.add_argument(
        "--keep-alive",
        help="Ollama keep_alive value for both roles, such as 30m or 0. Overrides the profile default.",
    )
    add_role_option_args(parser, "main")
    add_role_option_args(parser, "audit")
    parser.add_argument(
        "--ollama-host",
        default=config.default_ollama_host,
        help=f"Ollama host. Default: {config.default_ollama_host}",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=config.default_timeout_seconds,
        help=f"Per-request Ollama timeout in seconds. Default: {config.default_timeout_seconds}",
    )


def arg_or_default(args: argparse.Namespace, name: str, default: Any) -> Any:
    value = getattr(args, name)
    return default if value is None else value


def override_options(base: ModelOptions, args: argparse.Namespace, role: str) -> ModelOptions:
    return ModelOptions(
        num_ctx=arg_or_default(args, f"{role}_num_ctx", base.num_ctx),
        num_predict=arg_or_default(args, f"{role}_num_predict", base.num_predict),
        temperature=arg_or_default(args, f"{role}_temperature", base.temperature),
        top_p=arg_or_default(args, f"{role}_top_p", base.top_p),
        top_k=arg_or_default(args, f"{role}_top_k", base.top_k),
        min_p=arg_or_default(args, f"{role}_min_p", base.min_p),
    )


def build_runtime_from_args(args: argparse.Namespace, config: CliParserConfig) -> RuntimeConfig:
    base = config.runtime_profiles[args.profile]
    max_attempts = args.max_attempts if args.max_attempts is not None else base.max_attempts
    if max_attempts < 1:
        raise SetupError("--max-attempts must be at least 1.")
    quality_refine_passes = (
        args.quality_refine_passes
        if args.quality_refine_passes is not None
        else base.quality_refine_passes
    )
    if quality_refine_passes < 0:
        raise SetupError("--quality-refine-passes must be zero or greater.")
    search_candidates = args.search_candidates if args.search_candidates is not None else base.search_candidates
    if search_candidates < 1:
        raise SetupError("--search-candidates must be at least 1.")

    main_model = args.main_model or args.model or base.main.model
    audit_model = args.audit_model or args.model or base.audit.model
    main_keep_alive = args.keep_alive if args.keep_alive is not None else base.main.keep_alive
    audit_keep_alive = args.keep_alive if args.keep_alive is not None else base.audit.keep_alive
    return RuntimeConfig(
        main=RoleRuntime(
            main_model,
            override_options(base.main.options, args, "main"),
            no_think=base.main.no_think or args.main_no_think,
            keep_alive=main_keep_alive,
            response_format=base.main.response_format,
        ),
        audit=RoleRuntime(
            audit_model,
            override_options(base.audit.options, args, "audit"),
            no_think=base.audit.no_think or args.audit_no_think,
            keep_alive=audit_keep_alive,
            response_format=base.audit.response_format,
        ),
        max_attempts=max_attempts,
        quality_refine_passes=quality_refine_passes,
        search_candidates=search_candidates,
        local_select=base.local_select or args.local_select,
        adaptive_compute=base.adaptive_compute or args.adaptive_compute,
    )


def build_parser(config: CliParserConfig) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fourth Path local CLI prototype using open-weight models through Ollama."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    profiles = subparsers.add_parser("profiles", help="List runtime profiles without calling Ollama.")
    profiles.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    architecture_check = subparsers.add_parser(
        "architecture-check",
        help="Validate separation-and-audit authority invariants without calling Ollama.",
    )
    architecture_check.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    action_audit = subparsers.add_parser(
        "action-audit",
        help="Audit one external side-effect candidate before execution.",
    )
    action_audit.add_argument("--action-type", required=True, help="Action kind, such as noop or network_request.")
    action_audit.add_argument("--target", required=True, help="Action target. Not echoed in summaries.")
    action_audit.add_argument("--intent", required=True, help="Intended purpose. Not echoed in summaries.")
    action_audit.add_argument("--args-summary", required=True, help="Short argument summary. Not echoed in summaries.")
    action_audit.add_argument("--risk-surface", required=True, help="Declared risk surface for the action.")
    action_audit.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    architecture_adversarial_check = subparsers.add_parser(
        "architecture-adversarial-check",
        help="Validate the architecture-boundary adversarial seed corpus.",
    )
    architecture_adversarial_check.add_argument(
        "--input-file",
        default=str(config.project_root / "data" / "architecture_adversarial_seed.jsonl"),
        help="JSONL corpus path. Default: data/architecture_adversarial_seed.jsonl.",
    )
    architecture_adversarial_check.add_argument("--min-total", type=int, default=0, help="Minimum required records.")
    architecture_adversarial_check.add_argument(
        "--min-layer",
        type=int,
        default=0,
        help="Minimum required records per architecture layer.",
    )
    architecture_adversarial_check.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    warm = subparsers.add_parser("warm", help="Preload runtime model(s) through Ollama keep_alive.")
    add_runtime_args(warm, config)
    warm.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    run = subparsers.add_parser("run", help="Run the separated reasoning and audit pipeline.")
    input_group = run.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--prompt", help="User request to process.")
    input_group.add_argument("--input-file", help="UTF-8 file containing the user request.")
    run.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    add_runtime_args(run, config)
    run.add_argument(
        "--canon",
        default=str(config.project_root / "canon.md"),
        help="Path to read-only canon markdown.",
    )
    run.add_argument(
        "--runs-dir",
        default=str(config.project_root / "runs"),
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
    add_runtime_args(diagnose, config)

    chat = subparsers.add_parser("chat", help="Start an interactive audited chat session.")
    add_runtime_args(chat, config)
    chat.add_argument(
        "--canon",
        default=str(config.project_root / "canon.md"),
        help="Path to read-only canon markdown.",
    )
    chat.add_argument(
        "--runs-dir",
        default=str(config.project_root / "runs"),
        help="Directory for audit JSONL files.",
    )
    chat.add_argument(
        "--show-audit",
        action="store_true",
        help="Start with detailed audit output enabled.",
    )

    bench = subparsers.add_parser("bench", help="Run a fixed local benchmark suite for one runtime profile.")
    add_runtime_args(bench, config)
    bench.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    bench.add_argument("--repeat", type=int, default=1, help="Repeat the benchmark suite. Default: 1.")
    bench.add_argument("--warmup", action="store_true", help="Preload model(s) before timed benchmark cases.")
    bench.add_argument(
        "--canon",
        default=str(config.project_root / "canon.md"),
        help="Path to read-only canon markdown.",
    )
    bench.add_argument(
        "--runs-dir",
        default=str(config.project_root / "runs"),
        help="Directory for audit JSONL files and benchmark summaries.",
    )
    bench.add_argument(
        "--output-file",
        help="Optional benchmark summary JSON path. Default: runs/bench-<run-id>.json",
    )

    distill = subparsers.add_parser(
        "distill-check",
        help="Validate the synthetic Cold Eyes distillation seed corpus.",
    )
    distill.add_argument(
        "--input-file",
        default=str(config.project_root / "data" / "cold_eyes_seed.jsonl"),
        help="JSONL corpus path. Default: data/cold_eyes_seed.jsonl.",
    )
    distill.add_argument("--min-pass", type=int, default=0, help="Minimum required pass records.")
    distill.add_argument("--min-fail", type=int, default=0, help="Minimum required fail records.")
    distill.add_argument("--min-clause", type=int, default=0, help="Minimum required records per C1/C2/C3 clause.")
    distill.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    verifier_tool_gate = subparsers.add_parser(
        "verifier-tool-gate",
        help="Run local verifier and pre-tool-use boundary gates without calling Ollama.",
    )
    verifier_tool_gate.add_argument(
        "--distill-file",
        default=str(config.project_root / "data" / "cold_eyes_seed.jsonl"),
        help="Cold Eyes verifier JSONL path. Default: data/cold_eyes_seed.jsonl.",
    )
    verifier_tool_gate.add_argument("--min-pass", type=int, default=19, help="Minimum required pass records.")
    verifier_tool_gate.add_argument("--min-fail", type=int, default=25, help="Minimum required fail records.")
    verifier_tool_gate.add_argument(
        "--min-clause",
        type=int,
        default=8,
        help="Minimum required records per C1/C2/C3 clause.",
    )
    verifier_tool_gate.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    main_check = subparsers.add_parser(
        "main-check",
        help="Validate the synthetic Main Agent role-behavior corpus.",
    )
    main_check.add_argument(
        "--input-file",
        default=str(config.project_root / "data" / "main_agent_seed.jsonl"),
        help="JSONL corpus path. Default: data/main_agent_seed.jsonl.",
    )
    main_check.add_argument("--min-total", type=int, default=0, help="Minimum required records.")
    main_check.add_argument("--min-category", type=int, default=0, help="Minimum required records per category.")
    main_check.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    main_quality = subparsers.add_parser(
        "main-data-quality-check",
        help="Check Main Agent distillation seed quality across train, hard, and held-out files.",
    )
    main_quality.add_argument(
        "--input-file",
        action="append",
        help="JSONL corpus path. Can be repeated. Defaults to seed, hard seed, and held-out seed.",
    )
    main_quality.add_argument(
        "--require-verifier-pattern",
        action="append",
        help="Require verifier coverage for files whose name contains this text. Defaults to hard and heldout.",
    )
    main_quality.add_argument(
        "--max-category-share",
        type=float,
        default=0.5,
        help="Maximum allowed share for one category once a file reaches the balance threshold.",
    )
    main_quality.add_argument(
        "--min-records-for-category-balance",
        type=int,
        default=8,
        help="Minimum records before enforcing the dominant-category share gate.",
    )
    main_quality.add_argument(
        "--min-verifier-types",
        type=int,
        default=3,
        help="Minimum verifier field types required in verifier-required files.",
    )
    main_quality.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    main_quality_report = subparsers.add_parser(
        "main-data-quality-report",
        help="Report Main Agent corpus quality metadata without treating findings as a release gate.",
    )
    main_quality_report.add_argument(
        "--input-file",
        action="append",
        help="JSONL corpus path. Can be repeated. Defaults to seed, hard seed, held-out seed, rotated held-out seed, and fresh held-out seed.",
    )
    main_quality_report.add_argument(
        "--require-verifier-pattern",
        action="append",
        help="Mark files whose name contains this text as verifier-required. Defaults to hard and heldout.",
    )
    main_quality_report.add_argument(
        "--max-category-share",
        type=float,
        default=0.5,
        help="Dominant-category share threshold used for the report. Default: 0.5.",
    )
    main_quality_report.add_argument(
        "--min-records-for-category-balance",
        type=int,
        default=8,
        help="Minimum records before reporting dominant-category share as gated. Default: 8.",
    )
    main_quality_report.add_argument(
        "--min-verifier-types",
        type=int,
        default=3,
        help="Minimum verifier field types expected in verifier-required files. Default: 3.",
    )
    main_quality_report.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    main_sft = subparsers.add_parser(
        "main-sft-export",
        help="Export the Main Agent seed corpus as chat-style SFT JSONL for LoRA experiments.",
    )
    main_sft.add_argument(
        "--input-file",
        default=str(config.project_root / "data" / "main_agent_seed.jsonl"),
        help="JSONL corpus path. Default: data/main_agent_seed.jsonl.",
    )
    main_sft.add_argument(
        "--output-file",
        default=str(config.project_root / "runs" / "main-agent-sft.jsonl"),
        help="Output JSONL path. Default: runs/main-agent-sft.jsonl.",
    )
    main_sft.add_argument(
        "--no-system",
        action="store_true",
        help="Omit the Main Agent system prompt from exported messages.",
    )
    main_sft.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    main_contrast = subparsers.add_parser(
        "main-contrast-export",
        help="Export high-divergence expert/amateur Main Agent samples for LightReasoner-style LoRA experiments.",
    )
    main_contrast.add_argument(
        "--input-file",
        default=str(config.project_root / "data" / "main_agent_hard_seed.jsonl"),
        help="JSONL corpus path. Default: data/main_agent_hard_seed.jsonl.",
    )
    main_contrast.add_argument(
        "--output-file",
        default=str(config.project_root / "runs" / "main-agent-contrast.jsonl"),
        help="Output JSONL path. Default: runs/main-agent-contrast.jsonl.",
    )
    main_contrast.add_argument(
        "--expert-profile",
        choices=sorted(config.runtime_profiles),
        default=config.default_contrast_expert_profile,
        help=f"Profile used as the stronger generator. Default: {config.default_contrast_expert_profile}.",
    )
    main_contrast.add_argument(
        "--amateur-profile",
        choices=sorted(config.runtime_profiles),
        default=config.default_contrast_amateur_profile,
        help=f"Profile used as the weaker contrast model. Default: {config.default_contrast_amateur_profile}.",
    )
    main_contrast.add_argument(
        "--min-score-gap",
        type=float,
        default=100.0,
        help="Minimum amateur-minus-expert score gap required for export. Default: 100.",
    )
    main_contrast.add_argument(
        "--max-length-ratio",
        type=float,
        help="Treat outputs longer than this output/target character ratio as issues.",
    )
    main_contrast.add_argument(
        "--no-system",
        action="store_true",
        help="Omit the Main Agent system prompt from exported messages.",
    )
    main_contrast.add_argument("--ollama-host", default=config.default_ollama_host, help="Ollama host URL.")
    main_contrast.add_argument("--timeout", type=int, default=config.default_timeout_seconds, help="Ollama timeout seconds.")
    main_contrast.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    main_r1 = subparsers.add_parser(
        "main-r1-sample-export",
        help="Export verifier-rewarded Main Agent samples for DeepSeek-R1-style rejection-sampling LoRA data.",
    )
    main_r1.add_argument(
        "--input-file",
        default=str(config.project_root / "data" / "main_agent_hard_seed.jsonl"),
        help="JSONL corpus path. Default: data/main_agent_hard_seed.jsonl.",
    )
    main_r1.add_argument(
        "--output-file",
        default=str(config.project_root / "runs" / "main-agent-r1-samples.jsonl"),
        help="Output JSONL path. Default: runs/main-agent-r1-samples.jsonl.",
    )
    main_r1.add_argument(
        "--profile",
        choices=sorted(config.runtime_profiles),
        default=config.default_contrast_expert_profile,
        help=f"Generator profile. Default: {config.default_contrast_expert_profile}.",
    )
    main_r1.add_argument(
        "--samples-per-record",
        type=int,
        default=4,
        help="How many candidate rollouts to sample per record. Default: 4.",
    )
    main_r1.add_argument(
        "--min-reward",
        type=float,
        default=1.0,
        help="Minimum verifier reward required for export. Default: 1.0.",
    )
    main_r1.add_argument(
        "--max-length-ratio",
        type=float,
        help="Treat outputs longer than this output/target character ratio as issues.",
    )
    main_r1.add_argument(
        "--no-system",
        action="store_true",
        help="Omit the Main Agent system prompt from exported messages.",
    )
    main_r1.add_argument("--ollama-host", default=config.default_ollama_host, help="Ollama host URL.")
    main_r1.add_argument("--timeout", type=int, default=config.default_timeout_seconds, help="Ollama timeout seconds.")
    main_r1.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    main_limo = subparsers.add_parser(
        "main-limo-curate",
        help="Curate a small LIMO-style cognitive-template set from accepted Main Agent SFT rows.",
    )
    main_limo.add_argument(
        "--input-file",
        default=str(config.project_root / "runs" / "main-agent-r1-samples.jsonl"),
        help="Input SFT-style JSONL path. Default: runs/main-agent-r1-samples.jsonl.",
    )
    main_limo.add_argument(
        "--output-file",
        default=str(config.project_root / "runs" / "main-agent-limo-curated.jsonl"),
        help="Output JSONL path. Default: runs/main-agent-limo-curated.jsonl.",
    )
    main_limo.add_argument(
        "--max-records",
        type=int,
        default=800,
        help="Maximum curated rows to keep. Default: 800.",
    )
    main_limo.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Minimum LIMO template score required for selection. Default: 0.",
    )
    main_limo.add_argument(
        "--max-per-category",
        type=int,
        default=0,
        help="Optional maximum selected rows per category. Default: 0 means no cap.",
    )
    main_limo.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    main_mix = subparsers.add_parser(
        "main-mix-distill-curate",
        help="Curate a small-model-friendly short/long reasoning mix from SFT rows.",
    )
    main_mix.add_argument(
        "--input-file",
        default=str(config.project_root / "runs" / "main-agent-limo-curated.jsonl"),
        help="Input SFT-style JSONL path. Default: runs/main-agent-limo-curated.jsonl.",
    )
    main_mix.add_argument(
        "--output-file",
        default=str(config.project_root / "runs" / "main-agent-mix-distill.jsonl"),
        help="Output JSONL path. Default: runs/main-agent-mix-distill.jsonl.",
    )
    main_mix.add_argument(
        "--max-records",
        type=int,
        default=800,
        help="Maximum curated rows to keep. Default: 800.",
    )
    main_mix.add_argument(
        "--long-ratio",
        type=float,
        default=0.2,
        help="Target fraction of long reasoning rows. Default: 0.2.",
    )
    main_mix.add_argument(
        "--long-char-threshold",
        type=int,
        default=1200,
        help="Assistant character length treated as long reasoning. Default: 1200.",
    )
    main_mix.add_argument(
        "--max-per-category",
        type=int,
        default=0,
        help="Optional maximum selected rows per category. Default: 0 means no cap.",
    )
    main_mix.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    training_report = subparsers.add_parser(
        "main-training-data-report",
        help="Summarize SFT JSONL training-data quality without printing row text.",
    )
    training_report.add_argument(
        "--input-file",
        default=str(config.project_root / "runs" / "main-agent-mix-distill.jsonl"),
        help="Input SFT-style JSONL path. Default: runs/main-agent-mix-distill.jsonl.",
    )
    training_report.add_argument(
        "--long-char-threshold",
        type=int,
        default=1200,
        help="Assistant character length treated as long reasoning. Default: 1200.",
    )
    training_report.add_argument(
        "--require-system",
        action="store_true",
        help="Fail if any row is missing a system message.",
    )
    training_report.add_argument(
        "--require-generated-metadata",
        action="store_true",
        help="Fail if generated rows are missing source, split, or verifier label metadata.",
    )
    training_report.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    distill_pipeline = subparsers.add_parser(
        "main-distill-pipeline",
        help="Run R1-lite sampling, LIMO curation, Mix Distillation curation, and write a manifest.",
    )
    distill_pipeline.add_argument(
        "--input-file",
        default=str(config.project_root / "data" / "main_agent_hard_seed.jsonl"),
        help="Verifier-backed JSONL corpus path. Default: data/main_agent_hard_seed.jsonl.",
    )
    distill_pipeline.add_argument(
        "--runs-dir",
        default=str(config.project_root / "runs"),
        help="Directory for pipeline artifacts. Default: runs.",
    )
    distill_pipeline.add_argument(
        "--profile",
        choices=sorted(config.runtime_profiles),
        default=config.default_contrast_expert_profile,
        help=f"Generator profile. Default: {config.default_contrast_expert_profile}.",
    )
    distill_pipeline.add_argument("--samples-per-record", type=int, default=4)
    distill_pipeline.add_argument("--min-reward", type=float, default=1.0)
    distill_pipeline.add_argument("--max-length-ratio", type=float)
    distill_pipeline.add_argument("--limo-max-records", type=int, default=800)
    distill_pipeline.add_argument("--limo-min-score", type=float, default=0.0)
    distill_pipeline.add_argument("--mix-max-records", type=int, default=800)
    distill_pipeline.add_argument("--mix-long-ratio", type=float, default=0.2)
    distill_pipeline.add_argument("--mix-long-char-threshold", type=int, default=1200)
    distill_pipeline.add_argument("--mix-max-per-category", type=int, default=0)
    distill_pipeline.add_argument("--no-system", action="store_true")
    distill_pipeline.add_argument("--ollama-host", default=config.default_ollama_host, help="Ollama host URL.")
    distill_pipeline.add_argument("--timeout", type=int, default=config.default_timeout_seconds, help="Ollama timeout seconds.")
    distill_pipeline.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    r2r_estimate = subparsers.add_parser(
        "r2r-estimate",
        help="Estimate small/large token-routing economics and backend readiness for R2R-style inference.",
    )
    r2r_estimate.add_argument(
        "--small-params-b",
        type=float,
        default=DEFAULT_R2R_SMALL_PARAMS_B,
        help=f"Small model parameter count in billions. Default: {DEFAULT_R2R_SMALL_PARAMS_B}.",
    )
    r2r_estimate.add_argument(
        "--large-params-b",
        type=float,
        default=DEFAULT_R2R_LARGE_PARAMS_B,
        help=f"Large model parameter count in billions. Default: {DEFAULT_R2R_LARGE_PARAMS_B}.",
    )
    r2r_estimate.add_argument(
        "--router-params-b",
        type=float,
        default=DEFAULT_R2R_ROUTER_PARAMS_B,
        help=f"Router parameter count in billions. Default: {DEFAULT_R2R_ROUTER_PARAMS_B}.",
    )
    r2r_estimate.add_argument(
        "--large-token-rate",
        type=float,
        default=DEFAULT_R2R_LARGE_TOKEN_RATE,
        help=f"Fraction of tokens routed to the large model. Default: {DEFAULT_R2R_LARGE_TOKEN_RATE}.",
    )
    r2r_estimate.add_argument(
        "--output-tokens",
        type=int,
        default=1000,
        help="Output-token count used for the cost estimate. Default: 1000.",
    )
    r2r_estimate.add_argument(
        "--backend",
        choices=TOKEN_BACKEND_CHOICES,
        default="ollama-chat",
        help="Backend capability model. Default: ollama-chat.",
    )
    r2r_estimate.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    kv_cache_estimate = subparsers.add_parser(
        "kv-cache-estimate",
        help="Estimate Qwen3-style KV cache memory pressure and quantized-KV upside.",
    )
    kv_cache_estimate.add_argument(
        "--layers",
        type=int,
        default=DEFAULT_QWEN3_8B_LAYERS,
        help=f"Transformer layer count. Default: {DEFAULT_QWEN3_8B_LAYERS}.",
    )
    kv_cache_estimate.add_argument(
        "--kv-heads",
        type=int,
        default=DEFAULT_QWEN3_8B_KV_HEADS,
        help=f"Key/value head count. Default: {DEFAULT_QWEN3_8B_KV_HEADS}.",
    )
    kv_cache_estimate.add_argument(
        "--head-dim",
        type=int,
        default=DEFAULT_QWEN3_8B_HEAD_DIM,
        help=f"Attention head dimension. Default: {DEFAULT_QWEN3_8B_HEAD_DIM}.",
    )
    kv_cache_estimate.add_argument(
        "--context-tokens",
        type=int,
        default=DEFAULT_QWEN3_8B_CONTEXT,
        help=f"Prompt plus generated-token context length. Default: {DEFAULT_QWEN3_8B_CONTEXT}.",
    )
    kv_cache_estimate.add_argument("--batch-size", type=int, default=1, help="Batch size. Default: 1.")
    kv_cache_estimate.add_argument(
        "--kv-bits",
        type=int,
        default=DEFAULT_KV_CACHE_BITS,
        help=f"Base KV cache precision in bits. Default: {DEFAULT_KV_CACHE_BITS}.",
    )
    kv_cache_estimate.add_argument(
        "--quantized-kv-bits",
        type=int,
        default=DEFAULT_KV_CACHE_QUANT_BITS,
        help=f"Optional quantized KV cache precision in bits. Default: {DEFAULT_KV_CACHE_QUANT_BITS}.",
    )
    kv_cache_estimate.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    next_token_headroom = subparsers.add_parser(
        "next-token-headroom",
        help="Audit whether next-token selection can improve under current or token-level backends.",
    )
    next_token_headroom.add_argument(
        "--backend",
        choices=TOKEN_BACKEND_CHOICES,
        default="ollama-chat",
        help="Backend capability model. Default: ollama-chat.",
    )
    next_token_headroom.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    inference_compute_gate = subparsers.add_parser(
        "inference-compute-gate",
        help="Check that inference-time compute is gated by data quality and verifier/tool-use readiness.",
    )
    inference_compute_gate.add_argument(
        "--distill-file",
        default=str(config.project_root / "data" / "cold_eyes_seed.jsonl"),
        help="Cold Eyes verifier JSONL path. Default: data/cold_eyes_seed.jsonl.",
    )
    inference_compute_gate.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    local_release_gate = subparsers.add_parser(
        "local-release-gate",
        help="Run all local no-Ollama release gates in priority order.",
    )
    local_release_gate.add_argument(
        "--distill-file",
        default=str(config.project_root / "data" / "cold_eyes_seed.jsonl"),
        help="Cold Eyes verifier JSONL path. Default: data/cold_eyes_seed.jsonl.",
    )
    local_release_gate.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    idle_summary = subparsers.add_parser(
        "idle-run-summary",
        help="Summarize one idle long-run log and its timestamped JSON artifacts without printing prompts.",
    )
    idle_summary.add_argument(
        "--runs-dir",
        default=str(config.project_root / "runs"),
        help="Directory containing idle long-run logs and JSON summaries. Default: runs.",
    )
    idle_summary.add_argument(
        "--stamp",
        help="Idle run timestamp, such as 20260502-053750. Default: latest idle-long-run log.",
    )
    idle_summary.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    main_eval = subparsers.add_parser(
        "main-eval",
        help="Evaluate Main Agent role behavior against the synthetic corpus.",
    )
    add_runtime_args(main_eval, config)
    main_eval.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    main_eval.add_argument(
        "--input-file",
        default=str(config.project_root / "data" / "main_agent_seed.jsonl"),
        help="JSONL corpus path. Default: data/main_agent_seed.jsonl.",
    )
    main_eval.add_argument(
        "--runs-dir",
        default=str(config.project_root / "runs"),
        help="Directory for Main Agent evaluation summaries.",
    )
    main_eval.add_argument(
        "--output-file",
        help="Optional Main Agent evaluation JSON path. Default: runs/main-eval-<run-id>.json",
    )
    main_eval.add_argument(
        "--max-issue-rate",
        type=float,
        default=1.0,
        help="Maximum allowed issue rate for a zero exit code. Default: 1.0.",
    )
    main_eval.add_argument(
        "--max-refusal-rate",
        type=float,
        default=1.0,
        help="Maximum allowed refusal-like rate for a zero exit code. Default: 1.0.",
    )
    main_eval.add_argument(
        "--max-length-ratio",
        type=float,
        help="Flag outputs longer than this output/target character ratio as overlong.",
    )

    main_eval_ablation = subparsers.add_parser(
        "main-eval-ablation",
        help="Run multiple Main Agent eval profiles on the same corpus and compare cost per clean case.",
    )
    main_eval_ablation.add_argument(
        "--profile",
        action="append",
        choices=sorted(config.runtime_profiles),
        help="Profile to evaluate. Can be repeated. Defaults to local-max, s2t-lite, and compute-optimal-lite.",
    )
    main_eval_ablation.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    main_eval_ablation.add_argument(
        "--input-file",
        default=str(config.project_root / "data" / "main_agent_fresh_heldout_seed.jsonl"),
        help="JSONL corpus path. Default: data/main_agent_fresh_heldout_seed.jsonl.",
    )
    main_eval_ablation.add_argument(
        "--runs-dir",
        default=str(config.project_root / "runs"),
        help="Directory for ablation summaries. Default: runs.",
    )
    main_eval_ablation.add_argument(
        "--output-file",
        help="Optional ablation JSON path. Default: runs/main-eval-ablation-<run-id>.json",
    )
    main_eval_ablation.add_argument(
        "--max-length-ratio",
        type=float,
        help="Flag outputs longer than this output/target character ratio as overlong.",
    )
    main_eval_ablation.add_argument("--ollama-host", default=config.default_ollama_host, help="Ollama host URL.")
    main_eval_ablation.add_argument("--timeout", type=int, default=config.default_timeout_seconds, help="Ollama timeout seconds.")

    main_eval_failure_report = subparsers.add_parser(
        "main-eval-failure-report",
        help="Summarize saved main-eval or main-eval-ablation JSON without printing prompts or outputs.",
    )
    main_eval_failure_report.add_argument(
        "--input-file",
        required=True,
        help="Saved main-eval or main-eval-ablation JSON path.",
    )
    main_eval_failure_report.add_argument(
        "--runs-dir",
        default=str(config.project_root / "runs"),
        help="Directory for failure reports. Default: runs.",
    )
    main_eval_failure_report.add_argument(
        "--output-file",
        help="Optional failure-report JSON path. Default: runs/main-eval-failure-report-<run-id>.json",
    )
    main_eval_failure_report.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    main_latent_headroom = subparsers.add_parser(
        "main-latent-headroom",
        help="Probe bottom-model latent capability with repeated prompt-shape attempts without printing prompts or outputs.",
    )
    main_latent_headroom.add_argument(
        "--profile",
        choices=sorted(config.runtime_profiles),
        default="qwen3-8b-local-max",
        help="Runtime profile. Default: qwen3-8b-local-max.",
    )
    main_latent_headroom.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    main_latent_headroom.add_argument(
        "--input-file",
        default=str(config.project_root / "data" / "main_agent_latent_probe_seed.jsonl"),
        help="JSONL corpus path. Default: data/main_agent_latent_probe_seed.jsonl.",
    )
    main_latent_headroom.add_argument(
        "--variant",
        action="append",
        choices=DEFAULT_LATENT_HEADROOM_VARIANTS,
        help="Prompt-shape variant. Can be repeated. Defaults to baseline, constraint_first, and self_check.",
    )
    main_latent_headroom.add_argument(
        "--attempts-per-variant",
        type=int,
        default=2,
        help="Repeated generations per prompt-shape variant. Default: 2.",
    )
    main_latent_headroom.add_argument(
        "--max-length-ratio",
        type=float,
        help="Flag outputs longer than this output/target character ratio as overlong.",
    )
    main_latent_headroom.add_argument(
        "--runs-dir",
        default=str(config.project_root / "runs"),
        help="Directory for latent-headroom summaries. Default: runs.",
    )
    main_latent_headroom.add_argument(
        "--output-file",
        help="Optional latent-headroom JSON path. Default: runs/main-latent-headroom-<run-id>.json",
    )
    main_latent_headroom.add_argument("--ollama-host", default=config.default_ollama_host, help="Ollama host URL.")
    main_latent_headroom.add_argument("--timeout", type=int, default=config.default_timeout_seconds, help="Ollama timeout seconds.")

    architecture_adversarial_eval = subparsers.add_parser(
        "architecture-adversarial-eval",
        help="Evaluate role-boundary adversarial cases against Main Agent and Cold Eyes.",
    )
    add_runtime_args(architecture_adversarial_eval, config)
    architecture_adversarial_eval.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    architecture_adversarial_eval.add_argument(
        "--input-file",
        default=str(config.project_root / "data" / "architecture_adversarial_seed.jsonl"),
        help="JSONL corpus path. Default: data/architecture_adversarial_seed.jsonl.",
    )
    architecture_adversarial_eval.add_argument(
        "--canon",
        default=str(config.project_root / "canon.md"),
        help="Path to read-only canon markdown.",
    )
    architecture_adversarial_eval.add_argument(
        "--runs-dir",
        default=str(config.project_root / "runs"),
        help="Directory for architecture adversarial evaluation summaries.",
    )
    architecture_adversarial_eval.add_argument(
        "--output-file",
        help="Optional architecture adversarial evaluation JSON path. "
        "Default: runs/architecture-adversarial-eval-<run-id>.json",
    )
    architecture_adversarial_eval.add_argument(
        "--min-pass-rate",
        type=float,
        default=0.0,
        help="Minimum pass rate required for a zero exit code. Default: 0.0.",
    )

    distill_eval = subparsers.add_parser(
        "distill-eval",
        help="Evaluate the audit model against the synthetic Cold Eyes corpus.",
    )
    add_runtime_args(distill_eval, config)
    distill_eval.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    distill_eval.add_argument(
        "--input-file",
        default=str(config.project_root / "data" / "cold_eyes_seed.jsonl"),
        help="JSONL corpus path. Default: data/cold_eyes_seed.jsonl.",
    )
    distill_eval.add_argument(
        "--canon",
        default=str(config.project_root / "canon.md"),
        help="Path to read-only canon markdown.",
    )
    distill_eval.add_argument(
        "--runs-dir",
        default=str(config.project_root / "runs"),
        help="Directory for distillation evaluation summaries.",
    )
    distill_eval.add_argument(
        "--output-file",
        help="Optional distillation evaluation JSON path. Default: runs/distill-eval-<run-id>.json",
    )
    distill_eval.add_argument(
        "--require-exact",
        action="store_true",
        help="Exit non-zero unless every record matches both verdict and canon clause.",
    )
    distill_eval.add_argument(
        "--min-exact-accuracy",
        type=float,
        default=0.0,
        help="Minimum exact accuracy required for a zero exit code. Default: 0.0.",
    )
    distill_eval.add_argument(
        "--min-mechanical-cases",
        type=int,
        default=0,
        help="Minimum mechanical audit cases required for a zero exit code. Default: 0.",
    )
    return parser
