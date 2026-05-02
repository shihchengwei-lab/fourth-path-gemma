from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from core_types import SetupError


DEFAULT_R2R_SMALL_PARAMS_B = 1.7
DEFAULT_R2R_LARGE_PARAMS_B = 8.0
DEFAULT_R2R_ROUTER_PARAMS_B = 0.056
DEFAULT_R2R_LARGE_TOKEN_RATE = 0.13
DEFAULT_QWEN3_8B_LAYERS = 36
DEFAULT_QWEN3_8B_KV_HEADS = 8
DEFAULT_QWEN3_8B_HEAD_DIM = 128
DEFAULT_QWEN3_8B_CONTEXT = 8192
DEFAULT_KV_CACHE_BITS = 16
DEFAULT_KV_CACHE_QUANT_BITS = 4
TOKEN_BACKEND_CHOICES = ("ollama-chat", "sglang-r2r", "llama-cpp-turboquant")

NEXT_TOKEN_FACTORS: tuple[tuple[str, str, str], ...] = (
    (
        "prompt_context",
        "current",
        "Prompting and task hints shift the conditional distribution, but do not change the base model.",
    ),
    (
        "decoding_parameters",
        "current",
        "Temperature, top_p, top_k, min_p, and token budget affect sampling from the distribution.",
    ),
    (
        "qwen3_thinking_prefix",
        "current_opt_in",
        "Thinking mode spends extra hidden tokens before the answer; local eval keeps it opt-in because it can overrun simple tasks.",
    ),
    (
        "token_level_logits",
        "backend_required",
        "True next-token diagnostics need logits or top-k probabilities, which Ollama chat does not expose.",
    ),
    (
        "token_replacement_routing",
        "backend_required",
        "R2R-style next-token routing needs accept/replace control and KV prefill update.",
    ),
    (
        "adapter_training",
        "offline_weight_change",
        "LoRA or other adapters can change the learned next-token distribution after held-out proof.",
    ),
)

R2R_REQUIREMENTS: tuple[tuple[str, str], ...] = (
    ("token_level_logits", "Expose SLM next-token logits or top-k probabilities."),
    ("hidden_states_or_router_features", "Expose router features such as hidden states, logits, and token ids."),
    ("single_token_routing", "Accept or replace one generated token before continuing."),
    ("large_model_prefill", "Update the large model KV cache from the mixed prefix."),
    ("co_resident_models", "Keep small model, large model, and router resident enough to avoid load thrash."),
    ("trained_router", "Provide a router checkpoint for the exact small/large model pair."),
)


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _prefixed_errors(prefix: str, errors: list[str]) -> list[str]:
    return [f"{prefix}: {error}" for error in errors]


def r2r_backend_requirement_status(backend: str) -> dict[str, str]:
    if backend == "sglang-r2r":
        return {name: "supported_by_backend" for name, _ in R2R_REQUIREMENTS}
    if backend == "ollama-chat":
        return {
            "token_level_logits": "not_exposed",
            "hidden_states_or_router_features": "not_exposed",
            "single_token_routing": "not_exposed",
            "large_model_prefill": "not_exposed",
            "co_resident_models": "partial_keep_alive_only",
            "trained_router": "external",
        }
    if backend == "llama-cpp-turboquant":
        return {
            "token_level_logits": "reference_implementation_only",
            "hidden_states_or_router_features": "reference_logits_only",
            "single_token_routing": "reference_decode_loop_only",
            "large_model_prefill": "reference_kv_api_only",
            "co_resident_models": "not_measured_locally",
            "trained_router": "external",
        }
    raise SetupError(f"Unknown R2R backend: {backend}")


def r2r_estimate_data(
    small_params_b: float,
    large_params_b: float,
    router_params_b: float,
    large_token_rate: float,
    output_tokens: int,
    backend: str,
) -> dict[str, Any]:
    if small_params_b <= 0 or large_params_b <= 0 or router_params_b < 0:
        raise SetupError("R2R parameter sizes must be positive, with router params zero or greater.")
    if not 0 <= large_token_rate <= 1:
        raise SetupError("--large-token-rate must be between 0 and 1.")
    if output_tokens < 1:
        raise SetupError("--output-tokens must be at least 1.")

    average_params_b = small_params_b + router_params_b + large_token_rate * large_params_b
    large_only_cost = large_params_b * output_tokens
    routed_cost = average_params_b * output_tokens
    statuses = r2r_backend_requirement_status(backend)
    ready = all(status == "supported_by_backend" for status in statuses.values())
    return {
        "backend": backend,
        "backend_ready_for_true_token_routing": ready,
        "small_params_b": small_params_b,
        "large_params_b": large_params_b,
        "router_params_b": router_params_b,
        "large_token_rate": large_token_rate,
        "output_tokens": output_tokens,
        "average_activated_params_b": round(average_params_b, 3),
        "parameter_ratio_vs_large": round(_safe_ratio(average_params_b, large_params_b), 3),
        "estimated_routed_cost_btok": round(routed_cost, 3),
        "large_only_cost_btok": round(large_only_cost, 3),
        "estimated_cost_ratio_vs_large": round(_safe_ratio(routed_cost, large_only_cost), 3),
        "estimated_cost_reduction_vs_large": round(1 - _safe_ratio(routed_cost, large_only_cost), 3),
        "requirements": [
            {
                "name": name,
                "status": statuses[name],
                "detail": detail,
            }
            for name, detail in R2R_REQUIREMENTS
        ],
    }


def render_r2r_estimate(data: dict[str, Any]) -> str:
    lines = [
        f"R2R estimate backend: {data['backend']}",
        f"Backend ready for true token routing: {data['backend_ready_for_true_token_routing']}",
        f"Small model params: {data['small_params_b']:.3g}B",
        f"Large model params: {data['large_params_b']:.3g}B",
        f"Router params: {data['router_params_b']:.3g}B",
        f"Large-token route rate: {data['large_token_rate']:.3f}",
        f"Average activated params/token: {data['average_activated_params_b']:.3f}B",
        f"Parameter ratio vs large-only: {data['parameter_ratio_vs_large']:.3f}",
        f"Estimated cost reduction vs large-only: {data['estimated_cost_reduction_vs_large']:.3f}",
        "",
        "Backend requirements:",
    ]
    lines.extend(
        f"- {item['name']}: {item['status']} ({item['detail']})"
        for item in data["requirements"]
    )
    return "\n".join(lines)


def kv_cache_estimate_data(
    layers: int,
    kv_heads: int,
    head_dim: int,
    context_tokens: int,
    batch_size: int,
    kv_bits: int,
    quantized_kv_bits: int | None,
) -> dict[str, Any]:
    if layers < 1 or kv_heads < 1 or head_dim < 1 or context_tokens < 1 or batch_size < 1:
        raise SetupError("KV cache dimensions and batch size must be positive.")
    if kv_bits < 1:
        raise SetupError("--kv-bits must be positive.")
    if quantized_kv_bits is not None and quantized_kv_bits < 1:
        raise SetupError("--quantized-kv-bits must be positive when provided.")

    values_per_token = layers * 2 * kv_heads * head_dim * batch_size
    bytes_per_token = values_per_token * kv_bits / 8
    total_bytes = bytes_per_token * context_tokens
    quantized_bytes_per_token = None
    quantized_total_bytes = None
    quantized_total_mib = None
    estimated_savings_ratio = None
    if quantized_kv_bits is not None:
        quantized_bytes_per_token = values_per_token * quantized_kv_bits / 8
        quantized_total_bytes = quantized_bytes_per_token * context_tokens
        estimated_savings_ratio = 1 - _safe_ratio(quantized_total_bytes, total_bytes)
        quantized_total_mib = quantized_total_bytes / (1024 * 1024)

    return {
        "model_assumption": "Qwen3-8B default architecture unless overridden",
        "layers": layers,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "context_tokens": context_tokens,
        "batch_size": batch_size,
        "kv_bits": kv_bits,
        "bytes_per_token": int(bytes_per_token),
        "total_mib": round(total_bytes / (1024 * 1024), 3),
        "quantized_kv_bits": quantized_kv_bits,
        "quantized_bytes_per_token": None if quantized_bytes_per_token is None else int(quantized_bytes_per_token),
        "quantized_total_mib": None if quantized_total_mib is None else round(quantized_total_mib, 3),
        "estimated_savings_ratio": None if estimated_savings_ratio is None else round(estimated_savings_ratio, 3),
        "ollama_chat_exposes_kv_quantization": False,
        "useful_if": [
            "long context approaches the local VRAM limit",
            "the backend can reuse shared prefixes or quantize KV cache",
            "quality holds on held-out math, code, and instruction-following checks",
        ],
    }


def render_kv_cache_estimate(data: dict[str, Any]) -> str:
    lines = [
        "KV cache estimate",
        f"Assumption: {data['model_assumption']}",
        (
            "Shape: "
            f"layers={data['layers']}, kv_heads={data['kv_heads']}, "
            f"head_dim={data['head_dim']}, context={data['context_tokens']}, "
            f"batch={data['batch_size']}"
        ),
        f"Base KV precision: {data['kv_bits']} bits",
        f"Base bytes/token: {data['bytes_per_token']}",
        f"Base total: {data['total_mib']:.3f} MiB",
    ]
    if data["quantized_kv_bits"] is not None:
        lines.extend(
            [
                f"Quantized KV precision: {data['quantized_kv_bits']} bits",
                f"Quantized bytes/token: {data['quantized_bytes_per_token']}",
                f"Quantized total: {data['quantized_total_mib']:.3f} MiB",
                f"Estimated KV memory reduction: {data['estimated_savings_ratio']:.3f}",
            ]
        )
    lines.append(f"Ollama chat exposes KV quantization controls: {data['ollama_chat_exposes_kv_quantization']}")
    lines.append("Useful if:")
    lines.extend(f"- {item}" for item in data["useful_if"])
    return "\n".join(lines)


def next_token_headroom_data(backend: str) -> dict[str, Any]:
    r2r_statuses = r2r_backend_requirement_status(backend)
    token_level_backend_ready = all(
        r2r_statuses[name] == "supported_by_backend"
        for name in ("token_level_logits", "single_token_routing", "large_model_prefill")
    )
    current_backend = backend == "ollama-chat"
    return {
        "backend": backend,
        "fixed_qwen3_8b_weights_changeable_by_prompt": False,
        "current_ollama_chat_can_expose_true_next_token_logits": (
            current_backend and r2r_statuses["token_level_logits"] == "supported_by_backend"
        ),
        "current_ollama_chat_can_replace_individual_tokens": (
            current_backend and r2r_statuses["single_token_routing"] == "supported_by_backend"
        ),
        "token_level_backend_ready": token_level_backend_ready,
        "continue_recommended": True,
        "why_continue": [
            "prompt and decoding controls can still shift outputs without changing weights",
            "adapter training can change the next-token distribution if held-out gates prove the need",
            "a logits/token-routing backend would open true next-token selection experiments",
        ],
        "factors": [
            {
                "name": name,
                "status": status,
                "detail": detail,
            }
            for name, status, detail in NEXT_TOKEN_FACTORS
        ],
        "backend_requirements": [
            {
                "name": name,
                "status": r2r_statuses[name],
                "detail": detail,
            }
            for name, detail in R2R_REQUIREMENTS
        ],
    }


def render_next_token_headroom(data: dict[str, Any]) -> str:
    lines = [
        f"Next-token headroom backend: {data['backend']}",
        f"Prompt can change fixed Qwen3-8B weights: {data['fixed_qwen3_8b_weights_changeable_by_prompt']}",
        (
            "Current Ollama chat exposes true next-token logits: "
            f"{data['current_ollama_chat_can_expose_true_next_token_logits']}"
        ),
        (
            "Current Ollama chat can replace individual tokens: "
            f"{data['current_ollama_chat_can_replace_individual_tokens']}"
        ),
        f"Selected backend ready for token-level experiments: {data['token_level_backend_ready']}",
        f"Continue recommended: {data['continue_recommended']}",
        "",
        "Factors:",
    ]
    lines.extend(f"- {item['name']}: {item['status']} ({item['detail']})" for item in data["factors"])
    lines.append("")
    lines.append("Backend requirements:")
    lines.extend(
        f"- {item['name']}: {item['status']} ({item['detail']})"
        for item in data["backend_requirements"]
    )
    return "\n".join(lines)


def inference_compute_gate_data(
    distill_path: Path,
    data_quality_paths: list[Path],
    data_quality_check: Callable[[list[Path]], dict[str, Any]],
    verifier_tool_gate: Callable[[Path], dict[str, Any]],
    adaptive_plan: Callable[..., Any],
) -> dict[str, Any]:
    data_quality = data_quality_check(data_quality_paths)
    verifier_tool = verifier_tool_gate(distill_path)
    plan_prompts = {
        "strict_output_shape": "Return exactly three bullet lines about local inference.",
        "parallel_explore": "Compare two architecture options for a local inference pipeline.",
        "sequential_refine": "If 25 ms is saved on each of 8 cases, how much is saved in total?",
    }
    plans = {
        name: adaptive_plan(prompt, quality_refine_passes=1, search_candidates=1)
        for name, prompt in plan_prompts.items()
    }
    plan_data = {
        name: {
            "quality_refine_passes": plan.quality_refine_passes,
            "search_candidates": plan.search_candidates,
            "strategy": plan.strategy,
        }
        for name, plan in plans.items()
    }
    ollama_headroom = next_token_headroom_data("ollama-chat")

    errors = _prefixed_errors("data_quality", data_quality["errors"])
    errors.extend(_prefixed_errors("verifier_tool", verifier_tool["errors"]))
    if plans["strict_output_shape"].quality_refine_passes != 0 or plans["strict_output_shape"].search_candidates != 1:
        errors.append("strict output-shape prompts should not spend extra compute")
    if plans["parallel_explore"].search_candidates < 2:
        errors.append("exploration prompts should use at least two candidates")
    if plans["sequential_refine"].quality_refine_passes < 1:
        errors.append("reasoning prompts should get at least one refinement pass")
    if ollama_headroom["token_level_backend_ready"]:
        errors.append("ollama-chat must not be reported as token-level backend-ready")
    if ollama_headroom["current_ollama_chat_can_expose_true_next_token_logits"]:
        errors.append("ollama-chat must not be reported as exposing true next-token logits")

    return {
        "data_quality_errors": data_quality["errors"],
        "verifier_tool_errors": verifier_tool["errors"],
        "data_quality": {
            "total_records": data_quality["total_records"],
            "total_verifier_records": data_quality["total_verifier_records"],
            "overall_verifier_rate": data_quality["overall_verifier_rate"],
            "duplicate_ids": data_quality["duplicate_ids"],
            "duplicate_prompt_hashes": data_quality["duplicate_prompt_hashes"],
        },
        "verifier_tool": {
            "distill_total": verifier_tool["distill"]["total"],
            "distill_pass_count": verifier_tool["distill"]["pass_count"],
            "distill_fail_count": verifier_tool["distill"]["fail_count"],
            "required_architecture_checks": verifier_tool["required_architecture_checks"],
            "action_expectations": verifier_tool["action_expectations"],
        },
        "adaptive_compute_plans": plan_data,
        "ollama_next_token": {
            "token_level_backend_ready": ollama_headroom["token_level_backend_ready"],
            "current_ollama_chat_can_expose_true_next_token_logits": (
                ollama_headroom["current_ollama_chat_can_expose_true_next_token_logits"]
            ),
            "current_ollama_chat_can_replace_individual_tokens": (
                ollama_headroom["current_ollama_chat_can_replace_individual_tokens"]
            ),
            "continue_recommended": ollama_headroom["continue_recommended"],
        },
        "errors": errors,
    }


def render_inference_compute_gate(data: dict[str, Any]) -> str:
    status = "ok" if not data["errors"] else "error"
    lines = [
        f"Inference compute gate: {status}",
        (
            "Data quality: records={total_records}, verifier={total_verifier_records} "
            "({overall_verifier_rate:.3f})"
        ).format(**data["data_quality"]),
        (
            "Verifier/tool: distill={distill_total}, pass={distill_pass_count}, "
            "fail={distill_fail_count}"
        ).format(**data["verifier_tool"]),
        "Adaptive compute plans:",
    ]
    for name, plan in data["adaptive_compute_plans"].items():
        lines.append(
            "- {name}: strategy={strategy}, refine={quality_refine_passes}, candidates={search_candidates}".format(
                name=name,
                **plan,
            )
        )
    lines.append(
        "Ollama token-level backend-ready: "
        f"{data['ollama_next_token']['token_level_backend_ready']}"
    )
    if data["errors"]:
        lines.extend(["", "Errors:"])
        lines.extend(f"- {error}" for error in data["errors"])
    return "\n".join(lines)
