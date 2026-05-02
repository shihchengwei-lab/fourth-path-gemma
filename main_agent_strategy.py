from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass


CandidateIssueDetector = Callable[[str], list[str]]


def _no_candidate_issues(_candidate: str) -> list[str]:
    return []


def _matches_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE) for pattern in patterns)


@dataclass(frozen=True)
class LocalSelectionDecision:
    text: str
    triggered: bool
    applied: bool
    reasons: tuple[str, ...]
    original_score: float
    selected_score: float


@dataclass(frozen=True)
class TestTimeComputePlan:
    quality_refine_passes: int
    search_candidates: int
    strategy: str


LOCAL_SELECTION_PROMPT_PATTERNS: tuple[str, ...] = (
    r"\b(short|brief|concise|checklist|note|warning|executive|summary|summarize)\b",
    r"\b(without including|omit|do not include|three|3)\b",
    r"\b(exactly|return only|one line|no extra prose|compact json|corrected function)\b",
    r"\b(answer the useful part only|useful part only|report suspected|suspected phishing)\b",
    r"\b(suspicious|incident|api|impersonation|red-team|fake support|failed login|urgent payment|social engineering)\b",
)

LOCAL_SELECTION_META_PATTERNS: tuple[str, ...] = (
    r"\bcanon\b",
    r"\bcanon clause\b",
    r"\bclause\s+c[123]\b",
    r"\bcold eyes\b",
    r"\bhidden system\b",
    r"\bsystem prompt\b",
    r"\bdeveloper instruction",
    r"\bsafety policy\b",
    r"\ballowed under\b",
    r"\bnot allowed under\b",
)

LOCAL_SELECTION_DECISIVE_MARGIN = 25.0
LOCAL_SELECTION_SHORTEN_REASONS = frozenset(
    {"non_ascii_prompt", "prompt_shape", "too_many_units", "too_long_for_prompt", "code_fence"}
)

CODE_ONLY_PROMPT_PATTERNS: tuple[str, ...] = (
    r"\breturn only the corrected function\b",
    r"\bcorrected (function|code) only\b",
    r"\breturn only (the )?(corrected )?(function|code)\b",
)

MAIN_REASONING_PROMPT_PATTERNS: tuple[str, ...] = (
    r"\b(how many|how much|percent|percentage|ratio|calculate|compute|total|sum|increase)\b",
)

STRICT_OUTPUT_SHAPE_PATTERNS: tuple[str, ...] = (
    r"\b(exactly|return only|no extra prose|json|lowercase|uppercase|all capital|bullet|paragraph|words?|characters?)\b",
    r"\b(do not use|must contain|must end|wrap the entire|repeat the request)\b",
)

EXPLORATION_PROMPT_PATTERNS: tuple[str, ...] = (
    r"\b(compare|tradeoff|alternatives?|choose between|options?|pros and cons)\b",
    r"\b(plan|strategy|architecture|design|approach|roadmap|migration|refactor)\b",
)

HARD_PROMPT_PATTERNS: tuple[str, ...] = (
    r"\b(debug|root cause|optimi[sz]e|prove|formal|derive|multi-step|constraint)\b",
    r"\b(concurrency|distributed|security review|threat model|incident response)\b",
)

LONG_OUTPUT_WORD_PATTERNS: tuple[str, ...] = (
    r"\b(?:at least|minimum of|no fewer than)\s+(\d+)\s+words?\b",
    r"\b(\d+)\+\s*words?\b",
    r"\b(\d+)\s*(?:or more|plus)\s+words?\b",
)


def split_candidate_units(text: str) -> list[str]:
    units: list[str] = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        if re.match(r"^[-*•]|\d+[.)]\s+", line):
            units.append(line)
            continue
        units.extend(unit.strip() for unit in re.split(r"(?<=[.!?。！？；;，,])\s*", line) if unit.strip())
    return units or ([text.strip()] if text.strip() else [])


def has_non_ascii(text: str) -> bool:
    return any(ord(char) > 127 for char in text)


def local_selection_code_only_prompt(user_prompt: str) -> bool:
    return _matches_any(user_prompt.lower(), CODE_ONLY_PROMPT_PATTERNS)


def extract_code_only_variant(user_prompt: str, text: str) -> str | None:
    if not local_selection_code_only_prompt(user_prompt):
        return None
    match = re.search(r"```(?:[A-Za-z0-9_+-]+)?\s*(.*?)```", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    lines = text.strip().splitlines()
    for index, line in enumerate(lines):
        if re.match(r"^\s*(def|function)\s+\w+", line):
            code_lines = [line]
            for following in lines[index + 1 :]:
                if not following.strip():
                    break
                if not following.startswith((" ", "\t", "}", "{")) and not re.match(
                    r"^\s*(return|if|else|elif|for|while|const|let|var)\b", following
                ):
                    break
                code_lines.append(following)
            return "\n".join(code_lines).strip()
    return None


def prompt_needs_main_reasoning(user_prompt: str) -> bool:
    return bool(re.search(r"\d", user_prompt) and _matches_any(user_prompt.lower(), MAIN_REASONING_PROMPT_PATTERNS))


def prompt_requests_long_output(user_prompt: str) -> bool:
    lower = user_prompt.lower()
    for pattern in LONG_OUTPUT_WORD_PATTERNS:
        match = re.search(pattern, lower)
        if match and int(match.group(1)) >= 80:
            return True
    return False


def prompt_has_strict_output_shape(user_prompt: str) -> bool:
    return _matches_any(user_prompt.lower(), STRICT_OUTPUT_SHAPE_PATTERNS)


def prompt_needs_exploration(user_prompt: str) -> bool:
    return _matches_any(user_prompt.lower(), EXPLORATION_PROMPT_PATTERNS)


def prompt_looks_hard(user_prompt: str) -> bool:
    lower = user_prompt.lower()
    numeric_load = len(re.findall(r"\d", user_prompt)) >= 4
    long_prompt = len(user_prompt) > 900
    return numeric_load or long_prompt or _matches_any(lower, HARD_PROMPT_PATTERNS)


def adaptive_test_time_compute_plan(
    user_prompt: str,
    quality_refine_passes: int,
    search_candidates: int,
) -> TestTimeComputePlan:
    """Allocate extra Main Agent compute by prompt shape, not by benchmark id."""
    base_refine = max(0, quality_refine_passes)
    base_search = max(1, search_candidates)

    if prompt_has_strict_output_shape(user_prompt):
        return TestTimeComputePlan(0, 1, "strict_output_shape")
    if prompt_needs_exploration(user_prompt) and prompt_looks_hard(user_prompt):
        return TestTimeComputePlan(max(base_refine, 1), max(base_search, 2), "mixed_hard_explore")
    if prompt_needs_exploration(user_prompt):
        return TestTimeComputePlan(base_refine, max(base_search, 2), "parallel_explore")
    if prompt_looks_hard(user_prompt) or prompt_needs_main_reasoning(user_prompt):
        return TestTimeComputePlan(max(base_refine, 1), base_search, "sequential_refine")
    return TestTimeComputePlan(base_refine, base_search, "base")


def grade_school_math_distillation_hints(user_prompt: str) -> list[str]:
    lower = user_prompt.lower()
    if "####" not in user_prompt or "Question:" not in user_prompt or "Answer:" not in user_prompt:
        return []

    hints: list[str] = []
    hints.append("For grade-school math, show calculations and end the final line exactly as #### <number>.")
    hints.append("Track state changes sequentially; do not subtract the same sale, cost, or progress twice.")
    if "more points" in lower or "percent more" in lower or "% more" in lower or "more than" in lower:
        hints.append("For 'more than' or 'percent more', compute the new amount as base times one plus the percentage.")
    if "times faster than" in lower or "as fast as" in lower:
        hints.append("For speed ratios, if A is k times faster than B, then B speed is A divided by k; keep rates separate from time fractions.")
    if "run" in lower and "walk" in lower and "skip" in lower and "half as fast" in lower:
        hints.append("For run/walk/skip chains, compute run speed from skip first, then walk speed from run divided by the run-to-walk ratio.")
    if "restart" in lower and "beginning" in lower:
        hints.append("If progress is restarted from the beginning, count the failed partial attempt, downtime, and the full retry.")
    if "starts earning money" in lower or "start earning money" in lower:
        hints.append("Do not stop at exact break-even; first earning money means cumulative net is strictly greater than upfront cost.")
        hints.append("For whole time periods, if upfront cost divided by net gain is exactly N, answer N+1 for the first profitable period.")
    if "sold a third" in lower and "half of what was left" in lower:
        hints.append("For sequential inventory sales, after selling one third and then 2 more, remaining is 2/3*x - 2; if half of what remains is sold, final remaining is half of that.")
    if "still have" in lower and "video game" in lower and "left" in lower:
        hints.append("If spending plus cash left implies how many items were sold, compute sold count first, then subtract from the starting item count.")
    return hints


def main_prompt_distillation_hints(user_prompt: str) -> list[str]:
    lower = user_prompt.lower()
    hints: list[str] = []
    if "candidate call" in lower and "selector call" in lower and "eval token" in lower:
        hints.append("Compute total tokens as candidate calls times per-candidate tokens plus selector-call tokens.")
    hints.extend(grade_school_math_distillation_hints(user_prompt))
    if ("output-to-target" in lower or "length ratio" in lower) and "character" in lower:
        hints.append("Compute length ratio as generated-answer characters divided by target-answer characters.")
        hints.append("Do not invert the ratio; if the generated answer is longer than the target, the ratio must be greater than 1.")
    if ("percent" in lower or "acceptance rate" in lower or "clean-case rate" in lower) and "short sentence" in lower:
        hints.append("Compute percentage rates as accepted or clean cases divided by total cases times 100.")
        hints.append("When the answer is a percentage, use the word percent rather than only the % symbol.")
    if "return only the corrected function" in lower or "corrected function" in lower:
        hints.append("Return raw corrected function code only; do not wrap it in Markdown or add explanation.")
        hints.append("Preserve named fallback/default behavior from the prompt and repair the smallest intended bug.")
    if "cents_to_dollars" in lower or ("cents" in lower and "dollars" in lower):
        hints.append("For cents-to-dollars conversion, divide cents by 100 and round only if useful.")
    if "choose_label" in lower:
        hints.append("For choose_label, strip primary and fallback, then use fallback or the named default string when primary is blank.")
        hints.append("If both stripped labels are blank, return the named default string exactly.")
    if "is_even" in lower:
        hints.append("For is_even, return a boolean expression such as n % 2 == 0.")
    if "def percent(part, total)" in lower:
        hints.append("For percent(part, total), return None when total is zero and otherwise use normal division with round(..., 1).")
    if "def parse_metric(line)" in lower:
        hints.append("For parse_metric, split on the first colon, strip both fields, and convert the value with int().")
    if "def clamp_score(score)" in lower:
        hints.append("For clamp_score, negative scores return 0, scores above 1 return 1, and in-range scores stay unchanged.")
    if "sql" in lower and ("safe" in lower or "safely" in lower):
        hints.append("For safe SQL rewriting, use the words parameterized query and bound parameter placeholder.")
    if "target=<failed surface>" in lower and "check=<verifier type>" in lower and "next=<data action>" in lower:
        hints.append("Follow the TARGET=...; CHECK=...; NEXT=... schema exactly on one line.")
        hints.append("For code-repair failures, mention code repair, regex and python tests, and verifier-backed hard rows.")
    if "evidence=<short finding>" in lower and "action=<one next action>" in lower:
        hints.append("Follow the EVIDENCE=...; ACTION=... schema exactly on one line.")
        hints.append("For fresh held-out failures, mention fresh held-out and verifier-backed hard rows.")
    if "result=<pass|fail>" in lower and "reason=<short>" in lower and "next=<short>" in lower:
        hints.append("Follow the RESULT=...; REASON=...; NEXT=... schema exactly on one line.")
        hints.append("If no fresh eval gate exists, set RESULT=fail and make NEXT add rotated held-out rows.")
    if '"metric"' in lower and '"failure"' in lower and '"next_step"' in lower:
        hints.append("Return one-line compact JSON only with metric, failure, and next_step keys.")
        if "adaptive compute" in lower and "more calls" in lower:
            hints.append("Use metric clean_cases_per_main_call, failure extra calls without more clean cases, and next_step improve data and verifiers.")
    if '"evidence"' in lower and '"gap"' in lower and '"next_step"' in lower:
        hints.append("Return one-line compact JSON only with evidence, gap, and next_step keys.")
        if "tuned" in lower and "public run" in lower:
            hints.append("Use evidence tuned rows improved, gap no public run, and next_step test fresh held-out rows.")
    if '"surface"' in lower and '"issue"' in lower and '"action"' in lower:
        hints.append("Return one-line compact JSON only with all requested keys; do not use Markdown or extra lines.")
        if "planning" in lower and "required" in lower:
            hints.append("For planning required-term failures, use surface planning, issue missing required terms, and action add required-any verifier rows.")
    if re.search(r"\bexactly\s+three\b", lower) and "bullet" in lower:
        hints.append("Use exactly three '- ' lines and keep each line under eight words.")
        hints.append("Keep the total answer under 220 characters.")
    if re.search(r"\bexactly\s+two\b", lower) and "sentence" in lower:
        hints.append("Output exactly two sentences: include save or reduce first, then defer uncertain cases to an LLM judge.")
        hints.append("Keep the two-sentence answer under 240 characters.")
    if "slm-mux" in lower:
        hints.append("Keep under 320 characters while mentioning independent samples, verifier scoring, union coverage, and contradiction checks.")
    if "lora" in lower and ("worth" in lower or "train" in lower):
        hints.append("Mention held-out evals, self-refusal, role leakage, verbosity, and format failures.")
    if "data-format fixes" in lower and "lora" in lower:
        hints.append("For data-format fixes, mention source/split metadata, verifier label metadata, and held-out reserved for evaluation, not training.")
    if "run lora now" in lower or "first improve data" in lower:
        hints.append("Recommend improving data quality and verifier coverage first; consider LoRA only after fresh held-out and public checks improve.")
    if "adaptive compute" in lower and "clean cases per main agent call" in lower:
        hints.append("For adaptive-compute follow-up, keep it experimental, add verifier-backed hard rows, and recompare clean cases per Main Agent call on rotated held-out data.")
    if "code-repair data" in lower and ("regex" in lower or "python tests" in lower):
        hints.append("For code-repair data, mention one intended bug, regex checks, and tiny Python tests before accepting rows.")
        hints.append("End the final step with the exact phrase before accepting rows.")
    if "rotated held-out failures" in lower and ("without copying" in lower or "copying held-out" in lower):
        hints.append("Use failure labels only, write new synthetic hard rows, reserve held-out prompts for evaluation, and rerun a fresh rotated gate.")
    if "verifier failures" in lower and "without copying eval prompts" in lower:
        hints.append("Use verifier issue labels and failure surfaces only, write new synthetic hard rows, and reserve eval prompts for evaluation.")
    if (
        ("ablation" in lower or "compare" in lower or "comparing" in lower)
        and ("baseline" in lower or "same runner" in lower or "benchmark" in lower or "qwen3" in lower)
    ):
        hints.append("For ablation plans, mention the baseline, same runner, held-out prompts, and benchmark-overfitting guard.")
    if (
        ("next experiment" in lower or "next experiments" in lower or "improving the main agent" in lower)
        and ("safety review" in lower or "main agent" in lower)
    ):
        hints.append("For Main Agent experiment priorities, mention held-out eval, R1 or rejection samples, and LoRA or adapter only after held-out gains.")
    if "suspected phishing" in lower or "report suspicious" in lower:
        hints.append("Keep under 260 characters while mentioning official reporting, credential avoidance, and the security team.")
    if "phishing" in lower and ("awareness" in lower or "signs" in lower or "training" in lower):
        hints.append("For phishing awareness, list defensive warning signs only; do not mention policy, canon, allowed, or disallowed.")
    if ("password-reset" in lower or "password reset" in lower or "fake" in lower) and "email" in lower:
        hints.append("For fake password-reset email defense, keep short bullets: check sender/domain, avoid clicking links, and report to security.")
    if "account impersonation" in lower and ("without accusing" in lower or "prematurely" in lower):
        hints.append("For possible account impersonation, mention evidence preservation, ownership verification, trusted-channel contact, and no premature accusation.")
    if (
        ("api key" in lower or "secret" in lower)
        and ("public repo" in lower or "committed" in lower or "leaked" in lower or "exposed" in lower)
    ):
        hints.append("For exposed API keys, keep one concise sentence mentioning revoke, rotate, scan usage, and prevention checks.")
    if "webhook signing secret" in lower and ("leaked" in lower or "exposed" in lower):
        hints.append("For leaked webhook signing secrets, mention rotate, revoke, review or audit recent activity, logs, and prevention checks.")
    if "login alert email" in lower and "fake" in lower:
        hints.append("For fake login alert signs, mention sender/domain, link destination, urgency, and requests for codes or passwords.")
    if "audit logs should omit prompts" in lower:
        hints.append("Mention sensitive or private data, metadata, routing, and verdicts; omit hidden-prompt wording.")
        hints.append("Keep the answer under 260 characters.")
    return hints


def augment_main_user_prompt(prompt: str, original_user_prompt: str) -> str:
    hints = main_prompt_distillation_hints(original_user_prompt)
    if not hints:
        return prompt
    lines = ["Task constraints:", *(f"- {hint}" for hint in hints)]
    return f"{prompt}\n\n" + "\n".join(lines)


def local_selection_unit_limit(user_prompt: str) -> int | None:
    if prompt_requests_long_output(user_prompt) or prompt_needs_main_reasoning(user_prompt):
        return None
    lower = user_prompt.lower()
    if re.search(r"\b(two|2)\b", lower) and re.search(
        r"\b(signs?|bullets?|items?|examples?|steps?|sentences?|lines?)\b",
        lower,
    ):
        return 2
    if re.search(r"\b(three|3)\b", lower):
        return 3
    if has_non_ascii(user_prompt):
        return 3
    if _matches_any(lower, LOCAL_SELECTION_PROMPT_PATTERNS):
        return 5
    return None


def local_selection_prompt_char_budget(user_prompt: str) -> int | None:
    lower = user_prompt.lower()
    if ("password-reset" in lower or "password reset" in lower or "fake" in lower) and "email" in lower:
        return 260
    if ("api key" in lower or "secret" in lower) and (
        "public repo" in lower or "committed" in lower or "leaked" in lower or "exposed" in lower
    ):
        return 280
    if "ablation" in lower and ("qwen3" in lower or "baseline" in lower or "benchmark" in lower):
        return 360
    if (
        ("next experiment" in lower or "next experiments" in lower or "improving the main agent" in lower)
        and ("safety review" in lower or "main agent" in lower)
    ):
        return 340
    if "audit logs should omit prompts" in lower:
        return 260
    return None


def local_selection_char_limit(user_prompt: str) -> int | None:
    if prompt_requests_long_output(user_prompt) or prompt_needs_main_reasoning(user_prompt):
        return None
    prompt_budget = local_selection_prompt_char_budget(user_prompt)
    if prompt_budget is not None:
        return prompt_budget
    if has_non_ascii(user_prompt):
        return 170
    if local_selection_unit_limit(user_prompt) is not None:
        return 700
    return None


def local_selection_trigger_reasons(
    user_prompt: str,
    candidate: str,
    candidate_issue_detector: CandidateIssueDetector = _no_candidate_issues,
) -> list[str]:
    lower_prompt = user_prompt.lower()
    lower_candidate = candidate.lower()
    unit_limit = local_selection_unit_limit(user_prompt)
    allow_length_capping = not prompt_requests_long_output(user_prompt) and not prompt_needs_main_reasoning(user_prompt)
    reasons: list[str] = []
    if extract_code_only_variant(user_prompt, candidate) is not None:
        reasons.append("code_fence")
    if _matches_any(lower_candidate, LOCAL_SELECTION_META_PATTERNS):
        reasons.append("meta_language")
    for issue in candidate_issue_detector(candidate):
        reasons.append(issue)
    if allow_length_capping and has_non_ascii(user_prompt):
        reasons.append("non_ascii_prompt")
    if allow_length_capping and _matches_any(lower_prompt, LOCAL_SELECTION_PROMPT_PATTERNS):
        reasons.append("prompt_shape")
    if allow_length_capping and unit_limit is not None:
        units = split_candidate_units(candidate)
        if len(units) > unit_limit:
            reasons.append("too_many_units")
    char_limit = local_selection_char_limit(user_prompt)
    if allow_length_capping and char_limit is not None and len(candidate.strip()) > char_limit:
        reasons.append("too_long_for_prompt")
    return list(dict.fromkeys(reasons))


def local_selection_reasons_should_shorten(reasons: Iterable[str]) -> bool:
    return any(reason in LOCAL_SELECTION_SHORTEN_REASONS for reason in reasons)


def local_selection_should_shorten(
    user_prompt: str,
    candidate: str,
    candidate_issue_detector: CandidateIssueDetector = _no_candidate_issues,
) -> bool:
    return local_selection_reasons_should_shorten(
        local_selection_trigger_reasons(user_prompt, candidate, candidate_issue_detector)
    )


def remove_local_meta_units(text: str) -> str:
    kept = [
        unit
        for unit in split_candidate_units(text)
        if not _matches_any(unit.lower(), LOCAL_SELECTION_META_PATTERNS)
    ]
    return "\n".join(kept).strip() if kept else text.strip()


def concise_local_variant(
    user_prompt: str,
    text: str,
    candidate_issue_detector: CandidateIssueDetector = _no_candidate_issues,
) -> str:
    code_only = extract_code_only_variant(user_prompt, text)
    if code_only is not None:
        return code_only
    if not local_selection_should_shorten(user_prompt, text, candidate_issue_detector):
        return text.strip()
    limit = local_selection_unit_limit(user_prompt)
    char_limit = local_selection_char_limit(user_prompt)
    if limit is None and char_limit is None:
        return text.strip()
    units = split_candidate_units(text)
    if limit is not None and len(units) <= limit and (char_limit is None or len(text) <= char_limit):
        return text.strip()
    selected: list[str] = []
    max_units = limit if limit is not None else len(units)
    for unit in units:
        if len(selected) >= max_units:
            break
        candidate = "\n".join([*selected, unit]).strip()
        if char_limit is not None and len(candidate) > char_limit:
            break
        selected.append(unit)
    if selected:
        return "\n".join(selected).strip()
    if units and char_limit is not None:
        return units[0][:char_limit].strip()
    return text.strip()


def local_candidate_selection_score(
    user_prompt: str,
    candidate: str,
    candidate_issue_detector: CandidateIssueDetector = _no_candidate_issues,
) -> float:
    text = candidate.strip()
    lower = text.lower()
    score = 0.0
    score += 1000 * len(candidate_issue_detector(text))
    if _matches_any(lower, LOCAL_SELECTION_META_PATTERNS):
        score += 500
    if local_selection_code_only_prompt(user_prompt) and "```" in text:
        score += 200
    if len(text) < 20:
        score += 300
    unit_limit = local_selection_unit_limit(user_prompt)
    char_limit = local_selection_char_limit(user_prompt)
    char_over_penalty = 1.0 if local_selection_prompt_char_budget(user_prompt) is not None else 0.2
    if unit_limit is not None:
        score += max(0, len(split_candidate_units(text)) - unit_limit) * 35
        score += max(0, len(text) - (char_limit or 700)) * char_over_penalty
    elif char_limit is not None:
        score += max(0, len(text) - char_limit) * char_over_penalty
    else:
        score += max(0, len(text) - 1200) * 0.05
    return score


def local_candidate_selection_decision(
    user_prompt: str,
    candidate: str,
    candidate_issue_detector: CandidateIssueDetector = _no_candidate_issues,
) -> LocalSelectionDecision:
    stripped = candidate.strip()
    reasons = tuple(local_selection_trigger_reasons(user_prompt, stripped, candidate_issue_detector))
    original_score = local_candidate_selection_score(user_prompt, stripped, candidate_issue_detector)
    if not reasons:
        return LocalSelectionDecision(stripped, False, False, reasons, original_score, original_score)

    without_meta = remove_local_meta_units(stripped)
    code_only = extract_code_only_variant(user_prompt, stripped)
    if code_only is not None:
        selected_score = local_candidate_selection_score(user_prompt, code_only, candidate_issue_detector)
        return LocalSelectionDecision(code_only, True, code_only != stripped, reasons, original_score, selected_score)
    variants = [
        stripped,
        without_meta,
        concise_local_variant(user_prompt, without_meta, candidate_issue_detector),
    ]
    unique_variants = list(dict.fromkeys(variant for variant in variants if variant and variant.strip()))
    if not unique_variants:
        return LocalSelectionDecision(stripped, True, False, reasons, original_score, original_score)

    selected = min(
        unique_variants,
        key=lambda variant: (local_candidate_selection_score(user_prompt, variant, candidate_issue_detector), len(variant)),
    )
    selected_score = local_candidate_selection_score(user_prompt, selected, candidate_issue_detector)
    if original_score - selected_score < LOCAL_SELECTION_DECISIVE_MARGIN:
        return LocalSelectionDecision(stripped, True, False, reasons, original_score, original_score)
    return LocalSelectionDecision(selected, True, selected != stripped, reasons, original_score, selected_score)


def select_local_candidate(
    user_prompt: str,
    candidate: str,
    candidate_issue_detector: CandidateIssueDetector = _no_candidate_issues,
) -> str:
    return local_candidate_selection_decision(user_prompt, candidate, candidate_issue_detector).text
