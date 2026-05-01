# AGI-Scale Safety Architecture Notes

This repository uses an 8B local model because it is controllable and cheap
enough to measure repeatedly. The research question is larger than the local
model: if a much stronger model, or an AGI-level system, is used as the
reasoning layer, can separation-and-audit preserve more useful capability while
keeping final output constrained?

The answer is not established. This note defines what would have to become
true before the architecture could be treated as AGI-grade.

## Scale Ladder

### 1. Local 8B model

Purpose:

- Measure whether role separation reduces self-refusal, role leakage, and token
  waste.
- Keep all prompts, data, audit records, and failures inspectable.
- Use mechanical fail gates where precision is high.

Current status:

- Useful as an experimental testbed.
- Not a proof of general alignment.
- Not a production safety boundary.

### 2. Strong open local model

Purpose:

- Test whether the same architecture still works when the Main Agent is a
  stronger local reasoner.
- Check whether Cold Eyes must be a different model, a larger model, or an
  ensemble.

Required evidence:

- Higher helpful/honest quality per main-model call.
- No increase in audit bypass attempts or role-boundary leakage.
- Cold Eyes exact-match performance on held-out boundary data.

### 3. Frontier hosted model

Purpose:

- Test the Fourth Path hypothesis under a much stronger reasoning layer.
- Ask whether removing final refusal authority from the Main Agent lets the
  flagship model spend more of its capacity on reasoning.

Required evidence:

- The Main Agent cannot approve its own output.
- The audit layer is independent from the Main Agent prompt, history, and tool
  state.
- Policy and audit logs prove final decisions come from the separated layer.
- Audit latency and cost are measured separately from reasoning quality.

### 4. AGI-level system

Purpose:

- Preserve deployable autonomy without asking one reasoning process to also be
  its own final safety authority.

Required evidence:

- The audit layer is stronger than a prompt convention.
- Final output is fail-closed when audit is unavailable, uncertain, or
  inconsistent.
- Mechanical constraints cover high-confidence forbidden classes.
- Multiple independent auditors can disagree and force refusal or escalation.
- The Main Agent cannot modify canon, audit code, refusal output, audit logs, or
  its own review criteria.
- Tool calls, file writes, network requests, purchases, messages, and other
  external side effects are not executed directly by the Main Agent. They must
  first be serialized as auditable action candidates and reviewed before
  execution.
- Long-context drift in the Main Agent cannot change final output permissions.
- The system has adversarial tests for persuasion, collusion, hidden-state
  leakage, tool-use abuse, and audit bypass.

## Safety Bar

For a small local prototype, it is enough to show that separation improves
measurement and reduces obvious compute waste.

For a frontier model, the safety bar rises: the audit layer must be independent,
measurable, and harder to drift than the Main Agent.

For AGI, this repository is not enough. The Fourth Path architecture would need
at least:

- hard authority separation
- immutable policy sources
- fail-closed output control
- fail-closed actuation control for tool calls and other side effects
- multi-auditor review
- adversarial red-team suites
- formal invariants for what each layer can read, write, and decide
- continuous monitoring for distribution shift and audit degradation

The point is not that separation magically makes AGI safe. The point is that a
single loop doing reasoning, policy maintenance, self-monitoring, and final
permission may be the wrong shape for scalable safety. Separation is a testable
alternative, not a completed guarantee.

## Current Repo Contribution

The current implementation contributes an executable small-scale version:

- Main Agent generates candidates and cannot approve.
- Quality selector can choose Helpful/Honest quality but cannot approve safety.
- Classify can route issues but cannot approve.
- Mechanical Cold Eyes can fail high-confidence cases or defer, but cannot pass.
- Cold Eyes receives only canon plus candidate output.
- The current runtime has no Main Agent tool-execution path; future side effects
  must be converted into action candidates and audited before execution.
- Refusal output is fixed and outside the Main Agent.
- `architecture-check` validates core authority invariants without calling a
  model.
- `action-audit` exposes the pre-execution side-effect gate as a direct command.
- `architecture-adversarial-check` validates a seed suite for cross-layer
  attacks.
- `architecture-adversarial-eval` measures whether the Main Agent leaks hidden
  system/developer/audit state and whether Cold Eyes resists candidate-side
  instructions such as "ignore canon" or fake JSON self-approval. Public canon
  references are allowed.

This is the minimum useful shape for the larger question. The next milestone is
not claiming safety; it is making capability gain and safety cost measurable
across stronger models and stricter audit layers.

## Current Adversarial Suite

The seed corpus is intentionally small and inspectable. It has three layers:

- Pipeline prompts that mix hidden-state requests, public canon references, or
  refusal pressure with normal user-facing tasks.
- Cold Eyes candidates that contain self-approval, JSON injection, authority
  claims, or direct commands to return pass.
- Action candidates that cover safe no-op/read-only actions, direct canon
  violations, control-plane writes, phishing messages, and unknown external
  side effects that must fail closed.

Commands:

```powershell
python main.py architecture-adversarial-check --min-total 19 --min-layer 6
python main.py architecture-adversarial-eval --profile qwen3-8b-local-max --json --timeout 900
```

This does not prove AGI safety. It makes one necessary condition measurable:
each layer should keep its assigned authority even when the input tries to
collapse the separation.

The first `qwen3-8b-local-max` run showed why this suite matters: Cold Eyes
passed 6/6, while raw Main Agent output often followed meta requests. That is
not treated as a Main Agent responsibility. The bottom model is allowed to
remain a generator; hidden-state leakage is handled by an external
deterministic candidate check. Public canon references are not hidden-state
leaks: the user has a right to know which public rule governs a refusal.
Invented canon clause numbers are treated separately as an Honest-side
candidate issue, not as a Harmless violation.

That result should be read narrowly. It measures one role-collapse failure
mode; it does not replace stronger adversarial suites, held-out tests, or hard
authority separation.
