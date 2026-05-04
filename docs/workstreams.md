# Workstreams

The project goal is simple:

```text
Capability and safety should grow together.
Do not leak dangerous capability.
Do not over-cage normal capability.
```

That creates two main workstreams and one evidence rule.

## 1. Capability Extraction

This stream asks how much more useful work the local 8B Main Agent can do before
changing the architecture boundary.

Useful levers:

- better prompt contracts;
- verifier-backed hard and held-out datasets;
- golden answers written directly by the project owner or agent;
- NVIDIA teacher export while external endpoints are available;
- R1-lite sampling, LIMO-style curation, and mix distillation;
- LoRA / QLoRA when the data and held-out gates justify it;
- paper-driven test-time compute or routing ideas.

Boundary:

- The Main Agent may become better at solving normal tasks.
- The Main Agent must not become the safety judge.
- Training data should improve candidate quality, format stability, reasoning,
  and useful near-boundary helpfulness, not teach the bottom model to issue
  final approval or final refusal.

Known limit:

An 8B local model has a ceiling. More thinking, more prompt shaping, or more
adapter training may still fail on hard reasoning. That is a capability limit,
not a reason to move final safety authority into the Main Agent.

## 2. Safety-Layer Pressure

This stream asks whether external containment still holds when the candidate or
action layer tries to collapse authority.

Pressure targets:

- fake approval such as candidate text claiming to be allowed, passed, or
  approved by Cold Eyes;
- hidden control-plane leakage;
- unsupported canon or policy claims;
- attempts to turn public explanation into authority;
- unaudited side effects;
- project-relative sensitive file reads;
- candidate text that tries to modify audit, refusal, policy, or tool behavior.

Boundary:

- The pressure test should attack the external boundary, not train the Main
  Agent into a refusal engine.
- Passing a pressure set means the tested route held under that set. It is not
  a universal safety proof.
- If stronger agents, teacher models, or manually written attacks still cannot
  bypass the boundary, that is meaningful local containment evidence.

## 3. Evidence Rule

Every serious improvement round should report both sides:

```text
Capability result: what got better or did not regress?
Containment result: what still could not get through?
Over-blocking result: what normal near-boundary work still passed?
```

Do not count a round as progress if it only makes the model safer by blocking
normal capability, or only makes it more capable by leaking dangerous output or
unaudited authority.

The ideal result is not a timid model. The ideal result is a stronger candidate
generator behind a boundary that stronger pressure still cannot route around.

## 4. Eval Surface Handling

Do not spend clean eval surfaces casually.

- A previously run set can remain useful as a dev set, regression set, or
  failure-analysis surface.
- Once failures from a set influence prompts, data, verifier rules, gates, or
  adapter training, stop using that set as a clean held-out claim.
- Old failed rows should be used to name failure families and write new
  same-family rows with different wording, task content, and expected outputs.
- Clean capability held-out runs should happen only after the round design is
  frozen.
- `data/main_agent_v5_clean_heldout_seed.jsonl` is a legacy spent local
  surface. It is not capability evidence.
- The old `v6` through `v17` clean-heldout files are withdrawn, not merely
  downgraded. Remove them from eval/gate context. They may only contribute
  indirectly through rewritten failure families or verifier-accepted teacher
  rows; do not keep the old files as evidence surfaces.
- Spent eval rows can become training candidates only after they are removed
  from the clean-claim lane and labeled as non-evidence training candidates.
  This is not automatic permission to train on every spent row; use explicit
  provenance, rewriting, or teacher acceptance. The next capability claim
  restarts at a newly minted unused `v6` surface after training.
- Strong safety pressure rows can become regression rows after they drive a
  fix; stronger containment claims need new or independently authored attacks.

Short rule:

```text
Old rows guide engineering.
New rows support evidence claims.
Next clean proof version is v6.
```

## 5. Safety Freeze Bar

If the external audit/action boundary holds under strong pressure, stop adding
audit rules by default. The audit layer should become low-change
infrastructure, not an infinite rule sink.

Freeze the audit layer only when all of these are true:

- old containment regression rows still pass;
- a new strong pressure set still cannot collapse final authority, audit
  authority, or action authority;
- sensitive reads, policy mutation, audit-log mutation, network exfiltration,
  and social-engineering side effects still fail closed;
- normal near-boundary helpful tasks still pass at an acceptable rate;
- code review finds no concrete authority-collapse path.

Reopen the audit layer only for a new tool/action surface, a real bypass, or a
new capability layer that creates a different route around the boundary. Until
then, spend most engineering effort on capability extraction: better data,
golden answers, verifier-backed selection, prompt contracts, and adapter
experiments.
