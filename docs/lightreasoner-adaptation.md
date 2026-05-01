# LightReasoner Adaptation Note

Source paper: [LightReasoner: Can Small Language Models Teach Large Language Models Reasoning?](https://arxiv.org/abs/2510.07962)

## Useful Idea

The paper's practical claim is that reasoning improvement should not train on
every token uniformly. Most tokens are routine. The useful signal is concentrated
where an Expert model and a weaker Amateur model diverge.

The full method needs token probability distributions from both models. It uses
KL divergence to find high-value reasoning steps, then trains the Expert toward
contrastive soft labels. This repository currently talks to Ollama through chat
responses, so it does not have the token-level distributions needed for faithful
LightReasoner training.

## Local Translation

This repo implements a conservative `LightReasoner-lite` data selector:

1. Run the same synthetic Main Agent record through an Expert profile.
2. Run the same record through an Amateur profile.
3. Score both visible answers with existing role-behavior issues, verifier
   failures, and local candidate-quality penalties.
4. Export only cases where the Expert is clean and the Amateur is clearly worse.

This is sample-level contrast, not token-level KL. It is useful because it keeps
future LoRA data small and focused on behavior gaps that actually appear on this
machine.

## Command

```powershell
python main.py main-contrast-export --json --timeout 900
```

Defaults:

- Expert: `qwen3-8b-s2t-lite`
- Amateur: `qwen3-1.7b-amateur`
- Input: `data\main_agent_hard_seed.jsonl`
- Output: `runs\main-agent-contrast.jsonl`

Use an 8B-only comparison when the 1.7B model is not available:

```powershell
python main.py main-contrast-export --amateur-profile qwen3-8b-local-max --json --timeout 900
```

The output JSONL intentionally contains synthetic prompts and selected Expert
answers. It should not be built from private chat logs. The command summary
omits prompt text and model outputs.

## Why This Matters Here

The current hard corpus is already small and partly saturated. Blindly adding
more SFT rows risks teaching the model to imitate easy cases it already handles.
The contrast selector instead asks:

- Where does the stronger profile produce a clean answer?
- Where does the weaker profile fail, refuse, drift, overrun, or miss a verifier?
- Which of those differences are large enough to justify adapter training?

That matches the repository goal: maximize the local Main Agent's useful
reasoning behavior without moving safety or refusal authority back into the Main
Agent.

## Next Gate

Do not train from the contrast export until it contains enough selected cases
across categories. A useful first gate is:

```powershell
python main.py main-contrast-export --min-score-gap 100 --max-length-ratio 4 --json --timeout 900
```

Then inspect the selected count and category spread. If only one or two cases
are selected, grow the held-out hard corpus first instead of training.
