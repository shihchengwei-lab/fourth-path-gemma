# Fourth Path Gemma Prototype

Local open-weight model prototype for the Fourth Path architecture described in
[`separation-and-audit-alignment`](https://github.com/shihchengwei-lab/separation-and-audit-alignment).

This repository is a reference implementation, not a proven safety system. It is
intended to make the architecture concrete enough to inspect, run, and test on a
local machine.

## What It Implements

The prototype separates several roles that are often coupled inside one
assistant loop:

- **Main Agent**: produces candidate answers.
- **Input prefilter**: routes obvious canon violations to the refusal module; it
  cannot approve output.
- **Classify**: routes candidate-output issues; it cannot approve output.
- **Cold Eyes**: performs a single-pass review against `canon.md`.
- **Retry**: allows a bounded repair loop.
- **Refusal Module**: emits a fixed refusal after a blocked request.
- **Chat mode**: provides a simple natural-language interface over the audited
  pipeline.

Cold Eyes receives only the candidate output and the canon. It does not receive
the full chat history, the Main Agent system prompt, or hidden reasoning traces.

## Current Model Target

The local baseline targets:

```text
gemma4:e4b
```

The original test machine had 16GB RAM and an RTX 4060 Laptop GPU with 8GB
VRAM, so larger local baselines such as `gemma4:26b` or `gemma4:31b` were not
used for the first version.

## Setup

Install [Ollama](https://ollama.com/) and download the model:

```powershell
ollama pull gemma4:e4b
```

This repository has no Python package dependencies. It uses Python 3.12 standard
library only.

The included Windows helper scripts assume this local layout:

```text
E:\Ollama\ollama.exe
E:\ollama-models
```

Adjust the scripts if your Ollama executable or model directory lives elsewhere.

## Usage

Start the natural-language chat interface:

```powershell
.\chat.cmd
```

Inside chat mode:

```text
/help   Show commands.
/audit  Toggle detailed audit output.
/reset  Clear this chat session memory.
/exit   Leave chat.
```

Run one audited request:

```powershell
python main.py run --prompt "Summarize what this prototype does." --json --timeout 900
```

Inspect the Main Agent by itself, without the audit pipeline:

```powershell
python main.py diagnose-main --prompt "Write a simple Python function." --json --show-system-prompt --timeout 900
```

## Logs

Audit logs are written to:

```text
runs\
```

The logs record run id, attempt, route, Cold Eyes verdict, canon clause, and
final status. They do not store the original prompt, Main Agent system prompt,
full candidate output, or reasoning trace.

Local run logs are ignored by git.

## Tests

Run:

```powershell
python -m unittest discover -s tests -v
```

## Limitations

- This is a research prototype, not production safety infrastructure.
- The canon is intentionally small and only demonstrates the data flow.
- The Gemma model is instruction-tuned and may still carry safety behavior in
  its weights.
- The chat mode keeps memory only for the current process.
- The Windows helper scripts are convenience wrappers, not a cross-platform
  installer.

## License

MIT

> Pipeline integration validated.
