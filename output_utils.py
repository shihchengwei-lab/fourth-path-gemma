from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any


def new_run_id() -> str:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}-{uuid.uuid4().hex[:8]}"


def elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def print_json_or_text(data: Any, as_json: bool, text: str) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2) if as_json else text)


def write_json_summary(
    data: dict[str, Any],
    output_file: Path | None,
    runs_dir: Path,
    prefix: str,
    path_key: str,
) -> Path:
    path = output_file or runs_dir / f"{prefix}-{new_run_id()}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    data[path_key] = str(path)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
