from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training_data import (
    load_sft_jsonl_rows,
    training_data_quality_errors,
    training_data_quality_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge validated SFT JSONL files without duplicate row ids.")
    parser.add_argument("--input-file", action="append", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--allow-duplicate-ids", action="store_true")
    return parser.parse_args()


def merge_rows(input_files: list[Path], allow_duplicate_ids: bool = False) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    file_reports: list[dict[str, Any]] = []
    errors: list[str] = []
    seen_ids: set[str] = set()
    duplicate_ids: list[str] = []

    for input_file in input_files:
        loaded, load_errors, total = load_sft_jsonl_rows(input_file)
        if load_errors:
            errors.extend(f"{input_file}: {error}" for error in load_errors)
        for row in loaded:
            row_id = str(row.get("id"))
            if row_id in seen_ids:
                duplicate_ids.append(row_id)
            seen_ids.add(row_id)
            rows.append(row)
        file_reports.append({"path": str(input_file), "total_lines": total, "valid_rows": len(loaded)})

    report = training_data_quality_report(rows) if rows else {"rows": 0, "duplicate_ids": []}
    error_report = dict(report)
    if allow_duplicate_ids:
        error_report["duplicate_ids"] = []
    quality_errors = training_data_quality_errors(
        error_report,
        require_system=True,
        require_generated_metadata=True,
    )
    if duplicate_ids and not allow_duplicate_ids:
        quality_errors.append(f"duplicate row ids across inputs: {', '.join(sorted(set(duplicate_ids)))}")
    errors.extend(quality_errors)
    return rows, {
        "input_files": file_reports,
        "rows": len(rows),
        "duplicate_ids": sorted(set(duplicate_ids)),
        "errors": errors,
        "category_counts": report.get("category_counts", {}),
        "source_counts": report.get("source_counts", {}),
        "split_counts": report.get("split_counts", {}),
    }


def main() -> int:
    args = parse_args()
    input_files = [Path(value) for value in args.input_file]
    output_file = Path(args.output_file)
    rows, summary = merge_rows(input_files, allow_duplicate_ids=args.allow_duplicate_ids)
    if summary["errors"]:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 1
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    summary["output_file"] = str(output_file)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
