#!/usr/bin/env python3
"""Validate hand-seeded expected_composition.json gold files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from psaltica_ocr.ocr_validator import validate_composition


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, default=[Path("data/corrections")])
    parser.add_argument("--min-count", type=int, default=5)
    return parser.parse_args()


def gold_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.rglob("expected_composition.json")))
        elif path.name == "expected_composition.json":
            files.append(path)
    return files


def main() -> None:
    args = parse_args()
    files = gold_files(args.paths)
    if len(files) < args.min_count:
        raise SystemExit(f"Expected at least {args.min_count} gold files, found {len(files)}.")
    failures: list[str] = []
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        errors = validate_composition(payload)
        if errors:
            failures.append(f"{path}: {errors}")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"Validated {len(files)} gold composition files.")


if __name__ == "__main__":
    main()
