#!/usr/bin/env python3
"""Audit masked rendered pages against a fixed content/blank page sample."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from psaltica_ocr.rendering import compute_ink_ratio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=Path("config/mask_audit_pages.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/pages/mask_audit.csv"))
    parser.add_argument("--min-content-ratio", type=float, default=0.0008)
    parser.add_argument("--max-blank-ratio", type=float, default=0.0003)
    parser.add_argument("--min-precision", type=float, default=0.90)
    return parser.parse_args()


def read_audit_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def audit_rows(
    rows: list[dict[str, str]],
    *,
    min_content_ratio: float,
    max_blank_ratio: float,
) -> list[dict[str, object]]:
    audited: list[dict[str, object]] = []
    for row in rows:
        path = Path(row["image_path"])
        expected = row["expected"]
        ratio = compute_ink_ratio(path)
        if expected == "content":
            passed = ratio >= min_content_ratio
        elif expected == "blank":
            passed = ratio <= max_blank_ratio
        else:
            raise ValueError(f"Unsupported expected value: {expected}")
        audited.append(
            {
                "image_path": str(path),
                "expected": expected,
                "ink_ratio": ratio,
                "passed": passed,
            }
        )
    return audited


def write_report(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "expected", "ink_ratio", "passed"])
        writer.writeheader()
        writer.writerows(rows)


def precision(rows: list[dict[str, object]]) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if row["passed"]) / len(rows)


def main() -> None:
    args = parse_args()
    rows = audit_rows(
        read_audit_rows(args.audit),
        min_content_ratio=args.min_content_ratio,
        max_blank_ratio=args.max_blank_ratio,
    )
    score = precision(rows)
    write_report(args.output, rows)
    print(f"mask audit precision={score:.3f} ({sum(1 for row in rows if row['passed'])}/{len(rows)})")
    if score < args.min_precision:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
