#!/usr/bin/env python3
"""Audit layout segmentation against a hand-labelled gold fixture.

Two subcommands:

* ``bootstrap`` — emit precise ink line-bands per page so a gold fixture can be
  labelled by assigning roles to bands instead of reading pixel coordinates.
* ``score`` — run segmentation on every gold page and report the gate metrics
  (chant-mask precision and lyric-pairing precision, both must be >= 0.90).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from psaltica_ocr.layout_audit import audit_gold, extract_line_bands, load_gold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    bootstrap = sub.add_parser("bootstrap", help="Emit line-band skeleton for labelling")
    bootstrap.add_argument("--pages", nargs="+", type=Path, required=True)
    bootstrap.add_argument("--output", type=Path, default=Path("data/layout_audit_bands.json"))
    bootstrap.add_argument("--min-gap", type=int, default=10)

    score = sub.add_parser("score", help="Score segmentation against a gold fixture")
    score.add_argument("--gold", type=Path, default=Path("config/layout_audit_gold.json"))
    score.add_argument("--notation-direction", choices=["ltr", "rtl", "auto"], default="ltr")
    score.add_argument("--output", type=Path, default=Path("data/layout_audit_report.json"))
    score.add_argument("--min-precision", type=float, default=0.90)
    return parser.parse_args()


def run_bootstrap(args: argparse.Namespace) -> None:
    pages: list[dict[str, object]] = []
    for page in args.pages:
        image = cv2.imread(str(page), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"skip unreadable page: {page}")
            continue
        bands = extract_line_bands(image, min_gap=args.min_gap)
        pages.append(
            {
                "image_path": str(page),
                "bands": [{"y1": band.y1, "y2": band.y2} for band in bands],
            }
        )
        print(f"{page}: {len(bands)} line bands")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"pages": pages}, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


def run_score(args: argparse.Namespace) -> None:
    gold_pages = load_gold(args.gold)
    report = audit_gold(gold_pages, notation_direction=args.notation_direction)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")

    print(f"pages={len(report.pages)}")
    print(
        f"chant_mask_precision={report.chant_mask_precision:.3f} "
        f"recall={report.chant_mask_recall:.3f}"
    )
    print(
        f"lyric_pairing_precision={report.lyric_pairing_precision:.3f} "
        f"recall={report.lyric_pairing_recall:.3f}"
    )
    for page in report.pages:
        print(
            f"  {page.image_path}: "
            f"chant_p={page.chant_mask_precision:.2f} "
            f"lyric_p={page.lyric_pairing_precision:.2f} "
            f"({page.lyric_correct}/{page.lyric_pred} pairings, gold={page.lyric_gold})"
        )
    print(f"wrote {args.output}")
    if not report.passes_gate(min_precision=args.min_precision):
        raise SystemExit(1)


def main() -> None:
    args = parse_args()
    if args.command == "bootstrap":
        run_bootstrap(args)
    else:
        run_score(args)


if __name__ == "__main__":
    main()
