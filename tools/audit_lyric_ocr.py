#!/usr/bin/env python3
"""Audit lyric OCR accuracy against an ``expected_lyrics.json`` gold fixture.

Two subcommands:

* ``bootstrap`` — run layout segmentation on pages and emit a lyric-row
  skeleton (one entry per detected lyric row) so a human only fills in the
  ``script`` and expected ``text`` instead of reading pixel coordinates.
* ``score`` — run the OCR backend on every gold row crop and report character
  and word error rates bucketed by script. Requires the tesseract binary plus
  language packs (``brew install tesseract tesseract-lang``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from psaltica_ocr.layout_segmentation import segment_page_layout
from psaltica_ocr.lyric_ocr import LyricOcr, TesseractBackend
from psaltica_ocr.lyric_ocr_audit import audit_gold, load_gold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    bootstrap = sub.add_parser("bootstrap", help="Emit lyric-row skeleton for labelling")
    bootstrap.add_argument("--pages", nargs="+", type=Path, required=True)
    bootstrap.add_argument("--output", type=Path, default=Path("data/expected_lyrics_bands.json"))
    bootstrap.add_argument("--notation-direction", choices=["ltr", "rtl", "auto"], default="ltr")

    score = sub.add_parser("score", help="Score OCR text accuracy against a gold fixture")
    score.add_argument("--gold", type=Path, default=Path("config/expected_lyrics_gold.json"))
    score.add_argument("--output", type=Path, default=Path("data/lyric_ocr_report.json"))
    score.add_argument("--psm", type=int, default=7)
    return parser.parse_args()


def run_bootstrap(args: argparse.Namespace) -> None:
    pages: list[dict[str, object]] = []
    for page in args.pages:
        image = cv2.imread(str(page), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"skip unreadable page: {page}")
            continue
        layout = segment_page_layout(image, notation_direction=args.notation_direction)
        rows: list[dict[str, object]] = []
        for chant_row in layout.chant_rows:
            for region in chant_row.lyric_rows:
                rows.append(_row_skeleton(region.bbox, chant_index=chant_row.index))
        for region in layout.unpaired_lyric_rows:
            rows.append(_row_skeleton(region.bbox, chant_index=None))
        pages.append({"image_path": str(page), "rows": rows})
        print(f"{page}: {len(rows)} lyric rows")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"pages": pages}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


def _row_skeleton(bbox, *, chant_index: int | None) -> dict[str, object]:
    return {
        "chant_row_index": chant_index,
        "bbox": {"x1": bbox.x1, "y1": bbox.y1, "x2": bbox.x2, "y2": bbox.y2},
        "script": "",
        "direction": "",
        "text": "",
    }


def run_score(args: argparse.Namespace) -> None:
    gold_pages = load_gold(args.gold)
    adapter = LyricOcr(TesseractBackend(psm=args.psm))
    report = audit_gold(gold_pages, adapter)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    overall = report.overall
    print(f"engine={report.engine} rows={len(report.rows)}")
    print(f"overall char_acc={overall.char_accuracy:.3f} word_acc={overall.word_accuracy:.3f}")
    for score in report.by_script():
        print(
            f"  {score.script}: rows={score.rows} "
            f"char_acc={score.char_accuracy:.3f} word_acc={score.word_accuracy:.3f}"
        )
    print(f"wrote {args.output}")


def main() -> None:
    args = parse_args()
    if args.command == "bootstrap":
        run_bootstrap(args)
    else:
        run_score(args)


if __name__ == "__main__":
    main()
