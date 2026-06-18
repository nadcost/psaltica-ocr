#!/usr/bin/env python3
"""Seed per-page lyric gold DRAFTS under data/corrections/ for hand correction.

For each page it runs layout segmentation + the lyric OCR adapter over the
paired lyric rows and writes ``data/corrections/<book>_p<page>/`` containing:

* ``expected_lyrics.json`` — one entry per detected lyric row with the OCR text,
  bbox, script, and direction, all marked ``needs_review: true``. A human
  corrects the text/script rather than typing from scratch.
* ``expected_composition.json`` / ``expected_lyric_alignment.json`` — empty
  TODO stubs. These are deliberately NOT auto-filled: the composition string
  needs the neume cluster grammar (expert reading or the trained detector), and
  alignment depends on it. Filling them with OCR guesses would be false gold.

Lyric OCR is rough, so the output is a starting point for review, not gold yet.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from psaltica_ocr.layout_segmentation import segment_page_layout
from psaltica_ocr.lyric_ocr import LyricOcr, TesseractBackend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pages", nargs="+", type=Path)
    group.add_argument("--pages-file", type=Path, help="One page image path per line")
    parser.add_argument("--out-root", type=Path, default=Path("data/corrections"))
    parser.add_argument("--notation-direction", choices=["ltr", "rtl", "auto"], default="ltr")
    return parser.parse_args()


def collect_pages(args: argparse.Namespace) -> list[Path]:
    if args.pages:
        return [p for p in args.pages if p.exists()]
    lines = args.pages_file.read_text(encoding="utf-8").splitlines()
    return [Path(line.strip()) for line in lines if line.strip() and Path(line.strip()).exists()]


def page_key(image_path: Path) -> str:
    book = image_path.parent.name.replace(" ", "_")
    return f"{book}_{image_path.stem}"


def seed_page(image_path: Path, adapter: LyricOcr, out_root: Path, notation_direction: str) -> int:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"skip unreadable: {image_path}")
        return 0
    layout = segment_page_layout(image, notation_direction=notation_direction)

    rows: list[dict[str, object]] = []
    for chant_row in layout.chant_rows:
        for region in chant_row.lyric_rows:
            line = adapter.recognize_line(image, region.bbox)
            rows.append(
                {
                    "chant_row_index": chant_row.index,
                    "bbox": {"x1": region.bbox.x1, "y1": region.bbox.y1, "x2": region.bbox.x2, "y2": region.bbox.y2},
                    "script": line.script,
                    "direction": line.direction,
                    "text": line.text,
                    "raw_text": line.raw_text,
                    "ocr_confidence": round(line.confidence, 4),
                    "needs_review": True,
                }
            )

    out_dir = out_root / page_key(image_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "expected_lyrics.json").write_text(
        json.dumps({"image_path": str(image_path), "source": "ocr_draft", "rows": rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_stub(out_dir / "expected_composition.json", {
        "_todo": "Hand-fill segments[].composition with Psaltica insert strings (needs neume cluster grammar / trained detector). Validate with tools/validate_gold_compositions.py.",
        "image_path": str(image_path),
        "segments": [],
    })
    _write_stub(out_dir / "expected_lyric_alignment.json", {
        "_todo": "Hand-fill after composition: map each lyric unit to segment/cluster indexes. Depends on expected_composition.json.",
        "image_path": str(image_path),
        "alignments": [],
    })
    print(f"{page_key(image_path)}: {len(rows)} lyric-row drafts")
    return len(rows)


def _write_stub(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        return  # never overwrite human-edited gold
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    adapter = LyricOcr(TesseractBackend())
    total = 0
    for page in collect_pages(args):
        total += seed_page(page, adapter, args.out_root, args.notation_direction)
    print(f"\nSeeded {total} lyric-row drafts under {args.out_root}")


if __name__ == "__main__":
    main()
