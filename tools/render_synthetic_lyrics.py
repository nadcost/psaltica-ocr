#!/usr/bin/env python3
"""Render a reproducible synthetic lyric-OCR gold set.

Each line of known text is rendered as a single printed lyric row (black on
white) in its script's font and recorded in an ``expected_lyrics`` gold fixture
with the true bbox, script, direction, and normalized text. This gives the
Phase-4 lyric-OCR gate a deterministic, regenerable accuracy benchmark without
hand-transcribing real pages; real labelled pages can be appended later.

Arabic is pre-shaped (``arabic_reshaper``) and bidi-reordered (``python-bidi``)
before rendering because Pillow on this platform has no raqm complex-text
shaping. Run, then score with ``tools/audit_lyric_ocr.py score
--gold config/expected_lyrics_synthetic.json``.
"""

from __future__ import annotations

import argparse
import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from psaltica_ocr.reading_order import Direction
from psaltica_ocr.lyric_ocr import Script


GREEK_FONT = "/System/Library/Fonts/Supplemental/Times New Roman.ttf"
LATIN_FONT = "/System/Library/Fonts/Supplemental/Times New Roman.ttf"
ARABIC_FONT = "/System/Library/Fonts/SFArabic.ttf"

MARGIN = 24
FONT_SIZE = 48


@dataclass(frozen=True)
class Line:
    script: Script
    direction: Direction
    text: str
    font_path: str


# Liturgical lyric lines; the text is the gold transcription.
LINES: tuple[Line, ...] = (
    Line("Greek", "ltr", "Κυριε ελεησον", GREEK_FONT),
    Line("Greek", "ltr", "Αγιος ο Θεος", GREEK_FONT),
    Line("Greek", "ltr", "Δοξα Πατρι και Υιω", GREEK_FONT),
    Line("Latin", "ltr", "Lord have mercy", LATIN_FONT),
    Line("Latin", "ltr", "Holy God Holy Mighty", LATIN_FONT),
    Line("Latin", "ltr", "Glory to the Father", LATIN_FONT),
    Line("Arabic", "rtl", "يا رب ارحم", ARABIC_FONT),
    Line("Arabic", "rtl", "قدوس الله", ARABIC_FONT),
    Line("Arabic", "rtl", "المجد للاب", ARABIC_FONT),
)


def _shape_for_render(line: Line) -> str:
    """Return the glyph order Pillow should draw (Arabic needs reshape+bidi)."""

    if line.script != "Arabic":
        return line.text
    import arabic_reshaper
    from bidi.algorithm import get_display

    return get_display(arabic_reshaper.reshape(line.text))


def render_line(line: Line, *, font_size: int) -> tuple[Image.Image, tuple[int, int, int, int]]:
    font = ImageFont.truetype(line.font_path, font_size)
    display = _shape_for_render(line)

    probe = Image.new("L", (10, 10), 255)
    left, top, right, bottom = ImageDraw.Draw(probe).textbbox((0, 0), display, font=font)
    text_w, text_h = right - left, bottom - top

    width = text_w + 2 * MARGIN
    height = text_h + 2 * MARGIN
    image = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(image)
    draw.text((MARGIN - left, MARGIN - top), display, fill=0, font=font)

    bbox = (MARGIN, MARGIN, MARGIN + text_w, MARGIN + text_h)
    return image, bbox


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, default=Path("data/lyric_synth"))
    parser.add_argument("--gold", type=Path, default=Path("config/expected_lyrics_synthetic.json"))
    parser.add_argument("--font-size", type=int, default=FONT_SIZE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.image_dir.mkdir(parents=True, exist_ok=True)
    pages: list[dict[str, object]] = []
    for index, line in enumerate(LINES):
        image, bbox = render_line(line, font_size=args.font_size)
        image_path = args.image_dir / f"row_{index:02d}_{line.script.lower()}.png"
        image.save(image_path)
        pages.append(
            {
                "image_path": str(image_path),
                "rows": [
                    {
                        "bbox": {"x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3]},
                        "script": line.script,
                        "direction": line.direction,
                        "text": unicodedata.normalize("NFC", line.text),
                    }
                ],
            }
        )
        print(f"{image_path}: {line.script} {line.direction!s} {line.text!r}")

    args.gold.parent.mkdir(parents=True, exist_ok=True)
    args.gold.write_text(
        json.dumps({"pages": pages}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.gold} ({len(pages)} rows)")


if __name__ == "__main__":
    main()
