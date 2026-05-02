#!/usr/bin/env python3
"""Run OCR inference for a page image.

Detector inference is implemented in a later phase; this CLI establishes the
shared direction contract used by rendering and cluster assembly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from psaltica_ocr.reading_order import detect_page_direction, load_direction_map, normalize_direction, resolve_direction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--book-id", default="page")
    parser.add_argument("--page-number", type=int, default=1)
    parser.add_argument("--direction", choices=["ltr", "rtl", "auto"], default="ltr")
    parser.add_argument("--direction-map", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    direction = normalize_direction(args.direction)
    direction_map = load_direction_map(args.direction_map)
    resolved = resolve_direction(
        args.book_id,
        args.page_number,
        default=direction,
        direction_map=direction_map,
    )
    if resolved == "auto":
        image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
        if image is None:
            raise SystemExit(f"Could not read image: {args.image}")
        resolved = detect_page_direction(image)
    raise SystemExit(f"infer_page detector phase is not implemented yet; resolved direction={resolved}")


if __name__ == "__main__":
    main()
