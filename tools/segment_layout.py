#!/usr/bin/env python3
"""Segment rendered pages into chant rows, lyric rows, and non-score regions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from psaltica_ocr.layout_segmentation import segment_page_layout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    page_group = parser.add_mutually_exclusive_group(required=True)
    page_group.add_argument("--pages", nargs="+", type=Path, help="Rendered page image paths")
    page_group.add_argument("--book", help="Book ID under data/pages/<book>/")
    page_group.add_argument("--all", dest="all_books", action="store_true", help="Run across all books under data/pages/")
    parser.add_argument("--pages-per-book", type=int, default=0, help="Limit pages per book; 0 = all")
    parser.add_argument("--notation-direction", choices=["ltr", "rtl", "auto"], default="ltr")
    parser.add_argument("--output", type=Path, default=Path("data/layout_segments.json"))
    return parser.parse_args()


def collect_pages(args: argparse.Namespace) -> list[Path]:
    if args.pages:
        return [page for page in args.pages if page.exists()]
    if args.all_books:
        book_dirs = sorted(path for path in Path("data/pages").iterdir() if path.is_dir())
    else:
        book_dirs = [Path("data/pages") / args.book]

    pages: list[Path] = []
    for book_dir in book_dirs:
        book_pages = sorted(book_dir.glob("page_*.png"))
        if args.pages_per_book:
            book_pages = book_pages[: args.pages_per_book]
        pages.extend(book_pages)
    return pages


def main() -> None:
    args = parse_args()
    pages = collect_pages(args)
    results: list[dict[str, object]] = []
    for page in pages:
        image = cv2.imread(str(page), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"skip unreadable page: {page}")
            continue
        layout = segment_page_layout(image, notation_direction=args.notation_direction)
        record = layout.to_dict()
        record["image_path"] = str(page)
        results.append(record)
        lyric_rows = sum(len(row.lyric_rows) for row in layout.chant_rows)
        print(
            f"{page}: chant_rows={len(layout.chant_rows)} "
            f"paired_lyrics={lyric_rows} non_score={len(layout.non_score_regions)} "
            f"unpaired_lyrics={len(layout.unpaired_lyric_rows)}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump({"pages": results}, handle, indent=2)
        handle.write("\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
