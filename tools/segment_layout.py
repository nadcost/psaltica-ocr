#!/usr/bin/env python3
"""Segment rendered pages into chant rows, lyric rows, and non-score regions."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import cv2

from psaltica_ocr.layout_segmentation import segment_page_layout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    page_group = parser.add_mutually_exclusive_group(required=True)
    page_group.add_argument("--pages", nargs="+", type=Path, help="Rendered page image paths")
    page_group.add_argument("--book", help="Book ID under page root")
    page_group.add_argument("--all", dest="all_books", action="store_true", help="Run across all books under page root")
    parser.add_argument("--page-root", type=Path, default=Path("data/pages_full"))
    parser.add_argument("--pages-per-book", type=int, default=0, help="Limit pages per book; 0 = all")
    parser.add_argument("--notation-direction", choices=["ltr", "rtl", "auto"], default="ltr")
    parser.add_argument("--output", type=Path, default=Path("data/layout_segments.json"))
    parser.add_argument("--html", type=Path, help="Optional visual overlay report")
    return parser.parse_args()


def collect_pages(args: argparse.Namespace) -> list[Path]:
    if args.pages:
        return [page for page in args.pages if page.exists()]
    if args.all_books:
        book_dirs = sorted(path for path in args.page_root.iterdir() if path.is_dir())
    else:
        book_dirs = [args.page_root / args.book]

    pages: list[Path] = []
    for book_dir in book_dirs:
        book_pages = sorted(book_dir.glob("page_*.png"))
        if args.pages_per_book:
            book_pages = book_pages[: args.pages_per_book]
        pages.extend(book_pages)
    return pages


def write_html_report(path: Path, pages: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    parts = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        "<title>Psaltica layout segmentation audit</title>",
        "<style>",
        "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:24px;background:#f6f7f8;color:#172026}",
        "h1{font-size:22px;margin:0 0 16px}",
        "h2{font-size:16px;margin:24px 0 8px}",
        ".legend{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:18px;font-size:13px}",
        ".swatch{display:inline-block;width:12px;height:12px;border:2px solid currentColor;margin-right:4px;vertical-align:-2px}",
        ".page{margin:0 0 28px;padding:12px;background:#fff;border:1px solid #d7dde2;border-radius:6px}",
        ".canvas{position:relative;display:inline-block;max-width:100%;border:1px solid #c9d0d7;background:#fff}",
        ".canvas img{display:block;max-width:100%;height:auto}",
        ".box{position:absolute;box-sizing:border-box;border:3px solid;pointer-events:none}",
        ".chant{border-color:#14883f;background:rgba(20,136,63,.08)}",
        ".lyrics{border-color:#1169d8;background:rgba(17,105,216,.10)}",
        ".unpaired{border-color:#d47600;background:rgba(212,118,0,.14)}",
        ".non-score{border-color:#747b83;background:rgba(116,123,131,.12)}",
        ".label{position:absolute;left:0;top:-20px;padding:2px 5px;background:currentColor;color:#fff;font-size:12px;white-space:nowrap}",
        ".metrics{font-size:13px;color:#46515c;margin-bottom:8px}",
        "</style>",
        "</head>",
        "<body>",
        "<h1>Psaltica layout segmentation audit</h1>",
        '<div class="legend">',
        '<span><span class="swatch" style="color:#14883f"></span>chant row</span>',
        '<span><span class="swatch" style="color:#1169d8"></span>paired lyrics</span>',
        '<span><span class="swatch" style="color:#d47600"></span>unpaired lyrics</span>',
        '<span><span class="swatch" style="color:#747b83"></span>non-score</span>',
        "</div>",
    ]

    for page in pages:
        image_path = Path(str(page["image_path"]))
        width = int(page["width"])
        height = int(page["height"])
        chant_rows = page["chant_rows"]
        unpaired = page["unpaired_lyric_rows"]
        non_score = page["non_score_regions"]
        paired_count = sum(len(row["lyric_rows"]) for row in chant_rows)  # type: ignore[index]
        title = html.escape(str(image_path))
        parts.extend(
            [
                '<section class="page">',
                f"<h2>{title}</h2>",
                (
                    '<div class="metrics">'
                    f"chant_rows={len(chant_rows)} paired_lyrics={paired_count} "
                    f"unpaired_lyrics={len(unpaired)} non_score={len(non_score)} "
                    f"notation_direction={html.escape(str(page['notation_direction']))}"
                    "</div>"
                ),
                '<div class="canvas">',
                f'<img src="{html.escape(image_path.resolve().as_uri())}" width="{width}" height="{height}" alt="{title}">',
            ]
        )
        for row in chant_rows:  # type: ignore[assignment]
            parts.append(_box_html(row["bbox"], width, height, "chant", f"chant {row['index']}"))
            for lyric in row["lyric_rows"]:
                parts.append(_box_html(lyric["bbox"], width, height, "lyrics", f"lyrics -> {row['index']}"))
        for index, lyric in enumerate(unpaired):  # type: ignore[assignment]
            parts.append(_box_html(lyric["bbox"], width, height, "unpaired", f"unpaired {index}"))
        for index, region in enumerate(non_score):  # type: ignore[assignment]
            parts.append(_box_html(region["bbox"], width, height, "non-score", f"non-score {index}"))
        parts.extend(["</div>", "</section>"])

    parts.extend(["</body>", "</html>"])
    path.write_text("\n".join(parts), encoding="utf-8")


def _box_html(bbox: list[int], width: int, height: int, class_name: str, label: str) -> str:
    x1, y1, x2, y2 = bbox
    style = (
        f"left:{x1 / width * 100:.4f}%;"
        f"top:{y1 / height * 100:.4f}%;"
        f"width:{(x2 - x1) / width * 100:.4f}%;"
        f"height:{(y2 - y1) / height * 100:.4f}%;"
    )
    escaped_label = html.escape(label)
    return f'<div class="box {class_name}" style="{style}"><span class="label">{escaped_label}</span></div>'


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
    if args.html:
        write_html_report(args.html, results)
        print(f"wrote {args.html}")


if __name__ == "__main__":
    main()
