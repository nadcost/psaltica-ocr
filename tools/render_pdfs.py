#!/usr/bin/env python3
"""Render source PDFs to deterministic page PNGs and a manifest CSV."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

from psaltica_ocr.rendering import RenderedPage, iter_pdf_paths, render_pdf_pages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="PDF files or directories containing PDFs")
    parser.add_argument("--output-root", type=Path, default=Path("data/pages"))
    parser.add_argument("--manifest", type=Path, default=Path("data/pages/manifest.csv"))
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--first-page", type=int)
    parser.add_argument("--last-page", type=int)
    parser.add_argument("--mask-lyrics", action="store_true")
    parser.add_argument("--force", action="store_true", help="Overwrite existing PNGs")
    return parser.parse_args()


def write_manifest(path: Path, pages: list[RenderedPage]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "book_id",
        "page_number",
        "image_path",
        "sha256",
        "dpi",
        "width",
        "height",
        "masked",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for page in pages:
            row = asdict(page)
            row["image_path"] = str(page.image_path)
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    pdf_paths = iter_pdf_paths(args.inputs)
    if not pdf_paths:
        raise SystemExit("No PDF inputs found.")

    pages: list[RenderedPage] = []
    for pdf_path in pdf_paths:
        pages.extend(
            render_pdf_pages(
                pdf_path,
                args.output_root,
                dpi=args.dpi,
                first_page=args.first_page,
                last_page=args.last_page,
                mask=args.mask_lyrics,
                force=args.force,
            )
        )

    write_manifest(args.manifest, pages)
    print(f"Rendered {len(pages)} pages from {len(pdf_paths)} PDFs.")


if __name__ == "__main__":
    main()
