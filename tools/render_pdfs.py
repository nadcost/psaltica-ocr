#!/usr/bin/env python3
"""Render source PDFs to deterministic page PNGs and a manifest CSV."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

from pdf2image import pdfinfo_from_path

from psaltica_ocr.rendering import RenderedPage, iter_pdf_paths, render_pdf_pages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="PDF files or directories containing PDFs")
    parser.add_argument("--output-root", type=Path, default=Path("data/pages"))
    parser.add_argument("--manifest", type=Path, default=Path("data/pages/manifest.csv"))
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--first-page", type=int)
    parser.add_argument("--last-page", type=int)
    parser.add_argument(
        "--page-range",
        help="Comma-separated page ranges, e.g. 50-150,200,220-225. Overrides --first-page/--last-page.",
    )
    parser.add_argument("--chunk-size", type=int, default=25, help="Maximum pages rendered per pdf2image call")
    parser.add_argument("--resume-from-manifest", action="store_true", help="Skip pages already present in manifest")
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


def parse_page_range(value: str | None) -> list[int] | None:
    if not value:
        return None
    pages: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if start > end:
                raise ValueError(f"Invalid descending page range: {part}")
            pages.update(range(start, end + 1))
        else:
            pages.add(int(part))
    if any(page < 1 for page in pages):
        raise ValueError("Page numbers are 1-based and must be positive")
    return sorted(pages)


def pages_from_bounds(first_page: int | None, last_page: int | None, total_pages: int) -> list[int]:
    start = first_page or 1
    end = last_page or total_pages
    if start > end:
        raise ValueError("--first-page cannot be greater than --last-page")
    if start < 1:
        raise ValueError("Page numbers are 1-based and must be positive")
    return list(range(start, min(end, total_pages) + 1))


def contiguous_runs(pages: list[int]) -> list[tuple[int, int]]:
    if not pages:
        return []
    runs: list[tuple[int, int]] = []
    start = previous = pages[0]
    for page in pages[1:]:
        if page == previous + 1:
            previous = page
            continue
        runs.append((start, previous))
        start = previous = page
    runs.append((start, previous))
    return runs


def chunk_pages(pages: list[int], chunk_size: int) -> list[tuple[int, int]]:
    if chunk_size < 1:
        raise ValueError("--chunk-size must be at least 1")
    chunks: list[tuple[int, int]] = []
    for start, end in contiguous_runs(pages):
        chunk_start = start
        while chunk_start <= end:
            chunk_end = min(end, chunk_start + chunk_size - 1)
            chunks.append((chunk_start, chunk_end))
            chunk_start = chunk_end + 1
    return chunks


def manifest_pages(path: Path) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return {
            (row["book_id"], int(row["page_number"]))
            for row in reader
            if row.get("book_id") and row.get("page_number")
        }


def plan_pages(
    pdf_path: Path,
    *,
    page_range: str | None,
    first_page: int | None,
    last_page: int | None,
    manifest: Path,
    resume: bool,
    chunk_size: int,
) -> list[tuple[int, int]]:
    explicit_pages = parse_page_range(page_range)
    if explicit_pages is None:
        total_pages = int(pdfinfo_from_path(pdf_path)["Pages"])
        pages = pages_from_bounds(first_page, last_page, total_pages)
    else:
        pages = explicit_pages

    if resume:
        completed = manifest_pages(manifest)
        book_id = pdf_path.stem
        pages = [page for page in pages if (book_id, page) not in completed]

    return chunk_pages(pages, chunk_size)


def main() -> None:
    args = parse_args()
    pdf_paths = iter_pdf_paths(args.inputs)
    if not pdf_paths:
        raise SystemExit("No PDF inputs found.")

    pages: list[RenderedPage] = []
    for pdf_path in pdf_paths:
        chunks = plan_pages(
            pdf_path,
            page_range=args.page_range,
            first_page=args.first_page,
            last_page=args.last_page,
            manifest=args.manifest,
            resume=args.resume_from_manifest,
            chunk_size=args.chunk_size,
        )
        for first_page, last_page in chunks:
            pages.extend(
                render_pdf_pages(
                    pdf_path,
                    args.output_root,
                    dpi=args.dpi,
                    first_page=first_page,
                    last_page=last_page,
                    mask=args.mask_lyrics,
                    force=args.force,
                )
            )

    write_manifest(args.manifest, pages)
    print(f"Rendered {len(pages)} pages from {len(pdf_paths)} PDFs.")


if __name__ == "__main__":
    main()
