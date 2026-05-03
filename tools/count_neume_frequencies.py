#!/usr/bin/env python3
"""Count per-symbol detection frequency across pages to inform keyboard layout design.

Runs font-based template matching on a sample of rendered pages and reports how
often each neume class appears, grouped by category and ranked by frequency.

Usage:
  # Sample 100 pages spread across all books (default)
  python tools/count_neume_frequencies.py

  # Specific book
  python tools/count_neume_frequencies.py --book Mass --sample 80

  # All pages (slow — ~1120 pages)
  python tools/count_neume_frequencies.py --all

  # Specific pages
  python tools/count_neume_frequencies.py --pages data/pages/Mass/page_0025.png
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

from psaltica_ocr.template_matching import (
    MATCH_THRESHOLD,
    MATCHABLE_GROUPS,
    NMS_IOU_THRESHOLD,
    build_templates,
    load_symbol_map,
    match_template_on_page,
    nms,
)

GROUP_ORDER = [
    "base_neume",
    "rest",
    "modifier_modulation",
    "mode",
    "key_signature",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--pages", nargs="+", type=Path, help="Specific page image paths")
    source.add_argument("--book", help="Single book ID (data/pages/<book>/)")
    source.add_argument("--all", action="store_true", help="Use all non-blank pages from manifest")
    parser.add_argument("--sample", type=int, default=100,
                        help="Pages to sample when no explicit list given (default: 100)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--manifest", type=Path, default=Path("data/pages/manifest.csv"))
    parser.add_argument("--classes", type=Path, default=Path("config/classes.yaml"))
    parser.add_argument("--symbol-map", type=Path, default=Path("config/symbol_map.json"))
    parser.add_argument("--output", type=Path, default=Path("data/neume_frequencies.csv"))
    parser.add_argument("--threshold", type=float, default=MATCH_THRESHOLD)
    return parser.parse_args()


def collect_pages(args: argparse.Namespace) -> list[Path]:
    if args.pages:
        return [p for p in args.pages if p.exists()]

    if args.book:
        candidates = sorted((Path("data/pages") / args.book).glob("page_*.png"))
    else:
        candidates = []
        if args.manifest.exists():
            with args.manifest.open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if float(row.get("ink_ratio", "1") or 1) > 0:
                        p = Path(row["image_path"])
                        if p.exists():
                            candidates.append(p)
        else:
            for book_dir in sorted(Path("data/pages").iterdir()):
                if book_dir.is_dir():
                    candidates.extend(sorted(book_dir.glob("page_*.png")))

    if args.all:
        return candidates

    if len(candidates) <= args.sample:
        return candidates

    rng = random.Random(args.seed)
    return sorted(rng.sample(candidates, args.sample))


def count_page(
    image_path: Path,
    templates: dict[str, list[tuple[float, np.ndarray]]],
    threshold: float,
) -> dict[str, int]:
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {}
    _, img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)

    all_detections: list[tuple[int, int, int, int, float, str]] = []
    for label, variants in templates.items():
        for _pt, tmpl in variants:
            for x, y, w, h, score in match_template_on_page(img, tmpl, threshold):
                all_detections.append((x, y, w, h, score, label))

    kept = nms(all_detections, NMS_IOU_THRESHOLD)
    counts: dict[str, int] = defaultdict(int)
    for *_, label in kept:
        counts[label] += 1
    return dict(counts)


def print_report(
    counts: dict[str, int],
    classes: list[str],
    pages_processed: int,
) -> None:
    total = sum(counts.values())
    if total == 0:
        print("No detections.")
        return

    print(f"\n{'═' * 72}")
    print(f"  Neume frequency report — {pages_processed} pages, {total} total detections")
    print(f"{'═' * 72}")

    for group in GROUP_ORDER:
        group_classes = [c for c in classes if c.split(".", 1)[0] == group and c in counts]
        if not group_classes:
            continue
        group_total = sum(counts[c] for c in group_classes)
        group_pct = group_total / total * 100
        print(f"\n  {group.upper().replace('_', ' ')}  ({group_total:,}  {group_pct:.1f}% of total)")
        print(f"  {'─' * 60}")
        ranked = sorted(group_classes, key=lambda c: counts[c], reverse=True)
        for cls in ranked:
            icon = cls.split(".", 1)[1]
            n = counts[cls]
            per_page = n / pages_processed
            bar = "█" * min(30, int(n / total * 300))
            print(f"  {icon:<30}  {n:>6,}  {per_page:>5.1f}/pg  {bar}")

    # Zero-count classes (not detected)
    undetected = [c for c in classes if c.split(".", 1)[0] in MATCHABLE_GROUPS and c not in counts]
    if undetected:
        print(f"\n  NOT DETECTED ({len(undetected)} classes): "
              + ", ".join(c.split(".", 1)[1] for c in undetected[:10])
              + ("…" if len(undetected) > 10 else ""))

    print(f"\n{'═' * 72}\n")


def write_csv(
    output: Path,
    counts: dict[str, int],
    classes: list[str],
    pages_processed: int,
) -> None:
    total = sum(counts.values())
    rows = []
    for cls in classes:
        group = cls.split(".", 1)[0]
        if group not in MATCHABLE_GROUPS:
            continue
        n = counts.get(cls, 0)
        group_total = sum(counts.get(c, 0) for c in classes if c.split(".", 1)[0] == group)
        rows.append({
            "group": group,
            "class": cls,
            "icon": cls.split(".", 1)[1],
            "count": n,
            "per_page": round(n / pages_processed, 3) if pages_processed else 0,
            "pct_of_group": round(n / group_total * 100, 2) if group_total else 0,
            "pct_of_total": round(n / total * 100, 2) if total else 0,
        })
    rows.sort(key=lambda r: (-r["count"], r["group"], r["class"]))
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["rank", "group", "class", "icon", "count", "per_page",
                        "pct_of_group", "pct_of_total"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    with args.classes.open(encoding="utf-8") as f:
        classes = yaml.safe_load(f)["names"]
    icon_to_insert = load_symbol_map(args.symbol_map)

    print("Rendering templates…")
    templates = build_templates(classes, icon_to_insert)
    matchable = [c for c in classes if c.split(".", 1)[0] in MATCHABLE_GROUPS]
    print(f"  {len(templates)}/{len(matchable)} matchable classes have templates")

    pages = collect_pages(args)
    if not pages:
        raise SystemExit("No pages found.")
    print(f"Processing {len(pages)} pages…")

    totals: dict[str, int] = defaultdict(int)
    for i, page_path in enumerate(pages, 1):
        page_counts = count_page(page_path, templates, args.threshold)
        for label, n in page_counts.items():
            totals[label] += n
        total_so_far = sum(totals.values())
        print(f"  [{i:>{len(str(len(pages)))}}/{len(pages)}] {page_path.name}: "
              f"{sum(page_counts.values())} detections  (running total: {total_so_far:,})")

    print_report(dict(totals), classes, len(pages))
    write_csv(args.output, dict(totals), classes, len(pages))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
