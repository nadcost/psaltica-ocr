#!/usr/bin/env python3
"""Count per-symbol detection frequency across pages to inform keyboard layout design.

Uses cascade (composite-first) template matching: composite glyphs whose templates
are physically larger suppress their individual components in the same region, so
OnePlusOneUp is counted once as OnePlusOneUp rather than as Oligon + Kendima.
If no composite match is found at a location the individual components are still
detected independently.

Outputs:
  data/neume_frequencies.csv   — ranked table (rank, group, class, count, per_page, %)
  data/neume_frequencies.html  — visual report with font glyph images and frequency bars

Usage:
  # 100-page random sample across all books (default, ~2–3 min)
  python tools/count_neume_frequencies.py

  # Specific book
  python tools/count_neume_frequencies.py --book Mass --sample 80

  # All pages
  python tools/count_neume_frequencies.py --all

  # Quick smoke test
  python tools/count_neume_frequencies.py --pages data/pages/Mass/page_0025.png
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

import cv2
import yaml

from psaltica_ocr.template_matching import (
    MATCH_THRESHOLD,
    MATCHABLE_GROUPS,
    NMS_IOU_THRESHOLD,
    build_templates,
    load_symbol_map,
    match_cascade_page,
    render_glyph_b64,
)

GROUP_ORDER = [
    "base_neume",
    "rest",
    "modifier_modulation",
    "mode",
    "key_signature",
]

GROUP_LABELS = {
    "base_neume": "Base Neumes",
    "rest": "Rests",
    "modifier_modulation": "Modulation Markers",
    "mode": "Mode Indicators",
    "key_signature": "Key Signatures",
}


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
    parser.add_argument("--output-csv", type=Path, default=Path("data/neume_frequencies.csv"))
    parser.add_argument("--output-html", type=Path, default=Path("data/neume_frequencies.html"))
    parser.add_argument("--threshold", type=float, default=MATCH_THRESHOLD)
    parser.add_argument("--no-html", action="store_true", help="Skip HTML report generation")
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
    templates: dict,
    threshold: float,
) -> dict[str, int]:
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {}
    _, img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)

    kept = match_cascade_page(img, templates, threshold, NMS_IOU_THRESHOLD)
    counts: dict[str, int] = defaultdict(int)
    for *_, label in kept:
        counts[label] += 1
    return dict(counts)


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------

def print_report(counts: dict[str, int], classes: list[str], pages_processed: int) -> None:
    total = sum(counts.values())
    if total == 0:
        print("No detections.")
        return

    print(f"\n{'═' * 72}")
    print(f"  Neume frequency report — {pages_processed} pages, {total:,} total detections")
    print(f"{'═' * 72}")

    for group in GROUP_ORDER:
        group_classes = [c for c in classes if c.split(".", 1)[0] == group and c in counts]
        if not group_classes:
            continue
        group_total = sum(counts[c] for c in group_classes)
        group_pct = group_total / total * 100
        label = GROUP_LABELS.get(group, group)
        print(f"\n  {label.upper()}  ({group_total:,}  {group_pct:.1f}% of total)")
        print(f"  {'─' * 60}")
        ranked = sorted(group_classes, key=lambda c: counts[c], reverse=True)
        for cls in ranked:
            icon = cls.split(".", 1)[1]
            n = counts[cls]
            per_page = n / pages_processed
            bar = "█" * min(30, max(1, int(n / total * 300)))
            print(f"  {icon:<32}  {n:>6,}  {per_page:>6.1f}/pg  {bar}")

    undetected = [
        c for c in classes
        if c.split(".", 1)[0] in MATCHABLE_GROUPS and c not in counts
    ]
    if undetected:
        names = ", ".join(c.split(".", 1)[1] for c in undetected[:12])
        suffix = "…" if len(undetected) > 12 else ""
        print(f"\n  NOT DETECTED ({len(undetected)} classes): {names}{suffix}")

    print(f"\n{'═' * 72}\n")


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

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
            fieldnames=["rank", "group", "class", "icon", "count",
                        "per_page", "pct_of_group", "pct_of_total"],
        )
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def write_html(
    output: Path,
    counts: dict[str, int],
    classes: list[str],
    icon_to_insert: dict[str, str],
    pages_processed: int,
) -> None:
    total = sum(counts.values())
    if total == 0:
        return

    def bar_html(n: int, group_max: int) -> str:
        pct = int(n / group_max * 100) if group_max else 0
        return (
            f'<div class="bar-wrap">'
            f'<div class="bar" style="width:{pct}%"></div>'
            f'<span class="bar-num">{n:,}</span>'
            f'</div>'
        )

    sections = []
    for group in GROUP_ORDER:
        group_label = GROUP_LABELS.get(group, group)
        group_classes = [c for c in classes if c.split(".", 1)[0] == group]
        group_total = sum(counts.get(c, 0) for c in group_classes)
        if group_total == 0:
            continue
        group_max = max(counts.get(c, 0) for c in group_classes)
        group_pct = group_total / total * 100

        rows_html = []
        ranked = sorted(group_classes, key=lambda c: counts.get(c, 0), reverse=True)
        for cls in ranked:
            icon = cls.split(".", 1)[1]
            insert = icon_to_insert.get(icon, "")
            n = counts.get(cls, 0)
            per_page = n / pages_processed if pages_processed else 0
            b64 = render_glyph_b64(insert) if insert else ""
            img_html = (
                f'<img src="data:image/png;base64,{b64}" width="48" height="48" alt="{icon}"/>'
                if b64 else '<span class="no-glyph">?</span>'
            )
            row_class = "zero" if n == 0 else ""
            rows_html.append(
                f'<tr class="{row_class}">'
                f'<td class="glyph">{img_html}</td>'
                f'<td class="name" title="{cls}">{icon}</td>'
                f'<td class="rate">{per_page:.1f}/pg</td>'
                f'<td class="bar-cell">{bar_html(n, group_max)}</td>'
                f'</tr>'
            )

        sections.append(
            f'<section>'
            f'<h2>{group_label}'
            f' <span class="group-total">{group_total:,} detections'
            f' &nbsp;·&nbsp; {group_pct:.1f}% of total</span></h2>'
            f'<table>{"".join(rows_html)}</table>'
            f'</section>'
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>Psaltica OCR — Neume Frequencies</title>
<style>
  body {{ font-family: sans-serif; background: #f5f5f5; padding: 16px 24px; color: #222; }}
  h1 {{ font-size: 1.3em; margin-bottom: 4px; }}
  .subtitle {{ color: #666; font-size: 0.9em; margin-bottom: 32px; }}
  section {{ background: white; border-radius: 6px; padding: 16px 20px;
             margin-bottom: 24px; box-shadow: 0 1px 3px rgba(0,0,0,.1); }}
  h2 {{ font-size: 1em; margin: 0 0 12px 0; border-bottom: 2px solid #e0e0e0;
        padding-bottom: 6px; }}
  .group-total {{ font-weight: normal; color: #888; font-size: 0.85em; }}
  table {{ border-collapse: collapse; width: 100%; }}
  tr {{ border-bottom: 1px solid #f0f0f0; }}
  tr:last-child {{ border-bottom: none; }}
  tr.zero {{ opacity: 0.35; }}
  td {{ padding: 4px 8px; vertical-align: middle; }}
  td.glyph {{ width: 56px; text-align: center; }}
  td.name {{ width: 220px; font-size: 0.82em; font-family: monospace; color: #444; }}
  td.rate {{ width: 70px; text-align: right; font-size: 0.8em; color: #888; }}
  td.bar-cell {{ }}
  .bar-wrap {{ display: flex; align-items: center; gap: 8px; }}
  .bar {{ height: 16px; background: #4a90d9; border-radius: 2px; min-width: 2px; }}
  .bar-num {{ font-size: 0.8em; color: #555; white-space: nowrap; }}
  .no-glyph {{ font-size: 1.4em; color: #ccc; }}
  img {{ display: block; margin: auto; }}
</style>
</head>
<body>
<h1>Psaltica OCR — Neume Frequency Report</h1>
<p class="subtitle">{pages_processed} pages sampled &nbsp;·&nbsp;
{total:,} total detections &nbsp;·&nbsp;
Cascade matching (composite glyphs suppress components)</p>
{"".join(sections)}
</body>
</html>
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
    print(f"Processing {len(pages)} pages (cascade matching)…")

    totals: dict[str, int] = defaultdict(int)
    width = len(str(len(pages)))
    for i, page_path in enumerate(pages, 1):
        page_counts = count_page(page_path, templates, args.threshold)
        for label, n in page_counts.items():
            totals[label] += n
        print(f"  [{i:{width}}/{len(pages)}] {page_path.name}: "
              f"{sum(page_counts.values())} detections  "
              f"(running total: {sum(totals.values()):,})")

    counts = dict(totals)
    print_report(counts, classes, len(pages))

    write_csv(args.output_csv, counts, classes, len(pages))
    print(f"Wrote {args.output_csv}")

    if not args.no_html:
        write_html(args.output_html, counts, classes, icon_to_insert, len(pages))
        print(f"Wrote {args.output_html}")


if __name__ == "__main__":
    main()
