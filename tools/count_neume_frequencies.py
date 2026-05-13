#!/usr/bin/env python3
"""Count per-glyph detection frequency across pages using only the font file.

Reads every glyph directly from PsalticaPraxisUnified.ttf (no dependency on the
app's symbol map or classes.yaml), renders templates, and runs cascade
(largest-first) matching on a sample of rendered pages.

Composite glyphs whose templates are physically wider/taller automatically
suppress their component glyphs in the same region — the largest matching
template wins.

Outputs:
  data/neume_frequencies.csv   — rank, U+XXXX, count, per_page, pct_of_total
  data/neume_frequencies.html  — rendered glyph image + frequency bar per entry

Usage:
  # 100-page random sample (default, ~10–15 min for 675 PUA glyphs)
  python tools/count_neume_frequencies.py

  # Specific pages (fast smoke test)
  python tools/count_neume_frequencies.py --pages data/pages/Mass/page_0025.png

  # Specific book
  python tools/count_neume_frequencies.py --book Mass --sample 50

  # All non-blank pages (slow)
  python tools/count_neume_frequencies.py --all
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

import cv2

from psaltica_ocr.template_matching import (
    FONT_PATH,
    MATCH_THRESHOLD,
    NMS_IOU_THRESHOLD,
    NEUME_CODEPOINT_RANGES,
    build_templates_from_font,
    match_cascade_page,
    render_glyph_b64,
    to_gradient,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    source = parser.add_mutually_exclusive_group()
    source.add_argument("--pages", nargs="+", type=Path)
    source.add_argument("--book", help="Book ID (data/pages/<book>/)")
    source.add_argument("--all", action="store_true", help="All non-blank pages")

    parser.add_argument("--sample", type=int, default=100,
                        help="Pages to sample when no explicit list given (default: 100)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--manifest", type=Path, default=Path("data/pages/manifest.csv"))
    parser.add_argument("--font", type=Path, default=FONT_PATH)
    parser.add_argument("--symbol-map", type=Path, default=Path("config/symbol_map.json"),
                        help="Optional symbol map for icon name labels (default: config/symbol_map.json)")
    parser.add_argument("--threshold", type=float, default=MATCH_THRESHOLD)
    parser.add_argument("--min-area", type=int, default=80,
                        help="Min template pixel area; filters whitespace/tiny marks (default: 80)")
    parser.add_argument("--min-ink", type=float, default=0.04,
                        help="Min ink fraction per template (default: 0.04)")
    parser.add_argument("--max-ink", type=float, default=0.55,
                        help="Max ink fraction; >0.55 = solid fills / thick bars (default: 0.55)")
    parser.add_argument("--min-shape-var", type=float, default=0.08,
                        help="Min row-ink std-dev; <0.08 = horizontal line / rectangle (default: 0.08)")
    parser.add_argument("--min-aspect", type=float, default=0.3,
                        help="Min width/height ratio; <0.3 = vertical slivers (default: 0.3)")
    parser.add_argument("--max-aspect", type=float, default=3.5,
                        help="Max width/height ratio; >3.5 = horizontal bars (default: 3.5)")
    parser.add_argument("--min-ink-rows", type=int, default=4,
                        help="Min rows with ink; <4 = degenerate bar glyphs (default: 4)")
    parser.add_argument("--sizes", type=float, nargs="+",
                        default=[7.0, 8.5, 10.0, 11.5, 13.0],
                        help="Font sizes in pt to render templates at (default: 7 8.5 10 11.5 13)")
    parser.add_argument("--families", type=Path, default=Path("data/neume_families.json"),
                        help="Family map from group_neume_shapes.py; count per family when present")
    parser.add_argument("--output-csv", type=Path, default=Path("data/neume_frequencies.csv"))
    parser.add_argument("--output-html", type=Path, default=Path("data/neume_frequencies.html"))
    parser.add_argument("--no-html", action="store_true")
    parser.add_argument("--gradient", action="store_true", default=True,
                        help="Use Sobel edge matching instead of brightness (default: on)")
    parser.add_argument("--no-gradient", dest="gradient", action="store_false",
                        help="Disable gradient matching, use raw brightness")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Page collection
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Per-page counting
# ---------------------------------------------------------------------------

def count_page(
    image_path: Path,
    templates: dict,
    threshold: float,
    use_gradient: bool = False,
) -> dict[str, int]:
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {}
    _, img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
    if use_gradient:
        img = to_gradient(img)
    kept = match_cascade_page(img, templates, threshold, NMS_IOU_THRESHOLD, score_only=True)
    counts: dict[str, int] = defaultdict(int)
    for *_, label in kept:
        counts[label] += 1
    return dict(counts)


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------

def load_families(families_path: Path) -> list[dict] | None:
    """Load family definitions from group_neume_shapes.py output, or None."""
    if not families_path.exists():
        return None
    import json
    return json.loads(families_path.read_text(encoding="utf-8"))


def apply_families(
    counts: dict[str, int],
    metadata: dict[str, int],
    families: list[dict],
) -> tuple[dict[str, int], dict[str, int]]:
    """Fold per-codepoint counts and metadata into per-family aggregates.

    Returns new counts and metadata keyed by the family representative codepoint.
    The representative is the first member listed in each family.
    """
    # Build codepoint-key → family representative key
    key_to_rep: dict[str, str] = {}
    for fam in families:
        rep = fam["representative"]
        for key in fam["codepoints"]:
            key_to_rep[key] = rep

    new_counts: dict[str, int] = defaultdict(int)
    new_meta: dict[str, int] = {}

    for key, cp in metadata.items():
        rep = key_to_rep.get(key, key)
        new_meta[rep] = int(rep[2:], 16)  # codepoint from "U+XXXX"
        new_counts[rep] += counts.get(key, 0)

    return dict(new_counts), new_meta


def load_icon_map(symbol_map_path: Path) -> dict[int, str]:
    """Return {codepoint: icon_name} from symbol_map.json, or {} if unavailable."""
    if not symbol_map_path.exists():
        return {}
    import json
    sm = json.loads(symbol_map_path.read_text(encoding="utf-8"))
    result: dict[int, str] = {}
    for entry in sm.get("symbols", []):
        icon = entry.get("icon", "")
        for ch in entry.get("insert", ""):
            result.setdefault(ord(ch), icon)
    return result


def print_report(
    counts: dict[str, int],
    metadata: dict[str, int],
    pages_processed: int,
    icon_map: dict[int, str] | None = None,
) -> None:
    total = sum(counts.values())
    if total == 0:
        print("No detections.")
        return

    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    top_n = ranked[:40]

    print(f"\n{'═' * 75}")
    print(f"  Neume frequency — {pages_processed} pages, {total:,} detections, "
          f"{len(counts)} glyphs detected")
    print(f"{'═' * 75}")
    print(f"  {'Codepoint':<12}  {'Icon':<24}  {'Count':>7}  {'Per page':>8}  Freq")
    print(f"  {'─' * 67}")
    for key, n in top_n:
        per_page = n / pages_processed
        bar = "█" * min(20, max(1, int(n / total * 200)))
        cp = metadata.get(key, 0)
        icon = (icon_map or {}).get(cp, "")
        print(f"  {key:<12}  {icon:<24}  {n:>7,}  {per_page:>7.1f}/pg  {bar}")
    if len(ranked) > 40:
        print(f"  … {len(ranked) - 40} more glyphs detected (see HTML/CSV for full list)")
    undetected = len(metadata) - len(counts)
    if undetected:
        print(f"\n  {undetected} glyphs in font not detected on sampled pages")
    print(f"\n{'═' * 75}\n")


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def write_csv(
    output: Path,
    counts: dict[str, int],
    metadata: dict[str, int],
    pages_processed: int,
    icon_map: dict[int, str] | None = None,
) -> None:
    total = sum(counts.values())
    rows = []
    for key, codepoint in metadata.items():
        n = counts.get(key, 0)
        rows.append({
            "rank": 0,
            "codepoint": key,
            "codepoint_dec": codepoint,
            "icon": (icon_map or {}).get(codepoint, ""),
            "count": n,
            "per_page": round(n / pages_processed, 3) if pages_processed else 0,
            "pct_of_total": round(n / total * 100, 3) if total else 0,
        })
    rows.sort(key=lambda r: (-r["count"], r["codepoint_dec"]))
    for i, row in enumerate(rows, 1):
        row["rank"] = i

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["rank", "codepoint", "codepoint_dec", "icon",
                           "count", "per_page", "pct_of_total"],
        )
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def write_html(
    output: Path,
    counts: dict[str, int],
    metadata: dict[str, int],
    font_path: Path,
    pages_processed: int,
    icon_map: dict[int, str] | None = None,
) -> None:
    total = sum(counts.values())
    if total == 0:
        return

    rows_by_count = sorted(
        metadata.items(),
        key=lambda kv: (-counts.get(kv[0], 0), kv[1]),
    )
    max_count = max(counts.values()) if counts else 1

    def bar_html(n: int) -> str:
        pct = int(n / max_count * 100) if max_count else 0
        return (
            f'<div class="bar-wrap">'
            f'<div class="bar" style="width:{pct}%"></div>'
            f'<span class="bar-num">{n:,}</span>'
            f'</div>'
        )

    detected_rows = []
    zero_rows = []

    for key, codepoint in rows_by_count:
        n = counts.get(key, 0)
        char = chr(codepoint)
        per_page = n / pages_processed if pages_processed else 0
        b64 = render_glyph_b64(char, size_px=96, thumb_px=48)
        img_html = (
            f'<img src="data:image/png;base64,{b64}" width="48" height="48" alt="{key}"/>'
            if b64 else '<span class="no-glyph">?</span>'
        )
        icon = (icon_map or {}).get(codepoint, "")
        icon_cell = f'<td class="icon">{icon}</td>' if icon else '<td class="icon" style="color:#ccc">—</td>'
        row_html = (
            f'<tr>'
            f'<td class="glyph">{img_html}</td>'
            f'<td class="cp" title="decimal: {codepoint}">{key}</td>'
            f'{icon_cell}'
            f'<td class="rate">{per_page:.1f}/pg</td>'
            f'<td class="bar-cell">{bar_html(n)}</td>'
            f'</tr>'
        )
        if n > 0:
            detected_rows.append(row_html)
        else:
            zero_rows.append(row_html)

    # Show undetected collapsed by default
    undetected_section = ""
    if zero_rows:
        undetected_section = (
            f'<details><summary style="cursor:pointer;padding:8px 0;color:#888;">'
            f'{len(zero_rows)} glyphs not detected on sampled pages</summary>'
            f'<table class="zero-table">{"".join(zero_rows)}</table>'
            f'</details>'
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>Psaltica — Neume Glyph Frequencies</title>
<style>
  body {{ font-family: sans-serif; background: #f5f5f5; padding: 16px 24px; color: #222; }}
  h1 {{ font-size: 1.3em; margin-bottom: 4px; }}
  .subtitle {{ color: #666; font-size: 0.9em; margin-bottom: 24px; }}
  .card {{ background: white; border-radius: 6px; padding: 16px 20px;
           margin-bottom: 24px; box-shadow: 0 1px 3px rgba(0,0,0,.1); }}
  table {{ border-collapse: collapse; width: 100%; }}
  tr {{ border-bottom: 1px solid #f0f0f0; }}
  tr:last-child {{ border-bottom: none; }}
  td {{ padding: 4px 8px; vertical-align: middle; }}
  td.glyph {{ width: 56px; text-align: center; }}
  td.cp {{ width: 100px; font-size: 0.82em; font-family: monospace; color: #555; }}
  td.icon {{ width: 140px; font-size: 0.82em; color: #333; }}
  td.rate {{ width: 70px; text-align: right; font-size: 0.8em; color: #888; }}
  .bar-wrap {{ display: flex; align-items: center; gap: 8px; }}
  .bar {{ height: 14px; background: #4a90d9; border-radius: 2px; min-width: 2px; }}
  .bar-num {{ font-size: 0.8em; color: #555; white-space: nowrap; }}
  .no-glyph {{ font-size: 1.4em; color: #ccc; }}
  .zero-table tr {{ opacity: 0.4; }}
  img {{ display: block; margin: auto; }}
  details {{ margin-top: 8px; }}
</style>
</head>
<body>
<h1>Psaltica — Neume Glyph Frequency Report</h1>
<p class="subtitle">
  {pages_processed} pages sampled &nbsp;·&nbsp;
  {total:,} total detections &nbsp;·&nbsp;
  {len(counts)}/{len(metadata)} PUA glyphs detected &nbsp;·&nbsp;
  Cascade matching (larger glyphs suppress components)
</p>
<div class="card">
<table>{"".join(detected_rows)}</table>
{undetected_section}
</div>
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
    icon_map = load_icon_map(args.symbol_map)

    print(f"Loading font glyphs from {args.font.name}…")
    templates, metadata = build_templates_from_font(
        font_path=args.font,
        sizes_pt=args.sizes,
        min_area=args.min_area,
        min_ink_frac=args.min_ink,
        max_ink_frac=args.max_ink,
        min_shape_variance=args.min_shape_var,
        min_aspect=args.min_aspect,
        max_aspect=args.max_aspect,
        min_ink_rows=args.min_ink_rows,
        use_gradient=args.gradient,
    )
    mode = "gradient edge" if args.gradient else "brightness"
    print(f"  {len(templates)} glyphs with renderable templates (sizes: {args.sizes} pt, mode: {mode})")

    pages = collect_pages(args)
    if not pages:
        raise SystemExit("No pages found.")
    print(f"Processing {len(pages)} pages…")

    totals: dict[str, int] = defaultdict(int)
    width = len(str(len(pages)))
    for i, page_path in enumerate(pages, 1):
        page_counts = count_page(page_path, templates, args.threshold, use_gradient=args.gradient)
        for label, n in page_counts.items():
            totals[label] += n
        print(f"  [{i:{width}}/{len(pages)}] {page_path.name}: "
              f"{sum(page_counts.values())} detections  "
              f"(running total: {sum(totals.values()):,})")

    counts = dict(totals)

    families = load_families(args.families)
    if families:
        counts, metadata = apply_families(counts, metadata, families)
        print(f"  Folded into {len(metadata)} families (from {args.families.name})")

    print_report(counts, metadata, len(pages), icon_map)

    write_csv(args.output_csv, counts, metadata, len(pages), icon_map)
    print(f"Wrote {args.output_csv}")

    if not args.no_html:
        print("Rendering glyph images for HTML report…")
        write_html(args.output_html, counts, metadata, args.font, len(pages), icon_map)
        print(f"Wrote {args.output_html}")


if __name__ == "__main__":
    main()
