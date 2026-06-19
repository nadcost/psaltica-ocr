#!/usr/bin/env python3
"""Per-class instance counts over saved review-UI corrections.

Shows which classes are saturated (let the detector handle them / stop seeking
them out) vs starved (choose pages for them, or generate synthetically) — the
data-driven version of "how many pages should I annotate?". Counts ground-truth
boxes in data/corrections/*/detections.yolo, mapped through config/classes.yaml.

    uv run python tools/count_annotation_classes.py
    uv run python tools/count_annotation_classes.py --min 50 --all
    uv run python tools/count_annotation_classes.py --csv data/annotations/class_counts.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from review_ui.review_io import count_class_instances, group_of, load_class_names

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORRECTIONS = REPO_ROOT / "data/corrections"
DEFAULT_CLASSES = REPO_ROOT / "config/classes.yaml"
DEFAULT_SYMBOL_MAP = REPO_ROOT / "config/symbol_map.json"


def _inserts() -> dict[str, str]:
    from psaltica_ocr.template_matching import load_symbol_map

    return load_symbol_map(DEFAULT_SYMBOL_MAP) if DEFAULT_SYMBOL_MAP.exists() else {}


def write_synthetic_targets(path: Path, class_names: list[str], counts: dict[str, int], limit: int) -> int:
    """Composition-ready list of the rarest glyphs to put on synthetic pages.

    Includes every class with fewer than ``limit`` real instances (zeros first),
    with the icon name + insert string so they can be composed in the app.
    """
    inserts = _inserts()
    rows = sorted((n for n in class_names if counts[n] < limit), key=lambda n: (counts[n], n))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["count", "group", "class", "icon", "insert", "insert_unicode"])
        for n in rows:
            icon = n.split(".", 1)[1]
            insert = inserts.get(icon, "")
            uni = "".join(f"U+{ord(c):04X}" for c in insert) if insert else ""
            writer.writerow([counts[n], group_of(n), n, icon, insert, uni])
    return len(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corrections", type=Path, default=DEFAULT_CORRECTIONS)
    parser.add_argument("--classes", type=Path, default=DEFAULT_CLASSES)
    parser.add_argument("--min", type=int, default=50,
                        help="Instances below this count a class as 'starved' (default 50).")
    parser.add_argument("--all", action="store_true", help="List every class, not just top + starved.")
    parser.add_argument("--csv", type=Path, default=None, help="Also write the full table to this CSV.")
    parser.add_argument("--targets-out", type=Path, default=None,
                        help="Write the rare-glyph list (for composing synthetic pages) to this CSV.")
    parser.add_argument("--targets-min", type=int, default=10,
                        help="Glyphs below this count go on the synthetic-targets list (default 10).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    class_names = load_class_names(args.classes)
    counts, coverage, out_of_range = count_class_instances(args.corrections, class_names)

    pages = sorted(p.name for p in args.corrections.glob("*") if (p / "detections.yolo").exists())
    total = sum(counts.values())
    used = [n for n in class_names if counts[n] > 0]
    zero = [n for n in class_names if counts[n] == 0]
    starved = sorted((n for n in used if counts[n] < args.min), key=lambda n: counts[n])

    print(f"Pages annotated : {len(pages)}")
    print(f"Total boxes     : {total}")
    print(f"Classes covered : {len(used)} / {len(class_names)}  ({len(zero)} never annotated)")
    if out_of_range:
        print(f"WARNING         : {out_of_range} labels point to no class (stale indices?)")

    # Per-group instance totals.
    group_tot: dict[str, int] = defaultdict(int)
    for n in class_names:
        group_tot[group_of(n)] += counts[n]
    print("\nBy group:")
    for g, c in sorted(group_tot.items(), key=lambda kv: -kv[1]):
        print(f"  {g:22s} {c:6d}")

    def table(title: str, rows: list[str]) -> None:
        print(f"\n{title} ({len(rows)}):")
        print(f"  {'count':>6} {'pages':>5}  class")
        for n in rows:
            print(f"  {counts[n]:6d} {coverage[n]:5d}  {n}")

    if args.all:
        table("All classes (by count)", sorted(class_names, key=lambda n: (-counts[n], n)))
    else:
        table("Top 20 (saturated — let the model predict these)",
              sorted(used, key=lambda n: -counts[n])[:20])
        table(f"Starved (< {args.min} — choose pages for these or synthesize)", starved)
        print(f"\nNever annotated: {len(zero)} classes "
              f"(run with --all to list them).")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["class", "group", "count", "pages"])
            for n in sorted(class_names, key=lambda n: (-counts[n], n)):
                writer.writerow([n, group_of(n), counts[n], coverage[n]])
        print(f"\nWrote {args.csv}")

    if args.targets_out:
        n = write_synthetic_targets(args.targets_out, class_names, counts, args.targets_min)
        print(f"\nWrote {n} synthetic targets (count < {args.targets_min}) -> {args.targets_out}")


if __name__ == "__main__":
    main()
