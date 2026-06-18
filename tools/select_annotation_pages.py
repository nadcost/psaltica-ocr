#!/usr/bin/env python3
"""Propose a coverage-maximising 50-page annotation set for psaltica-ocr-mgr.

Selecting 50 near-identical full-music pages would train a detector blind to
rare glyphs, scripts, and layouts. This tool ranks the whole corpus for
diversity and greedily picks a set that maximises class + script + layout +
book/section coverage, emitting a candidate list for human confirm/swap.

Three stages (intermediate results are cached so reruns are cheap):

* Stage A (fast, all pages) — structural features from layout segmentation and
  the manifest: chant-row / lyric-row / non-score counts, glyph-size variety,
  ink ratio. Blanks and prose pages are filtered out.
* Stage B (shortlist) — a cheap half-resolution, 2-size template probe for the
  per-page matchable-class set, and a lyric-row script probe (Greek/Latin/
  Arabic). Skipped with --no-coverage or when tesseract is unavailable.
* Stage C — greedy selection that, at each step, takes the page adding the most
  new classes, plus bonuses for under-covered scripts, layout types, and
  book/section buckets.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml

from psaltica_ocr.layout_segmentation import segment_page_layout
from psaltica_ocr.rendering import binarize


@dataclass
class PageFeatures:
    book: str
    page_number: int
    image_path: str
    ink_ratio: float
    width: int
    height: int
    chant_rows: int = 0
    lyric_rows: int = 0
    unpaired_lyrics: int = 0
    non_score: int = 0
    components: int = 0
    height_variety: float = 0.0
    # Stage B (may stay empty):
    classes: list[str] = field(default_factory=list)
    scripts: list[str] = field(default_factory=list)

    @property
    def is_candidate(self) -> bool:
        # Drop blanks and prose: a chant page has at least one chant row and ink.
        return self.ink_ratio >= 0.02 and self.chant_rows >= 1

    @property
    def layout_type(self) -> str:
        if self.chant_rows >= 3:
            return "dense_multi_row"
        if self.chant_rows == 2:
            return "multi_row"
        return "single_row"

    @property
    def section(self) -> str:
        # Front/middle/back third by page number — sections often change font.
        return f"{self.book}:{'front' if self.page_number < 200 else 'mid' if self.page_number < 600 else 'back'}"

    @property
    def structural_score(self) -> float:
        return (
            min(self.chant_rows, 4) * 2.0
            + min(self.lyric_rows, 4) * 1.5
            + min(self.components, 400) / 100.0
            + self.height_variety / 10.0
            + (1.0 if self.non_score else 0.0)
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/pages_full/manifest.csv"))
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--shortlist", type=int, default=140)
    parser.add_argument("--classes", type=Path, default=Path("config/classes.yaml"))
    parser.add_argument("--symbol-map", type=Path, default=Path("config/symbol_map.json"))
    parser.add_argument("--features-cache", type=Path, default=Path("data/annotation_features.json"))
    parser.add_argument("--probe-cache", type=Path, default=Path("data/annotation_probe.json"))
    parser.add_argument("--output", type=Path, default=Path("data/annotation_candidates"))
    parser.add_argument("--no-coverage", action="store_true", help="Skip the slow template/script probe")
    parser.add_argument("--refresh", action="store_true", help="Recompute caches")
    parser.add_argument("--limit", type=int, default=0, help="Process only the first N manifest rows (debug)")
    return parser.parse_args()


def load_manifest(path: Path, limit: int) -> list[PageFeatures]:
    rows: list[PageFeatures] = []
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                PageFeatures(
                    book=row["book_id"],
                    page_number=int(row["page_number"]),
                    image_path=row["image_path"],
                    ink_ratio=float(row.get("ink_ratio") or 0.0),
                    width=int(row.get("width") or 0),
                    height=int(row.get("height") or 0),
                )
            )
            if limit and len(rows) >= limit:
                break
    return rows


def compute_structural(page: PageFeatures) -> None:
    image = cv2.imread(page.image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return
    layout = segment_page_layout(image, notation_direction="ltr")
    page.chant_rows = len(layout.chant_rows)
    page.lyric_rows = sum(len(row.lyric_rows) for row in layout.chant_rows)
    page.unpaired_lyrics = len(layout.unpaired_lyric_rows)
    page.non_score = len(layout.non_score_regions)

    binary = binarize(image)
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    heights = [int(stats[label, cv2.CC_STAT_HEIGHT]) for label in range(1, count) if stats[label, cv2.CC_STAT_AREA] >= 3]
    page.components = len(heights)
    page.height_variety = float(np.std(heights)) if heights else 0.0


def run_stage_a(pages: list[PageFeatures], cache: Path, refresh: bool) -> None:
    if cache.exists() and not refresh:
        cached = {row["image_path"]: row for row in json.loads(cache.read_text())}
        for page in pages:
            data = cached.get(page.image_path)
            if data:
                for key in ("chant_rows", "lyric_rows", "unpaired_lyrics", "non_score", "components", "height_variety"):
                    setattr(page, key, data[key])
        if all(page.components or page.chant_rows for page in pages):
            print(f"Stage A: loaded {len(pages)} pages from cache")
            return

    for index, page in enumerate(pages, 1):
        compute_structural(page)
        if index % 100 == 0:
            print(f"Stage A: {index}/{len(pages)} pages")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps([asdict(p) for p in pages], ensure_ascii=False, indent=1))
    print(f"Stage A: wrote {cache}")


def spread_shortlist(candidates: list[PageFeatures], size: int) -> list[PageFeatures]:
    """Top by structural score, but capped per section so no part dominates."""
    by_score = sorted(candidates, key=lambda p: p.structural_score, reverse=True)
    per_section_cap = max(8, size // 6)
    counts: Counter[str] = Counter()
    shortlist: list[PageFeatures] = []
    for page in by_score:
        if counts[page.section] >= per_section_cap:
            continue
        shortlist.append(page)
        counts[page.section] += 1
        if len(shortlist) >= size:
            break
    return shortlist


def run_stage_b(shortlist: list[PageFeatures], args: argparse.Namespace) -> None:
    cache_path = args.probe_cache
    if cache_path.exists() and not args.refresh:
        cached = {row["image_path"]: row for row in json.loads(cache_path.read_text())}
        for page in shortlist:
            data = cached.get(page.image_path)
            if data:
                page.classes = data.get("classes", [])
                page.scripts = data.get("scripts", [])
        if all(page.classes or page.scripts for page in shortlist):
            print(f"Stage B: loaded {len(shortlist)} probes from cache")
            return

    from psaltica_ocr.template_matching import (
        NMS_IOU_THRESHOLD,
        build_templates,
        load_symbol_map,
        match_cascade_page,
    )

    classes = yaml.safe_load(args.classes.read_text())["names"]
    templates = build_templates(classes, load_symbol_map(args.symbol_map), sizes_pt=[7.5, 9.0])
    script_probe = _make_script_probe()

    for index, page in enumerate(shortlist, 1):
        image = cv2.imread(page.image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        small = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        _, binary = cv2.threshold(small, 200, 255, cv2.THRESH_BINARY)
        kept = match_cascade_page(binary, templates, 0.6, NMS_IOU_THRESHOLD)
        page.classes = sorted({label for *_, label in kept})
        if script_probe is not None:
            page.scripts = script_probe(image, page)
        print(f"Stage B: [{index}/{len(shortlist)}] {Path(page.image_path).name}: {len(page.classes)} classes, {page.scripts}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            [{"image_path": p.image_path, "classes": p.classes, "scripts": p.scripts} for p in shortlist],
            ensure_ascii=False,
            indent=1,
        )
    )
    print(f"Stage B: wrote {cache_path}")


def _make_script_probe():
    try:
        from psaltica_ocr.lyric_ocr import LyricOcr, TesseractBackend

        adapter = LyricOcr(TesseractBackend())
    except Exception as exc:  # noqa: BLE001 - tesseract optional
        print(f"Stage B: script probe disabled ({exc})")
        return None

    def probe(image: np.ndarray, page: PageFeatures) -> list[str]:
        # Approximate: the script tag uses the no-hint OCR path, which conflates
        # Greek and Latin (the eng pack transliterates Greek with comparable
        # confidence), and only the first few rows are sampled. Arabic-vs-not is
        # the dependable signal; treat "Latin" on this Greek-Arabic corpus as
        # Greek. Good enough to bias selection toward script variety.
        layout = segment_page_layout(image, notation_direction="ltr")
        rows = [row for chant in layout.chant_rows for row in chant.lyric_rows][:3]
        scripts: set[str] = set()
        for region in rows:
            line = adapter.recognize_line(image, region.bbox)
            if line.script in ("Greek", "Latin", "Arabic"):
                scripts.add(line.script)
        return sorted(scripts)

    return probe


def greedy_select(candidates: list[PageFeatures], count: int) -> list[tuple[PageFeatures, int]]:
    covered_classes: set[str] = set()
    script_counts: Counter[str] = Counter()
    layout_counts: Counter[str] = Counter()
    section_counts: Counter[str] = Counter()
    remaining = list(candidates)
    selected: list[tuple[PageFeatures, int]] = []

    def gain(page: PageFeatures) -> tuple[float, int]:
        new_classes = len(set(page.classes) - covered_classes)
        script_bonus = sum(3.0 for s in page.scripts if script_counts[s] < 6)
        bilingual_bonus = 4.0 if len(page.scripts) >= 2 else 0.0
        layout_bonus = 3.0 if layout_counts[page.layout_type] < max(6, count // 6) else 0.0
        section_bonus = 2.0 if section_counts[page.section] < max(4, count // 8) else 0.0
        score = new_classes * 2.0 + script_bonus + bilingual_bonus + layout_bonus + section_bonus
        return score, new_classes

    while remaining and len(selected) < count:
        best = max(remaining, key=lambda p: gain(p)[0])
        _, new_classes = gain(best)
        remaining.remove(best)
        covered_classes.update(best.classes)
        for script in best.scripts:
            script_counts[script] += 1
        layout_counts[best.layout_type] += 1
        section_counts[best.section] += 1
        selected.append((best, len(covered_classes)))

    return selected


def write_output(selected: list[tuple[PageFeatures, int]], shortlist: list[PageFeatures], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["rank", "book", "page", "image_path", "scripts", "layout_type", "chant_rows", "lyric_rows", "n_classes", "cumulative_classes"]
        )
        for rank, (page, cumulative) in enumerate(selected, 1):
            writer.writerow(
                [rank, page.book, page.page_number, page.image_path, "+".join(page.scripts) or "?",
                 page.layout_type, page.chant_rows, page.lyric_rows, len(page.classes), cumulative]
            )
    json_path = output.with_suffix(".json")
    json_path.write_text(
        json.dumps([asdict(page) for page, _ in selected], ensure_ascii=False, indent=1), encoding="utf-8"
    )

    all_shortlist_classes = {c for p in shortlist for c in p.classes}
    chosen_classes = {c for page, _ in selected for c in page.classes}
    print(f"\nSelected {len(selected)} pages -> {csv_path}")
    print(f"class coverage: {len(chosen_classes)} (shortlist had {len(all_shortlist_classes)} distinct)")
    print(f"scripts: {dict(Counter(s for page, _ in selected for s in page.scripts))}")
    print(f"bilingual+ pages: {sum(1 for page, _ in selected if len(page.scripts) >= 2)}")
    print(f"layouts: {dict(Counter(page.layout_type for page, _ in selected))}")
    print(f"books: {dict(Counter(page.book for page, _ in selected))}")


def main() -> None:
    args = parse_args()
    pages = load_manifest(args.manifest, args.limit)
    print(f"manifest: {len(pages)} pages")
    run_stage_a(pages, args.features_cache, args.refresh)

    candidates = [page for page in pages if page.is_candidate]
    print(f"candidates after blank/prose filter: {len(candidates)}")
    shortlist = spread_shortlist(candidates, args.shortlist)
    print(f"shortlist: {len(shortlist)} pages")

    if not args.no_coverage:
        run_stage_b(shortlist, args)

    selected = greedy_select(shortlist, args.count)
    write_output(selected, shortlist, args.output)


if __name__ == "__main__":
    main()
