#!/usr/bin/env python3
"""Group same-shape font glyphs, then match each shape group on page images.

The grouping step normalizes each glyph by tight-cropping ink, scaling it to a
fixed canvas, and centering it. That removes glyph-bearing x/y differences, so
characters that draw the same mark in different attachment positions can share
one shape group.

Outputs:
  data/font_shape_groups.json
  data/annotations/font_shape_matches.json
  data/annotations/font_shape_matches.csv
  data/annotations/font_shape_matches.html
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from psaltica_ocr.font_shape_matching import (
    DEFAULT_ICON_THRESHOLDS,
    DEFAULT_SHAPE_FAMILY_ALIASES_PATH,
    build_glyph_shapes,
    build_group_templates,
    group_icon_names,
    group_priorities_from_icons,
    group_similar_shapes,
    group_thresholds_from_icons,
    groups_to_jsonable,
    load_shape_family_aliases,
    load_icon_map,
    match_shape_groups_on_page,
    merge_shape_group_aliases,
    normalize_codepoint_key,
    parse_codepoint_ranges,
    write_detections_csv,
    write_match_report_html,
)
from psaltica_ocr.template_matching import DPI, FONT_PATH, NMS_IOU_THRESHOLD


DEFAULT_SIZES_PT = [7.0, 8.5, 10.0, 11.5, 13.0]
DEFAULT_MATCH_THRESHOLD = 0.75


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    page_group = parser.add_mutually_exclusive_group(required=True)
    page_group.add_argument("--pages", nargs="+", type=Path, help="Rendered page image paths")
    page_group.add_argument("--book", help="Book ID under data/pages/<book>/")
    parser.add_argument("--pages-per-book", type=int, default=0, help="Limit pages for --book; 0 = all")
    parser.add_argument("--font", type=Path, default=FONT_PATH)
    parser.add_argument("--symbol-map", type=Path, default=Path("config/symbol_map.json"))
    parser.add_argument(
        "--codepoint-range",
        action="append",
        help="Inclusive hex range, e.g. E0D0-E127. Repeatable. Default: Psaltica neume ranges.",
    )
    parser.add_argument("--shape-threshold", type=float, default=0.86,
                        help="Similarity threshold for grouping same-shape glyphs")
    parser.add_argument("--family-alias", action="append", default=[],
                        help="Comma-separated codepoints to force into one family; first is representative.")
    parser.add_argument("--shape-family-aliases", type=Path, default=DEFAULT_SHAPE_FAMILY_ALIASES_PATH,
                        help="YAML file of default visual-family aliases.")
    parser.add_argument("--no-default-family-aliases", action="store_true",
                        help="Disable configured aliases for decorated composite variants.")
    parser.add_argument("--match-threshold", type=float, default=DEFAULT_MATCH_THRESHOLD,
                        help="Template-match score threshold")
    parser.add_argument("--icon-threshold", action="append", default=[],
                        help="Per-app-name threshold override, e.g. Oligon=0.88. Repeatable.")
    parser.add_argument("--no-icon-thresholds", action="store_true",
                        help="Disable built-in per-app thresholds for Apostrofos, Isson2, and Oligon.")
    parser.add_argument("--nms-iou", type=float, default=NMS_IOU_THRESHOLD,
                        help="NMS IoU threshold")
    parser.add_argument("--score-only-nms", action="store_true",
                        help="Sort NMS by score only instead of app-name priority.")
    parser.add_argument("--no-complex-first", action="store_true",
                        help="Disable larger-template-first cascade NMS.")
    parser.add_argument("--no-icon-size-filters", action="store_true",
                        help="Disable built-in app-name template-size filters, currently used for Oligon.")
    parser.add_argument("--canvas", type=int, default=48,
                        help="Normalized shape canvas size")
    parser.add_argument("--render-px", type=int, default=128,
                        help="Font pixel size used for shape grouping")
    parser.add_argument("--sizes", type=float, nargs="+", default=DEFAULT_SIZES_PT,
                        help="Font point sizes to render for page matching")
    parser.add_argument("--dpi", type=int, default=DPI)
    parser.add_argument("--groups-json", type=Path, default=Path("data/font_shape_groups.json"))
    parser.add_argument("--output-json", type=Path, default=Path("data/annotations/font_shape_matches.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("data/annotations/font_shape_matches.csv"))
    parser.add_argument("--output-html", type=Path, default=Path("data/annotations/font_shape_matches.html"))
    return parser.parse_args()


def parse_icon_thresholds(values: list[str]) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"Invalid --icon-threshold value {value!r}; expected Name=0.75")
        name, threshold = value.split("=", 1)
        thresholds[name] = float(threshold)
    return thresholds


def parse_family_aliases(values: list[str]) -> list[tuple[str, ...]]:
    aliases: list[tuple[str, ...]] = []
    for value in values:
        members = []
        for raw_member in value.split(","):
            text = raw_member.strip().upper()
            if not text:
                continue
            members.append(normalize_codepoint_key(text))
        if len(members) < 2:
            raise SystemExit(f"Invalid --family-alias value {value!r}; expected at least two codepoints")
        aliases.append(tuple(members))
    return aliases


def apply_icon_size_filters(
    templates: dict[str, list[tuple[float, object]]],
    names_by_group: dict[str, list[str]],
    *,
    max_size: float,
) -> None:
    for group_id, names in names_by_group.items():
        if "Oligon" not in names or group_id not in templates:
            continue
        largest = [variant for variant in templates[group_id] if variant[0] == max_size]
        if largest:
            templates[group_id] = largest


def collect_pages(args: argparse.Namespace) -> list[Path]:
    if args.pages:
        return [page for page in args.pages if page.exists()]
    book_dir = Path("data/pages") / args.book
    pages = sorted(book_dir.glob("page_*.png"))
    if args.pages_per_book:
        pages = pages[: args.pages_per_book]
    return pages


def detection_to_json(detection, members: tuple[str, ...]) -> dict:
    return {
        "groupId": detection.group_id,
        "representative": detection.representative,
        "members": list(members),
        "bbox": [detection.x, detection.y, detection.width, detection.height],
        "score": round(detection.score, 4),
        "sizePt": detection.size_pt,
    }


def main() -> None:
    args = parse_args()
    pages = collect_pages(args)
    if not pages:
        raise SystemExit("No page images found. Render PDFs first with tools/render_pdfs.py or pass --pages.")

    ranges = parse_codepoint_ranges(args.codepoint_range)
    print(f"Loading glyphs from {args.font}")
    shapes, key_to_codepoint = build_glyph_shapes(
        args.font,
        codepoint_ranges=ranges,
        canvas=args.canvas,
        render_px=args.render_px,
    )
    if not shapes:
        raise SystemExit("No renderable glyphs found for the requested codepoint ranges.")

    print(f"Grouping {len(shapes)} glyphs by shape at threshold {args.shape_threshold}")
    shape_groups = group_similar_shapes(shapes, threshold=args.shape_threshold)
    aliases = [] if args.no_default_family_aliases else list(load_shape_family_aliases(args.shape_family_aliases))
    aliases.extend(parse_family_aliases(args.family_alias))
    shape_groups = merge_shape_group_aliases(shape_groups, aliases)
    multi_member = sum(1 for group in shape_groups if len(group.members) > 1)
    print(f"  {len(shape_groups)} shape groups; {multi_member} groups contain multiple chars")

    icon_map = load_icon_map(args.symbol_map)
    groups_payload = {
        "font": str(args.font),
        "shapeThreshold": args.shape_threshold,
        "canvas": args.canvas,
        "renderPx": args.render_px,
        "codepointRanges": [[start, end] for start, end in ranges],
        "groups": groups_to_jsonable(shape_groups, shapes, icon_map=icon_map),
    }
    args.groups_json.parent.mkdir(parents=True, exist_ok=True)
    args.groups_json.write_text(json.dumps(groups_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote groups -> {args.groups_json}")

    names_by_group = group_icon_names(shape_groups, key_to_codepoint, icon_map)
    print(f"Rendering representative templates for sizes: {' '.join(str(size) for size in args.sizes)} pt")
    templates = build_group_templates(shape_groups, key_to_codepoint, args.font, sizes_pt=args.sizes, dpi=args.dpi)
    if not args.no_icon_size_filters:
        apply_icon_size_filters(templates, names_by_group, max_size=max(args.sizes))
    print(f"  {len(templates)}/{len(shape_groups)} groups have matchable templates")

    group_members = {group.id: group.members for group in shape_groups}
    icon_thresholds = {} if args.no_icon_thresholds else dict(DEFAULT_ICON_THRESHOLDS)
    icon_thresholds.update(parse_icon_thresholds(args.icon_threshold))
    thresholds = group_thresholds_from_icons(
        names_by_group,
        default_threshold=args.match_threshold,
        icon_thresholds=icon_thresholds,
    )
    priorities = {} if args.score_only_nms else group_priorities_from_icons(names_by_group)
    page_payloads = []
    for index, page in enumerate(pages, 1):
        image = cv2.imread(str(page), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"  [{index}/{len(pages)}] {page}: skipped (unreadable)")
            continue
        detections = match_shape_groups_on_page(
            image,
            shape_groups,
            templates,
            threshold=args.match_threshold,
            iou_threshold=args.nms_iou,
            thresholds=thresholds,
            priorities=priorities,
            complex_first=not args.no_complex_first,
        )
        page_payload = {
            "image": str(page),
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
            "detections": [
                detection_to_json(detection, group_members[detection.group_id])
                for detection in detections
            ],
        }
        page_payloads.append(page_payload)
        print(f"  [{index}/{len(pages)}] {page.name}: {len(detections)} detections")

    output_payload = {
        "font": str(args.font),
        "groupsJson": str(args.groups_json),
        "shapeThreshold": args.shape_threshold,
        "matchThreshold": args.match_threshold,
        "nmsIou": args.nms_iou,
        "sizesPt": args.sizes,
        "pages": page_payloads,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_detections_csv(args.output_csv, page_payloads)
    write_match_report_html(
        args.output_html,
        font_path=args.font,
        groups_payload=groups_payload,
        pages=page_payloads,
        match_threshold=args.match_threshold,
        shape_threshold=args.shape_threshold,
    )
    print(f"Wrote matches -> {args.output_json}")
    print(f"Wrote CSV -> {args.output_csv}")
    print(f"Wrote HTML -> {args.output_html}")


if __name__ == "__main__":
    main()
