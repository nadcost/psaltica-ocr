#!/usr/bin/env python3
"""Propose review-only aliases for glyphs that look like decorated variants.

This does not change matcher behavior. It writes candidate YAML for humans to
review, then accepted families can be added to config/shape_family_aliases.yaml.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from psaltica_ocr.font_shape_matching import (
    DEFAULT_SHAPE_FAMILY_ALIASES_PATH,
    GlyphShape,
    ShapeGroup,
    build_glyph_shapes,
    group_icon_names,
    group_similar_shapes,
    load_icon_map,
    load_shape_family_aliases,
    merge_shape_group_aliases,
    parse_codepoint_ranges,
    shape_similarity,
)
from psaltica_ocr.template_matching import FONT_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font", type=Path, default=FONT_PATH)
    parser.add_argument("--symbol-map", type=Path, default=Path("config/symbol_map.json"))
    parser.add_argument("--shape-family-aliases", type=Path, default=DEFAULT_SHAPE_FAMILY_ALIASES_PATH)
    parser.add_argument(
        "--codepoint-range",
        action="append",
        help="Inclusive hex range, e.g. E0D0-E127. Repeatable. Default: Psaltica neume ranges.",
    )
    parser.add_argument("--shape-threshold", type=float, default=0.86)
    parser.add_argument("--no-mirror-family-grouping", action="store_true",
                        help="Do not group horizontally flipped glyphs into the same visual family.")
    parser.add_argument("--min-similarity", type=float, default=0.45)
    parser.add_argument("--canvas", type=int, default=48)
    parser.add_argument("--render-px", type=int, default=128)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("data/proposed_shape_family_aliases.yaml"))
    return parser.parse_args()


def _center_band(image: np.ndarray, *, keep_fraction: float = 0.62) -> np.ndarray:
    margin = max(0, round(image.shape[0] * (1.0 - keep_fraction) / 2.0))
    if margin == 0:
        return image
    return image[margin:-margin, :]


def _ink_fraction(image: np.ndarray) -> float:
    return float(np.mean(image < 128))


def _score_decorated_candidate(candidate: GlyphShape, base: GlyphShape) -> dict[str, float]:
    full = shape_similarity(candidate.normalized, base.normalized, allow_mirror=True)
    central = shape_similarity(_center_band(candidate.normalized), _center_band(base.normalized), allow_mirror=True)
    candidate_ink = _ink_fraction(candidate.normalized)
    base_ink = _ink_fraction(base.normalized)
    if base_ink <= 0:
        ink_ratio = 0.0
    else:
        ink_ratio = candidate_ink / base_ink
    score = max(full, central)
    if not 0.55 <= ink_ratio <= 2.40:
        score *= 0.65
    return {
        "score": score,
        "fullSimilarity": full,
        "centralSimilarity": central,
        "inkRatio": ink_ratio,
    }


def propose_aliases(
    shapes: list[GlyphShape],
    groups: list[ShapeGroup],
    key_to_codepoint: dict[str, int],
    icon_map: dict[int, list[str]],
    *,
    min_similarity: float,
    limit: int,
) -> list[dict]:
    shape_by_key = {shape.key: shape for shape in shapes}
    names_by_group = group_icon_names(groups, key_to_codepoint, icon_map)
    base_groups = [group for group in groups if names_by_group.get(group.id)]
    candidate_groups = [
        group
        for group in groups
        if not names_by_group.get(group.id) and len(group.members) <= 2
    ]

    proposals_by_rep: dict[str, dict] = {}
    for candidate_group in candidate_groups:
        candidate_shape = shape_by_key[candidate_group.representative]
        best: tuple[ShapeGroup, dict[str, float]] | None = None
        for base_group in base_groups:
            base_shape = shape_by_key[base_group.representative]
            metrics = _score_decorated_candidate(candidate_shape, base_shape)
            if metrics["score"] < min_similarity:
                continue
            if best is None or metrics["score"] > best[1]["score"]:
                best = (base_group, metrics)
        if best is None:
            continue
        base_group, metrics = best
        proposal = proposals_by_rep.setdefault(
            base_group.representative,
            {
                "representative": base_group.representative,
                "representativeNames": names_by_group[base_group.id],
                "reason": "Review-only: unnamed glyph resembles this named base after modifier-tolerant comparison.",
                "members": [],
            },
        )
        for key in candidate_group.members:
            shape = shape_by_key[key]
            proposal["members"].append(
                {
                    "codepoint": key,
                    "glyphName": shape.glyph_name,
                    "score": round(metrics["score"], 4),
                    "fullSimilarity": round(metrics["fullSimilarity"], 4),
                    "centralSimilarity": round(metrics["centralSimilarity"], 4),
                    "inkRatio": round(metrics["inkRatio"], 4),
                }
            )

    proposals = list(proposals_by_rep.values())
    for proposal in proposals:
        proposal["members"] = sorted(proposal["members"], key=lambda member: -member["score"])
    proposals.sort(key=lambda item: -max(member["score"] for member in item["members"]))
    return proposals[:limit]


def main() -> None:
    args = parse_args()
    ranges = parse_codepoint_ranges(args.codepoint_range)
    shapes, key_to_codepoint = build_glyph_shapes(
        args.font,
        codepoint_ranges=ranges,
        canvas=args.canvas,
        render_px=args.render_px,
    )
    shape_groups = group_similar_shapes(
        shapes,
        threshold=args.shape_threshold,
        allow_mirror=not args.no_mirror_family_grouping,
    )
    shape_groups = merge_shape_group_aliases(shape_groups, load_shape_family_aliases(args.shape_family_aliases))
    proposals = propose_aliases(
        shapes,
        shape_groups,
        key_to_codepoint,
        load_icon_map(args.symbol_map),
        min_similarity=args.min_similarity,
        limit=args.limit,
    )
    payload = {
        "source": {
            "font": str(args.font),
            "symbolMap": str(args.symbol_map),
            "shapeFamilyAliases": str(args.shape_family_aliases),
            "shapeThreshold": args.shape_threshold,
            "mirrorFamilyGrouping": not args.no_mirror_family_grouping,
            "minSimilarity": args.min_similarity,
        },
        "proposals": proposals,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"Wrote {len(proposals)} proposal families -> {args.output}")


if __name__ == "__main__":
    main()
