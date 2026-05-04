#!/usr/bin/env python3
"""Group neume font glyphs into families of similar visual shapes.

Renders every glyph in the target codepoint ranges, normalizes each to a
fixed square canvas (tight ink crop → scale → center-pad), then computes
pairwise Pearson similarity and clusters with average-linkage hierarchical
clustering at a configurable threshold.

Outputs:
  data/neume_families.json  — [{id, codepoints, representative}, ...]
  data/neume_families.html  — visual grid of families for human review

Usage:
  python tools/group_neume_shapes.py              # default threshold 0.80
  python tools/group_neume_shapes.py --threshold 0.85 --canvas 48
  python tools/group_neume_shapes.py --show-matrix  # include similarity heatmap
"""

from __future__ import annotations

import argparse
import base64
import io
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from psaltica_ocr.template_matching import (
    FONT_PATH,
    NEUME_CODEPOINT_RANGES,
    get_font_codepoints,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font", type=Path, default=FONT_PATH)
    parser.add_argument("--threshold", type=float, default=0.80,
                        help="Similarity threshold for same-family (default: 0.80)")
    parser.add_argument("--canvas", type=int, default=48,
                        help="Normalized canvas size in px (default: 48)")
    parser.add_argument("--render-px", type=int, default=128,
                        help="Render size before normalization (default: 128)")
    parser.add_argument("--show-matrix", action="store_true",
                        help="Include similarity heatmap in HTML")
    parser.add_argument("--output-json", type=Path, default=Path("data/neume_families.json"))
    parser.add_argument("--output-html", type=Path, default=Path("data/neume_families.html"))
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Normalized rendering
# ---------------------------------------------------------------------------

def render_normalized(char: str, font_path: Path, canvas: int, render_px: int) -> np.ndarray | None:
    """Render char at render_px, crop to ink tight-bbox, resize+pad to canvas×canvas."""
    try:
        font = ImageFont.truetype(str(font_path), size=render_px)
    except Exception:
        return None
    dummy = Image.new("L", (1, 1))
    bbox = ImageDraw.Draw(dummy).textbbox((0, 0), char, font=font)
    w = max(bbox[2] - bbox[0] + 16, 4)
    h = max(bbox[3] - bbox[1] + 16, 4)
    img = Image.new("L", (w, h), 255)
    ImageDraw.Draw(img).text((-bbox[0] + 8, -bbox[1] + 8), char, font=font, fill=0)
    arr = np.array(img)
    _, arr = cv2.threshold(arr, 200, 255, cv2.THRESH_BINARY)

    ink_rows = np.any(arr == 0, axis=1)
    ink_cols = np.any(arr == 0, axis=0)
    if not ink_rows.any() or not ink_cols.any():
        return None

    r0, r1 = int(np.where(ink_rows)[0][0]), int(np.where(ink_rows)[0][-1])
    c0, c1 = int(np.where(ink_cols)[0][0]), int(np.where(ink_cols)[0][-1])
    cropped = arr[r0:r1 + 1, c0:c1 + 1]

    ch, cw = cropped.shape
    scale = canvas / max(ch, cw)
    new_h = max(1, round(ch * scale))
    new_w = max(1, round(cw * scale))
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)
    _, resized = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)

    result = np.full((canvas, canvas), 255, dtype=np.uint8)
    pad_top = (canvas - new_h) // 2
    pad_left = (canvas - new_w) // 2
    result[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
    return result


def glyph_b64(char: str, font_path: Path, size_px: int = 64) -> str:
    """Render char anti-aliased and return base64 PNG."""
    try:
        font = ImageFont.truetype(str(font_path), size=size_px)
    except Exception:
        return ""
    dummy = Image.new("RGB", (1, 1))
    bbox = ImageDraw.Draw(dummy).textbbox((0, 0), char, font=font)
    w = max(bbox[2] - bbox[0] + 8, 4)
    h = max(bbox[3] - bbox[1] + 8, 4)
    img = Image.new("RGB", (w, h), (255, 255, 255))
    ImageDraw.Draw(img).text((-bbox[0] + 4, -bbox[1] + 4), char, font=font, fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# Similarity and clustering
# ---------------------------------------------------------------------------

def pearson_similarity(a: np.ndarray, b: np.ndarray) -> float:
    fa = (a < 128).astype(float).ravel()
    fb = (b < 128).astype(float).ravel()
    std_a, std_b = fa.std(), fb.std()
    if std_a < 1e-6 or std_b < 1e-6:
        return 0.0
    return float(np.dot(fa - fa.mean(), fb - fb.mean()) / (len(fa) * std_a * std_b))


def cluster_glyphs(
    keys: list[str],
    images: dict[str, np.ndarray],
    threshold: float,
) -> tuple[list[list[str]], np.ndarray]:
    """Return (families, similarity_matrix). Each family is a list of keys."""
    n = len(keys)
    sim = np.zeros((n, n), dtype=float)
    for i in range(n):
        sim[i, i] = 1.0
        for j in range(i + 1, n):
            s = pearson_similarity(images[keys[i]], images[keys[j]])
            sim[i, j] = sim[j, i] = max(s, 0.0)

    dist = np.clip(1.0 - sim, 0.0, 1.0)
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")
    labels = fcluster(Z, t=1.0 - threshold, criterion="distance")

    families_map: dict[int, list[str]] = defaultdict(list)
    for key, label in zip(keys, labels):
        families_map[label].append(key)

    # Sort families by size desc, then by first member codepoint
    families = sorted(families_map.values(), key=lambda g: (-len(g), g[0]))
    return families, sim


# ---------------------------------------------------------------------------
# HTML output
# ---------------------------------------------------------------------------

def write_html(
    output: Path,
    families: list[list[str]],
    codepoints: dict[str, int],
    font_path: Path,
    similarity_matrix: np.ndarray | None,
    keys: list[str],
    threshold: float,
    icon_map: dict[int, str],
) -> None:
    family_rows = []
    for fid, members in enumerate(families, 1):
        cells = []
        for key in members:
            cp = codepoints[key]
            icon = icon_map.get(cp, "")
            b64 = glyph_b64(chr(cp), font_path, size_px=64)
            img_html = (
                f'<img src="data:image/png;base64,{b64}" width="48" height="48" alt="{key}"/>'
                if b64 else '<span style="color:#ccc">?</span>'
            )
            label = f'<div class="cp">{key}</div>'
            if icon:
                label += f'<div class="icon">{icon}</div>'
            cells.append(f'<div class="glyph-cell">{img_html}{label}</div>')

        cells_html = "".join(cells)
        member_count = len(members)
        rep = members[0]
        family_rows.append(
            f'<div class="family">'
            f'<div class="fam-header">Family {fid} &nbsp;·&nbsp; {member_count} glyph{"s" if member_count > 1 else ""}'
            f' &nbsp;·&nbsp; rep: {rep}</div>'
            f'<div class="glyph-row">{cells_html}</div>'
            f'</div>'
        )

    matrix_section = ""
    if similarity_matrix is not None:
        n = len(keys)
        # Build mini SVG heatmap (cap at 80 glyphs for size)
        show_n = min(n, 80)
        cell = 6
        svg_w = show_n * cell
        svg_h = show_n * cell
        rects = []
        for i in range(show_n):
            for j in range(show_n):
                v = similarity_matrix[i, j]
                r = int(255 * (1 - v))
                g = int(255 * (1 - v))
                b = 255
                if i == j:
                    r, g, b = 40, 40, 40
                rects.append(
                    f'<rect x="{j*cell}" y="{i*cell}" width="{cell}" height="{cell}" '
                    f'fill="rgb({r},{g},{b})"/>'
                )
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_w}" height="{svg_h}">'
            + "".join(rects) + "</svg>"
        )
        matrix_section = (
            f'<details><summary style="cursor:pointer;padding:8px 0;color:#888">'
            f'Similarity matrix (first {show_n} glyphs)</summary>'
            f'<div style="overflow:auto">{svg}</div>'
            f'</details>'
        )

    total_glyphs = sum(len(f) for f in families)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>Psaltica — Neume Shape Families</title>
<style>
  body {{ font-family: sans-serif; background: #f5f5f5; padding: 16px 24px; color: #222; }}
  h1 {{ font-size: 1.3em; margin-bottom: 4px; }}
  .subtitle {{ color: #666; font-size: 0.9em; margin-bottom: 16px; }}
  .family {{ background: white; border-radius: 6px; padding: 12px 16px;
             margin-bottom: 12px; box-shadow: 0 1px 3px rgba(0,0,0,.1); }}
  .fam-header {{ font-size: 0.8em; color: #888; margin-bottom: 8px; font-family: monospace; }}
  .glyph-row {{ display: flex; flex-wrap: wrap; gap: 8px; }}
  .glyph-cell {{ text-align: center; width: 56px; }}
  .glyph-cell img {{ display: block; margin: auto; }}
  .cp {{ font-size: 0.7em; font-family: monospace; color: #555; margin-top: 2px; }}
  .icon {{ font-size: 0.7em; color: #337ab7; margin-top: 1px; word-break: break-word; }}
</style>
</head>
<body>
<h1>Psaltica — Neume Shape Families</h1>
<p class="subtitle">
  {total_glyphs} glyphs → {len(families)} families &nbsp;·&nbsp;
  similarity threshold: {threshold} &nbsp;·&nbsp;
  average linkage clustering
</p>
{matrix_section}
{"".join(family_rows)}
</body>
</html>
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_icon_map(font_path: Path) -> dict[int, str]:
    sm_path = font_path.parent.parent.parent / "config" / "symbol_map.json"
    # Try relative to cwd too
    for p in [Path("config/symbol_map.json"), sm_path]:
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            result: dict[int, str] = {}
            for entry in data.get("symbols", []):
                icon = entry.get("icon", "")
                for ch in entry.get("insert", ""):
                    result.setdefault(ord(ch), icon)
            return result
    return {}


def main() -> None:
    args = parse_args()
    icon_map = load_icon_map(args.font)

    all_codepoints = get_font_codepoints(args.font)
    allowed = set()
    for lo, hi in NEUME_CODEPOINT_RANGES:
        for cp in range(lo, hi + 1):
            allowed.add(cp)

    print(f"Rendering normalized glyphs (canvas={args.canvas}px, render={args.render_px}px)…")
    keys: list[str] = []
    codepoints: dict[str, int] = {}
    images: dict[str, np.ndarray] = {}

    for cp in sorted(all_codepoints):
        if cp not in allowed or cp < 0x20:
            continue
        img = render_normalized(chr(cp), args.font, args.canvas, args.render_px)
        if img is None:
            continue
        key = f"U+{cp:04X}"
        keys.append(key)
        codepoints[key] = cp
        images[key] = img

    print(f"  {len(keys)} glyphs rendered")
    print(f"Computing {len(keys)}×{len(keys)} similarity matrix…")

    families, sim_matrix = cluster_glyphs(keys, images, args.threshold)

    singletons = sum(1 for f in families if len(f) == 1)
    grouped = sum(len(f) for f in families if len(f) > 1)
    print(f"  {len(families)} families  ({singletons} singletons, "
          f"{len(families)-singletons} multi-member families covering {grouped} glyphs)")

    # Build JSON output
    output_data = []
    for fid, members in enumerate(families, 1):
        output_data.append({
            "id": fid,
            "representative": members[0],
            "codepoints": members,
            "icons": [icon_map.get(codepoints[k], "") for k in members],
        })

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(output_data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.output_json}")

    print("Rendering HTML report…")
    write_html(
        args.output_html,
        families,
        codepoints,
        args.font,
        sim_matrix if args.show_matrix else None,
        keys,
        args.threshold,
        icon_map,
    )
    print(f"Wrote {args.output_html}")
    print(f"\nReview {args.output_html} and adjust --threshold if needed.")
    print(f"Then pass --families {args.output_json} to count_neume_frequencies.py")


if __name__ == "__main__":
    main()
