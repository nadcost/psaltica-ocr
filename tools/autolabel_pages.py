#!/usr/bin/env python3
"""Auto-label pages by template-matching font-rendered glyphs against printed scans.

Renders each class glyph from PsalticaPraxisUnified.ttf at several font sizes,
runs OpenCV template matching on the masked page images, applies NMS, and writes
a Label Studio predictions JSON that you import alongside your tasks.

Works best for: base_neume, rest, mode, modifier_modulation, key_signature.
Modifier glyphs (gorgon, isson) that sit atop other neumes are skipped by default.

Usage:
  python tools/autolabel_pages.py --pages data/pages/Mass/page_0025.png
  python tools/autolabel_pages.py --book Mass --pages-per-book 50
"""

from __future__ import annotations

import argparse
import io
import json
import math
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont


FONT_PATH = Path("/Users/nadcost/psaltica-praxis/app/assets/fonts/PsalticaPraxisUnified.ttf")

# Groups whose glyphs appear as distinct isolated marks (good for template matching).
# modifier_gorgon and modifier_isson sit on/near base neumes — skip them.
MATCHABLE_GROUPS = {"base_neume", "rest", "mode", "modifier_modulation", "key_signature"}

# Font sizes (pt) to try at 300 DPI. Cover the typical print range.
FONT_SIZES_PT = [6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5]
DPI = 300

MATCH_THRESHOLD = 0.65   # lower → more recalls, more false positives
NMS_IOU_THRESHOLD = 0.3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pages", nargs="+", type=Path, help="Specific page image paths")
    group.add_argument("--book", help="Book ID — match all pages in data/pages/<book>/")
    parser.add_argument("--pages-per-book", type=int, default=0,
                        help="Limit pages when using --book (0 = all)")
    parser.add_argument("--manifest", type=Path, default=Path("data/pages/manifest.csv"))
    parser.add_argument("--classes", type=Path, default=Path("config/classes.yaml"))
    parser.add_argument("--symbol-map", type=Path, default=Path("config/symbol_map.json"))
    parser.add_argument("--output", type=Path, default=Path("data/annotations/predictions.json"))
    parser.add_argument("--threshold", type=float, default=MATCH_THRESHOLD)
    parser.add_argument("--local-files-root", type=Path, default=Path("."))
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Template rendering
# ---------------------------------------------------------------------------

def pt_to_px(pt: float, dpi: int = DPI) -> int:
    return max(4, round(pt * dpi / 72))


def render_template(insert: str, pt: float) -> np.ndarray | None:
    """Render insert string; supersample 4x then downsample, then binarize.

    Supersampling then downscaling produces cleaner binary templates than
    rendering directly at the target size with anti-aliasing.
    """
    px = pt_to_px(pt)
    scale = 4
    try:
        font = ImageFont.truetype(str(FONT_PATH), size=px * scale)
    except Exception:
        return None
    dummy = Image.new("L", (1, 1))
    bbox = ImageDraw.Draw(dummy).textbbox((0, 0), insert, font=font)
    w = bbox[2] - bbox[0] + 8
    h = bbox[3] - bbox[1] + 8
    if w < 4 or h < 4:
        return None
    img = Image.new("L", (w, h), 255)
    ImageDraw.Draw(img).text((-bbox[0] + 4, -bbox[1] + 4), insert, font=font, fill=0)
    # Downsample to target size
    target = (max(1, w // scale), max(1, h // scale))
    img = img.resize(target, Image.LANCZOS)
    arr = np.array(img)
    # Binarize: ink pixels → 0, background → 255
    _, arr = cv2.threshold(arr, 200, 255, cv2.THRESH_BINARY)
    return arr


def build_templates(
    classes: list[str],
    icon_to_insert: dict[str, str],
    sizes_pt: list[float],
) -> dict[str, list[tuple[float, np.ndarray]]]:
    """Return class_label -> [(pt, template_array), ...]."""
    templates: dict[str, list[tuple[float, np.ndarray]]] = {}
    for cls in classes:
        group = cls.split(".", 1)[0]
        if group not in MATCHABLE_GROUPS:
            continue
        icon = cls.split(".", 1)[1]
        insert = icon_to_insert.get(icon)
        if not insert:
            continue
        variants: list[tuple[float, np.ndarray]] = []
        for pt in sizes_pt:
            tmpl = render_template(insert, pt)
            if tmpl is not None:
                variants.append((pt, tmpl))
        if variants:
            templates[cls] = variants
    return templates


# ---------------------------------------------------------------------------
# Matching + NMS
# ---------------------------------------------------------------------------

def match_template_on_page(
    page_gray: np.ndarray,
    tmpl: np.ndarray,
    threshold: float,
) -> list[tuple[int, int, int, int, float]]:
    """Return list of (x, y, w, h, score) in pixel coords."""
    if tmpl.shape[0] > page_gray.shape[0] or tmpl.shape[1] > page_gray.shape[1]:
        return []
    result = cv2.matchTemplate(page_gray, tmpl, cv2.TM_CCOEFF_NORMED)
    locs = np.where(result >= threshold)
    th, tw = tmpl.shape[:2]
    return [
        (int(x), int(y), tw, th, float(result[y, x]))
        for y, x in zip(*locs)
    ]


def iou(a: tuple, b: tuple) -> float:
    ax, ay, aw, ah = a[:4]
    bx, by, bw, bh = b[:4]
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def nms(
    detections: list[tuple[int, int, int, int, float, str]],
    iou_threshold: float,
) -> list[tuple[int, int, int, int, float, str]]:
    if not detections:
        return []
    detections = sorted(detections, key=lambda d: d[4], reverse=True)
    kept = []
    while detections:
        best = detections.pop(0)
        kept.append(best)
        detections = [d for d in detections if iou(best, d) < iou_threshold]
    return kept


# ---------------------------------------------------------------------------
# Label Studio output format
# ---------------------------------------------------------------------------

def to_ls_result(
    x_px: int, y_px: int, w_px: int, h_px: int,
    page_w: int, page_h: int,
    label: str,
    score: float,
) -> dict:
    return {
        "type": "rectanglelabels",
        "from_name": "label",
        "to_name": "image",
        "value": {
            "x": round(x_px / page_w * 100, 4),
            "y": round(y_px / page_h * 100, 4),
            "width": round(w_px / page_w * 100, 4),
            "height": round(h_px / page_h * 100, 4),
            "rectanglelabels": [label],
        },
        "score": round(score, 4),
    }


def image_url(image_path: Path, local_files_root: Path) -> str:
    try:
        rel = image_path.resolve().relative_to(local_files_root.resolve())
        return f"/data/local-files/?d={rel.as_posix()}"
    except ValueError:
        return str(image_path)


# ---------------------------------------------------------------------------
# Per-page processing
# ---------------------------------------------------------------------------

def process_page(
    image_path: Path,
    templates: dict[str, list[tuple[float, np.ndarray]]],
    threshold: float,
    local_files_root: Path,
) -> dict:
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read {image_path}")
    page_h, page_w = img.shape
    # Binarize page to match the binarized templates
    _, img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)

    all_detections: list[tuple[int, int, int, int, float, str]] = []
    for label, variants in templates.items():
        for _pt, tmpl in variants:
            for x, y, w, h, score in match_template_on_page(img, tmpl, threshold):
                all_detections.append((x, y, w, h, score, label))

    kept = nms(all_detections, NMS_IOU_THRESHOLD)
    results = [
        to_ls_result(x, y, w, h, page_w, page_h, label, score)
        for x, y, w, h, score, label in kept
    ]

    return {
        "data": {"image": image_url(image_path, local_files_root)},
        "predictions": [{"result": results, "score": float(np.mean([r["score"] for r in results])) if results else 0.0}],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_symbol_map(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8") as f:
        sm = json.load(f)
    result: dict[str, str] = {}
    for entry in sm["symbols"]:
        icon = entry["icon"]
        if icon not in result and entry.get("insert"):
            result[icon] = entry["insert"]
    return result


def collect_pages(args: argparse.Namespace) -> list[Path]:
    if args.pages:
        return [p for p in args.pages if p.exists()]
    book_dir = Path("data/pages") / args.book
    pages = sorted(book_dir.glob("page_*.png"))
    if args.pages_per_book:
        pages = pages[: args.pages_per_book]
    return pages


def main() -> None:
    args = parse_args()
    with args.classes.open(encoding="utf-8") as f:
        classes = yaml.safe_load(f)["names"]
    icon_to_insert = load_symbol_map(args.symbol_map)

    print("Rendering templates…")
    templates = build_templates(classes, icon_to_insert, FONT_SIZES_PT)
    matchable = [c for c in classes if c.split(".",1)[0] in MATCHABLE_GROUPS]
    print(f"  {len(templates)}/{len(matchable)} matchable classes have templates")

    pages = collect_pages(args)
    if not pages:
        raise SystemExit("No pages found.")

    tasks = []
    for i, page_path in enumerate(pages, 1):
        task = process_page(page_path, templates, args.threshold, args.local_files_root.resolve())
        n = len(task["predictions"][0]["result"])
        print(f"  [{i}/{len(pages)}] {page_path.name}: {n} detections")
        tasks.append(task)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(tasks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {len(tasks)} task predictions → {args.output}")
    print("Import this file into Label Studio (Import tab) to see pre-labeled suggestions.")


if __name__ == "__main__":
    main()
