"""Font-based template matching for Byzantine neume glyphs."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


FONT_PATH = Path("/Users/nadcost/psaltica-praxis/app/assets/fonts/PsalticaPraxisUnified.ttf")

# Groups whose glyphs appear as distinct isolated marks (good for template matching).
# modifier_gorgon and modifier_isson sit on/near base neumes — skip them.
MATCHABLE_GROUPS = {"base_neume", "rest", "mode", "modifier_modulation", "key_signature"}

FONT_SIZES_PT = [6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5]
DPI = 300
MATCH_THRESHOLD = 0.65
NMS_IOU_THRESHOLD = 0.3

# Reference size for glyph rendering (used for HTML output / reference sheet)
GLYPH_RENDER_PX = 96


def pt_to_px(pt: float, dpi: int = DPI) -> int:
    return max(4, round(pt * dpi / 72))


def render_template(insert: str, pt: float) -> np.ndarray | None:
    """Render insert string at 4x scale then downsample and binarize."""
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
    target = (max(1, w // scale), max(1, h // scale))
    img = img.resize(target, Image.LANCZOS)
    arr = np.array(img)
    _, arr = cv2.threshold(arr, 200, 255, cv2.THRESH_BINARY)
    return arr


def render_glyph_b64(insert: str, size_px: int = GLYPH_RENDER_PX, thumb_px: int = 64) -> str:
    """Render insert string anti-aliased and return a base64 PNG for HTML embedding."""
    try:
        font = ImageFont.truetype(str(FONT_PATH), size=size_px)
    except Exception:
        return ""
    dummy = Image.new("RGB", (1, 1))
    bbox = ImageDraw.Draw(dummy).textbbox((0, 0), insert, font=font)
    w = max(bbox[2] - bbox[0] + 8, 4)
    h = max(bbox[3] - bbox[1] + 8, 4)
    img = Image.new("RGB", (w, h), (255, 255, 255))
    ImageDraw.Draw(img).text((-bbox[0] + 4, -bbox[1] + 4), insert, font=font, fill=(0, 0, 0))
    img = img.resize((thumb_px, thumb_px), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def build_templates(
    classes: list[str],
    icon_to_insert: dict[str, str],
    sizes_pt: list[float] = FONT_SIZES_PT,
) -> dict[str, list[tuple[float, np.ndarray]]]:
    """Return class_label -> [(pt, template_array), ...].

    Templates for classes with longer insert strings (composite glyphs) are
    included alongside simple glyphs. Use match_cascade_page() to ensure
    composite matches suppress their components.
    """
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
    iou_threshold: float = NMS_IOU_THRESHOLD,
) -> list[tuple[int, int, int, int, float, str]]:
    """Standard NMS sorted by score descending."""
    if not detections:
        return []
    detections = sorted(detections, key=lambda d: d[4], reverse=True)
    kept = []
    while detections:
        best = detections.pop(0)
        kept.append(best)
        detections = [d for d in detections if iou(best, d) < iou_threshold]
    return kept


def match_cascade_page(
    page_gray: np.ndarray,
    templates: dict[str, list[tuple[float, np.ndarray]]],
    threshold: float,
    iou_threshold: float = NMS_IOU_THRESHOLD,
) -> list[tuple[int, int, int, int, float, str]]:
    """Cascade matching: composite (larger) glyphs suppress their components.

    All candidate detections are collected then sorted by (template_area desc,
    score desc) before NMS. This ensures that when a composite glyph and one of
    its component glyphs overlap at the same print location, the composite wins
    the NMS race and the component is suppressed — without requiring an explicit
    composite/simple classification.

    Falls back naturally: if the composite template scores below threshold at a
    given location, no composite detection is generated there, and the component
    template can still fire independently.
    """
    # Collect (x, y, w, h, score, label, template_area)
    candidates: list[tuple[int, int, int, int, float, str, int]] = []
    for label, variants in templates.items():
        for _pt, tmpl in variants:
            area = int(tmpl.shape[0]) * int(tmpl.shape[1])
            for x, y, w, h, score in match_template_on_page(page_gray, tmpl, threshold):
                candidates.append((x, y, w, h, score, label, area))

    if not candidates:
        return []

    # Sort: largest template area first, tie-break by score descending.
    # Composite glyphs have larger templates and thus win NMS over their components.
    candidates.sort(key=lambda d: (d[6], d[4]), reverse=True)

    kept: list[tuple[int, int, int, int, float, str]] = []
    while candidates:
        best = candidates.pop(0)
        kept.append(best[:6])
        candidates = [d for d in candidates if iou(best, d) < iou_threshold]

    return kept


def load_symbol_map(path: Path) -> dict[str, str]:
    """Return icon -> insert_string mapping (first match per icon)."""
    with path.open(encoding="utf-8") as f:
        sm = json.load(f)
    result: dict[str, str] = {}
    for entry in sm["symbols"]:
        icon = entry["icon"]
        if icon not in result and entry.get("insert"):
            result[icon] = entry["insert"]
    return result
