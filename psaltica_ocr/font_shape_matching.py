"""Group font glyphs by visual shape and match shape groups on page images."""

from __future__ import annotations

import base64
import csv
import html
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from psaltica_ocr.template_matching import DPI, FONT_PATH, NEUME_CODEPOINT_RANGES, get_font_codepoints


@dataclass(frozen=True)
class GlyphShape:
    key: str
    codepoint: int
    glyph_name: str
    normalized: np.ndarray


@dataclass(frozen=True)
class ShapeGroup:
    id: str
    representative: str
    members: tuple[str, ...]


@dataclass(frozen=True)
class MatchDetection:
    x: int
    y: int
    width: int
    height: int
    score: float
    group_id: str
    representative: str
    size_pt: float


DEFAULT_ICON_THRESHOLDS: dict[str, float] = {
    "Apostrofos": 0.65,
    "Isson2": 0.75,
    "Oligon": 0.78,
}

DEFAULT_ICON_PRIORITIES: dict[str, int] = {
    "Isson2": 40,
    "Apostrofos": 35,
    "Oligon": 5,
}


def parse_codepoint_ranges(values: Sequence[str] | None) -> list[tuple[int, int]]:
    if not values:
        return list(NEUME_CODEPOINT_RANGES)
    ranges: list[tuple[int, int]] = []
    for value in values:
        text = value.strip().upper().replace("U+", "")
        if "-" in text:
            start_text, end_text = text.split("-", 1)
            start = int(start_text, 16)
            end = int(end_text.replace("U+", ""), 16)
        else:
            start = end = int(text, 16)
        if end < start:
            raise ValueError(f"Invalid codepoint range: {value}")
        ranges.append((start, end))
    return ranges


def allowed_codepoints(ranges: Iterable[tuple[int, int]] | None) -> set[int] | None:
    if ranges is None:
        return None
    allowed: set[int] = set()
    for start, end in ranges:
        allowed.update(range(start, end + 1))
    return allowed


def codepoint_key(codepoint: int) -> str:
    return f"U+{codepoint:04X}"


def render_glyph_bitmap(
    char: str,
    font_path: Path = FONT_PATH,
    *,
    size_px: int,
    pad: int = 8,
    threshold: int = 200,
) -> np.ndarray | None:
    try:
        font = ImageFont.truetype(str(font_path), size=size_px)
    except OSError:
        return None
    dummy = Image.new("L", (1, 1), 255)
    bbox = ImageDraw.Draw(dummy).textbbox((0, 0), char, font=font)
    width = max(bbox[2] - bbox[0] + pad * 2, 4)
    height = max(bbox[3] - bbox[1] + pad * 2, 4)
    image = Image.new("L", (width, height), 255)
    ImageDraw.Draw(image).text((-bbox[0] + pad, -bbox[1] + pad), char, font=font, fill=0)
    array = np.array(image)
    _, binary = cv2.threshold(array, threshold, 255, cv2.THRESH_BINARY)
    return binary


def crop_to_ink(image: np.ndarray) -> np.ndarray | None:
    ink_rows = np.any(image < 128, axis=1)
    ink_cols = np.any(image < 128, axis=0)
    if not ink_rows.any() or not ink_cols.any():
        return None
    row_indexes = np.where(ink_rows)[0]
    col_indexes = np.where(ink_cols)[0]
    return image[row_indexes[0] : row_indexes[-1] + 1, col_indexes[0] : col_indexes[-1] + 1]


def normalize_shape(image: np.ndarray, *, canvas: int = 48) -> np.ndarray | None:
    """Return a tight-cropped, aspect-preserved, centered binary glyph image.

    This intentionally removes x/y placement differences inside the font. Two
    codepoints with the same ink shape but different glyph bearings should end up
    with the same normalized bitmap.
    """

    cropped = crop_to_ink(image)
    if cropped is None:
        return None
    height, width = cropped.shape[:2]
    scale = canvas / max(height, width)
    target_width = max(1, round(width * scale))
    target_height = max(1, round(height * scale))
    resized = cv2.resize(cropped, (target_width, target_height), interpolation=cv2.INTER_AREA)
    _, resized = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)
    result = np.full((canvas, canvas), 255, dtype=np.uint8)
    top = (canvas - target_height) // 2
    left = (canvas - target_width) // 2
    result[top : top + target_height, left : left + target_width] = resized
    return result


def shape_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_ink = (left < 128).astype(np.float32).ravel()
    right_ink = (right < 128).astype(np.float32).ravel()
    left_std = float(left_ink.std())
    right_std = float(right_ink.std())
    if left_std < 1e-6 or right_std < 1e-6:
        return 0.0
    left_centered = left_ink - float(left_ink.mean())
    right_centered = right_ink - float(right_ink.mean())
    return float(np.dot(left_centered, right_centered) / (len(left_ink) * left_std * right_std))


def group_similar_shapes(shapes: Sequence[GlyphShape], *, threshold: float) -> list[ShapeGroup]:
    """Group glyphs by connected components in the similarity graph."""

    parent = list(range(len(shapes)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left_index in range(len(shapes)):
        for right_index in range(left_index + 1, len(shapes)):
            if shape_similarity(shapes[left_index].normalized, shapes[right_index].normalized) >= threshold:
                union(left_index, right_index)

    grouped: dict[int, list[GlyphShape]] = {}
    for index, shape in enumerate(shapes):
        grouped.setdefault(find(index), []).append(shape)

    groups: list[ShapeGroup] = []
    sorted_members = sorted(grouped.values(), key=lambda members: (-len(members), members[0].codepoint))
    for group_index, members in enumerate(sorted_members, 1):
        representative = choose_representative(members)
        groups.append(
            ShapeGroup(
                id=f"shape_{group_index:04d}",
                representative=representative.key,
                members=tuple(shape.key for shape in sorted(members, key=lambda item: item.codepoint)),
            )
        )
    return groups


def choose_representative(members: Sequence[GlyphShape]) -> GlyphShape:
    if len(members) == 1:
        return members[0]
    best = members[0]
    best_score = -1.0
    for candidate in members:
        score = sum(shape_similarity(candidate.normalized, other.normalized) for other in members) / len(members)
        if score > best_score or (score == best_score and candidate.codepoint < best.codepoint):
            best = candidate
            best_score = score
    return best


def load_icon_map(path: Path = Path("config/symbol_map.json")) -> dict[int, list[str]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    icons: dict[int, list[str]] = {}
    for entry in payload.get("symbols", []):
        icon = entry.get("icon")
        insert = entry.get("insert") or ""
        if not icon:
            continue
        for char in insert:
            icons.setdefault(ord(char), [])
            if icon not in icons[ord(char)]:
                icons[ord(char)].append(icon)
    return icons


def render_glyph_png_b64(codepoint: int, font_path: Path = FONT_PATH, *, size_px: int = 96) -> str:
    bitmap = render_glyph_bitmap(chr(codepoint), font_path, size_px=size_px, pad=8)
    if bitmap is None:
        return ""
    cropped = crop_to_ink(bitmap)
    if cropped is None:
        return ""
    image = Image.fromarray(cropped).convert("RGB")
    image.thumbnail((64, 64), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (72, 72), (255, 255, 255))
    left = (canvas.width - image.width) // 2
    top = (canvas.height - image.height) // 2
    canvas.paste(image, (left, top))
    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def build_glyph_shapes(
    font_path: Path = FONT_PATH,
    *,
    codepoint_ranges: list[tuple[int, int]] | None = None,
    canvas: int = 48,
    render_px: int = 128,
) -> tuple[list[GlyphShape], dict[str, int]]:
    allowed = allowed_codepoints(codepoint_ranges)
    font_codepoints = get_font_codepoints(font_path)
    shapes: list[GlyphShape] = []
    key_to_codepoint: dict[str, int] = {}
    for codepoint, glyph_name in sorted(font_codepoints.items()):
        if codepoint < 0x20:
            continue
        if allowed is not None and codepoint not in allowed:
            continue
        bitmap = render_glyph_bitmap(chr(codepoint), font_path, size_px=render_px)
        if bitmap is None:
            continue
        normalized = normalize_shape(bitmap, canvas=canvas)
        if normalized is None:
            continue
        key = codepoint_key(codepoint)
        shapes.append(GlyphShape(key=key, codepoint=codepoint, glyph_name=glyph_name, normalized=normalized))
        key_to_codepoint[key] = codepoint
    return shapes, key_to_codepoint


def pt_to_px(pt: float, *, dpi: int = DPI) -> int:
    return max(4, round(pt * dpi / 72))


def render_match_template(
    char: str,
    font_path: Path = FONT_PATH,
    *,
    pt: float,
    dpi: int = DPI,
    scale: int = 4,
) -> np.ndarray | None:
    px = pt_to_px(pt, dpi=dpi)
    bitmap = render_glyph_bitmap(char, font_path, size_px=px * scale, pad=4)
    if bitmap is None:
        return None
    target = (max(1, bitmap.shape[1] // scale), max(1, bitmap.shape[0] // scale))
    resized = cv2.resize(bitmap, target, interpolation=cv2.INTER_AREA)
    _, binary = cv2.threshold(resized, 200, 255, cv2.THRESH_BINARY)
    return binary


def template_is_matchable(
    template: np.ndarray,
    *,
    min_area: int = 80,
    min_ink_frac: float = 0.04,
    max_ink_frac: float = 0.55,
    min_shape_variance: float = 0.08,
    min_aspect: float = 0.3,
    max_aspect: float = 3.5,
    min_ink_rows: int = 4,
) -> bool:
    if template.shape[0] * template.shape[1] < min_area:
        return False
    ink_fraction = float(np.mean(template == 0))
    if not (min_ink_frac <= ink_fraction <= max_ink_frac):
        return False
    row_ink = (template == 0).mean(axis=1).astype(float)
    if float(row_ink.std()) < min_shape_variance:
        return False
    if int(np.any(template == 0, axis=1).sum()) < min_ink_rows:
        return False
    height, width = template.shape[:2]
    aspect = width / height if height > 0 else 0.0
    return min_aspect <= aspect <= max_aspect


def build_group_templates(
    groups: Sequence[ShapeGroup],
    key_to_codepoint: dict[str, int],
    font_path: Path = FONT_PATH,
    *,
    sizes_pt: Sequence[float],
    dpi: int = DPI,
) -> dict[str, list[tuple[float, np.ndarray]]]:
    templates: dict[str, list[tuple[float, np.ndarray]]] = {}
    for group in groups:
        codepoint = key_to_codepoint[group.representative]
        variants: list[tuple[float, np.ndarray]] = []
        for pt in sizes_pt:
            template = render_match_template(chr(codepoint), font_path, pt=pt, dpi=dpi)
            if template is not None and template_is_matchable(template):
                variants.append((pt, template))
        if variants:
            templates[group.id] = variants
    return templates


def match_template_on_page(
    page_gray: np.ndarray,
    template: np.ndarray,
    *,
    threshold: float,
) -> list[tuple[int, int, int, int, float]]:
    if template.shape[0] > page_gray.shape[0] or template.shape[1] > page_gray.shape[1]:
        return []
    result = cv2.matchTemplate(page_gray, template, cv2.TM_CCOEFF_NORMED)
    height, width = template.shape[:2]
    peak_kernel = max(3, min(width, height) // 3)
    if peak_kernel % 2 == 0:
        peak_kernel += 1
    local_max = cv2.dilate(result, np.ones((peak_kernel, peak_kernel), dtype=np.uint8))
    locations = np.where((result >= threshold) & (result == local_max))
    return [(int(x), int(y), width, height, float(result[y, x])) for y, x in zip(*locations)]


def iou(left: MatchDetection, right: MatchDetection) -> float:
    left_x2 = left.x + left.width
    left_y2 = left.y + left.height
    right_x2 = right.x + right.width
    right_y2 = right.y + right.height
    intersect_width = max(0, min(left_x2, right_x2) - max(left.x, right.x))
    intersect_height = max(0, min(left_y2, right_y2) - max(left.y, right.y))
    intersection = intersect_width * intersect_height
    union = left.width * left.height + right.width * right.height - intersection
    return intersection / union if union else 0.0


def non_max_suppression(
    detections: Sequence[MatchDetection],
    *,
    iou_threshold: float,
    priorities: dict[str, int] | None = None,
) -> list[MatchDetection]:
    priorities = priorities or {}
    candidates = sorted(
        detections,
        key=lambda item: (priorities.get(item.group_id, 10), item.score),
        reverse=True,
    )
    kept: list[MatchDetection] = []
    while candidates:
        best = candidates.pop(0)
        kept.append(best)
        candidates = [candidate for candidate in candidates if iou(best, candidate) < iou_threshold]
    return kept


def match_shape_groups_on_page(
    page_gray: np.ndarray,
    groups: Sequence[ShapeGroup],
    templates: dict[str, list[tuple[float, np.ndarray]]],
    *,
    threshold: float,
    iou_threshold: float,
    thresholds: dict[str, float] | None = None,
    priorities: dict[str, int] | None = None,
) -> list[MatchDetection]:
    _, binary_page = cv2.threshold(page_gray, 200, 255, cv2.THRESH_BINARY)
    group_by_id = {group.id: group for group in groups}
    detections: list[MatchDetection] = []
    for group_id, variants in templates.items():
        group = group_by_id[group_id]
        group_threshold = (thresholds or {}).get(group_id, threshold)
        for size_pt, template in variants:
            for x, y, width, height, score in match_template_on_page(
                binary_page,
                template,
                threshold=group_threshold,
            ):
                detections.append(
                    MatchDetection(
                        x=x,
                        y=y,
                        width=width,
                        height=height,
                        score=score,
                        group_id=group_id,
                        representative=group.representative,
                        size_pt=size_pt,
                    )
                )
    return non_max_suppression(detections, iou_threshold=iou_threshold, priorities=priorities)


def group_icon_names(
    groups: Sequence[ShapeGroup],
    key_to_codepoint: dict[str, int],
    icon_map: dict[int, list[str]],
) -> dict[str, list[str]]:
    names_by_group: dict[str, list[str]] = {}
    for group in groups:
        names: list[str] = []
        for key in group.members:
            for icon in icon_map.get(key_to_codepoint[key], []):
                if icon not in names:
                    names.append(icon)
        names_by_group[group.id] = names
    return names_by_group


def group_thresholds_from_icons(
    names_by_group: dict[str, list[str]],
    *,
    default_threshold: float,
    icon_thresholds: dict[str, float] | None = None,
) -> dict[str, float]:
    icon_thresholds = icon_thresholds or DEFAULT_ICON_THRESHOLDS
    thresholds: dict[str, float] = {}
    for group_id, names in names_by_group.items():
        values = [icon_thresholds[name] for name in names if name in icon_thresholds]
        thresholds[group_id] = min(values) if values else default_threshold
    return thresholds


def group_priorities_from_icons(
    names_by_group: dict[str, list[str]],
    *,
    icon_priorities: dict[str, int] | None = None,
) -> dict[str, int]:
    icon_priorities = icon_priorities or DEFAULT_ICON_PRIORITIES
    priorities: dict[str, int] = {}
    for group_id, names in names_by_group.items():
        values = [icon_priorities[name] for name in names if name in icon_priorities]
        priorities[group_id] = max(values) if values else 10
    return priorities


def groups_to_jsonable(
    groups: Sequence[ShapeGroup],
    shapes: Sequence[GlyphShape],
    *,
    icon_map: dict[int, list[str]] | None = None,
) -> list[dict]:
    shape_by_key = {shape.key: shape for shape in shapes}
    icon_map = icon_map or {}
    rows: list[dict] = []
    for group in groups:
        members = []
        for key in group.members:
            shape = shape_by_key[key]
            members.append(
                {
                    "codepoint": key,
                    "char": chr(shape.codepoint),
                    "glyphName": shape.glyph_name,
                    "icons": icon_map.get(shape.codepoint, []),
                }
            )
        rows.append(
            {
                "id": group.id,
                "representative": group.representative,
                "memberCount": len(group.members),
                "members": members,
            }
        )
    return rows


def write_detections_csv(path: Path, pages: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "page",
                "group_id",
                "representative",
                "x",
                "y",
                "width",
                "height",
                "score",
                "size_pt",
                "members",
            ],
        )
        writer.writeheader()
        for page in pages:
            for detection in page["detections"]:
                writer.writerow(
                    {
                        "page": page["image"],
                        "group_id": detection["groupId"],
                        "representative": detection["representative"],
                        "x": detection["bbox"][0],
                        "y": detection["bbox"][1],
                        "width": detection["bbox"][2],
                        "height": detection["bbox"][3],
                        "score": detection["score"],
                        "size_pt": detection["sizePt"],
                        "members": " ".join(detection["members"]),
                    }
                )


def detection_frequencies(pages: Sequence[dict]) -> dict[str, int]:
    frequencies: dict[str, int] = {}
    for page in pages:
        for detection in page.get("detections", []):
            group_id = detection["groupId"]
            frequencies[group_id] = frequencies.get(group_id, 0) + 1
    return frequencies


def write_match_report_html(
    path: Path,
    *,
    font_path: Path,
    groups_payload: dict,
    pages: Sequence[dict],
    match_threshold: float,
    shape_threshold: float,
) -> None:
    """Write a self-contained HTML report for shape families and match counts."""

    frequencies = detection_frequencies(pages)
    total_detections = sum(frequencies.values())
    group_rows = []
    groups = groups_payload["groups"]
    sorted_groups = sorted(groups, key=lambda group: (-frequencies.get(group["id"], 0), group["id"]))
    max_frequency = max(frequencies.values(), default=1)

    for group in sorted_groups:
        frequency = frequencies.get(group["id"], 0)
        representative = group["representative"]
        representative_member = next(
            (member for member in group["members"] if member["codepoint"] == representative),
            group["members"][0],
        )
        codepoint = int(representative_member["codepoint"].replace("U+", ""), 16)
        glyph_b64 = render_glyph_png_b64(codepoint, font_path)
        glyph_html = (
            f'<img src="data:image/png;base64,{glyph_b64}" alt="{html.escape(representative)}"/>'
            if glyph_b64
            else "-"
        )

        member_rows = []
        family_names: list[str] = []
        for member in group["members"]:
            names = member.get("icons") or []
            name_text = ", ".join(names) if names else "-"
            if names:
                family_names.extend(names)
            member_rows.append(
                "<tr>"
                f"<td>{html.escape(member['codepoint'])}</td>"
                f"<td>{html.escape(member.get('glyphName') or '-')}</td>"
                f"<td>{html.escape(name_text)}</td>"
                "</tr>"
            )
        family_name_text = ", ".join(dict.fromkeys(family_names)) if family_names else "-"
        bar_width = round((frequency / max_frequency) * 180) if max_frequency else 0
        group_rows.append(
            "<tr>"
            f"<td class=\"glyph\">{glyph_html}</td>"
            f"<td class=\"group\"><strong>{html.escape(group['id'])}</strong>"
            f"<div class=\"muted\">rep {html.escape(representative)}</div></td>"
            f"<td class=\"names\">{html.escape(family_name_text)}</td>"
            f"<td class=\"members\"><details><summary>{len(group['members'])} codepoint"
            f"{'' if len(group['members']) == 1 else 's'}</summary>"
            "<table class=\"nested\"><thead><tr><th>Unicode</th><th>Glyph name</th><th>App name</th></tr></thead>"
            f"<tbody>{''.join(member_rows)}</tbody></table></details></td>"
            f"<td class=\"freq\"><span class=\"count\">{frequency}</span>"
            f"<span class=\"bar\"><span style=\"width:{bar_width}px\"></span></span></td>"
            "</tr>"
        )

    page_count = len(pages)
    html_text = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>Psaltica Font Shape Match Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #222; }}
  h1 {{ font-size: 22px; margin: 0 0 6px; }}
  .summary {{ color: #555; margin: 0 0 18px; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th {{ text-align: left; font-size: 12px; color: #555; border-bottom: 1px solid #ccc; padding: 8px; }}
  td {{ border-bottom: 1px solid #eee; padding: 8px; vertical-align: top; }}
  .glyph {{ width: 88px; text-align: center; }}
  .glyph img {{ width: 72px; height: 72px; image-rendering: pixelated; border: 1px solid #ddd; background: white; }}
  .group {{ width: 120px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 13px; }}
  .muted {{ color: #777; font-size: 11px; margin-top: 4px; }}
  .names {{ width: 220px; font-size: 13px; }}
  .members summary {{ cursor: pointer; color: #205c8a; }}
  .nested {{ margin-top: 8px; width: auto; min-width: 520px; }}
  .nested th, .nested td {{ padding: 5px 8px; border-bottom: 1px solid #eee; font-size: 12px; }}
  .nested td:first-child {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
  .freq {{ width: 250px; white-space: nowrap; }}
  .count {{ display: inline-block; min-width: 44px; text-align: right; margin-right: 10px; font-variant-numeric: tabular-nums; }}
  .bar {{ display: inline-block; width: 180px; height: 10px; background: #eee; vertical-align: middle; }}
  .bar span {{ display: block; height: 10px; background: #4577a8; }}
</style>
</head>
<body>
<h1>Psaltica Font Shape Match Report</h1>
<p class="summary">
  {len(groups)} shape families &middot; {total_detections} detections &middot; {page_count} page{"s" if page_count != 1 else ""} &middot;
  shape threshold {shape_threshold} &middot; match threshold {match_threshold}
</p>
<table>
<thead>
  <tr><th>Glyph matched</th><th>Family</th><th>App name</th><th>Unicode locations in family</th><th>Frequency</th></tr>
</thead>
<tbody>
  {''.join(group_rows)}
</tbody>
</table>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html_text, encoding="utf-8")
