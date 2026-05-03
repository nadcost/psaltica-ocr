#!/usr/bin/env python3
"""Generate a self-contained HTML reference sheet mapping each class label to its glyph.

Uses the existing GIF toolbar icons where available, falls back to font rendering.
Open the output HTML file alongside Label Studio while annotating.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFont


FONT_PATH = Path("/Users/nadcost/psaltica-praxis/app/assets/fonts/PsalticaPraxisUnified.ttf")
ICONS_DIR = Path("/Users/nadcost/psaltica-praxis/app/assets/icons")
KEYS_DIR  = Path("/Users/nadcost/psaltica-praxis/app/assets/keys")

GLYPH_SIZE = 96   # px for font-rendered fallback glyphs
THUMB_SIZE = 80   # px to display all thumbnails

GROUP_ORDER = [
    "base_neume",
    "rest",
    "modifier_gorgon",
    "modifier_isson",
    "modifier_modulation",
    "mode",
    "key_signature",
]

GROUP_LABELS = {
    "base_neume": "Base Neumes",
    "rest": "Rests",
    "modifier_gorgon": "Gorgon / Argon / Apli / Accidentals",
    "modifier_isson": "Isson (pitch indicators)",
    "modifier_modulation": "Modulation markers",
    "mode": "Mode indicators",
    "key_signature": "Key signatures",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classes", type=Path, default=Path("config/classes.yaml"))
    parser.add_argument("--symbol-map", type=Path, default=Path("config/symbol_map.json"))
    parser.add_argument("--output", type=Path, default=Path("data/reference_sheet.html"))
    return parser.parse_args()


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


def find_gif(icon: str) -> Path | None:
    p = ICONS_DIR / f"{icon}.gif"
    if p.exists():
        return p
    for sub in KEYS_DIR.iterdir():
        if sub.is_dir():
            p = sub / f"{icon}.gif"
            if p.exists():
                return p
    return None


def gif_to_b64(path: Path) -> str:
    img = Image.open(path).convert("RGB").resize((THUMB_SIZE, THUMB_SIZE), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def render_glyph_b64(insert: str, font: ImageFont.FreeTypeFont) -> str:
    """Render insert string with font onto white canvas, return base64 PNG."""
    dummy = Image.new("RGB", (1, 1))
    bbox = ImageDraw.Draw(dummy).textbbox((0, 0), insert, font=font)
    w = max(bbox[2] - bbox[0] + 8, 4)
    h = max(bbox[3] - bbox[1] + 8, 4)
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((-bbox[0] + 4, -bbox[1] + 4), insert, font=font, fill=(0, 0, 0))
    img = img.resize((THUMB_SIZE, THUMB_SIZE), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def build_html(classes: list[str], icon_to_insert: dict[str, str], font: ImageFont.FreeTypeFont) -> str:
    # Group classes
    groups: dict[str, list[str]] = {g: [] for g in GROUP_ORDER}
    for cls in classes:
        group = cls.split(".", 1)[0]
        groups.setdefault(group, []).append(cls)

    sections = []
    for group in GROUP_ORDER:
        items = groups.get(group, [])
        if not items:
            continue
        cards = []
        for cls in items:
            icon = cls.split(".", 1)[1]
            gif = find_gif(icon)
            if gif:
                b64 = gif_to_b64(gif)
            else:
                insert = icon_to_insert.get(icon, "?")
                b64 = render_glyph_b64(insert, font)
            short = icon
            cards.append(
                f'<div class="card" title="{cls}">'
                f'<img src="data:image/png;base64,{b64}" width="{THUMB_SIZE}" height="{THUMB_SIZE}" alt="{icon}"/>'
                f'<div class="label">{short}</div>'
                f'</div>'
            )
        title = GROUP_LABELS.get(group, group)
        sections.append(
            f'<h2>{title}</h2>'
            f'<div class="group">{"".join(cards)}</div>'
        )

    body = "\n".join(sections)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>Psaltica OCR — Symbol Reference</title>
<style>
  body {{ font-family: sans-serif; background: #f8f8f8; padding: 16px; }}
  h2 {{ border-bottom: 2px solid #ccc; padding-bottom: 4px; margin-top: 32px; color: #333; }}
  .group {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px; }}
  .card {{ background: white; border: 1px solid #ddd; border-radius: 4px;
           padding: 6px; text-align: center; width: {THUMB_SIZE + 16}px;
           cursor: default; transition: box-shadow 0.1s; }}
  .card:hover {{ box-shadow: 0 2px 8px rgba(0,0,0,0.2); }}
  .label {{ font-size: 10px; color: #555; word-break: break-all;
            margin-top: 4px; line-height: 1.2; }}
</style>
</head>
<body>
<h1>Psaltica OCR — Symbol Reference ({len(classes)} classes)</h1>
<p>Hover over any card to see the full class label in the tooltip.</p>
{body}
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    with args.classes.open(encoding="utf-8") as f:
        classes = yaml.safe_load(f)["names"]
    icon_to_insert = load_symbol_map(args.symbol_map)
    font = ImageFont.truetype(str(FONT_PATH), size=GLYPH_SIZE)

    html = build_html(classes, icon_to_insert, font)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(f"Wrote reference sheet ({len(classes)} classes) → {args.output}")


if __name__ == "__main__":
    main()
