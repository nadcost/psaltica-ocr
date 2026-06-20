"""IO and conversions for the review/correction UI (psaltica-ocr-0bg).

Kept separate from the Streamlit app so the coordinate maths and dataset export
can be unit-tested without a browser. Coordinate spaces:

* image pixels — the page's true resolution (YOLO and corrections live here),
* percent — Label Studio prediction format from the autolabeler,
* canvas — the displayed image, scaled by ``display_width / image_width``;
  st_canvas (fabric.js) rectangles carry ``left/top/width/height`` plus
  ``scaleX/scaleY`` applied on resize.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
from uuid import uuid4

import cv2
import numpy as np
import yaml

# Stroke colour per class group, so boxes are readable on the canvas.
GROUP_COLORS: dict[str, str] = {
    "base_neume": "#1f77b4",
    "modifier_gorgon": "#d62728",
    "modifier_isson": "#9467bd",
    "modifier_modulation": "#2ca02c",
    "key_signature": "#ff7f0e",
    "rest": "#8c564b",
    "lyrics": "#17becf",
}
DEFAULT_COLOR = "#555555"


@dataclass
class Box:
    cls: str
    x1: float
    y1: float
    x2: float
    y2: float
    score: float = 1.0
    source: str = "pred"  # pred | added | edited
    uid: str = field(default_factory=lambda: uuid4().hex[:8])  # stable widget key

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def group(self) -> str:
        return self.cls.split(".", 1)[0] if self.cls else ""

    def color(self) -> str:
        return GROUP_COLORS.get(self.group, DEFAULT_COLOR)


def group_of(cls: str) -> str:
    return cls.split(".", 1)[0] if cls else ""


# --------------------------------------------------------------------------- #
# Class list and glyphs
# --------------------------------------------------------------------------- #

def load_class_names(classes_path: str | Path) -> list[str]:
    payload = yaml.safe_load(Path(classes_path).read_text(encoding="utf-8"))
    names = payload["names"]
    if isinstance(names, dict):
        return [names[index] for index in sorted(names)]
    return list(names)


GLYPH_DESC_SIZE = 48


def _descriptor(gray: np.ndarray, size: int = GLYPH_DESC_SIZE) -> np.ndarray:
    """Shape descriptor: trim to ink, pad to square (keep aspect), edge-gradient.

    Trimming whitespace and padding (instead of stretching the raw crop to a
    square) makes a hand-drawn crop and a glyph comparable regardless of how
    much empty space the box includes or the glyph's aspect ratio.
    """
    from psaltica_ocr.template_matching import to_gradient

    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)  # ink = 0
    ys, xs = np.where(binary == 0)
    if len(xs):
        binary = binary[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]
    h, w = binary.shape
    side = max(h, w, 1)
    square = np.full((side, side), 255, np.uint8)
    square[(side - h) // 2: (side - h) // 2 + h, (side - w) // 2: (side - w) // 2 + w] = binary
    resized = cv2.resize(square, (size, size), interpolation=cv2.INTER_AREA)
    return to_gradient(resized).astype(np.float32)


def build_glyph_descriptors(
    class_names: list[str], icon_inserts: dict[str, str], size: int = GLYPH_DESC_SIZE
) -> dict[str, np.ndarray]:
    """Render each class's font glyph to a shape descriptor for crop matching."""
    from psaltica_ocr.template_matching import render_template

    descriptors: dict[str, np.ndarray] = {}
    for cls in class_names:
        if "." not in cls:
            continue
        insert = icon_inserts.get(cls.split(".", 1)[1])
        if not insert:
            continue
        template = render_template(insert, 9.0)
        if template is not None:
            descriptors[cls] = _descriptor(template, size)
    return descriptors


def crop_descriptor(crop_gray: np.ndarray, size: int = GLYPH_DESC_SIZE) -> np.ndarray:
    """Public shape descriptor for a box crop (used to learn exemplars)."""
    return _descriptor(crop_gray, size)


def guess_from_exemplars(
    crop_gray: np.ndarray, exemplars: list[tuple[str, np.ndarray]], size: int = GLYPH_DESC_SIZE
) -> tuple[str | None, float]:
    """Nearest-exemplar class for a crop — exemplars are (class, descriptor) of
    already-labelled crops, so this matches the real printed typeface."""
    if crop_gray is None or crop_gray.size == 0 or not exemplars:
        return None, 0.0
    crop_desc = _descriptor(crop_gray, size)
    best_cls, best_score = None, -2.0
    for cls, desc in exemplars:
        score = float(cv2.matchTemplate(crop_desc, desc, cv2.TM_CCOEFF_NORMED)[0, 0])
        if score > best_score:
            best_cls, best_score = cls, score
    return best_cls, best_score


def guess_class(
    crop_gray: np.ndarray, descriptors: dict[str, np.ndarray], size: int = GLYPH_DESC_SIZE
) -> tuple[str | None, float]:
    """Best-matching class for a box crop (normalized gradient correlation)."""
    if crop_gray is None or crop_gray.size == 0 or not descriptors:
        return None, 0.0
    crop_desc = _descriptor(crop_gray, size)
    best_cls, best_score = None, -2.0
    for cls, desc in descriptors.items():
        score = float(cv2.matchTemplate(crop_desc, desc, cv2.TM_CCOEFF_NORMED)[0, 0])
        if score > best_score:
            best_cls, best_score = cls, score
    return best_cls, best_score


def class_glyph_datauri(cls: str, icon_inserts: dict[str, str]) -> str | None:
    """Return a data: URI PNG of the class glyph, or None if unrenderable."""
    if "." not in cls:
        return None
    insert = icon_inserts.get(cls.split(".", 1)[1])
    if not insert:
        return None
    from psaltica_ocr.template_matching import render_glyph_b64

    b64 = render_glyph_b64(insert)
    return f"data:image/png;base64,{b64}" if b64 else None


def load_key_assets(path: str | Path) -> dict[str, str]:
    """Map key name (icon or label) -> repo-relative GIF path (or {} if absent)."""
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def key_asset_datauri(cls: str, key_assets: dict[str, str], assets_root: str | Path) -> str | None:
    """Data: URI of the app's GIF artwork for a key signature, else None.

    Key signatures (and mode martyria) render as GIFs in the app, not font
    glyphs, so this returns the real artwork when the class name matches; the
    font renderer is the fallback for everything else.
    """
    if "." not in cls:
        return None
    rel = key_assets.get(cls.split(".", 1)[1])
    if not rel:
        return None
    path = Path(assets_root) / rel
    if not path.exists():
        return None
    import base64

    b64 = base64.b64encode(path.read_bytes()).decode()
    return f"data:image/gif;base64,{b64}"


# --------------------------------------------------------------------------- #
# Page completion estimate
# --------------------------------------------------------------------------- #

# Printed neume ligatures fuse adjacent clusters, so one horizontal ink column
# spans ~1.6 glyph clusters on average. Calibrated on six fully-labelled pages
# (120–317 clusters); column/cluster held at 0.58–0.66 across that range, so the
# scaled estimate lands within ~±8% of the true cluster count.
SYMBOLS_PER_COLUMN = 1.63


def _ink_column_count(ink_mask: np.ndarray, *, min_gap: int = 4, min_width: int = 2) -> int:
    """Number of horizontal ink columns in a row mask.

    Clusters stack their modifiers vertically at one x-position, so columns of
    ink track clusters far better than raw connected components do. Runs of ink
    columns closer than ``min_gap`` are one column; runs thinner than
    ``min_width`` (stray specks) are dropped.
    """
    present = ink_mask.any(axis=0)
    runs: list[list[int]] = []
    x, n = 0, len(present)
    while x < n:
        if present[x]:
            start = x
            while x < n and present[x]:
                x += 1
            if runs and start - runs[-1][1] < min_gap:
                runs[-1][1] = x
            else:
                runs.append([start, x])
        else:
            x += 1
    return sum(1 for a, b in runs if b - a >= min_width)


def estimate_symbol_count(image_gray: np.ndarray) -> int:
    """Rough count of glyph clusters in a page's music (chant) rows.

    Used to give the review progress bar a real denominator (expected symbols on
    the page) instead of a moving one (boxes drawn so far). It is an estimate for
    a gauge, not a detector: it counts horizontal ink columns inside each chant
    row and scales by ``SYMBOLS_PER_COLUMN``.
    """
    from psaltica_ocr.layout_segmentation import segment_page_layout
    from psaltica_ocr.rendering import binarize

    binary = binarize(image_gray)  # ink = 255
    layout = segment_page_layout(image_gray)
    columns = 0
    for row in layout.chant_rows:
        b = row.bbox
        columns += _ink_column_count(binary[b.y1:b.y2, b.x1:b.x2] > 0)
    return round(columns * SYMBOLS_PER_COLUMN)


# --------------------------------------------------------------------------- #
# Label Studio predictions -> pixel boxes
# --------------------------------------------------------------------------- #

def _image_key(url: str) -> str:
    marker = "?d="
    return url.split(marker, 1)[1] if marker in url else url


def ls_predictions_by_image(predictions_path: str | Path) -> dict[str, list[dict]]:
    """Map image key (the ?d= path) -> list of raw percent-coord results."""
    tasks = json.loads(Path(predictions_path).read_text(encoding="utf-8"))
    by_image: dict[str, list[dict]] = {}
    for task in tasks:
        key = _image_key(str(task.get("data", {}).get("image", "")))
        results: list[dict] = []
        for prediction in task.get("predictions", []):
            results.extend(prediction.get("result", []))
        by_image[key] = results
    return by_image


def boxes_from_ls_results(results: Iterable[dict], width: int, height: int) -> list[Box]:
    boxes: list[Box] = []
    for result in results:
        value = result.get("value") or {}
        labels = value.get("rectanglelabels") or [""]
        x = float(value.get("x", 0)) / 100 * width
        y = float(value.get("y", 0)) / 100 * height
        w = float(value.get("width", 0)) / 100 * width
        h = float(value.get("height", 0)) / 100 * height
        boxes.append(Box(labels[0], x, y, x + w, y + h, float(result.get("score", 1.0))))
    return boxes


# --------------------------------------------------------------------------- #
# YOLO <-> pixel boxes
# --------------------------------------------------------------------------- #

def boxes_to_yolo_lines(boxes: Iterable[Box], class_to_id: dict[str, int], width: int, height: int) -> list[str]:
    lines: list[str] = []
    for box in boxes:
        if box.cls not in class_to_id:
            continue
        xc = (box.x1 + box.x2) / 2 / width
        yc = (box.y1 + box.y2) / 2 / height
        bw = box.width / width
        bh = box.height / height
        lines.append(f"{class_to_id[box.cls]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    return lines


def yolo_lines_to_boxes(lines: Iterable[str], class_names: list[str], width: int, height: int) -> list[Box]:
    boxes: list[Box] = []
    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            continue
        idx, xc, yc, bw, bh = int(parts[0]), *(float(p) for p in parts[1:])
        cls = class_names[idx] if 0 <= idx < len(class_names) else ""
        cx, cy, w, h = xc * width, yc * height, bw * width, bh * height
        boxes.append(Box(cls, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, source="edited"))
    return boxes


# --------------------------------------------------------------------------- #
# Canvas (fabric.js) <-> pixel boxes
# --------------------------------------------------------------------------- #

def boxes_to_canvas_objects(boxes: Iterable[Box], scale: float) -> list[dict]:
    objects: list[dict] = []
    for box in boxes:
        objects.append(
            {
                "type": "rect",
                "left": box.x1 * scale,
                "top": box.y1 * scale,
                "width": box.width * scale,
                "height": box.height * scale,
                "scaleX": 1,
                "scaleY": 1,
                "stroke": box.color(),
                "strokeWidth": 2,
                "fill": "rgba(0,0,0,0)",
            }
        )
    return objects


def canvas_objects_to_boxes(objects: Iterable[dict], scale: float, classes: list[str] | None = None) -> list[Box]:
    """Convert fabric rects back to pixel boxes; ``classes[i]`` assigns class i."""
    classes = classes or []
    boxes: list[Box] = []
    for index, obj in enumerate(objects):
        if obj.get("type") != "rect":
            continue
        aw = float(obj.get("width", 0)) * float(obj.get("scaleX", 1))
        ah = float(obj.get("height", 0)) * float(obj.get("scaleY", 1))
        x1 = float(obj.get("left", 0)) / scale
        y1 = float(obj.get("top", 0)) / scale
        cls = classes[index] if index < len(classes) else ""
        boxes.append(Box(cls, x1, y1, x1 + aw / scale, y1 + ah / scale, source="edited"))
    return boxes


# --------------------------------------------------------------------------- #
# Per-page correction persistence + dataset export
# --------------------------------------------------------------------------- #

def page_key(image_path: str | Path) -> str:
    p = Path(image_path)
    return f"{p.parent.name.replace(' ', '_')}_{p.stem}"


def save_page_detections(
    corrections_root: str | Path,
    image_path: str | Path,
    boxes: list[Box],
    class_names: list[str],
    width: int,
    height: int,
) -> Path:
    class_to_id = {name: i for i, name in enumerate(class_names)}
    out_dir = Path(corrections_root) / page_key(image_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = boxes_to_yolo_lines(boxes, class_to_id, width, height)
    label_path = out_dir / "detections.yolo"
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    (out_dir / "detections_meta.json").write_text(
        json.dumps({"image_path": str(image_path), "width": width, "height": height, "boxes": len(lines)}, indent=2),
        encoding="utf-8",
    )
    return label_path


def load_page_detections(
    corrections_root: str | Path,
    image_path: str | Path,
    class_names: list[str],
    width: int,
    height: int,
) -> list[Box] | None:
    label_path = Path(corrections_root) / page_key(image_path) / "detections.yolo"
    if not label_path.exists():
        return None
    return yolo_lines_to_boxes(label_path.read_text(encoding="utf-8").splitlines(), class_names, width, height)


def count_class_instances(
    corrections_root: str | Path, class_names: list[str]
) -> tuple[dict[str, int], dict[str, int], int]:
    """Tally saved corrections per class for dataset-balance decisions.

    Returns ``(counts, page_coverage, out_of_range)`` where ``counts[name]`` is
    the number of boxes of that class across all saved pages, ``page_coverage``
    is how many distinct pages each class appears on, and ``out_of_range`` is the
    count of label indices that don't map to a class (stale labels). Every class
    is present in the dicts, including those with zero instances.
    """
    counts = {name: 0 for name in class_names}
    pages: dict[str, set[str]] = {name: set() for name in class_names}
    out_of_range = 0
    for label_path in Path(corrections_root).glob("*/detections.yolo"):
        page = label_path.parent.name
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            idx = int(parts[0])
            if 0 <= idx < len(class_names):
                name = class_names[idx]
                counts[name] += 1
                pages[name].add(page)
            else:
                out_of_range += 1
    return counts, {name: len(seen) for name, seen in pages.items()}, out_of_range


def write_dataset_yaml(output: Path, class_names: list[str]) -> None:
    payload = {
        "path": str(output.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(class_names)},
    }
    (output / "dataset.yaml").write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
