"""Tailored review/correction UI for Psaltica OCR detections (psaltica-ocr-0bg).

Run with:
    uv run streamlit run review_ui/app.py

v1 scope: load each page with its predicted boxes (autolabel now, trained
detector later), add/move/resize/delete boxes on a canvas, assign each box's
class by sight (the Byzantine glyph is shown next to the crop), and save
corrections as YOLO so they feed detector retraining via the same dataset
layout as tools/import_labels.py. Lyric rows get a light text/script panel.

The geometry source of truth is the canvas; classes are tracked per box index
in session state. Heavy maths lives in review_ui/review_io.py (unit-tested).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image


def _install_canvas_compat() -> None:
    """Shim for streamlit-drawable-canvas 0.9.3 on Streamlit >= 1.30.

    The canvas calls ``streamlit.elements.image.image_to_url(image, width, ...)``,
    which moved to ``streamlit.elements.lib.image_utils`` and now takes a
    ``LayoutConfig`` instead of an int width. Re-expose the old name/signature.
    """

    import streamlit.elements.image as st_image

    if hasattr(st_image, "image_to_url"):
        return
    from streamlit.elements.lib.image_utils import image_to_url as _new_image_to_url
    from streamlit.elements.lib.layout_utils import LayoutConfig

    def image_to_url(image, width, clamp, channels, output_format, image_id):
        return _new_image_to_url(image, LayoutConfig(width=width), clamp, channels, output_format, image_id)

    st_image.image_to_url = image_to_url


_install_canvas_compat()

from streamlit_drawable_canvas import st_canvas  # noqa: E402

from review_ui import review_io as rio  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAGES_FILE = ROOT / "data/annotations/pages_50.txt"
DEFAULT_PREDICTIONS = ROOT / "data/annotations/predictions_50.json"
DEFAULT_CLASSES = ROOT / "config/classes.yaml"
DEFAULT_SYMBOL_MAP = ROOT / "config/symbol_map.json"
DEFAULT_CORRECTIONS = ROOT / "data/corrections"


@st.cache_data
def _class_names(path: str) -> list[str]:
    return rio.load_class_names(path)


@st.cache_data
def _icon_inserts(path: str) -> dict[str, str]:
    from psaltica_ocr.template_matching import load_symbol_map

    return load_symbol_map(Path(path))


@st.cache_data
def _predictions(path: str) -> dict[str, list[dict]]:
    return rio.ls_predictions_by_image(path) if Path(path).exists() else {}


@st.cache_data
def _glyph_uri(cls: str, symbol_map_path: str) -> str | None:
    return rio.class_glyph_datauri(cls, _icon_inserts(symbol_map_path))


def _load_pages(pages_file: Path) -> list[str]:
    if not pages_file.exists():
        return []
    return [line.strip() for line in pages_file.read_text(encoding="utf-8").splitlines() if line.strip()]


def _initial_boxes(image_path: str, width: int, height: int, classes: list[str], preds: dict) -> list[rio.Box]:
    saved = rio.load_page_detections(DEFAULT_CORRECTIONS, image_path, classes, width, height)
    if saved is not None:
        return saved
    key = image_path
    if key not in preds:  # predictions are keyed by the ?d= path
        key = next((k for k in preds if k.endswith(Path(image_path).name)), None)
    return rio.boxes_from_ls_results(preds.get(key, []), width, height) if key else []


def main() -> None:
    st.set_page_config(page_title="Psaltica OCR Review", layout="wide")
    st.title("Psaltica OCR — detection review")

    classes = _class_names(str(DEFAULT_CLASSES))
    preds = _predictions(str(DEFAULT_PREDICTIONS))
    pages = _load_pages(DEFAULT_PAGES_FILE)

    with st.sidebar:
        st.header("Page")
        if not pages:
            st.error(f"No pages listed in {DEFAULT_PAGES_FILE}")
            st.stop()
        image_path = st.selectbox("Page", pages, format_func=lambda p: rio.page_key(p))
        display_width = st.slider("Display width (px)", 700, 1600, 1000, 50)
        group_filter = st.selectbox("Class group filter", ["(all)"] + sorted(rio.GROUP_COLORS))
        per_view = st.slider("Boxes per panel page", 10, 60, 25, 5)
        saved_pages = sorted(p.name for p in DEFAULT_CORRECTIONS.glob("*/") if (p / "detections.yolo").exists())
        st.caption(f"Saved: {len(saved_pages)} pages")

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        st.error(f"Cannot read {image_path}")
        st.stop()
    height, width = image.shape
    scale = display_width / width
    pil = Image.fromarray(image).convert("RGB").resize((display_width, int(height * scale)))

    page = rio.page_key(image_path)
    state_key = f"classes::{page}"
    if state_key not in st.session_state:
        boxes = _initial_boxes(image_path, width, height, classes, preds)
        st.session_state[state_key] = [b.cls for b in boxes]
        st.session_state[f"init::{page}"] = rio.boxes_to_canvas_objects(boxes, scale)

    drawing_mode = st.radio("Tool", ["transform", "rect"], horizontal=True,
                            help="transform: select/move/resize/delete · rect: draw a new box")
    canvas = st_canvas(
        background_image=pil,
        drawing_mode=drawing_mode,
        initial_drawing={"version": "4.4.0", "objects": st.session_state[f"init::{page}"]},
        fill_color="rgba(0,0,0,0)",
        stroke_color="#555555",
        stroke_width=2,
        height=int(height * scale),
        width=display_width,
        update_streamlit=True,
        key=f"canvas::{page}",
    )

    objects = (canvas.json_data or {}).get("objects", []) if canvas.json_data else []
    stored = st.session_state[state_key]
    # Reconcile class list length with the current object count.
    if len(stored) < len(objects):
        stored += [""] * (len(objects) - len(stored))
    elif len(stored) > len(objects):
        stored = stored[: len(objects)]
    st.session_state[state_key] = stored

    st.markdown(f"**{len(objects)} boxes** on this page · classes assigned: "
                f"{sum(1 for c in stored if c)}/{len(objects)}")

    # ---- class assignment panel (paginated) ----
    indices = list(range(len(objects)))
    if group_filter != "(all)":
        indices = [i for i in indices if rio.group_of(stored[i]) == group_filter]
    total_views = max(1, (len(indices) + per_view - 1) // per_view)
    view = st.number_input("Panel page", 1, total_views, 1) - 1
    view_indices = indices[view * per_view : (view + 1) * per_view]

    for i in view_indices:
        obj = objects[i]
        box = rio.canvas_objects_to_boxes([obj], scale)[0]
        cols = st.columns([1, 1, 3])
        crop = image[max(0, int(box.y1)): int(box.y2), max(0, int(box.x1)): int(box.x2)]
        if crop.size:
            cols[0].image(crop, caption=f"#{i}", width=80)
        uri = _glyph_uri(stored[i], str(DEFAULT_SYMBOL_MAP)) if stored[i] else None
        if uri:
            cols[1].markdown(f"<img src='{uri}' width='56'>", unsafe_allow_html=True)
        default = classes.index(stored[i]) if stored[i] in classes else 0
        stored[i] = cols[2].selectbox(f"class #{i}", classes, index=default, key=f"cls::{page}::{i}",
                                      label_visibility="collapsed")
    st.session_state[state_key] = stored

    # ---- actions ----
    c1, c2 = st.columns(2)
    if c1.button("💾 Save page corrections", type="primary"):
        boxes = rio.canvas_objects_to_boxes(objects, scale, stored)
        boxes = [b for b in boxes if b.cls]  # drop unassigned
        path = rio.save_page_detections(DEFAULT_CORRECTIONS, image_path, boxes, classes, width, height)
        st.success(f"Saved {len(boxes)} boxes → {path}")
    if c2.button("📦 Export YOLO dataset"):
        out = _export_dataset(classes)
        st.success(f"Exported dataset → {out}")


def _export_dataset(class_names: list[str]) -> Path:
    """Assemble images/labels train split + dataset.yaml from saved corrections."""
    output = ROOT / "data/datasets/review_export"
    (output / "images/train").mkdir(parents=True, exist_ok=True)
    (output / "labels/train").mkdir(parents=True, exist_ok=True)
    (output / "images/val").mkdir(parents=True, exist_ok=True)
    (output / "labels/val").mkdir(parents=True, exist_ok=True)
    for meta_path in DEFAULT_CORRECTIONS.glob("*/detections_meta.json"):
        import json

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        src = Path(meta["image_path"])
        if not src.exists():
            continue
        stem = rio.page_key(src)
        img = Image.open(src).convert("RGB")
        img.save(output / "images/train" / f"{stem}.png")
        (output / "labels/train" / f"{stem}.txt").write_text(
            (meta_path.parent / "detections.yolo").read_text(encoding="utf-8"), encoding="utf-8"
        )
    rio.write_dataset_yaml(output, class_names)
    return output


if __name__ == "__main__":
    main()
