"""Tailored review/correction UI for Psaltica OCR detections (psaltica-ocr-0bg).

Run with:
    uv run streamlit run review_ui/app.py

v1 is correction-first and canvas-free: the page renders with its predicted
boxes overlaid (a plain image, so there is no click-coordinate offset), and each
box is corrected from a list — fix its class by sight (the Byzantine glyph is
shown beside the crop) or delete a false positive. Missed glyphs are added with
a small numeric form. Corrections save as YOLO so they feed detector retraining
via the same dataset layout as tools/import_labels.py.

(An earlier canvas-based draw/move/resize version was dropped: streamlit
-drawable-canvas mis-scales click coordinates inside Streamlit's iframe on
HiDPI displays. Geometry maths still lives, tested, in review_ui/review_io.py.)
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from review_ui import review_io as rio

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


def _prediction_boxes(image_path: str, width: int, height: int, preds: dict) -> list[rio.Box]:
    key = image_path if image_path in preds else next((k for k in preds if k.endswith(Path(image_path).name)), None)
    return rio.boxes_from_ls_results(preds.get(key, []), width, height) if key else []


def _initial_boxes(image_path: str, width: int, height: int, classes: list[str]) -> list[rio.Box]:
    # Saved corrections resume; otherwise start empty (from scratch). Autolabel
    # predictions are opt-in per page because template matching over-detects.
    saved = rio.load_page_detections(DEFAULT_CORRECTIONS, image_path, classes, width, height)
    return saved if saved is not None else []


def _hex_to_bgr(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    r, g, b = (int(color[i : i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


def _overlay(image_gray: np.ndarray, boxes: list[rio.Box], highlight: set[int], display_width: int) -> np.ndarray:
    canvas = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    for i, box in enumerate(boxes):
        color = _hex_to_bgr(box.color())
        thickness = 5 if i in highlight else 2
        cv2.rectangle(canvas, (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2)), color, thickness)
        cv2.putText(canvas, str(i), (int(box.x1), max(12, int(box.y1) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    scale = display_width / canvas.shape[1]
    resized = cv2.resize(canvas, (display_width, int(canvas.shape[0] * scale)))
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)


def main() -> None:
    st.set_page_config(page_title="Psaltica OCR Review", layout="wide")
    st.title("Psaltica OCR — detection review")

    classes = _class_names(str(DEFAULT_CLASSES))
    preds = _predictions(str(DEFAULT_PREDICTIONS))
    pages = _load_pages(DEFAULT_PAGES_FILE)
    if not pages:
        st.error(f"No pages listed in {DEFAULT_PAGES_FILE}")
        st.stop()

    with st.sidebar:
        st.header("Page")
        image_path = st.selectbox("Page", pages, format_func=rio.page_key)
        display_width = st.slider("Overlay width (px)", 500, 1500, 900, 50,
                                  help="Visual only — the overlay is a static image, no click mapping.")
        group_filter = st.selectbox("Class group filter", ["(all)"] + sorted(rio.GROUP_COLORS))
        per_view = st.slider("Boxes per panel page", 10, 60, 20, 5)
        saved = sorted(p.name for p in DEFAULT_CORRECTIONS.glob("*") if (p / "detections.yolo").exists())
        st.caption(f"Saved: {len(saved)} pages")

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        st.error(f"Cannot read {image_path}")
        st.stop()
    height, width = image.shape
    page = rio.page_key(image_path)

    box_key = f"boxes::{page}"
    if box_key not in st.session_state:
        st.session_state[box_key] = _initial_boxes(image_path, width, height, classes)
    boxes: list[rio.Box] = st.session_state[box_key]

    with st.sidebar:
        st.divider()
        n_pred = len(_prediction_boxes(image_path, width, height, preds))
        if st.button(f"Load autolabel predictions ({n_pred})", disabled=n_pred == 0,
                     help="Template-match boxes — they over-detect, so from-scratch is often easier."):
            st.session_state[box_key] = _prediction_boxes(image_path, width, height, preds)
            st.rerun()
        if st.button("Clear all boxes on this page"):
            st.session_state[box_key] = []
            st.rerun()

    indices = [i for i in range(len(boxes)) if group_filter == "(all)" or boxes[i].group == group_filter]
    total_views = max(1, (len(indices) + per_view - 1) // per_view)
    view = st.sidebar.number_input("Panel page", 1, total_views, 1) - 1
    view_indices = indices[view * per_view : (view + 1) * per_view]

    overlay = _overlay(image, boxes, set(view_indices), display_width)
    draw_mode = st.checkbox("✏️ Draw mode — drag a rectangle on the page to add a box")
    new_cls = st.selectbox("New-box class (for boxes you draw)", classes, key="add_cls")
    glyph = _glyph_uri(new_cls, str(DEFAULT_SYMBOL_MAP))
    if glyph:
        st.markdown(f"new box → <img src='{glyph}' width='40'> `{new_cls}`", unsafe_allow_html=True)

    if draw_mode:
        from streamlit_image_coordinates import streamlit_image_coordinates

        st.caption("Drag from one corner of the missed glyph to the opposite corner. "
                   "The new box gets the class selected above; fix it in the list if needed.")
        result = streamlit_image_coordinates(overlay, width=display_width, click_and_drag=True, key=f"draw::{page}")
        if result and result.get("x2") is not None:
            sig = (result["x1"], result["y1"], result["x2"], result["y2"])
            big = abs(result["x2"] - result["x1"]) > 2 and abs(result["y2"] - result["y1"]) > 2
            if big and st.session_state.get(f"lastdraw::{page}") != sig:
                st.session_state[f"lastdraw::{page}"] = sig
                factor = width / display_width  # coords come back in the displayed-image space
                xs = sorted([result["x1"] * factor, result["x2"] * factor])
                ys = sorted([result["y1"] * factor, result["y2"] * factor])
                boxes.append(rio.Box(new_cls, max(0, xs[0]), max(0, ys[0]),
                                     min(width, xs[1]), min(height, ys[1]), source="added"))
                st.rerun()
    else:
        st.image(overlay, width=display_width,
                 caption=f"{page} — {len(boxes)} boxes (current panel page highlighted)")

    st.markdown(f"**{len(boxes)} boxes** · classes assigned: {sum(1 for b in boxes if b.cls)} · "
                f"showing #{view_indices[0] if view_indices else 0}–{view_indices[-1] if view_indices else 0}")

    for i in view_indices:
        box = boxes[i]
        cols = st.columns([1, 1, 4, 1])
        crop = image[max(0, int(box.y1)): int(box.y2), max(0, int(box.x1)): int(box.x2)]
        if crop.size:
            cols[0].image(crop, caption=f"#{i}", width=64)
        default = classes.index(box.cls) if box.cls in classes else 0
        box.cls = cols[2].selectbox(f"class #{i}", classes, index=default,
                                    key=f"cls::{page}::{i}", label_visibility="collapsed")
        uri = _glyph_uri(box.cls, str(DEFAULT_SYMBOL_MAP))
        if uri:
            cols[1].markdown(f"<img src='{uri}' width='52'>", unsafe_allow_html=True)
        if cols[3].button("🗑", key=f"del::{page}::{i}", help="Delete this box"):
            boxes.pop(i)
            st.rerun()

    c1, c2 = st.columns(2)
    if c1.button("💾 Save page corrections", type="primary"):
        keep = [b for b in boxes if b.cls]
        path = rio.save_page_detections(DEFAULT_CORRECTIONS, image_path, keep, classes, width, height)
        st.success(f"Saved {len(keep)} boxes → {path}")
    if c2.button("📦 Export YOLO dataset"):
        st.success(f"Exported dataset → {_export_dataset(classes)}")


def _export_dataset(class_names: list[str]) -> Path:
    from PIL import Image

    output = ROOT / "data/datasets/review_export"
    for sub in ("images/train", "labels/train", "images/val", "labels/val"):
        (output / sub).mkdir(parents=True, exist_ok=True)
    for meta_path in DEFAULT_CORRECTIONS.glob("*/detections_meta.json"):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        src = Path(meta["image_path"])
        if not src.exists():
            continue
        stem = rio.page_key(src)
        Image.open(src).convert("RGB").save(output / "images/train" / f"{stem}.png")
        (output / "labels/train" / f"{stem}.txt").write_text(
            (meta_path.parent / "detections.yolo").read_text(encoding="utf-8"), encoding="utf-8"
        )
    rio.write_dataset_yaml(output, class_names)
    return output


if __name__ == "__main__":
    main()
