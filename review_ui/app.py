"""Tailored review/correction UI for Psaltica OCR detections (psaltica-ocr-0bg).

Run with:
    uv run streamlit run review_ui/app.py

The page renders inside a scrollable, zoomable drawing canvas
(streamlit-drawable-canvas). You click-drag a rectangle straight onto a glyph to
add a box; zooming scales the image itself and the pane scrolls both ways — there
is no panning UI. Existing boxes are baked into the canvas background (numbered),
and their class is set — or they're deleted — from the list below, where the
Byzantine glyph is shown beside the crop. A freshly drawn box can auto-fill its
class from glyphs you've already labelled this session. Corrections save as YOLO
so they feed detector retraining via the same dataset layout as import_labels.py.

The canvas only ever holds newly drawn rects: seeding it with its own boxes as
fabric objects sent it into an infinite remount loop, so existing boxes live in
the background image instead. Coordinate maths lives, tested, in review_io.py.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from review_ui import review_io as rio


def _patch_drawable_canvas() -> None:
    """streamlit-drawable-canvas 0.9.3 calls the pre-1.50 ``image_to_url(image,
    width, ...)``; Streamlit 1.57 relocated it to ``elements.lib.image_utils``
    and replaced the int ``width`` with a ``LayoutConfig``. Re-expose a
    width-compatible shim on the old path so the canvas background renders."""
    import streamlit.elements.image as st_image

    if hasattr(st_image, "image_to_url"):
        return
    from streamlit.elements.lib.image_utils import image_to_url as _new
    from streamlit.elements.lib.layout_utils import LayoutConfig

    def image_to_url(image, width, clamp, channels, output_format, image_id):  # noqa: ANN001
        return _new(image, LayoutConfig(width=width), clamp, channels, output_format, image_id)

    st_image.image_to_url = image_to_url


_patch_drawable_canvas()

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAGES_FILE = ROOT / "data/annotations/pages_50.txt"
DEFAULT_PREDICTIONS = ROOT / "data/annotations/predictions_50.json"
DEFAULT_CLASSES = ROOT / "config/classes.yaml"
DEFAULT_SYMBOL_MAP = ROOT / "config/symbol_map.json"
DEFAULT_KEY_ASSETS = ROOT / "config/key_assets.json"
CONFIG_DIR = ROOT / "config"
DEFAULT_CORRECTIONS = ROOT / "data/corrections"
UNASSIGNED = "— pick class —"


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
def _key_assets() -> dict[str, str]:
    return rio.load_key_assets(DEFAULT_KEY_ASSETS)


@st.cache_data
def _glyph_uri(cls: str, symbol_map_path: str) -> str | None:
    # Key signatures render as GIF artwork in the app; prefer that (the font
    # glyph for their insert often differs). Fall back to the font otherwise.
    asset = rio.key_asset_datauri(cls, _key_assets(), str(CONFIG_DIR))
    if asset:
        return asset
    return rio.class_glyph_datauri(cls, _icon_inserts(symbol_map_path))


@st.cache_resource
def _glyph_descriptors(classes_path: str, symbol_map_path: str) -> dict:
    return rio.build_glyph_descriptors(_class_names(classes_path), _icon_inserts(symbol_map_path))


@st.cache_data
def _symbol_estimate(image_path: str) -> int:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return rio.estimate_symbol_count(img) if img is not None else 0


def _session_exemplars(image, boxes, per_class: int = 6) -> list:
    """(class, descriptor) of crops you've labelled this session — across pages,
    capped per class. These match the real printed typeface, unlike font glyphs."""
    cache = st.session_state.setdefault("exemplar_desc", {})
    for box in boxes:
        if box.cls:
            key = (box.uid, box.cls)
            if key not in cache:
                crop = image[max(0, int(box.y1)): int(box.y2), max(0, int(box.x1)): int(box.x2)]
                if crop.size:
                    cache[key] = rio.crop_descriptor(crop)
    by_class: dict[str, list] = {}
    for (_uid, cls), desc in cache.items():
        by_class.setdefault(cls, []).append(desc)
    return [(cls, desc) for cls, descs in by_class.items() for desc in descs[-per_class:]]


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


def _background(image_gray: np.ndarray, boxes: list[rio.Box], disp_w: int, disp_h: int,
                scale: float, highlight: set[int]):
    """The zoomed page with existing boxes baked in, as an RGB PIL image.

    Boxes are drawn into the background (not handed to the canvas as fabric
    objects) so the canvas only ever holds freshly drawn rects — feeding its own
    output back as ``initial_drawing`` is what sent it into an infinite remount
    loop. Numbers tie each box back to its row in the correction list below.
    """
    from PIL import Image

    canvas = cv2.resize(cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR), (disp_w, disp_h))
    for i, box in enumerate(boxes):
        color = _hex_to_bgr(box.color())
        p1 = (int(box.x1 * scale), int(box.y1 * scale))
        p2 = (int(box.x2 * scale), int(box.y2 * scale))
        cv2.rectangle(canvas, p1, p2, color, 3 if i in highlight else 2)
        cv2.putText(canvas, str(i), (p1[0], max(12, p1[1] - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))


def _preserve_scroll(container_class: str) -> None:
    """Keep the scroll position of the canvas pane across reruns.

    Adding a box no longer remounts the canvas, but zooming/page changes do, and a
    rerun can still nudge the scrollable pane. This stores the pane's scroll offset
    on the window and restores it, retrying briefly while content reflows, so you
    keep your place when zoomed in.
    """
    st.html(
        f"""
        <script>
        (function() {{
            const cls = "{container_class}";
            const win = window.parent || window, doc = win.document;
            win.__canvasScroll = win.__canvasScroll || {{x: 0, y: 0}};
            const getEl = () => doc.getElementsByClassName(cls)[0];
            function restore(tries) {{
                const el = getEl();
                if (!el) {{ if (tries > 0) setTimeout(() => restore(tries - 1), 50); return; }}
                if (!el.__scrollBound) {{
                    el.__scrollBound = true;
                    el.addEventListener("scroll", () => {{
                        win.__canvasScroll = {{x: el.scrollLeft, y: el.scrollTop}};
                    }});
                }}
                el.scrollLeft = win.__canvasScroll.x;
                el.scrollTop = win.__canvasScroll.y;
                if (tries > 0) setTimeout(() => restore(tries - 1), 50);  // outlast reflow
            }}
            restore(24);
        }})();
        </script>
        """,
        unsafe_allow_javascript=True,
    )


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
        zoom = st.slider("🔍 Zoom", 0.25, 3.0, 1.0, 0.25,
                         help="Zooms the image itself. The view scrolls — drag inside it to draw.")
        viewport_h = st.slider("Viewport height (px)", 400, 1200, 750, 50,
                               help="Height of the scrollable image pane.")
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

    exemplars = _session_exemplars(image, boxes)
    glyph_descriptors = _glyph_descriptors(str(DEFAULT_CLASSES), str(DEFAULT_SYMBOL_MAP))

    def guess_cls(crop) -> str:
        # Score the crop against what you've labelled this session (real
        # typeface) AND against the font glyphs, then take the better match.
        # The font glyphs cover every class; without them an unlabelled class is
        # forced onto its nearest labelled neighbour — e.g. an Eteron with no
        # Eteron exemplar yet always lands on Antikenoma. Both descriptors share
        # the same space and metric, so their scores are directly comparable.
        ec, es = rio.guess_from_exemplars(crop, exemplars)
        gc, gs = rio.guess_class(crop, glyph_descriptors)
        cls, score = (gc, gs) if gs > es else (ec, es)
        return cls if cls and score >= 0.40 else ""

    undo_key, redo_key = f"undo::{page}", f"redo::{page}"
    clear_key = f"clearver::{page}"
    st.session_state.setdefault(clear_key, 0)

    def snapshot() -> None:
        st.session_state.setdefault(undo_key, []).append(copy.deepcopy(boxes))
        st.session_state[redo_key] = []

    def clear_canvas() -> None:
        # Wipe any drawn rect by bumping the canvas's initial_drawing version.
        # This keeps the SAME canvas key, so the iframe is NOT remounted — the
        # page doesn't flash or lose its scroll position; only the background
        # image (with the committed boxes baked in) updates in place.
        st.session_state[clear_key] += 1

    with st.sidebar:
        st.divider()
        n_pred = len(_prediction_boxes(image_path, width, height, preds))
        if st.button(f"Load autolabel predictions ({n_pred})", disabled=n_pred == 0,
                     help="Template-match boxes — they over-detect, so from-scratch is often easier."):
            snapshot()
            st.session_state[box_key] = _prediction_boxes(image_path, width, height, preds)
            clear_canvas()
            st.rerun()
        u, r = st.columns(2)
        if u.button("↩️ Undo", disabled=not st.session_state.get(undo_key)):
            st.session_state.setdefault(redo_key, []).append(copy.deepcopy(boxes))
            st.session_state[box_key] = st.session_state[undo_key].pop()
            clear_canvas()
            st.rerun()
        if r.button("↪️ Redo", disabled=not st.session_state.get(redo_key)):
            st.session_state.setdefault(undo_key, []).append(copy.deepcopy(boxes))
            st.session_state[box_key] = st.session_state[redo_key].pop()
            clear_canvas()
            st.rerun()
        if st.button("Clear all boxes on this page"):
            snapshot()
            st.session_state[box_key] = []
            clear_canvas()
            st.rerun()

    # Panel pages are derived from the box count — they grow/shrink automatically
    # as you add or delete. The current page is kept in session (not a widget) so
    # the ◀/▶ selector below the list drives it. Newest box first (reversed), so a
    # freshly drawn box lands at the top of page 1 where you can check it.
    indices = [i for i in range(len(boxes)) if group_filter == "(all)" or boxes[i].group == group_filter][::-1]
    total_views = max(1, (len(indices) + per_view - 1) // per_view)
    view_key = f"view::{page}"
    view = min(st.session_state.setdefault(view_key, 0), total_views - 1)
    st.session_state[view_key] = view
    view_indices = indices[view * per_view : (view + 1) * per_view]

    # Apply a pending re-guess BEFORE any class selectbox is created (Streamlit
    # forbids setting a widget's state after the widget exists).
    pending_uid = st.session_state.pop("guess_uid", None)
    if pending_uid:
        for b in boxes:
            if b.uid == pending_uid:
                guess = guess_cls(image[int(b.y1):int(b.y2), int(b.x1):int(b.x2)])
                if guess:
                    b.cls = guess
                    st.session_state[f"cls::{b.uid}"] = guess
                break

    auto_guess = st.checkbox("🔮 Auto-guess drawn boxes from your labels", value=True,
                             help="A drawn box is matched against glyphs you've already labelled this "
                                  "session; it auto-fills only on a confident match, else stays "
                                  "unassigned. Label the first of each glyph and the repeats fill in.")
    if exemplars:
        st.caption(f"learning from {len(exemplars)} labelled examples across "
                   f"{len({c for c, _ in exemplars})} glyph classes")

    from streamlit_drawable_canvas import st_canvas

    disp_w, disp_h = max(1, int(width * zoom)), max(1, int(height * zoom))
    scale = disp_w / width  # canvas pixels per image pixel (~zoom, exact after rounding)
    st.caption("Click-drag on the image to add a box · the pane scrolls both ways · zoom with the sidebar slider. "
               "Existing boxes are numbered — edit their class or delete them in the list below.")

    # The height-limited container scrolls vertically but clips horizontally, and
    # the component iframe is otherwise sized to the column width (clipping a
    # zoomed-in canvas). Let both scroll horizontally to the canvas width.
    st.markdown(
        f"<style>"
        f".st-key-canvasscroll {{ overflow: auto !important; }}"
        f".st-key-canvasscroll iframe {{ width: {disp_w}px !important; max-width: none !important; }}"
        f"</style>",
        unsafe_allow_html=True,
    )
    with st.container(height=viewport_h, border=True, key="canvasscroll"):
        # The canvas only ever holds NEW rects: existing boxes are baked into the
        # background, never seeded as fabric objects (that loops). The key is
        # STABLE across adds (only page/zoom change it) so the iframe isn't
        # remounted — instead a drawn rect is cleared by bumping initial_drawing's
        # version, which the component reloads in place without flashing/scrolling.
        canvas = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",
            stroke_width=2,
            stroke_color="#00cc00",
            background_image=_background(image, boxes, disp_w, disp_h, scale, set(view_indices)),
            update_streamlit=True,
            height=disp_h,
            width=disp_w,
            drawing_mode="rect",
            initial_drawing={"version": f"clear-{st.session_state[clear_key]}", "objects": []},
            display_toolbar=True,
            key=f"canvas::{page}::{disp_w}",
        )
    _preserve_scroll("st-key-canvasscroll")

    if canvas.json_data is not None:
        # Any rect on the (otherwise empty) canvas is a new box; drop stray clicks.
        drawn = [b for b in rio.canvas_objects_to_boxes(canvas.json_data.get("objects", []), scale)
                 if b.width >= 3 and b.height >= 3]
        # Guard: after committing, the canvas keeps emitting the same rect until it
        # clears, so skip if we've already committed exactly this set.
        sig = tuple(sorted((round(b.x1), round(b.y1), round(b.x2), round(b.y2)) for b in drawn))
        if drawn and sig != st.session_state.get(f"commitsig::{page}"):
            st.session_state[f"commitsig::{page}"] = sig
            snapshot()
            for nb in drawn:
                cls = guess_cls(image[int(nb.y1):int(nb.y2), int(nb.x1):int(nb.x2)]) if auto_guess else ""
                boxes.append(rio.Box(cls, nb.x1, nb.y1, nb.x2, nb.y2, source="added"))
            clear_canvas()  # wipe the drawn rect; it is now baked into the background
            st.session_state[view_key] = 0  # newest is first — show the top page
            st.rerun()
        elif not drawn:
            # Canvas cleared; forget the guard so an identical next box still adds.
            st.session_state[f"commitsig::{page}"] = None

    # Progress against the estimated number of symbols on the page (counted from
    # the music rows), so the bar reflects real remaining work rather than the
    # share of boxes drawn so far. The estimate is approximate, so a fully
    # labelled page can read slightly under or over — clamp to 100% and never let
    # it fall below what's already labelled.
    assigned = sum(1 for b in boxes if b.cls)
    expected = max(_symbol_estimate(image_path), assigned, 1)
    pct = min(assigned / expected, 1.0)
    st.progress(pct, text=f"~{pct * 100:.0f}% done · {assigned} labelled of ≈{expected} symbols")

    # Panel-page navigation: ◀/▶ step, a dropdown to jump anywhere, and a live
    # position readout. Pages are created/removed automatically with the boxes.
    nav_prev, nav_sel, nav_next = st.columns([1, 3, 1])
    if nav_prev.button("◀ Prev", disabled=view == 0, use_container_width=True):
        st.session_state[view_key] = view - 1
        st.rerun()
    def _page_label(v: int) -> str:
        chunk = indices[v * per_view : (v + 1) * per_view]
        return (f"Page {v + 1} of {total_views}  ·  boxes #{min(chunk)}–#{max(chunk)}"
                if chunk else f"Page {v + 1} of {total_views}")

    page_labels = [_page_label(v) for v in range(total_views)]
    chosen = nav_sel.selectbox("Panel page", page_labels, index=view, label_visibility="collapsed",
                               key=f"panelsel::{page}::{view}::{total_views}")
    if (sel := page_labels.index(chosen)) != view:
        st.session_state[view_key] = sel
        st.rerun()
    if nav_next.button("Next ▶", disabled=view >= total_views - 1, use_container_width=True):
        st.session_state[view_key] = view + 1
        st.rerun()

    for i in view_indices:
        box = boxes[i]
        cols = st.columns([1, 1, 4, 1, 1])
        crop = image[max(0, int(box.y1)): int(box.y2), max(0, int(box.x1)): int(box.x2)]
        if crop.size:
            cols[0].image(crop, caption=f"#{i}", width=64)
        options = [UNASSIGNED] + classes
        default = options.index(box.cls) if box.cls in classes else 0
        choice = cols[2].selectbox(f"class #{i}", options, index=default,
                                   key=f"cls::{box.uid}", label_visibility="collapsed")
        box.cls = "" if choice == UNASSIGNED else choice
        uri = _glyph_uri(box.cls, str(DEFAULT_SYMBOL_MAP))
        if uri:
            cols[1].markdown(f"<img src='{uri}' width='52'>", unsafe_allow_html=True)
        if cols[3].button("🔮", key=f"guess::{box.uid}", help="Guess from your labelled examples"):
            st.session_state["guess_uid"] = box.uid
            st.rerun()
        if cols[4].button("🗑", key=f"del::{box.uid}", help="Delete this box"):
            snapshot()
            boxes.pop(i)
            clear_canvas()
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
