# Psaltica OCR review UI

Tailored correction UI for printed-score detections — the front end of the OCR
refinement loop (MVP of `psaltica-ocr-7f0`, built under `psaltica-ocr-0bg`).

## Run

```bash
uv run streamlit run review_ui/app.py
```

It expects (defaults, all under the repo):

- `data/annotations/pages_50.txt` — page list (from the curated set;
  `config/annotation_pages.csv`).
- `data/annotations/predictions_50.json` — autolabel predictions (optional; the
  page just starts empty without it).
- `config/classes.yaml`, `config/symbol_map.json` — class list and glyphs.

## Workflow

The page renders inside a **scrollable, zoomable drawing canvas**
(`streamlit-drawable-canvas`). Pages start **empty** (from scratch). The
autolabel template-matcher over-detects badly (hundreds of wrong `base_neume`
boxes per page), so it is opt-in — the sidebar **"Load autolabel predictions"**
button pulls them in if you'd rather correct than draw. Saved corrections always
reload first.

1. Pick a page in the sidebar (it loads your saved corrections, or empty).
2. **Zoom** with the sidebar slider — it scales the *image itself*; the pane
   **scrolls both ways** (set its height with *Viewport height*). No panning UI.
3. **Draw a box**: **click-drag** straight onto a glyph — the rectangle is
   created where you drag, no corner-placement. It commits immediately and is
   baked into the image (numbered). Auto-guess fills its class from glyphs you've
   already labelled this session (label the first of each glyph, repeats fill in).
4. **Set classes / delete**: the list below shows each box's **crop** next to the
   **glyph** of its assigned class. Fix the class by sight via the searchable
   picker, **🔮** to re-guess, or **🗑** to delete. (Undo/redo/clear are in the
   sidebar.) For dense pages, narrow with the **Class group filter** first.
   - The list is **paginated** ("Boxes per panel page" in the sidebar sets the
     chunk size). Panel pages are created/removed automatically as you add or
     delete boxes — move between them with the **◀ Prev / Next ▶** buttons and
     the page dropdown above the list. Drawing a new box jumps the view to the
     page it lands on so you can label it straight away. The boxes on the current
     panel page are drawn with a thicker border on the image.
5. **Save page corrections** → `data/corrections/<page>/detections.yolo`.
6. **Export YOLO dataset** → `data/datasets/review_export/` (images/labels +
   `dataset.yaml`), ready for detector training.

## How the canvas stays loop-free

The canvas only ever holds **newly drawn rects**. Existing boxes are drawn into
the canvas *background image* (`app._background`), never handed to the canvas as
fabric objects — seeding the canvas with its own output makes it remount-loop
forever as boxes accumulate. On each draw, the new rect is converted to a typed
box (`review_io.canvas_objects_to_boxes`), appended, and the canvas is remounted
empty (a nonce in its key) so the box reappears baked into the background. The
same nonce bump re-seeds the background after load-predictions/undo/redo/clear.

## Notes / limits

- Existing boxes are edited (class) or removed via the list, not on the canvas;
  to reshape one, delete and redraw. This keeps the canvas single-purpose and
  avoids the fabric feedback loop.
- `streamlit-drawable-canvas` 0.9.3 predates Streamlit 1.57's `image_to_url`
  move/signature change; `app._patch_drawable_canvas()` re-exposes a
  width-compatible shim so the background renders. Revisit if either is upgraded.
- The autolabeler over-detects (hundreds of boxes/page); use the group filter
  and panel paging to work through them, deleting false positives.
- Coordinate/YOLO/canvas maths is in `review_io.py` and unit-tested
  (`tests/test_review_io.py`); the Streamlit wiring is smoke-tested with
  `streamlit.testing.v1.AppTest` but not the in-browser canvas interactions.
- Later versions add the other correction tracks (cluster grouping, reading
  order, composition, lyrics, alignment) as `8xr`/`yyt` land.
