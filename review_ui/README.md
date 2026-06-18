# Psaltica OCR review UI (v1)

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

## Workflow (canvas-free)

Pages start **empty** (from scratch): tick **Draw mode**, pick a class, and drag
boxes. The autolabel template-matcher over-detects badly (hundreds of wrong
`base_neume` boxes per page), so it is opt-in — the sidebar
**"Load autolabel predictions"** button pulls them in for a page if you want to
try correcting instead of drawing. Saved corrections always reload first.

1. Pick a page in the sidebar (it loads your saved corrections, or empty).
2. The left pane shows the page with every box overlaid and numbered — a static
   image, so there is **no click-coordinate offset**. The boxes for the current
   panel page are drawn thicker.
3. The right pane lists those boxes: each shows its **crop** next to the
   **glyph** of its assigned class. Fix the class by sight via the searchable
   picker, or **🗑** to delete a false positive. Use the group filter + panel
   paging for dense pages.
4. **Add a missed glyph**: tick **Draw mode**, drag the green resizable box (with
   handles, live preview) over the glyph, then **➕ Add**. The box stays put so
   you can slide it to the next glyph and add again. Auto-guess fills the class
   from glyphs you've already labelled this session (repeats auto-fill).
5. **Save page corrections** → `data/corrections/<page>/detections.yolo`.
6. **Export YOLO dataset** → `data/datasets/review_export/` (images/labels +
   `dataset.yaml`), ready for detector training.

## Drawing boxes: use makesense.ai (zoom/pan), classify here

Streamlit has no reliable pan/zoom box-drawing component (the canvas mis-scales
clicks on HiDPI; the cropper fits-to-container and can't pan). For dense
300-DPI pages, draw boxes in **makesense.ai** instead, then classify them here:

1. Open https://www.makesense.ai → **Object Detection**, drop the page image(s)
   (they stay local in the browser).
2. Import labels from `config/makesense_labels.txt` (order matches our 179-class
   taxonomy, so exported indices are correct).
3. Draw boxes with scroll-zoom + drag-pan. Label precisely, or coarsely/by
   group if you'd rather classify here.
4. **Export → YOLO**. Keep `data/reference_sheet.html` open to look up glyph
   names while labelling.

To finish classification here with the glyph picker + auto-guess, import the
makesense YOLO export into `data/corrections/` and review per box (no drawing
needed). The whole-page draw/zoom flow below remains for light touch-ups.

## Notes / v1 limits

- Box geometry uses `streamlit-image-coordinates` (click/drag → real image
  pixels), not `streamlit-drawable-canvas`, which mis-scales clicks inside
  Streamlit's iframe on HiDPI displays (boxes drifted from the cursor). Existing
  boxes are corrected from the list (class + delete); drawing is for additions.
- The autolabeler over-detects (hundreds of boxes/page); use the group filter
  and panel paging to work through them, deleting false positives.
- Coordinate/YOLO maths is in `review_io.py` and unit-tested
  (`tests/test_review_io.py`); the Streamlit wiring is not auto-tested.
- Later versions add the other correction tracks (cluster grouping, reading
  order, composition, lyrics, alignment) as `8xr`/`yyt` land.
