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

## Workflow

1. Pick a page in the sidebar. Predicted boxes load onto the canvas (or your
   previously-saved corrections if any).
2. **Tool = transform**: select / move / resize / delete boxes (canvas toolbar
   has undo/redo/trash). **Tool = rect**: draw a new box for a missed glyph.
3. In the panel below, each box shows its **crop** next to the **glyph** of its
   currently-assigned class — fix the class by sight via the searchable picker.
   Use the group filter + panel paging to work through dense pages.
4. **Save page corrections** → writes `data/corrections/<page>/detections.yolo`.
5. **Export YOLO dataset** → `data/datasets/review_export/` (images/labels +
   `dataset.yaml`), ready for detector training.

## Notes / v1 limits

- Geometry lives on the canvas; classes are tracked per box index in session
  state. Deleting a box mid-list can shift indices — re-check classes after
  large deletions.
- Coordinate/YOLO/canvas maths is in `review_io.py` and unit-tested
  (`tests/test_review_io.py`); the Streamlit wiring is not auto-tested.
- Later versions add the other correction tracks (cluster grouping, reading
  order, composition, lyrics, alignment) as `8xr`/`yyt` land.
