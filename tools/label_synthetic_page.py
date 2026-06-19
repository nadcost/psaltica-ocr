#!/usr/bin/env python3
"""Auto-label a composed synthetic page from the known glyph order.

You compose the rare glyphs in the app (in a known reading order), print to PDF,
and run this. It finds each glyph's box geometrically and assigns the class from
your order list — no visual class guessing, so it can't mislabel the way the old
template autolabeler did. A hard safeguard refuses to assign classes if the
detected box count doesn't match your list (it writes a preview so you can see
why, instead of producing garbage labels).

    # order file = one class name per line, in the order you composed them
    uv run python tools/label_synthetic_page.py page.pdf --order order.txt
    # or default to the synthetic_targets.csv order (one of each):
    uv run python tools/label_synthetic_page.py page.pdf --from-targets

Output: the page image + a *_preview.png (boxes + assigned class) under
data/synthetic/, and YOLO corrections under data/corrections/<page> so you can
review/fix in the UI and export with the rest. Inspect the preview first.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from psaltica_ocr.synthetic_labeling import detect_glyph_boxes
from review_ui.review_io import Box, load_class_names, save_page_detections

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLASSES = REPO_ROOT / "config/classes.yaml"
DEFAULT_TARGETS = REPO_ROOT / "data/annotations/synthetic_targets.csv"
SYNTHETIC_DIR = REPO_ROOT / "data/synthetic"
CORRECTIONS = REPO_ROOT / "data/corrections"
PAGES_FILE = REPO_ROOT / "data/annotations/pages_50.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("page", type=Path, help="Composed synthetic page (.pdf or image).")
    parser.add_argument("--order", type=Path, help="Text file: one class name per line, in composed order.")
    parser.add_argument("--from-targets", action="store_true",
                        help=f"Use the class order in {DEFAULT_TARGETS.name} (one of each).")
    parser.add_argument("--classes", type=Path, default=DEFAULT_CLASSES)
    parser.add_argument("--dpi", type=int, default=300, help="PDF render DPI (default 300).")
    parser.add_argument("--min-area", type=int, default=25)
    parser.add_argument("--col-gap", type=float, default=None, help="Override horizontal merge gap (px).")
    parser.add_argument("--row-gap", type=float, default=None, help="Override row split gap (px).")
    parser.add_argument("--register", action="store_true",
                        help="Append the page to the review UI's page list so it opens there.")
    return parser.parse_args()


def load_page_image(path: Path, dpi: int) -> tuple[np.ndarray, Path]:
    """Return (grayscale array, saved PNG path under data/synthetic)."""
    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".pdf":
        from pdf2image import convert_from_path

        pages = convert_from_path(str(path), dpi=dpi)
        if not pages:
            raise SystemExit(f"No pages rendered from {path}")
        if len(pages) > 1:
            print(f"Note: {len(pages)} PDF pages; using the first.")
        png = SYNTHETIC_DIR / f"{path.stem}.png"
        pages[0].convert("L").save(png)
    else:
        png = SYNTHETIC_DIR / f"{path.stem}.png"
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise SystemExit(f"Cannot read {path}")
        cv2.imwrite(str(png), img)
    gray = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
    return gray, png


def load_order(args: argparse.Namespace) -> list[str]:
    if args.order:
        return [ln.strip() for ln in args.order.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if args.from_targets:
        with DEFAULT_TARGETS.open(encoding="utf-8") as handle:
            return [row["class"] for row in csv.DictReader(handle)]
    raise SystemExit("Provide --order FILE or --from-targets so classes can be assigned by order.")


def write_preview(gray: np.ndarray, boxes: list, labels: list[str | None], out: Path) -> None:
    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for i, (box, label) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = box
        ok = label is not None
        color = (0, 150, 0) if ok else (0, 0, 220)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        text = f"{i}:{label.split('.', 1)[-1]}" if ok else str(i)
        cv2.putText(canvas, text, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.imwrite(str(out), canvas)


def main() -> None:
    args = parse_args()
    class_names = load_class_names(args.classes)
    order = load_order(args)
    unknown = [c for c in order if c not in class_names]
    if unknown:
        raise SystemExit(f"Order has {len(unknown)} unknown class(es), e.g. {unknown[:3]}")

    gray, png = load_page_image(args.page, args.dpi)
    boxes = detect_glyph_boxes(gray, min_area=args.min_area, col_gap=args.col_gap, row_gap=args.row_gap)
    height, width = gray.shape
    preview = SYNTHETIC_DIR / f"{png.stem}_preview.png"

    print(f"Detected {len(boxes)} glyph boxes; order list has {len(order)}.")
    if len(boxes) != len(order):
        write_preview(gray, boxes, [None] * len(boxes), preview)
        raise SystemExit(
            f"COUNT MISMATCH — not assigning classes (refusing to mislabel).\n"
            f"Inspect {preview}: likely glyphs merged/split. Adjust spacing in the\n"
            f"composition (or tune --col-gap/--row-gap) and re-run."
        )

    boxed = [Box(cls, *map(float, box), source="synthetic") for cls, box in zip(order, boxes)]
    write_preview(gray, boxes, order, preview)
    label_path = save_page_detections(CORRECTIONS, png, boxed, class_names, width, height)

    if args.register and PAGES_FILE.exists():
        lines = PAGES_FILE.read_text(encoding="utf-8").splitlines()
        rel = str(png.relative_to(REPO_ROOT))
        if rel not in lines:
            PAGES_FILE.write_text("\n".join(lines + [rel]) + "\n", encoding="utf-8")
            print(f"Registered {rel} in {PAGES_FILE.name} (open it in the review UI to verify).")

    print(f"Wrote {len(boxed)} labels -> {label_path}")
    print(f"Preview (inspect this!) -> {preview}")


if __name__ == "__main__":
    main()
