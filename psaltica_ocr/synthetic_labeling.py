"""Detect glyph boxes on a clean, composed synthetic page.

Synthetic pages are auto-labeled *without* visual class guessing: this module
only finds where the glyphs are (connected components grouped into glyph
clusters, returned in reading order); the class for each box comes from the
known order the page was composed in (see tools/label_synthetic_page.py). The
hard, failure-prone part of the old autolabeler — deciding *what* each blob is —
is removed entirely.

Grouping assumes glyphs are laid out isolated with clear spacing: ink blobs that
sit close horizontally (parts of one cluster) merge; the larger gaps between
glyphs separate them. It is heuristic by design, so callers should preview the
result and a count mismatch should be treated as "don't trust the ordering".
"""

from __future__ import annotations

import cv2
import numpy as np

Box = tuple[int, int, int, int]  # x1, y1, x2, y2


def detect_components(gray: np.ndarray, min_area: int) -> list[Box]:
    """Ink connected components (Otsu) as boxes, dropping specks under min_area."""
    _, binv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    count, _labels, stats, _ = cv2.connectedComponentsWithStats(binv, connectivity=8)
    boxes: list[Box] = []
    for i in range(1, count):  # 0 is background
        x, y, w, h, area = stats[i]
        if area >= min_area:
            boxes.append((int(x), int(y), int(x + w), int(y + h)))
    return boxes


def _cluster_rows(boxes: list[Box], row_gap: float) -> list[list[Box]]:
    """Group boxes into rows by vertical position (overlap or within row_gap)."""
    rows: list[list[Box]] = []
    for box in sorted(boxes, key=lambda b: (b[1] + b[3]) / 2):
        cy = (box[1] + box[3]) / 2
        placed = False
        for row in rows:
            top = min(b[1] for b in row)
            bottom = max(b[3] for b in row)
            if top - row_gap <= cy <= bottom + row_gap:
                row.append(box)
                placed = True
                break
        if not placed:
            rows.append([box])
    return rows


def _merge_row(row: list[Box], col_gap: float) -> list[Box]:
    """Merge horizontally-close boxes in a row into one glyph-cluster box."""
    merged: list[Box] = []
    cur: list[int] | None = None
    for box in sorted(row, key=lambda b: b[0]):
        if cur is not None and box[0] <= cur[2] + col_gap:
            cur = [min(cur[0], box[0]), min(cur[1], box[1]), max(cur[2], box[2]), max(cur[3], box[3])]
        else:
            if cur is not None:
                merged.append(tuple(cur))  # type: ignore[arg-type]
            cur = list(box)
    if cur is not None:
        merged.append(tuple(cur))  # type: ignore[arg-type]
    return merged


def detect_glyph_boxes(
    gray: np.ndarray,
    min_area: int = 25,
    col_gap: float | None = None,
    row_gap: float | None = None,
) -> list[Box]:
    """Glyph-cluster boxes on a clean page, in reading order (rows top→bottom,
    left→right). Gaps default to fractions of the median component size, which
    suits an evenly-spaced grid; override them for tighter or looser layouts."""
    comps = detect_components(gray, min_area)
    if not comps:
        return []
    med_w = float(np.median([b[2] - b[0] for b in comps]))
    med_h = float(np.median([b[3] - b[1] for b in comps]))
    col_gap = 0.6 * med_w if col_gap is None else col_gap
    row_gap = 0.5 * med_h if row_gap is None else row_gap

    ordered: list[Box] = []
    rows = _cluster_rows(comps, row_gap)
    for row in sorted(rows, key=lambda r: min(b[1] for b in r)):
        ordered.extend(_merge_row(row, col_gap))
    return ordered
