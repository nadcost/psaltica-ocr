"""Tests for synthetic-page glyph detection (no font needed — drawn shapes)."""

from __future__ import annotations

import cv2
import numpy as np

from psaltica_ocr.synthetic_labeling import detect_glyph_boxes


def _blank(h=400, w=600):
    return np.full((h, w), 255, np.uint8)


def test_grid_detected_in_reading_order() -> None:
    img = _blank()
    # 2 rows x 3 cols of well-separated blobs.
    centers = [(80, 100), (280, 100), (480, 100), (80, 300), (280, 300), (480, 300)]
    for cx, cy in centers:
        cv2.circle(img, (cx, cy), 25, 0, -1)
    boxes = detect_glyph_boxes(img)
    assert len(boxes) == 6
    # Reading order: row 1 left→right, then row 2.
    centers_out = [((b[0] + b[2]) // 2, (b[1] + b[3]) // 2) for b in boxes]
    assert [c[0] for c in centers_out[:3]] == sorted(c[0] for c in centers_out[:3])
    assert all(c[1] < 200 for c in centers_out[:3])  # first row on top
    assert all(c[1] > 200 for c in centers_out[3:])  # second row below


def test_close_parts_merge_far_parts_split() -> None:
    img = _blank(200, 600)
    # One glyph = two blobs 8px apart (a base + its accent); a far blob is separate.
    cv2.rectangle(img, (40, 80), (80, 140), 0, -1)
    cv2.rectangle(img, (88, 60), (110, 100), 0, -1)   # close → same glyph
    cv2.circle(img, (400, 100), 28, 0, -1)            # far → its own glyph
    boxes = detect_glyph_boxes(img)
    assert len(boxes) == 2
    first = boxes[0]
    assert first[0] <= 40 and first[2] >= 110  # merged box spans both close blobs


def test_count_safeguard_inputs() -> None:
    # No ink -> no boxes (caller will treat count mismatch as "don't assign").
    assert detect_glyph_boxes(_blank()) == []
