from __future__ import annotations

import numpy as np

from psaltica_ocr.font_shape_matching import (
    GlyphShape,
    MatchDetection,
    group_similar_shapes,
    non_max_suppression,
    normalize_shape,
    parse_codepoint_ranges,
    shape_similarity,
)


def test_normalize_shape_removes_position_offsets() -> None:
    left = np.full((40, 40), 255, dtype=np.uint8)
    right = np.full((40, 40), 255, dtype=np.uint8)
    left[4:18, 7:18] = 0
    right[18:32, 22:33] = 0

    left_normalized = normalize_shape(left, canvas=24)
    right_normalized = normalize_shape(right, canvas=24)

    assert left_normalized is not None
    assert right_normalized is not None
    assert shape_similarity(left_normalized, right_normalized) > 0.999


def test_group_similar_shapes_groups_by_normalized_ink_shape() -> None:
    square = np.full((24, 24), 255, dtype=np.uint8)
    square[6:18, 6:18] = 0
    same_square = square.copy()
    vertical = np.full((24, 24), 255, dtype=np.uint8)
    vertical[3:21, 10:14] = 0

    groups = group_similar_shapes(
        [
            GlyphShape("U+E001", 0xE001, "square.one", square),
            GlyphShape("U+E002", 0xE002, "square.two", same_square),
            GlyphShape("U+E003", 0xE003, "vertical", vertical),
        ],
        threshold=0.95,
    )

    member_sets = {group.members for group in groups}
    assert ("U+E001", "U+E002") in member_sets
    assert ("U+E003",) in member_sets


def test_non_max_suppression_keeps_best_overlapping_detection() -> None:
    detections = [
        MatchDetection(10, 10, 20, 20, 0.80, "shape_0001", "U+E001", 8.0),
        MatchDetection(11, 11, 20, 20, 0.95, "shape_0002", "U+E002", 8.0),
        MatchDetection(80, 80, 20, 20, 0.70, "shape_0003", "U+E003", 8.0),
    ]

    kept = non_max_suppression(detections, iou_threshold=0.3)

    assert [detection.group_id for detection in kept] == ["shape_0002", "shape_0003"]


def test_parse_codepoint_ranges() -> None:
    assert parse_codepoint_ranges(["E0D0-E0D2", "U+0174"]) == [(0xE0D0, 0xE0D2), (0x0174, 0x0174)]
