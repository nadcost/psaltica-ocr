import numpy as np

from psaltica_ocr.cluster_assembly import Detection, group_detections_by_row, sort_detections_reading_order
from psaltica_ocr.reading_order import detect_page_direction, normalize_direction_map, resolve_direction


def test_sort_detections_left_to_right_within_rows() -> None:
    detections = [
        Detection("b", (90, 10, 100, 20)),
        Detection("a", (10, 10, 20, 20)),
        Detection("c", (40, 50, 50, 60)),
    ]

    ordered = sort_detections_reading_order(detections, direction="ltr")

    assert [detection.class_name for detection in ordered] == ["a", "b", "c"]


def test_sort_detections_right_to_left_within_rows() -> None:
    detections = [
        Detection("left", (10, 10, 20, 20)),
        Detection("right", (90, 10, 100, 20)),
    ]

    ordered = sort_detections_reading_order(detections, direction="rtl")

    assert [detection.class_name for detection in ordered] == ["right", "left"]


def test_sort_detections_can_detect_direction_per_row() -> None:
    detections = [
        Detection("top_left", (10, 10, 20, 20)),
        Detection("top_right", (30, 10, 40, 20)),
        Detection("bottom_left", (160, 50, 170, 60)),
        Detection("bottom_right", (190, 50, 200, 60)),
    ]

    ordered = sort_detections_reading_order(detections, direction="ltr", page_width=200)

    assert [detection.class_name for detection in ordered] == [
        "top_left",
        "top_right",
        "bottom_right",
        "bottom_left",
    ]


def test_row_direction_overrides_detection() -> None:
    detections = [
        Detection("left", (10, 10, 20, 20)),
        Detection("right", (30, 10, 40, 20)),
    ]

    rows = group_detections_by_row(detections, page_width=200, row_directions={0: "rtl"})
    ordered = sort_detections_reading_order(detections, page_width=200, row_directions={0: "rtl"})

    assert rows[0].direction == "rtl"
    assert [detection.class_name for detection in ordered] == ["right", "left"]


def test_detect_page_direction_uses_ink_balance() -> None:
    image = np.full((40, 100), 255, dtype=np.uint8)
    image[10:30, 70:95] = 0

    assert detect_page_direction(image) == "rtl"


def test_direction_map_supports_default_and_page_override() -> None:
    direction_map = normalize_direction_map({"book": {"default": "rtl", "2": "ltr"}})

    assert resolve_direction("book", 1, default="ltr", direction_map=direction_map) == "rtl"
    assert resolve_direction("book", 2, default="rtl", direction_map=direction_map) == "ltr"
