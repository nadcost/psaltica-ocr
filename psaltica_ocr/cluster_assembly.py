"""Spatial helpers for assembling OCR detections into reading order."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from psaltica_ocr.reading_order import Direction


@dataclass(frozen=True)
class Detection:
    class_name: str
    xyxy: tuple[float, float, float, float]
    confidence: float | None = None

    @property
    def x_center(self) -> float:
        x1, _, x2, _ = self.xyxy
        return (x1 + x2) / 2

    @property
    def y_center(self) -> float:
        _, y1, _, y2 = self.xyxy
        return (y1 + y2) / 2


@dataclass(frozen=True)
class DetectionRow:
    index: int
    detections: list[Detection]
    direction: Direction


def sort_detections_reading_order(
    detections: Sequence[Detection],
    *,
    direction: Direction = "ltr",
    row_tolerance: float = 20.0,
    page_width: float | None = None,
    row_directions: Mapping[int, Direction] | None = None,
) -> list[Detection]:
    rows = group_detections_by_row(
        detections,
        row_tolerance=row_tolerance,
        default_direction=direction,
        page_width=page_width,
        row_directions=row_directions,
    )
    sorted_detections: list[Detection] = []
    for row in rows:
        reverse_x = row.direction == "rtl"
        sorted_detections.extend(sorted(row.detections, key=lambda detection: detection.x_center, reverse=reverse_x))
    return sorted_detections


def group_detections_by_row(
    detections: Sequence[Detection],
    *,
    row_tolerance: float = 20.0,
    default_direction: Direction = "ltr",
    page_width: float | None = None,
    row_directions: Mapping[int, Direction] | None = None,
) -> list[DetectionRow]:
    rows: list[list[Detection]] = []
    row_centers: list[float] = []
    for detection in sorted(detections, key=lambda item: item.y_center):
        for index, center in enumerate(row_centers):
            if abs(detection.y_center - center) <= row_tolerance:
                rows[index].append(detection)
                row_centers[index] = _mean_y(rows[index])
                break
        else:
            rows.append([detection])
            row_centers.append(detection.y_center)
    sorted_rows = [row for _, row in sorted(zip(row_centers, rows), key=lambda item: item[0])]
    return [
        DetectionRow(
            index=index,
            detections=row,
            direction=(row_directions or {}).get(
                index,
                detect_row_direction(row, default_direction=default_direction, page_width=page_width),
            ),
        )
        for index, row in enumerate(sorted_rows)
    ]


def detect_row_direction(
    row: Sequence[Detection],
    *,
    default_direction: Direction = "ltr",
    page_width: float | None = None,
) -> Direction:
    if not row or page_width is None or page_width <= 0:
        return default_direction
    row_center = sum(detection.x_center for detection in row) / len(row)
    return "rtl" if row_center > page_width / 2 else "ltr"


def _mean_y(row: Sequence[Detection]) -> float:
    return sum(detection.y_center for detection in row) / len(row)
