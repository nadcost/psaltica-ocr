"""Spatial helpers for assembling OCR detections into reading order."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

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


def sort_detections_reading_order(
    detections: Sequence[Detection],
    *,
    direction: Direction = "ltr",
    row_tolerance: float = 20.0,
) -> list[Detection]:
    rows = group_detections_by_row(detections, row_tolerance=row_tolerance)
    sorted_detections: list[Detection] = []
    reverse_x = direction == "rtl"
    for row in rows:
        sorted_detections.extend(sorted(row, key=lambda detection: detection.x_center, reverse=reverse_x))
    return sorted_detections


def group_detections_by_row(
    detections: Sequence[Detection],
    *,
    row_tolerance: float = 20.0,
) -> list[list[Detection]]:
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
    return [row for _, row in sorted(zip(row_centers, rows), key=lambda item: item[0])]


def _mean_y(row: Sequence[Detection]) -> float:
    return sum(detection.y_center for detection in row) / len(row)
