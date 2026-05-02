"""Reading-direction helpers for page-level and row-level OCR processing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Mapping

import cv2
import numpy as np


Direction = Literal["ltr", "rtl"]
DirectionOption = Literal["ltr", "rtl", "auto"]
DirectionMap = dict[str, dict[str, Direction]]


def normalize_direction(value: str) -> DirectionOption:
    normalized = value.lower()
    if normalized not in {"ltr", "rtl", "auto"}:
        raise ValueError(f"Unsupported direction: {value}")
    return normalized  # type: ignore[return-value]


def detect_page_direction(image: np.ndarray) -> Direction:
    """Infer a coarse page direction from horizontal ink balance."""

    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    if not np.any(binary):
        return "ltr"
    ys, xs = np.where(binary > 0)
    _ = ys
    ink_center = float(xs.mean())
    page_center = binary.shape[1] / 2
    return "rtl" if ink_center > page_center else "ltr"


def load_direction_map(path: Path | None) -> DirectionMap:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return normalize_direction_map(raw)


def normalize_direction_map(raw: Mapping[str, object]) -> DirectionMap:
    normalized: DirectionMap = {}
    for book_id, value in raw.items():
        if isinstance(value, str):
            normalized[str(book_id)] = {"default": _coerce_direction(value)}
            continue
        if not isinstance(value, Mapping):
            raise ValueError(f"Invalid direction map entry for {book_id}")
        book: dict[str, Direction] = {}
        for key, direction in value.items():
            book[str(key)] = _coerce_direction(str(direction))
        normalized[str(book_id)] = book
    return normalized


def resolve_direction(
    book_id: str,
    page_number: int,
    *,
    default: DirectionOption,
    direction_map: DirectionMap | None = None,
) -> DirectionOption:
    book_map = (direction_map or {}).get(book_id, {})
    page_keys = [str(page_number), f"page_{page_number:04d}"]
    for key in page_keys:
        if key in book_map:
            return book_map[key]
    return book_map.get("default", default)


def _coerce_direction(value: str) -> Direction:
    normalized = normalize_direction(value)
    if normalized == "auto":
        raise ValueError("Direction maps must resolve to ltr or rtl, not auto")
    return normalized
