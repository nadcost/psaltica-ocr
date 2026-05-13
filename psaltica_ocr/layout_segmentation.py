"""Page layout segmentation for chant notation and lyric rows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

from psaltica_ocr.reading_order import Direction, DirectionOption, detect_page_direction, normalize_direction
from psaltica_ocr.rendering import binarize


RegionKind = Literal["chant", "lyrics", "non_score"]
UnknownDirection = Literal["unknown"]
TextDirection = Direction | UnknownDirection
Script = Literal["unknown"]


@dataclass(frozen=True)
class BoundingBox:
    """Pixel-space bounding box using half-open coordinates."""

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def x_center(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def y_center(self) -> float:
        return (self.y1 + self.y2) / 2

    def padded(self, *, x: int, y: int, width: int, height: int) -> "BoundingBox":
        return BoundingBox(
            max(0, self.x1 - x),
            max(0, self.y1 - y),
            min(width, self.x2 + x),
            min(height, self.y2 + y),
        )

    def to_list(self) -> list[int]:
        return [self.x1, self.y1, self.x2, self.y2]


@dataclass(frozen=True)
class LayoutRegion:
    kind: RegionKind
    bbox: BoundingBox
    component_count: int
    long_component_count: int
    ink_ratio: float
    script: Script = "unknown"
    text_direction: TextDirection = "unknown"

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "bbox": self.bbox.to_list(),
            "component_count": self.component_count,
            "long_component_count": self.long_component_count,
            "ink_ratio": self.ink_ratio,
            "script": self.script,
            "text_direction": self.text_direction,
        }


@dataclass(frozen=True)
class ChantRow:
    index: int
    bbox: BoundingBox
    notation_direction: Direction
    lyric_rows: tuple[LayoutRegion, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "bbox": self.bbox.to_list(),
            "notation_direction": self.notation_direction,
            "lyric_rows": [row.to_dict() for row in self.lyric_rows],
        }


@dataclass(frozen=True)
class PageLayout:
    width: int
    height: int
    notation_direction: Direction
    chant_rows: tuple[ChantRow, ...]
    unpaired_lyric_rows: tuple[LayoutRegion, ...]
    non_score_regions: tuple[LayoutRegion, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "width": self.width,
            "height": self.height,
            "notation_direction": self.notation_direction,
            "chant_rows": [row.to_dict() for row in self.chant_rows],
            "unpaired_lyric_rows": [row.to_dict() for row in self.unpaired_lyric_rows],
            "non_score_regions": [row.to_dict() for row in self.non_score_regions],
        }


@dataclass(frozen=True)
class _BandStats:
    bbox: BoundingBox
    component_count: int
    long_component_count: int
    long_width_sum: int
    ink_ratio: float

    @property
    def is_chant(self) -> bool:
        if self.component_count > 75:
            return False
        if self.long_component_count >= 3 and self.long_width_sum >= 300:
            if self.component_count <= 18:
                return True
            return self.long_component_count / self.component_count >= 0.32
        if self.bbox.width < 300 and self.long_component_count >= 3 and self.long_width_sum >= 90:
            return True
        return self.component_count <= 18 and self.long_component_count >= 5 and self.long_width_sum >= 220

    @property
    def is_text_like(self) -> bool:
        if self.is_chant:
            return False
        return (self.component_count >= 2 or self.bbox.width >= 40) and self.bbox.height >= 4

    @property
    def box_key(self) -> tuple[int, int, int, int]:
        return (self.bbox.x1, self.bbox.y1, self.bbox.x2, self.bbox.y2)


def segment_page_layout(
    image: np.ndarray,
    *,
    notation_direction: DirectionOption = "ltr",
    min_row_height: int = 3,
) -> PageLayout:
    """Segment a rendered page into chant rows, lyric rows, and non-score rows.

    Lyric rows are paired only with the nearest chant row above them. Rows above
    the first chant row are always non-score for alignment purposes.
    """

    direction = _resolve_notation_direction(image, notation_direction)
    binary = binarize(image)
    height, width = binary.shape
    bands = _band_stats(binary, min_row_height=min_row_height)

    chant_bands, chant_source_boxes = _expanded_chant_bands(bands, height=height)
    text_bands = _filter_text_bands_against_chant(
        _merged_text_bands(
            [band for band in bands if band.box_key not in chant_source_boxes and band.is_text_like],
            height=height,
        ),
        chant_bands,
    )
    chant_rows: list[ChantRow] = []
    unpaired_lyrics: list[LayoutRegion] = []
    non_score: list[LayoutRegion] = []

    for index, chant in enumerate(chant_bands):
        next_chant_y = chant_bands[index + 1].bbox.y1 if index + 1 < len(chant_bands) else height
        lyric_rows: list[LayoutRegion] = []
        for band in text_bands:
            if _is_lyric_below_chant(band, chant, next_chant_y, height=height):
                lyric_rows.append(_region("lyrics", band))
        chant_rows.append(ChantRow(index, chant.bbox, direction, tuple(lyric_rows)))

    paired_band_boxes = {
        tuple(row.bbox.to_list())
        for chant in chant_rows
        for row in chant.lyric_rows
    }
    first_chant_y = chant_bands[0].bbox.y1 if chant_bands else height
    for band in text_bands:
        region = _region("lyrics" if band.is_text_like else "non_score", band)
        if tuple(region.bbox.to_list()) in paired_band_boxes:
            continue
        if band.bbox.y1 < first_chant_y:
            non_score.append(_region("non_score", band))
        elif band.is_text_like and _has_nearby_chant_above(band, chant_bands, height=height):
            unpaired_lyrics.append(region)
        else:
            non_score.append(_region("non_score", band))

    return PageLayout(
        width=width,
        height=height,
        notation_direction=direction,
        chant_rows=tuple(chant_rows),
        unpaired_lyric_rows=tuple(unpaired_lyrics),
        non_score_regions=tuple(non_score),
    )


def _expanded_chant_bands(
    bands: list[_BandStats],
    *,
    height: int,
) -> tuple[list[_BandStats], set[tuple[int, int, int, int]]]:
    strong_seeds = [band for band in bands if band.is_chant]
    weak_seeds = [
        band
        for band in bands
        if not band.is_chant and _is_isolated_chant_like(band, bands, height=height)
    ]
    seed_sources = strong_seeds + weak_seeds
    seeds = seed_sources
    source_boxes: set[tuple[int, int, int, int]] = {band.box_key for band in seed_sources}
    expanded: list[_BandStats] = []
    max_modifier_gap = max(28, int(height * 0.025))

    for seed in seeds:
        group = [seed]
        for candidate in bands:
            if candidate.is_chant:
                continue
            above_gap = seed.bbox.y1 - candidate.bbox.y2
            close_above = candidate.bbox.y_center <= seed.bbox.y_center and above_gap <= max_modifier_gap
            if close_above and _is_modifier_like(candidate, height=height):
                group.append(candidate)
                source_boxes.add(candidate.box_key)
        expanded.append(_pad_chant_band(_merge_band_group(group), height=height))

    return sorted(expanded, key=lambda band: band.bbox.y1), source_boxes


def _is_modifier_like(band: _BandStats, *, height: int) -> bool:
    max_modifier_height = max(24, int(height * 0.025))
    return (
        band.component_count <= 4 or (band.component_count <= 6 and band.long_component_count >= 1)
    ) and band.bbox.height <= max_modifier_height


def _is_isolated_chant_like(band: _BandStats, bands: list[_BandStats], *, height: int) -> bool:
    if band.long_component_count < 1 or band.long_width_sum < 60:
        return False
    if band.component_count > 8 or band.bbox.width > 700:
        return False
    max_lyric_gap = _max_lyric_gap(height)
    return any(
        candidate is not band
        and candidate.is_text_like
        and 8 <= candidate.bbox.y1 - band.bbox.y2 <= max_lyric_gap
        and _horizontal_overlap_ratio(band.bbox, candidate.bbox) >= 0.05
        for candidate in bands
    )


def _pad_chant_band(band: _BandStats, *, height: int) -> _BandStats:
    top_pad = max(12, int(height * 0.008))
    bottom_pad = max(24, int(height * 0.012))
    bbox = BoundingBox(
        band.bbox.x1,
        max(0, band.bbox.y1 - top_pad),
        band.bbox.x2,
        min(height, band.bbox.y2 + bottom_pad),
    )
    return _BandStats(
        bbox=bbox,
        component_count=band.component_count,
        long_component_count=band.long_component_count,
        long_width_sum=band.long_width_sum,
        ink_ratio=band.ink_ratio,
    )


def _is_lyric_below_chant(band: _BandStats, chant: _BandStats, next_chant_y: int, *, height: int) -> bool:
    gap = max(0, band.bbox.y1 - chant.bbox.y2)
    return (
        band.bbox.y_center > chant.bbox.y_center
        and gap <= _max_lyric_gap(height)
        and band.bbox.y1 < next_chant_y
        and band.is_text_like
    )


def _has_nearby_chant_above(band: _BandStats, chant_bands: list[_BandStats], *, height: int) -> bool:
    return any(
        band.bbox.y_center > chant.bbox.y_center
        and max(0, band.bbox.y1 - chant.bbox.y2) <= _max_lyric_gap(height)
        for chant in chant_bands
    )


def _max_lyric_gap(height: int) -> int:
    return max(125, int(height * 0.07))


def _merge_band_group(group: list[_BandStats]) -> _BandStats:
    x1 = min(band.bbox.x1 for band in group)
    y1 = min(band.bbox.y1 for band in group)
    x2 = max(band.bbox.x2 for band in group)
    y2 = max(band.bbox.y2 for band in group)
    area = max(1, (x2 - x1) * (y2 - y1))
    weighted_ink = sum(band.ink_ratio * band.bbox.width * band.bbox.height for band in group)
    return _BandStats(
        bbox=BoundingBox(x1, y1, x2, y2),
        component_count=sum(band.component_count for band in group),
        long_component_count=sum(band.long_component_count for band in group),
        long_width_sum=sum(band.long_width_sum for band in group),
        ink_ratio=float(weighted_ink / area),
    )


def _merged_text_bands(bands: list[_BandStats], *, height: int) -> list[_BandStats]:
    if not bands:
        return []
    max_gap = max(10, int(height * 0.01))
    merged: list[list[_BandStats]] = []
    for band in sorted(bands, key=lambda item: item.bbox.y1):
        if not merged:
            merged.append([band])
            continue
        previous_group = merged[-1]
        previous = _merge_band_group(previous_group)
        vertical_gap = band.bbox.y1 - previous.bbox.y2
        if vertical_gap <= max_gap and _horizontal_overlap_ratio(previous.bbox, band.bbox) >= 0.15:
            previous_group.append(band)
        else:
            merged.append([band])
    return [_merge_band_group(group) for group in merged]


def _filter_text_bands_against_chant(bands: list[_BandStats], chant_bands: list[_BandStats]) -> list[_BandStats]:
    filtered: list[_BandStats] = []
    for band in bands:
        if any(_text_is_inside_chant_box(band, chant) for chant in chant_bands):
            continue
        filtered.append(band)
    return filtered


def _text_is_inside_chant_box(text: _BandStats, chant: _BandStats) -> bool:
    y_overlap = max(0, min(text.bbox.y2, chant.bbox.y2) - max(text.bbox.y1, chant.bbox.y1))
    if y_overlap == 0:
        return False
    overlap_ratio = y_overlap / max(1, text.bbox.height)
    if overlap_ratio < 0.35:
        return False
    if text.bbox.y_center <= chant.bbox.y_center:
        return True
    small_lower_fragment = text.bbox.width <= max(45, int(chant.bbox.width * 0.25))
    return overlap_ratio >= 0.5 and small_lower_fragment


def _horizontal_overlap_ratio(a: BoundingBox, b: BoundingBox) -> float:
    overlap = max(0, min(a.x2, b.x2) - max(a.x1, b.x1))
    return overlap / max(1, min(a.width, b.width))


def chant_mask_from_layout(layout: PageLayout, *, pad_y: int = 8) -> np.ndarray:
    """Return a mask with chant rows as 255 and all other rows as 0."""

    mask = np.zeros((layout.height, layout.width), dtype=np.uint8)
    for row in layout.chant_rows:
        bbox = row.bbox.padded(x=0, y=pad_y, width=layout.width, height=layout.height)
        mask[bbox.y1 : bbox.y2, :] = 255
        for lyric in row.lyric_rows:
            mask[lyric.bbox.y1 : lyric.bbox.y2, :] = 0
    return mask


def _resolve_notation_direction(image: np.ndarray, direction: DirectionOption) -> Direction:
    normalized = normalize_direction(direction)
    if normalized == "auto":
        return detect_page_direction(image)
    return normalized


def _region(kind: RegionKind, band: _BandStats) -> LayoutRegion:
    return LayoutRegion(
        kind=kind,
        bbox=band.bbox,
        component_count=band.component_count,
        long_component_count=band.long_component_count,
        ink_ratio=band.ink_ratio,
    )


def _band_stats(binary: np.ndarray, *, min_row_height: int) -> list[_BandStats]:
    height, width = binary.shape
    component_count, _, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    components: list[tuple[int, int, int, int, int, float]] = []
    for label in range(1, component_count):
        x, y, component_width, component_height, area = stats[label]
        if area < 3 or component_height == 0:
            continue
        components.append((int(x), int(y), int(component_width), int(component_height), int(area), float(centroids[label][1])))

    row_groups = _component_row_groups(components, height=height)
    result: list[_BandStats] = []

    for group in row_groups:
        x1 = width
        x2 = 0
        y1 = height
        y2 = 0
        components = 0
        long_components = 0
        long_width_sum = 0
        min_width = max(12, int(width * 0.01))
        max_height = max(8, int(height * 0.025))

        for x, y, component_width, component_height, _, _ in group:
            components += 1
            x1 = min(x1, int(x))
            x2 = max(x2, int(x + component_width))
            y1 = min(y1, int(y))
            y2 = max(y2, int(y + component_height))
            aspect = component_width / component_height
            if component_width >= min_width and component_height <= max_height and aspect >= 3.0:
                long_components += 1
                long_width_sum += int(component_width)

        if components == 0:
            continue
        if y2 - y1 < min_row_height:
            continue
        bbox = BoundingBox(x1, y1, x2, y2)
        roi = binary[y1:y2, x1:x2] > 0
        ink_ratio = float(np.mean(roi)) if roi.size else 0.0
        result.append(_BandStats(bbox, components, long_components, long_width_sum, ink_ratio))

    return result


def _component_row_groups(
    components: list[tuple[int, int, int, int, int, float]],
    *,
    height: int,
) -> list[list[tuple[int, int, int, int, int, float]]]:
    tolerance = max(6, int(height * 0.006))
    groups: list[list[tuple[int, int, int, int, int, float]]] = []
    centers: list[float] = []
    for component in sorted(components, key=lambda item: item[5]):
        center_y = component[5]
        for index, center in enumerate(centers):
            if abs(center_y - center) <= tolerance:
                groups[index].append(component)
                centers[index] = sum(item[5] for item in groups[index]) / len(groups[index])
                break
        else:
            groups.append([component])
            centers.append(center_y)
    return [group for _, group in sorted(zip(centers, groups), key=lambda item: item[0])]
