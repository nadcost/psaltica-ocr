"""Page layout segmentation for chant notation and lyric rows."""

from __future__ import annotations

from bisect import bisect_right
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
    page_width: int = 0

    @property
    def _min_long_for_chant(self) -> int:
        # Neume rows are packed with long thin ligature strokes. A full-width
        # printed row carries ~9; a lyric row carries only a few kashida
        # elongations. The count threshold scales with page width so the same
        # rule holds for narrow synthetic fixtures (~3) and real pages (~6).
        return max(3, round(self.page_width / 400))

    @property
    def is_chant(self) -> bool:
        # Count of neume ligature strokes is the dominant chant/text
        # discriminator (gold-band precision 0.985 at the real-page threshold).
        return self.long_component_count >= self._min_long_for_chant and self.long_width_sum >= 90

    @property
    def _is_neume_like_fragment(self) -> bool:
        # A short neume row (too few ligatures to be a full chant seed) is still
        # dominated by horizontal strokes; keep it out of the lyric pool. Gauge
        # dominance by the stroke-to-component ratio so Arabic lyric rows, whose
        # few kashida elongations sit among many letters and dots, stay lyrics.
        if self.long_component_count < 3 or self.long_width_sum < 90:
            return False
        return self.long_component_count / max(1, self.component_count) >= 0.30

    @property
    def is_text_like(self) -> bool:
        if self.is_chant or self._is_neume_like_fragment:
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
    min_lyric_height = max(20, int(height * 0.006))
    text_bands = [
        band
        for band in _filter_text_bands_against_chant(
            _merged_text_bands(
                [band for band in bands if band.box_key not in chant_source_boxes and band.is_text_like],
                height=height,
            ),
            chant_bands,
        )
        if band.bbox.height >= min_lyric_height and band.component_count >= 4
    ]
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
    seeds = [band for band in bands if band.is_chant]
    source_boxes: set[tuple[int, int, int, int]] = {band.box_key for band in seeds}
    expanded: list[_BandStats] = []
    max_modifier_gap = max(28, int(height * 0.025))
    max_key_line_gap = max(70, int(height * 0.06))

    for seed in seeds:
        group = [seed]
        for candidate in bands:
            if candidate.is_chant:
                continue
            close_above = candidate.bbox.y_center <= seed.bbox.y_center
            if not close_above:
                continue
            above_gap = seed.bbox.y1 - candidate.bbox.y2
            if candidate.is_text_like:
                # A real lyric/title line has too many components to pass
                # _is_key_line_like below, so this only screens out genuine
                # text; a lone segment-start key glyph still gets a look.
                if not (above_gap <= max_key_line_gap and _is_key_line_like(candidate, height=height)):
                    continue
            elif above_gap <= max_modifier_gap and _is_modifier_like(candidate, height=height):
                pass
            elif above_gap <= max_key_line_gap and _is_key_line_like(candidate, height=height):
                pass
            else:
                continue
            group.append(candidate)
            source_boxes.add(candidate.box_key)
        expanded.append(_pad_chant_band(_merge_band_group(group), height=height))

    return _merge_overlapping_chant_bands(expanded), source_boxes


def _is_modifier_like(band: _BandStats, *, height: int) -> bool:
    max_modifier_height = max(24, int(height * 0.025))
    return (
        band.component_count <= 4 or (band.component_count <= 6 and band.long_component_count >= 1)
    ) and band.bbox.height <= max_modifier_height


def _is_key_line_like(band: _BandStats, *, height: int) -> bool:
    # A segment-start key glyph sits alone on an otherwise-blank line: unlike
    # a modifier accent (tight to the neume it decorates), it can be a full
    # line-height away, but it's a lone glyph, not a run of letters.
    max_key_height = max(60, int(height * 0.08))
    return band.component_count <= 2 and band.bbox.height <= max_key_height


def _pad_chant_band(band: _BandStats, *, height: int) -> _BandStats:
    top_pad = max(4, int(height * 0.0015))
    bottom_pad = max(4, int(height * 0.0015))
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
        page_width=band.page_width,
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
    # Real neume->lyric gaps are small (90th percentile ~27px on the audit set).
    # A tight bound stops a lyric from binding across a missed neume row to the
    # wrong chant above it.
    return max(50, int(height * 0.024))


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
        page_width=max(band.page_width for band in group),
    )


def _merge_overlapping_chant_bands(bands: list[_BandStats]) -> list[_BandStats]:
    if not bands:
        return []
    groups: list[list[_BandStats]] = []
    for band in sorted(bands, key=lambda item: item.bbox.y1):
        for group in groups:
            merged = _merge_band_group(group)
            if _chant_boxes_overlap(merged.bbox, band.bbox):
                group.append(band)
                break
        else:
            groups.append([band])
    return sorted((_merge_band_group(group) for group in groups), key=lambda item: item.bbox.y1)


def _chant_boxes_overlap(a: BoundingBox, b: BoundingBox) -> bool:
    y_overlap = max(0, min(a.y2, b.y2) - max(a.y1, b.y1))
    if y_overlap == 0:
        return False
    vertical_ratio = y_overlap / max(1, min(a.height, b.height))
    return vertical_ratio >= 0.15 and _horizontal_overlap_ratio(a, b) >= 0.05


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

    row_groups = _projection_row_groups(binary, components, min_height=min_row_height)
    result: list[_BandStats] = []

    for band_y1, band_y2, group in row_groups:
        x1 = width
        x2 = 0
        y1 = band_y2
        y2 = band_y1
        components = 0
        long_components = 0
        long_width_sum = 0
        min_width = max(12, int(width * 0.01))
        max_height = max(8, int(height * 0.025))

        for x, y, component_width, component_height, _, _ in group:
            components += 1
            x1 = min(x1, int(x))
            x2 = max(x2, int(x + component_width))
            # Clip vertical extent to the projection sub-band so a stroke that
            # dips across the neume/lyric valley does not inflate the band.
            y1 = min(y1, max(band_y1, int(y)))
            y2 = max(y2, min(band_y2, int(y + component_height)))
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
        result.append(
            _BandStats(bbox, components, long_components, long_width_sum, ink_ratio, page_width=width)
        )

    return result


def _projection_line_bands(
    binary: np.ndarray,
    *,
    min_gap: int,
    min_height: int,
    min_ink_fraction: float = 0.002,
) -> list[tuple[int, int]]:
    """Clean, non-overlapping ink line-bands from a horizontal projection.

    This replaces centroid clustering (which produced overlapping, fragmented
    bands) so a neume row and the lyric row beneath it land in distinct bands.
    """

    height, width = binary.shape
    row_ink = (binary > 0).sum(axis=1)
    threshold = max(1.0, min_ink_fraction * width)
    active = row_ink >= threshold
    bands: list[tuple[int, int]] = []
    start: int | None = None
    gap = 0
    for y in range(height):
        if active[y]:
            if start is None:
                start = y
            gap = 0
        elif start is not None:
            gap += 1
            if gap > min_gap:
                end = y - gap + 1
                if end - start >= min_height:
                    bands.append((start, end))
                start = None
                gap = 0
    if start is not None:
        end = height - gap if gap else height
        if end - start >= min_height:
            bands.append((start, end))
    return bands


def _split_band_at_valleys(
    row_ink: np.ndarray,
    y1: int,
    y2: int,
    *,
    min_gap: int,
    min_height: int,
) -> list[tuple[int, int]]:
    """Split a tall band at its single deepest ink valley.

    Touching neume/lyric rows merge in the projection (their gap is ~0 ink).
    Only over-tall bands are candidates, and only the deepest central valley is
    cut, so an ordinary neume or diacritic-laden lyric row is never shattered.
    """

    height = y2 - y1
    split_threshold = max(110, int(row_ink.shape[0] * 0.04))
    if height <= split_threshold:
        return [(y1, y2)]

    segment = row_ink[y1:y2].astype(float)
    lo = int(height * 0.30)
    hi = int(height * 0.75)
    if hi - lo < 2:
        return [(y1, y2)]
    valley = lo + int(np.argmin(segment[lo:hi]))
    # Require the valley to be a genuine separator: low ink relative to the
    # surrounding peaks, otherwise this is one continuous row.
    peak = max(segment[:valley].max(), segment[valley:].max(), 1.0)
    if segment[valley] > peak * 0.42:
        return [(y1, y2)]

    top = (y1, y1 + valley)
    bottom = (y1 + valley, y2)
    parts: list[tuple[int, int]] = []
    for a, b in (top, bottom):
        if b - a >= min_height:
            parts.extend(_split_band_at_valleys(row_ink, a, b, min_gap=min_gap, min_height=min_height))
    return parts or [(y1, y2)]


def _projection_row_groups(
    binary: np.ndarray,
    components: list[tuple[int, int, int, int, int, float]],
    *,
    min_height: int,
) -> list[list[tuple[int, int, int, int, int, float]]]:
    height = binary.shape[0]
    min_gap = max(4, int(height * 0.0015))
    row_ink = (binary > 0).sum(axis=1)
    coarse = _projection_line_bands(binary, min_gap=min_gap, min_height=min_height)
    bands: list[tuple[int, int]] = []
    for y1, y2 in coarse:
        bands.extend(_split_band_at_valleys(row_ink, y1, y2, min_gap=min_gap, min_height=min_height))
    if not bands:
        return []
    groups: list[list[tuple[int, int, int, int, int, float]]] = [[] for _ in bands]
    starts = [b[0] for b in bands]
    for component in components:
        center_y = component[5]
        # Place each component in the band whose span contains its centroid;
        # fall back to the nearest band for stray diacritics in the gutter.
        index = bisect_right(starts, center_y) - 1
        if index < 0:
            index = 0
        elif index + 1 < len(bands) and center_y >= bands[index][1]:
            below = bands[index + 1][0] - center_y
            above = center_y - bands[index][1]
            if below < above:
                index += 1
        groups[index].append(component)
    return [(bands[i][0], bands[i][1], group) for i, group in enumerate(groups) if group]
    return [group for _, group in sorted(zip(centers, groups), key=lambda item: item[0])]
