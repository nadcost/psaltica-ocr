"""Scoring of layout segmentation against a hand-labelled gold fixture.

The gold fixture records, per page, the true chant-row y-bands (each with the
lyric-row y-bands that belong to it) and optional non-score y-bands. Rows always
span the full page width, so matching is purely vertical.

Two gate metrics are produced:

* ``chant_mask_precision`` — pixel-row precision of the predicted chant mask
  against the union of gold chant bands. This penalises chant boxes that bleed
  into lyric/non-score space (the main over-segmentation failure mode).
* ``lyric_pairing_precision`` — fraction of predicted (chant -> lyric) pairings
  that attach a real lyric row to the correct chant row.

Recall is reported alongside each precision for diagnostics but is not gated.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from psaltica_ocr.layout_segmentation import PageLayout, chant_mask_from_layout, segment_page_layout
from psaltica_ocr.reading_order import DirectionOption
from psaltica_ocr.rendering import binarize


@dataclass(frozen=True)
class Band:
    """A full-width horizontal band identified by its vertical extent."""

    y1: int
    y2: int

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)


@dataclass(frozen=True)
class GoldChantRow:
    band: Band
    lyrics: tuple[Band, ...]


@dataclass(frozen=True)
class GoldPage:
    image_path: str
    chant_rows: tuple[GoldChantRow, ...]
    non_score: tuple[Band, ...]


@dataclass(frozen=True)
class PageScore:
    image_path: str
    chant_pred_px: int
    chant_gold_px: int
    chant_overlap_px: int
    lyric_pred: int
    lyric_gold: int
    lyric_correct: int
    lyric_matched_gold: int

    @property
    def chant_mask_precision(self) -> float:
        return _ratio(self.chant_overlap_px, self.chant_pred_px)

    @property
    def chant_mask_recall(self) -> float:
        return _ratio(self.chant_overlap_px, self.chant_gold_px)

    @property
    def lyric_pairing_precision(self) -> float:
        return _ratio(self.lyric_correct, self.lyric_pred)

    @property
    def lyric_pairing_recall(self) -> float:
        return _ratio(self.lyric_matched_gold, self.lyric_gold)


@dataclass(frozen=True)
class AuditReport:
    pages: tuple[PageScore, ...]

    @property
    def chant_mask_precision(self) -> float:
        return _ratio(
            sum(page.chant_overlap_px for page in self.pages),
            sum(page.chant_pred_px for page in self.pages),
        )

    @property
    def chant_mask_recall(self) -> float:
        return _ratio(
            sum(page.chant_overlap_px for page in self.pages),
            sum(page.chant_gold_px for page in self.pages),
        )

    @property
    def lyric_pairing_precision(self) -> float:
        return _ratio(
            sum(page.lyric_correct for page in self.pages),
            sum(page.lyric_pred for page in self.pages),
        )

    @property
    def lyric_pairing_recall(self) -> float:
        return _ratio(
            sum(page.lyric_matched_gold for page in self.pages),
            sum(page.lyric_gold for page in self.pages),
        )

    def passes_gate(self, *, min_precision: float = 0.90) -> bool:
        return (
            self.chant_mask_precision >= min_precision
            and self.lyric_pairing_precision >= min_precision
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "chant_mask_precision": self.chant_mask_precision,
            "chant_mask_recall": self.chant_mask_recall,
            "lyric_pairing_precision": self.lyric_pairing_precision,
            "lyric_pairing_recall": self.lyric_pairing_recall,
            "pages": [
                {
                    "image_path": page.image_path,
                    "chant_mask_precision": page.chant_mask_precision,
                    "chant_mask_recall": page.chant_mask_recall,
                    "lyric_pairing_precision": page.lyric_pairing_precision,
                    "lyric_pairing_recall": page.lyric_pairing_recall,
                    "lyric_pred": page.lyric_pred,
                    "lyric_gold": page.lyric_gold,
                    "lyric_correct": page.lyric_correct,
                }
                for page in self.pages
            ],
        }


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 1.0 if numerator == 0 else 0.0
    return numerator / denominator


def _band(raw: dict[str, object]) -> Band:
    return Band(int(raw["y1"]), int(raw["y2"]))


def load_gold(path: Path) -> list[GoldPage]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    pages: list[GoldPage] = []
    for page in data["pages"]:
        chant_rows = tuple(
            GoldChantRow(
                band=_band(row),
                lyrics=tuple(_band(lyric) for lyric in row.get("lyrics", [])),
            )
            for row in page.get("chant_rows", [])
        )
        non_score = tuple(_band(region) for region in page.get("non_score", []))
        pages.append(GoldPage(str(page["image_path"]), chant_rows, non_score))
    return pages


def _interval_union_length(intervals: Iterable[tuple[int, int]]) -> int:
    spans = sorted((max(0, a), max(0, b)) for a, b in intervals if b > a)
    if not spans:
        return 0
    total = 0
    current_start, current_end = spans[0]
    for start, end in spans[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    total += current_end - current_start
    return total


def _interval_overlap_length(
    a_intervals: Iterable[tuple[int, int]],
    b_intervals: Iterable[tuple[int, int]],
) -> int:
    a_spans = sorted((a, b) for a, b in a_intervals if b > a)
    b_spans = sorted((a, b) for a, b in b_intervals if b > a)
    total = 0
    j = 0
    for a_start, a_end in a_spans:
        while j < len(b_spans) and b_spans[j][1] <= a_start:
            j += 1
        k = j
        while k < len(b_spans) and b_spans[k][0] < a_end:
            total += max(0, min(a_end, b_spans[k][1]) - max(a_start, b_spans[k][0]))
            k += 1
    return total


def _vertical_overlap_ratio(a: Band, b: Band) -> float:
    overlap = max(0, min(a.y2, b.y2) - max(a.y1, b.y1))
    return overlap / max(1, min(a.height, b.height))


def _predicted_chant_intervals(layout: PageLayout) -> list[tuple[int, int]]:
    """Y-intervals actually marked chant by the downstream mask.

    Scoring the real ``chant_mask_from_layout`` output (not the raw chant
    bboxes) credits the segmenter for subtracting paired lyric rows and
    penalises chant boxes that swallow lyric/non-score space.
    """

    mask = chant_mask_from_layout(layout, pad_y=0)
    row_has_chant = mask.any(axis=1)
    intervals: list[tuple[int, int]] = []
    start: int | None = None
    for y, active in enumerate(row_has_chant):
        if active and start is None:
            start = y
        elif not active and start is not None:
            intervals.append((start, y))
            start = None
    if start is not None:
        intervals.append((start, len(row_has_chant)))
    return intervals


def score_page(layout: PageLayout, gold: GoldPage, *, match_threshold: float = 0.3) -> PageScore:
    pred_chant = _predicted_chant_intervals(layout)
    gold_chant = [(row.band.y1, row.band.y2) for row in gold.chant_rows]
    chant_pred_px = _interval_union_length(pred_chant)
    chant_gold_px = _interval_union_length(gold_chant)
    chant_overlap_px = _interval_overlap_length(pred_chant, gold_chant)

    # Map every predicted chant row to its best-overlapping gold chant row so a
    # predicted lyric pairing can be checked against that gold row's lyrics.
    pred_to_gold: dict[int, int] = {}
    for pred_index, row in enumerate(layout.chant_rows):
        pred_band = Band(row.bbox.y1, row.bbox.y2)
        best_ratio = match_threshold
        best_gold = -1
        for gold_index, gold_row in enumerate(gold.chant_rows):
            ratio = _vertical_overlap_ratio(pred_band, gold_row.band)
            if ratio >= best_ratio:
                best_ratio = ratio
                best_gold = gold_index
        if best_gold >= 0:
            pred_to_gold[pred_index] = best_gold

    lyric_pred = 0
    lyric_correct = 0
    matched_gold_lyrics: set[tuple[int, int]] = set()
    for pred_index, row in enumerate(layout.chant_rows):
        for lyric in row.lyric_rows:
            lyric_pred += 1
            gold_index = pred_to_gold.get(pred_index)
            if gold_index is None:
                continue
            lyric_band = Band(lyric.bbox.y1, lyric.bbox.y2)
            for gold_lyric_index, gold_lyric in enumerate(gold.chant_rows[gold_index].lyrics):
                if _vertical_overlap_ratio(lyric_band, gold_lyric) >= match_threshold:
                    lyric_correct += 1
                    matched_gold_lyrics.add((gold_index, gold_lyric_index))
                    break

    lyric_gold = sum(len(row.lyrics) for row in gold.chant_rows)
    return PageScore(
        image_path=gold.image_path,
        chant_pred_px=chant_pred_px,
        chant_gold_px=chant_gold_px,
        chant_overlap_px=chant_overlap_px,
        lyric_pred=lyric_pred,
        lyric_gold=lyric_gold,
        lyric_correct=lyric_correct,
        lyric_matched_gold=len(matched_gold_lyrics),
    )


def audit_gold(
    gold_pages: list[GoldPage],
    *,
    notation_direction: DirectionOption = "ltr",
    match_threshold: float = 0.3,
    image_loader=None,
) -> AuditReport:
    """Run segmentation on each gold page and score it.

    ``image_loader`` maps an image path to a grayscale ``np.ndarray``; the default
    reads from disk with OpenCV.
    """

    loader = image_loader or _default_image_loader
    scores: list[PageScore] = []
    for gold in gold_pages:
        image = loader(gold.image_path)
        if image is None:
            raise FileNotFoundError(f"unreadable gold page: {gold.image_path}")
        layout = segment_page_layout(image, notation_direction=notation_direction)
        scores.append(score_page(layout, gold, match_threshold=match_threshold))
    return AuditReport(tuple(scores))


def _default_image_loader(path: str) -> np.ndarray | None:
    import cv2

    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def extract_line_bands(
    image: np.ndarray,
    *,
    min_gap: int = 10,
    min_height: int = 4,
    min_ink_fraction: float = 0.002,
) -> list[Band]:
    """Return ink line-bands via a horizontal projection profile.

    Used to bootstrap a gold fixture: it yields precise pixel y-bands for every
    printed line so a human only has to assign labels, not read coordinates.
    """

    binary = binarize(image)
    height, width = binary.shape
    row_ink = (binary > 0).sum(axis=1)
    threshold = max(1.0, min_ink_fraction * width)
    active = row_ink >= threshold

    bands: list[Band] = []
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
                    bands.append(Band(start, end))
                start = None
                gap = 0
    if start is not None:
        end = height - gap if gap else height
        if end - start >= min_height:
            bands.append(Band(start, end))
    return bands
