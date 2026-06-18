"""Per-script lyric OCR accuracy against an ``expected_lyrics.json`` gold set.

The gold fixture records, per page, the lyric rows (box + true script/direction
+ normalized expected text). Scoring runs the configured OCR backend on each
row crop and reports character and word error rates bucketed by gold script, so
the Phase-4 gate ("lyric OCR word accuracy recorded by script") can be tracked
without committing to a fixed threshold yet.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from psaltica_ocr.layout_segmentation import BoundingBox
from psaltica_ocr.lyric_ocr import LyricOcr, Script, normalize_text
from psaltica_ocr.reading_order import Direction


@dataclass(frozen=True)
class GoldRow:
    bbox: BoundingBox
    script: Script
    direction: Direction | None
    text: str


@dataclass(frozen=True)
class GoldPage:
    image_path: str
    rows: tuple[GoldRow, ...]


@dataclass(frozen=True)
class RowScore:
    image_path: str
    script: Script
    expected: str
    predicted: str
    char_distance: int
    char_total: int
    word_distance: int
    word_total: int

    @property
    def char_error_rate(self) -> float:
        return _ratio(self.char_distance, self.char_total)

    @property
    def word_error_rate(self) -> float:
        return _ratio(self.word_distance, self.word_total)


@dataclass(frozen=True)
class ScriptScore:
    script: Script
    rows: int
    char_distance: int
    char_total: int
    word_distance: int
    word_total: int

    @property
    def char_error_rate(self) -> float:
        return _ratio(self.char_distance, self.char_total)

    @property
    def word_error_rate(self) -> float:
        return _ratio(self.word_distance, self.word_total)

    @property
    def char_accuracy(self) -> float:
        return 1.0 - self.char_error_rate

    @property
    def word_accuracy(self) -> float:
        return 1.0 - self.word_error_rate

    def to_dict(self) -> dict[str, object]:
        return {
            "script": self.script,
            "rows": self.rows,
            "char_error_rate": round(self.char_error_rate, 4),
            "word_error_rate": round(self.word_error_rate, 4),
            "char_accuracy": round(self.char_accuracy, 4),
            "word_accuracy": round(self.word_accuracy, 4),
        }


@dataclass(frozen=True)
class AuditReport:
    engine: str
    rows: tuple[RowScore, ...]

    def by_script(self) -> list[ScriptScore]:
        scripts: dict[Script, list[RowScore]] = {}
        for row in self.rows:
            scripts.setdefault(row.script, []).append(row)
        result: list[ScriptScore] = []
        for script, rows in sorted(scripts.items()):
            result.append(
                ScriptScore(
                    script=script,
                    rows=len(rows),
                    char_distance=sum(r.char_distance for r in rows),
                    char_total=sum(r.char_total for r in rows),
                    word_distance=sum(r.word_distance for r in rows),
                    word_total=sum(r.word_total for r in rows),
                )
            )
        return result

    @property
    def overall(self) -> ScriptScore:
        return ScriptScore(
            script="mixed",
            rows=len(self.rows),
            char_distance=sum(r.char_distance for r in self.rows),
            char_total=sum(r.char_total for r in self.rows),
            word_distance=sum(r.word_distance for r in self.rows),
            word_total=sum(r.word_total for r in self.rows),
        )

    def to_dict(self) -> dict[str, object]:
        overall = self.overall
        return {
            "engine": self.engine,
            "rows": len(self.rows),
            "overall": {
                "char_error_rate": round(overall.char_error_rate, 4),
                "word_error_rate": round(overall.word_error_rate, 4),
                "char_accuracy": round(overall.char_accuracy, 4),
                "word_accuracy": round(overall.word_accuracy, 4),
            },
            "by_script": [score.to_dict() for score in self.by_script()],
            "row_details": [
                {
                    "image_path": row.image_path,
                    "script": row.script,
                    "expected": row.expected,
                    "predicted": row.predicted,
                    "char_error_rate": round(row.char_error_rate, 4),
                    "word_error_rate": round(row.word_error_rate, 4),
                }
                for row in self.rows
            ],
        }


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0 if numerator == 0 else 1.0
    return numerator / denominator


def levenshtein(a: Sequence[object], b: Sequence[object]) -> int:
    """Standard edit distance over any comparable sequences."""

    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, item_a in enumerate(a, start=1):
        current = [i]
        for j, item_b in enumerate(b, start=1):
            cost = 0 if item_a == item_b else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1]


def char_error_rate(predicted: str, expected: str) -> float:
    expected_n = normalize_text(expected)
    return _ratio(levenshtein(normalize_text(predicted), expected_n), len(expected_n))


def word_error_rate(predicted: str, expected: str) -> float:
    expected_words = normalize_text(expected).split()
    predicted_words = normalize_text(predicted).split()
    return _ratio(levenshtein(predicted_words, expected_words), len(expected_words))


def _bbox(raw: dict[str, object]) -> BoundingBox:
    if isinstance(raw, dict) and "x1" in raw:
        return BoundingBox(int(raw["x1"]), int(raw["y1"]), int(raw["x2"]), int(raw["y2"]))
    values = list(raw)  # type: ignore[arg-type]
    return BoundingBox(int(values[0]), int(values[1]), int(values[2]), int(values[3]))


def load_gold(path: Path) -> list[GoldPage]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    pages: list[GoldPage] = []
    for page in data["pages"]:
        rows = tuple(
            GoldRow(
                bbox=_bbox(row["bbox"]),
                script=str(row.get("script", "unknown")),  # type: ignore[arg-type]
                direction=row.get("direction"),
                text=str(row.get("text", "")),
            )
            for row in page.get("rows", [])
        )
        pages.append(GoldPage(str(page["image_path"]), rows))
    return pages


def score_row(predicted: str, gold: GoldRow, image_path: str) -> RowScore:
    expected = normalize_text(gold.text)
    prediction = normalize_text(predicted)
    return RowScore(
        image_path=image_path,
        script=gold.script,
        expected=expected,
        predicted=prediction,
        char_distance=levenshtein(prediction, expected),
        char_total=len(expected),
        word_distance=levenshtein(prediction.split(), expected.split()),
        word_total=len(expected.split()),
    )


def audit_gold(
    gold_pages: list[GoldPage],
    adapter: LyricOcr,
    *,
    image_loader=None,
) -> AuditReport:
    """Run the adapter on every gold lyric row and score it per script."""

    loader = image_loader or _default_image_loader
    rows: list[RowScore] = []
    for page in gold_pages:
        image = loader(page.image_path)
        if image is None:
            raise FileNotFoundError(f"unreadable gold page: {page.image_path}")
        for gold in page.rows:
            direction_hint = gold.direction if gold.direction in ("ltr", "rtl") else None
            line = adapter.recognize_line(image, gold.bbox, direction_hint=direction_hint)
            rows.append(score_row(line.text, gold, page.image_path))
    return AuditReport(engine=adapter.engine_name, rows=tuple(rows))


def _default_image_loader(path: str) -> np.ndarray | None:
    import cv2

    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)
