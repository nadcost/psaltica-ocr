"""Local lyric OCR adapter for printed Greek, Latin/English, and Arabic rows.

The adapter separates two concerns so the OCR backend can be swapped (Tesseract
now, PaddleOCR or a specialized model later) without touching downstream
assembly or alignment:

* A :class:`LyricOcrBackend` does the only engine-specific work: turn a row
  crop into raw text plus optional crop-local word boxes and confidences.
* :class:`LyricOcr` does everything engine-independent: Unicode/script
  detection, NFC normalization (raw text preserved alongside), page-coordinate
  translation, and a geometry word-box fallback when the engine returns none.

v0 targets an offline engine. The default backend wraps Tesseract via a lazy
import so the rest of the pipeline (and its tests) runs without the binary
installed; the backend raises a clear error only when actually invoked.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Literal, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from psaltica_ocr.layout_segmentation import BoundingBox, LayoutRegion, PageLayout
from psaltica_ocr.reading_order import Direction


Script = Literal["Greek", "Latin", "Arabic", "mixed", "unknown"]


# Tesseract language codes per detected script. The adapter runs every pack at
# once (script is unknown before OCR) and classifies the result afterwards.
DEFAULT_LANGUAGES: dict[Script, tuple[str, ...]] = {
    "Greek": ("ell",),
    "Latin": ("eng",),
    "Arabic": ("ara",),
    "mixed": ("ell", "eng", "ara"),
    "unknown": ("ell", "eng", "ara"),
}
ALL_LANGUAGES: tuple[str, ...] = ("ell", "eng", "ara")


@dataclass(frozen=True)
class RawWord:
    """An engine-reported word in crop-local pixel coordinates."""

    text: str
    bbox: BoundingBox
    confidence: float


@dataclass(frozen=True)
class RawOcr:
    """Low-level OCR result for one row crop, in crop-local coordinates."""

    text: str
    words: tuple[RawWord, ...]
    confidence: float


@runtime_checkable
class LyricOcrBackend(Protocol):
    """Stable interface every OCR engine must implement.

    ``run`` receives a single binarizable row crop (grayscale or BGR) and the
    requested Tesseract-style language codes, and returns raw text plus optional
    word boxes in crop-local pixel coordinates. The adapter owns normalization,
    script detection, and page-coordinate translation.
    """

    name: str

    def run(self, crop: np.ndarray, *, languages: Sequence[str], direction: Direction) -> RawOcr: ...


@dataclass(frozen=True)
class LyricToken:
    """A word or syllable candidate in page coordinates, ready for alignment."""

    text: str
    raw_text: str
    bbox: BoundingBox
    confidence: float
    script: Script

    def to_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "raw_text": self.raw_text,
            "bbox": self.bbox.to_list(),
            "confidence": round(self.confidence, 4),
            "script": self.script,
        }


@dataclass(frozen=True)
class LyricLine:
    """OCR output for one lyric row, in page coordinates."""

    text: str
    raw_text: str
    script: Script
    direction: Direction
    confidence: float
    bbox: BoundingBox
    engine: str
    tokens: tuple[LyricToken, ...]
    boxes_from_geometry: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "raw_text": self.raw_text,
            "script": self.script,
            "direction": self.direction,
            "confidence": round(self.confidence, 4),
            "bbox": self.bbox.to_list(),
            "engine": self.engine,
            "boxes_from_geometry": self.boxes_from_geometry,
            "tokens": [token.to_dict() for token in self.tokens],
        }


def normalize_text(text: str) -> str:
    """NFC-normalize while preserving diacritics and punctuation."""

    return unicodedata.normalize("NFC", text)


def _classify_char(char: str) -> Script | None:
    code = ord(char)
    if (0x0370 <= code <= 0x03FF) or (0x1F00 <= code <= 0x1FFF):
        return "Greek"
    if (
        ("A" <= char <= "Z")
        or ("a" <= char <= "z")
        or (0x00C0 <= code <= 0x024F)  # Latin-1 Supplement + Latin Extended-A/B
    ):
        return "Latin"
    if (
        (0x0600 <= code <= 0x06FF)
        or (0x0750 <= code <= 0x077F)
        or (0x08A0 <= code <= 0x08FF)
        or (0xFB50 <= code <= 0xFDFF)
        or (0xFE70 <= code <= 0xFEFF)
    ):
        return "Arabic"
    return None


def detect_script(text: str) -> Script:
    """Classify a lyric row by dominant alphabetic script.

    Whitespace, digits, and punctuation are ignored. A single script with no
    competitors wins outright; otherwise the dominant script wins only if it
    holds the clear majority, else the row is ``mixed``.
    """

    counts: dict[Script, int] = {"Greek": 0, "Latin": 0, "Arabic": 0}
    for char in text:
        script = _classify_char(char)
        if script in counts:
            counts[script] += 1

    total = sum(counts.values())
    if total == 0:
        return "unknown"

    present = {script: count for script, count in counts.items() if count}
    if len(present) == 1:
        return next(iter(present))

    dominant_script, dominant_count = max(present.items(), key=lambda item: item[1])
    if dominant_count / total >= 0.85:
        return dominant_script
    return "mixed"


def script_direction(script: Script) -> Direction:
    """Default reading direction for a script (Arabic is RTL, others LTR).

    This is lyric-row direction only; notation encoding order is always L->R and
    is tracked separately on the segment.
    """

    return "rtl" if script == "Arabic" else "ltr"


class LyricOcr:
    """Engine-independent lyric OCR adapter.

    Wraps any :class:`LyricOcrBackend`, adding script/direction detection, NFC
    normalization, page-coordinate translation of word boxes, and a coarse
    geometry fallback when the engine reports no per-word boxes.
    """

    def __init__(
        self,
        backend: LyricOcrBackend,
        *,
        languages: Sequence[str] = ALL_LANGUAGES,
        languages_by_script: Mapping[Script, Sequence[str]] = DEFAULT_LANGUAGES,
    ) -> None:
        self.backend = backend
        self.languages = tuple(languages)
        self.languages_by_script = dict(languages_by_script)

    @property
    def engine_name(self) -> str:
        return getattr(self.backend, "name", type(self.backend).__name__)

    def recognize_line(
        self,
        image: np.ndarray,
        bbox: BoundingBox,
        *,
        direction_hint: Direction | None = None,
    ) -> LyricLine:
        """OCR a single lyric row given the page image and the row's box."""

        crop = image[bbox.y1 : bbox.y2, bbox.x1 : bbox.x2]
        raw = self.backend.run(crop, languages=self.languages, direction=direction_hint or "ltr")

        raw_text = raw.text
        text = normalize_text(raw_text)
        script = detect_script(text)
        direction = direction_hint or script_direction(script)

        boxes_from_geometry = not raw.words
        if raw.words:
            tokens = tuple(self._page_token(word, bbox) for word in raw.words)
        else:
            tokens = self._geometry_tokens(text, raw_text, bbox, direction, raw.confidence)

        return LyricLine(
            text=text,
            raw_text=raw_text,
            script=script,
            direction=direction,
            confidence=raw.confidence,
            bbox=bbox,
            engine=self.engine_name,
            tokens=tokens,
            boxes_from_geometry=boxes_from_geometry,
        )

    def recognize_layout(
        self,
        image: np.ndarray,
        layout: PageLayout,
    ) -> list[tuple[int | None, LyricLine]]:
        """OCR every lyric row in a page layout.

        Returns ``(chant_row_index, line)`` pairs; ``chant_row_index`` is
        ``None`` for unpaired lyric rows. Notation direction provides the row
        direction hint so segment metadata stays consistent.
        """

        results: list[tuple[int | None, LyricLine]] = []
        for chant_row in layout.chant_rows:
            for region in chant_row.lyric_rows:
                results.append((chant_row.index, self._recognize_region(image, region, layout.notation_direction)))
        for region in layout.unpaired_lyric_rows:
            results.append((None, self._recognize_region(image, region, layout.notation_direction)))
        return results

    def _recognize_region(
        self,
        image: np.ndarray,
        region: LayoutRegion,
        notation_direction: Direction,
    ) -> LyricLine:
        # Notation direction is only a hint; an Arabic row still flips to RTL via
        # script detection unless the layout already fixed the row direction.
        hint = region.text_direction if region.text_direction in ("ltr", "rtl") else None
        return self.recognize_line(image, region.bbox, direction_hint=hint)  # type: ignore[arg-type]

    def _page_token(self, word: RawWord, bbox: BoundingBox) -> LyricToken:
        page_bbox = BoundingBox(
            word.bbox.x1 + bbox.x1,
            word.bbox.y1 + bbox.y1,
            word.bbox.x2 + bbox.x1,
            word.bbox.y2 + bbox.y1,
        )
        text = normalize_text(word.text)
        return LyricToken(
            text=text,
            raw_text=word.text,
            bbox=page_bbox,
            confidence=word.confidence,
            script=detect_script(text),
        )

    def _geometry_tokens(
        self,
        text: str,
        raw_text: str,
        bbox: BoundingBox,
        direction: Direction,
        confidence: float,
    ) -> tuple[LyricToken, ...]:
        """Derive coarse per-word boxes from row geometry and text length.

        Boxes are laid out left-to-right and proportional to word character
        counts; for RTL rows the first reading-order word lands in the rightmost
        box so alignment still walks words in reading order.
        """

        words = text.split()
        raw_words = raw_text.split()
        if not words:
            return ()

        weights = [len(word) + 1 for word in words]
        total_weight = sum(weights)
        spans: list[tuple[int, int]] = []
        cursor = bbox.x1
        for index, weight in enumerate(weights):
            width = round(bbox.width * weight / total_weight)
            x1 = cursor
            x2 = bbox.x2 if index == len(weights) - 1 else min(bbox.x2, cursor + width)
            spans.append((x1, x2))
            cursor = x2

        if direction == "rtl":
            spans = list(reversed(spans))

        tokens: list[LyricToken] = []
        for index, word in enumerate(words):
            x1, x2 = spans[index]
            tokens.append(
                LyricToken(
                    text=word,
                    raw_text=raw_words[index] if index < len(raw_words) else word,
                    bbox=BoundingBox(x1, bbox.y1, x2, bbox.y2),
                    confidence=confidence,
                    script=detect_script(word),
                )
            )
        return tuple(tokens)


class TesseractBackend:
    """Offline Tesseract backend (pytesseract).

    Lazy-imports ``pytesseract`` so the binary is only required when OCR is
    actually run. Word boxes and per-word confidences come from
    ``image_to_data``; install the language packs with
    ``brew install tesseract tesseract-lang``.
    """

    name = "tesseract-v0"

    def __init__(self, *, psm: int = 7, min_confidence: float = 0.0) -> None:
        self.psm = psm
        self.min_confidence = min_confidence

    def run(self, crop: np.ndarray, *, languages: Sequence[str], direction: Direction) -> RawOcr:
        try:
            import pytesseract
            from pytesseract import Output
        except ImportError as exc:  # pragma: no cover - exercised only without the dep
            raise RuntimeError(
                "TesseractBackend requires pytesseract; add it to the project deps "
                "and install the tesseract binary (brew install tesseract tesseract-lang)."
            ) from exc

        lang = "+".join(languages)
        config = f"--psm {self.psm}"
        data = pytesseract.image_to_data(crop, lang=lang, config=config, output_type=Output.DICT)

        words: list[RawWord] = []
        confidences: list[float] = []
        for index, raw in enumerate(data["text"]):
            text = raw.strip()
            if not text:
                continue
            conf = float(data["conf"][index])
            if conf < 0:
                continue
            confidence = conf / 100.0
            if confidence < self.min_confidence:
                continue
            x, y, w, h = (data["left"][index], data["top"][index], data["width"][index], data["height"][index])
            words.append(RawWord(text=text, bbox=BoundingBox(x, y, x + w, y + h), confidence=confidence))
            confidences.append(confidence)

        ordered = sorted(words, key=lambda word: word.bbox.x1)
        text = " ".join(word.text for word in ordered)
        mean_conf = float(np.mean(confidences)) if confidences else 0.0
        return RawOcr(text=text, words=tuple(ordered), confidence=mean_conf)
