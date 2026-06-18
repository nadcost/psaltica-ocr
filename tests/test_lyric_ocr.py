"""Tests for the lyric OCR adapter, script detection, and accuracy metrics."""

from __future__ import annotations

import unicodedata
from pathlib import Path

import numpy as np
import pytest

from psaltica_ocr.layout_segmentation import BoundingBox
from psaltica_ocr.lyric_ocr import (
    LyricOcr,
    RawOcr,
    RawWord,
    TesseractBackend,
    detect_script,
    normalize_text,
    script_direction,
    syllable_spans,
    trim_to_text_band,
)
from psaltica_ocr.lyric_ocr_audit import (
    AuditReport,
    GoldPage,
    GoldRow,
    audit_gold,
    char_error_rate,
    fold_for_scoring,
    levenshtein,
    load_gold,
    word_error_rate,
)


class FakeBackend:
    """Returns a scripted result regardless of pixels, for deterministic tests."""

    name = "fake"

    def __init__(self, result: RawOcr) -> None:
        self.result = result
        self.calls: list[tuple[tuple[int, int], tuple[str, ...]]] = []

    def run(self, crop, *, languages, direction) -> RawOcr:
        self.calls.append((crop.shape[:2], tuple(languages)))
        return self.result


def _blank_page(height: int = 200, width: int = 400) -> np.ndarray:
    return np.zeros((height, width), dtype=np.uint8)


def test_detect_script_single_scripts() -> None:
    assert detect_script("Κυριε ελεησον") == "Greek"
    assert detect_script("Lord have mercy") == "Latin"
    assert detect_script("رب ارحم") == "Arabic"
    assert detect_script("12 . , !") == "unknown"


def test_detect_script_mixed_vs_dominant() -> None:
    # Stray Latin abbreviation in a Greek row stays Greek (dominant >= 85%).
    assert detect_script("Κυριε ελεησον ημας a") == "Greek"
    # Roughly balanced scripts are mixed.
    assert detect_script("Κυριε Lord ελεησον mercy") == "mixed"


def test_script_direction() -> None:
    assert script_direction("Arabic") == "rtl"
    assert script_direction("Greek") == "ltr"
    assert script_direction("Latin") == "ltr"
    assert script_direction("mixed") == "ltr"


def test_normalize_text_is_nfc() -> None:
    decomposed = unicodedata.normalize("NFD", "Άγιος")
    assert decomposed != "Άγιος"
    assert normalize_text(decomposed) == unicodedata.normalize("NFC", "Άγιος")


def test_recognize_line_translates_word_boxes_to_page_coords() -> None:
    raw = RawOcr(
        text="Κυ ρι ε",
        words=(
            RawWord("Κυ", BoundingBox(0, 2, 20, 18), 0.9),
            RawWord("ρι", BoundingBox(25, 2, 45, 18), 0.8),
            RawWord("ε", BoundingBox(50, 2, 60, 18), 0.95),
        ),
        confidence=0.88,
    )
    adapter = LyricOcr(FakeBackend(raw))
    bbox = BoundingBox(100, 150, 180, 172)
    line = adapter.recognize_line(_blank_page(), bbox)

    assert line.script == "Greek"
    assert line.direction == "ltr"
    assert line.engine == "fake"
    assert line.boxes_from_geometry is False
    assert line.tokens[0].bbox.to_list() == [100, 152, 120, 168]
    assert line.tokens[1].bbox.to_list() == [125, 152, 145, 168]
    assert [t.text for t in line.tokens] == ["Κυ", "ρι", "ε"]


def test_recognize_line_geometry_fallback_ltr() -> None:
    raw = RawOcr(text="alpha beta gamma", words=(), confidence=0.5)
    adapter = LyricOcr(FakeBackend(raw))
    bbox = BoundingBox(0, 0, 300, 20)
    line = adapter.recognize_line(_blank_page(), bbox)

    assert line.boxes_from_geometry is True
    assert [t.text for t in line.tokens] == ["alpha", "beta", "gamma"]
    xs = [t.bbox.x1 for t in line.tokens]
    assert xs == sorted(xs)  # words laid out left to right
    assert line.tokens[0].bbox.x1 == 0
    assert line.tokens[-1].bbox.x2 == 300
    assert all(t.confidence == 0.5 for t in line.tokens)


def test_recognize_line_geometry_fallback_rtl_orders_first_word_rightmost() -> None:
    raw = RawOcr(text="اول ثان", words=(), confidence=0.6)
    adapter = LyricOcr(FakeBackend(raw))
    bbox = BoundingBox(0, 0, 200, 20)
    line = adapter.recognize_line(_blank_page(), bbox)

    assert line.direction == "rtl"
    # First reading-order word sits in the rightmost box.
    assert line.tokens[0].bbox.x1 > line.tokens[1].bbox.x1


def test_recognize_line_direction_hint_overrides_script() -> None:
    raw = RawOcr(text="Lord have mercy", words=(), confidence=0.5)
    adapter = LyricOcr(FakeBackend(raw))
    line = adapter.recognize_line(_blank_page(), BoundingBox(0, 0, 100, 20), direction_hint="rtl")
    assert line.direction == "rtl"


def test_recognize_line_tries_each_focused_script_pack() -> None:
    backend = FakeBackend(RawOcr(text="x", words=(), confidence=0.0))
    adapter = LyricOcr(backend)
    adapter.recognize_line(_blank_page(), BoundingBox(0, 0, 50, 20))
    used = {langs for _, langs in backend.calls}
    assert used == {("ell",), ("eng",), ("ara",)}


def test_levenshtein_and_error_rates() -> None:
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein(["a", "b"], ["a", "b"]) == 0
    assert char_error_rate("kitten", "sitting") == pytest.approx(3 / 7)
    assert word_error_rate("the quick fox", "the slow fox") == pytest.approx(1 / 3)
    assert char_error_rate("", "abc") == 1.0
    assert char_error_rate("", "") == 0.0


def test_audit_gold_buckets_by_script() -> None:
    # Two gold rows; the fake backend always returns the same Greek text, so the
    # Greek row scores perfectly and the Arabic row scores zero word accuracy.
    raw = RawOcr(text="Κυριε ελεησον", words=(), confidence=0.9)
    adapter = LyricOcr(FakeBackend(raw))
    pages = [
        GoldPage(
            image_path="fake.png",
            rows=(
                GoldRow(BoundingBox(0, 0, 100, 20), "Greek", "ltr", "Κυριε ελεησον"),
                GoldRow(BoundingBox(0, 30, 100, 50), "Arabic", "rtl", "رب ارحم"),
            ),
        )
    ]
    report = audit_gold(pages, adapter, image_loader=lambda _: _blank_page())

    assert isinstance(report, AuditReport)
    by_script = {score.script: score for score in report.by_script()}
    assert by_script["Greek"].word_accuracy == 1.0
    assert by_script["Arabic"].word_accuracy == 0.0
    assert report.engine == "fake"
    assert report.to_dict()["overall"]["word_accuracy"] < 1.0


def test_fold_for_scoring_strips_diacritics_and_normalizes() -> None:
    # Greek accents/breathings removed; final sigma folded.
    assert fold_for_scoring("Κύριε ἐλέησον") == "Κυριε ελεησον"
    assert fold_for_scoring("λόγος") == "λογοσ"
    # Arabic harakat removed, alef/ya/hamza-seat variants folded, tatweel dropped.
    assert fold_for_scoring("لِتَرْ") == "لتر"
    assert fold_for_scoring("أحمد") == "احمد"
    assert fold_for_scoring("صلاـتي") == "صلاتي"
    # Whitespace collapsed.
    assert fold_for_scoring("a   b\tc") == "a b c"


def test_trim_to_text_band_keeps_densest_band() -> None:
    crop = np.full((60, 120), 255, dtype=np.uint8)
    crop[2:5, 40:46] = 0  # sparse neume speck near the top
    crop[30:44, 5:115] = 0  # dense lyric band lower down
    trimmed, y_offset = trim_to_text_band(crop, pad=2)
    assert 24 <= y_offset <= 30
    assert trimmed.shape[0] < crop.shape[0]
    # The dense band survives; the top speck is trimmed away.
    assert (trimmed < 128).sum() > 100


def test_trim_to_text_band_blank_crop_is_unchanged() -> None:
    crop = np.full((20, 50), 255, dtype=np.uint8)
    trimmed, y_offset = trim_to_text_band(crop)
    assert y_offset == 0
    assert trimmed.shape == crop.shape


def test_syllable_spans_splits_on_wide_gaps() -> None:
    crop = np.full((20, 120), 255, dtype=np.uint8)
    crop[5:15, 0:12] = 0
    crop[5:15, 70:82] = 0  # gap of ~58px >> min_gap
    spans = syllable_spans(crop)
    assert len(spans) == 2
    assert spans[0][0] < spans[1][0]


def test_syllable_spans_single_block_when_continuous() -> None:
    crop = np.full((20, 120), 255, dtype=np.uint8)
    crop[5:15, 5:115] = 0
    assert len(syllable_spans(crop)) == 1


def test_tesseract_backend_falls_back_to_psm6_when_psm7_empty(monkeypatch) -> None:
    import pytesseract

    empty = {"text": [""], "conf": ["-1"], "left": [0], "top": [0], "width": [0], "height": [0]}
    found = {"text": ["word"], "conf": ["90"], "left": [5], "top": [1], "width": [20], "height": [10]}
    seen_psm: list[str] = []

    def fake_image_to_data(crop, lang, config, output_type):
        seen_psm.append(config)
        return found if "--psm 6" in config else empty

    monkeypatch.setattr(pytesseract, "image_to_data", fake_image_to_data)
    backend = TesseractBackend()
    result = backend.run(np.zeros((20, 80), dtype=np.uint8), languages=["ell"], direction="ltr")

    assert any("--psm 7" in c for c in seen_psm)
    assert any("--psm 6" in c for c in seen_psm)
    assert result.text == "word"


@pytest.mark.parametrize(
    "fixture",
    ["config/expected_lyrics_synthetic.json", "config/expected_lyrics_real.json"],
)
def test_committed_gold_fixtures_are_well_formed(fixture: str) -> None:
    # Scoring needs tesseract + (for the real set) the local corpus, so this only
    # guards the fixture schema the audit harness depends on.
    pages = load_gold(Path(fixture))
    assert pages
    for page in pages:
        assert page.rows
        for row in page.rows:
            assert row.text
            assert row.script in ("Greek", "Latin", "Arabic", "mixed", "unknown")
            assert row.bbox.x2 > row.bbox.x1 and row.bbox.y2 > row.bbox.y1
