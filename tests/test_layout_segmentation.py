import numpy as np

from psaltica_ocr.layout_segmentation import chant_mask_from_layout, segment_page_layout


def _blank(height: int = 400, width: int = 400) -> np.ndarray:
    return np.full((height, width), 255, dtype=np.uint8)


def _text_row(image: np.ndarray, y: int, *, x0: int = 30, glyphs: int = 10, height: int = 26) -> None:
    """A lyric-like row: many separate glyph components, no long strokes."""
    for index in range(glyphs):
        x = x0 + index * 16
        image[y : y + height, x : x + 6] = 0


def _chant_row(image: np.ndarray, y: int, *, x0: int = 20, strokes: int = 6) -> None:
    """A neume-like row: several long thin horizontal ligature strokes."""
    for index in range(strokes):
        x = x0 + index * 34
        image[y : y + 5, x : x + 26] = 0


def _kashida_text_row(image: np.ndarray, y: int, *, glyphs: int = 12) -> None:
    """An Arabic-like lyric row: many glyphs plus a couple of kashida strokes."""
    _text_row(image, y, glyphs=glyphs)
    image[y + 12 : y + 16, 60:140] = 0
    image[y + 12 : y + 16, 200:280] = 0


def test_segment_page_pairs_lyrics_only_to_chant_row_above() -> None:
    image = _blank()
    _text_row(image, 40)
    _chant_row(image, 130)
    _text_row(image, 180)
    _chant_row(image, 270)
    _text_row(image, 320)

    layout = segment_page_layout(image)

    assert len(layout.chant_rows) == 2
    assert [len(row.lyric_rows) for row in layout.chant_rows] == [1, 1]
    assert layout.chant_rows[0].lyric_rows[0].bbox.y1 == 180
    assert layout.chant_rows[1].lyric_rows[0].bbox.y1 == 320
    # The text row above the first chant is never paired upward.
    assert [region.bbox.y1 for region in layout.non_score_regions] == [40]
    assert layout.unpaired_lyric_rows == ()


def test_lyrics_above_notes_are_not_paired_upward() -> None:
    image = _blank(height=300)
    _text_row(image, 80)
    _chant_row(image, 180)

    layout = segment_page_layout(image)

    assert len(layout.chant_rows) == 1
    assert layout.chant_rows[0].lyric_rows == ()
    assert [region.bbox.y1 for region in layout.non_score_regions] == [80]


def test_chant_mask_preserves_chant_rows_and_masks_lyrics() -> None:
    image = _blank()
    _chant_row(image, 130)
    _text_row(image, 180)

    layout = segment_page_layout(image)
    mask = chant_mask_from_layout(layout, pad_y=2)

    assert mask[131, 40] == 255
    assert mask[190, 40] == 0


def test_segment_page_preserves_notation_direction() -> None:
    image = _blank()
    _chant_row(image, 130)

    layout = segment_page_layout(image, notation_direction="rtl")

    assert layout.notation_direction == "rtl"
    assert layout.chant_rows[0].notation_direction == "rtl"


def test_far_text_below_last_chant_is_non_score_not_lyrics() -> None:
    image = _blank(height=600)
    _chant_row(image, 130)
    _text_row(image, 180)
    _text_row(image, 520)

    layout = segment_page_layout(image)

    assert len(layout.chant_rows) == 1
    assert len(layout.chant_rows[0].lyric_rows) == 1
    assert layout.chant_rows[0].lyric_rows[0].bbox.y1 == 180
    assert [region.bbox.y1 for region in layout.non_score_regions] == [520]
    assert layout.unpaired_lyric_rows == ()


def test_kashida_heavy_text_alone_is_not_chant() -> None:
    image = _blank(height=300)
    _kashida_text_row(image, 150)

    layout = segment_page_layout(image)

    # A few wide kashida strokes among many letters must not read as a neume row.
    assert layout.chant_rows == ()
    assert len(layout.non_score_regions) == 1


def test_kashida_text_below_chant_pairs_as_lyrics() -> None:
    image = _blank()
    _chant_row(image, 130)
    _kashida_text_row(image, 175)

    layout = segment_page_layout(image)

    assert len(layout.chant_rows) == 1
    assert len(layout.chant_rows[0].lyric_rows) == 1
    assert layout.chant_rows[0].lyric_rows[0].bbox.y1 == 175


def test_chant_row_requires_enough_ligature_strokes() -> None:
    image = _blank(height=300)
    _chant_row(image, 150, strokes=2)
    _text_row(image, 200)

    layout = segment_page_layout(image)

    # Two horizontal strokes are too few to be a neume row.
    assert layout.chant_rows == ()
