import numpy as np

from psaltica_ocr.layout_segmentation import chant_mask_from_layout, segment_page_layout


def _blank_page() -> np.ndarray:
    return np.full((180, 240), 255, dtype=np.uint8)


def _draw_text_like_row(image: np.ndarray, y1: int, *, x_start: int = 30, glyphs: int = 6) -> None:
    for index in range(glyphs):
        x = x_start + index * 14
        image[y1 : y1 + 12, x : x + 5] = 0


def _draw_chant_row(image: np.ndarray, y1: int, *, x_start: int = 28) -> None:
    for index in range(4):
        x = x_start + index * 45
        image[y1 : y1 + 4, x : x + 32] = 0
    for index in range(3):
        x = x_start + 8 + index * 48
        image[y1 - 10 : y1 - 4, x : x + 7] = 0


def test_segment_page_pairs_lyrics_only_to_chant_row_above() -> None:
    image = _blank_page()
    _draw_text_like_row(image, 20)
    _draw_chant_row(image, 62)
    _draw_text_like_row(image, 88)
    _draw_chant_row(image, 122)
    _draw_text_like_row(image, 148)

    layout = segment_page_layout(image)

    assert len(layout.chant_rows) == 2
    assert len(layout.chant_rows[0].lyric_rows) == 1
    assert layout.chant_rows[0].lyric_rows[0].bbox.y1 == 88
    assert len(layout.chant_rows[1].lyric_rows) == 1
    assert layout.chant_rows[1].lyric_rows[0].bbox.y1 == 148
    assert [region.bbox.y1 for region in layout.non_score_regions] == [20]
    assert layout.unpaired_lyric_rows == ()


def test_lyrics_above_notes_are_not_paired_upward() -> None:
    image = _blank_page()
    _draw_text_like_row(image, 48)
    _draw_chant_row(image, 92)

    layout = segment_page_layout(image)

    assert len(layout.chant_rows) == 1
    assert layout.chant_rows[0].lyric_rows == ()
    assert [region.kind for region in layout.non_score_regions] == ["non_score"]
    assert layout.non_score_regions[0].bbox.y1 == 48


def test_chant_mask_preserves_chant_rows_and_masks_lyrics() -> None:
    image = _blank_page()
    _draw_chant_row(image, 62)
    _draw_text_like_row(image, 88)

    layout = segment_page_layout(image)
    mask = chant_mask_from_layout(layout, pad_y=2)

    assert mask[62, 40] == 255
    assert mask[90, 40] == 0


def test_segment_page_preserves_notation_direction() -> None:
    image = _blank_page()
    _draw_chant_row(image, 62)

    layout = segment_page_layout(image, notation_direction="rtl")

    assert layout.notation_direction == "rtl"
    assert layout.chant_rows[0].notation_direction == "rtl"
