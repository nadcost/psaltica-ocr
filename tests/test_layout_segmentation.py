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


def _draw_long_text_row(image: np.ndarray, y1: int) -> None:
    image[y1 : y1 + 4, 40:100] = 0
    image[y1 + 16 : y1 + 20, 120:185] = 0
    for x in [60, 90, 145, 175]:
        image[y1 - 12 : y1 + 8, x : x + 5] = 0


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


def test_nearby_modifier_bands_merge_into_chant_row() -> None:
    image = _blank_page()
    image[54:59, 80:90] = 0
    image[70:74, 30:85] = 0
    image[70:74, 100:155] = 0
    image[70:74, 170:225] = 0
    image[80:86, 135:150] = 0
    _draw_text_like_row(image, 106)

    layout = segment_page_layout(image)

    assert len(layout.chant_rows) == 1
    assert layout.chant_rows[0].bbox.y1 <= 54
    assert layout.chant_rows[0].bbox.y2 >= 86
    assert len(layout.chant_rows[0].lyric_rows) == 1
    assert layout.unpaired_lyric_rows == ()


def test_long_arabic_like_text_is_not_chant_without_notation_context() -> None:
    image = _blank_page()
    _draw_long_text_row(image, 70)

    layout = segment_page_layout(image)

    assert layout.chant_rows == ()
    assert len(layout.non_score_regions) == 1


def test_fragmented_lyric_bands_merge_before_pairing() -> None:
    image = _blank_page()
    _draw_chant_row(image, 62)
    image[92:104, 30:95] = 0
    image[96:110, 115:170] = 0
    image[112:118, 45:160] = 0

    layout = segment_page_layout(image)

    assert len(layout.chant_rows) == 1
    assert len(layout.chant_rows[0].lyric_rows) == 1
    lyric = layout.chant_rows[0].lyric_rows[0]
    assert lyric.bbox.x1 == 30
    assert lyric.bbox.x2 == 170
    assert lyric.bbox.y1 == 92
    assert lyric.bbox.y2 == 118


def test_far_text_below_last_chant_is_non_score_not_lyrics() -> None:
    image = np.full((500, 240), 255, dtype=np.uint8)
    _draw_chant_row(image, 62)
    _draw_text_like_row(image, 92)
    _draw_text_like_row(image, 420)

    layout = segment_page_layout(image)

    assert len(layout.chant_rows) == 1
    assert len(layout.chant_rows[0].lyric_rows) == 1
    assert layout.chant_rows[0].lyric_rows[0].bbox.y1 == 92
    assert layout.unpaired_lyric_rows == ()
    assert [region.bbox.y1 for region in layout.non_score_regions] == [420]


def test_arabic_like_text_below_chant_pairs_as_lyrics_not_chant() -> None:
    image = _blank_page()
    _draw_chant_row(image, 62)
    _draw_long_text_row(image, 98)

    layout = segment_page_layout(image)

    assert len(layout.chant_rows) == 1
    assert len(layout.chant_rows[0].lyric_rows) == 1
    assert layout.chant_rows[0].lyric_rows[0].bbox.y1 == 86


def test_long_lyric_band_is_not_absorbed_as_lower_modifier() -> None:
    image = _blank_page()
    _draw_chant_row(image, 62)
    image[82:88, 40:195] = 0
    image[94:106, 55:65] = 0
    image[94:106, 170:180] = 0

    layout = segment_page_layout(image)

    assert len(layout.chant_rows) == 1
    assert len(layout.chant_rows[0].lyric_rows) == 1
    assert layout.chant_rows[0].lyric_rows[0].bbox.y1 == 82


def test_overlapping_chant_fragments_merge_into_one_row() -> None:
    image = _blank_page()
    image[54:58, 30:115] = 0
    image[54:58, 130:220] = 0
    image[82:86, 50:135] = 0
    image[82:86, 150:225] = 0
    _draw_text_like_row(image, 112)

    layout = segment_page_layout(image)

    assert len(layout.chant_rows) == 1
    assert layout.chant_rows[0].bbox.y1 <= 54
    assert layout.chant_rows[0].bbox.y2 >= 106
    assert len(layout.chant_rows[0].lyric_rows) == 1
