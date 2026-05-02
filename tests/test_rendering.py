from pathlib import Path

import numpy as np

from psaltica_ocr.rendering import apply_mask, binarize, mask_lyrics, page_output_path
from tools.render_pdfs import chunk_pages, manifest_pages, parse_page_range


def test_page_output_path_is_deterministic() -> None:
    assert page_output_path(Path("data/pages"), "book", 12) == Path("data/pages/book/page_0012.png")


def test_binarize_returns_ink_as_white_binary() -> None:
    image = np.full((20, 20), 255, dtype=np.uint8)
    image[5:15, 5:15] = 0

    binary = binarize(image)

    assert set(np.unique(binary)) <= {0, 255}
    assert binary[10, 10] == 255
    assert binary[0, 0] == 0


def test_mask_lyrics_keeps_tall_chant_rows_and_drops_short_text_rows() -> None:
    image = np.full((120, 160), 255, dtype=np.uint8)
    image[18:23, 20:140] = 0
    image[46:54, 25:36] = 0
    image[46:54, 45:56] = 0
    image[80:85, 20:140] = 0
    image[108:114, 25:36] = 0

    mask = mask_lyrics(image)

    chant_coverage = np.mean(mask[18:23, :] == 255)
    text_coverage = np.mean(mask[47:53, :] == 255)
    assert chant_coverage > 0.9
    assert text_coverage < 0.2


def test_apply_mask_paints_non_chant_rows_white() -> None:
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:5, :] = 255

    masked = apply_mask(image, mask)

    assert np.all(masked[0, 0] == 255)
    assert np.all(masked[3, 0] == 0)


def test_parse_page_range_expands_and_sorts_pages() -> None:
    assert parse_page_range("5-7,2,10-11") == [2, 5, 6, 7, 10, 11]


def test_chunk_pages_preserves_gaps_and_caps_chunk_size() -> None:
    assert chunk_pages([1, 2, 3, 4, 8, 9], 3) == [(1, 3), (4, 4), (8, 9)]


def test_manifest_pages_reads_completed_book_pages(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "book_id,page_number,image_path,sha256,dpi,width,height,masked\n"
        "book,1,data/pages/book/page_0001.png,abc,400,10,10,false\n",
        encoding="utf-8",
    )

    assert manifest_pages(manifest) == {("book", 1)}
