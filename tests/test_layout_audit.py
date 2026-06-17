import json
from pathlib import Path

import numpy as np

from psaltica_ocr.layout_audit import (
    Band,
    GoldChantRow,
    GoldPage,
    audit_gold,
    extract_line_bands,
    load_gold,
    score_page,
)
from psaltica_ocr.layout_segmentation import (
    BoundingBox,
    ChantRow,
    LayoutRegion,
    PageLayout,
)


def _region(y1: int, y2: int) -> LayoutRegion:
    return LayoutRegion(
        kind="lyrics",
        bbox=BoundingBox(0, y1, 200, y2),
        component_count=4,
        long_component_count=0,
        ink_ratio=0.1,
    )


def _layout(rows: list[tuple[int, int, list[tuple[int, int]]]]) -> PageLayout:
    chant_rows = tuple(
        ChantRow(
            index=index,
            bbox=BoundingBox(0, y1, 200, y2),
            notation_direction="ltr",
            lyric_rows=tuple(_region(*lyric) for lyric in lyrics),
        )
        for index, (y1, y2, lyrics) in enumerate(rows)
    )
    return PageLayout(
        width=200,
        height=1000,
        notation_direction="ltr",
        chant_rows=chant_rows,
        unpaired_lyric_rows=(),
        non_score_regions=(),
    )


def _gold(rows: list[tuple[int, int, list[tuple[int, int]]]]) -> GoldPage:
    return GoldPage(
        image_path="x.png",
        chant_rows=tuple(
            GoldChantRow(Band(y1, y2), tuple(Band(*lyric) for lyric in lyrics))
            for y1, y2, lyrics in rows
        ),
        non_score=(),
    )


def test_perfect_match_scores_one() -> None:
    layout = _layout([(100, 200, [(210, 260)]), (400, 500, [(510, 560)])])
    gold = _gold([(100, 200, [(210, 260)]), (400, 500, [(510, 560)])])

    score = score_page(layout, gold)

    assert score.chant_mask_precision == 1.0
    assert score.chant_mask_recall == 1.0
    assert score.lyric_pairing_precision == 1.0
    assert score.lyric_pairing_recall == 1.0


def test_chant_box_absorbing_lyrics_drops_precision() -> None:
    # Predicted chant box spans the lyric region too; the lyric is unpaired so
    # the mask does not subtract it -> half the predicted chant pixels are wrong.
    layout = _layout([(100, 300, [])])
    gold = _gold([(100, 200, [(210, 290)])])

    score = score_page(layout, gold)

    assert score.chant_mask_precision == 0.5
    assert score.lyric_pairing_precision == 1.0  # no pairings predicted
    assert score.lyric_pairing_recall == 0.0


def test_lyric_paired_to_wrong_chant_is_incorrect() -> None:
    layout = _layout([(100, 200, [(800, 850)])])
    gold = _gold([(100, 200, [(210, 260)])])

    score = score_page(layout, gold)

    assert score.lyric_pairing_precision == 0.0
    assert score.lyric_pairing_recall == 0.0


def test_audit_gold_aggregates_with_image_loader() -> None:
    gold_pages = [_gold([(100, 200, [(210, 260)])])]
    report = audit_gold(gold_pages, image_loader=lambda _path: np.full((1000, 200), 255, np.uint8))

    # A blank page yields no chant rows: zero predicted pixels -> precision 1.0,
    # zero recall against the gold band.
    assert report.chant_mask_precision == 1.0
    assert report.chant_mask_recall == 0.0


def test_load_gold_round_trip(tmp_path: Path) -> None:
    fixture = {
        "pages": [
            {
                "image_path": "p.png",
                "chant_rows": [{"y1": 10, "y2": 20, "lyrics": [{"y1": 22, "y2": 30}]}],
                "non_score": [{"y1": 0, "y2": 5}],
            }
        ]
    }
    path = tmp_path / "gold.json"
    path.write_text(json.dumps(fixture), encoding="utf-8")

    pages = load_gold(path)

    assert len(pages) == 1
    assert pages[0].chant_rows[0].band == Band(10, 20)
    assert pages[0].chant_rows[0].lyrics == (Band(22, 30),)
    assert pages[0].non_score == (Band(0, 5),)


def test_extract_line_bands_finds_separated_rows() -> None:
    image = np.full((300, 200), 255, np.uint8)
    image[40:60, 20:180] = 0
    image[120:140, 20:180] = 0
    image[200:220, 20:180] = 0

    bands = extract_line_bands(image, min_gap=10, min_height=4)

    assert len(bands) == 3
    assert bands[0].y1 <= 40 and bands[0].y2 >= 60
    assert bands[1].y1 <= 120 and bands[1].y2 >= 140
