from __future__ import annotations

import numpy as np

from psaltica_ocr.font_shape_matching import (
    GlyphShape,
    MatchDetection,
    ShapeGroup,
    group_icon_names,
    group_priorities_from_icons,
    group_similar_shapes,
    group_thresholds_from_icons,
    detection_frequencies,
    load_shape_family_aliases,
    merge_shape_group_aliases,
    non_max_suppression,
    normalize_shape,
    parse_codepoint_ranges,
    shape_similarity,
    write_match_report_html,
)
from tools.match_font_shape_groups import apply_icon_size_filters, parse_family_aliases, template_source_groups


def test_normalize_shape_removes_position_offsets() -> None:
    left = np.full((40, 40), 255, dtype=np.uint8)
    right = np.full((40, 40), 255, dtype=np.uint8)
    left[4:18, 7:18] = 0
    right[18:32, 22:33] = 0

    left_normalized = normalize_shape(left, canvas=24)
    right_normalized = normalize_shape(right, canvas=24)

    assert left_normalized is not None
    assert right_normalized is not None
    assert shape_similarity(left_normalized, right_normalized) > 0.999


def test_group_similar_shapes_groups_by_normalized_ink_shape() -> None:
    square = np.full((24, 24), 255, dtype=np.uint8)
    square[6:18, 6:18] = 0
    same_square = square.copy()
    vertical = np.full((24, 24), 255, dtype=np.uint8)
    vertical[3:21, 10:14] = 0

    groups = group_similar_shapes(
        [
            GlyphShape("U+E001", 0xE001, "square.one", square),
            GlyphShape("U+E002", 0xE002, "square.two", same_square),
            GlyphShape("U+E003", 0xE003, "vertical", vertical),
        ],
        threshold=0.95,
    )

    member_sets = {group.members for group in groups}
    assert ("U+E001", "U+E002") in member_sets
    assert ("U+E003",) in member_sets


def test_shape_similarity_can_compare_horizontal_flip() -> None:
    original = np.full((24, 24), 255, dtype=np.uint8)
    original[4:20, 5:9] = 0
    original[16:20, 5:18] = 0
    flipped = np.fliplr(original)

    assert shape_similarity(original, flipped) < 0.5
    assert shape_similarity(original, flipped, allow_mirror=True) > 0.999


def test_group_similar_shapes_groups_horizontal_flips_by_default() -> None:
    original = np.full((24, 24), 255, dtype=np.uint8)
    original[4:20, 5:9] = 0
    original[16:20, 5:18] = 0
    flipped = np.fliplr(original)

    groups = group_similar_shapes(
        [
            GlyphShape("U+E010", 0xE010, "left.form", original),
            GlyphShape("U+E011", 0xE011, "right.form", flipped),
        ],
        threshold=0.95,
    )

    assert [group.members for group in groups] == [("U+E010", "U+E011")]


def test_non_max_suppression_keeps_best_overlapping_detection() -> None:
    detections = [
        MatchDetection(10, 10, 20, 20, 0.80, "shape_0001", "U+E001", 8.0),
        MatchDetection(11, 11, 20, 20, 0.95, "shape_0002", "U+E002", 8.0),
        MatchDetection(80, 80, 20, 20, 0.70, "shape_0003", "U+E003", 8.0),
    ]

    kept = non_max_suppression(detections, iou_threshold=0.3, complex_first=False)

    assert [detection.group_id for detection in kept] == ["shape_0002", "shape_0003"]


def test_non_max_suppression_prefers_larger_complex_match() -> None:
    detections = [
        MatchDetection(10, 10, 20, 20, 0.99, "small_part", "U+E001", 8.0),
        MatchDetection(8, 8, 36, 24, 0.80, "complex", "U+E002", 8.0),
    ]

    kept = non_max_suppression(detections, iou_threshold=0.2)

    assert [detection.group_id for detection in kept] == ["complex"]


def test_non_max_suppression_protects_high_priority_group() -> None:
    detections = [
        MatchDetection(10, 10, 81, 62, 0.66, "one_plus", "U+004F", 13.0),
        MatchDetection(9, 8, 83, 66, 0.76, "plus_three", "U+0033", 13.0),
    ]

    kept = non_max_suppression(
        detections,
        iou_threshold=0.3,
        priorities={"one_plus": 50, "plus_three": 10},
    )

    assert [detection.group_id for detection in kept] == ["one_plus"]


def test_parse_codepoint_ranges() -> None:
    assert parse_codepoint_ranges(["E0D0-E0D2", "U+0174"]) == [(0xE0D0, 0xE0D2), (0x0174, 0x0174)]


def test_detection_frequencies() -> None:
    pages = [
        {"detections": [{"groupId": "shape_0001"}, {"groupId": "shape_0002"}]},
        {"detections": [{"groupId": "shape_0001"}]},
    ]

    assert detection_frequencies(pages) == {"shape_0001": 2, "shape_0002": 1}


def test_group_thresholds_and_priorities_from_app_names() -> None:
    names = {
        "shape_a": ["Apostrofos"],
        "shape_i": ["Isson2"],
        "shape_o": ["Oligon"],
        "shape_one_plus": ["OnePlusOneUp"],
        "shape_other": ["Apli"],
    }

    assert group_thresholds_from_icons(names, default_threshold=0.75) == {
        "shape_a": 0.65,
        "shape_i": 0.75,
        "shape_o": 0.78,
        "shape_one_plus": 0.65,
        "shape_other": 0.75,
    }
    assert group_priorities_from_icons(names) == {
        "shape_a": 35,
        "shape_i": 40,
        "shape_o": 5,
        "shape_one_plus": 50,
        "shape_other": 10,
    }


def test_group_icon_names_collects_unique_app_names() -> None:
    groups = [type("Group", (), {"id": "shape_1", "members": ("U+E001", "U+E002")})()]
    icons = {0xE001: ["Oligon"], 0xE002: ["Oligon", "Isson2"]}

    assert group_icon_names(groups, {"U+E001": 0xE001, "U+E002": 0xE002}, icons) == {
        "shape_1": ["Oligon", "Isson2"]
    }


def test_apply_icon_size_filters_keeps_only_largest_oligon_template() -> None:
    templates = {
        "shape_o": [(7.0, "small"), (13.0, "large")],
        "shape_one_plus": [(7.0, "small"), (13.0, "large")],
        "shape_i": [(7.0, "small"), (13.0, "large")],
    }

    apply_icon_size_filters(
        templates,
        {"shape_o": ["Oligon"], "shape_one_plus": ["OnePlusOneUp"], "shape_i": ["Isson2"]},
        max_size=13.0,
    )

    assert templates == {
        "shape_o": [(13.0, "large")],
        "shape_one_plus": [(13.0, "large")],
        "shape_i": [(7.0, "small"), (13.0, "large")],
    }


def test_template_source_groups_uses_only_one_plus_representative_for_matching() -> None:
    groups = [
        ShapeGroup("shape_0001", "U+004F", ("U+004C", "U+004F", "U+0150")),
        ShapeGroup("shape_0002", "U+0031", ("U+0031", "U+0131")),
    ]

    source_groups = template_source_groups(
        groups,
        {"shape_0001": ["OnePlusOneUp"], "shape_0002": ["Oligon"]},
    )

    assert source_groups[0].members == ("U+004F",)
    assert source_groups[1].members == ("U+0031", "U+0131")


def test_merge_shape_group_aliases_uses_first_alias_as_representative() -> None:
    groups = [
        type("Group", (), {"id": "shape_0001", "representative": "U+004C", "members": ("U+004C",)})(),
        type("Group", (), {"id": "shape_0002", "representative": "U+004F", "members": ("U+004F",)})(),
        type("Group", (), {"id": "shape_0003", "representative": "U+006C", "members": ("U+006C",)})(),
        type("Group", (), {"id": "shape_0004", "representative": "U+0031", "members": ("U+0031",)})(),
    ]

    merged = merge_shape_group_aliases(groups, aliases=(("U+004F", "U+004C", "U+006C"),))

    merged_by_rep = {group.representative: group for group in merged}
    assert merged_by_rep["U+004F"].members == ("U+004C", "U+004F", "U+006C")
    assert merged_by_rep["U+0031"].members == ("U+0031",)


def test_load_shape_family_aliases_normalizes_configured_members(tmp_path) -> None:
    path = tmp_path / "shape_family_aliases.yaml"
    path.write_text(
        """
families:
  - representative: 004f
    members:
      - U+004C
      - 0x006c
      - U+004C
""",
        encoding="utf-8",
    )

    assert load_shape_family_aliases(path) == (("U+004F", "U+004C", "U+006C"),)


def test_parse_family_aliases_normalizes_codepoints() -> None:
    assert parse_family_aliases(["004F,U+004C,006C"]) == [("U+004F", "U+004C", "U+006C")]


def test_write_match_report_html_includes_members_names_and_frequency(tmp_path) -> None:
    groups_payload = {
        "groups": [
            {
                "id": "shape_0001",
                "representative": "U+E001",
                "members": [
                    {"codepoint": "U+E001", "glyphName": "glyph.one", "icons": ["Oligon"]},
                    {"codepoint": "U+E002", "glyphName": "glyph.two", "icons": []},
                ],
            }
        ]
    }
    pages = [{"image": "page.png", "detections": [{"groupId": "shape_0001"}]}]
    output = tmp_path / "report.html"

    write_match_report_html(
        output,
        font_path=tmp_path / "missing.ttf",
        groups_payload=groups_payload,
        pages=pages,
        match_threshold=0.75,
        shape_threshold=0.86,
    )

    html = output.read_text(encoding="utf-8")
    assert "Glyph matched" in html
    assert "U+E001" in html
    assert "U+E002" in html
    assert "Oligon" in html
    assert "shape_0001" in html
    assert '<span class="count">1</span>' in html
