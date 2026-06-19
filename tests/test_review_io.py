"""Tests for review UI coordinate conversions and persistence (psaltica-ocr-0bg)."""

from __future__ import annotations

import json

import pytest

from review_ui.review_io import (
    Box,
    boxes_from_ls_results,
    boxes_to_canvas_objects,
    boxes_to_yolo_lines,
    canvas_objects_to_boxes,
    load_class_names,
    load_page_detections,
    ls_predictions_by_image,
    page_key,
    save_page_detections,
    yolo_lines_to_boxes,
)

CLASSES = ["base_neume.Oligon", "rest.Kratima", "key_signature.PaKey"]


def test_load_class_names_list_and_dict(tmp_path) -> None:
    p = tmp_path / "classes.yaml"
    p.write_text("names:\n- a\n- b\n", encoding="utf-8")
    assert load_class_names(p) == ["a", "b"]
    p.write_text("names:\n  1: b\n  0: a\n", encoding="utf-8")
    assert load_class_names(p) == ["a", "b"]


def test_ls_predictions_parsing(tmp_path) -> None:
    tasks = [
        {
            "data": {"image": "/data/local-files/?d=data/pages_full/Holy Week/page_0110.png"},
            "predictions": [
                {"result": [
                    {"value": {"x": 10, "y": 20, "width": 5, "height": 8, "rectanglelabels": ["base_neume.Oligon"]}, "score": 0.9}
                ]}
            ],
        }
    ]
    path = tmp_path / "preds.json"
    path.write_text(json.dumps(tasks), encoding="utf-8")
    by_image = ls_predictions_by_image(path)
    key = "data/pages_full/Holy Week/page_0110.png"
    assert key in by_image
    boxes = boxes_from_ls_results(by_image[key], width=1000, height=2000)
    assert len(boxes) == 1
    box = boxes[0]
    assert box.cls == "base_neume.Oligon"
    assert box.x1 == pytest.approx(100) and box.y1 == pytest.approx(400)
    assert box.x2 == pytest.approx(150) and box.y2 == pytest.approx(560)


def test_yolo_round_trip() -> None:
    boxes = [Box("base_neume.Oligon", 100, 200, 140, 260), Box("rest.Kratima", 300, 50, 320, 90)]
    class_to_id = {name: i for i, name in enumerate(CLASSES)}
    lines = boxes_to_yolo_lines(boxes, class_to_id, 1000, 2000)
    assert lines[0].split()[0] == "0" and lines[1].split()[0] == "1"
    restored = yolo_lines_to_boxes(lines, CLASSES, 1000, 2000)
    assert len(restored) == 2
    for original, back in zip(boxes, restored):
        assert back.cls == original.cls
        assert back.x1 == pytest.approx(original.x1, abs=0.5)
        assert back.y2 == pytest.approx(original.y2, abs=0.5)


def test_yolo_skips_unknown_class() -> None:
    lines = boxes_to_yolo_lines([Box("not_a_class", 0, 0, 10, 10)], {"base_neume.Oligon": 0}, 100, 100)
    assert lines == []


def test_canvas_round_trip_with_scale_and_resize() -> None:
    boxes = [Box("base_neume.Oligon", 100, 200, 140, 260)]
    scale = 0.5
    objects = boxes_to_canvas_objects(boxes, scale)
    assert objects[0]["left"] == pytest.approx(50) and objects[0]["width"] == pytest.approx(20)
    # Simulate a fabric resize via scaleX/scaleY.
    objects[0]["scaleX"] = 2.0
    restored = canvas_objects_to_boxes(objects, scale, classes=["base_neume.Oligon"])
    assert restored[0].cls == "base_neume.Oligon"
    assert restored[0].x1 == pytest.approx(100)
    assert restored[0].width == pytest.approx(80)  # 20px * scaleX 2 / scale 0.5


def test_canvas_classes_by_index() -> None:
    objects = boxes_to_canvas_objects([Box("a", 0, 0, 10, 10), Box("b", 20, 0, 30, 10)], 1.0)
    restored = canvas_objects_to_boxes(objects, 1.0, classes=["base_neume.Oligon", "rest.Kratima"])
    assert [b.cls for b in restored] == ["base_neume.Oligon", "rest.Kratima"]


def test_key_asset_datauri_reads_gif(tmp_path) -> None:
    from review_ui.review_io import key_asset_datauri, load_key_assets

    (tmp_path / "key_assets" / "chromaticHard").mkdir(parents=True)
    gif = tmp_path / "key_assets" / "chromaticHard" / "DhiKeyChromDure.gif"
    gif.write_bytes(b"GIF89a\x01\x00\x01\x00\x00\x00\x00;")  # tiny valid-ish gif bytes
    mapping = {"DhiKeyChromDure": "key_assets/chromaticHard/DhiKeyChromDure.gif",
               "DhiKeyChrom": "key_assets/chromaticHard/DhiKeyChromDure.gif"}
    (tmp_path / "key_assets.json").write_text(json.dumps(mapping), encoding="utf-8")

    loaded = load_key_assets(tmp_path / "key_assets.json")
    assert loaded == mapping
    uri = key_asset_datauri("key_signature.DhiKeyChromDure", loaded, tmp_path)
    assert uri is not None and uri.startswith("data:image/gif;base64,")
    # The label-aliased class resolves to the same artwork.
    assert key_asset_datauri("mode.DhiKeyChrom", loaded, tmp_path) == uri
    # Unmapped class falls through to None (font fallback handles it elsewhere).
    assert key_asset_datauri("base_neume.Oligon", loaded, tmp_path) is None
    assert load_key_assets(tmp_path / "missing.json") == {}


def test_count_class_instances(tmp_path) -> None:
    from review_ui.review_io import count_class_instances, save_page_detections

    save_page_detections(tmp_path, "book/page_0001.png",
                         [Box("base_neume.Oligon", 0, 0, 9, 9), Box("base_neume.Oligon", 10, 0, 19, 9),
                          Box("rest.Kratima", 20, 0, 29, 9)], CLASSES, 100, 100)
    save_page_detections(tmp_path, "book/page_0002.png",
                         [Box("base_neume.Oligon", 0, 0, 9, 9)], CLASSES, 100, 100)
    counts, coverage, out_of_range = count_class_instances(tmp_path, CLASSES)
    assert counts["base_neume.Oligon"] == 3 and coverage["base_neume.Oligon"] == 2
    assert counts["rest.Kratima"] == 1 and coverage["rest.Kratima"] == 1
    assert counts["key_signature.PaKey"] == 0  # present but unused
    assert out_of_range == 0


def test_canonical_class_name_merges_mode_and_aliases() -> None:
    from psaltica_ocr.symbol_map import canonical_class_name, group_for_class_name

    # Mode martyria fold into key signatures (same artwork).
    assert canonical_class_name("mode.PaKeyChromDure") == "key_signature.PaKeyChromDure"
    assert canonical_class_name("mode.DhiKeyChrom") == "key_signature.DhiKeyChromDure"  # via alias
    # Same-GIF alias within key signatures.
    assert canonical_class_name("key_signature.DhiKeyChrom") == "key_signature.DhiKeyChromDure"
    # Untouched classes pass through.
    assert canonical_class_name("base_neume.Oligon") == "base_neume.Oligon"
    # Group inverse mapping.
    assert group_for_class_name("base_neume.Oligon") == "neume"
    assert group_for_class_name("key_signature.PaKey") == "key_signature"
    assert group_for_class_name("modifier_gorgon.Gorgon") == "gorgon"


def test_page_key_handles_spaces() -> None:
    assert page_key("data/pages_full/Holy Week/page_0110.png") == "Holy_Week_page_0110"


def test_save_and_load_page_detections_round_trip(tmp_path) -> None:
    boxes = [Box("base_neume.Oligon", 100, 200, 140, 260), Box("key_signature.PaKey", 50, 60, 90, 110)]
    image_path = "data/pages_full/Holy Week/page_0110.png"
    save_page_detections(tmp_path, image_path, boxes, CLASSES, 1000, 2000)
    loaded = load_page_detections(tmp_path, image_path, CLASSES, 1000, 2000)
    assert loaded is not None and len(loaded) == 2
    assert {b.cls for b in loaded} == {"base_neume.Oligon", "key_signature.PaKey"}
    assert (tmp_path / "Holy_Week_page_0110" / "detections.yolo").exists()


def test_load_page_detections_missing_returns_none(tmp_path) -> None:
    assert load_page_detections(tmp_path, "x/y/page_0001.png", CLASSES, 100, 100) is None


def test_guess_class_recovers_rendered_glyph() -> None:
    from pathlib import Path

    from psaltica_ocr.template_matching import load_symbol_map, render_template

    from review_ui.review_io import build_glyph_descriptors, guess_class

    inserts = load_symbol_map(Path("config/symbol_map.json"))
    names = ["base_neume.Oligon", "mode.PaKey", "rest.Kratima", "base_neume.Petasti"]
    descriptors = build_glyph_descriptors(names, inserts)
    if not descriptors:
        pytest.skip("font unavailable")
    target = next(c for c in names if c in descriptors)
    crop = render_template(inserts[target.split(".", 1)[1]], 9.0)
    guess, score = guess_class(crop, descriptors)
    assert guess == target
    assert score > 0.5


def test_guess_from_exemplars_matches_same_glyph() -> None:
    from pathlib import Path

    from psaltica_ocr.template_matching import load_symbol_map, render_template

    from review_ui.review_io import crop_descriptor, guess_from_exemplars

    inserts = load_symbol_map(Path("config/symbol_map.json"))
    if "Oligon" not in inserts or "Petasti" not in inserts:
        pytest.skip("font unavailable")
    exemplars = [
        ("base_neume.Oligon", crop_descriptor(render_template(inserts["Oligon"], 8.0))),
        ("base_neume.Petasti", crop_descriptor(render_template(inserts["Petasti"], 8.0))),
    ]
    # A different rendering of Oligon should match the Oligon exemplar.
    guess, score = guess_from_exemplars(render_template(inserts["Oligon"], 9.5), exemplars)
    assert guess == "base_neume.Oligon"
