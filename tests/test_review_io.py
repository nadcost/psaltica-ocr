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
