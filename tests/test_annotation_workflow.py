import json
from pathlib import Path

import yaml

from tools.export_annotation_tasks import local_file_url, task_rows
from tools.export_label_studio_config import label_xml
from tools.import_labels import convert_export, load_class_names


def test_label_studio_config_escapes_labels() -> None:
    xml = label_xml(["base_neume.Oligon", "key_signature.Dhi<Key"])

    assert '<Label value="base_neume.Oligon"' in xml
    assert "Dhi&lt;Key" in xml


def test_import_labels_converts_rectanglelabels_to_yolo(tmp_path: Path) -> None:
    classes = tmp_path / "classes.yaml"
    export = tmp_path / "export.json"
    output = tmp_path / "dataset"
    classes.write_text(yaml.safe_dump({"names": ["base_neume.Oligon", "modifier_gorgon.Gorgon"]}), encoding="utf-8")
    image = tmp_path / "page_0001.png"
    image.write_bytes(b"fake")
    export.write_text(
        json.dumps(
            [
                {
                    "id": 1,
                    "data": {"image": str(image)},
                    "annotations": [
                        {
                            "result": [
                                {
                                    "type": "rectanglelabels",
                                    "value": {
                                        "x": 10,
                                        "y": 20,
                                        "width": 30,
                                        "height": 40,
                                        "rectanglelabels": ["base_neume.Oligon"],
                                    },
                                }
                            ]
                        }
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    count = convert_export(
        export,
        class_names=load_class_names(classes),
        output=output,
        split="train",
        image_root=tmp_path,
        copy_images=False,
    )

    assert count == 1
    assert (output / "labels" / "train" / "page_0001.txt").read_text(encoding="utf-8") == (
        "0 0.250000 0.400000 0.300000 0.400000\n"
    )
    dataset = yaml.safe_load((output / "dataset.yaml").read_text(encoding="utf-8"))
    assert dataset["names"] == ["base_neume.Oligon", "modifier_gorgon.Gorgon"]


def test_annotation_task_rows_include_page_metadata() -> None:
    rows = [
        {
            "image_path": "data/pages/book/page_0001.png",
            "book_id": "book",
            "page_number": "1",
            "direction": "rtl",
        }
    ]

    assert task_rows(rows, limit=50, skip_blank=False, local_files_root=None) == [
        {
            "data": {
                "image": "data/pages/book/page_0001.png",
                "book_id": "book",
                "page_number": 1,
                "direction": "rtl",
            }
        }
    ]


def test_annotation_task_rows_can_emit_label_studio_local_file_urls() -> None:
    rows = [
        {
            "image_path": "data/pages/book with spaces/page_0001.png",
            "book_id": "book",
            "page_number": "1",
            "direction": "ltr",
        }
    ]

    tasks = task_rows(rows, limit=50, skip_blank=False, local_files_root=Path(".").resolve())

    assert tasks[0]["data"]["image"] == "/data/local-files/?d=data/pages/book%20with%20spaces/page_0001.png"


def test_local_file_url_encodes_unicode_and_spaces() -> None:
    assert local_file_url("data/pages/Θεία Λειτουργία/page_0001.png", root=Path(".").resolve()) == (
        "/data/local-files/?d=data/pages/%CE%98%CE%B5%CE%B9%CC%81%CE%B1%20%CE%9B%CE%B5%CE%B9%CF%84%CE%BF%CF%85%CF%81%CE%B3%CE%B9%CC%81%CE%B1/page_0001.png"
    )
