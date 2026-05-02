import csv
from pathlib import Path

import yaml

from tools.symbol_stats import (
    cooccurrence_rows,
    frequency_rows,
    load_all_labels,
    load_class_names,
    write_reports,
)


def write_classes(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "names": [
                    "base_neume.Oligon",
                    "modifier_gorgon.Gorgon",
                    "modifier_isson.Pa",
                ]
            }
        ),
        encoding="utf-8",
    )


def test_symbol_stats_frequency_and_cooccurrence(tmp_path: Path) -> None:
    classes = tmp_path / "classes.yaml"
    labels_dir = tmp_path / "book" / "labels"
    labels_dir.mkdir(parents=True)
    label = labels_dir / "page_0001.txt"
    write_classes(classes)
    label.write_text(
        "0 0.20 0.50 0.10 0.10\n"
        "1 0.22 0.40 0.05 0.05\n"
        "0 0.80 0.50 0.10 0.10\n"
        "2 0.78 0.35 0.05 0.05\n",
        encoding="utf-8",
    )

    boxes = load_all_labels([label], load_class_names(classes))

    assert frequency_rows(boxes)[0]["count"] == 2
    assert cooccurrence_rows(boxes) == [
        {"base_icon": "Oligon", "modifier_icon": "Gorgon", "count": 1, "percentage": 0.5},
        {"base_icon": "Oligon", "modifier_icon": "Pa", "count": 1, "percentage": 0.5},
    ]


def test_symbol_stats_writes_reports(tmp_path: Path) -> None:
    classes = tmp_path / "classes.yaml"
    labels_dir = tmp_path / "book" / "labels"
    output_dir = tmp_path / "stats"
    labels_dir.mkdir(parents=True)
    write_classes(classes)
    label = labels_dir / "page_0001.txt"
    label.write_text("0 0.20 0.50 0.10 0.10\n1 0.22 0.40 0.05 0.05\n", encoding="utf-8")

    boxes = load_all_labels([label], load_class_names(classes))
    write_reports(output_dir, boxes)

    with (output_dir / "cooccurrence.csv").open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["base_icon"] == "Oligon"
    assert rows[0]["modifier_icon"] == "Gorgon"
