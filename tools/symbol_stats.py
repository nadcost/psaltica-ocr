#!/usr/bin/env python3
"""Aggregate symbol frequency and co-occurrence stats from YOLO labels."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml


@dataclass(frozen=True)
class LabeledBox:
    class_id: int
    class_name: str
    group: str
    icon: str
    x_center: float
    y_center: float
    width: float
    height: float
    book_id: str
    page_id: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels", nargs="+", type=Path, help="YOLO label files or directories")
    parser.add_argument("--classes", type=Path, default=Path("config/classes.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/annotations/stats"))
    return parser.parse_args()


def load_class_names(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    names = payload["names"]
    if isinstance(names, dict):
        return [names[index] for index in sorted(names)]
    return list(names)


def iter_label_paths(paths: Iterable[Path]) -> list[Path]:
    labels: list[Path] = []
    for path in paths:
        if path.is_dir():
            labels.extend(sorted(path.rglob("*.txt")))
        elif path.suffix == ".txt":
            labels.append(path)
    return labels


def parse_class_name(class_name: str) -> tuple[str, str]:
    group, _, icon = class_name.partition(".")
    return group, icon or class_name


def infer_book_id(label_path: Path) -> str:
    if label_path.parent.name in {"labels", "train", "val", "test"}:
        return label_path.parent.parent.name
    return label_path.parent.name


def load_yolo_labels(label_path: Path, class_names: list[str]) -> list[LabeledBox]:
    boxes: list[LabeledBox] = []
    with label_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) < 5:
                raise ValueError(f"{label_path}:{line_number}: expected at least 5 YOLO columns")
            class_id = int(parts[0])
            class_name = class_names[class_id]
            group, icon = parse_class_name(class_name)
            boxes.append(
                LabeledBox(
                    class_id=class_id,
                    class_name=class_name,
                    group=group,
                    icon=icon,
                    x_center=float(parts[1]),
                    y_center=float(parts[2]),
                    width=float(parts[3]),
                    height=float(parts[4]),
                    book_id=infer_book_id(label_path),
                    page_id=label_path.stem,
                )
            )
    return boxes


def load_all_labels(label_paths: Iterable[Path], class_names: list[str]) -> list[LabeledBox]:
    boxes: list[LabeledBox] = []
    for label_path in label_paths:
        boxes.extend(load_yolo_labels(label_path, class_names))
    return boxes


def frequency_rows(boxes: list[LabeledBox]) -> list[dict[str, object]]:
    total = len(boxes) or 1
    counts = Counter((box.class_name, box.group, box.icon) for box in boxes)
    return [
        {
            "class_name": class_name,
            "group": group,
            "icon": icon,
            "count": count,
            "percentage": count / total,
        }
        for (class_name, group, icon), count in counts.most_common()
    ]


def group_rows(boxes: list[LabeledBox]) -> list[dict[str, object]]:
    total = len(boxes) or 1
    counts = Counter(box.group for box in boxes)
    return [
        {"group": group, "count": count, "percentage": count / total}
        for group, count in counts.most_common()
    ]


def per_book_rows(boxes: list[LabeledBox]) -> list[dict[str, object]]:
    totals = Counter(box.book_id for box in boxes)
    counts = Counter((box.book_id, box.class_name, box.group, box.icon) for box in boxes)
    return [
        {
            "book_id": book_id,
            "class_name": class_name,
            "group": group,
            "icon": icon,
            "count": count,
            "percentage": count / totals[book_id],
        }
        for (book_id, class_name, group, icon), count in counts.most_common()
    ]


def cooccurrence_rows(boxes: list[LabeledBox]) -> list[dict[str, object]]:
    by_page: dict[tuple[str, str], list[LabeledBox]] = {}
    for box in boxes:
        by_page.setdefault((box.book_id, box.page_id), []).append(box)

    pair_counts: Counter[tuple[str, str]] = Counter()
    for page_boxes in by_page.values():
        bases = [box for box in page_boxes if box.group == "base_neume"]
        modifiers = [box for box in page_boxes if box.group.startswith("modifier_")]
        for modifier in modifiers:
            base = nearest_base(modifier, bases)
            if base is not None:
                pair_counts[(base.icon, modifier.icon)] += 1

    total = sum(pair_counts.values()) or 1
    return [
        {
            "base_icon": base_icon,
            "modifier_icon": modifier_icon,
            "count": count,
            "percentage": count / total,
        }
        for (base_icon, modifier_icon), count in pair_counts.most_common()
    ]


def nearest_base(modifier: LabeledBox, bases: list[LabeledBox]) -> LabeledBox | None:
    if not bases:
        return None
    return min(
        bases,
        key=lambda base: abs(base.x_center - modifier.x_center) + abs(base.y_center - modifier.y_center),
    )


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_reports(output_dir: Path, boxes: list[LabeledBox]) -> None:
    write_csv(output_dir / "frequency.csv", frequency_rows(boxes), ["class_name", "group", "icon", "count", "percentage"])
    write_csv(output_dir / "groups.csv", group_rows(boxes), ["group", "count", "percentage"])
    write_csv(
        output_dir / "per_book.csv",
        per_book_rows(boxes),
        ["book_id", "class_name", "group", "icon", "count", "percentage"],
    )
    write_csv(
        output_dir / "cooccurrence.csv",
        cooccurrence_rows(boxes),
        ["base_icon", "modifier_icon", "count", "percentage"],
    )


def main() -> None:
    args = parse_args()
    class_names = load_class_names(args.classes)
    label_paths = iter_label_paths(args.labels)
    if not label_paths:
        raise SystemExit("No YOLO label files found.")
    boxes = load_all_labels(label_paths, class_names)
    write_reports(args.output_dir, boxes)
    print(f"Wrote symbol stats for {len(boxes)} boxes from {len(label_paths)} label files to {args.output_dir}.")


if __name__ == "__main__":
    main()
