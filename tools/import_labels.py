#!/usr/bin/env python3
"""Convert Label Studio rectangle-label JSON exports to YOLO labels."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

import yaml


@dataclass(frozen=True)
class Box:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export_json", type=Path)
    parser.add_argument("--classes", type=Path, default=Path("config/classes.yaml"))
    parser.add_argument("--output", type=Path, default=Path("data/datasets/psaltica"))
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--copy-images", action="store_true")
    parser.add_argument("--image-root", type=Path, default=Path("."))
    return parser.parse_args()


def load_class_names(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    names = payload["names"]
    if isinstance(names, dict):
        return [names[index] for index in sorted(names)]
    return list(names)


def load_tasks(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError("Label Studio export must be a JSON list")
    return payload


def task_image_path(task: dict, *, image_root: Path) -> Path:
    raw = str(task.get("data", {}).get("image", ""))
    if not raw:
        raise ValueError(f"Task {task.get('id', '<unknown>')} has no data.image")
    parsed = urlparse(raw)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path))
    if parsed.scheme in {"http", "https"}:
        return image_root / Path(unquote(parsed.path)).name
    path = Path(unquote(raw))
    if path.is_absolute():
        return path
    return image_root / path


def selected_results(task: dict) -> list[dict]:
    annotations = task.get("annotations") or []
    if not annotations:
        return []
    completed = [annotation for annotation in annotations if not annotation.get("was_cancelled")]
    annotation = completed[-1] if completed else annotations[-1]
    return annotation.get("result") or []


def boxes_from_task(task: dict, class_to_id: dict[str, int]) -> list[Box]:
    boxes: list[Box] = []
    for result in selected_results(task):
        if result.get("type") != "rectanglelabels":
            continue
        value = result.get("value") or {}
        labels = value.get("rectanglelabels") or []
        if not labels:
            continue
        label = labels[0]
        if label not in class_to_id:
            raise ValueError(f"Unknown label {label!r} in task {task.get('id', '<unknown>')}")
        x = float(value["x"]) / 100
        y = float(value["y"]) / 100
        width = float(value["width"]) / 100
        height = float(value["height"]) / 100
        boxes.append(
            Box(
                class_id=class_to_id[label],
                x_center=min(max(x + width / 2, 0), 1),
                y_center=min(max(y + height / 2, 0), 1),
                width=min(max(width, 0), 1),
                height=min(max(height, 0), 1),
            )
        )
    return boxes


def yolo_line(box: Box) -> str:
    return f"{box.class_id} {box.x_center:.6f} {box.y_center:.6f} {box.width:.6f} {box.height:.6f}"


def write_dataset_yaml(output: Path, class_names: list[str]) -> None:
    payload = {
        "path": str(output.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": class_names,
    }
    with (output / "dataset.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def convert_export(
    export_json: Path,
    *,
    class_names: list[str],
    output: Path,
    split: str,
    image_root: Path,
    copy_images: bool,
) -> int:
    class_to_id = {name: index for index, name in enumerate(class_names)}
    labels_dir = output / "labels" / split
    images_dir = output / "images" / split
    labels_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    converted = 0
    for task in load_tasks(export_json):
        image_path = task_image_path(task, image_root=image_root)
        label_path = labels_dir / f"{image_path.stem}.txt"
        boxes = boxes_from_task(task, class_to_id)
        label_path.write_text("\n".join(yolo_line(box) for box in boxes) + ("\n" if boxes else ""), encoding="utf-8")
        if copy_images and image_path.exists():
            shutil.copy2(image_path, images_dir / image_path.name)
        converted += 1

    write_dataset_yaml(output, class_names)
    return converted


def main() -> None:
    args = parse_args()
    class_names = load_class_names(args.classes)
    count = convert_export(
        args.export_json,
        class_names=class_names,
        output=args.output,
        split=args.split,
        image_root=args.image_root,
        copy_images=args.copy_images,
    )
    print(f"Converted {count} Label Studio tasks to {args.output}.")


if __name__ == "__main__":
    main()
