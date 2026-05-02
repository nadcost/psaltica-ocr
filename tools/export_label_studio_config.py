#!/usr/bin/env python3
"""Export a Label Studio rectangle-label config from config/classes.yaml."""

from __future__ import annotations

import argparse
import html
from pathlib import Path

import yaml


PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classes", type=Path, default=Path("config/classes.yaml"))
    parser.add_argument("--output", type=Path, default=Path("config/label_studio.xml"))
    return parser.parse_args()


def load_class_names(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    names = payload["names"]
    if isinstance(names, dict):
        return [names[index] for index in sorted(names)]
    return list(names)


def label_xml(class_names: list[str]) -> str:
    labels = "\n".join(
        f'    <Label value="{html.escape(name, quote=True)}" background="{PALETTE[index % len(PALETTE)]}"/>'
        for index, name in enumerate(class_names)
    )
    return (
        "<View>\n"
        '  <Image name="image" value="$image" zoom="true" zoomControl="true" rotateControl="false"/>\n'
        '  <RectangleLabels name="label" toName="image" strokeWidth="2" canRotate="false">\n'
        f"{labels}\n"
        "  </RectangleLabels>\n"
        "</View>\n"
    )


def main() -> None:
    args = parse_args()
    class_names = load_class_names(args.classes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(label_xml(class_names), encoding="utf-8")
    print(f"Wrote {len(class_names)} labels to {args.output}.")


if __name__ == "__main__":
    main()
