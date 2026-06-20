#!/usr/bin/env python3
"""Generate and validate Psaltica OCR symbol taxonomy artifacts."""

from __future__ import annotations

import argparse
import difflib
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import yaml

from psaltica_ocr.symbol_map import (
    PLACEHOLDER_GROUP,
    SymbolMap,
    canonical_class_name,
    group_for_class_name,
    is_placeholder_class,
    load_symbol_map,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRAXIS_ROOT = Path("/Users/nadcost/psaltica-praxis")
DEFAULT_SYMBOL_MAP = REPO_ROOT / "config" / "symbol_map.json"
DEFAULT_CLASSES = REPO_ROOT / "config" / "classes.yaml"
DEFAULT_EXTRA_CLASSES = REPO_ROOT / "config" / "extra_classes.yaml"
EXTRACTOR = REPO_ROOT / "tools" / "_extract_symbol_map.ts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--praxis-root", type=Path, default=DEFAULT_PRAXIS_ROOT)
    parser.add_argument("--symbol-map", type=Path, default=DEFAULT_SYMBOL_MAP)
    parser.add_argument("--classes", type=Path, default=DEFAULT_CLASSES)
    parser.add_argument("--extra-classes", type=Path, default=DEFAULT_EXTRA_CLASSES)
    parser.add_argument("--check", action="store_true", help="Fail if generated artifacts differ")
    return parser.parse_args()


def load_extra_classes(path: Path) -> list[str]:
    """Detection-only placeholder class names merged into classes.yaml.

    These have no app glyph (see config/extra_classes.yaml); appending them here
    is what lets them survive a taxonomy re-sync instead of being regenerated
    away. Every name must use the placeholder prefix so export can skip them.
    """
    if not path.exists():
        return []
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    names = payload.get("classes", [])
    for name in names:
        if not is_placeholder_class(name):
            raise SystemExit(
                f"{path}: '{name}' must use the '{PLACEHOLDER_GROUP}.' prefix"
            )
    return names


def run_extractor(praxis_root: Path, output: Path) -> None:
    subprocess.run(
        ["npx", "--yes", "tsx", str(EXTRACTOR), "--out", str(output)],
        cwd=praxis_root,
        check=True,
    )


def class_name(symbol: dict[str, Any]) -> str:
    group = symbol["group"]
    icon = symbol["icon"]
    if group == "neume":
        return f"base_neume.{icon}"
    if group in {"gorgon", "modulation", "isson"}:
        return f"modifier_{group}.{icon}"
    return f"{group}.{icon}"


def build_classes(symbol_map: SymbolMap, extra_classes: list[str] | None = None) -> dict[str, Any]:
    names: list[str] = []
    groups: dict[str, list[str]] = {}
    raw = symbol_map.model_dump(by_alias=True)
    for symbol in raw["symbols"]:
        # Canonicalize so visually identical glyphs (mode vs key signature, and
        # same-GIF aliases) collapse to one class — see canonical_class_name.
        name = canonical_class_name(class_name(symbol))
        if name in names:
            continue
        names.append(name)
        groups.setdefault(group_for_class_name(name), []).append(name)

    # Detection-only placeholders are appended after the app-derived classes so a
    # re-sync keeps them (and their YOLO class ids) stable instead of dropping
    # them — see load_extra_classes / config/extra_classes.yaml.
    for name in extra_classes or []:
        if name not in names:
            names.append(name)
            groups.setdefault(group_for_class_name(name), []).append(name)

    return {
        "path": "../data/datasets",
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": names,
        "groups": groups,
        "source_meta": raw["_meta"],
    }


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def diff_text(expected_path: Path, generated_text: str) -> str:
    expected_text = expected_path.read_text(encoding="utf-8") if expected_path.exists() else ""
    return "".join(
        difflib.unified_diff(
            expected_text.splitlines(keepends=True),
            generated_text.splitlines(keepends=True),
            fromfile=str(expected_path),
            tofile=f"{expected_path} (generated)",
        ),
    )


def check_artifact(path: Path, generated_text: str) -> None:
    diff = diff_text(path, generated_text)
    if diff:
        raise SystemExit(diff)


def main() -> None:
    args = parse_args()
    if not args.praxis_root.exists():
        raise SystemExit(f"Praxis root not found: {args.praxis_root}")

    extra_classes = load_extra_classes(args.extra_classes)

    if args.check:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_symbol_map = Path(temp_dir) / "symbol_map.json"
            temp_classes = Path(temp_dir) / "classes.yaml"
            run_extractor(args.praxis_root, temp_symbol_map)
            generated_map = load_symbol_map(temp_symbol_map)
            write_yaml(temp_classes, build_classes(generated_map, extra_classes))
            check_artifact(args.symbol_map, temp_symbol_map.read_text(encoding="utf-8"))
            check_artifact(args.classes, temp_classes.read_text(encoding="utf-8"))
        return

    args.symbol_map.parent.mkdir(parents=True, exist_ok=True)
    run_extractor(args.praxis_root, args.symbol_map)
    generated_map = load_symbol_map(args.symbol_map)
    write_yaml(args.classes, build_classes(generated_map, extra_classes))


if __name__ == "__main__":
    main()
