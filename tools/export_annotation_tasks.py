#!/usr/bin/env python3
"""Create a Label Studio task JSON from rendered page manifest rows."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from urllib.parse import quote


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/pages/manifest.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/annotations/label_studio_tasks.json"))
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument(
        "--local-files-root",
        type=Path,
        default=Path("."),
        help="Label Studio local-files document root. Use the same path in LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT.",
    )
    parser.add_argument("--skip-blank", action="store_true")
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def local_file_url(path: str | Path, *, root: Path) -> str:
    relative = Path(path)
    if relative.is_absolute():
        relative = relative.relative_to(root.resolve())
    return f"/data/local-files/?d={quote(relative.as_posix())}"


def task_rows(
    rows: list[dict[str, str]],
    *,
    limit: int,
    skip_blank: bool,
    local_files_root: Path | None = None,
) -> list[dict[str, object]]:
    tasks: list[dict[str, object]] = []
    for row in rows:
        if skip_blank and float(row.get("ink_ratio", "1") or 1) == 0:
            continue
        image_path = row["image_path"]
        image_value = (
            local_file_url(image_path, root=local_files_root)
            if local_files_root is not None
            else image_path
        )
        tasks.append(
            {
                "data": {
                    "image": image_value,
                    "book_id": row["book_id"],
                    "page_number": int(row["page_number"]),
                    "direction": row.get("direction", "ltr"),
                }
            }
        )
        if len(tasks) >= limit:
            break
    return tasks


def main() -> None:
    args = parse_args()
    rows = read_manifest(args.manifest)
    tasks = task_rows(
        rows,
        limit=args.limit,
        skip_blank=args.skip_blank,
        local_files_root=args.local_files_root.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(tasks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(tasks)} Label Studio tasks to {args.output}.")


if __name__ == "__main__":
    main()
