#!/usr/bin/env python3
"""Copy Psaltica Praxis key-signature GIF artwork into the OCR repo.

Key signatures render as GIF images in the app, not as font glyphs, so the
font-based thumbnails in the review UI never match what the annotator sees
(e.g. DhiKeyChromDure). This runs the TS extractor to learn each key's asset
path, copies the GIFs under ``config/key_assets/`` (self-contained, version
controlled), and writes ``config/key_assets.json`` mapping each key NAME — by
both its ``icon`` and its ``label`` — to the copied file, so classes named after
either resolve. Re-run when the app's key set changes.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRAXIS_ROOT = Path("/Users/nadcost/psaltica-praxis")
EXTRACTOR = REPO_ROOT / "tools" / "_extract_key_assets.ts"
ASSET_DIR = REPO_ROOT / "config" / "key_assets"
MAP_PATH = REPO_ROOT / "config" / "key_assets.json"
ASSET_PREFIX = "app/assets/keys/"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--praxis-root", type=Path, default=DEFAULT_PRAXIS_ROOT)
    return parser.parse_args()


def extract_rows(praxis_root: Path) -> list[dict]:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "key_assets_raw.json"
        subprocess.run(
            ["npx", "--yes", "tsx", str(EXTRACTOR), "--out", str(out)],
            cwd=praxis_root,
            check=True,
        )
        return json.loads(out.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    if not args.praxis_root.exists():
        raise SystemExit(f"Praxis root not found: {args.praxis_root}")

    rows = extract_rows(args.praxis_root)
    if ASSET_DIR.exists():
        shutil.rmtree(ASSET_DIR)

    mapping: dict[str, str] = {}
    copied = 0
    missing: list[str] = []
    for row in rows:
        asset = row["asset"]
        src = args.praxis_root / asset
        # Mirror the path under config/key_assets/ to avoid basename collisions.
        rel = asset[len(ASSET_PREFIX):] if asset.startswith(ASSET_PREFIX) else Path(asset).name
        dest = ASSET_DIR / rel
        if not src.exists():
            missing.append(asset)
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copyfile(src, dest)
            copied += 1
        repo_rel = f"key_assets/{rel}"
        # Resolve by both the app's icon name and its human label.
        for name in {row["icon"], row["label"]}:
            mapping[name] = repo_rel

    MAP_PATH.write_text(json.dumps(mapping, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    print(f"Copied {copied} GIFs -> {ASSET_DIR}")
    print(f"Mapped {len(mapping)} key names -> {MAP_PATH}")
    if missing:
        print(f"WARNING: {len(missing)} asset files not found, e.g. {missing[:3]}")


if __name__ == "__main__":
    main()
