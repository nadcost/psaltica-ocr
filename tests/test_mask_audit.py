from pathlib import Path

import numpy as np
from PIL import Image

from tools.audit_mask_precision import audit_rows, precision


def test_mask_audit_scores_content_and_blank_pages(tmp_path: Path) -> None:
    content = tmp_path / "content.png"
    blank = tmp_path / "blank.png"
    content_pixels = np.full((100, 100), 255, dtype=np.uint8)
    content_pixels[40:60, 20:80] = 0
    Image.fromarray(content_pixels).save(content)
    Image.fromarray(np.full((100, 100), 255, dtype=np.uint8)).save(blank)

    rows = audit_rows(
        [
            {"image_path": str(content), "expected": "content"},
            {"image_path": str(blank), "expected": "blank"},
        ],
        min_content_ratio=0.01,
        max_blank_ratio=0.001,
    )

    assert precision(rows) == 1.0
