"""PDF rendering and page-image preprocessing primitives."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image

from psaltica_ocr.reading_order import DirectionMap, DirectionOption, detect_page_direction, resolve_direction

# Poppler can emit very large trusted local page images at 300-400 dpi.
Image.MAX_IMAGE_PIXELS = None


@dataclass(frozen=True)
class RenderedPage:
    book_id: str
    page_number: int
    image_path: Path
    sha256: str
    dpi: int
    width: int
    height: int
    masked: bool
    direction: str
    ink_ratio: float


def page_output_path(output_root: Path, book_id: str, page_number: int) -> Path:
    return output_root / book_id / f"page_{page_number:04d}.png"


def render_pdf_pages(
    pdf_path: Path,
    output_root: Path,
    *,
    book_id: str | None = None,
    dpi: int = 400,
    first_page: int | None = None,
    last_page: int | None = None,
    mask: bool = False,
    direction: DirectionOption = "ltr",
    direction_map: DirectionMap | None = None,
    force: bool = False,
) -> list[RenderedPage]:
    """Render one PDF into deterministic page PNGs under output_root/book_id."""

    pdf_path = Path(pdf_path)
    resolved_book_id = book_id or pdf_path.stem
    images = convert_from_path(
        pdf_path,
        dpi=dpi,
        first_page=first_page,
        last_page=last_page,
        fmt="png",
    )
    start_page = first_page or 1
    rendered: list[RenderedPage] = []

    for offset, image in enumerate(images):
        page_number = start_page + offset
        image_path = page_output_path(output_root, resolved_book_id, page_number)
        image_path.parent.mkdir(parents=True, exist_ok=True)
        if image_path.exists() and not force:
            with Image.open(image_path) as existing:
                width, height = existing.size
                page_direction = _page_direction(
                    pil_to_bgr(existing),
                    resolved_book_id,
                    page_number,
                    direction=direction,
                    direction_map=direction_map,
                )
            rendered.append(
                RenderedPage(
                    resolved_book_id,
                    page_number,
                    image_path,
                    sha256_file(image_path),
                    dpi,
                    width,
                    height,
                    mask,
                    page_direction,
                    compute_ink_ratio(image_path),
                )
            )
            continue

        page_direction = _page_direction(
            pil_to_bgr(image),
            resolved_book_id,
            page_number,
            direction=direction,
            direction_map=direction_map,
        )
        if mask:
            np_image = pil_to_bgr(image)
            chant_mask = mask_lyrics(np_image)
            np_image = apply_mask(np_image, chant_mask)
            image = bgr_to_pil(np_image)

        image.save(image_path)
        rendered.append(
            RenderedPage(
                resolved_book_id,
                page_number,
                image_path,
                sha256_file(image_path),
                dpi,
                image.width,
                image.height,
                mask,
                page_direction,
                compute_ink_ratio(image_path),
            )
        )

    return rendered


def compute_ink_ratio(path: Path) -> float:
    """Return the fraction of pixels darker than 245 in a grayscale image."""
    with Image.open(path) as img:
        pixels = np.asarray(img.convert("L"))
    return float(np.mean(pixels < 245))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_pil(image: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def ensure_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def binarize(image: np.ndarray) -> np.ndarray:
    """Return an Otsu-thresholded binary image with ink as 255."""

    gray = ensure_grayscale(image)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return thresholded


def deskew(image: np.ndarray, *, max_degrees: float = 8.0) -> np.ndarray:
    """Deskew a page image using the minimum-area rectangle of ink pixels."""

    binary = binarize(image)
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) == 0:
        return image.copy()

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    else:
        angle = -angle

    if abs(angle) > max_degrees:
        return image.copy()

    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def dewarp(image: np.ndarray) -> np.ndarray:
    """Placeholder dewarp primitive; v0 keeps geometry unchanged."""

    return image.copy()


def mask_lyrics(image: np.ndarray, *, min_band_height: int = 4) -> np.ndarray:
    """Return a row mask where probable chant rows are 255 and lyrics rows are 0.

    The v0 heuristic finds long, low horizontal components that are common in
    printed Byzantine neumes. Rows without these components, including prose-only
    or lyrics-only rows, are masked out before annotation/training.
    """

    binary = binarize(image)
    candidate_rows = _chant_row_candidates(binary)
    mask = np.zeros(binary.shape, dtype=np.uint8)
    if not candidate_rows:
        return mask

    pad = max(8, min(40, int(binary.shape[0] * 0.008)))
    for start, end in candidate_rows:
        mask[max(0, start - pad) : min(mask.shape[0], end + pad), :] = 255
    return mask


def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Paint masked-out rows white so text does not enter annotation/training."""

    result = image.copy()
    result[mask == 0] = 255
    return result


def _row_bands(active_rows: np.ndarray, *, min_height: int) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    start: int | None = None
    for index, active in enumerate(active_rows):
        if active and start is None:
            start = index
        elif not active and start is not None:
            if index - start >= min_height:
                bands.append((start, index))
            start = None
    if start is not None and len(active_rows) - start >= min_height:
        bands.append((start, len(active_rows)))
    return bands


def _chant_row_candidates(binary: np.ndarray) -> list[tuple[int, int]]:
    height, width = binary.shape
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    active_rows = np.zeros(height, dtype=bool)
    min_width = max(18, int(width * 0.012))
    max_height = max(10, int(height * 0.025))
    row_features: list[tuple[int, int, int, int]] = []

    for label in range(1, component_count):
        x, y, component_width, component_height, area = stats[label]
        _ = x
        if area < 6 or component_height == 0:
            continue
        aspect = component_width / component_height
        if component_width >= min_width and component_height <= max_height and aspect >= 3.0:
            active_rows[y : y + component_height] = True

    for start, end in _row_bands(active_rows, min_height=1):
        long_components = 0
        long_width_sum = 0
        total_components = 0
        roi_start = max(0, start - max_height)
        roi_end = min(height, end + max_height)
        for label in range(1, component_count):
            _, y, component_width, component_height, area = stats[label]
            if area < 6 or component_height == 0:
                continue
            if y + component_height < roi_start or y > roi_end:
                continue
            total_components += 1
            aspect = component_width / component_height
            if component_width >= min_width and component_height <= max_height and aspect >= 3.0:
                long_components += 1
                long_width_sum += int(component_width)
        if _is_probable_chant_row(long_components, long_width_sum, total_components):
            row_features.append((start, end, long_components, long_width_sum))

    if len(row_features) < 2 and not any(long_count >= 8 and width_sum >= min_width * 12 for _, _, long_count, width_sum in row_features):
        return []

    return [(start, end) for start, end, _, _ in row_features]


def _is_probable_chant_row(long_components: int, long_width_sum: int, total_components: int) -> bool:
    if total_components > 85:
        return False
    if long_components >= 8 and long_width_sum >= 300:
        return True
    return long_components >= 3 and long_width_sum >= 150 and total_components <= 75


def iter_pdf_paths(paths: Iterable[Path]) -> list[Path]:
    pdfs: list[Path] = []
    for path in paths:
        path = Path(path)
        if path.is_dir():
            pdfs.extend(sorted(path.glob("*.pdf")))
        elif path.suffix.lower() == ".pdf":
            pdfs.append(path)
    return pdfs


def _page_direction(
    image: np.ndarray,
    book_id: str,
    page_number: int,
    *,
    direction: DirectionOption,
    direction_map: DirectionMap | None,
) -> str:
    resolved = resolve_direction(book_id, page_number, default=direction, direction_map=direction_map)
    if resolved == "auto":
        return detect_page_direction(image)
    return resolved
