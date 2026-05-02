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
            )
        )

    return rendered


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

    The v0 heuristic uses horizontal ink projection. Taller and denser bands are
    treated as chant rows; short, sparse bands are treated as lyrics text.
    """

    binary = binarize(image)
    projection = binary.sum(axis=1) / 255
    active_rows = projection > max(2, projection.max() * 0.02)
    bands = _row_bands(active_rows, min_height=min_band_height)
    mask = np.zeros(binary.shape, dtype=np.uint8)
    if not bands:
        return np.full(binary.shape, 255, dtype=np.uint8)

    heights = np.array([end - start for start, end in bands], dtype=float)
    densities = np.array([projection[start:end].mean() for start, end in bands], dtype=float)
    height_cutoff = max(float(np.median(heights)), float(np.percentile(heights, 65)))
    density_cutoff = float(np.percentile(densities, 60))

    for (start, end), height, density in zip(bands, heights, densities):
        if height >= height_cutoff or density >= density_cutoff:
            pad = max(2, int(height * 0.2))
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
