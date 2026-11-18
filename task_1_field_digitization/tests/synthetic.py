"""SYNTHETIC test fixtures.

Everything here is drawn by code for unit testing. These images and PDFs are NOT the
assessment's sample forms or its template.jpg, and no result produced from them
represents real extraction output.
"""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
import pymupdf

PAGE_W, PAGE_H = 1654, 2339  # A4 at 200 DPI


def make_form_page(
    skew_degrees: float = 0.0, seed: int = 0, noise_sigma: float = 3.0
) -> np.ndarray:
    """A synthetic form-like page: printed lines, labels, checkboxes, pen-like strokes.

    `noise_sigma` is the standard deviation of the Gaussian sensor noise added at the end.
    Use 0 for a noise-free page with the same printed structure.
    """
    rng = np.random.default_rng(seed)
    img = np.full((PAGE_H, PAGE_W), 245, dtype=np.uint8)

    cv2.putText(img, "SYNTHETIC SAFETY FORM", (120, 140), cv2.FONT_HERSHEY_SIMPLEX, 2.0, 20, 4)
    cv2.putText(img, "Observer:", (120, 230), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 30, 2)
    cv2.putText(img, "Date:", (1000, 230), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 30, 2)
    cv2.line(img, (330, 240), (900, 240), 40, 2)
    cv2.line(img, (1130, 240), (1550, 240), 40, 2)

    for i, label in enumerate(["PPE worn", "Area clean", "Hazard observed"]):
        y = 460 + i * 140
        cv2.rectangle(img, (160, y), (240, y + 80), 30, 3)
        cv2.putText(img, label, (290, y + 60), cv2.FONT_HERSHEY_SIMPLEX, 1.3, 30, 2)

    cv2.putText(img, "Safety comments:", (120, 1180), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 30, 2)
    for y in range(1300, 1950, 90):
        cv2.line(img, (120, y), (1540, y), 60, 2)

    # Wobbly strokes standing in for handwriting.
    for _ in range(40):
        x0 = int(rng.integers(340, 880))
        y0 = int(rng.integers(190, 240))
        pts = np.array(
            [[x0 + k * 12, y0 + int(rng.integers(-8, 8))] for k in range(8)], dtype=np.int32
        )
        cv2.polylines(img, [pts], False, 25, 3)

    img = img.astype(np.float32) + rng.normal(0, noise_sigma, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)

    if skew_degrees:
        matrix = cv2.getRotationMatrix2D((PAGE_W / 2, PAGE_H / 2), skew_degrees, 1.0)
        img = cv2.warpAffine(img, matrix, (PAGE_W, PAGE_H), borderValue=245)
    return img


def make_blank_page(seed: int = 1) -> np.ndarray:
    """Clean white paper with faint sensor noise and nothing printed on it."""
    rng = np.random.default_rng(seed)
    img = np.full((PAGE_H, PAGE_W), 248, dtype=np.float32) + rng.normal(0, 2, (PAGE_H, PAGE_W))
    return np.clip(img, 0, 255).astype(np.uint8)


def make_dark_muddy_page(seed: int = 2) -> np.ndarray:
    """A page that is uniformly dark brown-grey with texture, like a muddy scan."""
    rng = np.random.default_rng(seed)
    base = np.full((PAGE_H, PAGE_W), 70, dtype=np.float32)
    blotches = cv2.GaussianBlur(rng.normal(0, 40, (PAGE_H, PAGE_W)).astype(np.float32), (0, 0), 25)
    img = base + blotches + rng.normal(0, 8, (PAGE_H, PAGE_W))
    return np.clip(img, 0, 255).astype(np.uint8)


def make_smeared_page(seed: int = 3) -> np.ndarray:
    """A form page with most of its surface covered by dark smears."""
    img = make_form_page(seed=seed).astype(np.float32)
    rng = np.random.default_rng(seed)
    for _ in range(40):
        center = (int(rng.integers(0, PAGE_W)), int(rng.integers(0, PAGE_H)))
        axes = (int(rng.integers(250, 500)), int(rng.integers(250, 500)))
        mask = np.zeros((PAGE_H, PAGE_W), dtype=np.uint8)
        cv2.ellipse(mask, center, axes, int(rng.integers(0, 180)), 0, 360, 255, -1)
        img[mask > 0] = rng.integers(20, 60)
    return np.clip(img, 0, 255).astype(np.uint8)


def make_low_contrast_page(seed: int = 4) -> np.ndarray:
    """A form page squashed into a narrow grey band, as from a faded or washed-out scan."""
    page = make_form_page(seed=seed).astype(np.float32)
    return np.clip(200 + (page - 245) * 0.15, 0, 255).astype(np.uint8)


def make_blurry_page(seed: int = 5) -> np.ndarray:
    return cv2.GaussianBlur(make_form_page(seed=seed), (0, 0), 10)


def make_small_form_page(seed: int = 0) -> np.ndarray:
    """A synthetic form page at roughly 100 DPI, so test PDFs stay small and fast."""
    return cv2.resize(make_form_page(seed=seed), (827, 1170), interpolation=cv2.INTER_AREA)


def make_pure_white_page() -> np.ndarray:
    """A page that is exactly 255 everywhere, like a bilevel or born-digital blank scan.

    Unlike `make_small_blank_page` it has no sensor noise, so it satisfies the strictest
    "uniform white" test and is the case most exposed to false damage verdicts.
    """
    return np.full((1170, 827), 255, dtype=np.uint8)


def make_small_blank_page(seed: int = 1) -> np.ndarray:
    return cv2.resize(make_blank_page(seed=seed), (827, 1170), interpolation=cv2.INTER_AREA)


def truncate_bytes(data: bytes, keep_fraction: float) -> bytes:
    """Simulate an incomplete download or copy by keeping only the first part of the file."""
    return data[: int(len(data) * keep_fraction)]


def zero_image_stream_span(
    data: bytes, image_number: int, start: float, end: float, image_count: int
) -> bytes:
    """Overwrite part of one embedded image's stream with zero bytes.

    `image_number` is 0-based, in file order, among the `image_count` largest streams
    (which are the page images in a PDF written by `write_pdf`). `start` and `end` are
    fractions of that stream.
    """
    streams = [
        (m.start(1), m.end(1))
        for m in re.finditer(rb"stream\r?\n(.*?)\r?\nendstream", data, re.DOTALL)
    ]
    largest = sorted(sorted(streams, key=lambda s: s[1] - s[0], reverse=True)[:image_count])
    lo, hi = largest[image_number]
    damaged = bytearray(data)
    first = lo + int((hi - lo) * start)
    last = lo + int((hi - lo) * end)
    damaged[first:last] = bytes(last - first)
    return bytes(damaged)


def write_pdf(images: list[np.ndarray], path: Path) -> Path:
    """Embed each grayscale image as a full page in a PDF, like a flatbed scan."""
    doc = pymupdf.open()
    for image in images:
        ok, encoded = cv2.imencode(".png", image)
        assert ok
        page = doc.new_page(width=595, height=842)
        page.insert_image(page.rect, stream=encoded.tobytes())
    doc.save(path)
    doc.close()
    return path
