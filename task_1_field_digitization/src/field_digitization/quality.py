"""Page quality gate.

Runs before any OCR so that blank, muddy or out-of-focus pages are stopped early.
That saves compute (and OCR/VLM cost) and, more importantly, prevents recognisers
from "reading" noise and returning plausible but invented text.

Configurable versus fixed
-------------------------
Configurable decision thresholds (`QualityConfig`, loaded from config/thresholds.yaml):
  ink_relative_threshold, mark_relative_threshold, min_content_extent_ratio,
  min_blank_paper_level, max_ink_ratio, min_mean_brightness, min_contrast, min_sharpness,
  sharpness_reference_width, max_noise_sigma, speckle_min_difference, max_speckle_ratio.

Fixed implementation constants (NOT configurable; changing them means editing the code):
  * background estimate: Gaussian blur with a kernel of max(height, width) // 20 pixels
  * contrast uses the 10th percentile of grey values inside content pixels
  * content components use 8-connectivity
  * the sharpness region is content dilated with a 5x5 kernel, Laplacian measured in float64
  * the noise estimator's 3x3 Immerkaer kernel and the 0.6745 median-to-sigma factor
  * the speckle test uses a 3x3 median and a 3x3 neighbourhood for the isolation check
  * 16-bit input is scaled linearly by 255/65535
Ingestion has its own fixed constants (see ingest.py: what counts as "uniform white" and as
a "zero-filled" row).

Every default, configurable or fixed, was chosen on synthetic images only. None has been
validated on real scans.

Measurements, all taken from one grayscale image:

  * mark mask       - pixels even faintly darker than their local background
                      (background = a heavy blur of the page, so uneven lighting and a
                      uniformly grey scan do not count as marks).
  * content         - connected groups of marked pixels that are long enough to be a
                      stroke or printed line (`min_content_extent_ratio` of the page).
                      Dust and speckle are small and compact, a word or a ruled line is
                      not. Total ink area is deliberately NOT used: one short handwritten
                      word has less ink than a few dust specks.
  * ink_ratio       - fraction of pixels clearly darker than the background (mud, stains).
  * mean_brightness / paper_level - mean and median grey level of the scan.
  * contrast        - paper level minus the 10th percentile of grey inside content pixels.
                      Measured on content only, so it works for sparse pages.
  * sharpness       - variance of the Laplacian, measured only around content, on a
                      width-normalised copy (independent of DPI and of how much of the
                      page is filled).
  * noise_sigma     - Immerkaer's Gaussian-noise estimator, using the median of the
                      response instead of the mean so printed edges do not inflate it.
  * speckle_ratio   - fraction of pixels that are ISOLATED impulses: a pixel differing from
                      its 3x3 median by at least `speckle_min_difference` while none of its
                      eight neighbours does. Catches salt-and-pepper noise, which the
                      median-based sigma ignores. The isolation rule keeps one-pixel wide
                      printed lines out of the count: a 3x3 median erases such a line, but
                      its pixels deviate together and so are not isolated.

Decision logic:
  * No content and a paper-like background and no noise  -> BLANK.
  * No content but a grey/dark background or noise        -> UNREADABLE. An empty page
    is only called blank when the scan itself looks trustworthy.
  * Otherwise any failed check makes the page UNREADABLE; if none fails it is OK.

Errs toward "not blank": wrongly passing an empty page on is harmless (later stages find
no fields), wrongly calling a written page blank loses data.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .config import QualityConfig
from .schema import PageQuality, QualityMetrics, QualityVerdict

# Immerkaer (1996) kernel. For white noise of std s the response has std 6*s.
_NOISE_KERNEL = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
_MEDIAN_ABS_TO_SIGMA = 0.6745  # median(|x|) / std for a zero-mean Gaussian


def compute_metrics(image: np.ndarray, config: QualityConfig) -> QualityMetrics:
    gray = _as_gray_uint8(image)

    # Local background estimate: a heavy blur is dominated by paper, not by thin strokes.
    kernel = _odd(max(gray.shape) // 20)
    background = cv2.GaussianBlur(gray, (kernel, kernel), 0).astype(np.float32)
    background = np.maximum(background, 1.0)
    relative = gray.astype(np.float32) / background
    ink_ratio = float(np.mean(relative < config.ink_relative_threshold))
    mark_mask = relative < config.mark_relative_threshold
    mark_ratio = float(np.mean(mark_mask))

    content_mask, content_components = _find_content(mark_mask, config)
    paper_level = float(np.median(gray))

    if content_mask is not None:
        ink_level = float(np.percentile(gray[content_mask], 10))
        contrast = max(0.0, paper_level - ink_level)
    else:
        contrast = 0.0

    return QualityMetrics(
        mark_ratio=mark_ratio,
        ink_ratio=ink_ratio,
        content_components=content_components,
        mean_brightness=float(gray.mean()),
        paper_level=paper_level,
        contrast=contrast,
        sharpness=_content_sharpness(gray, content_mask, config),
        noise_sigma=_estimate_noise_sigma(gray),
        speckle_ratio=_speckle_ratio(gray, config),
    )


def assess_quality(image: np.ndarray, config: Optional[QualityConfig] = None) -> PageQuality:
    """Classify a page as OK, BLANK or UNREADABLE and explain why."""
    config = config or QualityConfig()
    metrics = compute_metrics(image, config)

    too_dark = metrics.mean_brightness < config.min_mean_brightness
    noisy = (
        metrics.noise_sigma > config.max_noise_sigma
        or metrics.speckle_ratio > config.max_speckle_ratio
    )

    if metrics.content_components == 0:
        reasons: list[str] = []
        if too_dark:
            reasons.append("page is too dark")
        if metrics.paper_level < config.min_blank_paper_level:
            reasons.append("background is not paper-like (uniform grey or degraded scan)")
        if noisy:
            reasons.append("page is too noisy to confirm that it is empty")
        if not reasons:
            return PageQuality(
                verdict=QualityVerdict.BLANK,
                reasons=["no marks of stroke size detected on the page"],
                metrics=metrics,
            )
        return PageQuality(verdict=QualityVerdict.UNREADABLE, reasons=reasons, metrics=metrics)

    reasons = []
    if too_dark:
        reasons.append("page is too dark")
    if metrics.ink_ratio > config.max_ink_ratio:
        reasons.append("page is mostly covered by dark marks")
    if metrics.contrast < config.min_contrast:
        reasons.append("contrast is too low")
    if metrics.sharpness < config.min_sharpness:
        reasons.append("page is out of focus")
    if noisy:
        reasons.append("page is too noisy")

    verdict = QualityVerdict.UNREADABLE if reasons else QualityVerdict.OK
    return PageQuality(verdict=verdict, reasons=reasons, metrics=metrics)


def _find_content(
    mark_mask: np.ndarray, config: QualityConfig
) -> tuple[Optional[np.ndarray], int]:
    """Return (mask of content pixels or None, number of content components)."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mark_mask.astype(np.uint8), connectivity=8
    )
    if count <= 1:
        return None, 0
    extent = np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT])
    min_extent = max(2, int(round(config.min_content_extent_ratio * max(mark_mask.shape))))
    keep = extent >= min_extent
    keep[0] = False  # label 0 is the background
    kept = int(keep.sum())
    if kept == 0:
        return None, 0
    return keep[labels], kept


def _content_sharpness(
    gray: np.ndarray, content_mask: Optional[np.ndarray], config: QualityConfig
) -> float:
    if content_mask is None:
        return 0.0
    height, width = gray.shape
    ref_w = config.sharpness_reference_width
    ref_h = max(1, int(round(height * ref_w / float(width))))
    reference = cv2.resize(gray, (ref_w, ref_h), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(content_mask.astype(np.uint8) * 255, (ref_w, ref_h), interpolation=cv2.INTER_AREA)
    mask = cv2.dilate((mask > 0).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    laplacian = cv2.Laplacian(reference, cv2.CV_64F)
    return float(laplacian[mask].var())


def _estimate_noise_sigma(gray: np.ndarray) -> float:
    response = cv2.filter2D(gray.astype(np.float32), cv2.CV_32F, _NOISE_KERNEL)
    return float(np.median(np.abs(response)) / (_MEDIAN_ABS_TO_SIGMA * 6.0))


def _speckle_ratio(gray: np.ndarray, config: QualityConfig) -> float:
    """Share of pixels that are isolated impulses (salt-and-pepper noise).

    A pixel is an impulse when it differs strongly from its 3x3 median AND none of its
    eight neighbours does. The isolation test matters: a 3x3 median also erases one-pixel
    wide printed lines (common when a page is rendered at low DPI), but the pixels of a
    line deviate together, so they are not isolated and are not counted.
    """
    deviates = (cv2.absdiff(gray, cv2.medianBlur(gray, 3)) >= config.speckle_min_difference)
    deviating = deviates.astype(np.float32)
    neighbours = cv2.boxFilter(deviating, -1, (3, 3), normalize=False) - deviating
    return float(np.mean(deviates & (neighbours == 0)))


def _as_gray_uint8(image: np.ndarray) -> np.ndarray:
    """Return a 2-D uint8 image. 8-bit input is used as is, 16-bit input is scaled
    linearly to 0-255, and every other dtype is rejected rather than guessed at."""
    if image is None or image.size == 0:
        raise ValueError("image is empty")
    if image.dtype == np.uint16:
        image = np.rint(image.astype(np.float32) * (255.0 / 65535.0)).astype(np.uint8)
    elif image.dtype != np.uint8:
        raise ValueError(f"unsupported image dtype {image.dtype}; expected uint8 or uint16")
    if image.ndim == 3 and image.shape[2] == 1:
        image = image[:, :, 0]
    elif image.ndim == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    elif image.ndim != 2:
        raise ValueError(f"unsupported image shape: {image.shape}")
    return np.ascontiguousarray(image)


def _odd(value: int) -> int:
    value = max(3, value)
    return value if value % 2 == 1 else value + 1
