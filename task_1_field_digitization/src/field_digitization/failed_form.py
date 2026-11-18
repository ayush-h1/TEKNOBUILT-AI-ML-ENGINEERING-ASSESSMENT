"""Recognise a page explicitly marked "FAILED TEST FORM".

1. PDF text layer (preferred). The words are normalised (case, punctuation) and the phrase must occur
   as three consecutive words, at any angle. This is exact evidence, so no review is needed.
2. OCR fallback, used only when the page has no text layer AND could not be aligned to the template.
   The page is OCR'd at several rotations, because the marker may be diagonal. A match needs at least
   two of the three words, in order, each within one edit of the expected word. On the sample's page 4
   OCR at -45 degrees read only "ED TEST FORM", so OCR evidence is partial and always needs review.

A page without the marker is never classified as a failed test form.
"""

from __future__ import annotations

import math
import re
from typing import Optional

import cv2
import numpy as np

from .ingest import TextLayer
from .ocr import OcrFailure, OcrHint, OcrProvider, OcrUnavailable
from .schema import FailedFormEvidence, FailedFormEvidenceSource, TextKind, WordBox

PHRASE = ("FAILED", "TEST", "FORM")
OCR_ANGLES = (0, 45, -45, 90, -90, 30, -30, 60, -60)
_OCR_MAX_SIDE = 1400


def _norm(word: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", word.upper())


def from_text_layer(layer: Optional[TextLayer]) -> Optional[FailedFormEvidence]:
    if layer is None or layer.is_empty:
        return None
    words = list(layer.words)
    normalised = [_norm(w.text) for w in words]
    n = len(PHRASE)
    for i in range(len(words) - n + 1):
        if tuple(normalised[i:i + n]) == PHRASE:
            hit = words[i:i + n]
            direction = None
            for line in layer.lines:
                if "FAILEDTESTFORM" in _norm(line.text):
                    dx, dy = line.direction
                    direction = round(math.degrees(math.atan2(-dy, dx)), 1)  # PDF y points down
                    break
            return FailedFormEvidence(
                source=FailedFormEvidenceSource.TEXT_LAYER,
                matched_text=" ".join(w.text for w in hit),
                words=[WordBox(text=w.text, bbox=w.bbox) for w in hit],
                direction_degrees=direction,
                detail="phrase found in the PDF's own text layer",
            )
    return None


def _edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def phrase_match(tokens: list[str]) -> Optional[str]:
    """Return the matched tokens when at least two of FAILED, TEST, FORM appear in order (each within one
    edit) inside a window of three consecutive tokens; otherwise None."""
    tokens = [t for t in (_norm(t) for t in tokens) if t]
    for start in range(len(tokens)):
        matched, k = [], 0
        for token in tokens[start:start + len(PHRASE)]:
            fits = [j for j in range(k, len(PHRASE)) if _edit_distance(token, PHRASE[j]) <= 1]
            if not fits:
                break  # the matched words must be consecutive tokens
            matched.append(token)
            k = fits[0] + 1
        if len(matched) >= 2:
            return " ".join(matched)
    return None


def _rotate(gray: np.ndarray, angle: float) -> np.ndarray:
    h, w = gray.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(gray, m, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=255)


def from_ocr(gray: np.ndarray, provider: OcrProvider) -> Optional[FailedFormEvidence]:
    """Fallback for pages without a text layer. Returns None when OCR is unavailable or nothing matches."""
    scale = min(1.0, _OCR_MAX_SIDE / max(gray.shape))
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray
    for angle in OCR_ANGLES:
        try:
            reading = provider.read(_rotate(small, angle) if angle else small, OcrHint(TextKind.PRINTED, single_line=False))
        except OcrUnavailable:
            return None
        except OcrFailure:
            continue
        matched = phrase_match(reading.text.split())
        if matched:
            return FailedFormEvidence(
                source=FailedFormEvidenceSource.OCR,
                matched_text=matched,
                direction_degrees=float(-angle),
                ocr_engine=reading.engine,
                ocr_confidence=reading.confidence,
                detail=f"OCR of the page rotated by {angle} degrees read '{reading.text.strip()[:80]}'",
            )
    return None
