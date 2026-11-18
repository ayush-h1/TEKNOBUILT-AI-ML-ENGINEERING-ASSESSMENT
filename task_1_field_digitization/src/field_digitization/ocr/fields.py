"""Turn an OCR reading into a `TextField`, applying the trust policy.

The policy, in one place:
  * No ink in the region                -> empty_or_unreadable, value null (nothing to read).
  * Engine unavailable or failed        -> ocr_unavailable, value null, with the reason. Never invented.
  * HANDWRITTEN text that was read      -> needs_review ALWAYS. The value is kept as a candidate with
    the engine's confidence and the engine's name. A high confidence or a valid format never promotes
    it: on the supplied sample Tesseract returned wrong digits at 92% and a wrong date at 86%.
  * PRINTED text that was read          -> extracted when the confidence reaches the threshold,
    otherwise needs_review.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ..schema import FieldStatus, TextField, TextKind
from .base import OcrFailure, OcrHint, OcrProvider, OcrUnavailable

logger = logging.getLogger(__name__)

HANDWRITING_REASON = (
    "handwritten text is an OCR candidate and always needs human confirmation; "
    "the engine's confidence is not calibrated and does not prove correctness"
)


def read_text_field(
    provider: OcrProvider,
    image: Optional[np.ndarray],
    *,
    name: str,
    kind: TextKind,
    has_ink: bool = True,
    printed_min_confidence: float = 0.80,
    digits_only: bool = False,
    single_line: bool = True,
) -> TextField:
    if not has_ink or image is None or image.size == 0:
        return TextField(name=name, status=FieldStatus.EMPTY_OR_UNREADABLE, text_kind=kind)
    try:
        reading = provider.read(image, OcrHint(kind, single_line=single_line, digits_only=digits_only))
    except OcrUnavailable as exc:
        return TextField(name=name, status=FieldStatus.OCR_UNAVAILABLE, text_kind=kind, review_reason=str(exc)[:300])
    except OcrFailure as exc:
        return TextField(name=name, status=FieldStatus.OCR_UNAVAILABLE, text_kind=kind,
                         review_reason=f"OCR failed on this region: {exc}"[:300])
    except Exception as exc:  # one broken provider must not stop the page
        logger.exception("unexpected OCR error for field %s", name)
        return TextField(name=name, status=FieldStatus.OCR_UNAVAILABLE, text_kind=kind,
                         review_reason=f"unexpected OCR error: {type(exc).__name__}: {exc}"[:300])

    text = reading.text.strip()
    confidence = reading.confidence
    if not text:
        return TextField(name=name, status=FieldStatus.NEEDS_REVIEW, text_kind=kind, engine=reading.engine,
                         confidence=confidence, review_reason="ink is present but the OCR engine returned no text")
    note = ""
    if confidence is None:  # a value must carry a confidence; 0.0 says "no confidence was reported"
        confidence, note = 0.0, "; the engine reported no confidence (recorded as 0.0)"
    if kind is TextKind.HANDWRITTEN:
        return TextField(name=name, value=text, status=FieldStatus.NEEDS_REVIEW, confidence=confidence,
                         review_reason=HANDWRITING_REASON + note, text_kind=kind, engine=reading.engine)
    if confidence >= printed_min_confidence:
        return TextField(name=name, value=text, status=FieldStatus.EXTRACTED, confidence=confidence,
                         text_kind=kind, engine=reading.engine)
    return TextField(name=name, value=text, status=FieldStatus.NEEDS_REVIEW, confidence=confidence,
                     review_reason=f"confidence {confidence:.2f} is below the threshold {printed_min_confidence:.2f}" + note,
                     text_kind=kind, engine=reading.engine)
