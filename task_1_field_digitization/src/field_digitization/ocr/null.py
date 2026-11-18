"""The safe fallback: reads nothing, claims nothing."""

from __future__ import annotations

import numpy as np

from ..schema import OcrEngineInfo
from .base import OcrHint, OcrReading, OcrUnavailable


class NullOcr:
    """Used when no OCR engine is available or OCR is switched off. Every read is 'unavailable', so
    text fields become `ocr_unavailable` with a null value. Checkbox work does not need OCR."""

    def __init__(self, reason: str = "no OCR engine is configured") -> None:
        self.reason = reason

    def engine_info(self) -> OcrEngineInfo:
        return OcrEngineInfo(name="none", version=None)

    def read(self, image: np.ndarray, hint: OcrHint) -> OcrReading:
        raise OcrUnavailable(self.reason)
