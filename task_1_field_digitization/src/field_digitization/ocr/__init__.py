"""OCR providers and the field trust policy."""

from __future__ import annotations

from ..config import OcrConfig
from .base import OcrFailure, OcrHint, OcrProvider, OcrReading, OcrUnavailable
from .fields import HANDWRITING_REASON, read_text_field
from .null import NullOcr
from .tesseract import TesseractOcr, locate_tesseract
from .vlm import VLM_RESPONSE_SCHEMA, VLM_SYSTEM_PROMPT, VlmOcr


def make_provider(setting: str, config: OcrConfig) -> OcrProvider:
    """'none' -> NullOcr; 'tesseract' -> Tesseract (unavailability surfaces per field);
    'auto' -> Tesseract when it can be found and started, otherwise NullOcr with the reason."""
    if setting == "none":
        return NullOcr("OCR is switched off in the configuration")
    tess = TesseractOcr(config.tesseract_cmd, config.language, config.upscale)
    if setting == "tesseract" or tess.is_available():
        return tess
    try:
        tess._ensure()
    except OcrUnavailable as exc:
        return NullOcr(str(exc))
    return tess


__all__ = [
    "HANDWRITING_REASON", "NullOcr", "OcrFailure", "OcrHint", "OcrProvider", "OcrReading", "OcrUnavailable",
    "TesseractOcr", "VLM_RESPONSE_SCHEMA", "VLM_SYSTEM_PROMPT", "VlmOcr", "locate_tesseract", "make_provider",
    "read_text_field",
]
