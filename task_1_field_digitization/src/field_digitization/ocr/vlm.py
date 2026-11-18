"""Optional vision-language-model adapter for handwritten fields. A documented MOCK.

This adapter contains no network code and makes no API calls. It documents exactly how a paid
vision API would be configured, so the integration can be completed later by supplying a `client`
callable. With no client it is simply unavailable.

How it would be configured
--------------------------
Request (one per handwritten field crop):
  - model:        a vision-capable model name chosen by the operator
  - temperature:  0 (deterministic as far as the service allows)
  - max_tokens:   200
  - system:       VLM_SYSTEM_PROMPT below
  - image:        the field crop as a base64 PNG
  - response_schema: VLM_RESPONSE_SCHEMA below (structured output, so no free text is parsed)

Response contract:  {"text": <string, or null if nothing is legible>, "legible": <bool>, "confidence": <0..1, optional>}

Policy that does not change with the engine: whatever a VLM returns for a handwritten field is
still a candidate. It is stored as `needs_review` and never as `extracted`. A VLM's stated
confidence is not treated as calibrated.
"""

from __future__ import annotations

import base64
from typing import Any, Callable, Optional

import cv2
import numpy as np

from ..schema import OcrEngineInfo
from .base import OcrFailure, OcrHint, OcrReading, OcrUnavailable

VLM_SYSTEM_PROMPT = (
    "You transcribe handwriting from one cropped field of a construction safety observation form. "
    "Return exactly what is written, character for character, preserving capitals, digits and punctuation. "
    "Do not correct spelling, do not guess missing characters and do not complete partial words. "
    "If the field is empty or you cannot read it, return null for text and set legible to false. "
    "Never invent a value."
)

VLM_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "text": {"type": ["string", "null"]},
        "legible": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["text", "legible"],
    "additionalProperties": False,
}


class VlmOcr:
    def __init__(
        self,
        client: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
        model: str = "unconfigured",
        temperature: float = 0.0,
        max_tokens: int = 200,
    ) -> None:
        self._client = client
        self.model, self.temperature, self.max_tokens = model, temperature, max_tokens

    def engine_info(self) -> OcrEngineInfo:
        return OcrEngineInfo(name="vlm", version=self.model)

    def build_request(self, image: np.ndarray, hint: OcrHint, field_name: str = "") -> dict[str, Any]:
        """The exact request a real client would send. JSON-serialisable; contains no credentials."""
        ok, png = cv2.imencode(".png", image)
        if not ok:
            raise OcrFailure("could not encode the field image")
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "system": VLM_SYSTEM_PROMPT,
            "field": field_name,
            "digits_only": hint.digits_only,
            "response_schema": VLM_RESPONSE_SCHEMA,
            "image_png_base64": base64.b64encode(png.tobytes()).decode("ascii"),
        }

    def read(self, image: np.ndarray, hint: OcrHint) -> OcrReading:
        if self._client is None:
            raise OcrUnavailable("the VLM adapter is not configured; it makes no network calls by itself")
        request = self.build_request(image, hint)
        try:
            response = self._client(request)
        except Exception as exc:
            raise OcrFailure(f"VLM client failed: {type(exc).__name__}: {exc}") from exc
        if not isinstance(response, dict) or "legible" not in response:
            raise OcrFailure("VLM response did not match the required schema")
        text = response.get("text")
        if not response["legible"] or not text:
            return OcrReading("", None, self.engine_info())
        confidence = response.get("confidence")
        if confidence is not None and not (isinstance(confidence, (int, float)) and 0.0 <= confidence <= 1.0):
            raise OcrFailure("VLM confidence is outside 0..1")
        return OcrReading(str(text), None if confidence is None else float(confidence), self.engine_info())
