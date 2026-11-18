"""The OCR provider interface.

A provider turns an image of a text region into a candidate string. It never decides whether
that string is correct: that is the field policy's job (see `fields.py`).

Why a separate interface: OCR engines differ, may be absent, and for handwriting are unreliable.
The pipeline depends only on `OcrProvider`, so an engine can be swapped or missing without
touching extraction code. A missing engine raises `OcrUnavailable`; the pipeline then records
`ocr_unavailable` with a null value and carries on. It never invents text.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

import numpy as np

from ..schema import OcrEngineInfo, TextKind


class OcrUnavailable(Exception):
    """No usable OCR engine (not installed, not configured, or disabled)."""


class OcrFailure(Exception):
    """The engine exists but failed on this region."""


@dataclass(frozen=True)
class OcrHint:
    kind: TextKind
    single_line: bool = True
    digits_only: bool = False


@dataclass(frozen=True)
class OcrReading:
    text: str
    confidence: Optional[float]  # 0..1; None when the engine reports no confidence
    engine: OcrEngineInfo


@runtime_checkable
class OcrProvider(Protocol):
    def engine_info(self) -> OcrEngineInfo: ...

    def read(self, image: np.ndarray, hint: OcrHint) -> OcrReading:
        """Return the engine's text for `image`. Raise OcrUnavailable or OcrFailure, never invent text."""
        ...
