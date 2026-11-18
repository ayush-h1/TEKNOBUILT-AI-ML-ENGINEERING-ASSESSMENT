"""Tesseract adapter.

Tesseract is good at printed text and unreliable on handwriting: on the supplied sample it misread
script-font digits at confidences of 70 to 92. The adapter therefore only reports what the engine
said. Whether that text may be trusted is decided by the field policy, never here.

Locating the executable: an installed Tesseract is often not on PATH (a fresh winget install is not
visible to shells that were already open), so the adapter searches, in order:
  1. the explicit path from configuration,
  2. the TESSERACT_CMD environment variable,
  3. PATH,
  4. the default Windows install path C:\\Program Files\\Tesseract-OCR\\tesseract.exe.

Concurrency: pytesseract keeps the executable path in a module-level setting, so this adapter is
not safe to share between threads that use different executables.
"""

from __future__ import annotations

import os
import shutil
from typing import Any, Callable, Mapping, Optional

import cv2
import numpy as np

from ..schema import OcrEngineInfo
from .base import OcrFailure, OcrHint, OcrReading, OcrUnavailable

DEFAULT_WINDOWS_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def locate_tesseract(
    explicit: Optional[str] = None,
    environ: Optional[Mapping[str, str]] = None,
    which: Optional[Callable[[str], Optional[str]]] = None,
    exists: Optional[Callable[[str], bool]] = None,
    default: str = DEFAULT_WINDOWS_PATH,
) -> tuple[Optional[str], str]:
    """Return (path, description of where it was found). `path` is None when nothing was found, and
    the description then lists what was searched."""
    environ = os.environ if environ is None else environ
    which = shutil.which if which is None else which
    exists = os.path.isfile if exists is None else exists
    searched = []
    for source, candidate in (
        ("configuration", explicit),
        ("TESSERACT_CMD environment variable", environ.get("TESSERACT_CMD")),
        ("PATH", which("tesseract")),
        ("default install path", default),
    ):
        if not candidate:
            searched.append(f"{source}: not set")
            continue
        if exists(candidate):
            return candidate, source
        searched.append(f"{source}: '{candidate}' does not exist")
    return None, "; ".join(searched)


class _PytesseractBackend:
    """Thin wrapper over pytesseract. Tests replace it with a fake that has the same four methods."""

    def __init__(self) -> None:
        try:
            import pytesseract
        except ImportError as exc:
            raise OcrUnavailable("the 'pytesseract' package is not installed (pip install pytesseract)") from exc
        self._pt = pytesseract

    def set_cmd(self, path: str) -> None:
        self._pt.pytesseract.tesseract_cmd = path

    def version(self) -> str:
        return str(self._pt.get_tesseract_version())

    def languages(self) -> list[str]:
        return list(self._pt.get_languages(config=""))

    def image_to_data(self, image: np.ndarray, language: str, config: str) -> dict[str, Any]:
        try:
            return self._pt.image_to_data(image, lang=language, config=config, output_type=self._pt.Output.DICT)
        except self._pt.TesseractNotFoundError as exc:
            raise OcrUnavailable(f"Tesseract could not be started: {exc}") from exc
        except self._pt.TesseractError as exc:
            raise OcrFailure(f"Tesseract failed: {exc}") from exc


class TesseractOcr:
    def __init__(
        self,
        cmd: Optional[str] = None,
        language: str = "eng",
        upscale: float = 2.0,
        *,
        environ: Optional[Mapping[str, str]] = None,
        which: Optional[Callable[[str], Optional[str]]] = None,
        exists: Optional[Callable[[str], bool]] = None,
        backend: Any = None,
    ) -> None:
        self.cmd, self.language, self.upscale = cmd, language, upscale
        self._environ, self._which, self._exists = environ, which, exists
        self._backend = backend
        self._version: Optional[str] = None
        self._ready = False
        self._unavailable: Optional[str] = None

    def _ensure(self) -> None:
        if self._ready:
            return
        if self._unavailable:
            raise OcrUnavailable(self._unavailable)
        try:
            path, where = locate_tesseract(self.cmd, self._environ, self._which, self._exists)
            if path is None:
                raise OcrUnavailable(
                    "Tesseract executable not found (" + where + "). Install it (for example "
                    "'winget install UB-Mannheim.TesseractOCR'), set TESSERACT_CMD, or set ocr.tesseract_cmd."
                )
            backend = self._backend or _PytesseractBackend()
            backend.set_cmd(path)
            self._version = backend.version()
            if self.language not in backend.languages():
                raise OcrUnavailable(f"Tesseract language data '{self.language}' is not installed")
            self._backend = backend
            self._ready = True
        except OcrUnavailable as exc:
            self._unavailable = str(exc)
            raise
        except Exception as exc:  # a broken installation must not crash the pipeline
            self._unavailable = f"Tesseract could not be initialised: {type(exc).__name__}: {exc}"
            raise OcrUnavailable(self._unavailable) from exc

    def is_available(self) -> bool:
        try:
            self._ensure()
            return True
        except OcrUnavailable:
            return False

    def engine_info(self) -> OcrEngineInfo:
        return OcrEngineInfo(name="tesseract", version=self._version)

    def read(self, image: np.ndarray, hint: OcrHint) -> OcrReading:
        self._ensure()
        if image is None or image.size == 0:
            raise OcrFailure("empty image")
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.upscale != 1.0:
            gray = cv2.resize(gray, None, fx=self.upscale, fy=self.upscale, interpolation=cv2.INTER_CUBIC)
        gray = cv2.copyMakeBorder(gray, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
        config = f"--psm {7 if hint.single_line else 6}"
        if hint.digits_only:
            config += " -c tessedit_char_whitelist=0123456789"
        try:
            data = self._backend.image_to_data(gray, self.language, config)
        except (OcrUnavailable, OcrFailure):
            raise
        except Exception as exc:
            raise OcrFailure(f"{type(exc).__name__}: {exc}") from exc
        words = []
        for text, conf in zip(data.get("text", []), data.get("conf", [])):
            text = str(text).strip()
            try:
                conf = float(conf)
            except (TypeError, ValueError):
                continue
            if text and conf >= 0:
                words.append((text, conf))
        joined = " ".join(t for t, _ in words)
        confidence = (sum(c for _, c in words) / len(words) / 100.0) if words else None
        return OcrReading(joined, confidence, self.engine_info())
