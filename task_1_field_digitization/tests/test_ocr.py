"""OCR providers and the field trust policy. All provider behaviour is tested with fakes; one
smoke test uses the real Tesseract and is skipped when it is not installed."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from field_digitization.config import OcrConfig
from field_digitization.ocr import (
    HANDWRITING_REASON, NullOcr, OcrFailure, OcrHint, OcrReading, OcrUnavailable, TesseractOcr, VlmOcr,
    locate_tesseract, make_provider, read_text_field,
)
from field_digitization.schema import FieldStatus, OcrEngineInfo, TextKind

IMG = np.full((30, 120), 255, np.uint8)
IMG[10:14, 10:100] = 0
ENGINE = OcrEngineInfo(name="fake", version="1")


class FakeProvider:
    def __init__(self, text="", confidence=0.9, exc=None):
        self.text, self.confidence, self.exc = text, confidence, exc

    def engine_info(self):
        return ENGINE

    def read(self, image, hint):
        if self.exc:
            raise self.exc
        return OcrReading(self.text, self.confidence, ENGINE)


# ---------------------------------------------------------------- NullOcr
def test_null_ocr_is_unavailable():
    with pytest.raises(OcrUnavailable):
        NullOcr("off").read(IMG, OcrHint(TextKind.PRINTED))
    assert NullOcr().engine_info().name == "none"


# ---------------------------------------------------------------- locate_tesseract
def _loc(explicit=None, env=None, on_path=None, existing=(), default="C:/default/tesseract.exe"):
    return locate_tesseract(explicit, env or {}, lambda _n: on_path, lambda p: p in existing, default)


def test_locate_order_explicit_first():
    path, src = _loc("a", {"TESSERACT_CMD": "b"}, "c", existing=("a", "b", "c", "C:/default/tesseract.exe"))
    assert (path, src) == ("a", "configuration")


def test_locate_env_then_path_then_default():
    existing = ("b", "c", "C:/default/tesseract.exe")
    assert _loc(None, {"TESSERACT_CMD": "b"}, "c", existing)[0] == "b"
    assert _loc(None, {}, "c", existing)[0] == "c"
    assert _loc(None, {}, None, existing)[0] == "C:/default/tesseract.exe"


def test_locate_skips_nonexistent_candidates():
    assert _loc("missing", {"TESSERACT_CMD": "b"}, None, existing=("b",))[0] == "b"


def test_locate_not_found_lists_what_was_searched():
    path, searched = _loc("missing", {"TESSERACT_CMD": "nope"})
    assert path is None
    for fragment in ("configuration", "TESSERACT_CMD", "PATH", "default install path", "does not exist"):
        assert fragment in searched


# ---------------------------------------------------------------- TesseractOcr with a fake backend
class FakeBackend:
    def __init__(self, data=None, languages=("eng",), exc=None, init_exc=None):
        self.data = data if data is not None else {"text": ["Hello", ""], "conf": [90, -1]}
        self._languages, self.exc, self.init_exc = list(languages), exc, init_exc
        self.cmd = None
        self.calls = []

    def set_cmd(self, path):
        self.cmd = path

    def version(self):
        if self.init_exc:
            raise self.init_exc
        return "5.4.0"

    def languages(self):
        return self._languages

    def image_to_data(self, image, language, config):
        self.calls.append((image.shape, language, config))
        if self.exc:
            raise self.exc
        return self.data


def _tess(backend, exists=True, **kw):
    return TesseractOcr("fake.exe", "eng", 2.0, environ={}, which=lambda _n: None,
                        exists=lambda _p: exists, backend=backend, **kw)


def test_tesseract_success_joins_words_and_averages_confidence():
    backend = FakeBackend({"text": ["Hello", "World", "", "x"], "conf": [90, "80", 95, -1]})
    reading = _tess(backend).read(IMG, OcrHint(TextKind.PRINTED))
    assert reading.text == "Hello World"
    assert reading.confidence == pytest.approx(0.85)
    assert reading.engine == OcrEngineInfo(name="tesseract", version="5.4.0")
    assert backend.cmd == "fake.exe"


def test_tesseract_hint_controls_psm_whitelist_and_upscale():
    backend = FakeBackend()
    tess = _tess(backend)
    tess.read(IMG, OcrHint(TextKind.HANDWRITTEN, single_line=False, digits_only=True))
    shape, language, config = backend.calls[0]
    assert "--psm 6" in config and "tessedit_char_whitelist=0123456789" in config
    assert shape == (30 * 2 + 40, 120 * 2 + 40)  # upscaled then padded by 20 px each side
    tess.read(IMG, OcrHint(TextKind.PRINTED))
    assert "--psm 7" in backend.calls[1][2] and "whitelist" not in backend.calls[1][2]


def test_tesseract_no_words_gives_empty_text_and_none_confidence():
    reading = _tess(FakeBackend({"text": [""], "conf": [-1]})).read(IMG, OcrHint(TextKind.PRINTED))
    assert reading.text == "" and reading.confidence is None


def test_tesseract_missing_executable_is_unavailable():
    tess = _tess(FakeBackend(), exists=False)
    assert tess.is_available() is False
    with pytest.raises(OcrUnavailable, match="not found"):
        tess.read(IMG, OcrHint(TextKind.PRINTED))


def test_tesseract_missing_language_is_unavailable():
    with pytest.raises(OcrUnavailable, match="language data"):
        _tess(FakeBackend(languages=("osd",))).read(IMG, OcrHint(TextKind.PRINTED))


def test_tesseract_broken_init_is_unavailable_not_a_crash():
    with pytest.raises(OcrUnavailable, match="could not be initialised"):
        _tess(FakeBackend(init_exc=RuntimeError("boom"))).read(IMG, OcrHint(TextKind.PRINTED))


def test_tesseract_unavailable_is_remembered():
    tess = _tess(FakeBackend(init_exc=RuntimeError("boom")))
    assert tess.is_available() is False
    assert tess.is_available() is False


def test_tesseract_engine_failure_is_ocr_failure():
    with pytest.raises(OcrFailure):
        _tess(FakeBackend(exc=ValueError("bad"))).read(IMG, OcrHint(TextKind.PRINTED))
    with pytest.raises(OcrFailure):
        _tess(FakeBackend()).read(np.zeros((0, 0), np.uint8), OcrHint(TextKind.PRINTED))


def test_tesseract_passes_through_backend_unavailable_and_failure():
    with pytest.raises(OcrUnavailable):
        _tess(FakeBackend(exc=OcrUnavailable("gone"))).read(IMG, OcrHint(TextKind.PRINTED))
    with pytest.raises(OcrFailure):
        _tess(FakeBackend(exc=OcrFailure("bad"))).read(IMG, OcrHint(TextKind.PRINTED))


def test_tesseract_accepts_colour_images():
    reading = _tess(FakeBackend()).read(np.dstack([IMG] * 3), OcrHint(TextKind.PRINTED))
    assert reading.text == "Hello"


# ---------------------------------------------------------------- field policy
def _read(provider, kind, **kw):
    return read_text_field(provider, IMG, name="f", kind=kind, **kw)


def test_no_ink_is_empty_and_never_calls_the_engine():
    field = read_text_field(FakeProvider(exc=AssertionError("must not be called")), IMG, name="f",
                            kind=TextKind.HANDWRITTEN, has_ink=False)
    assert field.status is FieldStatus.EMPTY_OR_UNREADABLE and field.value is None


def test_missing_image_is_empty():
    assert read_text_field(FakeProvider("x"), None, name="f", kind=TextKind.PRINTED).status is FieldStatus.EMPTY_OR_UNREADABLE


@pytest.mark.parametrize("exc", [OcrUnavailable("no engine"), OcrFailure("bad crop"), RuntimeError("weird")])
def test_engine_problems_give_ocr_unavailable_with_null_value(exc):
    field = _read(FakeProvider(exc=exc), TextKind.HANDWRITTEN)
    assert field.status is FieldStatus.OCR_UNAVAILABLE and field.value is None
    assert field.review_reason


def test_null_provider_gives_ocr_unavailable():
    field = _read(NullOcr(), TextKind.PRINTED)
    assert field.status is FieldStatus.OCR_UNAVAILABLE and field.value is None


def test_handwritten_is_always_needs_review_even_at_full_confidence():
    field = _read(FakeProvider("1234", 0.99), TextKind.HANDWRITTEN)
    assert field.status is FieldStatus.NEEDS_REVIEW
    assert field.value == "1234" and field.confidence == 0.99
    assert field.engine == ENGINE and HANDWRITING_REASON in field.review_reason


def test_printed_extracted_at_or_above_threshold_and_review_below():
    assert _read(FakeProvider("Site A", 0.80), TextKind.PRINTED).status is FieldStatus.EXTRACTED
    low = _read(FakeProvider("Site A", 0.79), TextKind.PRINTED)
    assert low.status is FieldStatus.NEEDS_REVIEW and low.value == "Site A" and "threshold" in low.review_reason


def test_missing_confidence_becomes_zero_with_a_note():
    field = _read(FakeProvider("Site A", None), TextKind.PRINTED)
    assert field.status is FieldStatus.NEEDS_REVIEW and field.confidence == 0.0
    assert "no confidence" in field.review_reason
    hand = _read(FakeProvider("7", None), TextKind.HANDWRITTEN)
    assert hand.status is FieldStatus.NEEDS_REVIEW and "no confidence" in hand.review_reason


def test_ink_but_empty_text_needs_review():
    field = _read(FakeProvider("   ", 0.9), TextKind.PRINTED)
    assert field.status is FieldStatus.NEEDS_REVIEW and field.value is None


# ---------------------------------------------------------------- VLM mock
def test_vlm_unavailable_without_client():
    with pytest.raises(OcrUnavailable):
        VlmOcr().read(IMG, OcrHint(TextKind.HANDWRITTEN))


def test_vlm_request_is_serialisable_and_has_no_credentials():
    request = VlmOcr(model="m").build_request(IMG, OcrHint(TextKind.HANDWRITTEN, digits_only=True), "srin")
    text = json.dumps(request)
    assert request["temperature"] == 0 and request["digits_only"] is True and request["field"] == "srin"
    for forbidden in ("api_key", "authorization", "secret", "password", "bearer"):
        assert forbidden not in text.lower()
    assert not {"api_key", "key", "token", "authorization", "headers"} & set(request)  # max_tokens is not a credential


def test_vlm_response_is_mapped_but_handwriting_still_needs_review():
    vlm = VlmOcr(client=lambda req: {"text": "4821", "legible": True, "confidence": 0.99}, model="m")
    reading = vlm.read(IMG, OcrHint(TextKind.HANDWRITTEN))
    assert reading.text == "4821" and reading.confidence == 0.99 and reading.engine.name == "vlm"
    assert _read(vlm, TextKind.HANDWRITTEN).status is FieldStatus.NEEDS_REVIEW


def test_vlm_illegible_gives_empty_reading():
    assert VlmOcr(client=lambda r: {"text": None, "legible": False}).read(IMG, OcrHint(TextKind.HANDWRITTEN)).text == ""


@pytest.mark.parametrize("response", [None, "text", {"text": "x"}, {"text": "x", "legible": True, "confidence": 7}])
def test_vlm_invalid_response_is_ocr_failure(response):
    with pytest.raises(OcrFailure):
        VlmOcr(client=lambda r: response).read(IMG, OcrHint(TextKind.HANDWRITTEN))


def test_vlm_client_exception_is_ocr_failure():
    def boom(_):
        raise ConnectionError("down")

    with pytest.raises(OcrFailure):
        VlmOcr(client=boom).read(IMG, OcrHint(TextKind.HANDWRITTEN))


# ---------------------------------------------------------------- make_provider
def test_make_provider_none_gives_null():
    assert isinstance(make_provider("none", OcrConfig()), NullOcr)


def test_make_provider_tesseract_is_returned_even_if_unavailable():
    provider = make_provider("tesseract", OcrConfig(tesseract_cmd="does-not-exist.exe"))
    assert isinstance(provider, TesseractOcr)


def test_make_provider_auto_falls_back_to_null_with_reason(monkeypatch):
    monkeypatch.setattr(TesseractOcr, "is_available", lambda self: False)

    def fail(self):
        raise OcrUnavailable("nothing installed")

    monkeypatch.setattr(TesseractOcr, "_ensure", fail)
    provider = make_provider("auto", OcrConfig())
    assert isinstance(provider, NullOcr) and "nothing installed" in provider.reason


# ---------------------------------------------------------------- real engine smoke test
def _real_tesseract():
    path, _ = locate_tesseract()
    return path


@pytest.mark.skipif(_real_tesseract() is None, reason="Tesseract is not installed")
def test_real_tesseract_reads_clean_printed_text():
    import cv2

    canvas = np.full((80, 360), 255, np.uint8)
    cv2.putText(canvas, "SITE 42", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.6, 0, 3, cv2.LINE_AA)
    tess = TesseractOcr()
    assert tess.is_available()
    reading = tess.read(canvas, OcrHint(TextKind.PRINTED))
    assert "42" in reading.text and reading.engine.name == "tesseract" and reading.engine.version
