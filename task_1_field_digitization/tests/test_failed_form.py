"""Failed-test-form detection: text layer first, OCR as a fallback."""

from __future__ import annotations

import numpy as np
import pymupdf
import pytest

from field_digitization.failed_form import from_ocr, from_text_layer, phrase_match
from field_digitization.ingest import TextLayer, TextLine, TextWord, iter_pages, open_pdf
from field_digitization.ocr import NullOcr, OcrReading
from field_digitization.schema import FailedFormEvidenceSource, OcrEngineInfo


def _layer(*words, direction=(1.0, 0.0)):
    return TextLayer(words=tuple(TextWord(w, (0, 0, 1, 1)) for w in words),
                     lines=(TextLine(" ".join(words), direction),))


def test_text_layer_phrase_at_45_degrees():
    evidence = from_text_layer(_layer("FAILED", "TEST", "FORM", direction=(0.7071, -0.7071)))
    assert evidence.source is FailedFormEvidenceSource.TEXT_LAYER
    assert evidence.matched_text == "FAILED TEST FORM" and evidence.direction_degrees == pytest.approx(45, abs=0.1)
    assert len(evidence.words) == 3


def test_text_layer_is_case_and_punctuation_insensitive():
    assert from_text_layer(_layer("Failed", "test", "form.")) is not None


@pytest.mark.parametrize("words", [(), ("FAILED",), ("TEST", "FORM"), ("This", "test", "form", "has", "FAILED"),
                                   ("FAILED", "the", "TEST", "FORM")])
def test_text_layer_without_the_phrase_is_not_a_failed_form(words):
    assert from_text_layer(_layer(*words) if words else TextLayer()) is None
    assert from_text_layer(None) is None


@pytest.mark.parametrize("tokens,expected", [
    ("FAILED TEST FORM".split(), "FAILED TEST FORM"),
    ("ED TEST FORM".split(), "TEST FORM"),         # what OCR read on the sample's page 4 at -45 degrees
    ("FAILD TEST FORN".split(), "FAILD TEST FORN"),
    ("the test was fine".split(), None),
    ("FAILED and then FORM".split(), None),
])
def test_ocr_phrase_matching(tokens, expected):
    assert phrase_match(tokens) == expected


class _FakeOcr:
    def __init__(self, text_at):
        self.text_at, self.calls = text_at, 0

    def engine_info(self):
        return OcrEngineInfo(name="fake")

    def read(self, image, hint):
        self.calls += 1
        return OcrReading(self.text_at.get(self.calls, ""), 0.5, self.engine_info())


def test_ocr_fallback_tries_rotations_and_records_the_engine():
    fake = _FakeOcr({3: "ED TEST FORM"})  # third angle tried
    evidence = from_ocr(np.full((400, 300), 255, np.uint8), fake)
    assert evidence.source is FailedFormEvidenceSource.OCR and evidence.ocr_engine.name == "fake"
    assert fake.calls == 3


def test_ocr_fallback_finds_nothing_on_an_ordinary_page_or_without_an_engine():
    assert from_ocr(np.full((400, 300), 255, np.uint8), _FakeOcr({})) is None
    assert from_ocr(np.full((400, 300), 255, np.uint8), NullOcr()) is None


def test_phrase_in_a_real_pdf_text_layer(tmp_path):
    path = tmp_path / "failed.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((300, 600), "FAILED TEST FORM", fontsize=40, rotate=90)
    doc.save(path)
    doc.close()
    with open_pdf(path) as pdf:
        ingested = next(iter_pages(pdf))
    evidence = from_text_layer(ingested.text_layer)
    assert evidence is not None and evidence.direction_degrees == pytest.approx(90, abs=0.5)
