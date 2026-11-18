"""End-to-end runs on the supplied sample and on awkward pages built from the template.

The real files are git-ignored; these tests are skipped when they are absent. Agreement with the
ground-truth fixture is checked for tick boxes only (pixel evidence). The fixture is the assistant's
transcription and has not been confirmed by an independent person, so this is a consistency check,
not an accuracy measurement."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from field_digitization import pipeline
from field_digitization.__main__ import main
from field_digitization.ocr import TesseractOcr
from field_digitization.schema import (
    ChoiceStatus, DocumentResult, FailedFormEvidenceSource, FieldStatus, PageStatus, TextKind,
)

import synthetic

DATA = Path(__file__).resolve().parents[1] / "data"
TEMPLATE = DATA / "template" / "template.png"
SAMPLE = DATA / "input" / "Sample_input.pdf"
FIXTURE = DATA / "ground_truth" / "sample_input_ground_truth.json"

needs_template = pytest.mark.skipif(not TEMPLATE.exists(), reason="real template not present (git-ignored)")
needs_sample = pytest.mark.skipif(not (TEMPLATE.exists() and SAMPLE.exists()), reason="real sample not present (git-ignored)")


@pytest.fixture(scope="module")
def ctx_none():
    return pipeline.build_context(TEMPLATE, ocr="none")


@pytest.fixture(scope="module")
def sample_none(ctx_none):
    return pipeline.process_pdf(SAMPLE, ctx_none)


# ---------------------------------------------------------------- the supplied sample, OCR off
@needs_sample
def test_sample_pages_are_classified(sample_none):
    statuses = [p.status for p in sample_none.pages]
    assert statuses == [PageStatus.NEEDS_REVIEW] * 3 + [PageStatus.FAILED_TEST_FORM]
    assert [len(p.form.checklist) for p in sample_none.pages[:3]] == [8, 8, 9]
    assert all(len(p.form.expectation_answers) == 4 for p in sample_none.pages[:3])


@needs_sample
def test_sample_failed_test_form_page(sample_none):
    page = sample_none.pages[3]
    assert page.form is None and page.alignment is None  # never aligned, nothing extracted
    assert page.failed_form_evidence.source is FailedFormEvidenceSource.TEXT_LAYER
    assert page.failed_form_evidence.direction_degrees == pytest.approx(45, abs=1)
    assert page.review_required is False


@needs_sample
def test_without_ocr_no_text_is_invented(sample_none):
    for page in sample_none.pages[:3]:
        for field in page.form.text_fields():
            assert field.value is None
            assert field.status in (FieldStatus.OCR_UNAVAILABLE, FieldStatus.EMPTY_OR_UNREADABLE)


@needs_sample
def test_sample_tick_boxes_agree_with_the_fixture(sample_none):
    if not FIXTURE.exists():
        pytest.skip("ground-truth fixture not present (git-ignored)")
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    for page, truth in zip(sample_none.pages[:3], fixture["pages"][:3]):
        form = page.form
        pairs = [(form.header.location, truth["location"]), (form.header.shift, truth["shift"])]
        pairs += [(r.answer, t["answer"]) for r, t in zip(form.checklist, truth["checklist"])]
        pairs += [(q.answer, t["answer"]) for q, t in zip(form.expectation_answers, truth["expectation_answers"])]
        for group, expected in pairs:
            assert group.status is ChoiceStatus.ANSWERED
            assert group.selected == expected["selected"], (page.page_number, group.key)
            assert [o.state.value for o in group.options] == [o["state"] for o in expected["options"]]


@needs_sample
def test_output_is_valid_json_and_deterministic(sample_none, ctx_none, tmp_path):
    first = pipeline.write_json(sample_none, tmp_path / "a.json").read_bytes()
    second = pipeline.write_json(pipeline.process_pdf(SAMPLE, ctx_none), tmp_path / "b.json").read_bytes()
    assert first == second
    DocumentResult.model_validate_json(first)


@needs_sample
def test_summary_metrics(sample_none):
    summary = pipeline.summarize(sample_none)
    assert summary["processing_completed"] == 4
    assert summary["review_required"] == 3
    assert summary["fully_automatic_extraction"] == 0
    assert summary["failed_test_form_pages"] == [4]
    assert summary["checklist_rows_per_form_page"] == {1: 8, 2: 8, 3: 9}


# ---------------------------------------------------------------- the supplied sample, with Tesseract
@needs_sample
@pytest.mark.skipif(not TesseractOcr().is_available(), reason="Tesseract is not installed")
def test_sample_with_tesseract_keeps_handwriting_for_review(sample_none):
    result = pipeline.process_pdf(SAMPLE, pipeline.build_context(TEMPLATE, ocr="tesseract"))
    for page, plain in zip(result.pages[:3], sample_none.pages[:3]):
        hand = [f for f in page.form.text_fields() if f.text_kind is TextKind.HANDWRITTEN]
        assert len(hand) == 6
        for field in hand:
            assert field.status is FieldStatus.NEEDS_REVIEW and field.value and field.engine.name == "tesseract"
        assert page.form.section_title.value  # printed text was read
        # OCR never changes the tick-box results
        assert [g.model_dump() for g in page.form.choice_groups()] == [g.model_dump() for g in plain.form.choice_groups()]


# ---------------------------------------------------------------- awkward pages built from the template
def _template_page(marks=()):
    """The blank template as a 'filled' page, optionally with ticks drawn into given template cells."""
    page = cv2.imread(str(TEMPLATE), cv2.IMREAD_GRAYSCALE)
    for cell, kind in marks:
        x0, y0, x1, y1 = (int(round(v)) for v in (cell.x0, cell.y0, cell.x1, cell.y1))
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        if kind == "tick":
            cv2.line(page, (cx - 10, cy - 10), (cx + 10, cy + 10), 0, 3)
            cv2.line(page, (cx + 10, cy - 10), (cx - 10, cy + 10), 0, 3)
        else:  # a filled-in box
            page[y0 + 8:y1 - 8, x0 + 8:x1 - 8] = 0
    return page


@pytest.fixture(scope="module")
def awkward(ctx_none, tmp_path_factory):
    layout = ctx_none.layout
    slot1, slot2 = layout.checklist_slots[0], layout.checklist_slots[1]
    cols = layout.choice_columns_x
    from field_digitization.layout import Cell
    marks = [(Cell(cols[0], slot1.y0, cols[1], slot1.y1), "tick"), (Cell(cols[1], slot1.y0, cols[2], slot1.y1), "tick"),
             (Cell(cols[0], slot2.y0, cols[1], slot2.y1), "fill")]
    images = [synthetic.make_blank_page(), synthetic.make_dark_muddy_page(), synthetic.make_form_page(),
              _template_page(marks)]
    pdf = synthetic.write_pdf(images, tmp_path_factory.mktemp("awkward") / "awkward.pdf")
    return pdf, pipeline.process_pdf(pdf, ctx_none)


@needs_template
def test_blank_unreadable_and_foreign_pages_yield_no_form(awkward):
    _, result = awkward
    blank, muddy, foreign = result.pages[:3]
    assert blank.status is PageStatus.BLANK and blank.form is None
    assert muddy.status is PageStatus.UNREADABLE and muddy.form is None and muddy.review_required
    assert foreign.status is PageStatus.NEEDS_REVIEW and foreign.form is None
    assert foreign.alignment is not None and not foreign.alignment.accepted
    assert any("could not be aligned" in r for r in foreign.review_reasons)


@needs_template
def test_empty_form_conflicts_and_scribbles_are_flagged_not_guessed(awkward):
    page = awkward[1].pages[3]
    assert page.status is PageStatus.NEEDS_REVIEW and page.form is not None
    rows = page.form.checklist
    assert len(rows) == 15  # the template draws every slot
    assert rows[0].answer.status is ChoiceStatus.CONFLICT and rows[0].answer.selected is None
    assert rows[1].answer.status is ChoiceStatus.UNCERTAIN and rows[1].answer.selected is None
    assert all(r.answer.status is ChoiceStatus.UNANSWERED and r.answer.review_required for r in rows[2:])
    header = page.form.header
    assert header.location.status is ChoiceStatus.UNANSWERED and header.shift.status is ChoiceStatus.UNANSWERED
    assert header.observer.status is FieldStatus.EMPTY_OR_UNREADABLE and header.observer.value is None
    assert any("required header field 'observer' is empty" in r for r in page.review_reasons)
    assert page.form.comments.status is FieldStatus.EMPTY_OR_UNREADABLE


@needs_template
def test_an_error_on_one_page_does_not_stop_the_others(awkward, ctx_none, monkeypatch):
    pdf, _ = awkward

    def boom(*args, **kwargs):
        raise RuntimeError("simulated extraction bug")

    monkeypatch.setattr(pipeline, "extract_form", boom)
    result = pipeline.process_pdf(pdf, ctx_none)
    assert result.pages[3].status is PageStatus.FAILED and "simulated extraction bug" in result.pages[3].error
    assert [p.status for p in result.pages[:3]] == [PageStatus.BLANK, PageStatus.UNREADABLE, PageStatus.NEEDS_REVIEW]


# ---------------------------------------------------------------- command line
@needs_template
def test_cli_writes_output_and_report(awkward, tmp_path):
    pdf, _ = awkward
    out = tmp_path / "out" / "result.json"
    assert main(["--input", str(pdf), "--template", str(TEMPLATE), "--output", str(out), "--ocr", "none"]) == 0
    DocumentResult.model_validate_json(out.read_text(encoding="utf-8"))
    report = json.loads((out.parent / "processing_report.json").read_text(encoding="utf-8"))
    assert report["total_pages"] == 4


def test_cli_reports_document_level_errors(tmp_path):
    pdf = synthetic.write_pdf([synthetic.make_blank_page()], tmp_path / "one.pdf")
    assert main(["--input", str(pdf), "--template", str(tmp_path / "missing.png"),
                 "--output", str(tmp_path / "o.json"), "--ocr", "none"]) == 2
    bad_pdf = tmp_path / "bad.pdf"
    bad_pdf.write_bytes(b"not a pdf")
    if TEMPLATE.exists():
        assert main(["--input", str(bad_pdf), "--template", str(TEMPLATE),
                     "--output", str(tmp_path / "o.json"), "--ocr", "none"]) == 2
