"""Regression tests: damaged data must never be reported as a blank page."""

import numpy as np
import pytest

from field_digitization import page_gate
from field_digitization.config import IngestConfig
from field_digitization.ingest import IngestedPage, iter_pages, open_pdf
from field_digitization.page_gate import gate_page
from pydantic import ValidationError

from field_digitization.schema import DocumentResult, PageResult, PageStatus

import synthetic

CONFIG = IngestConfig(render_dpi=100)


def _build_pdf_bytes(tmp_path, images):
    return synthetic.write_pdf(images, tmp_path / "source.pdf").read_bytes()


def _gate_all(tmp_path, data):
    path = tmp_path / "input.pdf"
    path.write_bytes(data)
    doc = open_pdf(path)
    try:
        return [gate_page(page) for page in iter_pages(doc, CONFIG)]
    finally:
        doc.close()


@pytest.fixture
def three_form_pages():
    return [synthetic.make_small_form_page(seed=i) for i in (1, 2, 3)]


def test_intact_pdf_pages_proceed(tmp_path, three_form_pages):
    decisions = _gate_all(tmp_path, _build_pdf_bytes(tmp_path, three_form_pages))
    assert [d.proceed for d in decisions] == [True, True, True]
    assert all(d.review_reasons == () for d in decisions)


def test_genuinely_blank_scan_is_blank_without_review(tmp_path):
    data = _build_pdf_bytes(
        tmp_path, [synthetic.make_small_form_page(), synthetic.make_small_blank_page()]
    )
    decisions = _gate_all(tmp_path, data)
    assert decisions[0].proceed
    blank = decisions[1].result
    assert blank.status is PageStatus.BLANK
    assert blank.review_required is False


def test_truncated_pdf_loses_pages_as_corrupted_not_blank(tmp_path, three_form_pages):
    data = synthetic.truncate_bytes(_build_pdf_bytes(tmp_path, three_form_pages), 0.7)
    decisions = _gate_all(tmp_path, data)
    # The first two pages survive; the third lost its image and must not look "blank".
    assert decisions[0].proceed and decisions[1].proceed
    lost = decisions[2].result
    assert lost.status is PageStatus.CORRUPTED
    assert lost.review_required
    assert "plain white" in lost.error


@pytest.mark.parametrize("keep", [0.9, 0.7, 0.5, 0.3, 0.1])
def test_no_page_of_a_truncated_pdf_is_silently_blank(tmp_path, three_form_pages, keep):
    data = synthetic.truncate_bytes(_build_pdf_bytes(tmp_path, three_form_pages), keep)
    decisions = _gate_all(tmp_path, data)
    for decision in decisions:
        if not decision.proceed:
            status = decision.result.status
            assert status in (PageStatus.CORRUPTED, PageStatus.NEEDS_REVIEW), status
            assert decision.result.review_required
    # Something was lost, so at least one page must be flagged.
    assert any(not d.proceed for d in decisions)


def test_damaged_image_stream_is_corrupted(tmp_path, three_form_pages):
    data = _build_pdf_bytes(tmp_path, three_form_pages)
    damaged = synthetic.zero_image_stream_span(data, 1, 0.35, 0.65, image_count=3)
    decisions = _gate_all(tmp_path, damaged)
    assert decisions[0].proceed and decisions[2].proceed
    result = decisions[1].result
    assert result.status is PageStatus.CORRUPTED
    assert "zero-filled band" in result.error
    assert result.review_required


def test_fully_zeroed_image_stream_is_corrupted(tmp_path, three_form_pages):
    data = _build_pdf_bytes(tmp_path, three_form_pages)
    damaged = synthetic.zero_image_stream_span(data, 0, 0.0, 1.0, image_count=3)
    decisions = _gate_all(tmp_path, damaged)
    assert decisions[0].result.status is PageStatus.CORRUPTED
    assert decisions[1].proceed and decisions[2].proceed


def test_warning_with_visible_content_proceeds_with_a_review_note():
    page = IngestedPage(
        page_number=2, image=synthetic.make_small_form_page(), warnings=("benign warning",)
    )
    decision = gate_page(page)
    assert decision.proceed
    assert "1 warning" in decision.review_reasons[0]


@pytest.mark.parametrize("keep", [0.5, 0.4])
def test_intact_pure_white_first_page_is_not_corrupted_by_page_tree_damage(tmp_path, keep):
    pages = [
        synthetic.make_pure_white_page(),
        synthetic.make_small_form_page(1),
        synthetic.make_small_form_page(2),
    ]
    data = synthetic.truncate_bytes(_build_pdf_bytes(tmp_path, pages), keep)
    first, second, third = (d.result for d in _gate_all(tmp_path, data))

    # Page 1: intact, pure white. It must not be called corrupted because the page tree is
    # damaged. It is document-level uncertainty, so it goes to review, and it is not "blank".
    assert first.status is PageStatus.NEEDS_REVIEW
    assert first.status is not PageStatus.CORRUPTED
    assert first.error is None
    assert first.review_required
    assert any("page-tree" in reason for reason in first.review_reasons)

    # Page-specific corruption is still reported as corruption.
    assert second.status is PageStatus.CORRUPTED
    assert "plain white" in second.error
    assert third.status in (PageStatus.NEEDS_REVIEW, PageStatus.CORRUPTED)
    assert third.review_required


def test_page_tree_warnings_alone_never_produce_a_corrupted_verdict():
    white = synthetic.make_pure_white_page()
    page = IngestedPage(
        page_number=1, image=white, load_warnings=("format error: non-page object in page tree",)
    )
    result = gate_page(page).result
    assert result.status is PageStatus.NEEDS_REVIEW
    assert result.error is None
    assert "cannot be attributed to this page" in result.review_reasons[0]


def test_render_warnings_on_a_white_page_that_has_damage_evidence_stay_corrupted():
    # The page-specific rule is unchanged: the evidence comes from ingest as `damage`.
    page = IngestedPage(
        page_number=2,
        image=synthetic.make_pure_white_page(),
        warnings=("format error: object is not a stream",),
        load_warnings=("format error: non-page object in page tree",),
        damage="1 MuPDF warning(s) while rendering, and the page references content but "
               "rendered as plain white",
    )
    result = gate_page(page).result
    assert result.status is PageStatus.CORRUPTED
    assert "plain white" in result.error


def test_page_tree_warning_on_a_readable_page_proceeds_with_a_review_note():
    page = IngestedPage(
        page_number=1,
        image=synthetic.make_small_form_page(),
        load_warnings=("format error: non-page object in page tree",),
    )
    decision = gate_page(page)
    assert decision.proceed
    assert any("page-tree" in note for note in decision.review_reasons)


def test_blank_looking_page_with_warnings_needs_review_not_blank():
    page = IngestedPage(
        page_number=1, image=synthetic.make_small_blank_page(), warnings=("some warning",)
    )
    result = gate_page(page).result
    assert result.status is PageStatus.NEEDS_REVIEW
    assert result.review_required


def test_render_error_becomes_corrupted_page():
    result = gate_page(IngestedPage(page_number=4, error="RuntimeError: boom")).result
    assert result.status is PageStatus.CORRUPTED
    assert result.error == "RuntimeError: boom"
    assert result.review_required


def test_quality_failure_is_isolated_as_failed_page(monkeypatch):
    def explode(*args, **kwargs):
        raise RuntimeError("quality bug")

    monkeypatch.setattr(page_gate, "assess_quality", explode)
    result = gate_page(
        IngestedPage(page_number=1, image=synthetic.make_small_form_page())
    ).result
    assert result.status is PageStatus.FAILED
    assert "quality bug" in result.error
    assert result.review_required


def test_page_with_neither_image_nor_error_is_a_valid_failed_result():
    decision = gate_page(IngestedPage(page_number=3))  # used to raise a ValidationError
    result = decision.result
    assert not decision.proceed
    assert result.status is PageStatus.FAILED
    assert "neither an image nor an error" in result.error
    assert result.review_required
    assert any("neither an image nor an error" in reason for reason in result.review_reasons)
    assert PageResult.model_validate_json(result.model_dump_json()) == result


def test_page_with_neither_image_nor_error_keeps_any_damage_evidence():
    result = gate_page(IngestedPage(page_number=1, damage="zero-filled band of 99 rows")).result
    assert result.status is PageStatus.FAILED
    assert "zero-filled band of 99 rows" in result.error


@pytest.mark.parametrize("blank_error", ["", "   "])
def test_blank_error_text_on_an_unrendered_page_is_treated_as_missing(blank_error):
    result = gate_page(IngestedPage(page_number=1, error=blank_error)).result
    assert result.status is PageStatus.FAILED
    assert result.error.strip()


@pytest.mark.parametrize(
    "image_factory",
    [synthetic.make_pure_white_page, synthetic.make_small_blank_page, synthetic.make_small_form_page],
    ids=["pure-white", "noisy-blank", "form"],
)
def test_page_with_both_image_and_error_is_never_blank(image_factory):
    # D1: a blank-looking image that also carries an error used to be reported as BLANK.
    page = IngestedPage(page_number=1, image=image_factory(), error="RuntimeError: boom")
    decision = gate_page(page)
    result = decision.result
    assert not decision.proceed
    assert result.status is PageStatus.CORRUPTED
    assert "RuntimeError: boom" in result.error
    assert result.review_required
    assert any("image is present" in reason for reason in result.review_reasons)
    assert result.quality is None  # the contradictory image was not trusted or measured
    assert PageResult.model_validate_json(result.model_dump_json()) == result


def test_error_text_that_is_only_whitespace_does_not_trigger_the_error_path():
    # A blank error is "no error", so a normal blank page is still just blank.
    page = IngestedPage(page_number=1, image=synthetic.make_small_blank_page(), error="   ")
    assert gate_page(page).result.status is PageStatus.BLANK


def test_damage_evidence_is_kept_when_there_is_an_error_and_no_image():
    # D2: damage evidence used to be dropped on the "no image, with error" path.
    page = IngestedPage(page_number=2, error="RuntimeError: render failed", damage="zero-filled band")
    result = gate_page(page).result
    assert result.status is PageStatus.CORRUPTED
    assert "RuntimeError: render failed" in result.error  # the original error is kept
    assert "zero-filled band" in result.error  # and so is the damage evidence
    assert result.review_required
    assert PageResult.model_validate_json(result.model_dump_json()) == result


def test_damage_evidence_is_kept_when_there_is_an_error_and_an_image():
    page = IngestedPage(
        page_number=3, image=synthetic.make_pure_white_page(), error="ValueError: x",
        damage="plain white with warnings",
    )
    result = gate_page(page).result
    assert result.status is PageStatus.CORRUPTED
    assert "ValueError: x" in result.error and "plain white with warnings" in result.error


def test_error_without_damage_adds_no_damage_text():
    result = gate_page(IngestedPage(page_number=1, error="RuntimeError: boom")).result
    assert result.error == "RuntimeError: boom"  # unchanged from before this fix


def test_damage_evidence_survives_a_quality_assessment_exception(monkeypatch, caplog):
    def explode(*args, **kwargs):
        raise ValueError("odd image layout")

    monkeypatch.setattr(page_gate, "assess_quality", explode)
    damage = "image contains a zero-filled band of 350 rows"
    with caplog.at_level("ERROR", logger="field_digitization.page_gate"):
        result = gate_page(
            IngestedPage(page_number=2, image=synthetic.make_small_form_page(), damage=damage)
        ).result

    assert result.status is PageStatus.CORRUPTED  # not FAILED: the damage is not masked
    assert damage in result.error  # the damage evidence is kept...
    assert "ValueError: odd image layout" in result.error  # ...and so is the original exception
    assert result.review_required
    assert any("damaged" in reason for reason in result.review_reasons)
    assert any("quality assessment also failed" in reason for reason in result.review_reasons)
    assert result.quality is None  # nothing was measured
    assert "odd image layout" in caplog.text  # the traceback is logged too
    assert PageResult.model_validate_json(result.model_dump_json()) == result


def test_damage_with_working_quality_assessment_does_not_mention_a_failure():
    result = gate_page(
        IngestedPage(
            page_number=1, image=synthetic.make_small_form_page(), damage="zero-filled band"
        )
    ).result
    assert result.status is PageStatus.CORRUPTED
    assert "also failed" not in result.error
    assert result.quality is not None  # measurements are still recorded for the reviewer


def test_unreadable_page_requires_review_and_carries_reasons():
    page = IngestedPage(page_number=1, image=synthetic.make_blurry_page())
    result = gate_page(page).result
    assert result.status is PageStatus.UNREADABLE
    assert result.review_required
    assert any("out of focus" in reason for reason in result.review_reasons)
    assert result.text_fields == [] and result.checkboxes == []


def test_mixed_document_is_a_complete_schema_valid_result(tmp_path, three_form_pages):
    # Page 1 survives, page 2 is a genuine blank scan, page 3 loses its image to truncation.
    pages = [three_form_pages[0], synthetic.make_small_blank_page(), three_form_pages[2]]
    data = synthetic.truncate_bytes(_build_pdf_bytes(tmp_path, pages), 0.7)
    decisions = _gate_all(tmp_path, data)
    assert [d.proceed for d in decisions] == [True, False, False]

    # The extraction stage does not exist yet. A page that passes the gate is represented by
    # a stand-in with no fields, so this test checks only that the gate's own terminal
    # results fit into a complete document. It is not an extraction result.
    page_results = [
        decision.result
        if decision.result is not None
        else PageResult(page_number=number, status=PageStatus.OK, quality=decision.quality)
        for number, decision in enumerate(decisions, start=1)
    ]
    document = DocumentResult(
        source_file="synthetic.pdf", page_count=len(decisions), pages=page_results
    )

    assert [p.status for p in document.pages] == [
        PageStatus.OK, PageStatus.BLANK, PageStatus.CORRUPTED
    ]
    assert DocumentResult.model_validate_json(document.model_dump_json()) == document


def test_terminal_results_alone_cannot_pose_as_a_complete_document(tmp_path, three_form_pages):
    # Only the pages that did NOT proceed have a result. Presenting them as the whole
    # three-page document must be rejected, not accepted as a partial result.
    data = synthetic.truncate_bytes(_build_pdf_bytes(tmp_path, three_form_pages), 0.7)
    decisions = _gate_all(tmp_path, data)
    terminal = [d.result for d in decisions if d.result is not None]
    assert 0 < len(terminal) < 3
    with pytest.raises(ValidationError, match="page_count"):
        DocumentResult(source_file="synthetic.pdf", page_count=3, pages=terminal)
