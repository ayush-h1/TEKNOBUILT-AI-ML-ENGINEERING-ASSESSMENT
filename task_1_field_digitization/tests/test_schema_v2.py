"""Schema 2.0 additions: failed_test_form, handwriting policy, choice groups, form, alignment."""

import json

import pytest
from pydantic import ValidationError

from field_digitization.schema import (
    AlignmentInfo,
    AlignmentMethod,
    CheckboxField,
    CheckboxState,
    ChecklistRow,
    ChoiceGroup,
    ChoiceStatus,
    DocumentMetadata,
    DocumentResult,
    ExpectationAnswer,
    FailedFormEvidence,
    FailedFormEvidenceSource,
    FieldStatus,
    FormHeader,
    ObservationForm,
    OcrEngineInfo,
    PageResult,
    PageStatus,
    TextField,
    TextKind,
    WordBox,
)

ENGINE = OcrEngineInfo(name="tesseract", version="5.4.0")


# ---------------------------------------------------------------- builders ---------------------------------------
def cell(name, state, **kw):
    if state in (None, CheckboxState.UNCERTAIN):
        kw.setdefault("review_required", True)
        kw.setdefault("review_reason", "ambiguous mark")
    return CheckboxField(name=name, state=state, **kw)


def group(options, checked=None, required=True, status=None, **kw):
    cells = [cell(o, CheckboxState.CHECKED if o == checked else CheckboxState.UNCHECKED) for o in options]
    if status is None:
        status = ChoiceStatus.ANSWERED if checked else ChoiceStatus.UNANSWERED
    extra = dict(kw)
    if status is ChoiceStatus.ANSWERED:
        extra.setdefault("selected", checked)
    elif status is ChoiceStatus.UNANSWERED and required:
        extra.setdefault("review_required", True)
        extra.setdefault("review_reasons", ["no option is marked"])
    return ChoiceGroup(key="g", options=cells, required=required, status=status, **extra)


def printed(value="Question text?"):
    return TextField(name="q", value=value, status=FieldStatus.EXTRACTED, confidence=0.95,
                     text_kind=TextKind.PRINTED, engine=ENGINE)


def handwritten(name="observer", value="J Smith"):
    return TextField(name=name, value=value, status=FieldStatus.NEEDS_REVIEW, confidence=0.9,
                     review_reason="handwritten text is a candidate", text_kind=TextKind.HANDWRITTEN, engine=ENGINE)


def make_form(rows=3, checked="I", handwritten_fields=True):
    hand = handwritten if handwritten_fields else (lambda name="x", value="x": printed())
    header = FormHeader(
        observation_date=hand("observation_date"), observer=hand("observer"), company_observed=hand("company_observed"),
        timesheet_number=hand("timesheet_number"), employee_number=hand("employee_number"),
        location=group(["1000", "2000"], "1000"), shift=group(["Day", "Night"], "Day"))
    return ObservationForm(
        section_title=printed("Some Title"), header=header,
        checklist=[ChecklistRow(row_number=i, question=printed(), answer=group(["S", "R", "I", "N"], checked))
                   for i in range(1, rows + 1)],
        expectation_answers=[ExpectationAnswer(key=f"q{i}", question=printed(), answer=group(["Yes", "No", "N/A"], "No"))
                             for i in range(1, 5)],
        comments=hand("comments", "fine"))


def accepted_alignment():
    return AlignmentInfo(accepted=True, method=AlignmentMethod.SIFT_AFFINE, matrix=[[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]],
                         verification_chamfer_mean_px=0.2, verification_within_2px_fraction=0.99, inliers=40, matches=100)


# ---------------------------------------------------------------- text provenance --------------------------------
def test_handwritten_text_can_never_be_extracted():
    for confidence in (0.5, 0.99, 1.0):
        with pytest.raises(ValidationError, match="handwritten"):
            TextField(name="d", value="2024-10-02", status=FieldStatus.EXTRACTED, confidence=confidence,
                      text_kind=TextKind.HANDWRITTEN, engine=ENGINE)


def test_handwritten_candidate_keeps_value_confidence_engine_and_reason():
    f = handwritten()
    assert (f.value, f.confidence, f.engine.name, f.status) == ("J Smith", 0.9, "tesseract", FieldStatus.NEEDS_REVIEW)
    assert "candidate" in f.review_reason


def test_printed_text_may_be_extracted():
    assert printed().status is FieldStatus.EXTRACTED


def test_a_value_with_a_text_kind_needs_an_engine():
    with pytest.raises(ValidationError, match="engine"):
        TextField(name="d", value="x", status=FieldStatus.NEEDS_REVIEW, confidence=0.3, review_reason="r",
                  text_kind=TextKind.HANDWRITTEN)


def test_no_value_text_fields_need_no_engine():
    f = TextField(name="d", status=FieldStatus.OCR_UNAVAILABLE, text_kind=TextKind.HANDWRITTEN, review_reason="no OCR engine")
    assert f.value is None and f.engine is None


# ---------------------------------------------------------------- choice groups ----------------------------------
def test_answered_group_has_exactly_one_checked_option():
    g = group(["S", "R", "I", "N"], "R")
    assert g.selected == "R" and g.status is ChoiceStatus.ANSWERED and not g.review_required


@pytest.mark.parametrize("bad_selected", [None, "S", "Z"])
def test_answered_group_selected_must_be_the_checked_option(bad_selected):
    with pytest.raises(ValidationError):
        group(["S", "R"], "R", selected=bad_selected)


def test_answered_group_cannot_contain_an_uncertain_option():
    cells = [cell("S", CheckboxState.CHECKED), cell("R", CheckboxState.UNCERTAIN)]
    with pytest.raises(ValidationError):
        ChoiceGroup(key="g", options=cells, status=ChoiceStatus.ANSWERED, selected="S")


def test_answered_group_cannot_require_review():
    with pytest.raises(ValidationError, match="does not require review"):
        group(["S", "R"], "S", review_required=True, review_reasons=["x"])


def test_conflict_needs_two_checked_options_no_selection_and_review():
    cells = [cell("S", CheckboxState.CHECKED), cell("R", CheckboxState.CHECKED), cell("I", CheckboxState.UNCHECKED)]
    g = ChoiceGroup(key="g", options=cells, status=ChoiceStatus.CONFLICT, review_required=True, review_reasons=["two marks"])
    assert g.selected is None
    with pytest.raises(ValidationError):
        ChoiceGroup(key="g", options=cells, status=ChoiceStatus.CONFLICT, selected="S", review_required=True, review_reasons=["x"])
    with pytest.raises(ValidationError, match="flagged"):
        ChoiceGroup(key="g", options=cells, status=ChoiceStatus.CONFLICT)
    with pytest.raises(ValidationError, match="two or more"):
        ChoiceGroup(key="g", options=cells[:1] + cells[2:], status=ChoiceStatus.CONFLICT, review_required=True, review_reasons=["x"])


def test_uncertain_group_never_invents_a_selection():
    cells = [cell("S", CheckboxState.UNCERTAIN), cell("R", CheckboxState.UNCHECKED)]
    g = ChoiceGroup(key="g", options=cells, status=ChoiceStatus.UNCERTAIN, review_required=True, review_reasons=["faint mark"])
    assert g.selected is None
    with pytest.raises(ValidationError, match="undecided"):
        ChoiceGroup(key="g", options=[cell("S", CheckboxState.UNCHECKED), cell("R", CheckboxState.UNCHECKED)],
                    status=ChoiceStatus.UNCERTAIN, review_required=True, review_reasons=["x"])


def test_unanswered_required_group_must_be_flagged():
    g = group(["S", "R"], None, required=True)
    assert g.review_required and g.selected is None
    with pytest.raises(ValidationError, match="flagged"):
        ChoiceGroup(key="g", options=g.options, status=ChoiceStatus.UNANSWERED, required=True)


def test_unanswered_optional_group_is_not_an_error():
    g = group(["S", "R"], None, required=False)
    assert not g.review_required and g.review_reasons == []
    with pytest.raises(ValidationError, match="optional"):
        group(["S", "R"], None, required=False, review_required=True, review_reasons=["x"])


def test_unanswered_group_cannot_contain_marks():
    with pytest.raises(ValidationError):
        ChoiceGroup(key="g", options=[cell("S", CheckboxState.CHECKED), cell("R", CheckboxState.UNCHECKED)],
                    status=ChoiceStatus.UNANSWERED, review_required=True, review_reasons=["x"])


def test_group_needs_two_or_more_unique_options():
    with pytest.raises(ValidationError):
        ChoiceGroup(key="g", options=[cell("S", CheckboxState.UNCHECKED)], status=ChoiceStatus.UNANSWERED, review_required=True, review_reasons=["x"])
    with pytest.raises(ValidationError, match="unique"):
        ChoiceGroup(key="g", options=[cell("S", CheckboxState.UNCHECKED), cell("S", CheckboxState.UNCHECKED)],
                    status=ChoiceStatus.UNANSWERED, review_required=True, review_reasons=["x"])


def test_review_flag_and_reasons_agree_in_a_group():
    with pytest.raises(ValidationError):
        group(["S", "R"], None, review_required=True, review_reasons=[])
    with pytest.raises(ValidationError):
        group(["S", "R"], "S", review_reasons=["reason without flag"])


# ---------------------------------------------------------------- form -------------------------------------------
@pytest.mark.parametrize("rows", [1, 3, 8, 9, 15])
def test_form_accepts_any_number_of_checklist_rows(rows):
    assert len(make_form(rows).checklist) == rows


def test_form_needs_at_least_one_row_and_rows_numbered_in_order():
    with pytest.raises(ValidationError):
        ObservationForm(**{**make_form(2).model_dump(), "checklist": []})
    bad = make_form(3).model_dump()
    bad["checklist"][1]["row_number"] = 5
    with pytest.raises(ValidationError, match="numbered"):
        ObservationForm.model_validate(bad)


def test_form_with_handwritten_candidates_needs_review():
    assert make_form().needs_review()


def test_form_without_review_items_does_not_need_review():
    assert not make_form(handwritten_fields=False).needs_review()


def test_form_review_follows_a_choice_group_flag():
    form = make_form(handwritten_fields=False)
    form.checklist[0].answer = group(["S", "R", "I", "N"], None)
    assert form.needs_review()


# ---------------------------------------------------------------- page result ------------------------------------
def test_page_with_a_review_needing_form_must_be_flagged_and_not_ok():
    form = make_form()
    with pytest.raises(ValidationError, match="review_required"):
        PageResult(page_number=1, status=PageStatus.NEEDS_REVIEW, form=form)
    with pytest.raises(ValidationError):
        PageResult(page_number=1, status=PageStatus.OK, form=form, review_required=True, review_reasons=["handwriting"])
    page = PageResult(page_number=1, status=PageStatus.NEEDS_REVIEW, form=form, review_required=True,
                      review_reasons=["handwritten fields are OCR candidates"], alignment=accepted_alignment())
    assert page.form.header.observer.status is FieldStatus.NEEDS_REVIEW


def test_a_clean_form_page_may_be_ok():
    page = PageResult(page_number=1, status=PageStatus.OK, form=make_form(handwritten_fields=False))
    assert not page.review_required


@pytest.mark.parametrize("status", [PageStatus.BLANK, PageStatus.UNREADABLE, PageStatus.CORRUPTED, PageStatus.FAILED])
def test_non_form_statuses_cannot_hold_a_form(status):
    with pytest.raises(ValidationError):
        PageResult(page_number=1, status=status, form=make_form(), error="e", review_required=True, review_reasons=["x"])


def test_form_data_cannot_exist_when_alignment_was_rejected():
    rejected = AlignmentInfo(accepted=False, reasons=["too few inliers"])
    with pytest.raises(ValidationError, match="rejected"):
        PageResult(page_number=1, status=PageStatus.NEEDS_REVIEW, form=make_form(), alignment=rejected,
                   review_required=True, review_reasons=["x"])


def test_a_rejected_alignment_page_is_needs_review_without_a_form():
    rejected = AlignmentInfo(accepted=False, reasons=["too few inliers"], matches=5, inliers=2, opencv_version="5.1.0")
    page = PageResult(page_number=3, status=PageStatus.NEEDS_REVIEW, alignment=rejected, review_required=True,
                      review_reasons=["template alignment was rejected"])
    assert page.form is None and page.alignment.accepted is False


# ---------------------------------------------------------------- failed test form -------------------------------
def text_layer_evidence():
    return FailedFormEvidence(
        source=FailedFormEvidenceSource.TEXT_LAYER, matched_text="FAILED TEST FORM", direction_degrees=45.0,
        words=[WordBox(text="FAILED", bbox=(1, 2, 3, 4)), WordBox(text="TEST", bbox=(1, 2, 3, 4)), WordBox(text="FORM", bbox=(1, 2, 3, 4))])


def test_failed_test_form_status_value_and_text_layer_evidence_need_no_review():
    assert PageStatus.FAILED_TEST_FORM.value == "failed_test_form"
    page = PageResult(page_number=4, status=PageStatus.FAILED_TEST_FORM, failed_form_evidence=text_layer_evidence())
    assert page.review_required is False and page.form is None


def test_failed_test_form_requires_evidence():
    with pytest.raises(ValidationError, match="evidence"):
        PageResult(page_number=4, status=PageStatus.FAILED_TEST_FORM)


def test_ocr_only_failed_form_requires_review():
    ev = FailedFormEvidence(source=FailedFormEvidenceSource.OCR, matched_text="TEST FORM", ocr_engine=ENGINE, ocr_confidence=0.6)
    with pytest.raises(ValidationError, match="OCR must require review"):
        PageResult(page_number=4, status=PageStatus.FAILED_TEST_FORM, failed_form_evidence=ev)
    page = PageResult(page_number=4, status=PageStatus.FAILED_TEST_FORM, failed_form_evidence=ev, review_required=True,
                      review_reasons=["found only by OCR"])
    assert page.review_required


def test_ocr_evidence_must_name_its_engine():
    with pytest.raises(ValidationError, match="engine"):
        FailedFormEvidence(source=FailedFormEvidenceSource.OCR, matched_text="TEST FORM")


def test_failed_form_cannot_hold_observation_data_or_fields():
    with pytest.raises(ValidationError):
        PageResult(page_number=4, status=PageStatus.FAILED_TEST_FORM, failed_form_evidence=text_layer_evidence(), form=make_form(),
                   review_required=True, review_reasons=["x"])
    with pytest.raises(ValidationError):
        PageResult(page_number=4, status=PageStatus.FAILED_TEST_FORM, failed_form_evidence=text_layer_evidence(),
                   text_fields=[TextField(name="date", status=FieldStatus.EMPTY_OR_UNREADABLE)])


def test_evidence_is_only_valid_on_a_failed_test_form_page():
    with pytest.raises(ValidationError, match="only valid"):
        PageResult(page_number=1, status=PageStatus.BLANK, failed_form_evidence=text_layer_evidence())


def test_failed_test_form_is_distinct_from_operational_statuses():
    others = {PageStatus.FAILED, PageStatus.CORRUPTED, PageStatus.UNREADABLE, PageStatus.BLANK}
    assert PageStatus.FAILED_TEST_FORM not in others


def test_failed_form_evidence_needs_matched_text():
    with pytest.raises(ValidationError):
        FailedFormEvidence(source=FailedFormEvidenceSource.TEXT_LAYER, matched_text="  ")


# ---------------------------------------------------------------- alignment info ---------------------------------
def test_accepted_alignment_needs_method_matrix_and_verification():
    with pytest.raises(ValidationError):
        AlignmentInfo(accepted=True)
    with pytest.raises(ValidationError, match="verification"):
        AlignmentInfo(accepted=True, method=AlignmentMethod.SIFT_AFFINE, matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert accepted_alignment().accepted


def test_rejected_alignment_must_say_why():
    with pytest.raises(ValidationError, match="why"):
        AlignmentInfo(accepted=False)


# ---------------------------------------------------------------- document ---------------------------------------
def test_document_with_form_page_and_failed_form_page_round_trips_and_is_deterministic():
    doc = DocumentResult(
        source_file="x.pdf", page_count=2,
        pages=[PageResult(page_number=1, status=PageStatus.NEEDS_REVIEW, form=make_form(), alignment=accepted_alignment(),
                          review_required=True, review_reasons=["handwritten fields are OCR candidates"]),
               PageResult(page_number=2, status=PageStatus.FAILED_TEST_FORM, failed_form_evidence=text_layer_evidence())],
        metadata=DocumentMetadata(schema_version="2.0", pipeline_version="0.2.0", source_file="x.pdf", source_sha256="ab" * 32,
                                  template_file="t.png", template_sha256="cd" * 32, render_dpi=200, engines={"opencv": "5.1.0"}))
    text = doc.model_dump_json()
    assert DocumentResult.model_validate_json(text) == doc
    assert doc.model_dump_json() == text
    payload = json.loads(text)
    assert not any("time" in k or "date_run" in k for k in payload["metadata"])  # no timestamps in the output


def test_metadata_needs_a_render_dpi_of_at_least_one():
    with pytest.raises(ValidationError):
        DocumentMetadata(schema_version="2.0", pipeline_version="x", source_file="a", source_sha256="b", render_dpi=0)
