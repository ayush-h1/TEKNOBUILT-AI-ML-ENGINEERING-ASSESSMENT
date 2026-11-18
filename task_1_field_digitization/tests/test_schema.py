import json

import pytest
from pydantic import ValidationError

from field_digitization.schema import (
    SCHEMA_VERSION,
    CheckboxField,
    CheckboxState,
    DocumentResult,
    FieldStatus,
    PageResult,
    PageStatus,
    TextField,
)


def test_extracted_field_requires_value():
    with pytest.raises(ValidationError):
        TextField(name="date", status=FieldStatus.EXTRACTED, value=None)
    with pytest.raises(ValidationError):
        TextField(name="date", status=FieldStatus.EXTRACTED, value="")


@pytest.mark.parametrize("status", [FieldStatus.EMPTY_OR_UNREADABLE, FieldStatus.OCR_UNAVAILABLE])
def test_non_extracted_statuses_cannot_carry_a_value(status):
    with pytest.raises(ValidationError):
        TextField(name="observer", status=status, value="invented name")
    assert TextField(name="observer", status=status).value is None


def test_needs_review_requires_reason():
    with pytest.raises(ValidationError):
        TextField(name="date", status=FieldStatus.NEEDS_REVIEW, value="2O24")
    ok = TextField(
        name="date",
        status=FieldStatus.NEEDS_REVIEW,
        value="2O24",
        confidence=0.4,
        review_reason="low confidence",
    )
    assert ok.value == "2O24"


def test_confidence_must_be_between_zero_and_one():
    with pytest.raises(ValidationError):
        TextField(name="date", status=FieldStatus.EXTRACTED, value="x", confidence=1.2)


def test_uncertain_checkbox_must_be_flagged():
    with pytest.raises(ValidationError):
        CheckboxField(name="ppe_worn", state=CheckboxState.UNCERTAIN)
    flagged = CheckboxField(
        name="ppe_worn", state=CheckboxState.UNCERTAIN, review_required=True, review_reason="ambiguous"
    )
    assert flagged.review_required


def test_failed_pages_must_not_contain_fields_and_must_explain():
    with pytest.raises(ValidationError):
        PageResult(page_number=1, status=PageStatus.CORRUPTED)  # no error message
    with pytest.raises(ValidationError):
        PageResult(
            page_number=1,
            status=PageStatus.BLANK,
            text_fields=[TextField(name="date", status=FieldStatus.EMPTY_OR_UNREADABLE)],
        )


def test_any_text_value_must_carry_a_confidence():
    for status, extra in (
        (FieldStatus.EXTRACTED, {}),
        (FieldStatus.NEEDS_REVIEW, {"review_reason": "low confidence"}),
    ):
        with pytest.raises(ValidationError, match="confidence"):
            TextField(name="observer", status=status, value="J. Smith", **extra)
    assert TextField(
        name="observer", status=FieldStatus.EXTRACTED, value="J. Smith", confidence=0.93
    ).confidence == 0.93


def test_needs_review_reason_cannot_be_blank():
    with pytest.raises(ValidationError):
        TextField(name="date", status=FieldStatus.NEEDS_REVIEW, review_reason="   ")


@pytest.mark.parametrize("state", [None, CheckboxState.UNCERTAIN])
def test_checkbox_without_definite_state_must_be_flagged_with_a_reason(state):
    with pytest.raises(ValidationError):
        CheckboxField(name="ppe_worn", state=state)  # not flagged
    with pytest.raises(ValidationError):
        CheckboxField(name="ppe_worn", state=state, review_required=True)  # no reason
    with pytest.raises(ValidationError):
        CheckboxField(name="ppe_worn", state=state, review_required=True, review_reason="  ")
    ok = CheckboxField(
        name="ppe_worn", state=state, review_required=True, review_reason="mark overlaps the box edge"
    )
    assert ok.state is state  # the uncertainty is kept, no checked/unchecked value is invented


def test_definite_checkbox_flagged_for_review_still_needs_a_reason():
    with pytest.raises(ValidationError):
        CheckboxField(name="area_clean", state=CheckboxState.CHECKED, review_required=True)
    assert CheckboxField(name="area_clean", state=CheckboxState.CHECKED).review_required is False


def _review_field():
    return TextField(
        name="date",
        status=FieldStatus.NEEDS_REVIEW,
        value="2O24",
        confidence=0.3,
        review_reason="low confidence",
    )


def _flagged_checkbox():
    return CheckboxField(
        name="ppe_worn", state=CheckboxState.UNCERTAIN, review_required=True, review_reason="ambiguous"
    )


@pytest.mark.parametrize("field_kwargs", [
    lambda: {"text_fields": [_review_field()]},
    lambda: {"checkboxes": [_flagged_checkbox()]},
])
def test_page_must_be_flagged_when_a_field_needs_review(field_kwargs):
    with pytest.raises(ValidationError, match="review_required"):
        PageResult(page_number=1, status=PageStatus.NEEDS_REVIEW, **field_kwargs())
    with pytest.raises(ValidationError):  # flagged, but 'ok' status contradicts it
        PageResult(
            page_number=1, status=PageStatus.OK, review_required=True,
            review_reasons=["field needs review"], **field_kwargs(),
        )
    page = PageResult(
        page_number=1, status=PageStatus.NEEDS_REVIEW, review_required=True,
        review_reasons=["field needs review"], **field_kwargs(),
    )
    assert page.review_required


def test_page_review_flag_and_reasons_must_agree():
    with pytest.raises(ValidationError):
        PageResult(page_number=1, status=PageStatus.NEEDS_REVIEW, review_required=True)  # no reasons
    with pytest.raises(ValidationError):
        PageResult(page_number=1, status=PageStatus.BLANK, review_reasons=["why"])  # reasons, no flag
    with pytest.raises(ValidationError):
        PageResult(page_number=1, status=PageStatus.NEEDS_REVIEW)  # status without the flag
    with pytest.raises(ValidationError):
        PageResult(page_number=1, status=PageStatus.OK, review_required=True, review_reasons=["x"])


def test_clean_page_and_flagged_blank_page_are_valid():
    clean = PageResult(
        page_number=1,
        status=PageStatus.OK,
        text_fields=[TextField(name="date", status=FieldStatus.EMPTY_OR_UNREADABLE)],
    )
    flagged_blank_like = PageResult(
        page_number=2, status=PageStatus.NEEDS_REVIEW, review_required=True,
        review_reasons=["looks blank but MuPDF reported warnings"],
    )
    assert not clean.review_required and flagged_blank_like.review_required


# --- review policy for fields whose text was never read ---------------------------------

def _unread_field(name="date"):
    return TextField(name=name, status=FieldStatus.OCR_UNAVAILABLE)


def test_ocr_unavailable_cannot_sit_on_a_page_reported_as_fully_successful():
    # Nothing was read, so a page with such a field must not be 'ok' and not unflagged.
    with pytest.raises(ValidationError, match="ocr_unavailable"):
        PageResult(page_number=1, status=PageStatus.OK, text_fields=[_unread_field()])
    with pytest.raises(ValidationError, match="ocr_unavailable"):
        PageResult(page_number=1, status=PageStatus.NEEDS_REVIEW, text_fields=[_unread_field()])
    with pytest.raises(ValidationError):  # flagged, but 'ok' contradicts the flag
        PageResult(
            page_number=1, status=PageStatus.OK, text_fields=[_unread_field()],
            review_required=True, review_reasons=["text was not read"],
        )


def test_ocr_unavailable_page_is_valid_when_flagged_for_review_with_a_reason():
    page = PageResult(
        page_number=1,
        status=PageStatus.NEEDS_REVIEW,
        text_fields=[_unread_field("date"), _unread_field("observer")],
        review_required=True,
        review_reasons=["no OCR provider was available; text fields were not read"],
    )
    assert page.review_required
    assert all(f.value is None for f in page.text_fields)  # nothing was invented


def test_one_unread_field_among_extracted_fields_still_forces_review():
    extracted = TextField(name="observer", status=FieldStatus.EXTRACTED, value="J. Smith", confidence=0.95)
    with pytest.raises(ValidationError):
        PageResult(page_number=1, status=PageStatus.OK, text_fields=[extracted, _unread_field()])


def test_genuinely_empty_field_does_not_force_review():
    # EMPTY_OR_UNREADABLE keeps its documented meaning and does not flag the page.
    page = PageResult(
        page_number=1,
        status=PageStatus.OK,
        text_fields=[TextField(name="date", status=FieldStatus.EMPTY_OR_UNREADABLE)],
    )
    assert page.review_required is False


# --- DocumentResult completeness ---------------------------------------------------------

def _page(number, status=PageStatus.BLANK):
    return PageResult(page_number=number, status=status)


def test_complete_document_is_accepted_in_any_page_order():
    doc = DocumentResult(source_file="a.pdf", page_count=3, pages=[_page(2), _page(1), _page(3)])
    assert sorted(p.page_number for p in doc.pages) == [1, 2, 3]


def test_empty_document_with_zero_pages_is_accepted():
    assert DocumentResult(source_file="a.pdf", page_count=0, pages=[]).pages == []


@pytest.mark.parametrize("count, numbers", [(3, [1, 2]), (1, [1, 2]), (2, [])])
def test_page_count_must_match_the_number_of_pages(count, numbers):
    with pytest.raises(ValidationError, match="page_count"):
        DocumentResult(source_file="a.pdf", page_count=count, pages=[_page(n) for n in numbers])


def test_duplicate_page_numbers_are_rejected():
    with pytest.raises(ValidationError, match="unique"):
        DocumentResult(source_file="a.pdf", page_count=2, pages=[_page(1), _page(1)])


def test_page_numbers_must_be_exactly_one_to_page_count():
    with pytest.raises(ValidationError, match=r"exactly 1\.\.2"):
        DocumentResult(source_file="a.pdf", page_count=2, pages=[_page(1), _page(5)])


@pytest.mark.parametrize("status", [PageStatus.CORRUPTED, PageStatus.FAILED])
@pytest.mark.parametrize("error", [None, "", "   "])
def test_corrupted_and_failed_pages_need_a_non_blank_error(status, error):
    with pytest.raises(ValidationError, match="error message"):
        PageResult(page_number=1, status=status, error=error)


def test_unknown_keys_are_rejected():
    with pytest.raises(ValidationError):
        TextField(name="date", status=FieldStatus.EMPTY_OR_UNREADABLE, surprise=1)


def test_document_round_trips_through_json():
    doc = DocumentResult(
        source_file="sample.pdf",
        page_count=2,
        pages=[
            PageResult(page_number=1, status=PageStatus.BLANK),
            PageResult(page_number=2, status=PageStatus.CORRUPTED, error="decode failed"),
        ],
    )
    payload = json.loads(doc.model_dump_json())
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["pages"][0]["status"] == "blank"
    assert DocumentResult.model_validate(payload) == doc
