"""Backward-compatibility of schema 2.0 with schema 1.0.

Defaults alone are not treated as proof of compatibility. These tests freeze what 1.0 looked like
(field names, enum values, key order, a literal 1.0 payload) and check 2.0 against it.
"""

import json

import pytest
from pydantic import ValidationError

from field_digitization import schema
from field_digitization.schema import (
    SCHEMA_VERSION,
    CheckboxField,
    CheckboxState,
    DocumentResult,
    FieldStatus,
    PageQuality,
    PageResult,
    PageStatus,
    QualityMetrics,
    QualityVerdict,
    TextField,
)

# What schema 1.0 contained, written out by hand. Never derived from the current models.
V1_FIELDS = {
    "TextField": ["name", "value", "status", "confidence", "review_reason"],
    "CheckboxField": ["name", "state", "confidence", "review_required", "review_reason"],
    "QualityMetrics": ["mark_ratio", "ink_ratio", "content_components", "mean_brightness", "paper_level",
                       "contrast", "sharpness", "noise_sigma", "speckle_ratio"],
    "PageQuality": ["verdict", "reasons", "metrics"],
    "PageResult": ["page_number", "status", "quality", "text_fields", "checkboxes", "error",
                   "review_required", "review_reasons"],
    "DocumentResult": ["schema_version", "source_file", "page_count", "pages"],
}
V1_ENUMS = {
    "PageStatus": {"ok", "needs_review", "blank", "unreadable", "corrupted", "failed"},
    "FieldStatus": {"extracted", "empty_or_unreadable", "ocr_unavailable", "needs_review"},
    "CheckboxState": {"checked", "unchecked", "uncertain"},
    "QualityVerdict": {"ok", "blank", "unreadable"},
}


def test_schema_version_is_2_0_and_is_serialised():
    assert SCHEMA_VERSION == "2.0"
    doc = DocumentResult(source_file="a.pdf", page_count=0)
    assert json.loads(doc.model_dump_json())["schema_version"] == "2.0"


@pytest.mark.parametrize("model_name", sorted(V1_FIELDS))
def test_every_1_0_field_still_exists_in_the_same_relative_order(model_name):
    fields = list(getattr(schema, model_name).model_fields)
    for name in V1_FIELDS[model_name]:
        assert name in fields, f"{model_name}.{name} was removed"
    positions = [fields.index(n) for n in V1_FIELDS[model_name]]
    assert positions == sorted(positions), f"{model_name}: 1.0 fields were reordered"
    # 1.0 fields come first, so existing consumers reading by position or prefix are unaffected
    assert fields[: len(V1_FIELDS[model_name])] == V1_FIELDS[model_name]


@pytest.mark.parametrize("enum_name", sorted(V1_ENUMS))
def test_every_1_0_enum_value_still_exists_with_the_same_text(enum_name):
    values = {m.value for m in getattr(schema, enum_name)}
    assert V1_ENUMS[enum_name] <= values


def test_the_only_new_page_status_is_failed_test_form():
    new = {m.value for m in PageStatus} - V1_ENUMS["PageStatus"]
    assert new == {"failed_test_form"}


def test_1_0_style_objects_serialise_with_1_0_keys_unchanged_and_new_keys_empty():
    text = TextField(name="date", status=FieldStatus.EMPTY_OR_UNREADABLE)
    dump = text.model_dump(mode="json")
    assert {k: dump[k] for k in V1_FIELDS["TextField"]} == {
        "name": "date", "value": None, "status": "empty_or_unreadable", "confidence": None, "review_reason": None}
    assert dump["text_kind"] is None and dump["engine"] is None

    page = PageResult(page_number=1, status=PageStatus.OK, text_fields=[text])
    page_dump = page.model_dump(mode="json")
    assert page_dump["status"] == "ok" and page_dump["review_required"] is False
    assert page_dump["form"] is None and page_dump["alignment"] is None and page_dump["failed_form_evidence"] is None
    assert list(page_dump)[: len(V1_FIELDS["PageResult"])] == V1_FIELDS["PageResult"]


def test_a_literal_1_0_payload_validates_and_keeps_its_version():
    payload = {
        "schema_version": "1.0",
        "source_file": "legacy.pdf",
        "page_count": 2,
        "pages": [
            {"page_number": 1, "status": "blank", "quality": None, "text_fields": [], "checkboxes": [],
             "error": None, "review_required": False, "review_reasons": []},
            {"page_number": 2, "status": "needs_review", "quality": None,
             "text_fields": [{"name": "date", "value": "2O24", "status": "needs_review", "confidence": 0.4,
                              "review_reason": "low confidence"}],
             "checkboxes": [{"name": "ppe", "state": "uncertain", "confidence": None,
                             "review_required": True, "review_reason": "ambiguous"}],
             "error": None, "review_required": True, "review_reasons": ["a field needs review"]},
        ],
    }
    doc = DocumentResult.model_validate(payload)
    assert doc.schema_version == "1.0"            # a stored 1.0 document is not silently relabelled
    assert doc.metadata is None
    again = json.loads(doc.model_dump_json())
    for key in ("source_file", "page_count"):
        assert again[key] == payload[key]
    assert again["pages"][1]["text_fields"][0]["value"] == "2O24"
    assert DocumentResult.model_validate(again) == doc


def test_1_0_rules_still_hold_in_2_0():
    with pytest.raises(ValidationError):
        TextField(name="d", status=FieldStatus.EXTRACTED, value="x")                   # extracted needs confidence
    with pytest.raises(ValidationError):
        TextField(name="d", status=FieldStatus.OCR_UNAVAILABLE, value="invented")      # no value when unavailable
    with pytest.raises(ValidationError):
        CheckboxField(name="c", state=CheckboxState.UNCERTAIN)                         # uncertain must be flagged
    with pytest.raises(ValidationError):
        PageResult(page_number=1, status=PageStatus.CORRUPTED)                         # needs an error message
    with pytest.raises(ValidationError):
        PageResult(page_number=1, status=PageStatus.OK, text_fields=[TextField(name="x", status=FieldStatus.OCR_UNAVAILABLE)])


def test_legacy_extracted_text_without_a_kind_is_still_valid():
    field = TextField(name="q", status=FieldStatus.EXTRACTED, value="Has the job been stopped?", confidence=0.9)
    assert field.text_kind is None and field.engine is None


def test_unknown_keys_are_still_rejected_everywhere():
    with pytest.raises(ValidationError):
        PageResult.model_validate({"page_number": 1, "status": "blank", "surprise": 1})
    with pytest.raises(ValidationError):
        DocumentResult.model_validate({"source_file": "a", "page_count": 0, "surprise": 1})
    with pytest.raises(ValidationError):
        QualityMetrics.model_validate({"surprise": 1})


def test_serialisation_is_deterministic_and_key_order_is_stable():
    def build():
        return DocumentResult(
            source_file="a.pdf", page_count=2,
            pages=[PageResult(page_number=1, status=PageStatus.BLANK),
                   PageResult(page_number=2, status=PageStatus.CORRUPTED, error="decode failed")])
    first, second = build().model_dump_json(), build().model_dump_json()
    assert first == second
    assert list(json.loads(first)) == ["schema_version", "source_file", "page_count", "pages", "metadata"]


def test_quality_models_are_untouched():
    q = PageQuality(verdict=QualityVerdict.BLANK, reasons=["x"], metrics=QualityMetrics(
        mark_ratio=0, ink_ratio=0, content_components=0, mean_brightness=250, paper_level=250,
        contrast=0, sharpness=0, noise_sigma=0, speckle_ratio=0))
    assert PageQuality.model_validate_json(q.model_dump_json()) == q
