"""Output schema for the digitization pipeline.

The schema is the contract with downstream systems. It encodes one rule on purpose:
a field only carries a value when extraction actually succeeded. Every other outcome
is a distinct status with a null value, so "nothing was read" can never be confused
with "the form said nothing".
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "2.0"
# 2.0 adds: the failed_test_form page status, text provenance (kind and OCR engine), choice groups,
# the structured observation form, alignment and failed-form evidence, and document metadata.
# Every 1.0 field, enum value and validation rule is unchanged; the new fields are optional.


class QualityVerdict(str, Enum):
    """Outcome of the page quality gate."""

    OK = "ok"
    BLANK = "blank"
    UNREADABLE = "unreadable"


class PageStatus(str, Enum):
    """Final status of one page."""

    OK = "ok"                      # processed, nothing flagged
    NEEDS_REVIEW = "needs_review"  # processed, but at least one item needs a human
    BLANK = "blank"                # no content found; nothing extracted
    UNREADABLE = "unreadable"      # image too poor to process; nothing extracted
    CORRUPTED = "corrupted"        # the page could not be decoded or rendered
    FAILED = "failed"              # unexpected processing error, isolated to this page
    # An explicitly marked "failed test form". A successful classification, not an error:
    # distinct from `failed`, `corrupted`, `unreadable` and `blank`. No observation data exists.
    FAILED_TEST_FORM = "failed_test_form"


class FieldStatus(str, Enum):
    """Outcome for a single text field. Only EXTRACTED fields may carry a value
    that is trusted; NEEDS_REVIEW may carry a candidate value for the reviewer.

    Review policy (enforced by `PageResult`): a page containing a NEEDS_REVIEW field or an
    OCR_UNAVAILABLE field cannot be reported as fully successful. OCR_UNAVAILABLE means the
    text was never read, so a person must read it. EMPTY_OR_UNREADABLE does not force a
    review because it also covers fields that are genuinely empty; it cannot tell empty
    from illegible, which a later stage should resolve."""

    EXTRACTED = "extracted"
    EMPTY_OR_UNREADABLE = "empty_or_unreadable"
    OCR_UNAVAILABLE = "ocr_unavailable"
    NEEDS_REVIEW = "needs_review"


class CheckboxState(str, Enum):
    CHECKED = "checked"
    UNCHECKED = "unchecked"
    UNCERTAIN = "uncertain"


class TextKind(str, Enum):
    """How a text field was written, which decides how far its OCR result may be trusted."""

    PRINTED = "printed"
    HANDWRITTEN = "handwritten"


class ChoiceStatus(str, Enum):
    ANSWERED = "answered"
    UNANSWERED = "unanswered"
    UNCERTAIN = "uncertain"
    CONFLICT = "conflict"


class FailedFormEvidenceSource(str, Enum):
    TEXT_LAYER = "text_layer"
    OCR = "ocr"


class AlignmentMethod(str, Enum):
    SIFT_AFFINE = "sift_affine"
    SIFT_HOMOGRAPHY = "sift_homography"
    ORB_AFFINE = "orb_affine"
    ORB_HOMOGRAPHY = "orb_homography"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _has_text(value: Optional[str]) -> bool:
    return value is not None and value.strip() != ""


class OcrEngineInfo(_Model):
    name: str
    version: Optional[str] = None


class QualityMetrics(_Model):
    mark_ratio: float = Field(ge=0.0, le=1.0)
    ink_ratio: float = Field(ge=0.0, le=1.0)
    content_components: int = Field(ge=0)
    mean_brightness: float = Field(ge=0.0, le=255.0)
    paper_level: float = Field(ge=0.0, le=255.0)
    contrast: float = Field(ge=0.0, le=255.0)
    sharpness: float = Field(ge=0.0)
    noise_sigma: float = Field(ge=0.0)
    speckle_ratio: float = Field(ge=0.0, le=1.0)


class PageQuality(_Model):
    verdict: QualityVerdict
    reasons: list[str] = Field(default_factory=list)
    metrics: QualityMetrics


class TextField(_Model):
    name: str
    value: Optional[str] = None
    status: FieldStatus
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    review_reason: Optional[str] = None
    # Provenance (schema 2.0). `text_kind` is None only for objects built the 1.0 way.
    text_kind: Optional[TextKind] = None
    engine: Optional[OcrEngineInfo] = None

    @model_validator(mode="after")
    def _check_consistency(self) -> "TextField":
        # Handwritten OCR is a candidate for a person to confirm. Neither a high confidence nor a
        # valid format may promote it, so the schema refuses 'extracted' for handwriting outright.
        if self.status is FieldStatus.EXTRACTED and self.text_kind is TextKind.HANDWRITTEN:
            raise ValueError("handwritten text can never be 'extracted'; it stays a candidate that needs review")
        if self.text_kind is not None and self.value is not None and self.engine is None:
            raise ValueError("a value with a text_kind must record the OCR engine that produced it")
        if self.status in (FieldStatus.EMPTY_OR_UNREADABLE, FieldStatus.OCR_UNAVAILABLE):
            if self.value is not None:
                raise ValueError(f"status '{self.status.value}' must not carry a value")
        if self.status is FieldStatus.EXTRACTED and not self.value:
            raise ValueError("status 'extracted' requires a non-empty value")
        if self.status is FieldStatus.NEEDS_REVIEW and not _has_text(self.review_reason):
            raise ValueError("status 'needs_review' requires a review_reason")
        # Any value must come from a recogniser that reported how sure it was. A value
        # with no confidence has no provenance and cannot be told apart from an invention.
        if self.value is not None and self.confidence is None:
            raise ValueError("a value must carry the recogniser's confidence")
        return self


class CheckboxField(_Model):
    name: str
    state: Optional[CheckboxState] = None
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    review_required: bool = False
    review_reason: Optional[str] = None
    # Supporting evidence (schema 2.0); never a substitute for the state's own validation.
    ink_pixels: Optional[int] = Field(default=None, ge=0)
    mark_area_fraction: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_consistency(self) -> "CheckboxField":
        # No definite state (unread or uncertain) always needs a human, with an explicit reason.
        if self.state in (None, CheckboxState.UNCERTAIN) and not self.review_required:
            raise ValueError("a checkbox without a definite state must be flagged for review")
        if self.review_required and not _has_text(self.review_reason):
            raise ValueError("review_required needs a non-empty review_reason")
        return self


class ChoiceGroup(_Model):
    """One question answered by ticking exactly one of several boxes (S/R/I/N, Yes/No/N/A,
    a CWA location, a shift). The state of each box is a `CheckboxField`.

    The group is only `answered` when exactly one box is checked and every other box is
    confidently empty. Anything else keeps `selected` null and asks for review, except an
    unanswered group that is not required."""

    key: str
    options: list[CheckboxField] = Field(min_length=2)
    required: bool = True
    status: ChoiceStatus
    selected: Optional[str] = None
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    review_required: bool = False
    review_reasons: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_consistency(self) -> "ChoiceGroup":
        names = [o.name for o in self.options]
        if len(names) != len(set(names)):
            raise ValueError("option names within a choice group must be unique")
        checked = [o.name for o in self.options if o.state is CheckboxState.CHECKED]
        undecided = [o for o in self.options if o.state in (None, CheckboxState.UNCERTAIN)]
        if self.status is ChoiceStatus.ANSWERED:
            if len(checked) != 1 or undecided:
                raise ValueError("an answered group needs exactly one checked option and no undecided option")
            if self.selected != checked[0]:
                raise ValueError("selected must be the checked option")
            if self.review_required:
                raise ValueError("an answered group does not require review")
        else:
            if self.selected is not None:
                raise ValueError("only an answered group may have a selected option")
            if self.status is ChoiceStatus.CONFLICT and len(checked) < 2:
                raise ValueError("a conflict needs two or more checked options")
            if self.status is ChoiceStatus.UNCERTAIN and not undecided:
                raise ValueError("an uncertain group needs at least one undecided option")
            if self.status is ChoiceStatus.UNANSWERED and (checked or undecided):
                raise ValueError("an unanswered group has no checked or undecided option")
            must_review = self.required if self.status is ChoiceStatus.UNANSWERED else True
            if must_review and not self.review_required:
                raise ValueError(f"a {self.status.value} group must be flagged for review")
            if not must_review and self.review_required:
                raise ValueError("an unanswered optional group must not be flagged for review")
        if self.review_required and not self.review_reasons:
            raise ValueError("review_required needs at least one entry in review_reasons")
        if self.review_reasons and not self.review_required:
            raise ValueError("review_reasons were given but review_required is false")
        return self


class ChecklistRow(_Model):
    row_number: int = Field(ge=1)
    printed_row_label: Optional[str] = None  # the label read from the form; a cross-check only
    question: TextField
    answer: ChoiceGroup


class ExpectationAnswer(_Model):
    key: str
    question: TextField
    answer: ChoiceGroup


class FormHeader(_Model):
    observation_date: TextField
    observer: TextField
    company_observed: TextField
    timesheet_number: TextField
    employee_number: TextField
    location: ChoiceGroup
    shift: ChoiceGroup


_NEEDS_HUMAN = (FieldStatus.NEEDS_REVIEW, FieldStatus.OCR_UNAVAILABLE)


class ObservationForm(_Model):
    """The content of one filled safety observation form. The number of checklist rows is data,
    not schema: it varies per form."""

    section_title: TextField
    header: FormHeader
    checklist: list[ChecklistRow] = Field(min_length=1)
    expectation_answers: list[ExpectationAnswer]
    comments: TextField

    @model_validator(mode="after")
    def _check_rows(self) -> "ObservationForm":
        numbers = [r.row_number for r in self.checklist]
        if numbers != list(range(1, len(numbers) + 1)):
            raise ValueError("checklist rows must be numbered 1..n in order")
        keys = [q.key for q in self.expectation_answers]
        if len(keys) != len(set(keys)):
            raise ValueError("expectation answer keys must be unique")
        return self

    def text_fields(self) -> list[TextField]:
        h = self.header
        fields = [self.section_title, h.observation_date, h.observer, h.company_observed,
                  h.timesheet_number, h.employee_number, self.comments]
        fields += [r.question for r in self.checklist] + [q.question for q in self.expectation_answers]
        return fields

    def choice_groups(self) -> list[ChoiceGroup]:
        groups = [self.header.location, self.header.shift]
        groups += [r.answer for r in self.checklist] + [q.answer for q in self.expectation_answers]
        return groups

    def needs_review(self) -> bool:
        return any(f.status in _NEEDS_HUMAN for f in self.text_fields()) or any(
            g.review_required for g in self.choice_groups()
        )


class AlignmentInfo(_Model):
    """How a page was registered to the template, and the evidence for accepting or rejecting it."""

    accepted: bool
    method: Optional[AlignmentMethod] = None
    reasons: list[str] = Field(default_factory=list)
    keypoints_template: int = Field(default=0, ge=0)
    keypoints_page: int = Field(default=0, ge=0)
    matches: int = Field(default=0, ge=0)
    inliers: int = Field(default=0, ge=0)
    inlier_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    inlier_spread_x: float = Field(default=0.0, ge=0.0)
    inlier_spread_y: float = Field(default=0.0, ge=0.0)
    median_residual_px: Optional[float] = Field(default=None, ge=0.0)
    scale_x: Optional[float] = None
    scale_y: Optional[float] = None
    rotation_degrees: Optional[float] = None
    shear_degrees: Optional[float] = None
    perspective_magnitude: Optional[float] = None
    verification_chamfer_mean_px: Optional[float] = Field(default=None, ge=0.0)
    verification_within_2px_fraction: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    matrix: Optional[list[list[float]]] = None
    opencv_version: str = ""

    @model_validator(mode="after")
    def _check_consistency(self) -> "AlignmentInfo":
        if self.accepted:
            if self.method is None or self.matrix is None:
                raise ValueError("an accepted alignment must record its method and matrix")
            if self.verification_chamfer_mean_px is None or self.verification_within_2px_fraction is None:
                raise ValueError("an accepted alignment must record the independent verification result")
        elif not self.reasons:
            raise ValueError("a rejected alignment must record why")
        return self


class WordBox(_Model):
    text: str
    bbox: tuple[float, float, float, float]


class FailedFormEvidence(_Model):
    source: FailedFormEvidenceSource
    matched_text: str
    words: list[WordBox] = Field(default_factory=list)
    direction_degrees: Optional[float] = None
    ocr_engine: Optional[OcrEngineInfo] = None
    ocr_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    detail: str = ""

    @model_validator(mode="after")
    def _check_consistency(self) -> "FailedFormEvidence":
        if not _has_text(self.matched_text):
            raise ValueError("failed-form evidence must record the matched text")
        if self.source is FailedFormEvidenceSource.OCR and self.ocr_engine is None:
            raise ValueError("OCR evidence must record the engine")
        return self


class PageResult(_Model):
    page_number: int = Field(ge=1)
    status: PageStatus
    quality: Optional[PageQuality] = None
    text_fields: list[TextField] = Field(default_factory=list)
    checkboxes: list[CheckboxField] = Field(default_factory=list)
    error: Optional[str] = None
    review_required: bool = False
    review_reasons: list[str] = Field(default_factory=list)
    # Schema 2.0. The flat lists above stay for compatibility; the structured form is used by the pipeline.
    form: Optional["ObservationForm"] = None
    alignment: Optional["AlignmentInfo"] = None
    failed_form_evidence: Optional["FailedFormEvidence"] = None

    @model_validator(mode="after")
    def _check_consistency(self) -> "PageResult":
        no_content = (
            PageStatus.BLANK,
            PageStatus.UNREADABLE,
            PageStatus.CORRUPTED,
            PageStatus.FAILED,
            PageStatus.FAILED_TEST_FORM,
        )
        if self.status in no_content and (self.text_fields or self.checkboxes or self.form is not None):
            raise ValueError(f"a '{self.status.value}' page must not contain extracted fields")
        if self.status is PageStatus.FAILED_TEST_FORM:
            evidence = self.failed_form_evidence
            if evidence is None:
                raise ValueError("a 'failed_test_form' page must record the evidence for the classification")
            if evidence.source is FailedFormEvidenceSource.OCR and not self.review_required:
                raise ValueError("a failed test form found only by OCR must require review")
        elif self.failed_form_evidence is not None:
            raise ValueError("failed_form_evidence is only valid on a 'failed_test_form' page")
        if self.form is not None and self.status not in (PageStatus.OK, PageStatus.NEEDS_REVIEW):
            raise ValueError("a page with extracted form data must be 'ok' or 'needs_review'")
        if self.form is not None and self.alignment is not None and not self.alignment.accepted:
            raise ValueError("form data must not exist for a page whose template alignment was rejected")
        if self.status in (PageStatus.CORRUPTED, PageStatus.FAILED) and not _has_text(self.error):
            raise ValueError(f"a '{self.status.value}' page must record a non-empty error message")

        # Page-level review status must agree with field-level review requirements.
        # A field that was flagged, or whose text was never read, needs a person.
        needs_human = (FieldStatus.NEEDS_REVIEW, FieldStatus.OCR_UNAVAILABLE)
        field_needs_review = (
            any(f.status in needs_human for f in self.text_fields)
            or any(c.review_required for c in self.checkboxes)
            or (self.form is not None and self.form.needs_review())
        )
        if field_needs_review and not self.review_required:
            raise ValueError(
                "a field needs review (needs_review or ocr_unavailable), "
                "so the page must have review_required=true"
            )
        if self.review_required and not self.review_reasons:
            raise ValueError("review_required needs at least one entry in review_reasons")
        if self.review_reasons and not self.review_required:
            raise ValueError("review_reasons were given but review_required is false")
        if self.status is PageStatus.NEEDS_REVIEW and not self.review_required:
            raise ValueError("status 'needs_review' requires review_required=true")
        if self.status is PageStatus.OK and self.review_required:
            raise ValueError("status 'ok' cannot require review; use 'needs_review'")
        return self


class DocumentMetadata(_Model):
    """Reproducibility metadata. Contains no timestamps so that output is deterministic."""

    schema_version: str
    pipeline_version: str
    source_file: str  # file name only
    source_sha256: str
    template_file: Optional[str] = None
    template_sha256: Optional[str] = None
    render_dpi: int = Field(ge=1)
    engines: dict[str, str] = Field(default_factory=dict)


class DocumentResult(_Model):
    """A COMPLETE result for one PDF: exactly one `PageResult` for each of its pages.

    There is deliberately no partial form. A run that did not finish every page must not be
    expressed as a `DocumentResult`; give each unprocessed page an explicit status instead.
    """

    schema_version: str = SCHEMA_VERSION
    source_file: str
    page_count: int = Field(ge=0)
    pages: list[PageResult] = Field(default_factory=list)
    metadata: Optional[DocumentMetadata] = None

    @model_validator(mode="after")
    def _check_pages_are_complete(self) -> "DocumentResult":
        numbers = [page.page_number for page in self.pages]
        if len(numbers) != self.page_count:
            raise ValueError(
                f"page_count is {self.page_count} but {len(numbers)} page result(s) were given"
            )
        duplicates = sorted({n for n in numbers if numbers.count(n) > 1})
        if duplicates:
            raise ValueError(f"page numbers must be unique; repeated: {duplicates}")
        if sorted(numbers) != list(range(1, self.page_count + 1)):
            raise ValueError(f"page numbers must be exactly 1..{self.page_count}, got {sorted(numbers)}")
        return self
