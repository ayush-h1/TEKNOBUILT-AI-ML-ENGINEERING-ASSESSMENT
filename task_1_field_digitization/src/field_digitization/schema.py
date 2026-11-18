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

SCHEMA_VERSION = "1.0"


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


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _has_text(value: Optional[str]) -> bool:
    return value is not None and value.strip() != ""


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

    @model_validator(mode="after")
    def _check_consistency(self) -> "TextField":
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

    @model_validator(mode="after")
    def _check_consistency(self) -> "CheckboxField":
        # No definite state (unread or uncertain) always needs a human, with an explicit reason.
        if self.state in (None, CheckboxState.UNCERTAIN) and not self.review_required:
            raise ValueError("a checkbox without a definite state must be flagged for review")
        if self.review_required and not _has_text(self.review_reason):
            raise ValueError("review_required needs a non-empty review_reason")
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

    @model_validator(mode="after")
    def _check_consistency(self) -> "PageResult":
        no_content = (
            PageStatus.BLANK,
            PageStatus.UNREADABLE,
            PageStatus.CORRUPTED,
            PageStatus.FAILED,
        )
        if self.status in no_content and (self.text_fields or self.checkboxes):
            raise ValueError(f"a '{self.status.value}' page must not contain extracted fields")
        if self.status in (PageStatus.CORRUPTED, PageStatus.FAILED) and not _has_text(self.error):
            raise ValueError(f"a '{self.status.value}' page must record a non-empty error message")

        # Page-level review status must agree with field-level review requirements.
        # A field that was flagged, or whose text was never read, needs a person.
        needs_human = (FieldStatus.NEEDS_REVIEW, FieldStatus.OCR_UNAVAILABLE)
        field_needs_review = any(f.status in needs_human for f in self.text_fields) or any(
            c.review_required for c in self.checkboxes
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


class DocumentResult(_Model):
    """A COMPLETE result for one PDF: exactly one `PageResult` for each of its pages.

    There is deliberately no partial form. A run that did not finish every page must not be
    expressed as a `DocumentResult`; give each unprocessed page an explicit status instead.
    """

    schema_version: str = SCHEMA_VERSION
    source_file: str
    page_count: int = Field(ge=0)
    pages: list[PageResult] = Field(default_factory=list)

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
