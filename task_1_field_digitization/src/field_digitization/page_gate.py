"""Decide whether a rendered page may proceed to extraction.

Combines the ingestion evidence (render errors, MuPDF warnings, suspected damage) with
the image quality verdict. The important rule: a page is only reported as BLANK when
nothing suggests data was lost. A blank-looking page that came with warnings is sent to
human review instead, because damage and genuine emptiness look identical in pixels.

Outcomes for a page that does not proceed, in the order they are decided:
  an error message, with or without an image
                                       -> CORRUPTED (review required). An image that comes
                                          with an error is not trusted and is never judged
                                          blank. Any damage evidence is appended to the error.
  no image and no error message        -> FAILED    (review required; ingestion broke its
                                                     contract, a fallback error is recorded)
  damage suspected                     -> CORRUPTED (review required). If the quality check
                                          also raised, the exception text is appended to
                                          the damage evidence; it never replaces it.
  quality check raised, no damage      -> FAILED    (review required)
  blank, with render warnings          -> NEEDS_REVIEW (page-level warnings)
  blank, with page-tree warnings only  -> NEEDS_REVIEW (document-level uncertainty: the
                                          warning cannot be attributed to this page, so the
                                          page is never called CORRUPTED because of it)
  blank, no warnings                   -> BLANK
  unreadable                           -> UNREADABLE (review required, needs a re-scan)

Only render-phase warnings (`IngestedPage.warnings`) take part in the damage decision, and
that decision is made in ingest.py. Load-phase warnings (`IngestedPage.load_warnings`) are
page-tree messages and only ever lead to a review request.

Every `PageResult` built here satisfies the schema's consistency rules, including an error
message on CORRUPTED and FAILED pages, so the gate cannot itself raise a validation error.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from .config import QualityConfig
from .ingest import IngestedPage
from .quality import assess_quality
from .schema import PageQuality, PageResult, PageStatus, QualityVerdict

logger = logging.getLogger(__name__)

_NO_IMAGE_NO_ERROR = "ingestion returned neither an image nor an error message for this page"
_NO_ERROR_DETAIL = "no error detail was recorded"


@dataclass(frozen=True)
class GateDecision:
    proceed: bool
    # Terminal result when `proceed` is False; None when extraction should continue.
    result: Optional[PageResult] = None
    quality: Optional[PageQuality] = None
    # When proceeding: reasons later stages must carry into the page's review reasons.
    review_reasons: tuple[str, ...] = ()


def gate_page(page: IngestedPage, config: Optional[QualityConfig] = None) -> GateDecision:
    has_error = bool(page.error and page.error.strip())

    if has_error:
        # An error always wins. Without an image the page was not rendered; with an image the
        # page contradicts itself (ingestion never produces this), so the image is not trusted.
        # Either way it must not be judged blank, and any damage evidence is kept.
        error = page.error
        if page.damage:
            error += f"; damage evidence: {page.damage}"
        if page.ok:
            reasons = ["page reported an error even though an image is present; image not trusted"]
        else:
            reasons = ["page could not be rendered"]
        return _stop(page, PageStatus.CORRUPTED, error=error, reasons=reasons)

    if not page.ok:
        # A page must carry an image or an error. Fail visibly and keep any evidence we have.
        error = _NO_IMAGE_NO_ERROR
        if page.damage:
            error += f"; damage evidence: {page.damage}"
        return _stop(
            page, PageStatus.FAILED, error=error,
            reasons=["ingestion produced neither an image nor an error for this page"],
        )

    quality: Optional[PageQuality] = None
    quality_error: Optional[str] = None
    try:
        quality = assess_quality(page.image, config)
    except Exception as exc:  # isolate: a bug or odd image must not stop the batch
        quality_error = f"{type(exc).__name__}: {exc}"
        logger.exception("quality assessment failed for page %d", page.page_number)

    # Known damage evidence is decided first so that a failing quality check cannot hide it.
    if page.damage:
        error = page.damage
        reasons = ["page appears damaged; its content may have been lost"]
        if quality_error:
            error += f"; quality assessment also failed: {quality_error}"
            reasons.append("quality assessment also failed")
        return _stop(page, PageStatus.CORRUPTED, error=error, quality=quality, reasons=reasons)

    if quality_error:
        return _stop(
            page, PageStatus.FAILED, error=quality_error,
            reasons=["quality assessment failed"],
        )

    if quality.verdict is QualityVerdict.BLANK:
        if page.warnings:
            return _stop(
                page, PageStatus.NEEDS_REVIEW, quality=quality,
                reasons=["page looks blank but MuPDF reported warnings while rendering it"],
            )
        if page.load_warnings:
            return _stop(
                page, PageStatus.NEEDS_REVIEW, quality=quality,
                reasons=[
                    "page looks blank, and MuPDF reported page-tree problems for this document; "
                    "they cannot be attributed to this page, so whether it is genuinely blank "
                    "cannot be confirmed"
                ],
            )
        return _stop(page, PageStatus.BLANK, quality=quality)

    if quality.verdict is QualityVerdict.UNREADABLE:
        return _stop(
            page, PageStatus.UNREADABLE, quality=quality,
            reasons=[f"unreadable scan: {reason}" for reason in quality.reasons],
        )

    notes: list[str] = []
    if page.warnings:
        notes.append(f"MuPDF reported {len(page.warnings)} warning(s) while rendering this page")
    if page.load_warnings:
        notes.append(
            f"MuPDF reported {len(page.load_warnings)} page-tree warning(s) for this document "
            f"when loading this page; the file may be damaged elsewhere"
        )
    return GateDecision(proceed=True, quality=quality, review_reasons=tuple(notes))


def _stop(
    page: IngestedPage,
    status: PageStatus,
    *,
    error: Optional[str] = None,
    quality: Optional[PageQuality] = None,
    reasons: Optional[list[str]] = None,
) -> GateDecision:
    reasons = reasons or []
    if status in (PageStatus.CORRUPTED, PageStatus.FAILED) and not (error and error.strip()):
        error = _NO_ERROR_DETAIL  # the schema requires a message on these statuses
    result = PageResult(
        page_number=page.page_number,
        status=status,
        quality=quality,
        error=error,
        review_required=bool(reasons),
        review_reasons=reasons,
    )
    return GateDecision(proceed=False, result=result, quality=quality)
