"""Process a PDF of safety observation forms into a `DocumentResult`.

Per page, in this order, each page isolated so one failure never stops the others:
  ingest -> gate (corrupted / blank / unreadable) -> failed-test-form marker in the text layer
  -> alignment to the template -> (alignment rejected: OCR check for the failed-form marker)
  -> checklist rows, header, tick boxes and text -> page status.

Document-level problems (no template, a template whose structure cannot be read, a PDF that cannot
be opened) raise, because nothing on any page could be read safely without them.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import cv2
import pymupdf

from . import __version__
from .alignment import align_page
from .config import FormSemantics, Thresholds, load_semantics, load_thresholds
from .extract import Readers, extract_form, warp_to_template
from .failed_form import from_ocr, from_text_layer
from .ingest import IngestedPage, iter_pages, open_pdf
from .layout import TemplateLayout, detect_header_layout, detect_template_layout
from .ocr import NullOcr, make_provider
from .page_gate import gate_page
from .schema import (
    SCHEMA_VERSION, DocumentMetadata, DocumentResult, FailedFormEvidenceSource, PageResult, PageStatus,
)
from .template import Template, load_template

logger = logging.getLogger(__name__)


@dataclass
class Context:
    template: Template
    layout: TemplateLayout
    semantics: FormSemantics
    thresholds: Thresholds
    readers: Readers


def build_context(template_path: Union[str, Path], thresholds: Optional[Thresholds] = None,
                  semantics: Optional[FormSemantics] = None, ocr: Optional[str] = None) -> Context:
    """`ocr` overrides both provider settings: 'auto', 'tesseract' or 'none'."""
    thresholds = thresholds or load_thresholds()
    semantics = semantics or load_semantics()
    template = load_template(template_path)
    layout = detect_template_layout(template, semantics)
    layout = dataclasses.replace(layout, header=detect_header_layout(template, layout, semantics))
    cfg = thresholds.ocr
    printed_setting = ocr or cfg.printed_provider
    hand_setting = ocr or cfg.handwritten_provider
    printed = make_provider(printed_setting, cfg)
    handwritten = printed if hand_setting == printed_setting else make_provider(hand_setting, cfg)
    return Context(template, layout, semantics, thresholds, Readers(printed, handwritten, cfg))


def process_page(page: IngestedPage, ctx: Context) -> PageResult:
    decision = gate_page(page, ctx.thresholds.quality)
    if not decision.proceed:
        return decision.result
    notes = list(decision.review_reasons)
    number, quality = page.page_number, decision.quality

    evidence = from_text_layer(page.text_layer)
    if evidence is not None:
        return PageResult(page_number=number, status=PageStatus.FAILED_TEST_FORM, quality=quality,
                          failed_form_evidence=evidence, review_required=bool(notes), review_reasons=notes)

    alignment = align_page(page.image, ctx.template, ctx.layout, ctx.thresholds.alignment)
    if not alignment.accepted:
        if page.text_layer is None or page.text_layer.is_empty:
            evidence = from_ocr(page.image, ctx.readers.printed)
            if evidence is not None:
                reasons = notes + ["classified as a failed test form from OCR only; please confirm"]
                return PageResult(page_number=number, status=PageStatus.FAILED_TEST_FORM, quality=quality,
                                  failed_form_evidence=evidence, review_required=True, review_reasons=reasons)
        reasons = notes + ["page could not be aligned to the template, so nothing was read: "
                           + "; ".join(alignment.reasons)]
        return PageResult(page_number=number, status=PageStatus.NEEDS_REVIEW, quality=quality,
                          alignment=alignment, review_required=True, review_reasons=reasons)

    aligned = warp_to_template(page.image, alignment, ctx.template, ctx.thresholds.extraction)
    form, reasons = extract_form(aligned, ctx.layout, ctx.semantics, ctx.readers, ctx.thresholds.extraction)
    reasons = notes + reasons
    if form is None:
        reasons = reasons or ["no form content could be read"]
    status = PageStatus.NEEDS_REVIEW if reasons else PageStatus.OK
    return PageResult(page_number=number, status=status, quality=quality, alignment=alignment, form=form,
                      review_required=bool(reasons), review_reasons=reasons)


def _failed(page_number: int, exc: BaseException) -> PageResult:
    return PageResult(page_number=page_number, status=PageStatus.FAILED, error=f"{type(exc).__name__}: {exc}",
                      review_required=True, review_reasons=["unexpected processing error on this page"])


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def process_pdf(pdf_path: Union[str, Path], ctx: Context) -> DocumentResult:
    pdf_path = Path(pdf_path)
    doc = open_pdf(pdf_path)
    try:
        pages: list[PageResult] = []
        for page in iter_pages(doc, ctx.thresholds.ingest):
            try:
                pages.append(process_page(page, ctx))
            except Exception as exc:  # isolate: one bad page never stops the batch
                logger.exception("page %d failed", page.page_number)
                pages.append(_failed(page.page_number, exc))
        page_count = doc.page_count
    finally:
        doc.close()
    engines = {"opencv": cv2.__version__, "pymupdf": pymupdf.__version__}
    for provider in {id(p): p for p in (ctx.readers.printed, ctx.readers.handwritten)}.values():
        info = provider.engine_info()
        if not isinstance(provider, NullOcr):
            engines[info.name] = info.version or "unknown"
    metadata = DocumentMetadata(
        schema_version=SCHEMA_VERSION, pipeline_version=__version__, source_file=pdf_path.name,
        source_sha256=_sha256(pdf_path), template_file=Path(ctx.template.path).name,
        template_sha256=ctx.template.sha256, render_dpi=ctx.thresholds.ingest.render_dpi, engines=engines,
    )
    return DocumentResult(source_file=pdf_path.name, page_count=page_count, pages=pages, metadata=metadata)


def summarize(result: DocumentResult) -> dict[str, Any]:
    """Counts for the processing report. No accuracy figure: the pipeline cannot know whether a value is right."""
    statuses = Counter(p.status.value for p in result.pages)
    text_status: Counter = Counter()
    group_status: Counter = Counter()
    boxes: Counter = Counter()
    for p in result.pages:
        if p.form is None:
            continue
        text_status.update(f.status.value for f in p.form.text_fields())
        for g in p.form.choice_groups():
            group_status[g.status.value] += 1
            boxes.update(o.state.value if o.state else "unread" for o in g.options)
    forms = [p for p in result.pages if p.form is not None]
    return {
        "source_file": result.source_file,
        "total_pages": result.page_count,
        "page_status": dict(sorted(statuses.items())),
        "processing_completed": sum(p.status is not PageStatus.FAILED for p in result.pages),
        "review_required": sum(p.review_required for p in result.pages),
        "fully_automatic_extraction": sum(not p.review_required for p in forms),
        "failed_test_form_pages": [p.page_number for p in result.pages if p.status is PageStatus.FAILED_TEST_FORM],
        "failed_test_form_from_ocr_only": [p.page_number for p in result.pages if p.failed_form_evidence is not None
                                           and p.failed_form_evidence.source is FailedFormEvidenceSource.OCR],
        "checklist_rows_per_form_page": {p.page_number: len(p.form.checklist) for p in forms},
        "text_fields": dict(sorted(text_status.items())),
        "choice_groups": dict(sorted(group_status.items())),
        "checkboxes": dict(sorted(boxes.items())),
        "errors": {p.page_number: p.error for p in result.pages if p.error},
    }


def write_json(data: Union[DocumentResult, dict], path: Union[str, Path]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = data.model_dump_json(indent=2) if isinstance(data, DocumentResult) else json.dumps(data, indent=2)
    path.write_text(text + "\n", encoding="utf-8")
    return path
