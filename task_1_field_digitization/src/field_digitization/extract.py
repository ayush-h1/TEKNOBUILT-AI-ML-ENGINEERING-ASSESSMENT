"""Read one aligned form page: checklist rows, header fields, tick boxes and text.

Everything happens in the template's pixel frame. The page is warped there using the accepted
alignment, so every cell comes from the template layout and nothing is a fixed coordinate.

Ink evidence. A pixel is ink when it is darker than `dark_ratio` of the paper level. Two masks are
removed before anything is counted:
  * the page's own ruled lines (dilated by `line_tolerance_px`), so grid lines never count as marks;
  * for marks and handwriting, everything the blank template already prints there (dilated the same
    way), so printed labels such as "Comments:" or row numbers never count as filled-in content.
This whole-cell subtraction replaces a fixed inset, which on the sample clipped 31 of 43 marks.

Nothing is guessed: a row without its grid is not a row, an ambiguous box is `uncertain`, a group
with no or several marks is flagged, and text the OCR policy does not trust stays a candidate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .config import ExtractionConfig, FormSemantics, OcrConfig
from .layout import Band, Cell, TemplateLayout
from .ocr import OcrProvider, read_text_field
from .schema import (
    AlignmentInfo, CheckboxField, CheckboxState, ChecklistRow, ChoiceGroup, ChoiceStatus, ExpectationAnswer,
    FieldStatus, FormHeader, ObservationForm, TextField, TextKind,
)
from .template import Template, line_masks

_NEEDS_HUMAN = (FieldStatus.NEEDS_REVIEW, FieldStatus.OCR_UNAVAILABLE)
_THICK_PX = 9  # ink at least this thick in both directions is not a ruled line (lines are 2-5 px)


@dataclass
class AlignedPage:
    gray: np.ndarray          # the page in the template frame
    ink: np.ndarray           # bool: dark pixels
    h_lines: np.ndarray       # bool: the page's own horizontal rules (not dilated)
    v_lines: np.ndarray       # bool: the page's own vertical rules (not dilated)
    lines: np.ndarray         # bool: page rules, dilated by the tolerance
    printed: np.ndarray       # bool: what the blank template prints, dilated by the tolerance

    @property
    def residual(self) -> np.ndarray:
        """Ink that is neither a ruled line nor printed on the blank template: marks and handwriting."""
        return self.ink & ~self.lines & ~self.printed

    @property
    def unruled(self) -> np.ndarray:
        """Ink that is not a ruled line: printed text that differs per form, plus marks."""
        return self.ink & ~self.lines


def warp_to_template(gray: np.ndarray, alignment: AlignmentInfo, template: Template, cfg: ExtractionConfig) -> AlignedPage:
    if not alignment.accepted or alignment.matrix is None:
        raise ValueError("only an accepted alignment can be used to read a page")
    paper = float(np.median(gray))
    warped = cv2.warpPerspective(gray, np.asarray(alignment.matrix, np.float64), (template.width, template.height),
                                 flags=cv2.INTER_LINEAR, borderValue=int(round(paper)))
    page_paper = float(np.median(warped))
    ink = warped < cfg.dark_ratio * page_paper
    # A ruled line is thin. Solid ink (a filled box, a heavy scribble) can be long enough to pass the
    # line detector, so anything thicker than a line is excluded from the line masks; otherwise a filled
    # box would be erased and read as empty.
    thick = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_OPEN, np.ones((_THICK_PX, _THICK_PX), np.uint8))
    thick = cv2.dilate(thick, np.ones((5, 5), np.uint8)) > 0
    h, v = line_masks(warped)
    h, v = (h > 0) & ~thick, (v > 0) & ~thick
    k = 2 * cfg.line_tolerance_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    lines = cv2.dilate((h | v).astype(np.uint8), kernel) > 0
    printed = cv2.dilate((template.gray < cfg.dark_ratio * template.paper).astype(np.uint8), kernel) > 0
    return AlignedPage(warped, ink, h, v, lines, printed)


def _box(cell: Cell, shape: tuple[int, int], inset: int = 0) -> tuple[int, int, int, int]:
    h, w = shape
    x0 = max(int(round(cell.x0)) + inset, 0)
    y0 = max(int(round(cell.y0)) + inset, 0)
    x1 = min(int(round(cell.x1)) - inset, w)
    y1 = min(int(round(cell.y1)) - inset, h)
    return x0, y0, max(x1, x0), max(y1, y0)


# ---------------------------------------------------------------- tick boxes
def read_checkbox(page: AlignedPage, cell: Cell, name: str, cfg: ExtractionConfig) -> CheckboxField:
    x0, y0, x1, y1 = _box(cell, page.gray.shape)
    area = max((x1 - x0) * (y1 - y0), 1)
    ink = int(page.residual[y0:y1, x0:x1].sum())
    fraction = ink / area
    evidence = dict(ink_pixels=ink, mark_area_fraction=round(fraction, 5))
    if fraction <= cfg.checkbox_empty_max_fraction:
        return CheckboxField(name=name, state=CheckboxState.UNCHECKED, **evidence)
    if cfg.checkbox_checked_min_fraction <= fraction <= cfg.checkbox_checked_max_fraction:
        return CheckboxField(name=name, state=CheckboxState.CHECKED, **evidence)
    if fraction < cfg.checkbox_checked_min_fraction:
        reason = f"faint mark: {fraction:.2%} of the box is inked, between the empty and checked thresholds"
    else:
        reason = f"{fraction:.2%} of the box is inked, more than a tick (scribble, correction or filled box)"
    return CheckboxField(name=name, state=CheckboxState.UNCERTAIN, review_required=True, review_reason=reason, **evidence)


def choice_group(key: str, options: list[CheckboxField], required: bool) -> ChoiceGroup:
    checked = [o.name for o in options if o.state is CheckboxState.CHECKED]
    uncertain = [o.name for o in options if o.state is CheckboxState.UNCERTAIN]
    if uncertain:
        return ChoiceGroup(key=key, options=options, required=required, status=ChoiceStatus.UNCERTAIN,
                           review_required=True, review_reasons=[f"uncertain mark in: {', '.join(uncertain)}"])
    if len(checked) > 1:
        return ChoiceGroup(key=key, options=options, required=required, status=ChoiceStatus.CONFLICT,
                           review_required=True, review_reasons=[f"more than one option is marked: {', '.join(checked)}"])
    if checked:
        return ChoiceGroup(key=key, options=options, required=required, status=ChoiceStatus.ANSWERED, selected=checked[0])
    if required:
        return ChoiceGroup(key=key, options=options, required=True, status=ChoiceStatus.UNANSWERED,
                           review_required=True, review_reasons=["required question has no mark"])
    return ChoiceGroup(key=key, options=options, required=False, status=ChoiceStatus.UNANSWERED)


# ---------------------------------------------------------------- checklist rows
def _rule_coverage_h(mask: np.ndarray, y: float, x0: float, x1: float, tol: int) -> float:
    yi = int(round(y))
    strip = mask[max(yi - tol - 1, 0):yi + tol + 2, int(x0):int(x1)]
    return float(strip.any(axis=0).mean()) if strip.size else 0.0


def _rule_coverage_v(mask: np.ndarray, x: float, y0: float, y1: float, tol: int) -> float:
    xi = int(round(x))
    a, b = int(y0) + tol + 2, int(y1) - tol - 2
    strip = mask[a:b, max(xi - tol - 1, 0):xi + tol + 2]
    return float(strip.any(axis=1).mean()) if strip.size else 0.0


def detect_rows(page: AlignedPage, layout: TemplateLayout, cfg: ExtractionConfig) -> tuple[list[Band], list[str]]:
    """Checklist rows actually drawn on this page. A template slot is a row only when its bottom rule and
    every choice-column rule are present on the page. Rows are read from the top; a drawn slot after an
    undrawn one, or ink in an undrawn slot, is reported for review and not read."""
    tol = cfg.line_tolerance_px
    cols = layout.choice_columns_x
    present = []
    for slot in layout.checklist_slots:
        bottom = _rule_coverage_h(page.h_lines, slot.y1, cols[0], cols[-1], tol)
        sides = min(_rule_coverage_v(page.v_lines, x, slot.y0, slot.y1, tol) for x in cols)
        present.append(bottom >= cfg.row_rule_min_coverage and sides >= cfg.row_rule_min_coverage)
    n_rows = present.index(False) if False in present else len(present)
    reasons: list[str] = []
    if n_rows == 0:
        reasons.append("no checklist row with a complete grid was found on this page")
    later = [i + 1 for i in range(n_rows, len(present)) if present[i]]
    if later:
        reasons.append(f"checklist grid found again in slot(s) {later} after an undrawn slot; those slots were not read")
    x0, x1 = layout.checklist_rules_x[0], layout.checklist_rules_x[-1]
    for i in range(n_rows, len(present)):
        slot = layout.checklist_slots[i]
        a, b, c, d = _box(Cell(x0, slot.y0, x1, slot.y1), page.gray.shape, inset=tol + 2)
        ink = int(page.residual[b:d, a:c].sum())
        if ink >= cfg.orphan_ink_min_px and not present[i]:
            reasons.append(f"ink found in checklist slot {i + 1}, which has no grid on this page; it was not read")
    return list(layout.checklist_slots[:n_rows]), reasons


# ---------------------------------------------------------------- text
def _text_crop(page: AlignedPage, cell: Cell, mode: str, inset: int) -> tuple[np.ndarray, int]:
    """A cleaned crop for OCR plus its ink count. mode 'residual' removes lines and template print
    (handwriting, per-form question text); 'unruled' removes lines only (printed text that the
    template also prints, such as the expectation questions)."""
    x0, y0, x1, y1 = _box(cell, page.gray.shape, inset=inset)
    source = {"residual": page.residual, "unruled": page.unruled, "raw": page.ink}[mode]
    keep = source[y0:y1, x0:x1]
    crop = page.gray[y0:y1, x0:x1].copy()
    crop[~keep & (crop < 255)] = 255  # keep only the ink that belongs to this field
    return crop, int(keep.sum())


@dataclass
class Readers:
    printed: OcrProvider
    handwritten: OcrProvider
    ocr: OcrConfig


def _read(readers: Readers, page: AlignedPage, cell: Cell, name: str, kind: TextKind, mode: str,
          cfg: ExtractionConfig, single_line: bool) -> TextField:
    crop, ink = _text_crop(page, cell, mode, inset=cfg.line_tolerance_px + 2)
    provider = readers.handwritten if kind is TextKind.HANDWRITTEN else readers.printed
    return read_text_field(provider, crop, name=name, kind=kind, has_ink=ink >= cfg.min_text_ink_px,
                           printed_min_confidence=readers.ocr.printed_min_confidence, single_line=single_line)


# ---------------------------------------------------------------- the whole form
def extract_form(page: AlignedPage, layout: TemplateLayout, semantics: FormSemantics, readers: Readers,
                 cfg: ExtractionConfig) -> tuple[Optional[ObservationForm], list[str]]:
    """Return the form (None when no checklist row exists) and the page-level review reasons."""
    if layout.header is None:
        raise ValueError("the template layout has no header layout")
    reasons: list[str] = []
    rows, row_reasons = detect_rows(page, layout, cfg)
    reasons += row_reasons
    if not rows:
        return None, reasons

    w = float(layout.width)
    hand, printed = TextKind.HANDWRITTEN, TextKind.PRINTED

    title = _read(readers, page, Cell(0, layout.title_bar.y0, w, layout.title_bar.y1),
                  semantics.section_title.key, printed, "raw", cfg, single_line=True)

    texts = {}
    for spec, cell in zip(semantics.header_text_fields, layout.header.text_cells):
        texts[spec.key] = _read(readers, page, cell, spec.key, hand, "residual", cfg, single_line=True)
        if spec.required and texts[spec.key].status is FieldStatus.EMPTY_OR_UNREADABLE:
            reasons.append(f"required header field '{spec.key}' is empty")

    def group(key: str, names: list[str], cells, required: bool) -> ChoiceGroup:
        return choice_group(key, [read_checkbox(page, c, n, cfg) for n, c in zip(names, cells)], required)

    location = group(semantics.location.key, semantics.location.options, layout.header.location_boxes,
                     semantics.location.required)
    shift = group(semantics.shift.key, semantics.shift.options, layout.header.shift_boxes, semantics.shift.required)

    cols = layout.choice_columns_x
    rule_index = layout.checklist_rules_x.index(cols[0])
    question_x0 = layout.checklist_rules_x[rule_index - 1] if rule_index > 0 else 0.0
    checklist = []
    for number, slot in enumerate(rows, start=1):
        question = _read(readers, page, Cell(question_x0, slot.y0, cols[0], slot.y1),
                         f"checklist_row_{number}_question", printed, "residual", cfg, single_line=False)
        if question.status is FieldStatus.EMPTY_OR_UNREADABLE:
            reasons.append(f"checklist row {number} has no question text")
        cells = [Cell(a, slot.y0, b, slot.y1) for a, b in zip(cols[:-1], cols[1:])]
        answer = group(f"checklist_row_{number}", semantics.checklist.options, cells, semantics.checklist.required)
        checklist.append(ChecklistRow(row_number=number, question=question, answer=answer))

    ex_cols = layout.cczjv_rules_x
    expectations = []
    for key, band in zip(semantics.expectation.question_keys, layout.cczjv_rows):
        question = _read(readers, page, Cell(0, band.y0, ex_cols[0], band.y1), f"expectation_{key}_question",
                         printed, "unruled", cfg, single_line=True)
        cells = [Cell(a, band.y0, b, band.y1) for a, b in zip(ex_cols[:-1], ex_cols[1:])]
        answer = group(f"expectation_{key}", semantics.expectation.options, cells, semantics.expectation.required)
        expectations.append(ExpectationAnswer(key=key, question=question, answer=answer))

    comments = _read(readers, page, Cell(0, layout.comments.y0, w, layout.comments.y1), semantics.comments.key,
                     hand, "residual", cfg, single_line=False)

    header = FormHeader(location=location, shift=shift, **texts)
    form = ObservationForm(section_title=title, header=header, checklist=checklist,
                           expectation_answers=expectations, comments=comments)
    for field in form.text_fields():
        if field.status in _NEEDS_HUMAN:
            reasons.append(f"{field.name}: {field.status.value}")
    for g in form.choice_groups():
        if g.review_required:
            reasons.append(f"{g.key}: {g.status.value} ({'; '.join(g.review_reasons)})")
    return form, reasons
