"""Tick-box reading, choice-group rules, checklist row detection and header layout."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from field_digitization.config import DEFAULT_SEMANTICS_PATH, ExtractionConfig, load_semantics
from field_digitization.extract import AlignedPage, choice_group, detect_rows, read_checkbox, warp_to_template
from field_digitization.layout import Cell, LayoutError, detect_header_layout, detect_template_layout
from field_digitization.schema import AlignmentInfo, AlignmentMethod, CheckboxState, ChoiceStatus
from field_digitization.template import load_template, template_from_gray

import synthetic_form as sf

CFG = ExtractionConfig()
SEMANTICS = load_semantics(DEFAULT_SEMANTICS_PATH)
REAL_TEMPLATE = Path(__file__).resolve().parents[1] / "data" / "template" / "template.png"
IDENTITY = AlignmentInfo(accepted=True, method=AlignmentMethod.SIFT_AFFINE, matrix=np.eye(3).tolist(),
                         verification_chamfer_mean_px=0.0, verification_within_2px_fraction=1.0)


def _cell_page(draw) -> AlignedPage:
    gray = np.full((100, 100), 255, np.uint8)
    draw(gray)
    none = np.zeros(gray.shape, bool)
    return AlignedPage(gray, gray < 140, none, none, none, none)


# ---------------------------------------------------------------- tick boxes (synthetic 100x100 cell)
def _tick(g):
    cv2.line(g, (40, 40), (60, 60), 0, 3)
    cv2.line(g, (60, 40), (40, 60), 0, 3)


@pytest.mark.parametrize("draw,state", [
    (lambda g: None, CheckboxState.UNCHECKED),
    (_tick, CheckboxState.CHECKED),
    (lambda g: g.__setitem__((slice(50, 53), slice(50, 60)), 0), CheckboxState.UNCERTAIN),   # faint: 0.3%
    (lambda g: g.__setitem__((slice(10, 90), slice(10, 90)), 0), CheckboxState.UNCERTAIN),   # filled box
])
def test_checkbox_states(draw, state):
    box = read_checkbox(_cell_page(draw), Cell(0, 0, 100, 100), "S", CFG)
    assert box.state is state
    assert box.review_required is (state is CheckboxState.UNCERTAIN)
    assert box.ink_pixels is not None and box.mark_area_fraction is not None


def test_ruled_lines_and_template_print_are_not_marks():
    gray = np.full((100, 100), 255, np.uint8)
    gray[48:52, :] = 0  # a ruled line through the cell
    gray[10:20, 10:20] = 0  # something the blank template prints here
    ink = gray < 140
    lines = np.zeros_like(ink); lines[46:54, :] = True
    printed = np.zeros_like(ink); printed[8:22, 8:22] = True
    page = AlignedPage(gray, ink, lines, lines, lines, printed)
    assert read_checkbox(page, Cell(0, 0, 100, 100), "S", CFG).state is CheckboxState.UNCHECKED


# ---------------------------------------------------------------- choice groups
def _box(name, state):
    from field_digitization.schema import CheckboxField
    review = state is CheckboxState.UNCERTAIN
    return CheckboxField(name=name, state=state, review_required=review, review_reason="faint" if review else None)


U, C, Q = CheckboxState.UNCHECKED, CheckboxState.CHECKED, CheckboxState.UNCERTAIN


@pytest.mark.parametrize("states,required,status,selected,review", [
    ((U, C, U, U), True, ChoiceStatus.ANSWERED, "R", False),
    ((U, U, U, U), True, ChoiceStatus.UNANSWERED, None, True),
    ((U, U, U, U), False, ChoiceStatus.UNANSWERED, None, False),
    ((C, C, U, U), True, ChoiceStatus.CONFLICT, None, True),
    ((C, Q, U, U), True, ChoiceStatus.UNCERTAIN, None, True),
])
def test_choice_group_rules(states, required, status, selected, review):
    group = choice_group("row", [_box(n, s) for n, s in zip("SRIN", states)], required)
    assert (group.status, group.selected, group.review_required) == (status, selected, review)


# ---------------------------------------------------------------- checklist rows (synthetic template)
@pytest.fixture(scope="module")
def synthetic():
    template = template_from_gray(sf.make_template(12))
    return template, detect_template_layout(template, SEMANTICS)


def _form_with_rows(template, layout, keep):
    page = template.gray.copy()
    top, h = int(layout.checklist_slots[0].y0), int(round(layout.checklist_slots[0].height))
    page[top + keep * h + 4: top + 12 * h - 4, 70:] = 255  # remove the grid of the unused slots
    return page


def _aligned(gray, template):
    return warp_to_template(gray, IDENTITY, template, CFG)


@pytest.mark.parametrize("keep", [1, 5, 12])
def test_rows_are_counted_from_the_page_grid(synthetic, keep):
    template, layout = synthetic
    rows, reasons = detect_rows(_aligned(_form_with_rows(template, layout, keep), template), layout, CFG)
    assert len(rows) == keep
    assert reasons == []


def test_no_grid_gives_no_rows_and_a_reason(synthetic):
    template, layout = synthetic
    page = template.gray.copy()
    page[int(layout.checklist_slots[0].y0) + 4:int(layout.checklist_slots[-1].y1) - 4, 70:] = 255
    rows, reasons = detect_rows(_aligned(page, template), layout, CFG)
    assert rows == [] and any("no checklist row" in r for r in reasons)


def test_ink_in_an_undrawn_slot_is_reported_not_read(synthetic):
    template, layout = synthetic
    page = _form_with_rows(template, layout, 5)
    slot = layout.checklist_slots[8]
    cv2.putText(page, "stray note", (200, int(slot.y0) + 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)
    rows, reasons = detect_rows(_aligned(page, template), layout, CFG)
    assert len(rows) == 5 and any("slot 9" in r for r in reasons)


def test_grid_after_a_gap_is_reported_not_read(synthetic):
    template, layout = synthetic
    page = _form_with_rows(template, layout, 5)
    a, b = int(layout.checklist_slots[7].y0) - 3, int(layout.checklist_slots[7].y1) + 3
    page[a:b] = template.gray[a:b]  # slot 8 drawn again
    rows, reasons = detect_rows(_aligned(page, template), layout, CFG)
    assert len(rows) == 5 and any("found again" in r for r in reasons)


# ---------------------------------------------------------------- header layout
def test_synthetic_template_without_boxes_has_no_header_layout(synthetic):
    template, layout = synthetic
    with pytest.raises(LayoutError):
        detect_header_layout(template, layout, SEMANTICS)


@pytest.mark.skipif(not REAL_TEMPLATE.exists(), reason="real template not present (git-ignored)")
def test_real_template_header_cells():
    template = load_template(REAL_TEMPLATE)
    layout = detect_template_layout(template, SEMANTICS)
    header = detect_header_layout(template, layout, SEMANTICS)
    assert len(header.text_cells) == 5 and len(header.location_boxes) == 10 and len(header.shift_boxes) == 2
    # reading order: location boxes go left to right in the first band, then the second band
    ys = [b.y0 for b in header.location_boxes]
    assert ys[:5] == sorted(ys[:5]) and len(set(ys[:5])) == 1 and ys[5] > ys[4]
    assert header.shift_boxes[0].x0 < header.shift_boxes[1].x0
    # boxes are roughly square
    for b in header.location_boxes + header.shift_boxes:
        assert 0.8 < (b.x1 - b.x0) / (b.y1 - b.y0) < 1.3
