"""Template loading and template-level layout detection.

Synthetic tests use code-drawn forms (see synthetic_form.py). The real-template tests are skipped
when the git-ignored template file is absent."""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
import pytest

from field_digitization.config import DEFAULT_SEMANTICS_PATH, load_semantics
from field_digitization.layout import LayoutError, detect_template_layout
from field_digitization.template import TemplateError, load_template, template_from_gray

import synthetic_form as sf

SEMANTICS = load_semantics(DEFAULT_SEMANTICS_PATH)
REAL_TEMPLATE = Path(__file__).resolve().parents[1] / "data" / "template" / "template.png"


def _layout(gray):
    return detect_template_layout(template_from_gray(gray), SEMANTICS)


# ---------------------------------------------------------------- template loading
def test_load_template_records_sha256(tmp_path):
    path = tmp_path / "t.png"
    cv2.imwrite(str(path), sf.make_template(6))
    template = load_template(path)
    assert template.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert template.gray.ndim == 2 and template.line_mask.shape == template.gray.shape


def test_load_template_missing_file(tmp_path):
    with pytest.raises(TemplateError, match="not found"):
        load_template(tmp_path / "nope.png")


def test_load_template_not_an_image(tmp_path):
    path = tmp_path / "bad.png"
    path.write_bytes(b"this is not an image")
    with pytest.raises(TemplateError, match="not a readable image"):
        load_template(path)


def test_template_too_small_or_blank():
    with pytest.raises(TemplateError, match="too small"):
        template_from_gray(np.zeros((50, 50), np.uint8))
    with pytest.raises(TemplateError, match="blank"):
        template_from_gray(np.full((600, 400), 255, np.uint8))
    with pytest.raises(TemplateError, match="8-bit"):
        template_from_gray(np.zeros((600, 400), np.uint16))


# ---------------------------------------------------------------- synthetic layout
@pytest.mark.parametrize("n_slots", [3, 6, 12, 15, 20])
def test_slot_count_is_measured_not_configured(n_slots):
    layout = _layout(sf.make_template(n_slots))
    assert len(layout.checklist_slots) == n_slots
    assert len(layout.cczjv_rows) == 4
    assert len(layout.choice_columns_x) == 5 and len(layout.cczjv_rules_x) == 4


def test_layout_is_ordered_and_consistent():
    layout = _layout(sf.make_template(10))
    assert layout.header_rules == tuple(sorted(layout.header_rules))
    order = [layout.title_bar.y1, layout.legend_row.y1, layout.checklist_slots[-1].y1,
             layout.cczjv_header.y1, layout.cczjv_rows[-1].y1, layout.comments.y1]
    assert order == sorted(order)
    for a, b in zip(layout.checklist_slots[:-1], layout.checklist_slots[1:]):
        assert a.y1 == b.y0
    assert layout.comments.y0 == layout.cczjv_rows[-1].y1


def test_layout_values_follow_the_drawing():
    gray = sf.make_template(12, w=1000)
    layout = _layout(gray)
    assert layout.choice_columns_x == pytest.approx((690, 765, 840, 915, 990), abs=1.5)
    assert layout.checklist_slots[0].height == pytest.approx(70, abs=1.5)
    assert layout.cczjv_header.height == pytest.approx(40, abs=1.5)
    assert layout.cczjv_rows[0].height == pytest.approx(70, abs=1.5)


def test_layout_does_not_depend_on_resolution():
    gray = sf.make_template(12)
    big = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    a, b = _layout(gray), _layout(big)
    assert (len(a.checklist_slots), len(a.cczjv_rows)) == (len(b.checklist_slots), len(b.cczjv_rows))
    assert b.choice_columns_x == pytest.approx(tuple(x * 1.5 for x in a.choice_columns_x), abs=3)


def test_double_rule_that_merges_at_low_resolution_fails_closed():
    """Limitation: the gap between the two tables must stay resolvable. When it does not, the layout is
    refused instead of guessed."""
    small = cv2.resize(sf.make_template(12), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    with pytest.raises(LayoutError):
        _layout(small)


def test_stable_bands_exclude_the_checklist():
    layout = _layout(sf.make_template(12))
    header, lower = layout.stable_bands()
    assert header.y1 <= layout.title_bar.y0
    assert lower.y0 >= layout.checklist_slots[-1].y1
    assert all(s.y0 >= header.y1 and s.y1 <= lower.y0 + 7 for s in layout.checklist_slots)  # bands never overlap the slots


# ---------------------------------------------------------------- fail closed
def test_no_title_bar_fails():
    with pytest.raises(LayoutError, match="title bar"):
        _layout(sf.make_template(8, title_bar=False))


def test_no_vertical_rules_fails():
    with pytest.raises(LayoutError, match="vertical rules"):
        _layout(sf.make_template(8, vertical_rules=False))


def test_no_double_rule_between_tables_fails():
    with pytest.raises(LayoutError):
        _layout(sf.make_template(8, double_rule=False))


def test_too_few_question_rows_fails():
    with pytest.raises(LayoutError, match="question rows"):
        _layout(sf.make_template(8, n_questions=2))


def test_missing_rule_makes_slot_heights_inconsistent():
    gray = sf.make_template(8)
    layout = _layout(gray)
    y = int(layout.checklist_slots[3].y1)
    gray[y - 3:y + 4, 70:700] = 255  # erase one slot divider (the choice columns keep their own rules)
    with pytest.raises(LayoutError, match="slot heights"):
        _layout(gray)


def test_semantics_that_expect_more_questions_fail():
    template = template_from_gray(sf.make_template(8))
    bigger = SEMANTICS.model_copy(deep=True)
    bigger.expectation.question_keys.append("extra_question")
    with pytest.raises(LayoutError, match="question rows|comments area"):
        detect_template_layout(template, bigger)


# ---------------------------------------------------------------- the real template (git-ignored)
@pytest.mark.skipif(not REAL_TEMPLATE.exists(), reason="real template not present (git-ignored)")
def test_real_template_structure():
    template = load_template(REAL_TEMPLATE)
    layout = detect_template_layout(template, SEMANTICS)
    assert (template.width, template.height) == (1200, 2550)
    assert len(layout.checklist_slots) == 15
    assert len(layout.cczjv_rows) == 4
    assert layout.choice_columns_x == pytest.approx((835, 924, 1013.5, 1102, 1188), abs=3)
    assert layout.cczjv_rules_x == pytest.approx((921.5, 1010.5, 1099.5, 1187), abs=3)
    assert layout.checklist_slots[0].y0 == pytest.approx(721, abs=3)
    assert layout.checklist_slots[-1].y1 == pytest.approx(1998.5, abs=3)
    assert layout.comments.y0 == pytest.approx(2384.5, abs=3)
