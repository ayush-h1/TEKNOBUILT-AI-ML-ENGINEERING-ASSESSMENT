"""Template-level layout, derived from the template's own ruled lines.

No coordinate in this module is a constant. Everything is measured from the template image, and the
only inputs from configuration are COUNTS that come from the form semantics (how many choice
columns a checklist has, how many expectation questions exist). The number of checklist slots is
not an input: it is whatever the template contains.

This is the TEMPLATE layout. Per-page structure (how many rows a given page actually has, header
cell snapping, location and shift box mapping) is detected separately on each page in Stage 4b.3.

Failure is closed: if the template does not show the structure the semantics expect, `LayoutError`
is raised and nothing is guessed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import FormSemantics
from .template import Template


class LayoutError(Exception):
    """The template's structure could not be established."""


@dataclass(frozen=True)
class Band:
    y0: float
    y1: float

    @property
    def height(self) -> float:
        return self.y1 - self.y0


@dataclass(frozen=True)
class Cell:
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass(frozen=True)
class HeaderLayout:
    """Header cells in reading order. Text value cells map to `header_text_fields` in order; boxes map
    to the location options and then the shift options, in order."""

    text_cells: tuple[Cell, ...]
    location_boxes: tuple[Cell, ...]
    shift_boxes: tuple[Cell, ...]


@dataclass(frozen=True)
class TemplateLayout:
    width: int
    height: int
    header_rules: tuple[float, ...]        # y of long horizontal rules above the title bar
    title_bar: Band
    legend_row: Band                       # between the title bar and the first checklist rule
    checklist_slots: tuple[Band, ...]
    checklist_rules_x: tuple[float, ...]   # every long vertical rule across the checklist
    choice_columns_x: tuple[float, ...]    # the rules that bound the S/R/I/N cells (count + 1)
    cczjv_header: Band
    cczjv_rows: tuple[Band, ...]
    cczjv_rules_x: tuple[float, ...]       # the rules that bound the Yes/No/N-A cells (count + 1)
    comments: Band
    header: Optional[HeaderLayout] = None

    def stable_bands(self) -> tuple[Band, ...]:
        """Regions whose content does not depend on how many checklist rows a page has. Used to verify
        alignment independently of the checklist, which varies from form to form."""
        return (Band(0, self.title_bar.y0), Band(self.cczjv_header.y0, float(self.height)))


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    idx = np.flatnonzero(flags)
    runs: list[list[int]] = []
    for i in idx:
        if runs and i - runs[-1][1] <= 1:
            runs[-1][1] = int(i)
        else:
            runs.append([int(i), int(i)])
    return [(a, b) for a, b in runs]


def _bands_and_rules(profile: np.ndarray, min_coverage: float, max_rule: int) -> tuple[list[tuple[int, int]], list[float]]:
    """Split a 1-D coverage profile into thick bands and thin rules (rule centres)."""
    bands, rules = [], []
    for a, b in _runs(profile >= min_coverage):
        if b - a + 1 > max_rule:
            bands.append((a, b))
        else:
            rules.append((a + b) / 2.0)
    return bands, rules


def _uniform(values: list[float], tolerance: float) -> bool:
    gaps = np.diff(values)
    return bool(len(gaps) and np.all(gaps > 0) and np.all(np.abs(gaps / np.median(gaps) - 1.0) <= tolerance))


def detect_template_layout(template: Template, semantics: FormSemantics) -> TemplateLayout:
    h, w = template.gray.shape
    n_choice = len(semantics.checklist.options)
    n_expect_choice = len(semantics.expectation.options)
    n_questions = len(semantics.expectation.question_keys)
    if n_choice < 1 or n_expect_choice < 1 or n_questions < 1:
        raise LayoutError("the form semantics define no checklist options, expectation options or questions")

    max_rule = max(3, h // 200)
    min_band = max(4, h // 64)
    double_rule_gap = h / 125.0

    # --- horizontal structure -------------------------------------------------------------------
    h_cover = template.h_mask.sum(axis=1) / 255.0 / w
    bands, rules = _bands_and_rules(h_cover, 0.5, max_rule)
    bands = [(a, b) for a, b in bands if b - a + 1 >= min_band]
    if len(bands) != 1:
        raise LayoutError(f"expected exactly one title bar (a thick grey band), found {len(bands)}")
    title = Band(float(bands[0][0]), float(bands[0][1]) + 1.0)
    if not (0.05 * h < title.y0 < 0.5 * h):
        raise LayoutError("the title bar is not in the upper part of the template")

    header_rules = tuple(r for r in rules if r < title.y0)
    below = sorted(r for r in rules if r > title.y1)
    # A new section starts wherever two rules sit closer than a double-rule gap. The checklist table
    # and the expectation table are separated by such a double rule.
    sections: list[list[float]] = []
    for r in below:
        if sections and r - sections[-1][-1] <= double_rule_gap:
            sections.append([r])
        elif sections:
            sections[-1].append(r)
        else:
            sections.append([r])
    if len(sections) < 2:
        raise LayoutError("could not separate the checklist table from the expectation table")
    checklist_rules, expect_rules = sections[0], sections[1]

    if len(checklist_rules) < 2:
        raise LayoutError("fewer than two checklist rules were found")
    slot_heights = np.diff(checklist_rules)
    median_slot = float(np.median(slot_heights))
    if np.any(slot_heights < 0.5 * median_slot) or np.any(slot_heights > 1.6 * median_slot):
        raise LayoutError("checklist slot heights are not consistent; the table structure is unclear")
    slots = tuple(Band(a, b) for a, b in zip(checklist_rules[:-1], checklist_rules[1:]))
    legend = Band(title.y1, checklist_rules[0])
    if legend.height < 0.01 * h:
        raise LayoutError("no legend row was found between the title bar and the checklist")

    if len(expect_rules) < n_questions + 2:
        raise LayoutError(
            f"the expectation table has {max(len(expect_rules) - 2, 0)} question rows; the semantics expect {n_questions}")
    cczjv_header = Band(expect_rules[0], expect_rules[1])
    question_rows = tuple(Band(a, b) for a, b in zip(expect_rules[1:n_questions + 1], expect_rules[2:n_questions + 2]))
    comments_top = expect_rules[n_questions + 1]
    later = [r for r in below if r > comments_top + 0.02 * h]
    comments = Band(comments_top, later[0] if later else float(h))
    if comments.height < 0.02 * h:
        raise LayoutError("no comments area was found below the expectation questions")

    # --- vertical structure ---------------------------------------------------------------------
    max_vrule = max(3, w // 100)

    def vertical_rules(y0: float, y1: float) -> list[float]:
        a, b = int(round(y0)), int(round(y1))
        cover = template.v_mask[a:b].sum(axis=0) / 255.0 / max(b - a, 1)
        _, found = _bands_and_rules(cover, 0.8, max_vrule)
        return found

    checklist_x = vertical_rules(checklist_rules[0], checklist_rules[-1])
    if len(checklist_x) < n_choice + 1:
        raise LayoutError(f"found {len(checklist_x)} vertical rules across the checklist; need at least {n_choice + 1}")
    choice_x = checklist_x[-(n_choice + 1):]
    if not _uniform(choice_x, 0.2):
        raise LayoutError("the checklist choice columns are not evenly spaced")

    expect_x_all = vertical_rules(question_rows[0].y0, question_rows[-1].y1)
    if len(expect_x_all) < n_expect_choice + 1:
        raise LayoutError(
            f"found {len(expect_x_all)} vertical rules across the expectation rows; need at least {n_expect_choice + 1}")
    expect_x = expect_x_all[-(n_expect_choice + 1):]
    if not _uniform(expect_x, 0.2):
        raise LayoutError("the expectation choice columns are not evenly spaced")

    return TemplateLayout(
        width=w, height=h, header_rules=header_rules, title_bar=title, legend_row=legend,
        checklist_slots=slots, checklist_rules_x=tuple(checklist_x), choice_columns_x=tuple(choice_x),
        cczjv_header=cczjv_header, cczjv_rows=question_rows, cczjv_rules_x=tuple(expect_x), comments=comments,
    )


def detect_header_layout(template: Template, layout: TemplateLayout, semantics: FormSemantics) -> HeaderLayout:
    """Find the header's value cells and tick boxes from the template's own lines.

    Each band between two header rules is split into cells by the vertical rules that span it. A cell is
    a tick BOX when all four of its borders are dark, which is how the form draws boxes (the other cells
    have light grey borders). Bands that contain boxes provide the location boxes (in reading order) and
    then the shift boxes. Bands without boxes alternate label cell, value cell, label cell, value cell;
    their value cells are the handwritten header fields in reading order. Every count must match the
    semantics exactly, otherwise `LayoutError` is raised."""
    gray = template.gray
    # Box borders are near black (0 on the template); other cell borders are grey rules (about half the
    # paper level). A quarter of the paper level sits well between the two.
    dark_level = 0.25 * template.paper
    max_vrule = max(3, layout.width // 100)
    rules = layout.header_rules
    if len(rules) < 2:
        raise LayoutError("no header rows were found above the title bar")

    def column_is_dark(x: float, a: int, b: int) -> bool:
        xi = int(round(x))
        strip = gray[a:b, max(xi - 2, 0):xi + 3]
        return strip.size > 0 and float(np.median(strip.min(axis=1))) < dark_level

    def row_is_dark(y: float, x0: float, x1: float) -> bool:
        yi = int(round(y))
        strip = gray[max(yi - 2, 0):yi + 3, int(x0) + 3:int(x1) - 3]
        return strip.size > 0 and float(np.median(strip.min(axis=0))) < dark_level

    text_cells: list[Cell] = []
    boxes_per_band: list[list[Cell]] = []
    for y0, y1 in zip(rules[:-1], rules[1:]):
        a, b = int(round(y0)) + 3, int(round(y1)) - 3
        if b - a < 4:
            continue
        cover = template.v_mask[a:b].sum(axis=0) / 255.0 / (b - a)
        _, xs = _bands_and_rules(cover, 0.8, max_vrule)
        cells = [Cell(x0, y0, x1, y1) for x0, x1 in zip(xs[:-1], xs[1:])]
        boxes = [c for c in cells if column_is_dark(c.x0, a, b) and column_is_dark(c.x1, a, b)
                 and row_is_dark(c.y0, c.x0, c.x1) and row_is_dark(c.y1, c.x0, c.x1)]
        if boxes:
            boxes_per_band.append(boxes)
        else:
            text_cells += cells[1::2]  # label, value, label, value ...

    n_location, n_shift = len(semantics.location.options), len(semantics.shift.options)
    flat = [box for band in boxes_per_band for box in band]
    location, rest = flat[:n_location], flat[n_location:]
    if len(location) != n_location or len(rest) != n_shift:
        raise LayoutError(
            f"found {len(flat)} tick boxes in the header; the semantics expect {n_location} location "
            f"and {n_shift} shift boxes")
    used = 0
    for band in boxes_per_band:  # the location must end on a band boundary, so shift is its own row
        used += len(band)
        if used == n_location:
            break
    else:
        raise LayoutError("the location boxes and the shift boxes are not on separate header rows")
    n_text = len(semantics.header_text_fields)
    if len(text_cells) != n_text:
        raise LayoutError(f"found {len(text_cells)} header value cells; the semantics expect {n_text}")
    return HeaderLayout(tuple(text_cells), tuple(location), tuple(rest))
