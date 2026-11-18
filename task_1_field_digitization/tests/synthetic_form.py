"""SYNTHETIC form and template generator for layout and alignment tests.

Everything here is drawn by code. It mimics the STRUCTURE of the supplied form (header rules, a
grey title bar, a legend row, a checklist table with a variable number of slots, a double rule, an
expectation table with a header row and four question rows, and a comments box) but it is not the
real form, and results on it say nothing about accuracy on real scans.
"""

from __future__ import annotations

import cv2
import numpy as np

RULE = 110
WORDS = ["Observation", "Observer", "Company", "Location", "Shift", "Timesheet", "Employee", "Safe", "Risk",
         "Hazard", "Permit", "Barrier", "Supervisor", "Inspection", "Isolation", "Procedure", "Feedback",
         "Stopped", "Investigation", "Corrective", "Actions", "Comments", "Equipment", "Contractor"]


def _text(img, rng, x0, y0, x1, y1, n=2):
    for _ in range(n):
        word = WORDS[int(rng.integers(len(WORDS)))] + " " + WORDS[int(rng.integers(len(WORDS)))]
        x = int(rng.integers(x0, max(x0 + 1, x1 - 190)))
        y = int(rng.integers(y0 + 18, max(y0 + 19, y1 - 6)))
        cv2.putText(img, word, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55 + 0.1 * float(rng.random()), 60, 1, cv2.LINE_AA)


def make_template(n_slots: int = 12, w: int = 1000, seed: int = 0, n_questions: int = 4,
                  title_bar: bool = True, vertical_rules: bool = True, double_rule: bool = True) -> np.ndarray:
    rng = np.random.default_rng(seed)
    slot, q_row = 70, 70
    header_y = [10 + 60 * k for k in range(6)]
    title_y0 = header_y[-1] + 20
    title_y1 = title_y0 + 70
    legend_y1 = title_y1 + 70
    slots_y = [legend_y1 + slot * k for k in range(n_slots + 1)]
    gap = 6 if double_rule else 70
    cz_y0 = slots_y[-1] + gap
    cz_rules = [cz_y0, cz_y0 + 40] + [cz_y0 + 40 + q_row * k for k in range(1, n_questions + 1)]
    comments_y1 = cz_rules[-1] + 140
    h = comments_y1 + 10
    img = np.full((h, w), 255, np.uint8)

    def hline(y, x0=10, x1=None):
        cv2.line(img, (x0, y), (x1 or w - 10, y), RULE, 3)

    def vline(x, y0, y1):
        cv2.line(img, (x, y0), (x, y1), RULE, 3)

    for y in header_y:
        hline(y)
    if title_bar:
        img[title_y0:title_y1, 10:w - 10] = 207
    for y in slots_y:
        hline(y)
    for y in cz_rules:
        hline(y)
    hline(comments_y1 - 5)
    if vertical_rules:
        for x in (10, 60, w - 310, w - 235, w - 160, w - 85, w - 10):
            vline(x, slots_y[0], slots_y[-1])
        for x in (w - 235, w - 160, w - 85, w - 10):
            vline(x, cz_rules[1], cz_rules[-1])
        for x in (10, 300, 600, w - 10):
            vline(x, header_y[0], header_y[-1])
    # texture so that feature matching has something to hold on to
    for a, b in zip(header_y[:-1], header_y[1:]):
        _text(img, rng, 20, a, w - 20, b, 2)
    _text(img, rng, 20, title_y0, w - 20, title_y1, 1)
    _text(img, rng, 20, title_y1, w - 20, legend_y1, 2)
    for a, b in zip(cz_rules[:-1], cz_rules[1:]):
        _text(img, rng, 20, a, w - 250, b, 2)
    _text(img, rng, 20, cz_rules[-1], w - 20, comments_y1, 2)
    cv2.circle(img, (w - 120, 40), 28, 90, 3)  # a logo-like blob
    return img


def fit_to_canvas(template_shape: tuple[int, int], forward: np.ndarray | None, margin: int = 30):
    """Shift `forward` so the whole warped template lands inside the page with a margin.
    Returns (effective template -> page matrix, page size (w, h))."""
    h, w = template_shape
    m = np.eye(3) if forward is None else np.asarray(forward, np.float64)
    corners = cv2.perspectiveTransform(np.float64([[0, 0], [w, 0], [0, h], [w, h]]).reshape(-1, 1, 2), m).reshape(-1, 2)
    shift = np.array([[1, 0, margin - corners[:, 0].min()], [0, 1, margin - corners[:, 1].min()], [0, 0, 1.0]])
    m = shift @ m
    span = corners.max(axis=0) - corners.min(axis=0)
    return m, (int(np.ceil(span[0])) + 2 * margin, int(np.ceil(span[1])) + 2 * margin)


def make_page(template: np.ndarray, forward: np.ndarray | None = None, keep_slots: int | None = None,
              seed: int = 0, noise: float = 2.0, slot_bounds: tuple[int, int, int] | None = None):
    """Warp `template` into a page that contains the whole form. Returns (page, effective forward matrix).
    With `keep_slots` and `slot_bounds` (checklist top y, slot height, slot count), checklist slots past
    that many are painted out, which imitates a form that has fewer rows than the template."""
    src = template.copy()
    if keep_slots is not None and slot_bounds is not None:
        top, height, count = slot_bounds
        src[top + keep_slots * height + 4: top + count * height - 4, 70:] = 255  # stable bands untouched
    m, size = fit_to_canvas(template.shape, forward)
    page = cv2.warpPerspective(src, m, size, flags=cv2.INTER_LINEAR, borderValue=255)
    rng = np.random.default_rng(seed)
    page = np.clip(page.astype(np.float32) + rng.normal(0, noise, page.shape), 0, 255).astype(np.uint8)
    return page, m


def affine(scale_x=1.0, scale_y=1.0, rotation=0.0, tx=0.0, ty=0.0, shear_deg=0.0, centre=(500, 1000)) -> np.ndarray:
    cx, cy = centre
    t1 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1.0]])
    r = np.radians(rotation)
    rot = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1.0]])
    sh = np.array([[1, np.tan(np.radians(shear_deg)), 0], [0, 1, 0], [0, 0, 1.0]])
    sc = np.diag([scale_x, scale_y, 1.0])
    t2 = np.array([[1, 0, cx + tx], [0, 1, cy + ty], [0, 0, 1.0]])
    return t2 @ rot @ sh @ sc @ t1


def corner_error(returned: np.ndarray, forward: np.ndarray, shape: tuple[int, int]) -> float:
    """Largest distance, in template pixels, between where the returned page->template matrix and the true
    inverse of `forward` send the template's corners (after going to the page and back)."""
    h, w = shape
    forward = np.eye(3) if forward is None else forward
    corners = np.float64([[0, 0], [w, 0], [0, h], [w, h]]).reshape(-1, 1, 2)
    on_page = cv2.perspectiveTransform(corners, forward)
    back = cv2.perspectiveTransform(on_page, np.asarray(returned, np.float64))
    return float(np.linalg.norm(back - corners, axis=2).max())
