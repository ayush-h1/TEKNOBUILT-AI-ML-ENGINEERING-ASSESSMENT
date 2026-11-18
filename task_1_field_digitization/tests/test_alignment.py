"""Page-to-template alignment.

Synthetic cases (code-drawn forms, see synthetic_form.py) test the mechanism and its gates. They are
NOT evidence of accuracy on real scans. The real-sample tests are skipped when the git-ignored files
are absent and check only that the supplied sample behaves as measured in Stage 4b.1."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pymupdf
import pytest

from field_digitization import alignment as alignment_module
from field_digitization.alignment import align_page
from field_digitization.config import AlignmentConfig, DEFAULT_SEMANTICS_PATH, load_semantics
from field_digitization.layout import detect_template_layout
from field_digitization.schema import AlignmentMethod
from field_digitization.template import load_template, template_from_gray

import synthetic_form as sf

SEMANTICS = load_semantics(DEFAULT_SEMANTICS_PATH)
DATA = Path(__file__).resolve().parents[1] / "data"
REAL_TEMPLATE = DATA / "template" / "template.png"
REAL_PDF = DATA / "input" / "Sample_input.pdf"


@pytest.fixture(scope="module")
def tmpl():
    template = template_from_gray(sf.make_template(12))
    return template, detect_template_layout(template, SEMANTICS)


def _align(tmpl, forward, cfg=None, **page_kw):
    """Returns (AlignmentInfo, effective template -> page matrix)."""
    template, layout = tmpl
    page, effective = sf.make_page(template.gray, forward, **page_kw)
    return align_page(page, template, layout, cfg or AlignmentConfig()), effective


def _err(info, effective, tmpl):
    return sf.corner_error(info.matrix, effective, tmpl[0].gray.shape)


# ---------------------------------------------------------------- accepted
def test_identity_page_is_accepted_and_recovered(tmpl):
    info, effective = _align(tmpl, None)
    assert info.accepted and info.method is AlignmentMethod.SIFT_AFFINE
    assert _err(info, effective, tmpl) < 0.5
    assert info.verification_chamfer_mean_px < 0.5 and info.verification_within_2px_fraction > 0.97


@pytest.mark.parametrize("params", [
    dict(scale_x=0.8, scale_y=1.1, rotation=3),
    dict(scale_x=0.725, scale_y=1.09, rotation=0),   # the stretch measured on the supplied sample
    dict(scale_x=1.2, scale_y=1.2, rotation=-5),
    dict(scale_x=0.8, scale_y=1.1, rotation=12),
    dict(scale_x=0.6, scale_y=0.6, rotation=1),
])
def test_affine_pages_are_recovered_within_a_few_pixels(tmpl, params):
    info, effective = _align(tmpl, sf.affine(**params))
    assert info.accepted, info.reasons
    assert info.method is AlignmentMethod.SIFT_AFFINE  # homography is not used when affine passes
    assert _err(info, effective, tmpl) < 3.0
    assert info.inliers >= 12 and info.inlier_spread_x >= 0.35 and info.inlier_spread_y >= 0.35


def test_sideways_scan_of_the_same_form_is_accepted(tmpl):
    info, effective = _align(tmpl, sf.affine(rotation=90))
    assert info.accepted, info.reasons
    assert abs(abs(info.rotation_degrees) - 90) < 1 and _err(info, effective, tmpl) < 3.0


def test_page_with_fewer_checklist_rows_than_the_template_is_still_aligned(tmpl):
    template, layout = tmpl
    bounds = (int(layout.checklist_slots[0].y0), int(round(layout.checklist_slots[0].height)), len(layout.checklist_slots))
    info, effective = _align(tmpl, sf.affine(0.8, 1.1, 1), keep_slots=6, slot_bounds=bounds)
    assert info.accepted, info.reasons
    assert _err(info, effective, tmpl) < 3.0


def test_mild_perspective_uses_homography_only_because_affine_fails(tmpl):
    forward = sf.affine(0.9, 1.0, 2)
    forward[2, 0], forward[2, 1] = 4e-5, 2e-5
    info, effective = _align(tmpl, forward)
    assert info.accepted, info.reasons
    assert info.method is AlignmentMethod.SIFT_HOMOGRAPHY
    assert info.perspective_magnitude > 0
    assert _err(info, effective, tmpl) < 3.0
    no_h, _ = _align(tmpl, forward, AlignmentConfig(allow_homography=False))
    assert not no_h.accepted and no_h.reasons


# ---------------------------------------------------------------- rejected, fail closed
def test_non_form_pages_are_rejected_without_raising(tmpl):
    template, layout = tmpl
    rng = np.random.default_rng(1)
    for page in (rng.integers(0, 255, (1500, 1000)).astype(np.uint8), np.full((1500, 1000), 255, np.uint8)):
        info = align_page(page, template, layout, AlignmentConfig())
        assert not info.accepted and info.reasons and info.matrix is None


def test_a_page_with_rearranged_structure_is_rejected(tmpl):
    """Same printed content, different arrangement: the lower half is moved to the top. Features match
    locally but no single plausible transform explains the page."""
    template, layout = tmpl
    gray = template.gray
    cut = int(layout.checklist_slots[5].y0)
    rearranged = np.vstack([gray[cut:], gray[:cut]])
    info = align_page(rearranged, template, layout, AlignmentConfig())
    assert not info.accepted and info.reasons


def test_mirrored_page_is_rejected(tmpl):
    template, layout = tmpl
    info = align_page(np.ascontiguousarray(template.gray[:, ::-1]), template, layout, AlignmentConfig())
    assert not info.accepted


@pytest.mark.parametrize("bad", [None, np.zeros((0, 0), np.uint8), np.zeros((50, 50), np.uint8),
                                 np.zeros((600, 400), np.uint16)])
def test_invalid_inputs_are_rejected_not_raised(tmpl, bad):
    info = align_page(bad, *tmpl, AlignmentConfig())
    assert not info.accepted and info.reasons


def test_strong_perspective_is_rejected(tmpl):
    forward = sf.affine(0.9, 1.0, 2)
    forward[2, 0], forward[2, 1] = 3e-4, 2e-4
    info, _ = _align(tmpl, forward)
    assert not info.accepted


# ---------------------------------------------------------------- each gate really gates
def test_shear_gate(tmpl):
    forward = sf.affine(0.8, 1.1, 0, shear_deg=10)
    assert _align(tmpl, forward)[0].accepted
    info, _ = _align(tmpl, forward, AlignmentConfig(max_shear_degrees=5))
    assert not info.accepted and any("shear" in r for r in info.reasons)


def test_scale_gate(tmpl):
    info, _ = _align(tmpl, sf.affine(0.8, 1.1, 0), AlignmentConfig(min_scale=0.95, max_scale=3.0))  # page->template y scale is 1/1.1 = 0.909
    assert not info.accepted and any("scale" in r for r in info.reasons)


def test_inlier_count_gate(tmpl):
    info, _ = _align(tmpl, None, AlignmentConfig(min_inliers=100000, min_matches=6))
    assert not info.accepted and any("inliers" in r for r in info.reasons)


def test_residual_gate(tmpl):
    info, _ = _align(tmpl, sf.affine(0.8, 1.1, 3), AlignmentConfig(max_median_residual_px=0.01))
    assert not info.accepted and any("residual" in r for r in info.reasons)


def test_independent_line_check_can_veto_a_good_feature_fit(tmpl):
    """Strong feature agreement is not enough: the ruled lines must also coincide. Here the page's
    lower table rules are shifted by a few pixels while the text stays put."""
    template, layout = tmpl
    page = template.gray.copy()
    y0, y1 = int(layout.cczjv_header.y0) - 4, int(layout.comments.y1) + 4
    page[y0:y1] = np.maximum(page[y0:y1], 255 - template.line_mask[y0:y1])  # remove the rules there
    shifted = np.roll(template.line_mask[y0:y1], 6, axis=0)
    page[y0:y1] = np.minimum(page[y0:y1], np.where(shifted > 0, 110, 255).astype(np.uint8))
    info = align_page(page, template, layout, AlignmentConfig())
    assert not info.accepted and any("independent line check" in r for r in info.reasons)


# ---------------------------------------------------------------- detectors
def _count_orb(monkeypatch):
    calls = []
    real = cv2.ORB_create
    monkeypatch.setattr(alignment_module.cv2, "ORB_create", lambda *a, **k: calls.append(1) or real(*a, **k))
    return calls


def test_orb_is_never_used_by_default(tmpl, monkeypatch):
    calls = _count_orb(monkeypatch)
    rng = np.random.default_rng(2)
    info = align_page(rng.integers(0, 255, (1500, 1000)).astype(np.uint8), *tmpl, AlignmentConfig())
    assert not info.accepted and calls == []
    assert AlignmentConfig().detectors == ["sift"]


def test_orb_used_only_when_listed_and_held_to_stricter_gates(tmpl, monkeypatch):
    calls = _count_orb(monkeypatch)
    info, effective = _align(tmpl, sf.affine(0.8, 1.1, 3), AlignmentConfig(detectors=["orb"]))
    assert calls, "ORB was listed, so it should have been created"
    if info.accepted:
        assert info.method in (AlignmentMethod.ORB_AFFINE, AlignmentMethod.ORB_HOMOGRAPHY)
        assert _err(info, effective, tmpl) < 3.0
        assert info.verification_chamfer_mean_px <= AlignmentConfig().verification_max_mean_px * AlignmentConfig().orb_strictness
    else:
        assert all(r.startswith("orb:") for r in info.reasons)


def test_orb_is_not_an_automatic_rescue_when_sift_fails(tmpl, monkeypatch):
    calls = _count_orb(monkeypatch)
    info, _ = _align(tmpl, sf.affine(0.8, 1.1, 0, shear_deg=10), AlignmentConfig(max_shear_degrees=5))
    assert not info.accepted and calls == []


def test_config_rejects_duplicate_detectors():
    with pytest.raises(ValueError):
        AlignmentConfig(detectors=["sift", "sift"])


# ---------------------------------------------------------------- determinism
def test_repeated_runs_give_identical_results(tmpl):
    results = {_align(tmpl, sf.affine(0.8, 1.1, 3))[0].model_dump_json() for _ in range(3)}
    assert len(results) == 1


def test_result_is_json_serialisable_and_records_opencv_version(tmpl):
    info, _ = _align(tmpl, None)
    assert json.loads(info.model_dump_json())["opencv_version"] == cv2.__version__


def test_ransac_repeatability_regression():
    """On the tested OpenCV the RANSAC fits were repeatable with and without seeding. This pins that
    behaviour so a different OpenCV build that changes it is noticed."""
    rng = np.random.default_rng(0)
    src = rng.uniform(0, 1000, (200, 2)).astype(np.float32)
    true = np.array([[0.9, 0.05, 12.0], [-0.04, 1.1, -7.0]])
    dst = (src @ true[:, :2].T + true[:, 2]).astype(np.float32)
    dst += rng.normal(0, 0.3, dst.shape).astype(np.float32)
    dst[:60] = rng.uniform(0, 1000, (60, 2)).astype(np.float32)  # 30 percent outliers
    fits = []
    for seed in (0, 0, 1, 99, None):
        if seed is not None:
            cv2.setRNGSeed(seed)
        fits.append(cv2.estimateAffine2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)[0].round(6).tolist())
    assert all(f == fits[0] for f in fits)


# ---------------------------------------------------------------- real sample (git-ignored inputs)
needs_real = pytest.mark.skipif(not (REAL_TEMPLATE.exists() and REAL_PDF.exists()),
                                reason="real template and sample PDF are not present (git-ignored)")


@pytest.fixture(scope="module")
def real():
    template = load_template(REAL_TEMPLATE)
    layout = detect_template_layout(template, SEMANTICS)
    pages = []
    with pymupdf.open(REAL_PDF) as doc:
        for page in doc:
            pix = page.get_pixmap(dpi=200, colorspace=pymupdf.csGRAY)
            pages.append(np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width).copy())
    return template, layout, pages


@needs_real
def test_real_pages_1_to_3_are_accepted_by_sift_affine(real):
    template, layout, pages = real
    assert len(pages) == 4
    for page in pages[:3]:
        info = align_page(page, template, layout, AlignmentConfig())
        assert info.accepted, info.reasons
        assert info.method is AlignmentMethod.SIFT_AFFINE
        assert info.inliers >= 40
        assert info.verification_chamfer_mean_px < 1.0 and info.verification_within_2px_fraction > 0.95
        assert 0.6 < info.scale_x < 0.9 and 0.9 < info.scale_y < 1.3  # the sample is anisotropically stretched


@needs_real
def test_real_page_4_the_failed_test_form_is_not_aligned(real):
    template, layout, pages = real
    info = align_page(pages[3], template, layout, AlignmentConfig())
    assert not info.accepted and info.matrix is None and info.reasons


@needs_real
def test_real_alignment_is_repeatable_in_process(real):
    template, layout, pages = real
    results = {align_page(pages[0], template, layout, AlignmentConfig()).model_dump_json() for _ in range(3)}
    assert len(results) == 1
