import cv2
import numpy as np
import pytest

from field_digitization.config import QualityConfig
from field_digitization.quality import assess_quality, compute_metrics
from field_digitization.schema import QualityVerdict

import synthetic


def test_normal_form_page_is_ok():
    result = assess_quality(synthetic.make_form_page())
    assert result.verdict is QualityVerdict.OK, result
    assert result.reasons == []


@pytest.mark.parametrize("angle", [-8.0, 5.0, 12.0])
def test_skewed_page_is_still_ok(angle):
    # Skew is corrected later by preprocessing, so the quality gate must not reject it.
    result = assess_quality(synthetic.make_form_page(skew_degrees=angle))
    assert result.verdict is QualityVerdict.OK, result


def test_blank_page_is_blank():
    result = assess_quality(synthetic.make_blank_page())
    assert result.verdict is QualityVerdict.BLANK
    assert result.metrics.content_components == 0


def test_all_black_page_is_unreadable_not_blank():
    result = assess_quality(np.zeros((1000, 800), dtype=np.uint8))
    assert result.verdict is QualityVerdict.UNREADABLE
    assert "page is too dark" in result.reasons


def test_dark_muddy_page_is_unreadable():
    result = assess_quality(synthetic.make_dark_muddy_page())
    assert result.verdict is QualityVerdict.UNREADABLE
    assert "page is too dark" in result.reasons


def test_smeared_page_is_unreadable():
    result = assess_quality(synthetic.make_smeared_page())
    assert result.verdict is QualityVerdict.UNREADABLE


def test_low_contrast_page_is_unreadable():
    result = assess_quality(synthetic.make_low_contrast_page())
    assert result.verdict is QualityVerdict.UNREADABLE
    assert "contrast is too low" in result.reasons


def test_blurry_page_is_unreadable():
    result = assess_quality(synthetic.make_blurry_page())
    assert result.verdict is QualityVerdict.UNREADABLE
    assert "page is out of focus" in result.reasons


def test_color_image_is_accepted():
    color = cv2.cvtColor(synthetic.make_form_page(), cv2.COLOR_GRAY2BGR)
    assert assess_quality(color).verdict is QualityVerdict.OK


@pytest.mark.parametrize("bad", [np.zeros((0, 0), dtype=np.uint8), np.zeros((4, 4, 4, 4))])
def test_invalid_image_raises_value_error(bad):
    with pytest.raises(ValueError):
        assess_quality(bad)


def test_thresholds_are_configurable():
    strict = QualityConfig(min_sharpness=1e9)
    assert assess_quality(synthetic.make_form_page(), strict).verdict is QualityVerdict.UNREADABLE


# --- B: image data validation -------------------------------------------------------

def test_uint16_image_is_scaled_not_clipped():
    form8 = synthetic.make_form_page()
    form16 = form8.astype(np.uint16) * 257  # exact 8-bit to 16-bit scaling
    result = assess_quality(form16)
    # Silent clipping used to turn every 16-bit pixel into 255, i.e. an all-white "blank" page.
    assert result.verdict is QualityVerdict.OK
    expected = assess_quality(form8).metrics
    assert result.metrics.content_components == expected.content_components
    assert result.metrics.paper_level == pytest.approx(expected.paper_level, abs=1)


def test_dark_uint16_image_is_not_clipped_to_white():
    dark16 = np.full((1000, 800), 100 * 257, dtype=np.uint16)
    result = assess_quality(dark16)
    assert result.verdict is QualityVerdict.UNREADABLE  # it is a grey page, not a blank white one
    assert result.metrics.paper_level == pytest.approx(100, abs=1)


@pytest.mark.parametrize(
    "dtype", [np.float32, np.float64, np.int8, np.int16, np.int32, np.uint32, np.bool_]
)
def test_unsupported_dtypes_are_rejected(dtype):
    with pytest.raises(ValueError, match="unsupported image dtype"):
        assess_quality(np.ones((50, 50), dtype=dtype))


def test_four_channel_image_is_accepted_and_five_channel_rejected():
    form = synthetic.make_form_page()
    bgra = cv2.cvtColor(form, cv2.COLOR_GRAY2BGRA)
    assert assess_quality(bgra).verdict is QualityVerdict.OK
    with pytest.raises(ValueError, match="unsupported image shape"):
        assess_quality(np.zeros((50, 50, 5), dtype=np.uint8))


# --- C: blank versus degraded or sparse pages ---------------------------------------

@pytest.mark.parametrize("level", [60, 120, 160])
def test_uniform_grey_scan_is_unreadable_not_blank(level):
    rng = np.random.default_rng(level)
    page = np.clip(level + rng.normal(0, 2, (2339, 1654)), 0, 255).astype(np.uint8)
    result = assess_quality(page)
    assert result.verdict is QualityVerdict.UNREADABLE
    assert any("not paper-like" in reason or "too dark" in reason for reason in result.reasons)


def test_clean_light_paper_is_still_blank():
    rng = np.random.default_rng(1)
    page = np.clip(200 + rng.normal(0, 2, (2339, 1654)), 0, 255).astype(np.uint8)
    assert assess_quality(page).verdict is QualityVerdict.BLANK


def test_dust_specks_do_not_prevent_a_blank_verdict():
    page = synthetic.make_blank_page().copy()
    rng = np.random.default_rng(3)
    for _ in range(300):
        center = (int(rng.integers(0, 1654)), int(rng.integers(0, 2339)))
        cv2.circle(page, center, 3, 90, -1)
    result = assess_quality(page)
    assert result.verdict is QualityVerdict.BLANK
    assert result.metrics.mark_ratio > 0.002  # ink area alone would have said "not blank"


@pytest.mark.parametrize("text, scale, thickness", [("Hi", 1.5, 3), ("OK", 1.0, 2), ("OK", 0.8, 2)])
def test_sparse_handwriting_is_not_lost_as_blank(text, scale, thickness):
    page = synthetic.make_blank_page().copy()
    cv2.putText(page, text, (600, 900), cv2.FONT_HERSHEY_SIMPLEX, scale, 40, thickness)
    result = assess_quality(page)
    assert result.verdict is QualityVerdict.OK, result  # not BLANK, and not rejected for sparseness
    assert result.metrics.content_components >= 1
    assert result.metrics.contrast > 100 and result.metrics.sharpness > 100


def test_a_thin_line_is_treated_as_content_not_ignored():
    # A fibre or hairline is long and thin. Treating it as content errs towards keeping data.
    page = synthetic.make_blank_page().copy()
    cv2.line(page, (300, 300), (340, 330), 120, 1)
    assert assess_quality(page).verdict is not QualityVerdict.BLANK


# --- D: noise detection -------------------------------------------------------------

def _with_gaussian_noise(page, sigma, seed=0):
    rng = np.random.default_rng(seed)
    return np.clip(page.astype(np.float32) + rng.normal(0, sigma, page.shape), 0, 255).astype(np.uint8)


def _with_salt_and_pepper(page, fraction, seed=0):
    rng = np.random.default_rng(seed)
    noisy = page.copy()
    draw = rng.random(page.shape)
    noisy[draw < fraction / 2] = 0
    noisy[draw > 1 - fraction / 2] = 255
    return noisy


def test_printed_form_structure_is_not_mistaken_for_noise():
    result = assess_quality(synthetic.make_form_page())
    assert result.verdict is QualityVerdict.OK
    assert result.metrics.noise_sigma < 5
    assert result.metrics.speckle_ratio < 0.001


@pytest.mark.parametrize("true_sigma", [5, 10, 20])
def test_noise_estimator_tracks_the_true_gaussian_sigma(true_sigma):
    blank = _with_gaussian_noise(np.full((1200, 900), 240, dtype=np.uint8), true_sigma)
    estimated = compute_metrics(blank, QualityConfig()).noise_sigma
    assert estimated == pytest.approx(true_sigma, rel=0.2)


@pytest.mark.parametrize("added_sigma", [4, 8])
def test_noise_estimate_is_the_same_with_and_without_printed_structure(added_sigma):
    # Same added Gaussian noise (same seed, same sigma) on a flat page and on a page with
    # printed lines, text and boxes. The printed edges must not inflate the estimate.
    config = QualityConfig()
    flat = np.full((2339, 1654), 245, dtype=np.uint8)
    structured = synthetic.make_form_page(seed=9, noise_sigma=0)

    sigma_flat = compute_metrics(_with_gaussian_noise(flat, added_sigma, seed=11), config).noise_sigma
    sigma_structured = compute_metrics(
        _with_gaussian_noise(structured, added_sigma, seed=11), config
    ).noise_sigma

    assert sigma_flat == pytest.approx(added_sigma, rel=0.2)  # the estimator is calibrated
    assert sigma_structured == pytest.approx(sigma_flat, rel=0.15)  # edges do not move it


def test_printed_structure_alone_contributes_almost_no_noise_estimate():
    structured = synthetic.make_form_page(seed=9, noise_sigma=0)
    assert compute_metrics(structured, QualityConfig()).noise_sigma < 1.5


def test_one_pixel_wide_printed_lines_are_not_counted_as_speckle():
    # At low render DPI the form's ruled lines shrink to one pixel. A 3x3 median erases such
    # lines, so a naive speckle count flagged this clean page as noisy (ratio ~0.0076).
    page = synthetic.make_small_form_page()
    result = assess_quality(page)
    assert result.metrics.speckle_ratio < 0.001
    assert result.verdict is QualityVerdict.OK


@pytest.mark.parametrize("sigma", [25, 40])
def test_heavy_gaussian_noise_is_unreadable(sigma):
    result = assess_quality(_with_gaussian_noise(synthetic.make_form_page(), sigma))
    assert result.verdict is QualityVerdict.UNREADABLE
    assert "page is too noisy" in result.reasons


def test_moderate_gaussian_noise_still_passes():
    assert assess_quality(_with_gaussian_noise(synthetic.make_form_page(), 10)).verdict is QualityVerdict.OK


@pytest.mark.parametrize("fraction", [0.03, 0.06])
def test_salt_and_pepper_noise_is_unreadable(fraction):
    # The median-based Gaussian estimator is blind to impulses, so this needs the speckle metric.
    noisy = _with_salt_and_pepper(synthetic.make_form_page(), fraction)
    result = assess_quality(noisy)
    assert result.metrics.noise_sigma < 15
    assert result.metrics.speckle_ratio > QualityConfig().max_speckle_ratio
    assert result.verdict is QualityVerdict.UNREADABLE
    assert "page is too noisy" in result.reasons


def test_noisy_empty_page_is_not_confirmed_blank():
    noisy = _with_salt_and_pepper(synthetic.make_blank_page(), 0.06)
    result = assess_quality(noisy)
    assert result.verdict is QualityVerdict.UNREADABLE
    assert any("too noisy" in reason for reason in result.reasons)


def test_noise_thresholds_are_configurable():
    page = synthetic.make_form_page()
    assert assess_quality(page, QualityConfig(max_noise_sigma=1.0)).verdict is QualityVerdict.UNREADABLE
    assert assess_quality(page, QualityConfig(max_speckle_ratio=1e-6)).verdict is QualityVerdict.UNREADABLE
