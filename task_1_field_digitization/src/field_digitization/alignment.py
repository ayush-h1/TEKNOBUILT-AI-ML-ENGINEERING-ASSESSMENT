"""Register a page to the template, and refuse when the evidence is not good enough.

Method (chosen from measurements on the supplied sample, not assumed):
  * SIFT features + ratio test + RANSAC AFFINE fit (6 degrees of freedom). A similarity model fails
    because the sample pages are anisotropically stretched.
  * Homography is tried only when the affine fit fails verification, and is kept only when it is
    clearly better. ORB + homography was unstable on the sample.
  * ORB is an opt-in alternative detector (list it in `alignment.detectors`). It is never an automatic
    rescue, and it must pass the same gates with stricter margins (`orb_strictness`).

A fit is accepted only if ALL of these hold: enough matches and inliers, inliers spread over the
template, small residuals, a plausible scale, shear and perspective, a positive determinant (no
reflection), AND an independent check: the warped page's ruled lines must coincide with the
template's ruled lines in the regions that do not depend on the number of checklist rows. The
independent check does not use the feature matches, so a lucky feature fit cannot pass on its own.

`align_page` never raises. Every failure is returned as a rejected `AlignmentInfo` with reasons.
Randomness: RANSAC is seeded (`cv2.setRNGSeed`) as a cheap guard; on the tested OpenCV it was
repeatable regardless, and a test pins that down.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np

from .config import AlignmentConfig
from .layout import TemplateLayout
from .schema import AlignmentInfo, AlignmentMethod
from .template import Template, line_masks, paper_level

_METHODS = {
    ("sift", "affine"): AlignmentMethod.SIFT_AFFINE,
    ("sift", "homography"): AlignmentMethod.SIFT_HOMOGRAPHY,
    ("orb", "affine"): AlignmentMethod.ORB_AFFINE,
    ("orb", "homography"): AlignmentMethod.ORB_HOMOGRAPHY,
}


@dataclass
class _Fit:
    kind: str                      # 'affine' or 'homography'
    matrix: np.ndarray             # 3x3, page -> template
    inlier_mask: np.ndarray
    page_pts: np.ndarray
    tmpl_pts: np.ndarray
    metrics: dict[str, Any]
    reasons: list[str]
    verification: Optional[tuple[float, float]] = None


def _detector(name: str, max_features: int):
    if name == "sift":
        return cv2.SIFT_create(nfeatures=max_features), cv2.NORM_L2
    if name == "orb":
        return cv2.ORB_create(nfeatures=max_features), cv2.NORM_HAMMING
    raise ValueError(f"unknown detector '{name}'")


def _features(gray: np.ndarray, name: str, cfg: AlignmentConfig):
    det, norm = _detector(name, cfg.max_features)
    keypoints, descriptors = det.detectAndCompute(gray, None)
    return keypoints, descriptors, norm


def _template_features(template: Template, name: str, cfg: AlignmentConfig):
    key = f"{name}:{cfg.max_features}"
    if key not in template.feature_cache:
        template.feature_cache[key] = _features(template.gray, name, cfg)
    return template.feature_cache[key]


def _matches(page_desc, tmpl_desc, norm: int, ratio: float):
    if page_desc is None or tmpl_desc is None or len(page_desc) < 2 or len(tmpl_desc) < 2:
        return []
    pairs = cv2.BFMatcher(norm).knnMatch(page_desc, tmpl_desc, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < ratio * p[1].distance]
    return sorted(good, key=lambda m: (m.queryIdx, m.trainIdx))  # a fixed order keeps RANSAC reproducible


def _decompose(matrix: np.ndarray) -> dict[str, float]:
    a, b, c, d = matrix[0, 0], matrix[0, 1], matrix[1, 0], matrix[1, 1]
    sx = math.hypot(a, c)
    col2 = math.hypot(b, d)
    det = a * d - b * c
    sy = det / sx if sx > 0 else 0.0
    cos_angle = (a * b + c * d) / (sx * col2) if sx > 0 and col2 > 0 else 0.0
    shear = abs(90.0 - math.degrees(math.acos(max(-1.0, min(1.0, cos_angle)))))
    return {"scale_x": sx, "scale_y": sy, "rotation": math.degrees(math.atan2(c, a)), "shear": shear, "det": det}


def _perspective(matrix: np.ndarray, page_shape: tuple[int, int]) -> float:
    """Largest relative departure of the homogeneous weight from 1 over the page corners."""
    h, w = page_shape
    weights = [matrix[2, 0] * x + matrix[2, 1] * y + matrix[2, 2] for x, y in ((0, 0), (w, 0), (0, h), (w, h))]
    base = matrix[2, 2] if matrix[2, 2] != 0 else 1.0
    return float(max(abs(wt / base - 1.0) for wt in weights))


def _project(matrix: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float64), matrix).reshape(-1, 2)


def _gate_scale(cfg: AlignmentConfig, strict: float) -> dict[str, float]:
    """Thresholds, tightened for ORB by `strict` (1.0 for SIFT, orb_strictness < 1 for ORB)."""
    return {
        "min_matches": cfg.min_matches / strict,
        "min_inliers": cfg.min_inliers / strict,
        "max_residual": cfg.max_median_residual_px * strict,
        "verify_mean": cfg.verification_max_mean_px * strict,
        "verify_frac": 1.0 - (1.0 - cfg.verification_min_within_2px) * strict,
    }


def _evaluate(kind: str, matrix: Optional[np.ndarray], inliers: Optional[np.ndarray], page_pts, tmpl_pts,
              matches: int, template: Template, page_shape, cfg: AlignmentConfig, strict: float) -> Optional[_Fit]:
    if matrix is None or inliers is None:
        return None
    mask = inliers.ravel().astype(bool)
    n_in = int(mask.sum())
    gates = _gate_scale(cfg, strict)
    reasons: list[str] = []
    metrics: dict[str, Any] = {"inliers": n_in, "matches": matches}
    metrics["inlier_ratio"] = n_in / matches if matches else 0.0
    if n_in:
        projected = _project(matrix, page_pts[mask])
        residuals = np.linalg.norm(projected - tmpl_pts[mask], axis=1)
        metrics["median_residual"] = float(np.median(residuals))
        tx, ty = tmpl_pts[mask][:, 0], tmpl_pts[mask][:, 1]
        metrics["spread_x"] = float((tx.max() - tx.min()) / template.width)
        metrics["spread_y"] = float((ty.max() - ty.min()) / template.height)
    else:
        metrics.update(median_residual=None, spread_x=0.0, spread_y=0.0)
    dec = _decompose(matrix)
    persp = _perspective(matrix, page_shape) if kind == "homography" else 0.0
    metrics.update(dec)
    metrics["perspective"] = persp

    if matches < gates["min_matches"]:
        reasons.append(f"only {matches} feature matches (need {gates['min_matches']:.0f})")
    if n_in < gates["min_inliers"]:
        reasons.append(f"only {n_in} RANSAC inliers (need {gates['min_inliers']:.0f})")
    if n_in and (metrics["spread_x"] < cfg.min_inlier_spread or metrics["spread_y"] < cfg.min_inlier_spread):
        reasons.append(f"inliers cover only {metrics['spread_x']:.2f} x {metrics['spread_y']:.2f} of the template (need {cfg.min_inlier_spread})")
    if metrics["median_residual"] is not None and metrics["median_residual"] > gates["max_residual"]:
        reasons.append(f"median residual {metrics['median_residual']:.2f}px exceeds {gates['max_residual']:.2f}px")
    if dec["det"] <= 0:
        reasons.append("the transform reflects or collapses the page (determinant <= 0)")
    for axis in ("scale_x", "scale_y"):
        if not (cfg.min_scale <= dec[axis] <= cfg.max_scale):
            reasons.append(f"{axis} {dec[axis]:.3f} is outside {cfg.min_scale}..{cfg.max_scale}")
    if dec["shear"] > cfg.max_shear_degrees:
        reasons.append(f"shear {dec['shear']:.1f} degrees exceeds {cfg.max_shear_degrees}")
    if persp > cfg.max_perspective:
        reasons.append(f"perspective {persp:.3f} exceeds {cfg.max_perspective}")
    return _Fit(kind, matrix, mask, page_pts, tmpl_pts, metrics, reasons)


def _chamfer(warped: np.ndarray, template: Template, layout: TemplateLayout) -> tuple[float, float]:
    """Compare the warped page's ruled lines with the template's, both ways, in the stable bands.
    Returns (worst mean distance in px, worst share of line pixels within 2 px)."""
    h_page, v_page = line_masks(warped)
    page_mask = cv2.bitwise_or(h_page, v_page)
    template_mask = template.line_mask
    band = np.zeros(template_mask.shape, np.uint8)
    for b in layout.stable_bands():
        band[int(b.y0):int(math.ceil(b.y1))] = 255
    results = []
    for source, target in ((template_mask, page_mask), (page_mask, template_mask)):
        points = (source > 0) & (band > 0)
        if not points.any():
            return float("inf"), 0.0
        distance = cv2.distanceTransform((target == 0).astype(np.uint8), cv2.DIST_L2, 3)
        d = distance[points]
        results.append((float(d.mean()), float((d <= 2.0).mean())))
    return max(r[0] for r in results), min(r[1] for r in results)


def _verify(fit: _Fit, gray: np.ndarray, template: Template, layout: TemplateLayout,
            cfg: AlignmentConfig, strict: float) -> None:
    size = (template.width, template.height)
    paper = int(round(paper_level(gray)))
    if fit.kind == "affine":
        warped = cv2.warpAffine(gray, fit.matrix[:2], size, flags=cv2.INTER_LINEAR, borderValue=paper)
    else:
        warped = cv2.warpPerspective(gray, fit.matrix, size, flags=cv2.INTER_LINEAR, borderValue=paper)
    mean_px, within = _chamfer(warped, template, layout)
    fit.verification = (mean_px, within)
    gates = _gate_scale(cfg, strict)
    if not math.isfinite(mean_px) or mean_px > gates["verify_mean"]:
        fit.reasons.append(f"independent line check: mean distance {mean_px:.2f}px exceeds {gates['verify_mean']:.2f}px")
    if within < gates["verify_frac"]:
        fit.reasons.append(f"independent line check: only {within:.2f} of line pixels within 2px (need {gates['verify_frac']:.2f})")


def _fit_with(detector: str, gray: np.ndarray, template: Template, layout: TemplateLayout, cfg: AlignmentConfig):
    """Return (accepted_fit_or_None, best_diagnostic_fit_or_None, counts, reasons)."""
    strict = cfg.orb_strictness if detector == "orb" else 1.0
    cv2.setRNGSeed(cfg.rng_seed)
    t_kp, t_desc, norm = _template_features(template, detector, cfg)
    p_kp, p_desc, _ = _features(gray, detector, cfg)
    counts = {"kp_template": len(t_kp), "kp_page": len(p_kp), "matches": 0}
    good = _matches(p_desc, t_desc, norm, cfg.ratio_test)
    counts["matches"] = len(good)
    if len(good) < 4:
        return None, None, counts, [f"only {len(good)} feature matches"]
    page_pts = np.float32([p_kp[m.queryIdx].pt for m in good])
    tmpl_pts = np.float32([t_kp[m.trainIdx].pt for m in good])

    cv2.setRNGSeed(cfg.rng_seed)
    m_aff, in_aff = cv2.estimateAffine2D(page_pts, tmpl_pts, method=cv2.RANSAC,
                                          ransacReprojThreshold=cfg.ransac_threshold_px,
                                          maxIters=cfg.ransac_max_iters, confidence=cfg.ransac_confidence)
    affine = None
    if m_aff is not None:
        m3 = np.vstack([m_aff, [0, 0, 1]]).astype(np.float64)
        affine = _evaluate("affine", m3, in_aff, page_pts, tmpl_pts, len(good), template, gray.shape, cfg, strict)
    if affine is None:
        return None, None, counts, ["RANSAC could not fit an affine transform"]
    if not affine.reasons:
        _verify(affine, gray, template, layout, cfg, strict)
    if not affine.reasons:
        return affine, affine, counts, []

    best = affine
    if cfg.allow_homography:
        cv2.setRNGSeed(cfg.rng_seed)
        m_h, in_h = cv2.findHomography(page_pts, tmpl_pts, cv2.RANSAC, cfg.ransac_threshold_px,
                                       maxIters=cfg.ransac_max_iters, confidence=cfg.ransac_confidence)
        homography = _evaluate("homography", m_h, in_h, page_pts, tmpl_pts, len(good), template, gray.shape, cfg, strict) \
            if m_h is not None else None
        if homography is not None and not homography.reasons:
            _verify(homography, gray, template, layout, cfg, strict)
            if not homography.reasons:
                aff_mean = affine.verification[0] if affine.verification else float("inf")
                if aff_mean - homography.verification[0] >= cfg.homography_min_improvement_px:
                    return homography, homography, counts, []
                affine.reasons.append("a homography fit was possible but not clearly better than the affine fit")
    return None, best, counts, list(best.reasons)


def _rejected(reasons: list[str], **extra: Any) -> AlignmentInfo:
    return AlignmentInfo(accepted=False, reasons=reasons or ["alignment failed"], opencv_version=cv2.__version__, **extra)


def _info(fit: _Fit, detector: str, counts: dict[str, int], accepted: bool, reasons: list[str]) -> AlignmentInfo:
    m = fit.metrics
    ver = fit.verification
    return AlignmentInfo(
        accepted=accepted,
        method=_METHODS[(detector, fit.kind)] if accepted else None,
        reasons=reasons,
        keypoints_template=counts["kp_template"], keypoints_page=counts["kp_page"], matches=counts["matches"],
        inliers=m["inliers"], inlier_ratio=round(min(1.0, m["inlier_ratio"]), 4),
        inlier_spread_x=round(m["spread_x"], 4), inlier_spread_y=round(m["spread_y"], 4),
        median_residual_px=None if m["median_residual"] is None else round(m["median_residual"], 4),
        scale_x=round(m["scale_x"], 5), scale_y=round(m["scale_y"], 5),
        rotation_degrees=round(m["rotation"], 4), shear_degrees=round(m["shear"], 4),
        perspective_magnitude=round(m["perspective"], 5),
        verification_chamfer_mean_px=None if ver is None or not math.isfinite(ver[0]) else round(ver[0], 4),
        verification_within_2px_fraction=None if ver is None else round(ver[1], 4),
        matrix=[[round(float(v), 6) for v in row] for row in fit.matrix] if accepted else None,
        opencv_version=cv2.__version__,
    )


def align_page(page: np.ndarray, template: Template, layout: TemplateLayout, cfg: AlignmentConfig) -> AlignmentInfo:
    """Register `page` (grayscale or BGR) to `template`. Never raises; a failure is a rejected result."""
    try:
        if page is None or not isinstance(page, np.ndarray) or page.size == 0:
            return _rejected(["the page image is empty"])
        gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY) if page.ndim == 3 else page
        if gray.dtype != np.uint8:
            return _rejected(["the page image is not 8-bit"])
        if min(gray.shape) < 100:
            return _rejected([f"the page image is too small ({gray.shape[1]}x{gray.shape[0]} px)"])
        all_reasons: list[str] = []
        first_diag: Optional[tuple[_Fit, str, dict[str, int]]] = None
        counts_seen: dict[str, int] = {"kp_template": 0, "kp_page": 0, "matches": 0}
        for detector in cfg.detectors:
            accepted, diag, counts, reasons = _fit_with(detector, gray, template, layout, cfg)
            if accepted is not None:
                return _info(accepted, detector, counts, True, [])
            all_reasons += [f"{detector}: {r}" for r in reasons]
            counts_seen = counts
            if diag is not None and first_diag is None:
                first_diag = (diag, detector, counts)
        if first_diag is not None:
            fit, detector, counts = first_diag
            return _info(fit, detector, counts, False, all_reasons or ["alignment failed"])
        return _rejected(all_reasons, keypoints_template=counts_seen["kp_template"],
                         keypoints_page=counts_seen["kp_page"], matches=counts_seen["matches"])
    except Exception as exc:  # fail closed, never crash the pipeline on one bad page
        return _rejected([f"alignment error: {type(exc).__name__}: {exc}"])
