"""Typed loading of the YAML configuration files."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
DEFAULT_THRESHOLDS_PATH = CONFIG_DIR / "thresholds.yaml"
DEFAULT_SEMANTICS_PATH = CONFIG_DIR / "form_semantics.yaml"


class ConfigError(Exception):
    """Raised when a configuration file is missing or invalid."""


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class IngestConfig(_Model):
    render_dpi: int = Field(default=200, ge=50, le=600)
    max_pixels: int = Field(default=40_000_000, gt=0)
    # A run of full-width, pure-black rows covering at least this share of the page height
    # is treated as zero-filled image data (a truncated or damaged image stream).
    min_zero_band_ratio: float = Field(default=0.02, gt=0.0, le=1.0)


class QualityConfig(_Model):
    ink_relative_threshold: float = Field(default=0.6, gt=0.0, lt=1.0)
    mark_relative_threshold: float = Field(default=0.93, gt=0.0, lt=1.0)
    min_content_extent_ratio: float = Field(default=0.006, gt=0.0, lt=1.0)
    min_blank_paper_level: float = Field(default=170, ge=0.0, le=255.0)
    max_ink_ratio: float = Field(default=0.45, gt=0.0, le=1.0)
    min_mean_brightness: float = Field(default=90, ge=0.0, le=255.0)
    min_contrast: float = Field(default=40, ge=0.0, le=255.0)
    min_sharpness: float = Field(default=40, ge=0.0)
    sharpness_reference_width: int = Field(default=1000, ge=100)
    max_noise_sigma: float = Field(default=15.0, gt=0.0)
    speckle_min_difference: int = Field(default=60, ge=1, le=255)
    max_speckle_ratio: float = Field(default=0.005, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_ordering(self) -> "QualityConfig":
        if self.mark_relative_threshold <= self.ink_relative_threshold:
            raise ValueError("mark_relative_threshold must be above ink_relative_threshold")
        if self.min_blank_paper_level < self.min_mean_brightness:
            raise ValueError("min_blank_paper_level must not be below min_mean_brightness")
        return self


class AlignmentConfig(_Model):
    """Registration of a page to the template. Every default is a starting value taken from one
    supplied sample plus labelled synthetic cases; none is validated on real scans."""

    # Feature detectors to try, in order. SIFT is the primary. ORB is an opt-in alternative that is
    # never added automatically and must pass stricter verification (`orb_strictness`).
    detectors: list[Literal["sift", "orb"]] = Field(default_factory=lambda: ["sift"], min_length=1)
    max_features: int = Field(default=4000, ge=100)
    ratio_test: float = Field(default=0.75, gt=0.0, lt=1.0)
    ransac_threshold_px: float = Field(default=4.0, gt=0.0)      # in template pixels
    ransac_max_iters: int = Field(default=5000, ge=100)
    ransac_confidence: float = Field(default=0.999, gt=0.5, lt=1.0)
    rng_seed: int = 0
    # gates, all of which must pass
    min_matches: int = Field(default=12, ge=6)
    min_inliers: int = Field(default=12, ge=6)
    min_inlier_spread: float = Field(default=0.35, gt=0.0, le=1.0)   # share of template width and height covered by the inliers
    max_median_residual_px: float = Field(default=2.5, gt=0.0)
    min_scale: float = Field(default=0.2, gt=0.0)
    max_scale: float = Field(default=5.0, gt=0.0)
    max_shear_degrees: float = Field(default=30.0, gt=0.0, le=89.0)  # loose: a rotated, anisotropically stretched page shears legitimately
    max_perspective: float = Field(default=0.25, gt=0.0)             # relative change of the homogeneous weight across the page
    # independent verification against the template's own line mask
    verification_max_mean_px: float = Field(default=1.5, gt=0.0)
    verification_min_within_2px: float = Field(default=0.90, gt=0.0, le=1.0)
    orb_strictness: float = Field(default=0.5, gt=0.0, le=1.0)       # ORB must beat the thresholds by this factor
    # homography is tried only when the affine fit fails verification, and must be clearly better
    allow_homography: bool = True
    homography_min_improvement_px: float = Field(default=0.25, ge=0.0)

    @model_validator(mode="after")
    def _check(self) -> "AlignmentConfig":
        if self.min_scale >= self.max_scale:
            raise ValueError("min_scale must be below max_scale")
        if len(set(self.detectors)) != len(self.detectors):
            raise ValueError("detectors must not repeat")
        return self


class OcrConfig(_Model):
    tesseract_cmd: Optional[str] = None            # explicit path; otherwise auto-located
    language: str = "eng"
    upscale: float = Field(default=2.0, ge=1.0, le=6.0)
    # printed text at or above this confidence may be 'extracted'; below it needs review.
    printed_min_confidence: float = Field(default=0.80, ge=0.0, le=1.0)
    printed_provider: Literal["auto", "tesseract", "none"] = "auto"
    handwritten_provider: Literal["auto", "tesseract", "none"] = "auto"


class ExtractionConfig(_Model):
    """Reading cells in the aligned template frame. Starting values from the supplied sample (marked cells
    held 82-112 dark pixels, about 1.1-1.5% of a cell; every empty cell held 0) plus synthetic cases."""

    dark_ratio: float = Field(default=0.55, gt=0.0, lt=1.0)          # ink = darker than this share of the paper level
    line_tolerance_px: int = Field(default=2, ge=0, le=10)           # ink this close to a ruled line is the line
    checkbox_empty_max_fraction: float = Field(default=0.002, ge=0.0, lt=1.0)
    checkbox_checked_min_fraction: float = Field(default=0.005, gt=0.0, lt=1.0)
    checkbox_checked_max_fraction: float = Field(default=0.08, gt=0.0, le=1.0)   # more ink is a scribble or a filled box
    min_text_ink_px: int = Field(default=25, ge=1)                   # fewer dark pixels in a text cell = empty
    row_rule_min_coverage: float = Field(default=0.8, gt=0.0, le=1.0)
    orphan_ink_min_px: int = Field(default=50, ge=1)                 # ink in a slot with no grid that triggers review

    @model_validator(mode="after")
    def _check(self) -> "ExtractionConfig":
        if not (self.checkbox_empty_max_fraction < self.checkbox_checked_min_fraction < self.checkbox_checked_max_fraction):
            raise ValueError("checkbox thresholds must satisfy empty_max < checked_min < checked_max")
        return self


class Thresholds(_Model):
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    alignment: AlignmentConfig = Field(default_factory=AlignmentConfig)
    ocr: OcrConfig = Field(default_factory=OcrConfig)


class FieldKind(str, Enum):
    TEXT = "text"
    CHECKBOX = "checkbox"


class FieldRegion(_Model):
    name: str
    kind: FieldKind
    # Fractions of page width/height: (x0, y0, x1, y1), origin at the top-left.
    bbox: tuple[float, float, float, float]

    @model_validator(mode="after")
    def _check_bbox(self) -> "FieldRegion":
        x0, y0, x1, y1 = self.bbox
        if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
            raise ValueError(f"bbox for '{self.name}' must satisfy 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1")
        return self


class FormLayout(_Model):
    form_name: str
    fields: list[FieldRegion]

    @model_validator(mode="after")
    def _check_unique_names(self) -> "FormLayout":
        names = [f.name for f in self.fields]
        if len(names) != len(set(names)):
            raise ValueError("field names in the layout must be unique")
        return self


def _read_yaml(path: Path) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"config file is not valid YAML: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"config file must contain a mapping at the top level: {path}")
    return data


def load_thresholds(path: Union[str, Path, None] = None) -> Thresholds:
    path = Path(path) if path else DEFAULT_THRESHOLDS_PATH
    data = _read_yaml(path)
    try:
        return Thresholds.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"invalid thresholds in {path}: {exc}") from exc


def load_layout(path: Union[str, Path]) -> FormLayout:
    """Load an explicit, coordinate-based layout. There is no default file: the pipeline derives
    geometry from the template and the page, and only semantics come from `form_semantics.yaml`."""
    path = Path(path)
    data = _read_yaml(path)
    try:
        return FormLayout.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"invalid form layout in {path}: {exc}") from exc


# --- form semantics: names, kinds and options, never coordinates ---------------------------------------------

class TextFieldSpec(_Model):
    key: str
    kind: Literal["printed", "handwritten"]
    required: bool = True


class OptionGroupSpec(_Model):
    key: str
    options: list[str] = Field(min_length=2)
    required: bool = True

    @model_validator(mode="after")
    def _unique(self) -> "OptionGroupSpec":
        if len(set(self.options)) != len(self.options):
            raise ValueError(f"options of '{self.key}' must be unique")
        return self


class ExpectationSpec(OptionGroupSpec):
    question_keys: list[str] = Field(min_length=1)


class FormSemantics(_Model):
    form_name: str
    section_title: TextFieldSpec
    header_text_fields: list[TextFieldSpec] = Field(min_length=1)
    location: OptionGroupSpec
    shift: OptionGroupSpec
    checklist: OptionGroupSpec
    expectation: ExpectationSpec
    comments: TextFieldSpec

    @model_validator(mode="after")
    def _unique_keys(self) -> "FormSemantics":
        keys = [f.key for f in self.header_text_fields] + [self.section_title.key, self.comments.key,
                                                           self.location.key, self.shift.key, self.checklist.key,
                                                           self.expectation.key]
        if len(keys) != len(set(keys)):
            raise ValueError("semantic keys must be unique")
        if len(set(self.expectation.question_keys)) != len(self.expectation.question_keys):
            raise ValueError("expectation question keys must be unique")
        return self


def load_semantics(path: Union[str, Path, None] = None) -> FormSemantics:
    path = Path(path) if path else DEFAULT_SEMANTICS_PATH
    data = _read_yaml(path)
    try:
        return FormSemantics.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"invalid form semantics in {path}: {exc}") from exc
