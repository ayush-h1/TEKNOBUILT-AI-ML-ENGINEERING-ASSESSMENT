"""Typed loading of the YAML configuration files."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
DEFAULT_THRESHOLDS_PATH = CONFIG_DIR / "thresholds.yaml"
DEFAULT_LAYOUT_PATH = CONFIG_DIR / "form_layout.yaml"


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


class Thresholds(_Model):
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)


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


def load_layout(path: Union[str, Path, None] = None) -> FormLayout:
    path = Path(path) if path else DEFAULT_LAYOUT_PATH
    data = _read_yaml(path)
    try:
        return FormLayout.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"invalid form layout in {path}: {exc}") from exc
