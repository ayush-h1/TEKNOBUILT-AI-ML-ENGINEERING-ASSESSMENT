"""The blank form template: loading, identity and its ruled-line mask.

The template is the single source of geometry. Nothing here knows where any field is; that is
derived by `layout.py` from the template's own lines.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

import cv2
import numpy as np

MIN_TEMPLATE_SIDE = 200
MIN_TEMPLATE_CONTRAST = 20.0


class TemplateError(Exception):
    """The template is missing, unreadable or unusable. The pipeline cannot run without one."""


def paper_level(gray: np.ndarray) -> float:
    """Typical background brightness. The median is robust because most of a form is paper."""
    return float(np.median(gray))


def line_masks(gray: np.ndarray, paper: float | None = None, rule_ratio: float = 0.90) -> tuple[np.ndarray, np.ndarray]:
    """Horizontal and vertical ruled-line masks (uint8 0/255) of a grayscale form image.

    A pixel counts as ink when it is darker than `rule_ratio` of the paper level, which keeps the
    light grey rules of this form (about 168 on 255 paper). Lines are then isolated by morphological
    opening with kernels sized relative to the image (width // 20 and height // 64), so the result
    does not depend on a particular resolution."""
    if gray.ndim != 2:
        raise ValueError("line_masks expects a single-channel image")
    h, w = gray.shape
    paper = paper_level(gray) if paper is None else paper
    ink = (gray < rule_ratio * paper).astype(np.uint8) * 255
    horizontal = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, w // 20), 1)))
    vertical = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(3, h // 64))))
    return horizontal, vertical


@dataclass(eq=False)
class Template:
    gray: np.ndarray
    sha256: str
    path: str
    paper: float
    h_mask: np.ndarray
    v_mask: np.ndarray
    # per-detector feature cache, filled by alignment.py so features are computed once per run
    feature_cache: dict[str, Any] = field(default_factory=dict)

    @property
    def height(self) -> int:
        return int(self.gray.shape[0])

    @property
    def width(self) -> int:
        return int(self.gray.shape[1])

    @property
    def line_mask(self) -> np.ndarray:
        return cv2.bitwise_or(self.h_mask, self.v_mask)


def template_from_gray(gray: np.ndarray, sha256: str = "", path: str = "<memory>") -> Template:
    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise TemplateError("the template must be an 8-bit grayscale image")
    if min(gray.shape) < MIN_TEMPLATE_SIDE:
        raise TemplateError(f"the template is too small ({gray.shape[1]}x{gray.shape[0]} px)")
    if float(gray.std()) < MIN_TEMPLATE_CONTRAST:
        raise TemplateError("the template has almost no contrast; it looks blank")
    paper = paper_level(gray)
    h_mask, v_mask = line_masks(gray, paper)
    return Template(gray=gray, sha256=sha256, path=path, paper=paper, h_mask=h_mask, v_mask=v_mask)


def load_template(path: Union[str, Path]) -> Template:
    path = Path(path)
    if not path.is_file():
        raise TemplateError(f"template file not found: {path}")
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise TemplateError(f"template file could not be read: {exc}") from exc
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise TemplateError(f"template file is not a readable image: {path}")
    return template_from_gray(image, hashlib.sha256(data).hexdigest(), str(path))
