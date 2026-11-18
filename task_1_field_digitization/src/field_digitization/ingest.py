"""PDF ingestion: render each page to a grayscale image.

Failure is handled at three levels:
  * Document level: the file is missing, is not a PDF, is encrypted or has no pages.
    Nothing can be processed, so `open_pdf` raises IngestError.
  * Page level, loud: one page cannot be rendered. The error is captured on that page's
    `IngestedPage` and iteration continues, so one bad page never stops the batch.
  * Page level, silent: MuPDF repairs damaged files instead of raising, and a page whose
    image stream is lost renders as plain white paper. Left alone, that would look exactly
    like a genuinely blank page and real data loss would be reported as "blank".
    To catch it, the MuPDF warnings raised while rendering each page are recorded on
    that page. A warning alone proves little (MuPDF emits benign ones), so it is not
    treated as corruption by itself. A page is flagged as damaged only when the evidence
    agrees: warnings were raised, the page references content (an image, text or
    drawings), and yet it rendered as uniform white.

Damaged image data does not always warn. A damaged or truncated image stream can decode
without any warning, with the lost part zero-filled, i.e. a band of pure black rows. That
signature is checked separately and does not need a warning.

Where a warning comes from decides what it can prove. There are three phases:
  * Opening the document (xref repair and similar): document-level, only logged.
  * Loading a page object: MuPDF resolves the page tree lazily and reports problems with
    it ("non-page object in page tree", "Page tree load failed") on whichever page is loaded
    first. They are recorded in `IngestedPage.load_warnings`, are NOT evidence that this page
    is damaged, and cannot be attributed to it. They make a blank-looking page "needs
    review" (document-level uncertainty), never "corrupted".
  * Rendering the page content ("object is not a stream", "cannot load content stream"):
    reproducible for that page, recorded in `IngestedPage.warnings`, and the only warnings
    used by the damage rule.
If a page's own object is lost, MuPDF repeats its page-tree warning when the page is loaded
again during rendering, so such a warning can appear in both lists. That is harmless: a
page with no object references no content, so the damage rule does not fire for it.

MuPDF prints its errors to stdout by default. They are captured as warnings instead, so
nothing is lost and the process's output stays clean.

Global MuPDF state and concurrency
----------------------------------
MuPDF keeps two pieces of PROCESS-GLOBAL state that this module touches: the warning
buffer and the "display errors" flag. Both are shared by every document and every thread.

What this module does about it:
  * The display flag is switched off only inside `_mupdf_errors_captured()`, and every use
    of that context manager is a plain synchronous block. It is never held across a
    `yield`, so a suspended or abandoned generator cannot leave the flag changed, and
    several generators interleaved in ONE thread cannot corrupt each other's save/restore.
    Restoration happens in a `finally`, so it also holds when the block raises.
  * The warning buffer is reset before and drained right after each page's MuPDF work,
    inside the same block, so each page gets only its own warnings.

What it does NOT do: it takes no lock and is NOT thread-safe. Two threads using MuPDF at
the same time can overwrite each other's display flag and mix warnings between pages, and
code outside this module that uses MuPDF concurrently has the same effect. Run ingestion
from one thread at a time, or parallelise with separate processes (each process has its
own MuPDF state). This has not been tested under threads.

Page-tree warnings that MuPDF raises lazily on the first page loaded are separated from that
page's render warnings (see above), so they no longer feed the damage rule. This separation
relies on the same process-global warning buffer and is subject to the same limits.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Union

import numpy as np
import pymupdf

from .config import IngestConfig

logger = logging.getLogger(__name__)

# A render is "uniform white" when every pixel is within this range of white.
_WHITE_MIN_LEVEL = 250
_WHITE_MAX_SPREAD = 3
# A row is "zero-filled" when its brightest pixel is at most this.
_BLACK_MAX_LEVEL = 2


class IngestError(Exception):
    """The PDF as a whole cannot be processed."""


@dataclass(frozen=True)
class TextWord:
    text: str
    bbox: tuple[float, float, float, float]  # PDF points, origin top-left


@dataclass(frozen=True)
class TextLine:
    text: str
    direction: tuple[float, float]  # MuPDF line direction (cos, sin); (1, 0) is horizontal left to right


@dataclass(frozen=True)
class TextLayer:
    """The text a PDF page carries itself. A scanned form has an empty layer; a vector page has text.

    An empty layer means "this page has no text of its own". `None` on `IngestedPage.text_layer`
    means the layer could not be read, which is a different statement.
    """

    words: tuple[TextWord, ...] = ()
    lines: tuple[TextLine, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not self.words

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)


@dataclass(frozen=True)
class IngestedPage:
    """One rendered page, or the reason it could not be rendered."""

    page_number: int                     # 1-based
    image: Optional[np.ndarray] = None   # uint8 grayscale, shape (H, W)
    error: Optional[str] = None
    # MuPDF warnings raised while RENDERING this page's content. These are reproducible for
    # the page and are the only warnings used as evidence of page-specific damage.
    warnings: tuple[str, ...] = ()
    # MuPDF warnings raised while LOADING the page object, i.e. while MuPDF walks the
    # document's page tree. They describe the document's structure, are emitted lazily on
    # whichever page is loaded first, and cannot be attributed to this page. They are a
    # document-level uncertainty: never evidence of damage to this page by themselves.
    load_warnings: tuple[str, ...] = ()
    damage: Optional[str] = None         # set when data loss is suspected, with the evidence
    # The page's own text (schema 2.0). Read separately from rendering and never changes `warnings`.
    text_layer: Optional[TextLayer] = None

    @property
    def ok(self) -> bool:
        return self.image is not None


def open_pdf(path: Union[str, Path]) -> pymupdf.Document:
    """Open a PDF, raising IngestError for any document-level problem."""
    path = Path(path)
    if not path.is_file():
        raise IngestError(f"input file not found: {path}")
    with _mupdf_errors_captured():
        pymupdf.TOOLS.reset_mupdf_warnings()
        try:
            doc = pymupdf.open(path)
        except Exception as exc:  # PyMuPDF raises several exception types for bad files
            raise IngestError(f"cannot open '{path.name}' as a PDF: {exc}") from exc
        if not doc.is_pdf:
            doc.close()
            raise IngestError(f"'{path.name}' is not a PDF document")
        if doc.needs_pass:
            doc.close()
            raise IngestError(f"'{path.name}' is password protected")
        if doc.page_count == 0:
            doc.close()
            raise IngestError(f"'{path.name}' contains no pages")
        for warning in _drain_warnings():
            logger.warning("document-level MuPDF warning for '%s': %s", path.name, warning)
    return doc


@contextmanager
def _mupdf_errors_captured() -> Iterator[None]:
    """Stop MuPDF printing errors to stdout while it runs; they stay available as warnings."""
    previous = pymupdf.TOOLS.mupdf_display_errors()
    pymupdf.TOOLS.mupdf_display_errors(False)
    try:
        yield
    finally:
        pymupdf.TOOLS.mupdf_display_errors(previous)


def _drain_warnings() -> tuple[str, ...]:
    text = pymupdf.TOOLS.mupdf_warnings(reset=True)
    return tuple(line.strip() for line in text.splitlines() if line.strip())


def _render_page(doc: pymupdf.Document, index: int, config: IngestConfig) -> np.ndarray:
    page = doc.load_page(index)
    zoom = config.render_dpi / 72.0
    width_px = page.rect.width * zoom
    height_px = page.rect.height * zoom
    if width_px * height_px > config.max_pixels:
        raise ValueError(
            f"page would render to {int(width_px)}x{int(height_px)} px, "
            f"above the limit of {config.max_pixels} px"
        )
    pix = page.get_pixmap(
        matrix=pymupdf.Matrix(zoom, zoom),
        colorspace=pymupdf.csGRAY,
        alpha=False,
    )
    image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
    return image.copy()  # detach from the pixmap buffer


def _read_text_layer(doc: pymupdf.Document, index: int) -> Optional[TextLayer]:
    """Read the page's words and line directions. Returns None (and logs) if that fails; it never raises."""
    try:
        page = doc.load_page(index)
        words = tuple(
            TextWord(str(w[4]), (float(w[0]), float(w[1]), float(w[2]), float(w[3])))
            for w in page.get_text("words")
        )
        lines: list[TextLine] = []
        for block in page.get_text("dict").get("blocks", []):
            for line in block.get("lines", []):
                text = "".join(span.get("text", "") for span in line.get("spans", []))
                if text.strip():
                    lines.append(TextLine(text, (float(line["dir"][0]), float(line["dir"][1]))))
        return TextLayer(words=words, lines=tuple(lines))
    except Exception as exc:  # the text layer is evidence only; its failure must not fail the page
        logger.warning("text layer of page %d could not be read: %s", index + 1, exc)
        return None


def _page_references_content(doc: pymupdf.Document, index: int) -> bool:
    """True if the PDF says this page has an image, text or vector drawings.

    Fails closed: if the page cannot even be inspected, assume it had content.
    """
    try:
        page = doc.load_page(index)
        if page.get_images(full=True):
            return True
        if page.get_text("text").strip():
            return True
        return bool(page.get_drawings())
    except Exception:
        return True


def _is_uniform_white(image: np.ndarray) -> bool:
    low, high = int(image.min()), int(image.max())
    return low >= _WHITE_MIN_LEVEL and (high - low) <= _WHITE_MAX_SPREAD


def _longest_zero_band(image: np.ndarray) -> int:
    """Number of rows in the longest run of consecutive full-width pure-black rows."""
    black_rows = image.max(axis=1) <= _BLACK_MAX_LEVEL
    if not black_rows.any():
        return 0
    padded = np.concatenate(([False], black_rows, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return int((edges[1::2] - edges[0::2]).max())


def _assess_damage(
    doc: pymupdf.Document,
    index: int,
    image: np.ndarray,
    warnings: tuple[str, ...],
    config: IngestConfig,
) -> Optional[str]:
    """Return the evidence for suspected data loss on this page, or None."""
    evidence: list[str] = []

    # Silent loss, case 1: warnings AND the page references content AND it rendered white.
    if warnings and _is_uniform_white(image) and _page_references_content(doc, index):
        evidence.append(
            f"{len(warnings)} MuPDF warning(s) while rendering, and the page references "
            f"content but rendered as plain white (first warning: {warnings[0]})"
        )

    # Silent loss, case 2: lost image data is zero-filled, giving a band of black rows.
    band = _longest_zero_band(image)
    if band / image.shape[0] >= config.min_zero_band_ratio:
        evidence.append(
            f"image contains a zero-filled band of {band} rows "
            f"({band / image.shape[0]:.1%} of the page height), typical of a truncated or "
            f"damaged image stream"
        )
    return "; ".join(evidence) or None


def iter_pages(
    doc: pymupdf.Document, config: Optional[IngestConfig] = None
) -> Iterator[IngestedPage]:
    """Yield pages lazily, one at a time, so memory stays bounded for large PDFs.

    The MuPDF display flag is changed only while a page is being rendered and assessed,
    never while the generator is suspended at a `yield`.
    """
    config = config or IngestConfig()
    for index in range(doc.page_count):
        page_number = index + 1
        with _mupdf_errors_captured():
            pymupdf.TOOLS.reset_mupdf_warnings()
            load_warnings: tuple[str, ...] = ()
            try:
                # Phase 1, load the page object. MuPDF resolves the page tree lazily here and
                # reports page-tree problems for the document, not for this page's content.
                doc.load_page(index)
                load_warnings = _drain_warnings()
                # Phase 2, render the content. Warnings from now on belong to this page.
                image = _render_page(doc, index, config)
            except Exception as exc:  # isolate: any page failure is recorded, not raised
                logger.warning("page %d could not be rendered: %s", page_number, exc)
                page = IngestedPage(
                    page_number=page_number,
                    error=f"{type(exc).__name__}: {exc}",
                    warnings=_drain_warnings(),
                    load_warnings=load_warnings,
                )
            else:
                warnings = _drain_warnings()
                try:
                    damage = _assess_damage(doc, index, image, warnings, config)
                except Exception as exc:  # isolate, and fail closed: unknown means suspect
                    logger.exception("damage assessment failed for page %d", page_number)
                    damage = (
                        f"damage assessment itself failed ({type(exc).__name__}: {exc}); "
                        f"the page cannot be trusted"
                    )
                if damage:
                    logger.warning("page %d looks damaged: %s", page_number, damage)
                elif warnings:
                    logger.info(
                        "page %d rendered with %d MuPDF warning(s)", page_number, len(warnings)
                    )
                if load_warnings:
                    logger.info(
                        "page %d: %d document-structure (page tree) warning(s) while loading",
                        page_number, len(load_warnings),
                    )
                # Read after the render warnings were drained, so it cannot change them.
                text_layer = _read_text_layer(doc, index)
                page = IngestedPage(
                    page_number=page_number,
                    image=image,
                    warnings=warnings,
                    load_warnings=load_warnings,
                    damage=damage,
                    text_layer=text_layer,
                )
        yield page  # MuPDF global state is back to normal here, whatever the consumer does
