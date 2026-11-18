import numpy as np
import pymupdf
import pytest

from field_digitization import ingest
from field_digitization.config import IngestConfig
from field_digitization.ingest import IngestError, iter_pages, open_pdf

import synthetic


def test_valid_pdf_renders_all_pages(tmp_path):
    pdf = synthetic.write_pdf(
        [synthetic.make_form_page(), synthetic.make_blank_page()], tmp_path / "ok.pdf"
    )
    doc = open_pdf(pdf)
    pages = list(iter_pages(doc, IngestConfig(render_dpi=100)))
    doc.close()

    assert [p.page_number for p in pages] == [1, 2]
    assert all(p.ok for p in pages)
    image = pages[0].image
    assert image.ndim == 2 and image.dtype.name == "uint8"
    # A4 at 100 DPI is about 827 x 1170 pixels.
    assert abs(image.shape[1] - 827) <= 2 and abs(image.shape[0] - 1170) <= 2


def test_render_dpi_controls_resolution(tmp_path):
    pdf = synthetic.write_pdf([synthetic.make_form_page()], tmp_path / "ok.pdf")
    doc = open_pdf(pdf)
    low = next(iter_pages(doc, IngestConfig(render_dpi=72))).image
    high = next(iter_pages(doc, IngestConfig(render_dpi=144))).image
    doc.close()
    assert high.shape[0] == pytest.approx(low.shape[0] * 2, abs=2)


def test_missing_file_raises_ingest_error(tmp_path):
    with pytest.raises(IngestError, match="not found"):
        open_pdf(tmp_path / "missing.pdf")


def test_garbage_file_raises_ingest_error(tmp_path):
    bad = tmp_path / "garbage.pdf"
    bad.write_bytes(b"this is definitely not a pdf" * 10)
    with pytest.raises(IngestError):
        open_pdf(bad)


def test_empty_file_raises_ingest_error(tmp_path):
    empty = tmp_path / "empty.pdf"
    empty.write_bytes(b"")
    with pytest.raises(IngestError):
        open_pdf(empty)


def test_non_pdf_document_raises_ingest_error(tmp_path):
    # PyMuPDF opens PNG files too; they must not be accepted as PDFs.
    import cv2

    path = tmp_path / "image.pdf"
    cv2.imwrite(str(path.with_suffix(".png")), synthetic.make_blank_page())
    path.with_suffix(".png").rename(path)
    with pytest.raises(IngestError):
        open_pdf(path)


def test_encrypted_pdf_raises_ingest_error(tmp_path):
    source = tmp_path / "plain.pdf"
    synthetic.write_pdf([synthetic.make_blank_page()], source)
    locked = tmp_path / "locked.pdf"
    doc = pymupdf.open(source)
    doc.save(locked, encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="owner", user_pw="user")
    doc.close()
    with pytest.raises(IngestError, match="password"):
        open_pdf(locked)


def test_one_failing_page_does_not_stop_the_rest(tmp_path, monkeypatch):
    pdf = synthetic.write_pdf(
        [synthetic.make_blank_page(), synthetic.make_form_page(), synthetic.make_blank_page()],
        tmp_path / "three.pdf",
    )
    real_render = ingest._render_page

    def flaky(doc, index, config):
        if index == 1:
            raise RuntimeError("simulated decoder failure")
        return real_render(doc, index, config)

    monkeypatch.setattr(ingest, "_render_page", flaky)

    doc = open_pdf(pdf)
    pages = list(iter_pages(doc, IngestConfig(render_dpi=72)))
    doc.close()

    assert [p.ok for p in pages] == [True, False, True]
    assert "simulated decoder failure" in pages[1].error
    assert pages[1].image is None


def _pages_for(path, dpi=100):
    doc = open_pdf(path)
    try:
        return list(iter_pages(doc, IngestConfig(render_dpi=dpi)))
    finally:
        doc.close()


def test_intact_pdf_has_no_warnings_and_no_damage(tmp_path):
    pdf = synthetic.write_pdf([synthetic.make_small_form_page(i) for i in (1, 2)], tmp_path / "a.pdf")
    for page in _pages_for(pdf):
        assert page.warnings == ()
        assert page.damage is None


def test_truncated_pdf_records_warnings_on_the_damaged_page_only(tmp_path):
    pdf = synthetic.write_pdf(
        [synthetic.make_small_form_page(i) for i in (1, 2, 3)], tmp_path / "a.pdf"
    )
    pdf.write_bytes(synthetic.truncate_bytes(pdf.read_bytes(), 0.7))
    pages = _pages_for(pdf)
    assert [bool(p.damage) for p in pages] == [False, False, True]
    assert pages[2].warnings  # the evidence is attached to page 3, not spread over the file
    assert pages[0].warnings == () and pages[1].warnings == ()


def test_zero_filled_band_from_damaged_image_stream_is_flagged(tmp_path):
    pdf = synthetic.write_pdf(
        [synthetic.make_small_form_page(i) for i in (1, 2, 3)], tmp_path / "a.pdf"
    )
    pdf.write_bytes(synthetic.zero_image_stream_span(pdf.read_bytes(), 1, 0.35, 0.65, 3))
    pages = _pages_for(pdf)
    assert [bool(p.damage) for p in pages] == [False, True, False]
    assert "zero-filled band" in pages[1].damage
    assert pages[1].warnings == ()  # the damage is silent; the band is the only evidence


def _pure_white_first_pdf(tmp_path, keep):
    """Page 1 is an intact pure-white scan; pages 2 and 3 are forms. The file is truncated."""
    pdf = synthetic.write_pdf(
        [synthetic.make_pure_white_page(), synthetic.make_small_form_page(1),
         synthetic.make_small_form_page(2)],
        tmp_path / "white_first.pdf",
    )
    pdf.write_bytes(synthetic.truncate_bytes(pdf.read_bytes(), keep))
    return pdf


@pytest.mark.parametrize("keep", [0.5, 0.4, 0.3])
def test_page_tree_warnings_are_not_attributed_to_the_first_page_as_damage(tmp_path, keep):
    pages = _pages_for(_pure_white_first_pdf(tmp_path, keep))
    first, second, third = pages

    # Page 1 is intact, but MuPDF reports the damaged page tree while loading it.
    assert first.load_warnings, "the scenario must produce page-tree warnings on page 1"
    assert any("page tree" in w for w in first.load_warnings)
    assert first.warnings == ()  # nothing went wrong while rendering page 1's content
    assert first.damage is None  # so it is not marked as damaged
    assert int(first.image.min()) == int(first.image.max()) == 255  # and it is uniform white

    # Page-specific damage is still found: page 2 lost its image stream (render warnings).
    assert second.warnings
    assert "plain white" in second.damage
    assert second.load_warnings == ()

    # Page 3's own object is lost; that is page-tree damage and references no content.
    assert third.damage is None


def test_open_time_warnings_do_not_reach_any_page(tmp_path):
    # At 70% truncation the xref repair warnings occur while opening, before any page loads.
    pages = _pages_for(_pure_white_first_pdf(tmp_path, 0.7))
    assert pages[0].warnings == () and pages[0].load_warnings == ()
    assert pages[0].damage is None


def test_intact_pdf_has_no_load_warnings(tmp_path):
    pdf = synthetic.write_pdf(
        [synthetic.make_pure_white_page(), synthetic.make_small_form_page()], tmp_path / "a.pdf"
    )
    for page in _pages_for(pdf):
        assert page.load_warnings == () and page.warnings == ()


def test_warnings_alone_are_not_treated_as_damage(tmp_path, monkeypatch):
    pdf = synthetic.write_pdf([synthetic.make_small_form_page()], tmp_path / "a.pdf")
    monkeypatch.setattr(ingest, "_drain_warnings", lambda: ("harmless MuPDF warning",))
    page = _pages_for(pdf)[0]
    assert page.warnings == ("harmless MuPDF warning",)
    assert page.damage is None  # the page rendered with visible content


def test_blank_white_scan_with_warnings_but_no_content_reference_is_not_damage(tmp_path, monkeypatch):
    # An empty PDF page (no image, no text, no drawings) rendering white is just empty.
    doc = pymupdf.open()
    doc.new_page(width=595, height=842)
    path = tmp_path / "empty.pdf"
    doc.save(path)
    doc.close()
    monkeypatch.setattr(ingest, "_drain_warnings", lambda: ("some warning",))
    assert _pages_for(path)[0].damage is None


def test_mupdf_errors_are_captured_not_printed(tmp_path, capfd):
    pdf = synthetic.write_pdf([synthetic.make_small_form_page(i) for i in (1, 2)], tmp_path / "a.pdf")
    pdf.write_bytes(synthetic.truncate_bytes(pdf.read_bytes(), 0.6))
    before = pymupdf.TOOLS.mupdf_display_errors()
    _pages_for(pdf)
    captured = capfd.readouterr()
    assert "MuPDF error" not in captured.out + captured.err
    assert pymupdf.TOOLS.mupdf_display_errors() == before  # global setting restored


# --- MuPDF global display flag: it must never be left changed ------------------------

def _display():
    return pymupdf.TOOLS.mupdf_display_errors()


@pytest.fixture(params=[True, False], ids=["display-on", "display-off"])
def initial_display(request):
    """Run the test with the MuPDF display flag set to each value, then put it back."""
    original = _display()
    pymupdf.TOOLS.mupdf_display_errors(request.param)
    yield request.param
    pymupdf.TOOLS.mupdf_display_errors(original)


@pytest.fixture
def two_page_pdf(tmp_path):
    return synthetic.write_pdf(
        [synthetic.make_small_form_page(i) for i in (1, 2)], tmp_path / "two.pdf"
    )


def test_display_flag_is_restored_after_a_page_fails_to_render(
    two_page_pdf, monkeypatch, initial_display
):
    def fail(doc, index, config):
        raise RuntimeError("render failure")

    monkeypatch.setattr(ingest, "_render_page", fail)
    pages = _pages_for(two_page_pdf)
    assert [p.ok for p in pages] == [False, False]
    assert _display() == initial_display


def test_display_flag_is_restored_when_the_iterator_itself_raises(
    two_page_pdf, monkeypatch, initial_display
):
    # An exception that is not isolated per page must still leave the flag as it was,
    # immediately, without waiting for garbage collection.
    def explode():
        raise RuntimeError("unexpected failure while draining warnings")

    doc = open_pdf(two_page_pdf)  # open first: open_pdf drains warnings too
    monkeypatch.setattr(ingest, "_drain_warnings", explode)
    iterator = iter_pages(doc, IngestConfig(render_dpi=72))
    with pytest.raises(RuntimeError, match="draining warnings"):
        next(iterator)
    assert _display() == initial_display
    doc.close()


def test_a_failing_damage_assessment_is_isolated_and_fails_closed(
    two_page_pdf, monkeypatch, initial_display
):
    def explode(*args, **kwargs):
        raise RuntimeError("bug in damage assessment")

    monkeypatch.setattr(ingest, "_assess_damage", explode)
    pages = _pages_for(two_page_pdf)
    assert len(pages) == 2  # the batch continued
    for page in pages:
        assert page.ok  # the image is still available
        assert "damage assessment itself failed" in page.damage
        assert "bug in damage assessment" in page.damage  # the original exception is kept
    assert _display() == initial_display


def test_display_flag_is_not_held_while_the_generator_is_suspended(two_page_pdf, initial_display):
    doc = open_pdf(two_page_pdf)
    iterator = iter_pages(doc, IngestConfig(render_dpi=72))
    next(iterator)  # suspended at the first yield, with a second page still to come
    assert _display() == initial_display  # the caller's own MuPDF calls see the original flag
    iterator.close()
    assert _display() == initial_display
    doc.close()


def test_display_flag_is_unchanged_when_the_consumer_raises(two_page_pdf, initial_display):
    doc = open_pdf(two_page_pdf)
    iterator = iter_pages(doc, IngestConfig(render_dpi=72))  # keep a reference: no GC reliance
    with pytest.raises(RuntimeError, match="consumer"):
        for _ in iterator:
            raise RuntimeError("consumer failure")
    assert _display() == initial_display
    doc.close()


@pytest.mark.parametrize("finish_order", ["ab", "ba"])
def test_interleaved_iterators_leave_the_display_flag_unchanged(tmp_path, initial_display, finish_order):
    doc_a = open_pdf(synthetic.write_pdf(
        [synthetic.make_small_form_page(i) for i in (1, 2, 3)], tmp_path / "a.pdf"))
    doc_b = open_pdf(synthetic.write_pdf(
        [synthetic.make_small_form_page(i) for i in (4, 5, 6)], tmp_path / "b.pdf"))
    config = IngestConfig(render_dpi=72)
    iterators = {"a": iter_pages(doc_a, config), "b": iter_pages(doc_b, config)}

    # Start both, then advance them alternately, then finish them in the requested order.
    next(iterators["a"])
    next(iterators["b"])
    assert _display() == initial_display
    next(iterators["a"])
    next(iterators["b"])
    assert _display() == initial_display
    for key in finish_order:
        assert [p.page_number for p in iterators[key]] == [3]
        assert _display() == initial_display
    doc_a.close()
    doc_b.close()


@pytest.mark.parametrize("kind", ["missing", "garbage", "not_a_pdf"])
def test_display_flag_is_restored_when_open_pdf_fails(tmp_path, initial_display, kind):
    if kind == "missing":
        path = tmp_path / "missing.pdf"
    elif kind == "garbage":
        path = tmp_path / "garbage.pdf"
        path.write_bytes(b"not a pdf at all" * 5)
    else:
        import cv2

        png = tmp_path / "image.png"
        cv2.imwrite(str(png), synthetic.make_blank_page())
        path = tmp_path / "image.pdf"
        png.rename(path)
    with pytest.raises(IngestError):
        open_pdf(path)
    assert _display() == initial_display


def test_nested_capture_blocks_restore_the_flag_in_order(initial_display):
    with ingest._mupdf_errors_captured():
        assert _display() is False
        with ingest._mupdf_errors_captured():
            assert _display() is False
        assert _display() is False  # inner exit restores the OUTER state, which is "off"
    assert _display() == initial_display


def test_display_flag_is_restored_when_the_captured_block_raises(initial_display):
    with pytest.raises(ValueError):
        with ingest._mupdf_errors_captured():
            raise ValueError("boom")
    assert _display() == initial_display


@pytest.mark.parametrize(
    "rows, expected",
    [
        ("nnn", 0),
        ("BBnnnB", 2),
        ("nBBnBBBn", 3),
        ("BBBB", 4),
    ],
)
def test_longest_zero_band(rows, expected):
    # 'B' is a full row of pure black, 'n' a normal row (value 128).
    image = np.array([[0 if r == "B" else 128] * 5 for r in rows], dtype=np.uint8)
    assert ingest._longest_zero_band(image) == expected


def test_black_rows_that_are_not_full_width_do_not_count():
    image = np.full((100, 100), 128, dtype=np.uint8)
    image[40:60, :50] = 0  # black on half of the row only
    assert ingest._longest_zero_band(image) == 0


def test_oversized_page_is_rejected_not_rendered(tmp_path):
    pdf = synthetic.write_pdf([synthetic.make_blank_page()], tmp_path / "ok.pdf")
    doc = open_pdf(pdf)
    pages = list(iter_pages(doc, IngestConfig(render_dpi=200, max_pixels=1000)))
    doc.close()
    assert not pages[0].ok
    assert "above the limit" in pages[0].error
