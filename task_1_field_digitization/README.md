# Task 1: Field data digitization

Turns a PDF of filled PACE safety observation forms into structured JSON. For every page the output
says what was read, how sure the pipeline is, and what a person still has to check. A field only
carries a trusted value when extraction actually succeeded. Everything else has an explicit status
and, where relevant, a null value. Nothing is invented.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r task_1_field_digitization\requirements.txt
```

Tesseract is optional. It is needed only for text; tick boxes are read without OCR.

- Windows: `winget install UB-Mannheim.TesseractOCR`.
- The executable is found in this order: `ocr.tesseract_cmd` in `config/thresholds.yaml`, the
  `TESSERACT_CMD` environment variable, `PATH`, then `C:\Program Files\Tesseract-OCR\tesseract.exe`.
- Without Tesseract every text field is `ocr_unavailable` with a null value, and tick boxes are still read.

Tested environment: Python 3.11.9, OpenCV 5.1.0 (`opencv-python-headless` 5.1.0.95), PyMuPDF 1.28.2,
Tesseract 5.4.0.20240606. No other OpenCV version has been tested.

## Run

Put the inputs in place. Both are git-ignored because they contain real form data.

- `data/input/Sample_input.pdf`
- `data/template/template.png` (the blank form)

```powershell
cd task_1_field_digitization
$env:PYTHONPATH = "src"
..\.venv\Scripts\python.exe -m field_digitization            # uses the default paths above
..\.venv\Scripts\python.exe -m field_digitization --input other.pdf --output out.json --ocr none
```

| Option | Default | Meaning |
|---|---|---|
| `--input` | `data/input/Sample_input.pdf` | PDF to process |
| `--template` | `data/template/template.png` | blank form image |
| `--output` | `data/output/extracted_observations.json` | structured result |
| `--report` | `processing_report.json` next to `--output` | processing summary |
| `--ocr` | from config (`auto`) | `auto`, `tesseract` or `none` |
| `--dpi` | 200 | render resolution |
| `--thresholds` | `config/thresholds.yaml` | tunable thresholds |

Exit code 0 means every page was processed (each page has a status). Exit code 2 means a
document-level problem: the PDF cannot be opened, or the template is missing or does not have the
expected structure. The output is deterministic: two runs on the same input give byte-identical JSON.

Tests: `..\.venv\Scripts\python.exe -m pytest -p no:cacheprovider` from `task_1_field_digitization`.
Tests that need the real files are skipped when they are absent.

## How a page is processed

1. **Ingest** (`ingest.py`). Each page is rendered to grayscale, and its PDF text layer is read. Render
   errors, MuPDF warnings and signs of damage (zero-filled image bands, content that renders white)
   are recorded.
2. **Gate** (`page_gate.py`, `quality.py`). Damaged pages become `corrupted`. Pages with no content
   become `blank`, or `needs_review` if MuPDF warned while rendering them. Pages that are too dark,
   blurred, low-contrast or noisy become `unreadable`. None of these is read further.
3. **Failed test form** (`failed_form.py`). If the page's own text layer contains the words
   `FAILED TEST FORM` (any case, any angle), the page is `failed_test_form` and nothing else is
   extracted. This is exact evidence, so no review is requested. A page with no text layer that
   cannot be aligned is OCR'd at several rotations. If at least two of the three words are found in
   order, the page is also classified `failed_test_form`, but with a review request, because OCR
   evidence is partial.
4. **Alignment** (`alignment.py`). The page is registered to the template with SIFT features and a
   RANSAC affine fit. A homography is tried only if the affine fit fails, and kept only if it is
   clearly better. The fit must pass gates on matches, inliers, inlier spread, residual, scale, shear,
   reflection and perspective. It must also pass an independent check: the page's ruled lines must
   coincide with the template's lines. A page that fails is `needs_review` with the reasons, and
   nothing is read from it.
5. **Layout** (`template.py`, `layout.py`). Geometry comes from the template image itself, never from
   fixed coordinates: title bar, checklist slots and S/R/I/N columns, the CCZJV table, comments, and
   the header cells. Header tick boxes are told apart from label cells by their dark borders.
   `config/form_semantics.yaml` holds only names, kinds and options, in reading order.
6. **Extraction** (`extract.py`), in the template's pixel frame.
   - **Checklist rows.** A template slot counts as a row only when its grid lines are drawn on this
     page, so the row count comes from each page. Ink in an undrawn slot, or a grid that reappears
     after a gap, is reported for review and not read.
   - **Tick boxes.** Ruled lines and anything the blank template prints are removed, then the
     remaining ink is measured as a share of the box area. An empty box is `unchecked`, a mark of
     normal size is `checked`, and a faint mark or a box with too much ink (scribble, correction,
     filled box) is `uncertain`.
   - **Choice groups** (each checklist row, each CCZJV question, CWA location, shift). A group is
     `answered` only when exactly one box is checked and all others are clearly empty. Otherwise it
     is `unanswered`, `conflict` or `uncertain`, `selected` stays null, and the group is flagged for
     review. Every group on this form is required.
   - **Text.** Each field is OCR'd from its own cell after lines and template print are removed.
     - *Printed* text (section title, checklist and CCZJV questions) is `extracted` when Tesseract's
       confidence is at least 0.80, and `needs_review` otherwise.
     - *Handwritten* fields (date, observer, company, timesheet #, employee #, comments) are
       **always** `needs_review`. The value is kept as a candidate with its confidence, engine and
       reason. The schema refuses `extracted` for handwriting.
     - A cell with no ink is `empty_or_unreadable` with a null value. An empty required header field
       adds a page review reason; empty comments do not.
7. **Page status.** `ok` if nothing needs a person, otherwise `needs_review` with a list of reasons.
   An unexpected error on one page makes only that page `failed`, with the error message; the other
   pages continue.

## Output

`extracted_observations.json` is a `DocumentResult` (see `src/field_digitization/schema.py`,
schema version 2.0). It has one entry per page:

- `status`: one of `ok`, `needs_review`, `blank`, `unreadable`, `corrupted`, `failed`, `failed_test_form`.
- `review_required` and `review_reasons`.
- `quality`: the gate's metrics.
- `alignment`: method, gate metrics and the page-to-template matrix.
- `failed_form_evidence`: the matched words, their boxes and the text direction.
- `form`, for pages that were read:
  - `section_title`;
  - `header`: five text fields, plus `location` and `shift` as choice groups;
  - `checklist`: one entry per detected row, each with a question and an S/R/I/N answer;
  - `expectation_answers`: the four CCZJV questions with Yes/No/N/A answers;
  - `comments`.

Each text field carries `value`, `status`, `confidence`, `review_reason`, `text_kind` and `engine`.
Each tick box carries `state`, `ink_pixels` and `mark_area_fraction`. `metadata` records the input and
template SHA-256 and the engine versions. It contains no timestamps.

`processing_report.json` gives counts only: page statuses, pages processed, pages needing review,
fully automatic pages, failed-test-form pages, rows per form, and field, group and tick-box statuses.

## Result on the supplied sample

`Sample_input.pdf`, 4 pages, with Tesseract:

| Page | Status | Notes |
|---|---|---|
| 1 | `needs_review` | 8 checklist rows; all 12 choice groups answered; 6 handwritten candidates to confirm |
| 2 | `needs_review` | 8 rows; all 12 groups answered; 6 handwritten candidates |
| 3 | `needs_review` | 9 rows; all 13 groups answered; 6 handwritten candidates |
| 4 | `failed_test_form` | found in the text layer at 45 degrees; nothing extracted; no review needed |

Processing completed on 4/4 pages. Review is required on 3/4. Fully automatic extraction: 0/4, which
is expected, because every form has handwritten fields.

**Consistency check.** The output was compared with a hand transcription of the sample. That
transcription was made by the same developer and **has not been confirmed by an independent
person**, so the counts below are agreement with one reader on three pages. They are not an accuracy
measurement.

| Item | Agreement with the transcription |
|---|---|
| Page classification, rows per page | 4/4, 3/3 |
| Tick-box cells / choice-group selections | 172/172, 43/43 |
| Section titles (printed, `extracted`) | 3/3 |
| Checklist questions (printed, `extracted`) | 23/25 |
| CCZJV questions (printed, `extracted`) | 11/12 |
| Handwritten header candidates (all `needs_review`) | 8/15 |
| Handwritten comments candidates (all `needs_review`) | 2/3 |

The three printed mismatches were all accepted as `extracted` at confidence 0.80 or above:

- a dropped dash in "(e.g. – Line Break Permit)";
- "avalid" for "a valid";
- "Job" for "job".

The handwritten misreads are the reason those fields always need review. Some were wrong digits in a
valid format, for example a timesheet number with one digit changed, or a valid ISO date with the
wrong month. No format check can catch these.

## Limitations

- **One sample.** The thresholds come from three form pages and labelled synthetic cases. The
  sample pages are clean digital composites (no scanner noise, skew, folds or stains), so behaviour
  on real scans is untested. Synthetic tests show how the mechanisms and gates behave; they are not
  evidence of accuracy on real scans.
- **Handwriting.** Tesseract is unreliable on handwriting, and its confidence is not a usable signal:
  wrong values were reported at up to 0.92. Handwritten values are therefore candidates only. The
  sample's "handwriting" is a script font; real pen writing will be harder.
- **Printed text** is accepted on Tesseract's confidence alone, and small errors (spacing, a dropped
  dash, a capital letter) can pass as `extracted`. Use the question text for display and search, not
  as an exact key.
- **Layout assumptions.** Checklist rows must sit in the template's slots; rows of a different height
  are not supported. The double rule between the checklist and the CCZJV table must stay visible
  at the template's resolution, otherwise the template is refused. A long stroke drawn straight through a box may be taken for a ruled line and removed;
  the row then reads as unanswered and is flagged for review.
- **Tick-box thresholds** (empty up to 0.2% of the box, checked 0.5%–8%) are from the sample, where
  marks covered about 1.3% and empty boxes 0%. The sample has no crossed-out, faint or double marks;
  those paths are tested only on synthetic images.
- **Failed-form OCR fallback** applies only to pages with no text layer that fail alignment, and
  always needs review. Any other wording of a failed form is not recognised.
- **Rotation.** A sideways scan of the same form is accepted when it passes every alignment gate.
- **Environment.** Only OpenCV 5.1.0 and Tesseract 5.4.0 have been tested.
- **Optional VLM adapter.** `ocr/vlm.py` documents how a vision-language model could read
  handwriting. It makes no network calls and is not used by default. Anything it returned for
  handwriting would still be `needs_review`.
- **Sensitive output.** The output JSON contains the names and numbers from the forms. Inputs, the
  ground-truth file and debug output are git-ignored; decide separately whether output files may be
  shared.
