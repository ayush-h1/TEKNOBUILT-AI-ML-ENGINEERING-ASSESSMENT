"""Command line: python -m field_digitization --input form.pdf --template template.png --output out.json"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import ConfigError, load_thresholds
from .ingest import IngestError
from .layout import LayoutError
from .pipeline import build_context, process_pdf, summarize, write_json
from .template import TemplateError

ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="field_digitization", description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "input" / "Sample_input.pdf")
    parser.add_argument("--template", type=Path, default=ROOT / "data" / "template" / "template.png")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "output" / "extracted_observations.json")
    parser.add_argument("--report", type=Path, default=None,
                        help="processing report path (default: processing_report.json next to --output)")
    parser.add_argument("--thresholds", type=Path, default=None)
    parser.add_argument("--ocr", choices=["auto", "tesseract", "none"], default=None,
                        help="override both OCR provider settings")
    parser.add_argument("--dpi", type=int, default=None, help="override ingest.render_dpi")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    try:
        thresholds = load_thresholds(args.thresholds)
        if args.dpi is not None:
            thresholds = thresholds.model_copy(update={"ingest": thresholds.ingest.model_copy(update={"render_dpi": args.dpi})})
        ctx = build_context(args.template, thresholds=thresholds, ocr=args.ocr)
        result = process_pdf(args.input, ctx)
    except (ConfigError, TemplateError, LayoutError, IngestError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    write_json(result, args.output)
    summary = summarize(result)
    write_json(summary, args.report or args.output.with_name("processing_report.json"))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
