"""Generate a reproducible test PDF for the BluePebble OCR smoke test.

Renders a chart-like PNG (large, OCR-friendly text: uppercase words + digits,
no underscores which OCR mangles) with PIL, then embeds it into a one-page PDF
that also has real body text, using fpdf2. The embedded raster is what makes
Docling classify it as a *picture* (exercising extract_pictures -> caption ->
in-figure OCR), rather than a full-page scan.

The figure carries a distinctive sentinel (``REVENUE 2024 42 PERCENT``) that does
NOT appear in the body text, so finding it inside a document's "Text in image:"
section proves the OCR path ran. Run from the repo root:

    PYTHONPATH=src python scripts/make_ocr_smoke_pdf.py
"""

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

OUTPUT_DIR = Path("runs/ocr-smoke/input")
PNG_PATH = OUTPUT_DIR / "figure.png"
PDF_PATH = OUTPUT_DIR / "figure_report.pdf"

# Sentinel lines drawn INTO the figure only (never in the PDF body text).
FIGURE_TITLE = "QUARTERLY REVENUE REPORT"
FIGURE_LINES = ("REVENUE 2024 42 PERCENT", "GROWTH 2023 TO 2024 18 PERCENT")

BODY_TEXT = (
    "Annual results overview. The figure below summarises the reporting period; "
    "all headline metrics are rendered inside the embedded chart image so that "
    "the ingestion pipeline must read them from the picture, not from this text."
)


def _font(size: int) -> Any:
    """A large, scalable font for OCR-legible text, with fallbacks by Pillow age."""
    try:
        return ImageFont.load_default(size=size)  # Pillow >= 10.1
    except TypeError:
        pass
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _render_figure() -> None:
    image = Image.new("RGB", (960, 640), "white")
    draw = ImageDraw.Draw(image)
    draw.text((40, 30), FIGURE_TITLE, fill="black", font=_font(52))

    # A few labelled bars so it reads as a chart, not a text box.
    bars = ((160, 300), (320, 220), (480, 380), (640, 260))
    for index, (x, top) in enumerate(bars, start=1):
        draw.rectangle([x, top, x + 90, 520], fill="black")
        draw.text((x + 10, 530), f"Q{index}", fill="black", font=_font(34))

    for offset, line in enumerate(FIGURE_LINES):
        draw.text((40, 580 + offset * 30), line, fill="black", font=_font(30))

    image.save(PNG_PATH)


def _build_pdf() -> None:
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=16)
    pdf.cell(0, 12, "Evidence RAG - OCR smoke sample")
    pdf.ln(16)
    pdf.set_font("Helvetica", size=12)
    pdf.multi_cell(0, 7, BODY_TEXT)
    pdf.ln(6)
    pdf.image(str(PNG_PATH), w=170)
    pdf.output(str(PDF_PATH))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _render_figure()
    _build_pdf()
    print(f"wrote {PDF_PATH} (embedded figure sentinel: {FIGURE_LINES[0]!r})")


if __name__ == "__main__":
    main()
