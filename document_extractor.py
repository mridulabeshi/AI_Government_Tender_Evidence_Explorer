"""
Stage 3a — Tender Document Extraction.

Takes a tender document (PDF, or already-downloaded text) and turns it
into page-tagged text chunks ready for LLM structured extraction.

Pipeline:
    PDF -> pdfplumber text extraction
        -> if a page yields ~no text, fall back to PyMuPDF render + OCR
           (OCR itself needs `pytesseract` + the `tesseract-ocr` binary;
           see docstring below on how to enable it)
        -> chunk into ~page-sized units, each tagged with its page number
           so every later finding can cite "Page N".
"""

import os
import fitz  # PyMuPDF
import pdfplumber


class Page:
    __slots__ = ("page_number", "text", "ocr_used")

    def __init__(self, page_number: int, text: str, ocr_used: bool = False):
        self.page_number = page_number
        self.text = text
        self.ocr_used = ocr_used


def _ocr_page(pdf_path: str, page_index: int) -> str:
    """
    OCR fallback for scanned pages. Requires:
        pip install pytesseract pillow
        apt-get install tesseract-ocr
    Kept as an isolated, optional import so the rest of the pipeline works
    even when OCR isn't installed (most tender PDFs are text-based).
    """
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        return ""

    doc = fitz.open(pdf_path)
    page = doc[page_index]
    pix = page.get_pixmap(dpi=200)
    img_path = f"/tmp/_ocr_page_{page_index}.png"
    pix.save(img_path)
    text = pytesseract.image_to_string(Image.open(img_path))
    os.remove(img_path)
    doc.close()
    return text


def extract_pages(pdf_path: str, ocr_min_chars: int = 20) -> list:
    """
    Returns a list of Page objects, one per PDF page, with OCR fallback
    for pages where native text extraction returns almost nothing
    (a strong signal the page is a scanned image).
    """
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            ocr_used = False
            if len(text.strip()) < ocr_min_chars:
                ocr_text = _ocr_page(pdf_path, i)
                if len(ocr_text.strip()) > len(text.strip()):
                    text = ocr_text
                    ocr_used = True
            pages.append(Page(i + 1, text, ocr_used))
    return pages


def extract_pages_from_text(raw_text: str) -> list:
    """
    For tenders whose "document" is really just HTML/plaintext (some
    portals expose the notice inline rather than as a PDF). Splits on
    form-feed or double-newline runs into pseudo-pages so downstream code
    can treat both sources uniformly.
    """
    chunks = [c.strip() for c in raw_text.split("\f") if c.strip()]
    if len(chunks) <= 1:
        chunks = [c.strip() for c in raw_text.split("\n\n\n") if c.strip()]
    return [Page(i + 1, c) for i, c in enumerate(chunks)] or [Page(1, raw_text)]


def chunk_pages(pages: list, max_chars: int = 6000) -> list:
    """
    Groups consecutive pages into chunks under `max_chars` so each chunk
    fits comfortably in an LLM extraction call, while keeping a record of
    which page numbers each chunk spans (needed for clause/page citation).
    Returns list of dicts: {"text": str, "page_start": int, "page_end": int}
    """
    chunks = []
    buf_text, buf_start, buf_end, buf_len = [], None, None, 0

    for page in pages:
        if buf_start is None:
            buf_start = page.page_number
        if buf_len + len(page.text) > max_chars and buf_text:
            chunks.append({
                "text": "\n".join(buf_text),
                "page_start": buf_start,
                "page_end": buf_end,
            })
            buf_text, buf_len = [], 0
            buf_start = page.page_number
        buf_text.append(f"[Page {page.page_number}]\n{page.text}")
        buf_len += len(page.text)
        buf_end = page.page_number

    if buf_text:
        chunks.append({
            "text": "\n".join(buf_text),
            "page_start": buf_start,
            "page_end": buf_end,
        })

    return chunks


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.document_extractor <path_to_pdf>")
        sys.exit(0)
    pages = extract_pages(sys.argv[1])
    print(f"Extracted {len(pages)} pages")
    for p in pages[:2]:
        print(f"--- Page {p.page_number} (ocr={p.ocr_used}) ---")
        print(p.text[:300])
