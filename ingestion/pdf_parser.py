import re
import fitz  # PyMuPDF

def _normalize_whitespace(text: str) -> str:
    """
    Clean up common PDF-extraction whitespace artifacts:
    - non-breaking spaces -> regular spaces
    - collapse runs of spaces/tabs
    - collapse 3+ newlines down to 2 (preserve paragraph breaks)
    """
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def extract_pages(pdf_path: str) -> list[dict]:
    """
    Extract text from a PDF, page by page.

    Returns a list of dicts: [{"page_number": int, "text": str}, ...]
    page_number is 1-indexed (matches how humans refer to pages).
    """
    pages = []
    doc = fitz.open(pdf_path)
    for i, page in enumerate(doc):
        text = _normalize_whitespace(page.get_text())
        pages.append({"page_number": i + 1, "text": text})
    doc.close()
    return pages