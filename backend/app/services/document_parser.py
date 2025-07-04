import fitz  # PyMuPDF
from typing import List, Dict

def extract_text_from_pdf(file_path: str) -> List[Dict]:
    """
    Extracts text from a PDF and returns a list of pages with metadata.

    Returns:
        List[Dict] - Each item contains page number and its text content.
    """
    doc = fitz.open(file_path)
    pages = []

    for page_num, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            pages.append({
                "page_number": page_num + 1,
                "text": text.strip()
            })

    doc.close()
    return pages
