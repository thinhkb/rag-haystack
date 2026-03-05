# libs/ingestion/loader.py
from __future__ import annotations
from typing import Any
from pypdf import PdfReader

def load_pdf_pages(file_path: str) -> list[dict[str, Any]]:
    """
    Return list of pages: [{"page_num":1, "text":"..."}, ...]
    page_num is 1-indexed.
    """
    reader = PdfReader(file_path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.replace("\x00", "").strip()
        if text:
            pages.append({"page_num": i, "text": text})
    return pages