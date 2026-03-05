# libs/ingestion/chunking.py
from __future__ import annotations
from typing import Any

def chunk_by_pages(
    pages: list[dict[str, Any]],
    *,
    doc_id: str,
    title: str,
    base_metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    MVP: each page = one chunk
    """
    chunks = []
    for p in pages:
        chunks.append({
            "doc_id": doc_id,
            "title": title,
            "text": p["text"],
            "metadata": {
                **(base_metadata or {}),
                "page_start": p["page_num"],
                "page_end": p["page_num"],
                "section_path": None,  # v1: not parsing headings yet
            }
        })
    return chunks