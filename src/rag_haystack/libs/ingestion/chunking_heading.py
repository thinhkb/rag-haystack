# src/rag_haystack/libs/ingestion/chunking_heading.py
from __future__ import annotations
from typing import Any
from .parser import is_heading, split_page_to_blocks

def chunk_by_heading(
    pages: list[dict[str, Any]],
    *,
    doc_id: str,
    title: str,
    base_metadata: dict[str, Any],
    max_chars: int = 2500,
) -> list[dict[str, Any]]:
    """
    Build chunks grouped by detected headings. Each chunk keeps page_start/page_end.
    """
    chunks: list[dict[str, Any]] = []

    current_section = None
    current_heading = None
    buf: list[str] = []
    page_start = None
    page_end = None

    def flush():
        nonlocal buf, page_start, page_end, current_section, current_heading
        if not buf:
            return
        text = "\n".join(buf).strip()
        if not text:
            buf = []
            return

        chunks.append({
            "doc_id": doc_id,
            "title": title,
            "text": text,
            "metadata": {
                **(base_metadata or {}),
                "section_path": current_section,
                "section_title": current_heading,
                "page_start": page_start,
                "page_end": page_end,
            }
        })
        buf = []

    for p in pages:
        pnum = int(p["page_num"])
        blocks = split_page_to_blocks(p["text"])

        for line in blocks:
            sec, head = is_heading(line)
            if sec:
                # new heading -> flush previous
                flush()
                current_section = sec
                current_heading = head
                page_start = pnum
                page_end = pnum
                continue

            # normal content
            if page_start is None:
                page_start = pnum
            page_end = pnum
            buf.append(line)

            # keep chunks reasonably sized
            if sum(len(x) for x in buf) >= max_chars:
                flush()
                # continue same section, but next chunk continues
                page_start = pnum
                page_end = pnum

    flush()
    return chunks