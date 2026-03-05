# src/rag_haystack/libs/ingestion/parser.py
from __future__ import annotations
import re
from typing import Any

HEADING_PATTERNS = [
    re.compile(r"^\s*(\d+(\.\d+){0,4})\s+(.+?)\s*$"),        # 1 / 1.2 / 1.2.3 Title
    re.compile(r"^\s*(SECTION|CHAPTER)\s+(\d+)\s*[:\-]?\s*(.+)?\s*$", re.I),
]

def is_heading(line: str) -> tuple[str | None, str | None]:
    s = (line or "").strip()
    if not s:
        return (None, None)

    for pat in HEADING_PATTERNS:
        m = pat.match(s)
        if m:
            # Return a normalized section_path and heading text
            if m.group(1).strip().upper() in ["SECTION", "CHAPTER"]:
                sec = f"{m.group(1).strip().upper()} {m.group(2)}"
                title = (m.group(3) or "").strip() or sec
                return (sec, title)
            else:
                sec = m.group(1).strip()
                title = m.group(3).strip()
                return (sec, title)

    # ALL CAPS heuristic
    if s.isupper() and 6 <= len(s) <= 80:
        return ("HEADING", s)

    return (None, None)

def split_page_to_blocks(page_text: str) -> list[str]:
    # split into lines and keep non-empty
    lines = [ln.strip() for ln in (page_text or "").splitlines()]
    return [ln for ln in lines if ln]