# libs/ingestion/metadata.py
from __future__ import annotations

from typing import Any
from datetime import date

DEFAULT_DEPARTMENT = "UNKNOWN"
DEFAULT_CONFIDENTIALITY = "restricted"  # safest default
ALLOWED_CONFIDENTIALITY = {"public", "internal", "restricted"}

def normalize_metadata(
    metadata: dict[str, Any] | None,
    *,
    fallback_doc_id: str | None = None,
    fallback_title: str | None = None,
) -> dict[str, Any]:
    """
    Ensure ABAC/RBAC-required metadata exists and is well-formed.
    Production rule: missing fields -> conservative defaults.
    """
    m: dict[str, Any] = dict(metadata or {})

    # Required for traceability
    if fallback_doc_id and "doc_id" not in m:
        m["doc_id"] = fallback_doc_id
    if fallback_title and "title" not in m:
        m["title"] = fallback_title

    # ABAC required fields
    dept = str(m.get("department") or DEFAULT_DEPARTMENT).upper().strip()
    m["department"] = dept if dept else DEFAULT_DEPARTMENT

    conf = str(m.get("confidentiality_level") or DEFAULT_CONFIDENTIALITY).lower().strip()
    if conf not in ALLOWED_CONFIDENTIALITY:
        conf = DEFAULT_CONFIDENTIALITY
    m["confidentiality_level"] = conf

    # Optional fields (normalize types)
    if "allowed_roles" in m and isinstance(m["allowed_roles"], str):
        # allow comma-separated string
        m["allowed_roles"] = [r.strip().lower() for r in m["allowed_roles"].split(",") if r.strip()]

    # Version/effective
    # Keep as-is if provided; you can enforce format later.
    # Example: m["effective_date"] = "2025-01-01"
    # Example: m["version"] = "v1.2"

    return m