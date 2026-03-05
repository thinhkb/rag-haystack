from __future__ import annotations

import json
import re
from typing import Any

from google import genai
from rag_haystack.libs.retrieval.mcp_schema import MCPResponse

SYSTEM_INSTRUCTIONS = """You are an enterprise SOP assistant.

STRICT RULES:
- Only answer using the EVIDENCE below.
- If evidence is insufficient, add items to "missing_info".
- Output MUST be valid JSON only.
- Do NOT include markdown or explanations outside JSON.
"""

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _extract_json(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""

    # case 1: fenced JSON
    m = _JSON_FENCE_RE.search(text)
    if m:
        return m.group(1).strip()

    # case 2: find first {...} block
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        return text[first:last + 1].strip()

    return text


def _safe_json_loads(text: str) -> dict[str, Any]:
    extracted = _extract_json(text)
    if not extracted:
        raise ValueError("Model returned empty output (no JSON found).")
    return json.loads(extracted)


def _generate(client: genai.Client, model: str, prompt: str) -> str:
    """
    Try JSON response mode if supported; fallback to plain if not.
    """
    # Some SDK versions support config={"response_mime_type":"application/json"}.
    try:
        resp = client.models.generate_content(
            model=model,
            contents=prompt,
            config={"response_mime_type": "application/json"},
        )
        return (resp.text or "").strip()
    except Exception:
        resp = client.models.generate_content(model=model, contents=prompt)
        return (resp.text or "").strip()


def compose_mcp_json(
    *,
    api_key: str,
    model: str,
    question: str,
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    client = genai.Client(api_key=api_key)

    # Build compact evidence
    evidence_blocks = []
    for i, ev in enumerate(evidence, start=1):
        meta = ev.get("meta", {}) or {}
        evidence_blocks.append(
            f"[E{i}] doc_id={meta.get('doc_id')} "
            f"title={meta.get('title')} "
            f"section={meta.get('section_path')} "
            f"page={meta.get('page_start')}-{meta.get('page_end')}\n"
            f"CONTENT:\n{ev.get('content','')}\n"
        )

    # Keep schema simple + stable
    schema_hint = {
        "answer": {
            "summary": "",
            "steps": [],
            "requirements": [],
            "exceptions": []
        },
        "citations": [],
        "confidence": 0.0,
        "missing_info": [],
        "follow_up_questions": []
    }

    prompt = (
        f"{SYSTEM_INSTRUCTIONS}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"EVIDENCE:\n{'\n'.join(evidence_blocks)}\n\n"
        "Return ONLY JSON with this schema (no markdown):\n"
        f"{json.dumps(schema_hint, ensure_ascii=False)}"
    )

    raw = _generate(client, model, prompt)

    # Retry 1: if empty output, regenerate with shorter prompt
    if not raw:
        short_prompt = (
            f"{SYSTEM_INSTRUCTIONS}\n\n"
            f"QUESTION:\n{question}\n\n"
            "Return ONLY JSON with keys: answer,citations,confidence,missing_info,follow_up_questions.\n"
            "If insufficient evidence, fill missing_info.\n"
        )
        raw = _generate(client, model, short_prompt)

    # Try parse
    try:
        data = _safe_json_loads(raw)
    except Exception:
        # Retry 2: repair JSON
        repair_prompt = (
            "Convert the following content into VALID JSON ONLY. "
            "No markdown, no explanation. Must include keys: "
            "answer,citations,confidence,missing_info,follow_up_questions.\n\n"
            f"CONTENT:\n{raw}"
        )
        raw2 = _generate(client, model, repair_prompt)
        data = _safe_json_loads(raw2)
    # Build a lookup from evidence meta by doc_id + page range (best-effort)
    meta_lookup = {}
    for ev in evidence:
        m = ev.get("meta", {}) or {}
        key = (str(m.get("doc_id")), f"{m.get('page_start')}-{m.get('page_end')}")
        meta_lookup[key] = m

    # Fill missing citation fields (best-effort, safe)
    cit_list = data.get("citations", [])
    if not isinstance(cit_list, list):
        cit_list = []

    # Build lookup from evidence meta
    meta_lookup = {}
    for ev in evidence:
        m = ev.get("meta", {}) or {}
        if isinstance(m, dict):
            key = (str(m.get("doc_id")), f"{m.get('page_start')}-{m.get('page_end')}")
            meta_lookup[key] = m

    # Fill missing fields safely
    fixed_citations = []
    for c in cit_list:
        if not isinstance(c, dict):
            # skip non-dict citations (model sometimes returns strings)
            continue

        doc_id = str(c.get("doc_id", ""))
        page = c.get("page")
        key = (doc_id, page) if isinstance(page, str) else None

        if key and key in meta_lookup:
            m = meta_lookup[key]
            if not c.get("title"):
                c["title"] = m.get("title")
            if not c.get("section"):
                c["section"] = m.get("section_path")
            if not c.get("source_uri"):
                c["source_uri"] = m.get("source_uri")

        fixed_citations.append(c)

    data["citations"] = fixed_citations
    # Validate & fill defaults (prevents missing keys causing crashes)
    validated = MCPResponse.model_validate(data)
    return validated.model_dump()