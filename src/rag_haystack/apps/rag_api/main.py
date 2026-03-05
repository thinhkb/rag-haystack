from __future__ import annotations
from rag_haystack.libs.retrieval.gemini_composer import compose_mcp_json
import uuid
from fastapi import FastAPI, HTTPException
from .schemas import IndexRequest, QARequest, QAResponse, Citation, AnswerPayload
from .pipelines import RAGPipelines, to_haystack_documents
from .config import settings
from .security import build_access_predicate
from rag_haystack.libs.ingestion.metadata import normalize_metadata
from fastapi import UploadFile, File, Form
import tempfile
import os

from rag_haystack.libs.ingestion.loader import load_pdf_pages
from rag_haystack.libs.ingestion.chunking_heading import chunk_by_heading

app = FastAPI(title=settings.app_name)
pipelines = RAGPipelines()

def build_citations(docs):
    citations = []
    seen = set()
    for doc in docs:
        meta = doc.meta or {}
        doc_id = str(meta.get("doc_id", ""))
        title = str(meta.get("title", ""))
        section = meta.get("section_path")
        page = None
        if meta.get("page_start") is not None:
            page = f'{meta.get("page_start")}-{meta.get("page_end")}'
        source_uri = meta.get("source_uri")

        key = (doc_id, section, page, source_uri)
        if key in seen:
            continue
        seen.add(key)

        citations.append(Citation(
            doc_id=doc_id,
            title=title,
            section=str(section) if section else None,
            page=page,
            source_uri=str(source_uri) if source_uri else None,
        ))
    return citations

def extractive_answer(question: str, docs) -> AnswerPayload:
    # MVP fallback (no LLM): return top passage as "summary"
    if not docs:
        return AnswerPayload(
            summary="Không tìm thấy nội dung phù hợp trong tài liệu đã index.",
            steps=[],
            requirements=[],
            exceptions=[],
        )
    top = docs[0].content.strip()
    summary = top[:600] + ("..." if len(top) > 600 else "")
    return AnswerPayload(summary=summary)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/v1/index")
def index_docs(req: IndexRequest):
    try:
        normalized = []
        for d in req.documents:
            dd = d.model_dump()
            dd["metadata"] = normalize_metadata(
                dd.get("metadata"),
                fallback_doc_id=dd.get("doc_id"),
                fallback_title=dd.get("title"),
            )
            normalized.append(dd)

        docs = to_haystack_documents(normalized)
        n = pipelines.index_documents(docs)
        return {"indexed": n}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/index_files")
async def index_files(
    doc_id: str = Form(...),
    title: str = Form(...),

    # ABAC fields (form)
    department: str = Form("UNKNOWN"),
    confidentiality_level: str = Form("restricted"),
    allowed_roles: str = Form(""),  # optional: "staff,manager"
    source_uri: str = Form(""),

    file: UploadFile = File(...),
):
    suffix = os.path.splitext(file.filename or "")[1].lower()

    if suffix not in [".pdf"]:
        raise HTTPException(status_code=400, detail="MVP currently supports PDF only. Upload a .pdf file.")

    # save to temp
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Normalize ABAC metadata (safe defaults)
        base_meta = normalize_metadata({
            "department": department,
            "confidentiality_level": confidentiality_level,
            "allowed_roles": allowed_roles,   # normalize_metadata will convert str -> list
            "source_uri": source_uri or None,
        }, fallback_doc_id=doc_id, fallback_title=title)

        pages = load_pdf_pages(tmp_path)
        if not pages:
            raise HTTPException(status_code=400, detail="PDF has no extractable text (might be scanned).")

        chunks = chunk_by_heading(pages, doc_id=doc_id, title=title, base_metadata=base_meta)

        # reuse your existing indexing path
        docs = to_haystack_documents(chunks)
        n = pipelines.index_documents(docs)

        return {"indexed": n, "doc_id": doc_id, "filename": file.filename}
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

@app.post("/v1/qa", response_model=QAResponse)
def qa(req: QARequest):
    trace_id = str(uuid.uuid4())

    # 1) retrieve candidates
    candidates = pipelines.retrieve(req.question, top_k=req.top_k)

    # 2) ABAC filter (no leak)
    pred = build_access_predicate(user_context=req.user_context, filters=req.filters)
    filtered = [d for d in candidates if pred(d.meta or {})]

    # 3) rerank only on allowed docs
    docs = pipelines.rerank(req.question, filtered, top_k=req.top_k)

    citations = build_citations(docs)

    if not docs:
        return QAResponse(
            answer=AnswerPayload(summary="Không tìm thấy thông tin phù hợp trong phạm vi tài liệu bạn có quyền truy cập."),
            citations=[],
            confidence=0.0,
            missing_info=["Không có evidence phù hợp sau khi áp dụng policy/filters."],
            follow_up_questions=["Bạn thuộc phòng ban nào và cần SOP nào (mã SOP)?" ],
            trace_id=trace_id,
        )

    # Chuẩn bị evidence cho composer
    # Limit evidence to avoid long prompts => Gemini empty/non-JSON output
    MAX_EVIDENCE = 5
    MAX_CHARS_PER_EVIDENCE = 1500

    evidence = []
    for d in docs[:MAX_EVIDENCE]:
        content = (d.content or "").strip()
        if len(content) > MAX_CHARS_PER_EVIDENCE:
            content = content[:MAX_CHARS_PER_EVIDENCE] + "..."
        evidence.append({"content": content, "meta": d.meta or {}})

    if settings.gemini_api_key:
        try:
            out = compose_mcp_json(
                api_key=settings.gemini_api_key,
                model=settings.gemini_model,
                question=req.question,
                evidence=evidence,
            )
            return QAResponse(
                answer=AnswerPayload(**out["answer"]),
                citations=[Citation(**c) for c in out.get("citations", [])],
                confidence=float(out.get("confidence", 0.5)),
                missing_info=out.get("missing_info", []),
                follow_up_questions=out.get("follow_up_questions", []),
                trace_id=trace_id,
            )
        except Exception as e:
            # fallback extractive => never 500
            answer = extractive_answer(req.question, docs)
            return QAResponse(
                answer=answer,
                citations=citations,
                confidence=0.3,
                missing_info=[f"LLM composer failed: {type(e).__name__}: {str(e)}"],
                follow_up_questions=[],
                trace_id=trace_id,
            )

    # fallback (no key)
    answer = extractive_answer(req.question, docs)
    confidence = min(0.9, 0.4 + 0.1 * len(docs))
    return QAResponse(
        answer=answer,
        citations=citations,
        confidence=confidence,
        missing_info=[],
        follow_up_questions=[],
        trace_id=trace_id,
    )

