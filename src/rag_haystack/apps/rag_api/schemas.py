from pydantic import BaseModel, Field
from typing import Any, Literal

class IndexDoc(BaseModel):
    doc_id: str = Field(..., description="SOP id or unique doc id")
    title: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)

class IndexRequest(BaseModel):
    documents: list[IndexDoc]

class QARequest(BaseModel):
    question: str
    user_context: dict[str, Any] = Field(default_factory=dict)
    filters: dict[str, Any] = Field(default_factory=dict)
    top_k: int = 8
    response_format: Literal["mcp_json_v1"] = "mcp_json_v1"

class Citation(BaseModel):
    doc_id: str
    title: str
    section: str | None = None
    page: str | None = None
    source_uri: str | None = None

class AnswerPayload(BaseModel):
    summary: str
    steps: list[dict[str, Any]] = Field(default_factory=list)
    requirements: list[str] = Field(default_factory=list)
    exceptions: list[str] = Field(default_factory=list)

class QAResponse(BaseModel):
    answer: AnswerPayload
    citations: list[Citation]
    confidence: float
    missing_info: list[str] = Field(default_factory=list)
    follow_up_questions: list[str] = Field(default_factory=list)
    trace_id: str | None = None