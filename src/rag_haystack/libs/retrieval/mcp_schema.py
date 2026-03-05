# src/rag_haystack/libs/retrieval/mcp_schema.py
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Any

class AnswerPayload(BaseModel):
    summary: str = ""
    steps: list[dict[str, Any]] = Field(default_factory=list)
    requirements: list[str] = Field(default_factory=list)
    exceptions: list[str] = Field(default_factory=list)

class Citation(BaseModel):
    doc_id: str
    title: str
    section: str | None = None
    page: str | None = None
    source_uri: str | None = None

class MCPResponse(BaseModel):
    answer: AnswerPayload = Field(default_factory=AnswerPayload)
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = 0.5
    missing_info: list[str] = Field(default_factory=list)
    follow_up_questions: list[str] = Field(default_factory=list)