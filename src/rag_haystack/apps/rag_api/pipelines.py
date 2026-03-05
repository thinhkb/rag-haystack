from __future__ import annotations
from typing import Any
from haystack import Document, Pipeline
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from rag_haystack.libs.ingestion.metadata import normalize_metadata

try:
    from haystack.components.rankers import SentenceTransformersRanker
except Exception:
    SentenceTransformersRanker = None  # type: ignore

from .config import settings

def build_document_store() -> QdrantDocumentStore:
    return QdrantDocumentStore(
        url=settings.qdrant_url,
        index=settings.qdrant_collection,
        embedding_dim=384,
        recreate_index=False,
    )

class RAGPipelines:
    def __init__(self) -> None:
        self.document_store = build_document_store()
        self.doc_embedder = SentenceTransformersDocumentEmbedder(model=settings.embed_model)
        self.text_embedder = SentenceTransformersTextEmbedder(model=settings.embed_model)
        self.retriever = QdrantEmbeddingRetriever(document_store=self.document_store, top_k=16)

        self.reranker = None
        if SentenceTransformersRanker is not None:
            self.reranker = SentenceTransformersRanker(model=settings.rerank_model, top_k=8)

        self.query_pipeline = self._build_query_pipeline()

    def _build_query_pipeline(self) -> Pipeline:
        p = Pipeline()
        p.add_component("text_embedder", self.text_embedder)
        p.add_component("retriever", self.retriever)
        p.connect("text_embedder.embedding", "retriever.query_embedding")
        return p

    def index_documents(self, docs: list[Document]) -> int:
        embedded = self.doc_embedder.run(documents=docs)["documents"]
        self.document_store.write_documents(embedded)
        return len(embedded)

    def retrieve(self, question: str, *, top_k: int = 8) -> list[Document]:
        # pull more than needed, then filter + rerank, then cut to top_k
        self.retriever.top_k = max(16, top_k * 2)
        out = self.query_pipeline.run({"text_embedder": {"text": question}})
        return out["retriever"]["documents"]

    def rerank(self, question: str, docs: list[Document], *, top_k: int = 8) -> list[Document]:
        if self.reranker is None:
            return docs[:top_k]
        self.reranker.top_k = top_k
        return self.reranker.run(query=question, documents=docs)["documents"]


def to_haystack_documents(payload_docs: list[dict[str, Any]]) -> list[Document]:
    docs: list[Document] = []
    for d in payload_docs:
        meta = normalize_metadata(
            d.get("metadata"),
            fallback_doc_id=d.get("doc_id"),
            fallback_title=d.get("title"),
        )
        meta.update({
            "doc_id": d["doc_id"],
            "title": d["title"],
        })
        docs.append(Document(content=d["text"], meta=meta))
    return docs