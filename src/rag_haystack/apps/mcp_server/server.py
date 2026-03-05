import os
import httpx
from mcp.server.fastmcp import FastMCP

RAG_API_URL = os.getenv("RAG_API_URL", "http://rag_api:8000")

mcp = FastMCP("enterprise-rag-haystack")

@mcp.tool()
async def search_docs(query: str, top_k: int = 8) -> dict:
    """
    Retrieve relevant chunks (evidence) from indexed SOP documents.
    """
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(f"{RAG_API_URL}/v1/qa", json={
            "question": query,
            "top_k": top_k,
            "response_format": "mcp_json_v1"
        })
        r.raise_for_status()
        data = r.json()
    # Return only evidence-like info (citations + summary)
    return {
        "summary": data["answer"]["summary"],
        "citations": data["citations"],
        "confidence": data["confidence"],
        "trace_id": data.get("trace_id"),
    }

@mcp.tool()
async def index_docs(documents: list[dict]) -> dict:
    """
    Index documents. Each item must contain: doc_id, title, text, metadata (optional).
    """
    payload = {"documents": documents}
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(f"{RAG_API_URL}/v1/index", json=payload)
        r.raise_for_status()
        return r.json()

if __name__ == "__main__":
    # Default FastMCP uses stdio in many setups; run as HTTP if supported in your environment.
    # For now, just start the server (tooling depends on your MCP client).
    mcp.run()