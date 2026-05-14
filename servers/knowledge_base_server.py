"""Knowledge Base MCP Server — wraps the Production Hybrid RAG API as MCP tools.

This server makes HTTP calls to your running RAG instance, turning it into
a tool the agent can use to look up coding standards and review patterns.

Tools:
    search_coding_standards   — Query the RAG knowledge base for coding guidelines
    search_review_patterns    — Query for common code review issues and patterns
"""

from __future__ import annotations

import json
import os

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("knowledge-base")

RAG_URL = os.getenv("RAG_API_URL", "http://localhost:8000")
RAG_API_KEY = os.getenv("RAG_API_KEY", "")


def _rag_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if RAG_API_KEY:
        headers["X-API-Key"] = RAG_API_KEY
    return headers


@mcp.tool()
async def search_coding_standards(query: str) -> str:
    """Search the coding standards knowledge base for guidelines, best practices,
    and style rules relevant to a code review question.

    Args:
        query: Natural language question about coding standards
              (e.g. "What are the rules for error handling in async Python?")
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{RAG_URL}/ask",
            headers=_rag_headers(),
            json={"question": query, "top_k": 3},
        )

        if resp.status_code != 200:
            return json.dumps({"error": f"RAG API returned {resp.status_code}"})

        data = resp.json()

        if data.get("abstained", False):
            return json.dumps({
                "found": False,
                "message": "No relevant coding standards found for this query.",
            })

        return json.dumps(
            {
                "found": True,
                "answer": data["answer"],
                "sources": [
                    {"title": c["title"], "source": c["source"]}
                    for c in data.get("citations", [])
                ],
            },
            indent=2,
        )


@mcp.tool()
async def search_review_patterns(query: str) -> str:
    """Search for common code review patterns, anti-patterns, and past review
    feedback relevant to the current code.

    Args:
        query: Description of the code pattern to look up
              (e.g. "common issues with Python exception handling")
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{RAG_URL}/ask",
            headers=_rag_headers(),
            json={"question": f"code review pattern: {query}", "top_k": 5},
        )

        if resp.status_code != 200:
            return json.dumps({"error": f"RAG API returned {resp.status_code}"})

        data = resp.json()

        if data.get("abstained", False):
            return json.dumps({
                "found": False,
                "message": "No matching review patterns found.",
            })

        return json.dumps(
            {
                "found": True,
                "answer": data["answer"],
                "sources": [
                    {"title": c["title"], "source": c["source"]}
                    for c in data.get("citations", [])
                ],
            },
            indent=2,
        )


if __name__ == "__main__":
    mcp.run()
