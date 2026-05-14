"""FastAPI application — HTTP interface for the code review agent.

Endpoints:
    POST /review         — Run a full code review (JSON response)
    POST /review/stream  — Run a review with SSE streaming of agent steps
    GET  /health         — Liveness probe
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from agent.client import MCPClient
from agent.models import AgentEvent
from agent.reviewer import run_review, run_review_stream
from app.schemas import (
    FindingResponse,
    HealthResponse,
    ReviewRequest,
    ReviewResponse,
)
from config import settings

log = structlog.get_logger()

VERSION = "0.2.0"

# Global MCP client — initialized on startup, closed on shutdown
_mcp_client: MCPClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage MCP server connections on app startup/shutdown."""
    global _mcp_client
    _mcp_client = MCPClient()
    await _mcp_client.connect()
    log.info("mcp_servers_connected", servers=list(_mcp_client.servers.keys()))
    yield
    await _mcp_client.close()
    _mcp_client = None
    log.info("mcp_servers_disconnected")


app = FastAPI(
    title="MCP Code Review Agent",
    description="Autonomous code review agent using Model Context Protocol",
    version=VERSION,
    lifespan=lifespan,
)


@app.post("/review", response_model=ReviewResponse)
async def review_pr(request: ReviewRequest) -> ReviewResponse:
    """Run a full code review on a GitHub PR.

    The agent autonomously:
    1. Reads the PR diff and metadata
    2. Plans which files and checks to run
    3. Executes linters, type checkers, and complexity analysis
    4. Synthesizes findings into a structured review
    """
    if not _mcp_client:
        raise HTTPException(status_code=503, detail="MCP servers not connected")

    result = await run_review(
        pr_url=request.pr_url,
        mcp_client=_mcp_client,
        max_steps=request.max_steps,
    )

    # Optionally post the review to GitHub
    if request.post_review and result.findings:
        try:
            inline_comments = [
                {"path": f.file_path, "line": f.line or 1, "body": f"{f.title}: {f.description}"}
                for f in result.findings
                if f.line is not None
            ]
            await _mcp_client.call_tool(
                "post_review",
                {
                    "pr_url": request.pr_url,
                    "body": result.summary,
                    "event": result.event,
                    "comments": json.dumps(inline_comments),
                },
            )
        except Exception as e:
            log.error("failed_to_post_review", error=str(e))

    return ReviewResponse(
        pr_url=result.pr_url,
        pr_title=result.pr_title,
        summary=result.summary,
        findings=[
            FindingResponse(
                severity=f.severity,
                file_path=f.file_path,
                line=f.line,
                title=f.title,
                description=f.description,
                suggestion=f.suggestion,
            )
            for f in result.findings
        ],
        event=result.event,
        plan=result.plan,
        stats={
            "steps_taken": result.steps_taken,
            "total_tool_calls": result.total_tool_calls,
            "duration_seconds": result.duration_seconds,
            "blockers": result.blocker_count,
            "warnings": result.warning_count,
            "nits": result.nit_count,
            "praise": result.praise_count,
        },
    )


@app.post("/review/stream")
async def review_pr_stream(request: ReviewRequest) -> StreamingResponse:
    """Run a code review and stream agent events via SSE.

    Events:
        thinking  — agent is processing
        plan      — review plan created
        tool_call — agent calling an MCP tool
        finding   — structured finding extracted
        done      — review complete with summary
    """
    if not _mcp_client:
        raise HTTPException(status_code=503, detail="MCP servers not connected")

    async def event_generator():
        async for event in run_review_stream(
            pr_url=request.pr_url,
            mcp_client=_mcp_client,
            max_steps=request.max_steps,
        ):
            data = json.dumps({"event": event.event_type, "step": event.step, **event.data})
            yield f"data: {data}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness probe — is the agent running?"""
    servers = list(_mcp_client.servers.keys()) if _mcp_client else []
    return HealthResponse(
        status="ok" if _mcp_client else "degraded",
        version=VERSION,
        servers=servers,
    )
