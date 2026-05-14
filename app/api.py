"""FastAPI application — HTTP interface for the code review agent.

Endpoints:
    POST /review         — Run a full code review (JSON response)
    POST /review/stream  — Run a review with SSE streaming of agent steps
    POST /webhook        — GitHub webhook receiver for PR events
    GET  /health         — Liveness probe
    GET  /metrics        — Prometheus metrics
"""

from __future__ import annotations

import hashlib
import hmac
import json
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse, StreamingResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from agent.client import MCPClient
from agent.models import AgentEvent
from agent.observability import (
    record_review_start,
    record_review_end,
    record_findings,
    set_agent_info,
)
from agent.reviewer import run_review, run_review_stream
from app.schemas import (
    FindingResponse,
    HealthResponse,
    ReviewRequest,
    ReviewResponse,
)
from config import settings

log = structlog.get_logger()

VERSION = "0.4.0"

# Global MCP client — initialized on startup, closed on shutdown
_mcp_client: MCPClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage MCP server connections on app startup/shutdown."""
    global _mcp_client
    _mcp_client = MCPClient()
    await _mcp_client.connect()
    set_agent_info(VERSION, settings.llm_provider, settings.llm_model)
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

    start = record_review_start()
    try:
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

        record_review_end(start, success=True)
        record_findings(result.findings)
    except Exception:
        record_review_end(start, success=False)
        raise

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


@app.get("/metrics")
async def metrics() -> PlainTextResponse:
    """Prometheus metrics endpoint."""
    return PlainTextResponse(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.post("/webhook")
async def github_webhook(request: Request):
    """Receive GitHub webhook events for pull requests.

    Validates the webhook signature and triggers a review when a PR is
    opened or synchronized (new commits pushed).
    """
    body = await request.body()

    # Validate signature if secret is configured
    if settings.webhook_secret:
        signature = request.headers.get("X-Hub-Signature-256", "")
        expected = "sha256=" + hmac.new(
            settings.webhook_secret.encode(),
            body,
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(signature, expected):
            raise HTTPException(status_code=401, detail="Invalid signature")

    payload = json.loads(body)
    event_type = request.headers.get("X-GitHub-Event", "")

    if event_type != "pull_request":
        return {"status": "ignored", "reason": f"event type: {event_type}"}

    action = payload.get("action", "")
    if action not in ("opened", "synchronize", "reopened"):
        return {"status": "ignored", "reason": f"action: {action}"}

    pr_url = payload.get("pull_request", {}).get("html_url", "")
    if not pr_url:
        raise HTTPException(status_code=400, detail="Missing PR URL in payload")

    log.info("webhook_received", action=action, pr_url=pr_url)

    if not _mcp_client:
        raise HTTPException(status_code=503, detail="MCP servers not connected")

    # Run review asynchronously
    import asyncio
    asyncio.create_task(_webhook_review(pr_url))

    return {"status": "accepted", "pr_url": pr_url}


async def _webhook_review(pr_url: str) -> None:
    """Background task: review a PR triggered by webhook."""
    start = record_review_start()
    try:
        result = await run_review(pr_url=pr_url, mcp_client=_mcp_client)

        # Auto-post review to GitHub
        if result.findings:
            inline_comments = [
                {"path": f.file_path, "line": f.line or 1, "body": f"{f.title}: {f.description}"}
                for f in result.findings
                if f.line is not None
            ]
            await _mcp_client.call_tool(
                "post_review",
                {
                    "pr_url": pr_url,
                    "body": result.summary,
                    "event": result.event,
                    "comments": json.dumps(inline_comments),
                },
            )

        record_review_end(start, success=True)
        record_findings(result.findings)
        log.info("webhook_review_complete", pr_url=pr_url, findings=len(result.findings))
    except Exception as e:
        record_review_end(start, success=False)
        log.error("webhook_review_failed", pr_url=pr_url, error=str(e))
