"""Reviewer — high-level orchestrator that ties planning, agent loop, and parsing.

Flow:
    1. Fetch PR metadata + file list (deterministic)
    2. Build a ReviewPlan (deterministic heuristics)
    3. Inject plan into agent context
    4. Run agent loop (LLM-driven, autonomous)
    5. Parse structured findings from agent observations
    6. Produce final ReviewResult
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from datetime import datetime, timezone

import structlog

from agent.budget import TokenBudget
from agent.client import MCPClient
from agent.core import ReviewAgent
from agent.intelligence.pipeline import run_intelligence_pipeline
from agent.models import AgentEvent, Finding, ReviewResult
from agent.observability import TraceContext, record_intelligence_duration
from agent.parser import parse_tool_output
from agent.planner import create_review_plan, plan_to_prompt_context
from config import settings

log = structlog.get_logger()


async def run_review(
    pr_url: str,
    mcp_client: MCPClient,
    max_steps: int = 20,
) -> ReviewResult:
    """Run a full code review: plan → agent loop → structured result.

    Args:
        pr_url: GitHub PR URL
        mcp_client: Connected MCP client
        max_steps: Maximum agent steps

    Returns:
        Structured ReviewResult with findings
    """
    import uuid
    import time
    result = ReviewResult(pr_url=pr_url)
    trace = TraceContext(trace_id=str(uuid.uuid4()), pr_url=pr_url)
    budget = TokenBudget(
        max_input_tokens=settings.max_input_tokens,
        max_output_tokens=settings.max_output_tokens,
        max_total_tokens=settings.max_total_tokens,
    )

    # ── Step 1: Fetch PR context ─────────────────────────────
    log.info("fetching_pr_context", pr_url=pr_url)

    metadata_raw = await mcp_client.call_tool("get_pr_metadata", {"pr_url": pr_url})
    files_raw = await mcp_client.call_tool("list_pr_files", {"pr_url": pr_url})

    try:
        metadata = json.loads(metadata_raw)
        files = json.loads(files_raw)
    except json.JSONDecodeError as e:
        log.error("failed_to_parse_pr_context", error=str(e))
        result.summary = f"Failed to fetch PR context: {e}"
        result.completed_at = datetime.now(timezone.utc)
        return result

    result.pr_title = metadata.get("title", "")

    # ── Step 2: Build review plan ────────────────────────────
    plan = create_review_plan(metadata, files)
    result.plan = plan
    log.info(
        "review_plan_created",
        files_to_analyze=len(plan.files_to_analyze),
        checks=plan.checks_to_run,
        risks=len(plan.risk_areas),
    )

    # ── Step 3: Run agent with plan context ──────────────────
    agent = ReviewAgent(mcp_client=mcp_client, max_steps=max_steps, budget=budget)
    plan_context = plan_to_prompt_context(plan)
    async with trace.span("agent_loop", max_steps=max_steps):
        review_text = await agent.review_with_plan(pr_url, plan_context)

    result.raw_review_text = review_text
    result.steps_taken = len(agent.steps)
    result.total_tool_calls = len(agent.steps)

    # ── Step 4: Extract structured findings from observations ─
    raw_findings: list[Finding] = []
    file_contents: dict[str, str] = {}

    for step in agent.steps:
        if step.tool_name and step.tool_result:
            # Determine file context from tool args
            file_path = ""
            if step.tool_args:
                file_path = step.tool_args.get("file_path", step.tool_args.get("filename", ""))

            findings = parse_tool_output(step.tool_name, step.tool_result, file_path)
            raw_findings.extend(findings)

            # Collect file contents for cross-file analysis
            if step.tool_name == "get_file_contents" and file_path:
                file_contents[file_path] = step.tool_result

    # ── Step 5: Intelligence pipeline ────────────────────────
    intel_start = time.monotonic()
    result.findings = await run_intelligence_pipeline(
        findings=raw_findings,
        pr_files=files,
        file_contents=file_contents,
        mcp_client=mcp_client,
        changed_files={f.get("filename", "") for f in files},
    )

    record_intelligence_duration(time.monotonic() - intel_start)

    # ── Step 6: Determine review event ───────────────────────
    if result.blocker_count > 0:
        result.event = "REQUEST_CHANGES"
    elif result.warning_count == 0 and result.nit_count == 0:
        result.event = "COMMENT"  # Clean PR, but don't auto-approve
    else:
        result.event = "COMMENT"

    result.summary = review_text
    result.completed_at = datetime.now(timezone.utc)

    log.info(
        "review_complete",
        findings=len(result.findings),
        blockers=result.blocker_count,
        warnings=result.warning_count,
        nits=result.nit_count,
        praise=result.praise_count,
        duration_s=result.duration_seconds,
    )

    return result


async def run_review_stream(
    pr_url: str,
    mcp_client: MCPClient,
    max_steps: int = 20,
) -> AsyncGenerator[AgentEvent, None]:
    """Run a review and yield events as they happen (for SSE streaming).

    Yields AgentEvent objects for each phase:
        - plan: the review plan
        - tool_call: each tool invocation
        - tool_result: each tool response
        - finding: each structured finding extracted
        - done: final result
    """
    result = ReviewResult(pr_url=pr_url)

    # ── Fetch context ────────────────────────────────────────
    yield AgentEvent(event_type="thinking", step=0, data={"message": "Fetching PR metadata..."})

    metadata_raw = await mcp_client.call_tool("get_pr_metadata", {"pr_url": pr_url})
    files_raw = await mcp_client.call_tool("list_pr_files", {"pr_url": pr_url})

    try:
        metadata = json.loads(metadata_raw)
        files = json.loads(files_raw)
    except json.JSONDecodeError as e:
        yield AgentEvent(event_type="error", data={"message": f"Failed to parse PR: {e}"})
        return

    result.pr_title = metadata.get("title", "")

    # ── Plan ─────────────────────────────────────────────────
    plan = create_review_plan(metadata, files)
    result.plan = plan

    yield AgentEvent(
        event_type="plan",
        step=0,
        data={
            "summary": plan.summary,
            "files": plan.files_to_analyze,
            "checks": plan.checks_to_run,
            "risks": plan.risk_areas,
            "estimated_steps": plan.estimated_steps,
        },
    )

    # ── Agent loop with streaming ────────────────────────────
    agent = ReviewAgent(mcp_client=mcp_client, max_steps=max_steps)
    plan_context = plan_to_prompt_context(plan)
    review_text = await agent.review_with_plan(pr_url, plan_context)

    # Emit tool call events and collect raw findings
    raw_findings: list[Finding] = []
    file_contents: dict[str, str] = {}

    for step in agent.steps:
        yield AgentEvent(
            event_type="tool_call",
            step=step.step,
            data={
                "tool": step.tool_name or "",
                "args": list((step.tool_args or {}).keys()),
            },
        )

        if step.tool_name and step.tool_result:
            file_path = ""
            if step.tool_args:
                file_path = step.tool_args.get("file_path", step.tool_args.get("filename", ""))

            findings = parse_tool_output(step.tool_name, step.tool_result, file_path)
            raw_findings.extend(findings)

            if step.tool_name == "get_file_contents" and file_path:
                file_contents[file_path] = step.tool_result

    # ── Intelligence pipeline ────────────────────────────────
    yield AgentEvent(event_type="thinking", data={"message": "Running intelligence pipeline..."})

    enriched_findings = await run_intelligence_pipeline(
        findings=raw_findings,
        pr_files=files,
        file_contents=file_contents,
        mcp_client=mcp_client,
        changed_files={f.get("filename", "") for f in files},
    )
    result.findings = enriched_findings

    for finding in enriched_findings:
        yield AgentEvent(
            event_type="finding",
            step=0,
            data=finding.model_dump(),
        )

    # ── Done ─────────────────────────────────────────────────
    result.summary = review_text
    result.raw_review_text = review_text
    result.steps_taken = len(agent.steps)
    result.completed_at = datetime.now(timezone.utc)

    yield AgentEvent(
        event_type="done",
        data={
            "summary": result.summary,
            "findings_count": len(result.findings),
            "blockers": result.blocker_count,
            "warnings": result.warning_count,
            "nits": result.nit_count,
            "duration_s": result.duration_seconds,
        },
    )
