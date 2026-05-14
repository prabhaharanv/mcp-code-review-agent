"""Intelligence pipeline — orchestrates all enrichment steps on findings.

This is the single entry point for the intelligence layer. The reviewer
calls `run_intelligence_pipeline()` after the agent loop produces raw
findings, and gets back enriched findings with:

    1. Re-classified severity (context-aware)
    2. Cross-file analysis findings added
    3. Concrete fix suggestions attached
    4. Confidence scores computed
    5. RAG-enriched descriptions (for non-trivial findings)

The pipeline is designed to run in ~1-2 seconds for typical PRs (most
steps are pure computation, only RAG enrichment is async/networked).
"""

from __future__ import annotations

import structlog

from agent.client import MCPClient
from agent.intelligence.confidence import score_confidence_batch
from agent.intelligence.cross_file import analyze_cross_file_issues
from agent.intelligence.rag import enrich_findings_batch
from agent.intelligence.severity import classify_batch
from agent.intelligence.suggestions import suggest_fixes_batch
from agent.models import Finding

log = structlog.get_logger()


async def run_intelligence_pipeline(
    findings: list[Finding],
    pr_files: list[dict],
    file_contents: dict[str, str] | None = None,
    mcp_client: MCPClient | None = None,
    changed_files: set[str] | None = None,
) -> list[Finding]:
    """Run the full intelligence pipeline on a set of findings.

    Args:
        findings: Raw findings from the parser/agent
        pr_files: PR file list (from list_pr_files)
        file_contents: Optional map of filename → content for deeper analysis
        mcp_client: Optional MCP client for RAG enrichment
        changed_files: Set of files modified in the PR

    Returns:
        Enriched findings with improved severity, suggestions, and confidence
    """
    contents = file_contents or {}
    changed = changed_files or {f.get("filename", "") for f in pr_files}

    log.info("intelligence_pipeline_start", raw_findings=len(findings))

    # ── Step 1: Cross-file analysis ──────────────────────────
    # Adds NEW findings from multi-file reasoning
    cross_file_findings = analyze_cross_file_issues(pr_files, contents)
    all_findings = findings + cross_file_findings

    log.info(
        "cross_file_analysis_done",
        new_findings=len(cross_file_findings),
        total=len(all_findings),
    )

    # ── Step 2: Severity re-classification ───────────────────
    all_findings = classify_batch(all_findings, contents)

    # ── Step 3: Fix suggestions ──────────────────────────────
    all_findings = suggest_fixes_batch(all_findings, contents)

    # ── Step 4: RAG enrichment (async — calls knowledge base) ─
    all_findings = await enrich_findings_batch(all_findings, mcp_client)

    # ── Step 5: Confidence scoring (runs last, incorporates all signals)
    all_findings = score_confidence_batch(all_findings, changed)

    log.info(
        "intelligence_pipeline_done",
        total_findings=len(all_findings),
        from_tools=len(findings),
        from_cross_file=len(cross_file_findings),
    )

    return all_findings
