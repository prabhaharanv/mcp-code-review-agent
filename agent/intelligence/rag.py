"""RAG integration — enriches findings with knowledge base context.

Uses the Knowledge Base MCP server (which wraps the Production Hybrid RAG API)
to look up relevant coding standards, best practices, and review patterns
for each finding.

This is the bridge between the code review agent and the RAG project:
    Agent finds issue → RAG provides authoritative reference → richer review comment

The enrichment is selective: only non-trivial findings (WARNING+) get enriched,
and results are cached to avoid redundant API calls.
"""

from __future__ import annotations

import structlog

from agent.client import MCPClient
from agent.models import Finding, Severity

log = structlog.get_logger()

# Minimum severity to enrich with RAG (skip nits and praise)
_MIN_SEVERITY_FOR_RAG = Severity.WARNING

# Maximum findings to enrich per review (to control cost/latency)
_MAX_ENRICHMENTS = 10

# Cache: query → response (within a single review session)
_query_cache: dict[str, str] = {}


async def enrich_finding(
    finding: Finding,
    mcp_client: MCPClient,
) -> Finding:
    """Enrich a single finding with knowledge base context.

    Queries the RAG API for coding standards relevant to the finding,
    and appends the reference to the finding's description.

    Args:
        finding: The finding to enrich
        mcp_client: Connected MCP client (with knowledge-base server)

    Returns:
        Finding with enriched description (or unchanged if no relevant standards found)
    """
    query = _build_rag_query(finding)
    if not query:
        return finding

    # Check cache first
    if query in _query_cache:
        rag_response = _query_cache[query]
    else:
        try:
            rag_response = await mcp_client.call_tool(
                "search_coding_standards",
                {"query": query},
            )
            _query_cache[query] = rag_response
        except Exception as e:
            log.warning("rag_enrichment_failed", error=str(e), finding=finding.title)
            return finding

    # Parse and append if relevant
    enrichment = _extract_enrichment(rag_response)
    if enrichment:
        updated_desc = (
            f"{finding.description}\n\n"
            f"**Reference (from coding standards):** {enrichment}"
        )
        return finding.model_copy(update={"description": updated_desc})

    return finding


async def enrich_findings_batch(
    findings: list[Finding],
    mcp_client: MCPClient | None = None,
) -> list[Finding]:
    """Selectively enrich a batch of findings with RAG context.

    Only enriches findings above the minimum severity threshold,
    capped at _MAX_ENRICHMENTS to control API costs.

    Args:
        findings: List of findings to potentially enrich
        mcp_client: Connected MCP client. If None, returns findings unchanged.
    """
    if mcp_client is None:
        return findings

    # Check if knowledge-base server is available
    if "knowledge-base" not in mcp_client.servers:
        log.info("rag_enrichment_skipped", reason="knowledge-base server not connected")
        return findings

    # Clear per-review cache
    _query_cache.clear()

    enrichment_count = 0
    result = []

    for finding in findings:
        if (
            enrichment_count < _MAX_ENRICHMENTS
            and _should_enrich(finding)
        ):
            enriched = await enrich_finding(finding, mcp_client)
            if enriched is not finding:  # enrichment was added
                enrichment_count += 1
            result.append(enriched)
        else:
            result.append(finding)

    log.info("rag_enrichment_complete", enriched=enrichment_count, total=len(findings))
    return result


def _should_enrich(finding: Finding) -> bool:
    """Decide if a finding is worth enriching with RAG context."""
    if finding.severity == Severity.PRAISE:
        return False
    if finding.severity == Severity.NIT:
        return False
    # Skip findings that already have rich descriptions
    if len(finding.description) > 500:
        return False
    return True


def _build_rag_query(finding: Finding) -> str | None:
    """Build a RAG search query from a finding."""
    if finding.tool_source == "run_ruff":
        return f"Python linting best practice: {finding.title}. {finding.description}"

    if finding.tool_source == "run_mypy":
        return f"Python type annotation guideline: {finding.description}"

    if finding.tool_source == "analyze_complexity":
        return f"Code complexity and refactoring: {finding.description}"

    if finding.tool_source == "run_pytest":
        return f"Testing best practice: {finding.title}"

    if finding.tool_source == "cross_file_analysis":
        return f"Code architecture guideline: {finding.description}"

    # Generic query for unknown tools
    return f"Code review guideline: {finding.title}. {finding.description}"


def _extract_enrichment(rag_response: str) -> str | None:
    """Extract useful text from the RAG API response.

    Returns None if the response indicates no relevant standards were found.
    """
    import json

    try:
        data = json.loads(rag_response)
    except (json.JSONDecodeError, TypeError):
        return None

    # Check if RAG found relevant content
    if data.get("found") is False:
        return None

    answer = data.get("answer", "")
    if not answer or len(answer) < 20:
        return None

    # Truncate very long responses
    if len(answer) > 300:
        answer = answer[:297] + "..."

    return answer
