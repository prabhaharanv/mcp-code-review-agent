"""Tests for the RAG integration module."""

from unittest.mock import AsyncMock, MagicMock
import json
import pytest

from agent.intelligence.rag import (
    enrich_finding,
    enrich_findings_batch,
    _build_rag_query,
    _extract_enrichment,
    _should_enrich,
)
from agent.models import Finding, Severity


def _make_finding(**kwargs) -> Finding:
    defaults = {
        "severity": Severity.WARNING,
        "file_path": "app.py",
        "line": 10,
        "title": "Test finding",
        "description": "Something happened",
        "tool_source": "run_ruff",
    }
    defaults.update(kwargs)
    return Finding(**defaults)


class TestShouldEnrich:
    def test_enriches_warning(self):
        assert _should_enrich(_make_finding(severity=Severity.WARNING))

    def test_enriches_blocker(self):
        assert _should_enrich(_make_finding(severity=Severity.BLOCKER))

    def test_skips_nit(self):
        assert not _should_enrich(_make_finding(severity=Severity.NIT))

    def test_skips_praise(self):
        assert not _should_enrich(_make_finding(severity=Severity.PRAISE))

    def test_skips_long_description(self):
        assert not _should_enrich(_make_finding(description="x" * 501))


class TestBuildRagQuery:
    def test_ruff_query(self):
        finding = _make_finding(tool_source="run_ruff", title="ruff E711")
        query = _build_rag_query(finding)
        assert "linting" in query.lower()

    def test_mypy_query(self):
        finding = _make_finding(tool_source="run_mypy")
        query = _build_rag_query(finding)
        assert "type annotation" in query.lower()

    def test_complexity_query(self):
        finding = _make_finding(tool_source="analyze_complexity")
        query = _build_rag_query(finding)
        assert "complexity" in query.lower()

    def test_pytest_query(self):
        finding = _make_finding(tool_source="run_pytest")
        query = _build_rag_query(finding)
        assert "testing" in query.lower()

    def test_cross_file_query(self):
        finding = _make_finding(tool_source="cross_file_analysis")
        query = _build_rag_query(finding)
        assert "architecture" in query.lower()

    def test_unknown_tool_generic_query(self):
        finding = _make_finding(tool_source="custom_tool")
        query = _build_rag_query(finding)
        assert "code review" in query.lower()


class TestExtractEnrichment:
    def test_extracts_answer(self):
        response = json.dumps({"found": True, "answer": "Use parameterized queries for SQL."})
        result = _extract_enrichment(response)
        assert result == "Use parameterized queries for SQL."

    def test_returns_none_when_not_found(self):
        response = json.dumps({"found": False, "message": "No results"})
        assert _extract_enrichment(response) is None

    def test_returns_none_for_short_answer(self):
        response = json.dumps({"answer": "short"})
        assert _extract_enrichment(response) is None

    def test_truncates_long_answer(self):
        response = json.dumps({"answer": "x" * 500})
        result = _extract_enrichment(response)
        assert len(result) <= 300

    def test_handles_invalid_json(self):
        assert _extract_enrichment("not json") is None

    def test_handles_none(self):
        assert _extract_enrichment(None) is None


class TestEnrichFinding:
    @pytest.mark.asyncio
    async def test_enriches_with_rag_response(self):
        mcp_client = MagicMock()
        mcp_client.call_tool = AsyncMock(return_value=json.dumps({
            "found": True,
            "answer": "Always use parameterized queries to prevent SQL injection.",
        }))

        finding = _make_finding(
            severity=Severity.BLOCKER,
            description="SQL injection risk",
        )
        result = await enrich_finding(finding, mcp_client)
        assert "Reference (from coding standards)" in result.description
        assert "parameterized" in result.description

    @pytest.mark.asyncio
    async def test_returns_unchanged_on_no_results(self):
        mcp_client = MagicMock()
        mcp_client.call_tool = AsyncMock(return_value=json.dumps({
            "found": False,
            "message": "No results",
        }))

        finding = _make_finding()
        result = await enrich_finding(finding, mcp_client)
        assert "Reference" not in result.description

    @pytest.mark.asyncio
    async def test_handles_mcp_error_gracefully(self):
        mcp_client = MagicMock()
        mcp_client.call_tool = AsyncMock(side_effect=Exception("Connection failed"))

        finding = _make_finding()
        result = await enrich_finding(finding, mcp_client)
        assert result is finding  # unchanged


class TestEnrichFindingsBatch:
    @pytest.mark.asyncio
    async def test_returns_unchanged_without_client(self):
        findings = [_make_finding(), _make_finding()]
        result = await enrich_findings_batch(findings, mcp_client=None)
        assert result is findings

    @pytest.mark.asyncio
    async def test_skips_when_no_kb_server(self):
        mcp_client = MagicMock()
        mcp_client.servers = {}  # no knowledge-base server
        findings = [_make_finding()]
        result = await enrich_findings_batch(findings, mcp_client)
        assert result is findings

    @pytest.mark.asyncio
    async def test_enriches_only_warnings_and_above(self):
        mcp_client = MagicMock()
        mcp_client.servers = {"knowledge-base": MagicMock()}
        mcp_client.call_tool = AsyncMock(return_value=json.dumps({
            "found": True,
            "answer": "This is a relevant coding standard reference here.",
        }))

        findings = [
            _make_finding(severity=Severity.NIT),       # skipped
            _make_finding(severity=Severity.PRAISE),     # skipped
            _make_finding(severity=Severity.WARNING),    # enriched
            _make_finding(severity=Severity.BLOCKER),    # enriched
        ]
        results = await enrich_findings_batch(findings, mcp_client)
        assert len(results) == 4
        # NIT and PRAISE unchanged
        assert "Reference" not in results[0].description
        assert "Reference" not in results[1].description
        # WARNING and BLOCKER enriched
        assert "Reference" in results[2].description
        assert "Reference" in results[3].description
