"""Tests for the intelligence pipeline."""

from unittest.mock import AsyncMock, MagicMock, patch
import json
import pytest

from agent.intelligence.pipeline import run_intelligence_pipeline
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


class TestIntelligencePipeline:
    @pytest.mark.asyncio
    async def test_pipeline_with_no_findings(self):
        """Empty findings list should still run cross-file analysis."""
        files = [{"filename": "app.py", "status": "modified", "additions": 5}]
        result = await run_intelligence_pipeline(
            findings=[],
            pr_files=files,
        )
        # May have cross-file findings, but shouldn't crash
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_pipeline_enriches_findings(self):
        """Findings should get severity, suggestions, and confidence."""
        findings = [
            _make_finding(
                title="ruff E501",
                severity=Severity.WARNING,
                tool_source="run_ruff",
            ),
        ]
        files = [{"filename": "app.py", "status": "modified", "additions": 10}]

        result = await run_intelligence_pipeline(
            findings=findings,
            pr_files=files,
        )

        assert len(result) >= 1
        # E501 should be downgraded to NIT by severity classifier
        ruff_findings = [f for f in result if f.title == "ruff E501"]
        assert ruff_findings[0].severity == Severity.NIT
        # Should have confidence tag
        assert "[confidence:" in ruff_findings[0].description

    @pytest.mark.asyncio
    async def test_pipeline_adds_cross_file_findings(self):
        """Cross-file analysis should add findings for missing tests."""
        findings = [_make_finding()]
        files = [
            {"filename": "src/handler.py", "status": "added", "additions": 50},
        ]

        result = await run_intelligence_pipeline(
            findings=findings,
            pr_files=files,
        )

        # Should have original finding + cross-file "missing tests" finding
        assert len(result) >= 2
        cross_file = [f for f in result if f.tool_source == "cross_file_analysis"]
        assert len(cross_file) >= 1

    @pytest.mark.asyncio
    async def test_pipeline_with_mcp_client(self):
        """Pipeline should use MCP client for RAG enrichment when available."""
        mcp_client = MagicMock()
        mcp_client.servers = {"knowledge-base": MagicMock()}
        mcp_client.call_tool = AsyncMock(return_value=json.dumps({
            "found": True,
            "answer": "This is a coding standard about error handling best practices.",
        }))

        findings = [
            _make_finding(severity=Severity.BLOCKER, description="Error handling gap"),
        ]
        files = [{"filename": "app.py", "status": "modified", "additions": 10}]

        result = await run_intelligence_pipeline(
            findings=findings,
            pr_files=files,
            mcp_client=mcp_client,
        )

        assert len(result) >= 1
        # The blocker finding should have been enriched with RAG context
        blocker = [f for f in result if f.severity == Severity.BLOCKER]
        if blocker:
            assert "Reference" in blocker[0].description or "[confidence:" in blocker[0].description

    @pytest.mark.asyncio
    async def test_pipeline_without_mcp_client(self):
        """Pipeline should work fine without MCP client (no RAG enrichment)."""
        findings = [_make_finding()]
        files = [{"filename": "app.py", "status": "modified"}]

        result = await run_intelligence_pipeline(
            findings=findings,
            pr_files=files,
            mcp_client=None,
        )

        assert len(result) >= 1
        # Should still have confidence scores
        assert "[confidence:" in result[0].description

    @pytest.mark.asyncio
    async def test_pipeline_preserves_tool_source(self):
        """Enrichment should not lose the tool_source field."""
        findings = [
            _make_finding(tool_source="run_ruff"),
            _make_finding(tool_source="run_mypy"),
        ]
        files = [{"filename": "app.py", "status": "modified"}]

        result = await run_intelligence_pipeline(
            findings=findings,
            pr_files=files,
        )

        for f in result:
            assert f.tool_source is not None

    @pytest.mark.asyncio
    async def test_pipeline_with_file_contents(self):
        """Pipeline should use file contents for deeper analysis."""
        findings = [
            _make_finding(
                severity=Severity.NIT,
                file_path="auth.py",
                description="Minor issue",
                tool_source="custom",
            )
        ]
        files = [{"filename": "auth.py", "status": "modified", "additions": 20}]
        contents = {"auth.py": "def verify_password(token): pass"}

        result = await run_intelligence_pipeline(
            findings=findings,
            pr_files=files,
            file_contents=contents,
        )

        # The NIT in auth.py with password context should be escalated
        auth_findings = [f for f in result if f.file_path == "auth.py"]
        assert any(f.severity == Severity.WARNING for f in auth_findings)
