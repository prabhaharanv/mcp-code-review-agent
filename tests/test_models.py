"""Tests for agent data models."""

from __future__ import annotations

from datetime import datetime, timezone

from agent.models import AgentEvent, Finding, ReviewPlan, ReviewResult, Severity


class TestSeverity:
    def test_enum_values(self):
        assert Severity.BLOCKER == "blocker"
        assert Severity.WARNING == "warning"
        assert Severity.NIT == "nit"
        assert Severity.PRAISE == "praise"


class TestFinding:
    def test_minimal_finding(self):
        f = Finding(
            severity=Severity.WARNING,
            file_path="src/app.py",
            title="Unused import",
            description="os is imported but never used",
        )
        assert f.line is None
        assert f.suggestion is None
        assert f.tool_source is None

    def test_full_finding(self):
        f = Finding(
            severity=Severity.BLOCKER,
            file_path="auth.py",
            line=42,
            title="SQL injection",
            description="User input passed directly to query",
            suggestion="Use parameterized queries",
            tool_source="run_ruff",
        )
        assert f.line == 42
        assert f.tool_source == "run_ruff"


class TestReviewResult:
    def test_empty_result(self):
        r = ReviewResult(pr_url="https://github.com/o/r/pull/1")
        assert r.blocker_count == 0
        assert r.warning_count == 0
        assert r.nit_count == 0
        assert r.praise_count == 0
        assert r.duration_seconds is None

    def test_counts(self):
        r = ReviewResult(
            pr_url="https://github.com/o/r/pull/1",
            findings=[
                Finding(severity=Severity.BLOCKER, file_path="a.py", title="bug", description="x"),
                Finding(severity=Severity.BLOCKER, file_path="b.py", title="bug2", description="y"),
                Finding(severity=Severity.WARNING, file_path="c.py", title="warn", description="z"),
                Finding(severity=Severity.NIT, file_path="d.py", title="nit", description="w"),
                Finding(severity=Severity.PRAISE, file_path="e.py", title="nice", description="v"),
            ],
        )
        assert r.blocker_count == 2
        assert r.warning_count == 1
        assert r.nit_count == 1
        assert r.praise_count == 1

    def test_duration(self):
        start = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        end = datetime(2026, 1, 1, 0, 0, 30, tzinfo=timezone.utc)
        r = ReviewResult(
            pr_url="https://github.com/o/r/pull/1",
            started_at=start,
            completed_at=end,
        )
        assert r.duration_seconds == 30.0


class TestReviewPlan:
    def test_plan_creation(self):
        p = ReviewPlan(
            summary="Test PR",
            files_to_analyze=["a.py", "b.py"],
            checks_to_run=["run_ruff"],
            risk_areas=["touches auth"],
            estimated_steps=5,
        )
        assert len(p.files_to_analyze) == 2
        assert p.estimated_steps == 5


class TestAgentEvent:
    def test_event(self):
        e = AgentEvent(event_type="tool_call", step=3, data={"tool": "run_ruff"})
        assert e.event_type == "tool_call"
        assert e.step == 3
