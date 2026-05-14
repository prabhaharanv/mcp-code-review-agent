"""Tests for agent/observability.py — Prometheus metrics + trace context."""

import time

import pytest

from agent.observability import (
    ACTIVE_REVIEWS,
    FINDINGS_TOTAL,
    REVIEW_DURATION,
    REVIEW_REQUESTS,
    TOKEN_USAGE,
    TOOL_CALLS,
    TraceContext,
    record_findings,
    record_review_end,
    record_review_start,
    record_tool_call,
    record_token_usage,
    set_agent_info,
)


class TestTraceContext:
    """Tests for the lightweight trace context."""

    @pytest.fixture
    def trace(self):
        return TraceContext(trace_id="test-trace-123", pr_url="https://github.com/o/r/pull/1")

    async def test_span_records_duration(self, trace):
        async with trace.span("test_span", key="value") as span:
            pass  # near-zero duration
        assert len(trace.spans) == 1
        assert trace.spans[0]["name"] == "test_span"
        assert "duration_ms" in trace.spans[0]
        assert trace.spans[0]["duration_ms"] >= 0

    async def test_span_attributes(self, trace):
        async with trace.span("fetch", tool="github") as span:
            span["custom"] = "data"
        assert trace.spans[0]["attributes"]["tool"] == "github"

    async def test_multiple_spans(self, trace):
        async with trace.span("step1"):
            pass
        async with trace.span("step2"):
            pass
        assert len(trace.spans) == 2

    def test_summary(self, trace):
        s = trace.summary()
        assert s["trace_id"] == "test-trace-123"
        assert s["pr_url"] == "https://github.com/o/r/pull/1"
        assert s["span_count"] == 0
        assert s["total_duration_ms"] >= 0

    async def test_summary_after_spans(self, trace):
        async with trace.span("s1"):
            pass
        s = trace.summary()
        assert s["span_count"] == 1
        assert len(s["spans"]) == 1
        assert s["spans"][0]["name"] == "s1"


class TestMetricsHelpers:
    """Tests for Prometheus metric helper functions."""

    def test_record_review_start_returns_float(self):
        start = record_review_start()
        assert isinstance(start, float)
        assert start > 0
        # Clean up
        ACTIVE_REVIEWS.dec()

    def test_record_review_end_success(self):
        start = time.monotonic()
        ACTIVE_REVIEWS.inc()  # simulate start
        record_review_end(start, success=True)
        # Just verify no exceptions raised

    def test_record_review_end_failure(self):
        start = time.monotonic()
        ACTIVE_REVIEWS.inc()
        record_review_end(start, success=False)

    def test_record_tool_call(self):
        record_tool_call("run_ruff")
        record_tool_call("get_pr_metadata")
        # Verify counters increment without error

    def test_record_token_usage(self):
        record_token_usage(1000, 500)
        # Verify no exceptions

    def test_record_findings_with_severity(self):
        from agent.models import Finding, Severity

        findings = [
            Finding(severity=Severity.BLOCKER, file_path="a.py", title="Bug", description="Bad"),
            Finding(severity=Severity.WARNING, file_path="b.py", title="Warn", description="Meh"),
            Finding(severity=Severity.NIT, file_path="c.py", title="Nit", description="Minor"),
        ]
        record_findings(findings)

    def test_set_agent_info(self):
        set_agent_info("0.4.0", "anthropic", "claude-sonnet-4-20250514")
