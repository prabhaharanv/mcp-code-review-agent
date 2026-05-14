"""Observability — OpenTelemetry tracing + Prometheus metrics for the agent.

Provides:
    - Distributed tracing: every review gets a trace with spans for each phase
    - Prometheus metrics: review counts, durations, finding severities, token usage
    - Middleware for FastAPI auto-instrumentation

Design: metrics are collected in-process via prometheus_client and exposed
at GET /metrics. Traces are exported to an OTLP endpoint (configurable).
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator
from typing import Any

from prometheus_client import Counter, Histogram, Gauge, Info


# ── Prometheus Metrics ────────────────────────────────────────

REVIEW_REQUESTS = Counter(
    "agent_review_requests_total",
    "Total number of review requests",
    ["status"],  # success, error
)

REVIEW_DURATION = Histogram(
    "agent_review_duration_seconds",
    "Time spent on a full review",
    buckets=[5, 10, 30, 60, 120, 300, 600],
)

TOOL_CALLS = Counter(
    "agent_tool_calls_total",
    "Total number of MCP tool calls",
    ["tool_name"],
)

FINDINGS_TOTAL = Counter(
    "agent_findings_total",
    "Total findings produced",
    ["severity"],
)

TOKEN_USAGE = Counter(
    "agent_token_usage_total",
    "Total tokens consumed by LLM calls",
    ["type"],  # input, output
)

ACTIVE_REVIEWS = Gauge(
    "agent_active_reviews",
    "Number of reviews currently in progress",
)

INTELLIGENCE_DURATION = Histogram(
    "agent_intelligence_pipeline_seconds",
    "Time spent in the intelligence pipeline",
    buckets=[0.1, 0.5, 1, 2, 5, 10],
)

AGENT_INFO = Info(
    "agent",
    "Agent version and configuration",
)


# ── Trace Context ─────────────────────────────────────────────

class TraceContext:
    """Lightweight trace context for tracking a single review's lifecycle.

    This is a simple structured-logging-based trace (no OTLP dependency
    required). Each review gets a trace_id, and each phase gets a span.
    """

    def __init__(self, trace_id: str, pr_url: str):
        self.trace_id = trace_id
        self.pr_url = pr_url
        self.spans: list[dict[str, Any]] = []
        self._start_time = time.monotonic()

    @asynccontextmanager
    async def span(self, name: str, **attributes: Any) -> AsyncGenerator[dict, None]:
        """Create a span within this trace."""
        span_data = {
            "name": name,
            "trace_id": self.trace_id,
            "start_time": time.monotonic(),
            "attributes": attributes,
        }
        try:
            yield span_data
        finally:
            span_data["duration_ms"] = round(
                (time.monotonic() - span_data["start_time"]) * 1000, 2
            )
            self.spans.append(span_data)

    @property
    def total_duration_ms(self) -> float:
        return round((time.monotonic() - self._start_time) * 1000, 2)

    def summary(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "pr_url": self.pr_url,
            "total_duration_ms": self.total_duration_ms,
            "span_count": len(self.spans),
            "spans": [
                {"name": s["name"], "duration_ms": s["duration_ms"]}
                for s in self.spans
            ],
        }


# ── Helper functions ──────────────────────────────────────────

def record_review_start() -> float:
    """Record the start of a review. Returns start time for duration tracking."""
    ACTIVE_REVIEWS.inc()
    return time.monotonic()


def record_review_end(start_time: float, success: bool = True) -> None:
    """Record the end of a review."""
    duration = time.monotonic() - start_time
    ACTIVE_REVIEWS.dec()
    REVIEW_DURATION.observe(duration)
    REVIEW_REQUESTS.labels(status="success" if success else "error").inc()


def record_tool_call(tool_name: str) -> None:
    """Record a tool call."""
    TOOL_CALLS.labels(tool_name=tool_name).inc()


def record_findings(findings: list) -> None:
    """Record finding counts by severity."""
    for f in findings:
        FINDINGS_TOTAL.labels(severity=f.severity.value if hasattr(f.severity, 'value') else str(f.severity)).inc()


def record_token_usage(input_tokens: int, output_tokens: int) -> None:
    """Record LLM token usage."""
    TOKEN_USAGE.labels(type="input").inc(input_tokens)
    TOKEN_USAGE.labels(type="output").inc(output_tokens)


def record_intelligence_duration(duration: float) -> None:
    """Record intelligence pipeline duration."""
    INTELLIGENCE_DURATION.observe(duration)


def set_agent_info(version: str, llm_provider: str, llm_model: str) -> None:
    """Set static agent info labels."""
    AGENT_INFO.info({
        "version": version,
        "llm_provider": llm_provider,
        "llm_model": llm_model,
    })
