"""Tests for the FastAPI app schemas."""

from __future__ import annotations

from app.schemas import (
    FindingResponse,
    HealthResponse,
    ReviewRequest,
    ReviewResponse,
)
from agent.models import Severity


class TestReviewRequest:
    def test_defaults(self):
        req = ReviewRequest(pr_url="https://github.com/o/r/pull/1")
        assert req.post_review is False
        assert req.max_steps == 20

    def test_custom_values(self):
        req = ReviewRequest(
            pr_url="https://github.com/o/r/pull/1",
            post_review=True,
            max_steps=10,
        )
        assert req.post_review is True
        assert req.max_steps == 10


class TestFindingResponse:
    def test_creation(self):
        f = FindingResponse(
            severity=Severity.WARNING,
            file_path="app.py",
            line=10,
            title="Issue",
            description="Something wrong",
            suggestion="Fix it",
        )
        assert f.severity == Severity.WARNING
        assert f.line == 10


class TestHealthResponse:
    def test_creation(self):
        h = HealthResponse(status="ok", version="0.2.0", servers=["github", "code-analysis"])
        assert h.status == "ok"
        assert len(h.servers) == 2


class TestReviewResponse:
    def test_creation(self):
        r = ReviewResponse(
            pr_url="https://github.com/o/r/pull/1",
            pr_title="Test PR",
            summary="Looks good",
            findings=[],
            event="COMMENT",
            stats={"steps_taken": 5},
        )
        assert r.event == "COMMENT"
        assert r.stats["steps_taken"] == 5
