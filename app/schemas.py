"""Request/response schemas for the Code Review Agent API."""

from __future__ import annotations

from pydantic import BaseModel, Field

from agent.models import Finding, ReviewPlan, Severity


class ReviewRequest(BaseModel):
    """Request body for the /review endpoint."""

    pr_url: str = Field(
        description="Full GitHub PR URL (e.g. https://github.com/owner/repo/pull/123)"
    )
    post_review: bool = Field(
        default=False,
        description="If true, post the review as a comment on the PR",
    )
    max_steps: int = Field(
        default=20,
        ge=1,
        le=50,
        description="Maximum agent reasoning steps",
    )


class FindingResponse(BaseModel):
    """A single review finding."""

    severity: Severity
    file_path: str
    line: int | None = None
    title: str
    description: str
    suggestion: str | None = None
    confidence: float | None = None


class ReviewResponse(BaseModel):
    """Response body for the /review endpoint."""

    pr_url: str
    pr_title: str
    summary: str
    findings: list[FindingResponse]
    event: str = Field(description="COMMENT, APPROVE, or REQUEST_CHANGES")
    plan: ReviewPlan | None = None
    stats: dict = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """Response body for the /health endpoint."""

    status: str
    version: str
    servers: list[str] = Field(default_factory=list)
