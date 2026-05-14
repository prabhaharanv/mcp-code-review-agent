"""Structured data models for the code review agent."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class Severity(str, Enum):
    BLOCKER = "blocker"
    WARNING = "warning"
    NIT = "nit"
    PRAISE = "praise"


class Finding(BaseModel):
    """A single review finding tied to a specific file and line."""

    severity: Severity
    file_path: str
    line: int | None = None
    title: str
    description: str
    suggestion: str | None = None
    tool_source: str | None = None  # which MCP tool produced the evidence


class ReviewPlan(BaseModel):
    """Agent's plan for reviewing the PR before it starts analyzing."""

    summary: str = Field(description="One-sentence summary of what the PR does")
    files_to_analyze: list[str] = Field(description="Files that need deep analysis")
    checks_to_run: list[str] = Field(description="Tools/checks to run (e.g. ruff, mypy)")
    risk_areas: list[str] = Field(description="Potential risk areas to focus on")
    estimated_steps: int = Field(description="Estimated number of tool calls needed")


class ReviewResult(BaseModel):
    """The final structured output of a code review."""

    pr_url: str
    pr_title: str = ""
    summary: str = Field(default="", description="Top-level review summary (3-10 sentences)")
    findings: list[Finding] = Field(default_factory=list)
    event: str = Field(
        default="COMMENT",
        description="Review action: COMMENT, APPROVE, or REQUEST_CHANGES",
    )
    plan: ReviewPlan | None = None
    steps_taken: int = 0
    total_tool_calls: int = 0
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    raw_review_text: str = ""

    @property
    def blocker_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.BLOCKER)

    @property
    def warning_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.WARNING)

    @property
    def nit_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.NIT)

    @property
    def praise_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.PRAISE)

    @property
    def duration_seconds(self) -> float | None:
        if self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


class AgentEvent(BaseModel):
    """An event emitted during the agent's execution (for streaming)."""

    event_type: str  # plan, tool_call, tool_result, finding, thinking, done
    step: int = 0
    data: dict = Field(default_factory=dict)
