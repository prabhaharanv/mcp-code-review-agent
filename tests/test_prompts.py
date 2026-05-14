"""Tests for the agent prompts module."""

from __future__ import annotations

from agent.prompts import build_system_prompt, REVIEW_TASK_TEMPLATE


class TestBuildSystemPrompt:
    def test_default_max_steps(self):
        prompt = build_system_prompt()
        assert "20" in prompt

    def test_custom_max_steps(self):
        prompt = build_system_prompt(max_steps=10)
        assert "10" in prompt
        assert "20" not in prompt

    def test_contains_workflow_sections(self):
        prompt = build_system_prompt()
        assert "Understand" in prompt
        assert "Plan" in prompt
        assert "Analyze" in prompt
        assert "Synthesize" in prompt
        assert "Post" in prompt

    def test_contains_severity_levels(self):
        prompt = build_system_prompt()
        assert "Blocker" in prompt
        assert "Warning" in prompt
        assert "Nit" in prompt
        assert "Praise" in prompt


class TestReviewTaskTemplate:
    def test_template_formatting(self):
        url = "https://github.com/owner/repo/pull/42"
        task = REVIEW_TASK_TEMPLATE.format(pr_url=url)
        assert url in task
