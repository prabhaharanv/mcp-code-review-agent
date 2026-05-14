"""Tests for the review planner."""

from __future__ import annotations

from agent.planner import (
    create_review_plan,
    plan_to_prompt_context,
    _get_extension,
    _is_high_risk,
    _is_test_file,
)


class TestHelpers:
    def test_get_extension(self):
        assert _get_extension("foo.py") == ".py"
        assert _get_extension("path/to/bar.ts") == ".ts"
        assert _get_extension("Makefile") == ""
        assert _get_extension("archive.tar.gz") == ".gz"

    def test_is_high_risk(self):
        assert _is_high_risk("src/auth/login.py")
        assert _is_high_risk("config/settings.py")
        assert _is_high_risk("middleware/rate_limit.py")
        assert _is_high_risk("utils/security.py")
        assert not _is_high_risk("utils/helpers.py")
        assert not _is_high_risk("README.md")

    def test_is_test_file(self):
        assert _is_test_file("test_foo.py")
        assert _is_test_file("tests/test_bar.py")
        assert _is_test_file("foo_test.py")
        assert not _is_test_file("foo.py")
        assert not _is_test_file("testing_utils.py")


class TestCreateReviewPlan:
    def _make_metadata(self, **overrides):
        base = {
            "title": "Add feature X",
            "author": "dev",
            "additions": 50,
            "deletions": 10,
            "changed_files": 3,
        }
        base.update(overrides)
        return base

    def _make_file(self, filename, status="modified", additions=20, deletions=5):
        return {
            "filename": filename,
            "status": status,
            "additions": additions,
            "deletions": deletions,
            "patch": "",
        }

    def test_basic_plan(self):
        metadata = self._make_metadata()
        files = [
            self._make_file("src/app.py", additions=30),
            self._make_file("src/utils.py", additions=10),
        ]
        plan = create_review_plan(metadata, files)

        assert len(plan.files_to_analyze) == 2
        assert "run_ruff" in plan.checks_to_run
        assert plan.estimated_steps > 0

    def test_skips_deleted_files(self):
        metadata = self._make_metadata()
        files = [
            self._make_file("old_code.py", status="removed"),
            self._make_file("new_code.py", status="added", additions=25),
        ]
        plan = create_review_plan(metadata, files)

        assert "old_code.py" not in plan.files_to_analyze
        assert "new_code.py" in plan.files_to_analyze

    def test_flags_high_risk(self):
        metadata = self._make_metadata()
        files = [self._make_file("src/auth/handler.py", additions=20)]
        plan = create_review_plan(metadata, files)

        assert any("security" in r or "auth" in r for r in plan.risk_areas)

    def test_flags_missing_tests(self):
        metadata = self._make_metadata()
        files = [self._make_file("src/feature.py", additions=50)]
        plan = create_review_plan(metadata, files)

        assert any("test" in r.lower() for r in plan.risk_areas)

    def test_flags_large_pr(self):
        metadata = self._make_metadata(additions=400, deletions=200)
        files = [self._make_file("big.py", additions=400)]
        plan = create_review_plan(metadata, files)

        assert any("Large PR" in r for r in plan.risk_areas)

    def test_mypy_for_big_changes(self):
        metadata = self._make_metadata()
        files = [self._make_file("src/app.py", additions=25)]
        plan = create_review_plan(metadata, files)

        assert "run_mypy" in plan.checks_to_run

    def test_complexity_for_very_big_changes(self):
        metadata = self._make_metadata()
        files = [self._make_file("src/app.py", additions=35)]
        plan = create_review_plan(metadata, files)

        assert "analyze_complexity" in plan.checks_to_run

    def test_no_analysis_for_trivial_changes(self):
        metadata = self._make_metadata(additions=2, deletions=1, changed_files=1)
        files = [self._make_file("readme.md", additions=2)]
        plan = create_review_plan(metadata, files)

        # Non-Python files don't trigger linters
        assert "run_ruff" not in plan.checks_to_run


class TestPlanToPromptContext:
    def test_contains_summary(self):
        plan = create_review_plan(
            {"title": "Test", "author": "dev", "additions": 10, "deletions": 5, "changed_files": 1},
            [{"filename": "x.py", "status": "modified", "additions": 10, "deletions": 5, "patch": ""}],
        )
        context = plan_to_prompt_context(plan)

        assert "Review Plan" in context
        assert "Files to analyze" in context
        assert "Estimated tool calls" in context
