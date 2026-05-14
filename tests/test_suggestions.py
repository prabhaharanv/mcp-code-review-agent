"""Tests for the fix suggestion engine."""

from agent.intelligence.suggestions import (
    suggest_fix,
    suggest_fixes_batch,
    _generate_fix,
)
from agent.models import Finding, Severity


def _make_finding(**kwargs) -> Finding:
    defaults = {
        "severity": Severity.WARNING,
        "file_path": "app.py",
        "line": 10,
        "title": "Test finding",
        "description": "Something happened",
    }
    defaults.update(kwargs)
    return Finding(**defaults)


class TestGenerateFix:
    def test_ruff_f401_unused_import(self):
        finding = _make_finding(
            title="ruff F401",
            description="`os` imported but unused",
            tool_source="run_ruff",
        )
        fix = _generate_fix(finding, "")
        assert fix is not None
        assert "Remove unused import" in fix.summary
        assert fix.confidence >= 0.9

    def test_ruff_f401_in_init(self):
        finding = _make_finding(
            title="ruff F401",
            description="`MyClass` imported but unused",
            file_path="mypackage/__init__.py",
            tool_source="run_ruff",
        )
        fix = _generate_fix(finding, "")
        assert fix is not None
        assert "re-export" in fix.summary.lower() or "noqa" in fix.summary.lower()

    def test_ruff_b006_mutable_default(self):
        finding = _make_finding(
            title="ruff B006",
            description="Do not use mutable data structures",
            tool_source="run_ruff",
        )
        fix = _generate_fix(finding, "")
        assert fix is not None
        assert "None" in fix.summary
        assert fix.before is not None
        assert fix.after is not None

    def test_ruff_e711_none_comparison(self):
        finding = _make_finding(
            title="ruff E711",
            tool_source="run_ruff",
        )
        fix = _generate_fix(finding, "")
        assert fix is not None
        assert "is None" in fix.summary

    def test_ruff_s608_sql_injection(self):
        finding = _make_finding(
            title="ruff S608",
            tool_source="run_ruff",
        )
        fix = _generate_fix(finding, "")
        assert fix is not None
        assert "parameterized" in fix.summary.lower()

    def test_complexity_long_function(self):
        finding = _make_finding(
            title="Function `process_data` too long (80 lines)",
            tool_source="analyze_complexity",
        )
        fix = _generate_fix(finding, "")
        assert fix is not None
        assert "process_data" in fix.summary

    def test_complexity_too_many_params(self):
        finding = _make_finding(
            title="Function `build_query` has 8 parameters",
            tool_source="analyze_complexity",
        )
        fix = _generate_fix(finding, "")
        assert fix is not None
        assert "dataclass" in fix.summary.lower() or "config" in fix.summary.lower()

    def test_test_failure_suggestion(self):
        finding = _make_finding(
            title="Test failure",
            tool_source="run_pytest",
        )
        fix = _generate_fix(finding, "")
        assert fix is not None
        assert fix.confidence < 0.5  # low confidence for test failures

    def test_mypy_suggestion(self):
        finding = _make_finding(
            title="mypy type error",
            tool_source="run_mypy",
        )
        fix = _generate_fix(finding, "")
        assert fix is not None
        assert "type annotation" in fix.summary.lower()

    def test_unknown_tool_no_fix(self):
        finding = _make_finding(tool_source="unknown_tool")
        fix = _generate_fix(finding, "")
        assert fix is None

    def test_unknown_ruff_rule_no_fix(self):
        finding = _make_finding(
            title="ruff Z999",  # unknown rule
            tool_source="run_ruff",
        )
        fix = _generate_fix(finding, "")
        assert fix is None


class TestSuggestFix:
    def test_attaches_suggestion_to_finding(self):
        finding = _make_finding(
            title="ruff E711",
            tool_source="run_ruff",
        )
        result = suggest_fix(finding)
        assert result.suggestion is not None
        assert "is None" in result.suggestion

    def test_includes_before_after(self):
        finding = _make_finding(
            title="ruff B006",
            tool_source="run_ruff",
        )
        result = suggest_fix(finding)
        assert "Before" in result.suggestion
        assert "After" in result.suggestion

    def test_no_fix_returns_unchanged(self):
        finding = _make_finding(tool_source="unknown_tool")
        result = suggest_fix(finding)
        assert result is finding


class TestSuggestFixesBatch:
    def test_batch_processes_all(self):
        findings = [
            _make_finding(title="ruff E711", tool_source="run_ruff"),
            _make_finding(title="ruff B006", tool_source="run_ruff"),
            _make_finding(tool_source="unknown_tool"),
        ]
        results = suggest_fixes_batch(findings)
        assert len(results) == 3
        assert results[0].suggestion is not None
        assert results[1].suggestion is not None
        assert results[2].suggestion is None  # unknown tool → no fix

    def test_empty_list(self):
        assert suggest_fixes_batch([]) == []
