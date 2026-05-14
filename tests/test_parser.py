"""Tests for the observation parser."""

from __future__ import annotations

import json

from agent.models import Severity
from agent.parser import (
    parse_complexity_output,
    parse_mypy_output,
    parse_ruff_output,
    parse_test_output,
    parse_tool_output,
)


class TestParseRuffOutput:
    def test_no_issues(self):
        raw = json.dumps({"diagnostics": [], "summary": "No issues found"})
        findings = parse_ruff_output(raw, "app.py")
        assert findings == []

    def test_single_issue(self):
        raw = json.dumps({
            "diagnostics": [
                {"rule": "F401", "message": "os imported but unused", "line": 1, "severity": "error"}
            ],
            "summary": "1 issue(s) found",
        })
        findings = parse_ruff_output(raw, "app.py")
        assert len(findings) == 1
        assert findings[0].severity == Severity.BLOCKER
        assert findings[0].file_path == "app.py"
        assert findings[0].line == 1
        assert findings[0].tool_source == "run_ruff"

    def test_multiple_issues(self):
        raw = json.dumps({
            "diagnostics": [
                {"rule": "E501", "message": "line too long", "line": 10, "severity": "warning"},
                {"rule": "W291", "message": "trailing whitespace", "line": 20, "severity": "warning"},
            ],
            "summary": "2 issue(s) found",
        })
        findings = parse_ruff_output(raw, "utils.py")
        assert len(findings) == 2
        assert all(f.severity == Severity.WARNING for f in findings)

    def test_invalid_json(self):
        findings = parse_ruff_output("not json", "x.py")
        assert findings == []


class TestParseMypyOutput:
    def test_no_issues(self):
        raw = json.dumps({"diagnostics": [], "summary": "No type errors found"})
        findings = parse_mypy_output(raw, "app.py")
        assert findings == []

    def test_error(self):
        raw = json.dumps({
            "diagnostics": [
                {"raw": "app.py:10:5: error: Incompatible types"}
            ],
            "summary": "1 type issue(s) found",
        })
        findings = parse_mypy_output(raw, "app.py")
        assert len(findings) == 1
        assert findings[0].severity == Severity.BLOCKER
        assert findings[0].line == 10


class TestParseComplexityOutput:
    def test_clean_code(self):
        raw = json.dumps({
            "total_lines": 20,
            "function_count": 2,
            "class_count": 0,
            "functions": [],
            "classes": [],
            "warnings": {"long_functions": [], "too_many_args": []},
        })
        findings = parse_complexity_output(raw, "clean.py")
        assert findings == []

    def test_long_function(self):
        raw = json.dumps({
            "total_lines": 100,
            "function_count": 1,
            "class_count": 0,
            "functions": [{"name": "do_everything", "line": 1, "length": 80, "args": 2, "too_long": True}],
            "classes": [],
            "warnings": {
                "long_functions": [{"name": "do_everything", "line": 1, "length": 80}],
                "too_many_args": [],
            },
        })
        findings = parse_complexity_output(raw, "big.py")
        assert len(findings) == 1
        assert findings[0].severity == Severity.WARNING
        assert "do_everything" in findings[0].title

    def test_too_many_args(self):
        raw = json.dumps({
            "total_lines": 10,
            "function_count": 1,
            "class_count": 0,
            "functions": [{"name": "bloated", "line": 5, "length": 10, "args": 8, "too_long": False}],
            "classes": [],
            "warnings": {
                "long_functions": [],
                "too_many_args": [{"name": "bloated", "line": 5, "args": 8}],
            },
        })
        findings = parse_complexity_output(raw, "args.py")
        assert len(findings) == 1
        assert findings[0].severity == Severity.NIT

    def test_syntax_error(self):
        raw = json.dumps({"error": "SyntaxError: unexpected EOF"})
        findings = parse_complexity_output(raw, "broken.py")
        assert len(findings) == 1
        assert findings[0].severity == Severity.BLOCKER


class TestParseTestOutput:
    def test_all_passed(self):
        raw = json.dumps({"passed": 10, "failed": 0, "errors": 0, "summary": "10 passed"})
        findings = parse_test_output(raw)
        assert findings == []

    def test_failures(self):
        raw = json.dumps({
            "passed": 8,
            "failed": 2,
            "errors": 0,
            "summary": "8 passed, 2 failed",
            "failures": ["FAILED test_foo - AssertionError", "FAILED test_bar - ValueError"],
        })
        findings = parse_test_output(raw)
        assert len(findings) == 2
        assert all(f.severity == Severity.BLOCKER for f in findings)

    def test_errors(self):
        raw = json.dumps({
            "passed": 0,
            "failed": 0,
            "errors": 3,
            "summary": "3 errors",
            "failures": [],
        })
        findings = parse_test_output(raw)
        assert len(findings) == 1
        assert findings[0].severity == Severity.BLOCKER


class TestParseToolOutput:
    def test_routes_to_ruff(self):
        raw = json.dumps({"diagnostics": [], "summary": "clean"})
        findings = parse_tool_output("run_ruff", raw, "x.py")
        assert findings == []

    def test_unknown_tool(self):
        findings = parse_tool_output("unknown_tool", "{}", "x.py")
        assert findings == []

    def test_routes_to_pytest(self):
        raw = json.dumps({"passed": 5, "failed": 0, "errors": 0, "summary": "5 passed", "failures": []})
        findings = parse_tool_output("run_pytest", raw)
        assert findings == []
