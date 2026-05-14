"""Tests for the severity classifier."""

from agent.intelligence.severity import (
    classify_batch,
    classify_severity,
    _extract_rule_code,
)
from agent.models import Finding, Severity


def _make_finding(**kwargs) -> Finding:
    defaults = {
        "severity": Severity.WARNING,
        "file_path": "app.py",
        "line": 10,
        "title": "Test finding",
        "description": "Something happened",
        "tool_source": "run_ruff",
    }
    defaults.update(kwargs)
    return Finding(**defaults)


class TestExtractRuleCode:
    def test_extracts_ruff_rule(self):
        assert _extract_rule_code("ruff E501") == "E501"

    def test_extracts_complex_rule(self):
        assert _extract_rule_code("ruff S608") == "S608"

    def test_no_match(self):
        assert _extract_rule_code("mypy type error") is None


class TestClassifySeverity:
    def test_cosmetic_rule_downgraded_to_nit(self):
        finding = _make_finding(
            severity=Severity.WARNING,
            title="ruff E501",
            description="Line too long",
        )
        result = classify_severity(finding)
        assert result.severity == Severity.NIT

    def test_bug_rule_upgraded_to_blocker(self):
        finding = _make_finding(
            severity=Severity.WARNING,
            title="ruff F821",
            description="Undefined name 'foo'",
        )
        result = classify_severity(finding)
        assert result.severity == Severity.BLOCKER

    def test_security_rule_in_sensitive_file_is_blocker(self):
        finding = _make_finding(
            severity=Severity.WARNING,
            title="ruff S105",
            description="Hardcoded password",
            file_path="auth/handler.py",
        )
        result = classify_severity(finding)
        assert result.severity == Severity.BLOCKER

    def test_unused_import_in_init_is_nit(self):
        finding = _make_finding(
            severity=Severity.WARNING,
            title="ruff F401",
            description="`foo` imported but unused",
            file_path="mypackage/__init__.py",
        )
        result = classify_severity(finding)
        assert result.severity == Severity.NIT

    def test_warning_in_test_file_deescalated(self):
        finding = _make_finding(
            severity=Severity.BLOCKER,
            title="ruff B006",
            description="Mutable default",
            file_path="tests/test_utils.py",
            tool_source="run_ruff",
        )
        result = classify_severity(finding)
        assert result.severity == Severity.WARNING

    def test_security_context_escalation(self):
        finding = _make_finding(
            severity=Severity.NIT,
            title="Something",
            description="token validation issue",
            tool_source="custom",
        )
        result = classify_severity(finding)
        assert result.severity == Severity.WARNING

    def test_data_context_escalation(self):
        finding = _make_finding(
            severity=Severity.NIT,
            title="Something",
            description="database migration rollback",
            tool_source="custom",
        )
        result = classify_severity(finding)
        assert result.severity == Severity.WARNING

    def test_unchanged_finding_returned_as_is(self):
        finding = _make_finding(
            severity=Severity.WARNING,
            title="ruff W999",  # unknown rule, no override
            tool_source="run_ruff",
        )
        result = classify_severity(finding)
        assert result is finding  # exact same object

    def test_praise_not_modified(self):
        finding = _make_finding(severity=Severity.PRAISE)
        result = classify_severity(finding)
        # Praise doesn't go through security/data patterns since it's not NIT/WARNING
        assert result.severity == Severity.PRAISE


class TestClassifyBatch:
    def test_processes_multiple_findings(self):
        findings = [
            _make_finding(title="ruff E501", severity=Severity.WARNING),
            _make_finding(title="ruff F821", severity=Severity.WARNING),
        ]
        results = classify_batch(findings)
        assert len(results) == 2
        assert results[0].severity == Severity.NIT  # cosmetic
        assert results[1].severity == Severity.BLOCKER  # bug

    def test_empty_list(self):
        assert classify_batch([]) == []

    def test_with_file_contexts(self):
        findings = [
            _make_finding(
                severity=Severity.NIT,
                title="Something",
                file_path="app.py",
                tool_source="custom",
            )
        ]
        contexts = {"app.py": "def authenticate(password): pass"}
        results = classify_batch(findings, contexts)
        assert results[0].severity == Severity.WARNING  # password context escalation
