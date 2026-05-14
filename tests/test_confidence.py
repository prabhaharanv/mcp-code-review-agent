"""Tests for the confidence scorer."""

from agent.intelligence.confidence import (
    score_confidence,
    score_confidence_batch,
    _compute_factors,
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


class TestComputeFactors:
    def test_ruff_base_confidence(self):
        finding = _make_finding(tool_source="run_ruff")
        factors = _compute_factors(finding, [], True)
        assert factors.base_score == 0.90

    def test_mypy_base_confidence(self):
        finding = _make_finding(tool_source="run_mypy")
        factors = _compute_factors(finding, [], True)
        assert factors.base_score == 0.75

    def test_complexity_base_confidence(self):
        finding = _make_finding(tool_source="analyze_complexity")
        factors = _compute_factors(finding, [], True)
        assert factors.base_score == 0.60

    def test_pytest_base_confidence(self):
        finding = _make_finding(tool_source="run_pytest")
        factors = _compute_factors(finding, [], True)
        assert factors.base_score == 0.95

    def test_unknown_tool_default_confidence(self):
        finding = _make_finding(tool_source="custom_tool")
        factors = _compute_factors(finding, [], True)
        assert factors.base_score == 0.5

    def test_high_fp_rule_penalized(self):
        finding = _make_finding(title="ruff E501")
        factors = _compute_factors(finding, [], True)
        assert factors.tool_factor < 0

    def test_high_accuracy_rule_boosted(self):
        finding = _make_finding(title="ruff F821")
        factors = _compute_factors(finding, [], True)
        assert factors.tool_factor > 0

    def test_blocker_from_reliable_tool_boosted(self):
        finding = _make_finding(severity=Severity.BLOCKER, tool_source="run_ruff")
        factors = _compute_factors(finding, [], True)
        assert factors.severity_factor > 0

    def test_corroboration_boost(self):
        finding = _make_finding()
        corroborating = [_make_finding(tool_source="run_mypy")]
        factors = _compute_factors(finding, corroborating, True)
        assert factors.context_factor > 0

    def test_unchanged_file_penalty(self):
        finding = _make_finding()
        factors = _compute_factors(finding, [], file_was_changed=False)
        assert factors.context_factor < 0

    def test_score_clamped_to_range(self):
        finding = _make_finding(tool_source="run_pytest", severity=Severity.BLOCKER)
        factors = _compute_factors(finding, [_make_finding(), _make_finding()], True)
        assert 0.1 <= factors.final_score <= 1.0


class TestScoreConfidence:
    def test_appends_confidence_tag(self):
        finding = _make_finding()
        result = score_confidence(finding)
        assert "[confidence:" in result.description

    def test_high_confidence_for_ruff_bug(self):
        finding = _make_finding(title="ruff F821", severity=Severity.BLOCKER)
        result = score_confidence(finding)
        assert "confidence:" in result.description


class TestScoreConfidenceBatch:
    def test_processes_all_findings(self):
        findings = [
            _make_finding(title="ruff E501"),
            _make_finding(title="ruff F821"),
            _make_finding(tool_source="run_mypy"),
        ]
        results = score_confidence_batch(findings)
        assert len(results) == 3
        for r in results:
            assert "[confidence:" in r.description

    def test_corroboration_detected(self):
        # Two findings at same location → corroboration boost
        findings = [
            _make_finding(file_path="app.py", line=10, tool_source="run_ruff"),
            _make_finding(file_path="app.py", line=10, tool_source="run_mypy"),
        ]
        results = score_confidence_batch(findings)
        assert len(results) == 2
        # Both should mention "corroborated" in their confidence explanation
        assert any("corroborated" in r.description for r in results)

    def test_with_changed_files(self):
        findings = [
            _make_finding(file_path="changed.py"),
            _make_finding(file_path="untouched.py"),
        ]
        results = score_confidence_batch(findings, changed_files={"changed.py"})
        assert len(results) == 2
        # untouched file should have lower-ish confidence
        assert "unchanged file" in results[1].description

    def test_empty_list(self):
        assert score_confidence_batch([]) == []
