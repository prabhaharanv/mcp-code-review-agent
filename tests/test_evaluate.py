"""Tests for eval/evaluate.py — evaluation framework."""

import json
import tempfile

from agent.models import Finding, Severity
from eval.evaluate import (
    BenchmarkCase,
    EvalMetrics,
    EvalReport,
    GoldFinding,
    evaluate_findings,
    load_benchmark,
    run_evaluation,
)


class TestEvalMetrics:
    """Tests for the EvalMetrics dataclass."""

    def test_f1_score_perfect(self):
        m = EvalMetrics(case_id="t", finding_recall=1.0, finding_precision=1.0)
        assert m.f1_score == 1.0

    def test_f1_score_zero(self):
        m = EvalMetrics(case_id="t", finding_recall=0.0, finding_precision=0.0)
        assert m.f1_score == 0.0

    def test_f1_score_partial(self):
        m = EvalMetrics(case_id="t", finding_recall=0.5, finding_precision=0.5)
        assert m.f1_score == 0.5

    def test_summary_keys(self):
        m = EvalMetrics(case_id="test_case")
        s = m.summary()
        assert "case_id" in s
        assert "f1_score" in s
        assert "finding_recall" in s
        assert "finding_precision" in s


class TestEvalReport:
    """Tests for the EvalReport aggregate."""

    def test_empty_report(self):
        r = EvalReport()
        assert r.avg_f1 == 0.0
        assert r.avg_recall == 0.0
        assert r.avg_precision == 0.0

    def test_report_averages(self):
        r = EvalReport(
            cases=[
                EvalMetrics(case_id="a", finding_recall=1.0, finding_precision=1.0),
                EvalMetrics(case_id="b", finding_recall=0.5, finding_precision=0.5),
            ]
        )
        assert r.avg_recall == 0.75
        assert r.avg_precision == 0.75

    def test_event_accuracy(self):
        r = EvalReport(
            cases=[
                EvalMetrics(case_id="a", event_correct=True),
                EvalMetrics(case_id="b", event_correct=False),
                EvalMetrics(case_id="c", event_correct=True),
            ]
        )
        assert abs(r.event_accuracy - 2 / 3) < 0.01

    def test_summary_structure(self):
        r = EvalReport(cases=[EvalMetrics(case_id="x")])
        s = r.summary()
        assert s["total_cases"] == 1
        assert "cases" in s


class TestEvaluateFindings:
    """Tests for the core evaluation logic."""

    def _make_finding(self, file_path="test.py", line=10, severity="warning", title="Test"):
        return Finding(
            severity=Severity(severity),
            file_path=file_path,
            line=line,
            title=title,
            description="desc",
        )

    def _make_gold(self, file_path="test.py", line=10, severity="warning", keywords=None):
        return GoldFinding(
            file_path=file_path, line=line, severity=severity, keywords=keywords or []
        )

    def test_perfect_match(self):
        predicted = [self._make_finding(title="SQL injection")]
        expected = [self._make_gold(keywords=["SQL", "injection"])]
        m = evaluate_findings(predicted, expected, case_id="t1")
        assert m.finding_recall == 1.0
        assert m.finding_precision == 1.0
        assert m.matched_findings == 1

    def test_no_expected(self):
        predicted = []
        expected = []
        m = evaluate_findings(predicted, expected, case_id="t2")
        assert m.finding_recall == 1.0
        assert m.finding_precision == 1.0

    def test_no_expected_with_predictions(self):
        predicted = [self._make_finding()]
        expected = []
        m = evaluate_findings(predicted, expected, case_id="t3")
        assert m.finding_precision == 0.0

    def test_no_predictions(self):
        predicted = []
        expected = [self._make_gold()]
        m = evaluate_findings(predicted, expected, case_id="t4")
        assert m.finding_recall == 0.0
        assert m.false_negatives == 1

    def test_file_mismatch(self):
        predicted = [self._make_finding(file_path="other.py")]
        expected = [self._make_gold(file_path="test.py")]
        m = evaluate_findings(predicted, expected, case_id="t5")
        assert m.matched_findings == 0

    def test_line_tolerance(self):
        predicted = [self._make_finding(line=12)]
        expected = [self._make_gold(line=10)]
        m = evaluate_findings(predicted, expected, case_id="t6")
        assert m.matched_findings == 1  # within 5 lines

    def test_line_too_far(self):
        predicted = [self._make_finding(line=20, title="something")]
        expected = [self._make_gold(line=10, keywords=["something"])]
        m = evaluate_findings(predicted, expected, case_id="t7")
        assert m.matched_findings == 0  # > 5 lines apart

    def test_severity_accuracy(self):
        predicted = [self._make_finding(severity="warning", title="test issue")]
        expected = [self._make_gold(severity="warning", keywords=["test"])]
        m = evaluate_findings(predicted, expected, case_id="t8")
        assert m.severity_accuracy == 1.0

    def test_severity_mismatch(self):
        predicted = [self._make_finding(severity="nit", title="test issue")]
        expected = [self._make_gold(severity="blocker", keywords=["test"])]
        m = evaluate_findings(predicted, expected, case_id="t9")
        assert m.severity_accuracy == 0.0

    def test_event_correct(self):
        m = evaluate_findings([], [], review_event="APPROVE", expected_event="APPROVE", case_id="t10")
        assert m.event_correct is True

    def test_event_incorrect(self):
        m = evaluate_findings([], [], review_event="COMMENT", expected_event="APPROVE", case_id="t11")
        assert m.event_correct is False

    def test_file_coverage(self):
        predicted = [self._make_finding(file_path="auth/views.py")]
        expected = [self._make_gold(file_path="auth/views.py", keywords=["test"])]
        m = evaluate_findings(
            predicted,
            expected,
            expected_files=["auth/views.py", "tests/test_auth.py"],
            case_id="t12",
        )
        assert m.file_coverage == 0.5  # 1 of 2 files

    def test_suggestion_rate(self):
        f = self._make_finding(title="unused import")
        f.suggestion = "Remove the import"
        predicted = [f, self._make_finding(title="other issue")]
        expected = [
            self._make_gold(keywords=["unused"]),
            self._make_gold(keywords=["other"]),
        ]
        m = evaluate_findings(predicted, expected, case_id="t13")
        assert m.suggestion_rate == 0.5


class TestLoadBenchmark:
    """Tests for loading benchmark datasets."""

    def test_load_dataset_json(self):
        cases = load_benchmark("eval/dataset.json")
        assert len(cases) == 5
        assert cases[0].pr_title == "Add user authentication endpoint"
        assert len(cases[0].expected_findings) == 3

    def test_load_custom_dataset(self):
        data = [
            {
                "pr_url": "https://github.com/test/repo/pull/1",
                "pr_title": "Test PR",
                "expected_findings": [
                    {"file_path": "main.py", "severity": "warning", "keywords": ["test"]}
                ],
                "expected_event": "COMMENT",
            }
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            cases = load_benchmark(f.name)

        assert len(cases) == 1
        assert cases[0].expected_findings[0].file_path == "main.py"


class TestRunEvaluation:
    """Tests for the full evaluation pipeline."""

    def test_run_evaluation(self):
        cases = [
            BenchmarkCase(
                pr_url="https://github.com/t/r/pull/1",
                pr_title="Test",
                expected_findings=[
                    GoldFinding(file_path="a.py", severity="warning", keywords=["bug"])
                ],
                expected_event="COMMENT",
            )
        ]
        predicted = Finding(
            severity=Severity.WARNING,
            file_path="a.py",
            line=1,
            title="Found a bug",
            description="desc",
        )
        results = [([predicted], "COMMENT")]

        report = run_evaluation(cases, results)
        assert len(report.cases) == 1
        assert report.avg_f1 > 0
        assert report.cases[0].event_correct is True
