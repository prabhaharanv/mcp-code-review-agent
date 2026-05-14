"""Evaluation framework — measures code review quality against labeled benchmarks.

This module provides an offline evaluation pipeline to answer:
    "How good is the agent's review compared to a human-written gold standard?"

Metrics:
    1. Finding recall: What fraction of known issues did the agent find?
    2. Finding precision: What fraction of agent findings are real issues?
    3. Severity accuracy: Did the agent assign the correct severity?
    4. Suggestion quality: Did the agent provide actionable fix suggestions?
    5. Coverage: What fraction of files were analyzed?

Usage:
    python -m eval.benchmark --dataset eval/dataset.json
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from agent.models import Finding, Severity


@dataclass
class GoldFinding:
    """A human-labeled expected finding in the benchmark."""

    file_path: str
    line: int | None = None
    severity: str = "warning"
    title: str = ""
    description: str = ""
    keywords: list[str] = field(default_factory=list)  # alternative match terms


@dataclass
class BenchmarkCase:
    """One test case in the evaluation dataset."""

    pr_url: str
    pr_title: str = ""
    description: str = ""
    expected_findings: list[GoldFinding] = field(default_factory=list)
    expected_event: str = "COMMENT"
    expected_files_analyzed: list[str] = field(default_factory=list)


@dataclass
class EvalMetrics:
    """Evaluation metrics for one benchmark case."""

    case_id: str
    finding_recall: float = 0.0
    finding_precision: float = 0.0
    severity_accuracy: float = 0.0
    suggestion_rate: float = 0.0
    file_coverage: float = 0.0
    event_correct: bool = False
    matched_findings: int = 0
    total_expected: int = 0
    total_predicted: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    @property
    def f1_score(self) -> float:
        if self.finding_precision + self.finding_recall == 0:
            return 0.0
        return (
            2 * self.finding_precision * self.finding_recall
            / (self.finding_precision + self.finding_recall)
        )

    def summary(self) -> dict:
        return {
            "case_id": self.case_id,
            "f1_score": round(self.f1_score, 3),
            "finding_recall": round(self.finding_recall, 3),
            "finding_precision": round(self.finding_precision, 3),
            "severity_accuracy": round(self.severity_accuracy, 3),
            "suggestion_rate": round(self.suggestion_rate, 3),
            "file_coverage": round(self.file_coverage, 3),
            "event_correct": self.event_correct,
            "matched": self.matched_findings,
            "expected": self.total_expected,
            "predicted": self.total_predicted,
        }


@dataclass
class EvalReport:
    """Aggregate evaluation report across all benchmark cases."""

    cases: list[EvalMetrics] = field(default_factory=list)

    @property
    def avg_f1(self) -> float:
        if not self.cases:
            return 0.0
        return sum(c.f1_score for c in self.cases) / len(self.cases)

    @property
    def avg_recall(self) -> float:
        if not self.cases:
            return 0.0
        return sum(c.finding_recall for c in self.cases) / len(self.cases)

    @property
    def avg_precision(self) -> float:
        if not self.cases:
            return 0.0
        return sum(c.finding_precision for c in self.cases) / len(self.cases)

    @property
    def avg_severity_accuracy(self) -> float:
        if not self.cases:
            return 0.0
        return sum(c.severity_accuracy for c in self.cases) / len(self.cases)

    @property
    def event_accuracy(self) -> float:
        if not self.cases:
            return 0.0
        return sum(1 for c in self.cases if c.event_correct) / len(self.cases)

    def summary(self) -> dict:
        return {
            "total_cases": len(self.cases),
            "avg_f1_score": round(self.avg_f1, 3),
            "avg_recall": round(self.avg_recall, 3),
            "avg_precision": round(self.avg_precision, 3),
            "avg_severity_accuracy": round(self.avg_severity_accuracy, 3),
            "event_accuracy": round(self.event_accuracy, 3),
            "cases": [c.summary() for c in self.cases],
        }


# ── Evaluation Logic ──────────────────────────────────────────

def evaluate_findings(
    predicted: list[Finding],
    expected: list[GoldFinding],
    review_event: str = "COMMENT",
    expected_event: str = "COMMENT",
    expected_files: list[str] | None = None,
    case_id: str = "unknown",
) -> EvalMetrics:
    """Evaluate predicted findings against gold-standard expected findings.

    Matching is fuzzy: a predicted finding matches an expected finding if:
        - Same file path (or expected file_path is a substring of predicted)
        - Close line number (within 5 lines) OR line is None
        - Title/description contains any of the expected keywords

    Args:
        predicted: Agent's findings
        expected: Human-labeled gold findings
        review_event: Agent's review event (COMMENT, APPROVE, REQUEST_CHANGES)
        expected_event: Expected review event
        expected_files: Files that should have been analyzed
        case_id: Identifier for this benchmark case
    """
    metrics = EvalMetrics(
        case_id=case_id,
        total_expected=len(expected),
        total_predicted=len(predicted),
    )

    if not expected:
        # No expected findings — precision is 1.0 if agent also found nothing
        metrics.finding_precision = 1.0 if not predicted else 0.0
        metrics.finding_recall = 1.0
        metrics.event_correct = review_event == expected_event
        return metrics

    # Match predicted to expected
    matched_expected: set[int] = set()
    matched_predicted: set[int] = set()
    severity_correct = 0

    for i, gold in enumerate(expected):
        for j, pred in enumerate(predicted):
            if j in matched_predicted:
                continue
            if _findings_match(pred, gold):
                matched_expected.add(i)
                matched_predicted.add(j)
                if _severities_match(pred.severity, gold.severity):
                    severity_correct += 1
                break

    metrics.matched_findings = len(matched_expected)
    metrics.false_positives = len(predicted) - len(matched_predicted)
    metrics.false_negatives = len(expected) - len(matched_expected)

    # Recall: fraction of expected findings that were found
    metrics.finding_recall = len(matched_expected) / len(expected)

    # Precision: fraction of predicted findings that match expected
    metrics.finding_precision = (
        len(matched_predicted) / len(predicted) if predicted else 1.0
    )

    # Severity accuracy: among matched, how many had correct severity
    metrics.severity_accuracy = (
        severity_correct / len(matched_expected) if matched_expected else 0.0
    )

    # Suggestion rate: how many predicted findings have suggestions
    metrics.suggestion_rate = (
        sum(1 for f in predicted if f.suggestion) / len(predicted)
        if predicted else 0.0
    )

    # File coverage
    if expected_files:
        analyzed_files = {f.file_path for f in predicted}
        covered = sum(1 for ef in expected_files if any(ef in af for af in analyzed_files))
        metrics.file_coverage = covered / len(expected_files)
    else:
        metrics.file_coverage = 1.0

    # Event correctness
    metrics.event_correct = review_event == expected_event

    return metrics


def load_benchmark(dataset_path: str | Path) -> list[BenchmarkCase]:
    """Load a benchmark dataset from JSON.

    Expected format:
    [
        {
            "pr_url": "https://github.com/...",
            "pr_title": "Fix auth bug",
            "expected_findings": [
                {"file_path": "auth.py", "line": 42, "severity": "blocker",
                 "title": "SQL injection", "keywords": ["sql", "injection", "parameterized"]}
            ],
            "expected_event": "REQUEST_CHANGES",
            "expected_files_analyzed": ["auth.py", "tests/test_auth.py"]
        }
    ]
    """
    path = Path(dataset_path)
    with open(path) as f:
        data = json.load(f)

    cases = []
    for item in data:
        findings = [
            GoldFinding(
                file_path=ef["file_path"],
                line=ef.get("line"),
                severity=ef.get("severity", "warning"),
                title=ef.get("title", ""),
                description=ef.get("description", ""),
                keywords=ef.get("keywords", []),
            )
            for ef in item.get("expected_findings", [])
        ]
        cases.append(
            BenchmarkCase(
                pr_url=item["pr_url"],
                pr_title=item.get("pr_title", ""),
                description=item.get("description", ""),
                expected_findings=findings,
                expected_event=item.get("expected_event", "COMMENT"),
                expected_files_analyzed=item.get("expected_files_analyzed", []),
            )
        )
    return cases


def run_evaluation(
    cases: list[BenchmarkCase],
    results: list[tuple[list[Finding], str]],
) -> EvalReport:
    """Run evaluation across multiple benchmark cases.

    Args:
        cases: Benchmark cases with expected findings
        results: List of (predicted_findings, review_event) tuples,
                 one per case, in the same order
    """
    report = EvalReport()
    for i, (case, (predicted, event)) in enumerate(zip(cases, results)):
        metrics = evaluate_findings(
            predicted=predicted,
            expected=case.expected_findings,
            review_event=event,
            expected_event=case.expected_event,
            expected_files=case.expected_files_analyzed,
            case_id=f"case_{i}_{case.pr_title[:30]}",
        )
        report.cases.append(metrics)
    return report


# ── Matching Helpers ──────────────────────────────────────────

def _findings_match(predicted: Finding, gold: GoldFinding) -> bool:
    """Check if a predicted finding matches a gold finding (fuzzy)."""
    # File path match (substring)
    if gold.file_path not in predicted.file_path and predicted.file_path not in gold.file_path:
        return False

    # Line number match (within tolerance)
    if gold.line is not None and predicted.line is not None:
        if abs(gold.line - predicted.line) > 5:
            return False

    # Content match: check keywords in title + description
    if gold.keywords:
        pred_text = f"{predicted.title} {predicted.description}".lower()
        return any(kw.lower() in pred_text for kw in gold.keywords)

    # Fallback: title similarity
    if gold.title:
        return gold.title.lower() in predicted.title.lower()

    return True  # file + line matched, no keywords to check


def _severities_match(predicted: Severity, expected_str: str) -> bool:
    """Check if severities match (string comparison)."""
    return predicted.value == expected_str.lower()
