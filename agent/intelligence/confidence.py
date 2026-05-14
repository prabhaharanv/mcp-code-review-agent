"""Confidence scorer — rates how reliable each finding is.

Not all findings are equally trustworthy:
    - A ruff F821 (undefined name) is almost certainly a real bug → high confidence
    - A complexity warning about a 55-line function is debatable → medium confidence
    - A mypy error could be a false positive from missing stubs → lower confidence

The confidence score (0.0–1.0) helps the agent and the developer prioritize
which findings to act on vs. which to verify manually.

Factors:
    1. Tool reliability: Static analysis > heuristic checks
    2. Rule specificity: Bug-finding rules > style rules
    3. Context corroboration: Finding supported by multiple signals → higher
    4. File relevance: Finding in changed code > finding in untouched code
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from agent.models import Finding, Severity


@dataclass
class ConfidenceFactors:
    """Breakdown of what contributed to a confidence score."""

    base_score: float
    tool_factor: float
    severity_factor: float
    context_factor: float
    final_score: float
    explanation: str


# Base confidence by tool source
_TOOL_CONFIDENCE: dict[str, float] = {
    "run_ruff": 0.90,       # Static analysis — very reliable
    "run_mypy": 0.75,       # Type checking — sometimes false positives with missing stubs
    "analyze_complexity": 0.60,  # Heuristic — threshold-based, debatable
    "run_pytest": 0.95,     # Tests either pass or fail — very clear signal
    "find_test_files": 0.50,  # Just file existence checks
}

# Rules known to have high false-positive rates
_HIGH_FP_RULES = frozenset({
    "E501",   # line too long — style preference
    "D100", "D101", "D102", "D103",  # missing docstrings — common in internal code
    "I001",   # import order — auto-fixable, not a real issue
})

# Rules known to be very accurate
_HIGH_ACCURACY_RULES = frozenset({
    "F821",   # undefined name
    "F811",   # redefined unused
    "E711",   # comparison to None
    "B006",   # mutable default
    "S105", "S106", "S107",  # hardcoded secrets
    "S301",   # pickle
    "S608",   # SQL injection
})


def score_confidence(
    finding: Finding,
    corroborating_findings: list[Finding] | None = None,
    file_was_changed: bool = True,
) -> Finding:
    """Score the confidence of a finding.

    Args:
        finding: The finding to score
        corroborating_findings: Other findings about the same location
        file_was_changed: Whether this file was modified in the PR

    Returns:
        Finding with confidence info appended to description
    """
    factors = _compute_factors(finding, corroborating_findings or [], file_was_changed)

    # Append confidence to description
    confidence_tag = f" [confidence: {factors.final_score:.0%} — {factors.explanation}]"
    updated_desc = finding.description + confidence_tag

    return finding.model_copy(update={"description": updated_desc})


def score_confidence_batch(
    findings: list[Finding],
    changed_files: set[str] | None = None,
) -> list[Finding]:
    """Score confidence for a batch of findings.

    Automatically detects corroborating findings (same file + line).
    """
    changed = changed_files or set()

    # Index findings by (file, line) for corroboration
    location_index: dict[tuple[str, int | None], list[Finding]] = {}
    for f in findings:
        key = (f.file_path, f.line)
        location_index.setdefault(key, []).append(f)

    result = []
    for f in findings:
        key = (f.file_path, f.line)
        corroborating = [
            other for other in location_index.get(key, [])
            if other is not f
        ]
        file_changed = f.file_path in changed if changed else True
        result.append(score_confidence(f, corroborating, file_changed))

    return result


def _compute_factors(
    finding: Finding,
    corroborating: list[Finding],
    file_was_changed: bool,
) -> ConfidenceFactors:
    """Compute the confidence breakdown."""
    # 1. Base score from tool
    base = _TOOL_CONFIDENCE.get(finding.tool_source or "", 0.5)

    # 2. Tool-specific adjustments
    tool_factor = 0.0
    rule_code = _extract_rule(finding.title)
    if rule_code:
        if rule_code in _HIGH_FP_RULES:
            tool_factor = -0.20
        elif rule_code in _HIGH_ACCURACY_RULES:
            tool_factor = 0.10

    # 3. Severity factor: blockers from reliable tools get a boost
    severity_factor = 0.0
    if finding.severity == Severity.BLOCKER and base >= 0.8:
        severity_factor = 0.05
    elif finding.severity == Severity.PRAISE:
        severity_factor = 0.0  # praise doesn't need confidence

    # 4. Context factors
    context_factor = 0.0

    # Corroboration boost: multiple tools flagging the same location
    if len(corroborating) >= 1:
        context_factor += 0.10
    if len(corroborating) >= 2:
        context_factor += 0.05

    # Findings in unchanged code are less relevant
    if not file_was_changed:
        context_factor -= 0.15

    # Final score, clamped to [0.1, 1.0]
    raw = base + tool_factor + severity_factor + context_factor
    final = max(0.1, min(1.0, raw))

    explanation = _build_explanation(
        finding, base, tool_factor, severity_factor, context_factor, rule_code
    )

    return ConfidenceFactors(
        base_score=base,
        tool_factor=tool_factor,
        severity_factor=severity_factor,
        context_factor=context_factor,
        final_score=final,
        explanation=explanation,
    )


def _extract_rule(title: str) -> str | None:
    """Extract rule code from title like 'ruff E501'."""
    match = re.search(r"ruff\s+([A-Z]+\d+)", title)
    return match.group(1) if match else None


def _build_explanation(
    finding: Finding,
    base: float,
    tool_factor: float,
    severity_factor: float,
    context_factor: float,
    rule_code: str | None,
) -> str:
    """Build a human-readable explanation of the confidence score."""
    parts = []

    tool = finding.tool_source or "unknown"
    parts.append(f"{tool} base {base:.0%}")

    if tool_factor > 0:
        parts.append(f"high-accuracy rule {rule_code}")
    elif tool_factor < 0:
        parts.append(f"high-FP rule {rule_code}")

    if severity_factor > 0:
        parts.append("blocker boost")

    if context_factor > 0:
        parts.append("corroborated")
    elif context_factor < 0:
        parts.append("unchanged file")

    return ", ".join(parts)
