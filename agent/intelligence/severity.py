"""Severity classifier — re-evaluates finding severity using surrounding code context.

The parser assigns initial severity based on simple rules (e.g. ruff severity
mapping). This module uses code context + heuristics to upgrade or downgrade
findings that were mis-classified.

For example:
    - A ruff 'unused import' in __init__.py is a NIT (re-exports are common)
    - A type error in a function called 'handle_payment' is a BLOCKER
    - A long function in test code is a NIT, not a WARNING
"""

from __future__ import annotations

import re

from agent.models import Finding, Severity

# Patterns that indicate security-critical code
_SECURITY_PATTERNS = re.compile(
    r"(password|secret|token|api_key|auth|crypt|hash|jwt|session|cookie|csrf|xss|inject|sanitize)",
    re.IGNORECASE,
)

# Patterns indicating data-critical code
_DATA_PATTERNS = re.compile(
    r"(database|migration|delete|drop|truncate|commit|rollback|transaction|persist|save|write)",
    re.IGNORECASE,
)

# Files that are lower risk (test code, docs, configs)
_LOW_RISK_FILE_PATTERNS = re.compile(
    r"(test_|_test\.py|conftest|fixture|mock|stub|fake|\.md$|\.txt$|\.cfg$|\.ini$|\.toml$)",
    re.IGNORECASE,
)

# Init files often have re-exports
_INIT_FILE = re.compile(r"__init__\.py$")

# Rules that are almost always cosmetic
_COSMETIC_RULES = frozenset({
    "E501",  # line too long
    "W291",  # trailing whitespace
    "W292",  # no newline at end of file
    "W293",  # whitespace before ':'
    "E302",  # expected 2 blank lines
    "E303",  # too many blank lines
    "I001",  # import sorting
    "D100",  # missing docstring (module)
    "D101",  # missing docstring (class)
    "D102",  # missing docstring (method)
    "D103",  # missing docstring (function)
})

# Rules that indicate real bugs
_BUG_RULES = frozenset({
    "F821",  # undefined name
    "F811",  # redefined unused name
    "E711",  # comparison to None
    "E712",  # comparison to True/False
    "B006",  # mutable default argument
    "B007",  # unused loop variable
    "S101",  # use of assert in non-test
    "S105",  # hardcoded password
    "S106",  # hardcoded password in func arg
    "S107",  # hardcoded password default
    "S108",  # probable insecure usage of temp file
    "S301",  # pickle usage
    "S608",  # SQL injection
})


def classify_severity(finding: Finding, file_context: str = "") -> Finding:
    """Re-classify a finding's severity using contextual heuristics.

    Args:
        finding: The finding to re-classify
        file_context: Optional surrounding code or full file content

    Returns:
        A new Finding with potentially adjusted severity and a
        classification reason appended to the description.
    """
    new_severity = finding.severity
    reason = ""

    # ── Rule-based reclassification for linter findings ──────
    if finding.tool_source == "run_ruff":
        rule_code = _extract_rule_code(finding.title)
        if rule_code:
            new_severity, reason = _classify_ruff_rule(
                rule_code, finding, file_context
            )

    # ── Context-based escalation ─────────────────────────────
    if finding.severity in (Severity.NIT, Severity.WARNING):
        context_to_check = f"{finding.file_path} {finding.description} {file_context}"

        if _SECURITY_PATTERNS.search(context_to_check):
            new_severity = max(new_severity, Severity.WARNING, key=_severity_rank)
            reason = reason or "Escalated: touches security-sensitive code"

        if _DATA_PATTERNS.search(context_to_check):
            new_severity = max(new_severity, Severity.WARNING, key=_severity_rank)
            reason = reason or "Escalated: touches data-critical code"

    # ── Context-based de-escalation ──────────────────────────
    if finding.severity in (Severity.WARNING, Severity.BLOCKER):
        if _LOW_RISK_FILE_PATTERNS.search(finding.file_path):
            new_severity = min(new_severity, Severity.WARNING, key=_severity_rank)
            reason = reason or "De-escalated: finding is in test/config code"

    # ── Build updated finding ────────────────────────────────
    if new_severity != finding.severity:
        updated_desc = finding.description
        if reason:
            updated_desc = f"{finding.description} [{reason}]"
        return finding.model_copy(
            update={"severity": new_severity, "description": updated_desc}
        )

    return finding


def classify_batch(
    findings: list[Finding],
    file_contexts: dict[str, str] | None = None,
) -> list[Finding]:
    """Re-classify severity for a batch of findings.

    Args:
        findings: List of findings to process
        file_contexts: Optional map of file_path → file content
    """
    contexts = file_contexts or {}
    return [
        classify_severity(f, contexts.get(f.file_path, ""))
        for f in findings
    ]


# ── Internal helpers ──────────────────────────────────────────


def _extract_rule_code(title: str) -> str | None:
    """Extract ruff rule code from finding title like 'ruff E501'."""
    match = re.search(r"ruff\s+([A-Z]+\d+)", title)
    return match.group(1) if match else None


def _classify_ruff_rule(
    rule_code: str,
    finding: Finding,
    file_context: str,
) -> tuple[Severity, str]:
    """Classify a ruff rule into the appropriate severity."""
    if rule_code in _COSMETIC_RULES:
        return Severity.NIT, f"Rule {rule_code} is cosmetic"

    if rule_code in _BUG_RULES:
        # S-rules (bandit security) in security-sensitive files → BLOCKER
        if rule_code.startswith("S") and _SECURITY_PATTERNS.search(finding.file_path):
            return Severity.BLOCKER, f"Security rule {rule_code} in sensitive file"
        return Severity.BLOCKER, f"Rule {rule_code} indicates a likely bug"

    # Unused import in __init__.py is usually intentional (re-export)
    if rule_code == "F401" and _INIT_FILE.search(finding.file_path):
        return Severity.NIT, "Unused import in __init__.py is likely a re-export"

    return finding.severity, ""


_SEVERITY_RANK = {
    Severity.PRAISE: 0,
    Severity.NIT: 1,
    Severity.WARNING: 2,
    Severity.BLOCKER: 3,
}


def _severity_rank(s: Severity) -> int:
    return _SEVERITY_RANK.get(s, 1)
