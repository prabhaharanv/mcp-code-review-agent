"""Observation parser — converts raw MCP tool outputs into structured Findings.

The agent receives raw JSON strings from MCP tools. This module parses
them into typed Finding objects the reviewer can reason about.
"""

from __future__ import annotations

import json

from agent.models import Finding, Severity


def parse_ruff_output(raw: str, file_path: str) -> list[Finding]:
    """Parse ruff JSON output into Findings."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []

    diagnostics = data.get("diagnostics", [])
    findings = []
    for d in diagnostics:
        severity = _map_ruff_severity(d.get("severity", "info"))
        findings.append(
            Finding(
                severity=severity,
                file_path=file_path,
                line=d.get("line"),
                title=f"ruff {d.get('rule', 'unknown')}",
                description=d.get("message", ""),
                tool_source="run_ruff",
            )
        )
    return findings


def parse_mypy_output(raw: str, file_path: str) -> list[Finding]:
    """Parse mypy output into Findings."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []

    diagnostics = data.get("diagnostics", [])
    findings = []
    for d in diagnostics:
        raw_line = d.get("raw", "")
        severity = Severity.WARNING
        if ": error:" in raw_line:
            severity = Severity.BLOCKER

        findings.append(
            Finding(
                severity=severity,
                file_path=file_path,
                line=_extract_line_from_mypy(raw_line),
                title="mypy type error",
                description=raw_line,
                tool_source="run_mypy",
            )
        )
    return findings


def parse_complexity_output(raw: str, file_path: str) -> list[Finding]:
    """Parse analyze_complexity output into Findings."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []

    if "error" in data:
        return [
            Finding(
                severity=Severity.BLOCKER,
                file_path=file_path,
                line=None,
                title="Syntax error",
                description=data["error"],
                tool_source="analyze_complexity",
            )
        ]

    findings = []
    for fn in data.get("warnings", {}).get("long_functions", []):
        findings.append(
            Finding(
                severity=Severity.WARNING,
                file_path=file_path,
                line=fn.get("line"),
                title=f"Function `{fn['name']}` too long ({fn['length']} lines)",
                description="Functions longer than 50 lines are harder to test and maintain. Consider breaking into smaller functions.",
                suggestion=f"Split `{fn['name']}` into smaller, focused functions",
                tool_source="analyze_complexity",
            )
        )

    for fn in data.get("warnings", {}).get("too_many_args", []):
        findings.append(
            Finding(
                severity=Severity.NIT,
                file_path=file_path,
                line=fn.get("line"),
                title=f"Function `{fn['name']}` has {fn['args']} parameters",
                description="Functions with more than 5 parameters suggest the need for a config object or dataclass.",
                suggestion=f"Group `{fn['name']}` parameters into a dataclass or config object",
                tool_source="analyze_complexity",
            )
        )

    return findings


def parse_test_output(raw: str) -> list[Finding]:
    """Parse pytest output into Findings (failures become blockers)."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []

    findings = []
    if data.get("failed", 0) > 0:
        for failure in data.get("failures", []):
            findings.append(
                Finding(
                    severity=Severity.BLOCKER,
                    file_path="tests/",
                    line=None,
                    title="Test failure",
                    description=failure[:500] if isinstance(failure, str) else str(failure)[:500],
                    tool_source="run_pytest",
                )
            )
    elif data.get("errors", 0) > 0:
        findings.append(
            Finding(
                severity=Severity.BLOCKER,
                file_path="tests/",
                line=None,
                title=f"Test errors: {data['errors']} error(s)",
                description=data.get("summary", ""),
                tool_source="run_pytest",
            )
        )

    return findings


def parse_tool_output(tool_name: str, raw: str, file_path: str = "") -> list[Finding]:
    """Route tool output to the correct parser.

    Args:
        tool_name: The MCP tool that produced this output
        raw: Raw string output from the tool
        file_path: File path context (if applicable)

    Returns:
        List of structured Findings
    """
    parsers = {
        "run_ruff": lambda: parse_ruff_output(raw, file_path),
        "run_mypy": lambda: parse_mypy_output(raw, file_path),
        "analyze_complexity": lambda: parse_complexity_output(raw, file_path),
        "run_pytest": lambda: parse_test_output(raw),
    }

    parser = parsers.get(tool_name)
    if parser:
        return parser()
    return []


# ── Helpers ───────────────────────────────────────────────────


def _map_ruff_severity(severity: str) -> Severity:
    mapping = {
        "error": Severity.BLOCKER,
        "warning": Severity.WARNING,
        "info": Severity.NIT,
    }
    return mapping.get(severity, Severity.NIT)


def _extract_line_from_mypy(raw_line: str) -> int | None:
    """Extract line number from mypy output like 'file.py:10:5: error: ...'"""
    parts = raw_line.split(":")
    if len(parts) >= 2:
        try:
            return int(parts[1])
        except ValueError:
            pass
    return None
