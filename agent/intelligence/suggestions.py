"""Fix suggestion engine — generates concrete code fixes for findings.

Instead of vague advice like "consider refactoring", this module produces
actionable suggestions with before/after code snippets that the developer
can apply directly.

Strategies:
    1. Pattern-based fixes: Known linter rules have known fixes
    2. Context-aware fixes: Use surrounding code to generate targeted suggestions
    3. Structural fixes: For complexity issues, suggest specific decompositions
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from agent.models import Finding, Severity


@dataclass
class FixSuggestion:
    """A concrete fix suggestion with optional before/after code."""

    summary: str
    before: str | None = None
    after: str | None = None
    confidence: float = 0.8  # how confident we are this fix is correct
    automated: bool = False  # whether this can be auto-applied


# ── Known fixes for common linter rules ──────────────────────

_RUFF_FIXES: dict[str, callable] = {}


def _ruff_fix(rule_code: str):
    """Decorator to register a fix generator for a ruff rule."""
    def decorator(fn):
        _RUFF_FIXES[rule_code] = fn
        return fn
    return decorator


@_ruff_fix("F401")
def _fix_unused_import(finding: Finding, context: str) -> FixSuggestion:
    """Fix unused import by removing or marking as re-export."""
    # Extract the import name from description
    match = re.search(r"`(.+?)`\s+imported but unused", finding.description)
    import_name = match.group(1) if match else "the unused import"

    if "__init__.py" in finding.file_path:
        return FixSuggestion(
            summary=f"If `{import_name}` is a re-export, add `__all__` or use `noqa: F401`",
            before=f"from module import {import_name}",
            after=f"from module import {import_name}  # noqa: F401  (re-export)",
            confidence=0.7,
        )
    return FixSuggestion(
        summary=f"Remove unused import `{import_name}`",
        confidence=0.95,
        automated=True,
    )


@_ruff_fix("B006")
def _fix_mutable_default(finding: Finding, context: str) -> FixSuggestion:
    return FixSuggestion(
        summary="Replace mutable default argument with `None` and initialize inside the function",
        before="def func(items=[]):\n    items.append(x)",
        after="def func(items=None):\n    if items is None:\n        items = []\n    items.append(x)",
        confidence=0.95,
        automated=False,
    )


@_ruff_fix("E711")
def _fix_none_comparison(finding: Finding, context: str) -> FixSuggestion:
    return FixSuggestion(
        summary="Use `is None` / `is not None` instead of `== None` / `!= None`",
        before="if x == None:",
        after="if x is None:",
        confidence=0.99,
        automated=True,
    )


@_ruff_fix("E712")
def _fix_bool_comparison(finding: Finding, context: str) -> FixSuggestion:
    return FixSuggestion(
        summary="Use truthiness check instead of comparing to `True`/`False`",
        before="if x == True:",
        after="if x:",
        confidence=0.9,
        automated=True,
    )


@_ruff_fix("S105")
def _fix_hardcoded_password(finding: Finding, context: str) -> FixSuggestion:
    return FixSuggestion(
        summary="Move hardcoded secret to environment variable or secrets manager",
        before='password = "hunter2"',
        after='password = os.environ["DB_PASSWORD"]',
        confidence=0.85,
    )


@_ruff_fix("S608")
def _fix_sql_injection(finding: Finding, context: str) -> FixSuggestion:
    return FixSuggestion(
        summary="Use parameterized queries instead of string formatting for SQL",
        before='cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")',
        after='cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))',
        confidence=0.95,
    )


# ── Complexity fixes ─────────────────────────────────────────

def _suggest_function_split(finding: Finding, context: str) -> FixSuggestion:
    """Suggest splitting a long function into smaller pieces."""
    match = re.search(r"`(\w+)`.*?(\d+)\s*lines", finding.title)
    fn_name = match.group(1) if match else "the function"
    line_count = int(match.group(2)) if match else 0

    return FixSuggestion(
        summary=f"Extract logical sections of `{fn_name}` into helper functions",
        before=f"def {fn_name}(...):\n    # {line_count} lines of mixed concerns",
        after=(
            f"def {fn_name}(...):\n"
            f"    data = _validate_input(...)\n"
            f"    result = _process(data)\n"
            f"    return _format_output(result)"
        ),
        confidence=0.6,  # lower — we don't know the actual code
    )


def _suggest_param_object(finding: Finding, context: str) -> FixSuggestion:
    """Suggest grouping parameters into a config/dataclass."""
    match = re.search(r"`(\w+)`.*?(\d+)\s*param", finding.title)
    fn_name = match.group(1) if match else "the function"
    param_count = int(match.group(2)) if match else 0

    return FixSuggestion(
        summary=f"Group `{fn_name}`'s {param_count} parameters into a dataclass or config",
        before=f"def {fn_name}(a, b, c, d, e, f, g):",
        after=(
            f"@dataclass\n"
            f"class {fn_name.title()}Config:\n"
            f"    a: str\n    b: int\n    # ...\n\n"
            f"def {fn_name}(config: {fn_name.title()}Config):"
        ),
        confidence=0.5,
    )


# ── Test failure fixes ───────────────────────────────────────

def _suggest_test_fix(finding: Finding, context: str) -> FixSuggestion:
    """Provide guidance for test failures."""
    return FixSuggestion(
        summary="Investigate failing test — check assertion values and test fixtures",
        confidence=0.3,  # very low — we can't know the actual fix
    )


# ── Public API ───────────────────────────────────────────────

def suggest_fix(finding: Finding, file_context: str = "") -> Finding:
    """Generate a fix suggestion for a finding and attach it.

    If the finding already has a suggestion, this enriches it.
    If no fix strategy matches, the finding is returned unchanged.

    Args:
        finding: The finding to generate a fix for
        file_context: Optional surrounding code

    Returns:
        Finding with suggestion field populated
    """
    fix = _generate_fix(finding, file_context)
    if fix is None:
        return finding

    # Build a rich suggestion string
    parts = [fix.summary]
    if fix.before and fix.after:
        parts.append(f"\n**Before:**\n```python\n{fix.before}\n```")
        parts.append(f"**After:**\n```python\n{fix.after}\n```")

    suggestion_text = "\n".join(parts)

    return finding.model_copy(update={"suggestion": suggestion_text})


def suggest_fixes_batch(
    findings: list[Finding],
    file_contexts: dict[str, str] | None = None,
) -> list[Finding]:
    """Generate fix suggestions for a batch of findings."""
    contexts = file_contexts or {}
    return [
        suggest_fix(f, contexts.get(f.file_path, ""))
        for f in findings
    ]


def _generate_fix(finding: Finding, context: str) -> FixSuggestion | None:
    """Pick the right fix strategy for a finding."""
    # Ruff-specific fixes
    if finding.tool_source == "run_ruff":
        rule_match = re.search(r"ruff\s+([A-Z]+\d+)", finding.title)
        if rule_match:
            rule_code = rule_match.group(1)
            fix_fn = _RUFF_FIXES.get(rule_code)
            if fix_fn:
                return fix_fn(finding, context)

    # Complexity fixes
    if finding.tool_source == "analyze_complexity":
        if "too long" in finding.title.lower() or "lines" in finding.title.lower():
            return _suggest_function_split(finding, context)
        if "parameter" in finding.title.lower() or "arg" in finding.title.lower():
            return _suggest_param_object(finding, context)

    # Test failures
    if finding.tool_source == "run_pytest":
        return _suggest_test_fix(finding, context)

    # Type errors
    if finding.tool_source == "run_mypy":
        return FixSuggestion(
            summary="Add or fix type annotation to resolve the type error",
            confidence=0.5,
        )

    return None
