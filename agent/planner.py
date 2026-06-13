"""Review planner — analyzes PR metadata to create a structured review plan.

The planner runs BEFORE the main agent loop. It reads the PR metadata and
file list, then produces a ReviewPlan that guides the agent's tool usage.
"""

from __future__ import annotations

from agent.models import ReviewPlan

# File extensions that benefit from specific tools
PYTHON_EXTENSIONS = {".py"}
TYPE_CHECKABLE = {".py"}
LINTABLE = {".py"}

# Files that are high-risk and deserve extra attention
HIGH_RISK_PATTERNS = [
    "auth", "security", "crypt", "secret", "token", "password",
    "migration", "deploy", "config", "permission", "middleware",
]


def create_review_plan(
    pr_metadata: dict,
    pr_files: list[dict],
    max_steps: int = 20,
) -> ReviewPlan:
    """Create a review plan from PR metadata and file list.

    This is a deterministic planner (no LLM call) that uses heuristics
    to decide what to check. The agent can override this plan if the LLM
    decides something else is more important.

    Args:
        pr_metadata: Parsed output from get_pr_metadata
        pr_files: Parsed output from list_pr_files
    """
    files_to_analyze = []
    checks_to_run = set()
    risk_areas = []

    for f in pr_files:
        filename = f.get("filename", "")
        status = f.get("status", "")
        additions = f.get("additions", 0)

        # Skip deleted files — nothing to review
        if status == "removed":
            continue

        ext = _get_extension(filename)

        # Decide if this file needs deep analysis
        needs_analysis = (
            additions > 5  # non-trivial changes
            or status == "added"  # new files always need review
            or _is_high_risk(filename)
        )

        if needs_analysis:
            files_to_analyze.append((filename, additions, _is_high_risk(filename)))

        # Determine which tools to run
        if ext in LINTABLE:
            checks_to_run.add("run_ruff")
        if ext in TYPE_CHECKABLE and additions > 20:
            checks_to_run.add("run_mypy")
        if ext in PYTHON_EXTENSIONS and additions > 30:
            checks_to_run.add("analyze_complexity")

        # Flag high-risk areas
        if _is_high_risk(filename):
            risk_areas.append(f"{filename} — touches security/config/auth")

    # Check if tests exist for changed source files
    source_files = [name for name, _, _ in files_to_analyze if not _is_test_file(name)]
    test_files = [name for name, _, _ in files_to_analyze if _is_test_file(name)]
    if source_files and not test_files:
        risk_areas.append("No test files changed — missing test coverage?")
        checks_to_run.add("find_test_files")

    # Large PRs are riskier
    total_changes = pr_metadata.get("additions", 0) + pr_metadata.get("deletions", 0)
    if total_changes > 500:
        risk_areas.append(f"Large PR ({total_changes} lines changed) — higher review risk")
    if len(pr_files) > 15:
        risk_areas.append(f"Touches {len(pr_files)} files — consider splitting")

    # Prioritize high-risk files first, then by additions, before capping
    files_to_analyze.sort(key=lambda x: (x[2], x[1]), reverse=True)
    prioritized = [name for name, _, _ in files_to_analyze[:20]]

    # Estimate steps: 1 for metadata + 1 per file to read + 1 per check + 1 to post
    estimated_steps = 1 + len(prioritized) + len(checks_to_run) + 1

    summary = _build_summary(pr_metadata)

    return ReviewPlan(
        summary=summary,
        files_to_analyze=prioritized,
        checks_to_run=sorted(checks_to_run),
        risk_areas=risk_areas,
        estimated_steps=min(estimated_steps, max_steps),
    )


def plan_to_prompt_context(plan: ReviewPlan) -> str:
    """Convert a ReviewPlan into text the agent can use as context."""
    lines = [
        "## Review Plan",
        f"**Summary**: {plan.summary}",
        "",
        f"**Files to analyze** ({len(plan.files_to_analyze)}):",
    ]
    for f in plan.files_to_analyze:
        lines.append(f"  - {f}")

    lines.append("")
    lines.append(f"**Checks to run**: {', '.join(plan.checks_to_run) or 'none'}")

    if plan.risk_areas:
        lines.append("")
        lines.append("**Risk areas**:")
        for r in plan.risk_areas:
            lines.append(f"  ⚠ {r}")

    lines.append("")
    lines.append(f"**Estimated tool calls**: ~{plan.estimated_steps}")

    return "\n".join(lines)


# ── Helpers ───────────────────────────────────────────────────


def _get_extension(filename: str) -> str:
    dot = filename.rfind(".")
    return filename[dot:] if dot != -1 else ""


def _is_high_risk(filename: str) -> bool:
    lower = filename.lower()
    return any(pattern in lower for pattern in HIGH_RISK_PATTERNS)


def _is_test_file(filename: str) -> bool:
    base = filename.rsplit("/", 1)[-1] if "/" in filename else filename
    return base.startswith("test_") or base.endswith("_test.py")


def _build_summary(metadata: dict) -> str:
    title = metadata.get("title", "Unknown PR")
    author = metadata.get("author", "unknown")
    additions = metadata.get("additions", 0)
    deletions = metadata.get("deletions", 0)
    changed = metadata.get("changed_files", 0)
    return (
        f"PR '{title}' by {author}: "
        f"+{additions}/-{deletions} across {changed} file(s)"
    )
