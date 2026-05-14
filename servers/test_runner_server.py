"""Test Runner MCP Server — run tests and report results via MCP.

Tools:
    run_pytest        — Run pytest on a file or directory, return structured results
    find_test_files   — Find test files related to changed source files
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("test-runner")


async def _run_command(cmd: list[str], cwd: str | None = None) -> tuple[str, str, int]:
    """Run a shell command and return (stdout, stderr, returncode)."""
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
    )
    stdout, stderr = await proc.communicate()
    return stdout.decode(), stderr.decode(), proc.returncode


@mcp.tool()
async def run_pytest(
    test_path: str,
    repo_dir: str,
    markers: str = "",
    timeout: int = 120,
) -> str:
    """Run pytest on a test file or directory and return structured results.

    Args:
        test_path: Path to the test file or directory (relative to repo_dir)
        repo_dir: Absolute path to the cloned repository root
        markers: Optional pytest marker expression (e.g. 'not slow')
        timeout: Maximum seconds to wait for tests to complete
    """
    if not os.path.isdir(repo_dir):
        return json.dumps({"error": f"repo_dir does not exist: {repo_dir}"})

    # Write results to a temp JSON file
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        report_path = tmp.name

    cmd = [
        "python", "-m", "pytest",
        test_path,
        f"--json-report --json-report-file={report_path}",
        "--tb=short",
        "--no-header",
        "-q",
    ]
    if markers:
        cmd.extend(["-m", markers])

    # Fall back to plain output if json-report plugin isn't installed
    cmd_plain = [
        "python", "-m", "pytest",
        test_path,
        "--tb=short",
        "--no-header",
        "-q",
    ]
    if markers:
        cmd_plain.extend(["-m", markers])

    try:
        stdout, stderr, rc = await asyncio.wait_for(
            _run_command(cmd_plain, cwd=repo_dir),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        return json.dumps({"error": f"Tests timed out after {timeout}s"})

    # Parse the summary line (e.g., "5 passed, 2 failed in 1.23s")
    lines = stdout.strip().splitlines()
    summary_line = lines[-1] if lines else ""

    passed = failed = errors = 0
    for part in summary_line.split(","):
        part = part.strip()
        if "passed" in part:
            passed = int(part.split()[0])
        elif "failed" in part:
            failed = int(part.split()[0])
        elif "error" in part:
            errors = int(part.split()[0])

    # Extract failure details
    failures = []
    capture = False
    current_failure: list[str] = []
    for line in lines:
        if line.startswith("FAILED "):
            failures.append(line)
        elif line.startswith("_____") or line.startswith("====="):
            if current_failure:
                failures.append("\n".join(current_failure))
                current_failure = []
            capture = "FAILURES" in line
        elif capture:
            current_failure.append(line)

    if current_failure:
        failures.append("\n".join(current_failure))

    # Clean up temp file
    try:
        os.unlink(report_path)
    except OSError:
        pass

    return json.dumps(
        {
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "exit_code": rc,
            "summary": summary_line,
            "failures": failures[:10],  # Cap at 10 to avoid huge output
            "raw_output": stdout[-2000:] if len(stdout) > 2000 else stdout,
        },
        indent=2,
    )


@mcp.tool()
async def find_test_files(repo_dir: str, source_files: str) -> str:
    """Find test files that likely correspond to the given source files.

    Uses conventions: src/foo.py → tests/test_foo.py, test_foo.py, foo_test.py

    Args:
        repo_dir: Absolute path to the repository root
        source_files: JSON array of source file paths (relative to repo_dir)
    """
    if not os.path.isdir(repo_dir):
        return json.dumps({"error": f"repo_dir does not exist: {repo_dir}"})

    changed = json.loads(source_files)
    found: dict[str, list[str]] = {}

    # Build index of all test files in the repo
    test_files: list[str] = []
    for root, _dirs, files in os.walk(repo_dir):
        for f in files:
            if f.startswith("test_") and f.endswith(".py"):
                rel = os.path.relpath(os.path.join(root, f), repo_dir)
                test_files.append(rel)
            elif f.endswith("_test.py"):
                rel = os.path.relpath(os.path.join(root, f), repo_dir)
                test_files.append(rel)

    for src in changed:
        basename = os.path.basename(src).removesuffix(".py")
        candidates = [
            t
            for t in test_files
            if f"test_{basename}" in t or f"{basename}_test" in t
        ]
        found[src] = candidates

    return json.dumps(found, indent=2)


if __name__ == "__main__":
    mcp.run()
