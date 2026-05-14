"""Code Analysis MCP Server — static analysis tools exposed via MCP.

Tools:
    run_ruff       — Run ruff linter on a file or directory
    run_mypy       — Run mypy type checker on a file or directory
    analyze_complexity — Analyze code complexity (functions, classes, cyclomatic)
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("code-analysis")


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
async def run_ruff(code: str, filename: str = "review.py") -> str:
    """Run ruff linter on Python code and return diagnostics.

    Args:
        code: The Python source code to lint
        filename: Filename hint for ruff (affects rule selection)
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, filename)
        with open(filepath, "w") as f:
            f.write(code)

        stdout, stderr, rc = await _run_command(
            ["ruff", "check", "--output-format=json", "--no-fix", filepath]
        )

        if not stdout.strip():
            return json.dumps({"diagnostics": [], "summary": "No issues found"})

        try:
            diagnostics = json.loads(stdout)
        except json.JSONDecodeError:
            return json.dumps({"error": stderr or stdout})

        findings = []
        for d in diagnostics:
            findings.append(
                {
                    "rule": d.get("code", ""),
                    "message": d.get("message", ""),
                    "line": d.get("location", {}).get("row", 0),
                    "column": d.get("location", {}).get("column", 0),
                    "severity": _ruff_severity(d.get("code", "")),
                }
            )

        return json.dumps(
            {
                "diagnostics": findings,
                "summary": f"{len(findings)} issue(s) found",
            },
            indent=2,
        )


def _ruff_severity(code: str) -> str:
    """Map ruff rule codes to severity levels."""
    if code.startswith(("E9", "F")):
        return "error"
    if code.startswith(("E", "W")):
        return "warning"
    return "info"


@mcp.tool()
async def run_mypy(code: str, filename: str = "review.py") -> str:
    """Run mypy type checker on Python code.

    Args:
        code: The Python source code to type-check
        filename: Filename hint for mypy
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, filename)
        with open(filepath, "w") as f:
            f.write(code)

        stdout, stderr, rc = await _run_command(
            [
                "mypy",
                "--no-color-output",
                "--no-error-summary",
                "--show-column-numbers",
                "--no-pretty",
                filepath,
            ]
        )

        output = stdout.strip() or stderr.strip()
        if rc == 0 and not output:
            return json.dumps({"diagnostics": [], "summary": "No type errors found"})

        findings = []
        for line in output.splitlines():
            if ": error:" in line or ": warning:" in line or ": note:" in line:
                findings.append({"raw": line})

        return json.dumps(
            {
                "diagnostics": findings,
                "summary": f"{len(findings)} type issue(s) found",
            },
            indent=2,
        )


@mcp.tool()
async def analyze_complexity(code: str) -> str:
    """Analyze Python code structure: function count, class count, line count,
    and flag functions longer than 50 lines.

    Args:
        code: The Python source code to analyze
    """
    import ast

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return json.dumps({"error": f"SyntaxError: {e}"})

    functions = []
    classes = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end = getattr(node, "end_lineno", node.lineno)
            length = end - node.lineno + 1
            functions.append(
                {
                    "name": node.name,
                    "line": node.lineno,
                    "length": length,
                    "args": len(node.args.args),
                    "too_long": length > 50,
                }
            )
        elif isinstance(node, ast.ClassDef):
            classes.append({"name": node.name, "line": node.lineno})

    long_functions = [f for f in functions if f["too_long"]]
    many_args = [f for f in functions if f["args"] > 5]

    total_lines = len(code.splitlines())
    return json.dumps(
        {
            "total_lines": total_lines,
            "function_count": len(functions),
            "class_count": len(classes),
            "functions": functions,
            "classes": classes,
            "warnings": {
                "long_functions": long_functions,
                "too_many_args": many_args,
            },
        },
        indent=2,
    )


if __name__ == "__main__":
    mcp.run()
