"""Cross-file reasoning — detects issues that span multiple files.

Single-file analysis misses a class of bugs that only appear when you
look at how files interact:
    - Function signature changed but callers not updated
    - Import added in one file but the module isn't in requirements
    - API endpoint changed but client code still uses old shape
    - Shared constant renamed in one place but not another
    - Test file doesn't cover the newly added module

This module analyzes the PR's file set as a whole to surface these issues.
"""

from __future__ import annotations

import re
from collections import defaultdict

from agent.models import Finding, Severity


def analyze_cross_file_issues(
    pr_files: list[dict],
    file_contents: dict[str, str] | None = None,
) -> list[Finding]:
    """Analyze PR files for cross-cutting issues.

    Args:
        pr_files: List of file dicts from GitHub (filename, status, additions, etc.)
        file_contents: Optional map of filename → file content for deeper analysis

    Returns:
        List of cross-file findings
    """
    findings: list[Finding] = []
    contents = file_contents or {}

    filenames = [f.get("filename", "") for f in pr_files]
    statuses = {f.get("filename", ""): f.get("status", "") for f in pr_files}
    python_files = [f for f in filenames if f.endswith(".py")]
    source_files = [f for f in python_files if not _is_test_file(f)]
    test_files = [f for f in python_files if _is_test_file(f)]

    # ── Check 1: Missing test coverage ───────────────────────
    findings.extend(_check_missing_tests(source_files, test_files, statuses))

    # ── Check 2: Init file consistency ───────────────────────
    findings.extend(_check_init_exports(python_files, statuses))

    # ── Check 3: Import consistency (if we have file contents) ─
    if contents:
        findings.extend(_check_import_consistency(contents, statuses))

    # ── Check 4: API contract changes ────────────────────────
    if contents:
        findings.extend(_check_api_contract_changes(contents, statuses))

    # ── Check 5: Requirements consistency ────────────────────
    findings.extend(_check_requirements_consistency(filenames, statuses, contents))

    return findings


def _check_missing_tests(
    source_files: list[str],
    test_files: list[str],
    statuses: dict[str, str],
) -> list[Finding]:
    """Flag new or heavily modified source files with no corresponding test changes."""
    findings = []
    test_basenames = {_test_target(t) for t in test_files}

    for src in source_files:
        status = statuses.get(src, "")
        basename = _source_basename(src)

        # Only flag new files or significant modifications
        if status in ("added", "modified") and basename not in test_basenames:
            findings.append(
                Finding(
                    severity=Severity.WARNING,
                    file_path=src,
                    line=None,
                    title=f"No test coverage for `{_short_name(src)}`",
                    description=(
                        f"File `{src}` was {status} but no corresponding test file "
                        f"(e.g. `test_{basename}.py`) was changed in this PR."
                    ),
                    suggestion=f"Add or update tests for `{_short_name(src)}`",
                    tool_source="cross_file_analysis",
                )
            )

    return findings


def _check_init_exports(
    python_files: list[str],
    statuses: dict[str, str],
) -> list[Finding]:
    """Check if new modules were added without updating __init__.py."""
    findings = []

    # Group files by directory
    dirs_with_new_files: dict[str, list[str]] = defaultdict(list)
    init_files_changed: set[str] = set()

    for f in python_files:
        if f.endswith("__init__.py"):
            init_files_changed.add(_dir_of(f))
            continue

        if statuses.get(f) == "added":
            dirs_with_new_files[_dir_of(f)].append(f)

    for dir_path, new_files in dirs_with_new_files.items():
        if dir_path and dir_path not in init_files_changed:
            findings.append(
                Finding(
                    severity=Severity.NIT,
                    file_path=f"{dir_path}/__init__.py",
                    line=None,
                    title="New module(s) added without updating `__init__.py`",
                    description=(
                        f"New files added to `{dir_path}/`: {', '.join(_short_name(f) for f in new_files)}. "
                        f"Consider updating `__init__.py` if these should be public exports."
                    ),
                    tool_source="cross_file_analysis",
                )
            )

    return findings


def _check_import_consistency(
    contents: dict[str, str],
    statuses: dict[str, str],
) -> list[Finding]:
    """Check for imports of modules that were deleted or renamed."""
    findings = []

    removed_modules = set()
    for filepath, status in statuses.items():
        if status == "removed" and filepath.endswith(".py"):
            module = _filepath_to_module(filepath)
            if module:
                removed_modules.add(module)

    if not removed_modules:
        return findings

    for filepath, content in contents.items():
        if statuses.get(filepath) == "removed":
            continue

        for module in removed_modules:
            # Check for imports of the removed module
            patterns = [
                rf"from\s+{re.escape(module)}\s+import",
                rf"import\s+{re.escape(module)}",
            ]
            for pattern in patterns:
                match = re.search(pattern, content)
                if match:
                    # Find line number
                    line_num = content[:match.start()].count("\n") + 1
                    findings.append(
                        Finding(
                            severity=Severity.BLOCKER,
                            file_path=filepath,
                            line=line_num,
                            title=f"Import of deleted module `{module}`",
                            description=(
                                f"`{filepath}` imports `{module}` which was deleted in this PR. "
                                f"This will cause an ImportError at runtime."
                            ),
                            suggestion=f"Update or remove the import of `{module}`",
                            tool_source="cross_file_analysis",
                        )
                    )

    return findings


def _check_api_contract_changes(
    contents: dict[str, str],
    statuses: dict[str, str],
) -> list[Finding]:
    """Detect function signature changes that may break callers.

    This is a lightweight heuristic: if a function definition's parameter list
    changed, check if any other file in the PR calls that function.
    """
    findings = []

    # Collect defined functions from modified files
    defined_functions: dict[str, list[str]] = {}  # func_name → [defining_files]
    for filepath, content in contents.items():
        if statuses.get(filepath) not in ("modified", "added"):
            continue
        for match in re.finditer(r"def\s+(\w+)\s*\(", content):
            fn_name = match.group(1)
            if not fn_name.startswith("_"):  # only public functions
                defined_functions.setdefault(fn_name, []).append(filepath)

    # For each public function defined in a modified file, check if callers exist
    # but weren't updated
    for fn_name, defining_files in defined_functions.items():
        if len(defining_files) > 3:
            continue  # too common a name (e.g. 'get', 'run') — skip

        for filepath, content in contents.items():
            if filepath in defining_files:
                continue  # same file
            if statuses.get(filepath) == "modified":
                continue  # caller was updated too — probably fine

            # Check if this file calls the function
            call_pattern = rf"\b{re.escape(fn_name)}\s*\("
            match = re.search(call_pattern, content)
            if match:
                line_num = content[:match.start()].count("\n") + 1
                findings.append(
                    Finding(
                        severity=Severity.WARNING,
                        file_path=filepath,
                        line=line_num,
                        title=f"Caller of modified function `{fn_name}()` not updated",
                        description=(
                            f"`{filepath}` calls `{fn_name}()` which was modified in "
                            f"{', '.join(defining_files)}. Verify the call site is compatible "
                            f"with the new signature."
                        ),
                        suggestion=f"Review the call to `{fn_name}()` at line {line_num}",
                        tool_source="cross_file_analysis",
                    )
                )

    return findings


def _check_requirements_consistency(
    filenames: list[str],
    statuses: dict[str, str],
    contents: dict[str, str],
) -> list[Finding]:
    """Check if new imports were added without updating requirements.txt."""
    findings = []

    # Only check if requirements.txt exists and wasn't changed
    req_files = {"requirements.txt", "setup.py", "pyproject.toml", "setup.cfg"}
    req_changed = any(f in req_files for f in filenames)

    if req_changed:
        return findings  # requirements were updated — skip

    # Look for new third-party imports in added files
    stdlib_modules = _get_common_stdlib()
    new_imports: set[str] = set()

    for filepath, content in contents.items():
        if not filepath.endswith(".py"):
            continue
        if statuses.get(filepath) != "added":
            continue

        for match in re.finditer(r"^(?:from|import)\s+(\w+)", content, re.MULTILINE):
            module = match.group(1)
            if module not in stdlib_modules and not _is_local_module(module, filenames):
                new_imports.add(module)

    if new_imports:
        findings.append(
            Finding(
                severity=Severity.WARNING,
                file_path="requirements.txt",
                line=None,
                title="New dependencies may be missing from requirements",
                description=(
                    f"New files import these potentially third-party modules: "
                    f"{', '.join(sorted(new_imports))}. "
                    f"If any are new dependencies, add them to requirements.txt."
                ),
                suggestion="Verify these imports and update requirements.txt if needed",
                tool_source="cross_file_analysis",
            )
        )

    return findings


# ── Helpers ───────────────────────────────────────────────────


def _is_test_file(filename: str) -> bool:
    base = filename.rsplit("/", 1)[-1] if "/" in filename else filename
    return base.startswith("test_") or base.endswith("_test.py")


def _source_basename(filepath: str) -> str:
    """Get the module name from a source file path: 'src/utils/parser.py' → 'parser'."""
    base = filepath.rsplit("/", 1)[-1] if "/" in filepath else filepath
    return base.removesuffix(".py")


def _test_target(test_filepath: str) -> str:
    """Get the target module from a test file: 'tests/test_parser.py' → 'parser'."""
    base = test_filepath.rsplit("/", 1)[-1] if "/" in test_filepath else test_filepath
    base = base.removesuffix(".py")
    if base.startswith("test_"):
        return base[5:]
    if base.endswith("_test"):
        return base[:-5]
    return base


def _short_name(filepath: str) -> str:
    return filepath.rsplit("/", 1)[-1] if "/" in filepath else filepath


def _dir_of(filepath: str) -> str:
    return filepath.rsplit("/", 1)[0] if "/" in filepath else ""


def _filepath_to_module(filepath: str) -> str | None:
    """Convert 'src/utils/parser.py' to 'src.utils.parser'."""
    if not filepath.endswith(".py"):
        return None
    return filepath.removesuffix(".py").replace("/", ".")


def _is_local_module(module: str, filenames: list[str]) -> bool:
    """Check if a module name corresponds to a local file in the PR."""
    return any(
        f.endswith(f"{module}.py") or f.endswith(f"{module}/__init__.py")
        for f in filenames
    )


def _get_common_stdlib() -> frozenset[str]:
    """Return common Python stdlib module names (not exhaustive but covers common cases)."""
    return frozenset({
        "abc", "argparse", "ast", "asyncio", "base64", "bisect", "builtins",
        "calendar", "cgi", "cmath", "codecs", "collections", "colorsys",
        "concurrent", "configparser", "contextlib", "copy", "csv", "ctypes",
        "dataclasses", "datetime", "decimal", "difflib", "dis", "email",
        "enum", "errno", "exceptions", "fileinput", "fnmatch", "fractions",
        "ftplib", "functools", "gc", "getopt", "getpass", "gettext", "glob",
        "gzip", "hashlib", "heapq", "hmac", "html", "http", "imaplib",
        "importlib", "inspect", "io", "ipaddress", "itertools", "json",
        "keyword", "linecache", "locale", "logging", "lzma", "mailbox",
        "math", "mimetypes", "mmap", "multiprocessing", "numbers", "operator",
        "optparse", "os", "pathlib", "pdb", "pickle", "pkgutil", "platform",
        "plistlib", "pprint", "profile", "pstats", "py_compile", "queue",
        "quopri", "random", "re", "readline", "reprlib", "resource",
        "rlcompleter", "sched", "secrets", "select", "shelve", "shlex",
        "shutil", "signal", "site", "smtplib", "socket", "socketserver",
        "sqlite3", "ssl", "stat", "statistics", "string", "struct",
        "subprocess", "sys", "sysconfig", "syslog", "tabnanny", "tarfile",
        "tempfile", "textwrap", "threading", "time", "timeit", "token",
        "tokenize", "tomllib", "traceback", "tracemalloc", "turtle",
        "types", "typing", "unicodedata", "unittest", "urllib", "uuid",
        "venv", "warnings", "wave", "weakref", "webbrowser", "xml",
        "xmlrpc", "zipfile", "zipimport", "zlib",
        # Common aliases
        "os.path", "__future__",
    })
