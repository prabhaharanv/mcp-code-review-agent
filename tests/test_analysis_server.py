"""Tests for the Code Analysis MCP server tools."""

from __future__ import annotations

import json

import pytest

from servers.analysis_server import _ruff_severity


class TestRuffSeverity:
    def test_error_codes(self):
        assert _ruff_severity("E999") == "error"
        assert _ruff_severity("F401") == "error"
        assert _ruff_severity("F811") == "error"

    def test_warning_codes(self):
        assert _ruff_severity("E101") == "warning"
        assert _ruff_severity("E501") == "warning"
        assert _ruff_severity("W291") == "warning"

    def test_info_codes(self):
        assert _ruff_severity("I001") == "info"
        assert _ruff_severity("D100") == "info"
        assert _ruff_severity("N801") == "info"


@pytest.mark.asyncio
async def test_analyze_complexity_simple():
    from servers.analysis_server import analyze_complexity

    code = """\
def hello(name):
    return f"Hello, {name}!"

class Greeter:
    def greet(self, name):
        return hello(name)
"""
    result = json.loads(await analyze_complexity(code))

    assert result["function_count"] == 2  # hello + greet
    assert result["class_count"] == 1  # Greeter
    assert result["warnings"]["long_functions"] == []
    assert result["warnings"]["too_many_args"] == []


@pytest.mark.asyncio
async def test_analyze_complexity_syntax_error():
    from servers.analysis_server import analyze_complexity

    result = json.loads(await analyze_complexity("def broken("))
    assert "error" in result
    assert "SyntaxError" in result["error"]


@pytest.mark.asyncio
async def test_analyze_complexity_many_args():
    from servers.analysis_server import analyze_complexity

    code = "def too_many(a, b, c, d, e, f, g):\n    pass\n"
    result = json.loads(await analyze_complexity(code))

    assert len(result["warnings"]["too_many_args"]) == 1
    assert result["warnings"]["too_many_args"][0]["name"] == "too_many"
