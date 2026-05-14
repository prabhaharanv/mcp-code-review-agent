"""Tests for the GitHub MCP server tools."""

from __future__ import annotations

import pytest

from servers.github_server import _parse_pr_url


class TestParsePrUrl:
    def test_valid_url(self):
        owner, repo, number = _parse_pr_url(
            "https://github.com/octocat/hello-world/pull/42"
        )
        assert owner == "octocat"
        assert repo == "hello-world"
        assert number == 42

    def test_valid_url_with_trailing_parts(self):
        owner, repo, number = _parse_pr_url(
            "https://github.com/owner/repo/pull/123"
        )
        assert owner == "owner"
        assert repo == "repo"
        assert number == 123

    def test_invalid_url_raises(self):
        with pytest.raises(ValueError, match="Invalid PR URL"):
            _parse_pr_url("https://github.com/owner/repo/issues/1")

    def test_not_github_url(self):
        with pytest.raises(ValueError, match="Invalid PR URL"):
            _parse_pr_url("https://gitlab.com/owner/repo/pull/1")

    def test_empty_string(self):
        with pytest.raises(ValueError, match="Invalid PR URL"):
            _parse_pr_url("")

    def test_http_url(self):
        owner, repo, number = _parse_pr_url(
            "http://github.com/owner/repo/pull/99"
        )
        assert owner == "owner"
        assert number == 99
