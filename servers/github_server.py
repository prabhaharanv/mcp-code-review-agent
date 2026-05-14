"""GitHub MCP Server — exposes GitHub PR operations as MCP tools.

Tools:
    get_pr_metadata    — PR title, description, author, base/head branches
    list_pr_files      — files changed in the PR with status and patch
    get_pr_diff        — full unified diff of the PR
    get_file_contents  — read a file at the PR's head ref
    post_review        — post a structured code review on the PR
"""

from __future__ import annotations

import json
import os
import re

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("github")

GITHUB_API = "https://api.github.com"


def _headers() -> dict[str, str]:
    token = os.getenv("GITHUB_TOKEN", "")
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _parse_pr_url(pr_url: str) -> tuple[str, str, int]:
    """Extract (owner, repo, pr_number) from a GitHub PR URL."""
    match = re.match(
        r"https?://github\.com/([^/]+)/([^/]+)/pull/(\d+)", pr_url
    )
    if not match:
        raise ValueError(
            f"Invalid PR URL: {pr_url}. "
            "Expected format: https://github.com/owner/repo/pull/123"
        )
    return match.group(1), match.group(2), int(match.group(3))


@mcp.tool()
async def get_pr_metadata(pr_url: str) -> str:
    """Get PR metadata: title, description, author, branches, and labels.

    Args:
        pr_url: Full GitHub PR URL (e.g. https://github.com/owner/repo/pull/123)
    """
    owner, repo, number = _parse_pr_url(pr_url)
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{GITHUB_API}/repos/{owner}/{repo}/pulls/{number}",
            headers=_headers(),
        )
        resp.raise_for_status()
        pr = resp.json()

    return json.dumps(
        {
            "number": pr["number"],
            "title": pr["title"],
            "body": pr["body"] or "",
            "author": pr["user"]["login"],
            "state": pr["state"],
            "base_branch": pr["base"]["ref"],
            "head_branch": pr["head"]["ref"],
            "head_sha": pr["head"]["sha"],
            "labels": [label["name"] for label in pr["labels"]],
            "created_at": pr["created_at"],
            "updated_at": pr["updated_at"],
            "additions": pr["additions"],
            "deletions": pr["deletions"],
            "changed_files": pr["changed_files"],
        },
        indent=2,
    )


@mcp.tool()
async def list_pr_files(pr_url: str) -> str:
    """List all files changed in a PR with their status and patch.

    Args:
        pr_url: Full GitHub PR URL
    """
    owner, repo, number = _parse_pr_url(pr_url)
    async with httpx.AsyncClient() as client:
        files = []
        page = 1
        while True:
            resp = await client.get(
                f"{GITHUB_API}/repos/{owner}/{repo}/pulls/{number}/files",
                headers=_headers(),
                params={"per_page": 100, "page": page},
            )
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break
            for f in batch:
                files.append(
                    {
                        "filename": f["filename"],
                        "status": f["status"],  # added, removed, modified, renamed
                        "additions": f["additions"],
                        "deletions": f["deletions"],
                        "patch": f.get("patch", ""),
                    }
                )
            page += 1

    return json.dumps(files, indent=2)


@mcp.tool()
async def get_pr_diff(pr_url: str) -> str:
    """Get the full unified diff of a PR.

    Args:
        pr_url: Full GitHub PR URL
    """
    owner, repo, number = _parse_pr_url(pr_url)
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{GITHUB_API}/repos/{owner}/{repo}/pulls/{number}",
            headers={
                **_headers(),
                "Accept": "application/vnd.github.v3.diff",
            },
        )
        resp.raise_for_status()

    return resp.text


@mcp.tool()
async def get_file_contents(pr_url: str, file_path: str) -> str:
    """Read a file's full contents at the PR's head commit.

    Args:
        pr_url: Full GitHub PR URL
        file_path: Path to the file within the repository
    """
    owner, repo, number = _parse_pr_url(pr_url)

    # First get the head SHA
    async with httpx.AsyncClient() as client:
        pr_resp = await client.get(
            f"{GITHUB_API}/repos/{owner}/{repo}/pulls/{number}",
            headers=_headers(),
        )
        pr_resp.raise_for_status()
        head_sha = pr_resp.json()["head"]["sha"]

        # Get file at that ref
        resp = await client.get(
            f"{GITHUB_API}/repos/{owner}/{repo}/contents/{file_path}",
            headers={
                **_headers(),
                "Accept": "application/vnd.github.v3.raw",
            },
            params={"ref": head_sha},
        )
        resp.raise_for_status()

    return resp.text


@mcp.tool()
async def post_review(
    pr_url: str,
    body: str,
    event: str = "COMMENT",
    comments: str = "[]",
) -> str:
    """Post a code review on a PR.

    Args:
        pr_url: Full GitHub PR URL
        body: Top-level review summary
        event: Review action — COMMENT, APPROVE, or REQUEST_CHANGES
        comments: JSON array of inline comments, each with keys:
                  path (str), line (int), body (str)
    """
    if event not in ("COMMENT", "APPROVE", "REQUEST_CHANGES"):
        return f"Error: event must be COMMENT, APPROVE, or REQUEST_CHANGES, got {event}"

    owner, repo, number = _parse_pr_url(pr_url)
    parsed_comments = json.loads(comments)

    review_payload: dict = {"body": body, "event": event}
    if parsed_comments:
        review_payload["comments"] = [
            {"path": c["path"], "line": c["line"], "body": c["body"]}
            for c in parsed_comments
        ]

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{GITHUB_API}/repos/{owner}/{repo}/pulls/{number}/reviews",
            headers=_headers(),
            json=review_payload,
        )
        resp.raise_for_status()

    return json.dumps({"status": "review_posted", "event": event})


if __name__ == "__main__":
    mcp.run()
