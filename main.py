"""Entry point — run the code review agent on a PR."""

from __future__ import annotations

import asyncio
import sys

import structlog

from agent.client import MCPClient
from agent.core import ReviewAgent
from config import settings


def configure_logging() -> None:
    """Configure structlog for console or JSON output."""
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]
    if settings.log_json:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            structlog.get_level_from_name(settings.log_level)
        ),
    )


async def main(pr_url: str) -> None:
    log = structlog.get_logger()
    log.info("starting_review", pr_url=pr_url)

    mcp_client = MCPClient()
    try:
        await mcp_client.connect()
        agent = ReviewAgent(mcp_client=mcp_client)
        result = await agent.review(pr_url)
        print("\n" + "=" * 60)
        print("REVIEW RESULT")
        print("=" * 60)
        print(result)
        print("=" * 60)
        log.info("review_complete", steps=len(agent.steps))
    finally:
        await mcp_client.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <github-pr-url>")
        print("Example: python main.py https://github.com/owner/repo/pull/123")
        sys.exit(1)

    configure_logging()
    asyncio.run(main(sys.argv[1]))
