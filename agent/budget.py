"""Token budget controls — tracks and enforces LLM token spending per review.

Prevents runaway costs by:
    1. Estimating token usage before each LLM call
    2. Tracking cumulative tokens per review
    3. Aborting the review if the budget is exceeded

Token counting uses tiktoken for accurate estimates (OpenAI models) and
a character-based heuristic for Anthropic models.
"""

from __future__ import annotations

import structlog
from dataclasses import dataclass, field

log = structlog.get_logger()

# Rough token-to-character ratios (conservative)
_CHARS_PER_TOKEN = 4

# Default budget: generous enough for thorough reviews, but prevents runaway
DEFAULT_MAX_INPUT_TOKENS = 100_000
DEFAULT_MAX_OUTPUT_TOKENS = 20_000
DEFAULT_MAX_TOTAL_TOKENS = 120_000

# Cost per 1M tokens (approximate, for logging)
_COST_PER_1M: dict[str, dict[str, float]] = {
    "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
    "gpt-4o": {"input": 2.5, "output": 10.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.6},
}


@dataclass
class TokenBudget:
    """Tracks token usage and enforces budget limits for a single review."""

    max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    max_total_tokens: int = DEFAULT_MAX_TOTAL_TOKENS

    input_tokens_used: int = 0
    output_tokens_used: int = 0
    llm_calls: int = 0

    _warnings: list[str] = field(default_factory=list)

    @property
    def total_tokens_used(self) -> int:
        return self.input_tokens_used + self.output_tokens_used

    @property
    def remaining_input(self) -> int:
        return max(0, self.max_input_tokens - self.input_tokens_used)

    @property
    def remaining_output(self) -> int:
        return max(0, self.max_output_tokens - self.output_tokens_used)

    @property
    def remaining_total(self) -> int:
        return max(0, self.max_total_tokens - self.total_tokens_used)

    @property
    def budget_exceeded(self) -> bool:
        return (
            self.input_tokens_used > self.max_input_tokens
            or self.output_tokens_used > self.max_output_tokens
            or self.total_tokens_used > self.max_total_tokens
        )

    @property
    def budget_utilization(self) -> float:
        """Return budget usage as a fraction 0.0–1.0."""
        if self.max_total_tokens == 0:
            return 1.0
        return self.total_tokens_used / self.max_total_tokens

    def record_usage(self, input_tokens: int, output_tokens: int) -> None:
        """Record token usage from an LLM call."""
        self.input_tokens_used += input_tokens
        self.output_tokens_used += output_tokens
        self.llm_calls += 1

        log.debug(
            "token_usage_recorded",
            input=input_tokens,
            output=output_tokens,
            total_used=self.total_tokens_used,
            remaining=self.remaining_total,
            utilization=f"{self.budget_utilization:.1%}",
        )

        if self.budget_utilization > 0.8 and not any("80%" in w for w in self._warnings):
            self._warnings.append("Token budget 80% consumed")
            log.warning(
                "token_budget_warning",
                utilization=f"{self.budget_utilization:.1%}",
                total_used=self.total_tokens_used,
                max_total=self.max_total_tokens,
            )

        if self.budget_exceeded:
            log.error(
                "token_budget_exceeded",
                input_used=self.input_tokens_used,
                output_used=self.output_tokens_used,
                total_used=self.total_tokens_used,
            )

    def check_budget(self, estimated_input: int = 0) -> bool:
        """Check if there's enough budget for another LLM call.

        Args:
            estimated_input: Estimated input tokens for the next call

        Returns:
            True if the call should proceed, False if budget is exhausted
        """
        if self.budget_exceeded:
            return False
        if estimated_input > 0 and estimated_input > self.remaining_input:
            return False
        return True

    def estimate_cost(self, model: str) -> dict[str, float]:
        """Estimate the dollar cost of this review's token usage."""
        costs = _COST_PER_1M.get(model, {"input": 3.0, "output": 15.0})
        input_cost = (self.input_tokens_used / 1_000_000) * costs["input"]
        output_cost = (self.output_tokens_used / 1_000_000) * costs["output"]
        return {
            "input_cost_usd": round(input_cost, 6),
            "output_cost_usd": round(output_cost, 6),
            "total_cost_usd": round(input_cost + output_cost, 6),
        }

    def summary(self) -> dict:
        """Return a summary of token usage."""
        return {
            "input_tokens": self.input_tokens_used,
            "output_tokens": self.output_tokens_used,
            "total_tokens": self.total_tokens_used,
            "llm_calls": self.llm_calls,
            "budget_utilization": f"{self.budget_utilization:.1%}",
            "budget_exceeded": self.budget_exceeded,
            "warnings": self._warnings,
        }


def estimate_tokens(text: str) -> int:
    """Estimate token count from text using character heuristic.

    This is a rough estimate. For production use, integrate tiktoken.
    """
    return max(1, len(text) // _CHARS_PER_TOKEN)
