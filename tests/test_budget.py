"""Tests for agent/budget.py — token budget tracking and enforcement."""

import pytest

from agent.budget import TokenBudget, estimate_tokens


class TestTokenBudget:
    """Tests for the TokenBudget dataclass."""

    @pytest.fixture
    def budget(self):
        return TokenBudget(
            max_input_tokens=10_000,
            max_output_tokens=2_000,
            max_total_tokens=12_000,
        )

    def test_initial_state(self, budget):
        assert budget.input_tokens_used == 0
        assert budget.output_tokens_used == 0
        assert budget.total_tokens_used == 0
        assert budget.llm_calls == 0
        assert not budget.budget_exceeded
        assert budget.budget_utilization == 0.0

    def test_record_usage(self, budget):
        budget.record_usage(1000, 200)
        assert budget.input_tokens_used == 1000
        assert budget.output_tokens_used == 200
        assert budget.total_tokens_used == 1200
        assert budget.llm_calls == 1

    def test_cumulative_usage(self, budget):
        budget.record_usage(3000, 500)
        budget.record_usage(4000, 600)
        assert budget.input_tokens_used == 7000
        assert budget.output_tokens_used == 1100
        assert budget.total_tokens_used == 8100
        assert budget.llm_calls == 2

    def test_remaining_tokens(self, budget):
        budget.record_usage(6000, 1000)
        assert budget.remaining_input == 4000
        assert budget.remaining_output == 1000
        assert budget.remaining_total == 5000

    def test_budget_exceeded_input(self, budget):
        budget.record_usage(11000, 0)
        assert budget.budget_exceeded

    def test_budget_exceeded_output(self, budget):
        budget.record_usage(0, 3000)
        assert budget.budget_exceeded

    def test_budget_exceeded_total(self, budget):
        budget.record_usage(8000, 5000)
        assert budget.budget_exceeded

    def test_budget_not_exceeded(self, budget):
        budget.record_usage(5000, 1000)
        assert not budget.budget_exceeded

    def test_budget_utilization(self, budget):
        budget.record_usage(6000, 0)
        assert budget.budget_utilization == 0.5

    def test_check_budget_ok(self, budget):
        assert budget.check_budget() is True
        assert budget.check_budget(estimated_input=5000) is True

    def test_check_budget_exceeded(self, budget):
        budget.record_usage(11000, 0)
        assert budget.check_budget() is False

    def test_check_budget_estimated_too_high(self, budget):
        budget.record_usage(8000, 0)
        assert budget.check_budget(estimated_input=5000) is False

    def test_estimate_cost_known_model(self, budget):
        budget.record_usage(1_000_000, 100_000)
        cost = budget.estimate_cost("gpt-4o")
        assert cost["input_cost_usd"] == 2.5
        assert cost["output_cost_usd"] == 1.0
        assert cost["total_cost_usd"] == 3.5

    def test_estimate_cost_unknown_model(self, budget):
        budget.record_usage(1000, 100)
        cost = budget.estimate_cost("unknown-model")
        assert "total_cost_usd" in cost
        assert cost["total_cost_usd"] >= 0

    def test_summary(self, budget):
        budget.record_usage(5000, 1000)
        s = budget.summary()
        assert s["input_tokens"] == 5000
        assert s["output_tokens"] == 1000
        assert s["total_tokens"] == 6000
        assert s["llm_calls"] == 1
        assert s["budget_exceeded"] is False

    def test_zero_max_budget(self):
        budget = TokenBudget(max_total_tokens=0)
        assert budget.budget_utilization == 1.0

    def test_warning_at_80_percent(self, budget):
        budget.record_usage(9700, 0)
        assert any("80%" in w for w in budget._warnings)


class TestEstimateTokens:
    """Tests for the token estimation heuristic."""

    def test_empty_string(self):
        assert estimate_tokens("") == 1  # minimum 1

    def test_short_string(self):
        result = estimate_tokens("hello world")
        assert result >= 1

    def test_long_string(self):
        text = "x" * 4000
        result = estimate_tokens(text)
        assert result == 1000  # 4000 chars / 4 chars_per_token

    def test_returns_int(self):
        assert isinstance(estimate_tokens("some text"), int)
