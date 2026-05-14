"""Agent Core — ReAct loop that autonomously reviews PRs using MCP tools.

The agent:
1. Receives a PR URL
2. Plans its review approach
3. Iteratively calls MCP tools (GitHub, linters, tests, knowledge base)
4. Synthesizes findings into a structured review
5. Posts the review on the PR
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import anthropic
import openai
import structlog

from agent.budget import TokenBudget
from agent.client import MCPClient
from agent.observability import record_tool_call, record_token_usage
from agent.prompts import REVIEW_TASK_TEMPLATE, build_system_prompt
from config import settings

log = structlog.get_logger()


@dataclass
class AgentStep:
    """One step in the agent's reasoning loop."""

    step: int
    thought: str | None = None
    tool_name: str | None = None
    tool_args: dict | None = None
    tool_result: str | None = None


@dataclass
class ReviewAgent:
    """Autonomous code review agent using MCP tools."""

    mcp_client: MCPClient
    max_steps: int = field(default_factory=lambda: settings.max_agent_steps)
    budget: TokenBudget = field(default_factory=TokenBudget)
    steps: list[AgentStep] = field(default_factory=list)

    async def review(self, pr_url: str) -> str:
        """Run a full code review on a PR.

        Args:
            pr_url: GitHub PR URL to review

        Returns:
            The final review summary text
        """
        task = REVIEW_TASK_TEMPLATE.format(pr_url=pr_url)
        system_prompt = build_system_prompt(self.max_steps)

        if settings.llm_provider == "anthropic":
            return await self._run_anthropic(system_prompt, task)
        else:
            return await self._run_openai(system_prompt, task)

    async def review_with_plan(self, pr_url: str, plan_context: str) -> str:
        """Run a code review with a pre-built plan injected into context.

        Args:
            pr_url: GitHub PR URL to review
            plan_context: Formatted review plan text to prepend to the task

        Returns:
            The final review summary text
        """
        task = REVIEW_TASK_TEMPLATE.format(pr_url=pr_url)
        task_with_plan = f"{plan_context}\n\n---\n\n{task}"
        system_prompt = build_system_prompt(self.max_steps)

        if settings.llm_provider == "anthropic":
            return await self._run_anthropic(system_prompt, task_with_plan)
        else:
            return await self._run_openai(system_prompt, task_with_plan)

    # ── Anthropic ──────────────────────────────────────────────

    async def _run_anthropic(self, system_prompt: str, task: str) -> str:
        """Run the agent loop using the Anthropic API."""
        client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        tools = self.mcp_client.get_anthropic_tools()

        messages = [{"role": "user", "content": task}]

        for step_num in range(1, self.max_steps + 1):
            if not self.budget.check_budget():
                log.warning("budget_exceeded", provider="anthropic")
                return "Review stopped: token budget exceeded."

            log.info("agent_step", step=step_num, provider="anthropic")

            response = await client.messages.create(
                model=settings.llm_model,
                max_tokens=4096,
                system=system_prompt,
                tools=tools,
                messages=messages,
            )

            # Track token usage
            if hasattr(response, 'usage') and response.usage:
                inp = response.usage.input_tokens
                out = response.usage.output_tokens
                self.budget.record_usage(inp, out)
                record_token_usage(inp, out)

            # Check if the model wants to use a tool
            if response.stop_reason == "tool_use":
                # Process all tool calls in this response
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        step = AgentStep(
                            step=step_num,
                            tool_name=block.name,
                            tool_args=block.input,
                        )
                        log.info(
                            "tool_call",
                            step=step_num,
                            tool=block.name,
                            args=list(block.input.keys()),
                        )

                        result = await self.mcp_client.call_tool(
                            block.name, block.input
                        )
                        step.tool_result = result
                        self.steps.append(step)
                        record_tool_call(block.name)

                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": result,
                            }
                        )

                # Add assistant message and tool results
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})

            elif response.stop_reason == "end_turn":
                # Agent is done — extract final text
                final_text = ""
                for block in response.content:
                    if hasattr(block, "text"):
                        final_text += block.text

                log.info("agent_finished", steps=step_num)
                return final_text

        log.warning("agent_max_steps_reached", max_steps=self.max_steps)
        return "Review incomplete: reached maximum number of steps."

    # ── OpenAI ─────────────────────────────────────────────────

    async def _run_openai(self, system_prompt: str, task: str) -> str:
        """Run the agent loop using the OpenAI API."""
        client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
        tools = self.mcp_client.get_openai_tools()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task},
        ]

        for step_num in range(1, self.max_steps + 1):
            if not self.budget.check_budget():
                log.warning("budget_exceeded", provider="openai")
                return "Review stopped: token budget exceeded."

            log.info("agent_step", step=step_num, provider="openai")

            response = await client.chat.completions.create(
                model=settings.llm_model,
                messages=messages,
                tools=tools if tools else openai.NOT_GIVEN,
            )

            choice = response.choices[0]

            # Track token usage
            if hasattr(response, 'usage') and response.usage:
                inp = response.usage.prompt_tokens
                out = response.usage.completion_tokens
                self.budget.record_usage(inp, out)
                record_token_usage(inp, out)

            if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
                messages.append(choice.message)

                for tool_call in choice.message.tool_calls:
                    fn = tool_call.function
                    args = json.loads(fn.arguments)
                    step = AgentStep(
                        step=step_num,
                        tool_name=fn.name,
                        tool_args=args,
                    )
                    log.info(
                        "tool_call",
                        step=step_num,
                        tool=fn.name,
                        args=list(args.keys()),
                    )

                    result = await self.mcp_client.call_tool(fn.name, args)
                    step.tool_result = result
                    self.steps.append(step)
                    record_tool_call(fn.name)

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result,
                        }
                    )

            elif choice.finish_reason == "stop":
                log.info("agent_finished", steps=step_num)
                return choice.message.content or ""

        log.warning("agent_max_steps_reached", max_steps=self.max_steps)
        return "Review incomplete: reached maximum number of steps."
