# Copyright Sierra

import json
from typing import Any, Dict, List, Optional

from litellm import completion

from tau_bench.agents.base import Agent
from tau_bench.agents.tool_calling_agent import message_to_action
from tau_bench.envs.base import Env
from tau_bench.types import SolveResult, RESPOND_ACTION_NAME
from tau_bench.handoff.builder import build_structured_package
from tau_bench.handoff.tool_split import (
    HANDOFF_TOOL_NAME,
    get_write_tool_names,
    split_tools_info,
)


AGENT_A_ROLE = """

# Your role: Information Agent A
You are the upstream agent in a two-agent pipeline. You can ONLY gather information and verify
policy using read-only tools. You CANNOT perform any write or state-changing action (cancel,
modify, exchange, return, book, update, or transfer). You do not have access to those tools.

When you have gathered all the information needed, verified the relevant policy, and obtained the
user's explicit confirmation for the write action, you MUST call the `ready_for_execution` tool to
hand off to the execution agent. Do not claim the task is finished yourself; the execution agent
will perform the final actions."""

AGENT_B_ROLE = """

# Your role: Execution Agent B
You are the downstream agent in a two-agent pipeline. An information agent already interacted with
the user and gathered context, provided to you as the structured handoff package below. You did NOT
see the original conversation; rely on the package.

Use the package to complete the REMAINING subtasks by calling the appropriate tools. Rules:
- Do NOT repeat actions already listed in completed_subtasks / tool_trace_summary.
- Respect every item in kept_constraints.
- Before any write action, check execution_boundary. If confirmation_status is not "confirmed",
  confirm with the user before writing.
- If information is missing, you may re-query using read tools (see recover_hint).
- Your goal is to complete the original user task.

## Handoff package
"""


class TwoAgentHandoffAgent(Agent):
    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki: str,
        domain: str,
        agent_a_model: str,
        agent_a_provider: str,
        agent_b_model: str,
        agent_b_provider: str,
        builder_model: str,
        builder_provider: str,
        temperature: float = 0.0,
        handoff_method: str = "structured",
    ) -> None:
        self.tools_info = tools_info
        self.wiki = wiki
        self.domain = domain
        self.agent_a_model = agent_a_model
        self.agent_a_provider = agent_a_provider
        self.agent_b_model = agent_b_model
        self.agent_b_provider = agent_b_provider
        self.builder_model = builder_model
        self.builder_provider = builder_provider
        self.temperature = temperature
        self.handoff_method = handoff_method

    def _completion(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        model: str,
        provider: str,
    ):
        return completion(
            messages=messages,
            model=model,
            custom_llm_provider=provider,
            tools=tools,
            temperature=self.temperature,
        )

    @staticmethod
    def _append_tool_result(
        messages: List[Dict[str, Any]],
        assistant_message: Dict[str, Any],
        observation: str,
    ) -> None:
        assistant_message["tool_calls"] = assistant_message["tool_calls"][:1]
        messages.extend(
            [
                assistant_message,
                {
                    "role": "tool",
                    "tool_call_id": assistant_message["tool_calls"][0]["id"],
                    "name": assistant_message["tool_calls"][0]["function"]["name"],
                    "content": observation,
                },
            ]
        )

    @staticmethod
    def _last_user_message(messages: List[Dict[str, Any]]) -> str:
        for message in reversed(messages):
            if message.get("role") == "user" and message.get("content"):
                return message["content"]
        return ""

    def solve(
        self, env: Env, task_index: Optional[int] = None, max_num_steps: int = 30
    ) -> SolveResult:
        total_cost = 0.0
        env_reset_res = env.reset(task_index=task_index)
        obs = env_reset_res.observation
        info = env_reset_res.info.model_dump()
        reward = 0.0

        task_goal_hint = env.task.instruction
        write_tool_names = get_write_tool_names(self.domain)
        agent_a_tools, agent_b_tools = split_tools_info(self.tools_info, self.domain)

        agent_a_budget = max(1, max_num_steps // 2)
        agent_b_budget = max(1, max_num_steps - agent_a_budget)

        # ---- Agent A: information gathering ----
        agent_a_messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.wiki + AGENT_A_ROLE},
            {"role": "user", "content": obs},
        ]
        handoff_triggered = False
        done = False
        a_steps = 0
        for _ in range(agent_a_budget):
            a_steps += 1
            res = self._completion(
                agent_a_messages,
                agent_a_tools,
                self.agent_a_model,
                self.agent_a_provider,
            )
            next_message = res.choices[0].message.model_dump()
            total_cost += res._hidden_params["response_cost"] or 0
            action = message_to_action(next_message)

            if action.name == HANDOFF_TOOL_NAME or action.name in write_tool_names:
                # Intercept: Agent A is not allowed to execute writes. Hand off.
                note = action.kwargs.get("note", "") if action.kwargs else ""
                self._append_tool_result(
                    agent_a_messages,
                    next_message,
                    f"Acknowledged. Handing off to the execution agent. Note: {note}",
                )
                handoff_triggered = True
                break

            env_response = env.step(action)
            reward = env_response.reward
            info = {**info, **env_response.info.model_dump()}
            if action.name != RESPOND_ACTION_NAME:
                self._append_tool_result(
                    agent_a_messages, next_message, env_response.observation
                )
            else:
                agent_a_messages.extend(
                    [
                        next_message,
                        {"role": "user", "content": env_response.observation},
                    ]
                )
            if env_response.done:
                done = True
                break

        # Task already completed (or terminated) during Agent A: no handoff needed.
        if done:
            return SolveResult(
                reward=reward,
                info={**info, "handoff_triggered": False, "agent_a_steps": a_steps},
                messages=agent_a_messages,
                total_cost=total_cost,
            )

        # ---- Build structured handoff package (exclude the wiki system prompt) ----
        package = build_structured_package(
            agent_a_messages=agent_a_messages[1:],
            task_goal_hint=task_goal_hint,
            domain=self.domain,
            model=self.builder_model,
            provider=self.builder_provider,
            temperature=self.temperature,
            context_id=f"ctx-{task_index}",
            task_id=str(task_index),
        )

        # ---- Agent B: execution ----
        kickoff = self._last_user_message(agent_a_messages)
        agent_b_messages: List[Dict[str, Any]] = [
            {
                "role": "system",
                "content": self.wiki + AGENT_B_ROLE + package.model_dump_json(indent=2),
            },
            {
                "role": "user",
                "content": (
                    "You are now taking over to complete the remaining work. "
                    + (f"The user's last message was: {kickoff}\n" if kickoff else "")
                    + "Proceed to complete the remaining subtasks."
                ),
            },
        ]
        b_steps = 0
        for _ in range(agent_b_budget):
            b_steps += 1
            res = self._completion(
                agent_b_messages,
                agent_b_tools,
                self.agent_b_model,
                self.agent_b_provider,
            )
            next_message = res.choices[0].message.model_dump()
            total_cost += res._hidden_params["response_cost"] or 0
            action = message_to_action(next_message)
            env_response = env.step(action)
            reward = env_response.reward
            info = {**info, **env_response.info.model_dump()}
            if action.name != RESPOND_ACTION_NAME:
                self._append_tool_result(
                    agent_b_messages, next_message, env_response.observation
                )
            else:
                agent_b_messages.extend(
                    [
                        next_message,
                        {"role": "user", "content": env_response.observation},
                    ]
                )
            if env_response.done:
                break

        combined_messages = (
            agent_a_messages
            + [{"role": "system", "content": "--- HANDOFF ---"}]
            + agent_b_messages
        )
        info = {
            **info,
            "handoff_triggered": handoff_triggered,
            "handoff_method": self.handoff_method,
            "handoff_package": package.model_dump(),
            "compression_ratio": package.compression_ratio,
            "token_full": package.token_full,
            "token_count": package.token_count,
            "agent_a_steps": a_steps,
            "agent_b_steps": b_steps,
            "agent_a_model": self.agent_a_model,
            "agent_b_model": self.agent_b_model,
            "handoff_builder_model": self.builder_model,
        }
        return SolveResult(
            reward=reward,
            info=info,
            messages=combined_messages,
            total_cost=total_cost,
        )
