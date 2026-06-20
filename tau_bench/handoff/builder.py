# Copyright Sierra

import json
from typing import Any, Dict, List, Optional

from litellm import completion

from tau_bench.handoff.types import HandoffPackage


def estimate_tokens(text: str) -> int:
    """Rough token count. Uses tiktoken when available, else a char heuristic."""
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        # ~4 chars per token is a common rough approximation.
        return max(1, len(text) // 4)


def _messages_to_text(messages: List[Dict[str, Any]]) -> str:
    return json.dumps(messages, ensure_ascii=False, default=str)


PACKAGE_JSON_SCHEMA_HINT = """{
  "task_goal": "<one sentence describing what the user ultimately wants>",
  "completed_subtasks": ["<subtask already done by agent A>", ...],
  "remaining_subtasks": ["<subtask still to be done by agent B>", ...],
  "tool_trace_summary": [
    {"tool": "<tool name>", "result": "<concise result>", "ref": "<short id>"}
  ],
  "intermediate_state": {"<key>": "<value the executor needs, e.g. order_id, status, eligibility>"},
  "semantic_frame": {"intent": "<the action intent>", "slots": {"<slot>": "<value>"}},
  "kept_constraints": ["<user constraint or policy rule that must be respected>", ...],
  "execution_boundary": {
    "allowed_actions": ["<write action agent B is allowed to take>"],
    "forbidden_actions": ["<action that must NOT be taken>"],
    "confirmation_status": "confirmed" | "not_confirmed" | "unknown"
  },
  "refs": ["<short reference ids you used above>"],
  "recover_hint": "<short hint on how agent B could recover more context if needed>"
}"""

BUILDER_SYSTEM_PROMPT = f"""You are a context-compression module in a two-agent customer-service pipeline.

Agent A (an information-gathering agent) has interacted with a user and called read-only tools.
Agent A cannot perform any write/state-changing actions. Your job is to compress Agent A's full
conversation and tool history into a compact, structured handoff package for Agent B (the execution
agent), which will perform the remaining write actions WITHOUT seeing the original history.

Output requirements:
- Output ONLY a single valid JSON object, no markdown fences, no commentary.
- Preserve every fact Agent B needs to complete the task correctly: ids, statuses, eligibility,
  the user's exact constraints, and whether the user has confirmed the write action.
- Be concise: do not copy the raw conversation; summarize tool results.
- Do not invent information that is not present in Agent A's history.

Use exactly this JSON shape:
{PACKAGE_JSON_SCHEMA_HINT}"""


def _coerce_package(raw: Dict[str, Any]) -> HandoffPackage:
    # pydantic validates / fills defaults for any missing or malformed fields.
    return HandoffPackage.model_validate(raw)


def _minimal_fallback_package(
    task_goal_hint: str, domain: str, error: str
) -> HandoffPackage:
    return HandoffPackage(
        domain=domain,
        task_goal=task_goal_hint,
        remaining_subtasks=["complete_the_user_request"],
        recover_hint=f"package generation failed ({error}); reconstruct from tools as needed",
    )


def build_structured_package(
    agent_a_messages: List[Dict[str, Any]],
    task_goal_hint: str,
    domain: str,
    model: str,
    provider: str,
    temperature: float = 0.0,
    context_id: str = "",
    task_id: str = "",
) -> HandoffPackage:
    """Compress Agent A's history into a validated structured HandoffPackage.

    Falls back to a minimal package (and never raises) so a single bad
    compression does not crash an entire benchmark run.
    """
    full_text = _messages_to_text(agent_a_messages)
    token_full = estimate_tokens(full_text)

    builder_messages = [
        {"role": "system", "content": BUILDER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Domain: {domain}\n"
                f"Task goal hint: {task_goal_hint}\n\n"
                f"Agent A full history (messages + tool results) as JSON:\n{full_text}\n\n"
                "Produce the handoff package JSON now."
            ),
        },
    ]

    package: Optional[HandoffPackage] = None
    last_error = ""
    for _ in range(2):
        try:
            res = completion(
                messages=builder_messages,
                model=model,
                custom_llm_provider=provider,
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            content = res.choices[0].message.content or "{}"
            raw = json.loads(content)
            package = _coerce_package(raw)
            break
        except Exception as e:  # noqa: BLE001 - robustness over a single run
            last_error = str(e)
            builder_messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your previous output was invalid: "
                        f"{last_error}. Output ONLY the JSON object using the required shape."
                    ),
                }
            )

    if package is None:
        package = _minimal_fallback_package(task_goal_hint, domain, last_error)

    package.domain = package.domain or domain
    package.task_goal = package.task_goal or task_goal_hint
    package.context_id = context_id or package.context_id
    package.task_id = task_id or package.task_id

    token_count = estimate_tokens(package.model_dump_json())
    package.token_full = token_full
    package.token_count = token_count
    package.compression_ratio = (
        round(token_count / token_full, 4) if token_full > 0 else 0.0
    )
    return package
