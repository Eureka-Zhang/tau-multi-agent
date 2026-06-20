# Copyright Sierra

from typing import Any, Dict, List, Set, Tuple


HANDOFF_TOOL_NAME = "ready_for_execution"

HANDOFF_TOOL_INFO: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": HANDOFF_TOOL_NAME,
        "description": (
            "Hand off the task to the execution agent. Call this when you have gathered "
            "all necessary information, verified the relevant policy, and obtained explicit "
            "user confirmation, and are ready for a write/state-changing action (such as "
            "cancelling, modifying, exchanging, returning, booking, or updating). You do NOT "
            "have access to any write tools yourself, so you must hand off to proceed. After "
            "calling this, an execution agent will perform the remaining write actions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "note": {
                    "type": "string",
                    "description": (
                        "A short free-text note to the execution agent summarizing what is "
                        "ready to be executed and any final confirmation status."
                    ),
                }
            },
            "required": [],
        },
    },
}


# Read / query / policy tools owned by Agent A.
RETAIL_READ_TOOLS: Set[str] = {
    "calculate",
    "find_user_id_by_email",
    "find_user_id_by_name_zip",
    "get_order_details",
    "get_product_details",
    "get_user_details",
    "list_all_product_types",
    "think",
}

# Write / state-changing tools owned by Agent B.
RETAIL_WRITE_TOOLS: Set[str] = {
    "cancel_pending_order",
    "exchange_delivered_order_items",
    "modify_pending_order_address",
    "modify_pending_order_items",
    "modify_pending_order_payment",
    "modify_user_address",
    "return_delivered_order_items",
    "transfer_to_human_agents",
}

AIRLINE_READ_TOOLS: Set[str] = {
    "calculate",
    "get_reservation_details",
    "get_user_details",
    "list_all_airports",
    "search_direct_flight",
    "search_onestop_flight",
    "think",
}

AIRLINE_WRITE_TOOLS: Set[str] = {
    "book_reservation",
    "cancel_reservation",
    "send_certificate",
    "update_reservation_baggages",
    "update_reservation_flights",
    "update_reservation_passengers",
    "transfer_to_human_agents",
}


def get_read_tool_names(domain: str) -> Set[str]:
    if domain == "retail":
        return set(RETAIL_READ_TOOLS)
    elif domain == "airline":
        return set(AIRLINE_READ_TOOLS)
    raise ValueError(f"Unknown domain for tool split: {domain}")


def get_write_tool_names(domain: str) -> Set[str]:
    if domain == "retail":
        return set(RETAIL_WRITE_TOOLS)
    elif domain == "airline":
        return set(AIRLINE_WRITE_TOOLS)
    raise ValueError(f"Unknown domain for tool split: {domain}")


def _tool_name(tool_info: Dict[str, Any]) -> str:
    return tool_info["function"]["name"]


def split_tools_info(
    tools_info: List[Dict[str, Any]], domain: str
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split the full env tool schemas into Agent A and Agent B tool sets.

    Agent A receives read/policy tool schemas plus the handoff tool. Agent B
    receives all original tool schemas (read + write). The env keeps every tool
    executable; this only controls what is exposed to each sub-agent's LLM.
    """
    read_names = get_read_tool_names(domain)
    agent_a_tools = [t for t in tools_info if _tool_name(t) in read_names]
    agent_a_tools.append(HANDOFF_TOOL_INFO)
    agent_b_tools = list(tools_info)
    return agent_a_tools, agent_b_tools
