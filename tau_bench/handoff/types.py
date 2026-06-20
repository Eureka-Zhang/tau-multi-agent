# Copyright Sierra

from pydantic import BaseModel, Field
from typing import Any, Dict, List


class ToolTraceSummary(BaseModel):
    tool: str
    result: str
    ref: str = ""


class SemanticFrame(BaseModel):
    intent: str = ""
    slots: Dict[str, Any] = Field(default_factory=dict)


class ExecutionBoundary(BaseModel):
    allowed_actions: List[str] = Field(default_factory=list)
    forbidden_actions: List[str] = Field(default_factory=list)
    confirmation_status: str = "unknown"


class HandoffPackage(BaseModel):
    """Structured context package handed from Agent A to Agent B.

    Field layout follows section 9 of the design plan. The package is meant to
    replace Agent A's raw message history: Agent B only ever sees this object.
    """

    context_id: str = ""
    task_id: str = ""
    domain: str = ""
    task_goal: str = ""

    completed_subtasks: List[str] = Field(default_factory=list)
    remaining_subtasks: List[str] = Field(default_factory=list)

    tool_trace_summary: List[ToolTraceSummary] = Field(default_factory=list)
    intermediate_state: Dict[str, Any] = Field(default_factory=dict)

    semantic_frame: SemanticFrame = Field(default_factory=SemanticFrame)
    kept_constraints: List[str] = Field(default_factory=list)
    execution_boundary: ExecutionBoundary = Field(default_factory=ExecutionBoundary)

    refs: List[str] = Field(default_factory=list)
    recover_hint: str = ""

    # Compression analytics (filled in by the builder).
    token_full: int = 0
    token_count: int = 0
    compression_ratio: float = 0.0
