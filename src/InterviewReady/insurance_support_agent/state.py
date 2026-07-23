"""LangGraph state schema for the insurance support agent."""

from typing import Annotated, Any, Optional

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class SupportState(TypedDict):
    # Full conversation history. `add_messages` appends new messages instead
    # of overwriting -- this is what makes multi-turn conversation work.
    messages: Annotated[list, add_messages]

    # Sticky intent for the current request: one of
    # "check_status" | "report_claim" | "cancel_policy" | "unclear" | None
    intent: Optional[str]

    # Slots collected so far (policy_number, incident_date, etc.). Persists
    # across turns via the checkpointer until the request is completed.
    slots: dict[str, Any]

    # Required fields still missing for the current intent.
    missing_fields: list[str]

    # Result returned by the last tool call (or the "not approved" outcome).
    tool_result: Optional[dict]

    # Set when a tool call fails; cleared once handled.
    error: Optional[str]

    # Human-in-the-loop decision for cancellation requests.
    approved: Optional[bool]
