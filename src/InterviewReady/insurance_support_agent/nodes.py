"""
Graph nodes for the insurance support agent.

Design decision: the LLM is used ONLY for understanding the user (intent +
slot extraction via structured output). Tool dispatch is deterministic
Python routing, and user-facing confirmation text for completed
actions/clarifications is templated, not LLM-generated. Rationale:
  - Predictability: an insurance cancellation/claim confirmation should say
    exactly what happened -- not a paraphrase that could drop a detail.
  - Reliability: fewer LLM round-trips in the critical path = fewer places
    a live demo (or production traffic) can fail on network/latency.
  - Auditability: deterministic templates are easy to unit test and log.
This is a common production pattern: "LLM for understanding, code for
doing and confirming."
"""

from typing import Literal, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from .state import SupportState
from .tools import cancel_policy, check_policy_status, file_vehicle_claim

REQUIRED_FIELDS = {
    "check_status": ["policy_number"],
    "report_claim": [
        "policy_number",
        "incident_date",
        "incident_description",
        "incident_location",
    ],
    "cancel_policy": ["policy_number", "reason"],
}

FIELD_QUESTIONS = {
    "policy_number": "your policy number",
    "incident_date": "the date of the incident (e.g. 2026-07-20)",
    "incident_description": "a brief description of what happened",
    "incident_location": "where the incident happened",
    "reason": "the reason for the cancellation",
}


class IntentExtraction(BaseModel):
    """Structured extraction of intent + any details mentioned so far."""

    intent: Literal["check_status", "report_claim", "cancel_policy", "unclear"] = (
        Field(description="The customer's request type.")
    )
    policy_number: Optional[str] = Field(
        default=None, description="Policy number if mentioned, e.g. POL-1001."
    )
    incident_date: Optional[str] = Field(
        default=None, description="Date of the vehicle incident, if mentioned."
    )
    incident_description: Optional[str] = Field(
        default=None, description="Description of what happened, if mentioned."
    )
    incident_location: Optional[str] = Field(
        default=None, description="Location of the incident, if mentioned."
    )
    reason: Optional[str] = Field(
        default=None, description="Reason for policy cancellation, if mentioned."
    )


def _llm() -> ChatGroq:
    return ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_tokens=512)


def classify_intent_node(state: SupportState) -> dict:
    """Understand the user's latest message: intent + any slots mentioned."""
    structured_llm = _llm().with_structured_output(IntentExtraction)

    # Give the LLM the last user turn plus a hint of what's already known,
    # so it doesn't need to re-extract everything each turn.
    known_slots = state.get("slots", {})
    known_intent = state.get("intent")
    last_user_msg = state["messages"][-1].content

    system_hint = (
        "You are the NLU layer of an insurance support agent. "
        "Extract the customer's intent and any details they provide. "
        f"Known intent so far: {known_intent or 'none'}. "
        f"Known details so far: {known_slots or 'none'}. "
        "If the new message doesn't change the intent, keep the known intent. "
        "Only extract fields the customer actually stated in this message; "
        "leave others null."
    )

    result: IntentExtraction = structured_llm.invoke(
        [
            HumanMessage(content=system_hint),
            HumanMessage(content=f"Customer message: {last_user_msg}"),
        ]
    )

    new_slots = dict(known_slots)
    for field in ["policy_number", "incident_date", "incident_description", "incident_location", "reason"]:
        value = getattr(result, field)
        if value:
            new_slots[field] = value

    intent = result.intent if result.intent != "unclear" or not known_intent else known_intent

    missing = []
    if intent in REQUIRED_FIELDS:
        missing = [f for f in REQUIRED_FIELDS[intent] if not new_slots.get(f)]

    return {
        "intent": intent,
        "slots": new_slots,
        "missing_fields": missing,
        "error": None,
    }


def route_after_classify(
    state: SupportState,
) -> Literal["clarify", "ask_missing", "check_status", "report_claim", "human_approval"]:
    if state["intent"] == "unclear":
        return "clarify"
    if state["missing_fields"]:
        return "ask_missing"
    return {
        "check_status": "check_status",
        "report_claim": "report_claim",
        "cancel_policy": "human_approval",
    }[state["intent"]]


def clarify_node(state: SupportState) -> dict:
    msg = (
        "I can help with checking a policy status, reporting a vehicle claim, "
        "or cancelling a policy. Could you tell me which of these you'd like to do?"
    )
    return {"messages": [AIMessage(content=msg)]}


def ask_missing_node(state: SupportState) -> dict:
    friendly = [FIELD_QUESTIONS.get(f, f) for f in state["missing_fields"]]
    if len(friendly) == 1:
        ask = friendly[0]
    else:
        ask = ", ".join(friendly[:-1]) + f", and {friendly[-1]}"
    msg = f"Sure, I just need a bit more information: could you share {ask}?"
    return {"messages": [AIMessage(content=msg)]}


def check_status_node(state: SupportState) -> dict:
    try:
        result = check_policy_status.invoke({"policy_number": state["slots"]["policy_number"]})
        return {"tool_result": result, "error": None}
    except ValueError as exc:
        return {"tool_result": None, "error": str(exc)}


def report_claim_node(state: SupportState) -> dict:
    try:
        result = file_vehicle_claim.invoke(
            {
                "policy_number": state["slots"]["policy_number"],
                "incident_date": state["slots"]["incident_date"],
                "incident_description": state["slots"]["incident_description"],
                "incident_location": state["slots"]["incident_location"],
            }
        )
        return {"tool_result": result, "error": None}
    except ValueError as exc:
        return {"tool_result": None, "error": str(exc)}


def human_approval_node(state: SupportState) -> dict:
    """
    Human-in-the-loop gate. Policy cancellation is irreversible-ish and has
    billing consequences, so it requires an explicit human approval before
    the tool actually runs. interrupt() pauses the graph and surfaces the
    decision payload to the calling UI; execution resumes when the caller
    sends Command(resume=True/False) with the same thread id.
    """
    decision = interrupt(
        {
            "action": "approve_cancellation",
            "policy_number": state["slots"].get("policy_number"),
            "reason": state["slots"].get("reason"),
            "prompt": (
                f"Approval required: cancel policy "
                f"{state['slots'].get('policy_number')}? "
                f"Reason given: {state['slots'].get('reason')}"
            ),
        }
    )
    return {"approved": bool(decision)}


def route_after_approval(state: SupportState) -> Literal["cancel_execute", "respond"]:
    return "cancel_execute" if state.get("approved") else "respond"


def cancel_execute_node(state: SupportState) -> dict:
    try:
        result = cancel_policy.invoke(
            {
                "policy_number": state["slots"]["policy_number"],
                "reason": state["slots"]["reason"],
            }
        )
        return {"tool_result": result, "error": None}
    except ValueError as exc:
        return {"tool_result": None, "error": str(exc)}


def respond_node(state: SupportState) -> dict:
    """Compose the final, deterministic confirmation message for this turn."""
    if state.get("error"):
        msg = f"Sorry, I couldn't complete that: {state['error']}"

    elif state["intent"] == "cancel_policy" and not state.get("approved"):
        msg = (
            f"The cancellation request for policy "
            f"{state['slots'].get('policy_number')} was not approved, "
            "so no changes were made to the policy."
        )

    elif state["intent"] == "check_status":
        r = state["tool_result"]
        premium_note = " Note: a premium payment is due." if r["premium_due"] else ""
        msg = (
            f"Policy {r['policy_number']} is held by {r['holder_name']} for "
            f"vehicle {r['vehicle']}. Status: {r['status']}.{premium_note}"
        )

    elif state["intent"] == "report_claim":
        r = state["tool_result"]
        msg = (
            f"Your claim has been filed successfully. Claim ID: {r['claim_id']}, "
            f"status: {r['status']}. We'll follow up regarding policy "
            f"{r['policy_number']}."
        )

    elif state["intent"] == "cancel_policy":
        r = state["tool_result"]
        msg = (
            f"Policy {r['policy_number']} has been cancelled as approved. "
            f"Reason on file: {r['reason']}."
        )
    else:
        msg = "Is there anything else I can help you with?"

    # Reset per-request fields so the next user message starts a fresh
    # request, while keeping the full message history for context.
    return {
        "messages": [AIMessage(content=msg)],
        "intent": None,
        "slots": {},
        "missing_fields": [],
        "tool_result": None,
        "error": None,
        "approved": None,
    }
