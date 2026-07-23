"""
Mock 'backend API' tools for the insurance support agent.

These are real @tool-decorated LangChain tools (so they're introspectable,
have typed args, and could be swapped for live API calls without touching
the graph). The graph invokes them deterministically rather than handing
them to the LLM via bind_tools() -- see graph.py for why.
"""

import datetime
import uuid

from langchain_core.tools import tool

from .mock_data import CANCELLATIONS, CLAIMS, POLICIES


@tool
def check_policy_status(policy_number: str) -> dict:
    """Look up the status, holder, and vehicle details of an insurance policy."""
    key = policy_number.strip().upper()
    policy = POLICIES.get(key)
    if not policy:
        raise ValueError(
            f"No policy found with number '{policy_number}'. Please re-check the number."
        )
    return {"policy_number": key, **policy}


@tool
def file_vehicle_claim(
    policy_number: str,
    incident_date: str,
    incident_description: str,
    incident_location: str,
) -> dict:
    """File a vehicle insurance claim against an existing, active policy."""
    key = policy_number.strip().upper()
    policy = POLICIES.get(key)
    if not policy:
        raise ValueError(
            f"No policy found with number '{policy_number}'. Please re-check the number."
        )
    if policy["status"] != "active":
        raise ValueError(
            f"Policy '{key}' is currently '{policy['status']}', so a claim cannot be "
            "filed against it. Please renew the policy first."
        )
    claim = {
        "claim_id": f"CLM-{uuid.uuid4().hex[:8].upper()}",
        "policy_number": key,
        "incident_date": incident_date,
        "incident_description": incident_description,
        "incident_location": incident_location,
        "status": "submitted",
        "filed_at": datetime.datetime.utcnow().isoformat(timespec="seconds"),
    }
    CLAIMS.append(claim)
    return claim


@tool
def cancel_policy(policy_number: str, reason: str) -> dict:
    """Cancel an insurance policy. Must only be called after human approval."""
    key = policy_number.strip().upper()
    policy = POLICIES.get(key)
    if not policy:
        raise ValueError(
            f"No policy found with number '{policy_number}'. Please re-check the number."
        )
    if policy["status"] == "cancelled":
        raise ValueError(f"Policy '{key}' is already cancelled.")
    policy["status"] = "cancelled"
    record = {
        "policy_number": key,
        "reason": reason,
        "cancelled_at": datetime.datetime.utcnow().isoformat(timespec="seconds"),
    }
    CANCELLATIONS[key] = record
    return record
