"""
In-memory mock data standing in for real policy/claims/CRM systems.
In production these would be calls to a Policy Admin System, a Claims
Management System, and a Billing/CRM system respectively (likely via
internal REST/gRPC APIs, not direct DB access from the agent).
"""

POLICIES = {
    "POL-1001": {
        "holder_name": "Ravi Kumar",
        "status": "active",
        "vehicle": "Honda City - AP09XX1234",
        "premium_due": False,
    },
    "POL-1002": {
        "holder_name": "Sneha Reddy",
        "status": "active",
        "vehicle": "Hyundai Creta - TS10YY5678",
        "premium_due": True,
    },
    "POL-1003": {
        "holder_name": "Arjun Mehta",
        "status": "lapsed",
        "vehicle": "Maruti Swift - KA05ZZ9876",
        "premium_due": True,
    },
}

# Filled in at runtime by the file_vehicle_claim tool.
CLAIMS: list[dict] = []

# Filled in at runtime by the cancel_policy tool.
CANCELLATIONS: dict[str, dict] = {}
