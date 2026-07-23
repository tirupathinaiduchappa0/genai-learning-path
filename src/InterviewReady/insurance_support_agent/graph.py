"""
Graph assembly for the insurance support agent.

Flow:
    classify_intent
        -> clarify                (intent unclear)                -> END
        -> ask_missing             (slots missing)                 -> END
        -> check_status  -> respond                                -> END
        -> report_claim  -> respond                                -> END
        -> human_approval -> [interrupt] -> cancel_execute -> respond -> END
                                          -> respond (rejected)     -> END

Each END is a real graph end: one user turn = one pass through the graph,
using a checkpointer so state (slots, messages, intent) persists across
turns for the same thread_id. This is what makes multi-turn slot-filling
and the human-in-the-loop pause/resume work correctly.
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .nodes import (
    ask_missing_node,
    cancel_execute_node,
    check_status_node,
    clarify_node,
    classify_intent_node,
    human_approval_node,
    report_claim_node,
    respond_node,
    route_after_approval,
    route_after_classify,
)
from .state import SupportState


def build_graph():
    builder = StateGraph(SupportState)

    builder.add_node("classify_intent", classify_intent_node)
    builder.add_node("clarify", clarify_node)
    builder.add_node("ask_missing", ask_missing_node)
    builder.add_node("check_status", check_status_node)
    builder.add_node("report_claim", report_claim_node)
    builder.add_node("human_approval", human_approval_node)
    builder.add_node("cancel_execute", cancel_execute_node)
    builder.add_node("respond", respond_node)

    builder.add_edge(START, "classify_intent")

    builder.add_conditional_edges(
        "classify_intent",
        route_after_classify,
        {
            "clarify": "clarify",
            "ask_missing": "ask_missing",
            "check_status": "check_status",
            "report_claim": "report_claim",
            "human_approval": "human_approval",
        },
    )

    builder.add_conditional_edges(
        "human_approval",
        route_after_approval,
        {"cancel_execute": "cancel_execute", "respond": "respond"},
    )

    builder.add_edge("check_status", "respond")
    builder.add_edge("report_claim", "respond")
    builder.add_edge("cancel_execute", "respond")

    builder.add_edge("clarify", END)
    builder.add_edge("ask_missing", END)
    builder.add_edge("respond", END)

    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


# Module-level singleton graph, reused across turns (Streamlit reruns the
# script on every interaction, so this must not be re-created per message).
support_graph = build_graph()
