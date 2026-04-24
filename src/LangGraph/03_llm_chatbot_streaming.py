"""
💬 LangGraph Lesson 3 — LLM Chatbot with Streaming (Level 1 → 2)

═══════════════════════════════════════════════════════════════════
1. CONCEPT: Connecting an LLM to a LangGraph
═══════════════════════════════════════════════════════════════════

In Lesson 02, nodes were plain Python functions (no LLM).
Now we connect a REAL LLM (Groq) to a LangGraph node.

This is the bridge from "graph mechanics" to "AI chatbot".

═══════════════════════════════════════════════════════════════════
2. WHY THIS EXISTS
═══════════════════════════════════════════════════════════════════

A chatbot needs:
    - An LLM to generate responses
    - A message list (state) that grows with each turn
    - Optionally: conditional routing (search, tools, etc.)
    - Streaming for real-time token-by-token output

LangGraph gives you ALL of this with explicit control over the flow.

🔄 LANGCHAIN vs LANGGRAPH:
    LangChain: chain = prompt | llm | parser (linear, no cycles)
    LangGraph: graph with chatbot node → can loop, branch, stream

═══════════════════════════════════════════════════════════════════
3. REAL-WORLD ANALOGY
═══════════════════════════════════════════════════════════════════

Think of a CUSTOMER SERVICE DESK:

    Simple desk (Lesson 02):
        Customer arrives → Clerk reads form → Clerk fills response → Done
        No thinking, just data flow.

    Smart desk (This lesson):
        Customer arrives → AI Clerk THINKS (LLM) → Responds
        If customer asks to search → Search node → AI Clerk responds
        Customer sees words appearing in real-time (streaming)

TOPICS COVERED:
    1. Simple LLM Chatbot — single node with Groq
    2. Chatbot with Conditional Routing — LLM + search cycle
    3. Streaming with stream_mode="values" — full state per step
    4. Streaming with stream_mode="updates" — only changes per step
    5. Graph images saved as PNG

HOW TO RUN:
    $ python src/LangGraph/03_llm_chatbot_streaming.py

Author: GenAI Learner
Date: 2026-04-15
"""

import logging
import os

from dotenv import load_dotenv

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Graph image helper ───────────────────────────────────────────────────
GRAPH_DIR = os.path.join(os.path.dirname(__file__), "graphs")
os.makedirs(GRAPH_DIR, exist_ok=True)


def save_graph_image(compiled_graph, name: str) -> None:
    """Save the compiled graph as a PNG image using Mermaid rendering."""
    try:
        png_data = compiled_graph.get_graph().draw_mermaid_png()
        filepath = os.path.join(GRAPH_DIR, f"{name}.png")
        with open(filepath, "wb") as f:
            f.write(png_data)
        logger.info("  Graph image saved: %s", filepath)
    except Exception as e:
        logger.warning("  Could not save graph image: %s", e)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. DEMO 1: Simple LLM Chatbot — Single Node
# ═══════════════════════════════════════════════════════════════════════════════
#
# Graph:
#     [START] → [chatbot] → [END]
#
# The chatbot node calls the LLM with the message list from state.
# The LLM response is APPENDED to state (via operator.add).

def demo_simple_chatbot() -> None:
    """The simplest LLM-powered chatbot — one node, one LLM call."""
    import operator
    from typing import Annotated, TypedDict

    from langchain_core.messages import HumanMessage
    from langchain_groq import ChatGroq
    from langgraph.graph import END, START, StateGraph

    # ── State: messages list with append-only behavior ───────────────
    class ChatState(TypedDict):
        messages: Annotated[list, operator.add]

    # ── LLM ──────────────────────────────────────────────────────────
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_tokens=256)

    # ── Node: calls the LLM ──────────────────────────────────────────
    def chatbot_node(state: ChatState) -> dict:
        """Chatbot node — sends messages to LLM, appends response to state."""
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    # ── Build, compile, save image ───────────────────────────────────
    builder = StateGraph(ChatState)
    builder.add_node("chatbot", chatbot_node)
    builder.add_edge(START, "chatbot")
    builder.add_edge("chatbot", END)
    graph = builder.compile()

    save_graph_image(graph, "lesson03_simple_chatbot")

    # ── Invoke ───────────────────────────────────────────────────────
    result = graph.invoke({"messages": [HumanMessage(content="What is LangGraph in one sentence?")]})

    logger.info("--- Demo 1: Simple LLM Chatbot ---")
    logger.info("  Input: 'What is LangGraph in one sentence?'")
    logger.info("  Output: %s", result["messages"][-1].content[:150])
    logger.info("  Messages in state: %d", len(result["messages"]))


# ═══════════════════════════════════════════════════════════════════════════════
# 6. DEMO 2: Chatbot with Conditional Routing (LLM + Search Cycle)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Graph:
#     [START] → [chatbot]
#                   |
#                   ├── "search" in message? → [search] → [chatbot] (CYCLE!)
#                   |
#                   └── no search? → [END]
#
# This is the AGENT LOOP pattern — the chatbot can loop back after search.

def demo_chatbot_with_routing() -> None:
    """Chatbot with conditional routing — LLM decides, search loops back."""
    import operator
    from typing import Annotated, TypedDict

    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_groq import ChatGroq
    from langgraph.graph import END, START, StateGraph

    class AgentState(TypedDict):
        messages: Annotated[list, operator.add]
        needs_search: bool

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_tokens=256)

    def chatbot_node(state: AgentState) -> dict:
        """Chatbot — calls LLM and checks if search is needed."""
        response = llm.invoke(state["messages"])
        last_msg = state["messages"][-1]
        needs_search = isinstance(last_msg, HumanMessage) and "search" in last_msg.content.lower()
        return {"messages": [response], "needs_search": needs_search}

    def search_node(state: AgentState) -> dict:
        """Search — simulates web search, loops back to chatbot."""
        query = state["messages"][-2].content if len(state["messages"]) >= 2 else "general"
        return {"messages": [AIMessage(content=f"[Search Result] Found info for: {query}")], "needs_search": False}

    def route_after_chatbot(state: AgentState) -> str:
        """Router: search or end based on state."""
        return "search" if state.get("needs_search", False) else "end"

    # ── Build graph ──────────────────────────────────────────────────
    builder = StateGraph(AgentState)
    builder.add_node("chatbot", chatbot_node)
    builder.add_node("search", search_node)
    builder.add_edge(START, "chatbot")
    builder.add_conditional_edges("chatbot", route_after_chatbot, {"search": "search", "end": END})
    builder.add_edge("search", "chatbot")  # CYCLE — search loops back!
    graph = builder.compile()

    save_graph_image(graph, "lesson03_chatbot_with_routing")

    logger.info("--- Demo 2: Chatbot with Conditional Routing ---")

    # Test 1: Direct answer (no search)
    r1 = graph.invoke({"messages": [HumanMessage(content="What is Python?")], "needs_search": False})
    logger.info("  Q: 'What is Python?' → %d messages (direct)", len(r1["messages"]))
    logger.info("  A: %s", r1["messages"][-1].content[:100])

    # Test 2: Search triggered (cycle)
    r2 = graph.invoke({"messages": [HumanMessage(content="Please search for LangGraph tutorials")], "needs_search": False})
    logger.info("  Q: 'search for LangGraph tutorials' → %d messages (search cycle)", len(r2["messages"]))
    logger.info("  A: %s", r2["messages"][-1].content[:100])


# ═══════════════════════════════════════════════════════════════════════════════
# 7. DEMO 3: Streaming — stream_mode="values" vs stream_mode="updates"
# ═══════════════════════════════════════════════════════════════════════════════
#
# LangGraph supports TWO streaming modes:
#
# stream_mode="values":
#   Yields the FULL STATE after each node executes.
#   You see the complete state snapshot at every step.
#   Use when: you want to see the entire conversation at each step.
#
# stream_mode="updates":
#   Yields ONLY the CHANGES (delta) from each node.
#   You see what each node added/modified.
#   Use when: you want to see what each node did (debugging, logging).
#
# 🔄 LANGCHAIN vs LANGGRAPH:
#   LangChain: chain.stream() → yields output tokens one by one
#   LangGraph: graph.stream() → yields state snapshots per NODE step
#              (for token-level streaming, use astream_events)

def demo_streaming_values() -> None:
    """stream_mode='values' — full state after each node."""
    import operator
    from typing import Annotated, TypedDict

    from langchain_core.messages import HumanMessage
    from langchain_groq import ChatGroq
    from langgraph.graph import END, START, StateGraph

    class ChatState(TypedDict):
        messages: Annotated[list, operator.add]

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_tokens=128)

    def chatbot_node(state: ChatState) -> dict:
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    builder = StateGraph(ChatState)
    builder.add_node("chatbot", chatbot_node)
    builder.add_edge(START, "chatbot")
    builder.add_edge("chatbot", END)
    graph = builder.compile()

    logger.info("--- Demo 3: stream_mode='values' (full state per step) ---")

    # stream_mode="values" yields the FULL state after each node
    for i, state_snapshot in enumerate(graph.stream(
        {"messages": [HumanMessage(content="What is Python in one sentence?")]},
        stream_mode="values",
    )):
        msg_count = len(state_snapshot["messages"])
        last_msg = state_snapshot["messages"][-1]
        logger.info("  Step %d | Messages: %d | Last: [%s] %s",
                     i, msg_count, type(last_msg).__name__, str(last_msg.content)[:80])
    # Step 0: 1 message (HumanMessage — the input)
    # Step 1: 2 messages (HumanMessage + AIMessage — after chatbot node)


def demo_streaming_updates() -> None:
    """stream_mode='updates' — only changes from each node."""
    import operator
    from typing import Annotated, TypedDict

    from langchain_core.messages import HumanMessage
    from langchain_groq import ChatGroq
    from langgraph.graph import END, START, StateGraph

    class ChatState(TypedDict):
        messages: Annotated[list, operator.add]

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_tokens=128)

    def chatbot_node(state: ChatState) -> dict:
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    builder = StateGraph(ChatState)
    builder.add_node("chatbot", chatbot_node)
    builder.add_edge(START, "chatbot")
    builder.add_edge("chatbot", END)
    graph = builder.compile()

    logger.info("--- Demo 4: stream_mode='updates' (changes per node) ---")

    # stream_mode="updates" yields a dict: {node_name: {state_updates}}
    for update in graph.stream(
        {"messages": [HumanMessage(content="What is FastAPI in one sentence?")]},
        stream_mode="updates",
    ):
        for node_name, node_output in update.items():
            logger.info("  Node '%s' produced:", node_name)
            if "messages" in node_output:
                for msg in node_output["messages"]:
                    logger.info("    [%s] %s", type(msg).__name__, str(msg.content)[:100])
    # Only shows what the chatbot node ADDED — not the full state


# ═══════════════════════════════════════════════════════════════════════════════
# 8. COMMON MISTAKES & GOTCHAS
# ═══════════════════════════════════════════════════════════════════════════════
#
# Mistake                              | Fix
# ─────────────────────────────────────|──────────────────────────────────
# Forgetting operator.add on messages  | Messages overwrite instead of append
# Not passing needs_search in initial  | KeyError on first router call
#   state for conditional graphs       |
# Confusing stream_mode values/updates | values = full state, updates = delta
# Using graph.stream() for token-level | Use astream_events for token streaming
#   streaming                          |


# ═══════════════════════════════════════════════════════════════════════════════
# 🎯 INTERVIEW QUESTIONS
# ═══════════════════════════════════════════════════════════════════════════════
#
# THEORETICAL:
# Q: What is the difference between stream_mode="values" and "updates"?
# A: "values" yields the FULL state snapshot after each node executes —
#    you see the complete conversation at every step. "updates" yields
#    ONLY the changes (delta) that each node produced — you see what
#    each node added or modified. Use "values" for UI rendering,
#    "updates" for debugging and logging.
#
# HANDS-ON:
# Q: Build a chatbot graph that uses stream_mode="values" to print
#    the message count after each step.
# (Solution: see demo_streaming_values above)


# ═══════════════════════════════════════════════════════════════════════════════
# 📝 QUICK RECAP
# ═══════════════════════════════════════════════════════════════════════════════
#
# 1. Connect an LLM to a LangGraph node: node calls llm.invoke(state["messages"]),
#    returns {"messages": [response]} which appends via operator.add.
#
# 2. Conditional routing enables the AGENT LOOP: chatbot → search → chatbot → END.
#    The router function reads state and returns the next node name.
#
# 3. Streaming: stream_mode="values" for full state per step,
#    stream_mode="updates" for only the changes each node made.


# ═══════════════════════════════════════════════════════════════════════════════
# 🚀 Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("💬 LANGGRAPH LESSON 3 — LLM Chatbot with Streaming")
    logger.info("=" * 70)

    logger.info("\n🔹 Demo 1: Simple LLM Chatbot (single node)")
    demo_simple_chatbot()

    logger.info("\n🔹 Demo 2: Chatbot with Conditional Routing (search cycle)")
    demo_chatbot_with_routing()

    logger.info("\n🔹 Demo 3: Streaming — stream_mode='values'")
    demo_streaming_values()

    logger.info("\n🔹 Demo 4: Streaming — stream_mode='updates'")
    demo_streaming_updates()

    logger.info("\n" + "=" * 70)
    logger.info("✅ Lesson 3 complete — LLM Chatbot with Streaming")
    logger.info("Graph images saved in: src/LangGraph/graphs/")
    logger.info("Next: Lesson 4 — Memory & Checkpointing")
    logger.info("=" * 70)
