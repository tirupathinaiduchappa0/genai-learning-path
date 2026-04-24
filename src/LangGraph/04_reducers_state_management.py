"""
🔄 LangGraph Lesson 4 — Reducers: How State Gets Updated (Level 2)

═══════════════════════════════════════════════════════════════════
1. CONCEPT: Reducers — ONE-LINE DEFINITION
═══════════════════════════════════════════════════════════════════

A Reducer is a function that decides HOW a state field gets updated
when a node returns a new value for that field.

Without a reducer: new value OVERWRITES the old value.
With a reducer:    new value is COMBINED with the old value.

═══════════════════════════════════════════════════════════════════
2. WHY REDUCERS EXIST
═══════════════════════════════════════════════════════════════════

In a graph, MULTIPLE nodes write to the SAME state field.
The question is: what happens when Node B writes to a field
that Node A already wrote to?

    WITHOUT reducer (default):
        Node A writes: messages = ["Hello"]
        Node B writes: messages = ["Goodbye"]
        Final state:   messages = ["Goodbye"]  ← OVERWRITTEN! "Hello" is LOST!

    WITH reducer (operator.add):
        Node A writes: messages = ["Hello"]
        Node B writes: messages = ["Goodbye"]
        Final state:   messages = ["Hello", "Goodbye"]  ← APPENDED! Both kept!

For a chatbot, you NEED messages to accumulate — not overwrite.
That's why reducers are critical.

═══════════════════════════════════════════════════════════════════
3. WHERE IT FITS
═══════════════════════════════════════════════════════════════════

    State (TypedDict)
        ↓
    Each field can have a REDUCER (via Annotated)
        ↓
    When a node returns updates, the reducer decides HOW to merge
        ↓
    No reducer = overwrite | operator.add = append | add_messages = smart append

═══════════════════════════════════════════════════════════════════
4. REAL-WORLD ANALOGY
═══════════════════════════════════════════════════════════════════

Think of a SHARED WHITEBOARD in a meeting room:

    No reducer (overwrite):
        Person A writes notes → Person B ERASES everything and writes their notes.
        Only Person B's notes survive.

    operator.add (blind append):
        Person A writes notes → Person B writes BELOW Person A's notes.
        Both notes survive. But if someone writes the same thing twice, duplicates appear.

    add_messages (smart append):
        Person A writes notes → Person B writes below.
        If Person B corrects something Person A wrote, the OLD version is REPLACED.
        No duplicates. Smart merging.

═══════════════════════════════════════════════════════════════════
5. THE 3 REDUCERS YOU NEED TO KNOW
═══════════════════════════════════════════════════════════════════

    Reducer          | What It Does              | When to Use
    ─────────────────|───────────────────────────|──────────────────────
    (none)           | OVERWRITES old value      | Simple fields (str, int, bool)
    operator.add     | APPENDS to list           | Generic lists, step logs
    add_messages     | SMART APPEND for messages | Messages (production standard)

    add_messages is the ONE you'll use 90% of the time in production.
    It's specifically designed for LangChain message lists.

HOW TO RUN:
    $ python src/LangGraph/04_reducers_state_management.py

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

GRAPH_DIR = os.path.join(os.path.dirname(__file__), "graphs")
os.makedirs(GRAPH_DIR, exist_ok=True)


def save_graph_image(compiled_graph, name: str) -> None:
    """Save the compiled graph as a PNG image."""
    try:
        png_data = compiled_graph.get_graph().draw_mermaid_png()
        filepath = os.path.join(GRAPH_DIR, f"{name}.png")
        with open(filepath, "wb") as f:
            f.write(png_data)
        logger.info("  Graph image saved: %s", filepath)
    except Exception as e:
        logger.warning("  Could not save graph image: %s", e)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. DEMO 1: No Reducer (Default) — Overwrite Behavior
# ═══════════════════════════════════════════════════════════════════════════════
#
# When a state field has NO reducer, returning a value OVERWRITES the old one.
# This is fine for simple fields like status, current_step, etc.
# But DANGEROUS for lists — you lose all previous entries!

def demo_no_reducer() -> None:
    """Show that without a reducer, state fields get OVERWRITTEN."""
    from typing_extensions import TypedDict
    from langgraph.graph import END, START, StateGraph

    # No Annotated, no reducer — plain fields
    class State(TypedDict):
        status: str       # Overwrite is FINE here (we want the latest status)
        messages: list    # Overwrite is BAD here (we lose old messages!)

    def node_a(state: State) -> dict:
        return {"status": "processing", "messages": ["Message from Node A"]}

    def node_b(state: State) -> dict:
        return {"status": "complete", "messages": ["Message from Node B"]}

    builder = StateGraph(State)
    builder.add_node("node_a", node_a)
    builder.add_node("node_b", node_b)
    builder.add_edge(START, "node_a")
    builder.add_edge("node_a", "node_b")
    builder.add_edge("node_b", END)
    graph = builder.compile()

    result = graph.invoke({"status": "", "messages": []})

    logger.info("--- Demo 1: No Reducer (Overwrite) ---")
    logger.info("  status: '%s' (overwritten — correct, we want latest)", result["status"])
    logger.info("  messages: %s (OVERWRITTEN — Node A's message is LOST!)", result["messages"])
    # messages = ["Message from Node B"] — Node A's message is gone!


# ═══════════════════════════════════════════════════════════════════════════════
# 7. DEMO 2: operator.add — Generic List Append
# ═══════════════════════════════════════════════════════════════════════════════
#
# operator.add concatenates lists: [1, 2] + [3] = [1, 2, 3]
# It's a GENERIC Python operator — works on any list.
# It does NOT understand LangChain messages — it just blindly appends.
#
# GOOD FOR: step logs, generic lists, counters
# NOT IDEAL FOR: LangChain messages (use add_messages instead)

def demo_operator_add() -> None:
    """operator.add — blindly appends to lists."""
    import operator
    from typing import Annotated

    from typing_extensions import TypedDict
    from langgraph.graph import END, START, StateGraph

    class State(TypedDict):
        messages: Annotated[list, operator.add]  # Append, don't overwrite
        status: str                               # No reducer — overwrite is fine

    def node_a(state: State) -> dict:
        return {"messages": ["Message from Node A"], "status": "processing"}

    def node_b(state: State) -> dict:
        return {"messages": ["Message from Node B"], "status": "complete"}

    builder = StateGraph(State)
    builder.add_node("node_a", node_a)
    builder.add_node("node_b", node_b)
    builder.add_edge(START, "node_a")
    builder.add_edge("node_a", "node_b")
    builder.add_edge("node_b", END)
    graph = builder.compile()

    result = graph.invoke({"messages": [], "status": ""})

    logger.info("--- Demo 2: operator.add (Append) ---")
    logger.info("  messages: %s (APPENDED — both kept!)", result["messages"])
    logger.info("  status: '%s' (overwritten — correct)", result["status"])
    # messages = ["Message from Node A", "Message from Node B"] — both kept!


# ═══════════════════════════════════════════════════════════════════════════════
# 8. DEMO 3: add_messages — The Production Standard for Messages
# ═══════════════════════════════════════════════════════════════════════════════
#
# add_messages is from langgraph.graph.message and is SPECIFICALLY designed
# for LangChain message lists. It's SMARTER than operator.add:
#
#   1. APPENDS new messages (like operator.add)
#   2. DEDUPLICATES by message ID (if a message with the same ID exists,
#      it REPLACES the old one instead of duplicating)
#   3. Supports MESSAGE REMOVAL (return RemoveMessage to delete a message)
#
# This is what PRODUCTION LangGraph apps use for the messages field.
#
# HOW TO USE:
#   from langgraph.graph import MessagesState  ← pre-built state with add_messages
#   OR
#   from langgraph.graph.message import add_messages
#   class MyState(TypedDict):
#       messages: Annotated[list, add_messages]

def demo_add_messages() -> None:
    """add_messages — smart append with deduplication (production standard)."""
    from typing import Annotated

    from langchain_core.messages import AIMessage, HumanMessage
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages
    from typing_extensions import TypedDict

    # ── Using add_messages reducer ───────────────────────────────────
    class ChatState(TypedDict):
        messages: Annotated[list, add_messages]  # Smart append!

    def node_a(state: ChatState) -> dict:
        return {"messages": [AIMessage(content="Response from Node A")]}

    def node_b(state: ChatState) -> dict:
        return {"messages": [AIMessage(content="Response from Node B")]}

    builder = StateGraph(ChatState)
    builder.add_node("node_a", node_a)
    builder.add_node("node_b", node_b)
    builder.add_edge(START, "node_a")
    builder.add_edge("node_a", "node_b")
    builder.add_edge("node_b", END)
    graph = builder.compile()

    result = graph.invoke({"messages": [HumanMessage(content="Hello")]})

    logger.info("--- Demo 3: add_messages (Production Standard) ---")
    logger.info("  Total messages: %d", len(result["messages"]))
    for i, msg in enumerate(result["messages"]):
        logger.info("  [%d] %s: %s", i, type(msg).__name__, msg.content[:60])
    # [0] HumanMessage: Hello
    # [1] AIMessage: Response from Node A
    # [2] AIMessage: Response from Node B


# ═══════════════════════════════════════════════════════════════════════════════
# 9. DEMO 4: MessagesState — Pre-Built State (Shortcut)
# ═══════════════════════════════════════════════════════════════════════════════
#
# LangGraph provides MessagesState — a pre-built TypedDict with:
#   messages: Annotated[list, add_messages]
#
# This saves you from defining the state yourself for simple chatbots.
# It's the QUICKEST way to get started.
#
# You can EXTEND it by subclassing:
#   class MyState(MessagesState):
#       extra_field: str

def demo_messages_state_shortcut() -> None:
    """MessagesState — pre-built state with add_messages (the shortcut)."""
    from langchain_core.messages import HumanMessage
    from langchain_groq import ChatGroq
    from langgraph.graph import END, START, MessagesState, StateGraph

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_tokens=128)

    # MessagesState already has: messages: Annotated[list, add_messages]
    # No need to define your own state class for simple chatbots!
    def chatbot(state: MessagesState) -> dict:
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    builder = StateGraph(MessagesState)
    builder.add_node("chatbot", chatbot)
    builder.add_edge(START, "chatbot")
    builder.add_edge("chatbot", END)
    graph = builder.compile()

    save_graph_image(graph, "lesson04_messages_state")

    result = graph.invoke({"messages": [HumanMessage(content="What is a reducer in LangGraph?")]})

    logger.info("--- Demo 4: MessagesState (Pre-Built Shortcut) ---")
    logger.info("  Using MessagesState — no custom state class needed!")
    logger.info("  Q: What is a reducer in LangGraph?")
    logger.info("  A: %s", result["messages"][-1].content[:150])


# ═══════════════════════════════════════════════════════════════════════════════
# 10. DEMO 5: Custom Reducer — Build Your Own
# ═══════════════════════════════════════════════════════════════════════════════
#
# You can write ANY function as a reducer. It takes (old_value, new_value)
# and returns the merged result.
#
# Use cases:
#   - Keep only the last N items
#   - Deduplicate entries
#   - Sum counters
#   - Custom merge logic

def demo_custom_reducer() -> None:
    """Custom reducer — keep only the last 3 log entries."""
    from typing import Annotated

    from typing_extensions import TypedDict
    from langgraph.graph import END, START, StateGraph

    # Custom reducer: keeps only the last 3 entries
    def keep_last_3(old: list, new: list) -> list:
        """Append new items, but keep only the last 3."""
        combined = old + new
        return combined[-3:]  # Trim to last 3

    class State(TypedDict):
        logs: Annotated[list, keep_last_3]  # Custom reducer!

    def step_1(state: State) -> dict:
        return {"logs": ["step_1 done"]}

    def step_2(state: State) -> dict:
        return {"logs": ["step_2 done"]}

    def step_3(state: State) -> dict:
        return {"logs": ["step_3 done"]}

    def step_4(state: State) -> dict:
        return {"logs": ["step_4 done"]}

    builder = StateGraph(State)
    builder.add_node("step_1", step_1)
    builder.add_node("step_2", step_2)
    builder.add_node("step_3", step_3)
    builder.add_node("step_4", step_4)
    builder.add_edge(START, "step_1")
    builder.add_edge("step_1", "step_2")
    builder.add_edge("step_2", "step_3")
    builder.add_edge("step_3", "step_4")
    builder.add_edge("step_4", END)
    graph = builder.compile()

    result = graph.invoke({"logs": []})

    logger.info("--- Demo 5: Custom Reducer (keep last 3) ---")
    logger.info("  4 nodes ran, but only last 3 logs kept: %s", result["logs"])
    # ["step_2 done", "step_3 done", "step_4 done"] — step_1 was trimmed!


# ═══════════════════════════════════════════════════════════════════════════════
# 11. operator.add vs add_messages — THE KEY DIFFERENCE
# ═══════════════════════════════════════════════════════════════════════════════
#
# Feature              | operator.add          | add_messages
# ─────────────────────|───────────────────────|──────────────────────
# Type                 | Generic Python        | LangChain-specific
# Works on             | Any list              | Message lists only
# Deduplication        | No (blindly appends)  | Yes (by message ID)
# Message removal      | No                    | Yes (RemoveMessage)
# Message replacement  | No                    | Yes (same ID = replace)
# Production use       | Step logs, counters   | Messages (always)
#
# RULE: For messages, ALWAYS use add_messages (or MessagesState).
#       For other lists, operator.add is fine.


# ═══════════════════════════════════════════════════════════════════════════════
# 12. COMMON MISTAKES & GOTCHAS
# ═══════════════════════════════════════════════════════════════════════════════
#
# Mistake                              | Fix
# ─────────────────────────────────────|──────────────────────────────────
# No reducer on messages field         | Messages overwrite — use add_messages
# Using operator.add for messages      | Works but no dedup — use add_messages
# Forgetting Annotated[] syntax        | Field has no reducer — overwrites
# Custom reducer with wrong signature  | Must take (old_value, new_value)
# Returning full list from node        | Return ONLY new items to append


# ═══════════════════════════════════════════════════════════════════════════════
# 13. WHERE THIS CONNECTS
# ═══════════════════════════════════════════════════════════════════════════════
#
# Reducers → control HOW state fields merge
# add_messages → production standard for message lists
# MessagesState → pre-built state using add_messages
# operator.add → generic append for non-message lists
# Custom reducers → any merge logic (trim, dedup, sum)
# State → read by Nodes → updated via Reducers → flows through Edges


# ═══════════════════════════════════════════════════════════════════════════════
# 🎯 INTERVIEW QUESTIONS
# ═══════════════════════════════════════════════════════════════════════════════
#
# THEORETICAL:
# Q: What is the difference between operator.add and add_messages in LangGraph?
# A: operator.add is a generic Python function that blindly concatenates lists.
#    add_messages is a LangChain-specific reducer that handles message
#    deduplication by ID, supports message replacement (same ID = update),
#    and supports message removal via RemoveMessage. For message state fields,
#    add_messages is the production standard because it prevents duplicates
#    and supports message lifecycle management.
#
# HANDS-ON:
# Q: Create a state with a "scores" field that uses a custom reducer to
#    keep only the top 3 highest scores.
#
# SOLUTION:
#   def top_3_scores(old: list, new: list) -> list:
#       combined = old + new
#       return sorted(combined, reverse=True)[:3]
#
#   class GameState(TypedDict):
#       scores: Annotated[list, top_3_scores]


# ═══════════════════════════════════════════════════════════════════════════════
# 📝 QUICK RECAP
# ═══════════════════════════════════════════════════════════════════════════════
#
# 1. Reducers decide HOW state fields merge: no reducer = overwrite,
#    operator.add = blind append, add_messages = smart append with dedup.
#
# 2. For messages, ALWAYS use add_messages (or MessagesState shortcut).
#    operator.add works but lacks deduplication and message management.
#
# 3. Custom reducers take (old, new) → return merged. Use for trimming,
#    dedup, counters, or any custom merge logic.


# ═══════════════════════════════════════════════════════════════════════════════
# 🚀 Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("🔄 LANGGRAPH LESSON 4 — Reducers: How State Gets Updated")
    logger.info("=" * 70)

    logger.info("\n🔹 Demo 1: No Reducer (overwrite behavior)")
    demo_no_reducer()

    logger.info("\n🔹 Demo 2: operator.add (blind append)")
    demo_operator_add()

    logger.info("\n🔹 Demo 3: add_messages (production standard)")
    demo_add_messages()

    logger.info("\n🔹 Demo 4: MessagesState (pre-built shortcut)")
    demo_messages_state_shortcut()

    logger.info("\n🔹 Demo 5: Custom Reducer (keep last 3)")
    demo_custom_reducer()

    logger.info("\n" + "=" * 70)
    logger.info("✅ Lesson 4 complete — Reducers & State Management")
    logger.info("Rule: For messages, ALWAYS use add_messages or MessagesState")
    logger.info("Next: Lesson 5 — Memory & Checkpointing")
    logger.info("=" * 70)
