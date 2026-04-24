"""
🧠 LangGraph Lesson 8 — Memory & State Persistence: Checkpointing (Level 4)

═══════════════════════════════════════════════════════════════════
1. CONCEPT: Memory & Checkpointing — ONE-LINE DEFINITION
═══════════════════════════════════════════════════════════════════

A Checkpointer saves the graph's state after each step, so the agent
can REMEMBER previous conversations and resume where it left off.

    Without memory: each invoke() is independent — the agent forgets everything.
    With memory:    each invoke() builds on previous ones — the agent remembers.

═══════════════════════════════════════════════════════════════════
2. WHY MEMORY EXISTS (the problem it solves)
═══════════════════════════════════════════════════════════════════

LLMs are STATELESS. Every API call is independent.
The model doesn't remember what you said 5 seconds ago.

    Call 1: "My name is Krish"  → "Nice to meet you, Krish!"
    Call 2: "What is my name?"  → "I don't know your name."  ← FORGOT!

This is a fundamental limitation. In LangChain, you used ConversationBufferMemory
or ConversationSummaryMemory to work around this. But those were bolted on —
not part of the core architecture.

In LangGraph, memory is BUILT IN via checkpointers:
    - MemorySaver: saves state in memory (dev/testing only)
    - SqliteSaver: saves state to a SQLite file (simple persistence)
    - PostgresSaver: saves state to PostgreSQL (production standard)

HOW IT WORKS:
    1. Compile the graph with a checkpointer: graph.compile(checkpointer=memory)
    2. Pass a thread_id in config: {"configurable": {"thread_id": "user-1"}}
    3. Same thread_id = continued conversation (agent remembers)
    4. Different thread_id = fresh conversation (agent forgets)

═══════════════════════════════════════════════════════════════════
3. WHERE IT FITS IN THE LANGGRAPH ARCHITECTURE
═══════════════════════════════════════════════════════════════════

    State (messages with add_messages reducer)
        ↓
    LLM Node (calls the model)
        ↓
    Checkpointer (saves state after EACH step)
        ↓
    Next invoke() → Checkpointer loads previous state → LLM sees full history

    The checkpointer is INVISIBLE to the graph logic.
    You don't change any nodes or edges — just add it at compile time.
    graph.compile(checkpointer=MemorySaver())

═══════════════════════════════════════════════════════════════════
4. REAL-WORLD ANALOGY
═══════════════════════════════════════════════════════════════════

Think of a DOCTOR'S PATIENT FILE:

    Visit 1: Patient says "I have a headache." Doctor writes it down.
    Visit 2: Patient returns. Doctor reads the file: "Last time: headache."
             Doctor asks: "Is the headache still there?"
             → The doctor REMEMBERS because of the file.

    Without the file (no checkpointer):
    Visit 2: "What brings you in today?" → starts from scratch every time.

    The patient file IS the checkpointer.
    The patient's name IS the thread_id.
    Each visit IS an invoke() call.
    Reading previous notes IS loading the checkpoint.

═══════════════════════════════════════════════════════════════════
5. ASCII GRAPH STRUCTURE
═══════════════════════════════════════════════════════════════════

ReAct Agent WITHOUT Memory (Lesson 07):

    invoke("My name is Krish")  →  "Nice to meet you!"
    invoke("What is my name?")  →  "I don't know."  ← FORGOT!

    Each invoke() starts with EMPTY state.

ReAct Agent WITH Memory (This Lesson):

    invoke("My name is Krish", thread_id="user-1")  →  "Nice to meet you!"
        ↓ [checkpoint saved for thread "user-1"]
    invoke("What is my name?", thread_id="user-1")  →  "Your name is Krish!"
        ↓ [checkpoint loaded → LLM sees full history]

    Same thread_id = continued conversation.
    Different thread_id = fresh conversation (session isolation).

═══════════════════════════════════════════════════════════════════

🔄 LANGCHAIN vs LANGGRAPH
LangChain : ConversationBufferMemory — bolted on, separate from chain logic.
            Memory was a wrapper around the chain, not part of the graph.
LangGraph  : Checkpointer — built into the graph at compile time.
            State is saved/loaded automatically. No extra wrappers needed.

LangChain : Memory types were confusing (Buffer, Summary, Window, Entity...).
LangGraph  : One concept: Checkpointer. Swap implementations for different
            storage backends (memory, SQLite, PostgreSQL, Redis).

HOW TO RUN:
    $ python src/LangGraph/08_memory_checkpointing.py

Author: GenAI Learner
Date: 2025-07-15
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
# 6. DEMO 1: ReAct Agent WITHOUT Memory (No Checkpointer)
# ═══════════════════════════════════════════════════════════════════════════════
#
# First, we show the PROBLEM: without a checkpointer, each invoke() is
# completely independent. The agent has NO memory of previous conversations.
#
# Test:
#   invoke("My name is Krish")   → agent responds
#   invoke("What is my name?")   → agent DOESN'T remember!
#
# This proves that LLMs are stateless by default.
# Each invoke() starts with a fresh, empty state.

def demo_no_memory() -> None:
    """ReAct agent WITHOUT memory. Each invoke() is independent."""
    from typing import Annotated

    from langchain_core.messages import AnyMessage, HumanMessage
    from langchain_core.tools import tool
    from langchain_groq import ChatGroq
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import ToolNode, tools_condition
    from typing_extensions import TypedDict

    # ── State ────────────────────────────────────────────────────────────
    class State(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]

    # ── Simple tool (so it's a ReAct agent, not just a chatbot) ──────────
    @tool
    def add(a: int, b: int) -> int:
        """Add two integers together. Use this when the user asks to add numbers.

        Args:
            a: first integer
            b: second integer

        Returns:
            The sum of a and b
        """
        return a + b

    tools = [add]

    # ── LLM with tools bound ────────────────────────────────────────────
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_tokens=256)
    llm_with_tools = llm.bind_tools(tools)

    # ── Node ─────────────────────────────────────────────────────────────
    def tool_calling_llm(state: State) -> dict:
        """Call the LLM with tools bound."""
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    # ── Build ReAct graph (NO checkpointer) ──────────────────────────────
    builder = StateGraph(State)
    builder.add_node("tool_calling_llm", tool_calling_llm)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "tool_calling_llm")
    builder.add_conditional_edges("tool_calling_llm", tools_condition)
    builder.add_edge("tools", "tool_calling_llm")  # ReAct loop

    graph = builder.compile()  # NO checkpointer!
    save_graph_image(graph, "lesson08_no_memory")

    # ── Test 1: Tell the agent your name ─────────────────────────────────
    logger.info("--- Demo 1: ReAct Agent WITHOUT Memory ---")

    result1 = graph.invoke(
        {"messages": [HumanMessage(content="My name is Krish")]},
        config={"recursion_limit": 25},
    )
    logger.info("  Turn 1 — Tell name:")
    logger.info("  Q: My name is Krish")
    logger.info("  A: %s", result1["messages"][-1].content[:200])

    # ── Test 2: Ask the agent your name — it WON'T remember! ────────────
    result2 = graph.invoke(
        {"messages": [HumanMessage(content="What is my name?")]},
        config={"recursion_limit": 25},
    )
    logger.info("  Turn 2 — Ask name:")
    logger.info("  Q: What is my name?")
    logger.info("  A: %s", result2["messages"][-1].content[:200])
    logger.info("  WITHOUT memory, each invoke() is independent.")
    logger.info("  The agent FORGOT your name because there's no checkpointer.")



# ═══════════════════════════════════════════════════════════════════════════════
# 7. DEMO 2: ReAct Agent WITH MemorySaver (Checkpointer)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Now we add MemorySaver — the simplest checkpointer.
# It saves state in memory (Python dict) after each step.
#
# KEY CHANGES from Demo 1:
#   1. Import MemorySaver from langgraph.checkpoint.memory
#   2. Create: memory = MemorySaver()
#   3. Compile with: graph.compile(checkpointer=memory)
#   4. Pass thread_id in config: {"configurable": {"thread_id": "user-1"}}
#
# Same thread_id = continued conversation (agent remembers).
# The checkpointer loads the previous state before each invoke().
#
# Test:
#   invoke("My name is Krish", thread_id="user-1")  → agent responds
#   invoke("What is my name?", thread_id="user-1")   → agent REMEMBERS!

def demo_with_memory() -> None:
    """ReAct agent WITH MemorySaver. Same thread_id = continued conversation."""
    from typing import Annotated

    from langchain_core.messages import AnyMessage, HumanMessage
    from langchain_core.tools import tool
    from langchain_groq import ChatGroq
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import ToolNode, tools_condition
    from typing_extensions import TypedDict

    # ── State ────────────────────────────────────────────────────────────
    class State(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]

    # ── Simple tool ──────────────────────────────────────────────────────
    @tool
    def add(a: int, b: int) -> int:
        """Add two integers together. Use this when the user asks to add numbers.

        Args:
            a: first integer
            b: second integer

        Returns:
            The sum of a and b
        """
        return a + b

    tools = [add]

    # ── LLM with tools bound ────────────────────────────────────────────
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_tokens=256)
    llm_with_tools = llm.bind_tools(tools)

    # ── Node ─────────────────────────────────────────────────────────────
    def tool_calling_llm(state: State) -> dict:
        """Call the LLM with tools bound."""
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    # ── Build ReAct graph WITH MemorySaver ───────────────────────────────
    builder = StateGraph(State)
    builder.add_node("tool_calling_llm", tool_calling_llm)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "tool_calling_llm")
    builder.add_conditional_edges("tool_calling_llm", tools_condition)
    builder.add_edge("tools", "tool_calling_llm")  # ReAct loop

    # ┌─────────────────────────────────────────────────────────────────────┐
    # │  THIS IS THE KEY — MemorySaver as checkpointer!                    │
    # │  Without checkpointer: graph.compile()                             │
    # │  With checkpointer:    graph.compile(checkpointer=MemorySaver())   │
    # │  This ONE change gives the agent memory.                           │
    # └─────────────────────────────────────────────────────────────────────┘
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)
    save_graph_image(graph, "lesson08_with_memory")

    # ── Config with thread_id — this identifies the conversation ─────────
    config = {"configurable": {"thread_id": "user-1"}}

    # ── Turn 1: Tell the agent your name ─────────────────────────────────
    logger.info("--- Demo 2: ReAct Agent WITH MemorySaver ---")

    result1 = graph.invoke(
        {"messages": [HumanMessage(content="My name is Krish")]},
        config=config,  # thread_id="user-1"
    )
    logger.info("  Turn 1 (thread: user-1) — Tell name:")
    logger.info("  Q: My name is Krish")
    logger.info("  A: %s", result1["messages"][-1].content[:200])

    # ── Turn 2: Ask the agent your name — it REMEMBERS! ─────────────────
    result2 = graph.invoke(
        {"messages": [HumanMessage(content="What is my name?")]},
        config=config,  # SAME thread_id="user-1"
    )
    logger.info("  Turn 2 (thread: user-1) — Ask name:")
    logger.info("  Q: What is my name?")
    logger.info("  A: %s", result2["messages"][-1].content[:200])
    logger.info("  WITH MemorySaver, the agent REMEMBERS across invoke() calls!")
    logger.info("  The thread_id 'user-1' links both turns into one conversation.")
    logger.info("  Total messages in state: %d", len(result2["messages"]))



# ═══════════════════════════════════════════════════════════════════════════════
# 8. DEMO 3: Session Isolation (Different thread_ids)
# ═══════════════════════════════════════════════════════════════════════════════
#
# thread_id isolates conversations. Different thread_ids = different sessions.
#
# Test:
#   Thread "user-1": "My name is Krish"  → agent remembers
#   Thread "user-2": "What is my name?"  → agent DOESN'T know (different session!)
#
# This is how you handle MULTIPLE USERS in production:
#   - Each user gets a unique thread_id
#   - Their conversations are completely isolated
#   - User A's messages never leak into User B's session

def demo_session_isolation() -> None:
    """Session isolation: different thread_ids = different conversations."""
    from typing import Annotated

    from langchain_core.messages import AnyMessage, HumanMessage
    from langchain_core.tools import tool
    from langchain_groq import ChatGroq
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import ToolNode, tools_condition
    from typing_extensions import TypedDict

    # ── State ────────────────────────────────────────────────────────────
    class State(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]

    # ── Simple tool ──────────────────────────────────────────────────────
    @tool
    def add(a: int, b: int) -> int:
        """Add two integers together. Use this when the user asks to add numbers.

        Args:
            a: first integer
            b: second integer

        Returns:
            The sum of a and b
        """
        return a + b

    tools = [add]

    # ── LLM with tools bound ────────────────────────────────────────────
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_tokens=256)
    llm_with_tools = llm.bind_tools(tools)

    # ── Node ─────────────────────────────────────────────────────────────
    def tool_calling_llm(state: State) -> dict:
        """Call the LLM with tools bound."""
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    # ── Build ReAct graph WITH MemorySaver ───────────────────────────────
    builder = StateGraph(State)
    builder.add_node("tool_calling_llm", tool_calling_llm)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "tool_calling_llm")
    builder.add_conditional_edges("tool_calling_llm", tools_condition)
    builder.add_edge("tools", "tool_calling_llm")  # ReAct loop

    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)

    # ── Thread 1: Tell the agent your name ───────────────────────────────
    logger.info("--- Demo 3: Session Isolation (Different thread_ids) ---")

    config_user1 = {"configurable": {"thread_id": "user-1"}}
    config_user2 = {"configurable": {"thread_id": "user-2"}}

    # User 1 tells their name
    result1 = graph.invoke(
        {"messages": [HumanMessage(content="My name is Krish")]},
        config=config_user1,  # thread_id="user-1"
    )
    logger.info("  Thread 'user-1' — Tell name:")
    logger.info("  Q: My name is Krish")
    logger.info("  A: %s", result1["messages"][-1].content[:200])

    # User 1 asks their name — REMEMBERS (same thread)
    result2 = graph.invoke(
        {"messages": [HumanMessage(content="What is my name?")]},
        config=config_user1,  # SAME thread_id="user-1"
    )
    logger.info("  Thread 'user-1' — Ask name:")
    logger.info("  Q: What is my name?")
    logger.info("  A: %s", result2["messages"][-1].content[:200])

    # User 2 asks their name — DOESN'T know (different thread!)
    result3 = graph.invoke(
        {"messages": [HumanMessage(content="What is my name?")]},
        config=config_user2,  # DIFFERENT thread_id="user-2"
    )
    logger.info("  Thread 'user-2' — Ask name:")
    logger.info("  Q: What is my name?")
    logger.info("  A: %s", result3["messages"][-1].content[:200])

    logger.info("")
    logger.info("  Thread 'user-1': REMEMBERS (same session)")
    logger.info("  Thread 'user-2': DOESN'T KNOW (different session)")
    logger.info("  thread_id isolates conversations — perfect for multi-user apps.")



# ═══════════════════════════════════════════════════════════════════════════════
# 9. DEMO 4: Production Memory Patterns (Conceptual — Comments Only)
# ═══════════════════════════════════════════════════════════════════════════════
#
# MemorySaver is great for development, but it's IN-MEMORY — lost on restart.
# In production, you need PERSISTENT storage. Here are the options:
#
# ┌──────────────────┬──────────────────────────────────────────────────────────┐
# │ Checkpointer     │ Description                                              │
# ├──────────────────┼──────────────────────────────────────────────────────────┤
# │ MemorySaver      │ In-memory (Python dict). Fast, zero setup.               │
# │                  │ Lost on restart. Dev/testing ONLY.                        │
# │                  │ Use: graph.compile(checkpointer=MemorySaver())            │
# ├──────────────────┼──────────────────────────────────────────────────────────┤
# │ SqliteSaver      │ File-based (SQLite). Simple persistence.                 │
# │                  │ Survives restarts. Single server only.                    │
# │                  │ Use: from langgraph.checkpoint.sqlite import SqliteSaver  │
# │                  │      graph.compile(checkpointer=SqliteSaver("state.db")) │
# ├──────────────────┼──────────────────────────────────────────────────────────┤
# │ PostgresSaver    │ Database-backed (PostgreSQL). Production standard.        │
# │                  │ Distributed, scalable, reliable.                          │
# │                  │ Use: from langgraph.checkpoint.postgres import            │
# │                  │        PostgresSaver                                      │
# │                  │      graph.compile(checkpointer=PostgresSaver(conn_str))  │
# ├──────────────────┼──────────────────────────────────────────────────────────┤
# │ Redis            │ In-memory cache (Redis). Fast, distributed.              │
# │                  │ Good for high-traffic apps with TTL (auto-expire).        │
# │                  │ Community implementation available.                       │
# └──────────────────┴──────────────────────────────────────────────────────────┘
#
# WHEN TO USE EACH:
#
# - Local development / testing?     → MemorySaver (zero setup)
# - Simple app, single server?       → SqliteSaver (file-based, survives restart)
# - Production, multiple servers?    → PostgresSaver (distributed, scalable)
# - High-traffic, need speed?        → Redis (fast, distributed, auto-expire)
#
# HOW TO SWITCH:
#   The beauty of LangGraph's checkpointer design is that switching is trivial.
#   You only change ONE line — the checkpointer in compile():
#
#   # Development:
#   graph = builder.compile(checkpointer=MemorySaver())
#
#   # Production:
#   graph = builder.compile(checkpointer=PostgresSaver(conn_str))
#
#   The rest of the graph (nodes, edges, state) stays EXACTLY the same.
#   This is the power of the checkpointer abstraction.
#
# COMPARISON TABLE:
#
# Feature          | MemorySaver | SqliteSaver | PostgresSaver | Redis
# ─────────────────|─────────────|─────────────|───────────────|──────────
# Persistence      | No          | Yes (file)  | Yes (DB)      | Yes (cache)
# Survives restart | No          | Yes         | Yes           | Yes*
# Distributed      | No          | No          | Yes           | Yes
# Setup complexity | None        | Low         | Medium        | Medium
# Speed            | Fastest     | Fast        | Fast          | Very fast
# Use case         | Dev/test    | Simple apps | Production    | High-traffic
# Data safety      | None        | Good        | Excellent     | Good*
#
# * Redis persistence depends on configuration (RDB/AOF snapshots).


# ═══════════════════════════════════════════════════════════════════════════════
# 10. COMMON MISTAKES & GOTCHAS
# ═══════════════════════════════════════════════════════════════════════════════
#
# Mistake                                    | Fix
# ───────────────────────────────────────────|──────────────────────────────────────
# Forgetting the checkpointer at compile()  | graph.compile(checkpointer=memory)
#   (agent has no memory!)                   | Without it, each invoke() is fresh.
# Forgetting thread_id in config             | config={"configurable": {"thread_id": "x"}}
#   (MemorySaver requires thread_id!)        | Without it, you get an error.
# Using MemorySaver in production            | Use SqliteSaver or PostgresSaver
#   (data lost on restart!)                  | MemorySaver is dev/testing only.
# Same thread_id for all users               | Each user needs a UNIQUE thread_id
#   (conversations leak between users!)      | Use user ID, session ID, etc.
# Not setting recursion_limit                | Always set recursion_limit on cyclic graphs
#   (infinite loops with memory!)            | config={"recursion_limit": 25, ...}
# Forgetting add_messages reducer            | Messages overwrite instead of append
#   (memory seems broken!)                   | Use Annotated[list, add_messages]
#
# ANTI-PATTERN: Using MemorySaver in production.
#   MemorySaver stores state in a Python dict — it's lost when the process
#   restarts. For production, always use SqliteSaver (simple) or PostgresSaver
#   (scalable). The switch is ONE line of code.


# ═══════════════════════════════════════════════════════════════════════════════
# 11. WHY LANGGRAPH FOR THIS
# ═══════════════════════════════════════════════════════════════════════════════
#
# ✅ Built-in persistence — checkpointers are first-class, not bolted on
# ✅ Swappable backends — MemorySaver, SqliteSaver, PostgresSaver, Redis
# ✅ Thread isolation — thread_id separates conversations automatically
# ✅ Automatic save/load — state is saved after each step, loaded before next
# ✅ Works with any graph — add checkpointer to ANY graph (chatbot, ReAct, etc.)
# ✅ Enables human-in-the-loop — pause, review, resume (needs checkpointer)
# ✅ Production-ready — PostgresSaver scales to millions of conversations
# ✅ Zero code changes — switch backends by changing ONE line at compile()


# ═══════════════════════════════════════════════════════════════════════════════
# 12. WHERE THIS CONNECTS (Concept Linking)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Lesson 07 (ReAct Agent) → tools -> tool_calling_llm (LOOP, multi-step)
# Lesson 08 (Memory)       → ReAct + MemorySaver = Conversational Agent
# Next lessons             → Human-in-the-loop (requires checkpointer!)
#
# MemorySaver → enables → persistence → enables → conversation memory
# thread_id → isolates → conversations → enables → multi-user support
# Checkpointer → saves state after each step → loads before next invoke()
# add_messages reducer → appends messages → checkpointer saves the full list
#
# StateGraph → compiles with checkpointer → CompiledGraph with memory
# Nodes → read/write → State → saved by → Checkpointer
# Human-in-the-loop → requires → Checkpointer (interrupt_before/after)


# ═══════════════════════════════════════════════════════════════════════════════
# INTERVIEW QUESTIONS
# ═══════════════════════════════════════════════════════════════════════════════
#
# THEORETICAL:
# Q: What is the difference between MemorySaver and PostgresSaver in LangGraph?
#    When would you use each, and what happens if you use MemorySaver in production?
#
# A: MemorySaver stores graph state in a Python dictionary (in-memory). It's fast
#    and requires zero setup, but all data is lost when the process restarts.
#    PostgresSaver stores state in a PostgreSQL database — it's persistent,
#    distributed, and scalable. In production, MemorySaver means every server
#    restart wipes all conversation history. Users would lose their context
#    mid-conversation. PostgresSaver survives restarts and works across multiple
#    servers. Use MemorySaver for development/testing, PostgresSaver for production.
#    Switching is trivial: change one line at compile() time.
#
# HANDS-ON:
# Q: Build a ReAct agent with MemorySaver that:
#    1. Remembers the user's favorite color (thread "user-1")
#    2. A different user (thread "user-2") asks about the color and gets no answer
#    Test session isolation with two different thread_ids.
#
# SOLUTION:
#   from typing import Annotated
#   from langchain_core.messages import AnyMessage, HumanMessage
#   from langchain_core.tools import tool
#   from langchain_groq import ChatGroq
#   from langgraph.checkpoint.memory import MemorySaver
#   from langgraph.graph import END, START, StateGraph
#   from langgraph.graph.message import add_messages
#   from langgraph.prebuilt import ToolNode, tools_condition
#   from typing_extensions import TypedDict
#
#   class State(TypedDict):
#       messages: Annotated[list[AnyMessage], add_messages]
#
#   @tool
#   def add(a: int, b: int) -> int:
#       """Add two integers."""
#       return a + b
#
#   tools = [add]
#   llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_tokens=256)
#   llm_with_tools = llm.bind_tools(tools)
#
#   def tool_calling_llm(state: State) -> dict:
#       return {"messages": [llm_with_tools.invoke(state["messages"])]}
#
#   builder = StateGraph(State)
#   builder.add_node("tool_calling_llm", tool_calling_llm)
#   builder.add_node("tools", ToolNode(tools))
#   builder.add_edge(START, "tool_calling_llm")
#   builder.add_conditional_edges("tool_calling_llm", tools_condition)
#   builder.add_edge("tools", "tool_calling_llm")
#   graph = builder.compile(checkpointer=MemorySaver())
#
#   # User 1: set favorite color
#   r1 = graph.invoke(
#       {"messages": [HumanMessage(content="My favorite color is blue")]},
#       config={"configurable": {"thread_id": "user-1"}},
#   )
#   print(r1["messages"][-1].content)
#
#   # User 1: ask favorite color (REMEMBERS)
#   r2 = graph.invoke(
#       {"messages": [HumanMessage(content="What is my favorite color?")]},
#       config={"configurable": {"thread_id": "user-1"}},
#   )
#   print(r2["messages"][-1].content)  # "blue"
#
#   # User 2: ask favorite color (DOESN'T KNOW — different thread)
#   r3 = graph.invoke(
#       {"messages": [HumanMessage(content="What is my favorite color?")]},
#       config={"configurable": {"thread_id": "user-2"}},
#   )
#   print(r3["messages"][-1].content)  # "I don't know"
#
# BONUS (System Design):
# Q: Design a memory system for a customer support chatbot that serves 10,000
#    concurrent users. What checkpointer would you use? How would you handle
#    thread_id generation? What about conversation expiry?
#
# A: Use PostgresSaver for persistence and scalability across multiple servers.
#    Generate thread_id from user_id + session_id (e.g., "user-123-session-456").
#    For conversation expiry, implement a cleanup job that deletes checkpoints
#    older than 24 hours (or use Redis with TTL for automatic expiry).
#    For 10K concurrent users, PostgreSQL with connection pooling (pgbouncer)
#    handles the load. Each user's conversation is isolated by thread_id.


# ═══════════════════════════════════════════════════════════════════════════════
# QUICK RECAP
# ═══════════════════════════════════════════════════════════════════════════════
#
# 1. LLMs are stateless. Without a checkpointer, each invoke() starts fresh.
#    MemorySaver saves state after each step and loads it before the next invoke().
#    Add it with ONE line: graph.compile(checkpointer=MemorySaver())
#
# 2. thread_id isolates conversations. Same thread_id = continued conversation.
#    Different thread_id = fresh conversation. Use unique IDs per user/session.
#    Config: {"configurable": {"thread_id": "user-1"}}
#
# 3. MemorySaver is dev-only (lost on restart). For production, use SqliteSaver
#    (simple) or PostgresSaver (scalable). Switching is ONE line at compile().
#    The checkpointer abstraction makes backend swaps trivial.


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("LANGGRAPH LESSON 8 -- Memory & State Persistence")
    logger.info("   Checkpointing: MemorySaver, thread_id, session isolation")
    logger.info("=" * 70)

    logger.info("\n Demo 1: ReAct Agent WITHOUT Memory (No Checkpointer)")
    demo_no_memory()

    logger.info("\n Demo 2: ReAct Agent WITH MemorySaver")
    demo_with_memory()

    logger.info("\n Demo 3: Session Isolation (Different thread_ids)")
    demo_session_isolation()

    logger.info("\n Demo 4: Production Memory Patterns")
    logger.info("  (See comments in source code for the full comparison table)")

    logger.info("\n" + "=" * 70)
    logger.info("Lesson 8 complete -- Memory & State Persistence")
    logger.info("Key: graph.compile(checkpointer=MemorySaver()) adds memory")
    logger.info("Pattern: Same thread_id = remember, different = forget")
    logger.info("Production: Use PostgresSaver, not MemorySaver")
    logger.info("Next: Lesson 9 -- Human-in-the-Loop (interrupt_before/after)")
    logger.info("=" * 70)
