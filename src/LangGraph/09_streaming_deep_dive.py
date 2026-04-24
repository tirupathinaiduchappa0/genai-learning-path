"""
🌊 LangGraph Lesson 9 — Streaming Deep Dive: Real-Time Data Flow (Level 4+)

═══════════════════════════════════════════════════════════════════
1. CONCEPT: Streaming — ONE-LINE DEFINITION
═══════════════════════════════════════════════════════════════════

Streaming lets you observe graph execution IN REAL-TIME — from coarse
node-level snapshots down to individual tokens as the LLM generates them.

═══════════════════════════════════════════════════════════════════
2. WHY STREAMING EXISTS (the problem it solves)
═══════════════════════════════════════════════════════════════════

Without streaming, you call graph.invoke() and WAIT. The user stares at a
blank screen for 2-10 seconds until the ENTIRE response is ready. This is
a terrible user experience.

With streaming, words appear in real-time — just like ChatGPT. The user
sees the response being "typed out" token by token. This feels fast,
responsive, and interactive, even though the total time is the same.

WHY IT MATTERS:
    - User experience: seeing words appear feels 10x faster than waiting
    - Debugging: watch each node's output as it happens (not after)
    - Production monitoring: track tool calls, LLM decisions, errors in real-time
    - Perceived latency: first token in 200ms vs full response in 5s

THREE LEVELS of streaming in LangGraph:

    Level 1: stream_mode="values"
        → Full state snapshot after EACH node executes (coarse-grained)
        → Like refreshing the ENTIRE page after each step

    Level 2: stream_mode="updates"
        → Only the CHANGES from each node (fine-grained per node)
        → Like seeing a diff — what changed, not the whole state

    Level 3: stream_mode="messages" / astream_events
        → Individual TOKENS as the LLM generates them (finest grain)
        → Like watching someone type letter by letter

═══════════════════════════════════════════════════════════════════
3. WHERE IT FITS IN THE LANGGRAPH ARCHITECTURE
═══════════════════════════════════════════════════════════════════

    graph.invoke()   → blocks until done, returns FINAL state (no streaming)
    graph.stream()   → yields chunks as graph executes (sync streaming)
    graph.astream()  → async version of stream() (async streaming)
    graph.astream_events() → yields DETAILED events including tokens (advanced)

    Streaming is an EXECUTION mode — it doesn't change the graph structure.
    Same nodes, same edges, same state. Just a different way to OBSERVE.

═══════════════════════════════════════════════════════════════════
4. REAL-WORLD ANALOGY — A LIVE SPORTS SCOREBOARD
═══════════════════════════════════════════════════════════════════

Imagine watching a basketball game:

    stream_mode="values" = showing the FULL SCOREBOARD after each play
        "Team A: 45, Team B: 42, Quarter: 2, Time: 3:21"
        → You see EVERYTHING, but it's a lot of data each time.

    stream_mode="updates" = showing ONLY what changed
        "Team A scored 3 points"
        → Compact, efficient. Great for debugging — what did THIS play do?

    stream_mode="messages" / astream_events = showing EVERY MOMENT in real-time
        "Ball thrown... caught by #23... dribbles left... shoots... SCORES!"
        → You see every micro-event as it happens. Maximum detail.

    In production chat UIs, you want Level 3 (token streaming).
    For debugging agent behavior, Level 2 (updates) is ideal.
    For state inspection, Level 1 (values) gives the full picture.

═══════════════════════════════════════════════════════════════════
5. ASCII GRAPH STRUCTURE (used in all demos)
═══════════════════════════════════════════════════════════════════

Simple Chatbot (Demos 1-4):

    [START]
       │
    [chatbot]  ← LLM generates response
       │
     [END]

ReAct Agent with Tools (Demo 5):

    [START]
       │
    [agent]  ← LLM reasons + decides
       │
       ├── (tool_call?) ──→ [tools] ──→ [agent]  (loop back)
       │
       └── (done?) ──→ [END]

═══════════════════════════════════════════════════════════════════

🔄 LANGCHAIN vs LANGGRAPH — Streaming
LangChain : chain.stream() yields output chunks, but no visibility into
            intermediate steps. You see the final output streaming, not
            what happened inside the chain.
LangGraph  : graph.stream() yields per-NODE updates. You see what each
            node produced, when tools were called, what the LLM decided.
            Full observability into the execution pipeline.

LangChain : No built-in way to stream tokens AND see tool calls.
LangGraph  : stream_mode="messages" streams tokens. stream_mode="updates"
            shows tool calls. astream_events gives you EVERYTHING.

HOW TO RUN:
    $ python src/LangGraph/09_streaming_deep_dive.py

Author: GenAI Learner
Date: 2025-07-15
"""

import asyncio
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
# 6. DEMO 1: stream_mode="updates" — Node-Level Changes
# ═══════════════════════════════════════════════════════════════════════════════
#
# stream_mode="updates" yields ONLY the changes from each node.
# Each chunk is a dict: {node_name: {state_updates}}
#
# This is the DEBUGGING mode — you see exactly what each node produced,
# without the noise of the full state. Perfect for understanding agent behavior.
#
# Example output:
#   chunk = {"chatbot": {"messages": [AIMessage(content="Hello!")]}}
#   → The "chatbot" node added one AIMessage to the state.

def demo_stream_updates() -> None:
    """stream_mode='updates' — see only what each node CHANGED."""
    from typing import Annotated

    from langchain_core.messages import AnyMessage, HumanMessage
    from langchain_groq import ChatGroq
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages
    from typing_extensions import TypedDict

    # ── State ────────────────────────────────────────────────────────────
    class State(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]

    # ── LLM ──────────────────────────────────────────────────────────────
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_tokens=256)

    # ── Node ─────────────────────────────────────────────────────────────
    def chatbot(state: State) -> dict:
        """Call the LLM and return the response as a state update."""
        return {"messages": [llm.invoke(state["messages"])]}

    # ── Build graph: START → chatbot → END ───────────────────────────────
    builder = StateGraph(State)
    builder.add_node("chatbot", chatbot)
    builder.add_edge(START, "chatbot")
    builder.add_edge("chatbot", END)

    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)
    save_graph_image(graph, "lesson09_streaming_chatbot")

    # ── Stream with mode="updates" ───────────────────────────────────────
    logger.info("=" * 70)
    logger.info("--- Demo 1: stream_mode='updates' — Node-Level Changes ---")
    logger.info("=" * 70)

    config = {"configurable": {"thread_id": "stream-updates-1"}}
    input_msg = {"messages": [HumanMessage(content="Hi, I'm Krish!")]}

    # ┌─────────────────────────────────────────────────────────────────────┐
    # │  stream_mode="updates" yields {node_name: {state_changes}}         │
    # │  You see ONLY what each node produced — not the full state.        │
    # │  This is ideal for DEBUGGING: "what did this node do?"             │
    # └─────────────────────────────────────────────────────────────────────┘
    for chunk in graph.stream(input_msg, config, stream_mode="updates"):
        # chunk = {"chatbot": {"messages": [AIMessage(...)]}}
        logger.info("  Chunk (updates): %s", chunk)
        for node_name, state_update in chunk.items():
            logger.info("    Node: '%s'", node_name)
            logger.info("    State update keys: %s", list(state_update.keys()))
            if "messages" in state_update:
                for msg in state_update["messages"]:
                    logger.info("    Message type: %s", type(msg).__name__)
                    logger.info("    Content: %s", msg.content[:200])

    logger.info("")
    logger.info("  KEY INSIGHT: Each chunk shows ONLY what one node changed.")
    logger.info("  Format: {node_name: {state_key: new_value}}")
    logger.info("  Use case: DEBUGGING — see what each node produced.")



# ═══════════════════════════════════════════════════════════════════════════════
# 7. DEMO 2: stream_mode="values" — Full State Snapshots
# ═══════════════════════════════════════════════════════════════════════════════
#
# stream_mode="values" yields the FULL state after each node executes.
# Each chunk is the complete state dict (all messages, all fields).
#
# You'll see TWO snapshots:
#   Step 0: Just the HumanMessage (the input, before any node runs)
#   Step 1: HumanMessage + AIMessage (after the chatbot node runs)
#
# This is the UI RENDERING mode — at each step, you have the complete
# conversation to display. No need to track diffs or accumulate changes.

def demo_stream_values() -> None:
    """stream_mode='values' — see the FULL state after each node."""
    from typing import Annotated

    from langchain_core.messages import AnyMessage, HumanMessage
    from langchain_groq import ChatGroq
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages
    from typing_extensions import TypedDict

    # ── State ────────────────────────────────────────────────────────────
    class State(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]

    # ── LLM ──────────────────────────────────────────────────────────────
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_tokens=256)

    # ── Node ─────────────────────────────────────────────────────────────
    def chatbot(state: State) -> dict:
        """Call the LLM and return the response as a state update."""
        return {"messages": [llm.invoke(state["messages"])]}

    # ── Build graph: START → chatbot → END ───────────────────────────────
    builder = StateGraph(State)
    builder.add_node("chatbot", chatbot)
    builder.add_edge(START, "chatbot")
    builder.add_edge("chatbot", END)

    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)

    # ── Stream with mode="values" ────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("--- Demo 2: stream_mode='values' — Full State Snapshots ---")
    logger.info("=" * 70)

    config = {"configurable": {"thread_id": "stream-values-1"}}
    input_msg = {"messages": [HumanMessage(content="Hi, I'm Krish!")]}

    # ┌─────────────────────────────────────────────────────────────────────┐
    # │  stream_mode="values" yields the COMPLETE state after each step.   │
    # │  Step 0: input state (just the HumanMessage)                       │
    # │  Step 1: full state after chatbot node (HumanMessage + AIMessage)  │
    # │  This is ideal for UI rendering — show the full conversation.      │
    # └─────────────────────────────────────────────────────────────────────┘
    step = 0
    for chunk in graph.stream(input_msg, config, stream_mode="values"):
        # chunk = {"messages": [HumanMessage(...), AIMessage(...)]}
        logger.info("  Step %d — Full state snapshot:", step)
        logger.info("    Total messages in state: %d", len(chunk["messages"]))
        for i, msg in enumerate(chunk["messages"]):
            logger.info("    Message[%d] (%s): %s",
                        i, type(msg).__name__, msg.content[:150])
        step += 1

    logger.info("")
    logger.info("  KEY INSIGHT: Each chunk is the FULL state — all messages.")
    logger.info("  Step 0 = input only. Step 1 = input + LLM response.")
    logger.info("  Use case: UI RENDERING — display the complete conversation.")



# ═══════════════════════════════════════════════════════════════════════════════
# 8. DEMO 3: stream_mode="messages" — Message-Level Token Streaming
# ═══════════════════════════════════════════════════════════════════════════════
#
# stream_mode="messages" streams individual MESSAGE CHUNKS (tokens) as the
# LLM generates them. This is the PRODUCTION STANDARD for chat UIs.
#
# Each chunk is a tuple: (message_chunk, metadata)
#   - message_chunk.content = the token text ("Hello", " Krish", "!", etc.)
#   - metadata = info about which node generated it
#
# This is how ChatGPT, Claude, and every modern chat UI works:
# tokens appear one by one, giving the "typing" effect.
#
# NOTE: stream_mode="messages" is the newer, simpler way to get token-level
# streaming. It replaced the more complex astream_events approach for most
# use cases.

def demo_stream_messages() -> None:
    """stream_mode='messages' — token-level streaming for chat UIs."""
    from typing import Annotated

    from langchain_core.messages import AnyMessage, HumanMessage, AIMessageChunk
    from langchain_groq import ChatGroq
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages
    from typing_extensions import TypedDict

    # ── State ────────────────────────────────────────────────────────────
    class State(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]

    # ── LLM ──────────────────────────────────────────────────────────────
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_tokens=256)

    # ── Node ─────────────────────────────────────────────────────────────
    def chatbot(state: State) -> dict:
        """Call the LLM and return the response as a state update."""
        return {"messages": [llm.invoke(state["messages"])]}

    # ── Build graph: START → chatbot → END ───────────────────────────────
    builder = StateGraph(State)
    builder.add_node("chatbot", chatbot)
    builder.add_edge(START, "chatbot")
    builder.add_edge("chatbot", END)

    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)

    # ── Stream with mode="messages" ──────────────────────────────────────
    logger.info("=" * 70)
    logger.info("--- Demo 3: stream_mode='messages' — Token-Level Streaming ---")
    logger.info("=" * 70)

    config = {"configurable": {"thread_id": "stream-messages-1"}}
    input_msg = {"messages": [HumanMessage(content="Hello Krish! Tell me a short joke.")]}

    # ┌─────────────────────────────────────────────────────────────────────┐
    # │  stream_mode="messages" yields (message_chunk, metadata) tuples.   │
    # │  Each chunk contains a SINGLE TOKEN from the LLM response.         │
    # │  Filter for AIMessageChunk to get only the LLM's output tokens.    │
    # │  This is the PRODUCTION STANDARD for real-time chat UIs.           │
    # └─────────────────────────────────────────────────────────────────────┘
    logger.info("  Tokens arriving in real-time:")
    full_response = ""
    token_count = 0

    for msg_chunk, metadata in graph.stream(
        input_msg, config, stream_mode="messages"
    ):
        # Only process AI message chunks (skip HumanMessage echoes)
        if isinstance(msg_chunk, AIMessageChunk) and msg_chunk.content:
            token = msg_chunk.content
            full_response += token
            token_count += 1
            # Show each token as it arrives — this is the "typing" effect
            logger.info("    Token %03d: '%s'", token_count, token)

    logger.info("")
    logger.info("  Full response: %s", full_response[:200])
    logger.info("  Total tokens streamed: %d", token_count)
    logger.info("")
    logger.info("  KEY INSIGHT: Each token arrives individually — 'Hello', ' Krish', '!'")
    logger.info("  This is how ChatGPT works: tokens stream to the UI one by one.")
    logger.info("  Use case: PRODUCTION CHAT UIs — the standard for real-time responses.")



# ═══════════════════════════════════════════════════════════════════════════════
# 9. DEMO 4: astream_events — Detailed Event Streaming (Advanced / Async)
# ═══════════════════════════════════════════════════════════════════════════════
#
# astream_events is the MOST DETAILED streaming mode. It yields individual
# events for EVERYTHING that happens inside the graph:
#
#   - on_chain_start / on_chain_end  → graph lifecycle events
#   - on_chat_model_start            → LLM call begins
#   - on_chat_model_stream           → individual TOKENS from the LLM
#   - on_chat_model_end              → LLM call finishes
#
# Each event has:
#   - event["event"]  → event type (e.g., "on_chat_model_stream")
#   - event["data"]   → event payload (e.g., {"chunk": AIMessageChunk(...)})
#   - event["name"]   → name of the component (e.g., "ChatGroq")
#   - event["tags"]   → tags for filtering
#   - event["run_id"] → unique run identifier
#   - event["metadata"] → includes thread_id, langgraph_step, langgraph_node
#
# IMPORTANT: astream_events is ASYNC — requires asyncio.run() in a .py file.
# This is the advanced approach. For most chat UIs, stream_mode="messages"
# (Demo 3) is simpler and sufficient.

def demo_astream_events() -> None:
    """astream_events — detailed event streaming with token-level granularity."""
    from typing import Annotated

    from langchain_core.messages import AnyMessage, HumanMessage
    from langchain_groq import ChatGroq
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages
    from typing_extensions import TypedDict

    # ── State ────────────────────────────────────────────────────────────
    class State(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]

    # ── LLM ──────────────────────────────────────────────────────────────
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_tokens=256)

    # ── Node ─────────────────────────────────────────────────────────────
    def chatbot(state: State) -> dict:
        """Call the LLM and return the response as a state update."""
        return {"messages": [llm.invoke(state["messages"])]}

    # ── Build graph: START → chatbot → END ───────────────────────────────
    builder = StateGraph(State)
    builder.add_node("chatbot", chatbot)
    builder.add_edge(START, "chatbot")
    builder.add_edge("chatbot", END)

    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)

    # ── Async function for astream_events ────────────────────────────────
    async def stream_events() -> None:
        """Use astream_events to get detailed event-level streaming."""
        config = {"configurable": {"thread_id": "stream-events-1"}}
        input_msg = {"messages": [HumanMessage(content="What is LangGraph in one sentence?")]}

        logger.info("=" * 70)
        logger.info("--- Demo 4: astream_events — Detailed Event Streaming ---")
        logger.info("=" * 70)

        # ┌─────────────────────────────────────────────────────────────────┐
        # │  astream_events yields EVERY event in the graph execution.     │
        # │  We filter for "on_chat_model_stream" to get individual tokens.│
        # │  version="v2" is required for the latest event format.         │
        # └─────────────────────────────────────────────────────────────────┘

        # First pass: show ALL event types (for understanding)
        logger.info("  All event types in this execution:")
        event_types_seen: set[str] = set()

        async for event in graph.astream_events(
            input_msg, config, version="v2"
        ):
            event_type = event["event"]
            if event_type not in event_types_seen:
                event_types_seen.add(event_type)
                logger.info("    Event type: '%s' (name: '%s')",
                            event_type, event.get("name", "N/A"))

        logger.info("  Total unique event types: %d", len(event_types_seen))
        logger.info("")

        # Second pass: filter for TOKEN events only (production pattern)
        logger.info("  Filtered token events (on_chat_model_stream):")
        config2 = {"configurable": {"thread_id": "stream-events-2"}}
        full_response = ""
        token_count = 0

        async for event in graph.astream_events(
            input_msg, config2, version="v2"
        ):
            # ── Filter: only "on_chat_model_stream" events have tokens ───
            if event["event"] == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                token = chunk.content
                if token:  # skip empty tokens
                    full_response += token
                    token_count += 1
                    logger.info("    Token %03d: '%s' (node: %s, step: %s)",
                                token_count,
                                token,
                                event.get("metadata", {}).get("langgraph_node", "?"),
                                event.get("metadata", {}).get("langgraph_step", "?"))

        logger.info("")
        logger.info("  Full response: %s", full_response[:200])
        logger.info("  Total tokens: %d", token_count)
        logger.info("")
        logger.info("  KEY INSIGHT: astream_events gives you EVERYTHING.")
        logger.info("  Filter by event['event'] == 'on_chat_model_stream' for tokens.")
        logger.info("  Each event has metadata: node name, step number, run_id.")
        logger.info("  Use case: ADVANCED monitoring, logging, custom UIs.")

    # ── Run the async function ───────────────────────────────────────────
    asyncio.run(stream_events())



# ═══════════════════════════════════════════════════════════════════════════════
# 10. DEMO 5: Streaming with ReAct Agent (Production Pattern)
# ═══════════════════════════════════════════════════════════════════════════════
#
# In production, you don't just stream a simple chatbot — you stream AGENTS.
# Agents call tools, loop back, and make decisions. Streaming lets you
# OBSERVE this behavior in real-time.
#
# With stream_mode="updates", you see:
#   Step 1: agent node → LLM decides to call a tool (tool_calls in AIMessage)
#   Step 2: tools node → tool executes, returns result (ToolMessage)
#   Step 3: agent node → LLM uses tool result to generate final answer
#
# This is how you MONITOR agent behavior in production:
#   - Did the agent call the right tool?
#   - What arguments did it pass?
#   - What did the tool return?
#   - Did the agent loop too many times?

def demo_stream_react_agent() -> None:
    """Streaming a ReAct agent — observe tool calls and results in real-time."""
    from typing import Annotated

    from langchain_community.tools import WikipediaQueryRun
    from langchain_community.utilities import WikipediaAPIWrapper
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

    # ── Tools ────────────────────────────────────────────────────────────
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

    # Wikipedia tool for knowledge queries
    wikipedia_tool = WikipediaQueryRun(
        api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
    )

    tools = [add, wikipedia_tool]

    # ── LLM with tools bound ────────────────────────────────────────────
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_tokens=256)
    llm_with_tools = llm.bind_tools(tools)

    # ── Node ─────────────────────────────────────────────────────────────
    def agent(state: State) -> dict:
        """The agent node: LLM reasons and decides whether to call tools."""
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    # ── Build ReAct graph ────────────────────────────────────────────────
    builder = StateGraph(State)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)  # agent → tools or END
    builder.add_edge("tools", "agent")  # tools → agent (ReAct loop)

    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)
    save_graph_image(graph, "lesson09_react_streaming")

    # ── Stream the agent with mode="updates" ─────────────────────────────
    logger.info("=" * 70)
    logger.info("--- Demo 5: Streaming ReAct Agent (Production Pattern) ---")
    logger.info("=" * 70)

    config = {"configurable": {"thread_id": "react-stream-1"}, "recursion_limit": 25}
    input_msg = {"messages": [HumanMessage(content="What is 15 plus 27?")]}

    logger.info("  Query: 'What is 15 plus 27?'")
    logger.info("  Streaming agent execution with mode='updates':")
    logger.info("")

    # ┌─────────────────────────────────────────────────────────────────────┐
    # │  With a ReAct agent, stream_mode="updates" shows:                  │
    # │    1. agent node: LLM decides to call add(15, 27)                  │
    # │    2. tools node: add tool returns 42                              │
    # │    3. agent node: LLM generates final answer using tool result     │
    # │  This is how you MONITOR agent behavior in production.             │
    # └─────────────────────────────────────────────────────────────────────┘
    step = 0
    for chunk in graph.stream(input_msg, config, stream_mode="updates"):
        step += 1
        for node_name, state_update in chunk.items():
            logger.info("  Step %d — Node: '%s'", step, node_name)
            if "messages" in state_update:
                for msg in state_update["messages"]:
                    msg_type = type(msg).__name__
                    logger.info("    Message type: %s", msg_type)

                    # Show tool calls if present (agent deciding to use a tool)
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            logger.info("    → TOOL CALL: %s(%s)",
                                        tc["name"], tc["args"])

                    # Show content if present
                    if msg.content:
                        logger.info("    Content: %s", str(msg.content)[:200])
        logger.info("")

    logger.info("  KEY INSIGHT: stream_mode='updates' on a ReAct agent shows:")
    logger.info("    1. LLM deciding which tool to call (and with what args)")
    logger.info("    2. Tool execution results")
    logger.info("    3. LLM's final answer using the tool result")
    logger.info("  Use case: PRODUCTION MONITORING — track agent decisions in real-time.")



# ═══════════════════════════════════════════════════════════════════════════════
# 11. COMPARISON TABLE — Streaming Modes
# ═══════════════════════════════════════════════════════════════════════════════
#
# ┌──────────────────────┬────────────────────────────────────────────────────────┐
# │ Mode                 │ Description                                            │
# ├──────────────────────┼────────────────────────────────────────────────────────┤
# │ stream_mode="values" │ Full state snapshot after EACH node.                   │
# │                      │ Chunk = {"messages": [...all messages...]}             │
# │                      │ Granularity: per NODE (coarse)                         │
# │                      │ Use case: UI rendering — show full conversation        │
# │                      │ Sync: graph.stream()                                   │
# ├──────────────────────┼────────────────────────────────────────────────────────┤
# │ stream_mode="updates"│ Only the CHANGES from each node.                       │
# │                      │ Chunk = {"node_name": {"messages": [new_msg]}}         │
# │                      │ Granularity: per NODE (medium)                         │
# │                      │ Use case: Debugging — what did each node produce?      │
# │                      │ Sync: graph.stream()                                   │
# ├──────────────────────┼────────────────────────────────────────────────────────┤
# │ stream_mode="messages│ Individual tokens as the LLM generates them.           │
# │                      │ Chunk = (AIMessageChunk, metadata)                     │
# │                      │ Granularity: per TOKEN (fine)                          │
# │                      │ Use case: Production chat UIs (ChatGPT-style)          │
# │                      │ Sync: graph.stream()                                   │
# │                      │ NOTE: Newer, simpler alternative to astream_events     │
# ├──────────────────────┼────────────────────────────────────────────────────────┤
# │ astream_events       │ ALL events: chain start/end, LLM start/stream/end.    │
# │                      │ Chunk = {"event": "on_chat_model_stream", "data": ...} │
# │                      │ Granularity: per EVENT (finest)                        │
# │                      │ Use case: Advanced monitoring, logging, custom UIs     │
# │                      │ Async ONLY: graph.astream_events(version="v2")         │
# │                      │ Requires asyncio.run() in .py files                    │
# └──────────────────────┴────────────────────────────────────────────────────────┘
#
# WHEN TO USE EACH:
#
# - Building a chat UI?                → stream_mode="messages" (Demo 3)
# - Debugging agent behavior?          → stream_mode="updates" (Demo 1)
# - Rendering full conversation state? → stream_mode="values" (Demo 2)
# - Advanced monitoring / logging?     → astream_events (Demo 4)
# - Monitoring ReAct agent in prod?    → stream_mode="updates" (Demo 5)


# ═══════════════════════════════════════════════════════════════════════════════
# 12. COMMON MISTAKES & GOTCHAS
# ═══════════════════════════════════════════════════════════════════════════════
#
# Mistake                                    | Fix
# ───────────────────────────────────────────|──────────────────────────────────────
# Using invoke() when you need streaming     | Use graph.stream() or graph.astream()
#   (user waits for full response)           | Streaming shows tokens in real-time.
# Forgetting stream_mode parameter           | Default is "values". Explicitly set
#   (getting full state when you want tokens)| stream_mode="messages" for tokens.
# Using astream_events without asyncio       | astream_events is ASYNC only.
#   (SyntaxError: async for outside func)    | Wrap in async def + asyncio.run().
# Not filtering astream_events               | Filter for "on_chat_model_stream"
#   (drowning in internal events)            | to get only LLM tokens.
# Forgetting version="v2" in astream_events  | Always pass version="v2" for the
#   (getting deprecated event format)        | latest event format.
# Confusing "updates" and "values"           | "updates" = diff (what changed)
#   (expecting full state, getting diff)     | "values" = snapshot (full state)
# Not using MemorySaver with streaming       | Streaming works with or without
#   (thinking memory is required)            | checkpointer. Memory is optional.
# Forgetting thread_id when using memory     | If using MemorySaver, always pass
#   (error: missing thread_id)              | config={"configurable": {"thread_id": "x"}}
#
# ANTI-PATTERN: Using graph.invoke() in a production chat UI.
#   invoke() blocks until the ENTIRE response is ready. The user stares at
#   a blank screen for seconds. Always use graph.stream() with
#   stream_mode="messages" for real-time token streaming in chat UIs.


# ═══════════════════════════════════════════════════════════════════════════════
# 13. WHY LANGGRAPH FOR THIS
# ═══════════════════════════════════════════════════════════════════════════════
#
# ✅ First-class streaming — multiple modes for different use cases
# ✅ Node-level visibility — see what each node produced (not just final output)
# ✅ Token-level streaming — stream_mode="messages" for production chat UIs
# ✅ Event-level detail — astream_events for advanced monitoring and logging
# ✅ Works with agents — stream tool calls, results, and LLM decisions
# ✅ Sync AND async — graph.stream() for sync, graph.astream() for async
# ✅ Compatible with memory — streaming works with MemorySaver, SqliteSaver, etc.
# ✅ Production-ready — used in real chat applications with millions of users


# ═══════════════════════════════════════════════════════════════════════════════
# 14. WHERE THIS CONNECTS (Concept Linking)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Lesson 03 (Chatbot)     → simple chatbot, first use of graph.stream()
# Lesson 07 (ReAct Agent)  → agent loop, tool calls — now we STREAM them
# Lesson 08 (Memory)       → MemorySaver + streaming = conversational streaming
# Lesson 09 (Streaming)    → deep dive into ALL streaming modes
# Next lessons             → Human-in-the-loop (streaming + interrupt_before)
#
# graph.invoke()       → blocks, returns final state (no streaming)
# graph.stream()       → yields chunks per node (sync, configurable mode)
# graph.astream()      → async version of stream()
# graph.astream_events → yields ALL events including tokens (async, advanced)
#
# stream_mode="values"   → full state snapshot → UI rendering
# stream_mode="updates"  → node-level diff → debugging
# stream_mode="messages" → token-level streaming → production chat UIs
# astream_events         → event-level detail → advanced monitoring
#
# StateGraph → compiles to → CompiledGraph → .stream() / .astream_events()
# Nodes → produce updates → streamed to client in real-time
# MemorySaver → optional → streaming works with or without memory


# ═══════════════════════════════════════════════════════════════════════════════
# INTERVIEW QUESTIONS
# ═══════════════════════════════════════════════════════════════════════════════
#
# THEORETICAL:
# Q: What are the different streaming modes in LangGraph, and when would you
#    use each one? Explain the difference between "values", "updates",
#    "messages", and astream_events.
#
# A: LangGraph provides four streaming approaches:
#    1. stream_mode="values" yields the FULL state after each node executes.
#       Use it for UI rendering where you need the complete conversation.
#    2. stream_mode="updates" yields only the CHANGES from each node.
#       Use it for debugging — see exactly what each node produced.
#    3. stream_mode="messages" yields individual tokens as the LLM generates
#       them. This is the production standard for chat UIs (ChatGPT-style).
#    4. astream_events yields detailed events (chain start/end, LLM tokens,
#       tool calls) with metadata. Use it for advanced monitoring and logging.
#    The key difference is granularity: values (coarse, full state) → updates
#    (medium, per-node diff) → messages (fine, per-token) → events (finest,
#    every internal event). For production chat UIs, use "messages". For
#    debugging agents, use "updates".
#
# HANDS-ON:
# Q: Build a LangGraph chatbot that streams tokens to the console in real-time
#    using stream_mode="messages". The chatbot should use MemorySaver for memory
#    and print each token as it arrives (simulating a ChatGPT-style UI).
#
# SOLUTION:
#   from typing import Annotated
#   from langchain_core.messages import AnyMessage, HumanMessage, AIMessageChunk
#   from langchain_groq import ChatGroq
#   from langgraph.checkpoint.memory import MemorySaver
#   from langgraph.graph import END, START, StateGraph
#   from langgraph.graph.message import add_messages
#   from typing_extensions import TypedDict
#
#   class State(TypedDict):
#       messages: Annotated[list[AnyMessage], add_messages]
#
#   llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_tokens=256)
#
#   def chatbot(state: State) -> dict:
#       return {"messages": [llm.invoke(state["messages"])]}
#
#   builder = StateGraph(State)
#   builder.add_node("chatbot", chatbot)
#   builder.add_edge(START, "chatbot")
#   builder.add_edge("chatbot", END)
#   graph = builder.compile(checkpointer=MemorySaver())
#
#   config = {"configurable": {"thread_id": "demo-1"}}
#   for msg_chunk, metadata in graph.stream(
#       {"messages": [HumanMessage(content="Tell me about Python")]},
#       config,
#       stream_mode="messages",
#   ):
#       if isinstance(msg_chunk, AIMessageChunk) and msg_chunk.content:
#           print(msg_chunk.content, end="", flush=True)
#   print()  # newline after streaming completes
#
# BONUS (System Design):
# Q: Design a streaming architecture for a production chat application that
#    serves 10,000 concurrent users. How would you handle token streaming,
#    tool call visibility, and error monitoring?
#
# A: Use LangGraph with stream_mode="messages" for token streaming to the
#    frontend via Server-Sent Events (SSE) or WebSockets. For tool call
#    visibility, run a parallel stream_mode="updates" consumer that logs
#    each node's output to a monitoring system (e.g., LangSmith). Use
#    PostgresSaver for persistent memory across servers. For error monitoring,
#    use astream_events to capture on_chain_end events and check for errors.
#    Deploy behind a load balancer with sticky sessions (same thread_id →
#    same server) or use a shared checkpointer (PostgreSQL) for stateless
#    scaling. Rate-limit streaming connections per user to prevent abuse.


# ═══════════════════════════════════════════════════════════════════════════════
# QUICK RECAP
# ═══════════════════════════════════════════════════════════════════════════════
#
# 1. Streaming lets you observe graph execution in real-time — from full state
#    snapshots (values) to per-node diffs (updates) to individual tokens
#    (messages) to every internal event (astream_events).
#
# 2. For production chat UIs, use stream_mode="messages" — it streams tokens
#    one by one, giving the ChatGPT-style "typing" effect. For debugging
#    agents, use stream_mode="updates" to see tool calls and decisions.
#
# 3. astream_events is the most powerful but also the most complex. It's async-
#    only and yields every event (chain start/end, LLM tokens, tool calls).
#    Use it for advanced monitoring. For most use cases, "messages" is enough.


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN — Run all demos
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logger.info("🌊 LangGraph Lesson 9 — Streaming Deep Dive")
    logger.info("=" * 70)

    demo_stream_updates()       # Demo 1: stream_mode="updates"
    demo_stream_values()        # Demo 2: stream_mode="values"
    demo_stream_messages()      # Demo 3: stream_mode="messages"
    demo_astream_events()       # Demo 4: astream_events (async)
    demo_stream_react_agent()   # Demo 5: ReAct agent streaming

    logger.info("")
    logger.info("=" * 70)
    logger.info("🎉 All streaming demos complete!")
    logger.info("=" * 70)
