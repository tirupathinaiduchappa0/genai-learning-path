"""
🔬 LangSmith Lesson 2 — Hands-On: Agent with LangSmith Tracing

═══════════════════════════════════════════════════════════════════
WHAT THIS LESSON COVERS
═══════════════════════════════════════════════════════════════════

This is a FULLY RUNNABLE lesson that creates agents with LangSmith
tracing enabled. After running, go to https://smith.langchain.com
and see the traces in the "langgraph-agent-tracing-demo" project.

DEMOS:
    1. Simple Chatbot with LangSmith Tracing (run_name, tags, metadata)
    2. ReAct Agent with Multiple Tools + LangSmith (multi-step tracing)
    3. @traceable decorator for custom Python functions
    4. langgraph.json configuration (conceptual — for deployment)

IMPORTANT: Uses GROQ (not OpenAI) since we have the Groq API key.
Model: llama-3.1-8b-instant (fast, free tier available)

HOW TO RUN:
    $ python src/LangSmith/02_langsmith_hands_on.py

Author: GenAI Learner
Date: 2025-07-15
"""

import logging
import os

from dotenv import load_dotenv

# ── Load environment variables ───────────────────────────────────────────
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
# Use a CLEAN project name for this lesson's traces
os.environ["LANGCHAIN_PROJECT"] = "langgraph-agent-tracing-demo"

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
# DEMO 1: Simple Chatbot with LangSmith Tracing
# ═══════════════════════════════════════════════════════════════════════════════
#
# Graph:
#     [START] → [chatbot] → [END]
#
# This is the simplest possible graph — one LLM node.
# The KEY here is the config we pass: run_name, tags, metadata.
# These make the trace SEARCHABLE and FILTERABLE in LangSmith UI.
#
# WHAT YOU'LL SEE IN LANGSMITH UI:
#     - Run name: "simple-chatbot-trace"
#     - Tags: ["demo", "chatbot", "lesson-02"]
#     - Metadata: {"user_id": "krish", "lesson": "02"}
#     - Trace tree: START → chatbot (LLM call with tokens, latency) → END
#     - Token usage: input tokens, output tokens, total
#     - Latency: time for the LLM call

def demo_simple_chatbot_with_tracing() -> None:
    """Simple chatbot graph with LangSmith tracing config."""
    import operator
    from typing import Annotated, TypedDict

    from langchain_core.messages import HumanMessage
    from langchain_groq import ChatGroq
    from langgraph.graph import END, START, StateGraph

    # ── State ────────────────────────────────────────────────────────────
    class ChatState(TypedDict):
        messages: Annotated[list, operator.add]

    # ── LLM ──────────────────────────────────────────────────────────────
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_tokens=256)

    # ── Node ─────────────────────────────────────────────────────────────
    def chatbot_node(state: ChatState) -> dict:
        """Chatbot node — sends messages to LLM, returns response."""
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    # ── Build graph ──────────────────────────────────────────────────────
    builder = StateGraph(ChatState)
    builder.add_node("chatbot", chatbot_node)
    builder.add_edge(START, "chatbot")
    builder.add_edge("chatbot", END)
    graph = builder.compile()

    # ── Invoke WITH LangSmith config ─────────────────────────────────────
    # This is the KEY part — the config dict controls what LangSmith records
    langsmith_config = {
        "run_name": "simple-chatbot-trace",
        "tags": ["demo", "chatbot", "lesson-02"],
        "metadata": {"user_id": "krish", "lesson": "02"},
    }

    result = graph.invoke(
        {"messages": [HumanMessage(content="Explain LangSmith in 2 sentences.")]},
        config=langsmith_config,
    )

    logger.info("--- Demo 1: Simple Chatbot with LangSmith Tracing ---")
    logger.info("  Input:  'Explain LangSmith in 2 sentences.'")
    logger.info("  Output: %s", result["messages"][-1].content[:200])
    logger.info("  Messages in state: %d", len(result["messages"]))
    logger.info("")
    logger.info("  ✅ Check LangSmith UI at https://smith.langchain.com")
    logger.info("  Project: 'langgraph-agent-tracing-demo'")
    logger.info("  Look for run: 'simple-chatbot-trace'")
    logger.info("")
    logger.info("  WHAT YOU'LL SEE IN THE UI:")
    logger.info("    - Trace tree: chatbot node with LLM call inside")
    logger.info("    - Token usage: input tokens + output tokens")
    logger.info("    - Latency: total time for the LLM call")
    logger.info("    - Tags: ['demo', 'chatbot', 'lesson-02']")
    logger.info("    - Metadata: {'user_id': 'krish', 'lesson': '02'}")


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO 2: ReAct Agent with Multiple Tools + LangSmith
# ═══════════════════════════════════════════════════════════════════════════════
#
# Graph (ReAct loop):
#     [START] → [tool_calling_llm] ←─────────────┐
#                    |                             |
#                    ├── (tool_calls?) → [tools] ──┘  LOOPS BACK!
#                    |
#                    └── (no tool_calls) → [END]
#
# Tools: add, multiply, wikipedia
# Config: run_name, tags, metadata for LangSmith
#
# WHAT YOU'LL SEE IN LANGSMITH UI:
#     Run: "multi-tool-agent"
#     Trace tree:
#       ├── tool_calling_llm (LLM decides to call add)
#       ├── tools: add(5, 5) → 10
#       ├── tool_calling_llm (LLM sees 10, decides to call multiply)
#       ├── tools: multiply(10, 3) → 30
#       └── tool_calling_llm (LLM sees 30, returns final answer)
#     Each step shows: tokens used, latency, inputs/outputs

def demo_react_agent_with_tracing() -> None:
    """ReAct agent with add, multiply, Wikipedia tools — traced in LangSmith."""
    from typing import Annotated

    from langchain_community.tools import WikipediaQueryRun
    from langchain_community.utilities import WikipediaAPIWrapper
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

    # ── Tool 1: add ──────────────────────────────────────────────────────
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

    # ── Tool 2: multiply ─────────────────────────────────────────────────
    @tool
    def multiply(a: int, b: int) -> int:
        """Multiply two integers together. Use this when the user asks to multiply.

        Args:
            a: first integer
            b: second integer

        Returns:
            The product of a and b
        """
        return a * b

    # ── Tool 3: Wikipedia ────────────────────────────────────────────────
    wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
    wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

    tools = [add, multiply, wiki]

    # ── LLM with tools bound ────────────────────────────────────────────
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_tokens=512)
    llm_with_tools = llm.bind_tools(tools)

    # ── Node: LLM decides which tool to call ─────────────────────────────
    def tool_calling_llm(state: State) -> dict:
        """Call the LLM with tools bound. It decides: answer or call a tool."""
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    # ── Build the ReAct graph ────────────────────────────────────────────
    builder = StateGraph(State)
    builder.add_node("tool_calling_llm", tool_calling_llm)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "tool_calling_llm")
    builder.add_conditional_edges("tool_calling_llm", tools_condition)
    builder.add_edge("tools", "tool_calling_llm")  # LOOP BACK — ReAct!

    graph = builder.compile()
    save_graph_image(graph, "lesson_langsmith_react_agent")

    # ── Invoke with LangSmith config ─────────────────────────────────────
    langsmith_config = {
        "run_name": "multi-tool-agent",
        "tags": ["demo", "react-agent"],
        "metadata": {"user_id": "krish"},
        "recursion_limit": 25,
    }

    logger.info("--- Demo 2: ReAct Agent with Multiple Tools + LangSmith ---")
    logger.info("  Q: What is 5 plus 5 and then multiply by 3?")

    result = graph.invoke(
        {"messages": [HumanMessage(content="What is 5 plus 5 and then multiply by 3?")]},
        config=langsmith_config,
    )

    # ── Show the full message flow ───────────────────────────────────────
    logger.info("  Full message flow:")
    for i, msg in enumerate(result["messages"]):
        msg_type = type(msg).__name__
        content = str(msg.content)[:200]
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            tool_names = [tc["name"] for tc in msg.tool_calls]
            logger.info("    [%d] %s: %s (tool_calls: %s)", i, msg_type, content, tool_names)
        else:
            logger.info("    [%d] %s: %s", i, msg_type, content)

    logger.info("  Final answer: %s", result["messages"][-1].content[:200])
    logger.info("")
    logger.info("  ✅ Check LangSmith UI at https://smith.langchain.com")
    logger.info("  Project: 'langgraph-agent-tracing-demo'")
    logger.info("  Look for run: 'multi-tool-agent'")
    logger.info("")
    logger.info("  IN LANGSMITH UI, YOU'LL SEE:")
    logger.info("    LLM call → tool call (add) → LLM call → tool call (multiply) → LLM final answer")
    logger.info("    Each step shows: input messages, output, tokens used, latency")
    logger.info("    Tags: ['demo', 'react-agent'] — use these to filter traces")


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO 3: Using @traceable for Custom Functions
# ═══════════════════════════════════════════════════════════════════════════════
#
# LangSmith automatically traces LangChain/LangGraph components.
# But what about YOUR custom Python functions?
#
# Use @traceable from langsmith to wrap any function.
# It will appear in the LangSmith trace tree alongside LLM calls.
#
# This is useful for:
#   - Custom data processing steps
#   - Business logic between LLM calls
#   - External API calls you want to monitor
#   - Any function where you want visibility into inputs/outputs/latency
#
# WHAT YOU'LL SEE IN LANGSMITH UI:
#     Trace tree:
#       ├── custom_processing_step (your function!)
#       │   Input: {"data": "LangSmith is amazing"}
#       │   Output: "LANGSMITH IS AMAZING (processed, 23 chars)"
#       │   Latency: 0ms
#       └── format_for_display (another custom function!)
#           Input: {"processed_text": "LANGSMITH IS AMAZING..."}
#           Output: "=== LANGSMITH IS AMAZING (PROCESSED, 23 CHARS) ==="

def demo_traceable_decorator() -> None:
    """Show how @traceable wraps custom functions for LangSmith tracing."""
    from langsmith import traceable

    # ── Custom function 1: data processing ───────────────────────────────
    @traceable(name="custom_processing_step")
    def process_data(data: str) -> str:
        """A custom processing step that appears in LangSmith traces.

        Args:
            data: raw input string to process

        Returns:
            Processed string with metadata
        """
        processed = data.upper()
        return f"{processed} (processed, {len(data)} chars)"

    # ── Custom function 2: formatting ────────────────────────────────────
    @traceable(name="format_for_display")
    def format_for_display(processed_text: str) -> str:
        """Format processed text for display — also traced in LangSmith.

        Args:
            processed_text: the processed string from process_data

        Returns:
            Formatted string ready for display
        """
        return f"=== {processed_text} ==="

    # ── Run the custom pipeline ──────────────────────────────────────────
    logger.info("--- Demo 3: @traceable for Custom Functions ---")

    raw_data = "LangSmith is amazing for debugging agents"
    step1 = process_data(raw_data)
    step2 = format_for_display(step1)

    logger.info("  Raw input:  '%s'", raw_data)
    logger.info("  After process_data:     '%s'", step1)
    logger.info("  After format_for_display: '%s'", step2)
    logger.info("")
    logger.info("  ✅ Both functions appear in LangSmith traces!")
    logger.info("  @traceable(name='custom_processing_step') → visible in trace tree")
    logger.info("  @traceable(name='format_for_display')      → visible in trace tree")
    logger.info("")
    logger.info("  USE CASES for @traceable:")
    logger.info("    - Custom RAG preprocessing (cleaning, chunking)")
    logger.info("    - Business logic between LLM calls")
    logger.info("    - External API calls (database, search, etc.)")
    logger.info("    - Any function where you want input/output/latency visibility")


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO 4: langgraph.json Configuration (Conceptual)
# ═══════════════════════════════════════════════════════════════════════════════
#
# langgraph.json is the configuration file for deploying LangGraph agents
# as services using the LangGraph Platform (LangGraph Cloud).
#
# This is NOT needed for local tracing — LangSmith works with just env vars.
# langgraph.json is for DEPLOYMENT: turning your agent into a hosted API.
#
# FORMAT (from our project's langgraph.json):
#     {
#         "dependencies": ["."],
#         "graphs": {
#             "openai_agent": "./openai_agent.py:agent"
#         },
#         "env": "../.env"
#     }
#
# FIELDS:
#     dependencies: Python packages/paths to install
#     graphs: mapping of graph_name → module_path:variable_name
#             "openai_agent": "./openai_agent.py:agent"
#             means: import `agent` from `./openai_agent.py`
#     env: path to .env file for environment variables
#
# DEPLOYMENT FLOW:
#     1. Write your agent (e.g., openai_agent.py with `agent = graph.compile()`)
#     2. Create langgraph.json pointing to your agent
#     3. Deploy with: langgraph deploy (or push to LangGraph Cloud)
#     4. Your agent is now a REST API with:
#        - POST /runs/stream — stream agent responses
#        - POST /runs/wait — synchronous invocation
#        - GET /runs — list all runs
#        - Built-in LangSmith tracing for every request
#
# WHY THIS MATTERS:
#     Local development: python my_agent.py (traces go to LangSmith)
#     Production deployment: langgraph.json → LangGraph Platform → REST API
#     Both send traces to LangSmith automatically.

def demo_langgraph_json_explanation() -> None:
    """Explain the langgraph.json configuration format."""
    logger.info("--- Demo 4: langgraph.json Configuration (Conceptual) ---")
    logger.info("")
    logger.info("  langgraph.json is for DEPLOYING agents as hosted services.")
    logger.info("  It is NOT needed for local LangSmith tracing.")
    logger.info("")
    logger.info("  Our project's langgraph.json:")
    logger.info('    {')
    logger.info('      "dependencies": ["."],')
    logger.info('      "graphs": {')
    logger.info('        "openai_agent": "./openai_agent.py:agent"')
    logger.info('      },')
    logger.info('      "env": "../.env"')
    logger.info('    }')
    logger.info("")
    logger.info("  This tells LangGraph Platform:")
    logger.info("    - Install dependencies from current directory")
    logger.info("    - Expose the `agent` variable from openai_agent.py as a graph")
    logger.info("    - Load env vars from ../.env (includes LANGSMITH_API_KEY)")
    logger.info("")
    logger.info("  DEPLOYMENT: langgraph deploy → your agent becomes a REST API")
    logger.info("  TRACING: Both local and deployed agents send traces to LangSmith")


# ═══════════════════════════════════════════════════════════════════════════════
# 🎯 INTERVIEW QUESTIONS
# ═══════════════════════════════════════════════════════════════════════════════
#
# THEORETICAL:
# Q: How does LangSmith tracing work with LangGraph? What config options
#    should you always set for production traces?
#
# A: LangSmith tracing is AUTOMATIC when LANGCHAIN_TRACING_V2=true and
#    LANGSMITH_API_KEY are set as env vars. Every graph.invoke() or
#    graph.stream() call sends trace data to LangSmith asynchronously
#    (non-blocking). For production, always set:
#    - run_name: makes traces searchable ("RAG Query", "Support Agent")
#    - tags: groups related runs (["production", "v2"])
#    - metadata: enables user-level filtering ({"user_id": "u123"})
#    These go in the config dict: graph.invoke(input, config={...})
#
# HANDS-ON:
# Q: Build a ReAct agent with a "subtract" tool and a "square" tool.
#    Configure it to send traces to LangSmith with:
#    - run_name: "math-agent"
#    - tags: ["test", "math"]
#    - metadata: {"user_id": "interviewer"}
#    Test with: "Subtract 3 from 10 and square the result."
#    Then go to LangSmith UI and describe what you see in the trace.
#
# A: The trace tree would show:
#    ├── tool_calling_llm (LLM decides to call subtract)
#    ├── tools: subtract(10, 3) → 7
#    ├── tool_calling_llm (LLM sees 7, decides to call square)
#    ├── tools: square(7) → 49
#    └── tool_calling_llm (LLM returns "The answer is 49")
#    Each step shows tokens, latency, and full input/output.
#
# BONUS:
# Q: What is the @traceable decorator and when would you use it?
# A: @traceable from langsmith wraps custom Python functions so they
#    appear in the LangSmith trace tree alongside LLM/tool calls.
#    Use it for: custom preprocessing, business logic, external API calls,
#    or any function where you want input/output/latency visibility.
#    Example: @traceable(name="custom_rag_preprocessing")


# ═══════════════════════════════════════════════════════════════════════════════
# 📝 QUICK RECAP
# ═══════════════════════════════════════════════════════════════════════════════
#
# 1. LangSmith tracing is AUTOMATIC — set env vars, and every LangChain/LangGraph
#    call is traced. No code changes needed for basic tracing.
#
# 2. For PRODUCTION tracing, always pass config with run_name, tags, and metadata:
#    graph.invoke(input, config={"run_name": "...", "tags": [...], "metadata": {...}})
#
# 3. Use @traceable from langsmith to trace CUSTOM Python functions.
#    They appear in the trace tree alongside LLM and tool calls.
#
# 4. langgraph.json is for DEPLOYMENT (LangGraph Platform), not for tracing.
#    Both local and deployed agents send traces to LangSmith automatically.
#
# 5. After running this file, go to https://smith.langchain.com →
#    project "langgraph-agent-tracing-demo" → see all traces with full
#    execution trees, token usage, latency, and metadata.


# ═══════════════════════════════════════════════════════════════════════════════
# 🚀 Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("🔬 LANGSMITH LESSON 2 — Hands-On: Agent with LangSmith Tracing")
    logger.info("   All traces go to project: 'langgraph-agent-tracing-demo'")
    logger.info("=" * 70)

    logger.info("\n🔹 Demo 1: Simple Chatbot with LangSmith Tracing")
    demo_simple_chatbot_with_tracing()

    logger.info("\n🔹 Demo 2: ReAct Agent with Multiple Tools + LangSmith")
    demo_react_agent_with_tracing()

    logger.info("\n🔹 Demo 3: @traceable for Custom Functions")
    demo_traceable_decorator()

    logger.info("\n🔹 Demo 4: langgraph.json Configuration (Conceptual)")
    demo_langgraph_json_explanation()

    logger.info("\n" + "=" * 70)
    logger.info("✅ Lesson 2 complete — All traces sent to LangSmith!")
    logger.info("Go to: https://smith.langchain.com")
    logger.info("Project: 'langgraph-agent-tracing-demo'")
    logger.info("You should see traces for: 'simple-chatbot-trace' and 'multi-tool-agent'")
    logger.info("Next: Build your own agent and monitor it with LangSmith!")
    logger.info("=" * 70)
