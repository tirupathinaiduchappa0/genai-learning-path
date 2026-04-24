"""
🔁 LangGraph Lesson 7 — ReAct Agent: Reason + Act + Observe (Level 3)

═══════════════════════════════════════════════════════════════════
1. CONCEPT: ReAct Agent — ONE-LINE DEFINITION
═══════════════════════════════════════════════════════════════════

A ReAct Agent is a LOOPING graph where the LLM reasons about a problem,
acts by calling a tool, observes the result, and LOOPS BACK to reason again
— repeating until it has enough information to answer.

    ReAct = Reason + Act + Observe (the 3-step loop)

═══════════════════════════════════════════════════════════════════
2. WHY REACT EXISTS (and how it differs from Lesson 06)
═══════════════════════════════════════════════════════════════════

In Lesson 06, the graph was LINEAR:
    START → tool_calling_llm → (tools_condition) → tools → END

The LLM could call ONE tool, get the result, and that was it.
But what if the task requires MULTIPLE steps?

    "Add 5 plus 5 and then multiply by 10"
    Step 1: add(5, 5) = 10
    Step 2: multiply(10, 10) = 100  ← needs the RESULT of step 1!

The Lesson 06 graph CAN'T do this — after the tool runs, it goes to END.
There's no way to loop back and call another tool.

ReAct solves this with ONE critical edge:
    builder.add_edge("tools", "tool_calling_llm")  ← THE KEY LINE

This creates a LOOP:
    START → tool_calling_llm → tools → tool_calling_llm → tools → ... → END

The LLM keeps looping until it decides: "I have enough info to answer."
Then tools_condition routes to END (no more tool_calls).

═══════════════════════════════════════════════════════════════════
3. WHERE IT FITS IN THE LANGGRAPH ARCHITECTURE
═══════════════════════════════════════════════════════════════════

    State (messages with add_messages reducer)
        ↓
    LLM Node (calls the model, may produce tool_calls)
        ↓
    tools_condition (checks: did the LLM request a tool call?)
        ├── YES → ToolNode (executes the tool, returns result)
        │            ↓
        │         LOOP BACK → LLM Node (reason about the result!)
        │
        └── NO  → END (return the LLM's final answer)

    The LOOP is what makes it a ReAct agent.
    Without the loop, it's just a tool-calling chain (Lesson 06).

═══════════════════════════════════════════════════════════════════
4. REAL-WORLD ANALOGY
═══════════════════════════════════════════════════════════════════

Think of a DETECTIVE solving a case:

    1. REASON: "I need to find out who was at the scene."
    2. ACT:    Interview witness #1 → get a clue.
    3. OBSERVE: "Witness says they saw a red car."
    4. REASON: "I need to check the license plate database."
    5. ACT:    Search database → get owner info.
    6. OBSERVE: "Car belongs to John Doe."
    7. REASON: "I have enough info. John Doe is the suspect."
    8. → DONE (no more actions needed)

    Each step builds on the PREVIOUS result.
    The detective LOOPS until the case is solved.
    That's exactly how a ReAct agent works.

═══════════════════════════════════════════════════════════════════
5. ASCII GRAPH STRUCTURES
═══════════════════════════════════════════════════════════════════

Lesson 06 — Tool-Calling Chain (NO LOOP):

    [START]
       |
    [tool_calling_llm]
       |
       ├── (tool_calls?) ──→ [tools] ──→ [END]     ← STRAIGHT TO END
       |
       └── (no tool_calls) ──→ [END]

Lesson 07 — ReAct Agent (WITH LOOP):

    [START]
       |
    [tool_calling_llm]  ◄──────────────────┐
       |                                    |
       ├── (tool_calls?) ──→ [tools] ───────┘   ← LOOPS BACK!
       |
       └── (no tool_calls) ──→ [END]

    The ONLY difference is ONE edge: tools → tool_calling_llm
    But this ONE edge changes EVERYTHING — it enables multi-step reasoning.

═══════════════════════════════════════════════════════════════════

🔄 LANGCHAIN vs LANGGRAPH
LangChain : AgentExecutor runs the ReAct loop as a BLACK BOX.
            You can't see how many iterations happen or control the loop.
LangGraph  : You build the loop YOURSELF with one extra edge.
            tools → tool_calling_llm creates the cycle.
            You control recursion_limit, can add human-in-the-loop, etc.

LangChain : create_react_agent() → opaque, hard to customize.
LangGraph  : Explicit nodes + edges → full control over the agent loop.
            Every iteration is visible, debuggable, and customizable.

═══════════════════════════════════════════════════════════════════
AGENT TYPES OVERVIEW (Brief)
═══════════════════════════════════════════════════════════════════

- ReAct Agent: Reason → Act → Observe loop (THIS lesson)
- Plan-and-Execute: Plans ALL steps first, then executes sequentially
- Conversational Agent: ReAct + conversation memory (Lesson 08)
- Tool-Calling Agent: LLM outputs structured tool calls (foundation of ReAct)
- Multi-Agent: Multiple agents collaborating (supervisor, subgraphs)
- Reflexion Agent: Self-reflects on output quality, retries if needed

HOW TO RUN:
    $ python src/LangGraph/07_react_agent.py

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
# 6. DEMO 1: ReAct Agent with Math Tools (The LOOPING Graph)
# ═══════════════════════════════════════════════════════════════════════════════
#
# This is the CORE of this lesson. We build a ReAct agent with math tools
# (add, multiply, divide) and test it with a MULTI-STEP question:
#   "What is 5 plus 5 and then multiply by 3?"
#
# The LLM must:
#   1. Call add(5, 5) → get 10
#   2. LOOP BACK, reason about the result
#   3. Call multiply(10, 3) → get 30
#   4. LOOP BACK, reason — no more tools needed
#   5. Return the final answer: 30
#
# THE KEY DIFFERENCE FROM LESSON 06:
#   Lesson 06: builder.add_edge("tools", END)              ← goes to END
#   Lesson 07: builder.add_edge("tools", "tool_calling_llm")  ← LOOPS BACK!
#
# This ONE edge is what makes it a ReAct agent.
#
# Message flow for multi-step:
#   Human("add 5+5 then multiply by 3")
#   → AI(tool_call: add(5,5))
#   → Tool(10)
#   → AI(tool_call: multiply(10,3))   ← LLM reasons and calls another tool!
#   → Tool(30)
#   → AI("The answer is 30")          ← No more tool_calls → END

def demo_react_math_tools() -> None:
    """ReAct agent with math tools: add, multiply, divide. Multi-step reasoning."""
    from typing import Annotated

    from langchain_core.messages import AnyMessage, HumanMessage
    from langchain_core.tools import tool
    from langchain_groq import ChatGroq
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import ToolNode, tools_condition
    from typing_extensions import TypedDict

    # ── State: messages with add_messages reducer ────────────────────────
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
        """Multiply two integers together. Use this when the user asks to multiply numbers.

        Args:
            a: first integer
            b: second integer

        Returns:
            The product of a and b
        """
        return a * b

    # ── Tool 3: divide ───────────────────────────────────────────────────
    @tool
    def divide(a: int, b: int) -> float:
        """Divide two integers. Use this when the user asks to divide numbers.

        Args:
            a: numerator (the number being divided)
            b: denominator (the number to divide by)

        Returns:
            The result of a divided by b
        """
        if b == 0:
            return float("inf")  # Handle division by zero gracefully
        return a / b

    tools = [add, multiply, divide]

    # ── LLM with tools bound ────────────────────────────────────────────
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_tokens=512)
    llm_with_tools = llm.bind_tools(tools)

    # ── Node: calls the LLM (which may produce tool_calls) ──────────────
    def tool_calling_llm(state: State) -> dict:
        """Call the LLM with tools bound. It decides: answer or call a tool."""
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    # ── Build the LOOPING ReAct graph ────────────────────────────────────
    builder = StateGraph(State)
    builder.add_node("tool_calling_llm", tool_calling_llm)  # LLM node
    builder.add_node("tools", ToolNode(tools))               # Tool executor node

    builder.add_edge(START, "tool_calling_llm")              # Entry
    builder.add_conditional_edges(
        "tool_calling_llm",
        tools_condition,  # Router: tool_calls? → "tools" | no tool_calls? → END
    )
    # ┌─────────────────────────────────────────────────────────────────────┐
    # │  THIS IS THE KEY LINE — THE LOOP-BACK EDGE!                        │
    # │  In Lesson 06: builder.add_edge("tools", END)                      │
    # │  In Lesson 07: builder.add_edge("tools", "tool_calling_llm")       │
    # │  This ONE edge creates the ReAct loop.                             │
    # └─────────────────────────────────────────────────────────────────────┘
    builder.add_edge("tools", "tool_calling_llm")            # LOOP BACK!

    graph = builder.compile()
    save_graph_image(graph, "lesson07_react_math_tools")

    # ── Test: Multi-step math question ───────────────────────────────────
    logger.info("--- Demo 1: ReAct Agent with Math Tools ---")
    logger.info("  Q: What is 5 plus 5 and then multiply by 3?")

    result = graph.invoke(
        {"messages": [HumanMessage(content="What is 5 plus 5 and then multiply by 3?")]},
        config={"recursion_limit": 25},  # Safety guard against infinite loops
    )

    # ── Show the full message flow ───────────────────────────────────────
    logger.info("  Full message flow:")
    for i, msg in enumerate(result["messages"]):
        msg_type = type(msg).__name__
        content = str(msg.content)[:200]
        # Show tool_calls if present
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            tool_names = [tc["name"] for tc in msg.tool_calls]
            logger.info("    [%d] %s: %s (tool_calls: %s)", i, msg_type, content, tool_names)
        else:
            logger.info("    [%d] %s: %s", i, msg_type, content)

    logger.info("  Final answer: %s", result["messages"][-1].content[:200])
    logger.info("  The loop continues until the LLM has no more tool_calls.")
    logger.info("  Flow: START → LLM → tools → LLM → tools → LLM → END")

    # ── Comparison with Lesson 06 ────────────────────────────────────────
    logger.info("")
    logger.info("  ┌─────────────────────────────────────────────────────┐")
    logger.info("  │  LESSON 06 vs LESSON 07 — THE CRITICAL DIFFERENCE  │")
    logger.info("  │                                                     │")
    logger.info("  │  Lesson 06: tools → END (one tool call, done)       │")
    logger.info("  │  Lesson 07: tools → tool_calling_llm (LOOP BACK!)   │")
    logger.info("  │                                                     │")
    logger.info("  │  This ONE edge enables multi-step reasoning.        │")
    logger.info("  └─────────────────────────────────────────────────────┘")



# ═══════════════════════════════════════════════════════════════════════════════
# 7. DEMO 2: ReAct Agent with Multiple Tool Types (Math + Wikipedia + Search)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Same LOOPING graph, but with DIVERSE tools:
#   - add (custom math tool)
#   - Wikipedia (knowledge lookup)
#   - DuckDuckGo (web search)
#
# The LLM can chain DIFFERENT tool types in one query:
#   "What is machine learning and what is 10 plus 20?"
#   → wikipedia("machine learning") → add(10, 20) → final answer
#
# This shows the POWER of ReAct: the agent picks the right tool for each
# sub-task and chains them together automatically.

def demo_react_multi_tools() -> None:
    """ReAct agent with diverse tools: add + Wikipedia + DuckDuckGo."""
    from typing import Annotated

    from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
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

    # ── Tool 1: Custom add tool ──────────────────────────────────────────
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

    # ── Tool 2: Wikipedia search ─────────────────────────────────────────
    wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
    wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

    # ── Tool 3: DuckDuckGo web search ────────────────────────────────────
    ddg_search = DuckDuckGoSearchRun()

    # ── Combine all tools ────────────────────────────────────────────────
    tools = [add, wiki, ddg_search]

    # ── LLM with ALL tools bound ─────────────────────────────────────────
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_tokens=512)
    llm_with_tools = llm.bind_tools(tools)

    # ── Node: LLM decides which tool (if any) to call ───────────────────
    def tool_calling_llm(state: State) -> dict:
        """Call the LLM with all tools bound. It picks the right tool."""
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    # ── Build the LOOPING ReAct graph (same pattern as Demo 1) ───────────
    builder = StateGraph(State)
    builder.add_node("tool_calling_llm", tool_calling_llm)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "tool_calling_llm")
    builder.add_conditional_edges(
        "tool_calling_llm",
        tools_condition,
    )
    builder.add_edge("tools", "tool_calling_llm")  # LOOP BACK — ReAct!

    graph = builder.compile()
    save_graph_image(graph, "lesson07_react_multi_tools")

    # ── Test: Multi-tool query (knowledge + math in one question) ────────
    logger.info("--- Demo 2: ReAct Agent with Multiple Tool Types ---")
    logger.info("  Q: What is machine learning and what is 10 plus 20?")

    result = graph.invoke(
        {"messages": [HumanMessage(content="What is machine learning and what is 10 plus 20?")]},
        config={"recursion_limit": 25},
    )

    # ── Show which tools were called ─────────────────────────────────────
    tools_called = []
    logger.info("  Full message flow:")
    for i, msg in enumerate(result["messages"]):
        msg_type = type(msg).__name__
        content = str(msg.content)[:200]
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tools_called.append(tc["name"])
            tool_names = [tc["name"] for tc in msg.tool_calls]
            logger.info("    [%d] %s: %s (tool_calls: %s)", i, msg_type, content, tool_names)
        else:
            logger.info("    [%d] %s: %s", i, msg_type, content)

    logger.info("  Tools called in this query: %s", tools_called)
    logger.info("  Final answer: %s", result["messages"][-1].content[:300])
    logger.info("  The ReAct loop let the LLM use DIFFERENT tools for each sub-task.")



# ═══════════════════════════════════════════════════════════════════════════════
# 🎯 INTERVIEW QUESTIONS
# ═══════════════════════════════════════════════════════════════════════════════
#
# THEORETICAL:
# Q: What is the difference between a tool-calling agent (Lesson 06) and a
#    ReAct agent (Lesson 07) in LangGraph? What makes ReAct "agentic"?
#
# A: A tool-calling agent has a LINEAR graph: the LLM calls one tool, gets the
#    result, and the graph ends (tools -> END). A ReAct agent has a CYCLIC graph:
#    after the tool executes, the result goes BACK to the LLM (tools ->
#    tool_calling_llm). This loop lets the LLM reason about the tool result and
#    decide whether to call another tool or answer directly. The loop is what
#    makes it "agentic" -- the agent autonomously decides how many steps to take,
#    which tools to use, and when to stop. This enables multi-step reasoning
#    where each step builds on the previous result.
#
# HANDS-ON:
# Q: Build a ReAct agent with two tools:
#    - subtract(a, b) -> returns a - b
#    - square(n) -> returns n * n
#    Test with "Subtract 3 from 10 and then square the result."
#    Expected: subtract(10, 3) = 7, then square(7) = 49.
#
# SOLUTION:
#   from typing import Annotated
#   from langchain_core.messages import AnyMessage, HumanMessage
#   from langchain_core.tools import tool
#   from langchain_groq import ChatGroq
#   from langgraph.graph import END, START, StateGraph
#   from langgraph.graph.message import add_messages
#   from langgraph.prebuilt import ToolNode, tools_condition
#   from typing_extensions import TypedDict
#
#   class State(TypedDict):
#       messages: Annotated[list[AnyMessage], add_messages]
#
#   @tool
#   def subtract(a: int, b: int) -> int:
#       """Subtract b from a. Use when the user asks to subtract numbers."""
#       return a - b
#
#   @tool
#   def square(n: int) -> int:
#       """Square a number. Use when the user asks to square a number."""
#       return n * n
#
#   tools = [subtract, square]
#   llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_tokens=512)
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
#   builder.add_edge("tools", "tool_calling_llm")  # LOOP BACK! (ReAct)
#   graph = builder.compile()
#
#   result = graph.invoke(
#       {"messages": [HumanMessage(content="Subtract 3 from 10 and then square the result.")]},
#       config={"recursion_limit": 25},
#   )
#   print(result["messages"][-1].content)  # Expected: 49
#
# BONUS (System Design):
# Q: Design a ReAct agent for a customer support system that can:
#    - Look up order status (database tool)
#    - Search FAQ documents (RAG tool)
#    - Escalate to human agent (escalation tool)
#    How would you structure the graph? What safety guards would you add?
#
# A: Use a ReAct loop with 3 tools bound to the LLM. The graph structure is:
#    START -> tool_calling_llm -> (tools_condition) -> tools -> tool_calling_llm
#    Safety guards: recursion_limit=10 (prevent infinite loops), timeout per tool,
#    error handling in each tool (return friendly error messages, not exceptions),
#    and a "max_escalations" counter in state to prevent repeated escalations.
#    The escalation tool should set a flag in state that a custom router checks
#    to break out of the loop and route to a human handoff node.


# ═══════════════════════════════════════════════════════════════════════════════
# QUICK RECAP
# ═══════════════════════════════════════════════════════════════════════════════
#
# 1. ReAct = Reason + Act + Observe. The LLM calls a tool, observes the result,
#    and LOOPS BACK to reason again. The loop continues until no more tool_calls.
#
# 2. The ONLY difference from Lesson 06 is ONE edge:
#    Lesson 06: builder.add_edge("tools", END)
#    Lesson 07: builder.add_edge("tools", "tool_calling_llm")
#    This one edge enables multi-step reasoning and chained tool calls.
#
# 3. Always set recursion_limit on cyclic graphs to prevent infinite loops.
#    The ReAct pattern is the foundation for all advanced agent types:
#    Conversational (+ memory), Multi-Agent (+ subgraphs), Reflexion (+ critic).


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("ReAct LANGGRAPH LESSON 7 -- ReAct Agent: Reason + Act + Observe")
    logger.info("   The LOOPING agent that chains multiple tool calls")
    logger.info("=" * 70)

    logger.info("\n Demo 1: ReAct Agent with Math Tools (The LOOPING Graph)")
    demo_react_math_tools()

    logger.info("\n Demo 2: ReAct Agent with Multiple Tool Types")
    demo_react_multi_tools()

    logger.info("\n Demo 3: Agent Types Overview")
    logger.info("  (See comments in source code for the full comparison table)")

    logger.info("\n" + "=" * 70)
    logger.info("Lesson 7 complete -- ReAct Agent: Reason + Act + Observe")
    logger.info("Key: ONE edge (tools -> tool_calling_llm) creates the ReAct loop")
    logger.info("Pattern: LLM -> tools -> LLM -> tools -> ... -> END")
    logger.info("Safety: Always set recursion_limit on cyclic graphs")
    logger.info("Next: Lesson 8 -- Memory & Checkpointing (MemorySaver)")
    logger.info("=" * 70)
