"""
🧠 LangGraph Lesson 1 — Why LangGraph? (Foundation Level 1)

═══════════════════════════════════════════════════════════════════
1. CONCEPT NAME + ONE-LINE DEFINITION
═══════════════════════════════════════════════════════════════════

LangGraph is a library for building stateful, multi-actor applications
with LLMs using a graph-based architecture of Nodes, Edges, and State.

LangGraph = Lang (Language Models) + Graph (Directed Graph Structure)

═══════════════════════════════════════════════════════════════════
2. WHY IT EXISTS (The Problem It Solves)
═══════════════════════════════════════════════════════════════════

LangChain LCEL chains are SEQUENTIAL:
    prompt → llm → parser → done

This works for simple tasks. But real-world AI agents need:

    CYCLES:     LLM calls a tool → reads result → calls another tool → loops
    STATE:      Shared memory that ALL steps can read and write
    BRANCHING:  "If the user is angry, route to human. Else, auto-respond."
    PERSISTENCE: Save conversation state, resume later
    HUMAN CONTROL: Pause before dangerous actions, get approval

LangChain LCEL CANNOT do cycles. Once data flows through the pipe,
it's done. There's no way to loop back.

    prompt | llm | parser   ← data flows LEFT to RIGHT, never loops back

LangGraph solves this by replacing the linear pipe with a GRAPH:

    [START] → [Agent] → [Tool] → [Agent] → [END]
                  ↑                   |
                  └───────────────────┘   ← CYCLE! Agent can loop!

This is why LangGraph was created — to give developers FULL CONTROL
over the agent's execution flow, with cycles, state, and persistence.

═══════════════════════════════════════════════════════════════════
3. WHERE IT FITS IN THE LANGGRAPH ARCHITECTURE
═══════════════════════════════════════════════════════════════════

LangChain Ecosystem:

    ┌─────────────────────────────────────────────────────────┐
    │  LangGraph Platform (COMMERCIAL) — Deployment           │
    ├─────────────────────────────────────────────────────────┤
    │  Integrations (OSS) — Vector DBs, Logs                  │
    ├──────────────────────┬──────────────────────────────────┤
    │  LangChain (OSS)     │  LangGraph (OSS)                │
    │  Chains, Prompts,    │  State Graphs, Agents,           │
    │  RAG, LCEL           │  Multi-Agent, Workflows          │
    ├──────────────────────┴──────────────────────────────────┤
    │  LangSmith (COMMERCIAL) — Debugging, Tracing, Testing   │
    └─────────────────────────────────────────────────────────┘

    LangChain = building blocks (prompts, LLMs, tools, retrievers)
    LangGraph = orchestration layer (how those blocks connect and flow)
    LangSmith = observability (debugging, tracing, monitoring)

    You ALREADY know LangChain. LangGraph uses LangChain components
    but orchestrates them in a graph instead of a linear chain.

═══════════════════════════════════════════════════════════════════
4. REAL-WORLD ANALOGY
═══════════════════════════════════════════════════════════════════

Think of a HOSPITAL EMERGENCY ROOM:

    LangChain (linear chain):
        Patient arrives → Doctor examines → Prescribes medicine → Done
        Simple, one-pass, no loops.

    LangGraph (stateful graph):
        Patient arrives → Triage nurse assesses severity
            ├── Minor? → General doctor → Prescribe → Discharge
            ├── Serious? → Specialist → Run tests → Review results
            │                 ↑                          |
            │                 └──── Need more tests? ────┘  ← CYCLE
            └── Critical? → Surgery → ICU → Human doctor reviews → Discharge

    The patient's CHART (state) follows them through every step.
    Doctors can LOOP BACK for more tests.
    A human doctor can INTERVENE at any point.
    The chart is SAVED so the next shift can continue.

    That's LangGraph — stateful, cyclical, human-in-the-loop workflows.

═══════════════════════════════════════════════════════════════════
5. WHY LANGGRAPH IS TRENDING
═══════════════════════════════════════════════════════════════════

1. RISE OF AI AGENTS
   Agents need cycles (think → act → observe → think again).
   LangChain chains can't cycle. LangGraph can.

2. MULTI-AGENT SYSTEMS
   Multiple agents collaborating (researcher + writer + reviewer).
   Each agent is a node. Edges define who talks to whom.

3. PRODUCTION REQUIREMENTS
   Real apps need: persistence, human approval, error recovery.
   LangGraph has all of these built-in.

4. TRUSTED BY INDUSTRY
   LinkedIn, Uber, Klarna, GitLab use LangGraph in production.
   It's not experimental — it's battle-tested.

5. FINE-GRAINED CONTROL
   Unlike black-box agent frameworks, LangGraph lets you see
   and control EVERY step of the agent's execution.

Author: GenAI Learner
Date: 2026-04-14
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


# ═══════════════════════════════════════════════════════════════════════════════
# 6. CORE COMPONENTS OF LANGGRAPH
# ═══════════════════════════════════════════════════════════════════════════════
#
# LangGraph has THREE core building blocks:
#
# ┌──────────────────────────────────────────────────────────────────────────┐
# │                                                                          │
# │  1. STATE — The shared memory that flows through the entire graph        │
# │     - Defined as a TypedDict (typed dictionary)                          │
# │     - Every node can READ and WRITE to the state                         │
# │     - Persists across the entire execution                               │
# │     - Example: messages list, user info, intermediate results            │
# │                                                                          │
# │  2. NODES — Python functions that do the actual work                     │
# │     - Each node receives the state, does something, returns updates      │
# │     - A node can be: LLM call, tool execution, data processing, etc.    │
# │     - Nodes are registered with: graph.add_node("name", function)       │
# │                                                                          │
# │  3. EDGES — Connections between nodes (the "wiring")                     │
# │     - STATIC EDGE: Always goes from A to B                              │
# │       graph.add_edge("node_a", "node_b")                                │
# │     - CONDITIONAL EDGE: Goes to A or B based on a condition             │
# │       graph.add_conditional_edges("node_a", router_fn, {...})           │
# │     - This is what enables branching and cycles                          │
# │                                                                          │
# └──────────────────────────────────────────────────────────────────────────┘
#
# GRAPH = STATE + NODES + EDGES
#
# The graph is built with StateGraph, then compiled into a runnable:
#   builder = StateGraph(MyState)
#   builder.add_node(...)
#   builder.add_edge(...)
#   graph = builder.compile()   ← Now it's runnable!
#   result = graph.invoke(...)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. HOW LANGGRAPH WORKS — Step by Step
# ═══════════════════════════════════════════════════════════════════════════════
#
# STEP 1: Define the STATE (shared memory)
#   class AgentState(TypedDict):
#       messages: Annotated[list, operator.add]  # append-only list
#       next_step: str
#
# STEP 2: Define NODE functions (the workers)
#   def chatbot_node(state: AgentState) -> dict:
#       response = llm.invoke(state["messages"])
#       return {"messages": [response]}
#
# STEP 3: Build the GRAPH (connect nodes with edges)
#   builder = StateGraph(AgentState)
#   builder.add_node("chatbot", chatbot_node)
#   builder.set_entry_point("chatbot")
#   builder.add_edge("chatbot", END)
#
# STEP 4: COMPILE the graph
#   graph = builder.compile()
#
# STEP 5: RUN the graph
#   result = graph.invoke({"messages": [HumanMessage("Hello")]})
#
# EXECUTION FLOW:
#   [START] → chatbot_node reads state → LLM generates response
#   → chatbot_node writes response to state → [END]
#
# WITH TOOLS (the agent loop):
#   [START] → [agent_node] → LLM decides: call tool or respond?
#                |                                    |
#                ├── tool_call? → [tool_node] ────────┘  (loop back!)
#                |
#                └── done? → [END]
#
# This loop is what makes agents work — the LLM keeps calling tools
# until it has enough information to answer the user's question.


# ═══════════════════════════════════════════════════════════════════════════════
# 8. CODE EXAMPLES
# ═══════════════════════════════════════════════════════════════════════════════
#
# Lesson 01 is conceptual — no runnable demos here.
# See Lesson 02 for hands-on State, Nodes, Edges (no LLM).
# See Lesson 03 for LLM-powered chatbot with streaming.


# ═══════════════════════════════════════════════════════════════════════════════
# 9. LANGGRAPH vs OTHER FRAMEWORKS (Comparison)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Framework       | Architecture     | Best For              | Control Level
# ────────────────|──────────────────|───────────────────────|──────────────
# LangGraph       | State Graph      | Complex stateful      | FULL control
#                 | (DAG with cycles)| agents, multi-agent   | (every node/edge)
# ────────────────|──────────────────|───────────────────────|──────────────
# LangChain LCEL  | Linear pipe      | Simple chains, RAG    | Medium
#                 | (no cycles)      | one-shot tasks        | (pipe operator)
# ────────────────|──────────────────|───────────────────────|──────────────
# CrewAI          | Role-based       | Multi-agent teams     | Medium
#                 | agents           | with defined roles    | (role config)
# ────────────────|──────────────────|───────────────────────|──────────────
# AutoGen (MSFT)  | Conversational   | Agent-to-agent chat   | Low-Medium
#                 | agents           | collaborative tasks   | (chat-based)
# ────────────────|──────────────────|───────────────────────|──────────────
# OpenAI Swarm    | Lightweight      | Simple agent handoff  | Low
#                 | handoff          | quick prototyping     | (minimal config)
# ────────────────|──────────────────|───────────────────────|──────────────
# AgentScope      | Distributed      | Large-scale           | Medium
#                 | agents           | distributed systems   | (config-based)
#
# WHEN TO CHOOSE LANGGRAPH:
#   - You need CYCLES (agent loops, retry logic)
#   - You need FINE-GRAINED CONTROL over every step
#   - You need PERSISTENCE (save/resume conversations)
#   - You need HUMAN-IN-THE-LOOP (pause for approval)
#   - You're building PRODUCTION systems (not just prototypes)
#   - You want OBSERVABILITY via LangSmith
#
# WHEN TO CHOOSE SOMETHING ELSE:
#   - Simple one-shot chain → LangChain LCEL
#   - Quick multi-agent prototype → CrewAI
#   - Agent-to-agent conversation → AutoGen
#   - Minimal agent handoff → OpenAI Swarm


# ═══════════════════════════════════════════════════════════════════════════════
# 🔄 LANGCHAIN vs LANGGRAPH (The Bridge)
# ═══════════════════════════════════════════════════════════════════════════════
#
# You already know LangChain. Here's how LangGraph maps:
#
# LangChain Concept          → LangGraph Equivalent
# ──────────────────────────────────────────────────────────
# LCEL chain (prompt|llm)    → Node function (calls LLM)
# RunnablePassthrough         → State (data flows through state)
# RunnableParallel            → Parallel nodes (fan-out/fan-in)
# StrOutputParser             → Node that processes LLM output
# Retriever                   → Node that does retrieval
# ConversationMemory          → State + Checkpointer
# trim_messages               → SummarizationMiddleware
# AgentExecutor (deprecated)  → StateGraph with agent loop
# create_react_agent          → Manual graph with tool node
#
# KEY DIFFERENCE:
#   LangChain: Data flows LEFT → RIGHT through pipes. No loops.
#   LangGraph: Data flows through a GRAPH. Loops, branches, cycles.
#
#   LangChain: State is implicit (passed through pipe).
#   LangGraph: State is EXPLICIT (TypedDict, shared across all nodes).


# ═══════════════════════════════════════════════════════════════════════════════
# 10. COMMON MISTAKES & GOTCHAS
# ═══════════════════════════════════════════════════════════════════════════════
#
# Anti-Pattern                    | Why It's Wrong              | Correct Approach
# ────────────────────────────────|─────────────────────────────|──────────────────
# Using plain dict for state      | No type safety, hard debug  | Always use TypedDict
# Forgetting operator.add for     | State OVERWRITES instead    | Use Annotated[list,
#   list fields                   |   of APPENDING              |   operator.add]
# Hardcoding node names in edges  | Brittle, breaks on rename   | Use constants/Enum
# No recursion limit on cycles    | Infinite loops in prod      | Set recursion_limit
# Skipping .compile()             | Graph won't run             | Always compile first
# Not handling errors in nodes    | Crashes the entire graph    | try/except in nodes


# ═══════════════════════════════════════════════════════════════════════════════
# 11. WHERE THIS CONNECTS (Concept Linking Map)
# ═══════════════════════════════════════════════════════════════════════════════
#
# StateGraph → compiles to → CompiledGraph
# Nodes → read/write → State
# Edges → connect → Nodes (static or conditional)
# Conditional Edges → use → router functions → return → node names
# Checkpointer → enables → persistence → enables → human-in-the-loop
# ToolNode → wraps → LangChain tools → used inside → agent loop
# State (TypedDict) → shared memory → flows through ALL nodes


# ═══════════════════════════════════════════════════════════════════════════════
# 12. WHY LANGGRAPH FOR THIS (Benefits Callout)
# ═══════════════════════════════════════════════════════════════════════════════
#
# ✅ Stateful by design — shared State flows through all nodes
# ✅ Cycles and loops — not possible in LangChain LCEL
# ✅ Human-in-the-loop — pause, review, resume
# ✅ First-class streaming — stream tokens AND intermediate steps
# ✅ Built-in persistence — MemorySaver, SqliteSaver, custom
# ✅ Production-ready observability via LangSmith
# ✅ Composable — subgraphs plug into parent graphs
# ✅ Framework-agnostic nodes — any Python function is a node


# ═══════════════════════════════════════════════════════════════════════════════
# 🎯 INTERVIEW QUESTIONS
# ═══════════════════════════════════════════════════════════════════════════════
#
# THEORETICAL:
# Q: What is LangGraph and why was it created?
# A: LangGraph is a library for building stateful, multi-actor applications
#    with LLMs using a graph-based architecture. It was created because
#    LangChain's LCEL chains are sequential (no cycles), making them
#    unsuitable for agent loops where the LLM needs to call tools and
#    loop back. LangGraph introduces StateGraph with nodes, edges, and
#    shared state, enabling cycles, branching, persistence, and
#    human-in-the-loop — all critical for production AI agents.
#
# HANDS-ON:
# Q: Build a simple LangGraph with 2 nodes: a "greeter" that says hello,
#    and a "farewell" that says goodbye. The graph should go:
#    START → greeter → farewell → END
#
# SOLUTION:
#   from typing import TypedDict, Annotated
#   import operator
#   from langgraph.graph import StateGraph, END
#
#   class State(TypedDict):
#       messages: Annotated[list, operator.add]
#
#   def greeter(state): return {"messages": ["Hello! Welcome."]}
#   def farewell(state): return {"messages": ["Goodbye! See you."]}
#
#   builder = StateGraph(State)
#   builder.add_node("greeter", greeter)
#   builder.add_node("farewell", farewell)
#   builder.set_entry_point("greeter")
#   builder.add_edge("greeter", "farewell")
#   builder.add_edge("farewell", END)
#   graph = builder.compile()
#   result = graph.invoke({"messages": []})
#   print(result["messages"])  # ["Hello! Welcome.", "Goodbye! See you."]


# ═══════════════════════════════════════════════════════════════════════════════
# 📝 QUICK RECAP (3 bullets max)
# ═══════════════════════════════════════════════════════════════════════════════
#
# 1. LangGraph = State + Nodes + Edges — a graph-based orchestration layer
#    for building stateful, cyclical AI agent workflows.
#
# 2. It solves what LangChain LCEL can't: cycles (agent loops), shared state,
#    persistence, human-in-the-loop, and fine-grained execution control.
#
# 3. Core pattern: Define State (TypedDict) → Add Nodes (functions) →
#    Connect with Edges → Compile → Invoke. Always compile before running.


# ═══════════════════════════════════════════════════════════════════════════════
# 🚀 Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("🧠 LANGGRAPH LESSON 1 — Why LangGraph? (Conceptual)")
    logger.info("=" * 70)
    logger.info("")
    logger.info("This lesson is conceptual. Read the file for full content.")
    logger.info("")
    logger.info("Key takeaways:")
    logger.info("  - LangGraph = State + Nodes + Edges (graph-based orchestration)")
    logger.info("  - Solves what LCEL can't: cycles, shared state, persistence")
    logger.info("  - Used by LinkedIn, Uber, Klarna, GitLab in production")
    logger.info("")
    logger.info("Next lessons:")
    logger.info("  Lesson 02 → Hands-on: State, Nodes, Edges (no LLM)")
    logger.info("  Lesson 03 → LLM Chatbot with Streaming")
    logger.info("=" * 70)
