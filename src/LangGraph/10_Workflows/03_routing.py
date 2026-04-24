"""
🔀 LangGraph Workflows — Lesson 3: Routing (Level 7)

═══════════════════════════════════════════════════════════════════
1. CONCEPT: Routing — ONE-LINE DEFINITION
═══════════════════════════════════════════════════════════════════

Routing is a workflow pattern where an LLM (or logic function)
EXAMINES the input and DECIDES which specialized node should
handle it — conditionally choosing the next node based on
state or an LLM's structured decision.

═══════════════════════════════════════════════════════════════════
2. WHY ROUTING EXISTS
═══════════════════════════════════════════════════════════════════

Not all inputs should follow the same path. Why?
    - A joke request needs a DIFFERENT prompt than a poem request.
    - A billing question needs a DIFFERENT handler than a tech issue.
    - One-size-fits-all prompts produce MEDIOCRE results for all.

Routing solves this by:
    - Using an LLM to CLASSIFY the input (structured output).
    - Storing the classification in STATE (e.g., state["decision"]).
    - A router function reads state and RETURNS the next node name.
    - Each specialist node has a FOCUSED prompt for its domain.

═══════════════════════════════════════════════════════════════════
3. HOW ROUTING DIFFERS FROM tools_condition
═══════════════════════════════════════════════════════════════════

    tools_condition (Lessons 06/07):
        - Built-in LangGraph helper for TOOL-CALLING agents.
        - Checks if the LLM response contains tool_calls.
        - Routes to "tools" node or END — ONLY two destinations.
        - SPECIFIC to the ReAct agent loop pattern.

    route_decision (THIS lesson):
        - A CUSTOM function YOU write.
        - Reads ANY state field (not just tool_calls).
        - Routes to N DIFFERENT nodes based on ANY logic.
        - MORE FLEXIBLE — works for classification, triage, etc.

    In short:
        tools_condition = "Did the LLM want to call a tool?"
        route_decision  = "What TYPE of request is this?"

═══════════════════════════════════════════════════════════════════
4. REAL-WORLD ANALOGY — A HOSPITAL TRIAGE NURSE
═══════════════════════════════════════════════════════════════════

Think of a HOSPITAL TRIAGE NURSE:

    1. Patient arrives at the hospital with symptoms.
    2. The TRIAGE NURSE examines the patient (the router node).
    3. Based on symptoms, the nurse DECIDES:
        - Chest pain?     → Route to ER (emergency room)
        - Mild fever?     → Route to General Doctor
        - Broken bone?    → Route to Orthopedic Specialist
    4. The SPECIALIST handles the patient with domain expertise.

    The triage nurse does NOT treat the patient.
    The triage nurse CLASSIFIES and ROUTES.

    This is EXACTLY how routing works in LangGraph:
        - The router node CLASSIFIES the input.
        - The route_decision function ROUTES to the right specialist.
        - The specialist node HANDLES the request.

═══════════════════════════════════════════════════════════════════
5. ASCII GRAPH STRUCTURES
═══════════════════════════════════════════════════════════════════

Demo 1 — Content Type Router:

    [START]
       |
    [router]  ← LLM classifies input using structured output
       |
       ├── (decision="story") ──→ [write_story] → [END]
       |
       ├── (decision="joke")  ──→ [write_joke]  → [END]
       |
       └── (decision="poem")  ──→ [write_poem]  → [END]

Demo 2 — Customer Support Router:

    [START]
       |
    [support_router]  ← LLM classifies query into department
       |
       ├── (department="billing")   ──→ [handle_billing]   → [END]
       |
       ├── (department="technical") ──→ [handle_technical] → [END]
       |
       └── (department="general")   ──→ [handle_general]   → [END]

═══════════════════════════════════════════════════════════════════

🔄 LANGCHAIN vs LANGGRAPH — Routing
LangChain : RunnableBranch — static if/else routing on output.
            Limited to simple string matching. No shared state.
            The LLM doesn't DECIDE the route — you hardcode rules.
LangGraph  : StateGraph + structured output + conditional edges.
            The LLM CLASSIFIES the input via Pydantic model.
            The route function reads STATE and returns a node name.
            Full control, N-way routing, stateful, composable.

LangChain : Routing is a workaround (RunnableBranch).
LangGraph  : Routing is a FIRST-CLASS pattern (conditional edges).

═══════════════════════════════════════════════════════════════════
6. WHERE IT FITS IN THE LANGGRAPH ARCHITECTURE
═══════════════════════════════════════════════════════════════════

    Routing is a WORKFLOW PATTERN built on top of:
        StateGraph → Nodes → Conditional Edges → Structured Output

    HOW IT WORKS:
        1. A router node calls llm.with_structured_output(Route)
           to classify the input into a Pydantic model.
        2. The classification is stored in state (e.g., state["decision"]).
        3. add_conditional_edges uses a route function that reads
           state and returns the name of the next node.
        4. LangGraph routes to the correct specialist node.

    Concept Linking:
        StateGraph → compiles to → CompiledGraph
        Structured Output → Pydantic BaseModel → classification
        Conditional Edges → use → route functions → return → node names
        Routing → enables → N-way branching based on LLM decisions
        Routing → different from → tools_condition (tool-specific)
        Routing → combines with → Prompt Chaining, Parallelization

HOW TO RUN:
    $ python src/Workflows/03_routing.py

Author: GenAI Learner
"""

import logging
import os
import time
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Helper: Retry wrapper for Groq structured output ────────────────────
# Groq's tool_use method for structured output can intermittently fail
# with llama-3.1-8b-instant (400 Bad Request). A simple retry fixes this.
def invoke_with_retry(runnable, input_data, max_retries: int = 3):
    """Invoke a runnable with retry logic for intermittent Groq failures."""
    for attempt in range(max_retries):
        try:
            return runnable.invoke(input_data)
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning("  Retry %d/%d — Groq error: %s", attempt + 1, max_retries, str(e)[:100])
                time.sleep(1)
            else:
                raise

# ── Helper: Save graph as PNG image ──────────────────────────────────────
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
# 7. DEMO 1: Content Type Router (LLM-Driven Routing with Structured Output)
# ═══════════════════════════════════════════════════════════════════════════════
#
# This is the CORE routing pattern from the notebook, improved.
# An LLM classifies the user's input as "poem", "story", or "joke"
# using Pydantic structured output. The classification is stored in
# state["decision"]. A route function reads it and returns the
# specialist node name.
#
# KEY INSIGHT: The router node does NOT generate content.
# It ONLY classifies. The specialist nodes generate content.
# This separation of concerns is what makes routing powerful.
#
# HOW with_structured_output WORKS:
#   router = llm.with_structured_output(Route)
#   This wraps the LLM so it ALWAYS returns a Route Pydantic object.
#   The LLM is forced to pick from Literal["poem", "story", "joke"].
#   No parsing needed — you get route.step directly.
#
# WHY THIS MATTERS:
#   In production, you route customer queries, support tickets,
#   content requests, etc. to specialized handlers.
#   Each handler has a focused prompt → better quality output.

def demo_content_router() -> None:
    """Content type router: classify input → route to specialist."""

    # ── Pydantic Route Model: forces LLM to pick one of 3 types ─────
    # This is the STRUCTURED OUTPUT schema.
    # The LLM must return one of: "poem", "story", "joke".
    # NOTE: Field has no default — this is REQUIRED for Groq's
    # structured output to work reliably with tool_use.
    class Route(BaseModel):
        step: Literal["poem", "story", "joke"] = Field(
            ..., description="The content type to route to: poem, story, or joke"
        )

    # ── State: holds input, routing decision, and final output ───────
    class RouterState(TypedDict):
        input: str          # The user's request
        decision: str       # The routing decision: "poem", "story", or "joke"
        output: str         # The generated content from the specialist

    # ── LLM: ChatGroq for both routing and generation ────────────────
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_tokens=512)

    # ── Router: LLM with structured output (Pydantic model) ─────────
    # with_structured_output wraps the LLM so it returns a Route object.
    # The LLM is FORCED to classify — no free-form text allowed.
    # NOTE: We use llama-3.3-70b-versatile for routing because it has
    # the most reliable tool_use / structured output on Groq.
    # llama-3.1-8b-instant can intermittently fail with structured output.
    router_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, max_tokens=128)
    router = router_llm.with_structured_output(Route)

    # ── Node 1: llm_call_router — classifies the input ──────────────
    # This node does NOT generate content. It ONLY classifies.
    # It uses SystemMessage to instruct the LLM on routing rules,
    # and HumanMessage with the user's input.
    # The result is stored in state["decision"].
    def llm_call_router(state: RouterState) -> dict:
        """Classify the input into poem, story, or joke using structured output."""
        logger.info("  [router] Classifying input: '%s'", state["input"][:80])
        decision = invoke_with_retry(
            router,
            [
                SystemMessage(
                    content="Route the input to story, joke, or poem based on the user's request."
                ),
                HumanMessage(content=state["input"]),
            ],
        )
        logger.info("  [router] Decision: %s", decision.step)
        return {"decision": decision.step}

    # ── Route Function: reads state["decision"], returns node name ───
    # This is the CONDITIONAL EDGE function.
    # It reads the classification from state and returns the
    # name of the specialist node to execute next.
    #
    # KEY DIFFERENCE from tools_condition:
    #   tools_condition checks for tool_calls → routes to "tools" or END.
    #   route_decision reads a CUSTOM field → routes to ANY node.
    def route_decision(state: RouterState) -> str:
        """Read the routing decision from state and return the next node name."""
        if state["decision"] == "story":
            return "write_story"
        elif state["decision"] == "joke":
            return "write_joke"
        elif state["decision"] == "poem":
            return "write_poem"
        # Fallback — should never happen with structured output
        return "write_story"

    # ── Node 2: write_story — specialist for stories ─────────────────
    def write_story(state: RouterState) -> dict:
        """Generate a short story based on the user's input."""
        logger.info("  [write_story] ✍️  Generating story...")
        response = llm.invoke(
            f"Write a short, engaging story (3-4 sentences) based on this request: "
            f"{state['input']}"
        )
        logger.info("  [write_story] Done: %s", response.content[:120])
        return {"output": response.content}

    # ── Node 3: write_joke — specialist for jokes ────────────────────
    def write_joke(state: RouterState) -> dict:
        """Generate a joke based on the user's input."""
        logger.info("  [write_joke] 😂 Generating joke...")
        response = llm.invoke(
            f"Write a funny, clever joke based on this request: "
            f"{state['input']}"
        )
        logger.info("  [write_joke] Done: %s", response.content[:120])
        return {"output": response.content}

    # ── Node 4: write_poem — specialist for poems ────────────────────
    def write_poem(state: RouterState) -> dict:
        """Generate a poem based on the user's input."""
        logger.info("  [write_poem] 📝 Generating poem...")
        response = llm.invoke(
            f"Write a short, beautiful poem (4-6 lines) based on this request: "
            f"{state['input']}"
        )
        logger.info("  [write_poem] Done: %s", response.content[:120])
        return {"output": response.content}

    # ── Build the graph ──────────────────────────────────────────────
    builder = StateGraph(RouterState)

    # Register all nodes
    builder.add_node("llm_call_router", llm_call_router)
    builder.add_node("write_story", write_story)
    builder.add_node("write_joke", write_joke)
    builder.add_node("write_poem", write_poem)

    # Entry edge: START → router
    builder.add_edge(START, "llm_call_router")

    # CONDITIONAL EDGE: router → route_decision → specialist
    # The route_decision function reads state["decision"] and
    # returns the name of the next node to execute.
    # The mapping dict maps return values to node names.
    builder.add_conditional_edges(
        "llm_call_router",
        route_decision,
        {
            "write_story": "write_story",
            "write_joke": "write_joke",
            "write_poem": "write_poem",
        },
    )

    # All specialists → END
    builder.add_edge("write_story", END)
    builder.add_edge("write_joke", END)
    builder.add_edge("write_poem", END)

    # Compile the graph
    graph = builder.compile()

    # Save the graph visualization
    save_graph_image(graph, "workflow03_content_router")

    # ── Test with 3 different inputs ─────────────────────────────────
    test_inputs = [
        "Write me a joke about AI taking over the world",
        "Tell me a story about a robot who learns to love",
        "Write a poem about the beauty of nature in spring",
    ]

    logger.info("--- Demo 1: Content Type Router ---")
    for user_input in test_inputs:
        logger.info("  ─── Input: '%s' ───", user_input)
        result = graph.invoke(
            {"input": user_input, "decision": "", "output": ""}
        )
        logger.info("  Routed to: %s", result["decision"])
        logger.info("  Output: %s", result["output"][:200])
        logger.info("  Flow: START → router → (%s) → %s → END",
                     result["decision"],
                     f"write_{result['decision']}")
        logger.info("")

    logger.info("  Pattern: LLM-driven routing via structured output + conditional edges.")


# ═══════════════════════════════════════════════════════════════════════════════
# 8. DEMO 2: Production Customer Support Router (Real Business Use Case)
# ═══════════════════════════════════════════════════════════════════════════════
#
# A more PRACTICAL example of routing in a real business scenario.
# Customer queries are classified into departments: billing, technical,
# or general. Each department has a specialized handler.
#
# This shows routing in a REAL customer support use case:
#   - A customer submits a query (e.g., "My payment failed")
#   - The router LLM classifies it into a department
#   - The route function sends it to the right handler
#   - The handler generates a specialized response
#
# KEY DIFFERENCE from Demo 1:
#   Demo 1 routes CREATIVE content (poem/story/joke).
#   Demo 2 routes SUPPORT queries (billing/technical/general).
#   Same pattern, different domain — routing is UNIVERSAL.

def demo_support_router() -> None:
    """Customer support router: classify query → route to department."""

    # ── Pydantic Department Model: forces LLM to pick a department ───
    class Department(BaseModel):
        department: Literal["billing", "technical", "general"] = Field(
            ..., description="The department to route the customer query to"
        )

    # ── State: holds query, department classification, and response ───
    class SupportState(TypedDict):
        query: str          # The customer's query
        department: str     # The classified department
        response: str       # The department's response

    # ── LLM: ChatGroq for both routing and response generation ───────
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_tokens=512)

    # ── Router: LLM with structured output for department classification
    # NOTE: We use llama-3.3-70b-versatile for routing because it has
    # the most reliable tool_use / structured output on Groq.
    router_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, max_tokens=128)
    dept_router = router_llm.with_structured_output(Department)

    # ── Node 1: support_router — classifies the query ────────────────
    def support_router(state: SupportState) -> dict:
        """Classify the customer query into billing, technical, or general."""
        logger.info("  [support_router] Classifying query: '%s'", state["query"][:80])
        classification = invoke_with_retry(
            dept_router,
            [
                SystemMessage(
                    content=(
                        "You are a customer support classifier. "
                        "Route the customer's query to the correct department:\n"
                        "- billing: payment issues, invoices, refunds, subscriptions\n"
                        "- technical: bugs, crashes, errors, app issues, performance\n"
                        "- general: hours, locations, general info, greetings"
                    )
                ),
                HumanMessage(content=state["query"]),
            ],
        )
        logger.info("  [support_router] Department: %s", classification.department)
        return {"department": classification.department}

    # ── Route Function: reads state["department"], returns node name ──
    def route_to_department(state: SupportState) -> str:
        """Read the department from state and return the handler node name."""
        if state["department"] == "billing":
            return "handle_billing"
        elif state["department"] == "technical":
            return "handle_technical"
        elif state["department"] == "general":
            return "handle_general"
        return "handle_general"

    # ── Node 2: handle_billing — billing department specialist ────────
    def handle_billing(state: SupportState) -> dict:
        """Handle billing-related customer queries."""
        logger.info("  [handle_billing] 💳 Handling billing query...")
        response = llm.invoke(
            f"You are a billing support specialist. Respond helpfully and "
            f"professionally to this customer query. Offer specific next steps.\n\n"
            f"Customer query: {state['query']}"
        )
        logger.info("  [handle_billing] Response: %s", response.content[:120])
        return {"response": response.content}

    # ── Node 3: handle_technical — technical support specialist ───────
    def handle_technical(state: SupportState) -> dict:
        """Handle technical support queries."""
        logger.info("  [handle_technical] 🔧 Handling technical query...")
        response = llm.invoke(
            f"You are a technical support specialist. Diagnose the issue and "
            f"provide clear troubleshooting steps.\n\n"
            f"Customer query: {state['query']}"
        )
        logger.info("  [handle_technical] Response: %s", response.content[:120])
        return {"response": response.content}

    # ── Node 4: handle_general — general inquiries handler ───────────
    def handle_general(state: SupportState) -> dict:
        """Handle general customer inquiries."""
        logger.info("  [handle_general] ℹ️  Handling general query...")
        response = llm.invoke(
            f"You are a friendly customer service representative. Answer this "
            f"general inquiry warmly and helpfully.\n\n"
            f"Customer query: {state['query']}"
        )
        logger.info("  [handle_general] Response: %s", response.content[:120])
        return {"response": response.content}

    # ── Build the graph ──────────────────────────────────────────────
    builder = StateGraph(SupportState)

    # Register all nodes
    builder.add_node("support_router", support_router)
    builder.add_node("handle_billing", handle_billing)
    builder.add_node("handle_technical", handle_technical)
    builder.add_node("handle_general", handle_general)

    # Entry edge: START → support_router
    builder.add_edge(START, "support_router")

    # CONDITIONAL EDGE: support_router → route_to_department → handler
    builder.add_conditional_edges(
        "support_router",
        route_to_department,
        {
            "handle_billing": "handle_billing",
            "handle_technical": "handle_technical",
            "handle_general": "handle_general",
        },
    )

    # All handlers → END
    builder.add_edge("handle_billing", END)
    builder.add_edge("handle_technical", END)
    builder.add_edge("handle_general", END)

    # Compile the graph
    graph = builder.compile()

    # Save the graph visualization
    save_graph_image(graph, "workflow03_support_router")

    # ── Test with 3 different customer queries ───────────────────────
    test_queries = [
        "My payment failed and I was still charged twice",
        "The app keeps crashing every time I open the settings page",
        "What are your business hours on weekends?",
    ]

    logger.info("--- Demo 2: Production Customer Support Router ---")
    for query in test_queries:
        logger.info("  ─── Query: '%s' ───", query)
        result = graph.invoke(
            {"query": query, "department": "", "response": ""}
        )
        logger.info("  Routed to: %s", result["department"])
        logger.info("  Response: %s", result["response"][:200])
        logger.info("  Flow: START → support_router → (%s) → handle_%s → END",
                     result["department"],
                     result["department"])
        logger.info("")

    logger.info("  Pattern: LLM-driven department routing for customer support.")


# ═══════════════════════════════════════════════════════════════════════════════
# 9. COMPARISON TABLE: Routing Strategies in LangGraph
# ═══════════════════════════════════════════════════════════════════════════════
#
# Strategy          | What It Does                        | When to Use
# ──────────────────|─────────────────────────────────────|──────────────────────────────
# tools_condition   | Checks if LLM made tool_calls       | ReAct agent loop (tool use)
#                   | Routes to "tools" node or END        | ONLY 2 destinations
#                   | Built-in LangGraph helper            | Specific to tool-calling
# ──────────────────|─────────────────────────────────────|──────────────────────────────
# route_decision    | Reads a CUSTOM state field           | N-way classification routing
# (this lesson)     | Routes to ANY node based on logic    | Content type, department, etc.
#                   | Uses structured output for decision  | Flexible, general-purpose
# ──────────────────|─────────────────────────────────────|──────────────────────────────
# check_conflict    | Inspects state for a condition       | Quality gates (pass/fail)
# (lesson 01)       | Returns "Pass" or "Fail"             | Retry loops, fix branches
#                   | Typically NO LLM call in the gate    | Binary routing only
# ──────────────────|─────────────────────────────────────|──────────────────────────────
#
# KEY TAKEAWAY:
#   - tools_condition → tool-specific, binary (tools or END)
#   - route_decision  → general-purpose, N-way (any node)
#   - check_conflict  → quality gate, binary (pass or fail)
#
#   All three use add_conditional_edges under the hood.
#   The DIFFERENCE is what they CHECK and how many DESTINATIONS they support.


# ═══════════════════════════════════════════════════════════════════════════════
# 10. WHEN TO USE ROUTING
# ═══════════════════════════════════════════════════════════════════════════════
#
# USE routing when:
#   ✅ Different inputs need DIFFERENT processing paths
#   ✅ You want the LLM to CLASSIFY/TRIAGE before processing
#   ✅ Each path has a SPECIALIZED prompt or handler
#   ✅ You need N-way branching (more than pass/fail)
#   ✅ The classification can be expressed as a Pydantic model
#
# DON'T use routing when:
#   ❌ All inputs follow the same path (use prompt chaining)
#   ❌ You only need pass/fail gating (use quality gates)
#   ❌ You're building a tool-calling agent (use tools_condition)
#   ❌ The routing logic is too complex for a single classification
#      (use an orchestrator-workers pattern instead)


# ═══════════════════════════════════════════════════════════════════════════════
# 11. COMMON MISTAKES & GOTCHAS
# ═══════════════════════════════════════════════════════════════════════════════
#
# Mistake                                    | Fix
# ───────────────────────────────────────────|──────────────────────────────────────
# Router node ALSO generates content         | Router should ONLY classify.
#                                            | Keep routing and generation separate.
# Not using structured output for routing    | Always use with_structured_output()
#                                            | to force the LLM into valid categories.
# Missing fallback in route function         | Always have a default return value
#                                            | in case the LLM returns unexpected output.
# Hardcoding node names as strings           | Use constants or match the Literal values
#                                            | to avoid typos that silently break routing.
# Route function doesn't match mapping dict  | The return values of route_decision MUST
#                                            | match the keys in add_conditional_edges.
# Using plain dict for state                 | Always use TypedDict for type safety.
# Forgetting to compile before invoke        | Always call builder.compile().
#
# ANTI-PATTERN: Putting routing logic AND content generation in one node.
#   Instead, separate classification (router) from execution (specialist).
#   This makes each node focused, testable, and reusable.


# ═══════════════════════════════════════════════════════════════════════════════
# 12. WHY LANGGRAPH FOR THIS
# ═══════════════════════════════════════════════════════════════════════════════
#
# ✅ Conditional edges — N-way routing is NATIVE to the graph model
# ✅ Structured output — Pydantic models force valid classifications
# ✅ Stateful by design — routing decisions stored in shared state
# ✅ Explicit control — you SEE every route in the graph visualization
# ✅ Composable — routers can be subgraphs inside larger workflows
# ✅ Production-ready — combine with retry loops, parallelization, etc.


# ═══════════════════════════════════════════════════════════════════════════════
# 13. WHERE THIS CONNECTS (Concept Linking Map)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Routing → uses → StateGraph + Conditional Edges + Structured Output
# Structured Output → Pydantic BaseModel → forces valid classification
# Conditional Edges → use → route functions → return → node names
# Router Node → classifies input → stores decision in State
# Specialist Nodes → handle specific types → focused prompts
# Routing → different from → tools_condition (tool-specific, binary)
# Routing → different from → check_conflict (quality gate, binary)
# Routing → combines with → Prompt Chaining (route → then chain steps)
# Routing → combines with → Parallelization (route → then fan-out)
# Routing → foundation for → Orchestrator-Workers pattern


# ═══════════════════════════════════════════════════════════════════════════════
# 🎯 INTERVIEW QUESTIONS
# ═══════════════════════════════════════════════════════════════════════════════
#
# THEORETICAL:
# Q: What is the difference between tools_condition and a custom
#    route_decision function in LangGraph? When would you use each?
# A: tools_condition is a built-in LangGraph helper that checks if
#    the LLM's response contains tool_calls. It routes to a "tools"
#    node or END — only two destinations. It's specific to the ReAct
#    agent loop pattern. A custom route_decision function reads ANY
#    state field (not just tool_calls) and can route to N different
#    nodes based on any logic. It's used for classification/triage
#    scenarios where the LLM decides the TYPE of request using
#    structured output (Pydantic model). Use tools_condition for
#    tool-calling agents; use route_decision for N-way classification.
#
# HANDS-ON:
# Q: Build a routing graph that classifies user input as "positive",
#    "negative", or "neutral" sentiment, then routes to a specialist
#    node that generates an appropriate response for each sentiment.
#    Use Pydantic structured output for classification.
#
# SOLUTION:
#   from typing import Literal
#   from pydantic import BaseModel, Field
#   from typing_extensions import TypedDict
#   from langchain_core.messages import HumanMessage, SystemMessage
#   from langchain_groq import ChatGroq
#   from langgraph.graph import StateGraph, START, END
#
#   class Sentiment(BaseModel):
#       mood: Literal["positive", "negative", "neutral"] = Field(
#           description="The sentiment of the user's message"
#       )
#
#   class SentimentState(TypedDict):
#       input: str
#       sentiment: str
#       response: str
#
#   llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_tokens=512)
#   classifier = llm.with_structured_output(Sentiment)
#
#   def classify(state):
#       result = classifier.invoke([
#           SystemMessage(content="Classify the sentiment as positive, negative, or neutral."),
#           HumanMessage(content=state["input"]),
#       ])
#       return {"sentiment": result.mood}
#
#   def route_sentiment(state):
#       return f"handle_{state['sentiment']}"
#
#   def handle_positive(state):
#       r = llm.invoke(f"Respond enthusiastically to: {state['input']}")
#       return {"response": r.content}
#
#   def handle_negative(state):
#       r = llm.invoke(f"Respond empathetically to: {state['input']}")
#       return {"response": r.content}
#
#   def handle_neutral(state):
#       r = llm.invoke(f"Respond helpfully to: {state['input']}")
#       return {"response": r.content}
#
#   builder = StateGraph(SentimentState)
#   builder.add_node("classify", classify)
#   builder.add_node("handle_positive", handle_positive)
#   builder.add_node("handle_negative", handle_negative)
#   builder.add_node("handle_neutral", handle_neutral)
#   builder.add_edge(START, "classify")
#   builder.add_conditional_edges("classify", route_sentiment, {
#       "handle_positive": "handle_positive",
#       "handle_negative": "handle_negative",
#       "handle_neutral": "handle_neutral",
#   })
#   builder.add_edge("handle_positive", END)
#   builder.add_edge("handle_negative", END)
#   builder.add_edge("handle_neutral", END)
#   graph = builder.compile()
#   result = graph.invoke({"input": "I love this product!", "sentiment": "", "response": ""})
#   print(f"Sentiment: {result['sentiment']}")
#   print(f"Response: {result['response']}")


# ═══════════════════════════════════════════════════════════════════════════════
# 📝 QUICK RECAP
# ═══════════════════════════════════════════════════════════════════════════════
#
# 1. Routing uses an LLM with structured output (Pydantic BaseModel)
#    to CLASSIFY input, then a route function reads the classification
#    from state and returns the specialist node name.
#
# 2. route_decision is MORE FLEXIBLE than tools_condition.
#    tools_condition is binary (tools or END) and tool-specific.
#    route_decision supports N-way routing to ANY node based on
#    ANY state field — ideal for classification and triage.
#
# 3. ALWAYS separate routing (classification) from execution (generation).
#    The router node classifies. The specialist nodes generate.
#    This keeps each node focused, testable, and reusable.


# ═══════════════════════════════════════════════════════════════════════════════
# 🚀 Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("🔀 LANGGRAPH WORKFLOWS — Lesson 3: Routing")
    logger.info("=" * 70)

    logger.info("\n🔹 Demo 1: Content Type Router (Poem / Story / Joke)")
    demo_content_router()

    logger.info("\n🔹 Demo 2: Production Customer Support Router (Billing / Technical / General)")
    demo_support_router()

    logger.info("\n" + "=" * 70)
    logger.info("✅ Lesson 3 complete — Routing with LLM Classification")
    logger.info("You can now build N-way routing workflows using structured output.")
    logger.info("Next: Lesson 4 — Orchestrator-Workers Pattern")
    logger.info("=" * 70)
