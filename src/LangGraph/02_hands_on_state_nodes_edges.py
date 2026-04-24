"""
🏗️ LangGraph Lesson 2 — Hands-On: State, Nodes, Edges & Conditional Routing

═══════════════════════════════════════════════════════════════════
1. CONCEPT: StateGraph Building Blocks (Hands-On Practice)
═══════════════════════════════════════════════════════════════════

A LangGraph application is built from 3 pieces:
    STATE  — shared memory (TypedDict) that flows through the graph
    NODES  — Python functions that read/write state
    EDGES  — connections between nodes (static or conditional)

This lesson is HANDS-ON — no LLM integration yet.
We build graphs step by step, from the simplest to conditional routing.

═══════════════════════════════════════════════════════════════════
2. WHY THIS EXISTS
═══════════════════════════════════════════════════════════════════

In LangChain LCEL, data flows in ONE direction:
    prompt | llm | parser → done (no loops, no branching)

But real agents need DECISIONS:
    "Should I search the web or answer directly?"
    "Should I call a tool or end the conversation?"

StateGraph gives you FULL CONTROL over this flow:
    - Static edges: A always goes to B
    - Conditional edges: A goes to B OR C based on a condition
    - Cycles: B can loop back to A (impossible in LCEL)

═══════════════════════════════════════════════════════════════════
3. WHERE IT FITS
═══════════════════════════════════════════════════════════════════

    StateGraph → you define nodes + edges
         ↓
    .compile() → produces a CompiledGraph (runnable)
         ↓
    .invoke() → runs the graph with initial state
         ↓
    Result → final state after all nodes have executed

═══════════════════════════════════════════════════════════════════
4. REAL-WORLD ANALOGY
═══════════════════════════════════════════════════════════════════

Think of a SPORTS DAY at school:

    You arrive → Registration desk (START)
         ↓
    "What sport do you want to play?"
         ↓
    ├── Cricket field → Play cricket → Go home (END)
    └── Badminton court → Play badminton → Go home (END)

    The REGISTRATION DESK is the entry node.
    The CHOICE of sport is a conditional edge.
    The SPORT FIELDS are nodes that do work.
    Going HOME is the END node.
    Your NAME TAG (state) follows you everywhere.

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
        logger.info("  Falling back to ASCII:")
        compiled_graph.get_graph().print_ascii()


# ═══════════════════════════════════════════════════════════════════════════════
# 5. DEMO 1: Simplest Graph — Two Nodes, One Edge
# ═══════════════════════════════════════════════════════════════════════════════
#
# Graph structure:
#     [START] → [greet] → [farewell] → [END]
#
# This is the absolute minimum — two nodes connected by static edges.
# No branching, no conditions, no LLM. Just pure graph mechanics.
#
# 🔄 LANGCHAIN vs LANGGRAPH:
#   LangChain: chain = greet_fn | farewell_fn  (pipe operator, linear)
#   LangGraph: graph with nodes + edges (explicit connections, extensible)

def demo_simplest_graph() -> None:
    """Two nodes connected by static edges — the simplest possible graph."""
    from typing_extensions import TypedDict
    from langgraph.graph import END, START, StateGraph

    # ── STEP 1: Define State ─────────────────────────────────────────
    # State is a TypedDict — the shared memory for the entire graph.
    # Every node reads from state and returns updates to state.
    # Here we have ONE field: graph_info (a string that accumulates text).
    class State(TypedDict):
        graph_info: str  # Accumulates text as it flows through nodes

    # ── STEP 2: Define Node Functions ────────────────────────────────
    # A node is a Python function that:
    #   - Takes the current state as its FIRST argument
    #   - Returns a dict with the state keys to UPDATE
    #   - By default, returned values OVERWRITE the existing state
    def greet(state: State) -> dict:
        """Greet node — adds a greeting to the state."""
        logger.info("  [greet node] executing")
        return {"graph_info": state["graph_info"] + " → Hello!"}

    def farewell(state: State) -> dict:
        """Farewell node — adds a goodbye to the state."""
        logger.info("  [farewell node] executing")
        return {"graph_info": state["graph_info"] + " → Goodbye!"}

    # ── STEP 3: Build the Graph ──────────────────────────────────────
    # StateGraph(State) creates a graph builder with our state schema.
    graph = StateGraph(State)

    # add_node("name", function) registers a node.
    # The name is how you reference it in edges.
    graph.add_node("greet", greet)
    graph.add_node("farewell", farewell)

    # add_edge(from, to) creates a STATIC connection.
    # START is a special node — it's where execution begins.
    # END is a special node — it's where execution terminates.
    graph.add_edge(START, "greet")       # START → greet
    graph.add_edge("greet", "farewell")  # greet → farewell
    graph.add_edge("farewell", END)      # farewell → END

    # ── STEP 4: Compile ──────────────────────────────────────────────
    # compile() converts the builder into a runnable CompiledGraph.
    # COMMON MISTAKE: Forgetting to compile — the graph won't run!
    compiled = graph.compile()

    # ── STEP 5: Print the graph structure in terminal ────────────────
    logger.info("--- Demo 1: Simplest Graph ---")
    logger.info("  Graph visualization:")
    save_graph_image(compiled, "demo1_simplest_graph")

    # ── STEP 6: Invoke ───────────────────────────────────────────────
    # invoke() runs the graph with an initial state.
    # The state flows: START → greet (updates state) → farewell (updates state) → END
    result = compiled.invoke({"graph_info": "Journey"})

    logger.info("  Graph: [START] → [greet] → [farewell] → [END]")
    logger.info("  Final state: %s", result["graph_info"])
    # Output: "Journey → Hello! → Goodbye!"


# ═══════════════════════════════════════════════════════════════════════════════
# 6. DEMO 2: Conditional Edges — The Sports Day Example
# ═══════════════════════════════════════════════════════════════════════════════
#
# Graph structure:
#     [START] → [start_play]
#                    |
#                    ├── random > 0.5? → [cricket] → [END]
#                    |
#                    └── random <= 0.5? → [badminton] → [END]
#
# This is the KEY pattern — conditional routing based on state or logic.
# The router function decides which node to go to next.
#
# 🔄 LANGCHAIN vs LANGGRAPH:
#   LangChain: No built-in conditional routing in LCEL pipes
#   LangGraph: add_conditional_edges() with a router function

def demo_conditional_edges() -> None:
    """Conditional routing — the sports day example from the notebook."""
    import random
    from typing import Literal

    from typing_extensions import TypedDict
    from langgraph.graph import END, START, StateGraph

    # ── State ────────────────────────────────────────────────────────
    class State(TypedDict):
        graph_info: str

    # ── Node Functions ───────────────────────────────────────────────
    # Each node reads state["graph_info"] and appends to it.
    def start_play(state: State) -> dict:
        """Entry node — announces the intention to play."""
        logger.info("  [start_play] node executing")
        return {"graph_info": state["graph_info"] + " I am planning to play"}

    def cricket(state: State) -> dict:
        """Cricket node — chosen when random > 0.5."""
        logger.info("  [cricket] node executing")
        return {"graph_info": state["graph_info"] + " Cricket"}

    def badminton(state: State) -> dict:
        """Badminton node — chosen when random <= 0.5."""
        logger.info("  [badminton] node executing")
        return {"graph_info": state["graph_info"] + " Badminton"}

    # ── Router Function ──────────────────────────────────────────────
    # A router function:
    #   - Takes the current state
    #   - Returns a STRING that matches a node name
    #   - The return type hint Literal["cricket", "badminton"] tells
    #     LangGraph which nodes are possible targets
    #
    # This is how DECISIONS are made in LangGraph.
    # In a real agent, this would check if the LLM wants to call a tool.
    def random_play(state: State) -> Literal["cricket", "badminton"]:
        """Router: randomly picks cricket or badminton."""
        if random.random() > 0.5:
            return "cricket"
        else:
            return "badminton"

    # ── Build Graph ──────────────────────────────────────────────────
    graph = StateGraph(State)

    graph.add_node("start_play", start_play)
    graph.add_node("cricket", cricket)
    graph.add_node("badminton", badminton)

    # Static edge: START always goes to start_play
    graph.add_edge(START, "start_play")

    # CONDITIONAL EDGE: after start_play, the router decides where to go.
    # add_conditional_edges(source_node, router_function)
    # The router returns "cricket" or "badminton" — LangGraph routes accordingly.
    graph.add_conditional_edges("start_play", random_play)

    # Both sport nodes end the graph
    graph.add_edge("cricket", END)
    graph.add_edge("badminton", END)

    compiled = graph.compile()

    # ── Print graph structure ────────────────────────────────────────
    logger.info("--- Demo 2: Conditional Edges (Sports Day) ---")
    logger.info("  Graph visualization:")
    save_graph_image(compiled, "demo2_conditional_edges")
    logger.info("  Graph: [START] → [start_play] → cricket OR badminton → [END]")

    for i in range(3):
        result = compiled.invoke({"graph_info": "Hey, My name is Krish"})
        logger.info("  Run %d: %s", i + 1, result["graph_info"])


# ═══════════════════════════════════════════════════════════════════════════════
# 7. DEMO 3: State Accumulation with operator.add (Append, Don't Overwrite)
# ═══════════════════════════════════════════════════════════════════════════════
#
# COMMON MISTAKE: By default, returning a state key OVERWRITES the value.
# For lists (like messages), you want to APPEND, not overwrite.
# Use Annotated[list, operator.add] to make list fields append-only.
#
# This is CRITICAL for agent loops where messages accumulate.

def demo_state_accumulation() -> None:
    """Demonstrate operator.add for append-only list state."""
    import operator
    from typing import Annotated

    from typing_extensions import TypedDict
    from langgraph.graph import END, START, StateGraph

    # ── State with append-only list ──────────────────────────────────
    # Annotated[list, operator.add] means:
    #   When a node returns {"steps": ["step_2"]},
    #   it APPENDS to the existing list: ["step_1", "step_2"]
    #   Instead of OVERWRITING to: ["step_2"]
    class PipelineState(TypedDict):
        steps: Annotated[list, operator.add]  # Append-only!
        data: str

    def step_one(state: PipelineState) -> dict:
        logger.info("  [step_one] executing")
        return {"steps": ["step_one completed"], "data": "raw data processed"}

    def step_two(state: PipelineState) -> dict:
        logger.info("  [step_two] executing")
        return {"steps": ["step_two completed"], "data": state["data"] + " → analyzed"}

    def step_three(state: PipelineState) -> dict:
        logger.info("  [step_three] executing")
        return {"steps": ["step_three completed"], "data": state["data"] + " → reported"}

    graph = StateGraph(PipelineState)
    graph.add_node("step_one", step_one)
    graph.add_node("step_two", step_two)
    graph.add_node("step_three", step_three)

    graph.add_edge(START, "step_one")
    graph.add_edge("step_one", "step_two")
    graph.add_edge("step_two", "step_three")
    graph.add_edge("step_three", END)

    compiled = graph.compile()
    result = compiled.invoke({"steps": [], "data": ""})

    logger.info("--- Demo 3: State Accumulation (operator.add) ---")
    logger.info("  Graph: [START] → [step_one] → [step_two] → [step_three] → [END]")
    logger.info("  Steps (accumulated): %s", result["steps"])
    logger.info("  Data (overwritten each time): %s", result["data"])
    # steps = ["step_one completed", "step_two completed", "step_three completed"]
    # data = " → analyzed → reported" (overwritten, not accumulated)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. DEMO 4: Multi-Branch Conditional with Explicit Mapping
# ═══════════════════════════════════════════════════════════════════════════════
#
# Sometimes you want the router to return a KEY that maps to a node name.
# add_conditional_edges(source, router, {"key": "node_name"})
#
# This is useful when router return values don't match node names.

def demo_explicit_mapping() -> None:
    """Conditional edges with explicit key-to-node mapping."""
    import random
    from typing import Literal

    from typing_extensions import TypedDict
    from langgraph.graph import END, START, StateGraph

    class OrderState(TypedDict):
        order: str
        status: str

    def receive_order(state: OrderState) -> dict:
        logger.info("  [receive_order] Order received: %s", state["order"])
        return {"status": "received"}

    def process_express(state: OrderState) -> dict:
        logger.info("  [process_express] Express shipping!")
        return {"status": "shipped_express"}

    def process_standard(state: OrderState) -> dict:
        logger.info("  [process_standard] Standard shipping.")
        return {"status": "shipped_standard"}

    def confirm(state: OrderState) -> dict:
        logger.info("  [confirm] Order confirmed: %s", state["status"])
        return {"status": state["status"] + " → confirmed"}

    # Router returns "express" or "standard" — these are KEYS, not node names
    def shipping_router(state: OrderState) -> Literal["express", "standard"]:
        if random.random() > 0.5:
            return "express"
        return "standard"

    graph = StateGraph(OrderState)
    graph.add_node("receive", receive_order)
    graph.add_node("express_ship", process_express)
    graph.add_node("standard_ship", process_standard)
    graph.add_node("confirm", confirm)

    graph.add_edge(START, "receive")

    # Explicit mapping: router returns "express" → goes to "express_ship" node
    graph.add_conditional_edges(
        "receive",
        shipping_router,
        {"express": "express_ship", "standard": "standard_ship"},
    )

    graph.add_edge("express_ship", "confirm")
    graph.add_edge("standard_ship", "confirm")
    graph.add_edge("confirm", END)

    compiled = graph.compile()

    logger.info("--- Demo 4: Explicit Mapping ---")
    logger.info("  Graph visualization:")
    save_graph_image(compiled, "demo4_explicit_mapping")
    logger.info("  Graph: [START] → [receive] → express OR standard → [confirm] → [END]")

    for i in range(3):
        result = compiled.invoke({"order": f"Order-{i+1}", "status": ""})
        logger.info("  Run %d: %s → %s", i + 1, result["order"], result["status"])


# ═══════════════════════════════════════════════════════════════════════════════
# 9. COMMON MISTAKES & GOTCHAS
# ═══════════════════════════════════════════════════════════════════════════════
#
# Mistake                          | Fix
# ─────────────────────────────────|──────────────────────────────────
# Forgetting .compile()            | Always compile before invoke
# Using plain dict for state       | Always use TypedDict
# List state overwrites            | Use Annotated[list, operator.add]
# Router returns wrong string      | Must match node names or mapping keys
# No edge from node to END         | Graph hangs — always connect to END
# Forgetting set_entry_point/START | Graph doesn't know where to begin


# ═══════════════════════════════════════════════════════════════════════════════
# 10. WHERE THIS CONNECTS (Concept Linking Map)
# ═══════════════════════════════════════════════════════════════════════════════
#
# State (TypedDict) → shared memory → read/written by Nodes
# Nodes (functions) → connected by → Edges (static or conditional)
# Router functions → return node names → used by conditional edges
# StateGraph → .compile() → CompiledGraph → .invoke() → result
# operator.add → makes list fields → append-only (critical for messages)
# START → entry point | END → terminal node


# ═══════════════════════════════════════════════════════════════════════════════
# 🎯 INTERVIEW QUESTIONS
# ═══════════════════════════════════════════════════════════════════════════════
#
# THEORETICAL:
# Q: What is the difference between a static edge and a conditional edge?
# A: A static edge (add_edge) ALWAYS routes from node A to node B — the
#    path is fixed at build time. A conditional edge (add_conditional_edges)
#    uses a ROUTER FUNCTION that examines the current state and returns
#    a string indicating which node to go to next. This enables dynamic
#    branching and is the foundation of agent decision-making in LangGraph.
#
# HANDS-ON:
# Q: Build a 3-node graph: "check_weather" reads state, if weather is "rain"
#    route to "stay_home", if "sunny" route to "go_park". Both end the graph.
#
# SOLUTION:
#   from typing import Literal
#   from typing_extensions import TypedDict
#   from langgraph.graph import StateGraph, START, END
#
#   class State(TypedDict):
#       weather: str
#       plan: str
#
#   def check_weather(state):
#       return {"plan": f"Weather is {state['weather']}"}
#
#   def stay_home(state):
#       return {"plan": state["plan"] + " → Staying home"}
#
#   def go_park(state):
#       return {"plan": state["plan"] + " → Going to the park!"}
#
#   def weather_router(state) -> Literal["stay_home", "go_park"]:
#       return "stay_home" if state["weather"] == "rain" else "go_park"
#
#   g = StateGraph(State)
#   g.add_node("check_weather", check_weather)
#   g.add_node("stay_home", stay_home)
#   g.add_node("go_park", go_park)
#   g.add_edge(START, "check_weather")
#   g.add_conditional_edges("check_weather", weather_router)
#   g.add_edge("stay_home", END)
#   g.add_edge("go_park", END)
#   compiled = g.compile()
#   print(compiled.invoke({"weather": "sunny", "plan": ""}))
#   # {'weather': 'sunny', 'plan': 'Weather is sunny → Going to the park!'}


# ═══════════════════════════════════════════════════════════════════════════════
# 📝 QUICK RECAP
# ═══════════════════════════════════════════════════════════════════════════════
#
# 1. State (TypedDict) is shared memory. Nodes read it, update it, return it.
#    Use Annotated[list, operator.add] for append-only list fields.
#
# 2. Nodes are plain Python functions. They take state, do work, return updates.
#    No LLM needed — any logic works (this is what makes LangGraph flexible).
#
# 3. Edges connect nodes. Static edges are fixed paths. Conditional edges use
#    router functions to make decisions. This is the foundation of agent loops.


# ═══════════════════════════════════════════════════════════════════════════════
# 🚀 Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("🏗️ LANGGRAPH LESSON 2 — Hands-On: State, Nodes, Edges")
    logger.info("=" * 70)

    logger.info("\n🔹 Demo 1: Simplest Graph (two nodes, static edges)")
    demo_simplest_graph()

    logger.info("\n🔹 Demo 2: Conditional Edges (sports day — random routing)")
    demo_conditional_edges()

    logger.info("\n🔹 Demo 3: State Accumulation (operator.add)")
    demo_state_accumulation()

    logger.info("\n🔹 Demo 4: Explicit Mapping (router keys → node names)")
    demo_explicit_mapping()

    logger.info("\n" + "=" * 70)
    logger.info("✅ Lesson 2 complete — State, Nodes, Edges, Conditional Routing")
    logger.info("You can now build graphs with branching and state accumulation.")
    logger.info("Next: Lesson 3 — Building a Simple LLM Chatbot with LangGraph")
    logger.info("=" * 70)
