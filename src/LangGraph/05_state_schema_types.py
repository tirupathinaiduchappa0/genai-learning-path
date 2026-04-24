"""
🏗️ LangGraph Lesson 5 — State Schema Types: TypedDict vs dataclass vs Pydantic (Level 2)

═══════════════════════════════════════════════════════════════════
1. CONCEPT: State Schema Types — ONE-LINE DEFINITION
═══════════════════════════════════════════════════════════════════

A State Schema defines the SHAPE of data flowing through your graph.
LangGraph supports THREE ways to define it: TypedDict, dataclass, and Pydantic BaseModel.

═══════════════════════════════════════════════════════════════════
2. WHY STATE SCHEMA TYPES EXIST
═══════════════════════════════════════════════════════════════════

When you build a StateGraph, you MUST tell LangGraph what your state looks like.
But Python offers multiple ways to define structured data:

    TypedDict  → dict with type hints (most common in LangGraph)
    dataclass  → Python class with auto-generated __init__ (dot access)
    Pydantic   → BaseModel with RUNTIME VALIDATION (production standard)

The question is: which one should you use, and WHEN?

    TypedDict:  Fast to write, no validation, dict-style access.
    dataclass:  Dot notation, no validation, object-style invocation.
    Pydantic:   Dot notation, RUNTIME VALIDATION, catches bad data BEFORE it enters your graph.

The KEY difference: only Pydantic ENFORCES types at runtime.
TypedDict and dataclass type hints are IGNORED at runtime — they're just for IDEs and mypy.

═══════════════════════════════════════════════════════════════════
3. WHERE IT FITS IN THE LANGGRAPH ARCHITECTURE
═══════════════════════════════════════════════════════════════════

    State Schema (TypedDict / dataclass / Pydantic)
        ↓
    Passed to StateGraph(schema) at graph construction
        ↓
    Nodes READ state and RETURN dict updates
        ↓
    Reducers (Annotated[...]) decide HOW updates merge
        ↓
    Graph compiles and runs with that schema

    The schema is the FIRST thing you define. Everything else depends on it.

═══════════════════════════════════════════════════════════════════
4. REAL-WORLD ANALOGY
═══════════════════════════════════════════════════════════════════

Think of a SPORTS DAY REGISTRATION FORM:

    TypedDict:
        A paper form with labeled fields (Name, Game).
        You WRITE the labels, but nobody checks if you wrote a number
        in the "Name" field. It's just a suggestion.

    dataclass:
        A slightly fancier form — pre-printed with fields.
        Still no one checks your answers. You just fill it in.

    Pydantic:
        An ONLINE form with VALIDATION.
        If you type a number in the "Name" field, it REJECTS your submission
        with a clear error message. This is what production systems need.

═══════════════════════════════════════════════════════════════════
5. ASCII GRAPH STRUCTURE (same for all 3 demos)
═══════════════════════════════════════════════════════════════════

    [START]
       |
    [play_game]
       |
       ├── (random: 50/50) ──→ [cricket] ──→ [END]
       |
       └── (random: 50/50) ──→ [badminton] ──→ [END]

═══════════════════════════════════════════════════════════════════

🔄 LANGCHAIN vs LANGGRAPH
LangChain : State is implicit — data flows through chain inputs/outputs
LangGraph  : State is EXPLICIT — you define a schema, all nodes share it

LangChain : No built-in state validation
LangGraph  : Pydantic schema gives you runtime validation for free

HOW TO RUN:
    $ python src/LangGraph/05_state_schema_types.py

Author: GenAI Learner
Date: 2025-07-15
"""

import logging
import os
import random

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
# 6. DEMO 1: TypedDict State Schema — Dict-Style Access, NO Validation
# ═══════════════════════════════════════════════════════════════════════════════
#
# TypedDict is the MOST COMMON state schema in LangGraph.
# It's a dict with type hints — but those hints are NOT enforced at runtime.
#
# KEY BEHAVIORS:
#   - Access: state["name"] (dict-style bracket notation)
#   - Invocation: graph.invoke({"name": "Krish"}) — pass a plain dict
#   - Validation: NONE — passing name=123 works silently, no error!
#   - Type hints: Only for IDEs and mypy, ignored at runtime

def demo_typeddict_state() -> None:
    """TypedDict state — dict access, no runtime validation."""
    from typing import Literal

    from typing_extensions import TypedDict
    from langgraph.graph import END, START, StateGraph

    # ── Define state schema with TypedDict ───────────────────────────
    class TypedDictState(TypedDict):
        name: str                                  # Type hint: str (NOT enforced)
        game: Literal["cricket", "badminton"]      # Type hint: Literal (NOT enforced)

    # ── Node functions — access state with BRACKET notation ──────────
    def play_game(state: TypedDictState) -> dict:
        """Entry node: appends ' wants to play' to the name."""
        logger.info("    [play_game] called — state['name'] = %s", state["name"])
        return {"name": state["name"] + " wants to play"}  # dict-style READ: state["name"]

    def cricket(state: TypedDictState) -> dict:
        """Cricket branch: appends ' cricket' to the name."""
        logger.info("    [cricket] called")
        return {"name": state["name"] + " cricket", "game": "cricket"}

    def badminton(state: TypedDictState) -> dict:
        """Badminton branch: appends ' badminton' to the name."""
        logger.info("    [badminton] called")
        return {"name": state["name"] + " badminton", "game": "badminton"}

    # ── Router function — random 50/50 split ─────────────────────────
    def decide_play(state: TypedDictState) -> Literal["cricket", "badminton"]:
        """Randomly route to cricket or badminton."""
        return "cricket" if random.random() < 0.5 else "badminton"

    # ── Build the graph ──────────────────────────────────────────────
    builder = StateGraph(TypedDictState)
    builder.add_node("play_game", play_game)
    builder.add_node("cricket", cricket)
    builder.add_node("badminton", badminton)

    builder.add_edge(START, "play_game")
    builder.add_conditional_edges("play_game", decide_play)  # Random routing
    builder.add_edge("cricket", END)
    builder.add_edge("badminton", END)

    graph = builder.compile()
    save_graph_image(graph, "lesson05_typeddict_state")

    # ── Run 1: Valid input (string name) ─────────────────────────────
    logger.info("  Run 1: Valid input — name='Krish'")
    result = graph.invoke({"name": "Krish"})
    logger.info("  Result: %s", result)

    # ── Run 2: INVALID input — name=123 (integer, not string) ────────
    # TypedDict does NOT validate at runtime — 123 enters the graph silently!
    # The graph ACCEPTS the bad input. It only fails later when a node tries
    # to concatenate int + str. The point: NO ValidationError at the gate.
    logger.info("  Run 2: INVALID input — name=123 (should be str)")
    try:
        result_invalid = graph.invoke({"name": 123})
        logger.info("  Result: %s", result_invalid)
        logger.info("  ⚠️  No validation error! TypedDict accepted name=123 silently.")
    except TypeError as e:
        logger.warning("  ⚠️  TypeError (not ValidationError!): %s", e)
        logger.info("  ⚠️  TypedDict let name=123 INTO the graph — it only failed")
        logger.info("     during string concatenation. NO upfront validation!")
        logger.info("  ⚠️  Compare: Pydantic would have REJECTED this at the gate.")



# ═══════════════════════════════════════════════════════════════════════════════
# 7. DEMO 2: Dataclass State Schema — Dot Access, NO Validation
# ═══════════════════════════════════════════════════════════════════════════════
#
# Python's dataclasses provide another way to define structured data.
# They offer dot notation access (state.name instead of state["name"]).
#
# KEY BEHAVIORS:
#   - Access: state.name (dot notation — object-style)
#   - Invocation: graph.invoke(DataClassState(name="Krish", game="cricket"))
#                 — pass an OBJECT, not a dict!
#   - Validation: NONE — passing name=123 works silently, no error!
#   - Type hints: Only for IDEs and mypy, ignored at runtime
#
# IMPORTANT: Nodes still RETURN a dict (not a dataclass object).
#            You READ with dot notation, but WRITE with dict.

def demo_dataclass_state() -> None:
    """Dataclass state — dot access, object invocation, no runtime validation."""
    from dataclasses import dataclass
    from typing import Literal

    from langgraph.graph import END, START, StateGraph

    # ── Define state schema with @dataclass ──────────────────────────
    @dataclass
    class DataClassState:
        name: str                                  # Type hint: str (NOT enforced)
        game: Literal["cricket", "badminton"]      # Type hint: Literal (NOT enforced)

    # ── Node functions — access state with DOT notation ──────────────
    def play_game(state: DataClassState) -> dict:
        """Entry node: appends ' wants to play' to the name."""
        logger.info("    [play_game] called — state.name = %s", state.name)
        return {"name": state.name + " wants to play"}  # DOT notation READ: state.name

    def cricket(state: DataClassState) -> dict:
        """Cricket branch: appends ' cricket' to the name."""
        logger.info("    [cricket] called")
        return {"name": state.name + " cricket", "game": "cricket"}  # RETURN is still a dict!

    def badminton(state: DataClassState) -> dict:
        """Badminton branch: appends ' badminton' to the name."""
        logger.info("    [badminton] called")
        return {"name": state.name + " badminton", "game": "badminton"}

    # ── Router function — random 50/50 split ─────────────────────────
    def decide_play(state: DataClassState) -> Literal["cricket", "badminton"]:
        """Randomly route to cricket or badminton."""
        return "cricket" if random.random() < 0.5 else "badminton"

    # ── Build the graph ──────────────────────────────────────────────
    builder = StateGraph(DataClassState)
    builder.add_node("play_game", play_game)
    builder.add_node("cricket", cricket)
    builder.add_node("badminton", badminton)

    builder.add_edge(START, "play_game")
    builder.add_conditional_edges("play_game", decide_play)
    builder.add_edge("cricket", END)
    builder.add_edge("badminton", END)

    graph = builder.compile()
    save_graph_image(graph, "lesson05_dataclass_state")

    # ── Run 1: Valid input — pass a DataClassState OBJECT ────────────
    # NOTE: Invocation uses the class constructor, NOT a plain dict!
    logger.info("  Run 1: Valid input — DataClassState(name='Krish', game='cricket')")
    result = graph.invoke(DataClassState(name="Krish", game="cricket"))
    logger.info("  Result: %s", result)

    # ── Run 2: INVALID input — name=123 (integer, not string) ────────
    # Dataclass does NOT validate at runtime — 123 enters the graph silently!
    # Same behavior as TypedDict: no ValidationError at the gate.
    logger.info("  Run 2: INVALID input — DataClassState(name=123, game='cricket')")
    try:
        result_invalid = graph.invoke(DataClassState(name=123, game="cricket"))
        logger.info("  Result: %s", result_invalid)
        logger.info("  ⚠️  No validation error! Dataclass accepted name=123 silently.")
    except TypeError as e:
        logger.warning("  ⚠️  TypeError (not ValidationError!): %s", e)
        logger.info("  ⚠️  Dataclass let name=123 INTO the graph — same as TypedDict.")
        logger.info("     No upfront validation! Only fails during string operations.")
        logger.info("  ⚠️  Compare: Pydantic would have REJECTED this at the gate.")



# ═══════════════════════════════════════════════════════════════════════════════
# 8. DEMO 3: Pydantic State Schema — Dot Access, RUNTIME VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════
#
# Pydantic BaseModel is the PRODUCTION STANDARD for state schemas.
# It's the ONLY option that ENFORCES type hints at runtime.
#
# KEY BEHAVIORS:
#   - Access: state.name (dot notation — like dataclass)
#   - Invocation: graph.invoke({"name": "Krish"}) — pass a dict (like TypedDict!)
#   - Validation: YES! Passing name=123 RAISES ValidationError at runtime!
#   - Type hints: ENFORCED — Pydantic validates every field on construction
#
# THIS IS THE KEY DIFFERENCE:
#   TypedDict + dataclass → type hints are decorative (not enforced)
#   Pydantic BaseModel    → type hints are CONTRACTS (enforced at runtime)

def demo_pydantic_state() -> None:
    """Pydantic state — dot access, dict invocation, RUNTIME VALIDATION."""
    from pydantic import BaseModel, ValidationError
    from langgraph.graph import END, START, StateGraph

    # ── Define state schema with Pydantic BaseModel ──────────────────
    class PydanticState(BaseModel):
        name: str  # Type hint: str — ENFORCED at runtime by Pydantic!

    # ── Simple node function — access state with DOT notation ────────
    def example_node(state: PydanticState) -> dict:
        """Simple node that updates the name."""
        logger.info("    [example_node] called — state.name = %s", state.name)
        return {"name": "Hello " + state.name}  # DOT notation READ, dict WRITE

    # ── Build a simple graph ─────────────────────────────────────────
    builder = StateGraph(PydanticState)
    builder.add_node("example_node", example_node)
    builder.add_edge(START, "example_node")
    builder.add_edge("example_node", END)

    graph = builder.compile()
    save_graph_image(graph, "lesson05_pydantic_state")

    # ── Run 1: Valid input (string name) ─────────────────────────────
    logger.info("  Run 1: Valid input — {'name': 'Krish'}")
    result = graph.invoke({"name": "Krish"})
    logger.info("  Result: %s", result)

    # ── Run 2: INVALID input — name=123 (integer, not string) ────────
    # Pydantic VALIDATES at runtime — this RAISES ValidationError!
    logger.info("  Run 2: INVALID input — {'name': 123} (should be str)")
    try:
        result_invalid = graph.invoke({"name": 123})
        logger.info("  Result: %s", result_invalid)
    except ValidationError as e:
        logger.error("  🚨 ValidationError RAISED! Pydantic caught the bad input:")
        logger.error("  %s", e)
        logger.info("  ✅ This is the KEY difference — Pydantic ENFORCES types at runtime.")
        logger.info("  ✅ In production, this prevents bad data from corrupting your graph state.")
    except Exception as e:
        # Some LangGraph versions wrap the ValidationError
        logger.error("  🚨 Error RAISED! Pydantic validation caught the bad input:")
        logger.error("  %s", e)
        logger.info("  ✅ Pydantic ENFORCES types — bad data is rejected before entering the graph.")



# ═══════════════════════════════════════════════════════════════════════════════
# 9. COMPARISON TABLE
# ═══════════════════════════════════════════════════════════════════════════════
#
# Feature              | TypedDict          | dataclass          | Pydantic BaseModel
# ─────────────────────|────────────────────|────────────────────|────────────────────
# Access pattern       | state["key"]       | state.key          | state.key
# Runtime validation   | No                 | No                 | Yes ✅
# Type hints           | Yes (NOT enforced) | Yes (NOT enforced) | Yes (ENFORCED) ✅
# Default values       | Not supported      | Supported          | Supported
# Invocation style     | dict               | ClassName(...)     | dict
# Production use       | Common (simple)    | Rare               | Recommended ✅
# With reducers        | Annotated[...]     | Not standard       | Annotated[...]
# Import from          | typing_extensions  | dataclasses        | pydantic
# Error on bad type    | Silent pass        | Silent pass        | ValidationError ✅
#
# BOTTOM LINE:
#   - TypedDict: 80% of tutorials use this. Simple, fast, no overhead.
#   - dataclass: Works but offers no real advantage over TypedDict in LangGraph.
#   - Pydantic:  Production standard. Use when data integrity matters.


# ═══════════════════════════════════════════════════════════════════════════════
# 10. KEY INSIGHT — The Access Pattern Confusion
# ═══════════════════════════════════════════════════════════════════════════════
#
# This is the MOST CONFUSING part for beginners:
#
#   TypedDict:
#       state is a DICT → READ with state["key"] or state.get("key")
#
#   dataclass:
#       state is an OBJECT → READ with state.key
#
#   Pydantic:
#       state is an OBJECT → READ with state.key
#
#   BUT: ALL nodes RETURN a DICT (not an object) with updates!
#
#   Example (dataclass):
#       def my_node(state: DataClassState) -> dict:
#           name = state.name          # READ: dot notation (object)
#           return {"name": "updated"} # WRITE: dict notation (always!)
#
#   Example (TypedDict):
#       def my_node(state: TypedDictState) -> dict:
#           name = state["name"]       # READ: bracket notation (dict)
#           return {"name": "updated"} # WRITE: dict notation (always!)
#
#   Example (Pydantic):
#       def my_node(state: PydanticState) -> dict:
#           name = state.name          # READ: dot notation (object)
#           return {"name": "updated"} # WRITE: dict notation (always!)
#
#   RULE: You READ differently based on schema type.
#         You ALWAYS WRITE the same way — return a dict.
#
#   WHY? Because LangGraph's reducer system expects dict updates.
#   The dict you return is MERGED into the state using the reducer logic.


# ═══════════════════════════════════════════════════════════════════════════════
# 11. PRODUCTION RECOMMENDATION
# ═══════════════════════════════════════════════════════════════════════════════
#
# WHEN TO USE WHAT:
#
#   ┌─────────────────────────────────────────────────────────────────┐
#   │ Scenario                          │ Use This                   │
#   ├───────────────────────────────────┼────────────────────────────┤
#   │ Quick prototyping, tutorials      │ TypedDict (most common)    │
#   │ Simple graphs, no validation need │ TypedDict                  │
#   │ Production with strict validation │ Pydantic BaseModel ✅      │
#   │ API-facing graphs (user input)    │ Pydantic BaseModel ✅      │
#   │ Dataclass fans                    │ Works, but no advantage    │
#   │ Chatbot with messages             │ MessagesState (TypedDict   │
#   │                                   │   + add_messages reducer)  │
#   └───────────────────────────────────┴────────────────────────────┘
#
#   MessagesState from lesson 04 uses TypedDict + add_messages reducer.
#   It's the QUICKEST way to build a chatbot — no custom state needed.
#
#   For production systems handling user input, Pydantic is NON-NEGOTIABLE.
#   Bad data entering your graph can cause silent failures downstream.
#   Pydantic catches it at the GATE — before any node runs.


# ═══════════════════════════════════════════════════════════════════════════════
# 12. COMMON MISTAKES & GOTCHAS
# ═══════════════════════════════════════════════════════════════════════════════
#
# Mistake                                    | Fix
# ───────────────────────────────────────────|──────────────────────────────────────
# Using state.name with TypedDict            | TypedDict is a dict → use state["name"]
# Using state["name"] with dataclass         | dataclass is an object → use state.name
# Assuming TypedDict validates types         | It does NOT — use Pydantic for validation
# Returning a dataclass object from a node   | Always return a dict, not an object
# Passing a dict to a dataclass graph        | Use ClassName(field=value) for invocation
# Forgetting Pydantic catches errors early   | Wrap invoke() in try/except for user input
# Using dataclass when TypedDict suffices    | No real advantage — stick with TypedDict
#
# ANTI-PATTERN: Trusting TypedDict for data validation
#   TypedDict type hints are for DOCUMENTATION, not enforcement.
#   If you need runtime safety, use Pydantic. Period.


# ═══════════════════════════════════════════════════════════════════════════════
# 13. WHERE THIS CONNECTS (Concept Linking)
# ═══════════════════════════════════════════════════════════════════════════════
#
# State Schema (TypedDict/dataclass/Pydantic) → defines the SHAPE of state
# StateGraph(schema) → uses the schema to create the graph
# Nodes → READ state (bracket or dot) → WRITE state (always dict)
# Reducers (Annotated[...]) → decide HOW dict updates merge into state
# MessagesState → pre-built TypedDict with add_messages reducer
# Pydantic → VALIDATES state → catches bad data at the gate
#
# Lesson 04 (Reducers) → taught HOW state fields merge
# Lesson 05 (This)     → teaches WHAT TYPES define the state schema
# Next lessons          → Memory, Checkpointing, Human-in-the-loop


# ═══════════════════════════════════════════════════════════════════════════════
# 🎯 INTERVIEW QUESTIONS
# ═══════════════════════════════════════════════════════════════════════════════
#
# THEORETICAL:
# Q: What are the three ways to define a state schema in LangGraph, and what
#    is the KEY difference between them?
#
# A: LangGraph supports TypedDict, dataclass, and Pydantic BaseModel as state
#    schemas. TypedDict and dataclass both provide type hints but do NOT enforce
#    them at runtime — passing wrong types (e.g., int instead of str) works
#    silently. Pydantic BaseModel is the only option that ENFORCES types at
#    runtime via ValidationError. TypedDict uses dict-style access (state["key"]),
#    while dataclass and Pydantic use dot notation (state.key). In all three
#    cases, nodes return a plain dict with updates. For production systems,
#    Pydantic is recommended because it catches invalid data before it enters
#    the graph.
#
# HANDS-ON:
# Q: Create a LangGraph with a Pydantic state schema that has fields:
#    - email: str (must contain "@")
#    - age: int (must be >= 0)
#    Build a single-node graph that greets the user.
#    Show that passing age=-1 raises a ValidationError.
#
# SOLUTION:
#   from pydantic import BaseModel, field_validator
#   from langgraph.graph import END, START, StateGraph
#
#   class UserState(BaseModel):
#       email: str
#       age: int
#
#       @field_validator("email")
#       @classmethod
#       def email_must_contain_at(cls, v: str) -> str:
#           if "@" not in v:
#               raise ValueError("email must contain @")
#           return v
#
#       @field_validator("age")
#       @classmethod
#       def age_must_be_positive(cls, v: int) -> int:
#           if v < 0:
#               raise ValueError("age must be >= 0")
#           return v
#
#   def greet(state: UserState) -> dict:
#       return {"email": f"Hello {state.email}, age {state.age}"}
#
#   builder = StateGraph(UserState)
#   builder.add_node("greet", greet)
#   builder.add_edge(START, "greet")
#   builder.add_edge("greet", END)
#   graph = builder.compile()
#
#   # Valid: works fine
#   graph.invoke({"email": "user@example.com", "age": 25})
#
#   # Invalid: raises ValidationError — age must be >= 0
#   graph.invoke({"email": "user@example.com", "age": -1})


# ═══════════════════════════════════════════════════════════════════════════════
# 📝 QUICK RECAP
# ═══════════════════════════════════════════════════════════════════════════════
#
# 1. TypedDict = dict access (state["key"]), no validation, most common.
#    dataclass = dot access (state.key), no validation, object invocation.
#    Pydantic = dot access (state.key), RUNTIME VALIDATION, dict invocation.
#
# 2. ALL nodes return a DICT regardless of schema type. You READ differently
#    (bracket vs dot), but you ALWAYS WRITE the same way (return a dict).
#
# 3. For production, use Pydantic BaseModel — it's the only schema that
#    catches bad data BEFORE it enters your graph. TypedDict is fine for
#    prototyping and simple graphs.


# ═══════════════════════════════════════════════════════════════════════════════
# 🚀 Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("🏗️  LANGGRAPH LESSON 5 — State Schema Types")
    logger.info("   TypedDict vs dataclass vs Pydantic BaseModel")
    logger.info("=" * 70)

    logger.info("\n🔹 Demo 1: TypedDict State (dict access, NO validation)")
    demo_typeddict_state()

    logger.info("\n🔹 Demo 2: Dataclass State (dot access, NO validation)")
    demo_dataclass_state()

    logger.info("\n🔹 Demo 3: Pydantic State (dot access, RUNTIME VALIDATION)")
    demo_pydantic_state()

    logger.info("\n" + "=" * 70)
    logger.info("✅ Lesson 5 complete — State Schema Types")
    logger.info("Rule: TypedDict for prototyping, Pydantic for production")
    logger.info("Key insight: READ differs by schema, WRITE is always dict")
    logger.info("Next: Lesson 6 — Memory & Checkpointing")
    logger.info("=" * 70)
