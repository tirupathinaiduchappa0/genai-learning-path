"""
🔗 LangGraph Workflows — Lesson 1: Prompt Chaining (Level 7)

═══════════════════════════════════════════════════════════════════
1. CONCEPT: Prompt Chaining — ONE-LINE DEFINITION
═══════════════════════════════════════════════════════════════════

Prompt Chaining is a workflow pattern where a complex task is broken
into a SEQUENCE of smaller, focused LLM calls — each step's output
feeds into the next step's input, with optional QUALITY GATES that
can retry or reroute the flow.

═══════════════════════════════════════════════════════════════════
2. WHY PROMPT CHAINING EXISTS
═══════════════════════════════════════════════════════════════════

Single prompts FAIL for complex tasks. Why?
    - One prompt can't draft, review, revise, AND polish.
    - Quality degrades when you ask an LLM to do too much at once.
    - No way to CHECK intermediate results before proceeding.

Prompt chaining solves this by:
    - Breaking the task into discrete, manageable steps.
    - Allowing QUALITY GATES between steps (pass/fail checks).
    - Enabling RETRY LOOPS — if a step fails the gate, redo it.
    - Each step gets a focused, high-quality prompt.

═══════════════════════════════════════════════════════════════════
3. REAL-WORLD ANALOGY — A BOOK EDITOR'S PROCESS
═══════════════════════════════════════════════════════════════════

Think of how a BOOK EDITOR works:

    1. DRAFT    → Author writes the first draft.
    2. REVIEW   → Editor checks for plot holes, tone, grammar.
    3. REVISE   → If issues found, author rewrites. If not, proceed.
    4. POLISH   → Final pass for style, flow, and readability.
    5. PUBLISH  → Done!

    The REVIEW step is a QUALITY GATE.
    If the draft fails review → loop back to DRAFT (retry).
    If the draft passes review → move to POLISH.

    This is EXACTLY how prompt chaining works in LangGraph.

═══════════════════════════════════════════════════════════════════
4. ASCII GRAPH STRUCTURES
═══════════════════════════════════════════════════════════════════

Demo 1 — Story Generation Pipeline:

    [START]
       |
    [generate_story]
       |
    [check_conflict]  ← QUALITY GATE
       |
       ├── "Pass" ──→ [improve_story] → [polish_story] → [END]
       |
       └── "Fail" ──→ [generate_story]  (RETRY LOOP!)

Demo 2 — Email Drafting Pipeline:

    [START]
       |
    [draft_email]
       |
    [check_tone]  ← QUALITY GATE
       |
       ├── "Pass" ──→ [finalize_email] → [END]
       |
       └── "Fail" ──→ [fix_tone] → [finalize_email] → [END]

═══════════════════════════════════════════════════════════════════

🔄 LANGCHAIN vs LANGGRAPH — Prompt Chaining
LangChain : SequentialChain — linear only, no loops, no gates.
            If step 2 fails, the whole chain fails. No retry.
LangGraph  : StateGraph with conditional edges — supports CYCLES.
            Quality gates route to retry or next step.
            Full control over the chaining logic.

LangChain : RunnableSequence (LCEL) — pipe operator, one direction.
LangGraph  : Nodes + conditional edges — bidirectional, cyclic.

═══════════════════════════════════════════════════════════════════
5. WHERE IT FITS IN THE LANGGRAPH ARCHITECTURE
═══════════════════════════════════════════════════════════════════

    Prompt Chaining is a WORKFLOW PATTERN built on top of:
        StateGraph → Nodes → Edges → Conditional Edges → Cycles

    It's the simplest workflow pattern — sequential steps with gates.
    More advanced patterns (parallelization, orchestrator-workers)
    build on top of this foundation.

    Concept Linking:
        StateGraph → compiles to → CompiledGraph
        Nodes → read/write → State (TypedDict)
        Conditional Edges → use → gate functions → return → node names
        Cycles → enable → retry loops (what makes LangGraph unique)

HOW TO RUN:
    $ python src/Workflows/01_prompt_chaining.py

Author: GenAI Learner
"""

import logging
import os
from typing import Literal

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

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



# ═══════════════════════════════════════════════════════════════════════════════
# 6. DEMO 1: Story Generation Pipeline (Prompt Chaining with Retry Loop)
# ═══════════════════════════════════════════════════════════════════════════════
#
# This is the CORE prompt chaining pattern from the notebook.
# A story is generated, checked for "conflict" (quality gate),
# and either retried or improved and polished.
#
# KEY INSIGHT: The check_conflict node is a GATE — it doesn't call the LLM.
# It inspects the story and returns "Pass" or "Fail".
# The conditional edge routes based on the gate's return value.
#
# This demonstrates the RETRY LOOP pattern:
#   generate → gate → (Fail → generate again) or (Pass → improve → polish)
#
# WHY THIS MATTERS:
#   In production, you ALWAYS need quality checks between steps.
#   If the LLM produces bad output, you retry — not crash.
#   LangGraph makes this trivial with conditional edges + cycles.

def demo_story_pipeline() -> None:
    """Story generation pipeline: generate → gate → improve → polish."""

    # ── State: holds the topic and story at each stage ───────────────
    class StoryState(TypedDict):
        topic: str              # The topic to write about
        story: str              # The initial generated story
        improved_story: str     # The story after improvement
        final_story: str        # The final polished story

    # ── LLM: ChatGroq with creative temperature ─────────────────────
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7, max_tokens=512)

    # ── Node 1: generate_story — creates the initial draft ──────────
    def generate_story(state: StoryState) -> dict:
        """Generate an initial story draft based on the topic."""
        logger.info("  [generate_story] Generating story for topic: %s", state["topic"])
        response = llm.invoke(
            f"Write a short 2-3 sentence story about: {state['topic']}. "
            f"Make it engaging and creative."
        )
        story = response.content
        logger.info("  [generate_story] Generated: %s", story[:150])
        return {"story": story}

    # ── Node 2: check_conflict — QUALITY GATE (no LLM call) ────────
    # This is a GATE node. It inspects the story and decides:
    #   - "Pass" if the story has NO "?" or "!" (considered good)
    #   - "Fail" if the story HAS "?" or "!" (needs retry)
    #
    # NOTE: This is a SIMPLE heuristic for demonstration.
    # In production, you'd use a more sophisticated check.
    def check_conflict(state: StoryState) -> Literal["Pass", "Fail"]:
        """Gate: check if the story contains '?' or '!' — if yes, Fail (retry)."""
        story = state["story"]
        if "?" in story or "!" in story:
            logger.info("  [check_conflict] FAIL — story contains '?' or '!', retrying...")
            return "Fail"
        else:
            logger.info("  [check_conflict] PASS — story is clean, proceeding...")
            return "Pass"

    # ── Node 3: improve_story — enhances the draft ──────────────────
    def improve_story(state: StoryState) -> dict:
        """Improve the story by adding depth and detail."""
        logger.info("  [improve_story] Improving the story...")
        response = llm.invoke(
            f"Improve the following story by adding more depth, vivid details, "
            f"and emotional resonance. Keep it concise (3-4 sentences):\n\n"
            f"{state['story']}"
        )
        improved = response.content
        logger.info("  [improve_story] Improved: %s", improved[:150])
        return {"improved_story": improved}

    # ── Node 4: polish_story — final refinement ─────────────────────
    def polish_story(state: StoryState) -> dict:
        """Polish the story for publication-ready quality."""
        logger.info("  [polish_story] Polishing the story...")
        response = llm.invoke(
            f"Polish the following story for publication. Fix any grammar issues, "
            f"improve flow, and make it compelling. Keep it concise:\n\n"
            f"{state['improved_story']}"
        )
        final = response.content
        logger.info("  [polish_story] Final: %s", final[:150])
        return {"final_story": final}

    # ── Build the graph ──────────────────────────────────────────────
    builder = StateGraph(StoryState)

    # Register all nodes
    builder.add_node("generate_story", generate_story)
    builder.add_node("improve_story", improve_story)
    builder.add_node("polish_story", polish_story)

    # Entry edge: START → generate_story
    builder.add_edge(START, "generate_story")

    # CONDITIONAL EDGE: generate_story → gate → (Pass → improve) or (Fail → generate)
    # This is the RETRY LOOP — if the gate fails, we loop back to generate.
    builder.add_conditional_edges(
        "generate_story",
        check_conflict,
        {"Pass": "improve_story", "Fail": "generate_story"},
    )

    # Static edges: improve → polish → END
    builder.add_edge("improve_story", "polish_story")
    builder.add_edge("polish_story", END)

    # Compile with recursion_limit to prevent infinite retry loops
    # IMPORTANT: Always set recursion_limit on cyclic graphs!
    graph = builder.compile()

    # Save the graph visualization
    save_graph_image(graph, "workflow01_story_pipeline")

    # ── Invoke the graph ─────────────────────────────────────────────
    logger.info("--- Demo 1: Story Generation Pipeline ---")
    result = graph.invoke(
        {"topic": "Agentic AI Systems", "story": "", "improved_story": "", "final_story": ""},
        config={"recursion_limit": 10},
    )

    # Log each step's output
    logger.info("  Topic: %s", result["topic"])
    logger.info("  Initial Story: %s", result["story"][:200])
    logger.info("  Improved Story: %s", result["improved_story"][:200])
    logger.info("  Final Story: %s", result["final_story"][:200])
    logger.info("  Flow: START → generate → (gate) → improve → polish → END")
    logger.info("  Pattern: Prompt Chaining with RETRY LOOP via quality gate.")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. DEMO 2: Production Email Drafting Pipeline (Real Business Use Case)
# ═══════════════════════════════════════════════════════════════════════════════
#
# A more PRACTICAL example of prompt chaining in a real business scenario.
# The pipeline: draft an email → check tone → fix if needed → finalize.
#
# This shows prompt chaining in a REAL business use case:
#   - A user requests an email (e.g., "apologize to client for delay")
#   - The LLM drafts the email
#   - A tone checker (gate) verifies it's professional
#   - If tone is bad → fix_tone node rewrites it
#   - If tone is good → finalize_email polishes it
#
# KEY DIFFERENCE from Demo 1:
#   Demo 1 uses a RETRY LOOP (fail → regenerate from scratch).
#   Demo 2 uses a FIX BRANCH (fail → fix the existing draft, not regenerate).
#   Both are valid prompt chaining patterns.

def demo_email_pipeline() -> None:
    """Email drafting pipeline: draft → check tone → fix/finalize."""

    # ── State: holds the email at each stage ─────────────────────────
    class EmailState(TypedDict):
        request: str        # The user's email request
        draft: str          # The initial email draft
        tone_check: str     # Result of tone check: "Pass" or "Fail"
        final_email: str    # The finalized email

    # ── LLM: ChatGroq with creative temperature ─────────────────────
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7, max_tokens=512)

    # ── Node 1: draft_email — creates the initial email draft ───────
    def draft_email(state: EmailState) -> dict:
        """Draft an email based on the user's request."""
        logger.info("  [draft_email] Drafting email for: %s", state["request"][:80])
        response = llm.invoke(
            f"Draft a professional email based on this request: {state['request']}. "
            f"Keep it concise and professional."
        )
        draft = response.content
        logger.info("  [draft_email] Draft: %s", draft[:150])
        return {"draft": draft}

    # ── Node 2: check_tone — QUALITY GATE ────────────────────────────
    # Simple heuristic: checks if the draft contains informal words.
    # In production, you'd use sentiment analysis or another LLM call.
    def check_tone(state: EmailState) -> Literal["Pass", "Fail"]:
        """Gate: check if the email tone is professional."""
        draft = state["draft"].lower()
        informal_words = ["hey", "gonna", "wanna", "lol", "btw", "asap", "fyi"]
        for word in informal_words:
            if word in draft:
                logger.info("  [check_tone] FAIL — found informal word: '%s'", word)
                return "Fail"
        logger.info("  [check_tone] PASS — tone is professional.")
        return "Pass"

    # ── Node 3: fix_tone — rewrites the draft with better tone ──────
    def fix_tone(state: EmailState) -> dict:
        """Fix the tone of the email to be more professional."""
        logger.info("  [fix_tone] Fixing email tone...")
        response = llm.invoke(
            f"The following email draft has an unprofessional tone. "
            f"Rewrite it to be formal, polite, and business-appropriate:\n\n"
            f"{state['draft']}"
        )
        fixed = response.content
        logger.info("  [fix_tone] Fixed: %s", fixed[:150])
        return {"draft": fixed}

    # ── Node 4: finalize_email — final polish ────────────────────────
    def finalize_email(state: EmailState) -> dict:
        """Finalize the email for sending."""
        logger.info("  [finalize_email] Finalizing email...")
        response = llm.invoke(
            f"Finalize the following email. Ensure proper greeting, body, "
            f"and sign-off. Make it ready to send:\n\n{state['draft']}"
        )
        final = response.content
        logger.info("  [finalize_email] Final: %s", final[:150])
        return {"final_email": final}

    # ── Build the graph ──────────────────────────────────────────────
    builder = StateGraph(EmailState)

    # Register all nodes
    builder.add_node("draft_email", draft_email)
    builder.add_node("fix_tone", fix_tone)
    builder.add_node("finalize_email", finalize_email)

    # Entry edge: START → draft_email
    builder.add_edge(START, "draft_email")

    # CONDITIONAL EDGE: draft_email → gate → (Pass → finalize) or (Fail → fix_tone)
    builder.add_conditional_edges(
        "draft_email",
        check_tone,
        {"Pass": "finalize_email", "Fail": "fix_tone"},
    )

    # After fixing tone → finalize
    builder.add_edge("fix_tone", "finalize_email")

    # finalize → END
    builder.add_edge("finalize_email", END)

    # Compile the graph
    graph = builder.compile()

    # Save the graph visualization
    save_graph_image(graph, "workflow01_email_pipeline")

    # ── Invoke the graph ─────────────────────────────────────────────
    logger.info("--- Demo 2: Production Email Drafting Pipeline ---")
    result = graph.invoke(
        {
            "request": "Write an apology email to a client for a project delay of 2 weeks",
            "draft": "",
            "tone_check": "",
            "final_email": "",
        }
    )

    # Log each step's output
    logger.info("  Request: %s", result["request"])
    logger.info("  Draft: %s", result["draft"][:200])
    logger.info("  Final Email: %s", result["final_email"][:200])
    logger.info("  Flow: START → draft → (gate) → fix_tone/finalize → END")
    logger.info("  Pattern: Prompt Chaining with FIX BRANCH via quality gate.")


# ═══════════════════════════════════════════════════════════════════════════════
# 8. WHEN TO USE PROMPT CHAINING
# ═══════════════════════════════════════════════════════════════════════════════
#
# USE prompt chaining when:
#   ✅ The task has MULTIPLE DISTINCT STEPS (draft → review → revise → publish)
#   ✅ You need QUALITY GATES between steps (check before proceeding)
#   ✅ Each step benefits from a FOCUSED prompt (one job per LLM call)
#   ✅ You need RETRY LOGIC (if a step fails, redo it)
#   ✅ The output of one step is the INPUT of the next
#
# DON'T use prompt chaining when:
#   ❌ The task is simple enough for a single prompt
#   ❌ Steps are INDEPENDENT (use parallelization instead)
#   ❌ You need dynamic routing to many different paths (use orchestrator)


# ═══════════════════════════════════════════════════════════════════════════════
# 9. COMMON MISTAKES & GOTCHAS
# ═══════════════════════════════════════════════════════════════════════════════
#
# Mistake                                    | Fix
# ───────────────────────────────────────────|──────────────────────────────────────
# No recursion_limit on cyclic graphs        | ALWAYS set recursion_limit in config
# Gate function calls the LLM               | Gates should be FAST — use heuristics
#                                            | or lightweight checks, not full LLM calls
# Too many steps in the chain               | Keep chains SHORT (3-5 steps max)
#                                            | More steps = more latency + cost
# No fallback if retry loop exhausts limit  | Add a max-retry counter in state
# Using plain dict for state                | Always use TypedDict for type safety
# Forgetting to compile before invoke       | Always call builder.compile()
#
# ANTI-PATTERN: Putting ALL logic in one giant prompt.
#   Instead, break it into focused steps with quality gates.
#   Each step does ONE thing well.


# ═══════════════════════════════════════════════════════════════════════════════
# 10. WHY LANGGRAPH FOR THIS
# ═══════════════════════════════════════════════════════════════════════════════
#
# ✅ Cycles and loops — retry logic is NATIVE (impossible in LangChain LCEL)
# ✅ Conditional edges — quality gates route dynamically
# ✅ Stateful by design — each step reads/writes shared state
# ✅ Explicit control — you SEE every step, gate, and route
# ✅ Production-ready — recursion limits prevent infinite loops
# ✅ Composable — chains can be subgraphs inside larger workflows


# ═══════════════════════════════════════════════════════════════════════════════
# 11. WHERE THIS CONNECTS (Concept Linking Map)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Prompt Chaining → uses → StateGraph + Conditional Edges + Cycles
# Quality Gates → are → router functions that return node names
# Retry Loops → enabled by → cycles (conditional edge back to source)
# State (TypedDict) → carries data → through the entire chain
# Prompt Chaining → foundation for → more complex workflow patterns
# Prompt Chaining → simplest form of → LangGraph workflow orchestration


# ═══════════════════════════════════════════════════════════════════════════════
# 🎯 INTERVIEW QUESTIONS
# ═══════════════════════════════════════════════════════════════════════════════
#
# THEORETICAL:
# Q: What is prompt chaining, and why is it preferred over a single prompt
#    for complex tasks?
# A: Prompt chaining breaks a complex task into a sequence of smaller,
#    focused LLM calls where each step's output feeds into the next.
#    It's preferred because: (1) each step gets a focused prompt, improving
#    quality; (2) quality gates between steps catch errors early;
#    (3) retry loops allow recovery from bad outputs; (4) the pipeline
#    is debuggable — you can inspect each step's output independently.
#    A single prompt tries to do everything at once, leading to lower
#    quality and no ability to check intermediate results.
#
# HANDS-ON:
# Q: Build a prompt chaining graph with 3 nodes: "write_summary" generates
#    a summary, "check_length" (gate) checks if it's under 100 characters
#    (Pass) or over (Fail → retry), and "format_output" adds a title.
#    Use recursion_limit=5.
#
# SOLUTION:
#   from typing import Literal
#   from typing_extensions import TypedDict
#   from langgraph.graph import StateGraph, START, END
#
#   class SummaryState(TypedDict):
#       topic: str
#       summary: str
#       formatted: str
#
#   def write_summary(state):
#       # In production, call an LLM here
#       return {"summary": f"A brief overview of {state['topic']}."}
#
#   def check_length(state) -> Literal["Pass", "Fail"]:
#       return "Pass" if len(state["summary"]) < 100 else "Fail"
#
#   def format_output(state):
#       return {"formatted": f"# Summary\n{state['summary']}"}
#
#   builder = StateGraph(SummaryState)
#   builder.add_node("write_summary", write_summary)
#   builder.add_node("format_output", format_output)
#   builder.add_edge(START, "write_summary")
#   builder.add_conditional_edges("write_summary", check_length,
#       {"Pass": "format_output", "Fail": "write_summary"})
#   builder.add_edge("format_output", END)
#   graph = builder.compile()
#   result = graph.invoke(
#       {"topic": "AI", "summary": "", "formatted": ""},
#       config={"recursion_limit": 5}
#   )
#   print(result["formatted"])


# ═══════════════════════════════════════════════════════════════════════════════
# 📝 QUICK RECAP
# ═══════════════════════════════════════════════════════════════════════════════
#
# 1. Prompt chaining breaks complex tasks into sequential LLM calls.
#    Each step has a focused prompt. Quality gates check intermediate results.
#
# 2. LangGraph enables RETRY LOOPS via conditional edges + cycles.
#    If a gate fails, the graph routes back to regenerate. Always set
#    recursion_limit on cyclic graphs to prevent infinite loops.
#
# 3. Two patterns: RETRY (fail → regenerate from scratch) and
#    FIX BRANCH (fail → fix the existing output). Choose based on
#    whether the output is salvageable or needs a fresh start.


# ═══════════════════════════════════════════════════════════════════════════════
# 🚀 Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("🔗 LANGGRAPH WORKFLOWS — Lesson 1: Prompt Chaining")
    logger.info("=" * 70)

    logger.info("\n🔹 Demo 1: Story Generation Pipeline (Retry Loop Pattern)")
    demo_story_pipeline()

    logger.info("\n🔹 Demo 2: Production Email Drafting Pipeline (Fix Branch Pattern)")
    demo_email_pipeline()

    logger.info("\n" + "=" * 70)
    logger.info("✅ Lesson 1 complete — Prompt Chaining with Quality Gates")
    logger.info("You can now build sequential workflows with retry loops.")
    logger.info("Next: Lesson 2 — Parallelization (fan-out / fan-in)")
    logger.info("=" * 70)
