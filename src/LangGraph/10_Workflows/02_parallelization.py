"""
⚡ LangGraph Workflows — Lesson 2: Parallelization (Level 7)

═══════════════════════════════════════════════════════════════════
1. CONCEPT: Parallelization — ONE-LINE DEFINITION
═══════════════════════════════════════════════════════════════════

Parallelization is a workflow pattern where INDEPENDENT tasks run
SIMULTANEOUSLY — multiple nodes execute at the same time, and their
results are combined by a single downstream node.

    Fan-out: One source → multiple nodes (split work)
    Fan-in:  Multiple nodes → one target (combine results)

═══════════════════════════════════════════════════════════════════
2. WHY PARALLELIZATION EXISTS
═══════════════════════════════════════════════════════════════════

Sequential execution is SLOW when tasks don't depend on each other.

    Sequential (BAD for independent tasks):
        generate_characters → generate_setting → generate_premise
        Total time: T1 + T2 + T3 = 9 seconds

    Parallel (GOOD for independent tasks):
        generate_characters ─┐
        generate_setting    ─┤→ combine
        generate_premise    ─┘
        Total time: max(T1, T2, T3) = 3 seconds (3x faster!)

Parallelization solves this by:
    - Running independent tasks AT THE SAME TIME.
    - Reducing total latency to the SLOWEST task (not the sum).
    - LangGraph handles this AUTOMATICALLY — just connect multiple
      nodes from the same source, and they run in parallel.

═══════════════════════════════════════════════════════════════════
3. REAL-WORLD ANALOGY — A RESTAURANT KITCHEN
═══════════════════════════════════════════════════════════════════

Think of a RESTAURANT KITCHEN preparing a 3-course meal:

    Sequential (one chef does everything):
        Chef makes salad → Chef grills steak → Chef bakes dessert
        Total time: 15 + 30 + 45 = 90 minutes

    Parallel (three chefs work simultaneously):
        Chef 1 makes salad    ─┐
        Chef 2 grills steak   ─┤→ Plate together → Serve!
        Chef 3 bakes dessert  ─┘
        Total time: max(15, 30, 45) = 45 minutes (2x faster!)

    The PLATING step is the Fan-in — it waits for ALL chefs to finish,
    then combines everything into the final dish.

    This is EXACTLY how parallelization works in LangGraph.

═══════════════════════════════════════════════════════════════════
4. ASCII GRAPH STRUCTURES
═══════════════════════════════════════════════════════════════════

Demo 1 — Story Elements Generator:

    [START]
       |
       ├──→ [generate_characters]  ─┐
       |                            |
       ├──→ [generate_setting]     ─┤→ [combine_elements] → [END]
       |                            |
       └──→ [generate_premise]     ─┘

    Fan-out: START → 3 parallel nodes
    Fan-in:  3 parallel nodes → combine_elements

Demo 2 — Content Pipeline:

    [START]
       |
       ├──→ [generate_title]    ─┐
       |                         |
       ├──→ [generate_summary]  ─┤→ [combine_post] → [END]
       |                         |
       └──→ [generate_tags]     ─┘

═══════════════════════════════════════════════════════════════════

🔄 LANGCHAIN vs LANGGRAPH — Parallelization
LangChain : RunnableParallel — runs multiple chains in parallel,
            but limited to simple input/output mapping.
            No shared state, no complex fan-in logic.
LangGraph  : StateGraph with fan-out/fan-in — AUTOMATIC parallelism.
            Just connect multiple nodes from START (or any node).
            LangGraph detects independence and runs them in parallel.
            Shared state collects all results for the fan-in node.

LangChain : Parallelism is a special construct (RunnableParallel).
LangGraph  : Parallelism is NATURAL — it's just how graphs work.

═══════════════════════════════════════════════════════════════════
5. WHERE IT FITS IN THE LANGGRAPH ARCHITECTURE
═══════════════════════════════════════════════════════════════════

    Parallelization is a WORKFLOW PATTERN built on top of:
        StateGraph → Nodes → Edges (fan-out / fan-in)

    HOW IT WORKS:
        When multiple nodes have edges FROM the same source node,
        LangGraph runs them ALL in parallel automatically.
        When multiple nodes have edges TO the same target node,
        LangGraph waits for ALL of them to finish before proceeding.

    No special API needed — just connect the edges correctly!

    Concept Linking:
        StateGraph → compiles to → CompiledGraph
        Fan-out edges → enable → parallel execution
        Fan-in edges → enable → result combination
        State (TypedDict) → collects results → from all parallel nodes

HOW TO RUN:
    $ python src/Workflows/02_parallelization.py

Author: GenAI Learner
"""

import logging
import os

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
# 6. DEMO 1: Story Elements Generator (Parallelization from Notebook)
# ═══════════════════════════════════════════════════════════════════════════════
#
# This is the CORE parallelization pattern from the notebook.
# Three independent tasks run in parallel:
#   - generate_characters: creates character descriptions
#   - generate_setting: creates the world/environment
#   - generate_premise: creates the story premise/plot
#
# All three connect FROM START (fan-out) and TO combine_elements (fan-in).
# LangGraph handles parallel execution AUTOMATICALLY.
#
# KEY INSIGHT: You don't need any special API for parallelism.
# Just connect multiple nodes from the same source node.
# LangGraph detects that they're independent and runs them in parallel.
#
# WHY THIS MATTERS:
#   In production, many tasks are independent — generating different
#   sections of a report, analyzing different data sources, etc.
#   Parallelization cuts total time to the SLOWEST task, not the sum.

def demo_story_elements() -> None:
    """Story elements generator: 3 parallel nodes → combine."""

    # ── State: holds the topic and each generated element ────────────
    class StoryElementsState(TypedDict):
        topic: str          # The story topic
        characters: str     # Generated character descriptions
        settings: str       # Generated world/setting description
        premises: str       # Generated story premise/plot
        story_intro: str    # Combined story introduction

    # ── LLM: ChatGroq with creative temperature ─────────────────────
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7, max_tokens=512)

    # ── Node 1: generate_characters (runs in parallel) ──────────────
    def generate_characters(state: StoryElementsState) -> dict:
        """Generate character descriptions for the story."""
        logger.info("  [generate_characters] Creating characters for: %s", state["topic"])
        response = llm.invoke(
            f"Create 2-3 interesting characters for a story about: {state['topic']}. "
            f"Give each character a name, role, and one unique trait. Be concise."
        )
        characters = response.content
        logger.info("  [generate_characters] Done: %s", characters[:120])
        return {"characters": characters}

    # ── Node 2: generate_setting (runs in parallel) ─────────────────
    def generate_setting(state: StoryElementsState) -> dict:
        """Generate the world/setting for the story."""
        logger.info("  [generate_setting] Creating setting for: %s", state["topic"])
        response = llm.invoke(
            f"Describe a vivid, immersive setting for a story about: {state['topic']}. "
            f"Include time period, location, and atmosphere. Keep it to 2-3 sentences."
        )
        settings = response.content
        logger.info("  [generate_setting] Done: %s", settings[:120])
        return {"settings": settings}

    # ── Node 3: generate_premise (runs in parallel) ─────────────────
    def generate_premise(state: StoryElementsState) -> dict:
        """Generate the story premise/plot."""
        logger.info("  [generate_premise] Creating premise for: %s", state["topic"])
        response = llm.invoke(
            f"Write a compelling story premise about: {state['topic']}. "
            f"Include the central conflict and stakes. Keep it to 2-3 sentences."
        )
        premises = response.content
        logger.info("  [generate_premise] Done: %s", premises[:120])
        return {"premises": premises}

    # ── Node 4: combine_elements (fan-in — waits for all 3) ────────
    def combine_elements(state: StoryElementsState) -> dict:
        """Combine all parallel outputs into a story introduction."""
        logger.info("  [combine_elements] Merging all elements...")
        response = llm.invoke(
            f"Using the following elements, write a compelling story introduction "
            f"(3-4 sentences) that weaves them together:\n\n"
            f"Characters:\n{state['characters']}\n\n"
            f"Setting:\n{state['settings']}\n\n"
            f"Premise:\n{state['premises']}"
        )
        intro = response.content
        logger.info("  [combine_elements] Story intro: %s", intro[:150])
        return {"story_intro": intro}

    # ── Build the graph ──────────────────────────────────────────────
    builder = StateGraph(StoryElementsState)

    # Register all nodes
    builder.add_node("generate_characters", generate_characters)
    builder.add_node("generate_setting", generate_setting)
    builder.add_node("generate_premise", generate_premise)
    builder.add_node("combine_elements", combine_elements)

    # FAN-OUT: START → 3 parallel nodes
    # When multiple edges come FROM the same source, LangGraph runs
    # the target nodes IN PARALLEL automatically.
    builder.add_edge(START, "generate_characters")
    builder.add_edge(START, "generate_setting")
    builder.add_edge(START, "generate_premise")

    # FAN-IN: 3 parallel nodes → combine_elements
    # When multiple edges go TO the same target, LangGraph WAITS
    # for ALL source nodes to finish before running the target.
    builder.add_edge("generate_characters", "combine_elements")
    builder.add_edge("generate_setting", "combine_elements")
    builder.add_edge("generate_premise", "combine_elements")

    # combine → END
    builder.add_edge("combine_elements", END)

    # Compile the graph
    graph = builder.compile()

    # Save the graph visualization
    save_graph_image(graph, "workflow02_story_elements")

    # ── Invoke the graph ─────────────────────────────────────────────
    logger.info("--- Demo 1: Story Elements Generator (Parallel) ---")
    result = graph.invoke(
        {
            "topic": "time travel",
            "characters": "",
            "settings": "",
            "premises": "",
            "story_intro": "",
        }
    )

    # Log each element and the combined result
    logger.info("  Topic: %s", result["topic"])
    logger.info("  Characters: %s", result["characters"][:200])
    logger.info("  Settings: %s", result["settings"][:200])
    logger.info("  Premises: %s", result["premises"][:200])
    logger.info("  Story Intro: %s", result["story_intro"][:200])
    logger.info("  Flow: START → [characters, setting, premise] (parallel) → combine → END")
    logger.info("  Pattern: Fan-out / Fan-in parallelization.")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. DEMO 2: Production Content Pipeline (Real Content Production Use Case)
# ═══════════════════════════════════════════════════════════════════════════════
#
# A more PRACTICAL example of parallelization in content production.
# Generate a blog post's title, summary, and tags IN PARALLEL,
# then combine them into a structured blog post.
#
# This shows parallelization in a REAL content production use case:
#   - A user provides a topic for a blog post
#   - Three independent tasks run in parallel:
#     1. generate_title: creates a catchy blog title
#     2. generate_summary: writes a blog summary/abstract
#     3. generate_tags: creates relevant tags/keywords
#   - combine_post merges all into a structured blog post
#
# KEY INSIGHT: The three generation tasks are COMPLETELY INDEPENDENT.
# None of them needs the output of another. This is what makes
# parallelization possible and beneficial.

def demo_content_pipeline() -> None:
    """Content pipeline: 3 parallel generators → combine into blog post."""

    # ── State: holds the topic and each generated component ──────────
    class ContentState(TypedDict):
        topic: str          # The blog post topic
        title: str          # Generated blog title
        summary: str        # Generated blog summary
        tags: str           # Generated tags/keywords
        final_post: str     # Combined structured blog post

    # ── LLM: ChatGroq with creative temperature ─────────────────────
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7, max_tokens=512)

    # ── Node 1: generate_title (runs in parallel) ───────────────────
    def generate_title(state: ContentState) -> dict:
        """Generate a catchy blog post title."""
        logger.info("  [generate_title] Creating title for: %s", state["topic"])
        response = llm.invoke(
            f"Generate a catchy, SEO-friendly blog post title about: {state['topic']}. "
            f"Return ONLY the title, nothing else."
        )
        title = response.content.strip()
        logger.info("  [generate_title] Title: %s", title)
        return {"title": title}

    # ── Node 2: generate_summary (runs in parallel) ─────────────────
    def generate_summary(state: ContentState) -> dict:
        """Generate a blog post summary/abstract."""
        logger.info("  [generate_summary] Creating summary for: %s", state["topic"])
        response = llm.invoke(
            f"Write a compelling 3-4 sentence summary for a blog post about: "
            f"{state['topic']}. Make it informative and engaging."
        )
        summary = response.content
        logger.info("  [generate_summary] Summary: %s", summary[:120])
        return {"summary": summary}

    # ── Node 3: generate_tags (runs in parallel) ────────────────────
    def generate_tags(state: ContentState) -> dict:
        """Generate relevant tags/keywords for the blog post."""
        logger.info("  [generate_tags] Creating tags for: %s", state["topic"])
        response = llm.invoke(
            f"Generate 5-7 relevant tags/keywords for a blog post about: "
            f"{state['topic']}. Return them as a comma-separated list."
        )
        tags = response.content.strip()
        logger.info("  [generate_tags] Tags: %s", tags)
        return {"tags": tags}

    # ── Node 4: combine_post (fan-in — waits for all 3) ────────────
    def combine_post(state: ContentState) -> dict:
        """Combine all parallel outputs into a structured blog post."""
        logger.info("  [combine_post] Assembling final blog post...")
        final_post = (
            f"📝 BLOG POST\n"
            f"{'=' * 50}\n"
            f"Title: {state['title']}\n"
            f"{'=' * 50}\n\n"
            f"Summary:\n{state['summary']}\n\n"
            f"{'─' * 50}\n"
            f"Tags: {state['tags']}\n"
            f"{'=' * 50}"
        )
        logger.info("  [combine_post] Blog post assembled!")
        return {"final_post": final_post}

    # ── Build the graph ──────────────────────────────────────────────
    builder = StateGraph(ContentState)

    # Register all nodes
    builder.add_node("generate_title", generate_title)
    builder.add_node("generate_summary", generate_summary)
    builder.add_node("generate_tags", generate_tags)
    builder.add_node("combine_post", combine_post)

    # FAN-OUT: START → 3 parallel nodes
    builder.add_edge(START, "generate_title")
    builder.add_edge(START, "generate_summary")
    builder.add_edge(START, "generate_tags")

    # FAN-IN: 3 parallel nodes → combine_post
    builder.add_edge("generate_title", "combine_post")
    builder.add_edge("generate_summary", "combine_post")
    builder.add_edge("generate_tags", "combine_post")

    # combine → END
    builder.add_edge("combine_post", END)

    # Compile the graph
    graph = builder.compile()

    # Save the graph visualization
    save_graph_image(graph, "workflow02_content_pipeline")

    # ── Invoke the graph ─────────────────────────────────────────────
    logger.info("--- Demo 2: Production Content Pipeline (Parallel) ---")
    result = graph.invoke(
        {
            "topic": "The Future of Large Language Models in 2025",
            "title": "",
            "summary": "",
            "tags": "",
            "final_post": "",
        }
    )

    # Log each component and the final post
    logger.info("  Topic: %s", result["topic"])
    logger.info("  Title: %s", result["title"])
    logger.info("  Summary: %s", result["summary"][:200])
    logger.info("  Tags: %s", result["tags"])
    logger.info("  Final Post:\n%s", result["final_post"])
    logger.info("  Flow: START → [title, summary, tags] (parallel) → combine → END")
    logger.info("  Pattern: Fan-out / Fan-in parallelization for content production.")


# ═══════════════════════════════════════════════════════════════════════════════
# 8. WHEN TO USE PARALLELIZATION
# ═══════════════════════════════════════════════════════════════════════════════
#
# USE parallelization when:
#   ✅ Tasks are INDEPENDENT — none needs the output of another
#   ✅ You want to REDUCE LATENCY (total time = slowest task, not sum)
#   ✅ You're generating MULTIPLE COMPONENTS of a larger output
#   ✅ You're querying MULTIPLE DATA SOURCES simultaneously
#   ✅ You're running MULTIPLE ANALYSES on the same input
#
# DON'T use parallelization when:
#   ❌ Tasks DEPEND on each other (use prompt chaining instead)
#   ❌ Step 2 needs the output of Step 1 (that's sequential)
#   ❌ Order matters (parallelism doesn't guarantee execution order)
#
# COMPARISON — Sequential vs Parallel:
#
# Pattern       | When to Use                    | Latency
# ──────────────|────────────────────────────────|──────────────────
# Sequential    | Steps depend on each other     | T1 + T2 + T3
# Parallel      | Steps are independent          | max(T1, T2, T3)
# ──────────────|────────────────────────────────|──────────────────


# ═══════════════════════════════════════════════════════════════════════════════
# 9. COMMON MISTAKES & GOTCHAS
# ═══════════════════════════════════════════════════════════════════════════════
#
# Mistake                                    | Fix
# ───────────────────────────────────────────|──────────────────────────────────────
# Trying to parallelize DEPENDENT tasks     | If B needs A's output, they MUST be
#                                            | sequential. Only parallelize independent
#                                            | tasks.
# Forgetting fan-in edges                   | ALL parallel nodes must connect TO the
#                                            | combine node, or the graph won't wait.
# State key conflicts in parallel nodes     | Each parallel node should write to a
#                                            | DIFFERENT state key. If two nodes write
#                                            | to the same key, one overwrites the other.
# Using plain dict for state                | Always use TypedDict for type safety.
# Assuming execution order                  | Parallel nodes run in ANY order.
#                                            | Don't assume Node A finishes before B.
# Not compiling before invoke               | Always call builder.compile().
#
# ANTI-PATTERN: Parallelizing tasks that share dependencies.
#   If generate_setting needs characters to exist first,
#   you CANNOT parallelize them. Use chaining instead.


# ═══════════════════════════════════════════════════════════════════════════════
# 10. WHY LANGGRAPH FOR THIS
# ═══════════════════════════════════════════════════════════════════════════════
#
# ✅ AUTOMATIC parallelism — just connect edges, no special API
# ✅ Fan-in waits for ALL parallel nodes — no manual synchronization
# ✅ Stateful by design — shared state collects all parallel results
# ✅ Explicit control — you SEE the fan-out and fan-in in the graph
# ✅ Composable — parallel subgraphs can be part of larger workflows
# ✅ Production-ready — handles errors in individual parallel branches


# ═══════════════════════════════════════════════════════════════════════════════
# 11. WHERE THIS CONNECTS (Concept Linking Map)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Parallelization → uses → StateGraph + Fan-out/Fan-in edges
# Fan-out → one source node → multiple target nodes (parallel)
# Fan-in → multiple source nodes → one target node (combine)
# State (TypedDict) → each parallel node writes → different key
# Parallelization → reduces latency → for independent tasks
# Parallelization → combines with → Prompt Chaining for complex workflows
# Parallelization → foundation for → Map-Reduce pattern in LangGraph


# ═══════════════════════════════════════════════════════════════════════════════
# 🎯 INTERVIEW QUESTIONS
# ═══════════════════════════════════════════════════════════════════════════════
#
# THEORETICAL:
# Q: What is the difference between fan-out and fan-in in LangGraph
#    parallelization? How does LangGraph handle parallel execution?
# A: Fan-out is when one source node connects to MULTIPLE target nodes
#    via separate edges — LangGraph detects that these targets are
#    independent and runs them IN PARALLEL automatically. Fan-in is
#    when multiple source nodes connect to ONE target node — LangGraph
#    WAITS for ALL source nodes to complete before running the target.
#    No special API is needed; parallelism is a natural property of
#    the graph structure. Each parallel node writes to a different
#    state key, and the fan-in node reads all of them.
#
# HANDS-ON:
# Q: Build a parallelization graph that generates a "pros" list,
#    a "cons" list, and a "verdict" for a product review — all in
#    parallel — then combines them into a final review.
#
# SOLUTION:
#   from typing_extensions import TypedDict
#   from langgraph.graph import StateGraph, START, END
#
#   class ReviewState(TypedDict):
#       product: str
#       pros: str
#       cons: str
#       verdict: str
#       final_review: str
#
#   def gen_pros(state):
#       return {"pros": f"Pros of {state['product']}: Great quality, affordable."}
#
#   def gen_cons(state):
#       return {"cons": f"Cons of {state['product']}: Limited colors, slow shipping."}
#
#   def gen_verdict(state):
#       return {"verdict": f"Verdict: {state['product']} is worth buying."}
#
#   def combine(state):
#       review = f"{state['pros']}\n{state['cons']}\n{state['verdict']}"
#       return {"final_review": review}
#
#   builder = StateGraph(ReviewState)
#   builder.add_node("gen_pros", gen_pros)
#   builder.add_node("gen_cons", gen_cons)
#   builder.add_node("gen_verdict", gen_verdict)
#   builder.add_node("combine", combine)
#
#   # Fan-out: START → 3 parallel nodes
#   builder.add_edge(START, "gen_pros")
#   builder.add_edge(START, "gen_cons")
#   builder.add_edge(START, "gen_verdict")
#
#   # Fan-in: 3 parallel nodes → combine
#   builder.add_edge("gen_pros", "combine")
#   builder.add_edge("gen_cons", "combine")
#   builder.add_edge("gen_verdict", "combine")
#
#   builder.add_edge("combine", END)
#   graph = builder.compile()
#   result = graph.invoke({
#       "product": "Wireless Headphones",
#       "pros": "", "cons": "", "verdict": "", "final_review": ""
#   })
#   print(result["final_review"])


# ═══════════════════════════════════════════════════════════════════════════════
# 📝 QUICK RECAP
# ═══════════════════════════════════════════════════════════════════════════════
#
# 1. Parallelization runs INDEPENDENT tasks simultaneously using
#    fan-out (split) and fan-in (combine) edges. LangGraph handles
#    parallel execution AUTOMATICALLY — no special API needed.
#
# 2. Total latency = SLOWEST task (not the sum). This is a massive
#    speedup when you have multiple independent LLM calls.
#
# 3. NEVER parallelize DEPENDENT tasks. If B needs A's output,
#    use prompt chaining (sequential). Only parallelize tasks that
#    are truly independent and write to DIFFERENT state keys.


# ═══════════════════════════════════════════════════════════════════════════════
# 🚀 Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("⚡ LANGGRAPH WORKFLOWS — Lesson 2: Parallelization")
    logger.info("=" * 70)

    logger.info("\n🔹 Demo 1: Story Elements Generator (Fan-out / Fan-in)")
    demo_story_elements()

    logger.info("\n🔹 Demo 2: Production Content Pipeline (Parallel Generation)")
    demo_content_pipeline()

    logger.info("\n" + "=" * 70)
    logger.info("✅ Lesson 2 complete — Parallelization with Fan-out / Fan-in")
    logger.info("You can now build parallel workflows for independent tasks.")
    logger.info("Next: Lesson 3 — Orchestrator-Workers Pattern")
    logger.info("=" * 70)
