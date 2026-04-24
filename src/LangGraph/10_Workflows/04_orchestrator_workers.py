"""
LangGraph Workflows - Lesson 4: Orchestrator-Workers (Level 7)

CONCEPT: Orchestrator-Workers - ONE-LINE DEFINITION

The Orchestrator-Workers pattern is a workflow where a CENTRAL LLM
(the orchestrator) dynamically PLANS subtasks, DELEGATES them to
worker nodes via the Send API, and a SYNTHESIZER combines all
worker outputs into a final result.

WHY ORCHESTRATOR-WORKERS EXISTS

In Parallelization (Lesson 02), you PRE-DEFINE the parallel tasks
at graph-build time. But what if you DON'T KNOW the subtasks in
advance? What if the number and nature of subtasks depend on the
user's input?

Examples:
    - "Write a report on Agentic AI" -> How many sections? Which topics?
      The LLM must DECIDE the sections dynamically.
    - "Review this codebase" -> How many files? Which ones need review?
      The LLM must ANALYZE and PLAN the review dynamically.

Orchestrator-Workers solves this by:
    1. An ORCHESTRATOR node uses an LLM to PLAN the subtasks.
    2. The plan is a LIST of structured objects (Pydantic models).
    3. The Send API DYNAMICALLY creates one worker per subtask.
    4. Each worker runs IN PARALLEL with its own isolated state.
    5. All worker outputs are COLLECTED into a shared state key.
    6. A SYNTHESIZER node combines all outputs into the final result.

KEY DIFFERENCE from Parallelization:
    Parallelization : Tasks are FIXED at graph-build time (fan-out).
                      You know exactly 3 tasks: character, premise, setting.
    Orchestrator    : Tasks are DYNAMIC at runtime (Send API).
                      The LLM decides: maybe 3 sections, maybe 7.

TERMINOLOGY CLARIFICATION (Your Confusions Answered)

CONFUSION 1: Orchestrator vs Supervisor vs Parent Agent

    Orchestrator (THIS lesson):
        - A node that PLANS subtasks and DELEGATES via Send API.
        - It does NOT manage agents, it manages TASKS.
        - Workers are STATELESS functions, not autonomous agents.
        - The orchestrator runs ONCE, creates workers, then exits.
        - Think: PROJECT MANAGER who creates a task list.

    Supervisor (Multi-Agent pattern):
        - A node that MANAGES multiple AGENTS (not just tasks).
        - It decides WHICH AGENT to call next in a LOOP.
        - Agents are more autonomous, they have their own tools.
        - The supervisor runs in a LOOP, checking agent outputs.
        - Think: TEAM LEAD who assigns work and reviews results.

    Parent Agent (Subgraph pattern):
        - A graph that CONTAINS other graphs (subgraphs).
        - Each subgraph is a complete agent with its own state.
        - The parent orchestrates subgraphs, not individual tasks.
        - Think: DEPARTMENT HEAD who manages teams.

    Summary:
        Orchestrator -> plans TASKS -> delegates via Send -> one-shot
        Supervisor   -> manages AGENTS -> loops until done -> iterative
        Parent Agent -> composes SUBGRAPHS -> hierarchical -> nested

CONFUSION 2: How does the orchestrator decide the number of tasks?

    The orchestrator uses llm.with_structured_output(Sections) to
    ask the LLM to generate a PLAN. The LLM returns a Pydantic
    model with a LIST of Section objects. The LENGTH of this list
    determines the number of tasks/workers.

    Example:
        Input: "Write a report on Agentic AI"
        LLM Plan: Sections(sections=[
            Section(name="Introduction", description="..."),
            Section(name="Architecture", description="..."),
            Section(name="Use Cases", description="..."),
            Section(name="Challenges", description="..."),
        ])
        -> 4 sections -> 4 workers created dynamically.

    The LLM DECIDES. You don't hardcode the number.
    Different inputs -> different number of workers.

CONFUSION 3: Is this loop-based or fixed?

    FIXED, not loop-based. The flow is:
        START -> orchestrator -> [workers in parallel] -> synthesizer -> END

    Workers run ONCE each. There is NO loop back.
    The orchestrator runs ONCE to create the plan.
    Each worker runs ONCE to complete its assigned task.
    The synthesizer runs ONCE to combine all outputs.

    Compare with Supervisor pattern:
        START -> supervisor -> agent -> supervisor -> agent -> ... -> END
        (The supervisor LOOPS until all agents are done.)

CONFUSION 4: How does the Send API work internally?

    The Send API is LangGraph's mechanism for DYNAMIC fan-out.

    How it works:
        1. You define a function (assign_workers) that returns a
           LIST of Send objects.
        2. Each Send(node_name, state_dict) tells LangGraph:
           "Create a new execution of node_name with this state."
        3. LangGraph creates ONE worker instance per Send object.
        4. All workers run IN PARALLEL (not sequentially).
        5. Each worker has its OWN isolated WorkerState.
        6. Worker outputs are MERGED into the parent State via
           the Annotated[list, operator.add] reducer.

    Code:
        from langgraph.types import Send

        def assign_workers(state: State):
            return [Send("worker", {"section": s}) for s in state["sections"]]

    If state["sections"] has 4 items -> 4 Send objects -> 4 workers.
    If state["sections"] has 7 items -> 7 Send objects -> 7 workers.

    Think of Send as: "Hey LangGraph, spawn this node with this input."

CONFUSION 5: How many workers are created and who decides?

    The NUMBER of workers = the NUMBER of Send objects returned.
    The ORCHESTRATOR (via LLM planning) decides the number.

    Flow:
        1. Orchestrator asks LLM: "Plan sections for this topic."
        2. LLM returns: [Section1, Section2, Section3, Section4]
        3. assign_workers creates: [Send("worker", {section: S1}),
                                     Send("worker", {section: S2}),
                                     Send("worker", {section: S3}),
                                     Send("worker", {section: S4})]
        4. LangGraph spawns 4 worker instances.

    Workers are DYNAMIC, created at runtime, not at graph-build time.
    This is the KEY difference from Parallelization.

CONFUSION 6: How does synthesis work?

    All workers write to the SAME state key: completed_sections.
    This key uses Annotated[list, operator.add] as a REDUCER.
    The reducer APPENDS each worker's output to the list.

    After ALL workers finish:
        state["completed_sections"] = [
            "Section 1 content...",
            "Section 2 content...",
            "Section 3 content...",
            "Section 4 content...",
        ]

    The synthesizer node reads this list and combines them:
        final_report = "\\n\\n---\\n\\n".join(completed_sections)

    The synthesizer can also use an LLM to write an introduction,
    add transitions, or create an executive summary.

CONFUSION 7: Real-world production use cases

    1. Report Generation: LLM plans sections, workers write each.
    2. Code Review: LLM identifies files to review, workers review each.
    3. Research Pipeline: LLM breaks topic into areas, workers research each.
    4. Content Pipeline: LLM plans blog sections, workers draft each.
    5. Data Analysis: LLM identifies datasets, workers analyze each.
    6. Translation: LLM splits document into chunks, workers translate each.
    7. Test Generation: LLM identifies functions, workers write tests for each.

REAL-WORLD ANALOGY: A NEWSPAPER EDITOR

Think of a NEWSPAPER EDITOR planning tomorrow's edition:

    1. The EDITOR (orchestrator) decides the sections:
       Front page, Sports, Technology, Opinion, Weather.
    2. The editor ASSIGNS each section to a JOURNALIST (worker).
    3. Each journalist writes their section INDEPENDENTLY (parallel).
    4. The LAYOUT EDITOR (synthesizer) combines all articles into
       the final newspaper.

    The editor does NOT write articles. The editor PLANS and DELEGATES.
    The journalists do NOT coordinate with each other. They work alone.
    The layout editor does NOT write. They COMBINE and FORMAT.

    This is EXACTLY the orchestrator-workers pattern.

ASCII GRAPH STRUCTURES

Demo 1: Report Generation (from notebook, improved):

    [START]
       |
    [orchestrator]  <- LLM plans sections via structured output
       |
       |-- Send("llm_call", {section: S1}) --> [llm_call] --+
       |-- Send("llm_call", {section: S2}) --> [llm_call] --+
       |-- Send("llm_call", {section: S3}) --> [llm_call] --+--> [synthesizer] --> [END]
       |-- Send("llm_call", {section: S4}) --> [llm_call] --+
       (N workers, decided by LLM at runtime)

Demo 2: Production Code Review Pipeline:

    [START]
       |
    [plan_review]  <- LLM identifies review areas
       |
       |-- Send("review_worker", {area: A1}) --> [review_worker] --+
       |-- Send("review_worker", {area: A2}) --> [review_worker] --+
       |-- Send("review_worker", {area: A3}) --> [review_worker] --+--> [compile_review] --> [END]
       (N workers, decided by LLM at runtime)

LANGCHAIN vs LANGGRAPH: Orchestrator-Workers

LangChain : No native orchestrator pattern. You'd manually loop
            over tasks and call chains sequentially. No parallelism.
            No shared state. No dynamic task creation.
LangGraph  : Send API enables DYNAMIC fan-out at runtime.
            Workers run in PARALLEL with isolated state.
            Annotated[list, operator.add] merges outputs.
            Full control, stateful, composable, production-ready.

WHERE IT FITS IN THE LANGGRAPH ARCHITECTURE

    Orchestrator-Workers is a WORKFLOW PATTERN built on top of:
        StateGraph -> Nodes -> Send API -> Reducers -> Structured Output

    HOW IT WORKS:
        1. Orchestrator node calls llm.with_structured_output(Plan)
           to generate a dynamic list of subtasks.
        2. assign_workers function returns [Send(...)] for each subtask.
        3. LangGraph spawns N worker nodes in parallel.
        4. Each worker writes to Annotated[list, operator.add].
        5. Synthesizer reads all outputs and combines them.

    Concept Linking:
        StateGraph -> compiles to -> CompiledGraph
        Structured Output -> Pydantic BaseModel -> dynamic planning
        Send API -> creates -> dynamic worker instances at runtime
        Annotated[list, operator.add] -> merges -> worker outputs
        Orchestrator -> different from -> Parallelization (fixed fan-out)
        Orchestrator -> different from -> Supervisor (agent loop)
        Orchestrator -> combines with -> Routing, Prompt Chaining

HOW TO RUN:
    $ python src/Workflows/04_orchestrator_workers.py

Author: GenAI Learner
"""

import logging
import operator
import os
import time
from typing import Annotated, List

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
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


# -- Helper: Retry wrapper for Groq structured output -------------------------
# Groq's tool_use method for structured output can intermittently fail
# with some models (400 Bad Request). A simple retry fixes this.
def invoke_with_retry(runnable, input_data, max_retries: int = 3):
    """Invoke a runnable with retry logic for intermittent Groq failures."""
    for attempt in range(max_retries):
        try:
            return runnable.invoke(input_data)
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning("  Retry %d/%d - Groq error: %s", attempt + 1, max_retries, str(e)[:100])
                time.sleep(1)
            else:
                raise


# -- Helper: Save graph as PNG image ------------------------------------------
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


# ==============================================================================
# DEMO 1: Report Generation (Orchestrator-Workers with Send API)
# ==============================================================================
#
# This is the CORE orchestrator-workers pattern from the notebook, improved.
# An LLM dynamically plans report sections, workers write each section
# in parallel, and a synthesizer combines them into a final report.
#
# KEY CONCEPTS:
#   1. Pydantic models (Section, Sections) define the PLAN structure.
#   2. llm.with_structured_output(Sections) forces the LLM to return a plan.
#   3. Send API creates one worker per section DYNAMICALLY.
#   4. WorkerState is ISOLATED - each worker only sees its own section.
#   5. Annotated[list, operator.add] MERGES all worker outputs.
#   6. Synthesizer reads all outputs and formats the final report.
#
# WHY TWO STATE CLASSES?
#   State (parent): Holds the full picture - topic, sections, all outputs, final report.
#   WorkerState (child): Holds ONLY what one worker needs - its section + output key.
#   Workers don't need to see the full state. Isolation = cleaner, safer.
#
# HOW Send CONNECTS WorkerState TO State:
#   Workers write to "completed_sections" in WorkerState.
#   This key ALSO exists in State with the SAME reducer (operator.add).
#   LangGraph automatically merges WorkerState outputs into State.

def demo_report_generation() -> None:
    """Report generation: orchestrator plans sections, workers write, synthesizer combines."""

    # -- Pydantic Models: Define the PLAN structure ----------------------------
    # Section: ONE section of the report (name + description).
    # Sections: The FULL plan - a list of Section objects.
    # The LLM is FORCED to return this structure via with_structured_output.
    class Section(BaseModel):
        name: str = Field(description="Name for this section of the report")
        description: str = Field(
            description="Brief overview of the main topics and concepts of the section"
        )

    class Sections(BaseModel):
        sections: List[Section] = Field(description="Sections of the report")

    # -- State: Parent graph state (full picture) ------------------------------
    # topic: The user's input topic for the report.
    # sections: The LLM-generated plan (list of Section objects).
    # completed_sections: ALL worker outputs collected here via operator.add.
    # final_report: The synthesized final report string.
    class State(TypedDict):
        topic: str
        sections: list[Section]
        completed_sections: Annotated[list, operator.add]
        final_report: str

    # -- WorkerState: Isolated state for each worker ---------------------------
    # section: The ONE section this worker is responsible for.
    # completed_sections: This worker's output (merged into parent State).
    # NOTE: The key name "completed_sections" MUST match the parent State key.
    # This is how LangGraph knows where to merge worker outputs.
    class WorkerState(TypedDict):
        section: Section
        completed_sections: Annotated[list, operator.add]

    # -- LLMs: Separate models for planning vs generation ----------------------
    # Planning (structured output) needs a more capable model.
    # Generation (writing sections) can use a faster, smaller model.
    planner_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, max_tokens=1024)
    writer_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7, max_tokens=1024)

    # Augment the planner LLM with structured output schema.
    # This wraps the LLM so it ALWAYS returns a Sections Pydantic object.
    planner = planner_llm.with_structured_output(Sections)

    # -- Node 1: orchestrator - plans the report sections ----------------------
    # This node calls the planner LLM to generate a structured plan.
    # The plan is a list of Section objects (name + description).
    # The NUMBER of sections is decided by the LLM, not hardcoded.
    def orchestrator(state: State) -> dict:
        """Orchestrator that generates a plan for the report."""
        logger.info("  [orchestrator] Planning report for topic: '%s'", state["topic"])

        # Ask the LLM to generate a plan with structured output
        report_sections = invoke_with_retry(
            planner,
            [
                SystemMessage(
                    content=(
                        "Generate a plan for a detailed report. "
                        "Create 3-5 sections that cover the topic comprehensively. "
                        "Each section should have a clear name and description."
                    )
                ),
                HumanMessage(content=f"Here is the report topic: {state['topic']}"),
            ],
        )

        # Log the plan so we can see what the LLM decided
        for i, section in enumerate(report_sections.sections, 1):
            logger.info("  [orchestrator] Section %d: %s - %s", i, section.name, section.description[:80])

        logger.info("  [orchestrator] Plan complete: %d sections created.", len(report_sections.sections))

        # Return the sections to state - this triggers assign_workers
        return {"sections": report_sections.sections}

    # -- Node 2: llm_call (worker) - writes ONE section ------------------------
    # Each worker receives a WorkerState with ONE Section object.
    # It writes the section content and returns it to completed_sections.
    # NOTE: This function runs N times in PARALLEL (once per Send).
    def llm_call(state: WorkerState) -> dict:
        """Worker writes a section of the report."""
        section = state["section"]
        logger.info("  [worker] Writing section: '%s'", section.name)

        # Generate the section content using the writer LLM
        response = writer_llm.invoke(
            [
                SystemMessage(
                    content=(
                        "Write a report section following the provided name and description. "
                        "Include no preamble for each section. Use markdown formatting. "
                        "Write 2-3 paragraphs with clear, informative content."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Section name: {section.name}\n"
                        f"Section description: {section.description}"
                    )
                ),
            ]
        )

        logger.info("  [worker] Completed section: '%s' (%d chars)", section.name, len(response.content))

        # Return the content to completed_sections (merged via operator.add)
        return {"completed_sections": [response.content]}

    # -- Conditional Edge: assign_workers - creates Send objects ----------------
    # This function is used as a CONDITIONAL EDGE from orchestrator.
    # It reads state["sections"] and creates one Send per section.
    # Each Send("llm_call", {...}) spawns a worker with that section.
    #
    # THIS IS THE MAGIC OF THE ORCHESTRATOR PATTERN:
    #   The number of Send objects = the number of workers.
    #   The LLM decided the sections -> this function creates the workers.
    #   LangGraph handles the parallel execution automatically.
    def assign_workers(state: State) -> list:
        """Assign a worker to each section in the plan via Send API."""
        logger.info("  [assign_workers] Creating %d workers...", len(state["sections"]))
        # Each Send creates a new worker instance with its own WorkerState
        return [Send("llm_call", {"section": s}) for s in state["sections"]]

    # -- Node 3: synthesizer - combines all worker outputs ---------------------
    # After ALL workers finish, their outputs are in completed_sections.
    # The synthesizer reads them and formats the final report.
    def synthesizer(state: State) -> dict:
        """Synthesize full report from all completed sections."""
        completed_sections = state["completed_sections"]
        logger.info("  [synthesizer] Combining %d sections into final report...", len(completed_sections))

        # Join all sections with a separator
        completed_report = "\n\n---\n\n".join(completed_sections)

        logger.info("  [synthesizer] Final report: %d chars", len(completed_report))
        return {"final_report": completed_report}

    # -- Build the graph -------------------------------------------------------
    builder = StateGraph(State)

    # Register all nodes
    builder.add_node("orchestrator", orchestrator)
    builder.add_node("llm_call", llm_call)
    builder.add_node("synthesizer", synthesizer)

    # Edges: START -> orchestrator
    builder.add_edge(START, "orchestrator")

    # CONDITIONAL EDGE: orchestrator -> assign_workers -> [llm_call workers]
    # assign_workers returns a LIST of Send objects.
    # The second argument ["llm_call"] tells LangGraph which nodes
    # the Send objects can target (for graph visualization).
    builder.add_conditional_edges("orchestrator", assign_workers, ["llm_call"])

    # All workers -> synthesizer
    builder.add_edge("llm_call", "synthesizer")

    # synthesizer -> END
    builder.add_edge("synthesizer", END)

    # Compile the graph
    graph = builder.compile()

    # Save graph visualization
    save_graph_image(graph, "workflow04_report_generation")

    # -- Run the demo ----------------------------------------------------------
    logger.info("--- Demo 1: Report Generation (Orchestrator-Workers) ---")
    topic = "Agentic AI Systems in 2025: Architecture, Patterns, and Production Use Cases"
    logger.info("  Topic: '%s'", topic)

    state = graph.invoke({"topic": topic})

    logger.info("  Final Report Preview (first 500 chars):")
    logger.info("  %s", state["final_report"][:500])
    logger.info("")
    logger.info("  Flow: START -> orchestrator (plan) -> [N workers in parallel] -> synthesizer -> END")
    logger.info("  Pattern: LLM-driven dynamic planning + Send API for parallel worker execution.")


# ==============================================================================
# DEMO 2: Production Code Review Pipeline (Orchestrator-Workers)
# ==============================================================================
#
# A more PRACTICAL example: an LLM-powered code review system.
# The orchestrator analyzes a code snippet and identifies REVIEW AREAS
# (e.g., security, performance, readability). Workers review each area
# independently, and the synthesizer compiles a final review report.
#
# KEY DIFFERENCE from Demo 1:
#   Demo 1: Plans REPORT SECTIONS (content generation).
#   Demo 2: Plans REVIEW AREAS (code analysis).
#   Same pattern, different domain - orchestrator-workers is UNIVERSAL.
#
# WHY THIS IS PRODUCTION-RELEVANT:
#   In real code review tools (GitHub Copilot, CodeRabbit, etc.),
#   the system analyzes code from multiple angles simultaneously.
#   Each angle (security, performance, style) is a separate worker.
#   The final review combines all perspectives.

def demo_code_review_pipeline() -> None:
    """Code review pipeline: orchestrator identifies areas, workers review, synthesizer compiles."""

    # -- Pydantic Models: Define the REVIEW PLAN structure ---------------------
    class ReviewArea(BaseModel):
        name: str = Field(description="Name of the review area (e.g., Security, Performance)")
        focus: str = Field(
            description="What to focus on when reviewing this area"
        )

    class ReviewPlan(BaseModel):
        areas: List[ReviewArea] = Field(description="Areas to review in the code")

    # -- State: Parent graph state ---------------------------------------------
    class ReviewState(TypedDict):
        code: str                                           # The code to review
        areas: list[ReviewArea]                             # LLM-planned review areas
        completed_reviews: Annotated[list, operator.add]    # All worker review outputs
        final_review: str                                   # Combined final review

    # -- WorkerState: Isolated state for each review worker --------------------
    class ReviewWorkerState(TypedDict):
        code: str                                           # The code being reviewed
        area: ReviewArea                                    # This worker's review area
        completed_reviews: Annotated[list, operator.add]    # This worker's output

    # -- LLMs ------------------------------------------------------------------
    planner_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, max_tokens=1024)
    reviewer_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3, max_tokens=1024)

    # Planner with structured output
    review_planner = planner_llm.with_structured_output(ReviewPlan)

    # -- Node 1: plan_review - identifies review areas -------------------------
    def plan_review(state: ReviewState) -> dict:
        """Orchestrator that identifies areas to review in the code."""
        logger.info("  [plan_review] Analyzing code for review areas...")

        plan = invoke_with_retry(
            review_planner,
            [
                SystemMessage(
                    content=(
                        "You are a senior code reviewer. Analyze the given code and "
                        "identify 3-4 distinct areas that need review. "
                        "Common areas: Security, Performance, Readability, Error Handling, "
                        "Best Practices, Testing, Documentation. "
                        "Choose areas that are RELEVANT to the specific code provided."
                    )
                ),
                HumanMessage(content=f"Review this code:\n\n{state['code']}"),
            ],
        )

        for i, area in enumerate(plan.areas, 1):
            logger.info("  [plan_review] Area %d: %s - %s", i, area.name, area.focus[:80])

        logger.info("  [plan_review] Plan complete: %d review areas.", len(plan.areas))
        return {"areas": plan.areas}

    # -- Node 2: review_worker - reviews ONE area ------------------------------
    def review_worker(state: ReviewWorkerState) -> dict:
        """Worker reviews the code from one specific perspective."""
        area = state["area"]
        logger.info("  [review_worker] Reviewing: '%s'", area.name)

        response = reviewer_llm.invoke(
            [
                SystemMessage(
                    content=(
                        f"You are a code reviewer specializing in {area.name}. "
                        f"Focus on: {area.focus}. "
                        "Provide specific, actionable feedback with code examples where relevant. "
                        "Use markdown formatting. Be concise but thorough."
                    )
                ),
                HumanMessage(content=f"Review this code:\n\n{state['code']}"),
            ],
        )

        logger.info("  [review_worker] Completed: '%s' (%d chars)", area.name, len(response.content))
        return {"completed_reviews": [f"## {area.name}\n\n{response.content}"]}

    # -- Conditional Edge: assign_reviewers ------------------------------------
    def assign_reviewers(state: ReviewState) -> list:
        """Assign a reviewer worker to each review area via Send API."""
        logger.info("  [assign_reviewers] Creating %d review workers...", len(state["areas"]))
        return [
            Send("review_worker", {"code": state["code"], "area": area})
            for area in state["areas"]
        ]

    # -- Node 3: compile_review - combines all reviews -------------------------
    def compile_review(state: ReviewState) -> dict:
        """Compile all review area outputs into a final review report."""
        reviews = state["completed_reviews"]
        logger.info("  [compile_review] Compiling %d reviews into final report...", len(reviews))

        final_review = "# Code Review Report\n\n" + "\n\n---\n\n".join(reviews)

        logger.info("  [compile_review] Final review: %d chars", len(final_review))
        return {"final_review": final_review}

    # -- Build the graph -------------------------------------------------------
    builder = StateGraph(ReviewState)

    builder.add_node("plan_review", plan_review)
    builder.add_node("review_worker", review_worker)
    builder.add_node("compile_review", compile_review)

    builder.add_edge(START, "plan_review")
    builder.add_conditional_edges("plan_review", assign_reviewers, ["review_worker"])
    builder.add_edge("review_worker", "compile_review")
    builder.add_edge("compile_review", END)

    graph = builder.compile()

    save_graph_image(graph, "workflow04_code_review_pipeline")

    # -- Run the demo with sample code -----------------------------------------
    logger.info("--- Demo 2: Production Code Review Pipeline ---")

    sample_code = '''
def get_user_data(user_id):
    import sqlite3
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    query = f"SELECT * FROM users WHERE id = {user_id}"
    cursor.execute(query)
    result = cursor.fetchone()
    conn.close()
    return result

def process_items(items):
    results = []
    for item in items:
        processed = item.upper().strip()
        results.append(processed)
    return results

class UserService:
    def __init__(self):
        self.cache = {}

    def get_user(self, id):
        if id in self.cache:
            return self.cache[id]
        data = get_user_data(id)
        self.cache[id] = data
        return data
'''

    logger.info("  Code to review: (sample Python code with intentional issues)")
    state = graph.invoke({"code": sample_code})

    logger.info("  Final Review Preview (first 500 chars):")
    logger.info("  %s", state["final_review"][:500])
    logger.info("")
    logger.info("  Flow: START -> plan_review -> [N review workers in parallel] -> compile_review -> END")
    logger.info("  Pattern: LLM-driven code analysis + parallel review workers.")


# ==============================================================================
# COMPARISON TABLE: Workflow Patterns in LangGraph
# ==============================================================================
#
# Pattern              | Task Planning        | Execution         | Use Case
# ---------------------|----------------------|-------------------|---------------------------
# Prompt Chaining      | FIXED sequence       | Sequential        | Multi-step refinement
# (Lesson 01)          | A -> B -> C          | One at a time     | Story: generate->improve->polish
#                      | Gates: pass/fail     | May loop on fail  |
# ---------------------|----------------------|-------------------|---------------------------
# Parallelization      | FIXED fan-out        | Parallel          | Independent subtasks
# (Lesson 02)          | Pre-defined tasks    | All at once       | Story: character+premise+setting
#                      | Known at build time  | Fan-in to combine |
# ---------------------|----------------------|-------------------|---------------------------
# Routing              | CLASSIFICATION       | Single path       | Input triage/classification
# (Lesson 03)          | LLM picks ONE path   | One specialist    | Support: billing/tech/general
#                      | N-way branching      | No parallelism    |
# ---------------------|----------------------|-------------------|---------------------------
# Orchestrator-Workers | DYNAMIC planning     | Parallel          | Complex, unpredictable tasks
# (THIS lesson)        | LLM decides tasks    | Send API fan-out  | Report: LLM plans N sections
#                      | Unknown at build time| Workers + synth   | Code review: LLM plans areas
# ---------------------|----------------------|-------------------|---------------------------
#
# KEY INSIGHT:
#   Parallelization = you KNOW the tasks at build time.
#   Orchestrator    = the LLM DECIDES the tasks at runtime.
#   Both use parallel execution, but the PLANNING is different.


# ==============================================================================
# WHEN TO USE ORCHESTRATOR-WORKERS
# ==============================================================================
#
# USE orchestrator-workers when:
#   - You DON'T KNOW the subtasks in advance
#   - The number of subtasks depends on the INPUT
#   - Each subtask is INDEPENDENT (no dependencies between workers)
#   - You need an LLM to PLAN before executing
#   - The final output requires COMBINING multiple worker outputs
#
# DON'T use orchestrator-workers when:
#   - Tasks are FIXED and known at build time (use Parallelization)
#   - Tasks must run SEQUENTIALLY (use Prompt Chaining)
#   - You need to CLASSIFY input into one path (use Routing)
#   - Workers need to COMMUNICATE with each other (use Multi-Agent)
#   - The task is simple enough for a single LLM call


# ==============================================================================
# COMMON MISTAKES AND GOTCHAS
# ==============================================================================
#
# Mistake                                    | Fix
# -------------------------------------------|--------------------------------------
# Using plain dict for State                 | Always use TypedDict for type safety.
# Forgetting operator.add on list fields     | completed_sections MUST use
#                                            | Annotated[list, operator.add] or
#                                            | worker outputs will OVERWRITE.
# WorkerState key name doesn't match State   | The shared key (completed_sections)
#                                            | must have the SAME NAME in both
#                                            | State and WorkerState.
# Using same LLM for planning and generation | Use a CAPABLE model for planning
#                                            | (structured output) and a FAST model
#                                            | for generation (workers).
# Not handling empty plan from LLM           | Always check if sections list is
#                                            | empty before creating Send objects.
# Forgetting to compile before invoke        | Always call builder.compile().
# No recursion limit on complex graphs       | Set recursion_limit in config for
#                                            | production safety.
#
# ANTI-PATTERN: Putting planning AND execution in the orchestrator node.
#   The orchestrator should ONLY plan. Workers should ONLY execute.
#   Separation of concerns = cleaner, testable, reusable.


# ==============================================================================
# WHY LANGGRAPH FOR THIS
# ==============================================================================
#
# - Send API: DYNAMIC fan-out is NATIVE to LangGraph. No other framework
#   makes it this easy to spawn N workers at runtime.
# - Annotated reducers: operator.add automatically MERGES worker outputs.
#   No manual collection or synchronization needed.
# - Isolated WorkerState: Each worker has its own state. No interference.
# - Structured Output: Pydantic models force valid plans from the LLM.
# - Graph visualization: You can SEE the orchestrator pattern in the graph.
# - Composable: Orchestrator can be a subgraph inside a larger workflow.
# - Production-ready: Combine with checkpointing, streaming, tracing.


# ==============================================================================
# WHERE THIS CONNECTS (Concept Linking Map)
# ==============================================================================
#
# Orchestrator-Workers -> uses -> StateGraph + Send API + Structured Output
# Send API -> creates -> dynamic worker instances at runtime
# Send(node_name, state) -> spawns -> one worker with isolated WorkerState
# Annotated[list, operator.add] -> merges -> all worker outputs into parent State
# Orchestrator -> plans via -> llm.with_structured_output(Pydantic)
# Workers -> write to -> shared key (completed_sections)
# Synthesizer -> reads -> all worker outputs -> produces -> final result
# Orchestrator -> different from -> Parallelization (fixed fan-out)
# Orchestrator -> different from -> Supervisor (agent loop, iterative)
# Orchestrator -> different from -> Routing (classification, single path)
# Orchestrator -> combines with -> Prompt Chaining (workers can chain steps)


# ==============================================================================
# INTERVIEW QUESTIONS
# ==============================================================================
#
# THEORETICAL:
# Q: What is the difference between the Orchestrator-Workers pattern and
#    the Parallelization pattern in LangGraph? When would you use each?
# A: In Parallelization, the subtasks are PRE-DEFINED at graph-build time.
#    You know exactly which tasks to run in parallel (e.g., character,
#    premise, setting). In Orchestrator-Workers, the subtasks are DYNAMIC
#    - an LLM plans them at runtime using structured output. The Send API
#    creates one worker per planned subtask. Use Parallelization when tasks
#    are fixed and known. Use Orchestrator-Workers when the number and
#    nature of tasks depend on the input (e.g., report sections, code
#    review areas). Both use parallel execution, but the planning differs.
#
# HANDS-ON:
# Q: Build an orchestrator-workers graph where the orchestrator plans
#    3-5 quiz questions on a given topic, workers generate each question
#    with answer choices, and a synthesizer compiles the final quiz.
#    Use Pydantic structured output for planning and the Send API.
#
# SOLUTION:
#   from typing import Annotated, List
#   import operator
#   from pydantic import BaseModel, Field
#   from typing_extensions import TypedDict
#   from langchain_core.messages import HumanMessage, SystemMessage
#   from langchain_groq import ChatGroq
#   from langgraph.types import Send
#   from langgraph.graph import StateGraph, START, END
#
#   class QuizQuestion(BaseModel):
#       question: str = Field(description="The quiz question")
#       topic_hint: str = Field(description="Hint for generating answer choices")
#
#   class QuizPlan(BaseModel):
#       questions: List[QuizQuestion] = Field(description="Quiz questions to generate")
#
#   class QuizState(TypedDict):
#       topic: str
#       questions: list[QuizQuestion]
#       completed_questions: Annotated[list, operator.add]
#       final_quiz: str
#
#   class QuizWorkerState(TypedDict):
#       question: QuizQuestion
#       completed_questions: Annotated[list, operator.add]
#
#   planner_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
#   writer_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.5)
#   quiz_planner = planner_llm.with_structured_output(QuizPlan)
#
#   def plan_quiz(state):
#       plan = quiz_planner.invoke([
#           SystemMessage(content="Create 3-5 quiz questions for the topic."),
#           HumanMessage(content=f"Topic: {state['topic']}"),
#       ])
#       return {"questions": plan.questions}
#
#   def write_question(state):
#       q = state["question"]
#       r = writer_llm.invoke([
#           SystemMessage(content="Write a multiple-choice question with 4 options (A-D) and the correct answer."),
#           HumanMessage(content=f"Question: {q.question}\nHint: {q.topic_hint}"),
#       ])
#       return {"completed_questions": [r.content]}
#
#   def assign_question_workers(state):
#       return [Send("write_question", {"question": q}) for q in state["questions"]]
#
#   def compile_quiz(state):
#       quiz = "\n\n---\n\n".join(state["completed_questions"])
#       return {"final_quiz": f"# Quiz\n\n{quiz}"}
#
#   builder = StateGraph(QuizState)
#   builder.add_node("plan_quiz", plan_quiz)
#   builder.add_node("write_question", write_question)
#   builder.add_node("compile_quiz", compile_quiz)
#   builder.add_edge(START, "plan_quiz")
#   builder.add_conditional_edges("plan_quiz", assign_question_workers, ["write_question"])
#   builder.add_edge("write_question", "compile_quiz")
#   builder.add_edge("compile_quiz", END)
#   graph = builder.compile()
#   result = graph.invoke({"topic": "Python Data Structures"})
#   print(result["final_quiz"])
#
# BONUS (System Design):
# Q: Design an orchestrator-workers system for a multi-language
#    documentation generator. Given a codebase, the orchestrator
#    identifies modules to document, workers generate docs for each
#    module, and the synthesizer creates a unified documentation site.
#    Consider: How would you handle dependencies between modules?
#    How would you add quality checks? How would you scale this?


# ==============================================================================
# QUICK RECAP
# ==============================================================================
#
# 1. Orchestrator-Workers uses an LLM to DYNAMICALLY plan subtasks
#    via structured output (Pydantic), then the Send API creates
#    one worker per subtask. Workers run in PARALLEL with isolated
#    state, and a synthesizer combines all outputs.
#
# 2. The Send API is the KEY mechanism: Send(node_name, state_dict)
#    tells LangGraph to spawn a worker instance. The number of Send
#    objects = the number of workers. The LLM decides the count.
#
# 3. Orchestrator-Workers is DIFFERENT from Parallelization (fixed
#    tasks), Routing (single path), and Supervisor (agent loop).
#    Use it when subtasks are UNPREDICTABLE and depend on the input.


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("🎯 LANGGRAPH WORKFLOWS - Lesson 4: Orchestrator-Workers")
    logger.info("=" * 70)

    logger.info("\n Demo 1: Report Generation (Dynamic Section Planning)")
    demo_report_generation()

    # Small delay between demos to avoid Groq rate limiting
    # (both demos fire multiple parallel workers, which can hit TPM limits)
    logger.info("\n  Waiting 15s before Demo 2 to avoid Groq rate limits...")
    time.sleep(15)

    logger.info("\n Demo 2: Production Code Review Pipeline")
    demo_code_review_pipeline()

    logger.info("\n" + "=" * 70)
    logger.info("Lesson 4 complete - Orchestrator-Workers with Send API")
    logger.info("You can now build dynamic planning workflows with parallel workers.")
    logger.info("Next: Lesson 5 - Evaluator/Optimizer Pattern")
    logger.info("=" * 70)
