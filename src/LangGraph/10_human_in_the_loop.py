"""
LangGraph Lesson 10 - Human-in-the-Loop: Interrupts, Approval, and Editing (Level 5)

CONCEPT: Human-in-the-Loop (HITL) - ONE-LINE DEFINITION

Human-in-the-Loop is a pattern where the graph PAUSES execution at
a specific node, surfaces the current state to a human, and WAITS
for the human to approve, reject, or edit the state before resuming.

WHY HUMAN-IN-THE-LOOP EXISTS (the problem it solves)

Fully autonomous AI agents are powerful but DANGEROUS in production:

    Problem 1: IRREVERSIBLE ACTIONS
        An agent that sends emails, creates Jira tickets, or executes
        database queries can cause REAL DAMAGE if it makes a mistake.
        You need a human to APPROVE before the action executes.

    Problem 2: HALLUCINATION RISK
        LLMs can generate plausible but WRONG content. In medical,
        legal, or financial domains, wrong output = liability.
        A human must REVIEW before the output is finalized.

    Problem 3: INFINITE LOOPS
        A ReAct agent with tools can get stuck in a loop, calling
        the same tool repeatedly. A human can INTERVENE and redirect.

    Problem 4: CONTEXT CORRECTION
        The agent may misunderstand the user's intent. The human
        needs to EDIT the state (e.g., change the query) mid-execution.

In LangChain, there was NO built-in way to pause and resume.
You had to hack around it with callbacks or external queues.

In LangGraph, HITL is a FIRST-CLASS feature:
    - interrupt_before: Pause BEFORE a node executes.
    - interrupt_after: Pause AFTER a node executes (review output).
    - graph.get_state(): Inspect the current state while paused.
    - graph.update_state(): Edit the state before resuming.
    - graph.stream(None, config): Resume execution from where it paused.
    - Command(resume=...): Resume with explicit human feedback.

PREREQUISITE: Lesson 08 (Memory & Checkpointing).
    HITL REQUIRES a checkpointer. Without persistence, the graph
    cannot pause and resume. MemorySaver is the minimum requirement.

WHERE IT FITS IN THE LANGGRAPH ARCHITECTURE

    Level 1: StateGraph, Nodes, Edges, Compile, Invoke
    Level 2: Conditional Edges, Routing, END node
    Level 3: ReAct loop, ToolNode, LLM + Tools
    Level 4: Checkpointers, Thread IDs, State Reducers  <-- Lesson 08
    Level 5: HUMAN-IN-THE-LOOP (THIS LESSON)             <-- YOU ARE HERE
    Level 6: Multi-Agent, Supervisor, Subgraphs
    Level 7: Parallel execution, Map-reduce, Workflows

    HITL builds on top of checkpointing (Level 4).
    Without a checkpointer, there is NO way to pause and resume.

    Concept Linking:
        MemorySaver -> enables -> persistence -> enables -> HITL
        interrupt_before -> pauses -> BEFORE a node runs
        interrupt_after  -> pauses -> AFTER a node runs (review output)
        graph.get_state() -> inspects -> current state while paused
        graph.update_state() -> edits -> state before resuming
        graph.stream(None, config) -> resumes -> from the pause point
        HITL -> enables -> Approval, Editing, Debugging, Safety

REAL-WORLD ANALOGY: A DOCUMENT APPROVAL WORKFLOW

Think of a CORPORATE DOCUMENT APPROVAL process:

    1. An employee DRAFTS a proposal (agent generates content).
    2. The system PAUSES and sends it to the MANAGER for review.
    3. The manager can:
       a) APPROVE -> the proposal moves to the next stage.
       b) REJECT  -> the proposal is discarded or sent back.
       c) EDIT    -> the manager modifies the proposal, then it continues.
    4. After approval, the system RESUMES and sends the proposal
       to the client.

    The system does NOT send the proposal without approval.
    The manager has FULL CONTROL over what goes out.

    This is EXACTLY how interrupt_before works in LangGraph:
        - The graph pauses before the "send_to_client" node.
        - The human reviews the draft.
        - The human approves, rejects, or edits.
        - The graph resumes.

THE THREE HITL PATTERNS (from the notebook, explained clearly)

Pattern 1: APPROVAL (interrupt_before)
    Graph pauses BEFORE a critical node.
    Human reviews the pending action.
    Human approves (resume) or rejects (stop).
    Use case: Approve an email before sending, approve a DB query.

Pattern 2: EDITING (update_state)
    Graph pauses at a checkpoint.
    Human inspects the state via get_state().
    Human MODIFIES the state via update_state().
    Graph resumes with the EDITED state.
    Use case: Fix a wrong query, change parameters, correct a draft.

Pattern 3: FEEDBACK LOOP (human_feedback node)
    A dedicated "human_feedback" node exists in the graph.
    The graph pauses before this node (interrupt_before).
    The human provides NEW input (not just approve/reject).
    The graph resumes with the human's input as new state.
    Use case: Multi-turn agent where human guides each step.

HOW interrupt_before vs interrupt_after DIFFER

    interrupt_before=["node_name"]:
        Graph pauses BEFORE node_name executes.
        The node has NOT run yet.
        Human can approve (let it run) or edit state first.
        Use case: "Should I send this email?" -> Approve -> Send.

    interrupt_after=["node_name"]:
        Graph pauses AFTER node_name has executed.
        The node HAS already run. Its output is in state.
        Human can review the output and decide what's next.
        Use case: "Here's the draft I wrote." -> Review -> Continue.

    KEY INSIGHT: interrupt_before is for GATEKEEPING (prevent bad actions).
                 interrupt_after is for REVIEWING (inspect completed work).

ASCII GRAPH STRUCTURES

Demo 1: Email Drafting with Approval Gate

    [START]
       |
    [draft_email]  <- LLM drafts the email
       |
    --- INTERRUPT (human reviews draft) ---
       |
    [send_email]   <- Only runs if human approves
       |
    [END]

Demo 2: Customer Support with Human Escalation

    [START]
       |
    [classify_query]  <- LLM classifies the query
       |
       +-- (auto_resolve) --> [auto_respond] --> [END]
       |
       +-- (needs_human) --> [prepare_context]
                                  |
                             --- INTERRUPT (human reviews) ---
                                  |
                             [human_response]  <- Human provides response
                                  |
                             [END]

LANGCHAIN vs LANGGRAPH: Human-in-the-Loop

LangChain : No built-in HITL. You had to use callbacks, external
            queues, or custom middleware to pause and resume.
            AgentExecutor had no concept of "pause before tool call."
LangGraph  : interrupt_before / interrupt_after are FIRST-CLASS.
            Checkpointer saves state automatically.
            get_state() / update_state() give full control.
            Resume with stream(None, config) or Command(resume=...).

LangChain : HITL was a hack.
LangGraph  : HITL is a DESIGN PRINCIPLE.

HOW TO RUN:
    $ python src/LangGraph/10_human_in_the_loop.py

Author: GenAI Learner
"""

import logging
import operator
import os
from typing import Annotated, Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
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
# DEMO 1: Email Drafting Agent with Approval Gate (interrupt_before)
# ==============================================================================
#
# SCENARIO: A marketing team uses an AI agent to draft promotional emails.
# Before any email is SENT, a human manager must APPROVE the draft.
# The manager can:
#   - APPROVE: The email is sent as-is.
#   - EDIT: The manager modifies the draft, then it's sent.
#   - REJECT: The email is discarded.
#
# KEY CONCEPTS DEMONSTRATED:
#   1. interrupt_before=["send_email"] pauses before sending.
#   2. graph.get_state(config) inspects the draft while paused.
#   3. graph.update_state(config, new_values) edits the draft.
#   4. graph.stream(None, config) resumes after approval.
#   5. MemorySaver is required for pause/resume to work.

def demo_email_approval() -> None:
    """Email drafting agent with human approval gate before sending."""

    # -- State: holds the email workflow data ----------------------------------
    class EmailState(TypedDict):
        request: str        # What the human wants the email to be about
        draft: str          # The LLM-generated email draft
        status: str         # Workflow status: drafting/pending/approved/sent
        feedback: str       # Optional human feedback when editing

    # -- LLM for drafting emails -----------------------------------------------
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7, max_tokens=512)

    # -- Node 1: draft_email - LLM generates the email draft -------------------
    def draft_email(state: EmailState) -> dict:
        """LLM drafts a promotional email based on the request."""
        logger.info("  [draft_email] Drafting email for: '%s'", state["request"][:80])

        feedback_context = ""
        if state.get("feedback"):
            feedback_context = f"\n\nPrevious feedback to incorporate: {state['feedback']}"

        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a professional email copywriter. "
                        "Write a concise, engaging promotional email. "
                        "Include: Subject line, Greeting, Body (2-3 paragraphs), CTA, Sign-off. "
                        "Keep it under 150 words."
                    )
                ),
                HumanMessage(
                    content=f"Write an email about: {state['request']}{feedback_context}"
                ),
            ]
        )

        logger.info("  [draft_email] Draft generated (%d chars)", len(response.content))
        return {"draft": response.content, "status": "pending_approval"}

    # -- Node 2: send_email - sends the approved email -------------------------
    # This node ONLY runs after human approval.
    # In production, this would call an email API (SendGrid, SES, etc.).
    def send_email(state: EmailState) -> dict:
        """Send the approved email (simulated)."""
        logger.info("  [send_email] Sending approved email...")
        logger.info("  [send_email] Email content preview: %s", state["draft"][:100])
        return {"status": "sent"}

    # -- Build the graph -------------------------------------------------------
    builder = StateGraph(EmailState)

    builder.add_node("draft_email", draft_email)
    builder.add_node("send_email", send_email)

    # Flow: START -> draft_email -> send_email -> END
    builder.add_edge(START, "draft_email")
    builder.add_edge("draft_email", "send_email")
    builder.add_edge("send_email", END)

    # Compile with checkpointer AND interrupt_before
    # interrupt_before=["send_email"] means:
    #   The graph will PAUSE right before send_email runs.
    #   The human can then inspect, approve, edit, or reject.
    memory = MemorySaver()
    graph = builder.compile(
        checkpointer=memory,
        interrupt_before=["send_email"],  # PAUSE before sending
    )

    save_graph_image(graph, "lesson10_email_approval")

    # -- Scenario A: Human APPROVES the draft ----------------------------------
    logger.info("--- Demo 1A: Email Approval (APPROVE flow) ---")

    config_a = {"configurable": {"thread_id": "email-approve-1"}}

    # Step 1: Run the graph - it will draft the email and then PAUSE
    logger.info("  Step 1: Running graph (will pause before send_email)...")
    for event in graph.stream(
        {"request": "Summer sale - 30% off all courses this weekend", "draft": "", "status": "", "feedback": ""},
        config_a,
        stream_mode="values",
    ):
        logger.info("  State status: %s", event.get("status", "starting"))

    # Step 2: Inspect the state while paused
    paused_state = graph.get_state(config_a)
    logger.info("  Step 2: Graph is PAUSED. Next node: %s", paused_state.next)
    logger.info("  Draft preview: %s", paused_state.values["draft"][:200])

    # Step 3: Human APPROVES - just resume without changes
    logger.info("  Step 3: Human APPROVES the draft. Resuming...")
    for event in graph.stream(None, config_a, stream_mode="values"):
        logger.info("  State status: %s", event.get("status", ""))

    final_state = graph.get_state(config_a)
    logger.info("  Final status: %s", final_state.values["status"])
    logger.info("  Approval flow complete: draft -> pause -> approve -> send")
    logger.info("")

    # -- Scenario B: Human EDITS the draft before approving --------------------
    logger.info("--- Demo 1B: Email Approval (EDIT flow) ---")

    config_b = {"configurable": {"thread_id": "email-edit-1"}}

    # Step 1: Run the graph - it will draft and PAUSE
    logger.info("  Step 1: Running graph (will pause before send_email)...")
    for event in graph.stream(
        {"request": "New AI course launch announcement", "draft": "", "status": "", "feedback": ""},
        config_b,
        stream_mode="values",
    ):
        logger.info("  State status: %s", event.get("status", "starting"))

    # Step 2: Inspect the paused state
    paused_state = graph.get_state(config_b)
    logger.info("  Step 2: Graph is PAUSED. Reviewing draft...")
    logger.info("  Current draft: %s", paused_state.values["draft"][:200])

    # Step 3: Human EDITS the draft by updating state directly
    # update_state() lets you modify ANY field in the state.
    # Here the human replaces the draft with their own version.
    logger.info("  Step 3: Human EDITS the draft via update_state()...")
    edited_draft = (
        "Subject: Exciting News - Our New AI Mastery Course is Live!\n\n"
        "Hi there,\n\n"
        "We are thrilled to announce our brand-new AI Mastery Course! "
        "Learn LangChain, LangGraph, and production AI patterns from scratch. "
        "Enroll today and get 20%% off with code AI2025.\n\n"
        "Best regards,\nThe Learning Team"
    )
    graph.update_state(config_b, {"draft": edited_draft, "status": "edited_by_human"})

    # Verify the edit took effect
    edited_state = graph.get_state(config_b)
    logger.info("  Edited draft: %s", edited_state.values["draft"][:200])

    # Step 4: Resume - send_email will use the EDITED draft
    logger.info("  Step 4: Resuming with edited draft...")
    for event in graph.stream(None, config_b, stream_mode="values"):
        logger.info("  State status: %s", event.get("status", ""))

    final_state = graph.get_state(config_b)
    logger.info("  Final status: %s", final_state.values["status"])
    logger.info("  Edit flow complete: draft -> pause -> edit -> resume -> send")
    logger.info("")

    # -- Scenario C: Human REJECTS the draft -----------------------------------
    logger.info("--- Demo 1C: Email Approval (REJECT flow) ---")

    config_c = {"configurable": {"thread_id": "email-reject-1"}}

    # Step 1: Run the graph - it will draft and PAUSE
    logger.info("  Step 1: Running graph (will pause before send_email)...")
    for event in graph.stream(
        {"request": "Flash sale on outdated products", "draft": "", "status": "", "feedback": ""},
        config_c,
        stream_mode="values",
    ):
        logger.info("  State status: %s", event.get("status", "starting"))

    # Step 2: Human reviews and REJECTS
    paused_state = graph.get_state(config_c)
    logger.info("  Step 2: Graph is PAUSED. Human reviews draft...")
    logger.info("  Draft preview: %s", paused_state.values["draft"][:200])

    # Step 3: Human rejects by updating status - does NOT resume
    # In a rejection, you simply don't call stream(None, config).
    # The graph stays paused forever. The email is never sent.
    logger.info("  Step 3: Human REJECTS the draft. NOT resuming.")
    graph.update_state(config_c, {"status": "rejected_by_human"})
    rejected_state = graph.get_state(config_c)
    logger.info("  Final status: %s", rejected_state.values["status"])
    logger.info("  Reject flow complete: draft -> pause -> reject (no send)")
    logger.info("  The email was NEVER sent. This is the safety of HITL.")


# ==============================================================================
# DEMO 2: Customer Support Agent with Human Escalation (interrupt_after)
# ==============================================================================
#
# SCENARIO: A customer support system where:
#   - Simple queries are auto-resolved by the AI agent.
#   - Complex/sensitive queries are ESCALATED to a human agent.
#   - The AI prepares context for the human, then PAUSES.
#   - The human reviews the context and provides a response.
#   - The system delivers the human's response to the customer.
#
# This demonstrates:
#   1. interrupt_after=["prepare_context"] pauses AFTER context is prepared.
#   2. The human sees the AI's analysis and the customer's query.
#   3. The human provides a response via update_state().
#   4. The graph resumes and delivers the response.
#
# WHY interrupt_after (not interrupt_before)?
#   We WANT prepare_context to run first (it does useful work).
#   We pause AFTER it runs so the human can see its output.
#   The human then decides what response to give.

def demo_support_escalation() -> None:
    """Customer support with AI triage and human escalation."""

    # -- State: holds the support workflow data --------------------------------
    class SupportState(TypedDict):
        customer_query: str     # The customer's original query
        category: str           # AI classification: auto_resolve or needs_human
        ai_analysis: str        # AI's analysis/context for the human
        response: str           # Final response to the customer
        resolved_by: str        # "ai" or "human"

    # -- LLMs ------------------------------------------------------------------
    classifier_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, max_tokens=128)
    responder_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3, max_tokens=512)

    # -- Pydantic model for classification -------------------------------------
    class QueryClassification(BaseModel):
        category: Literal["auto_resolve", "needs_human"] = Field(
            ...,
            description=(
                "auto_resolve: simple FAQ, greeting, or straightforward question. "
                "needs_human: complaint, refund request, account issue, or sensitive topic."
            ),
        )

    classifier = classifier_llm.with_structured_output(QueryClassification)

    # -- Node 1: classify_query - AI classifies the query ----------------------
    def classify_query(state: SupportState) -> dict:
        """Classify the customer query as auto-resolvable or needing human."""
        logger.info("  [classify] Classifying: '%s'", state["customer_query"][:80])

        result = classifier.invoke(
            [
                SystemMessage(
                    content=(
                        "Classify the customer query. "
                        "auto_resolve: simple questions, greetings, FAQs, how-to questions. "
                        "needs_human: complaints, refund requests, account issues, "
                        "billing disputes, sensitive topics, angry customers."
                    )
                ),
                HumanMessage(content=state["customer_query"]),
            ]
        )

        logger.info("  [classify] Category: %s", result.category)
        return {"category": result.category}

    # -- Route function: decides next node based on classification -------------
    def route_query(state: SupportState) -> str:
        """Route to auto_respond or prepare_context based on classification."""
        if state["category"] == "auto_resolve":
            return "auto_respond"
        return "prepare_context"

    # -- Node 2a: auto_respond - AI handles simple queries ---------------------
    def auto_respond(state: SupportState) -> dict:
        """AI automatically responds to simple queries."""
        logger.info("  [auto_respond] Generating AI response...")

        response = responder_llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a friendly customer support agent. "
                        "Respond helpfully and concisely to this simple query."
                    )
                ),
                HumanMessage(content=state["customer_query"]),
            ]
        )

        logger.info("  [auto_respond] AI response generated (%d chars)", len(response.content))
        return {"response": response.content, "resolved_by": "ai"}

    # -- Node 2b: prepare_context - AI prepares context for human --------------
    # This node runs BEFORE the human sees anything.
    # It analyzes the query and prepares a summary for the human agent.
    # After this node, the graph PAUSES (interrupt_after).
    def prepare_context(state: SupportState) -> dict:
        """AI prepares context and analysis for the human agent."""
        logger.info("  [prepare_context] Preparing context for human escalation...")

        analysis = responder_llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are an AI assistant helping a human support agent. "
                        "Analyze this customer query and provide:\n"
                        "1. Summary of the issue\n"
                        "2. Customer sentiment (angry/frustrated/neutral)\n"
                        "3. Suggested response approach\n"
                        "4. Any relevant policies that may apply\n"
                        "Keep it concise - this is for the human agent's reference."
                    )
                ),
                HumanMessage(content=state["customer_query"]),
            ]
        )

        logger.info("  [prepare_context] Context prepared for human review")
        return {"ai_analysis": analysis.content}

    # -- Node 3: deliver_response - delivers the final response ----------------
    def deliver_response(state: SupportState) -> dict:
        """Deliver the response to the customer."""
        logger.info("  [deliver] Delivering response (resolved by: %s)", state["resolved_by"])
        return {}  # Response is already in state

    # -- Build the graph -------------------------------------------------------
    builder = StateGraph(SupportState)

    builder.add_node("classify_query", classify_query)
    builder.add_node("auto_respond", auto_respond)
    builder.add_node("prepare_context", prepare_context)
    builder.add_node("deliver_response", deliver_response)

    builder.add_edge(START, "classify_query")

    # Conditional routing: auto_resolve or needs_human
    builder.add_conditional_edges(
        "classify_query",
        route_query,
        {"auto_respond": "auto_respond", "prepare_context": "prepare_context"},
    )

    builder.add_edge("auto_respond", "deliver_response")
    builder.add_edge("prepare_context", "deliver_response")
    builder.add_edge("deliver_response", END)

    # Compile with interrupt_after on prepare_context
    # The graph pauses AFTER prepare_context runs, so the human
    # can see the AI's analysis before providing a response.
    memory = MemorySaver()
    graph = builder.compile(
        checkpointer=memory,
        interrupt_after=["prepare_context"],  # PAUSE after context is prepared
    )

    save_graph_image(graph, "lesson10_support_escalation")

    # -- Scenario A: Simple query - auto-resolved (no human needed) ------------
    logger.info("--- Demo 2A: Auto-Resolved Query (no human needed) ---")

    config_auto = {"configurable": {"thread_id": "support-auto-1"}}

    for event in graph.stream(
        {
            "customer_query": "What are your business hours on weekends?",
            "category": "", "ai_analysis": "", "response": "", "resolved_by": "",
        },
        config_auto,
        stream_mode="values",
    ):
        if event.get("response"):
            logger.info("  AI Response: %s", event["response"][:200])

    final = graph.get_state(config_auto)
    logger.info("  Resolved by: %s", final.values.get("resolved_by", ""))
    logger.info("  Flow: classify -> auto_resolve -> deliver -> END (no human)")
    logger.info("")

    # -- Scenario B: Complex query - escalated to human ------------------------
    logger.info("--- Demo 2B: Escalated Query (human intervention) ---")

    config_human = {"configurable": {"thread_id": "support-human-1"}}

    # Step 1: Run the graph - it will classify, prepare context, then PAUSE
    logger.info("  Step 1: Running graph (will pause after prepare_context)...")
    for event in graph.stream(
        {
            "customer_query": "I was charged twice for my subscription and I want a full refund immediately!",
            "category": "", "ai_analysis": "", "response": "", "resolved_by": "",
        },
        config_human,
        stream_mode="values",
    ):
        if event.get("ai_analysis"):
            logger.info("  AI Analysis prepared: %s", event["ai_analysis"][:200])

    # Step 2: Graph is PAUSED after prepare_context
    paused = graph.get_state(config_human)
    logger.info("  Step 2: Graph PAUSED. Next node: %s", paused.next)
    logger.info("  AI Analysis for human: %s", paused.values["ai_analysis"][:300])

    # Step 3: Human agent provides their response via update_state
    logger.info("  Step 3: Human agent provides response via update_state()...")
    human_response = (
        "Dear Customer,\n\n"
        "I sincerely apologize for the double charge on your subscription. "
        "I have initiated a full refund for the duplicate charge, which will "
        "appear in your account within 3-5 business days. I have also added "
        "a complimentary month to your subscription as an apology for the "
        "inconvenience.\n\n"
        "If you have any further concerns, please don't hesitate to reach out.\n\n"
        "Best regards,\nSupport Team"
    )
    graph.update_state(
        config_human,
        {"response": human_response, "resolved_by": "human"},
    )

    # Step 4: Resume - deliver_response will use the human's response
    logger.info("  Step 4: Resuming to deliver human's response...")
    for event in graph.stream(None, config_human, stream_mode="values"):
        if event.get("resolved_by"):
            logger.info("  Resolved by: %s", event["resolved_by"])

    final = graph.get_state(config_human)
    logger.info("  Final response: %s", final.values["response"][:200])
    logger.info("  Flow: classify -> prepare_context -> PAUSE -> human edits -> deliver -> END")


# ==============================================================================
# DEMO 3: Document Review Pipeline with State History (get_state_history)
# ==============================================================================
#
# SCENARIO: A content pipeline where:
#   1. AI generates a blog post draft.
#   2. AI reviews the draft for quality.
#   3. Human reviews the AI's assessment and the draft.
#   4. Human can APPROVE, EDIT, or ask for a REWRITE.
#
# This demonstrates:
#   1. interrupt_before for approval gating.
#   2. get_state_history() to see ALL previous states (time travel).
#   3. update_state() with as_node parameter to resume from a specific node.
#   4. How state history enables DEBUGGING and AUDITING.
#
# WHY get_state_history MATTERS:
#   In production, you need an AUDIT TRAIL. Who changed what? When?
#   get_state_history() returns every checkpoint, so you can:
#   - Debug: "What did the state look like at step 2?"
#   - Audit: "Who approved this? What was the original draft?"
#   - Rewind: "Go back to the state before the edit."

def demo_document_review() -> None:
    """Document review pipeline with state history and time travel."""

    # -- State -----------------------------------------------------------------
    class DocState(TypedDict):
        topic: str              # The blog post topic
        draft: str              # The generated draft
        quality_score: str      # AI's quality assessment
        final_content: str      # The approved final content
        status: str             # drafting/reviewed/approved/published

    # -- LLM -------------------------------------------------------------------
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7, max_tokens=512)

    # -- Node 1: generate_draft ------------------------------------------------
    def generate_draft(state: DocState) -> dict:
        """AI generates a blog post draft."""
        logger.info("  [generate_draft] Writing draft on: '%s'", state["topic"])

        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "Write a short blog post (3-4 paragraphs, under 200 words). "
                        "Make it informative and engaging. Use markdown formatting."
                    )
                ),
                HumanMessage(content=f"Topic: {state['topic']}"),
            ]
        )

        return {"draft": response.content, "status": "drafted"}

    # -- Node 2: review_draft --------------------------------------------------
    def review_draft(state: DocState) -> dict:
        """AI reviews the draft for quality."""
        logger.info("  [review_draft] AI reviewing draft quality...")

        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "Review this blog post draft. Provide:\n"
                        "1. Quality score (Good/Needs Improvement/Poor)\n"
                        "2. Brief feedback (1-2 sentences)\n"
                        "Be concise."
                    )
                ),
                HumanMessage(content=f"Draft to review:\n\n{state['draft']}"),
            ]
        )

        return {"quality_score": response.content, "status": "reviewed"}

    # -- Node 3: publish -------------------------------------------------------
    def publish(state: DocState) -> dict:
        """Publish the approved content."""
        logger.info("  [publish] Publishing final content...")
        # Use the draft as final content (human may have edited it)
        final = state.get("final_content") or state["draft"]
        return {"final_content": final, "status": "published"}

    # -- Build the graph -------------------------------------------------------
    builder = StateGraph(DocState)

    builder.add_node("generate_draft", generate_draft)
    builder.add_node("review_draft", review_draft)
    builder.add_node("publish", publish)

    builder.add_edge(START, "generate_draft")
    builder.add_edge("generate_draft", "review_draft")
    builder.add_edge("review_draft", "publish")
    builder.add_edge("publish", END)

    # Compile with interrupt_before publish (human must approve)
    memory = MemorySaver()
    graph = builder.compile(
        checkpointer=memory,
        interrupt_before=["publish"],
    )

    save_graph_image(graph, "lesson10_document_review")

    # -- Run the pipeline ------------------------------------------------------
    logger.info("--- Demo 3: Document Review with State History ---")

    config = {"configurable": {"thread_id": "doc-review-1"}}

    # Step 1: Run graph - generates draft, reviews it, then PAUSES before publish
    logger.info("  Step 1: Running pipeline (will pause before publish)...")
    for event in graph.stream(
        {"topic": "Why LangGraph is the Future of AI Agent Development", "draft": "", "quality_score": "", "final_content": "", "status": ""},
        config,
        stream_mode="values",
    ):
        if event.get("status"):
            logger.info("  Status: %s", event["status"])

    # Step 2: Inspect paused state
    paused = graph.get_state(config)
    logger.info("  Step 2: PAUSED before publish. Next: %s", paused.next)
    logger.info("  Draft: %s", paused.values["draft"][:200])
    logger.info("  Quality: %s", paused.values["quality_score"][:200])

    # Step 3: Demonstrate get_state_history - see ALL checkpoints
    logger.info("  Step 3: Inspecting state history (audit trail)...")
    history_count = 0
    for history_state in graph.get_state_history(config):
        history_count += 1
        status = history_state.values.get("status", "initial")
        logger.info(
            "    Checkpoint %d: status='%s', next=%s",
            history_count, status, history_state.next,
        )
        # Only show first 5 checkpoints to keep output manageable
        if history_count >= 5:
            logger.info("    ... (showing first 5 checkpoints)")
            break

    logger.info("  State history shows every step - useful for debugging and auditing.")

    # Step 4: Human approves and resumes
    logger.info("  Step 4: Human approves. Resuming to publish...")
    graph.update_state(config, {"final_content": paused.values["draft"]})

    for event in graph.stream(None, config, stream_mode="values"):
        if event.get("status"):
            logger.info("  Status: %s", event["status"])

    final = graph.get_state(config)
    logger.info("  Final status: %s", final.values["status"])
    logger.info("  Flow: generate -> review -> PAUSE -> approve -> publish -> END")


# ==============================================================================
# COMPARISON TABLE: HITL Patterns
# ==============================================================================
#
# Pattern              | Mechanism              | When It Pauses        | Use Case
# ---------------------|------------------------|-----------------------|---------------------------
# interrupt_before     | Pauses BEFORE a node   | Node has NOT run yet  | Approval gates: "Should
#                      | graph.compile(         | Human decides if it   | I send this email?"
#                      |   interrupt_before=[]) | should run at all     | Prevent irreversible actions
# ---------------------|------------------------|-----------------------|---------------------------
# interrupt_after      | Pauses AFTER a node    | Node HAS run already  | Review gates: "Here's what
#                      | graph.compile(         | Human reviews output  | I prepared. Is this OK?"
#                      |   interrupt_after=[])  | before next step      | Inspect AI analysis
# ---------------------|------------------------|-----------------------|---------------------------
# update_state         | Edits state while      | Graph is paused       | Corrections: "Change the
#                      | graph is paused        | (via either interrupt)| query from X to Y"
#                      | graph.update_state()   |                       | Edit drafts, fix errors
# ---------------------|------------------------|-----------------------|---------------------------
# get_state_history    | Views ALL checkpoints  | Anytime               | Debugging: "What happened
#                      | graph.get_state_       |                       | at step 3?"
#                      |   history(config)      |                       | Auditing, time travel
# ---------------------|------------------------|-----------------------|---------------------------
# human_feedback node  | Dedicated node for     | interrupt_before the  | Multi-turn: human provides
#                      | human input            | feedback node         | new instructions each turn
#                      | pass-through function  |                       | Guided agent workflows
# ---------------------|------------------------|-----------------------|---------------------------
#
# KEY INSIGHT:
#   interrupt_before = GATEKEEPING (prevent bad actions)
#   interrupt_after  = REVIEWING (inspect completed work)
#   update_state     = EDITING (modify state mid-execution)
#   get_state_history = AUDITING (full checkpoint trail)


# ==============================================================================
# REAL-WORLD PRODUCTION USE CASES
# ==============================================================================
#
# 1. EMAIL/COMMUNICATION AGENTS
#    Agent drafts emails, Slack messages, or customer responses.
#    Human approves before sending. Prevents embarrassing mistakes.
#
# 2. FINANCIAL TRANSACTION AGENTS
#    Agent prepares a trade, transfer, or payment.
#    Human approves before execution. Regulatory requirement.
#
# 3. CUSTOMER SUPPORT ESCALATION
#    AI handles simple queries automatically.
#    Complex/sensitive queries escalated to human agents.
#    AI prepares context to help the human respond faster.
#
# 4. CONTENT MODERATION
#    AI flags potentially harmful content.
#    Human reviews flagged content before action (remove/keep).
#
# 5. CODE DEPLOYMENT PIPELINES
#    AI generates code changes or infrastructure configs.
#    Human reviews before deploying to production.
#
# 6. MEDICAL/LEGAL AI ASSISTANTS
#    AI generates diagnoses or legal summaries.
#    Human expert reviews before delivering to patient/client.
#    Liability requires human sign-off.
#
# 7. DATA PIPELINE VALIDATION
#    AI processes and transforms data.
#    Human reviews sample outputs before full batch processing.


# ==============================================================================
# COMMON MISTAKES AND GOTCHAS
# ==============================================================================
#
# Mistake                                    | Fix
# -------------------------------------------|--------------------------------------
# Forgetting the checkpointer                | HITL REQUIRES a checkpointer.
#                                            | Without it, the graph cannot pause
#                                            | and resume. Always use MemorySaver
#                                            | (dev) or SqliteSaver (production).
# -------------------------------------------|--------------------------------------
# Forgetting thread_id in config             | Every HITL session needs a unique
#                                            | thread_id. Without it, you can't
#                                            | resume the right paused graph.
# -------------------------------------------|--------------------------------------
# Using interrupt_before when you need       | interrupt_before: node has NOT run.
# interrupt_after (or vice versa)            | interrupt_after: node HAS run.
#                                            | Choose based on whether you want
#                                            | to PREVENT or REVIEW.
# -------------------------------------------|--------------------------------------
# Calling graph.invoke() instead of          | invoke() runs to completion.
# graph.stream() for HITL                    | stream() respects interrupts.
#                                            | Always use stream() for HITL.
# -------------------------------------------|--------------------------------------
# Not checking state.next after pause        | state.next tells you which node
#                                            | will run when you resume. If it's
#                                            | empty, the graph is done.
# -------------------------------------------|--------------------------------------
# Resuming with input instead of None        | To resume a paused graph, call
#                                            | graph.stream(None, config).
#                                            | Passing new input starts a NEW run.
# -------------------------------------------|--------------------------------------
#
# ANTI-PATTERN: Using HITL for every node.
#   Only interrupt at CRITICAL decision points (send email, execute trade,
#   publish content). Too many interrupts = slow, frustrating workflow.
#   Let the AI handle routine steps autonomously.


# ==============================================================================
# WHY LANGGRAPH FOR THIS
# ==============================================================================
#
# - interrupt_before / interrupt_after are FIRST-CLASS compile options
# - Checkpointer automatically saves state at every step
# - get_state() / update_state() give full control over paused state
# - get_state_history() provides complete audit trail
# - stream(None, config) resumes exactly where it paused
# - Thread IDs isolate different HITL sessions
# - Works with any checkpointer: MemorySaver, SqliteSaver, PostgresSaver
# - Composable: HITL subgraphs inside larger autonomous workflows


# ==============================================================================
# WHERE THIS CONNECTS (Concept Linking Map)
# ==============================================================================
#
# MemorySaver -> enables -> persistence -> enables -> HITL
# interrupt_before -> pauses -> BEFORE a node runs -> GATEKEEPING
# interrupt_after  -> pauses -> AFTER a node runs  -> REVIEWING
# graph.get_state() -> inspects -> current state while paused
# graph.update_state() -> edits -> state before resuming
# graph.stream(None, config) -> resumes -> from the pause point
# get_state_history() -> returns -> all checkpoints -> AUDITING
# HITL -> builds on -> Checkpointing (Lesson 08)
# HITL -> enables -> Approval, Editing, Debugging, Safety
# HITL -> used in -> Email agents, Support bots, Financial systems
# HITL -> combines with -> Routing (escalation), Tools (tool approval)


# ==============================================================================
# INTERVIEW QUESTIONS
# ==============================================================================
#
# THEORETICAL:
# Q: What is the difference between interrupt_before and interrupt_after
#    in LangGraph? When would you use each in a production system?
# A: interrupt_before pauses the graph BEFORE a specified node executes.
#    The node has NOT run yet. This is used for GATEKEEPING - preventing
#    irreversible actions like sending emails or executing transactions
#    until a human approves. interrupt_after pauses AFTER a node has
#    executed. The node's output is already in state. This is used for
#    REVIEWING - letting a human inspect the AI's work before the next
#    step proceeds. In production, use interrupt_before for action nodes
#    (send, deploy, execute) and interrupt_after for analysis nodes
#    (classify, prepare, draft) where you want the human to see the
#    AI's output before deciding the next step.
#
# HANDS-ON:
# Q: Build a LangGraph workflow for a job application screening system.
#    The AI analyzes a resume and scores it. If the score is "strong",
#    it auto-advances to interview scheduling. If "borderline", it
#    pauses for a human recruiter to review. The recruiter can approve
#    (advance) or reject (send rejection email). Use interrupt_before.
#
# SOLUTION:
#   from typing_extensions import TypedDict
#   from langchain_groq import ChatGroq
#   from langchain_core.messages import HumanMessage, SystemMessage
#   from langgraph.checkpoint.memory import MemorySaver
#   from langgraph.graph import StateGraph, START, END
#
#   class ScreenState(TypedDict):
#       resume: str
#       score: str
#       decision: str
#       status: str
#
#   llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
#
#   def analyze_resume(state):
#       r = llm.invoke([
#           SystemMessage(content="Score this resume as 'strong', 'borderline', or 'weak'. Reply with just the score."),
#           HumanMessage(content=state["resume"]),
#       ])
#       return {"score": r.content.strip().lower(), "status": "analyzed"}
#
#   def route_score(state):
#       if "strong" in state["score"]:
#           return "schedule_interview"
#       return "human_review"
#
#   def human_review(state):
#       return {"status": "pending_human_review"}
#
#   def schedule_interview(state):
#       return {"decision": "interview_scheduled", "status": "advanced"}
#
#   def send_rejection(state):
#       return {"decision": "rejected", "status": "rejected"}
#
#   builder = StateGraph(ScreenState)
#   builder.add_node("analyze", analyze_resume)
#   builder.add_node("human_review", human_review)
#   builder.add_node("schedule_interview", schedule_interview)
#   builder.add_node("send_rejection", send_rejection)
#   builder.add_edge(START, "analyze")
#   builder.add_conditional_edges("analyze", route_score, {
#       "schedule_interview": "schedule_interview",
#       "human_review": "human_review",
#   })
#   # After human review, route to schedule or reject
#   builder.add_conditional_edges("human_review", lambda s: s.get("decision", "schedule_interview"), {
#       "schedule_interview": "schedule_interview",
#       "send_rejection": "send_rejection",
#   })
#   builder.add_edge("schedule_interview", END)
#   builder.add_edge("send_rejection", END)
#   memory = MemorySaver()
#   graph = builder.compile(checkpointer=memory, interrupt_after=["human_review"])
#   # Run: graph.stream({"resume": "...", ...}, config)
#   # Pause at human_review -> recruiter updates state -> resume
#
# BONUS (System Design):
# Q: Design a production HITL system for a financial trading agent.
#    The agent analyzes market data, proposes trades, and a human
#    trader must approve each trade above $10,000. Consider:
#    - How would you handle time-sensitive trades (timeout)?
#    - How would you implement different approval levels ($10K, $100K, $1M)?
#    - How would you audit all approvals for regulatory compliance?
#    - What checkpointer would you use in production?


# ==============================================================================
# QUICK RECAP
# ==============================================================================
#
# 1. Human-in-the-Loop uses interrupt_before (gatekeeping) or
#    interrupt_after (reviewing) to PAUSE the graph at critical
#    points. A checkpointer (MemorySaver) is REQUIRED for this.
#
# 2. While paused, use get_state() to inspect, update_state() to
#    edit, and stream(None, config) to resume. get_state_history()
#    provides a full audit trail of every checkpoint.
#
# 3. In production, use HITL for irreversible actions (send email,
#    execute trade, deploy code) and sensitive domains (medical,
#    legal, financial). Let AI handle routine steps autonomously.


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("LangGraph Lesson 10 - Human-in-the-Loop")
    logger.info("=" * 70)

    logger.info("\n Demo 1: Email Drafting with Approval Gate (interrupt_before)")
    demo_email_approval()

    logger.info("\n Demo 2: Customer Support with Human Escalation (interrupt_after)")
    demo_support_escalation()

    logger.info("\n Demo 3: Document Review with State History (get_state_history)")
    demo_document_review()

    logger.info("\n" + "=" * 70)
    logger.info("Lesson 10 complete - Human-in-the-Loop")
    logger.info("You can now build approval gates, escalation flows, and audit trails.")
    logger.info("=" * 70)
