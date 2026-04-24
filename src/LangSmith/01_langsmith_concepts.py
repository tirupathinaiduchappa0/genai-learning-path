"""
🔍 LangSmith Lesson 1 — LangSmith Concepts & Production Usage

═══════════════════════════════════════════════════════════════════
1. WHAT IS LANGSMITH?
═══════════════════════════════════════════════════════════════════

LangSmith is a platform by LangChain Inc. for DEBUGGING, TRACING,
MONITORING, and EVALUATING LLM applications.

Think of it as "Chrome DevTools for AI agents."

    Chrome DevTools → inspect HTTP requests, DOM, JS execution
    LangSmith       → inspect LLM calls, tool calls, agent loops, token usage

LangSmith is a SEPARATE product from LangChain/LangGraph.
LangChain = framework for building LLM apps (chains, agents, tools)
LangGraph = framework for building stateful agent workflows (graphs)
LangSmith = the OBSERVABILITY LAYER that sits on top of both

    URL: https://smith.langchain.com

Without LangSmith, you're flying blind. With it, you see EVERYTHING
your agent does — every LLM call, every tool invocation, every decision.

═══════════════════════════════════════════════════════════════════
2. WHY LANGSMITH IS NEEDED
═══════════════════════════════════════════════════════════════════

Problem 1: LLMs are BLACK BOXES
    You send a prompt, you get a response. But what happened inside?
    Which tokens were generated first? Why did the model choose that answer?
    LangSmith can't see INSIDE the model, but it shows you the INPUTS
    and OUTPUTS of every step — which is the next best thing.

Problem 2: Agents make DECISIONS
    A ReAct agent decides: which tool to call, what arguments to pass,
    when to stop looping. Without tracing, debugging an agent is like
    debugging code without logs — you see the final output but not
    the reasoning path.

    Example: "Why did my agent call Wikipedia instead of the calculator?"
    Without LangSmith: shrug
    With LangSmith: Open the trace → see the LLM's reasoning → see the
                    tool_calls in the AI message → understand the decision.

Problem 3: Production needs METRICS
    In production, you need:
    - Cost tracking: How much is each query costing me?
    - Latency monitoring: Are responses fast enough for users?
    - Error detection: Which runs are failing and why?
    - Quality monitoring: Are responses getting worse over time?

    LangSmith provides ALL of this out of the box.

═══════════════════════════════════════════════════════════════════
3. WHAT LANGSMITH TRACKS (with examples)
═══════════════════════════════════════════════════════════════════

TRACES — The complete execution path of a chain/agent:
    Every LLM call:
        Input: [SystemMessage("You are helpful"), HumanMessage("What is 5+5?")]
        Output: AIMessage("5 + 5 = 10")
        Tokens: input=25, output=8, total=33
        Latency: 340ms
        Model: llama-3.1-8b-instant

    Every tool call:
        Tool: add
        Arguments: {"a": 5, "b": 5}
        Result: 10
        Latency: 1ms

    Every node in a LangGraph:
        Node: "tool_calling_llm" entered at t=0ms, exited at t=350ms
        Node: "tools" entered at t=350ms, exited at t=352ms
        State at each step: full messages list visible

TOKEN USAGE:
    Input tokens: how many tokens in the prompt (system + history + user)
    Output tokens: how many tokens the model generated
    Total tokens: input + output (this is what you PAY for)
    Per-run breakdown: see which runs are token-heavy

COST ANALYSIS:
    Estimated cost per run based on model pricing
    Example: GPT-4 at $30/1M input tokens, $60/1M output tokens
    LangSmith calculates: "This run cost $0.003"
    Helps you optimize: switch to cheaper models for simple tasks

LATENCY:
    Time taken for each step:
    - LLM call: 200ms-2000ms (depends on model, tokens, provider)
    - Tool execution: 1ms-5000ms (depends on the tool)
    - Total run: sum of all steps
    Helps you find bottlenecks: "The Wikipedia tool takes 3 seconds!"

ERROR TRACKING:
    Failed runs with full stack traces
    Exceptions in tool calls
    Timeout errors
    Rate limit errors from LLM providers
    Retry attempts and their outcomes

═══════════════════════════════════════════════════════════════════
4. HOW TO SET UP LANGSMITH (3 steps)
═══════════════════════════════════════════════════════════════════

Step 1: Create account at https://smith.langchain.com
    - Sign up with GitHub or Google
    - Free tier includes generous trace limits

Step 2: Get API key from Settings → API Keys
    - Click "Create API Key"
    - Copy the key (starts with lsv2_pt_...)

Step 3: Add to your .env file:
    LANGSMITH_API_KEY=lsv2_pt_...
    LANGCHAIN_TRACING_V2=true
    LANGCHAIN_PROJECT=my-project-name

That's it! LangChain/LangGraph AUTO-SENDS traces when these env vars
are set. No code changes needed — just set the env vars and every
chain.invoke() or graph.invoke() is automatically traced.

HOW IT WORKS UNDER THE HOOD:
    1. Your code calls graph.invoke(...)
    2. LangChain detects LANGCHAIN_TRACING_V2=true
    3. It wraps every LLM call, tool call, and node execution with callbacks
    4. These callbacks send trace data to the LangSmith API (async, non-blocking)
    5. You see the traces in the LangSmith UI within seconds

IMPORTANT: Tracing is ASYNCHRONOUS — it does NOT slow down your app.
The trace data is sent in the background after each step completes.

═══════════════════════════════════════════════════════════════════
5. LANGSMITH UI FEATURES
═══════════════════════════════════════════════════════════════════

TRACES VIEW:
    See every run with a full execution tree.
    Expand each node to see inputs, outputs, tokens, latency.
    Filter by: project, tags, metadata, date range, status (success/error).

    Example trace tree for a ReAct agent:
    RunnableSequence (total: 1.2s)
    ├── tool_calling_llm (340ms, 33 tokens)
    ├── tools: add(5, 5) → 10 (1ms)
    ├── tool_calling_llm (280ms, 28 tokens)
    ├── tools: multiply(10, 3) → 30 (1ms)
    └── tool_calling_llm (250ms, 22 tokens) → "The answer is 30"

PLAYGROUND:
    Test prompts interactively without writing code.
    Change the system message, temperature, model — see results instantly.
    Great for prompt engineering and A/B testing.

DATASETS:
    Create test datasets with input/expected_output pairs.
    Run your chain against the dataset to measure quality.
    Track quality over time as you change prompts or models.

EVALUATORS:
    Automated quality scoring using LLM-as-judge or custom functions.
    Example: "Is the response factually correct?" → LLM scores 0-1.
    Run evaluators on datasets to get aggregate quality metrics.

MONITORING:
    Dashboards for production metrics:
    - Traces per day, success rate, average latency
    - Token usage trends, cost trends
    - Error rate by error type
    - P50/P95/P99 latency percentiles

═══════════════════════════════════════════════════════════════════
6. PRODUCTION USE CASES
═══════════════════════════════════════════════════════════════════

USE CASE 1: Debugging Agent Loops
    Problem: "My agent is stuck in an infinite loop calling the same tool."
    Solution: Open the trace in LangSmith → see the message flow →
              identify where the LLM keeps requesting the same tool_call →
              fix the prompt or add a recursion_limit.

USE CASE 2: Cost Optimization
    Problem: "My monthly LLM bill is $5,000. Which runs are expensive?"
    Solution: Filter traces by cost → find the top 10 most expensive runs →
              analyze: are they using GPT-4 when GPT-3.5 would suffice?
              Are prompts too long? Is conversation history not being trimmed?

USE CASE 3: Quality Monitoring
    Problem: "Users are complaining that responses are getting worse."
    Solution: Set up evaluators on a test dataset → run weekly →
              track scores over time → alert when quality drops below threshold.

USE CASE 4: A/B Testing
    Problem: "I have two prompt versions. Which one is better?"
    Solution: Tag runs with "prompt_v1" and "prompt_v2" →
              compare metrics (quality scores, latency, cost) side by side →
              pick the winner with data, not gut feeling.

USE CASE 5: Compliance & Audit Trail
    Problem: "We need an audit trail of all LLM interactions for compliance."
    Solution: LangSmith stores every run with full inputs/outputs →
              filter by date range, user_id, project → export for auditors.
              Metadata tags make filtering easy: config={"metadata": {"user_id": "u123"}}

═══════════════════════════════════════════════════════════════════
7. LANGSMITH BEST PRACTICES (from Important-Rules.md)
═══════════════════════════════════════════════════════════════════

RULE 1: Always name your runs
    config={"run_name": "RAG Query"}
    This makes traces searchable and readable in the UI.
    Without names, you get generic "RunnableSequence" labels everywhere.

RULE 2: Use tags to group related runs
    config={"tags": ["production", "v2"]}
    Tags let you filter traces: "Show me all production runs" or
    "Show me all v2 prompt runs." Essential for A/B testing.

RULE 3: Use metadata for filtering
    config={"metadata": {"user_id": "u123", "session_id": "s456"}}
    Metadata is key-value pairs attached to each run.
    Use it to trace runs back to specific users, sessions, or features.

RULE 4: Check token usage per run in the dashboard
    Token usage = cost. Monitor it to catch expensive runs early.
    Set alerts for runs that exceed a token threshold.

RULE 5: Set up evaluators for automated quality scoring
    Don't rely on manual review — automate quality checks.
    Use LLM-as-judge evaluators or custom Python functions.
    Run evaluators on datasets to get aggregate quality metrics.

COMBINED EXAMPLE:
    result = graph.invoke(
        {"messages": [HumanMessage(content="What is LangGraph?")]},
        config={
            "run_name": "LangGraph Explainer",
            "tags": ["production", "v2", "explainer"],
            "metadata": {"user_id": "krish", "feature": "chatbot"},
        },
    )

═══════════════════════════════════════════════════════════════════
8. INTERVIEW QUESTIONS
═══════════════════════════════════════════════════════════════════

Q1: What is LangSmith and how does it differ from LangChain?
A1: LangSmith is an OBSERVABILITY platform for debugging, tracing,
    monitoring, and evaluating LLM applications. LangChain is a
    FRAMEWORK for building LLM apps (chains, agents, tools).
    LangSmith sits ON TOP of LangChain/LangGraph — it doesn't build
    the app, it WATCHES the app run and gives you visibility into
    every LLM call, tool call, token usage, and latency.
    Setup: just set LANGCHAIN_TRACING_V2=true and LANGSMITH_API_KEY
    in your .env — LangChain auto-sends traces.

Q2: How would you use LangSmith to debug a ReAct agent that's
    calling the wrong tool?
A2: Open the trace in LangSmith UI → expand the execution tree →
    look at the AI message that contains tool_calls → read the LLM's
    reasoning (the content before the tool_call) → check if the tool
    descriptions are clear enough for the LLM to pick the right one.
    Common fix: improve tool docstrings so the LLM understands when
    to use each tool.

Q3: What LangSmith best practices would you follow in production?
A3: (1) Always name runs with config={"run_name": "..."} for searchability.
    (2) Use tags for grouping: config={"tags": ["production", "v2"]}.
    (3) Use metadata for user-level tracing: config={"metadata": {"user_id": "..."}}.
    (4) Monitor token usage per run to control costs.
    (5) Set up automated evaluators on test datasets for quality monitoring.
    (6) Set recursion_limit on agent graphs to prevent infinite loops.

═══════════════════════════════════════════════════════════════════

HOW TO RUN:
    $ python src/LangSmith/01_langsmith_concepts.py

Author: GenAI Learner
Date: 2025-07-15
"""

import logging
import os

from dotenv import load_dotenv

# ── Load environment variables ───────────────────────────────────────────
load_dotenv()

os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "langgraph-tutorial")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO: Verify LangSmith Connection
# ═══════════════════════════════════════════════════════════════════════════════
# This small demo verifies that LangSmith env vars are set correctly
# and makes a simple LLM call so a trace appears in the LangSmith UI.

def demo_verify_langsmith_connection() -> None:
    """Verify LangSmith is configured and send a test trace."""
    from langchain_core.messages import HumanMessage
    from langchain_groq import ChatGroq

    # ── Step 1: Check env vars ───────────────────────────────────────────
    langsmith_key = os.environ.get("LANGSMITH_API_KEY", "")
    tracing_enabled = os.environ.get("LANGCHAIN_TRACING_V2", "")
    project_name = os.environ.get("LANGCHAIN_PROJECT", "")

    logger.info("--- LangSmith Connection Check ---")
    logger.info("  LANGSMITH_API_KEY set: %s", "YES" if langsmith_key else "NO (traces will NOT be sent!)")
    logger.info("  LANGCHAIN_TRACING_V2:  %s", tracing_enabled)
    logger.info("  LANGCHAIN_PROJECT:     %s", project_name)

    if not langsmith_key:
        logger.warning("  LANGSMITH_API_KEY is not set! Add it to your .env file.")
        logger.warning("  Get your key at: https://smith.langchain.com → Settings → API Keys")
        return

    if tracing_enabled.lower() != "true":
        logger.warning("  LANGCHAIN_TRACING_V2 is not 'true'. Traces will NOT be sent.")
        return

    logger.info("  All env vars are set. Sending a test trace...")

    # ── Step 2: Make a simple LLM call (this sends a trace to LangSmith) ─
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_tokens=128)

    response = llm.invoke(
        [HumanMessage(content="What is LangSmith in one sentence?")],
        config={
            "run_name": "LangSmith Connection Test",
            "tags": ["test", "lesson-01"],
            "metadata": {"purpose": "verify_langsmith_connection"},
        },
    )

    logger.info("  LLM Response: %s", response.content[:200])
    logger.info("")
    logger.info("  ✅ Trace sent! Check LangSmith UI at: https://smith.langchain.com")
    logger.info("  Look for project: '%s'", project_name)
    logger.info("  Look for run named: 'LangSmith Connection Test'")
    logger.info("  You should see: the input message, output, token usage, and latency.")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("🔍 LANGSMITH LESSON 1 — Concepts & Production Usage")
    logger.info("=" * 70)

    logger.info("\n🔹 Verifying LangSmith Connection")
    demo_verify_langsmith_connection()

    logger.info("\n" + "=" * 70)
    logger.info("✅ Lesson 1 complete — LangSmith Concepts & Connection Verified")
    logger.info("Read the module docstring for the full conceptual guide.")
    logger.info("Next: Lesson 2 — Hands-On Agent with LangSmith Tracing")
    logger.info("=" * 70)
