"""
🛡️ Lesson 10.6 — Middleware & Summarization: Controlling Agent Behavior

═══════════════════════════════════════════════════════════════════
WHAT IS MIDDLEWARE?
═══════════════════════════════════════════════════════════════════

Middleware is like a SECURITY CHECKPOINT at an airport:

    Without middleware:
        Passenger (message) → Plane (LLM) → Destination (response)
        No checks, no control, no monitoring.

    With middleware:
        Passenger → Security → Passport → Plane → Customs → Destination
        Everything is monitored, filtered, and controlled!

In LangChain, middleware sits BETWEEN the user and the agent,
intercepting every interaction to:

    TRACKING:      Log messages, track tokens, debug behavior
    TRANSFORMING:  Modify prompts, format output, summarize history
    CONTROLLING:   Add retries, rate limits, guardrails, PII detection

═══════════════════════════════════════════════════════════════════
WHY MIDDLEWARE MATTERS IN PRODUCTION
═══════════════════════════════════════════════════════════════════

Without middleware, a 50-turn conversation sends ~100 messages
every turn (~50K tokens total). That's expensive and slow.

With SummarizationMiddleware, old messages are compressed:
    Before: [H1, A1, H2, A2, H3, A3, H4, A4, H5, A5] (10 messages)
    After:  [Summary of H1-A3, H4, A4, H5, A5]         (5 messages)

    Result: ~80% token savings for long conversations!

TOPICS COVERED:
    1. SummarizationMiddleware — Message-based trigger
    2. SummarizationMiddleware — Token-based trigger
    3. Checkpointing — Memory & state persistence
    4. Human-in-the-Loop — Pausing for human approval
    5. Cost Analysis — How summarization saves money
    6. Production Pattern — Complete agent with all features

NOTE: SummarizationMiddleware and create_agent() are LangChain v1.x
features that require specific versions. The demos use Groq as the LLM.
If create_agent or SummarizationMiddleware is not available in your
version, the lesson explains the concepts with detailed comments.

HOW TO RUN:
    $ python 06_middleware_summarization.py

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


# Model string for Groq — used across all demos
MODEL = "groq:llama-3.3-70b-versatile"


# ═══════════════════════════════════════════════════════════════════════════════
# 1️⃣  SummarizationMiddleware — Message-Based Trigger
# ═══════════════════════════════════════════════════════════════════════════════
#
# THE PROBLEM:
#   Every turn, you send ALL previous messages to the LLM.
#   Turn 1:  2 messages  (~50 tokens)
#   Turn 10: 20 messages (~500 tokens)
#   Turn 50: 100 messages (~2,500 tokens) — EXPENSIVE!
#
# THE SOLUTION:
#   SummarizationMiddleware auto-compresses old messages into a summary
#   when the conversation exceeds a threshold.
#
# HOW IT WORKS:
#   trigger=("messages", 10) → When message count exceeds 10, summarize
#   keep=("messages", 4)     → Keep the last 4 messages intact
#   Old messages (before the last 4) → compressed into ONE summary message
#
# REQUIRES:
#   - create_agent() from langchain.agents (LangChain v1.x)
#   - InMemorySaver() as checkpointer (needed for state management)
#   - A model for summarization (can be the same or a cheaper model)

def demo_message_based_summarization() -> None:
    """SummarizationMiddleware with message count trigger."""
    try:
        from langchain.agents import create_agent
        from langchain.agents.middleware import SummarizationMiddleware
        from langchain_core.messages import HumanMessage
        from langgraph.checkpoint.memory import InMemorySaver
    except ImportError as e:
        logger.warning("Required imports not available: %s", e)
        logger.info("SummarizationMiddleware requires LangChain v1.x with create_agent support.")
        return

    # Create agent with message-based summarization
    agent = create_agent(
        model=MODEL,
        checkpointer=InMemorySaver(),  # REQUIRED — stores conversation state
        middleware=[
            SummarizationMiddleware(
                model=MODEL,
                trigger=("messages", 10),  # Summarize when > 10 messages
                keep=("messages", 4),      # Keep last 4 messages intact
            )
        ],
    )

    config = {"configurable": {"thread_id": "msg-summarize-demo"}}

    questions = [
        "What is 2+2?",
        "What is 10*5?",
        "What is 100/4?",
        "What is 15-7?",
        "What is 3*3?",
        "What is 4*4?",
    ]

    logger.info("--- Message-Based Summarization ---")
    logger.info("Trigger: >10 messages | Keep: last 4 messages")

    for q in questions:
        response = agent.invoke(
            {"messages": [HumanMessage(content=q)]},
            config,
        )
        msg_count = len(response["messages"])
        last_answer = response["messages"][-1].content[:60]
        logger.info("  Q: %-20s → Messages: %d, A: %s", q, msg_count, last_answer)

    # After 6 turns (12 messages), summarization should have kicked in
    # Message count should DROP from 12 to ~6 (summary + last 4)
    logger.info("After 6 turns: messages went from growing to compressed!")


# ═══════════════════════════════════════════════════════════════════════════════
# 2️⃣  SummarizationMiddleware — Token-Based Trigger
# ═══════════════════════════════════════════════════════════════════════════════
#
# WHY TOKEN-BASED?
#   Message-based counts MESSAGES (all equal).
#   Token-based counts TOKENS (varies by message length).
#
#   10 short messages: "What is 2+2?" → ~50 tokens (cheap)
#   10 long messages:  "Here's a 500-word essay..." → ~5000 tokens (expensive!)
#
#   Token-based is MORE ACCURATE for cost control.
#
# trigger=("tokens", 1000) → Summarize when token count exceeds 1000
# keep=("tokens", 400)     → Keep ~400 tokens of recent messages

def demo_token_based_summarization() -> None:
    """SummarizationMiddleware with token count trigger — better for cost control."""
    try:
        from langchain.agents import create_agent
        from langchain.agents.middleware import SummarizationMiddleware
        from langchain_core.messages import HumanMessage
        from langchain_core.tools import tool
        from langgraph.checkpoint.memory import InMemorySaver
    except ImportError as e:
        logger.warning("Required imports not available: %s", e)
        return

    @tool
    def search_hotels(city: str) -> str:
        """Search hotels in a city. Returns hotel listings."""
        return f"""Hotels in {city}:
    1. Grand Hotel - 5 star, $350/night, spa, pool, gym
    2. City Inn - 4 star, $180/night, business center
    3. Budget Stay - 3 star, $75/night, free wifi"""

    agent = create_agent(
        model=MODEL,
        tools=[search_hotels],
        checkpointer=InMemorySaver(),
        middleware=[
            SummarizationMiddleware(
                model=MODEL,
                trigger=("tokens", 1000),  # Summarize when > 1000 tokens
                keep=("tokens", 400),      # Keep ~400 tokens of recent messages
            ),
        ],
    )

    config = {"configurable": {"thread_id": "token-summarize-demo"}}

    def approx_tokens(messages: list) -> int:
        """Approximate token count (~4 chars per token)."""
        return sum(len(str(m.content)) for m in messages) // 4

    cities = ["Paris", "London", "Tokyo"]

    logger.info("--- Token-Based Summarization ---")
    logger.info("Trigger: >1000 tokens | Keep: ~400 tokens")

    for city in cities:
        response = agent.invoke(
            {"messages": [HumanMessage(content=f"Find hotels in {city}")]},
            config,
        )
        msgs = response["messages"]
        tokens = approx_tokens(msgs)
        logger.info("  %s: ~%d tokens, %d messages", city, tokens, len(msgs))

    logger.info("Tool responses are LONG — token-based trigger is more precise!")


# ═══════════════════════════════════════════════════════════════════════════════
# 3️⃣  Checkpointing — Memory & State Persistence
# ═══════════════════════════════════════════════════════════════════════════════
#
# A CHECKPOINTER saves the agent's state (conversation history) so you can:
#   - Pause and resume conversations
#   - Maintain memory across multiple turns
#   - Go back to a previous state
#   - Support human-in-the-loop (pause for approval)
#
# Think of it as a SAVE GAME feature for your agent.
#
# CHECKPOINTER OPTIONS:
#   InMemorySaver()  → RAM only (dev/testing — lost on restart)
#   SqliteSaver()    → SQLite file (simple persistence)
#   PostgresSaver()  → PostgreSQL (production — distributed, durable)
#
# HOW thread_id WORKS:
#   Same thread_id    = continued conversation (agent remembers)
#   Different thread_id = fresh conversation (agent forgets)
#   This is how you support MULTIPLE users / conversations.

def demo_checkpointing() -> None:
    """Demonstrate conversation memory with checkpointing."""
    try:
        from langchain.agents import create_agent
        from langchain_core.messages import HumanMessage
        from langgraph.checkpoint.memory import InMemorySaver
    except ImportError as e:
        logger.warning("Required imports not available: %s", e)
        return

    agent = create_agent(
        model=MODEL,
        checkpointer=InMemorySaver(),
        system_prompt="You are a helpful assistant. Remember our conversation.",
    )

    config = {"configurable": {"thread_id": "memory-demo"}}

    logger.info("--- Checkpointing (Conversation Memory) ---")

    # Turn 1: Introduce yourself
    r1 = agent.invoke(
        {"messages": [HumanMessage(content="My name is Raj and I'm learning GenAI")]},
        config,
    )
    logger.info("  Turn 1 — Human: My name is Raj and I'm learning GenAI")
    logger.info("  Turn 1 — AI: %s", r1["messages"][-1].content[:120])

    # Turn 2: Agent should remember!
    r2 = agent.invoke(
        {"messages": [HumanMessage(content="What's my name and what am I learning?")]},
        config,
    )
    logger.info("  Turn 2 — Human: What's my name and what am I learning?")
    logger.info("  Turn 2 — AI: %s", r2["messages"][-1].content[:120])

    # Different thread_id = fresh conversation
    config_new = {"configurable": {"thread_id": "different-user"}}
    r3 = agent.invoke(
        {"messages": [HumanMessage(content="What's my name?")]},
        config_new,
    )
    logger.info("  New thread — AI: %s", r3["messages"][-1].content[:120])
    logger.info("  Different thread_id = no memory of previous conversation!")


# ═══════════════════════════════════════════════════════════════════════════════
# 4️⃣  Human-in-the-Loop — Pausing for Human Approval
# ═══════════════════════════════════════════════════════════════════════════════
#
# Human-in-the-Loop means the agent PAUSES for human approval before
# taking critical actions. Like a bank transaction:
#
#   Agent: "I want to transfer $10,000"
#   Human: "Wait, let me verify..."
#   Human: "OK, approved."
#   Agent: "Transfer complete!"
#
# CRITICAL FOR:
#   - Financial transactions (can't undo a wire transfer)
#   - Medical decisions (need doctor approval)
#   - Data deletion (can't undelete)
#   - Email sending (can't unsend)
#   - Compliance (audit trail required)
#
# HOW IT WORKS:
#   1. Agent reaches a checkpoint (e.g., before calling a dangerous tool)
#   2. Checkpointer SAVES the state
#   3. Agent PAUSES and returns control to your code
#   4. Your code shows the pending action to a human
#   5. Human approves/rejects
#   6. Agent RESUMES from the saved state
#
# NOTE: Full human-in-the-loop implementation requires LangGraph's
# interrupt_before/interrupt_after features. We'll cover this in depth
# when we learn LangGraph. For now, understand the CONCEPT and the
# role of checkpointers in enabling it.

def demo_human_in_the_loop_middleware() -> None:
    """
    HumanInTheLoopMiddleware — Pause agent for human approve/edit/reject.

    This is the REAL implementation using HumanInTheLoopMiddleware.
    The agent pauses BEFORE executing dangerous tools (like sending emails)
    and waits for human decision: approve, edit, or reject.
    """
    try:
        from langchain.agents import create_agent
        from langchain.agents.middleware import HumanInTheLoopMiddleware
        from langchain_core.messages import HumanMessage
        from langchain_core.tools import tool
        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph.types import Command
    except ImportError as e:
        logger.warning("Required imports not available: %s", e)
        logger.info("HumanInTheLoopMiddleware requires LangChain v1.x.")
        return

    # ── Define email tools ───────────────────────────────────────────
    # read_email_tool is SAFE — no human approval needed
    # send_email_tool is DANGEROUS — needs human approval before executing
    @tool
    def read_email_tool(email_id: str) -> str:
        """Mock function to read an email by its ID."""
        return f"Email content for ID: {email_id}"

    @tool
    def send_email_tool(recipient: str, subject: str, body: str) -> str:
        """Mock function to send an email."""
        return f"Email sent to {recipient} with subject '{subject}'"

    # ── Create agent with HumanInTheLoopMiddleware ───────────────────
    #
    # interrupt_on defines WHICH tools need human approval:
    #
    #   "send_email_tool": {
    #       "allowed_decisions": ["approve", "edit", "reject"]
    #   }
    #   → Agent PAUSES before send_email_tool
    #   → Human can: approve (send as-is), edit (modify args), reject (cancel)
    #
    #   "read_email_tool": False
    #   → Agent does NOT pause for read_email_tool (it's safe)
    #
    # This is like airport security:
    #   Carry-on bag (read email) → goes through without stopping
    #   Checked luggage (send email) → gets inspected before boarding
    agent = create_agent(
        model=MODEL,
        tools=[read_email_tool, send_email_tool],
        checkpointer=InMemorySaver(),
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "send_email_tool": {
                        "allowed_decisions": ["approve", "edit", "reject"]
                    },
                    "read_email_tool": False,  # No approval needed
                }
            )
        ],
    )

    logger.info("--- HumanInTheLoopMiddleware (Email Example) ---")
    logger.info("  send_email_tool → PAUSES for human approval")
    logger.info("  read_email_tool → runs freely (no pause)")

    # ══════════════════════════════════════════════════════════════════
    # SCENARIO 1: APPROVE — Send the email as-is
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n  --- Scenario 1: APPROVE ---")
    config_approve = {"configurable": {"thread_id": "test-approve"}}

    # Step 1: User requests to send an email
    result = agent.invoke(
        {"messages": [HumanMessage(
            content="Send email to john@test.com with subject 'Hello' and body 'How are you?'"
        )]},
        config=config_approve,
    )

    # Step 2: Check if agent paused (interrupt)
    if "__interrupt__" in result:
        logger.info("  PAUSED! Agent wants to send email. Approving...")

        # Resume with "approve" decision — send as-is
        result = agent.invoke(
            Command(
                resume={
                    "decisions": [
                        {"type": "approve"}
                    ]
                }
            ),
            config=config_approve,
        )
        logger.info("  APPROVED! Result: %s", result["messages"][-1].content[:100])
    else:
        logger.info("  No interrupt — agent completed: %s", result["messages"][-1].content[:100])

    # ══════════════════════════════════════════════════════════════════
    # SCENARIO 2: EDIT — Modify the email before sending
    # ══════════════════════════════════════════════════════════════════
    #
    # The human can CHANGE the tool arguments before the tool executes.
    # This is powerful — the agent proposed an action, but the human
    # corrects it (wrong recipient, typo in subject, etc.)
    logger.info("\n  --- Scenario 2: EDIT ---")
    config_edit = {"configurable": {"thread_id": "test-edit"}}

    result = agent.invoke(
        {"messages": [HumanMessage(
            content="Send email to john@test.com with subject 'Hello' and body 'How are you?'"
        )]},
        config=config_edit,
    )

    if "__interrupt__" in result:
        logger.info("  PAUSED! Editing the email before sending...")

        # Resume with "edit" decision — change the tool arguments
        # edited_action contains the MODIFIED tool call
        result = agent.invoke(
            Command(
                resume={
                    "decisions": [
                        {
                            "type": "edit",
                            "edited_action": {
                                "name": "send_email_tool",       # Tool name
                                "args": {                        # NEW arguments
                                    "recipient": "correct@email.com",
                                    "subject": "Corrected Subject",
                                    "body": "This was edited by human before sending",
                                },
                            },
                        }
                    ]
                }
            ),
            config=config_edit,
        )
        logger.info("  EDITED & SENT! Result: %s", result["messages"][-1].content[:100])
    else:
        logger.info("  No interrupt: %s", result["messages"][-1].content[:100])

    # ══════════════════════════════════════════════════════════════════
    # SCENARIO 3: REJECT — Cancel the email entirely
    # ══════════════════════════════════════════════════════════════════
    #
    # The human decides NOT to send the email. The agent is told
    # the action was rejected and can respond accordingly.
    logger.info("\n  --- Scenario 3: REJECT ---")
    config_reject = {"configurable": {"thread_id": "test-reject"}}

    result = agent.invoke(
        {"messages": [HumanMessage(
            content="Send email to john@test.com with subject 'Hello' and body 'How are you?'"
        )]},
        config=config_reject,
    )

    if "__interrupt__" in result:
        logger.info("  PAUSED! Rejecting the email...")

        result = agent.invoke(
            Command(
                resume={
                    "decisions": [
                        {"type": "reject"}
                    ]
                }
            ),
            config=config_reject,
        )
        logger.info("  REJECTED! Result: %s", result["messages"][-1].content[:100])
    else:
        logger.info("  No interrupt: %s", result["messages"][-1].content[:100])


# ═══════════════════════════════════════════════════════════════════════════════
# 5️⃣  Cost Analysis — How Summarization Saves Money
# ═══════════════════════════════════════════════════════════════════════════════

def demo_cost_analysis() -> None:
    """Show the math behind summarization cost savings."""
    logger.info("--- Cost Analysis ---")
    logger.info("")
    logger.info("  WITHOUT Summarization (50-turn conversation):")
    logger.info("    Turn 1:  2 messages   (~50 tokens)")
    logger.info("    Turn 10: 20 messages  (~500 tokens)")
    logger.info("    Turn 25: 50 messages  (~1,250 tokens)")
    logger.info("    Turn 50: 100 messages (~2,500 tokens)")
    logger.info("    Total tokens sent: ~50,000+ tokens")
    logger.info("")
    logger.info("  WITH Summarization (trigger=10, keep=4):")
    logger.info("    Turn 1:  2 messages   (~50 tokens)")
    logger.info("    Turn 10: Summarize! Now 5 messages (~150 tokens)")
    logger.info("    Turn 25: Multiple summaries, still ~5 messages")
    logger.info("    Turn 50: Still ~5 messages (~150 tokens)")
    logger.info("    Total tokens sent: ~10,000 tokens (80%% savings!)")
    logger.info("")
    logger.info("  Trade-off: Some detail lost in summarization.")
    logger.info("  For most chatbot use cases, this is perfectly fine.")


# ═══════════════════════════════════════════════════════════════════════════════
# 📊 OTHER MIDDLEWARE PATTERNS (Production Use Cases)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Beyond SummarizationMiddleware, production systems use these patterns:
#
# 1. LOGGING MIDDLEWARE
#    Log every message, tool call, and response for debugging.
#    In LangChain, use LangSmith tracing (LANGCHAIN_TRACING_V2=true)
#    or custom callbacks for this.
#
# 2. RATE LIMITING
#    Limit how many LLM calls per minute/hour to control costs.
#    Implement as a custom middleware or use provider-level limits.
#
# 3. PII DETECTION
#    Scan messages for personal info (emails, phone numbers, SSNs)
#    before sending to the LLM. Remove or mask PII.
#    Libraries: presidio, scrubadub
#
# 4. GUARDRAILS
#    Block harmful content, enforce topic boundaries, validate output.
#    Libraries: guardrails-ai, NeMo Guardrails
#
# 5. RETRY / FALLBACK
#    If the LLM call fails (rate limit, timeout), retry with backoff.
#    If primary model fails, fall back to a cheaper model.
#    LangChain v1.x has built-in retry middleware.
#
# 6. TOOL CALL LIMITING
#    Restrict the number of tool calls per turn to prevent infinite loops.
#    Example: max_tool_calls=5 — agent stops after 5 tool calls.
#
# 7. TOKEN BUDGET ENFORCEMENT
#    Set a hard limit on total tokens per conversation.
#    If exceeded, return an error instead of making the LLM call.


# ═══════════════════════════════════════════════════════════════════════════════
# 🏭 Production Pattern — Complete Agent with All Features
# ═══════════════════════════════════════════════════════════════════════════════
#
# This is the COMPLETE pattern for a production-ready agent:
#
#   agent = create_agent(
#       model="groq:llama-3.3-70b-versatile",
#       tools=[tool1, tool2],              # Real-world actions
#       system_prompt="You are...",        # Personality & rules
#       checkpointer=InMemorySaver(),      # Memory & state
#       middleware=[
#           SummarizationMiddleware(       # Cost control
#               model="groq:...",
#               trigger=("messages", 20),
#               keep=("messages", 6),
#           )
#       ],
#       response_format=MySchema,          # Structured output
#   )
#
# This ONE function call gives you:
#   - An agent that can use tools
#   - With a defined personality
#   - That remembers conversations
#   - That auto-summarizes to save tokens
#   - That returns structured data


# ═══════════════════════════════════════════════════════════════════════════════
# 🏭 INTERVIEW QUESTIONS — Middleware
# ═══════════════════════════════════════════════════════════════════════════════
#
# Q1: What is middleware in LangChain?
# A:  Middleware sits between the user and the agent, allowing you to
#     track, transform, and control agent behavior. Examples: logging,
#     summarization, rate limiting, PII detection, guardrails.
#
# Q2: What is SummarizationMiddleware?
# A:  It auto-compresses old conversation messages into a summary when
#     the conversation gets too long. Triggered by message count or
#     token count. Keeps recent messages intact while summarizing older ones.
#
# Q3: Message-based vs Token-based summarization trigger?
# A:  Message-based: trigger=("messages", 10) — counts messages (all equal).
#     Token-based: trigger=("tokens", 500) — counts tokens (varies by length).
#     Token-based is more accurate for cost control.
#
# Q4: What is a checkpointer and why is it needed?
# A:  Saves the agent's state (conversation history) so you can pause,
#     resume, and maintain memory across turns. InMemorySaver for dev,
#     PostgresSaver for production. Uses thread_id to identify conversations.
#
# Q5: What is human-in-the-loop and how does HumanInTheLoopMiddleware work?
# A:  HumanInTheLoopMiddleware pauses the agent BEFORE executing specified
#     tools. The human can: "approve" (execute as-is), "edit" (modify the
#     tool arguments then execute), or "reject" (cancel the action).
#     interrupt_on defines which tools need approval and which don't.
#     Uses Command(resume={"decisions": [{"type": "approve"}]}) to resume.
#
# Q6: What is the "edit" decision in HumanInTheLoopMiddleware?
# A:  The human can MODIFY the tool's arguments before execution. For example,
#     the agent wants to send an email to wrong@email.com, the human edits
#     it to correct@email.com using edited_action with new args. The tool
#     then executes with the corrected arguments.
#
# Q6: How does summarization save costs?
# A:  Without it, a 50-turn conversation sends ~100 messages every turn
#     (~50K tokens total). With summarization, old messages are compressed,
#     keeping ~5 messages per turn (~10K tokens total). ~80% savings.
#
# Q7: What other middleware patterns are used in production?
# A:  Logging (LangSmith), rate limiting, PII detection (presidio),
#     guardrails (NeMo), retry/fallback, tool call limiting, and
#     token budget enforcement.


# ═══════════════════════════════════════════════════════════════════════════════
# 🚀 Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("🛡️ MIDDLEWARE & SUMMARIZATION — Controlling Agent Behavior")
    logger.info("=" * 70)

    logger.info("\n🔹 1. SummarizationMiddleware — Message-Based Trigger")
    demo_message_based_summarization()

    logger.info("\n🔹 2. SummarizationMiddleware — Token-Based Trigger")
    demo_token_based_summarization()

    logger.info("\n🔹 3. Checkpointing — Conversation Memory")
    demo_checkpointing()

    logger.info("\n🔹 4. HumanInTheLoopMiddleware — Approve/Edit/Reject")
    demo_human_in_the_loop_middleware()

    logger.info("\n🔹 5. Cost Analysis")
    demo_cost_analysis()

    logger.info("\n" + "=" * 70)
    logger.info("✅ Middleware & Summarization lesson complete!")
    logger.info("You now understand: SummarizationMiddleware, checkpointing,")
    logger.info("human-in-the-loop, and production middleware patterns.")
    logger.info("=" * 70)
