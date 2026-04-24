"""
💬 Lesson 10.4 — Messages Deep Dive: The Language LLMs Speak

═══════════════════════════════════════════════════════════════════
WHAT ARE MESSAGES?
═══════════════════════════════════════════════════════════════════

Messages are the FUNDAMENTAL UNIT of communication with LLMs.
Every interaction — every question, every answer, every tool call —
is represented as a Message object.

Think of it like a chat app (WhatsApp, Slack):
    - Each bubble in the chat is a "message"
    - Each message has a SENDER (role) and CONTENT (text)
    - The full conversation is a LIST of messages in order

In LangChain, messages have 3 parts:
    ROLE     — Who sent it (system, human, ai, tool)
    CONTENT  — The actual text/data
    METADATA — Token usage, model info, tool call IDs, etc.

═══════════════════════════════════════════════════════════════════
THE 4 MESSAGE TYPES
═══════════════════════════════════════════════════════════════════

    Type            | Role    | Who Creates It | Purpose
    ────────────────|─────────|────────────────|──────────────────
    SystemMessage   | system  | YOU (developer)| Set LLM behavior/rules
    HumanMessage    | human   | The USER       | User's question/input
    AIMessage       | ai      | The LLM        | LLM's response
    ToolMessage     | tool    | YOUR CODE      | Tool execution result

    The conversation flows:
    [SystemMessage] → [HumanMessage] → [AIMessage] → ...

    With tools:
    [SystemMessage] → [HumanMessage] → [AIMessage(tool_calls)]
    → [ToolMessage] → [AIMessage(final answer)]

═══════════════════════════════════════════════════════════════════
WHY UNDERSTANDING MESSAGES MATTERS
═══════════════════════════════════════════════════════════════════

    - Conversation memory = managing a list of messages
    - Tool calling = reading AIMessage.tool_calls + creating ToolMessages
    - Prompt engineering = crafting the right SystemMessage
    - Debugging = inspecting the message flow to find issues
    - Cost optimization = counting tokens in messages

TOPICS COVERED:
    1. SystemMessage  — Setting the LLM's personality and rules
    2. HumanMessage   — User input (text, images, files)
    3. AIMessage      — LLM responses (text, tool calls, metadata)
    4. ToolMessage    — Tool execution results (the tricky one!)
    5. Text Prompts vs Message Prompts — When to use which
    6. Complete Message Flow — Real-world agent conversation trace


    
Lesson 10.4 — Messages Deep Dive (04_messages_deep_dive.py) — all 5 demos ran successfully:

SystemMessage — showed 3 different system messages producing completely different responses (generic, expert bullet points, poem format) for the same question.

HumanMessage — text input + multi-turn conversation where LLM remembered "Your name is Krish."

AIMessage Anatomy — inspected all fields: content, tool_calls (empty), tokens (input: 43, output: 32, total: 75), model name, provider, finish_reason.

ToolMessage (the tricky one) — the LLM requested BOTH tools simultaneously (get_weather + calculate). Showed every ToolMessage field: content ("Light rain, 19°C" / "345"), name, tool_call_id. Then demonstrated the UI Display Pattern with emojis: 👤 User → 🤖 Agent wants to use → 🔧 Tool returned → 🤖 Agent final answer.

Text vs Message Prompts — when to use which (text for quick one-offs, messages for production).



HOW TO RUN:
    $ python 04_messages_deep_dive.py

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


def _get_llm():
    from langchain_groq import ChatGroq
    return ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_tokens=256)


# ═══════════════════════════════════════════════════════════════════════════════
# 1️⃣  SystemMessage — Setting the LLM's Personality and Rules
# ═══════════════════════════════════════════════════════════════════════════════
#
# SystemMessage is the FIRST message in a conversation. It tells the LLM:
#   - WHO it is ("You are a Python expert")
#   - HOW to behave ("Be concise, answer in bullet points")
#   - WHAT rules to follow ("Only answer questions about Python")
#   - WHAT format to use ("Respond in JSON format")
#
# The LLM follows these instructions for ALL subsequent messages.
# Think of it as the LLM's "job description" for this conversation.
#
# PRODUCTION TIP: Keep system messages in separate files (prompts/system.txt)
# so they can be versioned, tested, and updated without code changes.

def demo_system_message() -> None:
    """Show how SystemMessage controls LLM behavior."""
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = _get_llm()

    # ── Without system message — generic response ────────────────────
    response_generic = llm.invoke([HumanMessage(content="What is Python?")])
    logger.info("--- SystemMessage ---")
    logger.info("Without system msg: %s", response_generic.content[:100])

    # ── With system message — controlled response ────────────────────
    response_expert = llm.invoke([
        SystemMessage(content="You are a senior Python developer. Answer in exactly 2 bullet points. Be technical."),
        HumanMessage(content="What is Python?"),
    ])
    logger.info("With system msg (expert): %s", response_expert.content[:150])

    # ── System message as a poet ─────────────────────────────────────
    response_poet = llm.invoke([
        SystemMessage(content="You are a poetry expert. Answer everything in the form of a short poem."),
        HumanMessage(content="What is Python?"),
    ])
    logger.info("With system msg (poet): %s", response_poet.content[:150])


# ═══════════════════════════════════════════════════════════════════════════════
# 2️⃣  HumanMessage — User Input
# ═══════════════════════════════════════════════════════════════════════════════
#
# HumanMessage represents what the USER says. It can contain:
#   - Text (most common)
#   - Images (for multimodal models like GPT-4V)
#   - Audio, files, documents (model-dependent)
#
# In a chat app, this is the user's chat bubble.
# In an API, this is the request body.

def demo_human_message() -> None:
    """Show HumanMessage usage — text input from the user."""
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = _get_llm()

    # ── Simple text input ────────────────────────────────────────────
    response = llm.invoke([
        SystemMessage(content="You are a helpful assistant. Be concise."),
        HumanMessage(content="Explain REST API in one sentence."),
    ])
    logger.info("--- HumanMessage ---")
    logger.info("Text input response: %s", response.content[:150])

    # ── Multiple turns (conversation) ────────────────────────────────
    # Each HumanMessage is a new user input in the conversation
    from langchain_core.messages import AIMessage
    conversation = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="My name is Krish."),
        AIMessage(content="Hello Krish! How can I help you?"),
        HumanMessage(content="What is my name?"),
    ]
    response = llm.invoke(conversation)
    logger.info("Multi-turn (remembers name): %s", response.content[:100])


# ═══════════════════════════════════════════════════════════════════════════════
# 3️⃣  AIMessage — LLM Responses (Text + Tool Calls + Metadata)
# ═══════════════════════════════════════════════════════════════════════════════
#
# AIMessage is what the LLM returns. It has rich fields:
#
#   .content           → The text response (string)
#   .tool_calls        → List of tool call requests (if any)
#   .response_metadata → Provider-specific info (tokens, model, finish_reason)
#   .usage_metadata    → Standardized token counts
#   .id                → Unique run ID
#
# TWO MODES:
#   1. TEXT MODE:  content has text, tool_calls is empty
#   2. TOOL MODE:  content is empty, tool_calls has requests
#   It's EITHER text OR tool calls — never both.

def demo_ai_message() -> None:
    """Inspect all fields of an AIMessage response."""
    from langchain_core.messages import HumanMessage

    llm = _get_llm()

    response = llm.invoke([HumanMessage(content="What is FastAPI in one sentence?")])

    logger.info("--- AIMessage Anatomy ---")
    logger.info("type: %s", type(response).__name__)
    logger.info("content: %s", response.content[:120])
    logger.info("tool_calls: %s", response.tool_calls)  # Empty — no tools bound

    # Token usage
    usage = response.usage_metadata
    logger.info("Tokens — input: %d, output: %d, total: %d",
                usage.get("input_tokens", 0), usage.get("output_tokens", 0), usage.get("total_tokens", 0))

    # Provider metadata
    meta = response.response_metadata
    logger.info("Model: %s, Provider: %s, Finish: %s",
                meta.get("model_name", "?"), meta.get("model_provider", "?"), meta.get("finish_reason", "?"))


# ═══════════════════════════════════════════════════════════════════════════════
# 4️⃣  ToolMessage — Tool Execution Results (The Tricky One!)
# ═══════════════════════════════════════════════════════════════════════════════
#
# ToolMessage is the RESULT of executing a tool. It's created by YOUR CODE
# (not by the LLM) after you run a tool that the LLM requested.
#
# HOW ToolMessage IS GENERATED (step by step):
#
#   1. LLM receives a question: "What's the weather in Tokyo?"
#   2. LLM responds with AIMessage containing tool_calls:
#      tool_calls=[{"name": "get_weather", "args": {"city": "Tokyo"}, "id": "call_abc"}]
#   3. YOUR CODE executes the tool: get_weather.invoke(tool_call)
#   4. This returns a ToolMessage:
#      ToolMessage(content="Light rain, 19°C", name="get_weather", tool_call_id="call_abc")
#   5. You append this ToolMessage to the conversation
#   6. LLM reads it and generates the final answer
#
# ToolMessage FIELDS:
#   content       → The tool's output (string) — what the tool returned
#   name          → Which tool produced this result (e.g., "get_weather")
#   tool_call_id  → Links this result to the specific tool_call request
#                   (critical when multiple tools are called in one turn)
#
# WHERE ToolMessage IS USED:
#   - In the tool execution loop (Lesson 10.3)
#   - In agent frameworks (LangGraph, CrewAI) — they create ToolMessages internally
#   - In conversation history — ToolMessages are part of the message list
#   - In debugging — inspecting ToolMessages shows what tools returned
#
# HOW TO DISPLAY TOOL RESPONSES (for UIs):
#   When building a chat UI, you can show tool usage to the user:
#   "🔧 Used tool: get_weather(city='Tokyo') → Light rain, 19°C"
#   This builds trust — the user sees HOW the agent got its answer.

def demo_tool_message() -> None:
    """Show how ToolMessages are generated, structured, and used."""
    from langchain.tools import tool
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from langchain_groq import ChatGroq

    @tool
    def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        data = {"Tokyo": "Light rain, 19°C", "London": "Overcast, 14°C"}
        return data.get(city, f"No data for {city}")

    @tool
    def calculate(expression: str) -> str:
        """Evaluate a math expression."""
        try:
            return str(eval(expression))
        except Exception as e:
            return f"Error: {e}"

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_tokens=256)
    tools = [get_weather, calculate]
    tool_map = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)

    logger.info("--- ToolMessage Deep Dive ---")

    # ── Full conversation with tool messages ─────────────────────────
    messages = [HumanMessage(content="What is the weather in Tokyo and what is 15 * 23?")]
    ai_response = llm_with_tools.invoke(messages)
    messages.append(ai_response)

    logger.info("LLM requested %d tool call(s):", len(ai_response.tool_calls))

    # Execute each tool and create ToolMessages
    for tc in ai_response.tool_calls:
        logger.info("  Executing: %s(%s) [id=%s]", tc["name"], tc["args"], tc["id"])
        tool_result = tool_map[tc["name"]].invoke(tc)

        # ── Inspect the ToolMessage ──────────────────────────────────
        logger.info("  ToolMessage created:")
        logger.info("    content: '%s'", tool_result.content)
        logger.info("    name: '%s'", tool_result.name)
        logger.info("    tool_call_id: '%s'", tool_result.tool_call_id)
        logger.info("    type: '%s'", type(tool_result).__name__)

        messages.append(tool_result)

    # Get final answer
    final = llm_with_tools.invoke(messages)
    messages.append(final)
    logger.info("Final answer: %s", final.content[:200])

    # ── Display tool usage for UI (production pattern) ───────────────
    logger.info("--- UI Display Pattern ---")
    for msg in messages:
        if isinstance(msg, HumanMessage):
            logger.info("👤 User: %s", msg.content)
        elif isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                logger.info("🤖 Agent wants to use: %s(%s)", tc["name"], tc["args"])
        elif isinstance(msg, ToolMessage):
            logger.info("🔧 Tool [%s] returned: %s", msg.name, msg.content)
        elif isinstance(msg, AIMessage):
            logger.info("🤖 Agent: %s", msg.content[:150])


# ═══════════════════════════════════════════════════════════════════════════════
# 5️⃣  Text Prompts vs Message Prompts — When to Use Which
# ═══════════════════════════════════════════════════════════════════════════════

def demo_text_vs_message_prompts() -> None:
    """Compare text prompts (simple) vs message prompts (structured)."""
    llm = _get_llm()

    # ── Text prompt — just a string ──────────────────────────────────
    # Use when: single standalone question, no conversation history
    text_response = llm.invoke("What is LangChain in one sentence?")
    logger.info("--- Text vs Message Prompts ---")
    logger.info("Text prompt: %s", text_response.content[:100])

    # ── Message prompt — structured list ─────────────────────────────
    # Use when: you need system instructions, conversation history, or tools
    from langchain_core.messages import HumanMessage, SystemMessage
    msg_response = llm.invoke([
        SystemMessage(content="Answer in exactly 10 words."),
        HumanMessage(content="What is LangChain?"),
    ])
    logger.info("Message prompt (10 words): %s", msg_response.content[:100])

    # RULE OF THUMB:
    # Text prompts  → quick one-off questions, prototyping
    # Message prompts → production apps, conversations, agents, tools


# ═══════════════════════════════════════════════════════════════════════════════
# 🏭 INTERVIEW QUESTIONS — Messages
# ═══════════════════════════════════════════════════════════════════════════════
#
# Q1: What are the 4 message types in LangChain?
# A:  SystemMessage (sets LLM behavior), HumanMessage (user input),
#     AIMessage (LLM response — text or tool calls), ToolMessage
#     (result of executing a tool, created by your code).
#
# Q2: What is a ToolMessage and when is it created?
# A:  ToolMessage is created by YOUR CODE after executing a tool that
#     the LLM requested. It contains the tool's output (content),
#     the tool's name, and a tool_call_id that links it back to the
#     specific tool_call in the AIMessage. The LLM reads it to
#     formulate its final answer.
#
# Q3: What is the difference between AIMessage.content and AIMessage.tool_calls?
# A:  They're mutually exclusive. If the LLM can answer directly,
#     content has text and tool_calls is empty. If the LLM needs a tool,
#     content is empty and tool_calls has the request. Never both.
#
# Q4: Why is SystemMessage important in production?
# A:  It controls the LLM's behavior for the entire conversation:
#     role, tone, rules, output format, safety guardrails. Without it,
#     the LLM uses its default behavior which may not match your needs.
#
# Q5: How would you display tool usage in a chat UI?
# A:  Parse the message list: for AIMessages with tool_calls, show
#     "Agent is using [tool_name]...". For ToolMessages, show
#     "Tool [name] returned: [content]". This builds user trust by
#     showing HOW the agent arrived at its answer.


# ═══════════════════════════════════════════════════════════════════════════════
# 🚀 Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("💬 MESSAGES DEEP DIVE — The Language LLMs Speak")
    logger.info("=" * 70)

    logger.info("\n🔹 1. SystemMessage — Setting LLM behavior")
    demo_system_message()

    logger.info("\n🔹 2. HumanMessage — User input")
    demo_human_message()

    logger.info("\n🔹 3. AIMessage — LLM responses")
    demo_ai_message()

    logger.info("\n🔹 4. ToolMessage — Tool execution results (the tricky one)")
    demo_tool_message()

    logger.info("\n🔹 5. Text Prompts vs Message Prompts")
    demo_text_vs_message_prompts()

    logger.info("\n" + "=" * 70)
    logger.info("✅ Messages Deep Dive complete!")
    logger.info("Upcoming: Structured Output, Human-in-the-Loop, Middleware")
    logger.info("=" * 70)
