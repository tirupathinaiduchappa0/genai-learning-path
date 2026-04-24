"""
🆕 Lesson 10.1 — Updated LangChain v1: Introduction & Agent Basics

═══════════════════════════════════════════════════════════════════
WHAT CHANGED IN LANGCHAIN v1.x?
═══════════════════════════════════════════════════════════════════

LangChain v1.x is a MAJOR rewrite. Key changes:

    OLD (v0.2.x)                          NEW (v1.x)
    ─────────────────────────────         ─────────────────────────
    from langchain.chains import ...      REMOVED — use LCEL instead
    LLMChain(llm=llm, prompt=prompt)      chain = prompt | llm | parser
    AgentExecutor                         create_agent() (LangGraph-based)
    create_react_agent (langchain)        create_agent (langchain.agents)
    langchain.chat_models.ChatGroq        langchain_groq.ChatGroq (same)
    langchain.chat_models.init_chat_model NEW — unified model init

WHAT'S NEW:
    1. init_chat_model() — ONE function to init ANY provider's model
    2. create_agent() — Simplified agent creation (uses LangGraph internally)
    3. Cleaner imports — provider-specific packages (langchain_groq, etc.)
    4. Better streaming — chunk.text for cleaner stream output
    5. Batch improvements — parallel processing built-in

WHAT'S THE SAME:
    - LCEL (prompt | llm | parser) — still the core pattern
    - Pydantic models for validation
    - Depends on provider packages (langchain-groq, langchain-openai, etc.)
    - .invoke(), .stream(), .batch() — same Runnable interface

NOTE ON AGENTS:
    The notebook shows create_agent() which is the NEW v1.x way to create
    agents. It uses LangGraph internally. We will learn LangGraph in depth
    in a future lesson. For now, understand the PATTERN:
        1. Define tools (Python functions with docstrings)
        2. Call create_agent(model, tools, system_prompt)
        3. Agent uses ReAct loop: Think → Tool Call → Observe → Answer

HOW TO RUN:
    $ python 01_langchain_v1_introduction_and_agents.py

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


# ═══════════════════════════════════════════════════════════════════════════════
# 1️⃣  LangChain Version Check
# ═══════════════════════════════════════════════════════════════════════════════

def demo_version_check() -> None:
    """Check the installed LangChain version."""
    import langchain
    logger.info("--- LangChain Version ---")
    logger.info("Installed version: %s", langchain.__version__)
    # v1.x = the updated version with create_agent, init_chat_model, etc.


# ═══════════════════════════════════════════════════════════════════════════════
# 2️⃣  Agent Basics — create_agent() (NEW in v1.x)
# ═══════════════════════════════════════════════════════════════════════════════
#
# In the notebook, the instructor uses:
#   from langchain.agents import create_agent
#   agent = create_agent(model="gpt-5", tools=[get_weather], system_prompt="...")
#
# This is the NEW simplified agent API in LangChain v1.x.
# Internally, it creates a LangGraph-based ReAct agent.
#
# HOW IT WORKS:
#   1. You define TOOLS — regular Python functions with docstrings
#      The docstring tells the LLM what the tool does and when to use it
#   2. You call create_agent() with a model, tools, and system prompt
#   3. The agent uses the ReAct loop:
#      THOUGHT: "I need to check the weather. I'll use get_weather."
#      ACTION:  get_weather(city="New York")
#      OBSERVATION: "The weather in New York is sunny."
#      THOUGHT: "I have the answer now."
#      FINAL: "It's sunny in New York right now."
#
# THE MESSAGE FLOW (from the notebook output):
#   [HumanMessage]  → User's question
#   [AIMessage]     → Agent decides to call a tool (tool_calls field)
#   [ToolMessage]   → Tool's response
#   [AIMessage]     → Agent's final answer to the user
#
# NOTE: create_agent() requires an LLM that supports tool-calling
# (OpenAI, Groq, Anthropic). Not all models support this.
#
# We will learn ReAct agents in depth when we study LangGraph.
# For now, here's the pattern using Groq:

def demo_agent_basics() -> None:
    """
    Demonstrate the basic agent pattern from LangChain v1.x.

    NOTE: create_agent() from langchain.agents may require specific
    LangChain v1.x versions. If not available, we use the equivalent
    LangGraph create_react_agent which does the same thing.
    """
    from langchain_groq import ChatGroq

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

    # ── Step 1: Define tools ─────────────────────────────────────────
    # A tool is just a Python function with a docstring.
    # The docstring is CRITICAL — the LLM reads it to decide when to use the tool.
    def get_weather(city: str) -> str:
        """Get the current weather for a city. Use this when the user asks about weather."""
        # In production, this would call a real weather API
        weather_data = {
            "New York": "sunny, 22°C",
            "London": "cloudy, 15°C",
            "Tokyo": "rainy, 18°C",
        }
        return weather_data.get(city, f"Weather data not available for {city}")

    def calculate(expression: str) -> str:
        """Calculate a math expression. Use this when the user asks a math question."""
        try:
            result = eval(expression)  # In production, use a safe math parser
            return str(result)
        except Exception as e:
            return f"Error: {e}"

    # ── Step 2: Create the agent ─────────────────────────────────────
    # Using LangGraph's create_react_agent (equivalent to create_agent in v1.x)
    try:
        from langgraph.prebuilt import create_react_agent

        agent = create_react_agent(
            model=llm,
            tools=[get_weather, calculate],
            prompt="You are a helpful assistant. Use tools when needed.",
        )

        # ── Step 3: Run the agent ────────────────────────────────────
        logger.info("--- Agent Basics ---")

        # Agent with tool call
        response = agent.invoke(
            {"messages": [{"role": "user", "content": "What is the weather in Tokyo?"}]}
        )
        # Extract the final AI message
        final_msg = response["messages"][-1]
        logger.info("Q: What is the weather in Tokyo?")
        logger.info("A: %s", final_msg.content)

        # Show the full message flow
        logger.info("Message flow (%d messages):", len(response["messages"]))
        for msg in response["messages"]:
            logger.info("  [%s]: %s", type(msg).__name__, str(msg.content)[:100])

        # Agent with math
        response2 = agent.invoke(
            {"messages": [{"role": "user", "content": "What is 25 * 47?"}]}
        )
        final_msg2 = response2["messages"][-1]
        logger.info("Q: What is 25 * 47?")
        logger.info("A: %s", final_msg2.content)

    except ImportError:
        logger.warning("langgraph not installed. Install with: uv add langgraph")
        logger.info("Agent creation requires langgraph. Skipping demo.")
        logger.info("We will cover agents in depth in the LangGraph lesson.")


# ═══════════════════════════════════════════════════════════════════════════════
# 📖 UPCOMING TOPICS (will be covered in future lessons)
# ═══════════════════════════════════════════════════════════════════════════════
#
# These topics are part of the Updated LangChain curriculum but will be
# covered one by one in dedicated lessons:
#
# - Messages: HumanMessage, AIMessage, SystemMessage, ToolMessage
#   (partially covered in Lesson 8.1 — Conversational Memory)
#
# - Tools: Defining custom tools, tool-calling protocol, @tool decorator
#   (will be covered with LangGraph agents)
#
# - Middleware: Request/response interceptors for chains
#   (will be covered in advanced LangChain patterns)
#
# - Human-in-the-Loop: Pausing agent execution for human approval
#   (will be covered with LangGraph checkpoints)
#
# - Structured Output: Getting JSON/Pydantic objects from LLMs
#   (will be covered in a dedicated lesson)
#
# - Prompting Patterns: Few-shot, chain-of-thought, self-consistency
#   (will be covered in prompt engineering lesson)


# ═══════════════════════════════════════════════════════════════════════════════
# 🏭 INTERVIEW QUESTIONS — LangChain v1.x
# ═══════════════════════════════════════════════════════════════════════════════
#
# Q1: What changed in LangChain v1.x?
# A:  Major cleanup: LLMChain removed (use LCEL), AgentExecutor replaced
#     by LangGraph-based agents, init_chat_model() for unified model init,
#     cleaner provider-specific imports.
#
# Q2: What is create_agent() in LangChain v1.x?
# A:  A simplified function that creates a ReAct agent using LangGraph
#     internally. Takes a model, tools, and system prompt. Returns a
#     compiled graph that can be invoked with messages.
#
# Q3: What is a Tool in LangChain?
# A:  A Python function with a docstring that an agent can call. The LLM
#     reads the docstring to decide when and how to use the tool. Tools
#     let agents interact with the real world (APIs, databases, etc.).
#
# Q4: What is the message flow in an agent?
# A:  HumanMessage → AIMessage (with tool_calls) → ToolMessage (result)
#     → AIMessage (final answer). The agent loops until it has enough
#     information to answer.


# ═══════════════════════════════════════════════════════════════════════════════
# 🚀 Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("🆕 UPDATED LANGCHAIN v1 — Introduction & Agent Basics")
    logger.info("=" * 70)

    logger.info("\n🔹 1. Version Check")
    demo_version_check()

    logger.info("\n🔹 2. Agent Basics (ReAct pattern)")
    demo_agent_basics()

    logger.info("\n" + "=" * 70)
    logger.info("✅ Updated LangChain v1 — Introduction complete!")
    logger.info("Next: Model Integration (unified chat model interface)")
    logger.info("=" * 70)
