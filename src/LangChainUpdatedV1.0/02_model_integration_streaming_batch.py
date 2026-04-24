"""
🔌 Lesson 10.2 — Model Integration: Unified Chat Interface, Streaming & Batch

═══════════════════════════════════════════════════════════════════
THE BIG IDEA: ONE Interface, ANY Provider
═══════════════════════════════════════════════════════════════════

Before LangChain v1.x, switching between providers was painful:
    # OpenAI
    from langchain_openai import ChatOpenAI
    model = ChatOpenAI(model="gpt-4o")

    # Google
    from langchain_google_genai import ChatGoogleGenerativeAI
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    # Groq
    from langchain_groq import ChatGroq
    model = ChatGroq(model="llama-3.1-8b-instant")

Each provider had different imports, different class names, different params.

NOW with init_chat_model():
    from langchain.chat_models import init_chat_model

    model = init_chat_model("gpt-4o")                    # OpenAI
    model = init_chat_model("google_genai:gemini-2.5-flash")  # Google
    model = init_chat_model("groq:llama-3.1-8b-instant")     # Groq

ONE function. ANY provider. Same interface. Same .invoke(), .stream(), .batch().

WHY THIS MATTERS:
    - Switch providers by changing ONE string (no import changes)
    - A/B test different models easily
    - Fallback: if Groq is down, switch to OpenAI with one line
    - Config-driven: read model name from .env or config file

TOPICS COVERED:
    1. init_chat_model() — Unified model initialization
    2. Provider-specific classes — Direct imports (still valid)
    3. .stream() — Token-by-token streaming
    4. .batch() — Parallel processing of multiple inputs
    5. Response anatomy — Understanding AIMessage fields

HOW TO RUN:
    $ python 02_model_integration_streaming_batch.py

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
# 1️⃣  init_chat_model() — Unified Model Initialization (NEW in v1.x)
# ═══════════════════════════════════════════════════════════════════════════════
#
# FORMAT: init_chat_model("provider:model_name")
#
# Provider prefixes:
#   "gpt-4o"                      → OpenAI (auto-detected from model name)
#   "google_genai:gemini-2.5-flash" → Google Gemini
#   "groq:llama-3.1-8b-instant"    → Groq
#   "anthropic:claude-3-sonnet"     → Anthropic
#
# If no prefix, LangChain tries to auto-detect the provider from the model name.
# Explicit prefix is recommended for clarity.
#
# NOTE: init_chat_model() requires the provider package to be installed:
#   groq → pip install langchain-groq
#   openai → pip install langchain-openai
#   google → pip install langchain-google-genai

def demo_init_chat_model() -> None:
    """Demonstrate the unified init_chat_model() function."""
    from langchain.chat_models import init_chat_model

    # ── Groq (our primary — free API) ────────────────────────────────
    model = init_chat_model("groq:llama-3.1-8b-instant")
    response = model.invoke("What is Python in one sentence?")
    logger.info("--- init_chat_model() ---")
    logger.info("Groq response: %s", response.content[:100])
    logger.info("Model provider: %s", response.response_metadata.get("model_provider", "N/A"))
    logger.info("Model name: %s", response.response_metadata.get("model_name", "N/A"))

    # ── OpenAI (commented — needs OPENAI_API_KEY) ────────────────────
    # model_openai = init_chat_model("gpt-4o")
    # response = model_openai.invoke("What is Python?")

    # ── Google Gemini (commented — needs GOOGLE_API_KEY) ─────────────
    # model_google = init_chat_model("google_genai:gemini-2.5-flash")
    # response = model_google.invoke("What is Python?")


# ═══════════════════════════════════════════════════════════════════════════════
# 2️⃣  Provider-Specific Classes — Direct Imports (Still Valid)
# ═══════════════════════════════════════════════════════════════════════════════
#
# You can still use provider-specific classes directly.
# This gives you more control over provider-specific parameters.
#
# WHEN TO USE WHICH:
#   init_chat_model() → When you want provider-agnostic code
#   Direct import     → When you need provider-specific features

def demo_direct_provider() -> None:
    """Use provider-specific class directly (more control)."""
    from langchain_groq import ChatGroq

    # Direct instantiation — same as we've been doing in all lessons
    model = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.3,
        max_tokens=256,
    )
    response = model.invoke("What is LangChain in one sentence?")
    logger.info("--- Direct Provider (ChatGroq) ---")
    logger.info("Response: %s", response.content[:100])


# ═══════════════════════════════════════════════════════════════════════════════
# 3️⃣  Response Anatomy — Understanding AIMessage Fields
# ═══════════════════════════════════════════════════════════════════════════════
#
# Every LLM call returns an AIMessage with these fields:
#
#   response.content           → The actual text response (string)
#   response.response_metadata → Provider-specific metadata (dict)
#     - token_usage            → Input/output/total tokens used
#     - model_name             → Which model was used
#     - model_provider         → Which provider (openai, groq, etc.)
#     - finish_reason          → Why the model stopped (stop, length, tool_calls)
#   response.usage_metadata    → Standardized token usage (dict)
#     - input_tokens           → Tokens in the prompt
#     - output_tokens          → Tokens in the response
#     - total_tokens           → Total tokens consumed
#   response.id                → Unique run ID
#
# WHY THIS MATTERS:
#   - Token usage → cost tracking and optimization
#   - finish_reason → detect if response was cut off (length limit)
#   - model_provider → verify which provider handled the request

def demo_response_anatomy() -> None:
    """Inspect all fields of an AIMessage response."""
    from langchain_groq import ChatGroq

    model = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3, max_tokens=100)
    response = model.invoke("What is FastAPI in one sentence?")

    logger.info("--- Response Anatomy ---")
    logger.info("content: %s", response.content)
    logger.info("type: %s", type(response).__name__)

    # Token usage — critical for cost tracking
    usage = response.usage_metadata
    logger.info("Tokens — input: %d, output: %d, total: %d",
                usage.get("input_tokens", 0),
                usage.get("output_tokens", 0),
                usage.get("total_tokens", 0))

    # Provider metadata
    meta = response.response_metadata
    logger.info("Model: %s (provider: %s)",
                meta.get("model_name", "N/A"),
                meta.get("model_provider", "N/A"))
    logger.info("Finish reason: %s", meta.get("finish_reason", "N/A"))


# ═══════════════════════════════════════════════════════════════════════════════
# 4️⃣  .stream() — Token-by-Token Streaming
# ═══════════════════════════════════════════════════════════════════════════════

def demo_streaming() -> None:
    """Stream LLM responses token by token."""
    from langchain_groq import ChatGroq

    model = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3, max_tokens=200)

    logger.info("--- Streaming ---")
    full_response = ""
    token_count = 0
    for chunk in model.stream("Write a 50-word paragraph on Artificial Intelligence"):
        full_response += chunk.content
        token_count += 1

    logger.info("Received %d chunks", token_count)
    logger.info("Full response (%d chars): %s", len(full_response), full_response[:200])


# ═══════════════════════════════════════════════════════════════════════════════
# 5️⃣  .batch() — Parallel Processing of Multiple Inputs
# ═══════════════════════════════════════════════════════════════════════════════

def demo_batch() -> None:
    """Process multiple inputs in parallel with .batch()."""
    from langchain_groq import ChatGroq

    model = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3, max_tokens=100)

    questions = [
        "Why do parrots have colorful feathers? Answer in one sentence.",
        "How do airplanes fly? Answer in one sentence.",
        "Why is the sky blue? Answer in one sentence.",
    ]

    # .batch() processes all questions concurrently
    responses = model.batch(questions)

    logger.info("--- Batch Processing ---")
    logger.info("Sent %d questions in one batch call", len(questions))
    for i, (q, r) in enumerate(zip(questions, responses)):
        logger.info("  Q%d: %s", i + 1, q.split("?")[0] + "?")
        logger.info("  A%d: %s", i + 1, r.content[:120])


# ═══════════════════════════════════════════════════════════════════════════════
# 🏭 Production Pattern: Config-Driven Model Selection
# ═══════════════════════════════════════════════════════════════════════════════
#
# In production, you don't hardcode the model. You read it from config:
#
#   # .env
#   LLM_MODEL=groq:llama-3.1-8b-instant
#
#   # settings.py
#   from pydantic_settings import BaseSettings
#   class Settings(BaseSettings):
#       llm_model: str = "groq:llama-3.1-8b-instant"
#       model_config = {"env_file": ".env"}
#
#   # main.py
#   from langchain.chat_models import init_chat_model
#   settings = Settings()
#   model = init_chat_model(settings.llm_model)
#
# Now you can switch models by changing ONE env variable. No code changes.
# This is how production GenAI apps handle model selection.


# ═══════════════════════════════════════════════════════════════════════════════
# 🏭 INTERVIEW QUESTIONS — Model Integration
# ═══════════════════════════════════════════════════════════════════════════════
#
# Q1: What is init_chat_model() in LangChain v1.x?
# A:  A unified function that initializes any LLM provider's chat model
#     using a single string format "provider:model_name". It auto-detects
#     the provider and returns a standard ChatModel with invoke/stream/batch.
#
# Q2: What is the difference between .invoke(), .stream(), and .batch()?
# A:  invoke() = single input, single output (synchronous).
#     stream() = single input, output arrives token-by-token (iterator).
#     batch() = multiple inputs, processed in parallel, list of outputs.
#
# Q3: How do you track token usage in LangChain?
# A:  Every AIMessage has usage_metadata with input_tokens, output_tokens,
#     and total_tokens. Use this for cost tracking and optimization.
#
# Q4: How would you make model selection configurable in production?
# A:  Use Pydantic BaseSettings to read the model name from an env variable.
#     Pass it to init_chat_model(). Switching models = changing one env var.
#
# Q5: What is the Runnable interface?
# A:  The standard interface ALL LangChain components implement:
#     invoke(), stream(), batch(), ainvoke(), astream(), abatch().
#     This is what makes the pipe (|) operator work — every component
#     speaks the same language.


# ═══════════════════════════════════════════════════════════════════════════════
# 🚀 Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("🔌 MODEL INTEGRATION — Unified Interface, Streaming & Batch")
    logger.info("=" * 70)

    logger.info("\n🔹 1. init_chat_model() — Unified model init")
    demo_init_chat_model()

    logger.info("\n🔹 2. Direct Provider (ChatGroq)")
    demo_direct_provider()

    logger.info("\n🔹 3. Response Anatomy")
    demo_response_anatomy()

    logger.info("\n🔹 4. Streaming")
    demo_streaming()

    logger.info("\n🔹 5. Batch Processing")
    demo_batch()

    logger.info("\n" + "=" * 70)
    logger.info("✅ Model Integration lesson complete!")
    logger.info("Upcoming: Messages, Tools, Structured Output, Human-in-the-Loop")
    logger.info("=" * 70)
