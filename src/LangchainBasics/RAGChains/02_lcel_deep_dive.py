"""
⛓️ Lesson 6.2 — LCEL Deep Dive: Runnables, Parallel, Lambda, Stream, Batch

LCEL = LangChain Expression Language

In Lesson 6.1, we learned the basics: prompt | llm | parser
Now we go DEEPER into the building blocks that make LCEL powerful.

WHY DOES THIS MATTER?
    In production, you need more than a simple chain. You need to:
    - Run multiple things in parallel (e.g., retrieve + pass question)
    - Wrap custom Python functions into chains
    - Stream responses token-by-token for real-time UIs
    - Process batches of inputs efficiently
    - Compose complex multi-step workflows

TOPICS COVERED:
    1. Runnable Protocol   — The interface ALL LCEL components share
    2. RunnablePassthrough — Pass input through unchanged
    3. RunnableLambda      — Wrap ANY Python function as a Runnable
    4. RunnableParallel    — Run multiple branches simultaneously
    5. .stream()           — Token-by-token streaming
    6. .batch()            — Process multiple inputs at once
    7. itemgetter pattern  — Clean way to extract dict keys
    8. Direct LLM calls    — Using HumanMessage/SystemMessage directly

Reference:
    https://python.langchain.com/docs/concepts/lcel/

Author: GenAI Learner
Date: 2026-04-12
"""

import logging
import os
from operator import itemgetter

from dotenv import load_dotenv

# ─── Environment & Logging Setup ────────────────────────────────────────────
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _get_llm():
    """Get the Groq LLM instance (reused across demos)."""
    from langchain_groq import ChatGroq
    return ChatGroq(model="llama-3.1-8b-instant", temperature=0.3, max_tokens=256)


# ═══════════════════════════════════════════════════════════════════════════════
# 📖 CORE CONCEPT: The Runnable Protocol
# ═══════════════════════════════════════════════════════════════════════════════
#
# Every LCEL component (prompt, llm, parser, retriever, your custom functions)
# implements the SAME interface called "Runnable". This is what makes the
# pipe (|) operator work — every piece speaks the same language.
#
# The Runnable interface has these key methods:
#
# ┌──────────────────┬────────────────────────────────────────────────────────┐
# │ Method           │ What it does                                           │
# ├──────────────────┼────────────────────────────────────────────────────────┤
# │ .invoke(input)   │ Run with a SINGLE input, get a SINGLE output          │
# │ .batch(inputs)   │ Run with a LIST of inputs, get a LIST of outputs      │
# │ .stream(input)   │ Run with a SINGLE input, get output TOKEN BY TOKEN    │
# │ .ainvoke(input)  │ Async version of invoke (for FastAPI, async apps)     │
# │ .abatch(inputs)  │ Async version of batch                                │
# │ .astream(input)  │ Async version of stream                               │
# └──────────────────┴────────────────────────────────────────────────────────┘
#
# Because EVERY component has these methods, you can:
#   chain.invoke(...)   → runs the whole chain synchronously
#   chain.stream(...)   → streams the whole chain token by token
#   chain.batch([...])  → runs the chain on multiple inputs in parallel


# ═══════════════════════════════════════════════════════════════════════════════
# 1️⃣  Direct LLM Calls with HumanMessage / SystemMessage
# ═══════════════════════════════════════════════════════════════════════════════
#
# Before using prompt templates, you can call the LLM directly
# by passing a list of Message objects. This is the lowest-level way.

def demo_direct_llm_call() -> None:
    """Call the LLM directly with Message objects (no prompt template)."""
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.output_parsers import StrOutputParser

    llm = _get_llm()

    # ── Direct invoke with messages ──────────────────────────────────────
    # You pass a list of Message objects directly to the LLM.
    # SystemMessage = instructions for the LLM's behavior
    # HumanMessage  = the user's actual input
    messages = [
        SystemMessage(content="Translate the following from English to French"),
        HumanMessage(content="Hello, how are you?"),
    ]

    # model.invoke() returns an AIMessage object
    result = llm.invoke(messages)
    logger.info("--- Direct LLM call ---")
    logger.info("Raw AIMessage: %s", result.content)
    logger.info("Token usage: %s", result.usage_metadata)

    # ── Chain: model | parser (simplest possible LCEL chain) ─────────────
    # This is the minimal chain — just LLM + parser, no prompt template.
    parser = StrOutputParser()
    chain = llm | parser
    answer = chain.invoke(messages)
    logger.info("model | parser result: %s", answer)


# ═══════════════════════════════════════════════════════════════════════════════
# 2️⃣  RunnablePassthrough — Pass Input Through Unchanged
# ═══════════════════════════════════════════════════════════════════════════════
#
# RunnablePassthrough does ONE thing: takes the input and passes it through
# without any modification. Sounds useless? It's actually critical.
#
# WHY? In RunnableParallel (dict syntax), you need to "route" the same input
# to multiple branches. One branch might transform it (e.g., retriever),
# while another just passes it through (the original question).
#
# Example from our RAG chain:
#   {"context": retriever | format_docs, "input": RunnablePassthrough()}
#   ↑ "context" branch transforms the input (retrieves docs)
#   ↑ "input" branch passes the question through unchanged

def demo_runnable_passthrough() -> None:
    """Show how RunnablePassthrough works in isolation and in chains."""
    from langchain_core.runnables import RunnablePassthrough

    # ── Standalone usage ─────────────────────────────────────────────────
    passthrough = RunnablePassthrough()
    result = passthrough.invoke("Hello, I am unchanged!")
    logger.info("--- RunnablePassthrough ---")
    logger.info("Input:  'Hello, I am unchanged!'")
    logger.info("Output: '%s'", result)
    # Output is identical to input — that's the whole point.

    # ── In a parallel dict (the real use case) ───────────────────────────
    from langchain_core.runnables import RunnableParallel

    def to_upper(text: str) -> str:
        return text.upper()

    # This runs TWO branches with the SAME input:
    #   "original" → passes through unchanged
    #   "uppercased" → transforms to uppercase
    parallel = RunnableParallel(
        original=RunnablePassthrough(),
        uppercased=to_upper,
    )
    result = parallel.invoke("hello world")
    logger.info("Parallel with passthrough: %s", result)
    # Output: {"original": "hello world", "uppercased": "HELLO WORLD"}


# ═══════════════════════════════════════════════════════════════════════════════
# 3️⃣  RunnableLambda — Wrap ANY Python Function as a Runnable
# ═══════════════════════════════════════════════════════════════════════════════
#
# RunnableLambda wraps a regular Python function so it can be used
# in LCEL chains with the pipe (|) operator.
#
# WHEN TO USE:
#   - Custom data transformation between chain steps
#   - Logging/debugging intermediate values
#   - Any custom logic that doesn't fit existing LangChain components
#
# NOTE: In many cases, you can pass a plain function directly to
# RunnableParallel and it auto-wraps it. But RunnableLambda is
# explicit and works everywhere.

def demo_runnable_lambda() -> None:
    """Wrap custom Python functions into LCEL chains."""
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableLambda

    llm = _get_llm()

    # ── Custom function wrapped as Runnable ──────────────────────────────
    def word_count(text: str) -> str:
        """Count words in the LLM's response and append the count."""
        count = len(text.split())
        return f"{text}\n\n[Word count: {count}]"

    # Wrap it so it can be piped
    word_counter = RunnableLambda(word_count)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer in exactly 2 sentences."),
        ("human", "{input}"),
    ])

    # Chain: prompt → llm → parser → word_counter
    # The word_counter runs AFTER the parser, on the plain string output
    chain = prompt | llm | StrOutputParser() | word_counter

    result = chain.invoke({"input": "What is Python?"})
    logger.info("--- RunnableLambda ---")
    logger.info("Result with word count:\n%s", result)


# ═══════════════════════════════════════════════════════════════════════════════
# 4️⃣  RunnableParallel — Run Multiple Branches Simultaneously
# ═══════════════════════════════════════════════════════════════════════════════
#
# RunnableParallel runs multiple Runnables with the SAME input, in parallel.
# The output is a dict where each key maps to its branch's output.
#
# TWO WAYS TO CREATE IT:
#   1. Dict syntax (shorthand): {"key1": runnable1, "key2": runnable2}
#   2. Explicit: RunnableParallel(key1=runnable1, key2=runnable2)
#
# THIS IS THE PATTERN USED IN RAG:
#   {"context": retriever | format_docs, "input": RunnablePassthrough()}
#   ↑ This is a RunnableParallel that feeds into the prompt template.

def demo_runnable_parallel() -> None:
    """Run multiple chains in parallel with the same input."""
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableParallel, RunnablePassthrough

    llm = _get_llm()
    parser = StrOutputParser()

    # ── Two different prompts, same input, parallel execution ────────────
    joke_prompt = ChatPromptTemplate.from_messages([
        ("human", "Tell me a one-line joke about {topic}"),
    ])
    fact_prompt = ChatPromptTemplate.from_messages([
        ("human", "Tell me one interesting fact about {topic}"),
    ])

    joke_chain = joke_prompt | llm | parser
    fact_chain = fact_prompt | llm | parser

    # RunnableParallel runs both chains with the same {"topic": "Python"}
    parallel = RunnableParallel(
        joke=joke_chain,
        fact=fact_chain,
    )

    result = parallel.invoke({"topic": "Python programming"})
    logger.info("--- RunnableParallel ---")
    logger.info("Joke: %s", result["joke"])
    logger.info("Fact: %s", result["fact"])
    # Both ran in parallel — faster than running sequentially!


# ═══════════════════════════════════════════════════════════════════════════════
# 5️⃣  .stream() — Token-by-Token Streaming
# ═══════════════════════════════════════════════════════════════════════════════
#
# .stream() returns an iterator that yields output CHUNKS as they arrive.
# For LLMs, each chunk is typically one token (word/subword).
#
# WHY STREAMING MATTERS:
#   Without streaming: User waits 5 seconds, then sees the full answer.
#   With streaming: User sees words appearing one by one in real-time.
#   This is how ChatGPT, Claude, and all modern chat UIs work.
#
# PRODUCTION TIP:
#   Always use .stream() in user-facing apps (Streamlit, FastAPI).
#   Use .invoke() for background processing where latency doesn't matter.

def demo_streaming() -> None:
    """Stream LLM responses token by token."""
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    llm = _get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("human", "Explain what LCEL is in 3 sentences."),
    ])
    chain = prompt | llm | StrOutputParser()

    logger.info("--- .stream() — Token-by-token ---")
    full_response = ""
    for chunk in chain.stream({}):
        # Each chunk is a small piece of text (usually 1-3 tokens)
        full_response += chunk

    logger.info("Streamed response (%d chars): %s", len(full_response), full_response[:200])
    # In a real app, you'd print each chunk immediately:
    # for chunk in chain.stream({}):
    #     print(chunk, end="", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 6️⃣  .batch() — Process Multiple Inputs at Once
# ═══════════════════════════════════════════════════════════════════════════════
#
# .batch() takes a LIST of inputs and processes them all.
# Under the hood, it runs them concurrently for better throughput.
#
# WHEN TO USE:
#   - Processing a list of questions against a RAG pipeline
#   - Generating summaries for multiple documents
#   - Any time you have multiple independent inputs

def demo_batch() -> None:
    """Process multiple inputs in a single batch call."""
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    llm = _get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("human", "What is the capital of {country}? Answer in one word."),
    ])
    chain = prompt | llm | StrOutputParser()

    # .batch() takes a LIST of input dicts
    inputs = [
        {"country": "France"},
        {"country": "Japan"},
        {"country": "Brazil"},
    ]
    results = chain.batch(inputs)

    logger.info("--- .batch() — Multiple inputs ---")
    for inp, result in zip(inputs, results):
        logger.info("  %s → %s", inp["country"], result.strip())


# ═══════════════════════════════════════════════════════════════════════════════
# 7️⃣  itemgetter Pattern — Clean Dict Key Extraction
# ═══════════════════════════════════════════════════════════════════════════════
#
# When your chain input is a dict, you often need to extract specific keys
# and route them to different branches. Python's operator.itemgetter is
# a clean way to do this.
#
# itemgetter("key") creates a function that extracts "key" from a dict.
# It's an alternative to RunnablePassthrough for dict inputs.

def demo_itemgetter() -> None:
    """Use itemgetter to extract dict keys in LCEL chains."""
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    llm = _get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a translator. Translate the text to {language}."),
        ("human", "{text}"),
    ])

    # itemgetter extracts keys from the input dict and routes them
    # This is equivalent to: prompt receives {"language": ..., "text": ...}
    chain = (
        {"language": itemgetter("language"), "text": itemgetter("text")}
        | prompt
        | llm
        | StrOutputParser()
    )

    result = chain.invoke({"language": "Spanish", "text": "Good morning!"})
    logger.info("--- itemgetter pattern ---")
    logger.info("Translated to Spanish: %s", result)


# ═══════════════════════════════════════════════════════════════════════════════
# 🚀 Main — Run all demos
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("⛓️  LCEL DEEP DIVE — Runnables, Parallel, Lambda, Stream, Batch")
    logger.info("=" * 70)

    logger.info("\n🔹 1. Direct LLM call with HumanMessage/SystemMessage")
    demo_direct_llm_call()

    logger.info("\n🔹 2. RunnablePassthrough")
    demo_runnable_passthrough()

    logger.info("\n🔹 3. RunnableLambda — custom functions in chains")
    demo_runnable_lambda()

    logger.info("\n🔹 4. RunnableParallel — multiple branches")
    demo_runnable_parallel()

    logger.info("\n🔹 5. .stream() — token-by-token streaming")
    demo_streaming()

    logger.info("\n🔹 6. .batch() — multiple inputs at once")
    demo_batch()

    logger.info("\n🔹 7. itemgetter — clean dict key extraction")
    demo_itemgetter()

    logger.info("\n" + "=" * 70)
    logger.info("✅ LCEL Deep Dive complete!")
    logger.info("You now know: invoke, stream, batch, Passthrough, Lambda, Parallel")
    logger.info("=" * 70)
