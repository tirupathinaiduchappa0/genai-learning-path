"""
🧠 Lesson 8.1 — Conversational Memory: How Chatbots Remember (Theory + Demos)

═══════════════════════════════════════════════════════════════════
THE CORE PROBLEM: LLMs Have NO Memory
═══════════════════════════════════════════════════════════════════

Every time you call an LLM, it starts FRESH. It has ZERO memory of
previous conversations. Each call is completely independent.

    Call 1: "Hi, my name is Krish"  → "Nice to meet you, Krish!"
    Call 2: "What's my name?"       → "I don't know your name."  ← PROBLEM!

The LLM forgot! It doesn't "remember" Call 1 when processing Call 2.

THE SOLUTION: We manually pass the conversation history WITH every call.

    Call 2 (with history):
        messages = [
            HumanMessage("Hi, my name is Krish"),           ← Call 1 input
            AIMessage("Nice to meet you, Krish!"),          ← Call 1 output
            HumanMessage("What's my name?"),                ← Call 2 input
        ]
        → "Your name is Krish!"  ← Now it works!

The LLM reads ALL previous messages and uses them as context.
This is NOT real memory — it's just passing the full conversation
as input every time. The LLM is "stateless", WE manage the state.

═══════════════════════════════════════════════════════════════════
CONCEPTS COVERED (Basic → Advanced):
═══════════════════════════════════════════════════════════════════

    1. The Memory Problem     — Why LLMs forget (stateless nature)
    2. Manual History         — Passing messages list yourself
    3. ChatMessageHistory     — LangChain's history storage class
    4. RunnableWithMessageHistory — Auto-manages history for chains
    5. MessagesPlaceholder    — Dynamic slot in prompt templates
    6. Session-based Memory   — Different conversations via session_id
    7. trim_messages          — Token-based memory limiting
    8. Streaming with Memory  — Real-time responses with history

HOW TO RUN:
    $ python 01_chat_memory_concepts.py

Author: GenAI Learner
Date: 2026-04-13
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
    return ChatGroq(model="llama-3.1-8b-instant", temperature=0.3, max_tokens=256)


# ═══════════════════════════════════════════════════════════════════════════════
# 1️⃣  The Problem: LLMs Are Stateless
# ═══════════════════════════════════════════════════════════════════════════════

def demo_stateless_problem() -> None:
    """Show that LLMs have NO memory between calls."""
    from langchain_core.messages import HumanMessage

    llm = _get_llm()

    # Call 1: Tell the LLM your name
    response1 = llm.invoke([HumanMessage(content="Hi, my name is Krish and I am a Chief AI Engineer")])
    logger.info("--- The Stateless Problem ---")
    logger.info("Call 1 (introduce): %s", response1.content[:100])

    # Call 2: Ask what your name is — it WON'T remember!
    response2 = llm.invoke([HumanMessage(content="What's my name?")])
    logger.info("Call 2 (ask name): %s", response2.content[:100])
    # Output: "I don't know your name" — because each call is independent!


# ═══════════════════════════════════════════════════════════════════════════════
# 2️⃣  Manual History: The Simplest Solution
# ═══════════════════════════════════════════════════════════════════════════════
#
# The fix: pass ALL previous messages with every new call.
# The LLM reads the full conversation and uses it as context.
#
# MESSAGE TYPES:
#   HumanMessage  → What the user said
#   AIMessage     → What the LLM responded
#   SystemMessage → Instructions for the LLM's behavior
#
# The conversation is just a LIST of these messages, in order.

def demo_manual_history() -> None:
    """Fix the memory problem by manually passing conversation history."""
    from langchain_core.messages import AIMessage, HumanMessage

    llm = _get_llm()

    # Build the conversation manually
    # We include the FULL history with every call
    response = llm.invoke([
        HumanMessage(content="Hi, my name is Krish and I am a Chief AI Engineer"),
        AIMessage(content="Hello Krish! Nice to meet you. What can I help you with?"),
        HumanMessage(content="What's my name and what do I do?"),
    ])
    logger.info("--- Manual History ---")
    logger.info("With history, LLM remembers: %s", response.content[:150])
    # Now it knows: "Your name is Krish and you're a Chief AI Engineer"

    # PROBLEM: Managing this list manually is tedious and error-prone.
    # That's why LangChain provides ChatMessageHistory and RunnableWithMessageHistory.


# ═══════════════════════════════════════════════════════════════════════════════
# 3️⃣  ChatMessageHistory — LangChain's History Storage
# ═══════════════════════════════════════════════════════════════════════════════
#
# ChatMessageHistory is a simple class that stores messages in a list.
# It provides .add_user_message() and .add_ai_message() methods.
#
# Think of it as a "conversation log" that you can read and write to.
#
# IN PRODUCTION:
#   - In-memory (ChatMessageHistory) → for dev/testing only
#   - Redis → for distributed production systems
#   - PostgreSQL → for persistent storage
#   - MongoDB → for document-based storage

def demo_chat_message_history() -> None:
    """Show how ChatMessageHistory stores and retrieves messages."""
    from langchain_community.chat_message_histories import ChatMessageHistory

    # Create a history store
    history = ChatMessageHistory()

    # Add messages (like appending to a conversation log)
    history.add_user_message("Hi, my name is Krish")
    history.add_ai_message("Hello Krish! How can I help you?")
    history.add_user_message("What's my name?")

    logger.info("--- ChatMessageHistory ---")
    logger.info("Stored %d messages:", len(history.messages))
    for msg in history.messages:
        logger.info("  [%s]: %s", msg.type, msg.content)

    # history.messages is a list[BaseMessage] — you can pass it to the LLM
    # history.clear() removes all messages


# ═══════════════════════════════════════════════════════════════════════════════
# 4️⃣  RunnableWithMessageHistory — Auto-Managed Memory
# ═══════════════════════════════════════════════════════════════════════════════
#
# THIS IS THE KEY CONCEPT. RunnableWithMessageHistory wraps any chain
# and automatically:
#   1. Loads previous messages from the history store
#   2. Appends them to the input before calling the chain
#   3. Saves the new input + output to the history store
#
# You just call .invoke() normally — the history management is invisible.
#
# HOW IT WORKS:
#   ┌─────────────────────────────────────────────────────────────────┐
#   │ User calls: with_history.invoke("What's my name?", config)     │
#   │                                                                  │
#   │ Step 1: Load history for session_id from store                  │
#   │         → [HumanMsg("I'm Krish"), AIMsg("Hello Krish!")]       │
#   │                                                                  │
#   │ Step 2: Prepend history to current input                        │
#   │         → [HumanMsg("I'm Krish"), AIMsg("Hello!"),             │
#   │            HumanMsg("What's my name?")]                         │
#   │                                                                  │
#   │ Step 3: Call the wrapped chain with full message list           │
#   │         → AIMsg("Your name is Krish!")                          │
#   │                                                                  │
#   │ Step 4: Save new messages to history store                      │
#   │         → Store now has 4 messages total                        │
#   └─────────────────────────────────────────────────────────────────┘
#
# SESSION ID:
#   Each conversation has a unique session_id (like a thread_id).
#   Different session_ids = different conversation histories.
#   This is how you support MULTIPLE users / conversations.

def demo_runnable_with_message_history() -> None:
    """Auto-managed conversation memory with session isolation."""
    from langchain_community.chat_message_histories import ChatMessageHistory
    from langchain_core.chat_history import BaseChatMessageHistory
    from langchain_core.messages import HumanMessage
    from langchain_core.runnables.history import RunnableWithMessageHistory

    llm = _get_llm()

    # ── Session store: maps session_id → ChatMessageHistory ──────────
    # In production, this would be Redis, PostgreSQL, etc.
    store: dict[str, ChatMessageHistory] = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        """Factory function: returns history for a given session."""
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    # ── Wrap the LLM with message history ────────────────────────────
    with_history = RunnableWithMessageHistory(llm, get_session_history)

    # ── Session 1: User "Krish" ──────────────────────────────────────
    config_krish = {"configurable": {"session_id": "user-krish"}}

    r1 = with_history.invoke(
        [HumanMessage(content="Hi, my name is Krish and I am a Chief AI Engineer")],
        config=config_krish,
    )
    logger.info("--- RunnableWithMessageHistory ---")
    logger.info("[Krish] Intro: %s", r1.content[:100])

    r2 = with_history.invoke(
        [HumanMessage(content="What's my name?")],
        config=config_krish,
    )
    logger.info("[Krish] Remembers: %s", r2.content[:100])

    # ── Session 2: Different user "John" — separate history ──────────
    config_john = {"configurable": {"session_id": "user-john"}}

    r3 = with_history.invoke(
        [HumanMessage(content="What's my name?")],
        config=config_john,
    )
    logger.info("[John] No history: %s", r3.content[:100])
    # John's session has no history — LLM doesn't know his name!

    r4 = with_history.invoke(
        [HumanMessage(content="My name is John")],
        config=config_john,
    )
    r5 = with_history.invoke(
        [HumanMessage(content="What's my name?")],
        config=config_john,
    )
    logger.info("[John] Now remembers: %s", r5.content[:100])

    logger.info("Sessions in store: %s", list(store.keys()))
    logger.info("Krish's history: %d messages", len(store["user-krish"].messages))
    logger.info("John's history: %d messages", len(store["user-john"].messages))


# ═══════════════════════════════════════════════════════════════════════════════
# 5️⃣  MessagesPlaceholder — Dynamic Slot in Prompt Templates
# ═══════════════════════════════════════════════════════════════════════════════
#
# When you use ChatPromptTemplate, you define fixed messages:
#   ("system", "You are helpful"), ("human", "{input}")
#
# But where does the CONVERSATION HISTORY go? You can't hardcode it
# because it grows with every message.
#
# MessagesPlaceholder creates a DYNAMIC SLOT that expands to hold
# any number of messages at runtime.
#
# Template:
#   [SystemMessage, MessagesPlaceholder("messages")]
#
# At runtime with 3 history messages:
#   [SystemMessage, HumanMsg1, AIMsg1, HumanMsg2, AIMsg2, HumanMsg3]
#
# The placeholder EXPANDS to fit all the messages.

def demo_messages_placeholder() -> None:
    """Use MessagesPlaceholder for dynamic conversation history in prompts."""
    from langchain_community.chat_message_histories import ChatMessageHistory
    from langchain_core.chat_history import BaseChatMessageHistory
    from langchain_core.messages import HumanMessage
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.runnables.history import RunnableWithMessageHistory

    llm = _get_llm()

    # ── Prompt with MessagesPlaceholder ──────────────────────────────
    # "messages" is the variable name — it will be filled with history
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer all questions to the best of your ability."),
        MessagesPlaceholder(variable_name="messages"),
    ])

    chain = prompt | llm

    # ── Wrap with history ────────────────────────────────────────────
    store: dict[str, ChatMessageHistory] = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    with_history = RunnableWithMessageHistory(chain, get_session_history)

    config = {"configurable": {"session_id": "placeholder-demo"}}

    r1 = with_history.invoke(
        [HumanMessage(content="Hi, my name is Krish")],
        config=config,
    )
    logger.info("--- MessagesPlaceholder ---")
    logger.info("Intro: %s", r1.content[:100])

    r2 = with_history.invoke(
        [HumanMessage(content="What's my name?")],
        config=config,
    )
    logger.info("Remembers: %s", r2.content[:100])


# ═══════════════════════════════════════════════════════════════════════════════
# 6️⃣  Multi-Variable Prompts with History — Adding {language}
# ═══════════════════════════════════════════════════════════════════════════════
#
# When your prompt has MULTIPLE variables (not just messages), you need
# to tell RunnableWithMessageHistory WHICH key contains the messages.
# Use input_messages_key="messages" to specify this.

def demo_multi_variable_with_history() -> None:
    """Prompt with both {messages} and {language} — history on messages only."""
    from langchain_community.chat_message_histories import ChatMessageHistory
    from langchain_core.chat_history import BaseChatMessageHistory
    from langchain_core.messages import HumanMessage
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.runnables.history import RunnableWithMessageHistory

    llm = _get_llm()

    # Prompt with TWO variables: {language} (fixed) and {messages} (dynamic)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer all questions in {language}."),
        MessagesPlaceholder(variable_name="messages"),
    ])

    chain = prompt | llm

    store: dict[str, ChatMessageHistory] = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    # input_messages_key tells RunnableWithMessageHistory:
    # "The 'messages' key in the input dict is where the chat messages are.
    #  Save/load history for THAT key. Leave 'language' alone."
    with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="messages",
    )

    config = {"configurable": {"session_id": "hindi-chat"}}

    r1 = with_history.invoke(
        {"messages": [HumanMessage(content="Hi, I am Krish")], "language": "Hindi"},
        config=config,
    )
    logger.info("--- Multi-Variable with History ---")
    logger.info("Hindi intro: %s", r1.content[:100])

    r2 = with_history.invoke(
        {"messages": [HumanMessage(content="What's my name?")], "language": "Hindi"},
        config=config,
    )
    logger.info("Hindi remembers: %s", r2.content[:100])


# ═══════════════════════════════════════════════════════════════════════════════
# 7️⃣  trim_messages — Token-Based Memory Limiting (CRITICAL for Production)
# ═══════════════════════════════════════════════════════════════════════════════
#
# THE PROBLEM:
#   As conversations grow, the message list gets HUGE.
#   LLMs have a context window limit (e.g., 8K, 32K, 128K tokens).
#   If your history exceeds this, the LLM call FAILS.
#   Even if it fits, more tokens = slower response + higher cost.
#
# THE SOLUTION: trim_messages()
#   Keeps only the MOST RECENT messages that fit within a token budget.
#   Older messages are dropped. The LLM only sees recent context.
#
# PARAMETERS:
#   max_tokens      → Maximum tokens to keep (e.g., 100, 4000)
#   strategy="last" → Keep the LAST N tokens (most recent messages)
#   token_counter   → The LLM model (it knows how to count its own tokens)
#   include_system  → Always keep the system message (True recommended)
#   allow_partial   → Allow cutting a message in the middle (False = keep whole)
#   start_on="human"→ Always start the trimmed history on a human message
#
# EXAMPLE:
#   10 messages (500 tokens total), max_tokens=100
#   → trim_messages keeps only the last ~3 messages that fit in 100 tokens
#   → Older messages are DROPPED (the LLM won't know about them)
#
# PRODUCTION VALUES (from Important-Rules.md):
#   max_tokens=4000, strategy="last", include_system=True

def demo_trim_messages() -> None:
    """Limit conversation history to a token budget — prevents overflow."""
    from operator import itemgetter

    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, trim_messages
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.runnables import RunnablePassthrough

    llm = _get_llm()

    # ── Create the trimmer ───────────────────────────────────────────
    trimmer = trim_messages(
        max_tokens=45,           # Very small for demo — use 4000+ in production
        strategy="last",         # Keep the most RECENT messages
        token_counter=llm,       # LLM counts its own tokens accurately
        include_system=True,     # ALWAYS keep the system message
        allow_partial=False,     # Don't cut messages in half
        start_on="human",        # Trimmed history starts on a human message
    )

    # ── Sample long conversation ─────────────────────────────────────
    messages = [
        SystemMessage(content="you're a good assistant"),
        HumanMessage(content="hi! I'm bob"),
        AIMessage(content="hi!"),
        HumanMessage(content="I like vanilla ice cream"),
        AIMessage(content="nice"),
        HumanMessage(content="whats 2 + 2"),
        AIMessage(content="4"),
        HumanMessage(content="thanks"),
        AIMessage(content="no problem!"),
        HumanMessage(content="having fun?"),
        AIMessage(content="yes!"),
    ]

    # ── Trim the messages ────────────────────────────────────────────
    trimmed = trimmer.invoke(messages)
    logger.info("--- trim_messages ---")
    logger.info("Original: %d messages", len(messages))
    logger.info("Trimmed:  %d messages", len(trimmed))
    for msg in trimmed:
        logger.info("  [%s]: %s", msg.type, msg.content)
    # Notice: "hi! I'm bob" is GONE — trimmed away because of token limit
    # The system message is KEPT (include_system=True)

    # ── Chain with trimmer — the production pattern ──────────────────
    # RunnablePassthrough.assign() replaces the "messages" key with trimmed version
    # This runs the trimmer BEFORE the prompt, so the LLM only sees recent messages
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer in {language}."),
        MessagesPlaceholder(variable_name="messages"),
    ])

    chain = (
        RunnablePassthrough.assign(messages=itemgetter("messages") | trimmer)
        | prompt
        | llm
    )

    # Ask about something from EARLY in the conversation (trimmed away)
    r1 = chain.invoke({
        "messages": messages + [HumanMessage(content="What ice cream do I like?")],
        "language": "English",
    })
    logger.info("Ask about trimmed info (ice cream): %s", r1.content[:120])
    # LLM doesn't know — "I'm bob" and "I like vanilla" were trimmed!

    # Ask about something RECENT (still in the trimmed window)
    r2 = chain.invoke({
        "messages": messages + [HumanMessage(content="What math problem did I ask?")],
        "language": "English",
    })
    logger.info("Ask about recent info (math): %s", r2.content[:120])
    # LLM knows — "whats 2 + 2" is recent enough to survive trimming


# ═══════════════════════════════════════════════════════════════════════════════
# 8️⃣  Streaming with Memory — Real-Time Responses
# ═══════════════════════════════════════════════════════════════════════════════

def demo_streaming_with_memory() -> None:
    """Stream responses token-by-token while maintaining conversation history."""
    from langchain_community.chat_message_histories import ChatMessageHistory
    from langchain_core.chat_history import BaseChatMessageHistory
    from langchain_core.messages import HumanMessage
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.runnables.history import RunnableWithMessageHistory

    llm = _get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Be concise."),
        MessagesPlaceholder(variable_name="messages"),
    ])

    chain = prompt | llm | StrOutputParser()

    store: dict[str, ChatMessageHistory] = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    with_history = RunnableWithMessageHistory(chain, get_session_history)
    config = {"configurable": {"session_id": "stream-demo"}}

    # ── Stream the response ──────────────────────────────────────────
    logger.info("--- Streaming with Memory ---")
    full_response = ""
    for chunk in with_history.stream(
        [HumanMessage(content="Tell me 3 benefits of Python in one sentence each.")],
        config=config,
    ):
        full_response += chunk

    logger.info("Streamed response (%d chars): %s...", len(full_response), full_response[:150])

    # The history is saved — follow-up works
    full_followup = ""
    for chunk in with_history.stream(
        [HumanMessage(content="Now tell me 3 downsides, same format.")],
        config=config,
    ):
        full_followup += chunk

    logger.info("Follow-up streamed (%d chars): %s...", len(full_followup), full_followup[:150])


# ═══════════════════════════════════════════════════════════════════════════════
# 🏭 INTERVIEW QUESTIONS — Conversational Memory
# ═══════════════════════════════════════════════════════════════════════════════
#
# Q1: Do LLMs have memory?
# A:  No. LLMs are stateless — each call is independent. "Memory" is
#     simulated by passing the full conversation history as input.
#
# Q2: What is RunnableWithMessageHistory?
# A:  A wrapper that auto-manages conversation history for any chain.
#     It loads history before each call, appends new messages after,
#     and isolates conversations by session_id.
#
# Q3: What is MessagesPlaceholder?
# A:  A dynamic slot in ChatPromptTemplate that expands to hold any
#     number of messages at runtime. Used for conversation history.
#
# Q4: Why do we need trim_messages?
# A:  Conversations grow unbounded. Without trimming, the message list
#     can exceed the LLM's context window, causing failures. trim_messages
#     keeps only the most recent messages within a token budget.
#
# Q5: What is session_id and why is it important?
# A:  A unique identifier for each conversation. Different session_ids
#     have separate histories. This is how you support multiple users
#     or multiple conversation threads.
#
# Q6: What's the difference between ChatMessageHistory and RunnableWithMessageHistory?
# A:  ChatMessageHistory is the STORAGE (stores messages in a list).
#     RunnableWithMessageHistory is the WRAPPER that auto-loads/saves
#     from that storage. You need both — storage + automation.
#
# Q7: How do you persist chat history in production?
# A:  Use Redis (fast, distributed), PostgreSQL (persistent, queryable),
#     or MongoDB (document-based). In-memory ChatMessageHistory is for
#     dev/testing only — it's lost when the server restarts.


# ═══════════════════════════════════════════════════════════════════════════════
# 🚀 Main — Run all demos
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("🧠 CONVERSATIONAL MEMORY — How Chatbots Remember")
    logger.info("=" * 70)

    logger.info("\n🔹 1. The Stateless Problem")
    demo_stateless_problem()

    logger.info("\n🔹 2. Manual History Fix")
    demo_manual_history()

    logger.info("\n🔹 3. ChatMessageHistory Storage")
    demo_chat_message_history()

    logger.info("\n🔹 4. RunnableWithMessageHistory (auto-managed)")
    demo_runnable_with_message_history()

    logger.info("\n🔹 5. MessagesPlaceholder in Prompts")
    demo_messages_placeholder()

    logger.info("\n🔹 6. Multi-Variable Prompts with History")
    demo_multi_variable_with_history()

    logger.info("\n🔹 7. trim_messages — Token-Based Memory Limiting")
    demo_trim_messages()

    logger.info("\n🔹 8. Streaming with Memory")
    demo_streaming_with_memory()

    logger.info("\n" + "=" * 70)
    logger.info("✅ Conversational Memory lesson complete!")
    logger.info("Next: Production chatbot with persistent memory")
    logger.info("=" * 70)
