"""
DocSage Chat Interface — Streamlit chat UI with step-by-step streaming.

WHY THIS FILE EXISTS:
    This is the USER-FACING part of the application. It displays the
    conversation history, handles user input, streams graph execution
    step-by-step, and shows the response with source citations.

INTERVIEW POINT:
    "How do you handle streaming in your RAG application?"
    "I use graph.stream() with stream_mode='updates' to get real-time
    progress from each node. As each node completes, the UI updates
    with a status message: 'Selecting knowledge source...', 'Retrieving
    documents...', 'Grading relevance...', 'Generating answer...'. This
    gives the user feedback during the 10-30 second processing time
    instead of just a spinner."
"""

import logging
import os
import uuid
from typing import Any

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from docsage.config import settings

logger = logging.getLogger(__name__)

# Map node names to user-friendly status messages with emojis.
# These are shown in real-time as each node completes.
NODE_STATUS_MESSAGES = {
    "agent": "🤖 Selecting knowledge source...",
    "tools": "📄 Retrieving documents...",
    "grade_documents": "✅ Checking document relevance...",
    "generate": "✍️ Generating answer...",
    "rewrite": "🔄 Improving query and retrying...",
    "validate": "🔍 Validating answer quality...",
}


def initialize_chat_state() -> None:
    """
    Initialize session state variables for the chat interface.

    Called once at app startup. Sets up:
    - messages: list of dicts for display history.
    - thread_id: unique ID for conversation memory (MemorySaver).
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())


def display_chat_history() -> None:
    """
    Display all previous messages in the chat interface.
    """
    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]
        sources = msg.get("sources", [])

        with st.chat_message(role):
            st.markdown(content)
            if role == "assistant" and sources:
                with st.expander("📚 Sources", expanded=False):
                    for source in sources:
                        st.caption(source)


def handle_user_input(graph: Any) -> None:
    """
    Handle user chat input with step-by-step streaming progress.

    Uses graph.stream(stream_mode="updates") to show real-time
    progress as each node completes. The user sees status messages
    like "Retrieving documents..." instead of just a spinner.

    Args:
        graph: The compiled LangGraph (from GraphBuilder.build()).
    """
    user_message = st.chat_input("Ask anything about your documents...")

    if not user_message:
        return

    # -- Display user message --------------------------------------------------
    st.session_state.messages.append({"role": "user", "content": user_message})
    with st.chat_message("user"):
        st.markdown(user_message)

    # -- Stream the graph with step-by-step progress ---------------------------
    with st.chat_message("assistant"):
        answer, sources, web_used = _stream_graph(graph, user_message)

        # Display the final answer
        st.markdown(answer)

        if sources:
            with st.expander("📚 Sources", expanded=False):
                for source in sources:
                    st.caption(source)

        if web_used == "Yes":
            st.caption("🌐 Web search was used for this answer.")

    # -- Save to session state -------------------------------------------------
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })


def _stream_graph(graph: Any, user_message: str) -> tuple[str, list[str], str]:
    """
    Stream the graph execution with real-time step-by-step progress.

    Uses graph.stream(stream_mode="updates") to receive updates as
    each node completes. Shows a status placeholder that updates
    in real-time so the user sees what's happening.

    If streaming fails (tool_use_failed, etc.), falls back to a
    direct LLM call.

    Args:
        graph: The compiled LangGraph.
        user_message: The user's question.

    Returns:
        Tuple of (answer, sources, web_used).
    """
    config = {
        "configurable": {"thread_id": st.session_state.thread_id},
        "recursion_limit": settings.RECURSION_LIMIT,
    }

    # Create a status placeholder for real-time progress.
    # It will be cleared after completion so only the answer remains.
    status_placeholder = st.empty()
    status_container = status_placeholder.status("🔍 Processing your question...", expanded=True)

    try:
        final_state = {}

        # Stream node-by-node updates
        for update in graph.stream(
            {"messages": [HumanMessage(content=user_message)]},
            config=config,
            stream_mode="updates",
        ):
            for node_name, node_output in update.items():
                step_msg = NODE_STATUS_MESSAGES.get(node_name, f"⚙️ Running {node_name}...")
                status_container.update(label=step_msg)
                logger.info("  [stream] Node completed: %s", node_name)

                if isinstance(node_output, dict):
                    final_state.update(node_output)

        # Clear the status widget entirely — only the answer should remain
        status_placeholder.empty()

        return _extract_response(final_state)

    except RecursionError:
        status_placeholder.empty()
        logger.warning("Graph hit recursion limit (%d)", settings.RECURSION_LIMIT)
        return (
            "I wasn't able to find a satisfactory answer after multiple attempts. "
            "Try rephrasing your question or uploading a more relevant document.",
            [], "No",
        )

    except Exception as e:
        error_msg = str(e)
        logger.error("Graph streaming failed: %s", error_msg[:200])

        # Handle tool_use_failed (400) — retry with direct LLM call
        if "tool_use_failed" in error_msg or "400" in error_msg[:50]:
            status_container.update(label="🔄 Retrying with direct answer...")
            logger.info("Retrying with direct LLM call...")
            try:
                from langchain_groq import ChatGroq
                direct_llm = ChatGroq(
                    api_key=os.getenv("GROQ_API_KEY", ""),
                    model="llama-3.1-8b-instant",
                    temperature=0.3,
                    max_tokens=1024,
                )
                response = direct_llm.invoke([HumanMessage(content=user_message)])
                status_placeholder.empty()
                return response.content, [], "No"
            except Exception as retry_err:
                logger.error("Direct retry failed: %s", str(retry_err)[:100])

        status_placeholder.empty()

        if "rate_limit" in error_msg.lower() or "429" in error_msg:
            return "The API rate limit was reached. Please wait a few seconds and try again.", [], "No"
        elif "api_key" in error_msg.lower():
            return "API key error. Please check your Groq API key in the sidebar.", [], "No"
        elif "413" in error_msg[:50]:
            return "The request was too large. Try asking a shorter question.", [], "No"
        else:
            return "Sorry, an error occurred. Please try again.", [], "No"


def _extract_response(state: dict) -> tuple[str, list[str], str]:
    """
    Extract the answer, sources, and web search flag from the final state.

    Args:
        state: The accumulated state dict from streaming.

    Returns:
        Tuple of (answer_text, sources_list, web_search_flag).
    """
    # Try generation field first (set by generate_node)
    answer = state.get("generation", "")

    # Fall back to the last message in the messages list
    if not answer:
        messages = state.get("messages", [])
        if isinstance(messages, list):
            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and msg.content:
                    answer = msg.content
                    break
                # Also check ToolMessages (e.g., email sent confirmation)
                elif isinstance(msg, ToolMessage) and msg.content:
                    answer = msg.content
                    break
                elif isinstance(msg, str):
                    answer = msg
                    break

    if not answer:
        answer = "I couldn't generate a response. Please try rephrasing your question."

    sources = state.get("sources", [])
    web_used = state.get("web_search_used", "No")

    return answer, sources, web_used
