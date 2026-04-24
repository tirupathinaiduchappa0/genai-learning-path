"""
DocSage Main — The orchestrator that wires UI + LLM + Tools + Graph together.

WHY THIS FILE EXISTS:
    This is the GLUE that connects all the modular pieces:
    1. Sidebar collects user inputs (API keys, model, documents).
    2. LLM Factory creates task-specific LLM instances.
    3. Retriever Tool Builder ingests documents into vector stores.
    4. Web Search Tool provides external fallback.
    5. Graph Builder assembles the LangGraph workflow.
    6. Chat Interface handles user interaction.

    Each piece is independent and testable. This file just connects them.

INTERVIEW POINT:
    "How do you wire all the components together?"
    "I have a main orchestrator function that reads user inputs from the
    sidebar, builds retriever tools from uploaded documents, creates the
    LLM factory, assembles the LangGraph workflow, and passes the compiled
    graph to the chat interface. Each component is a separate module —
    the main function just connects them. This makes it easy to swap any
    component without touching the others."

THE FLOW:
    1. render_sidebar() -> user_inputs dict
    2. GroqLLMFactory(api_key) -> factory
    3. build_retriever_from_file() for each uploaded doc -> tools list
    4. create_web_search_tool() -> web tool (added to tools list)
    5. GraphBuilder(factory, tools).build() -> compiled graph
    6. handle_user_input(graph) -> chat interaction loop
"""

import logging
import hashlib
import os

import streamlit as st

from docsage.config import settings
from docsage.graph.graph_builder import GraphBuilder
from docsage.llms.groq_llm import GroqLLMFactory
from docsage.tools.retriever_tool import build_retriever_from_file, build_retriever_from_url
from docsage.tools.web_search_tool import create_web_search_tool
from docsage.tools.email_tool import send_email
from docsage.ui.chat_interface import (
    display_chat_history,
    handle_user_input,
    initialize_chat_state,
)
from docsage.ui.sidebar import render_sidebar

logger = logging.getLogger(__name__)


def run_docsage() -> None:
    """
    Main entry point for the DocSage application.

    Orchestrates the full flow: UI setup -> tool building -> graph
    compilation -> chat interaction. Uses Streamlit caching to avoid
    rebuilding tools and graphs on every rerun.
    """
    # -- Page config (must be first Streamlit call) ----------------------------
    st.set_page_config(
        page_title=settings.PAGE_TITLE,
        page_icon=settings.PAGE_ICON,
        layout=settings.PAGE_LAYOUT,
    )
    st.title(f"{settings.PAGE_ICON} {settings.PAGE_TITLE}")

    # -- Initialize chat state -------------------------------------------------
    initialize_chat_state()

    # -- Render sidebar and collect inputs -------------------------------------
    user_inputs = render_sidebar()

    # -- Show status below title -----------------------------------------------
    if user_inputs["is_ready"]:
        num_docs = len(user_inputs["uploaded_files"])
        if num_docs > 0:
            st.caption(f"✅ Ready — {num_docs} document(s) loaded. Ask anything!")
        else:
            st.caption("✅ Ready — Chat directly or upload documents for RAG.")

    # -- Display chat history (previous messages) ------------------------------
    display_chat_history()

    # -- Check if the app is ready to run --------------------------------------
    if not user_inputs["is_ready"]:
        if not user_inputs["groq_api_key"]:
            st.info("👋 Welcome to DocSage! Enter your Groq API key in the sidebar to get started.")
        return

    # -- Build the graph (cached to avoid rebuilding on every rerun) -----------
    graph = _get_or_build_graph(user_inputs)

    if graph is None:
        st.error("Failed to build the processing pipeline. Check the logs.")
        return

    # -- Handle user chat input ------------------------------------------------
    handle_user_input(graph)


def _get_or_build_graph(user_inputs: dict):
    """
    Get the cached graph or build a new one if inputs changed.

    Uses a fingerprint of the uploaded files + model selection to detect
    changes. If the fingerprint matches the cached version, reuse the
    graph. Otherwise, rebuild everything.

    This avoids re-ingesting documents on every Streamlit rerun (which
    happens on every user interaction). Only rebuilds when the user
    uploads new documents or changes the model.

    Args:
        user_inputs: Dictionary from render_sidebar().

    Returns:
        Compiled LangGraph, or None if building failed.
    """
    # Create a fingerprint from the current inputs
    fingerprint = _compute_fingerprint(user_inputs)

    # Check if we already have a cached graph with the same fingerprint
    if (
        "graph" in st.session_state
        and "graph_fingerprint" in st.session_state
        and st.session_state.graph_fingerprint == fingerprint
    ):
        return st.session_state.graph

    # Inputs changed — rebuild everything
    logger.info("Building new graph (fingerprint changed)...")

    try:
        with st.spinner("🔧 Building knowledge base from your documents..."):
            graph = _build_graph(user_inputs)

        # Cache the graph and fingerprint
        st.session_state.graph = graph
        st.session_state.graph_fingerprint = fingerprint

        logger.info("Graph built and cached successfully.")
        return graph

    except Exception as e:
        logger.error("Failed to build graph: %s", str(e))
        st.error(f"Error building pipeline: {str(e)[:200]}")
        return None


def _build_graph(user_inputs: dict):
    """
    Build the full DocSage graph from user inputs.

    Steps:
    1. Create retriever tools from uploaded documents.
    2. Create web search tool (if Tavily key provided).
    3. Create LLM factory.
    4. Build and compile the graph.

    Args:
        user_inputs: Dictionary from render_sidebar().

    Returns:
        Compiled LangGraph.
    """
    # -- Step 1: Build retriever tools from uploaded documents ------------------
    tools = []
    failed_files = []
    for file_bytes, file_name, description in user_inputs["uploaded_files"]:
        try:
            logger.info("Ingesting document: %s", file_name)
            tool = build_retriever_from_file(
                file_bytes=file_bytes,
                file_name=file_name,
                description=description,
            )
            tools.append(tool)
            logger.info("Tool created: %s", tool.name)
        except Exception as e:
            # If one document fails, continue with the others.
            # Don't let one bad file break the entire pipeline.
            logger.error("Failed to ingest '%s': %s", file_name, str(e)[:100])
            failed_files.append(file_name)

    # Warn the user about failed files (but don't block the app)
    if failed_files:
        st.warning(
            f"⚠️ Could not process: {', '.join(failed_files)}. "
            f"These files may be corrupted or empty. Other documents are still available."
        )

    # -- Step 1b: Build retriever from URL source (if provided) ----------------
    url_source = user_inputs.get("url_source")
    if url_source:
        url, url_desc = url_source
        try:
            logger.info("Scraping URL: %s", url[:80])
            url_tool = build_retriever_from_url(url=url, description=url_desc)
            tools.append(url_tool)
            logger.info("URL tool created: %s", url_tool.name)
        except Exception as e:
            logger.error("Failed to scrape URL '%s': %s", url[:50], str(e)[:100])
            st.warning(f"⚠️ Could not load URL: {str(e)[:150]}")

    # -- Step 2: Add web search tool (if Tavily key is available) --------------
    if user_inputs["tavily_api_key"]:
        web_tool = create_web_search_tool()
        tools.append(web_tool)
        logger.info("Web search tool added.")
    else:
        logger.info("No Tavily key — web search fallback disabled.")

    # -- Step 2b: Add email tool (if Gmail credentials are available) ----------
    gmail_address = os.getenv("GMAIL_ADDRESS", "")
    gmail_password = os.getenv("GMAIL_APP_PASSWORD", "")
    logger.info("Gmail check: address='%s', password_set=%s", gmail_address, bool(gmail_password))
    if gmail_address and gmail_password:
        tools.append(send_email)
        logger.info("Email tool added (sender: %s).", gmail_address)
    else:
        logger.info("No Gmail credentials — email tool disabled.")

    if not tools:
        # No documents uploaded and no Tavily key — the agent can still
        # answer directly from its own knowledge. We create a minimal
        # graph that just uses the agent without any tools.
        logger.info("No tools available — agent will answer directly without retrieval.")

    # -- Step 3: Create LLM factory --------------------------------------------
    # If the user selected a specific model in the sidebar, pass it as override.
    # Otherwise, each task uses its default model from settings.
    selected_model = user_inputs["selected_model"]
    model_override = None
    if selected_model != settings.GROQ_MODEL_OPTIONS[0]:
        # User picked a non-default model — override all tasks
        model_override = selected_model

    factory = GroqLLMFactory(
        api_key=user_inputs["groq_api_key"],
        model_name=model_override,
    )

    # -- Step 4: Build and compile the graph -----------------------------------
    logger.info("Total tools for graph: %d -> %s", len(tools), [t.name for t in tools])
    builder = GraphBuilder(llm_factory=factory, tools=tools)
    graph = builder.build()

    return graph


def _compute_fingerprint(user_inputs: dict) -> str:
    """
    Compute a fingerprint from user inputs to detect changes.

    The fingerprint includes:
    - Names and sizes of uploaded files (detect new/removed files).
    - Selected model (detect model change).
    - Tavily key presence (detect web search toggle).

    We do NOT include file content in the hash (too slow for large files).
    File name + size is a good enough proxy for change detection.

    Args:
        user_inputs: Dictionary from render_sidebar().

    Returns:
        A hex string fingerprint.
    """
    parts = []
    for _, file_name, description in user_inputs["uploaded_files"]:
        parts.append(f"{file_name}:{description}")
    parts.append(f"model:{user_inputs['selected_model']}")
    parts.append(f"tavily:{'yes' if user_inputs['tavily_api_key'] else 'no'}")
    # Include Gmail availability in fingerprint
    parts.append(f"gmail:{'yes' if os.getenv('GMAIL_ADDRESS') and os.getenv('GMAIL_APP_PASSWORD') else 'no'}")
    # Include URL source in fingerprint
    url_source = user_inputs.get("url_source")
    if url_source:
        parts.append(f"url:{url_source[0]}")

    raw = "|".join(sorted(parts))
    return hashlib.md5(raw.encode()).hexdigest()
