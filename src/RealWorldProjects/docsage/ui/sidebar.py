"""
DocSage Sidebar — Streamlit sidebar for configuration and document upload.

WHY THIS FILE EXISTS:
    The sidebar is the CONTROL PANEL of the application. Users configure
    API keys, select models, and upload documents here. All user inputs
    are collected and returned as a dictionary for the main app to use.

INTERVIEW POINT:
    "How do you handle user configuration in your Streamlit app?"
    "The sidebar has four sections: document upload at the top (so the
    interviewer immediately sees the RAG capability), model selection,
    API keys, and a new conversation button. I use st.session_state to
    persist data across reruns. Each uploaded document gets a description
    field so the agent knows what it contains."
"""

import os
import logging
import uuid

import streamlit as st

from docsage.config import settings

logger = logging.getLogger(__name__)


def render_sidebar() -> dict:
    """
    Render the Streamlit sidebar and collect all user inputs.

    Sidebar order (optimized for interview demos):
    1. Document Upload — first thing the interviewer sees (RAG showcase).
    2. Model Selection — choose the Groq model.
    3. API Keys — Groq and Tavily keys (password fields).
    4. New Conversation — reset button at the bottom.

    Returns:
        Dictionary with all user inputs.
    """
    user_inputs = {
        "groq_api_key": "",
        "tavily_api_key": "",
        "selected_model": settings.GROQ_MODEL_OPTIONS[0],
        "uploaded_files": [],
        "is_ready": False,
    }

    with st.sidebar:
        st.header("⚙️ Configuration")

        # ==============================================================
        # SECTION 1: Document Upload (TOP — first thing interviewer sees)
        # ==============================================================
        st.subheader("📄 Upload Documents")

        allowed_types = list(settings.SUPPORTED_FILE_TYPES.keys())
        allowed_extensions = [ext.lstrip(".") for ext in allowed_types]

        uploaded_files = st.file_uploader(
            "PDF, DOCX, TXT, CSV, MD",
            type=allowed_extensions,
            accept_multiple_files=True,
            help="Each document becomes a separate knowledge base. The agent decides which to search.",
        )

        file_entries = []
        if uploaded_files:
            st.caption(f"📎 {len(uploaded_files)} file(s) uploaded")
            for i, uploaded_file in enumerate(uploaded_files):
                with st.expander(f"📄 {uploaded_file.name}", expanded=(i == 0)):
                    description = st.text_input(
                        "Describe this document",
                        value=f"Document: {uploaded_file.name}",
                        key=f"desc_{uploaded_file.name}_{i}",
                        help="Helps the agent route queries to the right document.",
                    )
                    file_entries.append((
                        uploaded_file.getvalue(),
                        uploaded_file.name,
                        description,
                    ))

        user_inputs["uploaded_files"] = file_entries

        st.divider()

        # ==============================================================
        # SECTION 1b: URL Knowledge Sources (Web Scraping)
        # ==============================================================
        st.subheader("🌐 Add URL Sources")

        url_input = st.text_input(
            "Paste a webpage URL",
            value="",
            placeholder="https://linkedin.com/in/your-profile",
            help="Scrapes the webpage and adds it as a knowledge source. Works with LinkedIn, GitHub, blogs, docs.",
        )

        url_description = ""
        if url_input:
            url_description = st.text_input(
                "Describe this URL (optional)",
                value="",
                placeholder="My LinkedIn profile with professional experience",
                key="url_desc",
                help="Helps the agent understand what this page contains.",
            )

        user_inputs["url_source"] = (url_input.strip(), url_description.strip()) if url_input.strip() else None

        st.divider()

        # ==============================================================
        # SECTION 2: Model Selection
        # ==============================================================
        st.subheader("🤖 Model")

        selected_model = st.selectbox(
            "Select Groq Model",
            options=settings.GROQ_MODEL_OPTIONS,
            index=0,
            help="Used for the agent. Grading always uses llama-3.3-70b.",
        )
        user_inputs["selected_model"] = selected_model

        st.divider()

        # ==============================================================
        # SECTION 3: API Keys
        # ==============================================================
        st.subheader("🔑 API Keys")

        groq_key = st.text_input(
            "Groq API Key",
            type="password",
            value=os.getenv("GROQ_API_KEY", ""),
            help="Get your free key at https://console.groq.com/keys",
        )
        user_inputs["groq_api_key"] = groq_key
        if groq_key:
            os.environ["GROQ_API_KEY"] = groq_key

        tavily_key = st.text_input(
            "Tavily API Key (optional)",
            type="password",
            value=os.getenv("TAVILY_API_KEY", ""),
            help="Enables web search fallback. https://app.tavily.com/home",
        )
        user_inputs["tavily_api_key"] = tavily_key
        if tavily_key:
            os.environ["TAVILY_API_KEY"] = tavily_key

        if not groq_key:
            st.warning("⚠️ Enter your Groq API key to proceed.")

        st.divider()

        # ==============================================================
        # SECTION 4: New Conversation + Info (BOTTOM)
        # ==============================================================
        if st.button("🔄 New Conversation", use_container_width=True):
            if "messages" in st.session_state:
                st.session_state.messages = []
            if "thread_id" in st.session_state:
                st.session_state.thread_id = str(uuid.uuid4())
            st.rerun()

        if "messages" in st.session_state and st.session_state.messages:
            turn_count = len([m for m in st.session_state.messages if m["role"] == "user"])
            st.caption(f"💬 {turn_count} turn(s) in this conversation")

        # Ready check
        is_ready = bool(groq_key)
        user_inputs["is_ready"] = is_ready

    return user_inputs
