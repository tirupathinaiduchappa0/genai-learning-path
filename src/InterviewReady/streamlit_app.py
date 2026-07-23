"""
════════════════════════════════════════════════════════════════════════════
 STREAMLIT CHATBOT BASE  —  Groq (open-source Llama)
════════════════════════════════════════════════════════════════════════════
 A clean, generic chat UI ready to adapt to ANY interview use case.

 Run:  streamlit run src/InterviewReady/streamlit_app.py

 What it already handles:
   • truststore SSL fix (corporate network)      -> no CERTIFICATE errors
   • .env loading of GROQ_API_KEY
   • model + temperature picker in the sidebar
   • full multi-turn chat with memory
   • streaming responses (tokens appear live)
   • editable SYSTEM_PROMPT -> this is where you inject the use case's role
════════════════════════════════════════════════════════════════════════════
"""

# ── 0. Corporate-network SSL fix (MUST be first, before any HTTPS import) ────
try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

import os

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq

load_dotenv()

# ── 1. The role of the bot. EDIT THIS to match the interview use case. ───────
SYSTEM_PROMPT = (
    "You are a helpful, concise AI assistant. Answer clearly and accurately."
)

AVAILABLE_MODELS = [
    "llama-3.1-8b-instant",      # fast, low latency
    "llama-3.3-70b-versatile",   # stronger reasoning
]

# ── 2. Page setup ────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="centered")
st.title("🤖 AI Chatbot")
st.caption("Powered by Groq + Llama · built live")

# ── 3. Sidebar controls ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    model = st.selectbox("Model", AVAILABLE_MODELS, index=0)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1)
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

    key_ok = bool(os.getenv("GROQ_API_KEY"))
    st.markdown(f"**API key loaded:** {'✅' if key_ok else '❌ missing'}")

# ── 4. Session state: conversation history ───────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []


def build_llm() -> ChatGroq:
    return ChatGroq(model=model, temperature=temperature, max_tokens=1024)


def to_lc_messages():
    """Convert stored chat history into LangChain message objects."""
    msgs = [SystemMessage(content=SYSTEM_PROMPT)]
    for m in st.session_state.messages:
        if m["role"] == "user":
            msgs.append(HumanMessage(content=m["content"]))
        else:
            msgs.append(AIMessage(content=m["content"]))
    return msgs


# ── 5. Render existing conversation ──────────────────────────────────────────
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ── 6. Chat input + streamed response ────────────────────────────────────────
if prompt := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        llm = build_llm()
        placeholder = st.empty()
        full = ""
        for chunk in llm.stream(to_lc_messages()):
            full += chunk.content
            placeholder.markdown(full + "▌")
        placeholder.markdown(full)

    st.session_state.messages.append({"role": "assistant", "content": full})
