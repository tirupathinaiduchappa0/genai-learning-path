"""
Streamlit UI for the Insurance Support Agent.

Run:
    streamlit run src/InterviewReady/insurance_support_agent/app.py
"""

# ── 0. Corporate-network SSL fix (must run before any HTTPS client import) ──
try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

import os
import sys
import uuid

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langgraph.types import Command

# Allow `streamlit run` from anywhere by ensuring the package's parent dir
# (src/InterviewReady) is importable.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from insurance_support_agent.graph import support_graph  # noqa: E402

load_dotenv()

st.set_page_config(page_title="Insurance Support Assistant", page_icon="🚗", layout="centered")
st.title("🚗 Insurance Support Assistant")
st.caption("LangGraph agent · policy status · claims · cancellations")

with st.sidebar:
    st.header("Try these")
    st.markdown(
        "- *\"What's the status of policy POL-1001?\"*\n"
        "- *\"I want to report a claim for POL-1002, my car was hit on 2026-07-15 "
        "near MG Road, a side mirror was damaged\"*\n"
        "- *\"Please cancel policy POL-1003, I sold the car\"*"
    )
    st.divider()
    st.markdown("**Mock policies available:** POL-1001, POL-1002, POL-1003")
    key_ok = bool(os.getenv("GROQ_API_KEY"))
    st.markdown(f"**Groq API key loaded:** {'✅' if key_ok else '❌ missing'}")
    if st.button("🗑️ Reset conversation"):
        for k in ["thread_id", "history", "pending_interrupt"]:
            st.session_state.pop(k, None)
        st.rerun()

# ── Session state ────────────────────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "history" not in st.session_state:
    st.session_state.history = []  # list of {"role": "user"/"assistant", "content": str}
if "pending_interrupt" not in st.session_state:
    st.session_state.pending_interrupt = None

config = {"configurable": {"thread_id": st.session_state.thread_id}}


def run_graph(graph_input):
    """Invoke the graph, render any new AI messages, and detect interrupts."""
    result = support_graph.invoke(graph_input, config=config)

    if "__interrupt__" in result:
        # Graph paused for human approval.
        payload = result["__interrupt__"][0].value
        st.session_state.pending_interrupt = payload
        return

    # No interrupt: surface the assistant's new message(s).
    for m in result.get("messages", [])[-1:]:
        if isinstance(m, AIMessage):
            st.session_state.history.append({"role": "assistant", "content": m.content})
    st.session_state.pending_interrupt = None


# ── Render past turns ────────────────────────────────────────────────────────
for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

# ── Human-in-the-loop approval UI ────────────────────────────────────────────
if st.session_state.pending_interrupt:
    payload = st.session_state.pending_interrupt
    with st.chat_message("assistant"):
        st.warning(f"⚠️ **Human approval required**\n\n{payload['prompt']}")
        col1, col2 = st.columns(2)
        approve = col1.button("✅ Approve cancellation", use_container_width=True)
        reject = col2.button("❌ Reject", use_container_width=True)

    if approve or reject:
        decision = bool(approve)
        st.session_state.history.append(
            {
                "role": "user",
                "content": "(approved cancellation)" if decision else "(rejected cancellation)",
            }
        )
        run_graph(Command(resume=decision))
        st.rerun()

# ── Chat input ───────────────────────────────────────────────────────────────
elif prompt := st.chat_input("Type your request..."):
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.spinner("Thinking..."):
        run_graph({"messages": [("user", prompt)]})
    st.rerun()
