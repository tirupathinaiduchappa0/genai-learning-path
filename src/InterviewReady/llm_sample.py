"""
════════════════════════════════════════════════════════════════════════════
 LLM INTEGRATION SAMPLE  —  Groq (open-source Llama)
════════════════════════════════════════════════════════════════════════════
 Purpose : Ready-to-run "sample code to integrate and interact with the LLM"
           exactly as required by the interview prerequisites mail.

 Run     :  python src/InterviewReady/llm_sample.py
            python src/InterviewReady/llm_sample.py "your custom question"

 Requires:  GROQ_API_KEY in .env  (already configured)
════════════════════════════════════════════════════════════════════════════
"""

import os
import sys

# ── 0. Trust the OS (Windows) certificate store ─────────────────────────────
# On corporate networks an SSL-inspecting proxy re-signs HTTPS traffic with a
# company root CA that Python's bundled certifi does NOT know about. truststore
# makes Python use the Windows cert store (where IT installed that root CA),
# fixing "CERTIFICATE_VERIFY_FAILED". Must run BEFORE any HTTPS client is made.
try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

# ── 1. Load API key from .env ───────────────────────────────────────────────
load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    raise SystemExit(
        "GROQ_API_KEY not found. Add it to your .env file before running."
    )

# ── 2. Model choices (both verified working on Groq in this project) ─────────
FAST_MODEL = "llama-3.1-8b-instant"      # low latency, great for live demos
SMART_MODEL = "llama-3.3-70b-versatile"  # stronger reasoning


def build_llm(model: str = FAST_MODEL, temperature: float = 0.3) -> ChatGroq:
    """Create a ChatGroq LLM instance."""
    return ChatGroq(model=model, temperature=temperature, max_tokens=1024)


def ask(question: str, model: str = FAST_MODEL) -> str:
    """Single-shot: send a question, get an answer back as a string."""
    llm = build_llm(model)
    messages = [
        SystemMessage(content="You are a concise, helpful assistant."),
        HumanMessage(content=question),
    ]
    response = llm.invoke(messages)
    return response.content


def ask_streaming(question: str, model: str = FAST_MODEL) -> None:
    """Stream tokens as they arrive — nice for a live 'show me the output' demo."""
    llm = build_llm(model)
    print("Assistant: ", end="", flush=True)
    for chunk in llm.stream(question):
        print(chunk.content, end="", flush=True)
    print()


def interactive_chat() -> None:
    """A minimal multi-turn chatbot loop with conversation memory."""
    llm = build_llm()
    history = [SystemMessage(content="You are a helpful assistant.")]
    print("Chatbot ready. Type 'exit' to quit.\n")
    while True:
        user = input("You: ").strip()
        if user.lower() in {"exit", "quit"}:
            print("Bye!")
            break
        if not user:
            continue
        history.append(HumanMessage(content=user))
        reply = llm.invoke(history)
        print(f"Assistant: {reply.content}\n")
        history.append(reply)


if __name__ == "__main__":
    # If a question is passed on the command line, answer it. Otherwise run a
    # quick self-test proving the LLM integration works end-to-end.
    if len(sys.argv) > 1:
        user_question = " ".join(sys.argv[1:])
        print(f"Q: {user_question}\n")
        print(f"A: {ask(user_question)}")
    else:
        print("=" * 70)
        print("Groq LLM integration self-test")
        print("=" * 70)
        print(f"Model: {FAST_MODEL}\n")
        answer = ask("In one sentence, confirm you are working and name yourself.")
        print(f"A: {answer}\n")
        print("Integration OK. LLM is reachable and responding.")
