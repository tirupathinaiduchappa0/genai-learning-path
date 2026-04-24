"""
📡 Lesson 7.2 — LangServe Client: Calling LangServe APIs from Python

═══════════════════════════════════════════════════════════════════
HOW TO CALL A LANGSERVE API
═══════════════════════════════════════════════════════════════════

Once your LangServe server is running (lesson 01), you need to CALL it
from your application. There are TWO ways:

1. RemoteRunnable (LangServe SDK) — The recommended way
   - Treats the remote API as if it were a local chain
   - Supports .invoke(), .batch(), .stream() — same interface!
   - Type-safe, handles serialization automatically

2. requests / httpx (raw HTTP) — The universal way
   - Works from ANY language (Python, JavaScript, curl)
   - You manually construct the JSON body
   - Good for non-Python clients or simple scripts

PREREQUISITE:
    Start the server first (in a separate terminal):
    $ cd LangCGS/src/LangchainBasics/LangServe
    $ python 01_langserve_server.py

    Then run this client:
    $ python 02_langserve_client.py

Author: GenAI Learner
Date: 2026-04-13
"""

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Base URL of the LangServe server
BASE_URL = "http://127.0.0.1:8000"


# ═══════════════════════════════════════════════════════════════════════════════
# 1️⃣  Method 1: RemoteRunnable (LangServe SDK) — Recommended
# ═══════════════════════════════════════════════════════════════════════════════
#
# RemoteRunnable wraps a LangServe endpoint as a local Runnable.
# It has the SAME interface as any LCEL chain:
#   .invoke()  → single input, single output
#   .batch()   → multiple inputs, multiple outputs
#   .stream()  → token-by-token streaming
#
# This means you can use a remote chain EXACTLY like a local chain.
# You could even pipe it with other local components:
#   remote_chain = RemoteRunnable(url) | local_parser | local_function
#
# INSTALL: pip install langserve[client]
# (The [client] extra installs httpx and other client dependencies)

def demo_remote_runnable() -> None:
    """Call LangServe using RemoteRunnable — the recommended approach."""
    from langserve import RemoteRunnable

    # ── Connect to the translate chain ───────────────────────────────────
    # This creates a "proxy" that behaves like a local chain
    # but actually sends HTTP requests to the server.
    translate = RemoteRunnable(f"{BASE_URL}/translate")

    # ── .invoke() — Single request ──────────────────────────────────────
    # Input format matches the chain's prompt variables: {language, text}
    result = translate.invoke({
        "language": "French",
        "text": "Hello, how are you?",
    })
    logger.info("--- RemoteRunnable .invoke() ---")
    logger.info("Translate to French: %s", result)

    # ── .invoke() with different language ────────────────────────────────
    result_spanish = translate.invoke({
        "language": "Spanish",
        "text": "Good morning, welcome to the course!",
    })
    logger.info("Translate to Spanish: %s", result_spanish)

    # ── .batch() — Multiple requests at once ─────────────────────────────
    # Sends all inputs in one HTTP call — more efficient than looping
    results = translate.batch([
        {"language": "German", "text": "Thank you"},
        {"language": "Japanese", "text": "Good night"},
        {"language": "Hindi", "text": "How are you?"},
    ])
    logger.info("--- RemoteRunnable .batch() ---")
    for i, r in enumerate(results):
        logger.info("  Result %d: %s", i, r)

    # ── .stream() — Token-by-token streaming ─────────────────────────────
    # Each chunk arrives as it's generated — for real-time UIs
    logger.info("--- RemoteRunnable .stream() ---")
    full_response = ""
    for chunk in translate.stream({"language": "Italian", "text": "I love programming"}):
        full_response += chunk
    logger.info("Streamed Italian translation: %s", full_response)

    # ── Call the essay chain ─────────────────────────────────────────────
    essay = RemoteRunnable(f"{BASE_URL}/essay")
    essay_result = essay.invoke({"topic": "Why Python is great for AI"})
    logger.info("--- Essay chain ---")
    logger.info("Essay (first 200 chars): %s...", essay_result[:200])

    # ── Call the code explainer chain ────────────────────────────────────
    code = RemoteRunnable(f"{BASE_URL}/code")
    code_result = code.invoke({
        "code": "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
    })
    logger.info("--- Code explainer ---")
    logger.info("Explanation: %s", code_result[:200])


# ═══════════════════════════════════════════════════════════════════════════════
# 2️⃣  Method 2: Raw HTTP with requests — Universal Approach
# ═══════════════════════════════════════════════════════════════════════════════
#
# This works from ANY language or tool (Python, JavaScript, curl, Postman).
# You manually construct the JSON body following LangServe's format:
#
# POST /chain/invoke
# Body: {
#   "input": { ...your chain variables... },
#   "config": {},    ← optional LangChain config (tags, metadata, etc.)
#   "kwargs": {}     ← optional extra kwargs
# }
#
# Response: {
#   "output": "...the chain's response..."
# }

def demo_raw_http() -> None:
    """Call LangServe using raw HTTP requests — works from any language."""
    import requests

    # ── /invoke — Single request ─────────────────────────────────────────
    response = requests.post(
        f"{BASE_URL}/translate/invoke",
        json={
            "input": {
                "language": "Portuguese",
                "text": "Hello, welcome!",
            },
            # "config": {},   ← optional: add tags, metadata for LangSmith
            # "kwargs": {},   ← optional: extra arguments
        },
    )
    data = response.json()
    logger.info("--- Raw HTTP /invoke ---")
    logger.info("Status: %d", response.status_code)
    logger.info("Response: %s", data.get("output", data))

    # ── /batch — Multiple requests ───────────────────────────────────────
    response = requests.post(
        f"{BASE_URL}/translate/batch",
        json={
            "inputs": [
                {"language": "French", "text": "Good morning"},
                {"language": "Korean", "text": "Thank you very much"},
            ],
        },
    )
    data = response.json()
    logger.info("--- Raw HTTP /batch ---")
    logger.info("Batch results: %s", data.get("output", data))

    # ── /input_schema — Get the expected input format ────────────────────
    response = requests.get(f"{BASE_URL}/translate/input_schema")
    logger.info("--- Input Schema ---")
    logger.info("Schema: %s", response.json())
    # This tells you exactly what fields the chain expects.
    # Useful for building frontend forms dynamically.

    # ── Health check ─────────────────────────────────────────────────────
    response = requests.get(f"{BASE_URL}/health")
    logger.info("--- Health Check ---")
    logger.info("Health: %s", response.json())


# ═══════════════════════════════════════════════════════════════════════════════
# 📖 HOW TO CALL FROM JAVASCRIPT (for your React frontend)
# ═══════════════════════════════════════════════════════════════════════════════
#
# // Using fetch (vanilla JS)
# const response = await fetch("http://localhost:8000/translate/invoke", {
#     method: "POST",
#     headers: { "Content-Type": "application/json" },
#     body: JSON.stringify({
#         input: { language: "French", text: "Hello" }
#     }),
# });
# const data = await response.json();
# console.log(data.output);  // "Bonjour"
#
# // Streaming with EventSource (Server-Sent Events)
# const eventSource = new EventSource("http://localhost:8000/translate/stream");
# eventSource.onmessage = (event) => {
#     const chunk = JSON.parse(event.data);
#     console.log(chunk);  // Each token as it arrives
# };
#
# // Using axios (React)
# import axios from "axios";
# const { data } = await axios.post("http://localhost:8000/translate/invoke", {
#     input: { language: "Spanish", text: "Good morning" }
# });
# console.log(data.output);


# ═══════════════════════════════════════════════════════════════════════════════
# 📖 HOW TO CALL FROM CURL (for quick testing)
# ═══════════════════════════════════════════════════════════════════════════════
#
# # Invoke
# curl -X POST http://localhost:8000/translate/invoke \
#   -H "Content-Type: application/json" \
#   -d '{"input": {"language": "French", "text": "Hello"}}'
#
# # Batch
# curl -X POST http://localhost:8000/translate/batch \
#   -H "Content-Type: application/json" \
#   -d '{"inputs": [{"language": "French", "text": "Hi"}, {"language": "German", "text": "Hi"}]}'
#
# # Input schema
# curl http://localhost:8000/translate/input_schema


# ═══════════════════════════════════════════════════════════════════════════════
# 🚀 Main — Run demos (server must be running first!)
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    logger.info("=" * 70)
    logger.info("📡 LANGSERVE CLIENT — Calling LangServe APIs")
    logger.info("=" * 70)
    logger.info("Make sure the server is running: python 01_langserve_server.py")
    logger.info("")

    # Check if server is reachable
    try:
        import requests
        resp = requests.get(f"{BASE_URL}/health", timeout=3)
        if resp.status_code != 200:
            raise ConnectionError()
    except Exception:
        logger.error("Server not reachable at %s", BASE_URL)
        logger.error("Start the server first: python 01_langserve_server.py")
        sys.exit(1)

    logger.info("Server is running. Starting demos...\n")

    logger.info("🔹 1. RemoteRunnable (LangServe SDK)")
    demo_remote_runnable()

    logger.info("\n🔹 2. Raw HTTP with requests")
    demo_raw_http()

    logger.info("\n" + "=" * 70)
    logger.info("✅ LangServe Client lesson complete!")
    logger.info("=" * 70)
