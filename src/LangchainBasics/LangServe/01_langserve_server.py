"""
🚀 Lesson 7.1 — LangServe: Deploying LangChain Chains as REST APIs

═══════════════════════════════════════════════════════════════════
WHAT IS LANGSERVE?
═══════════════════════════════════════════════════════════════════

LangServe is a library that deploys ANY LangChain chain (LCEL) as a
production-ready REST API — in ONE line of code.

Think of it this way:
    - You built a chain: prompt | llm | parser
    - Now you want OTHER apps (React frontend, mobile app, another service)
      to call this chain over HTTP
    - LangServe wraps your chain in a FastAPI server automatically

WITHOUT LangServe (manual approach):
    You'd have to:
    1. Create a FastAPI app
    2. Define request/response Pydantic models
    3. Write an endpoint that calls chain.invoke()
    4. Handle streaming manually
    5. Write error handling
    6. Document the API

WITH LangServe (one line):
    add_routes(app, chain, path="/my-chain")
    Done. It auto-generates:
    - POST /my-chain/invoke        → Run the chain, get full response
    - POST /my-chain/batch         → Run on multiple inputs at once
    - POST /my-chain/stream        → Stream response token by token
    - POST /my-chain/stream_log    → Stream with intermediate steps
    - GET  /my-chain/playground    → Built-in web UI for testing!
    - GET  /my-chain/input_schema  → JSON schema of expected input
    - GET  /my-chain/output_schema → JSON schema of output

HOW IT RELATES TO FASTAPI:
    LangServe is built ON TOP of FastAPI. Your LangServe app IS a FastAPI app.
    You can add regular FastAPI endpoints alongside LangServe routes.
    All FastAPI features work: middleware, CORS, auth, dependency injection.

WHY LANGSERVE EXISTS:
    Before LangServe, every team wrote their own FastAPI wrapper for chains.
    Everyone was writing the same boilerplate. LangServe standardizes it:
    - Consistent API format across all LangChain deployments
    - Built-in streaming support (critical for LLM responses)
    - Auto-generated playground for testing without Postman/curl
    - Input/output schema validation via Pydantic (already in your chain)

PRODUCTION USE CASES:
    1. RAG API — Your React frontend sends questions, gets AI answers
    2. Translation service — Any app can call your translation chain
    3. Multi-chain server — Multiple chains on different paths (/rag, /chat, /translate)
    4. Microservice — One LangServe service per AI capability

HOW TO RUN THIS FILE:
    $ cd LangCGS/src/LangchainBasics/LangServe
    $ python 01_langserve_server.py

    Then open:
    http://127.0.0.1:8000/docs              → FastAPI Swagger UI
    http://127.0.0.1:8000/translate/playground → LangServe Playground (interactive!)
    http://127.0.0.1:8000/essay/playground    → Essay chain playground

Reference:
    https://python.langchain.com/docs/langserve/

Author: GenAI Learner
Date: 2026-04-13
"""

import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langserve import add_routes

# ─── Environment & Logging Setup ────────────────────────────────────────────
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
# 1️⃣  LLM Setup — Using Groq (our free API)
# ═══════════════════════════════════════════════════════════════════════════════

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3,
    max_tokens=512,
)
logger.info("LLM initialized: Groq llama-3.1-8b-instant")

# Output parser — converts AIMessage to plain string
parser = StrOutputParser()


# ═══════════════════════════════════════════════════════════════════════════════
# 2️⃣  Chain 1: Translation Chain
# ═══════════════════════════════════════════════════════════════════════════════
#
# This is the chain from the original notebook — translates text to any language.
# Variables: {language} and {text}
#
# When deployed via LangServe, the API expects:
#   POST /translate/invoke
#   Body: {"input": {"language": "French", "text": "Hello"}}

translate_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a professional translator. Translate the following into {language}:"),
    ("human", "{text}"),
])

# LCEL chain: prompt → llm → parser
translate_chain = translate_prompt | llm | parser


# ═══════════════════════════════════════════════════════════════════════════════
# 3️⃣  Chain 2: Essay Writer Chain
# ═══════════════════════════════════════════════════════════════════════════════
#
# A second chain to show that you can deploy MULTIPLE chains on one server.
# Each chain gets its own path and its own playground.

essay_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert essay writer. Write a concise, well-structured essay "
     "on the given topic. Keep it under 200 words."),
    ("human", "Write an essay about: {topic}"),
])

essay_chain = essay_prompt | llm | parser


# ═══════════════════════════════════════════════════════════════════════════════
# 4️⃣  Chain 3: Code Explainer Chain (GenAI use case)
# ═══════════════════════════════════════════════════════════════════════════════

code_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a senior software engineer. Explain the given code snippet "
     "in simple terms. Mention what it does, key concepts used, and any "
     "potential improvements. Be concise."),
    ("human", "Explain this code:\n```\n{code}\n```"),
])

code_chain = code_prompt | llm | parser


# ═══════════════════════════════════════════════════════════════════════════════
# 5️⃣  FastAPI App + LangServe Routes
# ═══════════════════════════════════════════════════════════════════════════════
#
# This is where the magic happens. We create a FastAPI app and use
# add_routes() to expose each chain as a full REST API.
#
# add_routes() auto-generates for EACH chain:
#   POST /path/invoke        → chain.invoke(input)
#   POST /path/batch         → chain.batch([input1, input2, ...])
#   POST /path/stream        → chain.stream(input) (Server-Sent Events)
#   POST /path/stream_log    → chain.stream(input) with intermediate logs
#   GET  /path/playground    → Interactive web UI for testing
#   GET  /path/input_schema  → Pydantic JSON schema of expected input
#   GET  /path/output_schema → Pydantic JSON schema of output

app = FastAPI(
    title="LangServe Demo Server",
    version="1.0.0",
    description=(
        "A multi-chain LangServe server demonstrating how to deploy "
        "LangChain LCEL chains as production REST APIs. "
        "Visit /translate/playground, /essay/playground, or /code/playground "
        "for interactive testing."
    ),
)


# ── Add chain routes ─────────────────────────────────────────────────────
# Each add_routes() call creates a full set of endpoints under the given path.
# This is the ONE LINE that replaces writing all the FastAPI boilerplate.

add_routes(app, translate_chain, path="/translate")
# Now available:
#   POST /translate/invoke   → {"input": {"language": "French", "text": "Hello"}}
#   GET  /translate/playground → Interactive UI

add_routes(app, essay_chain, path="/essay")
# POST /essay/invoke → {"input": {"topic": "Artificial Intelligence"}}

add_routes(app, code_chain, path="/code")
# POST /code/invoke → {"input": {"code": "def hello(): print('hi')"}}

logger.info("LangServe routes added: /translate, /essay, /code")


# ── Regular FastAPI endpoint (works alongside LangServe) ─────────────────
# You can mix regular FastAPI endpoints with LangServe routes.
# This is useful for health checks, metadata, etc.

@app.get("/")
def root():
    """Health check + API info."""
    return {
        "status": "running",
        "server": "LangServe Demo",
        "chains": {
            "translate": "/translate/playground",
            "essay": "/essay/playground",
            "code": "/code/playground",
        },
        "docs": "/docs",
    }


@app.get("/health")
def health_check():
    """Health check endpoint — used by load balancers and monitoring."""
    return {"status": "healthy"}


# ═══════════════════════════════════════════════════════════════════════════════
# 🏭 PRODUCTION TIPS
# ═══════════════════════════════════════════════════════════════════════════════
#
# 1. CORS — Allow your React frontend to call this API:
#    from fastapi.middleware.cors import CORSMiddleware
#    app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000"], ...)
#
# 2. RATE LIMITING — Prevent abuse (Important-Rules.md says this is a must):
#    Use slowapi or a custom middleware to limit requests per IP.
#
# 3. AUTH — Protect your chains:
#    Add a dependency to verify API keys or JWT tokens.
#    add_routes(app, chain, path="/chain", dependencies=[Depends(verify_token)])
#
# 4. MULTIPLE MODELS — Deploy different chains for different models:
#    add_routes(app, groq_chain, path="/groq")
#    add_routes(app, openai_chain, path="/openai")
#
# 5. DEPLOYMENT:
#    Dev:  python 01_langserve_server.py (uses uvicorn.run())
#    Prod: gunicorn 01_langserve_server:app -w 4 -k uvicorn.workers.UvicornWorker


# ═══════════════════════════════════════════════════════════════════════════════
# 🏭 INTERVIEW QUESTIONS — LangServe
# ═══════════════════════════════════════════════════════════════════════════════
#
# Q1: What is LangServe?
# A:  A library that deploys LangChain LCEL chains as REST APIs with one line.
#     Built on FastAPI. Auto-generates invoke, batch, stream, and playground
#     endpoints for any chain.
#
# Q2: What endpoints does add_routes() create?
# A:  /invoke (single run), /batch (multiple inputs), /stream (SSE streaming),
#     /stream_log (streaming with logs), /playground (interactive UI),
#     /input_schema, /output_schema (Pydantic schemas).
#
# Q3: How does LangServe handle streaming?
# A:  Uses Server-Sent Events (SSE) via the /stream endpoint. The client
#     receives tokens as they're generated, enabling real-time UI updates.
#
# Q4: Can you add authentication to LangServe?
# A:  Yes. Since it's FastAPI, use Depends() with a security dependency.
#     Pass it via the dependencies parameter in add_routes().
#
# Q5: How is LangServe different from just writing a FastAPI endpoint?
# A:  LangServe auto-generates 7+ endpoints per chain, handles streaming,
#     provides a playground UI, and validates input/output schemas. Writing
#     this manually for every chain would be hundreds of lines of boilerplate.
#
# Q6: Can you deploy multiple chains on one server?
# A:  Yes. Call add_routes() multiple times with different paths.
#     Each chain gets its own set of endpoints and playground.


# ═══════════════════════════════════════════════════════════════════════════════
# 🚀 Run the server
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn

    logger.info("Starting LangServe server...")
    logger.info("Swagger docs: http://127.0.0.1:8000/docs")
    logger.info("Translate playground: http://127.0.0.1:8000/translate/playground")
    logger.info("Essay playground: http://127.0.0.1:8000/essay/playground")
    logger.info("Code playground: http://127.0.0.1:8000/code/playground")

    # host="127.0.0.1" = localhost only (safe for dev)
    # host="0.0.0.0"   = accessible from network (for production/Docker)
    uvicorn.run(app, host="127.0.0.1", port=8000)
