"""
DocSage Configuration — All project constants and defaults in one place.

WHY THIS FILE EXISTS:
    In production, you NEVER scatter configuration across files.
    All constants, model names, UI labels, and defaults live here.
    If you need to change a model name or chunk size, you change ONE file.

INTERVIEW POINT:
    "How do you manage configuration in your project?"
    "I centralize all config in a settings module. Model names, chunk sizes,
    UI labels, and defaults are all in one place. No magic strings scattered
    across the codebase. This makes it easy to swap models, adjust parameters,
    or add new LLM providers without touching business logic."
"""


# ==============================================================================
# PAGE / UI CONFIGURATION
# ==============================================================================

PAGE_TITLE: str = "Enterprise Document Intelligence Agent"
PAGE_ICON: str = "📄"
PAGE_LAYOUT: str = "wide"


# ==============================================================================
# LLM CONFIGURATION
# ==============================================================================

# Available Groq models for the user to select in the sidebar.
#
# WHY SEPARATE MODELS FOR DIFFERENT TASKS:
#   - Agent/Routing: needs tool-calling capability. llama-3.1-8b-instant
#     works reliably with bind_tools on Groq.
#   - Grading/Validation: needs structured output (Pydantic). llama-3.3-70b
#     is the most reliable for with_structured_output on Groq.
#   - Generation: needs good text quality. llama-3.1-8b-instant is fast
#     and produces good answers for RAG.

GROQ_MODEL_OPTIONS: list[str] = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "gemma2-9b-it",
]

# Default model assignments per task
DEFAULT_AGENT_MODEL: str = "llama-3.1-8b-instant"
DEFAULT_GRADING_MODEL: str = "llama-3.3-70b-versatile"
DEFAULT_GENERATION_MODEL: str = "llama-3.1-8b-instant"


# ==============================================================================
# RAG / RETRIEVAL CONFIGURATION
# ==============================================================================

# Chunking parameters for PDF ingestion.
CHUNK_SIZE: int = 1000
CHUNK_OVERLAP: int = 200

# Embedding model — HuggingFace (free, runs locally).
EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

# Number of documents to retrieve per query.
RETRIEVER_K: int = 4

# Supported file types for document upload.
# Maps file extension to a human-readable label for the UI.
SUPPORTED_FILE_TYPES: dict[str, str] = {
    ".pdf": "PDF Document",
    ".docx": "Word Document",
    ".txt": "Text File",
    ".csv": "CSV Spreadsheet",
    ".md": "Markdown File",
}

# Tavily web search — max results when falling back to web.
WEB_SEARCH_MAX_RESULTS: int = 3


# ==============================================================================
# GRAPH CONFIGURATION
# ==============================================================================

# Recursion limit prevents infinite self-correction loops.
RECURSION_LIMIT: int = 20

# Max tokens for different LLM tasks.
AGENT_MAX_TOKENS: int = 512
GRADING_MAX_TOKENS: int = 128
GENERATION_MAX_TOKENS: int = 1024
REWRITE_MAX_TOKENS: int = 256


# ==============================================================================
# EMAIL CONFIGURATION (Gmail SMTP)
# ==============================================================================
#
# Used by the send_email tool. The agent can send answers via email
# when the user requests it (e.g., "Email this to john@company.com").
#
# SENDER credentials come from .env (your Gmail account).
# RECIPIENT is extracted from the user's message by the agent.
#
# HOW TO SET UP:
#   1. Enable 2-Step Verification on your Google Account.
#   2. Go to https://myaccount.google.com/apppasswords
#   3. Generate an App Password for "Mail".
#   4. Add to .env: GMAIL_ADDRESS=you@gmail.com
#                   GMAIL_APP_PASSWORD=xxxx xxxx xxxx xxxx

GMAIL_SMTP_SERVER: str = "smtp.gmail.com"
GMAIL_SMTP_PORT: int = 587
