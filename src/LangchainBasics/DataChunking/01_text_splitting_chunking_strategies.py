"""
✂️ Lesson 3.3 — Text Splitting & Chunking Strategies (Production-Grade)

This module covers the SECOND step in the RAG pipeline:

    Load → ✂️ Split → Embed → Store

WHY do we split?
    LLMs have a limited context window (e.g., GPT-4 = 128K tokens).
    But even if a document FITS, stuffing everything in wastes tokens and
    reduces retrieval quality. Splitting into smaller, semantically meaningful
    chunks gives the vector store better "search targets" to match against.

GOLDEN RULE:
    Each chunk should contain ONE coherent idea/topic.
    Too small = loses context. Too large = dilutes relevance.

Splitters Covered:
    1. RecursiveCharacterTextSplitter — THE default for generic text (most used)
    2. CharacterTextSplitter          — Simple single-separator splitting
    3. HTMLHeaderTextSplitter          — Structure-aware HTML splitting
    4. RecursiveJsonSplitter           — JSON-aware splitting (keeps structure)

Key Concepts:
    - split_documents() vs create_documents() vs split_text()
    - chunk_size & chunk_overlap tuning
    - How separators work and their priority order
    - Production pattern: universal chunking function

Reference:
    https://python.langchain.com/docs/how_to/#text-splitters

Author: GenAI Learner
Date: 2026-04-10
"""

import json
import logging
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# ─── Environment & Logging Setup ────────────────────────────────────────────
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Base path for sample data files (reusing from DataIngestion lesson)
DATA_DIR: Path = Path(__file__).parent.parent / "DataIngestion" / "data"


# ═══════════════════════════════════════════════════════════════════════════════
# 📖 CORE CONCEPT: split_documents() vs create_documents() vs split_text()
# ═══════════════════════════════════════════════════════════════════════════════
#
# LangChain splitters have THREE methods. This confuses everyone at first.
# Here's the simple rule:
#
# ┌─────────────────────┬──────────────────────┬──────────────────────────────┐
# │ Method              │ Input                │ Output                       │
# ├─────────────────────┼──────────────────────┼──────────────────────────────┤
# │ split_documents()   │ list[Document]       │ list[Document] (with meta)   │
# │ create_documents()  │ list[str] (raw text) │ list[Document] (new docs)    │
# │ split_text()        │ str (single string)  │ list[str] (plain strings)    │
# └─────────────────────┴──────────────────────┴──────────────────────────────┘
#
# WHEN TO USE WHICH:
#
# 1. split_documents() — You already loaded files with a Loader (PyPDFLoader, etc.)
#    The loader gave you list[Document] with metadata (source, page, etc.)
#    split_documents() PRESERVES that metadata on every chunk.
#    → This is what you use 90% of the time in production.
#
# 2. create_documents() — You have raw strings (not Document objects).
#    Maybe you read a file with open(), or got text from an API.
#    create_documents() wraps each chunk into a Document object for you.
#    → Use when you have raw text and need Document objects.
#
# 3. split_text() — You just want plain strings back, no Document wrapper.
#    → Rarely used in RAG pipelines. Useful for debugging or previewing.


# ═══════════════════════════════════════════════════════════════════════════════
# 1️⃣  RecursiveCharacterTextSplitter — THE #1 Splitter for Generic Text
# ═══════════════════════════════════════════════════════════════════════════════
#
# WHY "Recursive"?
#   It tries to split using a LIST of separators, in ORDER of priority:
#     ["\n\n", "\n", " ", ""]
#
#   Step 1: Try splitting on "\n\n" (paragraph breaks) first
#   Step 2: If chunks are still too big, split on "\n" (line breaks)
#   Step 3: If still too big, split on " " (spaces / words)
#   Step 4: Last resort: split on "" (individual characters)
#
#   This "recursive fallback" keeps paragraphs together when possible,
#   then sentences, then words — preserving semantic meaning.
#
# WHY is this the default?
#   Because it works well for ANY text — PDFs, articles, code, docs.
#   It's the one recommended by LangChain for generic use cases.
#
# KEY PARAMETERS:
#   chunk_size    = Maximum characters per chunk (e.g., 1000)
#   chunk_overlap = Characters shared between consecutive chunks (e.g., 200)
#                   Overlap prevents losing context at chunk boundaries.
#                   If a sentence is split across two chunks, the overlap
#                   ensures both chunks have part of that sentence.
#
# PRODUCTION VALUES (from Important-Rules.md):
#   chunk_size=1000, chunk_overlap=200

def demo_recursive_character_splitter_with_documents() -> None:
    """
    Split PDF documents using RecursiveCharacterTextSplitter.

    This demo shows split_documents() — the most common production pattern.
    We load a PDF first (which gives us list[Document] with metadata),
    then split those documents into smaller chunks.
    """
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # ── Step 1: Load the PDF (each page = 1 Document) ───────────────────
    loader = PyPDFLoader(str(DATA_DIR / "attention.pdf"))
    documents = loader.load()
    logger.info("Loaded %d pages from PDF", len(documents))

    # ── Step 2: Create the splitter with production settings ─────────────
    # chunk_size=500   → each chunk will be at most 500 characters
    # chunk_overlap=50 → last 50 chars of chunk N appear at start of chunk N+1
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    # NOTE: We did NOT specify separators — it uses the default:
    # ["\n\n", "\n", " ", ""]
    # This default works great for most text.

    # ── Step 3: Split the documents ──────────────────────────────────────
    # split_documents() takes list[Document] and returns list[Document]
    # Each output Document KEEPS the original metadata (source, page, etc.)
    # plus the chunk's text in page_content
    chunks = text_splitter.split_documents(documents)

    logger.info("Split %d pages into %d chunks", len(documents), len(chunks))
    logger.info("Chunk 0 content (%d chars): %s...", len(chunks[0].page_content), chunks[0].page_content[:150])
    logger.info("Chunk 0 metadata: %s", chunks[0].metadata)
    # metadata still has {"source": "attention.pdf", "page": 0} — preserved!

    logger.info("Chunk 1 content (%d chars): %s...", len(chunks[1].page_content), chunks[1].page_content[:150])


def demo_recursive_character_splitter_with_raw_text() -> None:
    """
    Split raw text strings using create_documents().

    This demo shows what to do when you have plain strings
    (not Document objects) — like text from an API or open().
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # ── Step 1: Read raw text from file ──────────────────────────────────
    speech_path = DATA_DIR / "speech.txt"
    speech: str = speech_path.read_text(encoding="utf-8")
    logger.info("Read raw text: %d characters", len(speech))

    # ── Step 2: Create the splitter ──────────────────────────────────────
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
    )

    # ── Step 3: create_documents() wraps raw strings into Documents ──────
    # Input:  list[str]       → ["The world must be made safe..."]
    # Output: list[Document]  → [Document(page_content="The world...", metadata={})]
    #
    # WHY list[str]? Because you might have multiple raw texts to split.
    # Each string is split independently, then all chunks are combined.
    documents = text_splitter.create_documents([speech])

    logger.info("Created %d Document chunks from raw text", len(documents))
    logger.info("Chunk 0: %s", documents[0].page_content)
    logger.info("Chunk 1: %s", documents[1].page_content)
    # Notice: metadata is empty {} because raw text has no source info


# ═══════════════════════════════════════════════════════════════════════════════
# 2️⃣  CharacterTextSplitter — Simple Single-Separator Splitting
# ═══════════════════════════════════════════════════════════════════════════════
#
# HOW IT DIFFERS FROM RecursiveCharacterTextSplitter:
#
# ┌──────────────────────────────┬────────────────────────────────────────────┐
# │ RecursiveCharacterTextSplitter│ CharacterTextSplitter                     │
# ├──────────────────────────────┼────────────────────────────────────────────┤
# │ Tries MULTIPLE separators    │ Uses only ONE separator                   │
# │ Falls back: \n\n → \n → " "  │ Splits ONLY on the one you specify        │
# │ Better semantic preservation │ Simpler but less intelligent              │
# │ ✅ Use this 90% of the time  │ Use when you KNOW the exact separator     │
# └──────────────────────────────┴────────────────────────────────────────────┘
#
# WHEN TO USE CharacterTextSplitter:
#   - Log files split by newlines
#   - CSV-like text split by specific delimiters
#   - Text where you KNOW paragraphs are separated by "\n\n"
#
# In production, RecursiveCharacterTextSplitter is almost always preferred.

def demo_character_text_splitter() -> None:
    """
    Split documents using CharacterTextSplitter with a single separator.

    Shows both split_documents() and create_documents() approaches.
    """
    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import CharacterTextSplitter

    # ── Approach 1: split_documents() — from loaded Documents ────────────
    logger.info("--- CharacterTextSplitter with split_documents() ---")

    loader = TextLoader(str(DATA_DIR / "speech.txt"), encoding="utf-8")
    docs = loader.load()

    # separator="\n\n" means: ONLY split on double-newlines (paragraph breaks)
    # If a paragraph is bigger than chunk_size, it stays as one chunk
    # (unlike Recursive, which would fall back to \n, then " ", then "")
    text_splitter = CharacterTextSplitter(
        separator="\n\n",    # Split ONLY on paragraph breaks
        chunk_size=100,
        chunk_overlap=20,
    )
    chunks = text_splitter.split_documents(docs)

    logger.info("split_documents() produced %d chunks", len(chunks))
    for i, chunk in enumerate(chunks[:3]):
        logger.info("Chunk %d (%d chars): %s...", i, len(chunk.page_content), chunk.page_content[:80])

    # ── Approach 2: create_documents() — from raw strings ────────────────
    logger.info("--- CharacterTextSplitter with create_documents() ---")

    speech: str = (DATA_DIR / "speech.txt").read_text(encoding="utf-8")

    # Default separator is "\n\n" if not specified
    text_splitter_2 = CharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
    )
    # create_documents() takes list[str] and returns list[Document]
    text_docs = text_splitter_2.create_documents([speech])

    logger.info("create_documents() produced %d chunks", len(text_docs))
    logger.info("Chunk 0: %s", text_docs[0].page_content)
    logger.info("Chunk 1: %s", text_docs[1].page_content)


# ═══════════════════════════════════════════════════════════════════════════════
# 3️⃣  HTMLHeaderTextSplitter — Structure-Aware HTML Splitting
# ═══════════════════════════════════════════════════════════════════════════════
#
# WHY is this special?
#   Normal splitters treat HTML as plain text — they don't understand structure.
#   HTMLHeaderTextSplitter is "structure-aware":
#     - It splits text at HTML header boundaries (h1, h2, h3, h4)
#     - It adds the header hierarchy as METADATA on each chunk
#     - So each chunk knows "I belong under Section X > Subsection Y"
#
# WHY does this matter for RAG?
#   When a user asks "What does Section 3.2 say?", the metadata lets you
#   filter chunks by header, giving much more precise retrieval.
#
# TWO WAYS TO USE IT:
#   1. split_text(html_string)      — from an HTML string you already have
#   2. split_text_from_url(url)     — fetches a URL and splits the HTML
#
# PRODUCTION TIP:
#   Combine HTMLHeaderTextSplitter with RecursiveCharacterTextSplitter:
#   First split by headers (semantic sections), then split large sections
#   into smaller chunks by character count.

def demo_html_header_splitter() -> None:
    """
    Split HTML content by header tags, preserving header hierarchy in metadata.

    Shows both split_text() from a string and split_text_from_url() from a URL.
    """
    from langchain_text_splitters import HTMLHeaderTextSplitter

    # ── Example 1: Split an HTML string ──────────────────────────────────
    logger.info("--- HTMLHeaderTextSplitter from HTML string ---")

    html_string = """\
<!DOCTYPE html>
<html>
<body>
    <div>
        <h1>Machine Learning Basics</h1>
        <p>Machine learning is a subset of artificial intelligence.</p>
        <p>It allows systems to learn from data without being explicitly programmed.</p>
        <div>
            <h2>Supervised Learning</h2>
            <p>Supervised learning uses labeled training data.</p>
            <p>Common algorithms: Linear Regression, Decision Trees, SVM.</p>
            <h3>Classification</h3>
            <p>Classification predicts discrete labels (spam/not spam).</p>
            <h3>Regression</h3>
            <p>Regression predicts continuous values (house prices).</p>
        </div>
        <div>
            <h2>Unsupervised Learning</h2>
            <p>Unsupervised learning finds patterns in unlabeled data.</p>
            <p>Common algorithms: K-Means, DBSCAN, PCA.</p>
        </div>
    </div>
</body>
</html>"""

    # Define WHICH headers to split on and what to call them in metadata
    # Each tuple = (html_tag, metadata_key_name)
    # The splitter will split text whenever it encounters these tags
    headers_to_split_on: list[tuple[str, str]] = [
        ("h1", "Header 1"),   # Split on <h1> tags, store as "Header 1" in metadata
        ("h2", "Header 2"),   # Split on <h2> tags, store as "Header 2" in metadata
        ("h3", "Header 3"),   # Split on <h3> tags, store as "Header 3" in metadata
    ]

    html_splitter = HTMLHeaderTextSplitter(headers_to_split_on)

    # split_text() takes an HTML string and returns list[Document]
    # Each Document's metadata contains the header hierarchy
    html_chunks = html_splitter.split_text(html_string)

    logger.info("Split HTML into %d chunks", len(html_chunks))
    for i, chunk in enumerate(html_chunks):
        logger.info(
            "Chunk %d | Metadata: %s | Content: %s...",
            i, chunk.metadata, chunk.page_content[:80],
        )
    # Example output:
    # Chunk 0 | Metadata: {"Header 1": "Machine Learning Basics"}
    #         | Content: "Machine learning is a subset of artificial intelligence..."
    # Chunk 1 | Metadata: {"Header 1": "Machine Learning Basics", "Header 2": "Supervised Learning"}
    #         | Content: "Supervised learning uses labeled training data..."
    # Chunk 2 | Metadata: {"Header 1": "...", "Header 2": "Supervised Learning", "Header 3": "Classification"}
    #         | Content: "Classification predicts discrete labels..."

    # ── Example 2: Split from a URL (uncomment when online) ──────────────
    # logger.info("--- HTMLHeaderTextSplitter from URL ---")
    # url = "https://plato.stanford.edu/entries/goedel/"
    # headers_to_split_on_url = [
    #     ("h1", "Header 1"),
    #     ("h2", "Header 2"),
    #     ("h4", "Header 4"),
    # ]
    # html_splitter_url = HTMLHeaderTextSplitter(headers_to_split_on_url)
    # url_chunks = html_splitter_url.split_text_from_url(url)
    # logger.info("Split URL into %d chunks", len(url_chunks))
    # for chunk in url_chunks[:3]:
    #     logger.info("Metadata: %s | Content: %s...", chunk.metadata, chunk.page_content[:80])


# ═══════════════════════════════════════════════════════════════════════════════
# 4️⃣  RecursiveJsonSplitter — JSON-Aware Splitting (The Tricky One)
# ═══════════════════════════════════════════════════════════════════════════════
#
# WHY IS THIS NEEDED?
#   Normal text splitters would break JSON structure. Imagine splitting this:
#     {"name": "Alice", "age": 30, "skills": ["Python",
#   If you split here ↑, you get INVALID JSON. The vector store would have
#   garbage data. RecursiveJsonSplitter understands JSON structure and splits
#   at valid boundaries (between keys, between array items).
#
# HOW IT WORKS (step-by-step):
#   1. Takes a JSON object (Python dict/list, NOT a string)
#   2. Walks through the JSON tree recursively
#   3. Tries to keep related keys together in one chunk
#   4. If a nested object is too big, it splits WITHIN that object
#   5. Each chunk is a VALID JSON object (never broken syntax)
#
# KEY PARAMETER:
#   max_chunk_size = Maximum SIZE of each JSON chunk (in characters)
#                    The splitter measures the JSON string length of each chunk
#                    and splits if it exceeds this limit.
#
# THREE METHODS (same pattern as text splitters):
#
# ┌─────────────────────┬──────────────────────┬──────────────────────────────┐
# │ Method              │ Input                │ Output                       │
# ├─────────────────────┼──────────────────────┼──────────────────────────────┤
# │ split_json()        │ dict (JSON object)   │ list[dict] (JSON chunks)     │
# │ create_documents()  │ texts=[dict]         │ list[Document] (for RAG)     │
# │ split_text()        │ dict (JSON object)   │ list[str] (JSON strings)     │
# └─────────────────────┴──────────────────────┴──────────────────────────────┘
#
# WHEN TO USE:
#   - Splitting large API responses (OpenAPI specs, REST API data)
#   - Splitting NoSQL database exports
#   - Splitting configuration files
#   - Any time you need to chunk JSON while keeping it VALID

def demo_recursive_json_splitter() -> None:
    """
    Split a large JSON object into smaller valid JSON chunks.

    This demo uses a sample JSON that mimics a real-world API response.
    We show all three methods: split_json(), create_documents(), split_text().
    """
    from langchain_text_splitters import RecursiveJsonSplitter

    # ── Step 1: Create a sample JSON (mimicking an API response) ─────────
    # In the original notebook, this was fetched from LangChain's OpenAPI spec:
    #   json_data = requests.get("https://api.smith.langchain.com/openapi.json").json()
    # Here we use a local example so it works offline.

    json_data: dict = {
        "openapi": "3.0.0",
        "info": {
            "title": "GenAI Product API",
            "version": "1.0.0",
            "description": "API for managing AI-powered products and services. "
                           "This API provides endpoints for product search, "
                           "recommendation engines, and inventory management.",
        },
        "paths": {
            "/products": {
                "get": {
                    "summary": "List all products",
                    "description": "Returns a paginated list of all available products "
                                   "with filtering options for category, price range, "
                                   "and availability status.",
                    "parameters": [
                        {"name": "category", "in": "query", "type": "string"},
                        {"name": "min_price", "in": "query", "type": "number"},
                        {"name": "max_price", "in": "query", "type": "number"},
                        {"name": "page", "in": "query", "type": "integer"},
                    ],
                    "responses": {
                        "200": {"description": "Successful response with product list"},
                        "400": {"description": "Invalid query parameters"},
                        "500": {"description": "Internal server error"},
                    },
                },
                "post": {
                    "summary": "Create a new product",
                    "description": "Creates a new product entry in the catalog. "
                                   "Requires authentication and admin privileges.",
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "price": {"type": "number"},
                                        "category": {"type": "string"},
                                        "description": {"type": "string"},
                                    },
                                },
                            },
                        },
                    },
                },
            },
            "/products/{id}": {
                "get": {
                    "summary": "Get product by ID",
                    "description": "Returns detailed information about a specific product "
                                   "including pricing history, reviews, and related items.",
                    "parameters": [
                        {"name": "id", "in": "path", "type": "string", "required": True},
                    ],
                },
                "put": {
                    "summary": "Update a product",
                    "description": "Updates an existing product. Partial updates supported.",
                },
                "delete": {
                    "summary": "Delete a product",
                    "description": "Soft-deletes a product from the catalog.",
                },
            },
            "/recommendations": {
                "get": {
                    "summary": "Get AI recommendations",
                    "description": "Returns AI-powered product recommendations based on "
                                   "user purchase history and browsing behavior.",
                },
            },
        },
    }

    logger.info("Original JSON size: %d characters", len(json.dumps(json_data)))

    # ── Step 2: Create the JSON splitter ─────────────────────────────────
    # max_chunk_size=300 means each chunk should be at most ~300 characters
    # when serialized back to a JSON string.
    json_splitter = RecursiveJsonSplitter(max_chunk_size=300)

    # ── Step 3: split_json() — Returns list[dict] ────────────────────────
    # This is the most intuitive method.
    # Input:  one big dict (your JSON data)
    # Output: list of smaller dicts, each is VALID JSON
    #
    # HOW IT DECIDES WHERE TO SPLIT:
    #   - It walks the JSON tree top-down
    #   - At each level, it checks: "Is this subtree < max_chunk_size?"
    #   - If YES → keep it as one chunk
    #   - If NO  → go deeper and split the children
    #   - It tries to keep sibling keys together when possible
    #
    # EXAMPLE: If the "/products" path object is 800 chars but max is 300,
    #   it will split "get" and "post" into separate chunks.
    #   Each chunk still has the parent path context.
    json_chunks: list[dict] = json_splitter.split_json(json_data=json_data)

    logger.info("split_json() produced %d chunks", len(json_chunks))
    for i, chunk in enumerate(json_chunks):
        chunk_str = json.dumps(chunk, indent=2)
        logger.info(
            "Chunk %d (%d chars): %s...",
            i, len(chunk_str), chunk_str[:120],
        )

    # ── Step 4: create_documents() — Returns list[Document] ──────────────
    # This is what you use in a RAG pipeline.
    # It converts each JSON chunk into a LangChain Document object.
    #
    # IMPORTANT: The parameter is texts=[json_data], NOT texts=json_data
    #   texts expects a LIST of JSON objects to split.
    #   Even if you have just one JSON, wrap it in a list: [json_data]
    #
    # Each Document's page_content will be the JSON chunk as a string.
    # metadata will contain {"source": "...", "seq_num": 1, 2, 3...}
    docs = json_splitter.create_documents(texts=[json_data])

    logger.info("create_documents() produced %d Documents", len(docs))
    for doc in docs[:3]:
        logger.info("Document content preview: %s...", doc.page_content[:100])
        logger.info("Document metadata: %s", doc.metadata)

    # ── Step 5: split_text() — Returns list[str] ────────────────────────
    # Returns plain JSON strings (not dicts, not Documents).
    # Each string is a valid JSON that can be parsed back with json.loads().
    # Useful for debugging or when you need raw strings.
    text_chunks: list[str] = json_splitter.split_text(json_data=json_data)

    logger.info("split_text() produced %d text chunks", len(text_chunks))
    logger.info("Text chunk 0: %s", text_chunks[0][:150])
    if len(text_chunks) > 1:
        logger.info("Text chunk 1: %s", text_chunks[1][:150])


# ═══════════════════════════════════════════════════════════════════════════════
# 🏭 Production Pattern: Universal Chunking Function
# ═══════════════════════════════════════════════════════════════════════════════
# In production RAG pipelines, you want ONE function that:
#   1. Takes Documents from ANY loader
#   2. Applies the right splitting strategy
#   3. Returns chunks ready for embedding
#
# This is the pattern used in real-world applications.

def chunk_documents(
    documents: list,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    splitter_type: str = "recursive",
) -> list:
    """
    Universal chunking function — splits documents using the specified strategy.

    In production, RecursiveCharacterTextSplitter is the default because it
    works well for any text type. Use other splitters only when you have a
    specific reason (HTML structure, JSON data, etc.).

    Args:
        documents: list[Document] from any LangChain loader.
        chunk_size: Maximum characters per chunk (default: 1000).
                    Recommended range: 500-1500 for most RAG use cases.
        chunk_overlap: Characters shared between chunks (default: 200).
                       Recommended: 10-20% of chunk_size.
        splitter_type: Which splitter to use.
                       "recursive" (default) — RecursiveCharacterTextSplitter
                       "character"           — CharacterTextSplitter

    Returns:
        list[Document]: Chunked documents with preserved metadata.

    Example:
        # Load → Chunk in two lines
        docs = PyPDFLoader("report.pdf").load()
        chunks = chunk_documents(docs, chunk_size=1000, chunk_overlap=200)
    """
    from langchain_text_splitters import (
        CharacterTextSplitter,
        RecursiveCharacterTextSplitter,
    )

    # ── Pick the right splitter based on splitter_type ───────────────────
    # In 90% of cases, "recursive" is the right choice.
    # "character" is only for when you KNOW the exact separator.
    splitter_map: dict = {
        "recursive": lambda: RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            # Default separators: ["\n\n", "\n", " ", ""]
            # These work for most text types (PDFs, articles, docs)
        ),
        "character": lambda: CharacterTextSplitter(
            separator="\n\n",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        ),
    }

    if splitter_type not in splitter_map:
        raise ValueError(
            f"Unknown splitter_type: '{splitter_type}'. "
            f"Supported: {list(splitter_map.keys())}"
        )

    splitter = splitter_map[splitter_type]()
    chunks = splitter.split_documents(documents)

    logger.info(
        "Chunked %d documents into %d chunks (size=%d, overlap=%d, type=%s)",
        len(documents), len(chunks), chunk_size, chunk_overlap, splitter_type,
    )
    return chunks


# ═══════════════════════════════════════════════════════════════════════════════
# 🚀 Main — Run all demos
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("✂️  TEXT SPLITTING & CHUNKING STRATEGIES LESSON")
    logger.info("=" * 70)

    logger.info("\n🔹 1a. RecursiveCharacterTextSplitter — split_documents() (PDF)")
    demo_recursive_character_splitter_with_documents()

    logger.info("\n🔹 1b. RecursiveCharacterTextSplitter — create_documents() (raw text)")
    demo_recursive_character_splitter_with_raw_text()

    logger.info("\n🔹 2. CharacterTextSplitter — split_documents() & create_documents()")
    demo_character_text_splitter()

    logger.info("\n🔹 3. HTMLHeaderTextSplitter — structure-aware HTML splitting")
    demo_html_header_splitter()

    logger.info("\n🔹 4. RecursiveJsonSplitter — JSON-aware splitting (detailed)")
    demo_recursive_json_splitter()

    # ── Production pattern demo ──────────────────────────────────────────
    logger.info("\n🏭 Production Pattern: Universal Chunking")
    from langchain_community.document_loaders import PyPDFLoader

    docs = PyPDFLoader(str(DATA_DIR / "attention.pdf")).load()
    chunks = chunk_documents(docs, chunk_size=1000, chunk_overlap=200)
    logger.info("Production chunking: %d pages → %d chunks", len(docs), len(chunks))

    logger.info("\n" + "=" * 70)
    logger.info("✅ Text Splitting lesson complete!")
    logger.info("Next up: Embeddings & Vector Stores")
    logger.info("=" * 70)
