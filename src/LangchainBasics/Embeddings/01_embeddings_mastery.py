"""
🧮 Lesson 4.1 — Embeddings: Converting Text to Vectors (Production-Grade)

This module covers the THIRD step in the RAG pipeline:

    Load → Split → 🧮 Embed → Store

WHAT ARE EMBEDDINGS?
    Embeddings convert text into dense numerical vectors (lists of floats).
    Each vector captures the MEANING of the text in a high-dimensional space.

    Example:
        "king"  → [0.12, -0.34, 0.56, ..., 0.78]   (1536 dimensions)
        "queen" → [0.11, -0.33, 0.55, ..., 0.77]   (very similar vector!)
        "car"   → [0.89, 0.12, -0.67, ..., 0.01]   (very different vector)

    Similar meanings → similar vectors → close in vector space.
    This is HOW similarity search works in RAG.

WHY DO WE NEED EMBEDDINGS?
    Computers can't compare text directly for meaning.
    "happy" and "joyful" look completely different as strings.
    But as embeddings, they're VERY close in vector space.
    This lets us find relevant documents even when exact words don't match.

KEY CONCEPTS:
    - embed_query()     → Embed a SINGLE text (the user's question)
    - embed_documents() → Embed MULTIPLE texts (your document chunks)
    - Dimensions        → Length of the vector (384, 768, 1024, 1536, 3072)
                          More dimensions = more detail = more storage/compute

EMBEDDING MODELS COVERED:
    1. HuggingFace (all-MiniLM-L6-v2)  — FREE, local, 384 dimensions
    2. Groq                             — FREE API, fast inference
    3. OpenAI (commented)               — Paid, highest quality, 1536/3072 dims
    4. Ollama (commented)               — FREE, local, needs Ollama running

Reference:
    https://python.langchain.com/docs/integrations/text_embedding/

Author: GenAI Learner
Date: 2026-04-11
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# ─── Environment & Logging Setup ────────────────────────────────────────────
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR: Path = Path(__file__).parent.parent / "DataIngestion" / "data"


# ═══════════════════════════════════════════════════════════════════════════════
# 1️⃣  HuggingFace Embeddings — FREE, Local, No API Key Needed
# ═══════════════════════════════════════════════════════════════════════════════
#
# Model: all-MiniLM-L6-v2 (from sentence-transformers)
#   - 384 dimensions (compact, fast)
#   - Runs 100% locally on your machine (no internet after first download)
#   - Good quality for most use cases
#   - First run downloads ~80MB model, then cached locally
#
# PRODUCTION USE:
#   - Great for prototyping and small-to-medium apps
#   - No API costs, no rate limits, no data leaving your machine
#   - For higher quality, use "all-mpnet-base-v2" (768 dims) or
#     "BAAI/bge-large-en-v1.5" (1024 dims)

def demo_huggingface_embeddings() -> None:
    """Demonstrate HuggingFace sentence-transformer embeddings."""
    from langchain_huggingface import HuggingFaceEmbeddings

    logger.info("Initializing HuggingFace embeddings (all-MiniLM-L6-v2)...")

    # ── Create the embedding model ───────────────────────────────────────
    # model_name: which sentence-transformer model to use
    # First call downloads the model (~80MB), subsequent calls use cache
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # ── embed_query() — Embed a SINGLE text ──────────────────────────────
    # Use this for the USER'S QUESTION in a RAG pipeline.
    # Returns: list[float] — one vector
    text: str = "What is machine learning?"
    query_vector: list[float] = embeddings.embed_query(text)

    logger.info("embed_query() result:")
    logger.info("  Input text: '%s'", text)
    logger.info("  Vector dimensions: %d", len(query_vector))
    logger.info("  First 5 values: %s", query_vector[:5])
    # Output: Vector dimensions: 384

    # ── embed_documents() — Embed MULTIPLE texts at once ─────────────────
    # Use this for your DOCUMENT CHUNKS in a RAG pipeline.
    # Returns: list[list[float]] — one vector per document
    #
    # WHY separate methods?
    #   Some models apply different processing for queries vs documents.
    #   For example, some models prepend "query: " or "passage: " internally.
    #   Always use embed_query() for questions and embed_documents() for docs.
    documents: list[str] = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with many layers.",
        "Python is a popular programming language.",
    ]
    doc_vectors: list[list[float]] = embeddings.embed_documents(documents)

    logger.info("embed_documents() result:")
    logger.info("  Number of documents: %d", len(documents))
    logger.info("  Vectors generated: %d", len(doc_vectors))
    logger.info("  Each vector has %d dimensions", len(doc_vectors[0]))

    return embeddings  # Return for use in other demos


# ═══════════════════════════════════════════════════════════════════════════════
# 2️⃣  Groq Embeddings — FREE API, Fast Inference
# ═══════════════════════════════════════════════════════════════════════════════
#
# Groq provides fast LLM inference. We use it here as our PRIMARY embedding
# provider since we have a GROQ_API_KEY in .env.
#
# NOTE: Groq's embedding support depends on available models.
# We use HuggingFace as the reliable default and show Groq for LLM calls.
# For embeddings in this project, HuggingFace is our go-to.

def demo_groq_info() -> None:
    """Explain Groq's role in our stack."""
    groq_key: str = os.getenv("GROQ_API_KEY", "")
    logger.info("Groq API Key found: %s", "Yes" if groq_key else "No")
    logger.info("Groq is primarily used for LLM inference (chat models).")
    logger.info("For embeddings, we use HuggingFace (free, local, reliable).")
    logger.info("Groq will be used in later lessons for chains and agents.")


# ═══════════════════════════════════════════════════════════════════════════════
# 3️⃣  OpenAI Embeddings — Paid, Highest Quality (COMMENTED)
# ═══════════════════════════════════════════════════════════════════════════════
#
# OpenAI offers the best embedding models but requires a paid API key.
# Uncomment when you have an OPENAI_API_KEY in your .env file.
#
# Models:
#   text-embedding-3-small  → 1536 dimensions (cheaper, good quality)
#   text-embedding-3-large  → 3072 dimensions (best quality, more expensive)
#   text-embedding-3-large with dimensions=1024 → reduced to 1024 dims (saves storage)
#
# PRODUCTION TIP:
#   OpenAI embeddings are the industry standard for production RAG.
#   Use text-embedding-3-small for cost efficiency.
#   Use text-embedding-3-large with dimensions=1024 for quality + storage balance.

# def demo_openai_embeddings() -> None:
#     """Demonstrate OpenAI embeddings (requires OPENAI_API_KEY)."""
#     from langchain_openai import OpenAIEmbeddings
#
#     # Default: text-embedding-3-large with 3072 dimensions
#     embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
#     query_vector = embeddings.embed_query("What is machine learning?")
#     logger.info("OpenAI full dimensions: %d", len(query_vector))  # 3072
#
#     # Reduced dimensions: saves storage, slightly less quality
#     embeddings_1024 = OpenAIEmbeddings(
#         model="text-embedding-3-large",
#         dimensions=1024,  # Reduce from 3072 to 1024
#     )
#     query_vector_1024 = embeddings_1024.embed_query("What is machine learning?")
#     logger.info("OpenAI reduced dimensions: %d", len(query_vector_1024))  # 1024


# ═══════════════════════════════════════════════════════════════════════════════
# 4️⃣  Ollama Embeddings — FREE, Local, Needs Ollama Running (COMMENTED)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Ollama runs LLMs locally on your machine. It also supports embedding models.
# Requires: Ollama installed and running (ollama serve)
# Models: gemma:2b, mxbai-embed-large, nomic-embed-text, llama2
#
# PROS: Free, private (data never leaves your machine), no API limits
# CONS: Uses GPU/RAM, needs Ollama installed, slower than API-based
#
# Install: https://ollama.com/download
# Pull model: ollama pull mxbai-embed-large

# def demo_ollama_embeddings() -> None:
#     """Demonstrate Ollama embeddings (requires Ollama running locally)."""
#     from langchain_community.embeddings import OllamaEmbeddings
#
#     # Default model is llama2, but mxbai-embed-large is better for embeddings
#     embeddings = OllamaEmbeddings(model="mxbai-embed-large")
#
#     # embed_query — single text
#     query_vector = embeddings.embed_query("What is deep learning?")
#     logger.info("Ollama dimensions: %d", len(query_vector))
#
#     # embed_documents — multiple texts
#     doc_vectors = embeddings.embed_documents([
#         "Alpha is the first letter of Greek alphabet",
#         "Beta is the second letter of Greek alphabet",
#     ])
#     logger.info("Ollama embedded %d documents", len(doc_vectors))


# ═══════════════════════════════════════════════════════════════════════════════
# 🏭 Production Pattern: Full Pipeline — Load → Split → Embed
# ═══════════════════════════════════════════════════════════════════════════════
# This is how production apps convert documents into embeddings.
# The embeddings are then stored in a vector store (next lesson).

def demo_production_embedding_pipeline() -> None:
    """
    Full production pipeline: Load PDF → Split into chunks → Embed each chunk.

    This is the pattern used in every real-world RAG application.
    After this step, the embeddings go into a vector store for retrieval.
    """
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # ── Step 1: Load ─────────────────────────────────────────────────────
    loader = PyPDFLoader(str(DATA_DIR / "attention.pdf"))
    documents = loader.load()
    logger.info("Step 1 — Loaded %d pages from PDF", len(documents))

    # ── Step 2: Split ────────────────────────────────────────────────────
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    logger.info("Step 2 — Split into %d chunks", len(chunks))

    # ── Step 3: Embed ────────────────────────────────────────────────────
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # In production, you embed ALL chunks at once using embed_documents()
    # This is more efficient than calling embed_query() in a loop
    chunk_texts: list[str] = [chunk.page_content for chunk in chunks]
    vectors: list[list[float]] = embeddings.embed_documents(chunk_texts)

    logger.info("Step 3 — Embedded %d chunks into %d-dimensional vectors", len(vectors), len(vectors[0]))
    logger.info("Each chunk is now a %d-dim vector ready for vector store", len(vectors[0]))

    # ── Preview: What a single chunk + its vector looks like ─────────────
    logger.info("--- Sample chunk → vector ---")
    logger.info("Chunk text (first 100 chars): %s...", chunks[0].page_content[:100])
    logger.info("Vector (first 5 dims): %s", vectors[0][:5])
    logger.info("Metadata preserved: %s", chunks[0].metadata)

    # NEXT STEP: Store these vectors in a vector store (Chroma, FAISS, etc.)
    # That's covered in the VectorStore lesson.


# ═══════════════════════════════════════════════════════════════════════════════
# 🚀 Main — Run all demos
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("🧮 EMBEDDINGS MASTERY LESSON")
    logger.info("=" * 70)

    logger.info("\n🔹 1. HuggingFace Embeddings (FREE, local)")
    demo_huggingface_embeddings()

    logger.info("\n🔹 2. Groq Info")
    demo_groq_info()

    logger.info("\n🔹 3. OpenAI Embeddings — COMMENTED (needs API key)")
    logger.info("   Uncomment demo_openai_embeddings() when you have OPENAI_API_KEY")

    logger.info("\n🔹 4. Ollama Embeddings — COMMENTED (needs Ollama running)")
    logger.info("   Uncomment demo_ollama_embeddings() when Ollama is installed")

    logger.info("\n🏭 Production Pipeline: Load → Split → Embed")
    demo_production_embedding_pipeline()

    logger.info("\n" + "=" * 70)
    logger.info("✅ Embeddings lesson complete!")
    logger.info("Next up: Vector Stores & Retrieval")
    logger.info("=" * 70)
