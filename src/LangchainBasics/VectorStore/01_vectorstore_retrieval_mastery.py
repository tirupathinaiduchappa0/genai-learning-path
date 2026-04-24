"""
🗄️ Lesson 5.1 — Vector Stores, Retrieval & Similarity Search (Production-Grade)

This module covers the FOURTH step in the RAG pipeline:

    Load → Split → Embed → 🗄️ Store & Retrieve

WHAT IS A VECTOR STORE?
    A vector store is a specialized database that stores embeddings (vectors)
    and lets you find the MOST SIMILAR vectors to a given query vector.

    Think of it like this:
        Regular DB:  SELECT * FROM docs WHERE title = 'machine learning'  (exact match)
        Vector Store: "Find me documents whose MEANING is closest to this question"

WHY DO WE NEED VECTOR STORES?
    After embedding your document chunks, you need somewhere to STORE them
    and a way to SEARCH them. Vector stores do both.

VECTOR STORES COVERED:
    1. FAISS     — Facebook's library, in-memory, blazing fast, great for dev
    2. Chroma    — AI-native DB, persistent, great for dev & small production

    PRODUCTION OPTIONS (explained in comments):
    3. Pinecone  — Fully managed cloud vector DB (most popular in production)
    4. Weaviate  — Open-source, self-hosted or cloud
    5. Milvus    — Open-source, built for billion-scale vectors
    6. Qdrant    — Open-source, Rust-based, very fast

SIMILARITY METRICS EXPLAINED:
    - Cosine Similarity    — Measures angle between vectors (most common)
    - Euclidean Distance   — Measures straight-line distance
    - Manhattan Distance   — Measures grid-like distance
    - Dot Product          — Measures alignment (used by some models)

RETRIEVAL METHODS:
    - similarity_search()            — Top-K most similar documents
    - similarity_search_with_score() — Top-K with distance/similarity scores
    - as_retriever()                 — LangChain Retriever interface (for chains)
    - MMR (Maximal Marginal Relevance) — Balances relevance + diversity

PERSISTENCE:
    - FAISS: save_local() / load_local() — saves .faiss + .pkl files
    - Chroma: persist_directory — auto-saves to SQLite on disk
    - What is .pkl (pickle)? — Python's serialization format (explained below)

Reference:
    https://python.langchain.com/docs/integrations/vectorstores/

Author: GenAI Learner
Date: 2026-04-11
"""

import logging
import os
import shutil
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
PERSIST_DIR: Path = Path(__file__).parent / "persist"


# ═══════════════════════════════════════════════════════════════════════════════
# 📐 CORE CONCEPT: Similarity Metrics — How "Closeness" is Measured
# ═══════════════════════════════════════════════════════════════════════════════
#
# When you search a vector store, it compares your query vector against all
# stored vectors. But HOW does it measure "closeness"? There are several metrics:
#
# ┌────────────────────┬──────────────────────────────────────────────────────┐
# │ Metric             │ How it works                                        │
# ├────────────────────┼──────────────────────────────────────────────────────┤
# │ Cosine Similarity  │ Measures the ANGLE between two vectors.             │
# │                    │ Score range: -1 to 1                                │
# │                    │ 1 = identical direction (most similar)              │
# │                    │ 0 = perpendicular (unrelated)                       │
# │                    │ -1 = opposite (most dissimilar)                     │
# │                    │ ✅ MOST COMMON in RAG. Scale-invariant.             │
# │                    │ Used by: Chroma (default), Pinecone, most models    │
# ├────────────────────┼──────────────────────────────────────────────────────┤
# │ Euclidean Distance │ Measures STRAIGHT-LINE distance between vectors.    │
# │ (L2 Distance)      │ Score range: 0 to ∞                                │
# │                    │ 0 = identical (most similar)                        │
# │                    │ Higher = more different                             │
# │                    │ ⚠️ LOWER score = BETTER match (opposite of cosine!) │
# │                    │ Used by: FAISS (default)                            │
# ├────────────────────┼──────────────────────────────────────────────────────┤
# │ Manhattan Distance │ Measures GRID-LIKE distance (sum of absolute diffs) │
# │ (L1 Distance)      │ Like walking city blocks instead of flying straight │
# │                    │ Score range: 0 to ∞                                 │
# │                    │ 0 = identical, Higher = more different              │
# │                    │ Less common in RAG, used in some specialized cases  │
# ├────────────────────┼──────────────────────────────────────────────────────┤
# │ Dot Product        │ Measures ALIGNMENT of vectors.                      │
# │                    │ Higher = more similar                               │
# │                    │ Used when vectors are normalized (unit length)       │
# │                    │ Equivalent to cosine similarity for normalized vecs │
# └────────────────────┴──────────────────────────────────────────────────────┘
#
# THE CONFUSION: "Is higher better or lower better?"
#
#   COSINE SIMILARITY:  Higher is better (1 = perfect match)
#   EUCLIDEAN DISTANCE: Lower is better  (0 = perfect match)
#
#   FAISS returns L2 (Euclidean) distance → LOWER score = better match
#   Chroma returns cosine distance (1 - cosine_similarity) → LOWER = better
#
#   So in BOTH cases when using similarity_search_with_score():
#     → LOWER score = MORE similar document
#
#   But conceptually:
#     Cosine SIMILARITY: closer to 1 = more similar
#     Cosine DISTANCE:   closer to 0 = more similar (distance = 1 - similarity)


# ═══════════════════════════════════════════════════════════════════════════════
# 🔧 Helper: Load and prepare documents + embeddings
# ═══════════════════════════════════════════════════════════════════════════════

def _prepare_chunks_and_embeddings():
    """Load speech.txt, split into chunks, and create embedding model."""
    from langchain_community.document_loaders import TextLoader
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # Load
    loader = TextLoader(str(DATA_DIR / "speech.txt"), encoding="utf-8")
    docs = loader.load()

    # Split
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    logger.info("Prepared %d chunks from speech.txt", len(chunks))

    # Embedding model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    return chunks, embeddings


# ═══════════════════════════════════════════════════════════════════════════════
# 1️⃣  FAISS — Facebook AI Similarity Search
# ═══════════════════════════════════════════════════════════════════════════════
#
# FAISS is a library by Meta (Facebook) for efficient similarity search.
# It's IN-MEMORY — all vectors live in RAM. Blazing fast.
#
# PROS: Extremely fast, supports billions of vectors, battle-tested
# CONS: In-memory (needs RAM), no built-in filtering by metadata
#
# DEFAULT METRIC: L2 (Euclidean) distance → LOWER score = better match
#
# PERSISTENCE:
#   save_local("folder_name") → creates TWO files:
#     - index.faiss  → the actual vector index (binary)
#     - index.pkl    → metadata + document texts (Python pickle format)
#
#   load_local("folder_name", embeddings) → loads both files back
#
# WHAT IS .pkl (PICKLE)?
#   Pickle is Python's built-in serialization format.
#   It converts Python objects (dicts, lists, classes) into bytes
#   that can be saved to disk and loaded back later.
#
#   In FAISS context:
#     - index.faiss = the raw vectors (efficient binary format)
#     - index.pkl   = the Document objects (page_content + metadata)
#
#   ⚠️ SECURITY: Pickle files can execute arbitrary code when loaded.
#   That's why FAISS requires allow_dangerous_deserialization=True.
#   Only load pickle files YOU created or from trusted sources.

def demo_faiss_vectorstore() -> None:
    """
    Full FAISS demo: create, search, save, load.

    Shows similarity_search, similarity_search_with_score, as_retriever,
    and persistence with save_local/load_local.
    """
    from langchain_community.vectorstores import FAISS

    chunks, embeddings = _prepare_chunks_and_embeddings()

    # ── Create the FAISS vector store ────────────────────────────────────
    # from_documents() does THREE things in one call:
    #   1. Embeds all chunk texts using the embedding model
    #   2. Builds the FAISS index with those vectors
    #   3. Stores the Document objects (text + metadata) alongside
    db = FAISS.from_documents(documents=chunks, embedding=embeddings)
    logger.info("Created FAISS vector store with %d vectors", len(chunks))

    # ── Query the vector store ───────────────────────────────────────────
    query: str = "What does the speaker believe about democracy?"

    # --- similarity_search() — Returns top-K most similar Documents ---
    # Default k=4 (returns 4 most similar chunks)
    results = db.similarity_search(query, k=3)
    logger.info("--- similarity_search() (top 3) ---")
    for i, doc in enumerate(results):
        logger.info("Result %d: %s...", i, doc.page_content[:100])

    # --- similarity_search_with_score() — Returns Documents + scores ---
    # FAISS uses L2 (Euclidean) distance: LOWER score = MORE similar
    # A score of 0.0 would mean identical vectors
    results_with_score = db.similarity_search_with_score(query, k=3)
    logger.info("--- similarity_search_with_score() (FAISS = L2 distance) ---")
    for i, (doc, score) in enumerate(results_with_score):
        logger.info(
            "Result %d | L2 Distance: %.4f (lower=better) | Text: %s...",
            i, score, doc.page_content[:80],
        )

    # --- as_retriever() — LangChain Retriever interface ---
    # This wraps the vector store as a Retriever, which can be used
    # in LCEL chains: retriever | prompt | llm | parser
    # invoke() returns list[Document] (same as similarity_search)
    retriever = db.as_retriever(search_kwargs={"k": 2})
    retriever_results = retriever.invoke(query)
    logger.info("--- as_retriever().invoke() ---")
    logger.info("Retriever returned %d documents", len(retriever_results))
    logger.info("Top result: %s...", retriever_results[0].page_content[:100])

    # ── Save to disk (persistence) ───────────────────────────────────────
    # This creates a folder with:
    #   faiss_persist/index.faiss  → vector index (binary)
    #   faiss_persist/index.pkl    → document data (pickle)
    faiss_dir: str = str(PERSIST_DIR / "faiss_index")
    db.save_local(faiss_dir)
    logger.info("--- Saved FAISS to disk: %s ---", faiss_dir)
    logger.info("Files created: index.faiss (vectors) + index.pkl (documents)")

    # ── Load from disk (no re-embedding needed!) ─────────────────────────
    # This is the KEY benefit of persistence:
    #   First time: Load docs → Split → Embed → Save (slow, costs money if using API)
    #   Next time:  Just load_local() → instant! No re-embedding!
    #
    # allow_dangerous_deserialization=True is required because .pkl files
    # use Python's pickle format, which can execute arbitrary code.
    # Only set this to True for files YOU created.
    new_db = FAISS.load_local(
        faiss_dir,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    loaded_results = new_db.similarity_search(query, k=1)
    logger.info("--- Loaded FAISS from disk and searched ---")
    logger.info("Result from loaded DB: %s...", loaded_results[0].page_content[:100])


# ═══════════════════════════════════════════════════════════════════════════════
# 2️⃣  Chroma — AI-Native Vector Database with Persistence
# ═══════════════════════════════════════════════════════════════════════════════
#
# Chroma is an open-source vector database designed specifically for AI apps.
# It stores vectors in a SQLite database on disk (persistent by default).
#
# PROS: Easy to use, persistent, supports metadata filtering, good for dev
# CONS: Not designed for billion-scale (use Pinecone/Milvus for that)
#
# DEFAULT METRIC: Cosine distance (1 - cosine_similarity)
#   → LOWER score = MORE similar (because distance, not similarity)
#   → Score of 0.0 = identical vectors
#   → Score of 1.0 = completely unrelated
#
# PERSISTENCE:
#   persist_directory="./chroma_db" → auto-saves to disk as SQLite
#   Next time, just create Chroma(persist_directory=...) to load it back
#   No save/load methods needed — it's automatic!

def demo_chroma_vectorstore() -> None:
    """
    Full Chroma demo: create, search, persist, reload.

    Shows similarity_search, similarity_search_with_score, as_retriever,
    MMR search, and automatic persistence.
    """
    from langchain_chroma import Chroma

    chunks, embeddings = _prepare_chunks_and_embeddings()

    chroma_dir: str = str(PERSIST_DIR / "chroma_db")

    # ── Create Chroma with persistence ───────────────────────────────────
    # persist_directory tells Chroma WHERE to save the database on disk.
    # It creates a SQLite file + binary data files in that folder.
    # Every time you add documents, they're automatically persisted.
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=chroma_dir,
    )
    logger.info("Created Chroma vector store with %d vectors", len(chunks))
    logger.info("Persisted to: %s", chroma_dir)

    query: str = "What is the duty of the Congress?"

    # --- similarity_search() — Top-K most similar ---
    results = db.similarity_search(query, k=3)
    logger.info("--- Chroma similarity_search() (top 3) ---")
    for i, doc in enumerate(results):
        logger.info("Result %d: %s...", i, doc.page_content[:100])

    # --- similarity_search_with_score() — With cosine distance ---
    # Chroma returns COSINE DISTANCE (not cosine similarity!)
    # Cosine distance = 1 - cosine_similarity
    # So: LOWER score = MORE similar
    #   0.0 = identical, 1.0 = completely different
    results_with_score = db.similarity_search_with_score(query, k=3)
    logger.info("--- Chroma similarity_search_with_score() (cosine distance) ---")
    for i, (doc, score) in enumerate(results_with_score):
        logger.info(
            "Result %d | Cosine Distance: %.4f (lower=better) | Text: %s...",
            i, score, doc.page_content[:80],
        )

    # --- MMR Search (Maximal Marginal Relevance) ---
    # Regular similarity search might return 3 chunks that are all very similar
    # to each other (redundant). MMR balances RELEVANCE with DIVERSITY.
    #
    # lambda_mult controls the balance:
    #   1.0 = pure relevance (same as similarity_search)
    #   0.0 = pure diversity (most different from each other)
    #   0.5 = balanced (recommended)
    #
    # PRODUCTION TIP: Use MMR when you want diverse context for the LLM.
    retriever_mmr = db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "lambda_mult": 0.5},
    )
    mmr_results = retriever_mmr.invoke(query)
    logger.info("--- MMR Retriever (relevance + diversity) ---")
    for i, doc in enumerate(mmr_results):
        logger.info("MMR Result %d: %s...", i, doc.page_content[:100])

    # --- as_retriever() — Standard retriever ---
    retriever = db.as_retriever(search_kwargs={"k": 2})
    retriever_results = retriever.invoke(query)
    logger.info("--- Chroma as_retriever().invoke() ---")
    logger.info("Top result: %s...", retriever_results[0].page_content[:100])

    # ── Load from disk (no re-embedding!) ────────────────────────────────
    # Since Chroma auto-persists, loading is just creating a new Chroma
    # instance pointing to the same directory. No special load method needed.
    logger.info("--- Loading Chroma from disk (no re-embedding) ---")
    db_loaded = Chroma(
        persist_directory=chroma_dir,
        embedding_function=embeddings,
    )
    loaded_results = db_loaded.similarity_search(query, k=1)
    logger.info("Result from loaded Chroma: %s...", loaded_results[0].page_content[:100])


# ═══════════════════════════════════════════════════════════════════════════════
# 📐 DEEP DIVE: How Retrieval Actually Works (Step by Step)
# ═══════════════════════════════════════════════════════════════════════════════
#
# When you call similarity_search("What is democracy?"), here's what happens:
#
# Step 1: EMBED THE QUERY
#   The vector store calls embeddings.embed_query("What is democracy?")
#   This converts your question into a vector: [0.12, -0.34, 0.56, ...]
#
# Step 2: COMPARE AGAINST ALL STORED VECTORS
#   The vector store compares the query vector against every stored vector
#   using the configured distance metric (cosine, L2, etc.)
#
#   For cosine similarity between vectors A and B:
#     cos(A, B) = (A · B) / (||A|| × ||B||)
#     where A · B = sum of element-wise products
#     and ||A|| = square root of sum of squares (magnitude)
#
# Step 3: RANK BY SCORE
#   All documents are ranked by their similarity score.
#   The top-K most similar documents are returned.
#
# Step 4: RETURN DOCUMENTS
#   The original Document objects (page_content + metadata) are returned.
#   The vectors themselves are NOT returned (you don't need them).
#
# IMPORTANT: The embedding model used for QUERYING must be the SAME model
# used for STORING. If you embed documents with all-MiniLM-L6-v2 (384 dims)
# but query with OpenAI (1536 dims), the dimensions won't match and it will fail.


# ═══════════════════════════════════════════════════════════════════════════════
# 🏭 Production: Which Vector Store to Use?
# ═══════════════════════════════════════════════════════════════════════════════
#
# ┌──────────────┬────────────────────────────────────────────────────────────┐
# │ Stage        │ Recommended Vector Store                                  │
# ├──────────────┼────────────────────────────────────────────────────────────┤
# │ Learning/Dev │ FAISS or Chroma (both free, local, easy)                  │
# │ Prototype    │ Chroma (persistent, metadata filtering)                   │
# │ Small Prod   │ Chroma or Qdrant (self-hosted, <1M vectors)              │
# │ Medium Prod  │ Qdrant or Weaviate (self-hosted, 1M-100M vectors)        │
# │ Large Prod   │ Pinecone or Milvus (managed/distributed, 100M+ vectors)  │
# │ Enterprise   │ Pinecone (fully managed, SOC2, zero ops)                 │
# └──────────────┴────────────────────────────────────────────────────────────┘
#
# PRODUCTION VECTOR STORES (commented code for reference):
#
# --- Pinecone (most popular managed vector DB) ---
# from langchain_pinecone import Pinecone
# vectorstore = Pinecone.from_documents(
#     docs, embeddings, index_name="my-index"
# )
# # Pinecone stores data in the cloud — no local files
# # Supports metadata filtering, namespaces, serverless
#
# --- Qdrant (open-source, Rust-based, very fast) ---
# from langchain_qdrant import Qdrant
# vectorstore = Qdrant.from_documents(
#     docs, embeddings, url="http://localhost:6333", collection_name="my_docs"
# )
#
# --- Weaviate (open-source, GraphQL API) ---
# from langchain_weaviate import WeaviateVectorStore
# vectorstore = WeaviateVectorStore.from_documents(
#     docs, embeddings, client=weaviate_client, index_name="MyDocs"
# )
#
# --- Milvus (open-source, built for billion-scale) ---
# from langchain_milvus import Milvus
# vectorstore = Milvus.from_documents(
#     docs, embeddings, connection_args={"host": "localhost", "port": "19530"}
# )


# ═══════════════════════════════════════════════════════════════════════════════
# 📦 DEEP DIVE: Pickle Files & Persistence Explained
# ═══════════════════════════════════════════════════════════════════════════════
#
# WHAT IS PICKLE (.pkl)?
#   Pickle is Python's built-in way to save ANY Python object to a file.
#   It converts objects (dicts, lists, classes, etc.) into bytes (serialization)
#   and can convert them back (deserialization).
#
#   import pickle
#   # Save
#   with open("data.pkl", "wb") as f:
#       pickle.dump(my_object, f)
#   # Load
#   with open("data.pkl", "rb") as f:
#       my_object = pickle.load(f)
#
# WHY DOES FAISS USE PICKLE?
#   FAISS stores two things:
#     1. The vector index (index.faiss) — raw vectors in FAISS's own format
#     2. The document data (index.pkl) — Document objects with text + metadata
#
#   The .pkl file contains the mapping: vector_id → Document(page_content, metadata)
#   Without it, FAISS would have vectors but no way to return the original text.
#
# WHY IS PICKLE "DANGEROUS"?
#   Pickle can execute arbitrary Python code during deserialization.
#   A malicious .pkl file could run harmful code when you load it.
#   That's why FAISS requires allow_dangerous_deserialization=True.
#   Rule: Only load .pkl files YOU created or from TRUSTED sources.
#
# CHROMA vs FAISS PERSISTENCE:
#   FAISS:  Manual — call save_local() / load_local()
#   Chroma: Automatic — just set persist_directory, it handles everything
#
# WHEN TO RE-EMBED vs LOAD FROM DISK:
#   Re-embed when: Documents changed, embedding model changed, first time
#   Load from disk when: Same documents, same model, just restarting the app
#   This saves TIME and MONEY (especially with paid embedding APIs like OpenAI)


# ═══════════════════════════════════════════════════════════════════════════════
# 🏭 Production Pattern: Universal Vector Store Function
# ═══════════════════════════════════════════════════════════════════════════════

def create_or_load_vectorstore(
    chunks: list,
    embeddings,
    store_type: str = "chroma",
    persist_path: str = "./vectorstore_db",
    force_recreate: bool = False,
) -> object:
    """
    Create a new vector store or load an existing one from disk.

    In production, you check if a persisted store exists first.
    If it does, load it (fast, no re-embedding). If not, create it.

    Args:
        chunks: list[Document] — chunked documents to embed and store.
        embeddings: Embedding model instance.
        store_type: "chroma" or "faiss" (default: "chroma").
        persist_path: Where to save/load the vector store.
        force_recreate: If True, always recreate even if persisted store exists.

    Returns:
        Vector store instance (Chroma or FAISS).
    """
    persist = Path(persist_path)

    if store_type == "chroma":
        from langchain_chroma import Chroma

        if persist.exists() and not force_recreate:
            logger.info("Loading existing Chroma from: %s", persist_path)
            return Chroma(persist_directory=persist_path, embedding_function=embeddings)

        logger.info("Creating new Chroma at: %s", persist_path)
        return Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_path,
        )

    elif store_type == "faiss":
        from langchain_community.vectorstores import FAISS

        faiss_file = persist / "index.faiss"
        if faiss_file.exists() and not force_recreate:
            logger.info("Loading existing FAISS from: %s", persist_path)
            return FAISS.load_local(
                persist_path, embeddings, allow_dangerous_deserialization=True
            )

        logger.info("Creating new FAISS at: %s", persist_path)
        db = FAISS.from_documents(documents=chunks, embedding=embeddings)
        db.save_local(persist_path)
        return db

    else:
        raise ValueError(f"Unknown store_type: '{store_type}'. Use 'chroma' or 'faiss'.")


# ═══════════════════════════════════════════════════════════════════════════════
# 🚀 Main — Run all demos
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Clean up any previous persist directories for a fresh run
    if PERSIST_DIR.exists():
        shutil.rmtree(PERSIST_DIR)

    logger.info("=" * 70)
    logger.info("🗄️  VECTOR STORES, RETRIEVAL & SIMILARITY SEARCH LESSON")
    logger.info("=" * 70)

    logger.info("\n🔹 1. FAISS — In-Memory Vector Store")
    demo_faiss_vectorstore()

    logger.info("\n🔹 2. Chroma — Persistent Vector Database")
    demo_chroma_vectorstore()

    # ── Production pattern demo ──────────────────────────────────────────
    logger.info("\n🏭 Production Pattern: create_or_load_vectorstore()")
    chunks, embeddings = _prepare_chunks_and_embeddings()

    prod_path = str(PERSIST_DIR / "production_chroma")
    # First call: creates and persists
    db = create_or_load_vectorstore(chunks, embeddings, persist_path=prod_path)
    results = db.similarity_search("What is the speaker's view on war?", k=2)
    logger.info("First call (create): %s...", results[0].page_content[:80])

    # Second call: loads from disk (no re-embedding!)
    db2 = create_or_load_vectorstore(chunks, embeddings, persist_path=prod_path)
    results2 = db2.similarity_search("What is the speaker's view on war?", k=2)
    logger.info("Second call (load): %s...", results2[0].page_content[:80])

    logger.info("\n" + "=" * 70)
    logger.info("✅ Vector Store & Retrieval lesson complete!")
    logger.info("Next up: RAG Chains — connecting retrieval to LLMs")
    logger.info("=" * 70)
