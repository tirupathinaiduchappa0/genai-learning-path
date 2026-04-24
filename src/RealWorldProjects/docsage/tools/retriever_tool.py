"""
DocSage Retriever Tool — Builds FAISS retriever tools from documents and URLs.

WHY THIS FILE EXISTS:
    This is the RAG INGESTION PIPELINE. It takes document files or web URLs,
    processes them into searchable chunks, and wraps each as a LangChain
    retriever tool that the agent can call.

SUPPORTED SOURCES:
    Files:  .pdf, .docx, .txt, .csv, .md
    URLs:   Any public webpage (LinkedIn, GitHub, blogs, docs)

ARCHITECTURE (DRY — Don't Repeat Yourself):
    Both file and URL ingestion share the same core pipeline:
        load -> chunk -> embed -> FAISS -> retriever -> tool

    The ONLY difference is the loading step:
    - Files: detect extension -> pick loader (PyPDFLoader, Docx2txtLoader, etc.)
    - URLs:  WebBaseLoader scrapes the page

    The shared pipeline is in _build_tool_from_documents().
    This avoids code duplication between build_retriever_from_file()
    and build_retriever_from_url().

INTERVIEW POINT:
    "Walk me through your data ingestion pipeline."
    "The system supports multiple document formats and web URLs. The loading
    step is format-specific, but the rest of the pipeline is shared: chunk
    with RecursiveCharacterTextSplitter, embed with HuggingFace, store in
    FAISS, and wrap as a named retriever tool. This DRY design means adding
    a new format is just adding one loader — the pipeline stays the same."
"""

import logging
import os
import tempfile
from typing import Optional

from langchain_community.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    WebBaseLoader,
)
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.tools import BaseTool
from langchain_core.tools.retriever import create_retriever_tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from docsage.config import settings

logger = logging.getLogger(__name__)

# Module-level embedding model — initialized once, reused across all sources.
_embeddings: Optional[HuggingFaceEmbeddings] = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """Get or create the shared HuggingFace embedding model (singleton)."""
    global _embeddings
    if _embeddings is None:
        logger.info("Loading embedding model: %s", settings.EMBEDDING_MODEL)
        _embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
        logger.info("Embedding model loaded.")
    return _embeddings


# ==============================================================================
# SHARED CORE PIPELINE (used by both file and URL ingestion)
# ==============================================================================

def _build_tool_from_documents(
    pages: list[Document],
    tool_name: str,
    description: str,
    source_label: str,
) -> BaseTool:
    """
    Shared pipeline: chunk -> embed -> FAISS -> retriever -> tool.

    Both build_retriever_from_file() and build_retriever_from_url()
    call this after loading their documents. This eliminates duplication.

    Args:
        pages: Loaded Document objects (from any loader).
        tool_name: Sanitized tool name (e.g., "search_attention").
        description: Tool description for the agent.
        source_label: Human-readable source name for logging.

    Returns:
        A LangChain BaseTool wrapping the FAISS retriever.
    """
    # -- Chunk -----------------------------------------------------------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(pages)
    logger.info("  Chunks: %d (size=%d, overlap=%d)",
                len(chunks), settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)

    if not chunks:
        raise ValueError(f"'{source_label}' produced no chunks after splitting.")

    # -- Embed and store in FAISS ----------------------------------------------
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    logger.info("  FAISS index built (%d vectors)", len(chunks))

    # -- Create retriever and wrap as tool -------------------------------------
    retriever = vectorstore.as_retriever(search_kwargs={"k": settings.RETRIEVER_K})

    tool = create_retriever_tool(retriever, name=tool_name, description=description)

    logger.info("  Tool created: name='%s'", tool_name)
    logger.info("  Description: '%s'", description[:100])
    return tool


# ==============================================================================
# PUBLIC API: Build from File
# ==============================================================================

def build_retriever_from_file(
    file_bytes: bytes,
    file_name: str,
    description: str,
) -> BaseTool:
    """
    Build a FAISS retriever tool from an uploaded document file.

    Pipeline: file bytes -> temp file -> detect type -> loader -> [shared pipeline]

    Args:
        file_bytes: Raw bytes of the uploaded file.
        file_name: Original filename (used for type detection and tool name).
        description: Human-readable description. If generic, auto-improved.

    Returns:
        A LangChain BaseTool wrapping the FAISS retriever.
    """
    file_extension = os.path.splitext(file_name)[1].lower()
    logger.info("Building retriever for: '%s' (type: %s)", file_name, file_extension)

    if file_extension not in settings.SUPPORTED_FILE_TYPES:
        supported = ", ".join(settings.SUPPORTED_FILE_TYPES.keys())
        raise ValueError(f"Unsupported file type: '{file_extension}'. Supported: {supported}")

    # -- Load: save to temp file, use format-specific loader -------------------
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        pages = _load_document(tmp_path, file_extension)

        if not pages:
            raise ValueError(f"'{file_name}' produced no content. Is it a valid file?")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    # -- Build tool using shared pipeline --------------------------------------
    tool_name = _sanitize_tool_name(file_name)
    final_description = _improve_description(description, file_name, pages)

    return _build_tool_from_documents(pages, tool_name, final_description, file_name)


# ==============================================================================
# PUBLIC API: Build from URL
# ==============================================================================

def build_retriever_from_url(url: str, description: str = "") -> BaseTool:
    """
    Build a FAISS retriever tool from a web URL.

    Pipeline: URL -> WebBaseLoader -> [shared pipeline]

    Args:
        url: The webpage URL to scrape.
        description: Optional description. If empty, auto-generated from content.

    Returns:
        A LangChain BaseTool wrapping the FAISS retriever.
    """
    logger.info("Building retriever from URL: '%s'", url[:80])

    # -- Load: scrape webpage --------------------------------------------------
    try:
        pages = WebBaseLoader(url).load()
    except Exception as e:
        raise ValueError(f"Failed to load URL '{url}': {str(e)[:100]}")

    if not pages or not any(p.page_content.strip() for p in pages):
        raise ValueError(f"URL '{url}' produced no text content.")

    logger.info("  Loaded %d page(s) from URL", len(pages))

    # -- Build tool using shared pipeline --------------------------------------
    tool_name = _sanitize_url_tool_name(url)

    if not description or len(description.strip()) < 10:
        content_preview = " ".join(pages[0].page_content.split())[:250]
        description = f"Search content from {url}. Contains: {content_preview}..."

    return _build_tool_from_documents(pages, tool_name, description, url)


# ==============================================================================
# PRIVATE HELPERS
# ==============================================================================

def _load_document(file_path: str, file_extension: str) -> list[Document]:
    """Load a document using the appropriate loader based on file type."""
    if file_extension == ".pdf":
        loader = PyPDFLoader(file_path)
    elif file_extension == ".docx":
        loader = Docx2txtLoader(file_path)
    elif file_extension == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    elif file_extension == ".csv":
        loader = CSVLoader(file_path, encoding="utf-8")
    elif file_extension == ".md":
        loader = UnstructuredMarkdownLoader(file_path)
    else:
        supported = ", ".join(settings.SUPPORTED_FILE_TYPES.keys())
        raise ValueError(f"Unsupported file type: '{file_extension}'. Supported: {supported}")

    documents = loader.load()
    logger.info("  Loaded %d document(s) with %s", len(documents), type(loader).__name__)
    return documents


def _improve_description(description: str, file_name: str, pages: list[Document]) -> str:
    """Auto-improve generic descriptions using the document's actual content."""
    generic_prefixes = ["document:", "document :", "file:"]
    is_generic = any(description.lower().strip().startswith(p) for p in generic_prefixes)

    if not is_generic and len(description.strip()) > 20:
        return description

    content_preview = ""
    for page in pages[:2]:
        if page.page_content:
            content_preview += page.page_content + " "
            if len(content_preview) > 300:
                break

    content_preview = " ".join(content_preview.split())[:250]

    if content_preview:
        return f"Search the document '{file_name}'. Contains: {content_preview}..."
    return f"Search the document '{file_name}' for relevant information."


def _sanitize_tool_name(name: str) -> str:
    """Convert a filename into a valid tool name (e.g., 'report.pdf' -> 'search_report')."""
    base = os.path.splitext(name)[0].lower()
    clean = base.replace(" ", "_").replace("-", "_")
    clean = "".join(c for c in clean if c.isalnum() or c == "_")
    while "__" in clean:
        clean = clean.replace("__", "_")
    return f"search_{clean.strip('_')}"


def _sanitize_url_tool_name(url: str) -> str:
    """Convert a URL into a valid tool name (e.g., 'linkedin.com/in/john' -> 'search_linkedin_com_in_john')."""
    clean = url.lower().replace("https://", "").replace("http://", "").replace("www.", "")
    clean = clean.rstrip("/")
    clean = clean.replace(".", "_").replace("/", "_").replace("-", "_")
    clean = "".join(c for c in clean if c.isalnum() or c == "_")
    while "__" in clean:
        clean = clean.replace("__", "_")
    clean = clean.strip("_")
    if len(clean) > 40:
        clean = clean[:40].rstrip("_")
    return f"search_{clean}"


# ==============================================================================
# REFERENCE NOTES (for interview revision)
# ==============================================================================
#
# PyPDFLoader LIMITATIONS:
#   Plain text -> Excellent | Tables -> Flat text (okay) | Images -> Nothing | Scanned -> Fails
#
# INTERVIEW ANSWER for "What about images in PDFs?":
#   "I'd add OCR via UnstructuredLoader with Tesseract, or use GPT-4V
#   to describe images before indexing."
#
# FORMATS NOT ADDED (too complex):
#   JSON (custom schema), Excel (openpyxl), Images (OCR/vision), HTML (messy DOM)
