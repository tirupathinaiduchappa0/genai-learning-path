"""
📚 Lesson 3.2 — Data Ingestion: Document Loaders (Production-Grade)

This module covers the complete Data Ingestion pipeline in LangChain.
Data Ingestion is the FIRST step in any RAG pipeline:

    Load → Split → Embed → Store

In this lesson, we focus on the "Load" step — getting data FROM various
sources INTO LangChain Document objects.

Loaders Covered (Production-Relevant):
    1. TextLoader         — Plain text files (.txt, .log, .md)
    2. PyPDFLoader        — PDF documents (research papers, reports)
    3. WebBaseLoader      — Web pages via URL (with BeautifulSoup filtering)
    4. ArxivLoader        — Academic papers from arXiv by paper ID
    5. WikipediaLoader    — Wikipedia articles by search query
    6. UnstructuredXMLLoader — XML files (configs, data exports)
    7. CSVLoader          — CSV/tabular data
    8. JSONLoader         — JSON/JSONL files with jq-style extraction
    9. DirectoryLoader    — Bulk-load entire folders of mixed file types

Each loader returns a list[Document], where Document has:
    - page_content: str   → the actual text
    - metadata: dict      → source, page number, etc.

Reference:
    https://python.langchain.com/docs/integrations/document_loaders/

Author: GenAI Learner
Date: 2026-04-10
"""

import logging
import os
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

# Base path for all sample data files shipped with this lesson
DATA_DIR: Path = Path(__file__).parent / "data"


# ═══════════════════════════════════════════════════════════════════════════════
# 1️⃣  TextLoader — Load plain text files
# ═══════════════════════════════════════════════════════════════════════════════
# When to use: .txt, .log, .md, any plain-text file
# Production tip: Always specify encoding to avoid UnicodeDecodeError on servers

def demo_text_loader() -> None:
    """Load a plain text file and inspect the Document object."""
    from langchain_community.document_loaders import TextLoader

    file_path: str = str(DATA_DIR / "speech.txt")
    logger.info("Loading text file: %s", file_path)

    # encoding="utf-8" prevents issues on Windows/Linux with different defaults
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()  # Returns list[Document]

    logger.info("Loaded %d document(s) from TextLoader", len(documents))
    logger.info("Content preview (first 200 chars): %s", documents[0].page_content[:200])
    logger.info("Metadata: %s", documents[0].metadata)

    # Key takeaway:
    # - TextLoader loads the ENTIRE file as ONE Document
    # - metadata contains {"source": "<file_path>"}
    # - For large files, you MUST split after loading (covered in next lesson)


# ═══════════════════════════════════════════════════════════════════════════════
# 2️⃣  PyPDFLoader — Load PDF documents (page-by-page)
# ═══════════════════════════════════════════════════════════════════════════════
# When to use: Research papers, invoices, reports, contracts
# Production tip: Each page becomes a separate Document with page number in metadata
# Install: pip install pypdf

def demo_pdf_loader() -> None:
    """Load a PDF file — each page becomes a separate Document."""
    from langchain_community.document_loaders import PyPDFLoader

    file_path: str = str(DATA_DIR / "attention.pdf")
    logger.info("Loading PDF file: %s", file_path)

    loader = PyPDFLoader(file_path)
    documents = loader.load()  # Each page = 1 Document

    logger.info("Loaded %d pages from PDF", len(documents))

    # Inspect first page
    first_page = documents[0]
    logger.info("Page 0 content preview: %s", first_page.page_content[:200])
    logger.info("Page 0 metadata: %s", first_page.metadata)
    # metadata = {"source": "attention.pdf", "page": 0}

    # Inspect last page
    last_page = documents[-1]
    logger.info("Last page (%d) preview: %s", last_page.metadata.get("page"), last_page.page_content[:200])

    # Key takeaway:
    # - PyPDFLoader splits by PAGE automatically
    # - metadata["page"] gives you the page number (0-indexed)
    # - Great for citation tracking in RAG ("Answer from page 3")


# ═══════════════════════════════════════════════════════════════════════════════
# 3️⃣  WebBaseLoader — Load web pages via URL
# ═══════════════════════════════════════════════════════════════════════════════
# When to use: Blog posts, documentation pages, news articles
# Production tip: Use BeautifulSoup's SoupStrainer to extract ONLY relevant content
#                 (skip navbars, footers, ads — saves tokens and improves RAG quality)
# Install: pip install beautifulsoup4 lxml

def demo_web_loader() -> None:
    """Load a web page and filter content using BeautifulSoup."""
    import bs4
    from langchain_community.document_loaders import WebBaseLoader

    url: str = "https://lilianweng.github.io/posts/2023-06-23-agent/"
    logger.info("Loading web page: %s", url)

    # SoupStrainer filters HTML BEFORE parsing — much faster than post-filtering
    # Only extract elements with these CSS classes (the actual blog content)
    loader = WebBaseLoader(
        web_paths=(url,),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-title", "post-content", "post-header")
            )
        ),
    )
    documents = loader.load()

    logger.info("Loaded %d document(s) from web", len(documents))
    logger.info("Content preview: %s", documents[0].page_content[:300])
    logger.info("Metadata: %s", documents[0].metadata)
    # metadata = {"source": "<url>", "title": "...", "language": "en"}

    # Key takeaway:
    # - Without SoupStrainer, you get ALL HTML text (nav, footer, scripts = noise)
    # - With SoupStrainer, you get ONLY the content you care about
    # - In production RAG, clean input = better retrieval = better answers


# ═══════════════════════════════════════════════════════════════════════════════
# 4️⃣  ArxivLoader — Load academic papers from arXiv
# ═══════════════════════════════════════════════════════════════════════════════
# When to use: Research paper ingestion, academic RAG systems
# Production tip: Use paper ID for exact match, query string for search
# Install: pip install arxiv pymupdf

def demo_arxiv_loader() -> None:
    """Load an academic paper from arXiv by paper ID."""
    from langchain_community.document_loaders import ArxivLoader

    paper_id: str = "1706.03762"  # "Attention Is All You Need"
    logger.info("Loading arXiv paper: %s", paper_id)

    loader = ArxivLoader(query=paper_id, load_max_docs=2)
    documents = loader.load()

    logger.info("Loaded %d document(s) from arXiv", len(documents))
    logger.info("Content preview: %s", documents[0].page_content[:300])
    logger.info("Metadata keys: %s", list(documents[0].metadata.keys()))
    # metadata includes: Title, Authors, Summary, Published date

    # Key takeaway:
    # - ArxivLoader downloads the PDF and extracts text automatically
    # - load_max_docs controls how many papers to fetch
    # - metadata is rich — title, authors, abstract, published date
    # - Great for building research assistant RAG pipelines


# ═══════════════════════════════════════════════════════════════════════════════
# 5️⃣  WikipediaLoader — Load Wikipedia articles
# ═══════════════════════════════════════════════════════════════════════════════
# When to use: General knowledge RAG, fact-checking, entity enrichment
# Production tip: Combine with a vector store for a knowledge base
# Install: pip install wikipedia

def demo_wikipedia_loader() -> None:
    """Load Wikipedia articles by search query."""
    from langchain_community.document_loaders import WikipediaLoader

    query: str = "Generative AI"
    logger.info("Loading Wikipedia articles for: '%s'", query)

    loader = WikipediaLoader(query=query, load_max_docs=2)
    documents = loader.load()

    logger.info("Loaded %d article(s) from Wikipedia", len(documents))
    for i, doc in enumerate(documents):
        logger.info(
            "Article %d — Title: %s | Content length: %d chars",
            i, doc.metadata.get("title", "N/A"), len(doc.page_content),
        )

    # Key takeaway:
    # - WikipediaLoader searches Wikipedia and returns top matches
    # - load_max_docs limits the number of articles
    # - metadata includes: title, summary, source URL
    # - Useful as a fallback knowledge source in agentic RAG


# ═══════════════════════════════════════════════════════════════════════════════
# 6️⃣  UnstructuredXMLLoader — Load XML files
# ═══════════════════════════════════════════════════════════════════════════════
# When to use: Config files, data exports, SOAP responses, RSS feeds
# Production tip: For structured XML, consider parsing with ElementTree first
#                 and then creating Documents manually for better control
# Install: pip install unstructured lxml

def demo_xml_loader() -> None:
    """Load an XML file into LangChain Documents."""
    from langchain_community.document_loaders import UnstructuredXMLLoader

    file_path: str = str(DATA_DIR / "records.xml")
    logger.info("Loading XML file: %s", file_path)

    loader = UnstructuredXMLLoader(file_path)
    documents = loader.load()

    logger.info("Loaded %d document(s) from XML", len(documents))
    logger.info("Content preview: %s", documents[0].page_content[:300])
    logger.info("Metadata: %s", documents[0].metadata)

    # Key takeaway:
    # - UnstructuredXMLLoader extracts text content from XML tags
    # - For highly structured XML, you may want custom parsing
    # - metadata contains {"source": "<file_path>"}


# ═══════════════════════════════════════════════════════════════════════════════
# 7️⃣  CSVLoader — Load CSV/tabular data
# ═══════════════════════════════════════════════════════════════════════════════
# When to use: Spreadsheets, database exports, analytics data
# Production tip: Each ROW becomes a separate Document
#                 Use source_column to set which column appears in metadata

def demo_csv_loader() -> None:
    """Load a CSV file — each row becomes a Document."""
    from langchain_community.document_loaders.csv_loader import CSVLoader

    # Create a sample CSV for demonstration
    sample_csv: Path = DATA_DIR / "sample_products.csv"
    if not sample_csv.exists():
        sample_csv.write_text(
            "product_id,name,category,price\n"
            "1,Laptop Pro,Electronics,1299.99\n"
            "2,Wireless Mouse,Accessories,29.99\n"
            "3,USB-C Hub,Accessories,49.99\n"
            "4,Monitor 27in,Electronics,399.99\n",
            encoding="utf-8",
        )

    logger.info("Loading CSV file: %s", sample_csv)

    loader = CSVLoader(
        file_path=str(sample_csv),
        source_column="product_id",  # This column value goes into metadata["source"]
    )
    documents = loader.load()

    logger.info("Loaded %d row(s) as Documents from CSV", len(documents))
    for doc in documents[:2]:
        logger.info("Content: %s", doc.page_content)
        logger.info("Metadata: %s", doc.metadata)

    # Key takeaway:
    # - Each CSV row → one Document
    # - page_content = "column1: value1\ncolumn2: value2\n..."
    # - source_column lets you control what goes into metadata["source"]
    # - Great for product catalogs, FAQ databases, etc.


# ═══════════════════════════════════════════════════════════════════════════════
# 8️⃣  JSONLoader — Load JSON/JSONL with jq-style extraction
# ═══════════════════════════════════════════════════════════════════════════════
# When to use: API responses, NoSQL exports, structured data
# Production tip: Use jq_schema to extract exactly the fields you need
# Install: pip install jq

def demo_json_loader() -> None:
    """Load a JSON file using jq-style content extraction."""
    import json

    from langchain_community.document_loaders import JSONLoader

    # Create a sample JSON for demonstration
    sample_json: Path = DATA_DIR / "sample_articles.json"
    if not sample_json.exists():
        data = [
            {"title": "Intro to RAG", "content": "RAG combines retrieval with generation...", "author": "Alice"},
            {"title": "LangChain 101", "content": "LangChain is a framework for LLM apps...", "author": "Bob"},
            {"title": "Vector DBs", "content": "Vector databases store embeddings for similarity search...", "author": "Charlie"},
        ]
        sample_json.write_text(json.dumps(data, indent=2), encoding="utf-8")

    logger.info("Loading JSON file: %s", sample_json)

    # jq_schema tells the loader HOW to extract content from each JSON object
    # ".[].content" means: from the root array, for each item, take the "content" field
    loader = JSONLoader(
        file_path=str(sample_json),
        jq_schema=".[].content",       # Extract only the "content" field
        text_content=False,             # Content is not raw text, it's from JSON
    )
    documents = loader.load()

    logger.info("Loaded %d document(s) from JSON", len(documents))
    for doc in documents:
        logger.info("Content: %s", doc.page_content[:100])
        logger.info("Metadata: %s", doc.metadata)

    # Key takeaway:
    # - jq_schema is powerful — you can extract nested fields, filter, etc.
    # - ".[].content" = array → each item → content field
    # - ".messages[].text" = messages array → each → text field
    # - metadata_func can be used to extract custom metadata from each JSON object


# ═══════════════════════════════════════════════════════════════════════════════
# 9️⃣  DirectoryLoader — Bulk-load an entire folder of mixed files
# ═══════════════════════════════════════════════════════════════════════════════
# When to use: Loading all documents from a folder (mixed PDFs, TXTs, etc.)
# Production tip: Use glob patterns to filter file types
#                 Use show_progress=True for large directories

def demo_directory_loader() -> None:
    """Load all .txt files from a directory at once."""
    from langchain_community.document_loaders import DirectoryLoader, TextLoader

    logger.info("Loading all .txt files from: %s", DATA_DIR)

    loader = DirectoryLoader(
        path=str(DATA_DIR),
        glob="**/*.txt",                # Only load .txt files (recursive)
        loader_cls=TextLoader,          # Use TextLoader for each file
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,             # Progress bar for large dirs
        use_multithreading=True,        # Parallel loading for speed
    )
    documents = loader.load()

    logger.info("Loaded %d document(s) from directory", len(documents))
    for doc in documents:
        logger.info("Source: %s | Length: %d chars", doc.metadata.get("source"), len(doc.page_content))

    # Key takeaway:
    # - DirectoryLoader wraps any other loader and applies it to a folder
    # - glob="**/*.pdf" + loader_cls=PyPDFLoader → load all PDFs
    # - use_multithreading=True speeds up loading for many files
    # - show_progress=True gives a tqdm progress bar
    # - In production, this is how you ingest entire document repositories


# ═══════════════════════════════════════════════════════════════════════════════
# 🏭 Production Pattern: Unified Ingestion Function
# ═══════════════════════════════════════════════════════════════════════════════
# In real projects, you want ONE function that handles any file type.
# This is the pattern used in production RAG pipelines.

def load_document(file_path: str, encoding: str = "utf-8") -> list:
    """
    Universal document loader — auto-detects file type and uses the right loader.

    HOW IT WORKS (step-by-step):
        1. Takes a file path like "data/speech.txt" or "data/attention.pdf"
        2. Uses Python's Path to extract the file extension (.txt, .pdf, etc.)
        3. Looks up the extension in a dictionary (loader_map) to find the right loader
        4. Creates the loader and calls .load() to get Documents

    Args:
        file_path: Path to the file to load.
        encoding: Text encoding (default: utf-8).

    Returns:
        list[Document]: Loaded documents.

    Raises:
        ValueError: If file type is not supported.
        FileNotFoundError: If file does not exist.
    """
    from langchain_community.document_loaders import (
        CSVLoader,
        JSONLoader,
        PyPDFLoader,
        TextLoader,
        UnstructuredXMLLoader,
    )

    # ── Step 1: Convert the string path to a Path object ─────────────────
    # Path is from Python's built-in pathlib module.
    # It gives us powerful methods to work with file paths.
    #
    # Example:
    #   path = Path("C:/projects/data/attention.pdf")
    #   path.exists()  → True/False (does the file exist?)
    #   path.name      → "attention.pdf" (just the filename)
    #   path.stem      → "attention" (filename without extension)
    #   path.suffix    → ".pdf" (JUST the extension, including the dot)
    #   path.parent    → Path("C:/projects/data") (the folder)
    path = Path(file_path)

    # ── Step 2: Check if the file actually exists ────────────────────────
    # In production, files might be missing, moved, or path might be wrong.
    # Always validate before processing.
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # ── Step 3: Extract the file extension ───────────────────────────────
    # path.suffix returns the extension WITH the dot:
    #   Path("speech.txt").suffix      → ".txt"
    #   Path("attention.pdf").suffix   → ".pdf"
    #   Path("records.xml").suffix     → ".xml"
    #   Path("data.CSV").suffix        → ".CSV"
    #
    # .lower() converts it to lowercase so ".CSV" and ".csv" both work.
    # This is how we know WHAT TYPE of file the user gave us.
    extension: str = path.suffix.lower()
    logger.info("Auto-detected file type: %s for file: %s", extension, path.name)

    # ── Step 4: The loader_map dictionary ────────────────────────────────
    # This is the CORE of the universal loader pattern.
    #
    # It's a dictionary where:
    #   KEY   = file extension string (e.g., ".txt", ".pdf")
    #   VALUE = a lambda function that CREATES the right loader
    #
    # WHY lambda? Because we don't want to create ALL loaders upfront.
    # We only want to create the ONE loader we actually need.
    # lambda delays the creation until we call it with ().
    #
    # Think of it like a menu at a restaurant:
    #   ".txt"  → "If you pick this, I'll make a TextLoader for you"
    #   ".pdf"  → "If you pick this, I'll make a PyPDFLoader for you"
    #   ".csv"  → "If you pick this, I'll make a CSVLoader for you"
    #
    # The lambda is NOT executed yet — it's just a recipe waiting to be used.
    loader_map: dict = {
        # Text-based files → TextLoader (with encoding for safety)
        ".txt": lambda: TextLoader(file_path, encoding=encoding),
        ".md": lambda: TextLoader(file_path, encoding=encoding),
        ".log": lambda: TextLoader(file_path, encoding=encoding),

        # PDF files → PyPDFLoader (splits by page automatically)
        ".pdf": lambda: PyPDFLoader(file_path),

        # Tabular data → CSVLoader (each row = one Document)
        ".csv": lambda: CSVLoader(file_path=file_path),

        # XML files → UnstructuredXMLLoader (extracts text from tags)
        ".xml": lambda: UnstructuredXMLLoader(file_path),

        # JSON files → JSONLoader (uses jq to extract content)
        ".json": lambda: JSONLoader(file_path=file_path, jq_schema=".", text_content=False),
    }

    # ── Step 5: Check if we support this file type ───────────────────────
    # If someone passes "data/image.png", extension would be ".png"
    # which is NOT in our loader_map, so we raise a clear error.
    if extension not in loader_map:
        raise ValueError(
            f"Unsupported file type: '{extension}'. "
            f"Supported: {list(loader_map.keys())}"
        )

    # ── Step 6: Create the loader and load the documents ─────────────────
    # loader_map[extension] → gets the lambda function for this extension
    # ()                    → CALLS the lambda, which creates the actual loader
    #
    # EXAMPLE walkthrough for file_path = "data/attention.pdf":
    #   extension = ".pdf"
    #   loader_map[".pdf"] → lambda: PyPDFLoader(file_path)  (the recipe)
    #   loader_map[".pdf"]() → PyPDFLoader("data/attention.pdf")  (the actual loader)
    #
    # EXAMPLE walkthrough for file_path = "data/speech.txt":
    #   extension = ".txt"
    #   loader_map[".txt"] → lambda: TextLoader(file_path, encoding="utf-8")
    #   loader_map[".txt"]() → TextLoader("data/speech.txt", encoding="utf-8")
    loader = loader_map[extension]()

    # ── Step 7: Call .load() to get the list of Documents ────────────────
    # Every LangChain loader has a .load() method that returns list[Document].
    # Each Document has .page_content (the text) and .metadata (source info).
    documents = loader.load()
    logger.info("Loaded %d document(s) from %s", len(documents), path.name)
    return documents


# ═══════════════════════════════════════════════════════════════════════════════
# 🚀 Main — Run all demos
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("📚 DATA INGESTION — DOCUMENT LOADERS LESSON")
    logger.info("=" * 70)

    # --- Local file loaders (always work, no internet needed) ---
    logger.info("\n🔹 1. TextLoader Demo")
    demo_text_loader()

    logger.info("\n🔹 2. PyPDFLoader Demo")
    demo_pdf_loader()

    logger.info("\n🔹 6. UnstructuredXMLLoader Demo")
    demo_xml_loader()

    logger.info("\n🔹 7. CSVLoader Demo")
    demo_csv_loader()

    logger.info("\n🔹 8. JSONLoader Demo")
    demo_json_loader()

    logger.info("\n🔹 9. DirectoryLoader Demo")
    demo_directory_loader()

    # --- Internet-dependent loaders (uncomment when online) ---
    # logger.info("\n🔹 3. WebBaseLoader Demo")
    # demo_web_loader()

    # logger.info("\n🔹 4. ArxivLoader Demo")
    # demo_arxiv_loader()

    # logger.info("\n🔹 5. WikipediaLoader Demo")
    # demo_wikipedia_loader()

    # --- Production pattern ---
    logger.info("\n🏭 Production Pattern: Universal Loader")
    docs = load_document(str(DATA_DIR / "speech.txt"))
    logger.info("Universal loader returned %d doc(s)", len(docs))

    logger.info("\n" + "=" * 70)
    logger.info("✅ Data Ingestion lesson complete!")
    logger.info("Next up: Lesson 3.3 — Text Splitting & Chunking Strategies")
    logger.info("=" * 70)
