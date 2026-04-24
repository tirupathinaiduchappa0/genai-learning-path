"""
LangGraph RAG - Lesson 2: Corrective RAG (CRAG) (Level 5+)

CONCEPT: Corrective RAG - ONE-LINE DEFINITION

Corrective RAG (CRAG) is a self-correcting RAG pattern where
retrieved documents are GRADED for relevance, and if any document
is irrelevant, the query is REWRITTEN and a WEB SEARCH is triggered
as a fallback to bring fresh, correct information before generation.

WHY CORRECTIVE RAG EXISTS (the problem it solves)

Basic RAG blindly trusts the retriever. If the vector store returns
irrelevant chunks (stale data, poor embeddings, ambiguous query),
the LLM generates a hallucinated answer from bad context.

Agentic RAG (Lesson 01) solved the TOOL SELECTION problem: the agent
decides WHICH knowledge source to query. But what if the chosen
source returns BAD documents? Agentic RAG rewrites the query and
retries the SAME source. What if that source simply doesn't have
the answer?

Corrective RAG solves this with a DIFFERENT strategy:

    1. RETRIEVE from the local knowledge base (vector store).
    2. GRADE each retrieved document individually for relevance.
    3. If ALL documents are relevant -> GENERATE the answer directly.
    4. If ANY document is irrelevant -> CORRECTION PHASE:
       a. REWRITE the query (transform for better web search).
       b. WEB SEARCH using Tavily (external fallback).
       c. COMBINE local relevant docs + web results.
       d. GENERATE from the enriched context.

KEY DIFFERENCE: Agentic RAG vs Corrective RAG

    Agentic RAG (Lesson 01):
        - Agent DECIDES which tool to call (routing).
        - Grades docs, rewrites query, retries SAME source.
        - Self-correction = retry the same retriever.
        - No external fallback.

    Corrective RAG (THIS lesson):
        - ALWAYS retrieves from local knowledge base first.
        - Grades EACH document individually.
        - If bad docs found -> EXTERNAL fallback (web search).
        - Self-correction = bring NEW information from the web.
        - Combines local + web results for richer context.

    In short:
        Agentic RAG  = "Which source should I ask?"
        Corrective RAG = "Are these docs good? If not, search the web."

REAL-WORLD ANALOGY: A FACT-CHECKER AT A NEWSPAPER

Think of a FACT-CHECKER reviewing an article before publication:

    1. The journalist writes an article using internal archives (retrieval).
    2. The fact-checker REVIEWS each source cited (grading).
    3. If all sources check out -> article goes to print (generate).
    4. If any source is outdated or wrong -> the fact-checker:
       a. Reformulates the question (transform query).
       b. Searches external databases and the web (web search).
       c. Combines verified internal + external sources.
       d. The journalist rewrites with corrected information (generate).

    The fact-checker does NOT just retry the same archives.
    They go OUTSIDE to find better information. That's CRAG.

REAL-WORLD PRODUCTION USE CASES

    1. ENTERPRISE KNOWLEDGE BASE + WEB FALLBACK
       Internal docs may be outdated. CRAG grades them and falls
       back to web search for current information.

    2. LEGAL RESEARCH SYSTEM
       Case law database may miss recent rulings. CRAG detects
       gaps and searches legal databases online.

    3. CUSTOMER SUPPORT WITH LIVE DATA
       FAQ database may not cover new product features. CRAG
       detects irrelevant FAQ answers and searches product docs online.

    4. MEDICAL INFORMATION SYSTEM
       Drug database may lack new drug interactions. CRAG grades
       retrieved info and falls back to PubMed/web for updates.

    5. FINANCIAL ANALYSIS
       Historical data may not reflect current market conditions.
       CRAG detects stale data and searches for live market info.

ASCII GRAPH STRUCTURE

    [START]
       |
    [retrieve]  <- Retrieve from local vector store (PDF knowledge base)
       |
    [grade_documents]  <- Grade EACH document for relevance
       |
       +-- (all relevant) ---------> [generate] --> [END]
       |
       +-- (any irrelevant) -------> [transform_query]  <- Rewrite for web
                                          |
                                     [web_search_node]  <- Tavily web search
                                          |
                                     [generate] --> [END]

    KEY: The correction path (transform -> web search) only triggers
    when the grader finds irrelevant documents. Otherwise, it's
    a straight retrieve -> grade -> generate pipeline.

LANGCHAIN vs LANGGRAPH: Corrective RAG

LangChain : No built-in correction mechanism. You'd have to manually
            chain a grader, then conditionally call web search.
            No graph structure. No conditional routing. Messy.
LangGraph  : StateGraph with conditional edges makes CRAG natural.
            grade_documents -> decide_to_generate -> generate or transform.
            Clean, visual, debuggable, production-ready.

WHERE IT FITS IN THE LANGGRAPH ARCHITECTURE

    Corrective RAG combines:
        - Retrieval (vector store + retriever)
        - Structured Output (Pydantic grading model)
        - Conditional Edges (grade -> generate or transform)
        - Web Search (Tavily as external fallback)
        - Query Rewriting (transform_query node)

    Concept Linking:
        Retrieve Node -> queries -> FAISS vector store
        Grade Documents -> structured output -> binary relevance score
        decide_to_generate -> conditional edge -> generate or transform
        Transform Query -> rewrites -> question for web search
        Web Search Node -> Tavily API -> external fallback
        Generate Node -> RAG prompt -> final answer
        CRAG -> builds on -> Basic RAG + adds correction layer
        CRAG -> different from -> Agentic RAG (no agent, no tool selection)

THIS LESSON'S SETUP

    Knowledge Base: "Attention Is All You Need" PDF (same as Lesson 01).
    Loaded with PyPDFLoader, chunked, embedded with HuggingFace (free),
    stored in FAISS.

    Web Search Fallback: Tavily Search API (free tier, 1000 searches/month).
    Used when local documents are graded as irrelevant.

    LLMs: Groq API (free).
    - llama-3.3-70b-versatile for grading (structured output).
    - llama-3.1-8b-instant for generation and query rewriting.

HOW TO RUN:
    $ python src/RAG/02_corrective_rag.py

    REQUIRES: TAVILY_API_KEY in .env file.

Author: GenAI Learner
"""

import logging
import os
import time
from typing import List

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY", "")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Suppress noisy HTTP logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("langchain_community").setLevel(logging.WARNING)


# -- Helper: Save graph as PNG image ------------------------------------------
GRAPH_DIR = os.path.join(os.path.dirname(__file__), "graphs")
os.makedirs(GRAPH_DIR, exist_ok=True)


def save_graph_image(compiled_graph, name: str) -> None:
    """Save the compiled graph as a PNG image using Mermaid rendering."""
    try:
        png_data = compiled_graph.get_graph(xray=True).draw_mermaid_png()
        filepath = os.path.join(GRAPH_DIR, f"{name}.png")
        with open(filepath, "wb") as f:
            f.write(png_data)
        logger.info("  Graph image saved: %s", filepath)
    except Exception as e:
        logger.warning("  Could not save graph image: %s", e)


# ==============================================================================
# STEP 1: BUILD THE KNOWLEDGE BASE
# ==============================================================================
#
# We load the "Attention Is All You Need" PDF into a FAISS vector store.
# This is the LOCAL knowledge base that CRAG retrieves from first.
# If the retrieved docs are irrelevant, CRAG falls back to web search.

def build_knowledge_base():
    """Build FAISS vector store from the Attention PDF."""
    logger.info("  Building knowledge base...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100,
    )

    # Load the "Attention Is All You Need" PDF
    pdf_path = os.path.join(
        os.path.dirname(__file__),
        "..", "LangchainBasics", "DataIngestion", "data", "attention.pdf",
    )
    logger.info("  Loading PDF: %s", os.path.basename(pdf_path))
    pdf_loader = PyPDFLoader(pdf_path)
    pdf_docs = pdf_loader.load()
    pdf_splits = text_splitter.split_documents(pdf_docs)
    logger.info("  PDF chunks: %d", len(pdf_splits))

    vectorstore = FAISS.from_documents(pdf_splits, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    logger.info("  Knowledge base ready (FAISS + HuggingFace embeddings)")

    return retriever


# ==============================================================================
# STEP 2: BUILD THE CORRECTIVE RAG GRAPH
# ==============================================================================

def build_corrective_rag_graph(retriever):
    """Build the Corrective RAG graph with grading and web search fallback."""

    # -- State -----------------------------------------------------------------
    class GraphState(TypedDict):
        question: str
        documents: List[str]
        generation: str
        web_search: str

    # -- LLMs ------------------------------------------------------------------
    grade_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, max_tokens=128)
    gen_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3, max_tokens=1024)
    rewrite_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_tokens=256)

    # -- Web Search Tool -------------------------------------------------------
    web_search_tool = TavilySearchResults(k=3)

    # -- Grading Chain ---------------------------------------------------------
    class GradeDocuments(BaseModel):
        """Binary score for relevance check on retrieved documents."""
        binary_score: str = Field(
            description="Documents are relevant to the question, 'yes' or 'no'"
        )

    structured_grader = grade_llm.with_structured_output(GradeDocuments)

    grade_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a grader assessing relevance of a retrieved document to a user question. "
            "If the document contains keyword(s) or semantic meaning related to the question, "
            "grade it as relevant. Give a binary score 'yes' or 'no'.",
        ),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ])

    retrieval_grader = grade_prompt | structured_grader

    # -- Question Rewriter Chain -----------------------------------------------
    rewrite_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a question re-writer that converts an input question to a better version "
            "optimized for web search. Look at the input and try to reason about the underlying "
            "semantic intent / meaning.",
        ),
        ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question."),
    ])

    question_rewriter = rewrite_prompt | rewrite_llm | StrOutputParser()

    # -- RAG Generation Chain --------------------------------------------------
    rag_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an assistant for question-answering tasks. Use the following context "
            "to answer the question. If you don't know the answer, say so. "
            "Keep the answer concise (3-5 sentences).",
        ),
        ("human", "Question: {question}\n\nContext: {context}\n\nAnswer:"),
    ])

    rag_chain = rag_prompt | gen_llm | StrOutputParser()

    # -- Node 1: retrieve ------------------------------------------------------
    def retrieve(state: GraphState) -> dict:
        """Retrieve documents from the local knowledge base."""
        logger.info("  [retrieve] Querying local knowledge base...")
        question = state["question"]
        documents = retriever.invoke(question)
        logger.info("  [retrieve] Retrieved %d documents", len(documents))
        return {"documents": documents, "question": question}

    # -- Node 2: grade_documents -----------------------------------------------
    # Grades EACH retrieved document individually for relevance.
    # If ANY document is irrelevant, sets web_search = "Yes".
    # Keeps only the relevant documents in filtered_docs.
    def grade_documents(state: GraphState) -> dict:
        """Grade each retrieved document for relevance."""
        logger.info("  [grade] Grading %d documents...", len(state["documents"]))
        question = state["question"]
        documents = state["documents"]

        filtered_docs = []
        web_search = "No"

        for i, doc in enumerate(documents):
            try:
                score = retrieval_grader.invoke(
                    {"question": question, "document": doc.page_content}
                )
                grade = score.binary_score.lower().strip()
            except Exception as e:
                logger.warning("  [grade] Grading failed for doc %d: %s. Keeping it.", i, str(e)[:60])
                grade = "yes"

            if grade == "yes":
                logger.info("  [grade] Doc %d: RELEVANT", i + 1)
                filtered_docs.append(doc)
            else:
                logger.info("  [grade] Doc %d: NOT RELEVANT -> triggering web search", i + 1)
                web_search = "Yes"

        logger.info(
            "  [grade] Result: %d/%d relevant, web_search=%s",
            len(filtered_docs), len(documents), web_search,
        )
        return {"documents": filtered_docs, "question": question, "web_search": web_search}

    # -- Node 3: transform_query -----------------------------------------------
    # Rewrites the question for better web search results.
    def transform_query(state: GraphState) -> dict:
        """Rewrite the query for better web search results."""
        logger.info("  [transform] Rewriting query for web search...")
        question = state["question"]
        documents = state["documents"]

        better_question = question_rewriter.invoke({"question": question})
        logger.info("  [transform] Original: '%s'", question[:80])
        logger.info("  [transform] Rewritten: '%s'", better_question[:80])
        return {"documents": documents, "question": better_question}

    # -- Node 4: web_search_node -----------------------------------------------
    # Searches the web using Tavily API as an EXTERNAL FALLBACK.
    # This is the "corrective" part of CRAG: when local docs fail,
    # bring fresh information from the web.
    #
    # The web results are APPENDED to the existing relevant docs.
    # So the final context = local relevant docs + web results.
    def web_search_node(state: GraphState) -> dict:
        """Search the web for additional information."""
        logger.info("  [web_search] Searching the web via Tavily...")
        question = state["question"]
        documents = state.get("documents", [])

        try:
            web_results = web_search_tool.invoke({"query": question})
            # Combine web results into a single Document
            # Handle both dict and string response formats from Tavily
            parts = []
            for d in web_results:
                if isinstance(d, dict) and "content" in d:
                    parts.append(d["content"])
                elif isinstance(d, str):
                    parts.append(d)
                else:
                    parts.append(str(d))
            web_content = "\n\n".join(parts)
            web_doc = Document(page_content=web_content)
            documents.append(web_doc)
            logger.info("  [web_search] Found %d web results, appended to documents", len(web_results))
        except Exception as e:
            logger.warning("  [web_search] Web search failed: %s", str(e)[:80])

        return {"documents": documents, "question": question}

    # -- Node 5: generate ------------------------------------------------------
    # Produces the final answer using ALL available context:
    # relevant local docs + web results (if web search was triggered).
    def generate(state: GraphState) -> dict:
        """Generate the final answer from context."""
        logger.info("  [generate] Producing final answer...")
        question = state["question"]
        documents = state["documents"]

        # Format documents for the prompt
        context = "\n\n".join(
            doc.page_content if hasattr(doc, "page_content") else str(doc)
            for doc in documents
        )

        generation = rag_chain.invoke({"question": question, "context": context})
        logger.info("  [generate] Answer: %s", generation[:150])
        return {"documents": documents, "question": question, "generation": generation}

    # ==========================================================================
    # CONDITIONAL EDGE: decide_to_generate
    # ==========================================================================
    # This is the ROUTING LOGIC of CRAG.
    # It reads the web_search flag set by grade_documents:
    #   "No"  -> all docs relevant -> go straight to generate.
    #   "Yes" -> some docs irrelevant -> go to transform_query first.
    def decide_to_generate(state: GraphState) -> str:
        """Route to generate or transform_query based on grading results."""
        web_search = state["web_search"]

        if web_search == "Yes":
            logger.info("  [decide] Some docs irrelevant -> CORRECTION PATH (transform + web search)")
            return "transform_query"
        else:
            logger.info("  [decide] All docs relevant -> DIRECT GENERATION")
            return "generate"

    # ==========================================================================
    # BUILD THE GRAPH
    # ==========================================================================
    builder = StateGraph(GraphState)

    # Register all nodes
    builder.add_node("retrieve", retrieve)
    builder.add_node("grade_documents", grade_documents)
    builder.add_node("generate", generate)
    builder.add_node("transform_query", transform_query)
    builder.add_node("web_search_node", web_search_node)

    # Edges: the CRAG flow
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "grade_documents")

    # Conditional edge: grade -> generate OR grade -> transform
    builder.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        },
    )

    # Correction path: transform -> web search -> generate
    builder.add_edge("transform_query", "web_search_node")
    builder.add_edge("web_search_node", "generate")

    # generate -> END
    builder.add_edge("generate", END)

    # Compile
    graph = builder.compile()
    save_graph_image(graph, "rag02_corrective_rag")

    return graph


# ==============================================================================
# STEP 3: RUN THE CORRECTIVE RAG PIPELINE
# ==============================================================================

def run_corrective_rag(graph, query: str) -> str:
    """Run a single query through the Corrective RAG pipeline."""
    logger.info("  Query: '%s'", query)

    result = graph.invoke({"question": query})

    answer = result.get("generation", "No answer generated.")
    web_used = result.get("web_search", "No")

    logger.info("  Web search used: %s", web_used)
    return answer


# ==============================================================================
# COMPARISON TABLE: RAG Patterns
# ==============================================================================
#
# Pattern          | Retrieval        | Quality Check | Correction       | Routing
# -----------------|------------------|---------------|------------------|----------
# Basic RAG        | Fixed retriever  | None          | None             | None
# (LCEL chain)     | Single source    |               |                  |
# -----------------|------------------|---------------|------------------|----------
# Agentic RAG      | Agent decides    | Relevance     | Query rewrite    | Tool-based
# (Lesson 01)      | Multiple tools   | grading       | Retry same source| LLM decides
# -----------------|------------------|---------------|------------------|----------
# Corrective RAG   | Fixed retriever  | Per-document  | Query rewrite    | Grade-based
# (THIS lesson)    | + web fallback   | grading       | + WEB SEARCH     | If bad -> web
# -----------------|------------------|---------------|------------------|----------
# Adaptive RAG     | Route first      | Per-document  | Query rewrite    | Classify first
# (future lesson)  | then retrieve    | grading       | + web fallback   | then route
# -----------------|------------------|---------------|------------------|----------
#
# KEY INSIGHT:
#   Agentic RAG retries the SAME source with a better query.
#   Corrective RAG goes to an EXTERNAL source (web) when local fails.
#   This makes CRAG more robust for knowledge gaps in the local store.


# ==============================================================================
# COMMON MISTAKES AND GOTCHAS
# ==============================================================================
#
# Mistake                                    | Fix
# -------------------------------------------|--------------------------------------
# Grading the batch instead of each doc      | Grade EACH document individually.
#                                            | Some may be relevant, others not.
#                                            | Keep the good ones, supplement with web.
# -------------------------------------------|--------------------------------------
# Not handling grading failures              | Groq structured output can fail.
#                                            | Always fallback to "yes" (keep doc).
# -------------------------------------------|--------------------------------------
# Discarding ALL docs when one is bad        | Only discard irrelevant docs. Keep
#                                            | relevant ones and ADD web results.
# -------------------------------------------|--------------------------------------
# Using web search for every query           | Web search is a FALLBACK, not default.
#                                            | Only trigger when grading finds issues.
# -------------------------------------------|--------------------------------------
# Not rewriting query before web search      | The original query is optimized for
#                                            | vector search, not web search. Always
#                                            | transform the query first.
# -------------------------------------------|--------------------------------------
# Missing TAVILY_API_KEY in .env             | Tavily requires an API key. Free tier
#                                            | gives 1000 searches/month.
# -------------------------------------------|--------------------------------------
#
# ANTI-PATTERN: Skipping the grading step and always doing web search.
#   This wastes API calls and adds latency. Grade first, correct only when needed.


# ==============================================================================
# WHY LANGGRAPH FOR THIS
# ==============================================================================
#
# - Conditional edges: grade -> generate OR grade -> transform -> web -> generate
# - Structured output: Pydantic model forces binary relevance grading
# - Stateful: documents, question, web_search flag flow through the graph
# - Visual: graph image shows the correction branch clearly
# - Composable: add more correction strategies (e.g., different web sources)
# - Production-ready: add checkpointing, streaming, LangSmith tracing


# ==============================================================================
# WHERE THIS CONNECTS (Concept Linking Map)
# ==============================================================================
#
# Corrective RAG -> uses -> Retrieve + Grade + Conditional Routing
# Retrieve Node -> queries -> FAISS vector store (local knowledge)
# Grade Documents -> structured output -> per-document relevance score
# decide_to_generate -> conditional edge -> generate or transform
# Transform Query -> rewrites -> question optimized for web search
# Web Search Node -> Tavily API -> external fallback for fresh info
# Generate Node -> RAG prompt -> final answer from combined context
# CRAG -> builds on -> Basic RAG + adds correction layer
# CRAG -> different from -> Agentic RAG (no agent, no tool selection)
# CRAG -> foundation for -> Adaptive RAG (adds query classification)


# ==============================================================================
# INTERVIEW QUESTIONS
# ==============================================================================
#
# THEORETICAL:
# Q: What is the difference between Agentic RAG and Corrective RAG?
#    When would you choose one over the other?
# A: Agentic RAG uses an LLM agent to DECIDE which knowledge source
#    to query from multiple tools. If docs are irrelevant, it rewrites
#    the query and retries the SAME source. Corrective RAG ALWAYS
#    retrieves from the local knowledge base first, grades EACH document
#    individually, and falls back to WEB SEARCH when local docs fail.
#    Choose Agentic RAG when you have multiple specialized knowledge
#    sources and need intelligent routing. Choose Corrective RAG when
#    you have one primary knowledge base but need a safety net for
#    knowledge gaps (web search fallback).
#
# HANDS-ON:
# Q: Modify the Corrective RAG pipeline to add a SECOND correction
#    strategy: if web search also returns irrelevant results, fall back
#    to a "I don't know" response instead of hallucinating.
#    Add a second grading step after web_search_node.
#
# SOLUTION OUTLINE:
#   1. Add a grade_web_results node after web_search_node.
#   2. Grade the web results for relevance.
#   3. If relevant -> generate. If not -> return "I don't have enough
#      information to answer this question accurately."
#   4. Add conditional edge: grade_web_results -> generate or no_answer.
#
# BONUS (System Design):
# Q: Design a CRAG system for a customer support platform where the
#    local knowledge base is the product FAQ, and the web fallback
#    searches the company blog and documentation site. How would you
#    handle rate limiting on the web search API? How would you cache
#    web results to avoid redundant searches?


# ==============================================================================
# QUICK RECAP
# ==============================================================================
#
# 1. Corrective RAG adds a SELF-GRADING layer after retrieval. Each
#    document is graded individually for relevance using structured
#    output (Pydantic model with binary "yes"/"no" score).
#
# 2. If ANY document is irrelevant, CRAG triggers a CORRECTION PATH:
#    rewrite the query (transform_query) then search the web (Tavily)
#    as an external fallback. Relevant local docs are KEPT and combined
#    with web results for richer context.
#
# 3. CRAG is different from Agentic RAG: no agent, no tool selection.
#    CRAG always retrieves locally first, then falls back to the web.
#    Use CRAG when your local knowledge base may have gaps.


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("LangGraph RAG - Lesson 2: Corrective RAG (CRAG)")
    logger.info("=" * 70)

    # Step 1: Build the knowledge base
    logger.info("\n Step 1: Building Knowledge Base...")
    retriever = build_knowledge_base()

    # Step 2: Build the CRAG graph
    logger.info("\n Step 2: Building Corrective RAG Graph...")
    graph = build_corrective_rag_graph(retriever)
    logger.info("  Graph compiled and ready.")

    # Step 3: Test queries
    logger.info("\n Step 3: Running Queries...")
    logger.info("=" * 70)

    # Query 1: Should find relevant docs in the PDF (DIRECT path)
    logger.info("\n--- Query 1: In-domain question (local docs should be relevant) ---")
    answer1 = run_corrective_rag(graph, "What is multi-head attention and how does it work in the transformer?")
    logger.info("  Final Answer: %s", answer1[:300])

    time.sleep(3)

    # Query 2: Out-of-domain question (local docs will be irrelevant -> web search)
    logger.info("\n--- Query 2: Out-of-domain question (should trigger web search) ---")
    answer2 = run_corrective_rag(graph, "What are the latest features in LangGraph 2025?")
    logger.info("  Final Answer: %s", answer2[:300])

    time.sleep(3)

    # Query 3: Partially relevant (some docs relevant, some not)
    logger.info("\n--- Query 3: Partially relevant question ---")
    answer3 = run_corrective_rag(graph, "How does the transformer model compare to RNNs for sequence modeling?")
    logger.info("  Final Answer: %s", answer3[:300])

    logger.info("\n" + "=" * 70)
    logger.info("Lesson 2 complete - Corrective RAG (CRAG)")
    logger.info("The pipeline graded documents, triggered web search when needed,")
    logger.info("and generated answers from corrected context.")
    logger.info("Next: Lesson 3 - Adaptive RAG (query classification + routing)")
    logger.info("=" * 70)
