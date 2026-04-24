"""
LangGraph RAG - Lesson 1: Agentic RAG (Level 5+)

CONCEPT: Agentic RAG - ONE-LINE DEFINITION

Agentic RAG is a Retrieval-Augmented Generation pattern where an
LLM AGENT autonomously DECIDES which knowledge source to query,
EVALUATES the retrieved documents for relevance, and can REWRITE
the query if the results are not good enough - all within a
LangGraph state machine with conditional routing.

WHY AGENTIC RAG EXISTS (the problem it solves)

Standard RAG is a FIXED pipeline: query -> retrieve -> generate.
It has serious limitations in production:

    Problem 1: SINGLE KNOWLEDGE SOURCE
        Basic RAG queries ONE vector store. But real systems have
        MULTIPLE sources: PDFs, web docs, databases, APIs.
        The system needs to DECIDE which source to query.

    Problem 2: NO QUALITY CHECK
        Basic RAG blindly feeds retrieved chunks to the LLM.
        If the chunks are IRRELEVANT, the LLM hallucinates.
        There is no validation step between retrieval and generation.

    Problem 3: NO SELF-CORRECTION
        If the query is poorly worded, basic RAG returns bad results.
        There is no mechanism to REWRITE the query and retry.

    Problem 4: NO ROUTING INTELLIGENCE
        Basic RAG doesn't know WHEN to retrieve vs WHEN to answer
        directly. Some questions don't need retrieval at all.

Agentic RAG solves ALL of these by making the RAG pipeline an AGENT:

    1. TOOL SELECTION: The LLM decides WHICH retriever tool to call
       (PDF knowledge base, web docs, or no tool at all).
    2. RELEVANCE GRADING: After retrieval, a grader checks if the
       documents are actually relevant to the question.
    3. QUERY REWRITING: If documents are irrelevant, the agent
       rewrites the query and retries retrieval.
    4. CONDITIONAL GENERATION: The agent only generates an answer
       when it has RELEVANT context.

HOW AGENTIC RAG DIFFERS FROM BASIC RAG

    Basic RAG (LangChain LCEL):
        query -> retriever -> prompt + context -> LLM -> answer
        FIXED pipeline. No decisions. No quality checks.
        One retriever. One shot. Hope for the best.

    Agentic RAG (LangGraph):
        query -> AGENT (decides tool) -> retriever tool
              -> GRADER (checks relevance)
              -> if relevant: GENERATE answer
              -> if not relevant: REWRITE query -> loop back to AGENT
        DYNAMIC pipeline. Agent decides. Quality checked. Self-correcting.

REAL-WORLD ANALOGY: A RESEARCH LIBRARIAN

Think of a RESEARCH LIBRARIAN helping you find information:

    1. You ask: "How does the transformer architecture work?"
    2. The librarian DECIDES: "I should check the AI research papers
       section, not the web articles section."
    3. The librarian RETRIEVES relevant papers.
    4. The librarian EVALUATES: "Are these papers actually about
       transformers? Yes, this one is the original paper."
    5. The librarian SUMMARIZES the findings for you.

    But if the papers were NOT relevant:
    6. The librarian REPHRASES your question: "Let me search for
       'self-attention mechanism in neural networks' instead."
    7. The librarian RETRIES the search with the better query.

    This is EXACTLY the Agentic RAG pattern.

REAL-WORLD PRODUCTION USE CASES

    1. ENTERPRISE KNOWLEDGE ASSISTANT
       Multiple knowledge bases: HR policies (PDF), product docs (web),
       internal wiki (API). Agent decides which source to query.

    2. LEGAL RESEARCH SYSTEM
       Case law database + statute database + legal commentary.
       Agent routes queries to the right source, grades relevance.

    3. MEDICAL INFORMATION SYSTEM
       Drug databases + clinical guidelines + research papers.
       Agent retrieves from the right source, validates accuracy.

    4. CUSTOMER SUPPORT BOT
       FAQ database + product manuals + troubleshooting guides.
       Agent picks the right source based on the question type.

    5. CODE DOCUMENTATION ASSISTANT
       API docs + code examples + Stack Overflow answers.
       Agent decides which source has the best answer.

ASCII GRAPH STRUCTURE

    [START]
       |
    [agent]  <- LLM with tools (decides: call tool or answer directly)
       |
       +-- (tool_call?) --> [retrieve]  <- ToolNode executes retrieval
       |                       |
       |                    [grade_documents]  <- Check relevance
       |                       |
       |                       +-- (relevant) --> [generate] --> [END]
       |                       |
       |                       +-- (not relevant) --> [rewrite] --> [agent] (loop)
       |
       +-- (no tool_call) --> [END]  <- Agent answers directly

THIS LESSON'S SETUP

    Tool 1: "research_paper_search"
        Source: "Attention Is All You Need" PDF (the transformer paper).
        Loaded with PyPDFLoader, chunked, embedded with HuggingFace,
        stored in FAISS. Wrapped as a retriever tool.

    Tool 2: "langgraph_docs_search"
        Source: LangGraph documentation URLs.
        Loaded with WebBaseLoader, chunked, embedded with HuggingFace,
        stored in FAISS. Wrapped as a retriever tool.

    The AGENT decides which tool to call based on the question.
    After retrieval, documents are GRADED for relevance.
    If not relevant, the query is REWRITTEN and the agent retries.

LANGCHAIN vs LANGGRAPH: RAG

LangChain : LCEL chain (prompt | retriever | llm | parser).
            Fixed pipeline. One retriever. No quality checks.
            No self-correction. No tool selection.
LangGraph  : StateGraph with agent node, tool node, grader, rewriter.
            Dynamic routing. Multiple retrievers as tools.
            Relevance grading. Query rewriting. Full control.

WHERE IT FITS IN THE LANGGRAPH ARCHITECTURE

    Agentic RAG combines:
        - ReAct Agent pattern (Lesson 07): LLM decides tools
        - Tool Node (Lesson 06): ToolNode executes retriever tools
        - Conditional Edges (Lesson 06): grade -> generate or rewrite
        - Memory (Lesson 08): optional checkpointing for multi-turn

    Concept Linking:
        Agent Node -> calls LLM with bound tools -> decides retrieval
        ToolNode -> wraps retriever tools -> executes retrieval
        tools_condition -> routes -> to ToolNode or END
        grade_documents -> conditional edge -> generate or rewrite
        Rewrite -> loops back -> to agent (self-correction cycle)
        Agentic RAG -> builds on -> ReAct + Tools + Conditional Edges

HOW TO RUN:
    $ python src/RAG/01_agentic_rag.py

    NOTE: First run downloads the HuggingFace embedding model (~90MB).
    Subsequent runs use the cached model.

Author: GenAI Learner
"""

import logging
import os
import time
from typing import Annotated, Literal, Sequence

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.tools.retriever import create_retriever_tool
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Suppress noisy HTTP logs from sub-libraries
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
# STEP 1: BUILD THE KNOWLEDGE BASES (Two Retriever Tools)
# ==============================================================================
#
# We create TWO separate vector stores, each wrapped as a retriever tool.
# The agent will decide which tool to call based on the user's question.
#
# Tool 1: research_paper_search
#   Source: "Attention Is All You Need" PDF (transformer paper)
#   Use case: Questions about transformers, self-attention, architecture
#
# Tool 2: langgraph_docs_search
#   Source: LangGraph documentation (web pages)
#   Use case: Questions about LangGraph, workflows, state graphs
#
# WHY TWO SEPARATE VECTOR STORES?
#   In production, different knowledge domains live in different stores.
#   A legal system has case law DB + statute DB. A support bot has
#   FAQ DB + product manual DB. The agent must CHOOSE the right one.
#
# WHY HuggingFaceEmbeddings (not OpenAI)?
#   Free, runs locally, no API key needed. Uses the
#   "sentence-transformers/all-MiniLM-L6-v2" model (~90MB).
#   Good quality for learning. In production, you might use
#   OpenAI text-embedding-3-small for better accuracy.

def build_retriever_tools() -> list:
    """Build two retriever tools: PDF paper + web docs."""

    logger.info("  Building knowledge bases...")

    # -- Shared embedding model (free, local) ----------------------------------
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # -- Shared text splitter --------------------------------------------------
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    # -- Tool 1: Research Paper (PDF) ------------------------------------------
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

    # Build FAISS vector store for the PDF
    pdf_vectorstore = FAISS.from_documents(pdf_splits, embeddings)
    pdf_retriever = pdf_vectorstore.as_retriever(search_kwargs={"k": 3})

    # Wrap as a retriever tool
    # The name and description are CRITICAL - the LLM reads these
    # to decide WHICH tool to call for a given question.
    research_paper_tool = create_retriever_tool(
        pdf_retriever,
        name="research_paper_search",
        description=(
            "Search the 'Attention Is All You Need' research paper. "
            "Use this tool for questions about transformers, self-attention, "
            "multi-head attention, encoder-decoder architecture, positional "
            "encoding, and the original transformer paper."
        ),
    )
    logger.info("  Tool 1 ready: research_paper_search")

    # -- Tool 2: LangGraph Documentation (Web) --------------------------------
    # Load LangGraph documentation pages
    urls = [
        "https://langchain-ai.github.io/langgraph/tutorials/introduction/",
        "https://langchain-ai.github.io/langgraph/tutorials/workflows/",
        "https://langchain-ai.github.io/langgraph/how-tos/map-reduce/",
    ]
    logger.info("  Loading web docs from %d URLs...", len(urls))

    web_docs = []
    for url in urls:
        try:
            loaded = WebBaseLoader(url).load()
            web_docs.extend(loaded)
            logger.info("    Loaded: %s (%d docs)", url.split("/")[-2], len(loaded))
        except Exception as e:
            logger.warning("    Failed to load %s: %s", url, str(e)[:80])

    web_splits = text_splitter.split_documents(web_docs)
    logger.info("  Web doc chunks: %d", len(web_splits))

    # Build FAISS vector store for web docs
    web_vectorstore = FAISS.from_documents(web_splits, embeddings)
    web_retriever = web_vectorstore.as_retriever(search_kwargs={"k": 3})

    # Wrap as a retriever tool
    langgraph_docs_tool = create_retriever_tool(
        web_retriever,
        name="langgraph_docs_search",
        description=(
            "Search the LangGraph documentation. "
            "Use this tool for questions about LangGraph, state graphs, "
            "nodes, edges, workflows, map-reduce, agent patterns, "
            "and LangGraph tutorials."
        ),
    )
    logger.info("  Tool 2 ready: langgraph_docs_search")

    return [research_paper_tool, langgraph_docs_tool]


# ==============================================================================
# STEP 2: BUILD THE AGENTIC RAG GRAPH
# ==============================================================================

def build_agentic_rag_graph(tools: list):
    """Build the Agentic RAG graph with agent, retriever, grader, and generator."""

    # -- State: Messages-based state for the agent -----------------------------
    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]

    # -- LLMs ------------------------------------------------------------------
    # Agent LLM: llama-3.1-8b-instant works reliably with bind_tools on Groq.
    # NOTE: llama-3.3-70b-versatile works for structured_output but can fail
    # with bind_tools (tool_use_failed errors on Groq).
    agent_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_tokens=512)
    gen_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3, max_tokens=1024)
    grade_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, max_tokens=128)

    # Bind tools to the agent LLM
    agent_llm_with_tools = agent_llm.bind_tools(tools)

    # -- Node 1: agent ---------------------------------------------------------
    # The agent receives the question and decides which retriever tool to call.
    # It reads tool descriptions to pick: research_paper_search,
    # langgraph_docs_search, or no tool (answer directly).
    def agent(state: AgentState) -> dict:
        """Agent node: LLM decides which retriever tool to call."""
        logger.info("  [agent] Deciding tool for query...")
        messages = state["messages"]

        response = agent_llm_with_tools.invoke(
            [SystemMessage(content=(
                "You are a helpful research assistant with access to specialized knowledge bases. "
                "Use the research_paper_search tool for questions about transformers, attention mechanisms, "
                "and the 'Attention Is All You Need' paper. "
                "Use the langgraph_docs_search tool for questions about LangGraph, state graphs, and workflows. "
                "If the question is not related to these topics, answer directly without using tools."
            ))]
            + list(messages)
        )

        if response.tool_calls:
            tool_names = [tc["name"] for tc in response.tool_calls]
            logger.info("  [agent] Tool selected: %s", tool_names)
        else:
            logger.info("  [agent] No tool needed, answering directly.")

        return {"messages": [response]}

    # -- Node 2: retrieve (ToolNode) -------------------------------------------
    # Prebuilt node that executes the tool call from the agent's response.
    retrieve_node = ToolNode(tools)

    # -- Node 3: grade_documents (conditional edge) ----------------------------
    # After retrieval, checks if documents are relevant to the question.
    # Returns "generate" if relevant, "rewrite" if not.
    # This is the QUALITY GATE that prevents hallucination from bad context.
    def grade_documents(state: AgentState) -> Literal["generate", "rewrite"]:
        """Grade retrieved documents for relevance."""
        logger.info("  [grader] Checking document relevance...")

        class RelevanceGrade(BaseModel):
            binary_score: Literal["yes", "no"] = Field(
                description="Is the document relevant to the question? 'yes' or 'no'"
            )

        grader = grade_llm.with_structured_output(RelevanceGrade)

        messages = state["messages"]
        question = messages[0].content
        docs = messages[-1].content

        prompt = PromptTemplate(
            template=(
                "You are a grader assessing relevance of a retrieved document to a user question.\n\n"
                "Retrieved document:\n{context}\n\n"
                "User question: {question}\n\n"
                "If the document contains keywords or semantic meaning related to the question, "
                "grade it as relevant. Give a binary score 'yes' or 'no'."
            ),
            input_variables=["context", "question"],
        )

        chain = prompt | grader

        try:
            result = chain.invoke({"question": question, "context": docs})
            score = result.binary_score
        except Exception as e:
            logger.warning("  [grader] Grading failed: %s. Defaulting to 'yes'.", str(e)[:80])
            score = "yes"

        if score == "yes":
            logger.info("  [grader] Documents are RELEVANT -> generate")
            return "generate"
        else:
            logger.info("  [grader] Documents NOT relevant -> rewrite query")
            return "rewrite"

    # -- Node 4: generate ------------------------------------------------------
    # Produces the final answer using the relevant retrieved context.
    # Uses a focused RAG prompt that instructs the LLM to answer
    # based ONLY on the provided context.
    def generate(state: AgentState) -> dict:
        """Generate answer from relevant retrieved documents."""
        logger.info("  [generate] Producing final answer...")

        messages = state["messages"]
        question = messages[0].content
        docs = messages[-1].content

        rag_prompt = PromptTemplate(
            template=(
                "You are an assistant for question-answering tasks. "
                "Use the following retrieved context to answer the question. "
                "If you don't know the answer, say so. Keep the answer concise (3-5 sentences).\n\n"
                "Question: {question}\n\n"
                "Context: {context}\n\n"
                "Answer:"
            ),
            input_variables=["question", "context"],
        )

        chain = rag_prompt | gen_llm | StrOutputParser()
        response = chain.invoke({"question": question, "context": docs})

        logger.info("  [generate] Answer: %s", response[:150])
        return {"messages": [response]}

    # -- Node 5: rewrite -------------------------------------------------------
    # If documents were not relevant, this node rewrites the query
    # to improve retrieval on the next attempt.
    # The rewritten query goes back to the agent node (loop).
    def rewrite(state: AgentState) -> dict:
        """Rewrite the query to improve retrieval results."""
        logger.info("  [rewrite] Transforming query for better retrieval...")

        messages = state["messages"]
        question = messages[0].content

        rewrite_response = gen_llm.invoke(
            [HumanMessage(content=(
                f"Look at this question and try to reason about the underlying semantic intent.\n\n"
                f"Original question: {question}\n\n"
                f"Formulate an improved question that would retrieve better results from a knowledge base:"
            ))]
        )

        logger.info("  [rewrite] Rewritten query: %s", rewrite_response.content[:150])
        return {"messages": [rewrite_response]}

    # -- Build the graph -------------------------------------------------------
    builder = StateGraph(AgentState)

    # Register all nodes
    builder.add_node("agent", agent)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate", generate)
    builder.add_node("rewrite", rewrite)

    # Entry: START -> agent
    builder.add_edge(START, "agent")

    # Agent decides: call a tool (retrieve) or answer directly (END)
    # tools_condition checks if the agent's response has tool_calls.
    # If yes -> route to "retrieve" (ToolNode).
    # If no  -> route to END (agent answered directly).
    builder.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "retrieve", END: END},
    )

    # After retrieval: grade the documents
    # grade_documents returns "generate" or "rewrite"
    builder.add_conditional_edges(
        "retrieve",
        grade_documents,
    )

    # generate -> END (final answer produced)
    builder.add_edge("generate", END)

    # rewrite -> agent (loop back with improved query)
    builder.add_edge("rewrite", "agent")

    # Compile the graph (with recursion_limit for safety against infinite loops)
    graph = builder.compile()

    save_graph_image(graph, "rag01_agentic_rag")

    return graph


# ==============================================================================
# STEP 3: RUN THE AGENTIC RAG PIPELINE
# ==============================================================================

def run_agentic_rag(graph, query: str) -> str:
    """Run a single query through the Agentic RAG pipeline."""
    logger.info("  Query: '%s'", query)

    # Invoke with recursion_limit to prevent infinite rewrite loops
    result = graph.invoke(
        {"messages": [HumanMessage(content=query)]},
        config={"recursion_limit": 15},
    )

    # The final answer is the last message
    final_message = result["messages"][-1]
    # Handle both string and BaseMessage types
    if isinstance(final_message, str):
        answer = final_message
    else:
        answer = final_message.content

    return answer


# ==============================================================================
# COMPARISON TABLE: RAG Patterns
# ==============================================================================
#
# Pattern          | Retrieval        | Quality Check | Self-Correction | Routing
# -----------------|------------------|---------------|-----------------|----------
# Basic RAG        | Fixed retriever  | None          | None            | None
# (LCEL chain)     | Single source    |               |                 |
# -----------------|------------------|---------------|-----------------|----------
# Agentic RAG      | Agent decides    | Relevance     | Query rewrite   | Tool-based
# (THIS lesson)    | Multiple tools   | grading       | + retry loop    | LLM decides
# -----------------|------------------|---------------|-----------------|----------
# Corrective RAG   | Fixed retriever  | Relevance     | Web search      | Grade-based
# (future lesson)  | + web fallback   | grading       | fallback        | If bad -> web
# -----------------|------------------|---------------|-----------------|----------
# Adaptive RAG     | Route first      | Relevance     | Query rewrite   | Classify first
# (future lesson)  | then retrieve    | grading       | + web fallback  | then route
# -----------------|------------------|---------------|-----------------|----------


# ==============================================================================
# COMMON MISTAKES AND GOTCHAS
# ==============================================================================
#
# Mistake                                    | Fix
# -------------------------------------------|--------------------------------------
# Using the same LLM for agent + grading     | Use a capable model (70b) for agent
#                                            | routing and grading. Use a fast model
#                                            | (8b) for generation and rewriting.
# -------------------------------------------|--------------------------------------
# Poor tool descriptions                     | The agent reads tool descriptions to
#                                            | decide which tool to call. Vague
#                                            | descriptions = wrong tool selection.
# -------------------------------------------|--------------------------------------
# No recursion_limit on the rewrite loop     | The rewrite -> agent loop can go
#                                            | infinite. Always set recursion_limit.
# -------------------------------------------|--------------------------------------
# Grading with the generation LLM            | Use structured output (Pydantic) for
#                                            | grading. Don't parse free-text grades.
# -------------------------------------------|--------------------------------------
# Not handling grading failures              | Groq structured output can fail.
#                                            | Always have a fallback (default "yes").
# -------------------------------------------|--------------------------------------
# Using plain dict for state                 | Always use TypedDict with add_messages
#                                            | reducer for message-based agents.
# -------------------------------------------|--------------------------------------
#
# ANTI-PATTERN: Skipping the grading step.
#   Without grading, irrelevant documents go straight to the generator.
#   The LLM hallucinates from bad context. Always grade before generating.


# ==============================================================================
# WHY LANGGRAPH FOR THIS
# ==============================================================================
#
# - Tool-calling agent: LLM decides which retriever to use (bind_tools)
# - ToolNode: prebuilt node that executes retriever tools
# - Conditional edges: grade_documents routes to generate or rewrite
# - Cycles: rewrite -> agent loop enables self-correction
# - Recursion limit: safety guard against infinite loops
# - Composable: add more tools, more grading steps, more nodes
# - Production-ready: add checkpointing, streaming, LangSmith tracing


# ==============================================================================
# WHERE THIS CONNECTS (Concept Linking Map)
# ==============================================================================
#
# Agentic RAG -> uses -> ReAct Agent pattern (agent decides tools)
# Agent Node -> calls -> LLM with bind_tools -> decides retrieval
# ToolNode -> wraps -> retriever tools -> executes retrieval
# tools_condition -> routes -> to ToolNode or END
# grade_documents -> conditional edge -> generate or rewrite
# Rewrite Node -> loops back -> to agent (self-correction cycle)
# create_retriever_tool -> wraps -> FAISS retriever as LangChain tool
# FAISS -> stores -> document embeddings for similarity search
# HuggingFaceEmbeddings -> converts -> text to vectors (free, local)
# Agentic RAG -> foundation for -> Corrective RAG, Adaptive RAG


# ==============================================================================
# INTERVIEW QUESTIONS
# ==============================================================================
#
# THEORETICAL:
# Q: What is the difference between Basic RAG and Agentic RAG?
#    Why would you use Agentic RAG in a production system?
# A: Basic RAG is a fixed pipeline: query -> retrieve -> generate.
#    It uses a single retriever, has no quality checks, and cannot
#    self-correct. Agentic RAG uses an LLM agent that DECIDES which
#    retriever tool to call (from multiple sources), GRADES the
#    retrieved documents for relevance, and can REWRITE the query
#    if documents are not relevant. In production, you use Agentic
#    RAG when you have multiple knowledge sources, need quality
#    assurance on retrieved context, and want self-correction.
#
# HANDS-ON:
# Q: Add a third retriever tool to the Agentic RAG pipeline that
#    searches a Python documentation knowledge base. The agent should
#    route Python-related questions to this new tool.
#
# SOLUTION OUTLINE:
#   1. Load Python docs (from a URL or local file).
#   2. Chunk, embed, store in FAISS.
#   3. create_retriever_tool(retriever, "python_docs_search",
#      "Search Python docs for syntax, stdlib, and best practices.")
#   4. Add to tools list, update agent system prompt, rebuild graph.
#   The grader, generator, and rewriter stay the same.


# ==============================================================================
# QUICK RECAP
# ==============================================================================
#
# 1. Agentic RAG turns the RAG pipeline into an AGENT that DECIDES
#    which knowledge source to query (via bind_tools + tools_condition),
#    GRADES retrieved documents for relevance, and REWRITES the query
#    if results are not good enough (self-correction loop).
#
# 2. The key components are: agent node (tool selection), ToolNode
#    (retrieval execution), grade_documents (quality gate), generate
#    (answer production), and rewrite (query improvement loop).
#
# 3. In production, use Agentic RAG when you have MULTIPLE knowledge
#    sources, need QUALITY ASSURANCE on context, and want SELF-CORRECTION.
#    Always set recursion_limit to prevent infinite rewrite loops.


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("LangGraph RAG - Lesson 1: Agentic RAG")
    logger.info("=" * 70)

    # Step 1: Build the retriever tools (PDF + Web docs)
    logger.info("\n Step 1: Building Knowledge Bases...")
    tools = build_retriever_tools()
    logger.info("  Tools ready: %s", [t.name for t in tools])

    # Step 2: Build the Agentic RAG graph
    logger.info("\n Step 2: Building Agentic RAG Graph...")
    graph = build_agentic_rag_graph(tools)
    logger.info("  Graph compiled and ready.")

    # Step 3: Test with different queries
    logger.info("\n Step 3: Running Queries...")
    logger.info("=" * 70)

    # Query 1: Should route to research_paper_search (PDF tool)
    logger.info("\n--- Query 1: Transformer Architecture (PDF tool) ---")
    answer1 = run_agentic_rag(graph, "What is the transformer architecture and how does self-attention work?")
    logger.info("  Final Answer: %s", answer1[:300])

    # Small delay to avoid Groq rate limits
    time.sleep(5)

    # Query 2: Should route to langgraph_docs_search (Web tool)
    logger.info("\n--- Query 2: LangGraph Concepts (Web tool) ---")
    answer2 = run_agentic_rag(graph, "What is LangGraph and how do state graphs work?")
    logger.info("  Final Answer: %s", answer2[:300])

    time.sleep(5)

    # Query 3: Should answer directly (no tool needed)
    logger.info("\n--- Query 3: General Knowledge (No tool) ---")
    answer3 = run_agentic_rag(graph, "What is the capital of France?")
    logger.info("  Final Answer: %s", answer3[:300])

    logger.info("\n" + "=" * 70)
    logger.info("Lesson 1 complete - Agentic RAG")
    logger.info("The agent routed queries to the right knowledge source,")
    logger.info("graded document relevance, and generated answers.")
    logger.info("Next: Lesson 2 - Corrective RAG (web search fallback)")
    logger.info("=" * 70)
