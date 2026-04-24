"""
LangGraph RAG - Lesson 3: Adaptive RAG (Level 6)

CONCEPT: Adaptive RAG - ONE-LINE DEFINITION

Adaptive RAG dynamically ROUTES the query to the best strategy
(vector store OR web search), GRADES retrieved documents, GENERATES
an answer, then VALIDATES the answer for hallucination and relevance
before returning it - with a self-correction loop if validation fails.

WHY ADAPTIVE RAG EXISTS (the problem it solves)

Agentic RAG (Lesson 01) solved TOOL SELECTION: the agent picks the source.
Corrective RAG (Lesson 02) solved BAD RETRIEVAL: grade docs, fall back to web.

But NEITHER validates the GENERATED ANSWER itself. What if:
    - The retriever returns relevant docs, but the LLM hallucinates anyway?
    - The answer is grounded in docs but doesn't actually answer the question?
    - A simple question gets routed through expensive retrieval unnecessarily?

Adaptive RAG solves ALL of these by adding THREE new capabilities:

    1. QUERY ROUTING (front-gate):
       Classify the query FIRST. Route to vectorstore OR web search.
       Simple/general questions go to web. Domain questions go to vectorstore.
       This saves time and cost by picking the right strategy upfront.

    2. HALLUCINATION GRADING (post-generation check):
       After generating an answer, check: "Is this answer grounded in
       the retrieved documents?" If not, regenerate.

    3. ANSWER GRADING (post-generation check):
       After hallucination check passes, check: "Does this answer
       actually address the user's question?" If not, rewrite query and retry.

HOW ADAPTIVE RAG UNIFIES ALL THREE PATTERNS

    Basic RAG:     retrieve -> generate (no checks)
    Agentic RAG:   agent decides tool -> retrieve -> grade docs -> generate
    Corrective RAG: retrieve -> grade docs -> (web fallback) -> generate
    Adaptive RAG:   route query -> retrieve/web -> grade docs -> generate
                    -> check hallucination -> check answer -> (self-correct)

    Adaptive RAG = Routing + CRAG + Post-Generation Validation

REAL-WORLD ANALOGY: A SENIOR RESEARCH ANALYST

Think of a SENIOR RESEARCH ANALYST at a consulting firm:

    1. Client asks a question.
    2. The analyst CLASSIFIES the question:
       - "What's the weather?" -> Quick web search (no deep research).
       - "How does transformer attention work?" -> Internal research DB.
    3. The analyst RETRIEVES relevant documents.
    4. The analyst GRADES each document: "Is this actually relevant?"
    5. The analyst WRITES a report (generates answer).
    6. The analyst SELF-REVIEWS:
       a. "Is my report grounded in the sources?" (hallucination check)
       b. "Does my report actually answer the client's question?" (answer check)
    7. If either check fails -> REWRITE the approach and try again.

    This multi-layered quality assurance is what makes Adaptive RAG
    the most robust RAG pattern.

ASCII GRAPH STRUCTURE

    [START]
       |
    (route_question)  <- Classify: vectorstore or web_search
       |
       +-- (web_search) --> [web_search] --> [generate]
       |                                        |
       +-- (vectorstore) --> [retrieve]         |
                                |               |
                          [grade_documents]     |
                                |               |
                    +-----------+-----------+   |
                    |                       |   |
              (has relevant)          (all filtered)
                    |                       |
              [generate] <--+         [transform_query]
                    |       |               |
         (grade_generation) |          [retrieve] (retry)
                    |       |
           +--------+--------+
           |        |        |
        (useful) (not     (not
           |    supported) useful)
           |        |        |
         [END]  [generate] [transform_query]
                (retry)    (rewrite + retry)

LANGCHAIN vs LANGGRAPH: Adaptive RAG

LangChain : Impossible to build cleanly. You'd need nested if/else,
            manual loops, and no graph structure. Unmaintainable.
LangGraph  : StateGraph with conditional edges at EVERY decision point.
            Route at START, grade after retrieve, validate after generate.
            Clean, visual, debuggable, production-ready.

THIS LESSON'S SETUP

    Knowledge Base: "Attention Is All You Need" PDF.
    Web Search: Tavily API (free tier).
    LLMs: Groq API (free).
    - llama-3.3-70b-versatile for routing, grading (structured output).
    - llama-3.1-8b-instant for generation and query rewriting.

    The router knows the vectorstore contains information about
    transformers, attention mechanisms, and neural network architecture.
    Questions outside this domain are routed to web search.

HOW TO RUN:
    $ python src/RAG/03_adaptive_rag.py

    REQUIRES: TAVILY_API_KEY in .env file.

Author: GenAI Learner
"""

import logging
import os
import time
from typing import List, Literal

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
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

def build_knowledge_base():
    """Build FAISS vector store from the Attention PDF."""
    logger.info("  Building knowledge base...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100,
    )

    pdf_path = os.path.join(
        os.path.dirname(__file__),
        "..", "LangchainBasics", "DataIngestion", "data", "attention.pdf",
    )
    logger.info("  Loading PDF: %s", os.path.basename(pdf_path))
    pdf_docs = PyPDFLoader(pdf_path).load()
    pdf_splits = text_splitter.split_documents(pdf_docs)
    logger.info("  PDF chunks: %d", len(pdf_splits))

    vectorstore = FAISS.from_documents(pdf_splits, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    logger.info("  Knowledge base ready.")
    return retriever


# ==============================================================================
# STEP 2: BUILD THE ADAPTIVE RAG GRAPH
# ==============================================================================

def build_adaptive_rag_graph(retriever):
    """Build the Adaptive RAG graph with routing, grading, and post-generation validation."""

    class GraphState(TypedDict):
        question: str
        generation: str
        documents: List[str]

    router_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, max_tokens=128)
    grade_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, max_tokens=128)
    gen_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3, max_tokens=1024)
    rewrite_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_tokens=256)

    web_search_tool = TavilySearchResults(k=3)

    # -- Chain 1: Query Router --
    class RouteQuery(BaseModel):
        datasource: Literal["vectorstore", "web_search"] = Field(
            ..., description="Route to 'vectorstore' or 'web_search'."
        )

    route_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert at routing a user question to a vectorstore or web search. "
         "The vectorstore contains the 'Attention Is All You Need' research paper about "
         "transformers, self-attention, multi-head attention, encoder-decoder architecture, "
         "and positional encoding. Use the vectorstore for questions about these topics. "
         "For anything else, use web-search."),
        ("human", "{question}"),
    ])
    question_router = route_prompt | router_llm.with_structured_output(RouteQuery)

    # -- Chain 2: Retrieval Grader --
    class GradeDocuments(BaseModel):
        binary_score: str = Field(description="Relevant: 'yes' or 'no'")

    grade_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a grader assessing relevance of a retrieved document to a user question. "
         "If the document contains keyword(s) or semantic meaning related to the question, "
         "grade it as relevant. Give a binary score 'yes' or 'no'."),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ])
    retrieval_grader = grade_prompt | grade_llm.with_structured_output(GradeDocuments)

    # -- Chain 3: Hallucination Grader (NEW in Adaptive RAG) --
    # Checks if the generated answer is GROUNDED in the retrieved documents.
    class GradeHallucinations(BaseModel):
        binary_score: str = Field(description="Grounded in facts: 'yes' or 'no'")

    hallucination_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a grader assessing whether an LLM generation is grounded in / supported by "
         "a set of retrieved facts. Give a binary score 'yes' or 'no'. "
         "'Yes' means the answer is grounded in the facts."),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ])
    hallucination_grader = hallucination_prompt | grade_llm.with_structured_output(GradeHallucinations)

    # -- Chain 4: Answer Grader (NEW in Adaptive RAG) --
    # Checks if the answer actually ADDRESSES the user's question.
    class GradeAnswer(BaseModel):
        binary_score: str = Field(description="Addresses question: 'yes' or 'no'")

    answer_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a grader assessing whether an answer addresses / resolves a question. "
         "Give a binary score 'yes' or 'no'. 'Yes' means the answer resolves the question."),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ])
    answer_grader = answer_prompt | grade_llm.with_structured_output(GradeAnswer)

    # -- Chain 5: Question Rewriter --
    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a question re-writer that converts an input question to a better version "
         "optimized for vectorstore retrieval. Reason about the underlying semantic intent."),
        ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question."),
    ])
    question_rewriter = rewrite_prompt | rewrite_llm | StrOutputParser()

    # -- Chain 6: RAG Generation --
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an assistant for question-answering tasks. Use the following context "
         "to answer the question. If you don't know the answer, say so. "
         "Keep the answer concise (3-5 sentences)."),
        ("human", "Question: {question}\n\nContext: {context}\n\nAnswer:"),
    ])
    rag_chain = rag_prompt | gen_llm | StrOutputParser()

    # ==========================================================================
    # NODE DEFINITIONS
    # ==========================================================================

    def retrieve(state: GraphState) -> dict:
        """Retrieve documents from the local knowledge base."""
        logger.info("  [retrieve] Querying local knowledge base...")
        documents = retriever.invoke(state["question"])
        logger.info("  [retrieve] Retrieved %d documents", len(documents))
        return {"documents": documents, "question": state["question"]}

    def grade_documents(state: GraphState) -> dict:
        """Grade each retrieved document for relevance."""
        logger.info("  [grade_docs] Grading %d documents...", len(state["documents"]))
        question = state["question"]
        filtered_docs = []
        for i, doc in enumerate(state["documents"]):
            try:
                score = retrieval_grader.invoke({"question": question, "document": doc.page_content})
                grade = score.binary_score.lower().strip()
            except Exception:
                grade = "yes"
            if grade == "yes":
                logger.info("  [grade_docs] Doc %d: RELEVANT", i + 1)
                filtered_docs.append(doc)
            else:
                logger.info("  [grade_docs] Doc %d: NOT RELEVANT", i + 1)
        logger.info("  [grade_docs] Kept %d/%d documents", len(filtered_docs), len(state["documents"]))
        return {"documents": filtered_docs, "question": question}

    def generate(state: GraphState) -> dict:
        """Generate answer from retrieved context."""
        logger.info("  [generate] Producing answer...")
        question = state["question"]
        documents = state["documents"]
        context = "\n\n".join(
            doc.page_content if hasattr(doc, "page_content") else str(doc)
            for doc in documents
        )
        generation = rag_chain.invoke({"question": question, "context": context})
        logger.info("  [generate] Answer: %s", generation[:120])
        return {"documents": documents, "question": question, "generation": generation}

    def transform_query(state: GraphState) -> dict:
        """Rewrite the query for better retrieval."""
        logger.info("  [transform] Rewriting query...")
        better_question = question_rewriter.invoke({"question": state["question"]})
        logger.info("  [transform] Rewritten: %s", better_question[:100])
        return {"documents": state.get("documents", []), "question": better_question}

    def web_search(state: GraphState) -> dict:
        """Search the web via Tavily as external source."""
        logger.info("  [web_search] Searching the web...")
        question = state["question"]
        try:
            results = web_search_tool.invoke({"query": question})
            parts = []
            for d in results:
                if isinstance(d, dict) and "content" in d:
                    parts.append(d["content"])
                else:
                    parts.append(str(d))
            web_doc = Document(page_content="\n\n".join(parts))
            logger.info("  [web_search] Found %d results", len(results))
            return {"documents": [web_doc], "question": question}
        except Exception as e:
            logger.warning("  [web_search] Failed: %s", str(e)[:80])
            return {"documents": [], "question": question}

    # ==========================================================================
    # CONDITIONAL EDGE FUNCTIONS
    # ==========================================================================

    # Edge A: Route the query at START
    def route_question(state: GraphState) -> str:
        """Route question to vectorstore or web search."""
        logger.info("  [router] Classifying query...")
        try:
            source = question_router.invoke({"question": state["question"]})
            route = source.datasource
        except Exception:
            route = "web_search"
        logger.info("  [router] Route: %s", route)
        if route == "web_search":
            return "web_search"
        return "vectorstore"

    # Edge B: After grading, decide to generate or transform
    def decide_to_generate(state: GraphState) -> str:
        """If no relevant docs remain, transform query. Otherwise generate."""
        filtered = state["documents"]
        if not filtered:
            logger.info("  [decide] No relevant docs -> TRANSFORM QUERY")
            return "transform_query"
        logger.info("  [decide] Has relevant docs -> GENERATE")
        return "generate"

    # Edge C: After generation, validate the answer (NEW in Adaptive RAG)
    # This is the TRIPLE CHECK:
    #   1. Is the answer grounded in the documents? (hallucination check)
    #   2. Does the answer address the question? (answer check)
    #   3. If either fails, self-correct.
    def grade_generation(state: GraphState) -> str:
        """Validate generation: hallucination check + answer relevance check."""
        logger.info("  [validate] Checking generation quality...")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]

        # Check 1: Hallucination
        try:
            h_score = hallucination_grader.invoke(
                {"documents": documents, "generation": generation}
            )
            grounded = h_score.binary_score.lower().strip()
        except Exception:
            grounded = "yes"

        if grounded != "yes":
            logger.info("  [validate] HALLUCINATION detected -> regenerate")
            return "not supported"

        # Check 2: Answer addresses the question
        logger.info("  [validate] Grounded in docs. Checking answer relevance...")
        try:
            a_score = answer_grader.invoke(
                {"question": question, "generation": generation}
            )
            useful = a_score.binary_score.lower().strip()
        except Exception:
            useful = "yes"

        if useful == "yes":
            logger.info("  [validate] Answer is USEFUL -> done")
            return "useful"
        else:
            logger.info("  [validate] Answer NOT USEFUL -> transform query and retry")
            return "not useful"

    # ==========================================================================
    # BUILD THE GRAPH
    # ==========================================================================
    builder = StateGraph(GraphState)

    builder.add_node("web_search", web_search)
    builder.add_node("retrieve", retrieve)
    builder.add_node("grade_documents", grade_documents)
    builder.add_node("generate", generate)
    builder.add_node("transform_query", transform_query)

    # START -> route_question -> web_search or retrieve
    builder.add_conditional_edges(
        START, route_question,
        {"web_search": "web_search", "vectorstore": "retrieve"},
    )

    # web_search -> generate (web results go straight to generation)
    builder.add_edge("web_search", "generate")

    # retrieve -> grade_documents
    builder.add_edge("retrieve", "grade_documents")

    # grade_documents -> generate or transform_query
    builder.add_conditional_edges(
        "grade_documents", decide_to_generate,
        {"transform_query": "transform_query", "generate": "generate"},
    )

    # transform_query -> retrieve (retry with better query)
    builder.add_edge("transform_query", "retrieve")

    # generate -> grade_generation -> useful/not supported/not useful
    builder.add_conditional_edges(
        "generate", grade_generation,
        {"not supported": "generate", "useful": END, "not useful": "transform_query"},
    )

    graph = builder.compile()
    save_graph_image(graph, "rag03_adaptive_rag")
    return graph


# ==============================================================================
# STEP 3: RUN THE ADAPTIVE RAG PIPELINE
# ==============================================================================

def run_adaptive_rag(graph, query: str) -> str:
    """Run a single query through the Adaptive RAG pipeline."""
    logger.info("  Query: '%s'", query)
    result = graph.invoke({"question": query}, config={"recursion_limit": 20})
    return result.get("generation", "No answer generated.")


# ==============================================================================
# COMPARISON TABLE: All RAG Patterns
# ==============================================================================
#
# Pattern          | Front Gate     | Doc Grading | Post-Gen Check | Correction
# -----------------|----------------|-------------|----------------|------------------
# Basic RAG        | None           | None        | None           | None
# -----------------|----------------|-------------|----------------|------------------
# Agentic RAG      | Agent picks    | Yes (batch) | None           | Rewrite + retry
# (Lesson 01)      | tool/source    |             |                | same source
# -----------------|----------------|-------------|----------------|------------------
# Corrective RAG   | None           | Yes (each)  | None           | Rewrite + web
# (Lesson 02)      | Always local   |             |                | search fallback
# -----------------|----------------|-------------|----------------|------------------
# Adaptive RAG     | Router picks   | Yes (each)  | Hallucination  | Rewrite + retry
# (THIS lesson)    | vectorstore    |             | + Answer check | full pipeline
#                  | or web search  |             |                |
# -----------------|----------------|-------------|----------------|------------------
#
# KEY INSIGHT:
#   Adaptive RAG is the MOST COMPLETE pattern. It has:
#   - Front gate (routing) from Agentic RAG
#   - Document grading from Corrective RAG
#   - Post-generation validation (NEW: hallucination + answer checks)
#   - Self-correction loop that retries the FULL pipeline


# ==============================================================================
# COMMON MISTAKES AND GOTCHAS
# ==============================================================================
#
# Mistake                                    | Fix
# -------------------------------------------|--------------------------------------
# No recursion_limit                         | The self-correction loop can go
#                                            | infinite. Always set recursion_limit.
# -------------------------------------------|--------------------------------------
# Hallucination grader too strict            | If the grader always says "no", the
#                                            | graph loops forever. Use a lenient
#                                            | grading prompt and set recursion_limit.
# -------------------------------------------|--------------------------------------
# Router description doesn't match content   | The router prompt must accurately
#                                            | describe what the vectorstore contains.
#                                            | Wrong description = wrong routing.
# -------------------------------------------|--------------------------------------
# Not handling grading failures              | Structured output can fail on Groq.
#                                            | Always fallback to "yes" on error.
# -------------------------------------------|--------------------------------------
# Too many LLM calls per query               | Adaptive RAG can make 5+ LLM calls.
#                                            | Monitor costs. Use fast models for
#                                            | grading and capable models only for
#                                            | routing.
# -------------------------------------------|--------------------------------------
#
# ANTI-PATTERN: Using Adaptive RAG for simple chatbots.
#   The overhead of routing + grading + validation is only worth it
#   for knowledge-intensive applications. For simple Q&A, use Basic RAG.


# ==============================================================================
# WHY LANGGRAPH FOR THIS
# ==============================================================================
#
# - Multiple conditional edges: route at START, grade after retrieve,
#   validate after generate. Only LangGraph makes this clean.
# - Self-correction loop: generate -> validate -> transform -> retrieve -> ...
# - Structured output: Pydantic models for routing, grading, validation.
# - Recursion limit: built-in safety against infinite loops.
# - Visual graph: you can SEE every decision point in the graph image.


# ==============================================================================
# WHERE THIS CONNECTS (Concept Linking Map)
# ==============================================================================
#
# Adaptive RAG -> combines -> Routing + CRAG + Post-Generation Validation
# route_question -> classifies -> query at START -> vectorstore or web
# retrieve -> queries -> FAISS vector store
# grade_documents -> filters -> irrelevant docs (from CRAG)
# generate -> produces -> answer from context
# grade_generation -> validates -> hallucination + answer relevance (NEW)
# transform_query -> rewrites -> query for retry
# Self-correction loop -> transform -> retrieve -> grade -> generate -> validate
# Adaptive RAG -> most complete -> RAG pattern in this series


# ==============================================================================
# INTERVIEW QUESTIONS
# ==============================================================================
#
# THEORETICAL:
# Q: What makes Adaptive RAG different from Corrective RAG?
#    What additional checks does Adaptive RAG perform?
# A: Corrective RAG grades retrieved documents and falls back to web
#    search when local docs are irrelevant. Adaptive RAG adds THREE
#    capabilities on top: (1) Query routing at the START to pick the
#    best strategy (vectorstore vs web search) before retrieval.
#    (2) Hallucination grading after generation to check if the answer
#    is grounded in the retrieved documents. (3) Answer grading to
#    check if the answer actually addresses the question. If either
#    post-generation check fails, Adaptive RAG rewrites the query and
#    retries the full pipeline. This makes it the most robust RAG pattern.
#
# HANDS-ON:
# Q: Add a "direct_answer" route to the Adaptive RAG pipeline. If the
#    router classifies the query as "simple" (e.g., greetings, basic math),
#    skip retrieval entirely and let the LLM answer directly.
#
# SOLUTION OUTLINE:
#   1. Add "direct" to RouteQuery Literal: ["vectorstore", "web_search", "direct"]
#   2. Add a direct_answer node: def direct_answer(state): llm.invoke(question)
#   3. Update route_question to return "direct" for simple queries.
#   4. Add conditional edge: START -> "direct": "direct_answer"
#   5. Add edge: direct_answer -> END


# ==============================================================================
# QUICK RECAP
# ==============================================================================
#
# 1. Adaptive RAG ROUTES the query first (vectorstore or web search),
#    then GRADES retrieved documents, GENERATES an answer, and
#    VALIDATES the answer for hallucination and relevance.
#
# 2. The post-generation validation is what makes Adaptive RAG unique:
#    hallucination grading checks if the answer is grounded in docs,
#    answer grading checks if it addresses the question. If either
#    fails, the query is rewritten and the pipeline retries.
#
# 3. Adaptive RAG = Routing (Agentic) + Document Grading (CRAG) +
#    Post-Generation Validation (NEW). It is the most complete RAG
#    pattern, best for knowledge-intensive production applications.


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("LangGraph RAG - Lesson 3: Adaptive RAG")
    logger.info("=" * 70)

    # Step 1: Build knowledge base
    logger.info("\n Step 1: Building Knowledge Base...")
    retriever = build_knowledge_base()

    # Step 2: Build graph
    logger.info("\n Step 2: Building Adaptive RAG Graph...")
    graph = build_adaptive_rag_graph(retriever)
    logger.info("  Graph compiled and ready.")

    # Step 3: Test queries
    logger.info("\n Step 3: Running Queries...")
    logger.info("=" * 70)

    # Query 1: In-domain -> should route to vectorstore
    logger.info("\n--- Query 1: In-domain (vectorstore route) ---")
    a1 = run_adaptive_rag(graph, "How does self-attention work in the transformer model?")
    logger.info("  Final Answer: %s", a1[:300])

    time.sleep(5)

    # Query 2: Out-of-domain -> should route to web search
    logger.info("\n--- Query 2: Out-of-domain (web search route) ---")
    a2 = run_adaptive_rag(graph, "Who won the Cricket World Cup in 2023?")
    logger.info("  Final Answer: %s", a2[:300])

    time.sleep(5)

    # Query 3: In-domain but broad -> vectorstore + may need validation
    logger.info("\n--- Query 3: Broad in-domain question ---")
    a3 = run_adaptive_rag(graph, "What are the key contributions of the Attention Is All You Need paper?")
    logger.info("  Final Answer: %s", a3[:300])

    logger.info("\n" + "=" * 70)
    logger.info("Lesson 3 complete - Adaptive RAG")
    logger.info("The pipeline routed queries, graded documents, validated answers,")
    logger.info("and self-corrected when needed.")
    logger.info("=" * 70)
