"""
===================================================================================
RAG AGENT SKELETON — PDF Tool + Web Search Tool (Interview Ready)
===================================================================================

This is the skeleton ETech asked you to write.
It covers: LangGraph flow, two tools, edge cases, and business-level thinking.

WHAT THIS DEMONSTRATES:
    - Agentic RAG (agent DECIDES which tool to call)
    - PDF retriever tool (local knowledge base)
    - Web search tool (fallback for real-time/missing info)
    - Edge cases handled at every step
    - Production patterns (error handling, fallbacks, validation)

MEMORIZE THIS PATTERN. It's the most common interview skeleton.
===================================================================================
"""

# ============================================
# STEP 1: IMPORTS
# ============================================
import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent

# ============================================
# STEP 2: LOAD ENV + CREATE LLM
# ============================================
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY", "")

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# ============================================
# STEP 3: TOOL 1 — PDF RETRIEVER (Local Knowledge Base)
# ============================================
# Edge cases handled:
#   - File not found
#   - Empty PDF (no text extracted)
#   - PDF with only images (OCR needed)
#   - Very large PDF (chunking handles this)

def build_pdf_tool(pdf_path: str, tool_name: str, description: str):
    """Build a retriever tool from a PDF file with edge case handling."""

    # EDGE CASE 1: File doesn't exist
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Load PDF
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    # EDGE CASE 2: Empty PDF (no text extracted)
    if not pages or not any(p.page_content.strip() for p in pages):
        raise ValueError(f"PDF '{pdf_path}' produced no text. May need OCR.")

    # Chunk with overlap
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(pages)

    # EDGE CASE 3: Chunking produced nothing (very short doc)
    if not chunks:
        raise ValueError(f"PDF '{pdf_path}' produced no chunks after splitting.")

    # Embed and store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # Wrap as a tool the agent can call
    return create_retriever_tool(retriever, name=tool_name, description=description)


# ============================================
# STEP 4: TOOL 2 — WEB SEARCH (Fallback for real-time info)
# ============================================
# Edge cases handled:
#   - API key missing
#   - Rate limit exceeded
#   - No results found
#   - Network timeout

web_search_tool = TavilySearchResults(max_results=3)
# Tavily handles most edge cases internally and returns error messages.
# If TAVILY_API_KEY is missing, it will raise an error at call time.


# ============================================
# STEP 5: BUILD TOOLS LIST
# ============================================
# In production, you'd build PDF tools dynamically from uploaded files.
# For this skeleton, we show the pattern:

tools = []

# Add PDF tool (if PDF exists)
pdf_path = "company_handbook.pdf"  # Replace with actual path
try:
    pdf_tool = build_pdf_tool(
        pdf_path=pdf_path,
        tool_name="search_company_handbook",
        description="Search the company handbook for HR policies, leave rules, benefits, and procedures."
    )
    tools.append(pdf_tool)
except (FileNotFoundError, ValueError) as e:
    print(f"Warning: PDF tool not created — {e}")

# Add web search tool (always available as fallback)
tools.append(web_search_tool)


# ============================================
# STEP 6: CREATE AGENT WITH SYSTEM PROMPT
# ============================================
# The system prompt tells the agent WHEN to use which tool.
# This is the ROUTING LOGIC — the agent reads this and decides.

system_prompt = """You are an intelligent assistant with access to two knowledge sources:

1. search_company_handbook — Search internal company documents (HR policies, procedures, benefits).
   USE THIS FIRST for any question about company policies, leave, benefits, or internal procedures.

2. tavily_search_results — Search the web for real-time information.
   USE THIS ONLY when:
   - The question is NOT about company policies
   - The company handbook doesn't have the answer
   - The question needs current/real-time information (news, weather, prices)

RULES:
- Always try the company handbook FIRST for internal questions.
- If the handbook doesn't have the answer, THEN fall back to web search.
- If neither tool has the answer, say "I don't have enough information to answer this."
- NEVER make up information. Only answer from retrieved content.
- Cite your source: "According to the company handbook..." or "Based on web search..."

EDGE CASES TO HANDLE:
- If the user asks something ambiguous, ask for clarification.
- If retrieval returns irrelevant results, say so honestly.
- If web search fails, inform the user and suggest trying later.
"""

agent = create_react_agent(llm, tools, prompt=system_prompt)


# ============================================
# STEP 7: RUN WITH EDGE CASE EXAMPLES
# ============================================
def ask_agent(question: str):
    """Run the agent with error handling."""
    print(f"\n{'='*60}")
    print(f"Q: {question}")
    print(f"{'='*60}")

    try:
        result = agent.invoke(
            {"messages": [HumanMessage(content=question)]},
            {"recursion_limit": 15},  # EDGE CASE: prevent infinite loops
        )
        print(f"A: {result['messages'][-1].content}")

    except RecursionError:
        # EDGE CASE: Agent stuck in loop
        print("A: I wasn't able to find a satisfactory answer after multiple attempts.")

    except Exception as e:
        error_msg = str(e)
        # EDGE CASE: Rate limit
        if "rate_limit" in error_msg.lower() or "429" in error_msg:
            print("A: API rate limit reached. Please try again in a few seconds.")
        # EDGE CASE: Auth error
        elif "api_key" in error_msg.lower() or "401" in error_msg:
            print("A: API authentication error. Please check your API keys.")
        # EDGE CASE: Network error
        elif "timeout" in error_msg.lower() or "connection" in error_msg.lower():
            print("A: Network error. Please check your internet connection.")
        else:
            print(f"A: An error occurred: {error_msg[:200]}")


# Test questions that exercise different paths:
ask_agent("What is the leave policy for remote workers?")      # → PDF tool
ask_agent("What is the current price of Bitcoin?")              # → Web search
ask_agent("How many vacation days do I get per year?")          # → PDF tool
ask_agent("What happened in the news today?")                   # → Web search


# =================================================================================
# EDGE CASES SUMMARY (What interviewers look for)
# =================================================================================
"""
EDGE CASES HANDLED IN THIS SKELETON:

INPUT EDGE CASES:
    ✅ PDF file not found → FileNotFoundError with clear message
    ✅ Empty PDF (no text) → ValueError, suggest OCR
    ✅ No chunks after splitting → ValueError
    ✅ Missing API keys → error at tool call time
    ✅ Ambiguous user question → agent asks for clarification (via prompt)

RETRIEVAL EDGE CASES:
    ✅ PDF doesn't have the answer → agent falls back to web search
    ✅ Web search returns no results → agent says "I don't have enough info"
    ✅ Retrieved docs are irrelevant → agent acknowledges honestly (via prompt)
    ✅ Both tools fail → graceful "I can't answer this" response

RUNTIME EDGE CASES:
    ✅ Infinite loop (agent keeps retrying) → recursion_limit=15
    ✅ Rate limit exceeded (429) → user-friendly message
    ✅ Auth error (401) → check API keys message
    ✅ Network timeout → check connection message
    ✅ Unexpected error → generic error with details

BUSINESS EDGE CASES (what senior interviewers care about):
    ✅ Routing logic: internal questions → PDF first, external → web
    ✅ Source citation: always tell user WHERE the answer came from
    ✅ Honesty: never make up information, say "I don't know" if needed
    ✅ Fallback chain: PDF → Web → "I don't know" (graceful degradation)
    ✅ Security: agent can't access tools it shouldn't (controlled tool list)

THE FLOW (what to draw on whiteboard):

    [User Question]
         ↓
    [Agent LLM] — reads system prompt + tool descriptions
         ↓
    Decides: internal question? → [PDF Retriever Tool]
             external question? → [Web Search Tool]
         ↓
    [Tool returns results]
         ↓
    [Agent LLM] — generates answer from results
         ↓
    Edge case: results irrelevant? → try other tool
    Edge case: both fail? → "I don't have enough info"
         ↓
    [Final Answer with source citation]
"""
