"""
===================================================================================
DOCSAGE — COMPLETE PROJECT DEEP DIVE (Interview Revision Guide)
===================================================================================

This is your ONE-STOP revision document for the DocSage project.
Read this top-to-bottom before any interview. Covers:

    1. Project Overview & Elevator Pitch
    2. Architecture & Tech Stack
    3. Project Structure (8 packages, 15+ files)
    4. State Management (TypedDict + add_messages reducer)
    5. LLM Factory Pattern (4 task-specific LLMs)
    6. Tools — Retriever, Web Search, Email
    7. RAG Pipeline — Load → Chunk → Embed → Store → Retrieve
    8. Nodes — Agent, Grade, Generate, Rewrite, Validate
    9. Graph Builder — Full Agentic + Adaptive RAG Workflow
    10. Streamlit UI — Sidebar, Chat, Streaming
    11. Deployment — Hugging Face Spaces
    12. Key Design Decisions & Why
    13. RAG Patterns Used (Agentic, Corrective, Adaptive)
    14. 50+ Interview Q&A (project-specific)

Live Demo: https://huggingface.co/spaces/tirupathi0/docsage
===================================================================================
"""


# =================================================================================
# SECTION 1: ELEVATOR PITCH (30-second answer)
# =================================================================================
"""
QUESTION: "Tell me about your project."

ANSWER:
    "DocSage is an Enterprise Document Intelligence Agent I built using
    LangGraph with Agentic RAG. The user uploads documents — PDFs, DOCX,
    CSV, TXT, Markdown — or pastes a URL, and the agent autonomously
    selects the right knowledge base using tool calling. It uses FAISS
    vector search with HuggingFace embeddings for retrieval, grades
    documents for relevance, generates answers with source citations,
    and validates for hallucination. It also has web search fallback
    via Tavily and email integration via Gmail SMTP. The whole thing
    is deployed on Hugging Face Spaces with a Streamlit frontend.
    It's a production-grade implementation of Agentic + Corrective +
    Adaptive RAG patterns."

KEY NUMBERS TO REMEMBER:
    - 8 packages, 15+ files
    - 5 graph nodes: agent, tools, grade, generate, rewrite
    - 3 conditional edges: tools_condition, grade_documents, validate_answer
    - 5 file formats: PDF, DOCX, TXT, CSV, MD + URL scraping
    - 4 task-specific LLMs: agent, grading, generation, rewrite
    - 3 RAG patterns: Agentic, Corrective, Adaptive
    - 3 tools: document retriever, web search (Tavily), email (Gmail SMTP)
"""


# =================================================================================
# SECTION 2: TECH STACK
# =================================================================================
"""
FRAMEWORK / LIBRARY          PURPOSE                         WHY THIS ONE
─────────────────────────────────────────────────────────────────────────────
LangGraph                    Graph-based agent workflow       State machine for RAG
LangChain                    Chains, tools, prompts           Foundation for LLM apps
Groq API                     LLM provider (free)              Free, fast inference
  - llama-3.1-8b-instant     Agent + Generation               Good tool calling
  - llama-3.3-70b-versatile  Grading + Validation             Best structured output
FAISS                        Vector store                     Fast, local, no server
HuggingFace Embeddings       Text → vectors                   Free, runs locally
  - all-MiniLM-L6-v2         384-dim embeddings               Small, fast, accurate
Tavily                       Web search API                   Built for LLM apps
Streamlit                    Frontend UI                      Fast prototyping
Gmail SMTP                   Email sending                    Simple, free
python-dotenv                Environment variables            Secure key management
Hugging Face Spaces          Deployment                       Free hosting + GPU

WHY GROQ (not OpenAI)?
    - Free tier with generous limits (perfect for portfolio projects)
    - Anyone can clone and run without paying
    - In production, swap to OpenAI/Anthropic by changing ONE file (groq_llm.py)

WHY FAISS (not Pinecone/Chroma)?
    - No external server needed (runs in-memory)
    - Fast similarity search
    - Perfect for per-session document stores
    - In production, use Pinecone for persistence across sessions

WHY HuggingFace Embeddings (not OpenAI)?
    - Free (no API cost per embedding)
    - Runs locally (no network latency)
    - all-MiniLM-L6-v2 is small (80MB) but accurate
    - In production, use OpenAI text-embedding-3-small for better quality
"""


# =================================================================================
# SECTION 3: PROJECT STRUCTURE
# =================================================================================
"""
docsage/
├── app.py                  # Entry point (Streamlit runs this)
├── main.py                 # Orchestrator (wires everything together)
├── config/
│   └── settings.py         # All constants: models, chunk sizes, UI labels
├── llms/
│   └── groq_llm.py         # LLM Factory — 4 task-specific LLM creators
├── state/
│   └── state.py            # AgentState TypedDict with add_messages reducer
├── tools/
│   ├── retriever_tool.py   # RAG pipeline: load → chunk → embed → FAISS → tool
│   ├── web_search_tool.py  # Tavily web search fallback
│   └── email_tool.py       # Gmail SMTP @tool
├── nodes/
│   ├── agent_node.py       # Brain: decides which tool to call
│   ├── grade_node.py       # Quality gate: grades document relevance
│   ├── generate_node.py    # RAG generation with source citations
│   ├── rewrite_node.py     # Self-correction: rewrites query
│   └── validate_node.py    # Post-gen: hallucination + relevance checks
├── graph/
│   └── graph_builder.py    # Assembles the full LangGraph workflow
└── ui/
    ├── sidebar.py          # Config panel: uploads, model, API keys
    └── chat_interface.py   # Chat UI with step-by-step streaming

DESIGN PRINCIPLE: Each file has ONE responsibility.
    - Change the LLM? Edit groq_llm.py only.
    - Add a new file format? Edit retriever_tool.py only.
    - Change the graph flow? Edit graph_builder.py only.
    - Change the UI? Edit sidebar.py or chat_interface.py only.
"""


# =================================================================================
# SECTION 4: STATE MANAGEMENT
# =================================================================================
"""
FILE: state/state.py

The AgentState is a TypedDict that defines the SHAPE of data flowing
through the graph. Every node reads from and writes to this shared state.

    class AgentState(TypedDict):
        messages:         Annotated[Sequence[BaseMessage], add_messages]
        documents:        list[str]
        generation:       str
        sources:          list[str]
        web_search_used:  str

KEY CONCEPT — add_messages REDUCER:
    Without reducer:  state["messages"] = [new_msg]  → REPLACES the list
    With reducer:     state["messages"] = [new_msg]  → APPENDS to the list

    This is the #1 LangGraph mistake: forgetting the reducer.
    Without it, each node overwrites messages → agent loses context.

INTERVIEW Q: "What is a reducer in LangGraph?"
    "A reducer defines HOW state fields are updated. The add_messages
    reducer ensures that when a node returns {'messages': [new_msg]},
    the new message is APPENDED to the existing list instead of replacing
    it. Without this, each node would overwrite the conversation history."

INTERVIEW Q: "Why TypedDict and not a Pydantic model?"
    "LangGraph uses TypedDict because it's lightweight and doesn't add
    validation overhead on every state update. In a graph with 5+ nodes,
    each updating state, Pydantic validation on every update would add
    unnecessary latency. TypedDict gives you type hints for IDE support
    without runtime cost."
"""


# =================================================================================
# SECTION 5: LLM FACTORY PATTERN
# =================================================================================
"""
FILE: llms/groq_llm.py

Different tasks need different LLM configurations:

    TASK            MODEL                    TEMPERATURE   MAX_TOKENS   WHY
    ─────────────────────────────────────────────────────────────────────────
    Agent           llama-3.1-8b-instant     0             512          Deterministic tool selection
    Grading         llama-3.3-70b-versatile  0             128          Reliable structured output
    Generation      llama-3.1-8b-instant     0.3           1024         Slightly creative answers
    Rewrite         llama-3.1-8b-instant     0             256          Deterministic rewrites

WHY SEPARATE MODELS:
    - Agent needs TOOL CALLING → 8b-instant works reliably with bind_tools
    - Grading needs STRUCTURED OUTPUT (Pydantic) → 70b is most reliable
    - Generation needs TEXT QUALITY → 8b with temp=0.3 for grounded creativity
    - Rewrite needs DETERMINISTIC output → temp=0 for consistent rewrites

THE FACTORY PATTERN:
    factory = GroqLLMFactory(api_key="gsk_...")
    agent_llm = factory.create_agent_llm()
    grading_llm = factory.create_grading_llm()
    generation_llm = factory.create_generation_llm()
    rewrite_llm = factory.create_rewrite_llm()

    Change a model name in ONE place → all nodes get the update.

INTERVIEW Q: "Why not use one LLM for everything?"
    "Different tasks have different requirements. The agent needs
    deterministic tool selection (temp=0), but generation benefits
    from slight creativity (temp=0.3). Grading needs a larger model
    (70b) for reliable structured output, but the agent works fine
    with the faster 8b model. Using one model for everything would
    mean compromising on at least one task."

INTERVIEW Q: "What is temperature in LLMs?"
    "Temperature controls randomness. 0 = deterministic (same input →
    same output). 1 = creative (more random). For tool selection and
    grading, I use 0 because I want consistent decisions. For answer
    generation, I use 0.3 for slightly varied but still grounded responses."
"""


# =================================================================================
# SECTION 6: TOOLS
# =================================================================================
"""
DocSage has 3 types of tools that the agent can call:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TOOL 1: DOCUMENT RETRIEVER (retriever_tool.py)
    - One tool PER uploaded document (e.g., search_attention, search_contract)
    - Each tool wraps a FAISS vector store built from that document
    - The agent reads tool descriptions to decide which document to search

    Pipeline: file bytes → temp file → detect type → loader → chunk → embed → FAISS → retriever → tool

    Supported formats:
        .pdf  → PyPDFLoader
        .docx → Docx2txtLoader
        .txt  → TextLoader
        .csv  → CSVLoader
        .md   → UnstructuredMarkdownLoader
        URL   → WebBaseLoader (web scraping)

    DRY Design: Both file and URL ingestion share _build_tool_from_documents()
    The ONLY difference is the loading step. Adding a new format = adding one loader.

    Chunking: RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    Embedding: HuggingFace all-MiniLM-L6-v2 (384 dimensions, free, local)
    Vector Store: FAISS (in-memory, fast, no external server)
    Retriever: top-k=4 documents per query

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TOOL 2: WEB SEARCH (web_search_tool.py)
    - Tavily web search as fallback when local docs don't have the answer
    - Returns max 3 results (configurable in settings)
    - The agent calls this when no PDF retriever is relevant

    Why Tavily: Purpose-built for LLM apps, returns clean structured content
    (not raw HTML). Free tier = 1000 searches/month.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TOOL 3: EMAIL (email_tool.py)
    - Gmail SMTP tool using @tool decorator
    - Agent calls it when user says "email this to john@company.com"
    - Sends HTML-formatted email with DocSage branding
    - Uses App Password (not regular Gmail password)

    Setup: Enable 2-Step Verification → Generate App Password → Add to .env

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HOW AGENTIC RAG WORKS — WHY EACH PDF GETS ITS OWN VECTOR STORE
(This is the KEY differentiator from basic RAG)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BASIC RAG (what most tutorials teach):
    Upload 4 PDFs → ALL chunks go into ONE vector store → every question
    searches the SAME index → no routing, no selection, no intelligence.

    Problem: If you upload a legal contract AND a financial report, and
    ask "What was Q3 revenue?", the retriever might return chunks from
    the legal contract (because some words overlap). No way to control
    which document gets searched.

AGENTIC RAG (what DocSage does):
    Upload 4 PDFs → EACH PDF gets its OWN FAISS vector store → each
    vector store is wrapped as a SEPARATE retriever tool → the agent
    LLM DECIDES which tool to call based on the question.

    The agent is the ROUTER. It reads tool descriptions and picks the
    best match. This is what makes it "agentic" — the LLM makes the
    routing decision, not a hardcoded rule.

CONCRETE EXAMPLE — Upload 4 PDFs:

    ┌──────────────────────────────────────────────────────────────────┐
    │  FILE UPLOADED          TOOL CREATED              FAISS INDEX   │
    │  ─────────────────────────────────────────────────────────────── │
    │  attention_paper.pdf  → search_attention_paper  → FAISS Index 1 │
    │  legal_contract.pdf   → search_legal_contract   → FAISS Index 2 │
    │  financial_report.pdf → search_financial_report  → FAISS Index 3 │
    │  hr_policy.pdf        → search_hr_policy         → FAISS Index 4 │
    └──────────────────────────────────────────────────────────────────┘

    Each tool has a DESCRIPTION that the agent reads:
    - search_attention_paper:   "Search 'attention_paper.pdf'. Contains: transformer, self-attention..."
    - search_legal_contract:    "Search 'legal_contract.pdf'. Contains: partnership terms, liability..."
    - search_financial_report:  "Search 'financial_report.pdf'. Contains: Q3 revenue, profit margins..."
    - search_hr_policy:         "Search 'hr_policy.pdf'. Contains: leave policy, work from home..."

    Now the agent has 4 tools + web search + email = 6 tools total.

HOW THE AGENT DECIDES (step by step):

    User asks: "What is the leave policy for work from home?"

    Step 1: Agent LLM receives the question + ALL tool descriptions
    Step 2: LLM reads each description and REASONS:
            - search_attention_paper → "transformer, self-attention" → NOT relevant
            - search_legal_contract → "partnership, liability" → NOT relevant
            - search_financial_report → "revenue, profit" → NOT relevant
            - search_hr_policy → "leave policy, work from home" → MATCH!
            - tavily_search → web search → NOT needed (local doc has it)
    Step 3: LLM returns tool_call: { name: "search_hr_policy", args: { query: "work from home leave policy" } }
    Step 4: ToolNode executes search_hr_policy → searches ONLY FAISS Index 4
    Step 5: Returns top-4 chunks from hr_policy.pdf ONLY

    This is the PROOF that it's Agentic RAG:
    - The LLM made a DECISION (not a hardcoded rule)
    - Only ONE vector store was searched (not all 4)
    - The routing was based on SEMANTIC UNDERSTANDING of the question

WHERE THIS HAPPENS IN CODE:

    # retriever_tool.py — EACH file gets its own pipeline:
    def build_retriever_from_file(file_bytes, file_name, description):
        pages = _load_document(...)           # Load THIS file only
        chunks = splitter.split_documents(pages)  # Chunk THIS file only
        vectorstore = FAISS.from_documents(chunks, embeddings)  # OWN FAISS index
        retriever = vectorstore.as_retriever(k=4)  # OWN retriever
        tool = create_retriever_tool(retriever, name, description)  # OWN tool
        return tool  # This tool searches ONLY this document

    # main.py — Loop creates one tool per file:
    tools = []
    for file_bytes, file_name, description in uploaded_files:
        tool = build_retriever_from_file(file_bytes, file_name, description)
        tools.append(tool)  # Each file = separate tool

    # agent_node.py — Agent gets ALL tools:
    llm_with_tools = llm.bind_tools(tools)  # Agent sees all 4+ tools
    response = llm_with_tools.invoke(messages)  # LLM DECIDES which to call

    # graph_builder.py — ToolNode executes the chosen tool:
    tool_node = ToolNode(tools)  # Knows all tools, executes by name

BASIC RAG vs AGENTIC RAG — SIDE BY SIDE:

    BASIC RAG:
        4 PDFs → 1 vector store → 1 retriever → every question searches everything
        No routing. No selection. Chunks from all docs mixed together.
        If you ask about revenue, you might get legal contract chunks too.

    AGENTIC RAG (DocSage):
        4 PDFs → 4 vector stores → 4 retriever tools → agent PICKS the right one
        Smart routing. Targeted search. Only relevant document is searched.
        If you ask about revenue, ONLY the financial report is searched.

HOW PRODUCTION SYSTEMS DO IT:

    Small scale (DocSage): Each doc = separate FAISS index in memory
    Medium scale: Each doc = separate Pinecone namespace (same index, different partitions)
    Large scale: Each doc = metadata filter on a shared Pinecone index
        → retriever = vectorstore.as_retriever(filter={"doc_id": "financial_report"})
        → Agent decides the filter value, not the user

    The PATTERN is the same at every scale: the agent DECIDES which
    knowledge source to query. The implementation detail (separate FAISS
    vs Pinecone namespace vs metadata filter) changes, but the agentic
    routing stays the same.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

INTERVIEW Q: "How does the agent decide which tool to call?"
    "Each uploaded document becomes a SEPARATE retriever tool with its own
    FAISS vector store and its own description. The agent LLM has all tools
    bound via bind_tools(). When the user asks a question, the LLM reads
    ALL tool descriptions and picks the one whose description best matches
    the question. For example, if the user asks about 'leave policy', the
    agent picks the HR policy tool because its description mentions 'leave
    policy'. This is the same ReAct pattern — the LLM reasons about which
    tool to use."

INTERVIEW Q: "How can you prove it's Agentic RAG and not basic RAG?"
    "In basic RAG, all documents go into ONE vector store — every question
    searches everything. In DocSage, each document gets its OWN FAISS index
    wrapped as a SEPARATE tool. The agent LLM DECIDES which tool to call
    based on the question. I can prove this by uploading 4 different PDFs
    and asking a question about one specific topic — the agent will call
    ONLY that document's retriever tool, not all four. The tool_calls in
    the AIMessage show exactly which tool was selected."

INTERVIEW Q: "What if the agent picks the wrong document?"
    "That's where Corrective RAG kicks in. After retrieval, the grade node
    checks if the retrieved chunks are relevant to the question. If the
    agent picked the wrong document, the chunks will be irrelevant, the
    grader returns 'no', and the system rewrites the query and retries.
    The self-correction loop catches agent mistakes."

INTERVIEW Q: "What is bind_tools?"
    "bind_tools() wraps the LLM so its responses can include tool_calls.
    The LLM doesn't EXECUTE the tools — it just DECIDES which tool to
    call and with what arguments. The ToolNode in the graph actually
    executes the tool call. This separation is important: the LLM is
    the brain (decides), the ToolNode is the hands (executes)."

INTERVIEW Q: "What is create_retriever_tool?"
    "It's a LangChain utility that wraps a retriever (like FAISS) as a
    callable tool. It takes a retriever, a name, and a description, and
    returns a BaseTool that the agent can call. When called, it runs
    similarity search on the vector store and returns the top-k documents."

INTERVIEW Q: "How would this work at scale with 1000 documents?"
    "At scale, you wouldn't create 1000 separate FAISS indexes. Instead,
    you'd use a single Pinecone index with metadata filters. Each document
    gets a doc_id in its metadata. The agent decides which doc_id to filter
    on, and the retriever searches only that partition. The agentic routing
    pattern stays the same — only the storage backend changes."
"""


# =================================================================================
# SECTION 7: RAG PIPELINE — DEEP DIVE
# =================================================================================
"""
RAG = Retrieval-Augmented Generation

Instead of the LLM answering from its training data (which can be outdated
or hallucinated), we RETRIEVE relevant documents first, then GENERATE an
answer grounded in those documents.

THE PIPELINE (what happens when you upload a PDF):

    ┌─────────────────────────────────────────────────────────────────┐
    │  STEP 1: LOAD                                                   │
    │  file bytes → temp file → detect extension → pick loader        │
    │  .pdf → PyPDFLoader | .docx → Docx2txtLoader | .txt → TextLoader│
    │  .csv → CSVLoader   | .md → UnstructuredMarkdownLoader          │
    │  URL → WebBaseLoader (scrapes the webpage)                      │
    │  Output: list[Document] (each page = one Document object)       │
    └─────────────────────────────────────────────────────────────────┘
                                    ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │  STEP 2: CHUNK (Split)                                          │
    │  RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)│
    │  Splits by: \\n\\n → \\n → . → space → character                  │
    │  chunk_overlap=200 prevents context loss at boundaries          │
    │  Output: list[Document] (many smaller chunks)                   │
    └─────────────────────────────────────────────────────────────────┘
                                    ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │  STEP 3: EMBED                                                  │
    │  HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")          │
    │  Each chunk → 384-dimensional vector                            │
    │  Singleton pattern: model loaded ONCE, reused across all docs   │
    │  Output: list[vector] (one per chunk)                           │
    └─────────────────────────────────────────────────────────────────┘
                                    ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │  STEP 4: STORE                                                  │
    │  FAISS.from_documents(chunks, embeddings)                       │
    │  In-memory vector store (no external server)                    │
    │  Supports cosine similarity search                              │
    │  Output: FAISS index (searchable vector database)               │
    └─────────────────────────────────────────────────────────────────┘
                                    ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │  STEP 5: RETRIEVE                                               │
    │  vectorstore.as_retriever(search_kwargs={"k": 4})               │
    │  User query → embed → find 4 most similar chunks                │
    │  Uses cosine similarity (angle between vectors)                 │
    │  Output: top-4 Document objects                                 │
    └─────────────────────────────────────────────────────────────────┘
                                    ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │  STEP 6: WRAP AS TOOL                                           │
    │  create_retriever_tool(retriever, name, description)            │
    │  The agent can now CALL this tool by name                       │
    │  Output: BaseTool (callable by the agent)                       │
    └─────────────────────────────────────────────────────────────────┘

INTERVIEW Q: "Why chunk_size=1000 and chunk_overlap=200?"
    "1000 tokens is a good balance — small enough to be specific, large
    enough to contain meaningful context. The 200-token overlap ensures
    that if a sentence spans two chunks, both chunks have the full sentence.
    Without overlap, you'd lose context at chunk boundaries."

INTERVIEW Q: "What is cosine similarity?"
    "It measures the angle between two vectors. If two text chunks have
    similar meaning, their embedding vectors point in similar directions,
    giving a cosine similarity close to 1. It's scale-invariant — the
    length of the vector doesn't matter, only the direction."

INTERVIEW Q: "Why FAISS and not Pinecone?"
    "FAISS runs in-memory with no external server — perfect for a
    per-session document store where each user uploads their own docs.
    Pinecone is better for production with persistent, shared knowledge
    bases. For this project, FAISS keeps it simple and free."

INTERVIEW Q: "What embedding model do you use and why?"
    "all-MiniLM-L6-v2 from HuggingFace. It produces 384-dimensional
    vectors, runs locally (no API cost), and is only 80MB. It's one of
    the most popular models for semantic search. In production, I'd
    consider OpenAI's text-embedding-3-small for better quality."
"""


# =================================================================================
# SECTION 8: GRAPH NODES — WHAT EACH ONE DOES
# =================================================================================
"""
The graph has 5 nodes + 3 conditional edges. Here's what each does:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

NODE 1: AGENT (agent_node.py)
    WHAT: The brain — decides which tool to call
    HOW:  LLM with bind_tools() reads tool descriptions, picks the best match
    INPUT: User's question (from messages)
    OUTPUT: AIMessage with tool_calls (or direct answer if no tool needed)

    Key detail — CONTEXT WINDOW MANAGEMENT:
        The agent builds a COMPACT context from conversation history:
        - Collects only HumanMessages and AI text responses
        - Skips ToolMessages (large raw text that wastes tokens)
        - Keeps last 10 messages (5 Q&A turns)
        - This prevents token overflow on llama-3.1-8b (8K context)

    Why 10 messages? Enough for follow-up questions ("his age?", "her company?")
    without hitting the 8K token limit of llama-3.1-8b.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

NODE 2: TOOLS (ToolNode — LangGraph prebuilt)
    WHAT: Executes the tool call that the agent decided on
    HOW:  LangGraph's ToolNode reads tool_calls from the AIMessage and runs them
    INPUT: AIMessage with tool_calls
    OUTPUT: ToolMessage with the retrieval results (document chunks)

    This is NOT a custom node — it's LangGraph's prebuilt ToolNode.
    It automatically routes to the correct tool based on the tool name.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

NODE 3: GRADE (grade_node.py) — CONDITIONAL EDGE
    WHAT: Quality gate — checks if retrieved documents are relevant
    HOW:  LLM with structured output (Pydantic) returns "yes" or "no"
    INPUT: Retrieved documents + user's question
    OUTPUT: "generate" (docs relevant) or "rewrite" (docs not relevant)
            or "done" (for non-retrieval tools like email)

    Uses with_structured_output(GradeDocuments) to force binary yes/no.
    The 70b model is used here because structured output needs more reasoning.

    Special case: If the last tool was send_email, skip grading → return "done"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

NODE 4: GENERATE (generate_node.py)
    WHAT: Produces the final answer with source citations
    HOW:  RAG prompt + LLM + StrOutputParser
    INPUT: User's question + retrieved documents
    OUTPUT: generation (answer text) + sources (citations)

    The RAG prompt says: "Use ONLY the following context to answer.
    If you don't have enough info, say so." This prevents hallucination.

    Source citations are extracted from ToolMessage metadata (tool name,
    content preview). Displayed in the UI under an expandable "Sources" section.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

NODE 5: REWRITE (rewrite_node.py)
    WHAT: Self-correction — rewrites the query for better retrieval
    HOW:  LLM rewrites the question to be more specific
    INPUT: Original question
    OUTPUT: Rewritten question (as a new HumanMessage)

    The rewritten query goes back to the agent node → retry retrieval.
    This creates a SELF-CORRECTION LOOP: grade → rewrite → agent → tools → grade...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CONDITIONAL EDGE: VALIDATE (validate_node.py)
    WHAT: Post-generation quality checks (2 checks)
    HOW:  Two structured output chains (hallucination + answer relevance)

    CHECK 1 — HALLUCINATION: "Is the answer grounded in the documents?"
        If NO → return "not supported" → regenerate with same docs

    CHECK 2 — ANSWER RELEVANCE: "Does the answer address the question?"
        If YES → return "useful" → END (done!)
        If NO  → return "not useful" → rewrite query → full retry

    Both checks use the 70b model with structured output (Pydantic).
"""


# =================================================================================
# SECTION 9: THE COMPLETE GRAPH FLOW
# =================================================================================
"""
FILE: graph/graph_builder.py

THE FULL WORKFLOW (memorize this diagram):

    [START]
       │
    [agent]  ← LLM with bind_tools (PDF retrievers + web search + email)
       │
       ├── (has tool_call) ──→ [tools]  ← ToolNode executes the chosen tool
       │                          │
       │                    (grade_documents)  ← Conditional edge
       │                          │
       │                    ┌─────┴──────┐──────────┐
       │                    │            │          │
       │               "generate"    "rewrite"    "done"
       │                    │            │          │
       │               [generate]   [rewrite]    [END]
       │                    │            │      (email sent)
       │              (validate_answer)  │
       │                    │            │
       │              ┌─────┼─────┐      │
       │              │     │     │      │
       │          "useful" "not  "not    │
       │              │  supported" useful"
       │              │     │     │      │
       │            [END] [generate] ────┘
       │                  (retry)   (loops back to rewrite → agent)
       │
       └── (no tool_call) ──→ [END]  ← Agent answers directly (greetings, etc.)


THREE CONDITIONAL EDGES:
    1. tools_condition (LangGraph prebuilt):
       - Has tool_calls → go to "tools"
       - No tool_calls → go to END

    2. grade_documents (custom):
       - "generate" → docs are relevant, proceed to generation
       - "rewrite"  → docs not relevant, rewrite query and retry
       - "done"     → non-retrieval tool (email), skip to END

    3. validate_answer (custom):
       - "useful"        → answer is good, go to END
       - "not supported" → hallucinated, regenerate with same docs
       - "not useful"    → doesn't answer question, rewrite and full retry

RECURSION LIMIT: 20 (prevents infinite self-correction loops)

MEMORY: MemorySaver checkpointer
    - Each conversation gets a unique thread_id
    - Messages persist across turns within the same thread
    - Without this, each invoke() would be independent

INTERVIEW Q: "Walk me through what happens when a user asks a question."
    "1. The user types a question in the chat.
     2. The agent node receives it, reads all tool descriptions, and
        decides which retriever to call (or web search, or email).
     3. The ToolNode executes the chosen tool — runs similarity search
        on the FAISS vector store and returns top-4 chunks.
     4. The grade node checks if those chunks are relevant to the question.
        If not, it rewrites the query and loops back to the agent.
     5. If relevant, the generate node produces an answer using a RAG
        prompt that says 'answer ONLY from this context.'
     6. The validate node checks for hallucination (is the answer grounded
        in the docs?) and answer relevance (does it address the question?).
     7. If both checks pass, the answer is returned to the user with
        source citations. If either fails, it self-corrects."

INTERVIEW Q: "What is MemorySaver?"
    "MemorySaver is LangGraph's in-memory checkpointer. It persists the
    graph state (messages, documents, generation) across multiple invoke()
    calls within the same thread_id. This enables multi-turn conversations
    where the agent remembers previous questions and answers. In production,
    I'd use SqliteSaver or PostgresSaver for persistence across restarts."
"""


# =================================================================================
# SECTION 10: RAG PATTERNS USED
# =================================================================================
"""
DocSage implements THREE advanced RAG patterns:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PATTERN 1: AGENTIC RAG
    "The agent DECIDES which knowledge base to query."

    In basic RAG, there's one retriever and every question goes to it.
    In Agentic RAG, the agent has MULTIPLE retrievers (one per document)
    and CHOOSES which one to call based on the question.

    Example: User uploads a legal contract and a financial report.
    - "What are the partnership terms?" → agent calls search_legal_contract
    - "What was Q3 revenue?" → agent calls search_financial_report
    - "What is the weather?" → agent calls web_search (Tavily)

    Implementation: bind_tools() + ToolNode + tools_condition

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PATTERN 2: CORRECTIVE RAG (CRAG)
    "Grade retrieved documents. If irrelevant, CORRECT by rewriting the query."

    After retrieval, the grade node checks document relevance.
    If documents are NOT relevant → rewrite the query → retry retrieval.
    This self-correction loop prevents the system from generating answers
    from irrelevant context.

    Implementation: grade_node (conditional edge) + rewrite_node + loop back

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PATTERN 3: ADAPTIVE RAG
    "After generation, VALIDATE the answer. If bad, ADAPT by retrying."

    Even with relevant documents, the LLM can still hallucinate or
    miss the point. The validate node performs TWO post-generation checks:
    1. Hallucination check: Is the answer grounded in the documents?
    2. Answer relevance: Does the answer address the question?

    If hallucinated → regenerate with same docs.
    If not useful → rewrite query → full retry from agent.

    Implementation: validate_node (conditional edge) with 3 outcomes

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

INTERVIEW Q: "What's the difference between basic RAG and Agentic RAG?"
    "Basic RAG has one retriever — every question goes to the same vector
    store. Agentic RAG has an agent that DECIDES which retriever to call.
    The agent reads tool descriptions and routes the query to the most
    relevant knowledge base. This is critical when you have multiple
    documents covering different topics."

INTERVIEW Q: "What is Corrective RAG?"
    "After retrieval, we GRADE the documents for relevance. If they're
    not relevant, we don't just generate a bad answer — we CORRECT by
    rewriting the query and retrying retrieval. This self-correction
    loop significantly reduces hallucination from irrelevant context."

INTERVIEW Q: "What is Adaptive RAG?"
    "After generation, we VALIDATE the answer with two checks: is it
    grounded in the documents (hallucination check), and does it actually
    answer the question (relevance check). If either fails, the system
    ADAPTS — either regenerating or rewriting the query for a full retry.
    This post-generation validation is what makes it 'adaptive'."

INTERVIEW Q: "How do you prevent hallucination?"
    "Three layers of defense:
     1. The RAG prompt says 'answer ONLY from the provided context'
     2. The grade node filters out irrelevant documents before generation
     3. The validate node checks if the answer is grounded in the docs
     If the hallucination check fails, the system regenerates. This
     triple-layer approach makes hallucination very unlikely."
"""


# =================================================================================
# SECTION 11: STREAMLIT UI
# =================================================================================
"""
FILES: ui/sidebar.py, ui/chat_interface.py

SIDEBAR (sidebar.py):
    4 sections, ordered for interview demos:
    1. Document Upload — first thing interviewer sees (RAG showcase)
       - Multi-file upload (PDF, DOCX, TXT, CSV, MD)
       - Each file gets a description field (helps agent route queries)
    2. URL Sources — paste a webpage URL for web scraping
    3. Model Selection — dropdown with Groq models
    4. API Keys — Groq (required) + Tavily (optional)
    5. New Conversation — reset button at bottom

CHAT INTERFACE (chat_interface.py):
    - Uses st.chat_message() and st.chat_input() for chat UI
    - Step-by-step streaming with graph.stream(stream_mode="updates")
    - Real-time status messages as each node completes:
        "🤖 Selecting knowledge source..."
        "📄 Retrieving documents..."
        "✅ Checking document relevance..."
        "✍️ Generating answer..."
        "🔄 Improving query and retrying..."
    - Source citations in expandable section
    - Error handling with retry (tool_use_failed → direct LLM call)

KEY STREAMLIT PATTERNS:
    - st.session_state for conversation history (never global variables)
    - st.empty() for clearing status after completion
    - st.status() for real-time progress updates
    - uuid.uuid4() for unique thread_id per conversation

INTERVIEW Q: "How do you handle streaming?"
    "I use graph.stream() with stream_mode='updates'. As each node
    completes, I update a status widget with a friendly message like
    'Retrieving documents...' or 'Generating answer...'. This gives
    the user real-time feedback during the 10-30 second processing
    time instead of just a spinner. After completion, I clear the
    status widget so only the final answer remains."

INTERVIEW Q: "How do you handle errors?"
    "Three levels: (1) If a specific document fails to ingest, I skip
    it and warn the user but continue with other documents. (2) If the
    graph hits a tool_use_failed error (400), I retry with a direct LLM
    call as fallback. (3) For rate limits, auth errors, and other failures,
    I show a user-friendly error message instead of a stack trace."
"""


# =================================================================================
# SECTION 12: DEPLOYMENT — HUGGING FACE SPACES
# =================================================================================
"""
Live Demo: https://huggingface.co/spaces/tirupathi0/docsage

HOW IT'S DEPLOYED:
    1. Created a Hugging Face Space (Streamlit SDK)
    2. Added README.md with YAML metadata (title, sdk, app_file, etc.)
    3. Created app_hf.py (flat sys.path for HF's directory structure)
    4. Created requirements_deploy.txt (subset of dependencies)
    5. Added API keys as Secrets in HF Space Settings
    6. Pushed code to the Space repo → auto-deploys

WHY HUGGING FACE (not Streamlit Cloud)?
    - Free GPU access (for embedding model)
    - Better for ML/AI projects (community expects it)
    - Persistent URLs (good for resume links)
    - Secrets management built-in
    - Docker support if needed

INTERVIEW Q: "How did you deploy this?"
    "I deployed on Hugging Face Spaces using the Streamlit SDK. The Space
    auto-builds from the repo. API keys are stored as Secrets in the Space
    settings — they're injected as environment variables at runtime. The
    embedding model runs on the Space's CPU. For the LLM, I use Groq's
    API which is external. The live demo link is on my resume."

INTERVIEW Q: "What about scaling?"
    "For a portfolio project, HF Spaces is sufficient. For production,
    I'd containerize with Docker, deploy on AWS/GCP, use Pinecone for
    persistent vector storage, add Redis for caching, and put it behind
    an API gateway with rate limiting. The modular architecture makes
    this migration straightforward — swap FAISS for Pinecone in one file,
    swap Groq for OpenAI in one file."
"""


# =================================================================================
# SECTION 13: KEY DESIGN DECISIONS & WHY
# =================================================================================
"""
DECISION 1: One retriever tool PER document (not one shared vector store)
    WHY: The agent can CHOOSE which document to search based on the question.
    If all docs were in one vector store, the agent couldn't distinguish
    between a legal contract and a financial report.

DECISION 2: Separate LLMs for different tasks
    WHY: Agent needs fast tool calling (8b), grading needs reliable structured
    output (70b), generation needs slight creativity (temp=0.3). One model
    for everything would compromise at least one task.

DECISION 3: DRY pipeline for file and URL ingestion
    WHY: Both share the same chunk → embed → FAISS → tool pipeline. Only the
    loading step differs. Adding a new format = adding one loader function.

DECISION 4: Compact context window for agent (10 messages)
    WHY: llama-3.1-8b has 8K context. Sending full conversation history
    (including large ToolMessages) would overflow. Keeping only human
    questions and AI text answers (last 10) preserves follow-up context
    without hitting token limits.

DECISION 5: Grade node returns "done" for email tool
    WHY: When the agent sends an email, there are no documents to grade.
    Without this special case, the grader would try to grade the email
    confirmation message as a "document" and fail.

DECISION 6: Fingerprint-based graph caching
    WHY: Streamlit reruns the entire script on every interaction. Without
    caching, documents would be re-ingested on every message. The fingerprint
    (file names + model + API keys) detects changes — only rebuilds when
    the user uploads new documents or changes settings.

DECISION 7: MemorySaver for conversation memory
    WHY: Each invoke() is independent without a checkpointer. MemorySaver
    persists state across turns using thread_id. This enables multi-turn
    conversations where "his age?" refers to a person from a previous answer.

DECISION 8: Recursion limit = 20
    WHY: The self-correction loop (grade → rewrite → agent → tools → grade)
    could theoretically run forever. The recursion limit caps it at 20 steps.
    In practice, it usually resolves in 2-3 iterations.
"""


# =================================================================================
# SECTION 14: MAIN ORCHESTRATOR (main.py)
# =================================================================================
"""
FILE: main.py — The glue that connects everything.

FLOW:
    1. st.set_page_config() — must be first Streamlit call
    2. initialize_chat_state() — set up session state
    3. render_sidebar() → user_inputs dict
    4. display_chat_history() — show previous messages
    5. _get_or_build_graph(user_inputs) → compiled graph (cached)
    6. handle_user_input(graph) → chat interaction

GRAPH BUILDING (_build_graph):
    Step 1: Build retriever tools from uploaded documents
            - Loop through uploaded files, call build_retriever_from_file()
            - If a file fails, skip it and warn (don't break the pipeline)
    Step 1b: Build retriever from URL (if provided)
    Step 2: Add web search tool (if Tavily key available)
    Step 2b: Add email tool (if Gmail credentials available)
    Step 3: Create LLM factory (GroqLLMFactory)
    Step 4: Build and compile the graph (GraphBuilder)

CACHING:
    Uses fingerprint = hash(file_names + model + tavily_key + gmail + url)
    If fingerprint matches cached version → reuse graph
    If fingerprint changed → rebuild everything
    This avoids re-ingesting documents on every Streamlit rerun.
"""


# =================================================================================
# SECTION 15: INTERVIEW Q&A — 50+ QUESTIONS
# =================================================================================
"""
These are the most likely questions an interviewer will ask about DocSage.
Organized by topic. Practice answering each in 30-60 seconds.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PROJECT OVERVIEW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q1: "Tell me about your project in 30 seconds."
A: "DocSage is an Enterprise Document Intelligence Agent built with LangGraph.
   Users upload documents or paste URLs, and the agent autonomously selects
   the right knowledge base using tool calling. It uses FAISS for vector
   search, grades documents for relevance, generates answers with citations,
   and validates for hallucination. It also has web search fallback and
   email integration. Deployed on Hugging Face Spaces."

Q2: "What problem does it solve?"
A: "In enterprises, knowledge is scattered across PDFs, Word docs, CSVs,
   and web pages. DocSage lets users upload multiple documents and ask
   questions in natural language. The agent figures out which document
   has the answer, retrieves relevant chunks, and generates a grounded
   response with source citations. No manual searching needed."

Q3: "What makes it different from a basic chatbot?"
A: "Three things: (1) It's AGENTIC — the agent decides which knowledge
   base to query, not the user. (2) It has SELF-CORRECTION — if retrieval
   fails, it rewrites the query and retries. (3) It VALIDATES — after
   generation, it checks for hallucination and answer relevance. A basic
   chatbot just sends the question to an LLM and returns whatever it says."

Q4: "What's the tech stack?"
A: "LangGraph for the agent workflow, LangChain for tools and prompts,
   Groq API for LLM inference (free), FAISS for vector storage,
   HuggingFace embeddings (all-MiniLM-L6-v2), Tavily for web search,
   Streamlit for the frontend, and deployed on Hugging Face Spaces."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RAG & RETRIEVAL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q5: "What is RAG?"
A: "Retrieval-Augmented Generation. Instead of the LLM answering from its
   training data, we first RETRIEVE relevant documents from a knowledge
   base, then GENERATE an answer grounded in those documents. This reduces
   hallucination and lets the LLM answer questions about private data it
   was never trained on."

Q6: "Walk me through your RAG pipeline."
A: "Load the document using a format-specific loader, chunk it with
   RecursiveCharacterTextSplitter (1000 chars, 200 overlap), embed each
   chunk using HuggingFace all-MiniLM-L6-v2, store in FAISS, and wrap
   as a retriever tool. At query time, the user's question is embedded,
   FAISS finds the 4 most similar chunks via cosine similarity, and those
   chunks are passed to the LLM as context for answer generation."

Q7: "Why do you chunk documents?"
A: "LLMs have limited context windows. A 100-page PDF can't fit in one
   prompt. Chunking breaks it into smaller pieces (1000 chars each) so
   we can retrieve only the RELEVANT pieces. The 200-char overlap ensures
   sentences at chunk boundaries aren't cut in half."

Q8: "What is an embedding?"
A: "A dense vector representation of text. The embedding model converts
   text into a fixed-size vector (384 dimensions in my case) that captures
   semantic meaning. Similar texts have similar vectors. This lets us do
   semantic search — finding documents by MEANING, not just keywords."

Q9: "What is cosine similarity?"
A: "It measures the angle between two vectors. Cosine of 0° = 1 (identical),
   cosine of 90° = 0 (unrelated). It's scale-invariant — the magnitude
   doesn't matter, only the direction. This makes it ideal for comparing
   embeddings of different-length texts."

Q10: "What is FAISS?"
A: "Facebook AI Similarity Search. It's an in-memory vector database
    optimized for fast nearest-neighbor search. I use it to store document
    embeddings and find the most similar chunks to a user's query. It runs
    locally with no external server — perfect for per-session document stores."

Q11: "How do you handle multiple document formats?"
A: "I use format-specific loaders: PyPDFLoader for PDFs, Docx2txtLoader
    for Word docs, TextLoader for TXT, CSVLoader for CSV, and
    UnstructuredMarkdownLoader for Markdown. The loading step is the only
    format-specific part — the rest of the pipeline (chunk, embed, store)
    is shared. Adding a new format means adding one loader."

Q12: "What about images in PDFs?"
A: "PyPDFLoader extracts text only — it can't handle images or scanned
    PDFs. For images, I'd add OCR via UnstructuredLoader with Tesseract,
    or use a vision model like GPT-4V to describe images before indexing.
    For scanned PDFs, OCR is essential."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AGENTS & LANGGRAPH
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q13: "What is an AI agent?"
A: "An AI agent is an LLM that can DECIDE which tools to call and in what
    order to complete a task. Unlike a simple chain (fixed sequence), an
    agent reasons about the best action at each step. In DocSage, the agent
    decides whether to search a PDF, search the web, or send an email."

Q14: "What is LangGraph?"
A: "LangGraph is a framework for building stateful, multi-step agent
    workflows as directed graphs. Each node is a function, edges define
    the flow, and conditional edges enable dynamic routing. It supports
    checkpointing (conversation memory), streaming, and human-in-the-loop."

Q15: "Why LangGraph instead of AgentExecutor?"
A: "AgentExecutor is a simple loop: agent → tool → agent → tool. LangGraph
    gives you a full graph with conditional edges, parallel execution,
    and state management. I needed conditional routing (grade → generate
    OR rewrite) and post-generation validation — AgentExecutor can't do that."

Q16: "What is a StateGraph?"
A: "The main building block of LangGraph. You define a TypedDict for state,
    add nodes (functions), add edges (fixed or conditional), set an entry
    point, and compile. The compiled graph can be invoked or streamed."

Q17: "What is a conditional edge?"
A: "An edge where the next node depends on the output of the current node.
    For example, after grading, if documents are relevant → go to generate.
    If not → go to rewrite. The grade function returns a string ('generate'
    or 'rewrite') and the graph routes accordingly."

Q18: "What is tools_condition?"
A: "A LangGraph prebuilt function that checks if the agent's response
    contains tool_calls. If yes → route to the ToolNode. If no → route
    to END (agent answered directly without tools)."

Q19: "What is ToolNode?"
A: "A LangGraph prebuilt node that executes tool calls. It reads the
    tool_calls from the agent's AIMessage, finds the matching tool by
    name, calls it with the provided arguments, and returns the result
    as a ToolMessage."

Q20: "What is bind_tools?"
A: "It wraps the LLM so its responses can include tool_calls. The LLM
    doesn't execute tools — it just DECIDES which tool to call. bind_tools
    gives the LLM the tool schemas (name, description, parameters) so it
    can make informed decisions."

Q21: "What is with_structured_output?"
A: "It wraps the LLM to always return a Pydantic model instead of free
    text. I use it for grading (returns GradeDocuments with binary_score
    'yes' or 'no') and validation (returns GradeHallucinations). This
    makes the output deterministic and parseable."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HALLUCINATION & QUALITY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q22: "What is hallucination in LLMs?"
A: "When the LLM generates information that sounds plausible but is not
    grounded in the provided context or facts. For example, if the document
    says 'revenue was $10M' and the LLM says '$15M' — that's hallucination."

Q23: "How do you prevent hallucination?"
A: "Three layers: (1) The RAG prompt says 'answer ONLY from the provided
    context.' (2) The grade node filters out irrelevant documents before
    generation. (3) The validate node checks if the answer is grounded
    in the documents after generation. If hallucination is detected, the
    system regenerates."

Q24: "What happens if the documents don't have the answer?"
A: "The grade node detects that documents are irrelevant and triggers
    query rewriting. If the rewritten query still doesn't find relevant
    docs, the agent may fall back to web search (Tavily). If nothing
    works, the generate prompt instructs the LLM to say 'I don't have
    enough information' instead of making something up."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MEMORY & MULTI-TURN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q25: "How do you handle multi-turn conversations?"
A: "MemorySaver checkpointer persists state across turns using thread_id.
    The agent node builds a compact context from the last 10 messages
    (human questions + AI text answers only, skipping large ToolMessages).
    This lets follow-up questions like 'his age?' reference a person
    from a previous answer."

Q26: "Why only 10 messages?"
A: "llama-3.1-8b has an 8K token context window. ToolMessages contain
    large chunks of document text that would quickly overflow the context.
    By keeping only 10 compact messages (5 Q&A turns), I preserve enough
    context for follow-ups without hitting token limits."

Q27: "What is a thread_id?"
A: "A unique identifier for each conversation. MemorySaver uses it to
    isolate state between different conversations. Each new conversation
    gets a new UUID. This way, multiple users (or the same user with
    different topics) don't share conversation history."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOOLS & INTEGRATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q28: "How does the email feature work?"
A: "I built a custom LangChain @tool that sends emails via Gmail SMTP.
    When the user says 'email this to john@company.com', the agent
    extracts the recipient, uses the previous answer as the body, and
    calls the send_email tool. It uses Gmail App Passwords for auth."

Q29: "Why a @tool instead of MCP for email?"
A: "For simple, single-function tools, a direct @tool is simpler. MCP
    is better for complex integrations with multiple operations (like
    Jira with get_ticket, create_story, add_comment). The agent doesn't
    care — both appear as callable tools."

Q30: "What is Tavily?"
A: "A web search API purpose-built for LLM applications. It returns
    clean, structured content (not raw HTML). I use it as a fallback
    when local documents don't have the answer. Free tier gives 1000
    searches per month."

Q31: "How does URL scraping work?"
A: "WebBaseLoader from LangChain scrapes the webpage and extracts text.
    The scraped content goes through the same pipeline as uploaded files:
    chunk → embed → FAISS → retriever tool. The agent can then search
    the webpage content just like any uploaded document."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ARCHITECTURE & DESIGN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q32: "Why separate LLMs for different tasks?"
A: "Different tasks have different requirements. The agent needs fast,
    deterministic tool selection (8b, temp=0). Grading needs reliable
    structured output (70b, temp=0). Generation needs slight creativity
    (8b, temp=0.3). Using one model for everything would compromise
    at least one task."

Q33: "What is the Factory pattern you used?"
A: "GroqLLMFactory creates task-specific LLM instances. You pass the
    API key once, then call create_agent_llm(), create_grading_llm(),
    etc. Each method returns a ChatGroq configured for that task.
    Change a model name in ONE place → all nodes get the update."

Q34: "How do you handle errors?"
A: "Multiple levels: (1) If a document fails to ingest, skip it and
    continue with others. (2) If grading fails (Groq can be flaky),
    default to 'yes' to avoid blocking. (3) If the graph hits
    tool_use_failed, retry with a direct LLM call. (4) Rate limits
    and auth errors get user-friendly messages."

Q35: "What is the recursion limit for?"
A: "The self-correction loop (grade → rewrite → agent → tools → grade)
    could theoretically run forever. The recursion limit (20) caps the
    total number of graph steps. In practice, it resolves in 2-3 iterations."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DEPLOYMENT & PRODUCTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q36: "How did you deploy this?"
A: "On Hugging Face Spaces with the Streamlit SDK. API keys are stored
    as Secrets in the Space settings. The embedding model runs on the
    Space's CPU. The LLM calls go to Groq's API. The live demo link
    is on my resume."

Q37: "How would you scale this for production?"
A: "Containerize with Docker, deploy on AWS/GCP, swap FAISS for Pinecone
    (persistent vector storage), add Redis for caching, use PostgresSaver
    instead of MemorySaver, put behind an API gateway with rate limiting,
    and add authentication. The modular architecture makes each swap
    a one-file change."

Q38: "What about security?"
A: "API keys are in .env files (never committed). The sidebar uses
    password fields for key input. In production, I'd add input
    validation (prompt injection defense), rate limiting, and
    authentication. Document uploads would go through virus scanning."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LANGCHAIN SPECIFICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q39: "What is LCEL?"
A: "LangChain Expression Language. It uses the pipe operator (|) to
    compose chains: prompt | llm | output_parser. Each component is a
    Runnable that takes input and produces output. It's the modern way
    to build chains — the old LLMChain class is deprecated."

Q40: "What is a Runnable?"
A: "Any LCEL-compatible component that can be piped. Prompts, LLMs,
    output parsers, retrievers — they're all Runnables. You compose
    them with | to build chains."

Q41: "What is StrOutputParser?"
A: "It extracts the string content from an LLM response. Without it,
    you get an AIMessage object. With it, you get just the text string.
    I use it in the RAG chain: prompt | llm | StrOutputParser()."

Q42: "What is ChatPromptTemplate?"
A: "A template for chat-style prompts with system and human messages.
    I use ChatPromptTemplate.from_messages() with placeholders like
    {question} and {context} that get filled at runtime."

Q43: "What is create_retriever_tool?"
A: "A LangChain utility that wraps a retriever as a callable tool.
    It takes a retriever, name, and description, and returns a BaseTool.
    The agent can call it by name to run similarity search."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FUTURE SCOPE & IMPROVEMENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q44: "What would you improve?"
A: "Five things: (1) Add OCR for scanned PDFs and images. (2) Use
    Pinecone for persistent vector storage across sessions. (3) Add
    hybrid search (vector + keyword/BM25) for better retrieval.
    (4) Add a reranker (like Cohere) to re-score retrieved docs.
    (5) Add user authentication and document access control."

Q45: "What about multi-modal support?"
A: "I'd add GPT-4V or LLaVA to process images in documents. The image
    would be described by the vision model, and the description would
    be indexed alongside the text. This way, questions about charts
    or diagrams could be answered."

Q46: "What about evaluation?"
A: "I'd integrate LangSmith for tracing and evaluation. Set up automated
    evaluators for retrieval quality (precision@k, recall@k), answer
    quality (faithfulness, relevance), and latency. Run evaluation
    datasets to catch regressions before deployment."

Q47: "What about caching?"
A: "I'd add semantic caching — if a similar question was asked before,
    return the cached answer instead of running the full pipeline.
    LangChain has CacheBackedEmbeddings for this. Redis would be the
    cache backend in production."

Q48: "What about fine-tuning?"
A: "For domain-specific use cases (legal, medical), I'd fine-tune the
    embedding model on domain data for better retrieval. For the LLM,
    I'd use LoRA fine-tuning on domain Q&A pairs. But RAG is usually
    sufficient — fine-tuning is a last resort when RAG quality isn't
    enough."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REAL-TIME USE CASES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q49: "Where would this be used in a real company?"
A: "Legal teams querying contracts, HR teams searching policy documents,
    customer support agents finding answers in knowledge bases, research
    teams analyzing papers, compliance teams checking regulations. Any
    scenario where people need to find information in large document
    collections."

Q50: "How is this different from ChatGPT?"
A: "ChatGPT answers from its training data — it can't access your private
    documents. DocSage answers from YOUR uploaded documents with source
    citations. It's like having a personal research assistant that reads
    your specific files and gives grounded answers. Plus, it validates
    for hallucination, which ChatGPT doesn't do."

Q51: "Can multiple users use it simultaneously?"
A: "In the current Streamlit deployment, each user gets their own session
    with separate document stores and conversation history. For true
    multi-user production, I'd add authentication, shared document stores
    in Pinecone, and user-specific thread_ids in PostgresSaver."

Q52: "What's the latency like?"
A: "Typically 5-15 seconds for a full pipeline run (agent → retrieve →
    grade → generate → validate). The embedding step is the slowest
    (first query only, as the model loads). Subsequent queries are faster.
    The streaming UI shows progress so the user doesn't feel stuck."
"""


# =================================================================================
# SECTION 16: QUICK REVISION CHEAT SHEET
# =================================================================================
"""
MEMORIZE THESE FOR THE INTERVIEW:

ARCHITECTURE:  8 packages, 15+ files, 5 nodes, 3 conditional edges
MODELS:        llama-3.1-8b (agent/gen), llama-3.3-70b (grading/validation)
EMBEDDING:     all-MiniLM-L6-v2 (384 dims, free, local)
VECTOR STORE:  FAISS (in-memory, fast, no server)
CHUNKING:      1000 chars, 200 overlap, RecursiveCharacterTextSplitter
RETRIEVAL:     top-k=4, cosine similarity
TOOLS:         Document retriever + Tavily web search + Gmail email
RAG PATTERNS:  Agentic + Corrective + Adaptive
MEMORY:        MemorySaver with thread_id
DEPLOYMENT:    Hugging Face Spaces (Streamlit SDK)
LIVE DEMO:     https://huggingface.co/spaces/tirupathi0/docsage

GRAPH FLOW (one sentence):
    "Agent picks a tool → ToolNode retrieves docs → Grade checks relevance
    → Generate produces answer → Validate checks hallucination → END
    (or self-correct by rewriting and retrying)."

THREE RAG PATTERNS (one sentence each):
    Agentic:    "Agent DECIDES which knowledge base to query."
    Corrective: "Grade docs, CORRECT by rewriting if irrelevant."
    Adaptive:   "VALIDATE answer, ADAPT by retrying if hallucinated."

WHY SEPARATE LLMS (one sentence):
    "Agent needs fast tool calling (8b), grading needs reliable structured
    output (70b), generation needs slight creativity (temp=0.3)."
"""

print("=" * 70)
print("DocSage Deep Dive — Interview Revision Guide")
print("=" * 70)
print()
print("This file contains 16 sections covering the ENTIRE DocSage project.")
print("Read it top-to-bottom before any interview.")
print()
print("Sections:")
print("  1.  Elevator Pitch (30-second answer)")
print("  2.  Tech Stack (what and why)")
print("  3.  Project Structure (8 packages)")
print("  4.  State Management (TypedDict + add_messages)")
print("  5.  LLM Factory Pattern (4 task-specific LLMs)")
print("  6.  Tools (Retriever, Web Search, Email)")
print("  7.  RAG Pipeline (Load → Chunk → Embed → Store → Retrieve)")
print("  8.  Graph Nodes (Agent, Grade, Generate, Rewrite, Validate)")
print("  9.  Complete Graph Flow (diagram + explanation)")
print("  10. RAG Patterns (Agentic, Corrective, Adaptive)")
print("  11. Streamlit UI (Sidebar, Chat, Streaming)")
print("  12. Deployment (Hugging Face Spaces)")
print("  13. Key Design Decisions (8 decisions with reasoning)")
print("  14. Main Orchestrator (how everything connects)")
print("  15. Interview Q&A (52 questions with answers)")
print("  16. Quick Revision Cheat Sheet")
print()
print("Live Demo: https://huggingface.co/spaces/tirupathi0/docsage")
print("=" * 70)
