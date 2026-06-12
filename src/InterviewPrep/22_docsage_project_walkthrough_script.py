"""
===================================================================================
DOCSAGE PROJECT WALKTHROUGH — The 8-10 Minute Interview Script
===================================================================================

PURPOSE:
    When an interviewer says "walk me through your project" / "explain your
    architecture", you deliver THIS — a structured, professional, confident
    8-10 minute narrative that proves end-to-end understanding.

THE PROBLEM THIS SOLVES:
    You know the system, but you start from the UI, jump to folders, then
    components — inconsistent flow. This script fixes the ORDER so you sound
    like a senior engineer who designed the system on purpose.

THE GOLDEN RULE OF PROJECT WALKTHROUGHS:
    Go TOP-DOWN, OUTSIDE-IN:
      1. What it is + the problem (30s)
      2. The one-line architecture (30s)
      3. The request flow end-to-end (the spine — 3 min)
      4. Deep-dive the 2-3 impressive parts (3 min)
      5. Engineering decisions + trade-offs (1.5 min)
      6. Results, limits, what you'd improve (1 min)
    NEVER start with folder structure or UI widgets. Start with the PROBLEM
    and the BIG PICTURE, then drill down. Interviewers map your words onto a
    mental diagram — give them the diagram first.

HOW TO USE THIS FILE:
    - Sections 1-7 are the SPOKEN SCRIPT (memorize the bolded lines).
    - Section 8 is the 60-second version (for "quickly summarize it").
    - Section 9 is the Q&A bank (what they ask after the walkthrough).
    - Section 10 is the whiteboard diagram you draw while talking.
    - Section 11 is GOLDEN LESSONS (delivery tips).

GROUNDED IN YOUR REAL CODE:
    Every number/name here is from the actual repo so you never bluff:
    - LangGraph StateGraph, 5 nodes, MemorySaver checkpointer
    - Models: llama-3.1-8b-instant (agent/gen), llama-3.3-70b-versatile (grade)
    - Embeddings: sentence-transformers/all-MiniLM-L6-v2 (local, free)
    - Vector store: FAISS | Chunking: 1000/200 | Retriever k=4
    - Tools: per-document FAISS retrievers + Tavily web search + Gmail email
    - UI: Streamlit with stream_mode="updates" step-by-step streaming
    - Deploy: Hugging Face Spaces, keys via HF Secrets
===================================================================================
"""


# =================================================================================
# SECTION 1: THE OPENER — What it is + the problem (~30 seconds)
# =================================================================================
"""
Start here EVERY time. Calm, one breath, big picture. Do NOT dive into code.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SAY THIS (memorize the shape, not word-for-word):

    "Sure. The project is called DocSage — an Enterprise Document
    Intelligence Agent. The problem it solves: teams have large documents —
    contracts, reports, research papers — and they want to ask natural-
    language questions and get accurate, cited answers instead of manually
    searching through PDFs.

    What makes it more than a basic chatbot is that it's an AGENTIC RAG
    system. It doesn't just retrieve and answer — it decides which knowledge
    source to use, grades whether the retrieved content is actually relevant,
    validates its own answer for hallucination, and self-corrects if the
    answer isn't good enough. I built it with LangGraph as the orchestration
    layer."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHY THIS OPENER WORKS:
    - Names the project + category ("Agentic RAG") in the first 10 seconds.
    - States the USER PROBLEM (interviewers care about the "why").
    - Drops the 3 differentiators (decide, grade, self-correct) as a teaser
      you will expand later. This makes them WANT the deep dive.

DO NOT:
    - Don't open with "So there's a folder called docsage and inside it..."
    - Don't open with the Streamlit UI.
    - Don't list every library yet — that comes later.
"""


# =================================================================================
# SECTION 2: THE ONE-LINE ARCHITECTURE (~30 seconds)
# =================================================================================
"""
Give them the mental diagram in one sentence before any detail.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SAY THIS:

    "At a high level there are four layers:
      1. A Streamlit UI where users upload documents and chat.
      2. An ingestion pipeline that turns documents into a searchable
         vector index.
      3. A LangGraph agentic workflow — five nodes with conditional
         routing — that does the actual reasoning and retrieval.
      4. A set of tools the agent can call: per-document retrievers,
         a web-search fallback, and an email tool.

    The whole thing is modular — each piece is its own module, and a single
    orchestrator wires them together. I deployed it on Hugging Face Spaces."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE 4-LAYER MENTAL MODEL (this is your spine for the whole talk):

    [ UI: Streamlit ]
           |
    [ Ingestion: load -> chunk -> embed -> FAISS -> retriever tool ]
           |
    [ LangGraph workflow: agent -> tools -> grade -> generate -> validate ]
           |
    [ Tools: doc retrievers + Tavily web search + Gmail email ]

NOW you've earned the right to go deeper. The interviewer has the map.
"""


# =================================================================================
# SECTION 3: THE REQUEST FLOW — End to end (THE SPINE, ~3 minutes)
# =================================================================================
"""
This is the heart of the walkthrough. Trace ONE question through the system,
node by node. This single narrative proves you understand the whole pipeline.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SAY THIS (walk the path of a single user question):

    "Let me trace what happens when a user asks a question.

    STEP 1 — INGESTION (happens once, when documents are uploaded):
    When the user uploads a PDF, my ingestion pipeline loads it with the
    right loader — PyPDFLoader for PDFs, Docx2txt for Word, and so on. I
    split it with a RecursiveCharacterTextSplitter at 1000 characters with
    200 overlap, embed the chunks with a local HuggingFace model —
    all-MiniLM-L6-v2 — and store them in a FAISS vector index. Each document
    becomes its OWN retriever tool with a name and a description generated
    from its content. That description matters — it's how the agent later
    knows what's inside each document.

    STEP 2 — THE AGENT NODE (the brain):
    When a question comes in, it hits the agent node first. The agent is an
    LLM with all the tools bound to it via bind_tools. It reads the tool
    descriptions and DECIDES: do I call a specific document's retriever, the
    web-search tool, the email tool, or just answer directly? For example,
    if someone uploaded a contract and asks about payment terms, the agent
    matches that to the contract's retriever tool. This is the ReAct-style
    reasoning — the LLM picks the tool; it doesn't execute it.

    STEP 3 — THE TOOLS NODE (execution):
    A LangGraph ToolNode actually executes the chosen tool — runs the FAISS
    similarity search, returns the top 4 chunks.

    STEP 4 — THE GRADE NODE (quality gate):
    Here's where it stops being basic RAG. Before generating anything, a
    grading LLM checks: 'Are these retrieved chunks actually relevant to the
    question?' It returns a structured yes/no using Pydantic structured
    output. If the docs are relevant, we move to generation. If NOT, we go
    to the rewrite node instead. This is the Corrective RAG pattern.

    STEP 5 — THE GENERATE NODE (answer + citations):
    If the docs passed grading, the generate node produces the answer using
    a strict prompt that says 'answer ONLY from this context, and if it's not
    there, say you don't have enough information.' It also attaches source
    citations from the document metadata.

    STEP 6 — THE VALIDATE NODE (self-check):
    After generation I run TWO more checks. First, a hallucination check —
    'is this answer actually grounded in the retrieved documents?' Second,
    an answer-relevance check — 'does this answer actually address the
    question?' If it hallucinated, I regenerate. If it's grounded but off-
    topic, I rewrite the query and retry the whole loop. Only if both checks
    pass does the answer go back to the user. This is the Adaptive RAG part.

    STEP 7 — THE REWRITE NODE (self-correction):
    If grading or validation fails, the rewrite node asks an LLM to reword
    the query for better retrieval, and loops back to the agent. A recursion
    limit of 20 prevents infinite loops.

    Throughout all of this, the UI streams each node's progress in real time —
    'selecting source', 'retrieving', 'grading', 'generating' — so the user
    sees what's happening instead of staring at a spinner."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DELIVERY TIP:
    Use your hand or draw arrows as you say "agent -> tools -> grade ->
    generate -> validate". The physical motion sells that you SEE the graph.
    If you only memorize ONE thing, memorize this node sequence in order.
"""


# =================================================================================
# SECTION 4: DEEP DIVE — The 2-3 Impressive Parts (~3 minutes)
# =================================================================================
"""
After the spine, the interviewer is hooked. Now show DEPTH on the parts that
separate you from someone who just followed a tutorial. Pick 2-3 of these
based on what the interviewer reacts to. Don't dump all of them.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DEEP DIVE A — THE LANGGRAPH STATE & WHY THE REDUCER MATTERS:

    "The graph is built on a shared state — a TypedDict I call AgentState.
    It carries the conversation messages, the retrieved documents, the
    generated answer, and the source citations. The most important detail
    is the messages field: I annotate it with the add_messages reducer.
    Without that reducer, every node would OVERWRITE the message list and
    the agent would lose conversation context. With it, each node APPENDS.
    That one line is what makes multi-turn follow-ups like 'what about his
    age?' work — the reference person stays in context."

    WHY THIS IMPRESSES: it's the #1 LangGraph mistake; knowing it signals
    real hands-on experience, not tutorial copy-paste.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DEEP DIVE B — STRUCTURED OUTPUT FOR GRADING (anti-hallucination):

    "For the grading and validation nodes I use Pydantic structured output —
    with_structured_output on the LLM forces it to return a model with a
    binary yes/no score instead of free-form text. That makes the routing
    decision deterministic and parseable. If I parsed free text, I'd get
    'Well, it seems somewhat relevant...' which I can't route on. The
    structured score gives me a clean conditional edge: yes -> generate,
    no -> rewrite. I also added a defensive fallback — if Groq's structured
    output occasionally fails, I default to 'yes' so one flaky call doesn't
    break the whole pipeline."

    WHY THIS IMPRESSES: shows you think about reliability and edge cases,
    not just the happy path.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DEEP DIVE C — THE SELF-CORRECTION LOOP (Corrective + Adaptive RAG):

    "What I'm most proud of is the self-correction. Basic RAG retrieves once
    and hopes for the best. Mine has TWO correction points. First, after
    retrieval, the grader can reject irrelevant docs and send the query to
    the rewrite node, which rephrases it and tries again. Second, after
    generation, the validator catches hallucinations or off-topic answers
    and triggers either a regeneration or a full rewrite-and-retry. So the
    system actively recovers from bad retrieval and bad generation. These
    map to the named patterns — Corrective RAG (CRAG) on the retrieval side,
    and Adaptive/Self-RAG on the generation side."

    WHY THIS IMPRESSES: connecting your code to NAMED research patterns
    (CRAG, Self-RAG) is exactly what senior interviewers want to hear.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DEEP DIVE D — THE LLM FACTORY (right model for each task):

    "I don't use one LLM for everything. I built a factory that creates
    task-specific LLM instances. The agent and generation use the fast
    llama-3.1-8b-instant. The grading and validation use the larger
    llama-3.3-70b-versatile, because structured output needs stronger
    reasoning to reliably return valid JSON. Temperatures are tuned per
    task too — zero for tool selection and grading so they're deterministic,
    0.3 for generation so answers read naturally. Centralizing this in a
    factory means I can swap a model in one place instead of five nodes."

    WHY THIS IMPRESSES: shows cost/quality awareness and clean design.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DEEP DIVE E — MULTI-SOURCE TOOLS (docs + web + email + URL):

    "The agent isn't limited to uploaded files. Each document is its own
    retriever tool, but I also added a Tavily web-search tool as a fallback
    for questions the documents can't answer, a URL ingestion path so you
    can point it at a webpage, and even a Gmail tool so the agent can email
    an answer when the user says 'send this to so-and-so'. The agent chooses
    among all of these based on the tool descriptions — it's genuinely
    agentic, not a fixed pipeline."

    WHY THIS IMPRESSES: demonstrates true agent behavior (tool selection)
    and product thinking.

PICK 2-3, NOT ALL FIVE. A + C is the strongest pair (state + self-correction).
Add B or D if they probe reliability or cost.
"""


# =================================================================================
# SECTION 5: ENGINEERING DECISIONS & TRADE-OFFS (~1.5 minutes)
# =================================================================================
"""
Senior interviewers care MORE about WHY than WHAT. This section is where you
prove you made deliberate choices, not random ones. Frame each as a decision
with a reason and an honest trade-off.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SAY THIS (pick 3-4):

    "A few deliberate decisions:

    WHY LANGGRAPH over a plain chain: I needed conditional routing and loops —
    grade-then-branch, validate-then-retry. A linear LCEL chain can't loop
    back on itself. LangGraph's StateGraph with conditional edges models that
    cleanly, and the shared state makes the data flow explicit.

    WHY FAISS for the vector store: it's fast, local, and zero-cost — perfect
    for a portfolio project anyone can clone and run. The honest trade-off is
    FAISS is in-memory and not a production database. In production I'd move
    to Qdrant or Pinecone for persistence, metadata filtering, and scaling —
    and I'd keep the ingestion pipeline identical, just swap the store.

    WHY A LOCAL EMBEDDING MODEL (all-MiniLM-L6-v2): free, runs anywhere, no
    API cost or data egress. Trade-off is slightly lower quality than OpenAI
    embeddings — acceptable for this scale, and swappable in one line.

    WHY GROQ for inference: free with generous limits and very fast, so the
    self-correction loops don't feel slow. In production I'd swap to OpenAI
    or Bedrock — and because of the factory, that's a one-file change.

    WHY MODULAR STRUCTURE: each concern is its own module — nodes, tools,
    graph, llms, state, ui, config — and one orchestrator wires them. I can
    change the workflow without touching node logic, or swap a tool without
    touching the graph."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE PATTERN: "I chose X because Y. The trade-off is Z, and in production I'd
do W." This sentence shape makes you sound senior every single time.
"""


# =================================================================================
# SECTION 6: RESULTS, LIMITS & WHAT YOU'D IMPROVE (~1 minute)
# =================================================================================
"""
Close strong. Showing you know the LIMITS is a senior signal — juniors claim
everything is perfect; seniors know the gaps and have a roadmap.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SAY THIS:

    "In terms of outcome: it handles multi-format documents — PDF, Word, text,
    CSV, Markdown — plus web URLs, answers with citations, recovers from bad
    retrieval, and streams progress live. It's deployed on Hugging Face Spaces
    so anyone can use it with their own API key.

    I'm honest about the limits. FAISS is in-memory, so it's not built for
    millions of documents or multi-tenant persistence. PyPDF doesn't handle
    scanned or image-heavy PDFs — I'd add OCR or a vision model for that.
    And the grading is a binary relevance check, not a full reranker.

    If I took it to production, my roadmap would be: swap FAISS for Qdrant
    with metadata filtering and access control, add a cross-encoder reranker
    after retrieval, add hybrid search combining dense and keyword, add
    proper evaluation with RAGAS — faithfulness and context recall — and put
    observability with LangSmith tracing around the whole graph."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHY THIS CLOSING WORKS:
    - States concrete capabilities (not vague).
    - Names real limits WITHOUT undermining the project.
    - The roadmap shows you know what production-grade looks like (Qdrant,
      reranker, hybrid search, RAGAS, LangSmith) — these are senior keywords.
    - It naturally invites the next question, which you're ready for.

THE LANDING LINE (optional, confident close):
    "So in short — it's a full agentic RAG system with self-correction, built
    modular so every piece is swappable from portfolio-grade to production-
    grade. Happy to go deeper on any part."
"""


# =================================================================================
# SECTION 7: FOLDER STRUCTURE — Only if they ASK (~1 minute)
# =================================================================================
"""
IMPORTANT: Do NOT volunteer the folder structure in the main walkthrough.
It's boring and low-signal. But interviewers DO sometimes ask "how is your
project organized?" — so have this ready as a SEPARATE, on-demand answer.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SAY THIS (only when asked):

    "It's organized by RESPONSIBILITY, each as its own package:

      docsage/
        config/    -> all settings in one place: model names, chunk size,
                      retriever k, file types. No magic numbers scattered around.
        state/     -> the AgentState TypedDict, the shared graph state.
        llms/      -> the GroqLLMFactory that builds task-specific LLMs.
        tools/     -> retriever_tool (ingestion), web_search_tool, email_tool.
        nodes/     -> one file per node: agent, grade, generate, rewrite, validate.
        graph/     -> graph_builder, which wires nodes + edges into the StateGraph.
        ui/        -> sidebar (inputs) and chat_interface (streaming chat).
        main.py    -> the orchestrator that connects everything.
      app.py       -> the Hugging Face entry point.

    The principle is separation of concerns: nodes define WHAT each step does,
    the graph builder defines HOW they connect, and main just wires modules
    together. I can change the workflow without touching node logic, or swap
    a model without touching the graph."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DELIVERY TIP:
    Group folders by ROLE (config / state / llms / tools / nodes / graph / ui),
    not file-by-file. Say the PRINCIPLE (separation of concerns) — that's the
    point they're testing, not whether you can read a directory listing.
"""


# =================================================================================
# SECTION 8: THE 60-SECOND VERSION (for "quickly summarize it")
# =================================================================================
"""
Sometimes they want it short, or you're running out of time. Have a tight
one-minute version ready.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SAY THIS:

    "DocSage is an Enterprise Document Intelligence Agent — an agentic RAG
    system built with LangGraph. Users upload documents and ask questions in
    natural language. Under the hood: I ingest documents into a FAISS vector
    index, and a five-node LangGraph workflow handles each query. An agent
    node decides which tool to call, a grader checks if retrieved docs are
    relevant, a generator produces a cited answer, and a validator checks for
    hallucination and relevance — with self-correction loops that rewrite the
    query and retry when something's off. It supports multiple document
    formats, web-search fallback, streaming UI, and conversation memory. It's
    modular, so every component — LLM, vector store, embeddings — is swappable.
    Deployed on Hugging Face Spaces."

That's ~55 seconds at a normal pace. Practice it until it's automatic — it's
also a great answer to "tell me about a project you've built."
"""


# =================================================================================
# SECTION 9: Q&A BANK — What They Ask AFTER the Walkthrough
# =================================================================================
"""
The walkthrough triggers follow-ups. Here are the most likely, with tight
answers grounded in your code. Memorize the soundbites.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ARCHITECTURE / LANGGRAPH
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q: "Why LangGraph and not a simple LangChain chain?"
A: "I needed loops and conditional branching — grade then route, validate then
    retry. A linear chain can't loop back. LangGraph's StateGraph with
    conditional edges models that, and the shared state makes data flow explicit."

Q: "How do you maintain conversation memory?"
A: "I compile the graph with a MemorySaver checkpointer and give each
    conversation a unique thread_id. The state's messages field uses the
    add_messages reducer so turns append instead of overwrite."

Q: "What's in your graph state?"
A: "A TypedDict — messages (with add_messages reducer), documents, generation,
    sources, and a web-search flag. Every node reads and writes this state."

Q: "How does the agent decide which tool to call?"
A: "Each tool has a name and a description. The agent LLM has them bound via
    bind_tools and matches the question to the best tool description. It only
    DECIDES; a ToolNode executes. That's the ReAct pattern."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RAG / RETRIEVAL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q: "What's your chunking strategy?"
A: "RecursiveCharacterTextSplitter at 1000 chars with 200 overlap. The overlap
    keeps facts that straddle a boundary intact. I'd tune it by measuring
    retrieval recall on a test set."

Q: "How do you prevent hallucination?"
A: "Three layers: a relevance grader filters bad chunks before generation, a
    strict 'answer only from context' prompt, and a post-generation
    hallucination check that regenerates or rewrites if the answer isn't
    grounded. That's Corrective plus Adaptive RAG."

Q: "What embedding model and why?"
A: "all-MiniLM-L6-v2 from HuggingFace — local, free, fast, 384-dim. Trade-off
    is slightly lower quality than OpenAI embeddings, but swappable in one line."

Q: "FAISS in production?"
A: "No — FAISS is in-memory, great for a portfolio. In production I'd use
    Qdrant or Pinecone for persistence, metadata filtering, and scaling. The
    ingestion pipeline stays the same; I just swap the store."

Q: "You mentioned re-ranking / adaptive retrieval — explain."
A: "The grading node is my relevance filter, and the rewrite loop adapts the
    query when retrieval is weak. For true cross-encoder reranking I'd add a
    reranker after retrieval — that's on my production roadmap."

Q: "How do you handle a question your documents can't answer?"
A: "The agent falls back to the Tavily web-search tool. If even that doesn't
    help, the generate prompt makes the model say it doesn't have enough
    information rather than inventing an answer."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LLM / RELIABILITY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q: "Why different models for different nodes?"
A: "Tool selection and generation use fast llama-3.1-8b-instant; grading and
    validation use llama-3.3-70b-versatile because structured output needs
    stronger reasoning. A factory centralizes this so I swap models in one place."

Q: "What if the LLM's structured output fails?"
A: "I wrap grading in try/except and default to 'yes' so one flaky structured-
    output call doesn't block the pipeline. The UI also falls back to a direct
    LLM call if the whole graph stream errors out."

Q: "How do you stop infinite self-correction loops?"
A: "A recursion limit of 20 on the graph. If it can't converge, it returns a
    graceful 'try rephrasing' message instead of looping forever."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PRODUCT / DEPLOYMENT / SCALE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q: "How is it deployed?"
A: "Hugging Face Spaces. API keys come from HF Secrets as environment
    variables — no keys in code. The app.py at the root is the HF entry point."

Q: "How would you scale this to thousands of users / millions of docs?"
A: "Move FAISS to a managed vector DB with namespaces for multi-tenancy, run
    ingestion async with a queue and workers, add caching, put it behind an
    API with autoscaling, and add observability and rate-limit handling."

Q: "How do you handle large PDFs or images in PDFs?"
A: "Currently PyPDF handles text well but not scanned/image PDFs. I'd add OCR
    via an Unstructured loader with Tesseract, or use a vision model to
    describe images before indexing."

Q: "Is there security/access control?"
A: "Not in the portfolio version. For production I'd add metadata-filtered
    retrieval so users only search documents they're allowed to see, enforced
    at the retrieval layer, plus PII redaction before indexing."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THE "HOW DID YOU COME UP WITH IT?" QUESTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q: "How did you come up with this project / why did you build it?"
A: "I kept seeing the same pain — people drowning in documents, and basic RAG
    chatbots giving confident wrong answers because they retrieve once and
    never check themselves. I wanted to build something that behaves like a
    careful analyst: pick the right source, verify the evidence is relevant,
    answer with citations, and self-correct when unsure. LangGraph was the
    natural fit because that behavior needs branching and loops, not a
    straight line. So DocSage became my deep-dive into agentic, self-
    correcting RAG done properly."
"""


# =================================================================================
# SECTION 10: THE WHITEBOARD DIAGRAM (draw this while you talk)
# =================================================================================
"""
If there's a whiteboard or screen-share, DRAW as you speak Section 3. A drawn
diagram + narration is the single most senior-looking thing you can do.
Practice drawing this in under 60 seconds.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE DIAGRAM (draw top-to-bottom, left-to-right):

                    +------------------------+
                    |   Streamlit UI         |
                    | (upload + chat + stream)|
                    +-----------+------------+
                                |  user question
                                v
        INGESTION (once)        |
        per uploaded doc:       |
        load -> chunk(1000/200) |
        -> embed(MiniLM)        |
        -> FAISS -> retriever   |
        -> becomes a TOOL ------+--------------------+
                                |                    |
                                v                    |
                          +-----------+              |
              START ----> |  AGENT    |  (LLM + bind_tools)
                          +-----+-----+
                                | tool_call?
                  no tool       | yes
                +---------------+------------------+
                |                                  |
                v                                  v
              [END]                          +-----------+
          (direct answer)                    |  TOOLS    | (ToolNode runs
                                             | (retrieve)|  FAISS search, k=4)
                                             +-----+-----+
                                                   |
                                                   v
                                            +-------------+
                                            |   GRADE     | (relevant? yes/no,
                                            | (Pydantic)  |  structured output)
                                            +--+-------+--+
                                  relevant   |       |  not relevant
                                  "generate" |       | "rewrite"
                                             v       v
                                     +-----------+  +-----------+
                                     | GENERATE  |  |  REWRITE  | --> back to AGENT
                                     | (answer + |  | (reword   |     (loop)
                                     |  sources) |  |  query)   |
                                     +-----+-----+  +-----------+
                                           |
                                           v
                                     +-----------+
                                     | VALIDATE  | (2 checks:
                                     |           |  hallucination + relevance)
                                     +--+----+---+
                          "useful"      |    |  "not useful"
                                        |    +--------------> REWRITE (full retry)
                          "not supported"|
                          (regenerate) <-+
                                        |
                                "useful" v
                                      [END] --> answer + citations to UI

    SIDE TOOLS the agent can also pick:  Tavily web search  |  Gmail email
    CROSS-CUTTING:  MemorySaver checkpointer (thread_id) = conversation memory
                    recursion_limit = 20 (stops infinite loops)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MINIMAL VERSION (if short on time/space, draw just this line):

    START -> AGENT -> TOOLS -> GRADE --(ok)--> GENERATE -> VALIDATE -> END
                        ^         |                            |
                        |     (not ok)                    (not ok)
                        +----- REWRITE <----------------------+

    Then say: "the two backward arrows are my self-correction loops — that's
    what makes it agentic and adaptive rather than one-shot RAG."

DRAWING TIPS:
    - Draw the boxes FIRST, top to bottom, then add the arrows while narrating.
    - Say the node name as you draw each box (audio + visual = memorable).
    - Highlight the TWO backward arrows last and call them out explicitly —
      they are the "wow" of the diagram.
"""


# =================================================================================
# SECTION 11: GOLDEN LESSONS — Delivery That Sounds Senior
# =================================================================================
"""
The content is ready. These are the DELIVERY habits that make it land.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GOLDEN LESSON 1 — ALWAYS GO TOP-DOWN, NEVER BOTTOM-UP.
    Problem -> big picture -> request flow -> deep dive -> trade-offs -> limits.
    Never start with folders or UI widgets. Give the map before the streets.

GOLDEN LESSON 2 — TRACE ONE REQUEST. IT'S THE BEST STRUCTURE.
    "Let me follow one question through the system" is the clearest possible
    narrative. It forces order and proves end-to-end understanding.

GOLDEN LESSON 3 — NAME THE PATTERNS.
    Say "Agentic RAG", "Corrective RAG (CRAG)", "Adaptive/Self-RAG", "ReAct",
    "structured output", "checkpointer". Naming patterns = senior signal.

GOLDEN LESSON 4 — EVERY CHOICE GETS A "WHY" AND A "TRADE-OFF".
    "I chose X because Y; the trade-off is Z; in production I'd do W."
    This one sentence shape is what separates senior from junior.

GOLDEN LESSON 5 — KNOW YOUR NUMBERS COLD.
    5 nodes, chunk 1000/200, k=4, MiniLM (384-dim), recursion limit 20,
    8b for agent/gen, 70b for grading. Specifics = credibility.

GOLDEN LESSON 6 — OWN THE LIMITS.
    FAISS in-memory, no OCR for scanned PDFs, binary grading not reranking.
    Stating limits with a roadmap is MORE impressive than claiming perfection.

GOLDEN LESSON 7 — TIME-BOX YOURSELF.
    30s opener, 30s architecture, 3m flow, 3m deep dive, 1.5m decisions,
    1m results. Practice with a timer. Pause after the flow to let them steer.

GOLDEN LESSON 8 — PAUSE AND LET THEM DRIVE.
    After the request flow, pause: "I can go deeper on the self-correction or
    the retrieval pipeline — what's most useful?" This turns a monologue into
    a conversation and shows confidence.

GOLDEN LESSON 9 — DON'T DUMP EVERYTHING.
    You have 5 deep dives; pick 2-3. You have a full Q&A bank; let them ask.
    Restraint signals seniority; over-explaining signals nervousness.

GOLDEN LESSON 10 — PRACTICE OUT LOUD, NOT IN YOUR HEAD.
    Read Sections 1-6 aloud 5 times with a timer. Record once and listen.
    The goal: the structure becomes muscle memory so you stay calm under
    pressure and never wonder "what do I say next?"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE 6-BEAT SKELETON (memorize this order — it's the whole script):
    1. What + problem        (30s)  "Agentic RAG document assistant, solves X"
    2. 4-layer architecture  (30s)  "UI / ingestion / LangGraph / tools"
    3. Trace one request     (3m)   "agent -> tools -> grade -> generate -> validate"
    4. Deep dive 2-3 parts   (3m)   "state+reducer, self-correction, factory"
    5. Decisions + trade-offs(1.5m) "chose X because Y, trade-off Z"
    6. Results + limits + next(1m)  "works, here are limits, prod roadmap"

If you remember nothing else, remember these 6 beats in order. Everything
else hangs off them.
"""


# =================================================================================
# QUICK REFERENCE CARD (print this / glance before the interview)
# =================================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("DOCSAGE WALKTHROUGH — QUICK REFERENCE CARD")
    print("=" * 70)
    print()
    print("THE 6 BEATS (in order):")
    print("  1. What + problem            (30s)")
    print("  2. 4-layer architecture      (30s)")
    print("  3. Trace one request         (3m)  <- the spine")
    print("  4. Deep dive 2-3 parts       (3m)")
    print("  5. Decisions + trade-offs    (1.5m)")
    print("  6. Results + limits + next   (1m)")
    print()
    print("THE NODE SEQUENCE (say it like a chant):")
    print("  agent -> tools -> grade -> generate -> validate")
    print("            ^                                |")
    print("            +---------- rewrite <------------+  (self-correction)")
    print()
    print("NUMBERS TO KNOW COLD:")
    print("  Nodes: 5 (agent, tools/grade, generate, rewrite, validate)")
    print("  Chunking: 1000 size / 200 overlap | Retriever k: 4")
    print("  Embeddings: all-MiniLM-L6-v2 (local, free, 384-dim)")
    print("  Vector store: FAISS | Recursion limit: 20")
    print("  Models: 8b-instant (agent/gen), 70b-versatile (grade/validate)")
    print("  Memory: MemorySaver + thread_id | Deploy: HF Spaces")
    print()
    print("PATTERNS TO NAME:")
    print("  Agentic RAG | Corrective RAG (CRAG) | Adaptive/Self-RAG")
    print("  ReAct tool selection | structured output | checkpointer")
    print()
    print("DECISION SENTENCE SHAPE:")
    print('  "I chose X because Y. Trade-off is Z. In production I would do W."')
    print()
    print("=" * 70)
    print("Practice Sections 1-6 ALOUD with a timer until it's muscle memory.")
    print("=" * 70)
