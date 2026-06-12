"""
===================================================================================
ETECH GLOBAL — Sr. AI Application Developer (Interview Prep)
===================================================================================

Company: ETech Global Services / ETS Labs (Texas HQ, India office in Gandhinagar)
Role: Senior AI Application Developer & Architect (Remote)
Experience: 4-7 years
Focus: Production-grade GenAI applications — agentic workflows, RAG, LLM APIs

THIS IS YOUR BEST MATCH SO FAR. The JD literally describes what you've been building.

SECTIONS:
    1.  Company & Role Overview
    2.  Your Experience Mapped to EVERY JD Requirement
    3.  LangGraph Agentic Workflows — HITL, State Persistence, Crash Recovery
    4.  RAG Production Pipeline — Hybrid Retrieval, Reranking, Citations
    5.  LLM API Integration — Pluggable Provider Pattern
    6.  Streaming Responses — FastAPI WebSocket
    7.  Multi-Turn Conversational AI — Memory & Context Compaction
    8.  MCP Server & Tool Orchestration (Your Strength!)
    9.  Text-to-SQL & AI Analytics
    10. AWS Deployment (High-Level)
    11. AI Guardrails & Responsible AI
    12. Multi-Tenant Architecture
    13. 30+ Interview Q&A (Tier 1, 2, 3)
    14. GOLDEN LESSONS

ALSO REVISE:
    - 06_docsage_project_deep_dive.py (full file)
    - 07_rag_complete_guide.py (sections 7-8)
    - 10_jpmc_agentic_dev_prep.py (sections 2-5)
    - _rag_skeleton.py + _agent_skeleton.py
===================================================================================
"""


# =================================================================================
# SECTION 1: COMPANY & ROLE OVERVIEW
# =================================================================================
"""
ETECH GLOBAL SERVICES:
    - Service-based company, HQ in Texas, 3500+ employees, 8 global centers
    - Technology unit: ETS Labs (200+ developers in Gandhinagar, India)
    - Core business: AI-powered contact center solutions, analytics, SaaS products
    - Clients: Contact centers, healthcare, financial services
    - Flagship product: QEval (AI quality assurance platform)
    - SOC 2 Type II certified (security-focused)

THE ROLE (Why it's perfect for you):
    - "Application development role where AI is the core product layer"
    - "NOT a research or model-training position"
    - "Thinks like a software engineer first, uses LLMs as building blocks"
    - Remote position!
    - 4-7 years experience (you have 4 years — fits perfectly)

WHAT THEY BUILD:
    - Conversational AI chatbots for contact centers
    - Agentic assistants for customer support
    - Document intelligence systems (RAG)
    - Real-time AI analytics platforms
    - Text-to-SQL engines
    - Knowledge base platforms

YOUR INTRO (tailored for ETech):
    "Hi, I'm Tirupathi Naidu. I have 4 years of experience in software development,
    with the last 1.5 years focused on building production-grade AI applications.

    I built DocSage — an enterprise document intelligence agent using LangGraph
    with Agentic RAG. It has a 5-node workflow with conditional routing,
    self-correction loops, multi-turn memory, and streaming responses.
    It's deployed live on Hugging Face.

    I also built an MCP server with 17+ LLM-callable tools for ERP integration —
    that's tool orchestration with OAuth authentication, session management,
    and field-level token optimization.

    I think like a software engineer first and use LLMs as building blocks —
    which is exactly what this role describes. I work daily with Python,
    LangGraph, LangChain, FAISS, and Groq API."
"""


# =================================================================================
# SECTION 2: YOUR EXPERIENCE MAPPED TO EVERY JD REQUIREMENT
# =================================================================================
"""
JD REQUIREMENT                              YOUR EXPERIENCE (what to say)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LangGraph agentic workflows                 DocSage: StateGraph, 5 nodes, 3 conditional
                                            edges, tools_condition, MemorySaver.
                                            Agent decides which tool to call.

Multi-step reasoning                        DocSage: agent → retrieve → grade → generate
                                            → validate. Self-correction if quality fails.

Tool orchestration                          MCP server: 17+ tools (search_products,
                                            get_pricing, create_quote, etc.)
                                            Agent orchestrates tool calls autonomously.

HITL approval gates                         Know the concept: interrupt_before in LangGraph.
                                            "Pause before critical actions, wait for human."

LLM API integration (Bedrock, Groq)         Groq API daily. Know Bedrock concepts.
                                            Used pluggable pattern (GroqLLMFactory).

Prompt Engineering                          MCP: YAML agent instructions, few-shot,
                                            constrained output. DocSage: system prompts
                                            for agent, grading, generation, rewriting.

RAG pipeline (chunking, embedding, hybrid)  DocSage: RecursiveCharacterTextSplitter,
                                            HuggingFace embeddings, FAISS, top-k retrieval.
                                            Know hybrid (BM25 + vector) from theory.

Conversational AI (multi-turn memory)       DocSage: MemorySaver + thread_id + compact
                                            context window (last 10 messages).

Streaming responses                         DocSage: graph.stream(stream_mode="updates"),
                                            st.status() for real-time progress.

MCP server development                      Built 17+ tool MCP server with FastMCP!
                                            OAuth, session management, tool orchestration.

Vector databases                            FAISS hands-on. Know Pinecone, Weaviate concepts.

Python backend                              1.5 years daily Python. Know FastAPI basics.

Docker/AWS                                  Concepts: ECR, ECS, Bedrock. MCP server
                                            containerized. DocSage on HuggingFace.

AI guardrails                               DocSage: grade node (quality gate), validate
                                            node (hallucination check). Know prompt
                                            injection prevention from RAG lesson.

Team leadership                             4 years in agile teams. Mentored on GenAI.
                                            Code reviews, sprint planning, JIRA.

Multi-tenant                                Know concept: namespace per tenant in vector DB.
                                            MCP server: session-based user isolation.

"Nice to Have" — MCP server                 YOU HAVE THIS! 17+ tools. Mention prominently.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

KEY ADVANTAGE: The "Nice to Have" section mentions MCP server development.
YOU ALREADY HAVE THIS. Most candidates won't. This is your differentiator.
"""


# =================================================================================
# SECTION 3: LANGGRAPH AGENTIC WORKFLOWS — HITL, STATE, CRASH RECOVERY
# =================================================================================
"""
The JD specifically mentions: "LangGraph-based agentic workflows with multi-step
reasoning, tool orchestration, HITL approval gates, and crash-recovery state persistence."

You already know LangGraph basics (DocSage). Here's what's NEW for this interview:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HITL (Human-in-the-Loop) — APPROVAL GATES:

    WHAT: Pause the agent workflow BEFORE a critical action and wait for
    human approval. The agent suggests an action, human says "yes" or "no."

    WHY: In production, you can't let an agent send emails, create orders,
    or delete data without human confirmation. Too risky.

    HOW IN LANGGRAPH:
        # Compile with interrupt_before
        graph = builder.compile(
            checkpointer=MemorySaver(),
            interrupt_before=["send_email_node"]  # Pause BEFORE this node
        )

        # First invoke — runs until it hits the interrupt
        result = graph.invoke(input, config)
        # Graph is now PAUSED. State is saved in checkpointer.

        # Show user what the agent wants to do:
        # "Agent wants to send email to john@company.com. Approve?"

        # If user approves — resume:
        graph.invoke(Command(resume=True), config)

        # If user rejects — resume with override:
        graph.invoke(Command(resume={"action": "skip"}), config)

    REAL-WORLD EXAMPLE (contact center):
        Agent analyzes customer complaint → suggests refund of $500
        → PAUSES → supervisor reviews → approves/rejects → agent proceeds

    INTERVIEW ANSWER:
        "I implement HITL using LangGraph's interrupt_before parameter. The graph
        pauses before critical nodes (like sending emails or creating orders),
        saves state to the checkpointer, and waits for human approval. The human
        can approve, reject, or modify the action. This is essential for production
        systems where autonomous actions have real consequences."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CRASH-RECOVERY STATE PERSISTENCE:

    WHAT: If the server crashes mid-workflow, the agent can RESUME from where
    it left off instead of starting over.

    WHY: In production, servers crash. Network fails. If a 10-step workflow
    crashes at step 7, you don't want to redo steps 1-6.

    HOW: Use a PERSISTENT checkpointer (not MemorySaver which is in-memory):

        # In-memory (dev only — lost on crash):
        checkpointer = MemorySaver()

        # Persistent (production — survives crashes):
        from langgraph.checkpoint.postgres import PostgresSaver
        checkpointer = PostgresSaver(connection_string="postgresql://...")

        # Or SQLite for simpler setups:
        from langgraph.checkpoint.sqlite import SqliteSaver
        checkpointer = SqliteSaver("checkpoints.db")

    After crash:
        # The graph state is in PostgreSQL/SQLite
        # Just invoke with the same thread_id — it resumes from last checkpoint
        result = graph.invoke(None, {"configurable": {"thread_id": "thread-123"}})
        # Picks up from where it crashed!

    INTERVIEW ANSWER:
        "For crash recovery, I use a persistent checkpointer like PostgresSaver
        instead of MemorySaver. Every node completion saves state to the database.
        If the server crashes mid-workflow, I resume by invoking with the same
        thread_id — the graph picks up from the last completed node. This is
        critical for long-running agentic workflows in production."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STATE PERSISTENCE PATTERNS:

    PATTERN 1: Per-conversation persistence (what DocSage does)
        Each conversation gets a thread_id.
        State persists across multiple invoke() calls within same thread.
        Use case: Multi-turn chatbot.

    PATTERN 2: Per-workflow persistence (for long-running tasks)
        A document processing workflow takes 30 minutes.
        Each step saves state. If it crashes at step 5, resume from step 5.
        Use case: Batch processing, document ingestion pipelines.

    PATTERN 3: Cross-session persistence (for user preferences)
        User's preferences, history, and context persist across sessions.
        Even if they close the browser and come back tomorrow.
        Use case: Personalized assistants.

    CHECKPOINTER OPTIONS:
        MemorySaver → Dev/testing only (lost on restart)
        SqliteSaver → Simple production (single server)
        PostgresSaver → Production (multi-server, scalable)
        RedisSaver → High-performance (fast reads/writes)
"""


# =================================================================================
# SECTION 4: RAG PRODUCTION PIPELINE — HYBRID RETRIEVAL, RERANKING, CITATIONS
# =================================================================================
"""
The JD says: "Build production-grade RAG pipelines integrating vector databases
with hybrid dense + BM25 retrieval. Implement query transformation, reranking,
and LLM-based citation validation."

You know basic RAG. Here's the PRODUCTION-GRADE version:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PRODUCTION RAG PIPELINE (7 stages):

    ┌─────────────────────────────────────────────────────────────────┐
    │  1. QUERY TRANSFORMATION                                         │
    │     User query → improved query (or multiple sub-queries)        │
    │     Techniques: HyDE, query expansion, query decomposition       │
    └──────────────────────────┬──────────────────────────────────────┘
                               ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │  2. HYBRID RETRIEVAL (BM25 + Dense Vector)                       │
    │     BM25: keyword matching (exact terms, names, IDs)             │
    │     Dense: semantic similarity (meaning-based)                    │
    │     Combine with RRF (Reciprocal Rank Fusion)                    │
    │     Returns: top-20 candidates                                   │
    └──────────────────────────┬──────────────────────────────────────┘
                               ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │  3. RERANKING (CrossEncoder or Cohere Rerank)                    │
    │     Takes top-20 candidates → re-scores with cross-encoder       │
    │     Cross-encoder sees query AND document TOGETHER (more accurate)│
    │     Returns: top-4 most relevant                                 │
    └──────────────────────────┬──────────────────────────────────────┘
                               ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │  4. CONTEXT ASSEMBLY                                             │
    │     Format top-4 chunks into a clean context string              │
    │     Add metadata (source, page number, date)                     │
    │     Apply context window limits                                  │
    └──────────────────────────┬──────────────────────────────────────┘
                               ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │  5. GENERATION (LLM produces answer)                             │
    │     RAG prompt: "Answer ONLY from context. Cite sources."        │
    │     LLM generates grounded answer with inline citations          │
    └──────────────────────────┬──────────────────────────────────────┘
                               ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │  6. CITATION VALIDATION (LLM verifies its own citations)         │
    │     "Is claim X actually supported by source Y?"                 │
    │     Remove any citation that can't be verified                   │
    └──────────────────────────┬──────────────────────────────────────┘
                               ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │  7. OUTPUT + AUDIT LOG                                           │
    │     Return answer + sources + confidence                         │
    │     Log: query, retrieved docs, generation, latency, tokens      │
    └─────────────────────────────────────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HYBRID RETRIEVAL — HOW IT WORKS:

    BM25 (Sparse/Keyword):(Best Match 25), a ranking algorithm used in information retrieval systems
        - Traditional text search (like Google before AI)
        - Scores documents by term frequency × inverse document frequency
        - GREAT for: exact matches (invoice numbers, product codes, names)
        - BAD for: semantic similarity ("vacation" vs "time off")

    Dense Vector (Semantic):
        - Embedding-based similarity search
        - GREAT for: meaning-based matching
        - BAD for: exact keyword matches (might miss "INV-2024-0042")

    HYBRID = BOTH combined:
        - Run BM25 search → get top-20 by keywords
        - Run vector search → get top-20 by meaning
        - Combine using RRF (Reciprocal Rank Fusion):
            RRF_score = sum(1 / (k + rank_in_each_list))
        - Result: documents that are relevant by BOTH keyword AND meaning

    WHY HYBRID IS PRODUCTION STANDARD:
        Pure vector misses exact terms. Pure BM25 misses semantics.
        Hybrid catches BOTH. Essential for enterprise data with codes, IDs, names.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RERANKING — WHY AND HOW:

    PROBLEM: Initial retrieval (top-20) is fast but imprecise.
    Some irrelevant docs sneak in. Ordering might be wrong.

    SOLUTION: Reranker re-scores the top-20 with a more accurate model.

    HOW IT WORKS:
        Initial retrieval (bi-encoder): embeds query and docs SEPARATELY
            → fast but less accurate (can't see query+doc together)

        Reranker (cross-encoder): processes query AND document TOGETHER
            → slow but much more accurate (sees the full picture)

        Pipeline: Fast retrieval (top-20) → Accurate reranking (top-4)

    TOOLS:
        - Cohere Rerank API (managed, easy)
        - BGE-reranker (open-source, run locally)
        - CrossEncoder from sentence-transformers (open-source)

    CODE SKETCH:
        from langchain.retrievers import ContextualCompressionRetriever
        from langchain_cohere import CohereRerank

        base_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
        reranker = CohereRerank(top_n=4)
        retriever = ContextualCompressionRetriever(
            base_compressor=reranker,
            base_retriever=base_retriever
        )

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CITATION VALIDATION:

    WHAT: After the LLM generates an answer with citations, VERIFY that
    each citation actually supports the claim.

    WHY: LLMs sometimes cite the wrong source or make up citations.

    HOW:
        1. LLM generates: "Revenue was $10M [Source: Q3 Report, page 5]"
        2. Validation step: "Does Q3 Report page 5 actually mention $10M?"
        3. If YES → keep citation. If NO → remove or flag.

    This is what the JD means by "LLM-based citation validation."

INTERVIEW ANSWER:
    "My production RAG pipeline has 7 stages: query transformation, hybrid
    retrieval (BM25 + dense vector combined with RRF), reranking with a
    cross-encoder for precision, context assembly with metadata, generation
    with citation instructions, citation validation to verify each source
    actually supports the claim, and audit logging. The hybrid approach
    catches both exact keyword matches and semantic similarity, and the
    reranker ensures only the most relevant chunks reach the LLM."
"""


# =================================================================================
# SECTION 5: LLM API INTEGRATION — PLUGGABLE PROVIDER PATTERN
# =================================================================================
"""
The JD says: "Build and integrate LLM APIs (AWS Bedrock, OpenAI, Groq, Gemini, Ollama)
into production backends with pluggable provider abstraction."

WHAT IS PLUGGABLE PROVIDER ABSTRACTION?
    One interface, multiple LLM backends. Switch providers without changing app code.

    WHY: In production, you might need to:
    - Switch from OpenAI to Bedrock (for data privacy)
    - Fall back to Groq if OpenAI is down
    - Use different models for different tasks
    - Let clients choose their preferred provider

    HOW (the pattern you already use!):
        Your GroqLLMFactory in DocSage IS this pattern!
        You create LLMs through a factory — change the provider in ONE place.

    PRODUCTION VERSION:
        class LLMProvider:
            def __init__(self, provider: str, model: str, api_key: str):
                if provider == "groq":
                    self.llm = ChatGroq(model=model, api_key=api_key)
                elif provider == "openai":
                    self.llm = ChatOpenAI(model=model, api_key=api_key)
                elif provider == "bedrock":
                    self.llm = ChatBedrock(model_id=model, region="us-east-1")
                elif provider == "ollama":
                    self.llm = ChatOllama(model=model)  # local, no API key

            def invoke(self, messages):
                return self.llm.invoke(messages)

    PROVIDERS TO KNOW:
        AWS Bedrock — Claude, Llama, Mistral on AWS (data stays in your cloud)
        OpenAI — GPT-4o, GPT-4o-mini (best general quality)
        Groq — Llama, Mixtral (fastest inference, free tier)
        Gemini — Google's models (1M context window)
        Ollama — Run ANY open-source model locally (no API, no cost, full privacy)

    WHEN TO USE WHICH:
        Need privacy (healthcare, finance) → Bedrock or Ollama (data doesn't leave)
        Need best quality → OpenAI GPT-4o
        Need speed + free → Groq
        Need long context → Gemini (1M tokens)
        Need full local control → Ollama

INTERVIEW ANSWER:
    "I use a pluggable provider pattern — a factory class that creates LLM instances
    based on configuration. The application code calls llm.invoke() without knowing
    which provider is behind it. I can switch from Groq to Bedrock by changing one
    config value. In DocSage, my GroqLLMFactory creates task-specific LLMs (agent,
    grading, generation) — same pattern, just extended to multiple providers."
"""


# =================================================================================
# SECTION 6: STREAMING RESPONSES — FASTAPI WEBSOCKET
# =================================================================================
"""
The JD says: "Develop real-time AI features using FastAPI and WebSocket streaming,
delivering token-by-token LLM responses to end users."

WHAT IS STREAMING?
    Instead of waiting 10 seconds for the full response, show each word
    as it's generated. Like watching someone type in real-time.

    WITHOUT streaming: User waits 10 sec → sees full answer at once (bad UX)
    WITH streaming: User sees words appearing one by one (good UX, feels fast)

HOW STREAMING WORKS (3 levels):

    LEVEL 1: LLM streaming (what you already do in DocSage)
        for chunk in llm.stream("What is RAG?"):
            print(chunk.content, end="")  # prints word by word

    LEVEL 2: LangGraph streaming (what DocSage does)
        for update in graph.stream(input, config, stream_mode="updates"):
            # Get node-by-node progress
            for node_name, output in update.items():
                print(f"Node {node_name} completed")

    LEVEL 3: FastAPI WebSocket streaming (what ETech wants)
        The frontend connects via WebSocket. The backend streams tokens
        through the WebSocket in real-time.

        # Backend (FastAPI):
        from fastapi import FastAPI, WebSocket
        app = FastAPI()

        @app.websocket("/ws/chat")
        async def chat_websocket(websocket: WebSocket):
            await websocket.accept()
            user_message = await websocket.receive_text()

            # Stream LLM response token by token
            async for chunk in llm.astream(user_message):
                await websocket.send_text(chunk.content)

            await websocket.close()

        # Frontend (JavaScript):
        const ws = new WebSocket("ws://localhost:8000/ws/chat");
        ws.send("What is RAG?");
        ws.onmessage = (event) => {
            document.getElementById("answer").innerHTML += event.data;
        };

    WHY WEBSOCKET (not regular HTTP)?
        HTTP: Request → wait → full response (one-shot)
        WebSocket: Persistent connection → server pushes data continuously
        For streaming, WebSocket is essential because the server needs to
        PUSH tokens to the client as they're generated.

    ALTERNATIVE: Server-Sent Events (SSE)
        Simpler than WebSocket (one-way: server → client only).
        Works with regular HTTP. Good for streaming text.
        FastAPI supports this with StreamingResponse.

        from fastapi.responses import StreamingResponse

        @app.post("/chat")
        async def chat(request: ChatRequest):
            async def generate():
                async for chunk in llm.astream(request.message):
                    yield f"data: {chunk.content}\n\n"
            return StreamingResponse(generate(), media_type="text/event-stream")

WHAT YOU ALREADY KNOW (DocSage):
    You implemented streaming with Streamlit:
    - graph.stream(stream_mode="updates") for node-by-node progress
    - st.status() for real-time status messages
    - st.empty() to clear status after completion

    For ETech, the same concept but with FastAPI + WebSocket instead of Streamlit.

INTERVIEW ANSWER:
    "I implement streaming using FastAPI WebSocket connections. The client connects
    via WebSocket, sends a message, and the server streams LLM tokens back in
    real-time using async generators. Each token is sent as it's generated —
    the user sees the response building word by word. In DocSage, I used
    LangGraph's stream_mode='updates' with Streamlit for node-by-node progress.
    The same pattern applies with FastAPI — just different transport layer."
"""


# =================================================================================
# SECTION 7: MULTI-TURN CONVERSATIONAL AI
# =================================================================================
"""
The JD says: "Multi-turn conversational memory and sliding window context compaction."

YOU ALREADY KNOW THIS FROM DOCSAGE. Here's the production version:

WHAT IS MULTI-TURN MEMORY?
    The AI remembers previous questions and answers in the conversation.
    "What is RAG?" → answer → "How does it prevent hallucination?" → AI knows
    "it" refers to RAG from the previous turn.

THREE MEMORY STRATEGIES:

    STRATEGY 1: FULL HISTORY (simple but expensive)
        Send ALL previous messages to the LLM every time.
        Problem: Token count grows with every turn. Eventually overflows.

    STRATEGY 2: SLIDING WINDOW (what DocSage uses)
        Keep only the last N messages. Older messages are dropped.
        DocSage: last 10 messages (5 Q&A turns).
        Pros: Fixed token cost. Cons: Loses very old context.

    STRATEGY 3: SUMMARY MEMORY (best for long conversations)
        Periodically summarize older messages into a short summary.
        Keep: summary + last 5 messages.
        The summary preserves key facts without using many tokens.

        Example:
            Messages 1-20 → summarized to: "User asked about RAG, DocSage project,
            and deployment. Key facts: uses FAISS, Groq API, deployed on HuggingFace."
            Messages 21-25 → kept in full.

    STRATEGY 4: CONTEXT COMPACTION (what the JD mentions)
        Same as sliding window but smarter:
        - Keep human messages and AI text responses
        - SKIP ToolMessages (large raw document chunks)
        - SKIP AI messages that are just tool_calls (no text content)
        - This is EXACTLY what DocSage's agent_node does!

        compact = []
        for msg in messages:
            if msg.type == "human":
                compact.append(msg)
            elif msg.type == "ai" and msg.content.strip():
                compact.append(msg)
        compact = compact[-10:]  # last 10 relevant messages

SESSION MANAGEMENT:
    Each user/conversation gets a unique session_id (thread_id in LangGraph).
    State is isolated per session — User A's conversation doesn't leak into User B's.

    In production:
    - Generate UUID for each new conversation
    - Store in PostgresSaver (persistent across server restarts)
    - Clean up old sessions after 24 hours (TTL)

INTERVIEW ANSWER:
    "I implement multi-turn memory using LangGraph's MemorySaver with thread_id
    for session isolation. For context compaction, I keep only the last 10
    human/AI text messages and skip large ToolMessages to stay within token
    limits. In production, I'd use PostgresSaver for persistence and add
    summary memory for very long conversations — periodically summarizing
    older messages to preserve key facts without token overflow."
"""


# =================================================================================
# SECTION 8: MCP SERVER & TOOL ORCHESTRATION (YOUR STRENGTH!)
# =================================================================================
"""
The JD "Nice to Have" says: "Experience with MCP (Model Context Protocol) server
development and tool orchestration."

THIS IS YOUR DIFFERENTIATOR. Most candidates won't have this. Lead with it.

WHAT YOU BUILT:
    - MCP server using Python FastMCP
    - 17+ LLM-callable tools for ERP integration
    - Tools cover: product search, pricing, quote creation, order management,
      customer credit, AI recommendations
    - OAuth authentication with session management
    - Field-level token optimization (reduced tokens 60-70%)
    - YAML-based agent instructions for each agent type

HOW MCP WORKS (explain this clearly):
    MCP = Model Context Protocol (open standard by Anthropic)

    Architecture:
        LLM (Claude/GPT) ←→ MCP Protocol (JSON-RPC) ←→ MCP Server (your tools)

    The MCP server EXPOSES tools with:
    - Name: "search_products"
    - Description: "Search products by keyword, returns name, price, availability"
    - Parameters: {"query": str, "category": str, "max_results": int}

    The LLM reads these descriptions and DECIDES which tool to call.
    The MCP server EXECUTES the tool and returns results.

YOUR TOKEN OPTIMIZATION (the wow factor):
    Problem: ERP APIs return 50+ fields per object (2000 tokens).
    Solution: Field-level relevance filtering.
    - Before sending API response to LLM, filter to only relevant fields
    - User asked about credit → send only {credit_limit, credit_used, available}
    - Not all 50 fields (address, phone, fax, tax_id, etc.)
    - Result: 60-70% fewer tokens per tool call

    At scale (thousands of queries/day), this saves significant API cost.

INTERVIEW ANSWER:
    "I built a production MCP server with 17+ LLM-callable tools for ERP integration.
    The server uses FastMCP with OAuth authentication and session management. Each
    tool has a name, description, and typed parameters — the LLM reads descriptions
    and decides which to call. My key optimization was field-level relevance filtering:
    ERP APIs return 50+ fields, but I filter to only the 3-5 fields relevant to the
    user's question before sending to the LLM. This reduced token consumption by
    60-70%. I also used YAML-based agent instructions to define each agent's role,
    available tools, and output constraints."
"""


# =================================================================================
# SECTION 9: TEXT-TO-SQL & AI ANALYTICS
# =================================================================================
"""
The JD says: "Build Text-to-SQL engines, automated data visualization pipelines."

WHAT IS TEXT-TO-SQL?
    User asks in natural language → LLM converts to SQL → executes on database → returns results.

    User: "Show me total revenue by product category for Q3 2024"
    LLM generates: SELECT category, SUM(revenue) FROM sales
                   WHERE quarter = 'Q3' AND year = 2024
                   GROUP BY category ORDER BY SUM(revenue) DESC
    Execute → return table of results.

HOW TO BUILD IT:
    1. Give the LLM the DATABASE SCHEMA (table names, columns, types)
    2. User asks a question in natural language
    3. LLM generates SQL based on schema + question
    4. Execute SQL on the database
    5. Return results (or visualize as chart)

    Code sketch:
        from langchain_community.utilities import SQLDatabase
        from langchain.chains import create_sql_query_chain

        db = SQLDatabase.from_uri("postgresql://user:pass@localhost/mydb")
        chain = create_sql_query_chain(llm, db)
        sql_query = chain.invoke({"question": "Total revenue by category in Q3?"})
        result = db.run(sql_query)

CHALLENGES:
    - LLM might generate invalid SQL → add validation step
    - Security: SQL injection risk → use read-only DB connection
    - Complex queries: LLM struggles with JOINs across 5+ tables
    - Solution: Give the LLM only relevant table schemas, not the entire DB

INTERVIEW ANSWER:
    "Text-to-SQL works by giving the LLM the database schema and the user's
    natural language question. The LLM generates SQL, which is validated and
    executed on a read-only connection. I'd use LangChain's create_sql_query_chain
    with schema filtering — only show relevant tables to the LLM. For complex
    queries, I'd add a validation step that checks SQL syntax before execution
    and a retry mechanism if the query fails."
"""


# =================================================================================
# SECTION 10: AWS DEPLOYMENT (High-Level)
# =================================================================================
"""
The JD says: "Deploy on AWS (ECS Fargate, Lambda) via CI/CD pipelines."

YOU DON'T NEED DEEP AWS EXPERTISE. Know the concepts and what each service does.

AWS SERVICES FOR AI APPLICATIONS:

    SERVICE         WHAT IT DOES                        WHEN TO USE
    Bedrock         Managed LLM access (Claude, Llama)  LLM calls without managing GPUs
    ECS Fargate     Run Docker containers (serverless)  Deploy your FastAPI app
    Lambda          Serverless functions                 Lightweight tasks, triggers
    S3              Object storage                      Store documents, models, data
    Secrets Manager Store API keys securely             Never hardcode secrets
    CloudWatch      Monitoring and logging              Track errors, latency, costs
    SQS             Message queue                       Async task processing
    RDS             Managed PostgreSQL/MySQL            Application database

DEPLOYMENT FLOW:
    Code → Docker image → Push to ECR → Deploy on ECS Fargate → Load balancer → Users

    1. Write Dockerfile for your FastAPI app
    2. Build Docker image: docker build -t my-ai-app .
    3. Push to ECR (AWS container registry)
    4. ECS Fargate runs the container (no server management)
    5. Application Load Balancer routes traffic
    6. CloudWatch monitors health and logs

WHY BEDROCK (not direct OpenAI):
    - Data stays within AWS (compliance: HIPAA, SOC 2)
    - No data sent to external APIs
    - Access to Claude, Llama, Mistral — all in one place
    - Pay per token, no subscription

INTERVIEW ANSWER:
    "I'd deploy the AI application as a Docker container on ECS Fargate —
    serverless container orchestration with no server management. LLM calls
    go through AWS Bedrock for data privacy (data stays in AWS). Secrets
    in AWS Secrets Manager, monitoring in CloudWatch, documents in S3.
    CI/CD pipeline: code push → build Docker image → push to ECR → deploy
    to ECS automatically. For async tasks (document ingestion), I'd use
    SQS + Lambda or Celery + Redis."
"""


# =================================================================================
# SECTION 11: AI GUARDRAILS & RESPONSIBLE AI
# =================================================================================
"""
The JD says: "Implement AI guardrails, prompt injection prevention, output validation,
PII detection, audit logging, traceability."

GUARDRAILS = Safety checks BEFORE and AFTER LLM calls.

INPUT GUARDRAILS (before LLM processes the query):
    1. Prompt injection detection:
       - Detect patterns: "ignore previous instructions", "reveal system prompt"
       - Use a classifier or regex to flag suspicious inputs
       - Block or sanitize before sending to LLM

    2. PII detection in input:
       - Detect: credit cards (16 digits), SSN, phone numbers, emails
       - Redact before sending to LLM: "My card is ****-****-****-1234"

    3. Topic filtering:
       - Block queries outside the system's scope
       - "Tell me how to hack a server" → blocked

OUTPUT GUARDRAILS (after LLM generates response):
    1. PII detection in output:
       - Scan response for accidentally leaked PII
       - Mask before returning to user

    2. Hallucination check:
       - Verify claims against retrieved documents
       - Remove unsupported statements

    3. Content moderation:
       - Check for toxic, harmful, or inappropriate content
       - Block or flag for human review

    4. Format validation:
       - Ensure output matches expected format (JSON, specific structure)
       - Retry if format is wrong

AUDIT LOGGING (for compliance):
    Every AI interaction is logged:
    - Timestamp, user_id, session_id
    - Input query (redacted PII)
    - Retrieved documents
    - LLM response
    - Guardrail decisions (what was blocked/flagged)
    - Latency, token count, cost

    WHY: In healthcare/finance, you must be able to explain WHY the AI
    gave a specific answer. Audit logs provide traceability.

INTERVIEW ANSWER:
    "I implement guardrails at both input and output layers. Input: prompt
    injection detection (pattern matching + classifier), PII redaction before
    LLM processing, and topic filtering. Output: PII scanning, hallucination
    verification against source documents, content moderation, and format
    validation. Every interaction is audit-logged with timestamp, user, query,
    response, and guardrail decisions for compliance traceability. In DocSage,
    my grade node and validate node serve as guardrails — they catch irrelevant
    retrieval and hallucinated answers before they reach the user."
"""


# =================================================================================
# SECTION 12: MULTI-TENANT ARCHITECTURE
# =================================================================================
"""
The JD says: "Design multi-tenant document ingestion pipelines with per-user isolation."

WHAT IS MULTI-TENANT?
    Multiple customers (tenants) share the same application infrastructure,
    but their DATA is completely isolated. Customer A cannot see Customer B's documents.

    Example: ETech builds ONE RAG platform. 50 different companies use it.
    Each company's documents are isolated — Company A's invoices are invisible to Company B.

HOW TO IMPLEMENT IN VECTOR DATABASES:

    APPROACH 1: Separate collections per tenant
        tenant_a_collection = "docs_tenant_a"
        tenant_b_collection = "docs_tenant_b"
        # Each tenant has their own vector store
        # Simple but doesn't scale well (1000 tenants = 1000 collections)

    APPROACH 2: Namespace isolation (Pinecone)
        # One index, multiple namespaces
        index.upsert(vectors, namespace="tenant_a")
        index.query(query_vector, namespace="tenant_a")  # only searches tenant_a
        # Efficient, scalable, built-in isolation

    APPROACH 3: Metadata filtering (Weaviate, OpenSearch)
        # One collection, filter by tenant_id
        results = vectorstore.similarity_search(
            query,
            filter={"tenant_id": "tenant_a"}  # only returns tenant_a's docs
        )
        # Most scalable, but relies on filter correctness

    APPROACH 4: Row-level security (PostgreSQL)
        # Database enforces isolation at the query level
        # Even if code has a bug, wrong tenant's data can't be accessed

SECURITY CONSIDERATIONS:
    - NEVER trust client-side tenant_id (always verify from auth token)
    - Test: "Can tenant A access tenant B's data?" (must always be NO)
    - Audit: Log all cross-tenant access attempts
    - Encryption: Encrypt data at rest per tenant (separate keys)

INTERVIEW ANSWER:
    "For multi-tenant RAG, I use namespace isolation in the vector database —
    each tenant's documents are stored in a separate namespace. Queries are
    scoped to the tenant's namespace only, enforced server-side from the
    authenticated user's token (never from client input). I also add metadata
    filtering as a second layer and audit logging for any cross-tenant access
    attempts. The key principle: isolation must be enforced at the infrastructure
    level, not just the application level."
"""


# =================================================================================
# SECTION 13: 30+ INTERVIEW Q&A
# =================================================================================
"""
TIER 1 — MUST NAIL (these will definitely be asked):

Q1: "Walk me through an agentic system you built."
A: [Use your DocSage explanation from lesson 06 — 5 nodes, conditional routing,
   self-correction, MemorySaver, streaming. Keep it under 2 minutes.]

Q2: "How do you design a production RAG pipeline?"
A: "7 stages: query transformation, hybrid retrieval (BM25 + dense), reranking
   with cross-encoder, context assembly, generation with citation instructions,
   citation validation, and audit logging."

Q3: "What is LangGraph and why use it over LangChain agents?"
A: "LangGraph gives full control over workflow — conditional routing, state
   persistence, HITL gates, and crash recovery. LangChain's AgentExecutor is
   a simple loop that can't do conditional branching or post-generation validation."

Q4: "How do you handle multi-turn conversations?"
A: "MemorySaver with thread_id for session isolation. Context compaction: keep
   last 10 human/AI messages, skip ToolMessages. For long conversations, add
   summary memory to compress older context."

Q5: "Tell me about your MCP server."
A: "17+ LLM-callable tools for ERP integration. FastMCP with OAuth authentication.
   Field-level token optimization reduced tokens 60-70%. YAML agent instructions
   define each agent's role and constraints."

Q6: "How do you prevent hallucination?"
A: "Three layers: RAG prompt says 'answer ONLY from context', grade node filters
   irrelevant docs before generation, validate node checks if answer is grounded
   after generation. If hallucination detected, self-correction loop retries."

TIER 2 — SHOULD KNOW:

Q7: "What is HITL in LangGraph?"
A: "interrupt_before pauses the graph before critical nodes. State saves to
   checkpointer. Human reviews and approves/rejects. Graph resumes on approval."

Q8: "How do you handle crash recovery?"
A: "Persistent checkpointer (PostgresSaver). Every node saves state. On crash,
   resume with same thread_id — picks up from last completed node."

Q9: "How would you implement streaming with FastAPI?"
A: "WebSocket connection for bidirectional streaming. Client connects, sends
   message, server streams LLM tokens back using async generator. Each token
   sent as it's generated."

Q10: "What is hybrid retrieval?"
A: "BM25 (keyword) + dense vector (semantic) combined with RRF. BM25 catches
    exact terms, vector catches meaning. Together they outperform either alone."

Q11: "What is reranking?"
A: "Cross-encoder re-scores top-20 retrieved docs. More accurate than initial
    retrieval because it sees query AND document together. Returns top-4."

Q12: "How do you handle multi-tenant data isolation?"
A: "Namespace isolation in vector DB. Queries scoped to tenant's namespace.
    Tenant ID from auth token (never client input). Audit logging for access."

Q13: "What is Text-to-SQL?"
A: "LLM converts natural language to SQL using database schema as context.
    Validate SQL before execution. Use read-only connection for security."

TIER 3 — NICE TO HAVE:

Q14: "What AI guardrails do you implement?"
A: "Input: prompt injection detection, PII redaction, topic filtering.
    Output: PII scanning, hallucination check, content moderation, format validation.
    All interactions audit-logged for compliance."

Q15: "How do you optimize token costs at scale?"
A: "Field-level filtering (send only relevant fields), model selection per task
    (cheap model for simple, expensive for complex), batch processing, caching,
    output length constraints."

Q16: "What is the pluggable provider pattern?"
A: "Factory class creates LLM instances based on config. App code calls llm.invoke()
    without knowing the provider. Switch Groq to Bedrock by changing one config value."

Q17: "How do you test agentic systems?"
A: "Four levels: unit tests for tools, integration tests for routing, end-to-end
    tests for full workflows, evaluation at scale with metrics (faithfulness,
    tool selection accuracy)."

Q18: "What is context compaction?"
A: "Keep only relevant messages in the conversation history. Skip large ToolMessages,
    keep human questions and AI text answers. Prevents token overflow while
    preserving follow-up context."

Q19: "Difference between MemorySaver and PostgresSaver?"
A: "MemorySaver: in-memory, lost on restart (dev only). PostgresSaver: persistent,
    survives crashes, supports multi-server deployment (production)."

Q20: "How would you handle a document ingestion pipeline for 10,000 docs?"
A: "Async processing with Celery + Redis. Queue documents, process in parallel
    workers. Each worker: load → chunk → embed → store. Track progress per document.
    Retry failed documents. Notify user on completion."
"""


# =================================================================================
# SECTION 14: GOLDEN LESSONS
# =================================================================================
"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GOLDEN LESSON 1: "Think like a software engineer first, use LLMs as building blocks."
    This is LITERALLY in the JD. Don't position yourself as "AI researcher" or
    "prompt writer." You're a SOFTWARE ENGINEER who builds applications where
    AI is the core layer. You write production code, handle errors, deploy,
    monitor, and scale.

GOLDEN LESSON 2: "MCP is your secret weapon."
    The JD lists MCP as "Nice to Have." Most candidates won't have it.
    YOU BUILT A 17-TOOL MCP SERVER. Lead with this. It's your differentiator.

GOLDEN LESSON 3: "Production = reliability, not just accuracy."
    Demo: "It works!" (95%)
    Production: "It works, handles errors, recovers from crashes, scales,
    is secure, is auditable, and costs are controlled." (100%)
    Show you think about the FULL picture, not just the happy path.

GOLDEN LESSON 4: "Every system you add multiplies error probability."
    0.9 × 0.9 × 0.9 = 0.729. Keep architectures simple. Each component
    must justify its existence. Sometimes context stuffing beats complex RAG.

GOLDEN LESSON 5: "Hybrid retrieval is the production standard."
    Pure vector search misses exact keywords. Pure BM25 misses semantics.
    Hybrid + reranking is what production systems use. Know this cold.

GOLDEN LESSON 6: "HITL is non-negotiable in production."
    Agents that send emails, create orders, or modify data without human
    approval are DANGEROUS. Always add approval gates for critical actions.

GOLDEN LESSON 7: "Token optimization is a business requirement, not a nice-to-have."
    At scale (thousands of queries/day), every unnecessary token costs money.
    Field-level filtering, model selection per task, output constraints,
    caching — these are engineering decisions that save thousands per month.

GOLDEN LESSON 8: "Remote work = communication is everything."
    This is a remote role. They'll assess: Can you communicate clearly?
    Can you work independently? Can you document your decisions?
    Be clear, structured, and proactive in your answers.
"""

print("=" * 60)
print("ETech Sr. AI Application Developer — Interview Prep")
print("=" * 60)
print()
print("14 Sections:")
print("  1.  Company & Role Overview")
print("  2.  Your Experience Mapped to JD")
print("  3.  LangGraph — HITL, State Persistence, Crash Recovery")
print("  4.  RAG Production — Hybrid Retrieval, Reranking, Citations")
print("  5.  LLM API — Pluggable Provider Pattern")
print("  6.  Streaming — FastAPI WebSocket")
print("  7.  Multi-Turn Conversational AI")
print("  8.  MCP Server (Your Differentiator!)")
print("  9.  Text-to-SQL & AI Analytics")
print("  10. AWS Deployment (High-Level)")
print("  11. AI Guardrails & Responsible AI")
print("  12. Multi-Tenant Architecture")
print("  13. 20 Interview Q&A (Tier 1, 2, 3)")
print("  14. GOLDEN LESSONS (8 lessons)")
print()
print("MUST FOCUS: Sections 3, 4, 5, 8 (your strengths + new concepts)")
print("=" * 60)
