"""
===================================================================================
NEW CONCEPTS DEEP DIVE — Things You Don't Know Yet (Must Learn for ETech)
===================================================================================

These are concepts from the ETech JD that you haven't practiced hands-on.
Each section explains the concept deeply with examples and code sketches.

SECTIONS:
    1. HITL (Human-in-the-Loop) in LangGraph — Full Implementation
    2. FastAPI WebSocket Streaming — Token-by-Token Delivery
    3. Text-to-SQL — Natural Language to Database Queries
    4. Hybrid Retrieval with Reranking — BM25 + Dense + CrossEncoder
    5. Multi-Tenant Vector Database — Namespace Isolation
    6. Celery + Redis — Async Task Queuing for AI Jobs
    7. AI Guardrails Implementation — Input/Output Validation
    8. GOLDEN LESSONS
===================================================================================
"""


# =================================================================================
# SECTION 1: HITL (Human-in-the-Loop) — Full Implementation
# =================================================================================
"""
WHAT IS HITL?
    The agent workflow PAUSES before a critical action and waits for
    a human to approve, reject, or modify the action.

WHY IS IT NEEDED?
    In production, agents can:
    - Send wrong emails to customers
    - Create incorrect orders worth thousands of dollars
    - Delete important data
    - Make irreversible decisions

    HITL adds a safety gate: "Agent wants to do X. Do you approve?"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HOW IT WORKS IN LANGGRAPH (Step by Step):

    STEP 1: Define your graph with a "critical" node
        builder = StateGraph(AgentState)
        builder.add_node("analyze", analyze_node)
        builder.add_node("send_email", send_email_node)  # CRITICAL action
        builder.add_node("confirm", confirm_node)
        builder.add_edge("analyze", "send_email")
        builder.add_edge("send_email", "confirm")

    STEP 2: Compile with interrupt_before
        graph = builder.compile(
            checkpointer=MemorySaver(),
            interrupt_before=["send_email"]  # PAUSE before this node
        )

    STEP 3: First invoke — runs until interrupt
        config = {"configurable": {"thread_id": "thread-1"}}
        result = graph.invoke({"messages": [HumanMessage("Send report to john@co.com")]}, config)

        # Graph runs: START → analyze → PAUSES (before send_email)
        # State is saved in checkpointer

    STEP 4: Check what the agent wants to do
        # Get the current state
        state = graph.get_state(config)
        # state.next = ["send_email"]  ← this is what's about to execute
        # state.values = current state with the email details

        # Show to user: "Agent wants to send email to john@co.com. Approve?"

    STEP 5: Resume (if approved)
        # Option A: Approve — continue as planned
        graph.invoke(None, config)  # Resumes from where it paused

        # Option B: Reject — skip the action
        graph.invoke(Command(resume={"action": "skip"}), config)

        # Option C: Modify — change the action
        graph.invoke(Command(resume={"recipient": "jane@co.com"}), config)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

REAL-WORLD EXAMPLES:

    EXAMPLE 1: Customer refund agent
        Agent analyzes complaint → determines refund of $500 is appropriate
        → PAUSES → supervisor sees: "Refund $500 to customer #1234. Approve?"
        → Supervisor approves → agent processes refund

    EXAMPLE 2: Document publishing agent
        Agent generates report → formats it → ready to publish
        → PAUSES → editor reviews content for accuracy
        → Editor approves (or edits) → agent publishes

    EXAMPLE 3: Your MCP quotation agent
        Agent creates quote for customer → calculates total $5,000
        → PAUSES → sales manager reviews pricing
        → Manager approves → agent submits quote to ERP

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

COMPLETE CODE EXAMPLE:

    from langgraph.graph import StateGraph, END, START
    from langgraph.checkpoint.memory import MemorySaver
    from langchain_core.messages import HumanMessage
    from typing import TypedDict, Annotated
    from langgraph.graph.message import add_messages

    class State(TypedDict):
        messages: Annotated[list, add_messages]
        email_to: str
        email_body: str
        approved: bool

    def analyze_node(state):
        # Agent decides what email to send
        return {
            "email_to": "john@company.com",
            "email_body": "Your refund of $500 has been processed.",
        }

    def send_email_node(state):
        # This only runs AFTER human approval
        # Actually send the email here
        return {"messages": [AIMessage(content=f"Email sent to {state['email_to']}")]}

    # Build graph
    builder = StateGraph(State)
    builder.add_node("analyze", analyze_node)
    builder.add_node("send_email", send_email_node)
    builder.add_edge(START, "analyze")
    builder.add_edge("analyze", "send_email")
    builder.add_edge("send_email", END)

    # Compile with HITL
    graph = builder.compile(
        checkpointer=MemorySaver(),
        interrupt_before=["send_email"]  # Human must approve before sending
    )

    # Run
    config = {"configurable": {"thread_id": "refund-001"}}
    result = graph.invoke({"messages": [HumanMessage("Process refund for customer")]}, config)

    # At this point, graph is PAUSED. Show user:
    state = graph.get_state(config)
    print(f"Agent wants to email {state.values['email_to']}")
    print(f"Body: {state.values['email_body']}")
    print("Approve? (yes/no)")

    # If approved:
    graph.invoke(None, config)  # Resumes and sends the email

INTERVIEW ANSWER:
    "I implement HITL using LangGraph's interrupt_before. The graph pauses before
    critical nodes, saves state to the checkpointer, and waits for human input.
    The human can approve, reject, or modify the pending action. This is essential
    for production systems — you can't let an agent send emails or process refunds
    without human oversight. The state persistence means the approval can happen
    asynchronously — even hours later — and the graph resumes exactly where it paused."
"""


# =================================================================================
# SECTION 2: FASTAPI WEBSOCKET STREAMING
# =================================================================================
"""
WHAT: Deliver LLM responses token-by-token to the frontend in real-time.

WHY: Users don't want to wait 10 seconds staring at a blank screen.
Streaming shows words appearing as they're generated — feels instant.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

COMPLETE FASTAPI WEBSOCKET EXAMPLE:

    from fastapi import FastAPI, WebSocket
    from langchain_groq import ChatGroq
    from langchain_core.messages import HumanMessage
    import asyncio

    app = FastAPI()
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3, streaming=True)

    @app.websocket("/ws/chat")
    async def chat_endpoint(websocket: WebSocket):
        await websocket.accept()

        try:
            while True:
                # Receive message from client
                user_message = await websocket.receive_text()

                # Stream LLM response token by token
                async for chunk in llm.astream([HumanMessage(content=user_message)]):
                    if chunk.content:
                        await websocket.send_text(chunk.content)

                # Send end-of-message signal
                await websocket.send_text("[END]")

        except Exception as e:
            await websocket.close()

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ALTERNATIVE: SERVER-SENT EVENTS (SSE) — Simpler than WebSocket:

    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel

    app = FastAPI()

    class ChatRequest(BaseModel):
        message: str

    @app.post("/chat/stream")
    async def chat_stream(request: ChatRequest):
        async def generate():
            async for chunk in llm.astream([HumanMessage(content=request.message)]):
                if chunk.content:
                    yield f"data: {chunk.content}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WEBSOCKET vs SSE vs HTTP:

    HTTP:      Request → wait → full response (no streaming)
    SSE:       One-way streaming (server → client only). Simpler. Uses HTTP.
    WebSocket: Two-way streaming (server ↔ client). More complex. Persistent connection.

    For LLM streaming: SSE is usually enough (server sends tokens to client).
    For chat with typing indicators: WebSocket (client also sends "user is typing").

WHAT YOU ALREADY KNOW (DocSage):
    graph.stream(stream_mode="updates") + Streamlit st.status()
    Same concept, different transport. FastAPI WebSocket = production version.

INTERVIEW ANSWER:
    "I implement token-by-token streaming using FastAPI with either WebSocket
    (for bidirectional communication) or Server-Sent Events (for simpler one-way
    streaming). The backend uses LangChain's astream() to get tokens as they're
    generated and pushes each token to the client immediately. In DocSage, I used
    LangGraph's stream_mode='updates' with Streamlit — same concept, just different
    transport layer for production."
"""


# =================================================================================
# SECTION 3: TEXT-TO-SQL — Natural Language to Database Queries
# =================================================================================
"""
WHAT: User asks in English → LLM generates SQL → execute → return results.

EXAMPLE:
    User: "Show me top 5 customers by revenue this month"
    LLM generates: SELECT customer_name, SUM(amount) as revenue
                   FROM orders WHERE date >= '2026-05-01'
                   GROUP BY customer_name ORDER BY revenue DESC LIMIT 5
    Execute → return table.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HOW TO BUILD (Step by Step):

    STEP 1: Connect to database and get schema
        from langchain_community.utilities import SQLDatabase
        db = SQLDatabase.from_uri("postgresql://user:pass@localhost/mydb")
        schema = db.get_table_info()  # Returns CREATE TABLE statements

    STEP 2: Give schema to LLM with the user's question
        prompt = f'''You are a SQL expert. Given this database schema:
        {schema}

        Write a SQL query to answer: {user_question}

        Rules:
        - Return ONLY the SQL query, nothing else
        - Use PostgreSQL syntax
        - Always use LIMIT to prevent huge results
        - Never use DELETE, DROP, or UPDATE (read-only)'''

    STEP 3: LLM generates SQL
        sql_query = llm.invoke(prompt).content

    STEP 4: Validate SQL (IMPORTANT — security!)
        - Check it's a SELECT statement (not DELETE/DROP)
        - Check it doesn't have SQL injection patterns
        - Optionally: parse with sqlparse to verify syntax

    STEP 5: Execute on READ-ONLY connection
        result = db.run(sql_query)

    STEP 6: Return results (or generate natural language summary)
        summary_prompt = f"Summarize these results in plain English: {result}"
        answer = llm.invoke(summary_prompt).content

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LANGCHAIN'S BUILT-IN APPROACH:

    from langchain.chains import create_sql_query_chain
    from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool

    # Create the chain
    write_query = create_sql_query_chain(llm, db)
    execute_query = QuerySQLDataBaseTool(db=db)

    # Full pipeline: question → SQL → execute → answer
    chain = write_query | execute_query
    result = chain.invoke({"question": "Top 5 customers by revenue"})

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CHALLENGES AND SOLUTIONS:

    CHALLENGE 1: LLM generates invalid SQL
        Solution: Try/except around execution. If fails, ask LLM to fix:
        "This SQL failed with error: {error}. Fix it."

    CHALLENGE 2: Security (SQL injection)
        Solution: Read-only DB connection. Whitelist only SELECT statements.
        Never execute DELETE, DROP, UPDATE, INSERT from LLM-generated SQL.

    CHALLENGE 3: Large schemas (100+ tables)
        Solution: Don't give ALL tables to LLM. Use a "table selector" first:
        "Which tables are relevant to this question?" → then give only those schemas.

    CHALLENGE 4: Complex JOINs
        Solution: Provide example queries in the prompt (few-shot).
        "Here's an example of a JOIN query for this schema: ..."

INTERVIEW ANSWER:
    "Text-to-SQL works by giving the LLM the database schema and user's question.
    The LLM generates SQL, which I validate (must be SELECT only, no injection
    patterns) and execute on a read-only connection. For large schemas, I use a
    table selector to give the LLM only relevant tables. I add retry logic — if
    the SQL fails, I send the error back to the LLM to fix. LangChain has
    create_sql_query_chain for this, but I'd add validation and security layers on top."
"""


# =================================================================================
# SECTIONS 4-7: Covered in main lesson (13_etech). Key points here:
# =================================================================================
"""
SECTION 4 — HYBRID RETRIEVAL: See Section 4 of 13_etech lesson.
    Key: BM25 + Dense + RRF + CrossEncoder reranking.

SECTION 5 — MULTI-TENANT: See Section 12 of 13_etech lesson.
    Key: Namespace isolation, metadata filtering, auth-token-based scoping.

SECTION 6 — CELERY + REDIS (Async Task Queuing):
    WHAT: Process long-running AI tasks in the background.
    WHY: Document ingestion (chunking 1000 PDFs) takes minutes.
         You can't make the user wait. Process in background, notify when done.

    HOW:
        # Define a task
        from celery import Celery
        app = Celery('tasks', broker='redis://localhost:6379')

        @app.task
        def ingest_document(doc_id, file_path):
            # Load, chunk, embed, store — takes 30 seconds
            pages = load_document(file_path)
            chunks = split_documents(pages)
            vectorstore.add_documents(chunks)
            return {"status": "done", "chunks": len(chunks)}

        # Queue the task (returns immediately)
        result = ingest_document.delay(doc_id="doc-123", file_path="/uploads/report.pdf")
        # User gets instant response: "Processing... we'll notify you when ready."

        # Check status later
        if result.ready():
            print(result.get())  # {"status": "done", "chunks": 42}

    INTERVIEW ANSWER:
        "For long-running AI tasks like document ingestion, I use Celery with Redis
        as the message broker. The API endpoint queues the task and returns immediately.
        A background worker processes the document (load, chunk, embed, store) and
        updates the status. The user gets notified when processing is complete.
        This prevents API timeouts and keeps the application responsive."

SECTION 7 — AI GUARDRAILS: See Section 11 of 13_etech lesson.
    Key: Input validation (injection detection, PII redaction) +
         Output validation (hallucination check, PII scan, format check) +
         Audit logging for compliance.
"""


# =================================================================================
# SECTION 8: GOLDEN LESSONS
# =================================================================================
"""
GOLDEN LESSON 1: "HITL is what separates demos from production."
    Any agent can work in a demo. In production, critical actions MUST have
    human approval. interrupt_before is your safety net.

GOLDEN LESSON 2: "Streaming is a UX requirement, not a nice-to-have."
    Users expect real-time feedback. A 10-second blank screen = broken app.
    Token-by-token streaming is the minimum bar for production AI apps.

GOLDEN LESSON 3: "Text-to-SQL is powerful but dangerous."
    Always validate generated SQL. Always use read-only connections.
    Never trust LLM-generated queries with write access to your database.

GOLDEN LESSON 4: "Hybrid retrieval is the production standard."
    Pure vector misses keywords. Pure BM25 misses semantics.
    BM25 + Dense + Reranking = what serious RAG systems use.

GOLDEN LESSON 5: "Background processing keeps your app responsive."
    Long tasks (document ingestion, batch analysis) go to Celery workers.
    API returns immediately. User gets notified when done.
    Never block the main thread with heavy AI processing.

GOLDEN LESSON 6: "Multi-tenant isolation is a security requirement."
    In enterprise AI, data leakage between tenants = lawsuit.
    Enforce isolation at infrastructure level (namespaces, not just code).
    Always verify tenant from auth token, never from client input.

GOLDEN LESSON 7: "Your MCP server experience is RARE and VALUABLE."
    Most candidates have built RAG chatbots. Few have built MCP servers
    with 17+ tools, OAuth, and token optimization. This is your edge.
    Lead with it in every interview.
"""

print("=" * 60)
print("New Concepts Deep Dive — ETech Interview Prep")
print("=" * 60)
print()
print("7 Sections + Golden Lessons:")
print("  1. HITL in LangGraph (interrupt_before, approval gates)")
print("  2. FastAPI WebSocket Streaming (token-by-token)")
print("  3. Text-to-SQL (natural language to database)")
print("  4-7. Hybrid Retrieval, Multi-Tenant, Celery+Redis, Guardrails")
print("  8. GOLDEN LESSONS (7 lessons)")
print()
print("These fill your knowledge gaps for the ETech interview.")
print("=" * 60)
