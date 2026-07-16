"""
===================================================================================
PHASE 7 — SYSTEM DESIGN FOR AI / AGENTIC SYSTEMS (HashedIn by Deloitte — Lead)
===================================================================================

WHY THIS IS PHASE 7:
    The JD: "Own the end-to-end architecture and technical direction for
    multi-agent systems and AI-driven workflows" + "keen eye for architecture"
    + "translate POCs into well-architected, scalable, production-ready
    solutions." This is the 1-1.5 HOUR ARCHITECTURE ROUND your friends warned
    about — where they say "design X" and drill into every layer.

DEPTH LEVEL: Technical Lead / Architect. The whiteboard round.

HOW TO USE: Section 1 is the METHOD (the framework you apply to ANY design
    question). Sections 2-9 are the building blocks you assemble. Section 10
    is a full worked example. Practice drawing Section 11's reference diagram.

SECTIONS:
    1.  The System Design Method (a repeatable framework for ANY prompt)
    2.  Requirements & Scale Estimation (clarify first)
    3.  The Reference Architecture of a Production AI System (the layers)
    4.  The RAG/Ingestion Subsystem at Scale
    5.  The Serving Layer (API, LLM gateway, streaming)
    6.  Scaling, Caching & Performance
    7.  Reliability, Failure Modes & Graceful Degradation
    8.  Security, Privacy & Governance
    9.  Observability, Evaluation & Cost Management
    10. Worked Example — Design an Enterprise AI Assistant
    11. The Whiteboard Diagram (draw this)
    12. Interview Q&A + GOLDEN LESSONS
===================================================================================
"""


# =================================================================================
# SECTION 1: THE SYSTEM DESIGN METHOD (a repeatable framework)
# =================================================================================
'''
NEVER start drawing boxes randomly. Apply THIS sequence to any "design X"
prompt. Structure is what an architect interviewer scores most.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE 7-STEP METHOD (memorize this order):

    1. CLARIFY REQUIREMENTS (don't skip — interviewers test this)
       - Functional: what must it DO? (the use cases)
       - Non-functional: scale, latency, accuracy, availability, budget, security.
       - Constraints: on-prem vs cloud, data residency, existing stack.

    2. ESTIMATE SCALE (back-of-envelope)
       - Users, requests/sec, data volume, growth. Drives every later decision.

    3. DEFINE THE API / CONTRACT
       - The main endpoints + request/response shapes. The system's boundary.

    4. HIGH-LEVEL ARCHITECTURE (the boxes)
       - Draw the major components + data flow. Top-down, left-to-right.

    5. DRILL INTO EACH COMPONENT
       - Let the interviewer steer; go deep where they probe.

    6. ADDRESS CROSS-CUTTING CONCERNS
       - Scaling, caching, reliability, security, observability, cost.

    7. STATE TRADE-OFFS + EVOLUTION
       - "I chose X because Y, trade-off Z. Start simple; here's how it scales."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE GOLDEN RULES OF THE ARCHITECTURE ROUND:
    - THINK OUT LOUD. They score your reasoning, not just the final diagram.
    - START SIMPLE, THEN SCALE. Design the v1, then "now at 100x, I'd add...".
      Don't over-engineer upfront — that's a junior tell.
    - EVERY CHOICE = A TRADE-OFF. "X because Y, cost Z" on every decision.
    - DRIVE FROM REQUIREMENTS. Scale/latency/accuracy numbers justify choices.
    - IT'S A CONVERSATION. Pause, ask, let them direct the depth.

INTERVIEW ANSWER (how you open a design question):
    "Before I design, let me clarify the requirements — the core use cases, the
    scale in users and requests, latency and accuracy targets, and any
    constraints like cloud versus on-prem or data residency. Then I'll estimate
    scale, sketch the API contract, draw the high-level architecture, drill
    into the components you're most interested in, and cover scaling,
    reliability, security, and cost — stating trade-offs as I go and starting
    with a simple v1 that I then scale."
'''


# =================================================================================
# SECTION 2: REQUIREMENTS & SCALE ESTIMATION
# =================================================================================
'''
The questions to ASK, and the math to do. Skipping this is the #1 mistake.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CLARIFYING QUESTIONS (ask these before designing):
    FUNCTIONAL:
    - Who are the users and what are the top use cases?
    - What data sources feed it? (docs, DBs, APIs)
    - What ACTIONS can it take? (read-only vs write/transact — changes risk)
    - Single-turn or conversational? Multi-tenant?

    NON-FUNCTIONAL:
    - Scale: how many users / requests per second / documents?
    - Latency: interactive (<2s) or batch (minutes OK)?
    - Accuracy/quality bar? Tolerance for wrong answers (compliance domain)?
    - Availability target (99.9%?), DR requirements?
    - Budget: cost per query matters (LLM tokens = money)?
    - Security/privacy: PII, data residency, on-prem requirement?

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BACK-OF-ENVELOPE EXAMPLE:
    "10,000 users, each ~20 queries/day"
    -> 200,000 queries/day ~= 2.3 queries/sec average,
       but PEAK is ~5-10x average -> design for ~20-25 QPS peak.
    -> Each LLM call ~3s + ~2K tokens. At $X/1K tokens -> daily cost estimate.
    -> Docs: 1M documents x 5 chunks = 5M vectors -> sizes the vector DB.

    These numbers DRIVE decisions: 20 QPS with 3s LLM calls = ~60 concurrent
    in flight -> async + a few workers handle it; pgvector is fine at 5M
    vectors; cost estimate justifies caching + a smaller routing model.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

INTERVIEW POINT:
    "I always quantify scale first because it drives every choice. For 10k
    users at 20 queries a day that's roughly 2-3 QPS average but I'd design for
    ~20 QPS peak; with 3-second LLM calls that's about 60 concurrent requests,
    which async plus a few workers handles. A million documents at five chunks
    each is five million vectors, which pgvector handles, so I don't need a
    dedicated vector DB yet. And the token-cost estimate justifies caching and
    a small routing model. Numbers turn architecture from opinion into
    engineering."
'''


# =================================================================================
# SECTION 3: THE REFERENCE ARCHITECTURE OF A PRODUCTION AI SYSTEM
# =================================================================================
'''
The layered architecture you assemble for almost any AI system. Memorize the
layers — they're your mental checklist.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE LAYERS (top to bottom):

    1. CLIENT LAYER — web/mobile/API consumers. Streaming UI for chat.

    2. EDGE / GATEWAY LAYER
       - API Gateway / load balancer: routing, auth, rate limiting, TLS.
       - (Phase 3) the single front door.

    3. APPLICATION / ORCHESTRATION LAYER
       - FastAPI services (Phase 1) + the agent/workflow orchestrator
         (LangGraph). This is where reasoning, routing, and tool-calling live.
       - Stateless app servers (scale horizontally).

    4. AI / MODEL LAYER
       - LLM GATEWAY / abstraction: a single interface in front of providers
         (OpenAI, Anthropic, Azure, Gemini, self-hosted). Enables fallback,
         routing by cost/capability, retries, central rate-limit + token
         accounting. (Tools: LiteLLM, Portkey, or in-house.)
       - Embedding model service.
       - Reranker (optional).

    5. TOOLS / INTEGRATION LAYER
       - MCP servers / tool APIs the agent calls (ERP, CRM, search, email).
       - Web search, calculators, internal services.

    6. DATA / MEMORY LAYER
       - Vector DB (embeddings / RAG knowledge base).
       - Relational DB (source of truth, metadata, transactions) — Phase 4.
       - Cache (Redis: query cache, embedding cache, semantic cache, sessions).
       - Object store (S3: raw documents).
       - Conversation state store (checkpointer: Postgres/Redis).

    7. ASYNC / PIPELINE LAYER
       - Message queue (Kafka/SQS/RabbitMQ) + workers (Celery/Arq) for
         ingestion, embedding, long agentic jobs (Phase 3 async pattern).

    8. CROSS-CUTTING (spans all layers)
       - Observability (LangSmith tracing + Prometheus/Grafana + logs).
       - Security/guardrails (auth, PII redaction, input/output filters).
       - Config/secrets (vault), CI/CD, IaC.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHY THE LLM GATEWAY MATTERS (a senior-favorite component):
    Don't call providers directly from app code. A gateway gives you:
    - PROVIDER ABSTRACTION (swap models without code changes — JD: "LLM SDKs").
    - FALLBACK (provider down -> route to backup).
    - COST/CAPABILITY ROUTING (cheap model for easy, big for hard).
    - CENTRAL rate limiting, retries, caching, token accounting, logging.

INTERVIEW ANSWER:
    "My reference architecture has layers: a client layer with streaming UI; an
    edge layer with an API gateway for auth, routing, and rate limiting;
    stateless FastAPI app servers running the LangGraph orchestrator; an AI
    layer with an LLM gateway that abstracts providers and gives me fallback,
    cost-based routing, and central token accounting; a tools layer of MCP
    servers and integrations; a data layer with a vector DB for RAG, a
    relational DB as source of truth, Redis for caching, object storage for raw
    docs, and a checkpoint store for conversation state; and an async pipeline
    with a queue and workers for ingestion and long jobs. Observability,
    security, and config cut across all of it. The LLM gateway is a piece I
    always include — it lets me swap providers, fail over, and route by cost
    without touching app code."
'''


# =================================================================================
# SECTION 4: THE RAG / INGESTION SUBSYSTEM AT SCALE
# =================================================================================
'''
The data pipeline that feeds the knowledge base. Two paths: ingestion (offline)
and retrieval (online). Ties to your Lessons 21/24.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

INGESTION PIPELINE (offline, async — Phase 3 queue pattern):
    Source (S3/DB/Confluence) -> [queue] -> worker:
      load -> clean -> chunk -> embed (batch) -> upsert to vector DB + metadata.
    At scale:
    - Run as ASYNC workers off a queue (not in the request path).
    - INCREMENTAL: track doc hashes / updated_at; only re-process changes
      (Lesson 24). Avoid full re-index.
    - BATCH embedding calls; cache embeddings.
    - Idempotent upserts (deterministic IDs).
    - Handle failures per-document (one bad file doesn't break the batch).

RETRIEVAL PIPELINE (online, per query):
    query -> embed -> hybrid search (dense + BM25) -> rerank (cross-encoder)
          -> retrieval gate (threshold) -> top-k context -> LLM -> answer + cites.
    At scale:
    - ANN index (HNSW) for fast vector search.
    - Metadata filtering for multi-tenancy + access control.
    - Cache frequent queries (semantic cache).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SCALING THE VECTOR STORE:
    - Small/medium (< few M vectors): pgvector (one less system; metadata joins).
    - Large (10M-1B+): dedicated vector DB (Qdrant/Pinecone/Milvus) with
      sharding, replicas, and HNSW.
    - Multi-tenancy: namespaces (Pinecone) or payload filtering (Qdrant).
    - Keep the relational source of truth separate; vector DB holds embeddings
      + metadata for retrieval.

INTERVIEW POINT:
    "I split RAG into an offline ingestion pipeline and an online retrieval
    path. Ingestion runs as async workers off a queue — load, chunk, batch-embed,
    upsert — with incremental updates by hash so I never full re-index, and
    per-document error handling. Retrieval is embed, hybrid search, rerank, a
    confidence gate, then top-k to the LLM with citations. I start on pgvector
    and move to a dedicated vector DB with HNSW, sharding, and replicas past a
    few million vectors, using metadata filtering for multi-tenancy and access
    control."
'''


# =================================================================================
# SECTION 5: THE SERVING LAYER (API, LLM gateway, streaming)
# =================================================================================
'''
How requests are served in real time — ties Phases 1, 2, 3 together.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE SERVING PATH:
    Client -> Gateway (auth/rate limit) -> FastAPI (async) -> Orchestrator
    (LangGraph) -> [retrieval + LLM gateway + tools] -> stream response back.

KEY DESIGN DECISIONS:
    - STATELESS app servers: conversation state lives in a store (checkpointer),
      not in server memory -> any server handles any request -> scale out.
    - ASYNC end-to-end (Phase 2): async LLM/DB clients so one worker handles
      many concurrent in-flight calls.
    - STREAMING: SSE/WebSocket to stream tokens — critical UX for chat
      (Phase 3). Long agentic jobs -> async pattern (202 + job + poll/webhook).
    - LLM GATEWAY in front of providers (Section 3): fallback + routing + cost.
    - TIMEOUTS/RETRIES/CIRCUIT BREAKERS on every external call (Phase 3).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SYNC vs ASYNC REQUEST HANDLING:
    - Fast interactive query (chat) -> synchronous request, stream the answer.
    - Long pipeline (multi-step agent, report) -> async API: return a job ID,
      process via the queue, client polls or gets a webhook.

INTERVIEW POINT:
    "The serving layer is stateless async FastAPI behind a gateway, with
    conversation state in a checkpoint store so any server handles any request.
    Everything is async end-to-end so a worker keeps many LLM calls in flight,
    and I stream tokens over SSE for chat UX. An LLM gateway sits in front of
    providers for fallback and cost routing, and every external call has a
    timeout, retry, and circuit breaker. For long agentic jobs I switch to the
    async pattern — return a job ID and let the client poll."
'''


# =================================================================================
# SECTION 6: SCALING, CACHING & PERFORMANCE
# =================================================================================
'''
"It works for 100 users. Now make it work for 1 million." The scale-up story.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SCALING THE TIERS:
    - STATELESS APP SERVERS: scale horizontally behind a load balancer;
      Kubernetes HPA autoscales on CPU/latency/queue depth.
    - DATABASE (Phase 4): read replicas, caching, partitioning, then sharding.
    - VECTOR DB: replicas + sharding; HNSW for fast ANN.
    - QUEUE + WORKERS: scale workers independently of API servers.
    - LLM: the usual bottleneck (latency + rate limits + cost) -> caching,
      smaller models, batching, multiple provider keys/regions.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CACHING STRATEGIES (huge for AI cost + latency):
    1. EXACT-MATCH CACHE — identical query -> cached response (Redis).
    2. SEMANTIC CACHE — embed the query; if a past query is within a similarity
       threshold, return its cached answer. Catches paraphrases. (GPTCache.)
    3. EMBEDDING CACHE — cache embeddings of repeated text (don't re-embed).
    4. PROMPT/CONTEXT CACHE — provider-side prompt caching for repeated system
       prompts / context (cuts cost + latency).
    5. RETRIEVAL CACHE — cache retrieved chunks for common queries.
    Cache invalidation: TTL + invalidate on knowledge-base updates.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PERFORMANCE LEVERS:
    - Parallelize independent steps (asyncio.gather).
    - Smaller/faster models for simple steps (routing, grading).
    - Stream to reduce perceived latency.
    - Precompute/batch offline what you can.
    - Right-size context (don't stuff unnecessary tokens — slower + costlier).

INTERVIEW ANSWER:
    "To scale, the app servers are stateless so I scale them horizontally with
    Kubernetes autoscaling; the relational DB scales with replicas, caching,
    and partitioning; the vector DB with replicas and sharding. The real
    bottleneck is the LLM — latency, rate limits, and cost — so caching is my
    biggest lever: exact-match and semantic caching to avoid repeat calls,
    embedding caching, and provider-side prompt caching for repeated context.
    Beyond that, smaller models for simple steps, parallelizing independent
    calls, streaming to cut perceived latency, and right-sizing context so I'm
    not paying for tokens I don't need."
'''


# =================================================================================
# SECTION 7: RELIABILITY, FAILURE MODES & GRACEFUL DEGRADATION
# =================================================================================
'''
A lead designs for failure. AI systems have MORE failure modes than typical
apps (model downtime, hallucination, rate limits).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FAILURE MODES SPECIFIC TO AI SYSTEMS:
    1. LLM PROVIDER DOWN / SLOW / RATE-LIMITED (429/503).
       -> LLM gateway with FALLBACK to a backup provider/model; retries +
          backoff; circuit breaker; queue + retry for non-interactive.
    2. HALLUCINATION / BAD OUTPUT.
       -> grounding + faithfulness check (your Lesson 20); retrieval gate;
          "I don't know" fallback; HITL for high-stakes.
    3. VECTOR DB / TOOL DOWN.
       -> graceful degradation: answer from cache, or general knowledge with a
          disclosure, or a clear error — never crash.
    4. RUNAWAY AGENT LOOP.
       -> recursion/iteration limits, token budget, timeouts (Phase 6).
    5. CONTEXT OVERFLOW.
       -> truncation/summarization/retrieval memory.
    6. DOWNSTREAM (DB/integration) failure.
       -> timeouts, retries, circuit breakers, bulkheads (Phase 3).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RESILIENCE PATTERNS (the toolkit):
    - TIMEOUTS everywhere (no call hangs forever).
    - RETRIES with exponential backoff + jitter (transient errors only).
    - CIRCUIT BREAKER (stop calling a failing dependency; fail fast).
    - BULKHEAD (isolate resources so one failing dependency can't exhaust all).
    - FALLBACK / GRACEFUL DEGRADATION (cached answer, cheaper model, partial
      result, or honest error — degrade, don't crash).
    - IDEMPOTENCY (safe retries for actions).
    - HEALTH CHECKS + auto-restart (K8s liveness/readiness probes).
    - REDUNDANCY (multi-AZ, replicas) for HA; backups + DR plan.

THE PRINCIPLE: "DEGRADE GRACEFULLY." A partial or cached answer beats a 500.

INTERVIEW ANSWER:
    "AI systems have extra failure modes, so I design for them explicitly.
    Provider outages or rate limits — an LLM gateway with fallback to a backup
    model, plus retries, backoff, and a circuit breaker. Hallucination — a
    faithfulness check and retrieval gate with an honest 'I don't know'
    fallback. A vector DB or tool failing — degrade gracefully to a cached
    answer or a clear error rather than crashing. Runaway loops — iteration and
    token limits. Across the board: timeouts everywhere, circuit breakers,
    bulkheads to isolate failures, idempotency for actions, health checks with
    auto-restart, and multi-AZ redundancy. The principle is degrade gracefully —
    a partial or cached answer beats a 500."
'''


# =================================================================================
# SECTION 8: SECURITY, PRIVACY & GOVERNANCE
# =================================================================================
'''
Critical for a Big-4 / enterprise context. AI adds NEW security surfaces.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STANDARD API/APP SECURITY (Phase 3): AuthN/AuthZ, object-level authorization,
    input validation, rate limiting, HTTPS, secrets in vault, least privilege.

AI-SPECIFIC SECURITY SURFACES:
    1. PROMPT INJECTION (direct + indirect via retrieved docs/tool output).
       -> treat retrieved content as DATA not instructions; input/output
          guardrails; constrain tool permissions; sandboxing.
    2. PII / DATA LEAKAGE.
       -> PII detection + redaction (Presidio) before indexing and before
          sending to a third-party LLM; response models that don't over-expose;
          redact logs/traces.
    3. SENSITIVE DATA TO THIRD-PARTY LLMs.
       -> zero-retention agreements, or self-hosted/private models (Azure
          OpenAI, Bedrock) for regulated data; data residency controls.
    4. ACCESS CONTROL IN RAG.
       -> metadata-filtered retrieval so users only see docs they're allowed to
          (enforced at retrieval, not UI).
    5. AGENT ACTIONS.
       -> per-tool least-privilege permissions, HITL for high-risk actions,
          audit logs of every action.

GOVERNANCE (the enterprise/Big-4 angle):
    - Audit trails (who/what/when/why for every answer + action).
    - Data lineage (which doc version produced which answer).
    - Right to be forgotten (delete user data from vector store + logs).
    - Compliance: GDPR, SOC2, industry rules; model/data governance.
    - Guardrail frameworks: NeMo Guardrails, Llama Guard, Guardrails AI.

INTERVIEW ANSWER:
    "On top of standard API security — auth, object-level authorization,
    validation, rate limiting, secrets in a vault — AI adds new surfaces. The
    big one is prompt injection, including indirect injection through retrieved
    documents, so I treat retrieved content as data not instructions and put
    guardrails on inputs and outputs. PII gets detected and redacted before
    indexing and before any third-party LLM call, and for regulated data I use
    private or self-hosted models with data-residency controls. RAG access
    control is metadata-filtered at retrieval so users only see permitted docs,
    and agent actions are least-privilege with human-in-the-loop for risky ones.
    For a Big-4 context, governance is key — audit trails, data lineage, right
    to be forgotten, and compliance like GDPR and SOC2."
'''


# =================================================================================
# SECTION 9: OBSERVABILITY, EVALUATION & COST MANAGEMENT
# =================================================================================
'''
You can't operate what you can't measure. The JD: "performance benchmarks,
... POCs into production." Operations = lead territory.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

OBSERVABILITY (3 pillars + LLM tracing):
    - METRICS: latency p50/p95/p99, QPS, error rate, cache hit rate, queue
      depth (Prometheus/Grafana).
    - LOGS: structured, with request IDs (ELK/CloudWatch).
    - TRACES: distributed tracing (OpenTelemetry) + LLM-specific tracing
      (LangSmith/LangFuse) — see every prompt, tool call, token count, decision.
    - ALERTS: on latency, error rate, faithfulness drop, cost spikes.

AI-SPECIFIC METRICS (beyond infra):
    - Quality: faithfulness, answer relevance, retrieval recall (RAGAS).
    - Agent: steps/task, tool-error rate, loop frequency, success rate.
    - Cost: tokens/request, cost/request, cost/user.
    - User: thumbs up/down, escalation rate, deflection rate.

EVALUATION (continuous, not one-time — your Lessons 17/21):
    - Offline: golden eval set; run RAGAS + LLM-as-judge on every change
      (regression testing for AI).
    - Online: A/B test prompts/models; track real user signals; monitor drift.

COST MANAGEMENT (LLM tokens = real money — a lead concern):
    - Model routing (small for simple, big for hard), caching, context pruning,
      max_tokens caps, batch where possible. Track + budget cost per query;
      alert on spikes. Often the difference between a viable and unviable product.

INTERVIEW ANSWER:
    "Observability has the three pillars — metrics, logs, traces — plus
    LLM-specific tracing with LangSmith so I can see every prompt, tool call,
    and decision. Beyond infra metrics I track AI-specific ones: quality via
    RAGAS faithfulness and retrieval recall, agent metrics like steps per task
    and loop frequency, and cost as tokens and dollars per request. Evaluation
    is continuous — a golden set with RAGAS and LLM-as-judge runs on every
    change as regression testing, plus online A/B tests and drift monitoring.
    And because tokens are real money, I manage cost actively with model
    routing, caching, context pruning, and per-query budgets with alerts — it's
    often what makes an AI product viable."
'''


# =================================================================================
# SECTION 10: WORKED EXAMPLE — Design an Enterprise AI Assistant
# =================================================================================
'''
A full end-to-end design applying the Section 1 method. Adapt this skeleton to
any "design an AI/agentic system" prompt.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PROMPT: "Design an enterprise AI assistant that answers employee questions over
internal docs AND can take actions (create tickets, look up HR/IT data)."

STEP 1 — CLARIFY:
    "How many employees? ~20k. Latency? interactive, <3s to first token.
    Data? Confluence + SharePoint + an HR DB + an IT ticketing system.
    Actions? create tickets, look up records — writes need approval?
    Compliance? PII in HR data, must stay in our cloud (data residency).
    Multi-tenant? single org, but department-level access control."

STEP 2 — SCALE:
    20k employees x ~10 queries/day = 200k/day ~= 2.5 QPS avg, ~20 QPS peak.
    Docs: ~500k -> ~2.5M chunks. -> pgvector is viable; async + a few workers.

STEP 3 — API:
    POST /chat (streaming) {conversation_id, message} -> SSE token stream.
    POST /actions/* (HITL-gated). GET /jobs/{id} for long tasks.

STEP 4 — HIGH-LEVEL ARCHITECTURE:
    Client (web, streaming) -> API Gateway (auth/SSO, rate limit)
      -> FastAPI (async, stateless) -> LangGraph orchestrator:
           Router -> {Knowledge(RAG), HR-data(tools), IT-ticket(action)}
           -> reflection/validation -> respond OR escalate.
      -> LLM Gateway (Azure OpenAI for data residency, fallback model)
      -> Tools via MCP servers (HR DB, ticketing) ; Vector DB (pgvector) ;
         Postgres (metadata + checkpoint state) ; Redis (cache) ; S3 (docs).
      -> Async queue + workers for ingestion + long jobs.

STEP 5 — COMPONENT DRILL (examples):
    - RAG: hybrid search + rerank + retrieval gate + citations; metadata
      filter by department for access control.
    - Action agent: creates tickets via MCP tool; HITL approval for sensitive
      ops; audit-logged.
    - Memory: conversation state checkpointed per conversation_id.

STEP 6 — CROSS-CUTTING:
    Scale: stateless servers + HPA; cache (semantic) for repeat Qs.
    Reliability: LLM fallback, timeouts/retries/circuit breakers, "I don't know".
    Security: SSO + department-level RAG filtering, PII redaction, Azure OpenAI
      (data stays in-cloud), per-tool permissions, audit logs.
    Observability: LangSmith tracing, dashboards (latency, faithfulness, cost).
    Eval: golden Q&A set + RAGAS; thumbs up/down online.

STEP 7 — TRADE-OFFS + EVOLUTION:
    "v1: a single RAG agent + escalation, pgvector, one LLM. Add the action
    agent and HITL once read-only is solid. Move to a dedicated vector DB only
    past a few million vectors. Use Azure OpenAI for residency over cheaper
    APIs — trade cost for compliance. Least autonomy first; expand with need."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

INTERVIEW POINT:
    "I'd clarify scale, data sources, allowed actions, and compliance first,
    estimate ~20 QPS peak and ~2.5M chunks so pgvector suffices, then sketch a
    streaming FastAPI + LangGraph orchestrator with a router delegating to a
    RAG knowledge agent, an HR-data agent over MCP tools, and an HITL-gated
    action agent, behind an LLM gateway on Azure OpenAI for data residency.
    Department-level metadata filtering for access control, PII redaction,
    audit logs, semantic caching, LLM fallback, and LangSmith tracing. I'd ship
    a read-only RAG v1 first and add actions once it's solid — least autonomy
    that works, scaled with need."
'''


# =================================================================================
# SECTION 11: THE WHITEBOARD DIAGRAM (draw this)
# =================================================================================
'''
Practice drawing this in <90s while narrating. A drawn diagram + structured
talk is the strongest signal in an architecture round.

   [ Clients: web / mobile / API ]  (streaming UI)
                |
                v
   [ API GATEWAY ]  auth / SSO / rate limit / TLS / routing
                |
                v
   [ FastAPI app servers (async, STATELESS) ]  <-- horizontal scale (K8s HPA)
                |
                v
   [ ORCHESTRATOR (LangGraph) ]
        router -> [Knowledge/RAG] [Data/Tools] [Action(HITL)] -> validate
                |                 |                |
                v                 v                v
        +---------------+   [ TOOLS via MCP ]  [ Action APIs ]
        | LLM GATEWAY   |   (ERP/CRM/search)   (ticket/email)
        | fallback +    |
        | cost routing  |---> [ Providers: OpenAI/Anthropic/Azure/self-hosted ]
        +---------------+
                |
   ----------------- DATA / MEMORY LAYER -----------------
   [ Vector DB ]   [ Postgres ]   [ Redis cache ]   [ S3 docs ]
   (RAG/HNSW)      (truth +        (query/semantic   (raw
                   checkpoint)      /embedding cache)  documents)
                |
   [ ASYNC: Queue (Kafka/SQS) -> Workers ]  ingestion + long agentic jobs

   CROSS-CUTTING (around everything):
   [ Observability: LangSmith + Prometheus/Grafana + logs ]
   [ Security/Guardrails: authz, PII redaction, input/output filters, audit ]
   [ Config/Secrets (vault) | CI/CD | IaC ]

   DATA FLOW (one query): client -> gateway -> FastAPI -> orchestrator
     -> (retrieve from vector DB + call tools via MCP) -> LLM via gateway
     -> validate -> stream answer + citations back to client.

DRAWING TIPS:
    - Draw TOP-DOWN: clients -> gateway -> app -> orchestrator -> data.
    - Put the LLM GATEWAY as a distinct box (senior signal).
    - Show the ASYNC queue branch separately (ingestion isn't in the request path).
    - Add cross-cutting boxes LAST and say "observability/security span all layers."
    - Narrate the one-query data flow while tracing arrows.
'''


# =================================================================================
# SECTION 12: INTERVIEW Q&A + GOLDEN LESSONS
# =================================================================================
'''
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Q&A
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q1. How do you approach a system design question?
A:  "Clarify requirements + scale, define the API, draw high-level architecture,
    drill into components, cover scaling/reliability/security/cost, state
    trade-offs. Start simple, then scale."

Q2. Why an LLM gateway?
A:  "Provider abstraction + fallback + cost/capability routing + central rate
    limiting, retries, caching, and token accounting — without touching app code."

Q3. How do you make app servers scalable?
A:  "Stateless — conversation state in a checkpoint store, not memory — so any
    server handles any request and I scale horizontally with K8s autoscaling."

Q4. Biggest bottleneck in an AI system and how to handle it?
A:  "The LLM — latency, rate limits, cost. Caching (exact + semantic), smaller
    models for simple steps, parallelization, streaming, and a gateway with
    multiple providers."

Q5. How do you handle an LLM provider outage?
A:  "LLM gateway fails over to a backup provider/model; retries with backoff;
    circuit breaker; queue + retry for non-interactive; cached/degraded answer
    as last resort."

Q6. How do you design for data residency / regulated data?
A:  "Self-hosted or private models (Azure OpenAI/Bedrock), keep data in our
    cloud/VPC, PII redaction, no third-party retention, audit + lineage."

Q7. How do you control cost?
A:  "Model routing, caching, context pruning, max_tokens, batching; track
    cost/query with budgets and alerts."

Q8. How do you scale RAG to 100M+ documents?
A:  "Dedicated vector DB with sharding + replicas + HNSW, async batch
    ingestion off a queue, incremental updates, metadata filtering, retrieval
    caching."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GOLDEN LESSONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. CLARIFY + ESTIMATE SCALE FIRST. Numbers drive every decision. Never skip it.
2. START SIMPLE, THEN SCALE. Design v1, then "at 100x I'd add...". Over-
   engineering upfront is a junior tell.
3. THINK OUT LOUD; IT'S A CONVERSATION. They score reasoning, not the diagram.
4. EVERY CHOICE = TRADE-OFF. "X because Y, cost Z." The JD's #1 theme.
5. KNOW THE LAYERS (your checklist): client -> gateway -> app/orchestrator ->
   LLM gateway -> tools -> data/memory -> async pipeline -> cross-cutting.
6. ALWAYS INCLUDE AN LLM GATEWAY. Abstraction + fallback + cost routing.
7. STATELESS SERVERS + STATE IN A STORE = horizontal scale.
8. THE LLM IS THE BOTTLENECK. Caching + model routing + streaming are your levers.
9. DEGRADE GRACEFULLY. Fallbacks, circuit breakers; a partial answer beats a 500.
10. AI ADDS SECURITY + EVAL + COST SURFACES. Prompt injection, PII, faithfulness,
    tokens-as-money. Owning these = lead-level.

ONE-LINE CHEAT SHEET:
    Method: Clarify -> Scale -> API -> Architecture -> Drill -> Cross-cutting -> Trade-offs.
    Layers: client / gateway / app+orchestrator / LLM gateway / tools / data / async / x-cutting.
    Scale: stateless + autoscale; DB replicas/partition/shard; vector HNSW+shard.
    LLM bottleneck -> cache (exact+semantic) + model routing + stream.
    Reliability -> timeouts + retries + circuit breaker + fallback (degrade gracefully).
    Security -> authz + prompt-injection guards + PII redaction + private models + audit.
    Observe -> metrics+logs+traces+LangSmith; eval (RAGAS); cost/query budgets.
    Always: start simple, scale with need, trade-offs on every choice.
'''


# =================================================================================
# RUN SUMMARY
# =================================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 7 — SYSTEM DESIGN FOR AI / AGENTIC SYSTEMS")
    print("=" * 70)
    print()
    print("THE METHOD (apply to ANY 'design X' prompt):")
    print("  Clarify -> Scale -> API -> Architecture -> Drill -> Cross-cutting -> Trade-offs")
    print()
    print("THE LAYERS (your mental checklist):")
    print("  client -> gateway -> app+orchestrator -> LLM GATEWAY -> tools(MCP)")
    print("  -> data/memory(vector+SQL+cache+S3+checkpoint) -> async queue -> x-cutting")
    print()
    print("KEY MOVES:")
    print("  - Estimate scale FIRST (numbers drive design)")
    print("  - Stateless servers + state in a store -> horizontal scale")
    print("  - LLM gateway: abstraction + fallback + cost routing")
    print("  - LLM is the bottleneck -> cache + model routing + stream")
    print("  - Degrade gracefully (fallback/circuit breaker > 500)")
    print("  - Start simple, scale with need, trade-offs on EVERY choice")
    print()
    print("=" * 70)
    print("Next: Phase 8 — Architecture Trade-offs & Leadership")
    print("=" * 70)
