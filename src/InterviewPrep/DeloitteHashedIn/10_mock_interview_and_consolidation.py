"""
===================================================================================
PHASE 10 — MOCK INTERVIEW & CONSOLIDATION (HashedIn by Deloitte — Lead)
===================================================================================

WHY THIS PHASE:
    The final piece. HashedIn/Deloitte rounds run 1-1.5 hours and drill DEEP,
    mixing coding, design, theory, and follow-ups. This phase: (1) simulates
    that pressure with a full mock, (2) consolidates all 9 phases into a
    one-page glance card, (3) gives the delivery habits that win deep-dive rounds.

HOW TO USE:
    - Read Section 1 (the round map) so nothing surprises you.
    - Do the Section 2 mock OUT LOUD, timed, no notes. Then check answers.
    - Drill Section 6 (the consolidated cheat card) before the interview.

SECTIONS:
    1.  The HashedIn/Deloitte Round Map (what to expect)
    2.  Full Mock Interview (rapid-fire across all 9 phases)
    3.  A Coding Round Simulation (the live-coding pattern)
    4.  A System Design Round Simulation (the architecture deep-dive)
    5.  Handling Deep Follow-Ups & "I Don't Know"
    6.  THE CONSOLIDATED CHEAT CARD (all 9 phases, one page)
    7.  Final Delivery Habits + GOLDEN LESSONS
===================================================================================
"""


# =================================================================================
# SECTION 1: THE HASHEDIN/DELOITTE ROUND MAP (what to expect)
# =================================================================================
'''
Know the shape so the 1.5-hour deep-dive doesn't surprise you.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TYPICAL ROUNDS FOR A LEAD PYTHON/GenAI ROLE:
    1. SCREEN — recruiter + a quick technical filter.
    2. TECHNICAL DEEP-DIVE (1-1.5h) — Python + FastAPI + async + DB + coding;
       drills DEEP with follow-ups. (Phases 1-5.)
    3. AI / SYSTEM DESIGN (1h) — agentic systems, RAG, "design X", architecture.
       (Phases 6, 7.)
    4. MANAGERIAL / LEADERSHIP — project deep-dive, trade-offs, mentoring,
       estimation, behavioral (STAR). (Phase 8.)
    5. HR / culture fit.

    NOTE: exact structure varies; rounds often blend. Expect coding + design +
    theory + behavioral, all probing depth.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHAT THEY'RE TESTING (the meta-signals):
    - DEPTH: they keep asking "why" / "how" / "what if" until you hit your limit.
      They WANT to find the edge of your knowledge — that's the point.
    - TRADE-OFF THINKING: every decision justified, alternatives known.
    - COMMUNICATION: can you explain clearly, structure an answer, think aloud.
    - HONESTY: admitting "I don't know, here's how I'd find out" beats bluffing.
    - LEADERSHIP: ownership, mentoring, delivery — it's a LEAD role.

WHY PEOPLE FAIL (your friends' warning — and how to avoid it):
    - Shallow answers that collapse under the 2nd/3rd follow-up
      -> FIX: the depth in Phases 1-9; know the "why" beneath the "what".
    - Memorized facts, no reasoning -> FIX: think aloud, derive, not recite.
    - No trade-offs ("always use X") -> FIX: "it depends on [factors]".
    - Bluffing when stuck -> FIX: honest + a reasoned approach (Section 5).
    - Rambling, no structure -> FIX: frameworks (Phase 7 method, STAR).

INTERVIEW MINDSET:
    "They drill deep ON PURPOSE to find your ceiling — that's normal, not a
    sign you're failing. Stay calm, reason aloud, give trade-offs, and when you
    hit your edge, be honest and show how you'd reason it out. Depth + structure
    + honesty wins these rounds."
'''


# =================================================================================
# SECTION 2: FULL MOCK INTERVIEW (rapid-fire across all 9 phases)
# =================================================================================
'''
Do this OUT LOUD, timed (~1 min/answer), NO notes. Then read the model answer.
This simulates the rapid follow-up style. Cover answers; speak first.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ PHASE 1: FASTAPI ━━━━━━━━━━━━━━━━━━

Q: "Walk me through what happens when a request hits your FastAPI endpoint."
A: Server(uvicorn)->middleware->routing->dependency resolution->validation->
   handler (async on loop / sync in thread pool)->response_model->middleware->out.

Q: "async def or def for an endpoint that calls a sync DB driver?"
A: "def — FastAPI runs it in a thread pool so it won't block the loop. Or
   switch to an async driver and use async def. NEVER a sync blocking call
   inside async def."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ PHASE 2: ASYNC ━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q: "Why is asyncio good for LLM apps but not for image processing?"
A: "LLM calls are I/O-bound — async interleaves the waits, one worker handles
   thousands. Image processing is CPU-bound — it blocks the single-threaded
   loop; that needs multiprocessing for true parallelism."

Q: "What does the GIL mean for threading?"
A: "One thread runs Python bytecode at a time. No CPU speedup from threads, but
   I/O-bound threading helps because the GIL releases during I/O."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ PHASE 3: API DESIGN ━━━━━━━━━━━━━━━━━━━━━━

Q: "401 vs 403? Idempotency?"
A: "401 = not authenticated; 403 = authenticated but not authorized.
   Idempotent (GET/PUT/DELETE) = same result if retried; POST isn't, so use
   idempotency keys for safe create retries."

Q: "How do you make a resilient call to an external LLM API?"
A: "httpx async client, explicit timeout, retries with exponential backoff +
   jitter, circuit breaker, connection pooling, respect 429 Retry-After, cap
   concurrency with a semaphore."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ PHASE 4: DATABASES ━━━━━━━━━━━━━━━━━━━━━━━

Q: "A query is slow. What do you do?"
A: "EXPLAIN ANALYZE. A seq scan on a big table in WHERE/JOIN = missing index.
   Add the index, avoid SELECT * and functions on indexed columns, check for
   N+1, cache or materialize expensive aggregates."

Q: "What's the N+1 problem?"
A: "1 query for parents + 1 per parent for related data. Fix with eager loading —
   joinedload/selectinload, or select_related/prefetch_related in Django."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ PHASE 5: TESTING ━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q: "How do you test non-deterministic LLM output?"
A: "Separate the deterministic code — mock the LLM and unit-test prompt building,
   parsing, routing, error handling. Evaluate the model output as quality, not
   equality: golden set, semantic similarity, LLM-as-judge / RAGAS, separate suite."

Q: "Is 100% coverage the goal?"
A: "No — coverage is execution, not verification. Gate ~80%, care that tests
   assert meaningful behavior. Mutation testing measures real test quality."

━━━━━━━━━━━━━━━━━━━━━━━━━ PHASE 6: MULTI-AGENT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q: "Workflow vs agent — and MCP vs A2A?"
A: "Workflow follows a fixed path; an agent decides its own next step — use the
   least autonomy that works. MCP = agent-to-tools (vertical); A2A =
   agent-to-agent (horizontal). Complementary."

Q: "How do you debug an agent stuck in a loop?"
A: "Recursion/iteration limit + loop detection as guardrails; LangSmith tracing
   to see every step and where it repeats; then fix the cause — tool description,
   routing, or context loss."

━━━━━━━━━━━━━━━━━━━━━━━━━━ PHASE 7: SYSTEM DESIGN ━━━━━━━━━━━━━━━━━━━━━━━━━

Q: "How do you approach 'design an AI system'?"
A: "Clarify requirements + scale, define the API, draw the layered architecture
   (client/gateway/app+orchestrator/LLM gateway/tools/data/async/x-cutting),
   drill where they probe, cover scaling/reliability/security/cost, state
   trade-offs. Start simple, scale with need."

Q: "Biggest bottleneck in an AI system?"
A: "The LLM — latency, rate limits, cost. Caching (exact+semantic), model
   routing, streaming, a gateway with multiple providers."

━━━━━━━━━━━━━━━━━━━━━━━━ PHASE 8: TRADE-OFFS/LEAD ━━━━━━━━━━━━━━━━━━━━━━━━━

Q: "RAG or fine-tuning?"
A: "RAG for fresh, attributable knowledge; fine-tune for style/format/behavior,
   not facts. Prompt first; they combine."

Q: "How do you estimate a GenAI project?"
A: "Decompose into small tasks, estimate in ranges with velocity, add hidden
   work + buffer. For GenAI's uncertainty I spike the risky part first, then
   estimate confidently."

━━━━━━━━━━━━━━━━━━━━━━━━━━━ PHASE 9: CLOUD/DEVOPS ━━━━━━━━━━━━━━━━━━━━━━━━━

Q: "Zero-downtime deploy?"
A: "Rolling update + readiness probes + graceful shutdown on SIGTERM +
   backward-compatible migrations. Canary or blue-green for risky changes."

Q: "Managed LLM or self-hosted?"
A: "Managed (Bedrock/Azure OpenAI) for best models and zero GPU ops, in-region
   for compliance. Self-host only at high volume or strict data control."

SCORING: 18 questions. Strong = crisp answer + you could go DEEPER on a
follow-up. If any felt shaky, re-read that phase. Target: confident on 15+.
'''


# =================================================================================
# SECTION 3: A CODING ROUND SIMULATION (the live-coding pattern)
# =================================================================================
'''
HashedIn rounds include live coding (easy->medium->hard). Here's the PROCESS to
follow out loud — the process matters as much as the answer.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE LIVE-CODING PROCESS (narrate each step):
    1. CLARIFY: inputs, outputs, constraints, edge cases, expected scale.
       ("Can the list be empty? Negatives? How large — does O(n^2) matter?")
    2. EXAMPLES: state a couple of input/output examples to confirm understanding.
    3. APPROACH: describe the approach + complexity BEFORE coding. Get a nod.
       ("Brute force is O(n^2); a hash map gets O(n) — I'll do that.")
    4. CODE: write clean code, talk through it. Meaningful names, small functions.
    5. TEST: walk through an example + edge cases (empty, one element, dup, None).
    6. COMPLEXITY: state time + space. Discuss trade-offs / improvements.

    Interviewers score PROBLEM-SOLVING + COMMUNICATION, not just the final code.
    Think aloud; a partial well-reasoned solution beats silent perfection.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

A WORKED EXAMPLE (Python idioms a lead writes cleanly):

    # "Group anagrams together."
    from collections import defaultdict
    def group_anagrams(words: list[str]) -> list[list[str]]:
        groups = defaultdict(list)
        for w in words:
            key = "".join(sorted(w))      # anagrams share a sorted-letter key
            groups[key].append(w)
        return list(groups.values())
    # Time O(n*k log k), space O(n*k). Clean, Pythonic, explained.

    PYTHON IDIOMS THEY LIKE TO SEE:
    - Comprehensions, dict/defaultdict/Counter, set for dedup/membership,
      enumerate/zip, generators for large data, f-strings, type hints,
      context managers, sorted(key=...). (Your Python coding bank — Lessons in
      InterviewPrep/PythonCoding.)

THE COMMON TRAPS (your earlier Cognizant lessons apply):
    Mutable default args, late-binding closures, aliasing, is vs ==, slicing
    bounds. Don't get caught on these under pressure.

INTERVIEW POINT:
    "I clarify inputs and edge cases first, state examples, then describe my
    approach and complexity before coding so we agree on direction. I write
    clean, typed Python, narrate as I go, then test against edge cases like
    empty, single, duplicate, and None, and finish by stating time and space
    complexity and any trade-offs. Communicating the reasoning matters as much
    as the code."
'''


# =================================================================================
# SECTION 4: A SYSTEM DESIGN ROUND SIMULATION (architecture deep-dive)
# =================================================================================
'''
A full design prompt run end-to-end using the Phase 7 method. Practice
narrating this in ~15 minutes.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PROMPT: "Design a multi-agent system that automates IT support: answers from
internal KB, looks up ticket/asset data, and can create/update tickets."

(1) CLARIFY: "Users/scale? Latency? Data sources — KB, ticketing system, asset
    DB? Which actions are allowed, and do writes need approval? PII/compliance?
    Multi-tenant?" -> assume ~20k employees, interactive, must approve risky writes.

(2) SCALE: 20k x ~5/day = 100k/day ~= 1-2 QPS avg, ~15 QPS peak; ~300k KB docs
    -> ~1.5M chunks -> pgvector fine; async + a few workers.

(3) API: POST /chat (SSE stream) {conversation_id, message}; POST /actions/*
    (HITL-gated); GET /jobs/{id}.

(4) ARCHITECTURE (Phase 7 layers + Phase 6 topology):
    Client -> API Gateway (SSO/auth, rate limit) -> FastAPI (async, stateless)
    -> LangGraph SUPERVISOR (orchestrator-worker):
        Router -> [KB Agent (RAG)] [Data Agent (tools)] [Action Agent (HITL)]
        -> reflection/validate -> respond OR escalate to human.
    -> LLM Gateway (Azure OpenAI for residency, fallback model)
    -> Tools via MCP (ticketing, asset DB); Vector DB (pgvector); Postgres
       (metadata + checkpoint state); Redis (cache); S3 (KB docs).
    -> SQS + workers for ingestion + long jobs.

(5) DRILL (where they probe): RAG = hybrid + rerank + gate + citations,
    metadata-filtered by team for access control; Action Agent uses an MCP tool,
    HITL approval for risky writes, audit-logged.

(6) CROSS-CUTTING: stateless + HPA autoscale; semantic cache; LLM fallback +
    timeouts/retries/circuit breakers + "I don't know"; SSO + per-team RAG
    filtering + PII redaction + audit logs; LangSmith tracing + dashboards
    (latency/faithfulness/cost); golden eval set + RAGAS.

(7) TRADE-OFFS + EVOLUTION: "v1 = single RAG agent + escalation, pgvector, one
    LLM. Add the data and action agents once read-only is solid. Dedicated
    vector DB only past a few million vectors. Azure OpenAI over cheaper APIs —
    trade cost for compliance. Least autonomy first, scale with need."

INTERVIEW POINT:
    "I'd clarify scale, data sources, allowed actions, and compliance; estimate
    ~15 QPS and ~1.5M chunks so pgvector suffices; then sketch a streaming
    FastAPI + LangGraph supervisor routing to KB, data, and HITL-gated action
    agents, behind an LLM gateway on Azure OpenAI for residency. Team-level
    metadata filtering, audit logs, caching, fallback, and tracing — shipping a
    read-only RAG v1 first and adding actions once it's solid."
'''


# =================================================================================
# SECTION 5: HANDLING DEEP FOLLOW-UPS & "I DON'T KNOW"
# =================================================================================
'''
The make-or-break skill for a deep-dive round. They WILL push past your edge.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHEN THEY KEEP ASKING "WHY?" / "HOW?" / "WHAT IF?":
    - Stay calm — this is the design of the interview, not a fail signal.
    - Go one level deeper each time using your mechanism knowledge (Phases 1-9
      give you the layers beneath each concept).
    - If you reach your limit, transition gracefully (below) — don't bluff.

THE GRACEFUL "I DON'T KNOW" (this WINS points, doesn't lose them):
    Pattern: ACKNOWLEDGE + REASON + APPROACH.
    "I haven't worked with that specific [X] directly. Based on first
    principles, I'd expect it to [reasoned guess], because [reasoning]. To be
    sure, I'd [check the docs / prototype / measure]."

    WHY THIS WINS: it shows honesty (they trust you), reasoning ability (the
    real skill), and a path to the answer (how a senior actually operates).
    Bluffing a wrong fact confidently is the fastest way to fail — they'll
    catch it on the next follow-up.

DON'T:
    - Don't bluff specifics you don't know.
    - Don't freeze/go silent — narrate your thinking.
    - Don't argue if corrected — "good point, that changes my approach to...".

WHEN YOU NEED A MOMENT:
    "Good question — let me think about that for a second." (Then structure.)
    A 3-second pause to organize beats rushing into a mess.

WHEN STUCK ON A DESIGN/CODING PROBLEM:
    - Restate the problem; break it into smaller parts.
    - Start with the simplest brute-force, then optimize ("I'll get something
      working, then improve it").
    - Ask a clarifying question to unstick.

INTERVIEW ANSWER (meta — if asked "how do you handle not knowing?"):
    "I'm honest — I say I haven't used that directly, then reason from first
    principles about how I'd expect it to work and why, and state how I'd verify:
    docs, a spike, or a measurement. Pretending to know is risky and gets caught;
    showing how I reason and find answers is what actually matters day to day."
'''


# =================================================================================
# SECTION 6: THE CONSOLIDATED CHEAT CARD (all 9 phases, one page)
# =================================================================================
'''
Glance at THIS the morning of the interview. One line per concept, all phases.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 1 — FASTAPI
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ASGI + Starlette + Pydantic (async + validation + auto-docs).
  Lifecycle: server->middleware->routing->deps->validation->handler->
             response_model->middleware->out.
  async def = event loop | def = thread pool | NEVER block the loop.
  DI = reuse + testability (dependency_overrides) + lifecycle (yield).
  Separate request/response models; response_model filters output.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ PHASE 2 — ASYNC ━━━━━━━━━━━━━━━━━━━━━━━━━━
  Concurrency (interleave, 1 thread) != parallelism (N cores).
  GIL: one thread runs bytecode; releases on I/O (threads help I/O not CPU).
  await = pause, yield to loop, resume when I/O ready.
  I/O-bound -> asyncio | sync libs -> threads | CPU-bound -> multiprocessing.
  gather (concurrent) | Semaphore (cap) | wait_for (timeout) | to_thread (offload).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ PHASE 3 — API DESIGN ━━━━━━━━━━━━━━━━━━━━━━━
  REST = stateless + uniform interface -> scales horizontally.
  GET/PUT/DELETE idempotent; POST not (idempotency keys). 401 vs 403.
  Sync (wait) | async-API (202+poll/webhook) | event-driven (broker).
  Resilient call: httpx + timeout + backoff retry + circuit breaker + pooling.
  Gateway: routing + auth + rate limit + TLS at the front door.
  Security: AuthN + object-level AuthZ (#1 vuln) + validation + rate limit.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ PHASE 4 — DATABASES ━━━━━━━━━━━━━━━━━━━━━━━━
  SQL default (ACID/joins) | NoSQL for scale/shape; often polyglot.
  Normalize to 3NF; denormalize hot paths with evidence.
  Index WHERE/JOIN/ORDER BY + FKs; composite=leftmost-prefix; don't over-index.
  Slow query -> EXPLAIN ANALYZE -> kill seq scans / N+1 / SELECT *.
  ACID + isolation (read committed->repeatable read->serializable).
  Scale: optimize->cache->replicas->partition->shard(last). Pool connections.
  AI: pgvector (with metadata) or Qdrant/Pinecone; HNSW + cosine.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ PHASE 5 — TESTING ━━━━━━━━━━━━━━━━━━━━━━━━━━
  Pyramid: unit(many)/integration(some)/E2E(few); test behavior not impl.
  pytest: fixtures(yield/scope/conftest) + parametrize + raises/approx.
  Mock at the edges; patch where USED; return_value/side_effect/assert_called.
  Async/FastAPI: pytest-asyncio + AsyncMock + TestClient + dependency_overrides.
  LLM: mock LLM -> test logic; eval output (semantic/judge/RAGAS) separately.
  Coverage executes != verifies; ~80% gate; kill flaky tests.
  Tooling: venv | pip | Poetry | uv(fast); commit the LOCK file; pyproject.toml.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━ PHASE 6 — MULTI-AGENT ━━━━━━━━━━━━━━━━━━━━━━━━
  Agent = reason + tools + memory + autonomy. Workflow(fixed) vs agent(dynamic);
  use LEAST autonomy that works.
  Patterns: ReAct, Plan-Execute, Reflection, Router.
  Topologies: Orchestrator-Worker (default), Hierarchical, P2P, Hub-Spoke.
  MCP = agent<->tools (vertical); A2A = agent<->agent (horizontal). Complementary.
  Frameworks: LangGraph(control) | CrewAI(speed) | AutoGen(chat) | ADK(Google+A2A).
  Debug: bound loops + tracing (LangSmith) + replay + guardrails + evals.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━ PHASE 7 — SYSTEM DESIGN ━━━━━━━━━━━━━━━━━━━━━━
  Method: Clarify -> Scale -> API -> Architecture -> Drill -> X-cutting -> Trade-offs.
  Layers: client/gateway/app+orchestrator/LLM gateway/tools/data/async/x-cutting.
  Stateless servers + state in a store -> horizontal scale.
  LLM gateway: abstraction + fallback + cost routing.
  LLM is the bottleneck -> cache (exact+semantic) + model routing + stream.
  Degrade gracefully (fallback/circuit breaker > 500). Start simple, scale with need.

━━━━━━━━━━━━━━━━━━━━━━━━━━ PHASE 8 — TRADE-OFFS/LEAD ━━━━━━━━━━━━━━━━━━━━━━
  EVERY decision: "X because Y, trade-off Z, switch when W." No dogma.
  Monolith-first | SQL-default | RAG-facts/fine-tune-behavior | buy-non-core.
  Estimate: decompose + ranges + hidden work + buffer; spike GenAI risk first.
  Breakdown: goal -> story(INVEST) -> vertical slice -> trace to value.
  Review: correctness>security>design>tests; automate style; be kind.
  Mentor: teach the WHY, guide with questions, psychological safety.
  POC != production -> name the gap (scale/security/observability).

━━━━━━━━━━━━━━━━━━━━━━━━━━━ PHASE 9 — CLOUD/DEVOPS ━━━━━━━━━━━━━━━━━━━━━━━━
  Docker (cache layers, slim, non-root) -> registry -> K8s/ECS + probes + HPA.
  Container shares kernel (light) vs VM full guest OS (heavy).
  Zero-downtime: rolling + readiness + graceful shutdown + safe migrations.
  CI/CD: lint+type+test+coverage+scan -> build ONCE -> promote dev/stg/prod.
  IaC (Terraform). Git: trunk-based + PR + protected main + feature flags.
  Operate: metrics+logs+traces + golden signals + SLOs + post-mortems.
  AI: managed LLM (in-region) default; API tier I/O-bound; LLMOps=prompt/RAG/eval/cost.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ YOUR STORY ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  4 yrs: 2 Java full-stack -> 2 GenAI. At Infor on Distribution SX.e (ERP).
  3 pillars: MCP server (ERP tools) | product-attribute pipeline | DocSage
  (LangGraph agentic RAG: agent->tools->grade->generate->validate + self-correct).
  Lead the trade-off sentence on EVERY answer. Connect everything to the AI use case.
'''


# =================================================================================
# SECTION 7: FINAL DELIVERY HABITS + GOLDEN LESSONS
# =================================================================================
'''
The content is done across 10 phases. These habits make it LAND in a 1.5-hour
deep-dive.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE 10 DELIVERY HABITS:

    1. THINK ALOUD. They score reasoning, not just answers. Narrate your path.
    2. STRUCTURE EVERY ANSWER. Use a framework: Phase 7 method for design, STAR
       for behavioral, clarify-approach-code-test for coding.
    3. TRADE-OFF SENTENCE ON EVERYTHING: "X because Y, trade-off Z, switch when W."
    4. START SIMPLE, THEN SCALE. v1 first, then "at 100x I'd...". Avoid
       over-engineering upfront.
    5. CLARIFY BEFORE SOLVING. Ask about scale, constraints, edge cases. Shows
       maturity; prevents solving the wrong problem.
    6. CONNECT TO YOUR EXPERIENCE. Tie answers to DocSage / MCP / SX.e — concrete
       beats abstract.
    7. CONNECT TO THE AI USE CASE. This is a GenAI role — relate Python/DB/cloud
       answers back to LLM systems where natural.
    8. BE HONEST AT YOUR EDGE. "I haven't used that; I'd reason it as... and
       verify by...". Honesty + reasoning > bluffing.
    9. PAUSE WHEN NEEDED. "Let me think about that for a second." Calm > rushed.
    10. STAY CALM UNDER DEPTH. Deep follow-ups are the format, not a fail signal.
        Going deep until you hit a limit is EXPECTED.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE WEEK-BEFORE PLAN (you've built the material — now consolidate):
    - Re-read Phases 1-4 (primary, highest probability) — twice.
    - Re-read Phases 6, 7, 8 (your strength + design + lead).
    - Skim 5, 9 (lighter). Do the Section 2 mock OUT LOUD, timed.
    - Practice the DocSage walkthrough (Lesson 22) + your intro (Lesson 23).
    - Drill this cheat card (Section 6) the morning of.

THE NIGHT BEFORE / MORNING OF:
    - Sleep. A sharp, calm mind beats one more topic crammed.
    - Re-read ONLY the Section 6 cheat card + your story.
    - Test the video/screen-share setup; pen + paper ready for diagrams.
    - Join early, water nearby, resume open.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GOLDEN LESSONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. DEPTH IS THE TEST — they drill to find your ceiling; that's normal. The
   Phases gave you the "why" beneath every "what" so you don't collapse on the
   3rd follow-up (the #1 reason people fail here).
2. STRUCTURE WINS — a framework on every answer (design method, STAR, trade-off
   sentence) reads as lead-level and keeps you calm.
3. TRADE-OFFS, NOT DOGMA — "it depends on [factors], so I'd choose X." The JD's
   single biggest theme.
4. HONESTY AT THE EDGE BEATS BLUFFING — reason from first principles + how you'd
   verify. This wins points; bluffing loses them.
5. CONNECT TO YOUR WORK + THE AI USE CASE — concrete and role-relevant.
6. START SIMPLE, SCALE WITH NEED — least complexity / least autonomy that solves it.
7. COMMUNICATE — think aloud; the reasoning IS the evaluation.
8. IT'S A LEAD ROLE — show ownership, mentoring, delivery, not just coding.
9. CALM + CURIOUS — treat it as a technical conversation between peers.
10. YOU'VE DONE THE WORK — 10 deep phases + your real projects. Trust it.

ONE-LINE MANTRA:
    "Clarify -> structure -> reason aloud -> trade-offs -> connect to my work.
     Depth + honesty + structure. Start simple, scale with need. I've got this."
'''


# =================================================================================
# RUN SUMMARY
# =================================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 10 — MOCK INTERVIEW & CONSOLIDATION")
    print("=" * 70)
    print()
    print("THE ROUND: 1-1.5h deep-dive; coding + design + theory + behavioral.")
    print("They drill DEEP to find your ceiling — that's the format, not a fail.")
    print()
    print("WHY PEOPLE FAIL -> HOW YOU WON'T:")
    print("  shallow (collapses on follow-up) -> the depth in Phases 1-9")
    print("  no trade-offs -> 'X because Y, trade-off Z, switch when W'")
    print("  bluffing -> honest + reason from first principles + how to verify")
    print("  rambling -> frameworks (design method, STAR, clarify-approach-code)")
    print()
    print("DELIVERY: think aloud | structure | trade-offs | start simple |")
    print("          clarify first | connect to DocSage/MCP/SX.e | honest at edge |")
    print("          pause when needed | stay calm under depth.")
    print()
    print("MANTRA: Clarify -> structure -> reason aloud -> trade-offs ->")
    print("        connect to my work. Depth + honesty + structure. I've got this.")
    print()
    print("=" * 70)
    print("ALL 10 PHASES COMPLETE. Primary(1-4) + Medium(6,7,8) + Low(5,9,10).")
    print("Re-read 1-4 twice, 6-8 once, skim 5/9, do the mock aloud. Good luck!")
    print("=" * 70)
