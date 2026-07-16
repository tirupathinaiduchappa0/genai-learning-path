"""

Q1)Write a function (Python) 
that uses a GenAI API to perform entity extraction from input text,
then enriches each entity using a third-party knowledge API 
(e.g., DBpedia). Handle API failures, concurrency for enrichment calls,
and return a structured list of enriched entities.  
  
 Q2)Given a weekly project report with automated RAG status updates,
 describe the technical steps you'd take to ensure you maintain true 
 RAG ownership and prevent misleading statuses caused by data
 integration errors.


Q3)You have a GenAI model that returns overly verbose answers
to simple queries. Describe a practical approach, without retraining
, to make responses more concise.

Q4)A GenAI model is generating outputs that are too generic 
lack context relevance. What practical prompt engineering techniques
would you apply to improve the specificity and usefulness of its 
responses? 


Q5)A core service relies on a single external API with
occasional slowdowns. What simple, immediate changes would you
make to improve reliability without a major redesign?

Q6)Explain how you would implement background job processing
(e.g., sending emails or report generation) in a backend application
built with Java, Python. What patterns or 
components would you use to ensure reliability and scalability?
                

===================================================================================
POST-MORTEM — 6 REAL INTERVIEW QUESTIONS (GenAI + Backend, Senior/Lead)
===================================================================================

A real interview: Q1 was a coding skeleton; Q2-Q6 were "understand the question
and explain your approach." Mix: hands-on coding, GenAI practical, senior
backend/architecture, and techno-managerial.

For EACH question: (a) what the interviewer is REALLY testing, (b) the full
model answer / code, (c) the KEYWORDS to drop, (d) traps, (e) related concepts
to remember.

THE 6 QUESTIONS:
    Q1. Entity extraction (GenAI) + enrichment (DBpedia) + concurrency + errors.
    Q2. Weekly project report "RAG status" ownership + data-integration errors.
        ⚠️ RAG here = RED / AMBER / GREEN (project health), NOT retrieval!
    Q3. GenAI gives verbose answers -> make concise WITHOUT retraining.
    Q4. GenAI outputs too generic -> prompt engineering for specificity.
    Q5. Core service on a single slow external API -> immediate reliability wins.
    Q6. Background job processing (Java/Python) -> reliability + scalability.

THE CROSS-CUTTING THEME (say this if asked to reflect):
    Resilience — timeouts, retries+backoff, circuit breakers, DLQ, idempotency —
    runs through Q1, Q5, and Q6. The interviewer is probing whether you build
    ROBUST systems, not just happy-path code.

NOTE: Q1 code SIMULATES the GenAI + DBpedia APIs so this file runs anywhere
      (no keys/network). The PATTERN is what matters and is production-shaped.
===================================================================================
"""


# =================================================================================
# Q1 — ENTITY EXTRACTION + ENRICHMENT + CONCURRENCY + ERROR HANDLING
# =================================================================================
'''
THE QUESTION:
    Write a Python function that uses a GenAI API to extract entities from text,
    enriches each entity via a third-party knowledge API (e.g., DBpedia),
    handles API failures, uses concurrency for enrichment, and returns a
    structured list of enriched entities.

WHAT HE'S REALLY TESTING:
    Can you orchestrate TWO external APIs in production-grade async Python with
    proper structure, concurrency, and resilience? This is the synthesis of
    async (Phase 2), resilient API calls (Phase 3), and structured I/O (Pydantic).

THE 6 THINGS HE'S SCORING (your mental checklist):
    1. SEPARATION OF CONCERNS — distinct functions (extract / enrich one /
       orchestrate), not one mega-function.
    2. STRUCTURED I/O — Pydantic models, typed return, each entity has a status.
    3. CONCURRENCY — asyncio.gather to fan out enrichment; Semaphore to cap it.
    4. PER-ITEM ERROR HANDLING — one failure must NOT kill the batch.
    5. RESILIENCE — timeouts + retries with backoff on every external call.
    6. RIGHT TOOLS — httpx.AsyncClient (not requests), shared client.

KEYWORDS TO DROP:
    async/await, asyncio.gather, asyncio.Semaphore, httpx.AsyncClient, Pydantic,
    timeout, retry + exponential backoff, return_exceptions=True, "fail
    per-entity not per-batch", structured output, graceful degradation.

TRAPS (what loses points):
    - Using requests inside async (blocks the event loop).
    - Enriching in a sequential for-loop (no concurrency) -> slow.
    - One failed enrichment crashing the whole function.
    - Returning raw dicts instead of a typed model.
    - No timeout (a hung API hangs your function forever).

RELATED CONCEPTS TO REMEMBER (drop if probed):
    - Semaphore caps concurrency to respect the knowledge-API rate limit.
    - gather(return_exceptions=True) returns errors as results, doesn't raise.
    - Idempotency / caching enrichment results (same entity seen again).
    - tenacity library for retry/backoff in real code.
    - This is exactly the RAG "ingestion + enrichment" shape in production.
'''

import asyncio
import random
from typing import Literal, Optional
from dataclasses import dataclass, field, asdict


# ----- STRUCTURED MODELS (use Pydantic in real code; dataclass here = no deps) -----

@dataclass
class Entity:
    """A raw entity extracted by the GenAI model."""
    name: str
    type: str                       # PERSON, ORG, LOCATION, ...
    confidence: float = 1.0


@dataclass
class EnrichedEntity:
    """An entity after enrichment, carrying its own success/failure status."""
    name: str
    type: str
    confidence: float
    enrichment: Optional[dict] = None
    status: Literal["success", "failed"] = "success"
    error: Optional[str] = None


# ----- SIMULATED EXTERNAL APIs (replace with real GenAI + DBpedia calls) -----

async def _call_genai_extract(text: str) -> list[Entity]:
    """SIMULATES a GenAI API extracting entities. Real: OpenAI/Anthropic with
    a structured-output (JSON) prompt -> parse into Entity models."""
    await asyncio.sleep(0.05)                      # network latency
    # In production: structured-output prompt returning {entities: [...]}
    canned = [
        Entity("Alan Turing", "PERSON", 0.98),
        Entity("Bletchley Park", "LOCATION", 0.95),
        Entity("Enigma", "PRODUCT", 0.91),
        Entity("UNKNOWN_CORP", "ORG", 0.70),       # this one will fail enrichment
    ]
    return canned


async def _call_dbpedia(name: str) -> dict:
    """SIMULATES a knowledge API (DBpedia) lookup. Randomly fails/slows to
    exercise our error handling + timeout + retry."""
    await asyncio.sleep(random.uniform(0.02, 0.12))
    if name == "UNKNOWN_CORP":
        raise ConnectionError("404 / not found in knowledge base")
    return {"abstract": f"{name} — enriched description.", "uri": f"dbpedia.org/{name.replace(' ', '_')}"}


# ----- THE RESILIENT ENRICHMENT (one entity, with timeout + retry + per-item catch) -----

async def enrich_entity(entity: Entity, sem: asyncio.Semaphore,
                        retries: int = 2, timeout: float = 1.0) -> EnrichedEntity:
    """Enrich ONE entity. Bounded by the semaphore, with timeout + retry +
    backoff, and per-item error handling so a failure is isolated."""
    async with sem:                                          # cap concurrency
        for attempt in range(retries + 1):
            try:
                data = await asyncio.wait_for(_call_dbpedia(entity.name), timeout=timeout)
                return EnrichedEntity(entity.name, entity.type, entity.confidence,
                                      enrichment=data, status="success")
            except (ConnectionError, asyncio.TimeoutError) as e:
                if attempt < retries:
                    await asyncio.sleep(2 ** attempt * 0.05)  # exponential backoff
                    continue
                # final failure -> degrade gracefully, don't raise
                return EnrichedEntity(entity.name, entity.type, entity.confidence,
                                      enrichment=None, status="failed", error=str(e)[:60])


# ----- THE ORCHESTRATOR (extract -> fan-out enrich -> structured list) -----

async def extract_and_enrich(text: str, max_concurrency: int = 10) -> list[EnrichedEntity]:
    """Top-level: extract entities, then enrich them CONCURRENTLY, returning a
    structured list. One enrichment failure does not break the others."""
    entities = await _call_genai_extract(text)
    if not entities:
        return []
    sem = asyncio.Semaphore(max_concurrency)
    tasks = [enrich_entity(e, sem) for e in entities]
    # return_exceptions=True would also work; here each task already catches.
    return await asyncio.gather(*tasks)


# ----- DEMO (runs deterministically enough to show the pattern) -----

def _demo_q1():
    random.seed(0)
    results = asyncio.run(extract_and_enrich("Alan Turing worked at Bletchley Park."))
    print("Q1: extracted + enriched", len(results), "entities")
    for r in results:
        tag = "OK " if r.status == "success" else "FAIL"
        print(f"   [{tag}] {r.name} ({r.type}) -> {r.status}")
    n_ok = sum(1 for r in results if r.status == "success")
    print(f"Q1: {n_ok}/{len(results)} enriched successfully; failures isolated, batch survived.")


# Run the demo when the file runs.
print("=" * 70)
print("Q1 — ENTITY EXTRACTION + ENRICHMENT (runnable skeleton demo)")
print("=" * 70)
_demo_q1()


# =================================================================================
# Q2 — "RAG STATUS" OWNERSHIP IN A WEEKLY PROJECT REPORT (techno-managerial)
# =================================================================================
'''
THE QUESTION:
    Given a weekly project report with automated RAG status updates, describe
    the technical steps you'd take to maintain TRUE RAG ownership and prevent
    misleading statuses caused by data integration errors.

⚠️ THE CRITICAL INSIGHT — "RAG" HERE = RED / AMBER / GREEN ⚠️
    This is the project-health TRAFFIC-LIGHT status (Red=at risk, Amber=caution,
    Green=on track), NOT Retrieval-Augmented Generation. This is a
    TECHNO-MANAGERIAL / DELIVERY-OWNERSHIP question. If you answered it as
    retrieval-augmented generation, that's the miss to learn from. Recognizing
    "RAG = project status" is itself a signal of real delivery experience.

WHAT HE'S REALLY TESTING:
    Lead maturity: do you OWN the accuracy of project status, understand that
    automated dashboards LIE when their input data is broken, and apply
    data-quality engineering to reporting? Accountability + not blindly trusting
    automation.

THE MODEL ANSWER (structure: ownership stance -> validate the data -> safeguards):

    "First, the principle: automation assists, but I OWN the status — I'm
    accountable to stakeholders, not the dashboard. An automated Green never
    overrides what I know to be true about the project.

    The RAG status is COMPUTED from integrated data — Jira progress, test pass
    rates, burn-down, defect counts, CI results. If a data-integration error
    corrupts those inputs, the status misleads. So my technical steps:

    1. DATA QUALITY CHECKS on every source feed: schema validation, null/range
       checks, and FRESHNESS/STALENESS checks — if the Jira sync failed, the
       data is stale and I must NOT show a confident Green off stale data.
    2. FAIL-SAFE DEFAULT: if a feed is missing or broken, the status DEGRADES to
       Amber or 'Unknown' — never silently defaults to Green. Absence of bad
       news is not good news.
    3. RECONCILIATION: periodically cross-check the automated status against
       ground truth (the actual team/standup view) to catch divergence.
    4. ANOMALY DETECTION: a sudden Red->Green jump triggers investigation, not
       celebration — it's often a data glitch, not real recovery.
    5. DATA LINEAGE / AUDIT TRAIL: record which data produced which status, when,
       from where — so a misleading status is traceable and explainable.
    6. CONFIDENCE / COMPLETENESS FLAG: annotate the status with data coverage
       ('Green, based on 60% of feeds — treat with caution').
    7. ALERTING on integration failures: a broken feed is VISIBLE immediately,
       not discovered weeks later.
    8. HUMAN-IN-THE-LOOP SIGN-OFF: I review and approve the automated status
       before it goes to stakeholders. The lead validates; automation drafts.

    Net: automation drafts the status from validated, fresh, reconciled data;
    I own the final call and there's a fail-safe so broken data never produces
    false confidence."

KEYWORDS TO DROP:
    Red/Amber/Green, ownership/accountability, data quality, schema validation,
    freshness/staleness, FAIL-SAFE (degrade to Amber not Green), reconciliation,
    anomaly detection, data lineage/audit trail, confidence/completeness flag,
    alerting on integration failures, human-in-the-loop sign-off, single source
    of truth.

TRAPS:
    - Answering as Retrieval-Augmented Generation (the big one).
    - Only talking process ("I'll check it manually") with no DATA-QUALITY
      engineering — he said "technical steps" and "data integration errors."
    - Trusting automation blindly / no fail-safe.

RELATED CONCEPTS TO REMEMBER:
    - This is data-pipeline observability applied to reporting (ties to your
      monitoring/observability knowledge — Phase 9).
    - "Garbage in, garbage out" — the status is only as good as its feeds.
    - Same fail-safe philosophy as a RAG retrieval gate: when unsure, don't
      assert confidence — degrade. (Nice bridge if you want to show range.)
'''


# =================================================================================
# Q3 — VERBOSE GENAI ANSWERS -> CONCISE, WITHOUT RETRAINING
# =================================================================================
'''
THE QUESTION:
    A GenAI model returns overly verbose answers to simple queries. Describe a
    practical approach, WITHOUT retraining, to make responses more concise.

WHAT HE'S REALLY TESTING:
    Do you know the practical LLM control levers (prompt + params + routing +
    post-processing) beyond "fine-tune it"? Pragmatism with the toolkit.

THE MODEL ANSWER (organized by lever, strongest first):

    "Several levers, none requiring retraining:

    1. FEW-SHOT EXAMPLES (most effective): show 2-3 examples of a simple
       question with a short answer. The model mirrors the demonstrated style —
       far more reliable than just instructions.
    2. PROMPT INSTRUCTIONS: an explicit conciseness directive in the system
       prompt — 'Answer in 1-2 sentences. Be direct. No preamble or filler.'
       Set a brevity persona as the default.
    3. max_tokens CAP: hard-limit the output length as a backstop (bounds it,
       though it can cut off mid-sentence, so pair with instructions).
    4. OUTPUT FORMAT CONSTRAINTS: request a specific shape — 'one sentence',
       a single bullet, or structured output — which forces brevity.
    5. QUERY ROUTING by complexity: classify simple vs complex queries and route
       simple ones to a 'concise mode' prompt (or a smaller, faster model). A
       simple question shouldn't trigger the verbose reasoning chain.
    6. POST-PROCESSING: a cheap second pass — 'compress this to 2 sentences' —
       or truncate/summarize, when the source can't be controlled.

    I'd then VERSION the new prompt and EVALUATE it on examples, measuring both
    length AND whether quality held — treating the prompt like code."

KEYWORDS TO DROP:
    few-shot examples, system prompt, conciseness instruction, max_tokens,
    output format / structured output, query routing / complexity classification,
    'concise mode', post-processing / compression pass, prompt versioning + eval,
    "no fine-tuning needed."

TRAPS:
    - Only saying "tell it to be short" (one lever). He wants MULTIPLE.
    - Confusing this with Q4 — Q3 is about LENGTH, Q4 about RELEVANCE.
    - Suggesting fine-tuning (he explicitly excluded it).

RELATED CONCEPTS:
    - Generation params: temperature affects creativity/rambling; max_tokens,
      stop sequences bound output.
    - Smaller models are often naturally more concise + cheaper for simple Qs.
    - Ties to your prompt-management lesson (version + eval the prompt change).
'''


# =================================================================================
# Q4 — GENERIC OUTPUTS LACKING CONTEXT -> PROMPT ENGINEERING FOR SPECIFICITY
# =================================================================================
'''
THE QUESTION:
    A GenAI model generates outputs that are too generic and lack context
    relevance. What practical prompt engineering techniques would you apply to
    improve specificity and usefulness?

WHAT HE'S REALLY TESTING:
    Depth of prompt engineering — and that you DON'T just repeat Q3. Q3 = length;
    Q4 = RELEVANCE / GROUNDING. The core insight: generic output usually means
    the model has NO context to be specific about.

THE MODEL ANSWER (lead with grounding/context — that's the real fix):

    "Generic output almost always means missing context — the model falls back
    to platitudes because it has nothing specific to anchor to. So my techniques,
    strongest first:

    1. INJECT CONTEXT / GROUNDING (the #1 fix): feed the model the specific
       inputs — the user's data, domain details, conversation history, and most
       structurally, RETRIEVED DOCUMENTS via RAG. Grounding in real, relevant
       context is what kills generic answers.
    2. ROLE / PERSONA PROMPTING: 'You are a senior tax advisor specializing in
       Indian SMEs.' A specific persona narrows the model's frame and vocabulary.
    3. SPECIFIC, DETAILED INSTRUCTIONS: replace vague asks with precise ones —
       specify the audience, the angle, the format, the depth, and constraints.
    4. FEW-SHOT EXAMPLES of the SPECIFIC, context-rich style I want (not generic
       samples) — the model imitates the demonstrated specificity.
    5. CHAIN-OF-THOUGHT: 'Reason about THIS user's specific situation before
       answering' — step-by-step grounding reduces generic responses.
    6. CLARIFYING QUESTIONS: have the model ask for missing specifics before
       answering, instead of guessing generically.
    7. CONSTRAIN TO THE USER'S ACTUAL DATA so it can't retreat to generalities.

    Then I A/B test the prompt variants and measure RELEVANCE, not just fluency."

KEYWORDS TO DROP:
    context injection / grounding, RAG / retrieved documents, role/persona
    prompting, specific instructions, few-shot (specific style), chain-of-thought,
    clarifying questions, "ground in the user's data", relevance evaluation,
    structured prompt.

TRAPS:
    - Overlapping with Q3. SAY THE DISTINCTION: "Q3 was about length; this is
      about relevance — the fix is grounding/context, primarily RAG."
    - Listing only generic 'be more specific' with no mechanism.
    - Forgetting RAG — it's the structural answer to "lacks context relevance."

RELATED CONCEPTS:
    - This is literally why RAG exists: inject relevant, fresh context.
    - Context engineering > prompt wording: the right CONTEXT beats clever phrasing.
    - Pairs with Q3: real systems route simple->concise and complex->grounded.
'''


# =================================================================================
# Q5 — SINGLE SLOW EXTERNAL API -> IMMEDIATE RELIABILITY WINS (no redesign)
# =================================================================================
'''
THE QUESTION:
    A core service relies on a single external API with occasional slowdowns.
    What simple, IMMEDIATE changes would you make to improve reliability WITHOUT
    a major redesign?

WHAT HE'S REALLY TESTING:
    Resilience patterns under a pragmatism constraint. He wants QUICK WINS, not
    "rebuild the architecture." Ties to your Phase 3 resilience toolkit.

THE MODEL ANSWER (ordered by impact + simplicity, lead with the top 3):

    "Without redesigning anything, the highest-impact immediate changes:

    1. TIMEOUTS (the #1 fix): set aggressive connect + read timeouts so a slow
       upstream can't hang my service indefinitely and exhaust resources. Today,
       a single slow call can tie up workers and cascade.
    2. CACHING with TTL: cache responses (Redis or in-memory) so I'm not hitting
       the slow API on every request, and I can serve a slightly-stale cached
       result during slowdowns. Cheap, huge win for read-heavy calls.
    3. CIRCUIT BREAKER: when it's repeatedly slow/failing, 'open' the breaker and
       fail fast or serve a fallback for a cooldown, instead of piling requests
       onto a struggling dependency (which makes it worse).
    4. RETRIES with exponential backoff + jitter: for transient slowness, retry
       once or twice — but only for idempotent calls, and with backoff so I
       don't hammer it.
    5. GRACEFUL DEGRADATION / FALLBACK: return cached or default/partial data
       when the API is slow, rather than failing the whole user request.
    6. CONNECTION POOLING: reuse connections instead of reopening — immediate,
       low-effort latency win.
    7. BULKHEAD / CONCURRENCY CAP: limit concurrent calls to that API so it can't
       consume all my threads/workers and take down unrelated features.
    8. Make the call ASYNC / non-blocking so a slow call doesn't block a worker.

    My top three immediate wins: TIMEOUT + CACHE + CIRCUIT BREAKER."

KEYWORDS TO DROP:
    timeout, caching (TTL), circuit breaker, retry + exponential backoff + jitter,
    graceful degradation / fallback, connection pooling, bulkhead, concurrency
    cap, idempotency, async/non-blocking. (Top 3: timeout, cache, circuit breaker.)

TRAPS:
    - Proposing a big redesign / replacing the API (he said "no major redesign").
    - Forgetting timeouts — the single most important immediate change.
    - Retrying non-idempotent calls (can duplicate side effects).

RELATED CONCEPTS:
    - Circuit breaker states: closed -> open (failing) -> half-open (trial).
    - "Retry storms" / thundering herd — why jitter matters.
    - This is the consuming side of Q1's resilience and overlaps Q6's reliability.
    - Cache invalidation (TTL vs event-based) — the trade-off of staleness.
'''


# =================================================================================
# Q6 — BACKGROUND JOB PROCESSING (Java/Python) -> RELIABILITY + SCALABILITY
# =================================================================================
'''
THE QUESTION:
    How would you implement background job processing (e.g., sending emails or
    report generation) in a backend built with Java/Python? What patterns or
    components ensure reliability and scalability?

WHAT HE'S REALLY TESTING:
    Do you know the queue + workers architecture and the RELIABILITY properties
    (durability, retries, DLQ, idempotency, at-least-once)? Ties to Phase 3
    (BackgroundTasks vs real queue) and Phase 9.

THE ARCHITECTURE (draw/describe this):

    [API / Producer] --enqueue job--> [Message Broker / Queue] --pull--> [Workers]
         |  returns 202 + job_id              (Redis/RabbitMQ/                |
         |  immediately                        Kafka/SQS)                     |
         v                                                                    v
    [Job status store] <----------------- workers update status -------> [do work:
     (DB: queued/running/                                                 send email /
      done/failed)                                                        gen report]
                                                  |
                                       failed repeatedly
                                                  v
                                       [Dead-Letter Queue (DLQ)]

THE MODEL ANSWER:

    "The core pattern is PRODUCER -> MESSAGE BROKER -> WORKER CONSUMERS. The API
    enqueues a job and returns immediately (202 + a job id), so the request
    isn't blocked; separate worker processes pull from the queue and do the work.

    COMPONENTS:
    - Broker/queue: Redis, RabbitMQ, Kafka, or AWS SQS.
    - Workers: Python — Celery (or RQ, Arq, Dramatiq); Java — Spring @Async,
      Spring Batch, or queue consumers (@RabbitListener / Kafka), Quartz for
      scheduling.

    RELIABILITY (the heart of the question):
    - DURABILITY: jobs are PERSISTED in the broker, so they survive an API or
      worker crash — not held in process memory.
    - ACKNOWLEDGEMENTS: a job is removed only AFTER successful processing; if a
      worker dies mid-job, it's redelivered. This gives AT-LEAST-ONCE delivery.
    - IDEMPOTENCY: because of at-least-once, a job can run twice — so processing
      must be idempotent (an idempotency key so a retried 'send email' doesn't
      double-send).
    - RETRIES with backoff on transient failures.
    - DEAD-LETTER QUEUE: jobs that keep failing move to a DLQ for inspection,
      instead of blocking the queue or retrying forever.

    SCALABILITY:
    - HORIZONTAL SCALING of workers: add consumers to drain the queue faster;
      the queue BUFFERS spikes (backpressure) so the API stays responsive.
    - PRIORITY / MULTIPLE QUEUES: separate fast vs slow jobs; high-priority lanes.
    - SCHEDULING for periodic jobs (reports): Celery Beat / Quartz / cron.

    OBSERVABILITY: a job-status table users can poll (the 202 + job_id pattern),
    monitor queue depth, retry rates, DLQ size; alert on backlog.

    I would NOT use in-process background threads or FastAPI BackgroundTasks for
    critical work — they don't survive a restart and don't scale across machines."

KEYWORDS TO DROP:
    message queue/broker (Celery / RabbitMQ / Kafka / SQS), producer-consumer,
    worker pool, DURABILITY/persistence, ACKNOWLEDGEMENT, AT-LEAST-ONCE delivery,
    IDEMPOTENCY (idempotency key), retry + backoff, DEAD-LETTER QUEUE (DLQ),
    horizontal scaling, backpressure, priority queues, scheduling (Celery Beat /
    Quartz), job-status tracking (202 + job id). Java: Spring @Async / Batch /
    Quartz / Kafka consumers.

TRAPS:
    - Answering "a background thread" or FastAPI BackgroundTasks for CRITICAL
      work — not durable, doesn't scale. (Fine only for trivial fire-and-forget.)
    - Forgetting IDEMPOTENCY — the subtle reliability point under at-least-once.
    - No DLQ — failing jobs block or loop forever.

RELATED CONCEPTS:
    - At-least-once vs exactly-once vs at-most-once delivery semantics.
    - This IS the async-API pattern from Q-set / Phase 3 (202 + poll/webhook).
    - For long agentic AI jobs (ingestion, report gen) this is the deployment
      shape — overlaps Q1 (offload heavy work) and the system-design phase.
'''


# =================================================================================
# CROSS-CUTTING THEMES + MASTER KEYWORD LIST
# =================================================================================
'''
HOW THE 6 QUESTIONS CONNECT (say this if asked to reflect — shows synthesis):

    RESILIENCE is the spine across Q1, Q5, Q6:
        Q1 — per-entity error handling + timeout + retry on enrichment calls.
        Q5 — timeout + cache + circuit breaker for a slow dependency.
        Q6 — durability + retries + DLQ + idempotency for background jobs.
        -> The interviewer is repeatedly probing: "do you build ROBUST systems?"

    GENAI PRACTICALITY across Q3, Q4 (+ Q1):
        Q3 — control output LENGTH (prompt + few-shot + params + routing).
        Q4 — control output RELEVANCE (grounding/RAG + persona + specifics).
        Q1 — orchestrate GenAI + a tool API in production code.
        -> Can you make LLMs behave in production, not just call them?

    DELIVERY/OWNERSHIP in Q2:
        The lead dimension — own status accuracy, engineer data quality into
        reporting, don't trust automation blindly.

    THE THREE LENSES they tested: hands-on coding (Q1), GenAI practitioner
    (Q3/Q4), senior backend/architecture (Q5/Q6), techno-managerial (Q2).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MASTER KEYWORD LIST (the vocabulary that signals senior/lead across this set):

    RESILIENCE:   timeout | retry + exponential backoff + jitter | circuit
                  breaker | bulkhead | graceful degradation / fallback |
                  connection pooling | idempotency | dead-letter queue.
    CONCURRENCY:  async/await | asyncio.gather | Semaphore (rate-limit cap) |
                  httpx.AsyncClient | worker pool | horizontal scaling.
    STRUCTURE:    Pydantic / structured output | separation of concerns |
                  per-item error isolation | typed return.
    GENAI:        few-shot | system prompt | max_tokens | grounding / context |
                  RAG | persona prompting | chain-of-thought | query routing |
                  prompt versioning + eval.
    BACKGROUND:   message queue/broker | producer-consumer | durability |
                  acknowledgement | at-least-once | DLQ | Celery / Quartz |
                  backpressure | 202 + job id.
    DELIVERY:     Red/Amber/Green | data quality / freshness | fail-safe (degrade
                  to Amber) | reconciliation | data lineage | human-in-the-loop |
                  ownership/accountability.
'''


# =================================================================================
# GOLDEN LESSONS (from this real interview)
# =================================================================================
'''
1. THE Q2 "RAG = RED/AMBER/GREEN" TRAP — context decides the meaning. In a
   PROJECT-REPORT context, RAG is project health, not retrieval. Read the domain
   of the question, not just the acronym. This one catch separates delivery-
   experienced candidates from pure coders.

2. RESILIENCE IS THE RECURRING SIGNAL — timeouts, retries+backoff, circuit
   breakers, DLQ, idempotency. When 3 of 6 questions probe robustness, the
   interviewer cares about production-grade thinking. Lead with these.

3. FOR CODING (Q1): show STRUCTURE + CONCURRENCY + PER-ITEM ERROR HANDLING +
   TYPED OUTPUT. "One failure doesn't kill the batch" + gather + Semaphore +
   timeout is the senior shape. Never a sequential loop, never raw dicts.

4. DON'T CONFLATE Q3 AND Q4 — length vs relevance. Explicitly name the
   distinction; the fix for relevance is GROUNDING/RAG, for length is
   few-shot + instructions + max_tokens + routing.

5. RESPECT THE CONSTRAINT (Q5: "no major redesign", Q3: "no retraining") —
   answering with the excluded option signals you didn't listen. Pragmatic
   quick wins win.

6. Q6 RELIABILITY = the subtle words: DURABILITY, AT-LEAST-ONCE, IDEMPOTENCY,
   DLQ. In-process threads/BackgroundTasks are NOT the answer for critical work.

7. SHOW THE TRADE-OFF + "WHY" ON EVERY ANSWER — even practical ones. "Cache
   because it's the cheapest win; trade-off is staleness, mitigated with TTL."

8. CONNECT TO THE AI USE CASE where natural — these are GenAI roles; relate
   backend/resilience answers to LLM/agent systems.

9. ADMIT THE EDGE HONESTLY — if you'd missed Q2's RAG meaning live, the recovery
   is "Let me reconsider — in a project-report context, RAG is Red/Amber/Green,
   so here's how I'd own status accuracy..." Recovering well > never slipping.

10. THESE MAP TO YOUR PHASES — Q1/Q5/Q6 -> Phases 2,3,9; Q3/Q4 -> RAG/prompt
    lessons; Q2 -> delivery/leadership (Phase 8). You already have the depth;
    this post-mortem is the application.
'''


# =================================================================================
# RUN SUMMARY
# =================================================================================

if __name__ == "__main__":
    print()
    print("=" * 70)
    print("POST-MORTEM — 6 INTERVIEW QUESTIONS: SUMMARY")
    print("=" * 70)
    print()
    print("Q1 CODING: extract -> enrich CONCURRENTLY (gather + Semaphore) ->")
    print("   per-item error handling (one failure != batch failure) -> typed list.")
    print("   Keywords: async, gather, Semaphore, httpx, Pydantic, timeout, retry.")
    print()
    print("Q2 *** RAG = RED/AMBER/GREEN (project status), NOT retrieval! ***")
    print("   Own status accuracy; data-quality + fail-safe (degrade to Amber,")
    print("   never false Green); reconciliation; lineage; human sign-off.")
    print()
    print("Q3 VERBOSE->CONCISE (no retrain): few-shot + concise prompt + max_tokens")
    print("   + query routing + post-process. (LENGTH control.)")
    print()
    print("Q4 GENERIC->SPECIFIC: grounding/RAG + persona + specific instructions")
    print("   + few-shot + chain-of-thought. (RELEVANCE control — don't repeat Q3.)")
    print()
    print("Q5 SLOW API (no redesign): TIMEOUT + CACHE + CIRCUIT BREAKER (top 3),")
    print("   + retry/backoff + fallback + pooling + bulkhead.")
    print()
    print("Q6 BACKGROUND JOBS: queue + workers; durability + ACK + at-least-once")
    print("   + IDEMPOTENCY + retries + DLQ; scale workers horizontally; Celery/")
    print("   Quartz; 202 + job id. NOT in-process threads for critical work.")
    print()
    print("SPINE: resilience (timeout/retry/circuit-breaker/DLQ/idempotency)")
    print("       runs through Q1, Q5, Q6. They probe ROBUST systems.")
    print("=" * 70)
