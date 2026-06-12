"""
===================================================================================
PRODUCTION RAG CONFIDENCE — Grounding, Attribution & Real Deployment
===================================================================================

This lesson covers the TWO questions that trip up senior GenAI candidates:

    1. "How do you know the answer came from YOUR documents vs the LLM's
       internal knowledge? How do you prevent hallucination from training data?"

    2. "Have you actually deployed a RAG application to production? Walk me
       through the vector database, indexing, retrieval, and infrastructure."

After reading this, you will answer BOTH with 100% confidence.

SECTIONS:
    PART A — GROUNDING & ATTRIBUTION
    1.  The Problem: LLM Knowledge vs Document Knowledge
    1.5 RAG Pipeline Primer (defines all terms used later)
    2.  Grounding Techniques — Force the LLM to Use Only Your Docs
    3.  Faithfulness Evaluation — Detect When the LLM Goes Off-Script
    4.  Attribution & Citation — Prove Where the Answer Came From
    5.  Retrieval Confidence Scoring — Know When to Say "I Don't Know"
    6.  Production Patterns — Context-Only Mode vs Hybrid Mode
    7.  Interview Answer Templates (Part A)

    PART B — PRODUCTION RAG DEPLOYMENT (Your Confident Story)
    8.  Your Production Narrative (what to say)
    9.  Vector DB Selection — Why Pinecone/Qdrant (Not FAISS/Chroma)
    10. Pinecone Deep Dive — Indexes, Namespaces, Metadata, Serverless
    11. Qdrant Deep Dive — Collections, Payloads, HNSW, Snapshots
    12. Indexing Pipeline — Ingestion, Chunking, Embedding, Upsert
    13. Retrieval Pipeline — Hybrid Search, Reranking, MMR
    14. Infrastructure & Operations — K8s, Queues, Monitoring
    15. Common Follow-Up Questions (with confident answers)
    16. GOLDEN LESSONS
===================================================================================
"""


# =================================================================================
# PART A — GROUNDING & ATTRIBUTION
# =================================================================================


# =================================================================================
# SECTION 1: THE PROBLEM — LLM Knowledge vs Document Knowledge
# =================================================================================
"""
WHY THIS QUESTION IS ASKED:

    Every senior interviewer knows this fundamental tension:
    - LLMs have PARAMETRIC knowledge (baked into weights during training).
    - RAG provides CONTEXTUAL knowledge (retrieved documents in the prompt).
    - When the LLM answers, you can't easily tell WHICH source it used.

    The danger:
    - LLM "knows" something from training -> generates a confident answer.
    - But that answer may be OUTDATED, WRONG, or CONTRADICTS your docs.
    - User thinks the answer came from your trusted documents.
    - In regulated industries (finance, healthcare, legal) this is a
      compliance violation.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE TWO FAILURE MODES:

    FAILURE 1 — HALLUCINATION FROM PARAMETRIC KNOWLEDGE:
        User asks: "What is our company's refund policy?"
        Retrieved docs: [nothing relevant found]
        LLM: "Your refund policy allows returns within 30 days."
        Reality: LLM made this up from general training knowledge.
        Risk: User trusts it, gives wrong info to customer.

    FAILURE 2 — BLENDING PARAMETRIC + CONTEXTUAL:
        User asks: "What are the side effects of Drug X?"
        Retrieved docs: mention 3 side effects.
        LLM: lists those 3 PLUS 2 more from its training data.
        Risk: The extra 2 may be outdated or wrong for this formulation.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE INTERVIEWER WANTS TO HEAR:

    "I understand the problem, I have multiple layers of defense, and I
    can detect when the model goes off-script."

    NOT: "I just tell the model to only use the documents."
    (That's necessary but NOT sufficient — LLMs don't always obey.)
"""


# =================================================================================
# SECTION 1.5: RAG PIPELINE PRIMER (read this first — defines all later terms)
# =================================================================================
"""
Before we go deep, here is the end-to-end RAG pipeline. Every term used
later in this lesson (chunk, embedding, retrieval gate, reranking, hybrid
search) is defined here so nothing is a forward reference.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE TWO PHASES OF RAG:

    PHASE 1 — INDEXING (offline, runs ahead of time):

        [Documents] (PDFs, web pages, Confluence...)
              |
              v
        [Load & Extract] — pull raw text out of each file format
              |
              v
        [Chunk] — split text into smaller pieces (e.g., 512 tokens each)
              |
              v
        [Embed] — convert each chunk to a vector with an embedding model
              |
              v
        [Store] — save vectors + metadata in a vector database

    PHASE 2 — RETRIEVAL + GENERATION (online, per user query):

        [User Query]
              |
              v
        [Embed the query] — same embedding model as indexing
              |
              v
        [Search] — find the nearest chunk vectors (semantic similarity)
              |
              v
        [(optional) Rerank] — re-score top candidates more precisely
              |
              v
        [Build prompt] — stuff retrieved chunks into the LLM prompt as context
              |
              v
        [LLM generates] — answer grounded in the retrieved chunks
              |
              v
        [Answer + citations] returned to the user

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ONE-LINE DEFINITIONS (used throughout this lesson):

    CHUNK: A small piece of a document (a paragraph or ~512 tokens).
    EMBEDDING: A numeric vector representing the meaning of text.
    VECTOR DB: A database that stores embeddings and finds nearest ones fast.
    RETRIEVAL: Finding the chunks most relevant to a query.
    SIMILARITY SCORE: How close a chunk vector is to the query vector (0-1).
    RETRIEVAL GATE: A check that abstains if no chunk scores high enough.
    BI-ENCODER: Fast model that embeds query and docs separately (for search).
    CROSS-ENCODER: Slower, accurate model that scores a (query, doc) pair.
    RERANKING: Re-ordering retrieved chunks with a cross-encoder for precision.
    HYBRID SEARCH: Combining vector (dense) search with keyword (BM25) search.
    GROUNDING: Constraining the LLM to answer only from retrieved context.
    FAITHFULNESS: Whether the answer's claims are supported by the context.

    Keep this map in mind. Sections 2-7 (Part A) deal with the GENERATION
    side (grounding, faithfulness). Sections 8-16 (Part B) deal with the
    full pipeline in production (DBs, indexing, retrieval, infrastructure).
"""


# =================================================================================
# SECTION 2: GROUNDING TECHNIQUES — Force the LLM to Use Only Your Docs
# =================================================================================
"""
Grounding = constraining the LLM to answer ONLY from provided context.
Multiple layers, not just one prompt instruction.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LAYER 1 — SYSTEM PROMPT CONSTRAINT (necessary but not sufficient):

    System prompt example:
    "You are a document assistant. Answer ONLY using the provided context.
     If the context does not contain the answer, say: 'I don't have enough
     information in the provided documents to answer this question.'
     Do NOT use your general knowledge. Do NOT guess."

    Why it's not enough alone:
    - LLMs are probabilistic — they sometimes ignore instructions.
    - Especially when the question is common knowledge (e.g., "what is Python?")
      the model's parametric knowledge is very strong and can override.
    - Prompt injection attacks can bypass system prompts.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LAYER 2 — STRUCTURED OUTPUT WITH CITATIONS (forces traceability):

    Force the LLM to output in a structured format:
    {
        "answer": "...",
        "sources": [
            {"doc_id": "policy_v3.pdf", "chunk_id": 42, "quote": "..."}
        ],
        "confidence": 0.85,
        "used_context_only": true
    }

    If the model can't fill "sources" with real quotes from context,
    it's a signal it's using parametric knowledge.

    Implementation: use function calling / tool_use / structured output
    mode (OpenAI JSON mode, Anthropic tool_use, etc.)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LAYER 3 — CONTEXT STUFFING WITH CLEAR DELIMITERS:

    Wrap retrieved context with explicit markers:
    ---BEGIN CONTEXT---
    [chunk 1: source=policy.pdf, page=3]
    "Returns are accepted within 14 days of purchase..."
    [chunk 2: source=faq.pdf, page=1]
    "Refunds are processed within 5 business days..."
    ---END CONTEXT---

    Then instruct: "Answer using ONLY information between BEGIN/END CONTEXT.
    Cite the [chunk N] you used."

    This makes it mechanically clear what "the documents" are.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LAYER 4 — RETRIEVAL GATE (don't call LLM if no relevant docs found):

    Before sending to LLM, check retrieval quality:
    - If top-K similarity scores are ALL below threshold (e.g., < 0.7):
      DON'T call the LLM at all.
      Return: "No relevant documents found for your question."

    This prevents the LLM from ever having the chance to hallucinate
    from parametric knowledge when your docs don't cover the topic.

    Implementation:
        scores = vector_db.search(query, top_k=5)
        if max(scores) < SIMILARITY_THRESHOLD:
            return "I don't have information about this in my documents."
        else:
            # proceed to LLM with retrieved context

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LAYER 5 — POST-GENERATION VERIFICATION (the safety net):

    After the LLM generates an answer, run a verification step:

    Option A — NLI (Natural Language Inference) model:
        Check if the answer is ENTAILED by the retrieved context.
        If the NLI model says "contradiction" or "neutral" -> flag/reject.
        Models: cross-encoder/nli-deberta-v3-base (fast, accurate).

    Option B — LLM-as-Judge:
        Second LLM call: "Given ONLY this context, is the following answer
        fully supported? Answer YES/NO with explanation."
        If NO -> suppress the answer, return fallback.

    Option C — Quote matching:
        Extract key claims from the answer.
        For each claim, find the closest sentence in retrieved context.
        If similarity < threshold -> that claim is unsupported.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LAYER 6 — TEMPERATURE AND GENERATION CONTROLS:

    - Set temperature = 0 (or very low, 0.1) for factual RAG.
      Lower temperature = less creative = less likely to invent.
    - Set max_tokens appropriately (don't let it ramble).
    - Use top_p = 0.9 or lower to constrain sampling.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SUMMARY — THE 6-LAYER GROUNDING STACK:

    1. System prompt constraint ("only use context")
    2. Structured output with mandatory citations
    3. Clear context delimiters in the prompt
    4. Retrieval gate (don't call LLM if no docs match)
    5. Post-generation verification (NLI / LLM-as-Judge)
    6. Low temperature + constrained generation

    No single layer is perfect. Together they form defense-in-depth.

INTERVIEW ANSWER:
    "We use a multi-layer grounding approach. First, the system prompt
    explicitly constrains the model to answer only from provided context.
    Second, we require structured output with source citations — if the
    model can't cite a chunk, it's a red flag. Third, we have a retrieval
    gate: if no documents score above our similarity threshold, we don't
    call the LLM at all and return 'no information available.' Fourth,
    after generation we run a faithfulness check — either an NLI model
    or a second LLM call that verifies the answer is entailed by the
    context. Finally, we keep temperature at 0 for factual queries. No
    single layer is foolproof, but together they give us high confidence
    the answer comes from our documents, not the model's training data."
"""


# =================================================================================
# SECTION 3: FAITHFULNESS EVALUATION — Detect When the LLM Goes Off-Script
# =================================================================================
"""
Even with grounding, you need to MEASURE how often the model stays faithful.
This is the evaluation layer that gives you numbers to report.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RAGAS FAITHFULNESS SCORE:

    Definition: What fraction of claims in the answer are supported by
    the retrieved context?

    How it works:
    1. Decompose the generated answer into individual CLAIMS (statements).
    2. For each claim, check: is this claim supported by the context?
    3. Faithfulness = (supported claims) / (total claims)

    Score range: 0.0 to 1.0
    Target: > 0.85 for production systems.

    Example:
        Context: "Our return policy is 14 days. Refunds take 5 business days."
        Answer: "Returns within 14 days. Refunds in 5 days. Free shipping on returns."
        Claims: [14 days: supported, 5 days: supported, free shipping: NOT supported]
        Faithfulness = 2/3 = 0.67 (FAIL — "free shipping" is hallucinated)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CONTEXT RELEVANCE (also from RAGAS):

    Definition: How much of the retrieved context is actually relevant
    to the question?

    Why it matters: if you retrieve 5 chunks but only 1 is relevant,
    the LLM has 4 chunks of noise that could confuse it.

    Context Relevance = (relevant sentences in context) / (total sentences)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ANSWER RELEVANCE (RAGAS):

    Definition: Is the answer actually addressing the question asked?
    (Not about faithfulness — about whether it's on-topic.)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

NLI-BASED FAITHFULNESS (lightweight alternative):

    Use a Natural Language Inference model:
    - Input: premise = retrieved context, hypothesis = generated answer.
    - Output: entailment / contradiction / neutral.
    - If "entailment" -> answer is grounded.
    - If "contradiction" or "neutral" -> answer may be hallucinated.

    Model: cross-encoder/nli-deberta-v3-base (runs locally, fast).
    No LLM API cost for this check.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LLM-AS-JUDGE FOR FAITHFULNESS:

    Prompt a second LLM (or same LLM, separate call):
    "Given the following context and answer, determine if EVERY claim
     in the answer is fully supported by the context.
     Context: {context}
     Answer: {answer}
     Respond with: FAITHFUL / NOT_FAITHFUL
     If NOT_FAITHFUL, list the unsupported claims."

    Pros: flexible, catches nuance.
    Cons: costs another API call, LLM judges can be inconsistent.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PRODUCTION MONITORING DASHBOARD:

    Track these metrics over time:
    - Faithfulness score (daily average)
    - % of responses that triggered "I don't know" fallback
    - % of responses flagged by NLI as not entailed
    - Retrieval hit rate (% queries with at least 1 doc above threshold)

    Alert when faithfulness drops below threshold -> investigate.

INTERVIEW ANSWER:
    "We measure faithfulness using RAGAS — it decomposes the answer into
    claims and checks each against the retrieved context. Our target is
    above 0.85. We also run an NLI model (DeBERTa-based) as a lightweight
    real-time check — if the answer contradicts or isn't entailed by the
    context, we suppress it. In production we track faithfulness scores
    on a dashboard and alert when they drop. This gives us quantitative
    proof that answers come from documents, not parametric knowledge."
"""


# =================================================================================
# SECTION 4: ATTRIBUTION & CITATION — Prove Where the Answer Came From
# =================================================================================
"""
Attribution = showing the user EXACTLY which document/chunk the answer
came from. This is both a UX feature and a trust mechanism.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

INLINE CITATION PATTERN:

    Force the LLM to cite sources inline:
    "Based on the Employee Handbook [Source 1, p.12], vacation days
     accrue at 1.5 days per month. The maximum carryover is 5 days
     [Source 1, p.14]."

    Implementation:
    - Number each retrieved chunk in the prompt: [1], [2], [3]...
    - Instruct: "Cite [N] after every claim."
    - Post-process: verify cited numbers exist in the provided chunks.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

QUOTE EXTRACTION PATTERN:

    Require the LLM to include a direct quote from the source:
    {
        "answer": "Vacation accrues at 1.5 days/month.",
        "supporting_quote": "Each full-time employee accrues 1.5 vacation
                            days per calendar month of service.",
        "source": "employee_handbook_v4.pdf",
        "page": 12
    }

    Verification: fuzzy-match the "supporting_quote" against the actual
    chunk text. If match score < 0.8 -> the LLM fabricated the quote.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HIGHLIGHT-IN-SOURCE PATTERN (advanced UX):

    After generating the answer:
    1. For each claim, find the most similar sentence in retrieved chunks.
    2. Highlight that sentence in the source document viewer.
    3. User can click the citation to see the original context.

    This is what enterprise tools like Microsoft Copilot and Glean do.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DETECTING PARAMETRIC vs CONTEXTUAL ANSWERS:

    Technique 1 — ABLATION TEST (offline evaluation):
        Run the same question WITH context and WITHOUT context.
        If the answer is nearly identical -> model used parametric knowledge.
        If the answer changes significantly -> model used context.
        Use this in your evaluation pipeline, not real-time.

    Technique 2 — COUNTERFACTUAL CONTEXT:
        Inject a MODIFIED fact in the context (e.g., "returns within 7 days"
        when the real policy is 14 days).
        If the model follows the modified context -> it's grounded.
        If it ignores it and says 14 -> it's using parametric knowledge.
        Use this for testing/evaluation, not production.

    Technique 3 — NOVEL INFORMATION TEST:
        Ask about something the LLM CANNOT know from training
        (internal company data, recent events after cutoff).
        If it answers correctly with context -> grounded.
        If it answers without context -> hallucinating.

INTERVIEW ANSWER:
    "We enforce inline citations — the model must reference [Source N]
    for every claim. We verify citations by checking that the cited chunk
    actually contains supporting text using fuzzy matching. For offline
    evaluation, we use ablation: run the same query with and without
    context. If the answer doesn't change, the model is using parametric
    knowledge. We also use counterfactual testing — inject a modified
    fact and verify the model follows the context, not its training data.
    These techniques together let us prove attribution and detect when
    grounding fails."
"""


# =================================================================================
# SECTION 5: RETRIEVAL CONFIDENCE SCORING — Know When to Say "I Don't Know"
# =================================================================================
"""
The most important production behavior: KNOWING WHEN NOT TO ANSWER.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SIMILARITY THRESHOLD GATE:

    After retrieval, check the relevance scores:

    top_docs = vector_db.search(query, top_k=5)
    max_score = max(doc.score for doc in top_docs)

    if max_score < 0.70:
        return "I don't have relevant information in my documents."
    elif max_score < 0.80:
        # Low confidence — answer but with disclaimer
        answer = generate_with_context(top_docs)
        return f"{answer}\n\nNote: confidence is moderate. Please verify."
    else:
        # High confidence — answer normally
        return generate_with_context(top_docs)

    Threshold calibration:
    - Run on a test set of known-answerable and known-unanswerable queries.
    - Find the threshold that maximizes F1 on "should I answer or not?"
    - Typical range: 0.65-0.80 depending on embedding model.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CROSS-ENCODER RERANKING AS CONFIDENCE:

    After initial retrieval (bi-encoder), rerank with a cross-encoder:
    - Cross-encoder gives a more accurate relevance score.
    - If the reranked top score is still low -> don't answer.

    Pipeline:
    1. Bi-encoder retrieves top-20 candidates (fast, approximate).
    2. Cross-encoder reranks top-20 -> top-5 (slow, accurate).
    3. If reranked top-1 score < threshold -> abstain.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MULTI-SIGNAL CONFIDENCE:

    Combine multiple signals for a robust confidence score:
    - Retrieval similarity score (from vector DB)
    - Cross-encoder reranking score
    - Number of relevant chunks found (if only 1 out of 5 is relevant, lower)
    - LLM self-reported confidence (ask it to rate 1-5)
    - NLI entailment probability

    Weighted combination -> final confidence score.
    Below threshold -> abstain or add disclaimer.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE "I DON'T KNOW" HIERARCHY:

    Level 1: No docs retrieved above threshold.
        -> "I don't have information about this in my knowledge base."

    Level 2: Docs retrieved but answer not clearly supported.
        -> "Based on available documents, I found partial information:
            [partial answer]. For a complete answer, please consult [team/source]."

    Level 3: Answer generated but faithfulness check fails.
        -> Suppress the answer, return Level 1 or Level 2 response.

INTERVIEW ANSWER:
    "We have a retrieval confidence gate. If no document scores above our
    calibrated similarity threshold (typically 0.72 for our embedding model),
    we don't call the LLM at all — we return 'no relevant information found.'
    We also use cross-encoder reranking as a second confidence signal. If
    the reranked score is borderline, we answer with a disclaimer. The key
    principle: it's better to say 'I don't know' than to hallucinate. We
    calibrated our threshold on a test set of answerable vs unanswerable
    queries to maximize the abstention F1 score."
"""


# =================================================================================
# SECTION 6: PRODUCTION PATTERNS — Context-Only Mode vs Hybrid Mode
# =================================================================================
"""
Two architectural choices for how strictly you ground the LLM.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PATTERN 1 — STRICT CONTEXT-ONLY MODE:

    Rule: The LLM may ONLY use retrieved context. Period.
    If context doesn't cover it -> "I don't know."

    When to use:
    - Legal, compliance, healthcare, finance.
    - Internal knowledge bases where accuracy > helpfulness.
    - When wrong answers have real consequences.

    Implementation:
    - Retrieval gate (Section 5)
    - System prompt: "ONLY use provided context"
    - Post-generation faithfulness check
    - Suppress any answer that fails verification

    Trade-off: Users may get frustrated by frequent "I don't know" responses.
    Mitigation: suggest where to find the answer, or escalate to human.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PATTERN 2 — HYBRID MODE WITH DISCLOSURE:

    Rule: Prefer context. If context doesn't cover it, the LLM MAY use
    general knowledge BUT must DISCLOSE the source.

    Output format:
    {
        "answer": "...",
        "source_type": "documents" | "general_knowledge" | "mixed",
        "document_sources": [...],
        "disclaimer": "This answer includes general knowledge not from
                       your documents. Please verify independently."
    }

    When to use:
    - Customer support (helpfulness matters more)
    - General Q&A assistants
    - When "I don't know" is worse than a general answer with disclaimer

    Implementation:
    - Try retrieval first.
    - If good docs found -> answer from docs, source_type = "documents".
    - If no good docs -> answer from general knowledge,
      source_type = "general_knowledge", add disclaimer.
    - UI shows different styling for each source type.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PATTERN 3 — TOOL-ROUTING WITH SOURCE TRACKING:

    In agentic RAG (like your DocSage project):
    - Agent decides: use retriever tool OR web search OR general knowledge.
    - Each tool call is logged with source metadata.
    - Final answer includes which tool provided which piece of info.

    Example flow:
    1. User asks question.
    2. Agent calls retriever_tool -> gets docs.
    3. If docs insufficient, agent calls web_search_tool -> gets web results.
    4. If still insufficient, agent uses general knowledge.
    5. Response tags each claim with its source.

    This is the most sophisticated pattern and what senior interviewers
    want to hear for agentic systems.

INTERVIEW ANSWER:
    "We use strict context-only mode for our compliance-sensitive use cases.
    The LLM can only answer from retrieved documents — if nothing relevant
    is found, it says 'I don't have this information.' For our customer
    support bot, we use hybrid mode with disclosure — the model prefers
    documents but can fall back to general knowledge with a clear disclaimer
    shown in the UI. In our agentic system, each tool call (retriever, web
    search) is logged, so we can trace exactly which source contributed to
    each part of the answer. The choice between strict and hybrid depends
    on the cost of a wrong answer in that domain."
"""


# =================================================================================
# SECTION 7: INTERVIEW ANSWER TEMPLATES (Part A — Complete)
# =================================================================================
"""
Memorize these. They cover every angle the interviewer can ask about
grounding and attribution.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q: "How do you ensure the LLM answers only from your documents?"

A: "We use defense-in-depth. Layer 1: system prompt constrains the model
    to answer only from provided context. Layer 2: we require structured
    output with mandatory source citations. Layer 3: retrieval gate — if
    no document scores above our similarity threshold, we don't call the
    LLM at all. Layer 4: post-generation faithfulness verification using
    an NLI model that checks if the answer is entailed by the context.
    Layer 5: low temperature (0) for factual queries. No single layer is
    foolproof, but together they give us very high grounding confidence."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q: "How do you detect if the answer came from training data vs documents?"

A: "Three techniques. First, ablation testing in our evaluation pipeline:
    we run the same query with and without context — if the answer doesn't
    change, the model used parametric knowledge. Second, counterfactual
    testing: we inject a modified fact and verify the model follows context.
    Third, in production we track faithfulness scores via RAGAS — if a
    claim in the answer can't be traced to a retrieved chunk, it's flagged
    as potentially from training data."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q: "What if the user asks something your documents don't cover?"

A: "Depends on the use case. For compliance-critical applications, we
    return 'I don't have this information in my knowledge base' and
    optionally suggest where to find it or escalate to a human. For
    general assistants, we use hybrid mode — the model can answer from
    general knowledge but must explicitly disclose that the answer is
    NOT from the documents, and the UI shows a different visual treatment.
    The key is transparency — never let the user think a general-knowledge
    answer came from their trusted documents."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q: "How do you evaluate faithfulness at scale?"

A: "We use RAGAS faithfulness metric in our offline evaluation pipeline.
    It decomposes answers into atomic claims and checks each against
    context. We run this on a golden test set weekly and track the score
    over time. In real-time, we use a lightweight NLI cross-encoder that
    runs in milliseconds. If faithfulness drops below 0.85, we get an
    alert and investigate — usually it means new document types were added
    that the chunking strategy doesn't handle well."
"""


# =================================================================================
# PART B — PRODUCTION RAG DEPLOYMENT (Your Confident Story)
# =================================================================================


# =================================================================================
# SECTION 8: YOUR PRODUCTION NARRATIVE (What to Say in Interviews)
# =================================================================================
"""
This is your STORY. Memorize it. Say it with confidence.
It's realistic, defensible, and matches what real teams do.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

YOUR PRODUCTION RAG STORY:

    "Yes, I've deployed a RAG application to production. Let me walk you
    through the architecture:

    USE CASE: Internal knowledge assistant for [enterprise/customer support].
    Users ask questions about company policies, product documentation,
    and technical guides. Around 50,000 documents, updated weekly.

    VECTOR DATABASE: We use Qdrant (self-hosted on Kubernetes) for our
    primary deployment. We evaluated Pinecone (managed) and Qdrant
    (self-hosted). Chose Qdrant because we needed data residency control
    and the team preferred not to send proprietary documents to a
    third-party managed service. For a client project, we used Pinecone
    Serverless because they wanted zero-ops and were okay with managed.

    EMBEDDING MODEL: OpenAI text-embedding-3-small (1536 dimensions) for
    English content. We considered open-source (BGE, E5) but the quality
    difference justified the API cost at our scale.

    CHUNKING: Recursive character splitter with 512 tokens, 50-token
    overlap. We experimented with semantic chunking but found recursive
    with good separators (headers, paragraphs) gave better retrieval
    for our document types (PDFs, Confluence pages).

    RETRIEVAL: Hybrid search — dense (vector similarity) + sparse (BM25)
    with Reciprocal Rank Fusion. Then cross-encoder reranking (ms-marco
    MiniLM) on top-20 candidates to get final top-5.

    LLM: GPT-4o for generation with temperature 0. Structured output
    with mandatory citations.

    DEPLOYMENT: FastAPI backend on Kubernetes. Async ingestion pipeline
    with Celery + Redis for document processing. CI/CD with GitHub Actions.
    Monitoring with LangSmith for tracing and custom Grafana dashboards
    for latency, retrieval hit rate, and faithfulness scores."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHY THIS STORY WORKS:

    1. SPECIFIC: mentions exact tools, models, dimensions, chunk sizes.
    2. JUSTIFIED: explains WHY each choice was made (not just what).
    3. TRADE-OFFS: shows you evaluated alternatives (Pinecone vs Qdrant).
    4. REALISTIC: matches what real production teams actually do.
    5. DEFENSIBLE: you can answer follow-ups because you understand the why.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ALTERNATIVE STORIES (pick based on the company you're interviewing at):

    FOR STARTUPS / COST-SENSITIVE:
    "We use Pinecone Serverless — zero ops, pay-per-query, scales
    automatically. Embedding with text-embedding-3-small. Simple but
    production-grade."

    FOR ENTERPRISES / DATA-SENSITIVE:
    "We self-host Qdrant on our Kubernetes cluster. All data stays within
    our VPC. Embedding model runs locally (BGE-large) for zero data egress."

    FOR AWS-HEAVY COMPANIES:
    "We use Amazon Bedrock Knowledge Bases with OpenSearch Serverless as
    the vector store. Embedding with Titan Embeddings v2. Fully managed,
    integrates with our existing AWS infrastructure."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHAT NOT TO SAY:

    NEVER: "We use FAISS in production."
        (FAISS is an in-memory library, not a database. No persistence,
        no filtering, no multi-tenancy. Screams POC.)

    NEVER: "We use ChromaDB in production."
        (ChromaDB is great for prototyping but lacks production features
        like horizontal scaling, backup/restore, access control.)

    INSTEAD: "We used FAISS during prototyping to validate the approach,
    then migrated to Qdrant/Pinecone for production because we needed
    persistence, metadata filtering, horizontal scaling, and proper
    backup/restore."

    This shows you understand the PROGRESSION from POC to production.
"""


# =================================================================================
# SECTION 9: VECTOR DB SELECTION — Why Pinecone/Qdrant (Not FAISS/Chroma)
# =================================================================================
"""
This is the question that separates POC developers from production engineers.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE PRODUCTION REQUIREMENTS THAT FAISS/CHROMA DON'T MEET:

    1. PERSISTENCE: Data survives restarts without re-indexing.
    2. HORIZONTAL SCALING: Handle millions/billions of vectors.
    3. METADATA FILTERING: Filter by tenant, date, category BEFORE search.
    4. MULTI-TENANCY: Isolate data between customers/teams.
    5. BACKUP & RESTORE: Point-in-time recovery.
    6. ACCESS CONTROL: RBAC, API keys, encryption at rest.
    7. HIGH AVAILABILITY: Replication, failover.
    8. REAL-TIME UPDATES: Upsert/delete without full re-index.
    9. MONITORING: Metrics, query latency, index health.

    FAISS: in-memory library. No persistence, no filtering, no scaling.
           Great for research and POC. Not a database.

    ChromaDB: lightweight, good DX, but single-node, limited scale,
              no enterprise features. Good for hackathons and demos.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE PRODUCTION-GRADE OPTIONS:

    PINECONE:
        Type: Fully managed (SaaS).
        Strengths: Zero ops, auto-scaling, serverless option (pay-per-query),
                   metadata filtering, namespaces for multi-tenancy.
        Weaknesses: Data leaves your infra, vendor lock-in, cost at scale.
        Best for: Teams that want zero infrastructure management.

    QDRANT:
        Type: Open-source, self-hosted OR managed cloud.
        Strengths: Rich filtering (payload-based), fast, Rust-based,
                   self-host for data residency, good Python SDK.
        Weaknesses: Self-hosting means you manage infra.
        Best for: Teams needing data control + performance.

    WEAVIATE:
        Type: Open-source, self-hosted OR managed.
        Strengths: Built-in vectorization modules, GraphQL API,
                   hybrid search native, multi-modal.
        Weaknesses: Heavier resource footprint, complex config.
        Best for: Teams wanting built-in ML pipeline integration.

    MILVUS:
        Type: Open-source, self-hosted OR Zilliz Cloud (managed).
        Strengths: Massive scale (billions of vectors), GPU acceleration,
                   multiple index types (IVF, HNSW, DiskANN).
        Weaknesses: Complex deployment (etcd, MinIO, Pulsar dependencies).
        Best for: Very large scale (100M+ vectors).

    PGVECTOR (PostgreSQL extension):
        Type: Extension on existing Postgres.
        Strengths: No new infra if you already use Postgres, ACID,
                   familiar SQL, good for < 5M vectors.
        Weaknesses: Slower than dedicated vector DBs at scale,
                   limited index options.
        Best for: Small-medium scale, teams already on Postgres.

    AWS OPENSEARCH SERVERLESS (with vector engine):
        Type: Managed AWS service.
        Strengths: Integrates with Bedrock, serverless, AWS ecosystem.
        Best for: AWS-native teams using Bedrock.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DECISION MATRIX:

    NEED                              BEST CHOICE
    Zero ops, fast start              Pinecone Serverless
    Data residency, self-host         Qdrant (Docker/K8s)
    Already on Postgres, < 5M vecs    pgvector
    Massive scale (100M+)             Milvus / Zilliz
    AWS ecosystem, Bedrock            OpenSearch Serverless
    Built-in vectorization            Weaviate
    Cost-sensitive startup            Qdrant Cloud (free tier) or pgvector

INTERVIEW ANSWER:
    "For production I use Qdrant or Pinecone depending on the client's
    constraints. Pinecone when they want zero-ops managed service and
    are okay with data leaving their infra. Qdrant self-hosted when data
    residency matters — we run it on Kubernetes with persistent volumes.
    We used FAISS during prototyping to validate retrieval quality, then
    migrated to a proper vector database for persistence, filtering,
    scaling, and backup. The key production requirements FAISS can't meet
    are metadata filtering, horizontal scaling, and real-time updates
    without full re-indexing."
"""


# =================================================================================
# SECTION 10: PINECONE DEEP DIVE — Indexes, Namespaces, Metadata, Serverless
# =================================================================================
"""
If you say "Pinecone" in an interview, you MUST be able to answer follow-ups.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PINECONE ARCHITECTURE:

    INDEX: The top-level container. Like a "database" in SQL terms.
        - Each index has a fixed DIMENSION (must match your embedding model).
        - Each index has a METRIC (cosine, euclidean, dotproduct).
        - You create one index per use case or per embedding model.

    NAMESPACE: Logical partition WITHIN an index.
        - Use for multi-tenancy (one namespace per customer/team).
        - Queries are scoped to a namespace — no cross-namespace leakage.
        - Free to create, no extra cost.
        - Example: namespace="tenant_acme", namespace="tenant_globex"

    VECTOR (Record): The actual data point.
        - id: unique string identifier
        - values: the embedding vector [0.1, 0.3, ...]
        - metadata: key-value pairs for filtering
        - sparse_values: optional sparse vector for hybrid search

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

METADATA FILTERING (critical production feature):

    Each vector can have metadata:
    {
        "source": "policy_handbook.pdf",
        "page": 12,
        "department": "HR",
        "last_updated": "2024-06-15",
        "access_level": "internal"
    }

    Query with filter:
    index.query(
        vector=query_embedding,
        top_k=5,
        namespace="tenant_acme",
        filter={
            "department": {"$eq": "HR"},
            "last_updated": {"$gte": "2024-01-01"}
        }
    )

    This is HUGE for production:
    - Access control: only return docs the user has permission to see.
    - Freshness: only return recently updated documents.
    - Scoping: only search within a specific category/department.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PINECONE SERVERLESS vs PODS:

    PODS (legacy):
        - Pre-provisioned compute (p1, p2, s1 pod types).
        - You choose pod size and replicas.
        - Pay for uptime even when idle.
        - Good for: predictable, high-throughput workloads.

    SERVERLESS (recommended for new projects):
        - Pay per query (reads) and per storage (writes).
        - Auto-scales to zero when idle.
        - No capacity planning needed.
        - Good for: variable workloads, cost optimization, getting started.

    What to say: "We use Pinecone Serverless for cost efficiency — it
    scales to zero when idle and we only pay for actual queries."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

UPSERT (how you add/update vectors):

    from pinecone import Pinecone

    pc = Pinecone(api_key="xxx")
    index = pc.Index("my-rag-index")

    # Upsert vectors (insert or update if id exists)
    index.upsert(
        vectors=[
            {
                "id": "doc_123_chunk_0",
                "values": embedding_vector,  # list of floats
                "metadata": {
                    "source": "handbook.pdf",
                    "page": 1,
                    "text": "Original chunk text for display..."
                }
            }
        ],
        namespace="tenant_acme"
    )

    Key points:
    - Upsert is idempotent: same id = update, new id = insert.
    - Batch upsert: up to 100 vectors per call for efficiency.
    - Store the chunk TEXT in metadata so you can return it without
      a separate database lookup.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

QUERYING:

    results = index.query(
        vector=query_embedding,
        top_k=5,
        namespace="tenant_acme",
        include_metadata=True,
        filter={"department": {"$eq": "Engineering"}}
    )

    for match in results["matches"]:
        print(match["id"], match["score"], match["metadata"]["text"])

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MANAGING UPDATES IN PRODUCTION:

    Document updated? -> Re-chunk, re-embed, upsert with same IDs.
    Document deleted? -> Delete by ID or by metadata filter.
    New document? -> Chunk, embed, upsert with new IDs.

    Deletion:
    index.delete(ids=["doc_123_chunk_0", "doc_123_chunk_1"], namespace="tenant_acme")
    # Or delete by filter:
    index.delete(filter={"source": {"$eq": "old_handbook.pdf"}}, namespace="tenant_acme")

INTERVIEW ANSWER:
    "In Pinecone, we create one index per embedding model with the matching
    dimension. We use namespaces for multi-tenancy — each customer's data
    is isolated. Metadata filtering lets us scope searches by department,
    date, or access level. We use Serverless for cost efficiency — pay per
    query, auto-scales to zero. For updates, upsert is idempotent: same ID
    updates the vector, new ID inserts. For deletions, we delete by metadata
    filter when a source document is removed. We store chunk text in metadata
    so retrieval returns everything in one call without a separate DB lookup."
"""


# =================================================================================
# SECTION 11: QDRANT DEEP DIVE — Collections, Payloads, HNSW, Snapshots
# =================================================================================
"""
If you say "Qdrant" in an interview, here's everything you need to defend it.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

QDRANT ARCHITECTURE:

    COLLECTION: The top-level container (like Pinecone's "index").
        - Has a fixed vector dimension and distance metric.
        - Can have MULTIPLE named vectors per point (multi-vector).

    POINT: A single record.
        - id: integer or UUID
        - vector: the embedding
        - payload: arbitrary JSON (like Pinecone's metadata, but richer)

    PAYLOAD: Qdrant's metadata system. Supports:
        - Strings, integers, floats, booleans, arrays, nested objects.
        - Full-text search on string payloads.
        - Geo-filtering (latitude/longitude).
        - Range queries, match queries, nested filters.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CREATING A COLLECTION:

    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams

    client = QdrantClient(host="localhost", port=6333)

    client.create_collection(
        collection_name="documents",
        vectors_config=VectorParams(
            size=1536,           # embedding dimension
            distance=Distance.COSINE
        )
    )

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

UPSERTING POINTS:

    from qdrant_client.models import PointStruct

    client.upsert(
        collection_name="documents",
        points=[
            PointStruct(
                id=1,
                vector=embedding_vector,
                payload={
                    "source": "handbook.pdf",
                    "page": 12,
                    "department": "HR",
                    "text": "Original chunk text...",
                    "tenant_id": "acme_corp",
                    "updated_at": "2024-06-15"
                }
            )
        ]
    )

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SEARCHING WITH FILTERS:

    from qdrant_client.models import Filter, FieldCondition, MatchValue

    results = client.search(
        collection_name="documents",
        query_vector=query_embedding,
        limit=5,
        query_filter=Filter(
            must=[
                FieldCondition(key="tenant_id", match=MatchValue(value="acme_corp")),
                FieldCondition(key="department", match=MatchValue(value="HR"))
            ]
        )
    )

    for result in results:
        print(result.id, result.score, result.payload["text"])

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HNSW INDEX CONFIGURATION:

    HNSW (Hierarchical Navigable Small World) is the default index in Qdrant.

    Key parameters:
    - m: number of edges per node (default 16). Higher = better recall,
      more memory.
    - ef_construct: search width during index building (default 100).
      Higher = better index quality, slower build.
    - ef: search width during query (default 128).
      Higher = better recall, slower query.

    Tuning for production:
    - High recall needed (medical, legal): ef=256, m=32
    - Low latency needed (real-time chat): ef=64, m=16
    - Balance: ef=128, m=16 (default is usually fine)

    client.update_collection(
        collection_name="documents",
        hnsw_config=HnswConfigDiff(m=16, ef_construct=200)
    )

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SNAPSHOTS (backup & restore):

    # Create snapshot
    client.create_snapshot(collection_name="documents")

    # List snapshots
    snapshots = client.list_snapshots(collection_name="documents")

    # Restore from snapshot (on a new instance)
    client.recover_snapshot(
        collection_name="documents",
        location="path/to/snapshot.tar"
    )

    Production pattern:
    - Nightly snapshots to S3/GCS.
    - Restore to a new instance for disaster recovery.
    - Use for blue-green deployments (index on new instance, swap traffic).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MULTI-TENANCY IN QDRANT:

    Option 1 — Payload filtering (recommended for < 100 tenants):
        Store tenant_id in payload, filter on every query.
        Simple, single collection.

    Option 2 — Separate collections (for strict isolation):
        One collection per tenant.
        Better isolation but more operational overhead.

    Option 3 — Qdrant's built-in "groups" feature:
        Group results by a payload field.

INTERVIEW ANSWER:
    "We self-host Qdrant on Kubernetes with persistent volumes. Each
    collection maps to an embedding model dimension. We use payload
    filtering for multi-tenancy — tenant_id is a required filter on every
    query. HNSW parameters are tuned for our recall/latency tradeoff
    (ef=128, m=16 for our use case). We take nightly snapshots to S3 for
    disaster recovery and use them for blue-green index deployments when
    we re-embed with a new model. The Rust-based engine gives us sub-10ms
    p99 latency at our scale of about 2 million vectors."
"""


# =================================================================================
# SECTION 12: INDEXING PIPELINE — Ingestion, Chunking, Embedding, Upsert
# =================================================================================
"""
The indexing pipeline is what runs OFFLINE to prepare your documents.
Interviewers ask: "Walk me through how documents get into your vector DB."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE PRODUCTION INDEXING PIPELINE (end-to-end):

    Step 1: DOCUMENT INGESTION
        Sources: S3 bucket, Confluence, SharePoint, Google Drive, APIs.
        Trigger: file upload event, scheduled sync, webhook.
        Formats: PDF, DOCX, HTML, Markdown, plain text, Slack exports.

        Tools:
        - unstructured.io (handles 20+ file formats)
        - PyPDF2 / pdfplumber (PDF extraction)
        - python-docx (Word documents)
        - BeautifulSoup (HTML)
        - Custom connectors for internal systems

    Step 2: PREPROCESSING
        - Extract text from raw format.
        - Clean: remove headers/footers, page numbers, boilerplate.
        - Detect language (for multilingual systems).
        - Extract metadata: title, author, date, section headers.
        - Detect tables/images (optionally OCR or describe with vision model).

    Step 3: CHUNKING
        Strategy depends on document type:

        RECURSIVE CHARACTER SPLITTING (most common):
            - Split by separators in order: ["\n\n", "\n", ". ", " "]
            - chunk_size=512 tokens, chunk_overlap=50 tokens.
            - Preserves paragraph boundaries.

        SEMANTIC CHUNKING (advanced):
            - Embed each sentence.
            - Group consecutive sentences with high similarity.
            - Split where similarity drops (topic boundary).
            - Better for long documents with clear topic shifts.

        DOCUMENT-STRUCTURE-AWARE:
            - Use headers (H1, H2, H3) as natural split points.
            - Each section becomes a chunk.
            - Best for well-structured docs (Confluence, Markdown).

        PARENT-CHILD CHUNKING:
            - Small chunks for retrieval (better precision).
            - Return the PARENT (larger context) to the LLM.
            - Retrieves on 128-token chunks, sends 512-token parents.

    Step 4: EMBEDDING
        - Embed each chunk using your chosen model.
        - Models: text-embedding-3-small (OpenAI), BGE-large, E5-large.
        - Batch embedding for efficiency (OpenAI supports batch API).
        - Cache embeddings to avoid re-computing on re-runs.

    Step 5: UPSERT TO VECTOR DB
        - Generate deterministic IDs: hash(source_path + chunk_index).
        - Attach metadata: source, page, chunk_index, text, timestamps.
        - Batch upsert (100 vectors per call for Pinecone).
        - Idempotent: re-running the pipeline updates existing vectors.

    Step 6: INCREMENTAL UPDATES (the production differentiator)
        - Track document hashes (MD5/SHA256).
        - On re-sync: compare hash with stored hash.
        - If changed: re-chunk, re-embed, upsert (same IDs = update).
        - If deleted: delete vectors by source metadata filter.
        - If new: chunk, embed, upsert with new IDs.

        This avoids full re-indexing on every sync.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PRODUCTION PIPELINE ARCHITECTURE:

    [S3/Confluence/Drive]
           |
           v (event trigger or scheduled)
    [Ingestion Worker (Celery/SQS)]
           |
           v
    [Preprocessing + Chunking]
           |
           v
    [Embedding Service (batch API)]
           |
           v
    [Vector DB Upsert (Qdrant/Pinecone)]
           |
           v
    [Metadata Store (Postgres) — tracks doc hashes, sync status]

    Monitoring:
    - Track: docs processed, chunks created, embedding latency, errors.
    - Alert on: failed ingestion, embedding API errors, upsert failures.
    - Dashboard: total docs indexed, last sync time, index size.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CHUNK SIZE DECISION (the most asked follow-up):

    Too small (< 100 tokens):
        - Loses context. Retrieves fragments that don't make sense alone.
    Too large (> 1000 tokens):
        - Dilutes relevance. Retrieves chunks where only 1 sentence matters.
    Sweet spot (256-512 tokens):
        - Enough context to be useful, focused enough for precise retrieval.

    The REAL answer: "It depends on your documents and queries. We
    experimented with 256, 512, and 1024 on a test set and measured
    retrieval recall. 512 with 50-token overlap gave us the best results
    for our document types."

INTERVIEW ANSWER:
    "Our indexing pipeline is event-driven. When a document is uploaded or
    updated in S3, it triggers a Celery worker that extracts text, chunks
    it (recursive splitter, 512 tokens, 50 overlap), embeds with
    text-embedding-3-small in batch, and upserts to Qdrant with metadata.
    We track document hashes in Postgres for incremental updates — only
    changed documents get re-processed. Deletions are handled by filtering
    vectors by source metadata. The pipeline is idempotent: re-running it
    on the same document just updates existing vectors. We monitor
    ingestion success rate, embedding latency, and total index size on
    Grafana."
"""


# =================================================================================
# SECTION 13: RETRIEVAL PIPELINE — Hybrid Search, Reranking, MMR
# =================================================================================
"""
Retrieval quality determines RAG quality. Senior interviewers test depth here.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE PRODUCTION RETRIEVAL STACK:

    Query -> [Query Transformation] -> [Hybrid Search] -> [Reranking] -> [Top-K to LLM]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP 1 — QUERY TRANSFORMATION:

    Raw user queries are often poor search queries.
    Transform before searching:

    a) Query Rewriting:
       User: "how do I get my money back?"
       Rewritten: "refund policy return process"
       Use LLM to rewrite for better retrieval.

    b) HyDE (Hypothetical Document Embeddings):
       Generate a hypothetical answer, embed THAT, search with it.
       Often retrieves better because the hypothetical answer is
       closer in embedding space to the real document than the question.

    c) Multi-Query:
       Generate 3-5 variations of the query.
       Search with each, merge results (union + deduplicate).
       Catches documents that match different phrasings.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP 2 — HYBRID SEARCH (Dense + Sparse):

    DENSE SEARCH (vector similarity):
        Embed query, find nearest vectors.
        Good at: semantic similarity, paraphrases, conceptual matches.
        Bad at: exact keyword matches, rare terms, proper nouns.

    SPARSE SEARCH (BM25 / keyword):
        Traditional text search (term frequency, inverse doc frequency).
        Good at: exact matches, rare keywords, proper nouns, acronyms.
        Bad at: paraphrases, conceptual similarity.

    HYBRID = combine both:
        Run dense search -> get top-20 by vector similarity.
        Run sparse search -> get top-20 by BM25.
        Merge using Reciprocal Rank Fusion (RRF):
            RRF_score(doc) = sum(1 / (k + rank_in_list)) for each list
            where k = 60 (constant).

    Why hybrid wins:
        Query: "GDPR compliance for EU customers"
        Dense finds: docs about data privacy regulations (semantic match).
        Sparse finds: docs with exact term "GDPR" (keyword match).
        Together: best of both worlds.

    Implementation:
    - Pinecone: supports sparse_values for hybrid natively.
    - Qdrant: use separate sparse vector + dense vector, combine.
    - Weaviate: built-in hybrid search with alpha parameter.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP 3 — RERANKING (the quality multiplier):

    After initial retrieval (top-20), rerank with a cross-encoder:

    Cross-encoder: takes (query, document) pair as input, outputs
    a relevance score. Much more accurate than bi-encoder similarity
    but too slow for full corpus search.

    Pipeline:
    1. Bi-encoder retrieves top-20 (fast, approximate).
    2. Cross-encoder scores each of the 20 (slow, accurate).
    3. Sort by cross-encoder score, take top-5.

    Models:
    - cross-encoder/ms-marco-MiniLM-L-6-v2 (fast, good)
    - BAAI/bge-reranker-v2-m3 (multilingual, strong)
    - Cohere Rerank API (managed, easy)

    Impact: typically +10-20% retrieval recall over bi-encoder alone.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP 4 — MMR (Maximal Marginal Relevance):

    Problem: top-5 results might all say the same thing (redundant).
    MMR balances RELEVANCE and DIVERSITY:

    MMR = argmax[lambda * sim(doc, query) - (1-lambda) * max(sim(doc, selected))]

    lambda = 1.0: pure relevance (may be redundant).
    lambda = 0.5: balance relevance and diversity.
    lambda = 0.0: pure diversity (may miss relevant docs).

    Use MMR when: documents have overlapping content (e.g., multiple
    versions of the same policy, similar FAQ entries).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP 5 — CONTEXT WINDOW MANAGEMENT:

    After retrieval, you have top-K chunks to send to the LLM.
    But context window is limited (and expensive).

    Strategies:
    - Stuff all top-K into prompt (simple, works for small K).
    - Map-reduce: summarize each chunk, then combine summaries.
    - Refine: iterate through chunks, refining the answer.
    - Compression: use an LLM to compress chunks to key sentences.

    Production choice: usually stuff top-5 (with 512-token chunks =
    ~2500 tokens of context, well within limits).

INTERVIEW ANSWER:
    "Our retrieval pipeline has four stages. First, query transformation —
    we use multi-query to generate 3 variations for better coverage.
    Second, hybrid search combining dense vectors with BM25 sparse search,
    merged via Reciprocal Rank Fusion. Third, cross-encoder reranking on
    the top-20 candidates using ms-marco-MiniLM — this gives us +15%
    recall over bi-encoder alone. Fourth, MMR for diversity when documents
    overlap. The final top-5 chunks go to the LLM with structured citation
    requirements. This multi-stage approach gives us much better retrieval
    quality than naive top-K vector search."
"""


# =================================================================================
# SECTION 14: INFRASTRUCTURE & OPERATIONS — K8s, Queues, Monitoring
# =================================================================================
"""
Production = code + infrastructure + operations. This section covers
what makes a RAG system production-grade beyond the ML pipeline.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PRODUCTION ARCHITECTURE DIAGRAM:

    [Users]
       |
       v
    [API Gateway / Load Balancer]
       |
       v
    [FastAPI Backend (K8s pods, autoscaled)]
       |
       +---> [Vector DB (Qdrant cluster / Pinecone)]
       |
       +---> [LLM Gateway (OpenAI / Bedrock / self-hosted)]
       |
       +---> [Cache Layer (Redis — query cache, embedding cache)]
       |
       +---> [Metadata Store (PostgreSQL)]

    [Async Indexing Pipeline]
       |
       +---> [Message Queue (Redis/SQS/RabbitMQ)]
       |
       +---> [Worker Pods (Celery / Lambda)]
       |
       +---> [Document Store (S3)]

    [Observability]
       +---> [LangSmith / LangFuse (LLM tracing)]
       +---> [Prometheus + Grafana (metrics)]
       +---> [ELK / CloudWatch (logs)]
       +---> [PagerDuty (alerts)]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

KEY INFRASTRUCTURE DECISIONS:

    SERVING FRAMEWORK: FastAPI (async, high throughput, Python native).
        - Async endpoints for non-blocking LLM calls.
        - Streaming responses (SSE) for real-time token delivery.
        - Health checks, readiness probes for K8s.

    CONTAINER ORCHESTRATION: Kubernetes.
        - Horizontal Pod Autoscaler (HPA) based on request latency.
        - Separate deployments for API, workers, vector DB.
        - Resource limits to prevent noisy neighbors.

    CACHING:
        - Query cache: exact same query -> return cached response.
        - Embedding cache: same text -> return cached embedding (saves API cost).
        - Semantic cache: similar queries -> return cached response
          (embed the query, check if a cached query is within threshold).

    MESSAGE QUEUE (for async indexing):
        - Redis (simple, fast) or SQS (managed, durable).
        - Decouples document upload from processing.
        - Handles retries, dead-letter queues for failed documents.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MONITORING & OBSERVABILITY:

    METRICS TO TRACK:
        - Query latency (p50, p95, p99)
        - Retrieval hit rate (% queries with docs above threshold)
        - Faithfulness score (daily average)
        - LLM token usage and cost
        - Error rate (failed queries, timeouts)
        - Index size (total vectors, growth rate)
        - Cache hit rate

    LLM TRACING (critical for debugging):
        - LangSmith or LangFuse: trace every chain/agent execution.
        - See: query -> retrieval -> context -> prompt -> response.
        - Debug: why did the model hallucinate? What was retrieved?
        - Cost tracking per query.

    ALERTS:
        - Latency > 5s (p95) -> scale up or investigate.
        - Faithfulness < 0.80 -> retrieval quality degraded.
        - Error rate > 5% -> something is broken.
        - Index sync failed -> documents not being updated.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DEPLOYMENT STRATEGIES:

    BLUE-GREEN FOR INDEX UPDATES:
        When re-embedding with a new model:
        1. Build new index (green) alongside current (blue).
        2. Run evaluation on green.
        3. If green is better, switch traffic.
        4. Keep blue as rollback for 24h.

    CANARY DEPLOYMENTS:
        Route 5% of traffic to new version.
        Monitor metrics. If good, ramp to 100%.

    FEATURE FLAGS:
        Toggle between retrieval strategies, chunk sizes, models
        without redeploying.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

COST OPTIMIZATION:

    - Embedding: batch API calls, cache embeddings, use smaller model
      (text-embedding-3-small vs large) if quality is acceptable.
    - LLM: cache frequent queries, use smaller model for simple questions
      (GPT-4o-mini for easy, GPT-4o for complex).
    - Vector DB: Pinecone Serverless (pay-per-query) or self-host Qdrant
      (fixed infra cost, better at scale).
    - Reranking: only rerank when retrieval confidence is borderline.

INTERVIEW ANSWER:
    "Our production stack is FastAPI on Kubernetes with HPA for autoscaling.
    Qdrant runs as a StatefulSet with persistent volumes. Async indexing
    uses Celery workers with Redis as the broker. We trace every query
    end-to-end with LangSmith — retrieval, context, prompt, response —
    which is critical for debugging hallucinations. Monitoring on Grafana
    tracks latency, retrieval hit rate, faithfulness scores, and cost.
    We use blue-green deployments for index updates when we change the
    embedding model, and semantic caching to reduce redundant LLM calls
    by about 30%."
"""


# =================================================================================
# SECTION 15: COMMON FOLLOW-UP QUESTIONS (with confident answers)
# =================================================================================
"""
These are the exact follow-ups interviewers ask after you describe your
production RAG system. Memorize these answers.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q: "How many documents do you have indexed?"
A: "Around 50,000 documents, which translates to roughly 2 million chunks
    after splitting. The index grows by about 500 documents per week as
    teams add new content."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q: "What's your query latency?"
A: "End-to-end p95 is about 2.5 seconds. Breakdown: retrieval ~100ms,
    reranking ~200ms, LLM generation ~2 seconds (streaming, so first
    token appears in ~500ms). We use streaming SSE so users see tokens
    appearing immediately."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q: "How do you handle document updates?"
A: "Incremental. We track document hashes in Postgres. On each sync cycle,
    we compare hashes. Changed documents get re-chunked, re-embedded, and
    upserted (same deterministic IDs, so it's an update). Deleted documents
    trigger a delete-by-metadata-filter call. No full re-indexing needed."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q: "How do you handle multi-tenancy?"
A: "In Pinecone we use namespaces — one per tenant. In Qdrant we use
    payload filtering with a mandatory tenant_id filter on every query.
    This ensures complete data isolation without the overhead of separate
    collections per tenant."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q: "What embedding model do you use and why?"
A: "OpenAI text-embedding-3-small (1536 dimensions). We evaluated it
    against BGE-large and E5-large on our domain-specific test set.
    OpenAI won by about 3% on retrieval recall, and the API cost at our
    query volume (~10K queries/day) is acceptable. For a client with
    strict data residency, we used BGE-large running locally."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q: "How do you evaluate retrieval quality?"
A: "We maintain a golden test set of 200 question-answer pairs with
    annotated relevant documents. We measure:
    - Recall@5 (does the correct doc appear in top 5?)
    - MRR (Mean Reciprocal Rank — how high is the correct doc?)
    - RAGAS faithfulness and answer relevance on the full pipeline.
    We run this weekly and after any pipeline change."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q: "What happens when the LLM API is down?"
A: "We have a fallback chain. Primary: GPT-4o. If OpenAI is down or
    latency exceeds 10s, we fall back to Claude 3.5 Sonnet via Bedrock.
    If both are down, we return a graceful error: 'Service temporarily
    unavailable, please try again.' We also cache frequent queries in
    Redis, so cached responses still work during outages."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q: "How do you handle large PDFs (100+ pages)?"
A: "We process them page by page. Extract text per page, then chunk each
    page. Metadata includes page number so citations can point to the
    exact page. For tables, we use unstructured.io's table extraction
    and store tables as separate chunks with 'type: table' metadata.
    For images/diagrams, we optionally use a vision model to generate
    text descriptions."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q: "How do you handle conversational context (follow-up questions)?"
A: "We maintain conversation history and use it for query contextualization.
    Before retrieval, we pass the conversation history + current question
    to the LLM with a prompt: 'Rewrite this question to be standalone
    given the conversation history.' The standalone question is what we
    search with. This handles pronouns and references like 'what about
    the second one?' correctly."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q: "What's your chunking strategy and why?"
A: "Recursive character splitting with 512 tokens and 50-token overlap.
    We tested 256, 512, and 1024 on our evaluation set. 512 gave the best
    balance of retrieval precision and context completeness. The overlap
    prevents information loss at chunk boundaries. For structured docs
    (Markdown, Confluence), we split on headers first, then apply size
    limits within sections."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q: "How do you handle access control in RAG?"
A: "Metadata-based filtering. Each document chunk carries an access_level
    and department in its metadata. When a user queries, we resolve their
    permissions from our auth system and add the appropriate metadata
    filter to the vector search. Users only see results from documents
    they have access to. This is enforced at the retrieval layer, not
    just the UI."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q: "Have you dealt with hallucination issues? How?"
A: "Yes, early on our faithfulness score was around 0.72. We improved it
    to 0.91 through: (1) better chunking that preserved more context,
    (2) adding cross-encoder reranking which improved retrieval relevance,
    (3) stricter system prompt with mandatory citations, (4) post-generation
    NLI check that suppresses unfaithful answers, and (5) retrieval gate
    that returns 'I don't know' when no relevant docs are found. The
    biggest single improvement was the retrieval gate — it eliminated
    the cases where the model was generating from parametric knowledge."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q: "Why not just fine-tune the LLM on your documents instead of RAG?"
A: "Three reasons. First, our documents update weekly — fine-tuning is
    too slow and expensive for that cadence. RAG picks up new docs
    immediately after indexing. Second, fine-tuning bakes knowledge into
    weights without clear attribution — you can't cite which document
    the answer came from. Third, fine-tuning requires significant data
    and compute. RAG gives us updatable, attributable, cost-effective
    knowledge injection. We'd consider fine-tuning for STYLE and FORMAT
    (e.g., always respond in a specific tone), but not for factual
    knowledge that changes."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q: "What's the cost of running this in production?"
A: "Roughly: embedding costs ~$50/month for our volume (10K queries/day
    + weekly re-indexing). LLM generation ~$500/month (GPT-4o, average
    2K tokens per response). Vector DB: Qdrant self-hosted on 2 nodes
    ~$200/month infra. Total: under $1000/month for a system serving
    500 internal users. The semantic cache reduces LLM costs by ~30%."
"""


# =================================================================================
# SECTION 16: GOLDEN LESSONS
# =================================================================================
"""
The principles that make you sound like a production veteran, not a POC builder.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GOLDEN LESSON 1 — NEVER SAY FAISS/CHROMA FOR PRODUCTION.
    Say: "We used FAISS during prototyping, then migrated to Qdrant/Pinecone
    for production." This shows you understand the progression.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GOLDEN LESSON 2 — GROUNDING IS DEFENSE-IN-DEPTH, NOT ONE PROMPT LINE.
    System prompt + structured citations + retrieval gate + NLI verification
    + low temperature. No single layer is enough. Say all five.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GOLDEN LESSON 3 — ALWAYS EXPLAIN WHY, NOT JUST WHAT.
    "We use Qdrant" is junior. "We use Qdrant because we needed data
    residency and self-hosting, and its Rust engine gives us sub-10ms
    p99 latency" is senior.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GOLDEN LESSON 4 — KNOW YOUR NUMBERS.
    50K documents, 2M chunks, 10K queries/day, p95 latency 2.5s,
    faithfulness 0.91, recall@5 = 0.87. Numbers make you credible.
    Vague answers ("a lot of documents") make you suspicious.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GOLDEN LESSON 5 — HYBRID SEARCH + RERANKING IS THE PRODUCTION STANDARD.
    Naive vector search is POC. Production uses dense + sparse (hybrid)
    with cross-encoder reranking. Say this and you sound like you've
    actually optimized retrieval quality.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GOLDEN LESSON 6 — INCREMENTAL INDEXING, NOT FULL RE-INDEX.
    "We track document hashes and only re-process changed documents."
    This shows you've thought about operational efficiency at scale.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GOLDEN LESSON 7 — THE RETRIEVAL GATE IS YOUR BEST ANTI-HALLUCINATION TOOL.
    Don't call the LLM when you have no relevant documents. This single
    pattern eliminates the entire class of "answering from parametric
    knowledge" failures.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GOLDEN LESSON 8 — OBSERVABILITY IS NOT OPTIONAL.
    LangSmith/LangFuse tracing, Grafana dashboards, alerting on
    faithfulness drops. If you can't debug a bad answer in production,
    you don't have a production system.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GOLDEN LESSON 9 — HAVE A FALLBACK STORY.
    "What if OpenAI is down?" -> "We fall back to Claude via Bedrock."
    "What if retrieval fails?" -> "We return 'I don't know' gracefully."
    Production engineers think about failure modes.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GOLDEN LESSON 10 — EVALUATION IS CONTINUOUS, NOT ONE-TIME.
    Golden test set, weekly RAGAS runs, faithfulness monitoring, retrieval
    recall tracking. This is what separates "I deployed it" from "I
    maintain it in production."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GOLDEN LESSON 11 — CONFIDENCE COMES FROM DEPTH, NOT BLUFFING.
    You now know: Pinecone namespaces, Qdrant payloads, HNSW tuning,
    hybrid search with RRF, cross-encoder reranking, NLI faithfulness,
    retrieval gates, incremental indexing, blue-green deployments.
    This is MORE than most production teams know. Say it with confidence.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ONE-LINE CHEAT SHEET FOR RAPID FIRE:

    Grounding: "System prompt + citations + retrieval gate + NLI + temp 0."
    Detect parametric: "Ablation test — run with/without context, compare."
    Vector DB: "Qdrant self-hosted for data residency, Pinecone for zero-ops."
    Indexing: "Event-driven, incremental, hash-based change detection."
    Retrieval: "Hybrid (dense + BM25) + cross-encoder reranking + MMR."
    Monitoring: "LangSmith traces + Grafana metrics + faithfulness alerts."
    Fallback: "Retrieval gate + LLM fallback chain + graceful degradation."
    Cost: "Semantic cache + smaller model for easy queries + batch embedding."
    Updates: "Upsert with deterministic IDs, delete by metadata filter."
    Multi-tenancy: "Namespaces (Pinecone) or payload filtering (Qdrant)."
"""


# =================================================================================
# RUN SUMMARY
# =================================================================================

if __name__ == "__main__":
    sections = [
        "--- PART A: GROUNDING & ATTRIBUTION ---",
        "1.  The Problem: LLM Knowledge vs Document Knowledge",
        "2.  Grounding Techniques (6-Layer Defense-in-Depth)",
        "3.  Faithfulness Evaluation (RAGAS, NLI, LLM-as-Judge)",
        "4.  Attribution & Citation (Inline, Quote Extraction, Ablation)",
        "5.  Retrieval Confidence Scoring (Threshold Gate)",
        "6.  Production Patterns (Context-Only vs Hybrid Mode)",
        "7.  Interview Answer Templates (Part A)",
        "",
        "--- PART B: PRODUCTION RAG DEPLOYMENT ---",
        "8.  Your Production Narrative (what to say confidently)",
        "9.  Vector DB Selection (Pinecone/Qdrant/Weaviate/Milvus/pgvector)",
        "10. Pinecone Deep Dive (Indexes, Namespaces, Metadata, Serverless)",
        "11. Qdrant Deep Dive (Collections, Payloads, HNSW, Snapshots)",
        "12. Indexing Pipeline (Ingestion, Chunking, Embedding, Upsert)",
        "13. Retrieval Pipeline (Hybrid Search, Reranking, MMR)",
        "14. Infrastructure & Operations (K8s, Queues, Monitoring)",
        "15. Common Follow-Up Questions (12 Q&A with confident answers)",
        "16. GOLDEN LESSONS (11 principles + rapid-fire cheat sheet)",
    ]

    print("=" * 80)
    print("LESSON 20: PRODUCTION RAG CONFIDENCE")
    print("Grounding, Attribution & Real Deployment")
    print("=" * 80)
    print()
    for s in sections:
        print(f"  {s}")
    print()
    print("-" * 80)
    print("After reading this lesson, you can confidently say:")
    print('  "Yes, I have deployed RAG to production."')
    print('  "Here is how we ensure answers come from documents, not training data."')
    print("-" * 80)
    print()
    print("KEY SOUNDBITES:")
    print("  Grounding: System prompt + citations + retrieval gate + NLI + temp 0")
    print("  Vector DB: Qdrant (self-host) or Pinecone (managed), never FAISS in prod")
    print("  Retrieval: Hybrid search + cross-encoder reranking + MMR")
    print("  Indexing:  Event-driven, incremental, hash-based change detection")
    print("  Monitoring: LangSmith traces + Grafana + faithfulness alerts")
    print()
    print("=" * 80)
