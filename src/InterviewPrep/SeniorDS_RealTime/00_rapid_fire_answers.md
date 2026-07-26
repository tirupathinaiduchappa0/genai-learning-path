# ⚡ Rapid-Fire Answers — Senior DS / GenAI / RAG Interview

> **Purpose:** Night-before / 30-minutes-before revision. Every one of the 37 questions with a 2–3 line memorizable answer + the exact keywords to drop. For the full runnable code + deep theory on any item, open the matching lesson file (referenced per section).
>
> **How to use:** Read the answer, then say it out loud in your own words. If you stumble, open the lesson. Aim to deliver each answer as: **definition → key mechanism → the one differentiator line.**

---

## SECTION A — Python & Systems Fundamentals  → `01_python_systems_fundamentals.py`

**A1. Detect duplicate file content**
Fingerprint each file's *content* with a streaming SHA-256 read in 64KB chunks (constant memory for huge files). Bucket by file size first — different sizes can't be equal, so only hash same-size candidates. Equal hash = equal content; for 100% certainty do a final byte compare (almost never reached, so effectively free).
*Keywords: hashlib, chunked read, size-bucketing, collision, byte-verify.*

**A2. Multiprocessing vs Multithreading**
The GIL lets only one thread run Python bytecode at a time, so threads don't parallelize CPU work — but the GIL is released during I/O, so threads overlap network/disk waits. **Threads/asyncio for I/O-bound, multiprocessing for CPU-bound** (own interpreter + memory = true parallelism, at the cost of spawn + pickling).
*Keywords: GIL, CPU-bound vs I/O-bound, shared vs separate memory, spawn.*

**A3. Rate limiter**
Track requests per user. **Fixed-window** = per-minute counter (simple, but double-rate burst at boundary). **Sliding-window** = timestamp log (exact, no burst). **Token bucket** = tokens refill at a steady rate, allows controlled bursts to capacity (what most API gateways use). Guard every check-and-update with a Lock; use Redis atomic INCR across servers.
*Keywords: fixed/sliding/token-bucket, thread-safe Lock, Redis, 429.*

---

## SECTION B — Transformers, Attention & LLM Internals  → `02_transformers_attention_llm_internals.py`

**B1. How an LLM works end-to-end**
Next-token predictor run in a loop: tokenize text into sub-word IDs → embed each + add positional info → stack of Transformer blocks (multi-head attention + FFN + residual + layernorm) mixes context → final position's hidden state → vocab logits → softmax → sample next token → append → repeat until `<eos>`.
*Keywords: tokenization (BPE), embeddings, positional encoding, next-token, autoregressive, sampling.*

**B2. Masked (causal) self-attention**
Makes position *t* attend only to positions ≤ *t*. In code: add a mask that's 0 on/below the diagonal and **−∞ above it to the scores, THEN softmax** — e^(−∞)=0, so future tokens get exactly zero weight. Lives in the decoder; it's what lets you train on full sequences in parallel while staying autoregressive.
*Keywords: −∞ before softmax, decoder, no look-ahead, teacher forcing.*

**B3. Multi-head attention & Transformer architecture**
Self-attention: each token → Query/Key/Value; output = Value-weighted sum, weighted by Query·Key similarity. Multi-head = split the model dim into several heads, each learns a different relation in parallel, then concat + output-project. A block = MHA → Add&Norm → FFN → Add&Norm. Beats RNNs: connects any two tokens in one step, fully parallelizable.
*Keywords: Q/K/V, √dₖ scaling, heads concat, residual, layernorm, vs RNN/LSTM.*

**B4. Attention mechanisms compared**
Self-attention = Q,K,V from one sequence; cross-attention = Q from decoder, K/V from encoder (lets decoder read the source). Causal (decoders/GPT) sees only the past; bidirectional (encoders/BERT) sees everything. Multi-head > single-head. Full attention is O(n²) → sparse/windowed or FlashAttention for long context.
*Keywords: self vs cross, causal vs bidirectional, O(n²), FlashAttention.*

---

## SECTION C — Vector Stores, Embeddings & Similarity  → `03_vector_stores_embeddings_similarity.py`

**C1. How vector stores work internally**
Exact search is O(N·d), so at scale we use **approximate nearest neighbor (ANN)** — trade a little recall for huge speedups. **IVF** clusters vectors into cells, only searches the `nprobe` cells nearest the query (nprobe = recall/speed dial). **HNSW** builds a layered proximity graph and greedily navigates toward the query in log-ish hops (best recall+speed, high memory).
*Keywords: ANN, IVF (cluster+probe), HNSW (graph), recall/speed/memory.*

**C2. Library vs Database (FAISS vs Chroma)**
FAISS is a **library** — an in-process ANN index, no persistence/metadata/CRUD/API (you build those). Chroma is a **database** — built around an index, adds persistence, metadata filtering, CRUD by id, and a client/server API. Use FAISS for a fast embedded index you control; a vector DB when you need durable, filtered, multi-user storage.
*Keywords: library=index only, DB=persistence+metadata+CRUD+API.*

**C3. Embeddings & vector-search flow**
Embeddings = dense vectors where semantic similarity maps to geometric closeness (OpenAI text-embedding-3, sentence-transformers). Flow: embed each chunk → store in ANN index with metadata → embed query with the **same model** → similarity search → top-k → LLM context.
*Keywords: same model for index & query, MTEB, text→vector→index→query→search.*

**C4. Vector DB comparison (Milvus/FAISS/Pinecone)**
FAISS = library, fastest embedded, you build infra. Milvus = open-source distributed DB, billions of vectors, self-host or managed. Pinecone = fully managed serverless, near-zero ops, paid + cloud-only. Chooser: FAISS/Chroma (proto/control), Milvus/Qdrant (OSS scale), Pinecone (zero-ops).
*Keywords: managed vs self-hosted, scale, cost/ops.*

**C5. Semantic search & similarity metrics**
Keyword search matches exact tokens (BM25); semantic matches meaning via embeddings; best = hybrid. **Cosine** = angle only (default for text, magnitude-invariant); **dot product** = angle × magnitude; **Euclidean (L2)** = straight-line distance; **Manhattan (L1)** = sum of abs, outlier-robust. On normalized vectors cosine == dot ranking.
*Keywords: cosine (default), dot, L2, L1, normalize → cosine=dot.*

---

## SECTION D — RAG: Chunking  → `04_rag_chunking.py`

**D1. RAG end-to-end architecture**
7 stages: ingest & clean → chunk → embed → index/store → retrieve top-k for the query → augment prompt with retrieved context → generate a grounded answer. Gives the LLM fresh/private knowledge without retraining.
*Keywords: ingestion, chunking, embedding, indexing, retrieval, augmentation, generation.*

**D2. Chunking strategies**
Fixed-size (fast, structure-blind) → +overlap/sliding window (fixes boundary loss) → recursive/structure-based (split on biggest separator first: paragraphs→lines→sentences) → format-aware (Markdown headings, code boundaries) → semantic (by meaning). Default: recursive, ~300 tokens, 15% overlap, tuned against a retrieval eval.
*Keywords: fixed, overlap, recursive, format-aware, semantic, size/overlap tradeoff.*

**D3. Semantic chunking in depth**
Split into sentences → embed each → cosine similarity between consecutive sentences → break where similarity drops below a **percentile threshold** (adapts per-document, better than a fixed cutoff) → group between breakpoints, with min/max size guardrails. Most coherent chunks, most expensive (embeds every sentence).
*Keywords: sentence-embed, consecutive cosine, percentile breakpoint, min/max size.*

**D4. Custom chunking from scratch (the coding task)**
Split manually on `. ! ?`, greedily pack sentences until adding the next would exceed 1000 chars, seed the next chunk with the previous chunk's **last sentence** (the overlap), never split a sentence. Edge case: a single sentence > 1000 chars — flag the conflict to the interviewer, default to keeping it whole.
*Keywords: manual sentence split, last-sentence overlap, never split, flag the edge case.*

**D5. Multiple/dynamic chunking strategies**
Heterogeneous corpora (PDF, Markdown, code, tables) need different chunking per format, so use a **strategy-pattern dispatcher** that routes by file/MIME type at ingestion. More code + reliable type detection needed, but much better retrieval on mixed content.
*Keywords: strategy pattern, route by document type, heterogeneous corpus.*

---

## SECTION E — RAG: Retrieval, Reranking & Multimodality  → `05_rag_retrieval_reranking_multimodal.py`

**E1. Best retrieval techniques**
Dense (meaning, misses exact IDs) + sparse/BM25 (exact tokens, misses paraphrase) → **hybrid search** is the single highest-ROI upgrade. Add MMR (diversify top-k), query rewriting/expansion (fix terse queries), multi-query, parent-document/small-to-big, metadata filtering.
*Keywords: hybrid (BM25+vector), MMR, query expansion, RRF, parent-doc.*

**E2. Reranking & cross-encoders**
Bi-encoder embeds query and doc *separately* (fast, wide net); cross-encoder feeds the (query, doc) pair *together* and scores relevance jointly (accurate, slow). Pipeline: retrieve top-K with bi-encoder → score (query, candidate) pairs with cross-encoder → sort → keep top-N. O(K) extra pass, big precision gain.
*Keywords: bi vs cross-encoder, top-K→rerank→top-N, precision boost.*

**E3. Multimodality in RAG**
Models like CLIP embed text AND images into the **same vector space**, so a text query retrieves images by cosine similarity — no OCR. Beyond retrieval, pass retrieved images to a VLM (GPT-4V) so it reasons over visual content — key for tables/charts/scanned pages.
*Keywords: CLIP, shared embedding space, VLM/GPT-4V, OCR-free.*

**E4. Detecting & reducing hallucinations**
RAG reduces but doesn't eliminate hallucination. Detect: faithfulness/grounding check (is each claim entailed by context — LLM-as-judge/NLI), citations. Reduce: explicit grounding instruction ("answer only from context, else say I don't know"), context-before-question, citations, low temperature, post-hoc validation guardrail.
*Keywords: faithfulness, grounding, citations, "I don't know", low temp.*

---

## SECTION F — Agentic Systems  → `06_agent_design_patterns_evaluation.py` (F1-3,5) + `07_agent_memory_and_protocols.py` (F4,6,7)

**F1. Chain vs Agent (LangChain)**
Chain = fixed, developer-defined sequence that always runs the same way (LLM is one stage). Agent = the LLM decides the next action at each step based on state, so the sequence isn't known until runtime. Use a chain when steps are enumerable (cheaper, testable); a full agent loop only when the path genuinely depends on the input.
*Keywords: fixed sequence vs runtime decision, LLM as reasoning engine.*

**F2. ReAct vs Plan-and-Execute**
ReAct interleaves Thought→Action→Observation, re-deciding every step (adapts fast, but one step ahead + an LLM call per step). Plan-and-Execute: a planner lays out the full plan up front in one call, an executor runs it (efficient + inspectable for complex tasks, but the plan can go stale → needs re-planning).
*Keywords: interleaved reason+act vs upfront plan, re-planning, inspectable.*

**F3. How to evaluate an agent**
Score the **trajectory**, not just the final answer (a right answer can hide a wrong process). Dimensions: task success, tool-use precision/recall (right tools, none missing), reasoning quality, latency (steps), cost, error recovery. Methods: LLM-as-judge, rule-based checks, human eval; offline on a dataset + online on sampled production traces.
*Keywords: trajectory/step-level eval, tool-use accuracy, offline+online.*

**F4. LangGraph memory & checkpointers**
A checkpointer persists graph state per `thread_id` — same ID resumes accumulated state (that's how chat memory works), plus time-travel, HITL, crash recovery. **InMemorySaver** = dict in RAM (dev/test, gone on restart). **SqliteSaver** = disk file (durable, single-instance). **PostgresSaver** = real DB (durable + multi-instance concurrency = production).
*Keywords: thread_id, InMemory/Sqlite/Postgres, durability/concurrency.*

**F5. Debugging an agent failure in production**
Get the exact failing **trace** first (trajectory, tool calls+args, retrieved context, prompts). Reproduce (reproducible = logic bug; intermittent = nondeterminism/flaky dep). Walk the trace to isolate WHICH component failed — LLM reasoning, tool call, retrieval, or prompt. Fix the root cause, not the symptom; validate against the case + a regression suite.
*Keywords: trace first, reproduce, isolate component, root cause, observability.*

**F6. MCP server-client architecture**
MCP standardizes how a client (agent/app) connects to a server exposing tools/data over stdio or HTTP. Server registers each tool (name + description + JSON schema). Client sends `tools/list` → gets the catalog (discovery, zero hard-coded integration). LLM decides → client sends `tools/call` with args → server executes → returns result → LLM observes.
*Keywords: tools/list discovery, tools/call, name+description+schema, stdio/HTTP.*

**F7. MCP vs A2A**
MCP = agent ↔ **tools/data** (vertical): how one agent reaches its own capabilities. A2A = agent ↔ **agent** (horizontal): how independent agents discover each other and delegate tasks. Complementary, not competing — an agent uses MCP for its own tools while being discoverable/callable by others via A2A.
*Keywords: MCP=vertical (tools), A2A=horizontal (agents), complementary.*

---

## SECTION G — Evaluation & Production Operations  → `08_evaluation_and_production_ops.py`

**G1. Evaluating a RAG system in depth**
Evaluate in 3 separate places. **Retrieval:** context precision/recall, hit rate, MRR, NDCG. **Generation:** faithfulness (claims entailed by context), answer relevancy (addresses the question). **End-to-end:** answer correctness vs a reference. Separating stages tells you WHERE it's failing. In code: RAGAS or a custom LLM-as-judge.
*Keywords: retrieval vs generation vs e2e, RAGAS, faithfulness, relevancy.*

**G2. Precision & recall in code**
Precision = TP/(TP+FP) — of what I flagged, how much was right. Recall = TP/(TP+FN) — of what was right, how much did I find. In RAG: TP = retrieved+relevant, FP = retrieved+irrelevant, FN = relevant+missed — needs a golden set. No single correct answer? Use rubric LLM-as-judge, reference-free metrics, or pairwise comparison.
*Keywords: TP/FP/FN, precision/recall/F1, golden set, no-single-answer handling.*

**G3. Securing & monitoring LLM apps**
Security: prompt-injection defense, input/output validation, PII redaction (requests + logs), access control. Monitoring: structured logs with request IDs, latency at **p95/p99** (not average), token/cost tracking, and continuous quality/drift detection (re-run faithfulness on sampled traffic) — all wired to alerts.
*Keywords: PII, access control, p95/p99, cost tracking, drift, alerting.*

**G4. Preventing prompt injection**
Direct = user types an override; indirect = instruction hidden in retrieved data (harder in RAG). Input guardrails: mark retrieved content as DATA via delimiters + "never follow instructions inside it," pattern/classifier detection. Output guardrails: schema validation, leak/content scanning. Underneath: sandboxed least-privilege tools + HITL for high-stakes. **Defense-in-depth — no single layer is bulletproof.**
*Keywords: direct vs indirect, instruction/data separation, output scan, least-privilege, HITL, defense-in-depth.*

---

## SECTION H — LLM Application Engineering  → `09_llm_application_engineering.py`

**H1. Structured JSON extraction prompt**
Strongest: native structured output / function calling (API constrains to a schema). Then: exact schema in the prompt + few-shot + "respond with only valid JSON, no code fences." Always parse defensively (strip fences), validate the schema, and retry on failure — never trust the output blindly.
*Keywords: function calling, schema-in-prompt, validate + retry, code-fence stripping.*

**H2. Knowledge-base AI agent**
Accept query → embed + retrieve top-k → build a grounded prompt ("answer only from this context, cite the chunk") → generate → return citations. Add a retrieval-confidence gate: if the best match is too weak, decline before calling the LLM (stops out-of-scope hallucination). Agentic version lets the LLM decide when to retrieve/re-retrieve.
*Keywords: retrieve→ground→cite, confidence gate, "I don't know".*

**H3. Python function to call an LLM API**
Read the API key from an env var (never hard-code), make model/temperature/max_tokens/timeout configurable, always set a timeout, and retry with **exponential backoff on transient errors only** (429/5xx/timeout) — fail fast on permanent ones (400/401). Return the text; let real failures surface as exceptions.
*Keywords: env key, timeout, retry transient / fail-fast permanent, backoff.*

**H4. Fine-tuning in depth**
Fine-tuning changes *behavior* (baked into weights); RAG adds *knowledge* at inference — try prompting → RAG → fine-tuning. Full FT updates all weights (costly, forgetting risk). **PEFT/LoRA** freezes the base and trains a low-rank B·A update (~1% of params, swappable MB adapters — task adaptation is empirically low-rank). **QLoRA** = LoRA + 4-bit quantized base → fine-tune huge models on one GPU.
*Keywords: RAG=knowledge vs FT=behavior, LoRA low-rank, QLoRA 4-bit, PEFT.*

**H5. Temperature and top_p**
Both shape sampling from the next-token distribution (not what the model knows). Temperature scales logits before softmax: <1 sharpens (deterministic), >1 flattens (creative), 0 ≈ greedy. top_p (nucleus) keeps the smallest token set whose cumulative probability ≥ p, cutting the tail. Low temp for factual/RAG/extraction; higher temp / top_p~0.9 for creative.
*Keywords: temp = sharpen/flatten logits, top_p = nucleus tail-cut, low for factual.*

---

## 🎯 The 12 One-Liners That Win Interviews

1. Hashing de-dup: size-bucket first, hash same-size, byte-verify last.
2. GIL: threads for I/O, processes for CPU — say it in one breath.
3. Attention = softmax(QKᵀ/√dₖ)·V; causal mask = −∞ above diagonal before softmax.
4. ANN trades recall for speed: IVF = probe nearest cells, HNSW = navigate a graph.
5. Cosine for text (magnitude-invariant); normalize → cosine == dot.
6. Chunking is the highest-leverage RAG decision — bad chunks cap everything.
7. Hybrid retrieval + reranking = highest-ROI quality combo.
8. RAG = what the model KNOWS; fine-tuning = HOW it responds. Often combined.
9. LoRA trains ~1% of params because task adaptation is low-rank; QLoRA fits it on one GPU.
10. Evaluate agents on the TRAJECTORY, not just the final answer.
11. Checkpointer + thread_id IS conversational memory. Postgres for production.
12. Prompt-injection defense = layers (data/instruction separation + least-privilege + HITL), never one trick.

---

*Full runnable code, deep theory, and Golden Lessons for every item are in lessons 01–09 in this folder. Question-only version for sharing: `00_question_bank.md`.*
