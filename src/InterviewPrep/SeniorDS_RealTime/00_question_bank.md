# 🎯 Senior Data Scientist / GenAI / RAG — Real-Time Interview Question Bank

> **What this is:** A curated set of **37 real interview questions** (deduped from 41) asked in recent Senior Data Scientist / GenAI Engineer / Applied-RAG interviews. These are the deep, hands-on, "do you actually understand it" questions — a mix of runnable coding/skeleton tasks and in-depth theory.
>
> **How to use it:** Each question lists the exact sub-points the interviewer expected you to cover. Work through them on your own first — try to answer out loud or in code before looking anything up. If you can address every sub-bullet confidently, you're interview-ready on that topic.
>
> **Level:** Senior (3+ yrs). Expect follow-ups that go one layer deeper than the question itself.

---

## 📚 Contents

- **Section A** — Python & Systems Fundamentals (3)
- **Section B** — Transformers, Attention & LLM Internals (4)
- **Section C** — Vector Stores, Embeddings & Similarity (5)
- **Section D** — RAG: Chunking (5)
- **Section E** — RAG: Retrieval, Reranking & Multimodality (4)
- **Section F** — Agentic Systems (7)
- **Section G** — Evaluation & Production Operations (4)
- **Section H** — LLM Application Engineering (5)

**Total: 37 questions across 8 topic areas.**

---

## SECTION A — Python & Systems Fundamentals

### A1. Detecting Duplicate File Content
Write a program that identifies duplicate files within a directory (or set of directories) based on their **content**, not their filenames.
- Use a hashing approach (`hashlib` with MD5/SHA-256) to fingerprint file contents.
- Explain why hashing content is preferable to comparing byte-by-byte or comparing names/sizes alone.
- Handle large files efficiently (read in chunks rather than loading the whole file into memory).
- Optimization: group by file size first, then hash only same-size candidates.
- **Follow-up:** How would you handle hash collisions, and when would a full byte comparison still be needed?

### A2. Multiprocessing vs. Multithreading
Explain what multiprocessing and multithreading are in Python, and compare them across:
- Memory model (shared memory vs. separate process memory).
- Impact of the Global Interpreter Lock (GIL).
- Suitability for CPU-bound vs. I/O-bound workloads.
- Overhead, communication cost, and complexity.
- **Follow-up:** Give a concrete example of when you'd choose one over the other.

### A3. Implement a Rate Limiter
Design and implement a rate limiter: max **100 requests per minute, per user**; reject/throttle excess; window auto-resets after one minute.
- Data structure to track requests per user (sliding window, fixed window, token bucket).
- Thread-safety / concurrency considerations.
- Trade-offs between fixed-window, sliding-window, and token-bucket algorithms.

---

## SECTION B — Transformers, Attention & LLM Internals

### B1. How an LLM Works Internally (End-to-End Text Generation)
Walk through everything that happens from input text to generated output:
- **Tokenization** — how raw text is broken into tokens (and what a token actually is; BPE/subword).
- **Embeddings** — how tokens become numerical vectors; role of positional information.
- **Self-attention** — how it works, why it matters, how the model decides which tokens to focus on.
- **Transformer internals** — what happens across layers (multi-head attention, feed-forward networks, etc.).
- **Next-token prediction** — how probabilities are produced over the vocabulary and how the next token is selected.
- Any other key internal components involved.

### B2. Masked (Causal) Self-Attention
- What masked self-attention is and how it differs from standard (unmasked) self-attention.
- Why masking is needed and where it's applied (e.g., the decoder).
- How the mask prevents a position from "looking ahead" at future tokens during training.
- The math: how the mask is applied within the attention score computation (before/after softmax, what value is used).
- Why this is critical for autoregressive/causal generation.

### B3. Multi-Head Self-Attention & the Transformer Architecture
- Self-attention at a basic level (query, key, value).
- What "multi-head" adds over single-head, and why multiple heads.
- How the outputs of individual heads are combined.
- The overall architecture: encoder vs. decoder blocks and their roles; positional encoding and why it's needed; feed-forward layers, residual connections, layer normalization.
- What problem the Transformer solved compared to RNNs/LSTMs.

### B4. Attention / Transformer Mechanisms Compared
- Self-attention vs. cross-attention (encoder-decoder attention).
- Masked (causal) vs. bidirectional self-attention.
- Single-head vs. multi-head attention.
- When and where each mechanism is used within the architecture.
- Efficiency-oriented variants (sparse/windowed attention) and why they exist.

---

## SECTION C — Vector Stores, Embeddings & Similarity

### C1. How Vector Stores Work Internally
Explain how a vector store performs similarity search internally. Describe and compare:
- **IVF (Inverted File Index)** — partitioning/clustering vectors and searching a subset (probing).
- **HNSW (Hierarchical Navigable Small World graphs)** — graph-based approximate nearest neighbor search.
- How each balances search speed, accuracy (recall), and memory usage.
- What "approximate nearest neighbor" (ANN) means and why we accept approximation.

### C2. FAISS / Chroma — Library or Database?
- Is FAISS a vector database or a library? Is Chroma a vector database or a library?
- The difference between a vector **library** (indexing/search engine) and a full vector **database** (persistence, metadata filtering, CRUD, scaling, APIs).

### C3. Embeddings and Vector Search
- What are embeddings?
- How do you generate them (models, APIs) and use them in vector search?
- End-to-end flow: text → embedding vector → storage in index → query embedding → similarity search → results.

### C4. Vector Database Comparison
- Which vector databases have you used in practice?
- Compare **Milvus vs. FAISS vs. Pinecone**: managed vs. self-hosted; scalability & performance; feature set (metadata filtering, hybrid search, persistence); cost & operational overhead.

### C5. Semantic Search & Similarity/Distance Metrics
- What semantic search is and how it differs from keyword/lexical search.
- Metrics to measure relevance/similarity between vectors: **cosine similarity, dot product, Euclidean (L2) distance, Manhattan (L1) distance** (and others).
- When and why you'd choose one metric over another; the role of normalization.

---

## SECTION D — RAG: Chunking

### D1. RAG System Architecture (End-to-End)
Explain the full architecture of a Retrieval-Augmented Generation system, covering each stage: document ingestion, chunking, embedding, indexing/storage, retrieval, prompt augmentation, and generation.

### D2. Chunking Strategies (Comprehensive)
- What chunking is and why it matters.
- Compare: fixed-size (chars/tokens); fixed-size with overlap (sliding window); recursive/structure-based (paragraphs, sentences, headings); document/format-aware; semantic chunking.
- Pros, cons, and typical use cases of each.
- How to decide and optimize chunk size and overlap; too-small vs. too-large trade-offs.

### D3. Semantic Chunking in Depth
- What it is and how it differs from fixed-size/rule-based chunking.
- End-to-end: split into sentences → embed them → group by semantic similarity.
- Parameters/metrics: similarity metric (cosine) + breakpoint threshold; embedding model choice; min/max chunk size; sentence granularity; overlap handling.
- How these affect retrieval quality, chunk coherence, and cost; tuning challenges.

### D4. Implement Custom RAG Chunking (From Scratch, No Libraries) — CODE
Write a chunking function following these rules:
- Overlap between consecutive chunks = the **last sentence** of the previous chunk carried into the next.
- Each chunk contains at least one complete sentence.
- Sentences are never broken across chunks (boundaries only at sentence ends).
- Max chunk size ≤ 1000 characters.
- **No** built-in RAG library helpers (no `RecursiveCharacterTextSplitter` etc.) — implement manually.
- Input: a single raw string; Output: a list of chunk strings. Sentence splitting done manually (on `.`, `!`, `?`).
- **Edge case to clarify:** behavior when a single sentence exceeds 1000 characters.

### D5. Multiple Chunking Strategies in One RAG Application
- Why a single RAG app might use multiple chunking strategies.
- How the strategy is selected dynamically at runtime (by document type, structure, or content).
- The trade-offs involved.

---

## SECTION E — RAG: Retrieval, Reranking & Multimodality

### E1. Best Retrieval Techniques for Better Answer Generation
- The most popular, widely-used retrieval techniques.
- Which tend to produce the best output and why.
- Advanced techniques: hybrid search (BM25 + vector), re-ranking, query expansion/rewriting, MMR, multi-query, parent-document/small-to-big, metadata filtering.

### E2. Reranking & Cross-Encoders — CODE
- The role of reranking: why re-rank retrieved results after the initial vector search.
- Cross-encoder vs. bi-encoder (the model used for initial retrieval).
- What the cross-encoder does: inputs (query paired with each candidate) → output (relevance score) → reorder.
- Implementation: retrieve top-K → pass (query, candidate) pairs to cross-encoder → sort by score → select top-N.
- Trade-offs (latency, accuracy, cost) of adding a reranking stage.

### E3. Multimodality in RAG / AI
- What multimodal systems are (text, images, audio, video).
- How multimodality applies specifically to RAG pipelines (multimodal embeddings/CLIP, VLMs).
- How you've handled multimodality in your own projects (practical, project-based).

### E4. Detecting & Reducing Hallucinations
- How to detect and reduce hallucinations in LLM responses.
- Techniques: grounding in retrieved context, citations, confidence checks, guardrails, prompt design.

---

## SECTION F — Agentic Systems

### F1. Chain (LangChain) vs. Agent — In Depth
- Definition and purpose of each.
- Control flow: chain follows a fixed/predetermined sequence vs. agent dynamically decides its next action.
- The role of the LLM as a reasoning/decision engine in agents vs. chains.
- When to choose one over the other, with trade-offs (predictability, cost, reliability, flexibility).

### F2. ReAct vs. Plan-and-Execute — In Depth
- How each works internally (reasoning + acting interleaved vs. up-front planning then execution).
- Key differences in control flow and decision-making.
- Strengths, weaknesses, and failure modes of each.
- When you'd choose one pattern over the other.

### F3. How Do You Evaluate an Agent?
- Dimensions: task success/completion, correctness, tool-use accuracy, reasoning quality, latency, cost.
- Metrics/methods: automated eval, LLM-as-judge, human eval, trajectory/step-level evaluation.
- Evaluating multi-step behavior: right tools, right order, error recovery.
- Setup in practice: datasets, benchmarks, offline vs. online/production evaluation.

### F4. Handling Memory in LangGraph + Checkpointers/Savers Deep Dive
- The concept of checkpointers and how LangGraph persists conversational/graph state.
- **InMemorySaver** — how it works, when to use, limitations.
- **SQLiteSaver** — how it works and use cases.
- **PostgresSaver** — how it works and use cases.
- Comparison across durability, scalability, concurrency, production-readiness; which to choose for chat persistence in different scenarios.

### F5. Debugging an Agent Failure in Production
- The very first thing you observe/check when a failure is reported.
- Step-by-step process: reproduce, inspect traces/logs, examine the agent's step-by-step trajectory, isolate the failing component (LLM, tool call, retrieval, prompt).
- The role of observability/tracing tooling in diagnosing failures.
- How you identify root cause and validate a fix.

### F6. MCP Server–Client Tool Architecture
- Overall architecture: how an MCP server exposes tools and how a client connects.
- How tools are registered/defined on the server side.
- How a client discovers what tools a server provides (discovery/tool listing).
- How a tool call flows from client to server and back; transport (stdio/HTTP).

### F7. MCP vs. A2A — What's the Difference?
- What problem each protocol solves.
- MCP's role in connecting agents/models to tools, data, and context.
- A2A's role in enabling communication and collaboration between agents.
- How the two are complementary rather than competing, with example scenarios for each.

---

## SECTION G — Evaluation & Production Operations

### G1. Evaluating a RAG System in Depth (+ Metrics in Code)
- At which stages evaluation is applied: retrieval stage vs. generation stage vs. end-to-end.
- Retrieval metrics: context precision/recall, hit-rate, MRR, NDCG.
- Generation metrics: faithfulness, answer relevancy, context relevancy — what each measures.
- How it's implemented in code (RAGAS, and/or custom LLM-as-judge logic).

### G2. Evaluating AI Responses — Precision & Recall in Code
- Define precision and recall in this context and what each measures.
- Determine true positives / false positives / false negatives for a retrieval or classification component.
- Calculate precision and recall in code: formulas + concrete implementation (manual and/or library).
- How these tie into higher-level RAG evaluation (retrieval quality vs. answer quality).
- Handling evaluation when there is no single "correct" answer.

### G3. Securing & Monitoring LLM Applications in Production
- Prompt injection defense, input/output validation, PII handling, access control.
- Logging, observability, latency/cost monitoring, quality/drift detection.

### G4. Preventing Prompt Injection (Input & Output Guardrails) — In Depth
- What prompt injection is (direct and indirect injection).
- **Input guardrails** — techniques applied to user/external input before it reaches the model.
- **Output guardrails** — techniques applied to the model's output before it's used/returned.
- Additional defenses: input/output validation, sandboxing tool use, least-privilege, content filtering, system-prompt hardening, human-in-the-loop.
- How to layer these for defense-in-depth.

---

## SECTION H — LLM Application Engineering

### H1. Prompt Design — Support Ticket Extraction (JSON)
Create a prompt that extracts structured details from a customer support ticket and returns valid JSON.
- Must include at least: Ticket ID, Issue Category (plus priority, customer sentiment, summary, requested action for completeness).
- The prompt must reliably enforce valid JSON output.

### H2. Design a Simple Knowledge-Base AI Agent
Design an agent that answers questions using a knowledge base. Show (explanation/pseudocode) how it:
- Accepts the user's query.
- Retrieves relevant context from the knowledge base.
- Constructs a prompt with that context and generates a grounded answer.

### H3. Python Function to Call an LLM API
Write a function that accepts a user's query and sends it to an LLM API (OpenAI/Azure OpenAI/equivalent).
- Accept the prompt/query as input; send the request; handle the response; return generated text.
- Consider error handling, API key management, and configurable parameters.

### H4. Fine-Tuning LLMs (In Depth)
- What fine-tuning is and when it's the right choice (vs. prompting or RAG).
- Full fine-tuning vs. parameter-efficient fine-tuning (PEFT).
- Key techniques: **LoRA** (how it works, why efficient), **QLoRA** (quantized LoRA), adapters, prefix/prompt tuning.
- Data preparation, compute/memory requirements, evaluation, deployment trade-offs.
- Applied to models such as LLaMA or Falcon.

### H5. Temperature and top_p
- What temperature and top_p are in OpenAI (and similar) models.
- How each influences output, how they differ.
- When you'd adjust them (deterministic vs. creative tasks).

---

## ✅ Self-Assessment Checklist

For each question, rate yourself:
- 🟢 **Confident** — I can explain it clearly AND write the code/draw the diagram unprompted.
- 🟡 **Partial** — I know the concept but fumble the details or the code.
- 🔴 **Gap** — I need to study this from scratch.

Aim for all 🟢 before the interview. The 🔴 and 🟡 topics are where to focus first.

---

*Question bank compiled from real Senior DS / GenAI / RAG interviews. Detailed answer lessons (with runnable code) are maintained separately, section by section.*
