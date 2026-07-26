# Rapid-Fire Answers — Senior Data Scientist / GenAI Interview

> Concise, interview-ready answers to all 18 questions. Say these out loud in 60–90 seconds each.
> Deep-dive lessons for every question live in the numbered files in this folder.

---

## A. Python & Systems Fundamentals

### Q1. Detecting duplicate file content (hashlib)

Hash each file's **content** (not its name) with SHA-256 and group files by hash. Same hash = same bytes.
Optimize by grouping on file **size** first, then only hashing same-size files, and read in **chunks** so you never load a huge file fully into memory.

```python
import hashlib
from collections import defaultdict

def file_hash(path, chunk=8192):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()

def find_duplicates(paths):
    seen = defaultdict(list)
    for p in paths:
        seen[file_hash(p)].append(p)
    return {h: ps for h, ps in seen.items() if len(ps) > 1}
```

**One-liner:** "Content hashing gives O(n) dedup; I group by size first to avoid hashing unique files, and do a byte compare on hash matches only if I need collision-proof certainty."

---

### Q2. Multiprocessing vs multithreading

- **Threads** share memory, live in one process, but are throttled by the **GIL** for CPU work → great for **I/O-bound** tasks (network, disk).
- **Processes** each have their own memory and interpreter → true parallelism → great for **CPU-bound** tasks (number crunching). Cost: higher memory + inter-process communication (IPC) overhead.

**One-liner:** "I/O-bound → threads (or async). CPU-bound → processes, because the GIL prevents threads from running Python bytecode in parallel."

---

### Q3. Rate limiter (100 req/min per user)

Use a **token bucket** or **sliding-window log** keyed per user. For distributed systems, store counters in **Redis** with a TTL so it resets automatically.

```python
import time
from collections import deque, defaultdict

class SlidingWindowLimiter:
    def __init__(self, limit=100, window=60):
        self.limit, self.window = limit, window
        self.hits = defaultdict(deque)

    def allow(self, user):
        now = time.time()
        q = self.hits[user]
        while q and q[0] <= now - self.window:  # drop expired
            q.popleft()
        if len(q) < self.limit:
            q.append(now)
            return True
        return False
```

**One-liner:** "Fixed window is simplest but has burst-at-boundary problems; sliding window fixes that; token bucket allows controlled bursts. In production I put the counter in Redis with expiry for multi-instance correctness."

---

## B. Vector Stores & Embeddings

### Q4. How vector stores work internally (IVF, HNSW)

Exact nearest-neighbor search is O(n) per query, too slow at scale, so we use **Approximate Nearest Neighbor (ANN)** indexes:

- **IVF (Inverted File):** cluster vectors with k-means into `nlist` buckets; at query time only search the `nprobe` closest buckets. Trades recall for speed via `nprobe`.
- **HNSW:** a multi-layer navigable graph; search starts at a sparse top layer and descends, greedily hopping to closer neighbors. Fast, high recall, higher memory.

**One-liner:** "IVF partitions the space and probes a few cells; HNSW walks a layered proximity graph. Both are ANN: you tune a knob (nprobe / efSearch) to trade recall against latency."

---

### Q5. FAISS / Chroma — library or database?

- **FAISS** = a **library** (Facebook AI Similarity Search). Indexing + search only; no persistence, metadata, or server out of the box.
- **Chroma** = a **vector database**: it wraps an index but adds persistence, metadata filtering, collections, and an API.

**One-liner:** "FAISS is an ANN library you embed in your app; Chroma/Milvus/Pinecone are databases that add storage, metadata filtering, CRUD, and serving on top."

---

### Q6. Embeddings & vector search

An **embedding** is a dense vector that captures semantic meaning, so similar meaning → nearby vectors. Generate them with a model (OpenAI `text-embedding-3`, `sentence-transformers`, etc.). Store in a vector index; embed the query the same way; retrieve by **cosine / dot-product** similarity.

**One-liner:** "Embeddings turn text into geometry so semantic similarity becomes distance. Same model for docs and queries, then nearest-neighbor search."

---

### Q7. Milvus vs FAISS vs Pinecone

- **FAISS:** library, self-hosted, fastest raw ANN, you manage everything (no persistence/metadata layer).
- **Milvus:** open-source vector **DB**, self/managed, scalable, metadata filtering, distributed.
- **Pinecone:** fully **managed** SaaS, zero-ops, great for fast production, cost + vendor lock-in.

**One-liner:** "FAISS when I want an embeddable engine and control; Milvus when I want an open-source scalable DB; Pinecone when I want managed, zero-ops production."

---

## C. Retrieval-Augmented Generation (RAG)

### Q8. RAG architecture

Ingest → **chunk** → **embed** → **index** (offline). At query time: embed query → **retrieve** top-k → **augment** the prompt with retrieved context → **generate** grounded answer (optionally re-rank + cite).

**One-liner:** "RAG grounds the LLM in retrieved context so it answers from your data instead of parametric memory, reducing hallucination and enabling fresh/private knowledge."

---

### Q9. Chunking & chunk-size optimization

Split docs into chunks (e.g., 200–500 tokens) with **overlap** (10–20%) to preserve context across boundaries. Prefer **semantic / structural** splitting (by paragraph, heading) over naive fixed splits. Too small → lost context; too big → diluted relevance + wasted tokens.

**One-liner:** "I tune chunk size and overlap empirically against retrieval metrics; semantic-aware splitting beats fixed-size, and overlap prevents cutting ideas in half."

---

### Q10. RAG evaluation — faithfulness, relevancy

- **Faithfulness:** is the answer supported by the retrieved context? (claims grounded, not invented)
- **Answer relevancy:** does the answer address the question?
- **Context relevancy/precision:** were the retrieved chunks actually relevant?

Computed with **LLM-as-judge** or frameworks like **RAGAS**: decompose answer into claims, check each against context, score the ratio.

**One-liner:** "Faithfulness = grounded-claims / total-claims; relevancy = semantic match to the question; I automate this with RAGAS or an LLM judge and track it in CI."

---

### Q11. Handling hallucinations

Ground with RAG, force **citations**, restrict with "answer only from context; else say you don't know," lower **temperature**, add a **verification/self-check** pass, and use guardrails / output validation.

**One-liner:** "Best defense is grounding + 'refuse if not in context' + citations, then a faithfulness check to catch anything ungrounded before it reaches the user."

---

## D. LLM Application Engineering

### Q12. Prompt: support-ticket extraction → JSON

Give a role, explicit fields, a strict "return valid JSON only" instruction, and a schema/example. Use structured-output / JSON mode where available.

```
Extract the following from the support ticket and return ONLY valid JSON:
{ "ticket_id": string, "issue_category": string, "priority": "low|medium|high",
  "sentiment": "positive|neutral|negative", "summary": string }
Ticket: """{ticket_text}"""
```

**One-liner:** "Constrain the schema, demand JSON-only output, provide an example, and validate the parse — ideally with the model's native structured-output mode."

---

### Q13. Simple knowledge-base agent (pseudocode)

```
accept query
q_vec = embed(query)
context = vector_store.search(q_vec, k=5)
prompt = build(system + context + query)
answer = llm(prompt)
return answer with sources
```

**One-liner:** "It's a retrieve-then-generate loop: embed the query, pull top-k context, stuff it into a grounded prompt, generate, and return with citations."

---

### Q14. Python function calling an LLM API

```python
from openai import OpenAI
client = OpenAI()

def ask_llm(prompt: str, model="gpt-4o-mini", temperature=0.2) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return resp.choices[0].message.content
```

**One-liner:** "Accept the prompt, call chat.completions, handle errors/retries, keep the key in env vars, and expose temperature/model as params."

---

### Q15. Fine-tuning LLaMA / Falcon

Prepare an instruction dataset → choose **PEFT (LoRA/QLoRA)** instead of full fine-tune to save memory → train with HuggingFace + `peft` on quantized weights → evaluate → merge/serve adapters. Full fine-tune only when you have big data + compute.

**One-liner:** "For open models I default to QLoRA: quantize to 4-bit, train small low-rank adapters — 90% of the benefit for a fraction of the GPU cost."

---

### Q16. Temperature and top_p

- **Temperature:** scales randomness of the probability distribution. Low = deterministic/focused; high = creative/diverse.
- **top_p (nucleus):** sample only from the smallest set of tokens whose cumulative probability ≥ p.

Adjust temperature down for extraction/code/factual; up for brainstorming. Usually tune one, not both.

**One-liner:** "Temperature reshapes the whole distribution; top_p truncates its tail. Low temp for deterministic tasks, higher for creative ones."

---

## E. Evaluation & Production Ops

### Q17. Evaluating AI responses (precision/recall in code)

Define TP/FP/FN for your task, then:

```python
def precision_recall(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall    = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
    return precision, recall, f1
```

**One-liner:** "Precision = of what I flagged, how much was right; recall = of what mattered, how much I caught; F1 balances them. For generation I use them on retrieval hits or classification labels via sklearn."

---

### Q18. Securing & monitoring LLM apps in production

- **Security:** prompt-injection defenses, input/output validation, PII redaction, least-privilege on tools, secrets management, rate limiting.
- **Monitoring:** log prompts/responses, track latency, token cost, error rates, quality/faithfulness drift, and user feedback; add alerting + tracing.

**One-liner:** "Treat model I/O as untrusted: validate, sanitize, redact PII, guard tool access; then observe cost, latency, and quality drift with tracing and alerts."

---

*Deep dives with theory, real-time scenarios, and runnable code are in files 01–18.*
