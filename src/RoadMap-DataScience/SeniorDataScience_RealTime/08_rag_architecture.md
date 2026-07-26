# Lesson 08 — RAG System Architecture

## 1. The Question

> "Explain the end-to-end architecture of a Retrieval-Augmented Generation system. Cover ingestion, chunking, embedding, indexing, retrieval, augmentation, and generation."

---

## 2. Theory — why RAG exists

An LLM only knows what was in its training data (frozen, possibly stale) and what fits in its context window. Problems:

- **Stale knowledge** — no events after the training cutoff.
- **No private data** — it never saw your company's docs.
- **Hallucination** — it fills gaps with plausible-sounding fiction.
- **No provenance** — you can't cite where an answer came from.

**RAG (Retrieval-Augmented Generation)** fixes this by retrieving relevant text from an external knowledge base *at query time* and injecting it into the prompt. The model then answers **grounded** in that retrieved context instead of its parametric memory.

One-line mental model: **"Open-book exam instead of closed-book."**

---

## 3. The two phases

### Phase A — Ingestion / Indexing (offline, batch)

```
raw sources (PDF, HTML, DB, Confluence, code)
   → LOAD    (extract clean text)
   → CHUNK   (split into passages, with overlap)
   → EMBED   (each chunk -> vector, via embedding model)
   → STORE   (vectors + metadata + original text in vector DB)
```

### Phase B — Query / Serving (online, per request)

```
user query
   → EMBED query (same model)
   → RETRIEVE top-k chunks (ANN search, optional metadata filter)
   → (RE-RANK)  reorder by a cross-encoder for precision
   → AUGMENT    build prompt = system + context + query
   → GENERATE   LLM produces grounded answer
   → (CITE)     attach sources
```

---

## 4. Component-by-component

| Stage | Job | Common tools / choices |
|---|---|---|
| **Loaders** | Extract clean text from many formats | LangChain/LlamaIndex loaders, `unstructured`, PyMuPDF |
| **Chunking** | Split into retrievable passages | fixed+overlap, recursive, semantic (Lesson 09) |
| **Embedding** | Text → vector | `text-embedding-3`, `bge`, `e5`, MiniLM |
| **Vector store** | Index + ANN search + filter | Chroma, Qdrant, Milvus, Pinecone, pgvector |
| **Retriever** | Fetch top-k relevant chunks | dense, sparse (BM25), **hybrid** |
| **Re-ranker** | Precision reorder of candidates | cross-encoder (`bge-reranker`), Cohere Rerank |
| **Generator** | Produce the final answer | GPT-4o, Claude, Llama, etc. |
| **Orchestration** | Glue it all together | LangChain, LlamaIndex, custom |

---

## 5. Hands-on — a minimal but complete RAG

```python
from sentence_transformers import SentenceTransformer
import numpy as np
from openai import OpenAI

embedder = SentenceTransformer("all-MiniLM-L6-v2")
llm = OpenAI()

# ---------- Phase A: ingest ----------
documents = [
    "Our refund policy allows returns within 30 days of purchase.",
    "Premium support is available 24/7 for enterprise customers.",
    "The API rate limit is 100 requests per minute per key.",
]
doc_vecs = embedder.encode(documents, normalize_embeddings=True)

# ---------- Phase B: query ----------
def retrieve(query, k=2):
    q = embedder.encode([query], normalize_embeddings=True)[0]
    scores = doc_vecs @ q
    top = np.argsort(scores)[::-1][:k]
    return [documents[i] for i in top]

def rag_answer(query):
    context = retrieve(query)
    prompt = f"""Answer the question using ONLY the context below.
If the answer isn't in the context, say "I don't know."

Context:
{chr(10).join(f"- {c}" for c in context)}

Question: {query}
Answer:"""
    resp = llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,   # low temp => grounded, deterministic
    )
    return resp.choices[0].message.content

print(rag_answer("How many API requests can I make?"))
# -> grounded in the rate-limit chunk, not invented
```

The two things that make it "RAG": retrieval before generation, and the **"only use the context / else say I don't know"** instruction.

---

## 6. Advanced patterns (senior-level bonus)

- **Hybrid retrieval** — combine BM25 (keywords) + dense (semantics), fuse with Reciprocal Rank Fusion.
- **Re-ranking** — retrieve 50 cheaply, re-rank to top 5 with a cross-encoder for precision.
- **Query transformation** — rewrite/expand the query, or **HyDE** (generate a hypothetical answer, embed *that*).
- **Multi-query** — issue several rephrasings, union the results.
- **Agentic / iterative RAG** — the model decides when to retrieve, can retrieve multiple times.
- **GraphRAG** — retrieve over a knowledge graph for multi-hop reasoning.
- **Contextual retrieval** — prepend a short doc-level summary to each chunk before embedding (Anthropic's technique) to preserve context.

---

## 7. Real-time / production concerns

- **Freshness:** re-index changed documents; use incremental updates keyed on source + hash (ties to Lesson 01).
- **Latency budget:** embedding + ANN + LLM; cache query embeddings and frequent answers.
- **Evaluation:** track faithfulness / relevancy continuously (Lesson 10).
- **Guardrails:** the "say I don't know" rule + citations reduce hallucination (Lesson 11).
- **Security:** retrieved content is untrusted → risk of **prompt injection** via documents; sanitize / sandbox.
- **Access control:** filter retrieval by the user's permissions (don't leak docs across tenants).

---

## 8. Interview script

"RAG turns a closed-book model into an open-book one. Offline I load, chunk, embed, and index my documents. At query time I embed the question, retrieve the top-k most similar chunks, optionally re-rank them, stuff them into the prompt with a strict 'answer only from this context' instruction, and generate a cited answer. That grounds the model in current, private data and cuts hallucination. For quality I add hybrid retrieval and a cross-encoder re-ranker, and I continuously measure faithfulness and context relevance. Key production concerns are freshness, latency, per-tenant access control, and prompt injection from retrieved content."
