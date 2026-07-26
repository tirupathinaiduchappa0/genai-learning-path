# Lesson 05 — FAISS / Chroma: Library or Database?

## 1. The Question

> "Is FAISS a vector database or a library? Is Chroma? Explain the difference between a vector library and a vector database."

This is a "do you actually understand the stack" question. The trap is calling FAISS a database.

---

## 2. Theory — library vs database

A **vector library** gives you the **algorithm**: build an index, add vectors, search for nearest neighbors. That's it. No persistence server, no metadata engine, no network API, no CRUD semantics — you wrap it in your own application.

A **vector database** is a full **data system** built *around* an index (often the same algorithms). It adds:

- **Persistence & durability** — data survives restarts, WAL, snapshots.
- **CRUD** — insert/update/delete with consistency guarantees.
- **Metadata storage & filtering** — "find similar docs *where* tenant = X and date > Y".
- **A serving layer** — network API, auth, multi-tenancy.
- **Scaling** — sharding, replication, horizontal scale.
- **Operational features** — backups, monitoring, collections/namespaces.

Analogy: FAISS is like `SQLite's B-tree code` (an index structure); a vector DB is like `PostgreSQL` (a whole database that uses index structures).

---

## 3. Where each tool sits

| Tool | What it is | Persistence | Metadata filter | Server/API | Scale |
|---|---|---|---|---|---|
| **FAISS** | Library | No (you save/load files manually) | No | No | Single node |
| **hnswlib / ScaNN / Annoy** | Library | Manual | No | No | Single node |
| **Chroma** | Vector **DB** (embedded/lightweight) | Yes | Yes | Yes (client/server mode) | Small–medium |
| **Milvus / Qdrant / Weaviate** | Vector **DB** | Yes | Yes | Yes | Distributed |
| **Pinecone** | Managed vector **DB** (SaaS) | Yes | Yes | Yes | Managed, elastic |
| **pgvector** | Postgres **extension** | Yes (Postgres) | Yes (SQL) | Yes | Postgres-bound |

### FAISS — a library
- Facebook AI Similarity Search. Pure ANN engine (Flat, IVF, HNSW, PQ).
- No persistence layer (you `faiss.write_index` / `read_index` yourself), no metadata, no server.
- Blazing fast, GPU support. You embed it inside your service.

### Chroma — a (lightweight) database
- Wraps an ANN index (HNSW) but adds collections, persistence, metadata filtering, and a Python/REST API.
- Runs embedded (in-process) or client/server. Aimed at quick RAG prototypes → small production.

---

## 4. Hands-on: same task, both tools

### FAISS (you manage everything)

```python
import faiss, numpy as np, pickle

index = faiss.IndexFlatIP(384)          # you pick the metric
vectors = np.random.rand(3, 384).astype("float32")
index.add(vectors)                       # positions are your only "IDs"

faiss.write_index(index, "my.index")     # you handle persistence
# metadata? you keep a separate dict/DB and map row-id -> doc yourself
docs = {0: "doc a", 1: "doc b", 2: "doc c"}
```

### Chroma (DB handles IDs, metadata, persistence)

```python
import chromadb

client = chromadb.PersistentClient(path="./chroma_db")   # persistence built in
col = client.get_or_create_collection("docs")

col.add(
    ids=["1", "2", "3"],
    documents=["doc a", "doc b", "doc c"],
    metadatas=[{"lang": "en"}, {"lang": "fr"}, {"lang": "en"}],
)

# vector search + metadata filter in one call
col.query(query_texts=["something"], n_results=2, where={"lang": "en"})
```

Notice: with FAISS you hand-roll IDs, metadata mapping, and persistence. Chroma gives them to you.

---

## 5. Real-time / when to choose what

- **FAISS** — you need max control/speed, embed the index in your own service, small-to-medium corpus, and you're fine building the metadata + persistence layer yourself. Also great as the *engine inside* another system.
- **Chroma** — fast RAG prototypes, local dev, small production; batteries-included with LangChain/LlamaIndex.
- **Qdrant / Milvus / Weaviate** — production scale, rich filtering, self-host or managed.
- **Pinecone** — you want zero ops and pay for it.
- **pgvector** — you already run Postgres and want vectors next to relational data (transactional consistency, one system to operate).

---

## 6. Interview script

"FAISS is a **library** — it's the ANN algorithm engine, no persistence, metadata, or server; you embed it and build everything else around it. Chroma is a **vector database** — it uses an index like HNSW under the hood but adds collections, persistence, metadata filtering, and an API. The distinction is: a library gives you the search algorithm; a database gives you a whole data system — CRUD, durability, filtering, scaling, serving. Milvus, Qdrant, Weaviate, and Pinecone are databases; Annoy, hnswlib, ScaNN are libraries like FAISS."
