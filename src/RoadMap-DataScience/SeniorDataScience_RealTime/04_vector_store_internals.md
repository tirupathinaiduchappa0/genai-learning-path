# Lesson 04 — How Vector Stores Work Internally (IVF, HNSW)

## 1. The Question

> "Explain how a vector store performs similarity search internally. Compare IVF and HNSW. What does Approximate Nearest Neighbor mean, and how do you trade recall for speed?"

---

## 2. Theory — the problem

You have millions of embedding vectors (say 768–1536 dims). A query is also a vector. You want the **k nearest** vectors by cosine/dot/L2 distance.

**Exact (brute force / flat):** compare the query to every vector → O(n·d) per query. Accurate but too slow at scale (10M vectors × 1536 dims = billions of multiplications per query).

Solution: **ANN — Approximate Nearest Neighbor.** Accept a tiny loss in accuracy (**recall**) for orders-of-magnitude speedup. The core knob in every ANN index trades **recall ↔ latency**.

### Distance metrics
- **Cosine similarity** — angle; standard for normalized text embeddings.
- **Dot product** — magnitude matters (used with some models).
- **L2 (Euclidean)** — straight-line distance.

---

## 3. IVF — Inverted File Index

**Build (offline):**
1. Run **k-means** on all vectors to find `nlist` centroids (buckets/cells).
2. Assign each vector to its nearest centroid → an "inverted list" per cell.

**Search:**
1. Find the `nprobe` centroids closest to the query.
2. Only scan the vectors inside those cells (not the whole dataset).

**The knob:** `nprobe`.
- `nprobe=1` → fast, lower recall (might miss neighbors in adjacent cells).
- `nprobe=nlist` → equals brute force, perfect recall, no speedup.

**Analogy:** a library organized by section. Instead of scanning every book, you go to the 2–3 most relevant sections.

**IVF + PQ (Product Quantization):** compress vectors into codes so millions fit in RAM. Trades a bit more accuracy for huge memory savings. This is `IVF-PQ`, the workhorse for billion-scale.

---

## 4. HNSW — Hierarchical Navigable Small World

A **multi-layer proximity graph**:
- Bottom layer: all nodes, densely connected to near neighbors.
- Higher layers: exponentially fewer nodes with "long-range" links (like express lanes / skip list).

**Search:**
1. Enter at the top (sparse) layer, greedily hop toward the query.
2. Drop a layer, refine, repeat.
3. At the bottom layer, do a fine-grained greedy search for the k best.

**The knobs:**
- `M` — neighbors per node (graph connectivity). Higher = better recall, more memory.
- `efConstruction` — build-time search width (quality of the graph).
- `efSearch` — query-time candidate list size. **Higher = better recall, slower.** This is the recall↔latency dial.

**Analogy:** a highway system — you take the interstate (top layer) to get near, then local roads (bottom layer) to arrive exactly.

**Trade-off vs IVF:** HNSW usually gives the best recall/latency but uses **more memory** (stores the graph) and has slower/heavier index builds. Updates (inserts) are easy; deletes are awkward.

---

## 5. Comparison table

| | Flat (exact) | IVF(-PQ) | HNSW |
|---|---|---|---|
| Recall | 100% | tunable (nprobe) | tunable (efSearch), very high |
| Speed | slow | fast | fastest at high recall |
| Memory | high (raw) | low (with PQ) | high (graph) |
| Build time | none | medium (k-means) | slow |
| Best when | small data / ground truth | billion-scale, RAM-limited | latency-critical, moderate scale |

---

## 6. Hands-on code (FAISS)

```python
import numpy as np
import faiss

d = 128                       # embedding dim
xb = np.random.random((100_000, d)).astype("float32")   # database
xq = np.random.random((5, d)).astype("float32")         # queries

# --- Flat: exact baseline ---
flat = faiss.IndexFlatL2(d)
flat.add(xb)
D, I = flat.search(xq, k=5)   # ground truth

# --- IVF: cluster then probe ---
nlist = 100
quantizer = faiss.IndexFlatL2(d)
ivf = faiss.IndexIVFFlat(quantizer, d, nlist)
ivf.train(xb)                 # k-means to build cells
ivf.add(xb)
ivf.nprobe = 10               # <-- recall/latency knob
D, I = ivf.search(xq, k=5)

# --- HNSW: graph ---
hnsw = faiss.IndexHNSWFlat(d, 32)   # M=32
hnsw.hnsw.efConstruction = 200
hnsw.add(xb)
hnsw.hnsw.efSearch = 64             # <-- recall/latency knob
D, I = hnsw.search(xq, k=5)
```

### Measuring recall (so you can tune the knob)

```python
def recall_at_k(approx_I, truth_I):
    hits = sum(len(set(a) & set(t)) for a, t in zip(approx_I, truth_I))
    return hits / (len(truth_I) * truth_I.shape[1])
```

Sweep `nprobe` / `efSearch`, plot recall vs latency, pick the knee.

---

## 7. Real-time / production notes

- **Tuning is empirical:** build a ground-truth set with a Flat index, then raise `nprobe`/`efSearch` until recall meets your SLA at acceptable latency.
- **HNSW** is the default in Chroma, Qdrant, Weaviate, pgvector; **IVF-PQ** shines at billion scale (Milvus, FAISS on huge corpora).
- **Filtered search** (metadata + vector) is a real production pain point — pre-filter vs post-filter changes recall; mature DBs (Qdrant, Milvus) handle it natively.
- **Updates:** HNSW handles inserts well; heavy deletes may need periodic rebuilds.

---

## 8. Interview script

"Exact search is O(n) per query, so at scale we use approximate nearest neighbor. IVF clusters vectors with k-means and only probes the nearest cells — `nprobe` trades recall for speed. HNSW builds a layered proximity graph and you navigate from a sparse top layer down to a dense bottom layer — `efSearch` is the recall knob. HNSW usually gives the best recall-per-millisecond but costs more memory; IVF-PQ compresses vectors and wins at billion scale. I tune the knob by measuring recall against a flat-index ground truth and picking the point that meets my latency SLA."
