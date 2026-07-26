"""
===================================================================================
LESSON 3 — VECTOR STORES, EMBEDDINGS & SIMILARITY  (Senior DS Real-Time Deep-Dive)
===================================================================================

COVERS (Section C of the question bank):
  C1. How vector stores work internally  (ANN, IVF vs HNSW)
  C2. Library vs Database  (FAISS vs Chroma — what a full DB adds)
  C3. Embeddings & the vector-search flow  (text -> vector -> index -> query -> results)
  C4. Vector DB comparison  (Milvus vs FAISS vs Pinecone)
  C5. Semantic search & similarity/distance metrics  (cosine, dot, L2, L1)

HOW TO READ:  THEORY -> RUNNABLE NUMPY -> INTERVIEW ANSWER -> RELATED CONCEPTS.

Run:  & "C:\\Projects\\Gen AI Udemy\\LangCGS\\.venv\\Scripts\\python.exe" "03_vector_stores_embeddings_similarity.py"

NOTE: everything here is pure NumPy — we re-implement IVF and a greedy graph search
FROM SCRATCH so you understand what FAISS/Chroma/Pinecone do under the hood. No
external vector DB needed to run this.
===================================================================================
"""

from __future__ import annotations

import heapq
import sys
import time

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

np.random.seed(42)
np.set_printoptions(precision=3, suppress=True)


def banner(title: str) -> None:
    line = "=" * 83
    print("\n" + line + "\n" + title + "\n" + line)


def sub(title: str) -> None:
    print("\n" + "━" * 79 + "\n" + title + "\n" + "━" * 79)


# ===================================================================================
# CORE — similarity / distance metrics (reused everywhere)
# ===================================================================================

def l2_normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-12)


def cosine_similarity(q: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Angle-based: 1=same direction, 0=orthogonal, -1=opposite. Magnitude-invariant."""
    return l2_normalize(M) @ l2_normalize(q)


def dot_product(q: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Cosine * magnitudes. Rewards BOTH alignment AND large norms."""
    return M @ q


def euclidean_distance(q: np.ndarray, M: np.ndarray) -> np.ndarray:
    """L2 straight-line distance. Lower = closer."""
    return np.linalg.norm(M - q, axis=1)


def manhattan_distance(q: np.ndarray, M: np.ndarray) -> np.ndarray:
    """L1 city-block distance (sum of abs diffs). Lower = closer. Robust to outliers."""
    return np.sum(np.abs(M - q), axis=1)


def exact_topk(q: np.ndarray, M: np.ndarray, k: int) -> np.ndarray:
    """Brute-force exact nearest neighbors by cosine. The ground truth we compare ANN to."""
    sims = cosine_similarity(q, M)
    return np.argsort(sims)[::-1][:k]


# ===================================================================================
# C1. HOW VECTOR STORES WORK INTERNALLY  (ANN, IVF, HNSW)
# ===================================================================================
#
# THE PROBLEM
#   Exact nearest-neighbor = compare the query to EVERY vector: O(N·d). Fine for
#   thousands, fatal for billions. So we accept APPROXIMATE nearest neighbor (ANN):
#   trade a tiny bit of recall for orders-of-magnitude speedup.
#
# WHAT "APPROXIMATE" MEANS
#   ANN may occasionally miss a true top-k neighbor, but returns the right answers
#   ~95-99% of the time far faster. Recall@k measures how many true neighbors it found.
#   The whole game is balancing SPEED vs RECALL vs MEMORY.
#
# IVF (Inverted File Index) — partition then probe
#   1) Cluster all vectors into `nlist` cells (k-means) with a centroid each.
#   2) Each vector lives in its nearest cell's inverted list.
#   3) At query time, find the `nprobe` nearest CENTROIDS and search ONLY those cells.
#   -> You scan a small fraction of the data. nprobe↑ = higher recall, slower.
#   Memory: cheap (can add Product Quantization to compress vectors).
#
# HNSW (Hierarchical Navigable Small World) — graph navigation
#   A multi-layer proximity graph: sparse long-range links on top layers, dense local
#   links at the bottom. Search = greedily hop to the neighbor closest to the query,
#   descending layers ("zoom in"). Log-ish complexity, excellent recall/speed.
#   -> Fastest queries, highest recall in practice, but HIGH memory (stores the graph)
#      and slower to build. Default in Chroma, Qdrant, Weaviate, Milvus, pgvector.
#
# THE TRADE-OFF TABLE (memorize):
#   Flat (exact):  100% recall, slowest, O(N). Baseline for small data.
#   IVF:           tunable recall (nprobe), low memory, fast build. Great for huge sets.
#   HNSW:          top recall+speed, high memory, slow build. Default for most apps.


def kmeans(X: np.ndarray, k: int, iters: int = 15) -> tuple[np.ndarray, np.ndarray]:
    """Tiny k-means (used to build IVF cells). Returns (centroids, assignments)."""
    idx = np.random.choice(len(X), k, replace=False)
    centroids = X[idx].copy()
    assign = np.zeros(len(X), dtype=int)
    for _ in range(iters):
        dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        assign = np.argmin(dists, axis=1)
        for c in range(k):
            members = X[assign == c]
            if len(members):
                centroids[c] = members.mean(axis=0)
    return centroids, assign


def ivf_search(q, X, centroids, assign, k, nprobe):
    """IVF query: probe the nprobe nearest cells, exact-search only their members."""
    cell_dist = np.linalg.norm(centroids - q, axis=1)
    probe_cells = np.argsort(cell_dist)[:nprobe]
    candidate_ids = np.where(np.isin(assign, probe_cells))[0]
    if len(candidate_ids) == 0:
        return np.array([], dtype=int), 0
    sims = cosine_similarity(q, X[candidate_ids])
    top = candidate_ids[np.argsort(sims)[::-1][:k]]
    return top, len(candidate_ids)


def _demo_ivf() -> None:
    sub("C1a. IVF FROM SCRATCH — partition & probe (runnable)")
    N, d, k = 5000, 32, 10
    # Normalize (as real embedding pipelines do) so L2-cells align with cosine ranking.
    X = l2_normalize(np.random.randn(N, d).astype(np.float32))
    q = l2_normalize(np.random.randn(d).astype(np.float32))

    t0 = time.perf_counter()
    truth = exact_topk(q, X, k)
    exact_ms = (time.perf_counter() - t0) * 1000

    centroids, assign = kmeans(X, k=50)  # 50 cells
    for nprobe in (1, 5, 20):
        t0 = time.perf_counter()
        approx, scanned = ivf_search(q, X, centroids, assign, k, nprobe)
        ms = (time.perf_counter() - t0) * 1000
        recall = len(set(approx.tolist()) & set(truth.tolist())) / k
        print(f"  nprobe={nprobe:>2}: scanned {scanned:>4}/{N} vectors "
              f"({100*scanned/N:4.1f}%), recall@{k}={recall:.0%}, {ms:.2f}ms")
    print(f"  exact (flat) scanned {N}/{N} vectors, recall=100%, {exact_ms:.2f}ms")
    print("  -> more probes = more recall but more work. That's the ANN dial.")


def greedy_graph_search(q, X, graph, entry, k, ef=25):
    """HNSW-style best-first ('ef search') walk on a proximity graph.

    Expand the closest unvisited candidate, keep the ef best found; stop when the
    frontier can't beat the current worst result. This is how HNSW navigates.
    """
    d0 = float(np.linalg.norm(X[entry] - q))
    visited = {entry}
    frontier = [(d0, entry)]          # min-heap: closest candidate to expand next
    best = [(-d0, entry)]             # max-heap (negated): the ef best found so far
    hops = 0
    while frontier:
        dist, node = heapq.heappop(frontier)
        if -best[0][0] < dist and len(best) >= ef:
            break                     # closest candidate is worse than our worst keeper
        for nb in graph[node]:
            if nb in visited:
                continue
            visited.add(nb)
            hops += 1
            dnb = float(np.linalg.norm(X[nb] - q))
            heapq.heappush(frontier, (dnb, nb))
            heapq.heappush(best, (-dnb, nb))
            if len(best) > ef:
                heapq.heappop(best)
    ranked = sorted((n for _, n in best), key=lambda n: np.linalg.norm(X[n] - q))
    return ranked[:k], hops


def _demo_hnsw_walk() -> None:
    sub("C1b. HNSW-STYLE GREEDY GRAPH WALK (runnable illustration)")
    N, d, M, k = 800, 16, 12, 5
    X = l2_normalize(np.random.randn(N, d).astype(np.float32))  # normalized => L2≈cosine
    q = l2_normalize(np.random.randn(d).astype(np.float32))
    # Build a kNN proximity graph (each node -> M nearest neighbors).
    graph = {}
    for i in range(N):
        dists = np.linalg.norm(X - X[i], axis=1)
        graph[i] = np.argsort(dists)[1:M + 1].tolist()
    truth = set(exact_topk(q, X, k).tolist())
    found, hops = greedy_graph_search(q, X, graph, entry=0, k=k)
    recall = len(set(found) & truth) / k
    print(f"  best-first walk from entry node 0: explored {hops}/{N} nodes, "
          f"recall@{k}={recall:.0%}")
    print("  -> navigate the graph toward the query instead of scanning all N.")
    print("     (Real HNSW adds sparse upper layers, making this closer to log(N) hops.)")


# INTERVIEW ANSWER (C1):
#   "Exact search is O(N·d), so at scale we use approximate nearest neighbor. IVF
#    clusters vectors into cells and only searches the nprobe cells nearest the query
#    — nprobe trades recall for speed. HNSW builds a layered proximity graph and
#    greedily navigates toward the query in log-ish hops — best recall and speed but
#    high memory and slower builds. Flat/exact is the 100%-recall baseline for small
#    data. It's always a speed/recall/memory trade-off."
#
# RELATED CONCEPTS: Product Quantization (PQ) & IVF-PQ compression; recall@k; ef_search
#   / ef_construction / M knobs in HNSW; DiskANN; when exact (flat) is actually fine.


# ===================================================================================
# C2. LIBRARY vs DATABASE  (FAISS vs Chroma)
# ===================================================================================
#
# THE DISTINCTION (a very common "gotcha" question)
#   FAISS = a LIBRARY. It's an in-process ANN index (build, add, search vectors).
#           No persistence layer, no metadata store, no CRUD/update semantics, no
#           server/API, no auth — YOU wrap all that yourself. Blazing fast, low-level.
#   Chroma = a vector DATABASE. It's built AROUND an index (often HNSW) and adds the
#           database concerns: persistence to disk, metadata + filtering, collections,
#           CRUD (add/update/delete by id), a client/server API, and easy ops.
#
# LIBRARY vs FULL DATABASE — what the DB adds on top of the raw index:
#   - Persistence & durability (survive restarts; FAISS you must save/load manually).
#   - Metadata storage + FILTERED search ("vectors WHERE source='policy.pdf'").
#   - CRUD by id (upsert/delete) — libraries are largely append/rebuild oriented.
#   - Scaling, sharding, replication; a network API; auth; monitoring.
#   Rule of thumb: FAISS when you want a fast embedded index and control everything;
#   a vector DB (Chroma/Qdrant/Milvus/Pinecone) when you need persistence, metadata
#   filtering, updates, and multi-user access without building infra yourself.


def _demo_metadata_filtering() -> None:
    sub("C2. METADATA-FILTERED SEARCH (what a DB adds over a raw index)")
    # A raw library returns nearest vectors; a DB can pre-filter by metadata first.
    N, d, k = 400, 16, 3
    X = np.random.randn(N, d).astype(np.float32)
    q = np.random.randn(d).astype(np.float32)
    sources = np.array(["policy.pdf", "faq.md", "manual.txt"])[np.random.randint(0, 3, N)]

    # Unfiltered (library-style) top-k
    top_all = exact_topk(q, X, k)
    print(f"  Unfiltered top-{k}: ids {top_all.tolist()} "
          f"sources {sources[top_all].tolist()}")

    # DB-style: filter to source='policy.pdf' FIRST, then search within
    mask = np.where(sources == "policy.pdf")[0]
    sims = cosine_similarity(q, X[mask])
    top_filtered = mask[np.argsort(sims)[::-1][:k]]
    print(f"  Filtered (source=policy.pdf) top-{k}: ids {top_filtered.tolist()} "
          f"sources {sources[top_filtered].tolist()}")
    print("  -> metadata filtering is a DATABASE feature; a raw FAISS index has no idea"
          " about 'source'.")


# INTERVIEW ANSWER (C2):
#   "FAISS is a library — an in-process ANN index with no persistence, metadata, CRUD,
#    or API; you wrap those yourself. Chroma is a vector database built around an index
#    that adds persistence, metadata filtering, CRUD by id, and a client/server API.
#    Use FAISS when you want a fast embedded index and full control; use a vector DB
#    when you need durable storage, filtered search, updates, and multi-user access."
#
# RELATED CONCEPTS: pgvector (vectors inside Postgres); hybrid search (BM25 + vector);
#   'FAISS + SQLite/metadata sidecar' as a poor-man's vector DB (this is basically
#    what many DocSage-style apps do).


# ===================================================================================
# C3. EMBEDDINGS & THE VECTOR-SEARCH FLOW
# ===================================================================================
#
# WHAT EMBEDDINGS ARE
#   Dense numeric vectors where SEMANTIC similarity ≈ geometric closeness. "king" and
#   "queen" land near each other; "king" and "banana" don't. A model learned to map
#   text -> a point in R^d such that meaning is captured by direction/position.
#
# HOW YOU GENERATE THEM
#   - API models: OpenAI text-embedding-3, Cohere, Voyage.
#   - Local/open: sentence-transformers (all-MiniLM, BGE, E5), HuggingFace.
#   Pick by: quality (MTEB benchmark), dimension, speed, cost, and whether it matches
#   your domain. IMPORTANT: use the SAME model for indexing and querying.
#
# END-TO-END FLOW (the answer they want, in order):
#   text  ->  embedding model  ->  vector  ->  store in ANN index (with metadata)
#   query text  ->  SAME model  ->  query vector  ->  similarity search  ->  top-k
#   ->  (feed retrieved chunks to the LLM as context = RAG)


def _stable_hash(s: str) -> int:
    """Deterministic hash — Python's built-in hash() is SALTED per process, so it
    would make this embedder non-reproducible across runs. This fixes that."""
    import hashlib
    return int.from_bytes(hashlib.md5(s.encode()).digest()[:4], "little")


def toy_embed(text: str, d: int = 256) -> np.ndarray:
    """Deterministic TOY embedding (char-trigram hashing) — stands in for a real model.

    Real embeddings come from trained models; this just shows the FLOW with real text
    (similar strings -> somewhat similar vectors)."""
    v = np.zeros(d)
    t = text.lower()
    for i in range(len(t) - 2):
        v[_stable_hash(t[i:i + 3]) % d] += 1.0
    return l2_normalize(v)


def _demo_embedding_flow() -> None:
    sub("C3. EMBEDDING + VECTOR-SEARCH FLOW (runnable, toy embedder)")
    corpus = [
        "how to reset my password",
        "steps to change account password",
        "the cat sat on the mat",
        "refund policy for returned orders",
        "how do I get a refund",
    ]
    d = 256
    index = np.vstack([toy_embed(t, d) for t in corpus])   # text -> vectors -> index
    query = "i forgot my password, reset it"
    qv = toy_embed(query, d)                                # same embedder for query
    sims = cosine_similarity(qv, index)
    order = np.argsort(sims)[::-1][:3]
    print(f"  query: {query!r}")
    print("  top-3 retrieved:")
    for rank, i in enumerate(order, 1):
        print(f"    {rank}. (sim={sims[i]:.3f}) {corpus[i]!r}")
    print("  -> password-reset docs rank top. Real models capture far deeper meaning.")


# INTERVIEW ANSWER (C3):
#   "Embeddings are dense vectors where semantic similarity maps to geometric
#    closeness, produced by models like OpenAI text-embedding-3 or sentence-
#    transformers. The flow: embed every chunk and store it in an ANN index with
#    metadata; at query time embed the query with the SAME model, run a similarity
#    search, and return the top-k — which in RAG become the LLM's context. The
#    non-negotiable rule is using the same embedding model for indexing and querying."
#
# RELATED CONCEPTS: MTEB leaderboard; dimensionality vs cost; normalize for cosine;
#   domain fine-tuned embeddings; chunk-then-embed granularity (ties to Lesson 4).


# ===================================================================================
# C4. VECTOR DATABASE COMPARISON  (Milvus vs FAISS vs Pinecone)
# ===================================================================================
#
#   FAISS (Meta) — LIBRARY
#     Type:        in-process library (Python/C++), not a server.
#     Hosting:     you embed it in your app; you manage persistence/scaling yourself.
#     Scale:       single-machine (huge on one box with GPU); no built-in distribution.
#     Features:    richest index zoo (Flat, IVF, HNSW, PQ, IVF-PQ); NO metadata/CRUD/API.
#     Cost/ops:    free/OSS; ops burden is on you.
#     Use when:    max control + speed, embedded index, you build the surrounding infra.
#
#   Milvus (Zilliz) — OPEN-SOURCE DISTRIBUTED DB
#     Type:        full vector DB, self-hosted or managed (Zilliz Cloud).
#     Hosting:     self-host (k8s) or managed.
#     Scale:       distributed, billions of vectors, sharding/replication.
#     Features:    multiple index types, metadata filtering, hybrid search, persistence.
#     Cost/ops:    OSS but heavier to self-operate; managed option available.
#     Use when:    large scale + you want an open-source DB you can control.
#
#   Pinecone — FULLY MANAGED (SaaS)
#     Type:        managed vector DB; no infra to run.
#     Hosting:     cloud SaaS only.
#     Scale:       serverless auto-scaling; low ops.
#     Features:    metadata filtering, hybrid search, namespaces, high availability.
#     Cost/ops:    pay-per-use / subscription; minimal ops but vendor lock-in + cost.
#     Use when:    you want production-ready with near-zero ops and will pay for it.
#
#   QUICK CHOOSER:
#     Prototype / embedded / full control -> FAISS (or Chroma for a light local DB).
#     Self-hosted at big scale, open-source -> Milvus (or Qdrant/Weaviate).
#     Zero-ops managed, ship fast -> Pinecone.


def _demo_comparison_table() -> None:
    sub("C4. VECTOR DB COMPARISON (quick reference)")
    rows = [
        ("Dimension", "FAISS", "Milvus", "Pinecone"),
        ("Type", "library", "OSS DB (distributed)", "managed SaaS"),
        ("Hosting", "in-process", "self-host/managed", "cloud only"),
        ("Scale", "single node", "billions/distributed", "serverless auto"),
        ("Metadata filter", "no (DIY)", "yes", "yes"),
        ("Persistence", "manual save", "yes", "yes"),
        ("Ops burden", "high (you)", "medium-high", "very low"),
        ("Cost", "free/OSS", "free/OSS+infra", "paid"),
    ]
    widths = [16, 12, 22, 14]
    for r in rows:
        print("  " + "".join(str(c).ljust(w) for c, w in zip(r, widths)))


# INTERVIEW ANSWER (C4):
#   "FAISS is a library — fastest embedded index, no persistence/metadata/API, you
#    build the infra. Milvus is an open-source distributed vector DB for billions of
#    vectors with filtering and persistence, self-hosted or managed. Pinecone is a
#    fully managed serverless DB — near-zero ops, metadata filtering, HA, but paid and
#    cloud-only. I choose FAISS/Chroma for prototypes and control, Milvus/Qdrant for
#    open-source scale, and Pinecone when I want to ship fast with no ops."


# ===================================================================================
# C5. SEMANTIC SEARCH & SIMILARITY / DISTANCE METRICS
# ===================================================================================
#
# SEMANTIC vs KEYWORD (LEXICAL) SEARCH
#   Keyword/lexical (BM25, TF-IDF): matches exact words/tokens. "car" won't match
#   "automobile". Fast, exact, great for names/codes/rare terms.
#   Semantic (embeddings): matches MEANING via vector closeness. "car" ~ "automobile".
#   Best in practice = HYBRID (BM25 + vector) to get both exact and semantic matches.
#
# THE METRICS (know when to use each):
#   Cosine similarity: angle only, magnitude-invariant. THE default for text
#     embeddings (length shouldn't matter). Higher = more similar.
#   Dot product: cosine × magnitudes. Rewards big-norm vectors too. Used when the
#     model is trained for it (e.g., some retrieval models bake importance into norm)
#     — and on NORMALIZED vectors, dot product == cosine.
#   Euclidean (L2): straight-line distance. Lower = closer. Sensitive to magnitude.
#   Manhattan (L1): sum of absolute differences. More robust to outliers; used in
#     some high-dim / sparse settings.
#   KEY FACT: on L2-NORMALIZED vectors, cosine, dot, and (monotonically) Euclidean give
#   the SAME ranking. That's why people normalize and then use fast dot product.


def _demo_metrics() -> None:
    sub("C5. SIMILARITY / DISTANCE METRICS (runnable — how rankings differ)")
    # Two docs: same DIRECTION as query but very different MAGNITUDE.
    q = np.array([1.0, 1.0])
    docs = np.array([
        [2.0, 2.0],   # same direction as q, larger magnitude
        [1.0, 1.0],   # identical direction & magnitude to q
        [1.0, 0.0],   # 45° off
        [-1.0, -1.0], # opposite direction
    ])
    labels = ["same-dir big", "identical", "45°-off", "opposite"]

    cos = cosine_similarity(q, docs)
    dot = dot_product(q, docs)
    euc = euclidean_distance(q, docs)
    man = manhattan_distance(q, docs)

    print("  doc            cosine↑   dot↑    L2↓    L1↓")
    for i, lab in enumerate(labels):
        print(f"  {lab:<14} {cos[i]:6.3f}  {dot[i]:6.3f}  {euc[i]:6.3f}  {man[i]:6.3f}")
    print("\n  Note: cosine ranks 'same-dir big' and 'identical' EQUALLY (angle only),")
    print("  but dot product prefers 'same-dir big' (larger magnitude), and L2 prefers")
    print("  'identical' (closest point). Same data, different winners -> pick on purpose.")

    # Prove: normalize -> cosine and dot agree on ranking
    nd = l2_normalize(docs)
    print(f"\n  After L2-normalizing docs, dot == cosine: "
          f"{np.allclose(nd @ l2_normalize(q), cosine_similarity(q, docs))}")


# INTERVIEW ANSWER (C5):
#   "Keyword search matches exact tokens (BM25); semantic search matches meaning via
#    embedding closeness; the best systems hybridize both. For metrics: cosine is the
#    default for text because it's magnitude-invariant — only direction/meaning
#    matters; dot product also rewards magnitude and equals cosine on normalized
#    vectors; Euclidean is straight-line distance and Manhattan is sum-of-abs, more
#    outlier-robust. Since most pipelines L2-normalize, cosine and dot give identical
#    rankings, so people normalize and use fast dot product."


# ===================================================================================
# GOLDEN LESSONS
# ===================================================================================
GOLDEN_LESSONS = """
  1. Exact NN is O(N·d); at scale use ANN — trade a little recall for huge speedups.
  2. IVF = cluster into cells, probe only the nprobe nearest cells. nprobe = recall/speed dial.
  3. HNSW = layered proximity graph, greedy navigation, log-ish hops. Best recall+speed, high RAM.
  4. Flat/exact = 100% recall baseline; fine for small corpora.
  5. FAISS = LIBRARY (index only). Chroma/Milvus/Pinecone = DATABASES (persist, metadata, CRUD, API).
  6. Metadata-filtered search is a DB feature — raw indexes don't know your metadata.
  7. Embeddings map meaning -> geometry. Use the SAME model to index and to query. Always.
  8. Flow: text -> embed -> index; query -> embed -> search -> top-k -> LLM context (RAG).
  9. Cosine = angle (default for text); dot = angle×magnitude; L2 = distance; L1 = outlier-robust.
 10. On normalized vectors cosine == dot ranking — normalize, then use fast dot product.
 11. Chooser: FAISS/Chroma (proto/control), Milvus/Qdrant (OSS scale), Pinecone (zero-ops managed).
"""


if __name__ == "__main__":
    banner("LESSON 3 — VECTOR STORES, EMBEDDINGS & SIMILARITY")
    _demo_ivf()
    _demo_hnsw_walk()
    _demo_metadata_filtering()
    _demo_embedding_flow()
    _demo_comparison_table()
    _demo_metrics()
    banner("GOLDEN LESSONS")
    print(GOLDEN_LESSONS)
