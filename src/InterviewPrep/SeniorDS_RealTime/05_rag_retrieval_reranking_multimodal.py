"""
===================================================================================
LESSON 5 — RAG: RETRIEVAL, RERANKING & MULTIMODALITY  (Senior DS Real-Time Deep-Dive)
===================================================================================

COVERS (Section E of the question bank):
  E1. Best retrieval techniques for better answer generation (dense/sparse/hybrid/...)
  E2. Reranking & cross-encoders — the exact wiring, in code
  E3. Multimodality in RAG / AI
  E4. Detecting & reducing hallucinations

HOW TO READ:  THEORY -> RUNNABLE CODE -> INTERVIEW ANSWER -> RELATED CONCEPTS.

Run:  & "C:\\Projects\\Gen AI Udemy\\LangCGS\\.venv\\Scripts\\python.exe" "05_rag_retrieval_reranking_multimodal.py"

Pure standard library + NumPy only. BM25, MMR, and a cross-encoder scorer are all
implemented FROM SCRATCH so you understand the mechanism behind every retrieval
library you've ever imported.
===================================================================================
"""

from __future__ import annotations

import math
import re
import sys
from collections import Counter

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

np.random.seed(21)
np.set_printoptions(precision=3, suppress=True)


def banner(title: str) -> None:
    line = "=" * 83
    print("\n" + line + "\n" + title + "\n" + line)


def sub(title: str) -> None:
    print("\n" + "━" * 79 + "\n" + title + "\n" + "━" * 79)


# ===================================================================================
# SHARED HELPERS (toy embedder + cosine, same pattern as Lessons 3-4)
# ===================================================================================

def _stable_hash(s: str) -> int:
    """Deterministic hash — Python's built-in hash() is SALTED per process (would make
    this embedder non-reproducible across runs). This fixes that."""
    import hashlib
    return int.from_bytes(hashlib.md5(s.encode()).digest()[:4], "little")


def toy_embed(text: str, d: int = 256) -> np.ndarray:
    """Deterministic toy embedder (char-trigram hashing) — stands in for a real model."""
    v = np.zeros(d)
    t = text.lower()
    for i in range(len(t) - 2):
        v[_stable_hash(t[i:i + 3]) % d] += 1.0
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


# ===================================================================================
# E1. BEST RETRIEVAL TECHNIQUES FOR BETTER ANSWER GENERATION
# ===================================================================================
#
# THE MENU (know what each buys you, and be ready to say WHY it helps generation):
#
#   DENSE RETRIEVAL (embeddings + ANN, Lessons 3-4)
#     Matches by MEANING. Great for paraphrases/synonyms ("car" ~ "automobile").
#     Weak on exact identifiers — product codes, names, acronyms it's never seen.
#
#   SPARSE / LEXICAL RETRIEVAL (BM25, TF-IDF)
#     Matches by EXACT TOKEN overlap, weighted by term rarity (IDF) and saturation
#     (TF). Great for exact codes/names/rare terms; blind to paraphrase.
#
#   HYBRID SEARCH (BM25 + vector, fused)
#     Run both, COMBINE scores (weighted sum or Reciprocal Rank Fusion). Gets the
#     best of both: exact-match recall AND semantic recall. THE most reliable
#     single upgrade to retrieval quality in production RAG.
#
#   QUERY EXPANSION / REWRITING
#     Use an LLM to rewrite the user's (often terse/ambiguous) query into one or more
#     better search queries before retrieving — e.g., expanding acronyms, adding
#     synonyms, or decomposing a multi-part question into sub-queries.
#
#   MULTI-QUERY RETRIEVAL
#     Generate SEVERAL rephrasings of the query, retrieve for each, and merge/de-dup
#     results — reduces sensitivity to any single query's exact wording.
#
#   MMR (Maximal Marginal Relevance)
#     After the initial similarity ranking, re-order to balance RELEVANCE against
#     DIVERSITY, so top-k isn't 5 near-duplicate chunks saying the same thing.
#
#   PARENT-DOCUMENT / SMALL-TO-BIG RETRIEVAL
#     Embed and search SMALL chunks (precise matching), but return the LARGER parent
#     section/document as context to the LLM (richer surrounding context, less
#     fragmentation) — decouples "what you search on" from "what you feed the model."
#
#   METADATA FILTERING
#     Narrow the candidate set by structured fields (date, source, department) BEFORE
#     or alongside similarity search — huge precision win when metadata is available.
#
#   WHICH PRODUCES THE BEST OUTPUT, AND WHY
#     In practice: HYBRID SEARCH + RERANKING (Lesson section E2) is the highest-ROI
#     combo — hybrid maximizes recall of the right candidates, reranking maximizes
#     precision of the final top-k actually shown to the LLM.


def bm25_scores(query: str, corpus: list[str], k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    """BM25 from scratch — the standard sparse/lexical retrieval scoring function.

    score(q,d) = sum over query terms of IDF(t) * TF(t,d)*(k1+1) /
                 (TF(t,d) + k1*(1 - b + b*|d|/avgdl))
    """
    docs_tokens = [tokenize(d) for d in corpus]
    N = len(docs_tokens)
    avgdl = sum(len(d) for d in docs_tokens) / N

    df = Counter()
    for toks in docs_tokens:
        for term in set(toks):
            df[term] += 1
    idf = {t: math.log((N - df[t] + 0.5) / (df[t] + 0.5) + 1) for t in df}

    q_terms = tokenize(query)
    scores = np.zeros(N)
    for i, toks in enumerate(docs_tokens):
        tf = Counter(toks)
        dl = len(toks)
        s = 0.0
        for term in q_terms:
            if term not in idf:
                continue
            f = tf[term]
            s += idf[term] * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avgdl))
        scores[i] = s
    return scores


def hybrid_scores(query: str, corpus: list[str], embeddings: list[np.ndarray],
                  alpha: float = 0.5) -> np.ndarray:
    """Hybrid = weighted fusion of normalized BM25 (sparse) + cosine (dense) scores."""
    bm25 = bm25_scores(query, corpus)
    bm25_norm = bm25 / (bm25.max() + 1e-9)
    qv = toy_embed(query)
    dense = np.array([cosine(qv, e) for e in embeddings])
    dense_norm = (dense - dense.min()) / (dense.max() - dense.min() + 1e-9)
    return alpha * dense_norm + (1 - alpha) * bm25_norm


def mmr(query_vec: np.ndarray, doc_vecs: list[np.ndarray], k: int,
       lambda_param: float = 0.5) -> list[int]:
    """Maximal Marginal Relevance: greedily pick items that are relevant to the query
    but NOT redundant with what's already selected."""
    candidates = list(range(len(doc_vecs)))
    selected: list[int] = []
    while candidates and len(selected) < k:
        def mmr_score(i: int) -> float:
            relevance = cosine(query_vec, doc_vecs[i])
            if not selected:
                return relevance
            redundancy = max(cosine(doc_vecs[i], doc_vecs[j]) for j in selected)
            return lambda_param * relevance - (1 - lambda_param) * redundancy
        best = max(candidates, key=mmr_score)
        selected.append(best)
        candidates.remove(best)
    return selected


def _demo_retrieval_techniques() -> None:
    sub("E1. DENSE vs SPARSE vs HYBRID vs MMR (runnable)")
    corpus = [
        "SKU-4471 replacement filter for the X200 water purifier",
        "how to replace the water filter cartridge in your purifier",
        "automobile maintenance schedule for oil changes",
        "car engine oil change interval recommendations",
        "return policy for water purifier accessories",
    ]
    query = "SKU-4471 filter replacement"

    bm25 = bm25_scores(query, corpus)
    print("BM25 (sparse, exact-token) scores:")
    for i, s in sorted(enumerate(bm25), key=lambda x: -x[1]):
        print(f"  {s:6.2f}  {corpus[i]!r}")
    print("  -> BM25 nails the exact SKU code because it matches the literal token.\n")

    embeddings = [toy_embed(d) for d in corpus]
    qv = toy_embed(query)
    dense = [cosine(qv, e) for e in embeddings]
    print("Dense (semantic) scores:")
    for i, s in sorted(enumerate(dense), key=lambda x: -x[1]):
        print(f"  {s:6.3f}  {corpus[i]!r}")
    print("  -> dense also picks up the paraphrase ('replace the water filter').\n")

    hyb = hybrid_scores(query, corpus, embeddings, alpha=0.5)
    print("Hybrid (fused BM25 + dense) scores:")
    for i, s in sorted(enumerate(hyb), key=lambda x: -x[1]):
        print(f"  {s:6.3f}  {corpus[i]!r}")
    print("  -> hybrid keeps the exact-match win AND benefits from semantic recall.\n")

    print("MMR (diversify a candidate set, avoid near-duplicate top-k):")
    order = mmr(qv, embeddings, k=3, lambda_param=0.6)
    for rank, i in enumerate(order, 1):
        print(f"  {rank}. {corpus[i]!r}")
    print("  -> balances relevance with NOT repeating the same content twice.")


# INTERVIEW ANSWER (E1):
#   "Dense retrieval matches meaning via embeddings but misses exact identifiers;
#    sparse retrieval like BM25 matches exact tokens via term-frequency/inverse-
#    document-frequency but misses paraphrase. Hybrid search fuses both score sets —
#    normalized and weighted or via reciprocal rank fusion — and is the single highest
#    -ROI upgrade to retrieval quality. On top of that I use MMR to diversify the
#    final top-k so I'm not feeding the LLM five near-duplicate chunks, and query
#    rewriting/expansion to handle terse or ambiguous user queries before retrieval."
#
# RELATED CONCEPTS: Reciprocal Rank Fusion (RRF) as an alternative fusion method;
#   query decomposition for multi-hop questions; parent-document/small-to-big
#   retrieval; metadata pre-filtering; retrieval latency budget vs quality trade-off.


# ===================================================================================
# E2. RERANKING & CROSS-ENCODERS — the exact wiring, in code
# ===================================================================================
#
# WHY RE-RANK AFTER THE INITIAL VECTOR SEARCH
#   The initial ANN search (bi-encoder embeddings) is optimized for SPEED across
#   millions of vectors — it embeds the query and docs INDEPENDENTLY, so it can never
#   look at the two together. That's fast but approximate. Reranking adds a SLOWER,
#   MORE ACCURATE second pass over just the top-K candidates, where accuracy matters
#   more than speed because the candidate set is now small.
#
# CROSS-ENCODER vs BI-ENCODER
#   Bi-encoder (used for initial retrieval): embed query and document SEPARATELY into
#   vectors, compare with cosine/dot product. Fast (precompute doc vectors once,
#   compare with simple math) but loses fine-grained query-document interaction.
#   Cross-encoder (used for reranking): feed the (query, document) pair TOGETHER into
#   one model (e.g., a BERT-style model with [query, SEP, document] as input); the
#   model attends across BOTH texts jointly and outputs a single relevance score.
#   Much more accurate (it can reason about exact wording overlap, negation, etc.)
#   but too slow to run over millions of documents — hence: bi-encoder for the wide
#   first pass, cross-encoder for the narrow, accurate second pass.
#
# INPUTS / OUTPUTS
#   Input:  (query, candidate_document) — a PAIR, not independent vectors.
#   Output: a single scalar RELEVANCE SCORE per pair (often via a sigmoid on a
#           classification head — "how relevant is this doc to this query").
#   Reordering: sort candidates by this score, descending; keep the new top-N.
#
# THE EXACT WIRING (the code they want to see):
#   1) retrieve top-K candidates from the vector store (K > N, e.g., K=50)
#   2) build (query, candidate) pairs for ALL K candidates
#   3) score each pair with the cross-encoder
#   4) sort candidates by cross-encoder score, descending
#   5) keep the final top-N (e.g., N=5) — THESE go into the LLM prompt, not the
#      original bi-encoder ranking.
#
# TRADE-OFFS
#   Latency:  +1 extra scoring pass over K candidates (not the whole corpus) — small
#             if K is kept modest (20-100), since cross-encoders are O(K), not O(N).
#   Accuracy: usually a meaningful bump in precision@top-N — worth it whenever K is
#             cheap to fetch and the reranker is reasonably fast (e.g., a small
#             cross-encoder like ms-marco-MiniLM).
#   Cost:     an extra model call per query (or a batch of K calls) — real but small
#             compared to the LLM generation call that follows.


def toy_cross_encoder_score(query: str, document: str) -> float:
    """Illustrative cross-encoder: unlike a bi-encoder, it looks at query AND
    document TOGETHER (here: token overlap + order-sensitive bigram overlap) instead
    of comparing two independently-computed vectors. A real cross-encoder is a
    fine-tuned Transformer scoring [CLS] query [SEP] document [SEP]; this captures
    the SAME idea — joint reasoning over the pair — without needing model weights.
    """
    q_tokens, d_tokens = tokenize(query), tokenize(document)
    q_set, d_set = set(q_tokens), set(d_tokens)
    token_overlap = len(q_set & d_set) / (len(q_set) + 1e-9)

    q_bigrams = {(q_tokens[i], q_tokens[i + 1]) for i in range(len(q_tokens) - 1)}
    d_bigrams = {(d_tokens[i], d_tokens[i + 1]) for i in range(len(d_tokens) - 1)}
    bigram_overlap = len(q_bigrams & d_bigrams) / (len(q_bigrams) + 1e-9)

    # Joint signal: exact phrase overlap (bigrams) weighted higher than loose token
    # overlap — this "joint reasoning over the pair" is what a bi-encoder can't do.
    return 0.4 * token_overlap + 0.6 * bigram_overlap


def rerank(query: str, candidates: list[str], top_n: int) -> list[tuple[float, str]]:
    """The exact wiring: score every (query, candidate) pair, sort, keep top_n."""
    scored = [(toy_cross_encoder_score(query, doc), doc) for doc in candidates]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_n]


def _demo_reranking() -> None:
    sub("E2. RERANK PIPELINE — bi-encoder recall -> cross-encoder precision (runnable)")
    corpus = [
        "SKU-4471 replacement filter for the X200 water purifier",
        "how to replace the water filter cartridge in your purifier",
        "the annual filter replacement schedule for HVAC systems",
        "car engine oil filter replacement guide",
        "return policy for water purifier accessories",
        "water purifier warranty and support contact information",
    ]
    query = "how do I replace the water filter"

    # STEP 1: bi-encoder retrieval (fast, wide net) — top-K by embedding similarity
    embeddings = [toy_embed(d) for d in corpus]
    qv = toy_embed(query)
    dense = [(cosine(qv, e), d) for e, d in zip(embeddings, corpus)]
    dense.sort(key=lambda x: x[0], reverse=True)
    top_k = [d for _, d in dense[:4]]
    print("Step 1 — bi-encoder top-K (fast, approximate):")
    for score, doc in dense[:4]:
        print(f"  {score:.3f}  {doc!r}")

    # STEP 2-5: cross-encoder rerank of just those K candidates
    print("\nStep 2-5 — cross-encoder rerank of the K candidates -> final top-N:")
    reranked = rerank(query, top_k, top_n=2)
    for score, doc in reranked:
        print(f"  {score:.3f}  {doc!r}")
    print("\n  -> reranking is applied ONLY to the K candidates the bi-encoder already")
    print("     found — it refines precision, it does not replace the wide first pass.")


# INTERVIEW ANSWER (E2):
#   "Bi-encoders embed query and document independently for fast large-scale search
#    but can't model fine interaction between them. A cross-encoder takes the (query,
#    document) pair TOGETHER, attends across both jointly, and outputs one relevance
#    score — far more accurate but too slow for millions of documents. So the
#    pipeline is: bi-encoder retrieves top-K candidates from the vector store, I build
#    (query, candidate) pairs for all K, score each with the cross-encoder, sort
#    descending, and keep the final top-N for the LLM prompt. It costs one extra
#    O(K) scoring pass but meaningfully improves precision at the top."
#
# RELATED CONCEPTS: ms-marco-MiniLM-style rerankers; Cohere Rerank API; ColBERT (late
#   interaction — a middle ground between bi- and cross-encoders); reranking latency
#   budget vs K size; combining reranking with MMR (rerank for relevance, then
#   diversify) — order matters, rerank first, diversify second.


# ===================================================================================
# E3. MULTIMODALITY IN RAG / AI
# ===================================================================================
#
# WHAT MULTIMODAL SYSTEMS ARE
#   Systems that process/reason across MORE THAN ONE data type — text, images, audio,
#   video — often in the SAME embedding space, so you can compare a text query against
#   an image, or an image against another image, using the same similarity math.
#
# HOW MULTIMODALITY APPLIES TO RAG SPECIFICALLY
#   - MULTIMODAL EMBEDDINGS (e.g., CLIP): text and images are embedded into the SAME
#     vector space, so "a photo of a dog" (text) lands near actual dog PHOTOS. This
#     lets you retrieve images via a text query, or vice versa, using ordinary
#     cosine-similarity search — no architecture change to the vector store itself.
#   - VISION-LANGUAGE MODELS (VLMs) as the GENERATOR: instead of (or alongside) text
#     chunks, you retrieve relevant images/diagrams/screenshots and pass them directly
#     to a multimodal LLM (GPT-4V-class) that can read text AND look at the image to
#     answer — e.g., "what does the error screen in this screenshot mean?"
#   - DOCUMENT LAYOUT / TABLES / CHARTS: real-world documents aren't pure text — PDFs
#     have tables, charts, scanned pages. Multimodal RAG can embed a PAGE IMAGE
#     directly (skipping brittle OCR/table-parsing) and let a VLM read it visually.
#   - AUDIO/VIDEO: transcribe (speech-to-text) for text-based retrieval, OR embed
#     audio/video segments directly with multimodal encoders for cross-modal search.
#
# PRACTICAL PROJECT FRAMING (how to talk about it even without a shipped multimodal
# RAG project): "In DocSage-style projects I've worked with text/PDF/CSV ingestion.
# The natural multimodal extension is embedding page IMAGES with CLIP-family models
# so a user query can retrieve a diagram or chart directly, and passing that image to
# a VLM instead of relying purely on OCR'd text — which breaks on complex layouts."
#   (Be HONEST about what you've actually built vs. what you understand conceptually
#    — interviewers respect "I understand the architecture, haven't shipped it" far
#    more than a vague overclaim.)


def _demo_multimodal_shared_space() -> None:
    sub("E3. MULTIMODAL SHARED EMBEDDING SPACE (illustrative — CLIP-style idea)")
    # We can't run real CLIP here (needs a trained model + images), but we CAN show
    # the core idea: text describing an image lands NEAR that image's own embedding
    # in a SHARED space — exactly what lets a text query retrieve an image directly.
    # NOTE: our toy_embed is a crude trigram-hash, so we widen its dimension (d=256)
    # here to reduce hash collisions and keep the illustration honest — a real CLIP
    # model does this properly with learned, trained embeddings.
    d = 256
    image_captions = {
        "img_001.jpg": "a golden retriever dog sitting on a grassy lawn outdoors",
        "img_002.jpg": "a red sports car parked on a city highway at sunset",
        "img_003.jpg": "a bowl of fresh fruit on a wooden kitchen table indoors",
    }
    image_vecs = {k: toy_embed(v, d) for k, v in image_captions.items()}

    text_query = "a dog sitting outdoors on grass"
    qv = toy_embed(text_query, d)
    ranked = sorted(image_vecs.items(), key=lambda kv: -cosine(qv, kv[1]))
    print(f"  text query: {text_query!r}")
    print("  retrieved images by cross-modal similarity:")
    for img, vec in ranked:
        print(f"    sim={cosine(qv, vec):.3f}  {img}  ({image_captions[img]!r})")
    print("\n  -> the dog/outdoors query correctly ranks the dog image highest, purely")
    print("     from vector closeness in a SHARED space — no keyword matching rule was")
    print("     written for 'dog'. In a real CLIP-style system, the IMAGE PIXELS")
    print("     themselves are embedded into this space (not captions), so retrieval")
    print("     works even for images that were never manually tagged or OCR'd.")


# INTERVIEW ANSWER (E3):
#   "Multimodal RAG uses models like CLIP to embed text AND images into the SAME
#    vector space, so a text query can retrieve relevant images by plain cosine
#    similarity — no OCR needed. Beyond retrieval, you can pass retrieved images
#    directly to a vision-language model like GPT-4V so it reasons over the visual
#    content alongside text context, which matters a lot for documents with tables,
#    charts, or scanned pages that break traditional text extraction."
#
# RELATED CONCEPTS: CLIP/BLIP-2/Florence-2 embedding models; VLM-based OCR-free
#   document understanding; audio embeddings (Whisper for transcription vs. direct
#   audio embeddings for retrieval); video as sampled-frame retrieval.


# ===================================================================================
# E4. DETECTING & REDUCING HALLUCINATIONS
# ===================================================================================
#
# WHAT COUNTS AS A HALLUCINATION HERE
#   The LLM states something as fact that is NOT supported by (or contradicts) the
#   retrieved context — even though RAG gave it grounding material to work from.
#   RAG reduces but does NOT eliminate hallucination; the model can still ignore or
#   misread the context.
#
# DETECTION TECHNIQUES
#   - GROUNDING/FAITHFULNESS CHECK: for each claim in the answer, verify it's entailed
#     by the retrieved context — often via an LLM-as-judge prompt ("is this sentence
#     supported by this context? yes/no") or an NLI (entailment) model. This is
#     literally the "faithfulness" metric from Lesson 6 (RAGAS).
#   - CITATION-BASED CHECKS: force the model to cite WHICH chunk/source backs each
#     claim; if it can't produce one, or the cited chunk doesn't actually say that,
#     flag it. Citations also let a HUMAN verify quickly.
#   - CONFIDENCE / CONSISTENCY CHECKS: ask the same question multiple times (or with
#     paraphrased prompts) and check answer consistency — wildly different answers to
#     the same grounded question signal the model is guessing, not reading context.
#   - RETRIEVAL COVERAGE CHECK: if retrieval returned LOW-similarity chunks (below a
#     confidence threshold), that's a leading indicator the model has weak grounding
#     material and may fabricate to fill the gap.
#
# REDUCTION TECHNIQUES
#   - PROMPT DESIGN: explicit instruction "answer ONLY using the provided context; if
#     the answer isn't in the context, say you don't know" — dramatically cuts
#     fabrication vs. an unconstrained prompt.
#   - GROUNDING IN RETRIEVED CONTEXT: put the context BEFORE the question, clearly
#     delimited (e.g., inside XML-like tags), so the model's attention is anchored to
#     it.
#   - REQUIRE CITATIONS: forces the model toward context-grounded phrasing instead of
#     free recall from its parametric memory.
#   - GUARDRAILS / POST-HOC VALIDATION: run the faithfulness check (above) and, if it
#     fails, either regenerate with a stricter prompt, fall back to "I don't know," or
#     route to a human.
#   - LOWER TEMPERATURE for fact-based tasks (less creative sampling -> more
#     deterministic, context-anchored generation) — see Lesson 6/9 for the trade-off.


def simple_faithfulness_check(answer: str, context: str) -> dict:
    """A crude, dependency-free faithfulness proxy: what fraction of the answer's
    content words are actually attested in the retrieved context? A REAL system uses
    an LLM-as-judge or NLI model (see Lesson 6) — this shows the underlying IDEA.
    """
    answer_terms = set(tokenize(answer))
    context_terms = set(tokenize(context))
    stopwords = {"the", "a", "an", "is", "are", "to", "of", "and", "in", "on", "for"}
    content_terms = answer_terms - stopwords
    if not content_terms:
        return {"grounded_ratio": 1.0, "ungrounded_terms": []}
    grounded = content_terms & context_terms
    ungrounded = content_terms - context_terms
    return {
        "grounded_ratio": len(grounded) / len(content_terms),
        "ungrounded_terms": sorted(ungrounded),
    }


def _demo_hallucination_check() -> None:
    sub("E4. HALLUCINATION DETECTION — a simple faithfulness proxy (runnable)")
    context = ("Our refund policy allows returns within 30 days of purchase. "
               "Items must be unused and in original packaging.")

    grounded_answer = "You can return items within 30 days if they are unused."
    hallucinated_answer = "You can return items within 90 days and get double refunds."

    for label, answer in [("grounded", grounded_answer), ("hallucinated", hallucinated_answer)]:
        result = simple_faithfulness_check(answer, context)
        print(f"  [{label:>12}] answer: {answer!r}")
        print(f"                grounded_ratio={result['grounded_ratio']:.2f}, "
              f"ungrounded_terms={result['ungrounded_terms']}")
    print("\n  -> the hallucinated answer's invented specifics ('90 days', 'double')")
    print("     don't appear in the source context — exactly what a faithfulness")
    print("     check (RAGAS or LLM-as-judge, Lesson 6) is designed to catch.")


# INTERVIEW ANSWER (E4):
#   "RAG reduces hallucination but doesn't eliminate it — the model can still ignore
#    its context. I detect it with faithfulness/grounding checks that verify each
#    claim is entailed by the retrieved context, usually via an LLM-as-judge or NLI
#    model, plus requiring citations so claims are traceable to a source chunk. To
#    reduce it, I use explicit grounding instructions — answer only from context, say
#    'I don't know' otherwise — put context ahead of the question, lower temperature
#    for factual tasks, and add a post-hoc validation guardrail that can trigger a
#    regenerate or human fallback if the faithfulness check fails."
#
# RELATED CONCEPTS: RAGAS faithfulness metric (Lesson 6); NLI entailment models for
#   automated fact-checking; self-consistency sampling; retrieval-confidence
#   thresholds as an early-warning signal before generation even happens.


# ===================================================================================
# GOLDEN LESSONS
# ===================================================================================
GOLDEN_LESSONS = """
  1. Dense = meaning (misses exact IDs); sparse/BM25 = exact tokens (misses paraphrase); hybrid = both.
  2. Hybrid search (fused BM25 + vector) is the single highest-ROI retrieval upgrade in production.
  3. MMR trades pure relevance for diversity so top-k isn't 5 near-duplicate chunks.
  4. Bi-encoder = embed query/doc SEPARATELY (fast, wide net). Cross-encoder = embed the PAIR TOGETHER.
  5. Rerank wiring: retrieve top-K (bi-encoder) -> score (query,candidate) pairs (cross-encoder) -> sort -> top-N.
  6. Reranking is O(K) not O(N) — cheap because it only touches the candidates already retrieved.
  7. Multimodal RAG (CLIP-style) embeds text AND images into ONE shared space — text queries find images, no OCR.
  8. VLMs let the LLM literally LOOK at retrieved images/tables/charts, not just read OCR'd text.
  9. RAG reduces hallucination, does NOT eliminate it — the model can still ignore its own context.
 10. Faithfulness = is each claim entailed by the retrieved context? Detect via LLM-as-judge/NLI.
 11. Reduce hallucination: explicit grounding instructions + citations + context-before-question + guardrails.
 12. Best-practice stack for generation quality: hybrid retrieval -> rerank -> grounded prompt -> faithfulness check.
"""


if __name__ == "__main__":
    banner("LESSON 5 — RAG: RETRIEVAL, RERANKING & MULTIMODALITY")
    _demo_retrieval_techniques()
    _demo_reranking()
    _demo_multimodal_shared_space()
    _demo_hallucination_check()
    banner("GOLDEN LESSONS")
    print(GOLDEN_LESSONS)
