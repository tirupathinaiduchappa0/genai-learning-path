# Lesson 06 — Embeddings & Vector Search

## 1. The Question

> "What are embeddings? How do you generate them, and how do you use them in vector search? Walk through the full flow."

---

## 2. Theory — what is an embedding?

An **embedding** is a dense, fixed-length vector of floats that represents the **meaning** of an input (text, image, audio) in a continuous space. The defining property:

> Semantically similar inputs map to vectors that are **close** together; dissimilar inputs map far apart.

So "king" and "queen" land near each other; "king" and "banana" don't. Meaning becomes **geometry**, and "find similar" becomes "find nearest."

- Dimensions: typically 384 (MiniLM), 768 (BERT-base), 1536 (OpenAI `text-embedding-3-small`), 3072 (large).
- They're **learned** by models trained so that related items are pulled together and unrelated pushed apart (contrastive / masked objectives).

### Sparse vs dense
- **Sparse** (TF-IDF, BM25): keyword overlap; a huge mostly-zero vector. Great for exact terms, blind to synonyms.
- **Dense** (neural embeddings): capture semantics/synonyms; "car" ≈ "automobile".
- **Hybrid search** combines both — best recall in practice.

---

## 3. Distance / similarity metrics

- **Cosine similarity** — cares about *direction* (angle), not magnitude. Default for text. Range [-1, 1].
- **Dot product** — direction and magnitude; equals cosine when vectors are normalized.
- **Euclidean (L2)** — straight-line distance.

Tip: **normalize** vectors, then dot product == cosine, and it's cheaper.

---

## 4. Generating embeddings

### Local (open source, free, private)

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")   # 384-dim, fast
vecs = model.encode(
    ["The cat sat on the mat", "A feline rested on the rug"],
    normalize_embeddings=True,
)
# these two are near-identical despite sharing almost no words
```

### API (OpenAI / Azure OpenAI)

```python
from openai import OpenAI
client = OpenAI()

def embed(texts):
    resp = client.embeddings.create(
        model="text-embedding-3-small",   # 1536-dim
        input=texts,
    )
    return [d.embedding for d in resp.data]
```

**Golden rule:** use the **same model** to embed documents and queries. Mixing models = incomparable spaces = garbage results.

---

## 5. The full vector-search flow

```
OFFLINE (indexing):
  documents → chunk → embed → store vectors (+ metadata) in index

ONLINE (query):
  query text → embed (same model) → ANN search top-k → return docs
```

### End-to-end runnable example

```python
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

docs = [
    "Python is a programming language.",
    "The Eiffel Tower is in Paris.",
    "Pandas is a data analysis library.",
    "Paris is the capital of France.",
]

# OFFLINE: embed & store
doc_vecs = model.encode(docs, normalize_embeddings=True)

def search(query, k=2):
    q = model.encode([query], normalize_embeddings=True)[0]
    scores = doc_vecs @ q            # dot product == cosine (normalized)
    top = np.argsort(scores)[::-1][:k]
    return [(docs[i], float(scores[i])) for i in top]

print(search("What language do I code in?"))
# -> Python doc scores highest, even without the word "Python" in the query
print(search("Tell me about the French capital"))
# -> Paris docs win
```

Notice retrieval works on **meaning**, not keyword overlap — that's the whole point.

---

## 6. Real-time / production concerns

- **Cost & latency:** batch your embedding calls; cache embeddings (they're deterministic per text+model).
- **Dimensionality:** bigger isn't always better; smaller dims = cheaper storage/search. Some models (Matryoshka / `text-embedding-3`) let you truncate dims.
- **Chunking matters** (see Lesson 09): you embed chunks, not whole docs.
- **Normalization:** normalize once at write time so search is a cheap dot product.
- **Domain fit:** general embeddings can underperform on jargon (legal/medical/code) — consider domain-specific or fine-tuned embedding models.
- **Re-embedding:** if you change the embedding model, you must re-embed the entire corpus.

---

## 7. Interview script

"An embedding is a dense vector where semantic similarity becomes geometric closeness, so 'find related' turns into 'find nearest neighbor.' I generate them with a model — sentence-transformers locally or the OpenAI embeddings API — and the critical rule is using the same model for documents and queries. Offline I chunk, embed, and index; online I embed the query and do an ANN search by cosine similarity. I normalize vectors so cosine reduces to a dot product, cache embeddings since they're deterministic, and often add BM25 for hybrid search to catch exact keywords."
