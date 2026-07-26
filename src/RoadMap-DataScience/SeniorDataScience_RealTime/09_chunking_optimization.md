# Lesson 09 — Chunking & Chunk-Size Optimization in RAG

## 1. The Question

> "How do you implement chunking in a RAG pipeline? How do you choose and optimize chunk size and overlap? What are the trade-offs?"

---

## 2. Theory — why chunk at all?

You can't embed a whole 50-page document as one vector: it would blur many topics into one point, and it wouldn't fit useful context into the prompt. So you split documents into **chunks** — passages small enough to be a single coherent idea, big enough to be self-contained.

Chunking directly controls retrieval quality. It's often the **highest-leverage knob** in a RAG system — more than the choice of LLM.

### The core trade-off

- **Too small** (e.g., 1 sentence): high precision on the exact match, but loses surrounding context; the model gets fragments and can't reason. More chunks = more index cost.
- **Too large** (e.g., whole page): each chunk covers many topics, so the query vector matches it weakly (**dilution**), and you waste prompt tokens on irrelevant text.
- **Goldilocks:** one chunk ≈ one coherent idea. Typical sweet spot is **200–500 tokens** with **10–20% overlap**, but it's data-dependent — tune empirically.

---

## 3. Chunking strategies (worst → best)

### 3.1 Fixed-size (character/token count)
Split every N characters. Simple, but blindly cuts through sentences and ideas.

### 3.2 Fixed-size **with overlap**
Carry the last ~10–20% of each chunk into the next so an idea split across a boundary still appears whole in at least one chunk. The standard baseline.

### 3.3 Recursive character splitting
Try to split on natural separators in priority order: paragraphs `\n\n` → lines `\n` → sentences → words. Keeps semantic units intact where possible. (LangChain's `RecursiveCharacterTextSplitter`.)

### 3.4 Document-structure aware
Split on Markdown headings, HTML tags, code functions, PDF sections. Respects the author's structure. Great for docs/wikis/code.

### 3.5 Semantic chunking
Embed sentences, start a new chunk when the topic shifts (embedding distance between consecutive sentences spikes). Best coherence, higher compute cost.

### 3.6 Contextual / late chunking
Prepend a short doc- or section-level summary to each chunk before embedding (Anthropic "contextual retrieval") so a chunk like "It increased 20%" knows what "it" is. Big recall win.

---

## 4. Hands-on

### Overlap in plain Python (see the mechanics)

```python
def chunk_text(text, size=500, overlap=50):
    chunks, start = [], 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap        # step back by overlap
    return chunks
```

### Production splitters (LangChain)

```python
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)

# Recursive: respects paragraph/sentence boundaries, token-aware
splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=60,
    separators=["\n\n", "\n", ". ", " ", ""],
)
chunks = splitter.split_text(document_text)

# Structure-aware for Markdown docs
md_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")]
)
md_chunks = md_splitter.split_text(markdown_doc)  # keeps heading as metadata
```

### Contextual chunking sketch

```python
def contextualize(chunk, doc_summary):
    # store this enriched text's embedding; keep original for display
    return f"[Document: {doc_summary}]\n{chunk}"
```

---

## 5. How to *optimize* chunk size (the empirical loop)

Don't guess — measure. This is the senior answer.

1. Build an **eval set**: representative questions + the passages that should answer them.
2. Sweep configs: `chunk_size ∈ {128, 256, 512, 1024}` × `overlap ∈ {0, 10%, 20%}` × strategy.
3. For each config, run retrieval and score **context precision / recall** and end-to-end **faithfulness/relevancy** (Lesson 10).
4. Plot metric vs config; pick the knee (best quality within your token/latency budget).
5. Re-check when documents or the embedding model change.

```python
for size in [128, 256, 512, 1024]:
    for overlap in [0, size // 10, size // 5]:
        index = build_index(chunk(docs, size, overlap))
        score = evaluate(index, eval_questions)   # context recall / faithfulness
        print(size, overlap, score)
```

---

## 6. Real-time / production notes

- **Embedding model window:** don't exceed the model's max input; small models (MiniLM ~256 tokens) force smaller chunks.
- **Match chunk to content type:** prose → recursive/semantic; code → by function/class; tables → keep rows together; chat logs → by turn.
- **Store metadata per chunk:** source, section, page, doc summary → enables filtering and citations.
- **Parent-document / small-to-big:** retrieve on small precise chunks but feed the LLM the larger parent chunk for context — best of both.
- **Overlap costs storage** (duplicated text) — a real trade-off at scale.

---

## 7. Interview script

"Chunking is the highest-leverage knob in RAG. Too small loses context, too large dilutes relevance and wastes tokens — the sweet spot is usually 200–500 tokens with 10–20% overlap, but I tune it empirically. I prefer recursive or structure-aware splitting over naive fixed-size so I don't cut through ideas, and for tricky corpora I use semantic or contextual chunking, where each chunk carries a short document summary so references resolve. To optimize, I build an eval set and sweep size/overlap/strategy, scoring context recall and faithfulness, then pick the knee. I also like the parent-document trick: retrieve on small chunks, generate on the larger parent."
