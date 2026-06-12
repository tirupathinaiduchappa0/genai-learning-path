# DocSage Walkthrough — Pre-Interview Cheat Card

> Glance for 60 seconds before the call. Follow the 6 beats in order. Stay calm.

---

## THE 6 BEATS (your fixed order — never deviate)

| # | Beat | Time | One-liner to open with |
|---|------|------|------------------------|
| 1 | **What + problem** | 30s | "DocSage — Enterprise Document Intelligence Agent. Agentic RAG that decides, grades, validates, and self-corrects." |
| 2 | **4-layer architecture** | 30s | "UI → ingestion → LangGraph workflow → tools." |
| 3 | **Trace ONE request** | 3m | "Let me follow one question through the system…" |
| 4 | **Deep dive 2-3 parts** | 3m | state+reducer, self-correction, LLM factory |
| 5 | **Decisions + trade-offs** | 1.5m | "Chose X because Y, trade-off Z, in prod I'd do W." |
| 6 | **Results + limits + roadmap** | 1m | works → honest limits → prod roadmap |

> **Start top-down. NEVER start with folders or UI widgets.**

---

## THE SPINE (say it like a chant)

```
START → AGENT → TOOLS → GRADE → GENERATE → VALIDATE → END
                  ^                              |
                  +---------- REWRITE <----------+   (2 self-correction loops)
```

- **AGENT** — LLM + bind_tools, *decides* which tool (ReAct)
- **TOOLS** — ToolNode runs FAISS search (k=4)
- **GRADE** — relevant? yes→generate / no→rewrite (Pydantic structured output) → *Corrective RAG*
- **GENERATE** — answer from context only + citations
- **VALIDATE** — hallucination check + answer-relevance check → *Adaptive/Self-RAG*
- **REWRITE** — reword query, loop back to agent

---

## NUMBERS TO KNOW COLD

- **5 nodes** | recursion limit **20**
- Chunking **1000 / 200 overlap** | retriever **k=4**
- Embeddings **all-MiniLM-L6-v2** (local, free, 384-dim)
- Vector store **FAISS** | Memory **MemorySaver + thread_id**
- Models: **8b-instant** (agent/gen) · **70b-versatile** (grade/validate)
- Deploy **Hugging Face Spaces** (keys via HF Secrets)

---

## PATTERNS TO NAME (drop these = senior signal)

`Agentic RAG` · `Corrective RAG (CRAG)` · `Adaptive/Self-RAG` · `ReAct tool selection` · `structured output` · `add_messages reducer` · `checkpointer`

---

## DEEP-DIVE PICKS (choose 2-3, not all)

- **State + reducer** — `add_messages` appends, else context is overwritten (the #1 LangGraph bug).
- **Self-correction** — 2 loops: bad retrieval → rewrite; bad answer → regenerate/rewrite.
- **LLM factory** — right model per task, temp 0 for grading, 0.3 for gen, swap in one place.
- *(reliability)* structured output + try/except fallback to "yes".
- *(product)* multi-source tools: doc retrievers + Tavily web + Gmail + URL.

---

## DECISIONS (sentence shape: *chose X because Y, trade-off Z, prod → W*)

- **LangGraph** — need loops + branching; a linear chain can't loop back.
- **FAISS** — fast/local/free; in-memory → prod = **Qdrant/Pinecone**.
- **Local embeddings** — free, no egress; slightly lower quality, swappable.
- **Groq** — free + fast so loops don't drag; prod → OpenAI/Bedrock (one file).

---

## LIMITS + ROADMAP (own them confidently)

**Limits:** FAISS in-memory · no OCR for scanned PDFs · binary grading (not a reranker).
**Roadmap:** Qdrant + metadata filtering · cross-encoder **reranker** · **hybrid search** · **RAGAS** eval · **LangSmith** tracing.

> Say "grading + adaptive retrieval" — NOT "reranking" (you don't have a reranker yet).

---

## THE PAUSE (after the spine — sound senior)

> "I can go deeper on the self-correction loops or the retrieval pipeline — what's most useful for you?"

## 60-SEC VERSION (if rushed)

> "DocSage is an agentic RAG document assistant on LangGraph. Documents go into a FAISS index; a 5-node workflow handles each query — an agent picks the tool, a grader checks relevance, a generator gives a cited answer, a validator checks hallucination and relevance, with self-correction loops that rewrite and retry. Multi-format docs, web fallback, streaming UI, conversation memory. Modular and swappable. Deployed on HF Spaces."

---
**If you blank out:** breathe → say beat 1 → then "let me trace one question" → the spine carries you.
