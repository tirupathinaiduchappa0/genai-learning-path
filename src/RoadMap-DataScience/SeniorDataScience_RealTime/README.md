# SeniorDS_RealTime — Interview Prep Pack

Deep-dive prep for Senior Data Scientist / GenAI engineering interviews. Every file covers a question with **theory → real-time/production context → hands-on runnable code → an interview script** you can say out loud.

## How to use
1. Skim **`00_rapid_fire_answers.md`** for the 60–90 second version of all 18 answers.
2. Read the numbered lessons for depth, code, and the "what to say" script.

## Index

| # | Lesson | Theme |
|---|--------|-------|
| 00 | Rapid-fire answers (all 18) | Quick reference |
| 01 | Detecting duplicate file content (hashlib) | Python & Systems |
| 02 | Multiprocessing vs multithreading | Python & Systems |
| 03 | Implement a rate limiter | Python & Systems |
| 04 | Vector store internals (IVF, HNSW) | Vectors |
| 05 | FAISS / Chroma: library vs database | Vectors |
| 06 | Embeddings & vector search | Vectors |
| 07 | Milvus vs FAISS vs Pinecone | Vectors |
| 08 | RAG architecture | RAG |
| 09 | Chunking & chunk-size optimization | RAG |
| 10 | RAG evaluation (faithfulness, relevancy) | RAG |
| 11 | Handling hallucinations | RAG |
| 12 | Prompt design: ticket extraction → JSON | LLM Engineering |
| 13 | Simple knowledge-base agent | LLM Engineering |
| 14 | Python function to call an LLM API | LLM Engineering |
| 15 | Fine-tuning LLaMA / Falcon | LLM Engineering |
| 16 | Temperature and top_p | LLM Engineering |
| 17 | Evaluating AI responses (precision/recall) | Evaluation & Ops |
| 18 | Securing & monitoring LLM apps | Evaluation & Ops |

## Themes
- **A. Python & Systems Fundamentals** — 01, 02, 03
- **B. Vector Stores & Embeddings** — 04, 05, 06, 07
- **C. Retrieval-Augmented Generation** — 08, 09, 10, 11
- **D. LLM Application Engineering** — 12, 13, 14, 15, 16
- **E. Evaluation & Production Ops** — 17, 18

## Notes
- Code samples are illustrative; install the relevant libs (`openai`, `sentence-transformers`, `faiss-cpu`, `ragas`, `transformers`, `peft`, `scikit-learn`) to run them.
- Cross-references between lessons are noted inline (e.g., Lesson 10 ↔ Lesson 11).
