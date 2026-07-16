# 🟢 START HERE — Context Handoff for a New AI Model / New Laptop

> **If you are an AI assistant reading this for the first time: READ THIS FILE FULLY, then read the 4 conversation summary files in this same folder (`conversastation_summary-1.py` → `conversation_summary-4.py`). After that, you will have the full context of who this user is, what he already knows, and what exists in this repo. Do NOT start from scratch or explain things at a beginner level — he already has the foundation described below.**

---

## 1. WHO THIS IS

- **Name:** Tirupathi Naidu (Tirupathi).
- **Total experience:** ~4 years as a software engineer.
- **GenAI experience:** ~2 years (positioned/presented as GenAI on resume; the deep hands-on GenAI focus is the most recent ~1.5–2 yrs).
- **Origin skillset:** Java Full-Stack developer — **Java, Spring Boot, microservices, Spring Batch, REST APIs, MySQL/MongoDB**, and front-end with **React.js (Hooks, Redux Toolkit, TypeScript, Axios)**.
- **Current identity/direction:** **Generative AI Engineer.** Actively interviewing for GenAI / AI Engineer / Senior Associate roles.
- **Strongest coding language historically:** JavaScript. **Python** was ramped up recently — comfortable now, but earlier he practiced by converting JS solutions to Python. When helping with live Python syntax, prefer clear, standard, compilable Python.
- **Company context:** Worked at **Infor** (ERP domain — SXE/CSD). Built an internal **MCP server** exposing 17+ LLM-callable tools for ERP (product pricing, quote lifecycle, order management, customer credit, recommendations).
- **HuggingFace username:** `tirupathi0`. **Live project:** https://huggingface.co/spaces/tirupathi0/docsage

### How to work with him (learned preferences)
- Wants **deep-dive, senior/technical-lead level** content (20+ yr practitioner depth). **"Quality over coverage."**
- Reads lessons **line by line**. Prefers **bullet points over paragraphs**.
- Every concept should carry a **memorizable INTERVIEW ANSWER one-liner/soundbite**.
- Every lesson ends with a **GOLDEN LESSONS** section.
- Goes **step by step with review** — do not dump an entire large implementation without checkpoints.
- **Integrity rule (important):** He does **NOT** want live answers fed to him *during* an interview (he tried, then refused — it's cheating and some companies e.g. PwC prohibit AI use). Help fully with **prep and debrief**, never live cheating.
- **Never fabricate URLs** — give search terms or verified links.

---

## 2. THE PROJECT / REPO

- **Repo root:** `LangCGS` (this repo). Python GenAI learning + interview-prep workspace.
- **Sibling repo:** `docsage-deploy` (the deployed DocSage app, also cloned separately).
- **Language/tooling:** Python. Uses **`uv`** + `pyproject.toml`. Virtual env at `LangCGS/.venv`.
- **LLM provider:** **Groq API (free)** is primary — NOT OpenAI (paid).
  - `llama-3.1-8b-instant` → tool calling / generation.
  - `llama-3.3-70b-versatile` → structured output / grading / routing.
- **Other keys used:** `TAVILY_API_KEY` (web search), Gmail SMTP (email tool). All secrets live in `.env` (git-ignored — do not commit).

### Mandatory coding rules (in `LangCGS/kiro/`)
- `kiro/Important-Rules.md` — general rules: type hints, docstrings, logging, `.env`, error handling, modular code.
- `kiro/LangGraph-rules.md` — 12 LangGraph-specific rules.
- **Env loading pattern he wants:** `load_dotenv()` then `os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")`. Do **NOT** use the `Path(__file__).resolve().parents[...]` pattern.
- **Graph visualization:** use `draw_mermaid_png()` (saves PNG to a `graphs/` folder), NOT `print_ascii()`.

### Run / verify command (Windows)
```
& "C:\Projects\Gen AI Udemy\LangCGS\.venv\Scripts\python.exe" "path\to\file.py"
```
- His terminal sometimes shows `(Python Course)` venv — that's the **WRONG** venv. Always use `LangCGS/.venv`.
- Console mangles em-dashes / unicode (shows `ù`) — that's cosmetic; files are correct UTF-8.
- Note on file-writing: large files were built with append-in-chunks. Lessons are plain `.py` files that print their content when run.

---

## 3. THE LEARNING JOURNEY (topic-ordered — what he already knows)

This is the arc he went through. Treat all of this as **known foundation**.

1. **LangChain fundamentals** — document loaders/ingestion → text splitting/chunking → embeddings → vector stores/retrieval → prompts/chains/retrieval → **LCEL** → **LangServe** (server/client) → **conversational memory**.
2. **LangChain v1.x** — `create_agent`, `init_chat_model`, `@tool`, `bind_tools`, `ToolNode`, `tools_condition`, message types (Human/AI/Tool), **structured output** (Pydantic/TypedDict/dataclass), **middleware** (Summarization, Human-in-the-Loop approve/edit/reject).
3. **AI Agents vs Agentic AI** — conceptual distinction + resume/project use cases.
4. **LangGraph (core, 10 lessons)** — why LangGraph → state/nodes/edges → LLM chatbot + streaming → **reducers** (`operator.add` vs `add_messages` vs custom) → state schema types → chains/tools/routing → **ReAct agent** → **memory/checkpointing** (`MemorySaver`, `thread_id`) → streaming deep dive → **Human-in-the-Loop**.
5. **LangGraph Workflow patterns** — Prompt Chaining, Parallelization, Routing (`route_decision`), **Orchestrator-Workers** (`Send` API for dynamic workers).
6. **LangSmith** — tracing concepts + hands-on (`@traceable`, `langgraph.json`).
7. **RAG (advanced)** — **Agentic RAG, Corrective RAG (CRAG), Adaptive RAG** + advanced concepts (chunking, HNSW/IVF, separator precedence, vectorless RAG, knowledge-base updates).
8. **Real-world project — DocSage** — Enterprise Document Intelligence Agent (see §5).
9. **Agents Hands-On series** — universal pattern (LLM + tools + `create_react_agent`). Done: Code Analyzer, Database Query (Text-to-SQL). Skeletons for agent/RAG/RAG-with-tools. (Agents 2, 4–10 pending — Log Debugger, Code Reviewer, Doc Generator, Customer Support, Data Pipeline, Email Classifier, Resume Screener, Test Generator.)
10. **Interview preparation** — a large numbered lesson series (see §4) covering LLMs, ML/DL, GenAI, Python, HuggingFace, RAG end-to-end, RAG evaluation (RAGAS/LLM-as-Judge/drift), Agentic AI architecture (5 layers, multi-agent patterns, block/flow diagrams), ML fundamentals articulation, production RAG, structured-data RAG, prompt management/versioning.
11. **FastAPI** — 3 lessons with Spring Boot comparisons (these live in the separate `Python Course` repo, not here).
12. **Deloitte/HashedIn 10-phase lead-level prep** (see §4).

---

## 4. FOLDER MAP (where everything lives under `LangCGS/src/`)

| Folder | What it contains |
|---|---|
| `ConversastationSummary/` | **This file + the 4 session summaries.** Start here. |
| `LangchainBasics/` | LangChain foundations. Subfolders: `DataIngestion`, `DataChunking`, `Embeddings`, `VectorStore`, `RAGChains` (LCEL), `LangServe`, `ConversationalMemory`. |
| `LangChainUpdatedV1.0/` | LangChain v1.x: agents, model integration, tools, messages, structured output, middleware (6 lessons). |
| `AIAgentVsAgenticAI/` | 3 conceptual lessons (AI agents, agentic AI, resume/use-cases). |
| `LangGraph/` | 10 core LangGraph lessons (`01`–`10`) + `graphs/` (PNGs). |
| `LangGraph/10_Workflows/` | Workflow patterns: prompt chaining, parallelization, routing, orchestrator-workers. |
| `LangSmith/` | 2 lessons (concepts + hands-on tracing). |
| `RAG/` | Agentic RAG, Corrective RAG, Adaptive RAG, advanced RAG concepts (+ reference PDFs). |
| `RealWorldProjects/docsage/` | **The DocSage project** (also deployed; mirror of `docsage-deploy`). |
| `AgentHandsOn/` | Hands-on agents series (code analyzer, DB query agent) + skeletons. |
| `InterviewPrep/` | The big numbered interview lesson series `00`–`25` (see below) + role-specific subfolders. |
| `InterviewPrep/PythonCoding/` | Python coding practice: arrays, strings, dicts, searching/sorting/DSA, advanced patterns, 160-Q walkthrough. |
| `InterviewPrep/DeloitteHashedIn/` | 10-phase lead-level prep (FastAPI, async, API design, DBs, testing, multi-agent, system design, architecture/leadership, cloud/devops, mock) + a 6-question post-mortem. |
| `InterviewPrep/MockInterviews/` | GenAI mock interview deep dive. |
| `InterviewPrep/RoleReferences/` | Keyword reference cards for roles he might pursue later (e.g., Computer Vision Engineer). |

### `InterviewPrep/` numbered lessons (00–25)
- `00` Naukri statement · `01` How LLMs work · `02` ML/DL basics · `03` GenAI concepts · `04` Python interview · `05` HuggingFace basics
- `06` DocSage deep dive · `07` RAG complete guide · `08` VidaXL AI-prompter prep · `09` GenAI features in real project · `10` JPMC agentic-dev prep
- `11` VidaXL case-study PPT · `12` VidaXL L2 hands-on · `13` ETech Sr AI Developer prep · `14` New concepts deep dive · `15` Advanced RAG deep dive
- `16` Advanced Python deep dive · `17` RAG evaluation deep dive · `18` Agentic AI architecture deep dive · `19`/`19.1` ML fundamentals articulation
- `20` Production RAG confidence · `21` Advanced RAG mastery · `22` DocSage walkthrough script + cheat card (`.md`)
- `23` PwC R2 pack (battle plan, self-intro, product-attribute explainer, Infor context — all `.md`) · `24` RAG with structured data · `25` Prompt management & versioning

---

## 5. THE FLAGSHIP PROJECT — DocSage (know this well)

**DocSage = Enterprise Document Intelligence Agent.** Production-grade **Streamlit** app using **LangGraph + Agentic RAG**. Deployed on **Hugging Face Spaces**.

- **What it does:** Multi-document Q&A (PDF, DOCX, TXT, CSV, MD) + URL web scraping + web-search fallback (Tavily) + email integration (Gmail SMTP) + conversation memory + step-by-step streaming + document grading.
- **Architecture:** ~8 packages / 15+ files, modular. Node workflow with conditional routing.
  - `config/settings.py` — central config
  - `llms/groq_llm.py` — LLM factory (agent, grading, generation, rewrite)
  - `state/state.py` — TypedDict with `add_messages` reducer
  - `tools/retriever_tool.py` — multi-format ingestion + URL scraping
  - `tools/web_search_tool.py` — Tavily fallback
  - `tools/email_tool.py` — Gmail SMTP `@tool`
  - `nodes/agent_node.py` — tool selection with 10-message context window
  - `nodes/grade_node.py` — document grading (+ email tool bypass → "done" route)
  - `nodes/generate_node.py` — RAG generation with source citations
  - `nodes/rewrite_node.py` — query rewriting (self-correction)
  - `nodes/validate_node.py` — hallucination + answer-relevance checks
  - `graph/graph_builder.py` — full graph (with tools) or simple graph (no tools)
  - `ui/sidebar.py`, `ui/chat_interface.py` — upload/model/API-key UI + streaming
  - `main.py` — orchestrator with fingerprint-based graph caching · `app.py` — entry point
- **RAG patterns demonstrated:** Agentic, Corrective, Adaptive. **Embeddings:** HuggingFace. **Vector store:** FAISS.
- **This is his main resume project** — the walkthrough script + cheat card are `InterviewPrep/22_*`.

---

## 6. CURRENT STATUS SNAPSHOT (as of last session — early July 2026)

- **Learning:** Currently GenAI-focused. Open to learning **ML / Computer Vision / image-video-audio processing** later (that's why `RoleReferences/` exists — e.g. the Computer Vision Engineer keyword reference card is done).
- **Interviews (historical outcomes, informational):**
  - T-Mobile — did not clear (live Python coding task).
  - VidaXL — cleared L1 + case study + L2 → HR process.
  - JPMC — VP round went well.
  - ETech — went deep on RAG.
  - PwC — cleared R1, went to R2 (architect-level, Agentic AI diagrams).
  - Deloitte/HashedIn (Lead Python/GenAI) — prepped all 10 phases.
  - A Computer Vision Engineer founder chat — exploratory (he lacks CV skills; GenAI bridge is the angle).
- **Known weak spots he flagged to reinforce:** RAG **evaluation** (precision/recall, confusion matrix, RAGAS); **regularization**; **why MCP**; AWS agent frameworks; articulating **classification vs regression** cleanly; writing **compilable Python live** under pressure.
- **Also exploring:** no-code automation (**n8n** recommended first, then Make).

---

## 7. HOW TO USE THIS ON A NEW LAPTOP

1. Clone the repo, recreate `.venv`, restore `.env` (secrets are NOT in git — he keeps them separately).
2. Tell the model: **"Read `src/ConversastationSummary/00_START_HERE.md` first, then the 4 summary files in that folder. Then wait for my instructions."**
3. The model now knows his level, preferences, and the full file map — it can jump straight to deep work.

## 8. MAINTENANCE NOTE
Whenever significant new work is added to the repo, **update this file** (add to §3 journey, §4 folder map, §6 status). Keep the 4 session summaries as the frozen archive; keep THIS file as the living index.

---
*Living index — update as the repo grows. The 4 `*summary-*.py` files are the frozen per-session archive.*
