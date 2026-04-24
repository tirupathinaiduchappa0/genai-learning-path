# 🤖 LangCGS — Generative AI Learning Path & Projects

A comprehensive, hands-on learning repository covering **LangChain, LangGraph, RAG patterns, and production AI development** — from fundamentals to a deployable real-world project.

## 📁 Repository Structure

```
src/
├── LangchainBasics/         # LangChain fundamentals (data ingestion, chunking, embeddings, vector stores, RAG chains, LCEL, LangServe, memory)
├── LangChainUpdatedV1.0/    # LangChain v1.x features (agents, tools, messages, structured output, middleware)
├── AIAgentVsAgenticAI/      # Conceptual lessons on AI agents vs agentic AI
├── LangGraph/               # LangGraph deep dive (state, nodes, edges, routing, ReAct, memory, streaming, HITL, workflows)
├── LangSmith/               # LangSmith tracing and observability
├── RAG/                     # Advanced RAG patterns (Agentic, Corrective, Adaptive RAG + concepts guide)
├── RealWorldProjects/
│   └── docsage/             # 🚀 Enterprise Document Intelligence Agent (production project)
└── InterviewPrep/           # Interview preparation (LLMs, ML/DL basics, GenAI concepts, Python)
```

## 🚀 DocSage — Enterprise Document Intelligence Agent

The flagship project. A Streamlit-powered AI assistant with:
- **Multi-document RAG** — Upload PDFs, DOCX, TXT, CSV, MD + paste URLs
- **Agentic tool selection** — Agent decides which knowledge base to search
- **Document grading** — Filters irrelevant chunks before generation
- **Post-generation validation** — Hallucination and answer relevance checks
- **Web search fallback** — Tavily for questions outside uploaded docs
- **Email integration** — Send answers via Gmail SMTP
- **Conversation memory** — Multi-turn Q&A with MemorySaver
- **Step-by-step streaming** — Real-time progress as each node completes

## ⚡ Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/LangCGS.git
cd LangCGS
python -m venv .venv
.venv/Scripts/activate        # Windows
pip install -r requirements.txt
cp .env.example .env          # Add your API keys
```

**Run DocSage:**
```bash
cd src/RealWorldProjects
streamlit run docsage/app.py
```

**Run any lesson:**
```bash
python src/LangGraph/07_react_agent.py
```

## 🔑 Required API Keys

| Key | Free? | Get it at |
|-----|-------|-----------|
| Groq API | ✅ Yes | https://console.groq.com/keys |
| Tavily Search | ✅ Yes | https://app.tavily.com/home |
| LangSmith | ✅ Yes | https://smith.langchain.com |
| Gmail App Password | ✅ Yes | https://myaccount.google.com/apppasswords |

## 🛠 Tech Stack

LangChain · LangGraph · Groq (LLaMA) · FAISS · HuggingFace Embeddings · Tavily · Streamlit · Pydantic · Python 3.11+
