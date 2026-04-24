# 🤖 GenAI Developer Rules — (Read Once, Apply Always)

> **Who I am**: I'm a Java full-stack developer transitioning to Python + Generative AI.
> I have **mastered Python** (OOP, modules, decorators, async, etc.).
> My goal is to become a **production-grade Generative AI Developer**.
> Help me learn by doing — write real code, explain concepts deeply, and build toward production.

---

## 🎯 My Learning Stack (in priority order)

1. **LangChain** — chains, prompt templates, output parsers, LCEL
2. **LangGraph** — state graphs, workflows, human-in-the-loop, memory
3. **LangSmith** — tracing, evaluation, debugging, token optimization
4. **LangServe** — deploying chains as REST APIs
5. **LangFlow** — visual pipeline builder (low-code prototyping)
6. **LlamaIndex** — RAG pipelines, data connectors, query engines
7. **CrewAI** — multi-agent orchestration, role-based agents
8. **MCP (Model Context Protocol)** — tool servers, connecting external tools
9. **Streamlit** — frontend for all GenAI apps
10. **Vector Stores** — Chroma, Pinecone, FAISS, Weaviate
11. **Palantir AIP** — enterprise AI platform concepts

---

## 🧠 Core Philosophy for All Code

- Always write **production-ready** code, not just demos
- Use **type hints** on every function signature
- Write **docstrings** for every class and function
- Prefer **modular structure**: split into `chains/`, `agents/`, `tools/`, `utils/`, `config/`
- Use **`.env` files** for all API keys — never hardcode
- Use **`python-dotenv`** to load environment variables
- Always add **error handling** with meaningful messages
- Add **logging** using Python's `logging` module, not `print()`
- Write **comments** explaining *why*, not *what*

---

## 📁 Recommended Project Structure

```
my_genai_app/
├── .env                    # API keys (never commit)
├── .env.example            # Template for teammates
├── requirements.txt        # Pin all versions
├── README.md
├── main.py                 # Entry point
├── config/
│   └── settings.py         # All config via pydantic BaseSettings
├── chains/
│   └── rag_chain.py        # LangChain chains
├── agents/
│   └── research_agent.py   # LangGraph / CrewAI agents
├── tools/
│   └── search_tool.py      # Custom tools / MCP tools
├── memory/
│   └── conversation.py     # Memory management
├── vectorstore/
│   └── ingestion.py        # Data ingestion pipeline
├── prompts/
│   └── templates.py        # Prompt templates
├── api/
│   └── server.py           # LangServe FastAPI endpoints
└── app/
    └── streamlit_app.py    # Streamlit frontend
```

---

## 🔗 LangChain Rules

### Always Use LCEL (LangChain Expression Language)
```python
# ✅ CORRECT — Use pipe operator for chains
chain = prompt | llm | output_parser

# ❌ AVOID — Old-style LLMChain is deprecated
chain = LLMChain(llm=llm, prompt=prompt)
```

### Prompt Templates
- Always use `ChatPromptTemplate.from_messages()` for chat models
- Use `MessagesPlaceholder` for dynamic conversation history
- Use `HumanMessagePromptTemplate` and `SystemMessagePromptTemplate`
- Keep system prompts in separate files under `prompts/`

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant specialized in {domain}."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])
```

### Output Parsers
- Use `StrOutputParser` for plain text responses
- Use `JsonOutputParser` with a Pydantic model for structured output
- Use `PydanticOutputParser` when you need validation
- Chain parsers at the end: `chain = prompt | llm | StrOutputParser()`

### Token Optimization (Critical for Production)
- Always set `max_tokens` on the LLM to cap response size
- Use `RecursiveCharacterTextSplitter` with `chunk_size=1000, chunk_overlap=200`
- Trim conversation history with `trim_messages()` before sending to LLM
- Use `tiktoken` to count tokens before sending requests
- Implement **sliding window** for long conversations

```python
from langchain_core.messages import trim_messages

trimmed = trim_messages(
    messages,
    max_tokens=4000,
    strategy="last",         # Keep the most recent messages
    token_counter=llm,
    include_system=True,
)
```

---

## 🔁 LangGraph Rules

### State Graph Pattern
```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

# Always define state as TypedDict
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]  # append-only
    context: str
    next_step: str

# Build graph
builder = StateGraph(AgentState)
builder.add_node("researcher", researcher_node)
builder.add_node("writer", writer_node)
builder.set_entry_point("researcher")
builder.add_conditional_edges("researcher", route_fn, {"write": "writer", "end": END})
graph = builder.compile()  # Always compile before running
```

### LangGraph Components
- `StateGraph` — main graph builder
- `add_node(name, fn)` — register a node function
- `add_edge(from, to)` — fixed edge
- `add_conditional_edges(from, condition_fn, mapping)` — dynamic routing
- `set_entry_point(node)` — where execution starts
- `compile(checkpointer=...)` — returns a `CompiledGraph`
- `graph.stream(input, config)` — stream results step by step
- `graph.invoke(input, config)` — run synchronously

### Human-in-the-Loop
```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
graph = builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["human_review_node"],  # Pause before this node
)

# Resume after human input
graph.invoke(
    Command(resume={"human_feedback": "Looks good, proceed"}),
    config={"configurable": {"thread_id": "thread-1"}}
)
```

### Conversation Memory in LangGraph
- Use `MemorySaver` for in-memory (dev/testing only)
- Use `SqliteSaver` or `PostgresSaver` for persistence in production
- Always pass `thread_id` in config for per-conversation isolation

### Debugging LangGraph Apps
- Use `graph.get_graph().print_ascii()` to visualize graph structure
- Use `graph.stream(..., stream_mode="debug")` for step-by-step logs
- Integrate with LangSmith for full trace visibility
- Check `state["messages"]` at each node for debugging

---

## 🔍 LangSmith Rules (Tracing & Observability)

```python
# In .env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_key
LANGCHAIN_PROJECT=my-project-name

# Wrapping custom functions for tracing
from langsmith import traceable

@traceable(name="custom_retrieval_step")
def retrieve_documents(query: str) -> list:
    ...
```

- Always name your runs: `chain.invoke(input, config={"run_name": "RAG Query"})`
- Use **tags** to group related runs: `config={"tags": ["production", "v2"]}`
- Use **metadata** for filtering: `config={"metadata": {"user_id": "u123"}}`
- Check token usage per run in the LangSmith dashboard
- Set up **evaluators** for automated quality scoring

---

## 🗂️ RAG Pipeline Rules

### Data Ingestion Pipeline (always modular)
```
Load → Split → Embed → Store
```

### Document Loaders
```python
from langchain_community.document_loaders import (
    PyPDFLoader,          # PDFs
    WebBaseLoader,        # URLs
    CSVLoader,            # CSV files
    JSONLoader,           # JSON files
    DirectoryLoader,      # Entire folder
    UnstructuredFileLoader,  # Any file type
)
```

### Chunking Strategy
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""],  # Priority order
)
```
- Use `chunk_overlap=200` to prevent context loss at boundaries
- For code: use `Language.PYTHON` splitter
- For markdown: use `MarkdownHeaderTextSplitter` first, then character splitter

### Embeddings
```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

# OpenAI (paid, high quality)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# HuggingFace (free, good quality)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
```

### Vector Stores
```python
from langchain_community.vectorstores import Chroma, FAISS
from langchain_pinecone import Pinecone

# Local dev: Chroma or FAISS
vectorstore = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")

# Production: Pinecone
vectorstore = Pinecone.from_documents(docs, embeddings, index_name="my-index")
```

### Similarity Search Methods
- `vectorstore.similarity_search(query, k=4)` — returns top-k docs
- `vectorstore.similarity_search_with_score(query, k=4)` — returns docs + scores
- **Cosine similarity** — measures angle between vectors (most common, scale-invariant)
- **Euclidean distance** — measures straight-line distance (sensitive to magnitude)
- **MMR (Maximal Marginal Relevance)** — balances relevance + diversity

```python
# Use MMR to avoid redundant results
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 6, "lambda_mult": 0.5}
)
```

### RAG Chain (LCEL)
```python
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

---

## 🤖 AI Agents vs Agentic AI

| Concept | Description |
|---|---|
| **AI Agent** | Single LLM that calls tools to complete a task |
| **Agentic AI** | Multi-step, multi-model system with planning, memory, and reflection |
| **ReAct Agent** | Reason → Act → Observe loop |
| **Tool-calling Agent** | LLM decides which tools to call and when |
| **Multi-agent System** | Multiple agents collaborating (e.g., CrewAI) |

### Building Agents with LangGraph (preferred over legacy AgentExecutor)
```python
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(
    model=llm,
    tools=[search_tool, calculator_tool],
    state_modifier="You are a research assistant."
)
```

### CrewAI Multi-Agent Pattern
```python
from crewai import Agent, Task, Crew, Process

researcher = Agent(role="Researcher", goal="Find accurate info", llm=llm, tools=[search_tool])
writer = Agent(role="Writer", goal="Write clear summaries", llm=llm)

task1 = Task(description="Research {topic}", expected_output="Bullet points", agent=researcher)
task2 = Task(description="Write a blog post", expected_output="500-word post", agent=writer)

crew = Crew(agents=[researcher, writer], tasks=[task1, task2], process=Process.sequential)
result = crew.kickoff(inputs={"topic": "Quantum Computing"})
```

---

## 🔌 MCP (Model Context Protocol) Rules

### What is MCP
MCP is an open standard for connecting LLMs to external tools and data sources — like a USB-C port for AI tools.

### MCP Architecture
```
MCP Host (Claude / your app)
    ↕ MCP Protocol (JSON-RPC over stdio / HTTP)
MCP Server (Jira, GitHub, Slack, databases, etc.)
```

### Connecting an MCP Server in Python
```python
from langchain_mcp_adapters.client import MultiServerMCPClient

client = MultiServerMCPClient({
    "jira": {
        "command": "npx",
        "args": ["-y", "@atlassian/mcp-server-jira"],
        "env": {"JIRA_API_TOKEN": os.getenv("JIRA_API_TOKEN")}
    },
    "github": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-github"],
        "env": {"GITHUB_TOKEN": os.getenv("GITHUB_TOKEN")}
    }
})

tools = await client.get_tools()   # Returns LangChain-compatible tools
agent = create_react_agent(llm, tools)
```

### Writing a Custom MCP Server
```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("My Custom Tools")

@mcp.tool()
def get_jira_ticket(ticket_id: str) -> dict:
    """Fetch a Jira ticket by ID."""
    ...

if __name__ == "__main__":
    mcp.run()   # Starts stdio server
```

---

## 🎨 Streamlit Frontend Rules

- Always use `st.session_state` for conversation history — never use global variables
- Use `st.chat_message()` and `st.chat_input()` for chat UIs
- Use `st.spinner()` for loading states during LLM calls
- Cache expensive operations with `@st.cache_resource` (LLM, vectorstore init)
- Use `@st.cache_data` for data loading functions

```python
@st.cache_resource
def load_vectorstore():
    """Initialize vectorstore once, reuse across sessions."""
    return Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    response = chain.invoke({"input": prompt, "chat_history": st.session_state.messages})
    st.session_state.messages.append({"role": "assistant", "content": response})
```

---

## 🚀 LangServe Deployment Rules

```python
from fastapi import FastAPI
from langserve import add_routes

app = FastAPI(title="My GenAI API", version="1.0")

add_routes(app, rag_chain, path="/rag")
add_routes(app, chat_chain, path="/chat")

# Auto-generates:
# POST /rag/invoke
# POST /rag/stream
# GET  /rag/playground  ← Built-in UI for testing
```

---

## 🧩 Key Concepts Cheat Sheet

| Concept | One-liner |
|---|---|
| **Prompt Template** | Reusable prompt with variables |
| **LCEL** | Pipe `\|` operator to compose chains |
| **Output Parser** | Converts LLM string → structured Python object |
| **Retriever** | Interface to fetch relevant docs from vectorstore |
| **Runnable** | Any LCEL-compatible component (pipe-able) |
| **RunnablePassthrough** | Passes input unchanged (for parallel branches) |
| **RunnableParallel** | Runs multiple runnables in parallel |
| **Checkpointer** | Persists LangGraph state between runs |
| **Thread ID** | Unique ID per conversation in LangGraph |
| **Embeddings** | Dense vector representation of text |
| **Cosine Similarity** | Angle between vectors (1 = identical) |
| **Context Window** | Max tokens an LLM can process at once |
| **Vectorless Embeddings** | Keyword/BM25 search (no vectors needed) |
| **Hybrid Search** | Combine dense (vector) + sparse (keyword) retrieval |
| **Reranker** | Re-scores retrieved docs for better relevance |

---

## 🔐 Security Rules

- **Never** commit `.env` files — always add to `.gitignore`
- Always validate user input before passing to LLM (prompt injection)
- Set rate limits on LangServe APIs using FastAPI middleware
- Use Pydantic models to validate all inputs and outputs
- Sanitize retrieved documents before injecting into prompts

---

## 📝 Resume-Worthy Project Ideas

Each project below is a complete end-to-end app to put on your resume:

1. **Multi-PDF RAG Chatbot** — Upload PDFs, ask questions, cite sources (Streamlit + LangChain + Chroma)
2. **Agentic Research Assistant** — Web search + summarize + write report (LangGraph + CrewAI + Tavily)
3. **Customer Support Agent** — FAQ RAG + ticket creation via MCP + Jira integration
4. **Code Review Bot** — Review PRs using LangGraph multi-agent + GitHub MCP
5. **LangSmith Evaluation Dashboard** — Trace, score, and compare RAG pipelines
6. **Conversational SQL Agent** — Natural language to SQL with LangGraph + SQLite

---

## 🤝 How to Work With Me in This IDE

- **Explain a concept**: Say "Explain [concept] with code example and production tips"
- **Build a feature**: Say "Build [feature] using [stack], production-ready, with error handling"
- **Debug my code**: Paste the error + relevant code, I'll trace through it
- **Review my code**: I'll check for production issues, token efficiency, and best practices
- **Architecture help**: Ask me to design the system before coding it
- **LangSmith trace help**: Share the trace JSON, I'll diagnose the issue

> Always tell me which framework/tool is the focus and whether this is for learning or production.
