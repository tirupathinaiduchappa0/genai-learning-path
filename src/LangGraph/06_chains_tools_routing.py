"""
🔗 LangGraph Lesson 6 — Chains, Tools & Routing: From Simple Chain to Multi-Tool Agent (Level 3)

═══════════════════════════════════════════════════════════════════
1. CONCEPT: Chains, Tools & Routing — ONE-LINE DEFINITION
═══════════════════════════════════════════════════════════════════

A Chain is a sequence of nodes connected by edges.
A Tool is an external function the LLM can call.
Routing is the LLM deciding: answer directly OR call a tool.

Together they form the foundation of every LangGraph agent:
    Chain → Tool Binding → Router → Multi-Tool Agent

═══════════════════════════════════════════════════════════════════
2. WHY CHAINS, TOOLS & ROUTING EXIST
═══════════════════════════════════════════════════════════════════

LLMs are powerful — but they can't do math, search the web, or query
databases on their own. They need TOOLS for that.

The problem: how does the LLM DECIDE when to use a tool vs answer directly?
That's where ROUTING comes in.

    Chain (no tools):   User → LLM → Answer
    Chain (with tools): User → LLM → (needs tool?) → YES → Tool → Answer
                                                    → NO  → Answer directly

In LangChain, this was handled by AgentExecutor — a BLACK BOX.
In LangGraph, you build the EXACT same logic with EXPLICIT nodes and edges.
Full control. Full visibility. No magic.

═══════════════════════════════════════════════════════════════════
3. WHERE IT FITS IN THE LANGGRAPH ARCHITECTURE
═══════════════════════════════════════════════════════════════════

    State (messages with add_messages reducer)
        ↓
    LLM Node (calls the model, may produce tool_calls)
        ↓
    tools_condition (checks: did the LLM request a tool call?)
        ├── YES → ToolNode (executes the tool, returns result)
        └── NO  → END (return the LLM's direct answer)

    ToolNode wraps LangChain tools for use inside a graph.
    tools_condition is the ROUTER — it reads the LLM's response
    and decides the next step.

═══════════════════════════════════════════════════════════════════
4. REAL-WORLD ANALOGY
═══════════════════════════════════════════════════════════════════

Think of a RECEPTIONIST at a company:

    1. Someone calls in with a question.
    2. If the receptionist KNOWS the answer → responds directly.
    3. If the question needs a SPECIALIST → transfers the call
       to the right department (accounting, IT, legal...).
    4. The specialist answers → receptionist relays the answer.

    The receptionist IS the LLM.
    The departments ARE the tools.
    The decision to transfer IS the routing.
    The whole process IS the chain.

═══════════════════════════════════════════════════════════════════
5. ASCII GRAPH STRUCTURES
═══════════════════════════════════════════════════════════════════

Demo 1 — Simple Chain (no tools):

    [START]
       |
    [chatbot]
       |
    [END]

Demo 3 & 4 — Router Pattern (with tools):

    [START]
       |
    [tool_calling_llm]
       |
       ├── (tool_calls?) ──→ [tools] ──→ [END]
       |
       └── (no tool_calls) ──→ [END]

═══════════════════════════════════════════════════════════════════

🔄 LANGCHAIN vs LANGGRAPH
LangChain : AgentExecutor runs the LLM → tool → LLM loop as a BLACK BOX.
            You can't see or control the routing logic.
LangGraph  : You build the loop YOURSELF with nodes + conditional edges.
            tools_condition is the router. ToolNode executes tools.
            Every step is visible, debuggable, and customizable.

LangChain : Tools are passed to initialize_agent() — implicit binding.
LangGraph  : Tools are bound with llm.bind_tools() — explicit binding.
            ToolNode wraps them for graph execution.

HOW TO RUN:
    $ python src/LangGraph/06_chains_tools_routing.py

Author: GenAI Learner
Date: 2025-07-15
"""

import logging
import os

from dotenv import load_dotenv

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

GRAPH_DIR = os.path.join(os.path.dirname(__file__), "graphs")
os.makedirs(GRAPH_DIR, exist_ok=True)


def save_graph_image(compiled_graph, name: str) -> None:
    """Save the compiled graph as a PNG image."""
    try:
        png_data = compiled_graph.get_graph().draw_mermaid_png()
        filepath = os.path.join(GRAPH_DIR, f"{name}.png")
        with open(filepath, "wb") as f:
            f.write(png_data)
        logger.info("  Graph image saved: %s", filepath)
    except Exception as e:
        logger.warning("  Could not save graph image: %s", e)



# ═══════════════════════════════════════════════════════════════════════════════
# 6. DEMO 1: Simple Chain — LLM Node Only (No Tools)
# ═══════════════════════════════════════════════════════════════════════════════
#
# This is the FOUNDATION. A chain is just nodes connected by edges.
# Here: START → chatbot → END. The chatbot node calls the LLM and returns
# the response. No tools, no routing — just a straight-line chain.
#
# This is the simplest possible LangGraph application:
#   - State holds messages (with add_messages reducer for smart append)
#   - One node calls the LLM
#   - The graph runs from START to END
#
# WHY START HERE?
# Because every agent is built ON TOP of this pattern.
# If you understand this, you understand the skeleton of every LangGraph app.

def demo_simple_chain() -> None:
    """Simple chain: START → chatbot → END. No tools, no routing."""
    from typing import Annotated

    from langchain_core.messages import AnyMessage, HumanMessage
    from langchain_groq import ChatGroq
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages
    from typing_extensions import TypedDict

    # ── State: messages with add_messages reducer (production standard) ──
    class State(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]

    # ── LLM: ChatGroq with llama-3.1-8b-instant ─────────────────────────
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_tokens=512)

    # ── Node: chatbot — calls the LLM with the current messages ─────────
    def chatbot(state: State) -> dict:
        """Call the LLM and return its response as a new message."""
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    # ── Build the graph: START → chatbot → END ───────────────────────────
    builder = StateGraph(State)
    builder.add_node("chatbot", chatbot)       # Register the node
    builder.add_edge(START, "chatbot")          # Entry edge
    builder.add_edge("chatbot", END)            # Exit edge
    graph = builder.compile()                   # Compile to runnable

    save_graph_image(graph, "lesson06_simple_chain")

    # ── Invoke with a simple question ────────────────────────────────────
    result = graph.invoke({"messages": [HumanMessage(content="What is Python?")]})

    logger.info("--- Demo 1: Simple Chain (LLM Node Only) ---")
    logger.info("  Q: What is Python?")
    logger.info("  A: %s", result["messages"][-1].content[:200])
    logger.info("  Total messages in state: %d", len(result["messages"]))
    logger.info("  This is the FOUNDATION — just a chain, no tools yet.")



# ═══════════════════════════════════════════════════════════════════════════════
# 7. DEMO 2: Tool Binding — Teaching the LLM About Tools
# ═══════════════════════════════════════════════════════════════════════════════
#
# Tools are external functions the LLM can REQUEST to call.
# The LLM doesn't execute tools itself — it returns a STRUCTURED REQUEST
# (tool_calls) that says: "I want to call function X with arguments Y."
#
# HOW IT WORKS:
#   1. Define a Python function with @tool decorator
#   2. Bind it to the LLM: llm.bind_tools([add])
#   3. When the LLM sees a question that matches the tool's description,
#      it returns tool_calls instead of a text answer.
#
# KEY INSIGHT: This demo shows the LLM REQUESTING a tool call.
# It does NOT execute the tool. That's what ToolNode does (Demo 3).
#
# The @tool decorator converts a regular Python function into a LangChain
# Tool object. The docstring becomes the tool's description — the LLM reads
# this to decide WHEN to use the tool.

def demo_tool_binding() -> None:
    """Tool binding: teach the LLM about tools. Show tool_calls structure."""
    from langchain_core.messages import HumanMessage
    from langchain_core.tools import tool
    from langchain_groq import ChatGroq

    # ── Define a simple tool with @tool decorator ────────────────────────
    @tool
    def add(a: int, b: int) -> int:
        """Add two integers together. Use this when the user asks to add numbers.

        Args:
            a: first integer
            b: second integer

        Returns:
            The sum of a and b
        """
        return a + b

    # ── LLM with tool binding ────────────────────────────────────────────
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_tokens=512)
    llm_with_tools = llm.bind_tools([add])  # Teach the LLM about the add tool

    # ── Invoke with a math question — triggers tool_calls ────────────────
    response = llm_with_tools.invoke(
        [HumanMessage(content="What is 2 plus 2?")]
    )

    logger.info("--- Demo 2: Tool Binding (Teaching the LLM About Tools) ---")
    logger.info("  Q: What is 2 plus 2?")
    logger.info("  Content: '%s'", response.content)
    logger.info("  tool_calls: %s", response.tool_calls)

    if response.tool_calls:
        tc = response.tool_calls[0]
        logger.info("  Tool call details:")
        logger.info("    name: %s", tc["name"])
        logger.info("    args: %s", tc["args"])
        logger.info("    id:   %s", tc["id"])
        logger.info("  The LLM REQUESTED a tool call — but did NOT execute it.")
        logger.info("  Execution happens in ToolNode (Demo 3).")
    else:
        logger.info("  No tool calls — LLM answered directly.")

    # ── Invoke with a non-math question — NO tool_calls ──────────────────
    response2 = llm_with_tools.invoke(
        [HumanMessage(content="What is the capital of France?")]
    )

    logger.info("  Q: What is the capital of France?")
    logger.info("  Content: '%s'", response2.content[:150])
    logger.info("  tool_calls: %s", response2.tool_calls)
    logger.info("  No tool needed — LLM answered directly from its knowledge.")



# ═══════════════════════════════════════════════════════════════════════════════
# 8. DEMO 3: Router Pattern — tools_condition
# ═══════════════════════════════════════════════════════════════════════════════
#
# This is the CORE PATTERN of every LangGraph agent.
# The graph has TWO paths:
#   1. LLM returns tool_calls → route to ToolNode → execute tool → END
#   2. LLM returns content (no tool_calls) → route directly to END
#
# HOW tools_condition WORKS:
#   - It's a pre-built router function from langgraph.prebuilt
#   - It inspects the LAST message in state["messages"]
#   - If that message has tool_calls → returns "tools" (route to tools node)
#   - If no tool_calls → returns "__end__" (route to END)
#
# HOW ToolNode WORKS:
#   - It's a pre-built node from langgraph.prebuilt
#   - It receives the tool_calls from the LLM's response
#   - It looks up the correct tool by NAME, calls it with the ARGS
#   - It returns the result as a ToolMessage in the state
#
# ASCII:
#   [START]
#      |
#   [tool_calling_llm]
#      |
#      ├── (tool_calls?) ──→ [tools] ──→ [END]
#      |
#      └── (no tool_calls) ──→ [END]

def demo_router_pattern() -> None:
    """Router pattern: tools_condition routes to ToolNode or END."""
    from typing import Annotated

    from langchain_core.messages import AnyMessage, HumanMessage
    from langchain_core.tools import tool
    from langchain_groq import ChatGroq
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import ToolNode, tools_condition
    from typing_extensions import TypedDict

    # ── State ────────────────────────────────────────────────────────────
    class State(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]

    # ── Tool ─────────────────────────────────────────────────────────────
    @tool
    def add(a: int, b: int) -> int:
        """Add two integers together. Use this when the user asks to add numbers.

        Args:
            a: first integer
            b: second integer

        Returns:
            The sum of a and b
        """
        return a + b

    tools = [add]

    # ── LLM with tools bound ────────────────────────────────────────────
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_tokens=512)
    llm_with_tools = llm.bind_tools(tools)

    # ── Node: calls the LLM (which may produce tool_calls) ──────────────
    def tool_calling_llm(state: State) -> dict:
        """Call the LLM with tools bound. It decides: answer or call a tool."""
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    # ── Build the router graph ───────────────────────────────────────────
    builder = StateGraph(State)
    builder.add_node("tool_calling_llm", tool_calling_llm)  # LLM node
    builder.add_node("tools", ToolNode(tools))               # Tool executor node

    builder.add_edge(START, "tool_calling_llm")              # Entry
    builder.add_conditional_edges(
        "tool_calling_llm",
        tools_condition,  # Router: tool_calls? → "tools" | no tool_calls? → END
    )
    builder.add_edge("tools", END)                           # After tool → END

    graph = builder.compile()
    save_graph_image(graph, "lesson06_router_pattern")

    # ── Test 1: Math question → triggers tool call ───────────────────────
    logger.info("--- Demo 3: Router Pattern (tools_condition) ---")

    result1 = graph.invoke({"messages": [HumanMessage(content="What is 2 plus 2?")]})
    logger.info("  Test 1 — Math question (triggers tool):")
    logger.info("  Q: What is 2 plus 2?")
    for i, msg in enumerate(result1["messages"]):
        logger.info("    [%d] %s: %s", i, type(msg).__name__, str(msg.content)[:120])
    logger.info("  Flow: START → tool_calling_llm → tools → END")

    # ── Test 2: General question → direct answer (no tool) ──────────────
    result2 = graph.invoke(
        {"messages": [HumanMessage(content="What is Python?")]}
    )
    logger.info("  Test 2 — General question (direct answer):")
    logger.info("  Q: What is Python?")
    for i, msg in enumerate(result2["messages"]):
        logger.info("    [%d] %s: %s", i, type(msg).__name__, str(msg.content)[:120])
    logger.info("  Flow: START → tool_calling_llm → END (no tool needed)")



# ═══════════════════════════════════════════════════════════════════════════════
# 9. DEMO 4: Multi-Tool Chatbot (Production Pattern)
# ═══════════════════════════════════════════════════════════════════════════════
#
# In production, agents have MULTIPLE tools — not just one.
# The LLM decides WHICH tool to call based on the user's question.
#
# HOW IT WORKS:
#   1. Define multiple tools (custom add, Wikipedia, DuckDuckGo search)
#   2. Bind ALL tools: llm.bind_tools(tools)
#   3. ToolNode(tools) handles ALL tools — it reads the tool_call name
#      and dispatches to the correct tool automatically.
#
# The graph structure is IDENTICAL to Demo 3 — the only difference is
# the number of tools. ToolNode is smart enough to route to the right one.
#
# This is the PRODUCTION PATTERN for building tool-using agents.

def demo_multi_tool_chatbot() -> None:
    """Multi-tool chatbot: add + Wikipedia + DuckDuckGo search."""
    from typing import Annotated

    from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
    from langchain_community.utilities import WikipediaAPIWrapper
    from langchain_core.messages import AnyMessage, HumanMessage
    from langchain_core.tools import tool
    from langchain_groq import ChatGroq
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import ToolNode, tools_condition
    from typing_extensions import TypedDict

    # ── State ────────────────────────────────────────────────────────────
    class State(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]

    # ── Tool 1: Custom add tool ──────────────────────────────────────────
    @tool
    def add(a: int, b: int) -> int:
        """Add two integers together. Use this when the user asks to add numbers.

        Args:
            a: first integer
            b: second integer

        Returns:
            The sum of a and b
        """
        return a + b

    # ── Tool 2: Wikipedia search ─────────────────────────────────────────
    wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
    wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

    # ── Tool 3: DuckDuckGo web search ────────────────────────────────────
    ddg_search = DuckDuckGoSearchRun()

    # ── Combine all tools ────────────────────────────────────────────────
    tools = [add, wiki, ddg_search]

    # ── LLM with ALL tools bound ─────────────────────────────────────────
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_tokens=512)
    llm_with_tools = llm.bind_tools(tools)

    # ── Node: LLM decides which tool (if any) to call ───────────────────
    def tool_calling_llm(state: State) -> dict:
        """Call the LLM with all tools bound. It picks the right tool."""
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    # ── Build the multi-tool router graph ────────────────────────────────
    builder = StateGraph(State)
    builder.add_node("tool_calling_llm", tool_calling_llm)
    builder.add_node("tools", ToolNode(tools))  # Handles ALL tools

    builder.add_edge(START, "tool_calling_llm")
    builder.add_conditional_edges(
        "tool_calling_llm",
        tools_condition,
    )
    builder.add_edge("tools", END)

    graph = builder.compile()
    save_graph_image(graph, "lesson06_multi_tool_chatbot")

    logger.info("--- Demo 4: Multi-Tool Chatbot (Production Pattern) ---")

    # ── Test 1: Math question → add tool ─────────────────────────────────
    result1 = graph.invoke({"messages": [HumanMessage(content="What is 15 plus 27?")]})
    logger.info("  Test 1 — Math question:")
    logger.info("  Q: What is 15 plus 27?")
    for i, msg in enumerate(result1["messages"]):
        logger.info("    [%d] %s: %s", i, type(msg).__name__, str(msg.content)[:150])
    # Identify which tool was called
    ai_msg = result1["messages"][1] if len(result1["messages"]) > 1 else None
    if ai_msg and hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls:
        logger.info("  Tool called: %s", ai_msg.tool_calls[0]["name"])

    # ── Test 2: Knowledge question → Wikipedia tool ──────────────────────
    result2 = graph.invoke(
        {"messages": [HumanMessage(content="What is machine learning?")]}
    )
    logger.info("  Test 2 — Knowledge question:")
    logger.info("  Q: What is machine learning?")
    for i, msg in enumerate(result2["messages"]):
        logger.info("    [%d] %s: %s", i, type(msg).__name__, str(msg.content)[:150])
    ai_msg2 = result2["messages"][1] if len(result2["messages"]) > 1 else None
    if ai_msg2 and hasattr(ai_msg2, "tool_calls") and ai_msg2.tool_calls:
        logger.info("  Tool called: %s", ai_msg2.tool_calls[0]["name"])

    # ── Test 3: Current events → DuckDuckGo search tool ──────────────────
    result3 = graph.invoke(
        {"messages": [HumanMessage(content="Search for latest AI news 2025")]}
    )
    logger.info("  Test 3 — Web search question:")
    logger.info("  Q: Search for latest AI news 2025")
    for i, msg in enumerate(result3["messages"]):
        logger.info("    [%d] %s: %s", i, type(msg).__name__, str(msg.content)[:150])
    ai_msg3 = result3["messages"][1] if len(result3["messages"]) > 1 else None
    if ai_msg3 and hasattr(ai_msg3, "tool_calls") and ai_msg3.tool_calls:
        logger.info("  Tool called: %s", ai_msg3.tool_calls[0]["name"])

    logger.info("  ToolNode dispatches to the correct tool based on tool_call name.")
    logger.info("  Same graph structure — just more tools in the list!")



# ═══════════════════════════════════════════════════════════════════════════════
# 10. DEMO 5: Custom Tool + RAG Retriever as a Tool (Advanced)
# ═══════════════════════════════════════════════════════════════════════════════
#
# This is the ADVANCED pattern: RAG as just another tool in the agent's toolkit.
#
# Instead of building a separate RAG pipeline, you wrap the retriever as a @tool.
# The LLM decides WHEN to search the document — just like it decides when to
# call the add tool or Wikipedia.
#
# HOW IT WORKS:
#   1. Load speech.txt → split into chunks → embed with HuggingFace → store in FAISS
#   2. Wrap the retriever as a @tool function (search_speech_document)
#   3. Combine with other tools (add, Wikipedia)
#   4. Build the same router graph — the LLM picks the right tool
#
# WHY THIS MATTERS:
#   In production, RAG is NOT a separate system — it's a TOOL the agent can use.
#   The agent decides: "Do I need to search the document, or can I answer from
#   my own knowledge?" This is how real-world agents work.

def demo_rag_as_tool() -> None:
    """RAG retriever as a tool: speech.txt search + add + Wikipedia."""
    from typing import Annotated

    from langchain_community.document_loaders import TextLoader
    from langchain_community.tools import WikipediaQueryRun
    from langchain_community.utilities import WikipediaAPIWrapper
    from langchain_community.vectorstores import FAISS
    from langchain_core.messages import AnyMessage, HumanMessage
    from langchain_core.tools import tool
    from langchain_groq import ChatGroq
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import ToolNode, tools_condition
    from typing_extensions import TypedDict

    # ── State ────────────────────────────────────────────────────────────
    class State(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]

    # ── Load and index speech.txt ────────────────────────────────────────
    speech_path = os.path.join(
        os.path.dirname(__file__),
        "..", "LangchainBasics", "DataIngestion", "data", "speech.txt"
    )
    loader = TextLoader(speech_path, encoding="utf-8")
    documents = loader.load()

    # Split into chunks for retrieval
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    # Embed and store in FAISS
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    logger.info("  Indexed %d chunks from speech.txt into FAISS.", len(chunks))

    # ── Tool 1: RAG retriever wrapped as a tool ──────────────────────────
    @tool
    def search_speech_document(query: str) -> str:
        """Search the speech document for relevant information.
        Use this when the user asks about democracy, war, freedom, or the speech.

        Args:
            query: the search query about the speech content

        Returns:
            Relevant passages from the speech document
        """
        docs = retriever.invoke(query)
        return "\n\n".join(doc.page_content for doc in docs)

    # ── Tool 2: Custom add tool ──────────────────────────────────────────
    @tool
    def add(a: int, b: int) -> int:
        """Add two integers together. Use this when the user asks to add numbers.

        Args:
            a: first integer
            b: second integer

        Returns:
            The sum of a and b
        """
        return a + b

    # ── Tool 3: Wikipedia ────────────────────────────────────────────────
    wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
    wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

    # ── Combine all tools ────────────────────────────────────────────────
    tools = [search_speech_document, add, wiki]

    # ── LLM with ALL tools bound ─────────────────────────────────────────
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_tokens=512)
    llm_with_tools = llm.bind_tools(tools)

    # ── Node ─────────────────────────────────────────────────────────────
    def tool_calling_llm(state: State) -> dict:
        """Call the LLM with RAG + other tools bound."""
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    # ── Build the graph ──────────────────────────────────────────────────
    builder = StateGraph(State)
    builder.add_node("tool_calling_llm", tool_calling_llm)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "tool_calling_llm")
    builder.add_conditional_edges(
        "tool_calling_llm",
        tools_condition,
    )
    builder.add_edge("tools", END)

    graph = builder.compile()
    save_graph_image(graph, "lesson06_rag_as_tool")

    logger.info("--- Demo 5: RAG Retriever as a Tool (Advanced) ---")

    # ── Test: Ask about the speech → triggers search_speech_document ─────
    result = graph.invoke(
        {"messages": [HumanMessage(content="What does the speaker say about democracy?")]}
    )
    logger.info("  Q: What does the speaker say about democracy?")
    for i, msg in enumerate(result["messages"]):
        logger.info("    [%d] %s: %s", i, type(msg).__name__, str(msg.content)[:200])
    ai_msg = result["messages"][1] if len(result["messages"]) > 1 else None
    if ai_msg and hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls:
        logger.info("  Tool called: %s", ai_msg.tool_calls[0]["name"])

    logger.info("  RAG is just another tool — the LLM decides when to use it.")
    logger.info("  This is how production agents integrate document search.")



# ═══════════════════════════════════════════════════════════════════════════════
# 11. COMPARISON TABLE: Chain vs Router vs Multi-Tool Agent
# ═══════════════════════════════════════════════════════════════════════════════
#
# Pattern          | Tools? | Routing?  | Graph Structure              | Use Case
# ─────────────────|────────|───────────|──────────────────────────────|──────────────────────
# Simple Chain     | No     | No        | START → LLM → END            | Basic chatbot, Q&A
# Router (1 tool)  | Yes    | Yes       | START → LLM → (cond) → T/END | Calculator, single API
# Multi-Tool Agent | Yes    | Yes       | START → LLM → (cond) → T/END | Production agents
# RAG-as-Tool      | Yes    | Yes       | START → LLM → (cond) → T/END | Document Q&A + tools
#
# KEY INSIGHT:
#   The graph STRUCTURE is the same for Router, Multi-Tool, and RAG-as-Tool.
#   The only difference is the NUMBER and TYPE of tools in the list.
#   ToolNode handles dispatching to the correct tool automatically.
#
# PROGRESSION:
#   Simple Chain → add tools → add routing → add more tools → production agent
#   Each step builds on the previous one. No magic. Just more tools.


# ═══════════════════════════════════════════════════════════════════════════════
# 12. COMMON MISTAKES & GOTCHAS
# ═══════════════════════════════════════════════════════════════════════════════
#
# Mistake                                    | Fix
# ───────────────────────────────────────────|──────────────────────────────────────
# Forgetting to bind tools to LLM           | Always call llm.bind_tools(tools)
# Using llm instead of llm_with_tools       | The unbound LLM can't make tool calls
# Missing @tool decorator                   | Without it, the function isn't a Tool
# Bad tool docstring                        | LLM reads the docstring to decide WHEN
#                                            | to use the tool — make it descriptive!
# Not wrapping tools in ToolNode            | ToolNode executes tools; raw functions don't
# Forgetting add_messages reducer           | Messages overwrite instead of append
# Not handling ToolException in tool nodes  | Unhandled errors crash the agent
# Passing tools list to ToolNode but not    | Both must have the SAME tools list
#   to llm.bind_tools()                     |
#
# ANTI-PATTERN: Building a separate RAG pipeline instead of wrapping it as a tool.
#   In LangGraph, RAG should be a tool the agent can choose to use.
#   This gives the agent flexibility to answer from knowledge OR search documents.


# ═══════════════════════════════════════════════════════════════════════════════
# 13. WHY LANGGRAPH FOR THIS
# ═══════════════════════════════════════════════════════════════════════════════
#
# ✅ Explicit routing — you SEE the decision logic (tools_condition)
# ✅ Composable tools — add/remove tools without changing graph structure
# ✅ Stateful by design — messages accumulate via add_messages reducer
# ✅ Debuggable — save graph images, log every step
# ✅ Production-ready — same pattern scales from 1 tool to 100 tools
# ✅ RAG as a tool — document search is just another tool in the toolkit
# ✅ Built-in streaming — stream tokens AND intermediate tool results
# ✅ Framework-agnostic nodes — any Python function can be a node


# ═══════════════════════════════════════════════════════════════════════════════
# 14. WHERE THIS CONNECTS (Concept Linking)
# ═══════════════════════════════════════════════════════════════════════════════
#
# @tool decorator → creates a LangChain Tool → bound to LLM via bind_tools()
# llm.bind_tools(tools) → teaches the LLM about available tools
# LLM response → may contain tool_calls (structured requests)
# tools_condition → reads tool_calls → routes to "tools" or END
# ToolNode(tools) → executes the requested tool → returns ToolMessage
# add_messages reducer → appends all messages (Human, AI, Tool) to state
#
# Lesson 04 (Reducers) → add_messages keeps the full conversation
# Lesson 05 (State Schemas) → TypedDict defines the state shape
# Lesson 06 (This) → Chains + Tools + Routing = Agent foundation
# Next lessons → Memory (checkpointing), Human-in-the-loop, Multi-agent
#
# StateGraph → compiles to → CompiledGraph
# Nodes → read/write → State
# Conditional Edges → use → tools_condition → return → node names
# ToolNode → wraps → LangChain tools → used inside → Agent loop


# ═══════════════════════════════════════════════════════════════════════════════
# 🎯 INTERVIEW QUESTIONS
# ═══════════════════════════════════════════════════════════════════════════════
#
# THEORETICAL:
# Q: What is the difference between tools_condition and a custom router function
#    in LangGraph? When would you use each?
#
# A: tools_condition is a pre-built router from langgraph.prebuilt that checks
#    if the LLM's last message contains tool_calls. If yes, it routes to the
#    "tools" node; if no, it routes to END. It's designed specifically for the
#    tool-calling pattern. A custom router function is any Python function that
#    reads the state and returns a node name (string). You'd use tools_condition
#    for standard tool-calling agents, and a custom router when you need more
#    complex routing logic — e.g., routing based on message content, sentiment,
#    or custom state fields beyond just tool_calls.
#
# HANDS-ON:
# Q: Build a LangGraph agent with two custom tools:
#    - multiply(a, b) → returns a * b
#    - greet(name) → returns "Hello, {name}!"
#    The agent should route to the correct tool based on the user's question.
#    Test with "What is 3 times 7?" and "Greet Alice".
#
# SOLUTION:
#   from typing import Annotated
#   from langchain_core.messages import AnyMessage, HumanMessage
#   from langchain_core.tools import tool
#   from langchain_groq import ChatGroq
#   from langgraph.graph import END, START, StateGraph
#   from langgraph.graph.message import add_messages
#   from langgraph.prebuilt import ToolNode, tools_condition
#   from typing_extensions import TypedDict
#
#   class State(TypedDict):
#       messages: Annotated[list[AnyMessage], add_messages]
#
#   @tool
#   def multiply(a: int, b: int) -> int:
#       """Multiply two integers. Use when the user asks to multiply numbers."""
#       return a * b
#
#   @tool
#   def greet(name: str) -> str:
#       """Greet a person by name. Use when the user asks to greet someone."""
#       return f"Hello, {name}!"
#
#   tools = [multiply, greet]
#   llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_tokens=512)
#   llm_with_tools = llm.bind_tools(tools)
#
#   def tool_calling_llm(state: State) -> dict:
#       return {"messages": [llm_with_tools.invoke(state["messages"])]}
#
#   builder = StateGraph(State)
#   builder.add_node("tool_calling_llm", tool_calling_llm)
#   builder.add_node("tools", ToolNode(tools))
#   builder.add_edge(START, "tool_calling_llm")
#   builder.add_conditional_edges("tool_calling_llm", tools_condition)
#   builder.add_edge("tools", END)
#   graph = builder.compile()
#
#   # Test 1: multiply
#   r1 = graph.invoke({"messages": [HumanMessage(content="What is 3 times 7?")]})
#   print(r1["messages"][-1].content)  # 21
#
#   # Test 2: greet
#   r2 = graph.invoke({"messages": [HumanMessage(content="Greet Alice")]})
#   print(r2["messages"][-1].content)  # Hello, Alice!


# ═══════════════════════════════════════════════════════════════════════════════
# 📝 QUICK RECAP
# ═══════════════════════════════════════════════════════════════════════════════
#
# 1. A Chain is nodes + edges. Add tools via @tool + llm.bind_tools().
#    Add routing via tools_condition. That's the entire agent pattern.
#
# 2. tools_condition checks for tool_calls in the LLM's response.
#    ToolNode executes the tool and returns the result as a ToolMessage.
#    The graph structure is IDENTICAL whether you have 1 tool or 100.
#
# 3. RAG is just another tool. Wrap your retriever with @tool and add it
#    to the tools list. The LLM decides when to search vs answer directly.


# ═══════════════════════════════════════════════════════════════════════════════
# 🚀 Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("🔗 LANGGRAPH LESSON 6 — Chains, Tools & Routing")
    logger.info("   From Simple Chain to Multi-Tool Agent")
    logger.info("=" * 70)

    logger.info("\n🔹 Demo 1: Simple Chain (LLM Node Only)")
    demo_simple_chain()

    logger.info("\n🔹 Demo 2: Tool Binding (Teaching the LLM About Tools)")
    demo_tool_binding()

    logger.info("\n🔹 Demo 3: Router Pattern (tools_condition)")
    demo_router_pattern()

    logger.info("\n🔹 Demo 4: Multi-Tool Chatbot (Production Pattern)")
    demo_multi_tool_chatbot()

    logger.info("\n🔹 Demo 5: RAG Retriever as a Tool (Advanced)")
    demo_rag_as_tool()

    logger.info("\n" + "=" * 70)
    logger.info("✅ Lesson 6 complete — Chains, Tools & Routing")
    logger.info("Key: Chain → Tool Binding → Router → Multi-Tool Agent")
    logger.info("Pattern: tools_condition routes, ToolNode executes")
    logger.info("Advanced: RAG is just another tool in the agent's toolkit")
    logger.info("Next: Lesson 7 — Memory & Checkpointing")
    logger.info("=" * 70)
