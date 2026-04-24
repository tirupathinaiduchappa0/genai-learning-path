"""
DocSage Graph Builder — Assembles the full Agentic + Adaptive RAG workflow.

WHY THIS FILE EXISTS:
    This is the ORCHESTRATOR that wires all nodes, edges, and conditional
    routing into a single LangGraph StateGraph. It takes the LLM factory
    and tools, creates all node functions, registers them in the graph,
    defines the edges (fixed + conditional), and compiles the graph.

    Separation of concerns:
    - Nodes (Step 3) define WHAT each step does.
    - Graph Builder (this file) defines HOW steps connect.
    - You can change the workflow (add/remove nodes, change routing)
      without touching any node logic.

INTERVIEW POINT:
    "Walk me through your LangGraph workflow."
    "The graph has 5 nodes: agent, tools, grade, generate, rewrite,
    plus a validate conditional edge after generation. The agent decides
    which retriever tool to call. The ToolNode executes the retrieval.
    The grader checks document relevance — if relevant, we generate;
    if not, we rewrite the query and loop back to the agent. After
    generation, the validator checks for hallucination and answer
    relevance. If either fails, we self-correct. The graph compiles
    with MemorySaver for conversation persistence."

THE GRAPH FLOW:

    [START]
       |
    [agent]  <-- LLM with bind_tools (PDF retrievers + web search)
       |
       +-- (tool_call) --> [tools]  <-- ToolNode executes chosen retriever
       |                      |
       |                (grade_documents)  <-- Conditional edge
       |                      |
       |                      +-- ("generate") --> [generate]
       |                      |                        |
       |                      |                  (validate_answer)  <-- Conditional edge
       |                      |                        |
       |                      |                 +------+------+
       |                      |                 |      |      |
       |                      |              "useful" "not   "not
       |                      |                 |    supported" useful"
       |                      |                 |      |      |
       |                      |               [END] [generate][rewrite]
       |                      |                     (retry)  (full retry)
       |                      |
       |                      +-- ("rewrite") --> [rewrite] --> [agent] (loop)
       |
       +-- (no tool_call) --> [END]  <-- Agent answers directly
"""

import logging
from typing import Optional

from langchain_core.tools import BaseTool
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from docsage.config import settings
from docsage.llms.groq_llm import GroqLLMFactory
from docsage.nodes.agent_node import create_agent_node
from docsage.nodes.generate_node import create_generate_node
from docsage.nodes.grade_node import create_grade_node
from docsage.nodes.rewrite_node import create_rewrite_node
from docsage.nodes.validate_node import create_validate_node
from docsage.state.state import AgentState

logger = logging.getLogger(__name__)


class GraphBuilder:
    """
    Builds and compiles the DocSage LangGraph workflow.

    This class takes a configured LLM factory and a list of tools,
    creates all node functions, wires them into a StateGraph with
    conditional edges, and compiles with a MemorySaver checkpointer.

    Usage:
        factory = GroqLLMFactory(api_key="gsk_...")
        tools = [pdf_retriever_tool, web_search_tool]
        builder = GraphBuilder(llm_factory=factory, tools=tools)
        graph = builder.build()
        result = graph.invoke({"messages": [HumanMessage(content="...")]})
    """

    def __init__(
        self,
        llm_factory: GroqLLMFactory,
        tools: list[BaseTool],
    ) -> None:
        """
        Initialize the graph builder.

        Args:
            llm_factory: Factory for creating task-specific LLM instances.
            tools: List of LangChain tools (retriever tools + web search).
                   These are bound to the agent LLM via bind_tools().
        """
        self.llm_factory = llm_factory
        self.tools = tools
        logger.info(
            "GraphBuilder initialized with %d tools: %s",
            len(tools),
            [t.name for t in tools],
        )

    def build(self) -> "CompiledGraph":
        """
        Build and compile the full DocSage graph.

        Creates all LLM instances, node functions, registers them
        in the StateGraph, defines edges, and compiles with MemorySaver.

        Returns:
            A compiled LangGraph ready for invoke() or stream().
        """
        logger.info("Building DocSage graph...")

        # ==================================================================
        # STEP 1: Create task-specific LLM instances
        # ==================================================================
        # Each node gets an LLM optimized for its task.
        # The factory handles model selection and configuration.
        agent_llm = self.llm_factory.create_agent_llm()
        grading_llm = self.llm_factory.create_grading_llm()
        generation_llm = self.llm_factory.create_generation_llm()
        rewrite_llm = self.llm_factory.create_rewrite_llm()

        # ==================================================================
        # STEP 2: Create node functions using factories
        # ==================================================================
        agent_fn = create_agent_node(agent_llm, self.tools)

        # ==================================================================
        # STEP 3: Build the StateGraph
        # ==================================================================
        builder = StateGraph(AgentState)

        if self.tools:
            # FULL GRAPH: agent + tools + grade + generate + rewrite + validate
            # This is the complete Agentic + Adaptive RAG workflow.
            grade_fn = create_grade_node(grading_llm)
            generate_fn = create_generate_node(generation_llm)
            rewrite_fn = create_rewrite_node(rewrite_llm)
            validate_fn = create_validate_node(grading_llm)
            tool_node = ToolNode(self.tools)

            builder.add_node("agent", agent_fn)
            builder.add_node("tools", tool_node)
            builder.add_node("generate", generate_fn)
            builder.add_node("rewrite", rewrite_fn)

            builder.add_edge(START, "agent")
            builder.add_conditional_edges(
                "agent", tools_condition,
                {"tools": "tools", END: END},
            )
            builder.add_conditional_edges(
                "tools", grade_fn,
                {"generate": "generate", "rewrite": "rewrite", "done": END},
            )
            builder.add_conditional_edges(
                "generate", validate_fn,
                {"useful": END, "not supported": "generate", "not useful": "rewrite"},
            )
            builder.add_edge("rewrite", "agent")

            logger.info("  Full RAG graph built (agent + tools + grade + generate + validate)")
        else:
            # SIMPLE GRAPH: agent only (no tools, no retrieval).
            # The agent answers directly from its own knowledge.
            # This is used when no documents are uploaded and no web search key.
            builder.add_node("agent", agent_fn)
            builder.add_edge(START, "agent")
            builder.add_edge("agent", END)

            logger.info("  Simple graph built (agent only, no tools)")

        # ==================================================================
        # STEP 4: Compile with MemorySaver
        # ==================================================================
        # MemorySaver enables conversation memory across turns.
        # Each conversation gets a unique thread_id.
        # Without this, each invoke() would be independent.
        memory = MemorySaver()
        graph = builder.compile(checkpointer=memory)

        logger.info("DocSage graph compiled successfully.")
        logger.info(
            "  Nodes: agent, tools, generate, rewrite"
        )
        logger.info(
            "  Conditional edges: tools_condition, grade_documents, validate_answer"
        )
        logger.info("  Checkpointer: MemorySaver (conversation memory enabled)")

        return graph
