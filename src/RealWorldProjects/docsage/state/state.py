"""
DocSage State — The shared state definition for the LangGraph workflow.

WHY THIS FILE EXISTS:
    Every LangGraph workflow needs a TypedDict that defines the SHAPE
    of the data flowing through the graph. All nodes read from and
    write to this shared state. It's the single source of truth.

INTERVIEW POINT:
    "How do you define state in your LangGraph application?"
    "I use a TypedDict called AgentState. It holds the conversation
    messages (with add_messages reducer for append-only behavior),
    the retrieved documents, the generated answer, and metadata like
    source citations. The add_messages reducer is critical — without
    it, each node would OVERWRITE the messages instead of appending."

WHY add_messages REDUCER:
    Without a reducer, state["messages"] = [new_msg] REPLACES the list.
    With Annotated[list, add_messages], it APPENDS new_msg to the existing list.
    This preserves the full conversation history through the graph.

    This is one of the most common LangGraph mistakes:
    Forgetting the reducer -> messages get overwritten -> agent loses context.
"""

from typing import Annotated, Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """
    Shared state for the DocSage Agentic RAG workflow.

    Every node in the graph reads from and writes to this state.
    The add_messages reducer ensures messages APPEND, not overwrite.

    Attributes:
        messages: The conversation history (user messages, AI responses,
                  tool calls, tool results). Uses add_messages reducer
                  so each node appends to the list instead of replacing it.

        documents: Retrieved documents from the vector store or web search.
                   Each node can read/update this list. The grading node
                   filters it down to only relevant documents.

        generation: The final generated answer from the RAG pipeline.
                    Set by the generate node, read by the validate node.

        sources: Source citations extracted from retrieved documents.
                 Contains page numbers, chunk metadata, or URLs.
                 Displayed in the UI alongside the answer.

        web_search_used: Flag indicating whether web search was triggered.
                         "Yes" if the agent called the web search tool,
                         "No" if it only used local document retrieval.
    """

    messages: Annotated[Sequence[BaseMessage], add_messages]
    documents: list[str]
    generation: str
    sources: list[str]
    web_search_used: str
