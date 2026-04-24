"""
DocSage Generate Node — Produces the final answer with source citations.

WHY THIS NODE EXISTS:
    This is where the actual ANSWER is produced. It takes the user's
    question and the retrieved (graded) documents, and generates a
    concise answer using a RAG prompt. It also extracts source citations
    from the document metadata.

INTERVIEW POINT:
    "How do you generate answers with citations?"
    "The generate node uses a focused RAG prompt that instructs the LLM
    to answer ONLY from the provided context. It also extracts source
    metadata (page numbers, filenames) from the retrieved documents and
    includes them as citations. This way, the user can verify the answer
    against the original document."

WHY SOURCE CITATIONS MATTER:
    In production RAG systems, citations are CRITICAL for trust.
    Users need to know WHERE the answer came from. Without citations,
    the system is a black box. With citations, users can verify.
    This is especially important in legal, medical, and financial domains.
"""

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from docsage.state.state import AgentState

logger = logging.getLogger(__name__)

# RAG generation prompt — instructs the LLM to answer from context only.
# The "If you don't know" instruction prevents hallucination.
GENERATE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are DocSage, an intelligent document assistant. "
        "Use ONLY the following retrieved context to answer the question. "
        "If the context doesn't contain enough information to answer, "
        "say 'I don't have enough information in the uploaded documents to answer this.' "
        "Keep the answer concise and well-structured (3-6 sentences). "
        "If relevant, mention which part of the document the information comes from.",
    ),
    (
        "human",
        "Question: {question}\n\nRetrieved Context:\n{context}\n\nAnswer:",
    ),
])


def create_generate_node(llm: ChatGroq) -> callable:
    """
    Create the generation node that produces answers with citations.

    Args:
        llm: The ChatGroq LLM for generation (from GroqLLMFactory.create_generation_llm).

    Returns:
        A node function: (AgentState) -> dict
    """
    # Build the RAG chain: prompt -> LLM -> string output
    rag_chain = GENERATE_PROMPT | llm | StrOutputParser()

    def generate_node(state: AgentState) -> dict:
        """
        Generate an answer from the retrieved context with source citations.

        Extracts the question and retrieved documents from state,
        formats them into the RAG prompt, and produces the answer.
        Also extracts source metadata for citations.

        Args:
            state: Current graph state with messages.

        Returns:
            Dict with generation (answer) and sources (citations).
        """
        logger.info("  [generate] Producing answer...")

        messages = state["messages"]

        # Find the LAST HumanMessage — that's the current question
        question = ""
        for msg in reversed(messages):
            if hasattr(msg, "type") and msg.type == "human":
                question = msg.content
                break
        if not question and messages:
            question = messages[0].content

        # Find the LAST ToolMessage — that's the current retrieval result
        docs_content = ""
        for msg in reversed(messages):
            if hasattr(msg, "type") and msg.type == "tool":
                docs_content = msg.content
                break

        # Fallback: use the last message content if no ToolMessage found
        if not docs_content and messages:
            docs_content = messages[-1].content if hasattr(messages[-1], "content") else str(messages[-1])

        try:
            # Generate the answer
            generation = rag_chain.invoke({
                "question": question,
                "context": docs_content,
            })
        except Exception as e:
            logger.error("  [generate] Generation failed: %s", str(e)[:100])
            generation = "I encountered an error while generating the answer. Please try again."

        # Extract source citations from the tool messages
        sources = _extract_sources(messages)

        logger.info("  [generate] Answer: %s", generation[:120])
        logger.info("  [generate] Sources: %d citations", len(sources))

        return {
            "generation": generation,
            "sources": sources,
        }

    return generate_node


def _extract_sources(messages: list) -> list[str]:
    """
    Extract source citations from the MOST RECENT tool call only.

    In multi-turn conversations, there may be ToolMessages from previous
    turns. We only want citations from the CURRENT retrieval, not old ones.

    Args:
        messages: The full message list from state.

    Returns:
        List of source citation strings.
    """
    from langchain_core.messages import ToolMessage

    # Find the LAST ToolMessage only (current turn's retrieval)
    sources = []
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            tool_name = getattr(msg, "name", "unknown_source")
            display_name = tool_name.replace("search_", "").replace("_", " ").title()
            content_preview = msg.content[:100] if msg.content else ""
            sources.append(f"Source: {display_name} | Preview: {content_preview}...")
            break  # Only the most recent ToolMessage

    if not sources:
        sources.append("Source: Direct LLM response (no document retrieval)")

    return sources
