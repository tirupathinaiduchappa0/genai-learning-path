"""
DocSage Rewrite Node — Rewrites the query for better retrieval.

WHY THIS NODE EXISTS:
    When the grader determines that retrieved documents are NOT relevant,
    the original query may be poorly worded for the knowledge base.
    This node rewrites the query to improve retrieval on the next attempt.

    The rewritten query goes back to the agent node, which tries again
    with the improved question. This is the SELF-CORRECTION LOOP.

INTERVIEW POINT:
    "How does your system handle poor retrieval results?"
    "If the grader finds that retrieved documents are irrelevant, the
    rewrite node transforms the query. It asks the LLM to reason about
    the underlying semantic intent and produce a better question. The
    improved query goes back to the agent, which retries retrieval.
    This self-correction loop runs until relevant docs are found or
    the recursion limit is hit."

WHERE THIS NODE SITS IN THE GRAPH:
    [grade_documents] --(not relevant)--> [rewrite] --> [agent] (loop back)
    It's the self-correction mechanism that makes the system robust.
"""

import logging

from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from docsage.state.state import AgentState

logger = logging.getLogger(__name__)


def create_rewrite_node(llm: ChatGroq) -> callable:
    """
    Create the query rewrite node for self-correction.

    Args:
        llm: The ChatGroq LLM for rewriting (from GroqLLMFactory.create_rewrite_llm).

    Returns:
        A node function: (AgentState) -> dict
    """

    def rewrite_node(state: AgentState) -> dict:
        """
        Rewrite the user's query for better retrieval results.

        Takes the original question, asks the LLM to reason about
        the semantic intent, and produces an improved version.
        The rewritten query replaces the original in the message
        history so the agent retries with the better question.

        Args:
            state: Current graph state with messages.

        Returns:
            Dict with the rewritten query as a new HumanMessage.
        """
        logger.info("  [rewrite] Transforming query for better retrieval...")

        messages = state["messages"]
        question = messages[0].content

        # Ask the LLM to rewrite the query
        response = llm.invoke(
            [HumanMessage(content=(
                f"You are a query optimizer. Look at this question and reason "
                f"about the underlying semantic intent. Rewrite it to be more "
                f"specific and likely to retrieve relevant documents from a "
                f"knowledge base.\n\n"
                f"Original question: {question}\n\n"
                f"Rewritten question (just the question, nothing else):"
            ))]
        )

        rewritten = response.content.strip()
        logger.info("  [rewrite] Original: '%s'", question[:80])
        logger.info("  [rewrite] Rewritten: '%s'", rewritten[:80])

        # Return as a new HumanMessage so the agent sees the improved query
        return {"messages": [HumanMessage(content=rewritten)]}

    return rewrite_node
