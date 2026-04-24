"""
DocSage Grade Node — Grades retrieved documents for relevance.

WHY THIS NODE EXISTS:
    After the retriever tool returns documents, we need to check:
    "Are these documents actually relevant to the question?"
    This is the QUALITY GATE that prevents hallucination from bad context.

    Without grading, the LLM might receive irrelevant chunks and
    generate a plausible but WRONG answer. The grader catches this.

INTERVIEW POINT:
    "How do you prevent hallucination in your RAG pipeline?"
    "After retrieval, each document is graded individually for relevance
    using structured output. The grading LLM returns a Pydantic model
    with a binary yes/no score. Irrelevant documents are filtered out
    before generation. If ALL documents are irrelevant, the query is
    rewritten and retrieval is retried. This is the Corrective RAG
    pattern — grade first, correct if needed."

HOW STRUCTURED OUTPUT WORKS:
    llm.with_structured_output(GradeDocuments) wraps the LLM so it
    ALWAYS returns a GradeDocuments Pydantic object with binary_score.
    The LLM is forced to pick "yes" or "no" — no free-form text.
    This makes grading deterministic and parseable.

WHERE THIS NODE SITS IN THE GRAPH:
    [tools (ToolNode)] -> [grade_documents] -> generate or rewrite
    It's a CONDITIONAL EDGE source — its return value determines
    whether we proceed to generation or self-correction.
"""

import logging
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from docsage.state.state import AgentState

logger = logging.getLogger(__name__)


def create_grade_node(llm: ChatGroq) -> callable:
    """
    Create the document grading node (conditional edge function).

    This node grades the retrieved documents and returns a routing
    decision: "generate" if docs are relevant, "rewrite" if not.

    Args:
        llm: The ChatGroq LLM for grading (from GroqLLMFactory.create_grading_llm).

    Returns:
        A conditional edge function: (AgentState) -> Literal["generate", "rewrite"]
    """

    # Pydantic model for structured grading output.
    # Forces the LLM to return exactly "yes" or "no".
    class GradeDocuments(BaseModel):
        """Binary relevance score for a retrieved document."""
        binary_score: Literal["yes", "no"] = Field(
            description="Is the document relevant to the question? 'yes' or 'no'"
        )

    # Wrap the LLM with structured output
    structured_grader = llm.with_structured_output(GradeDocuments)

    # Grading prompt — instructs the LLM on how to assess relevance
    grade_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a grader assessing relevance of a retrieved document to a user question. "
            "If the document contains keyword(s) or semantic meaning related to the question, "
            "grade it as relevant. Give a binary score 'yes' or 'no'.",
        ),
        (
            "human",
            "Retrieved document: \n\n {document} \n\n User question: {question}",
        ),
    ])

    grading_chain = grade_prompt | structured_grader

    def grade_documents(state: AgentState) -> Literal["generate", "rewrite", "done"]:
        """
        Grade retrieved documents for relevance.

        Also handles non-retrieval tool results (like send_email).
        If the last tool result is from the email tool, skip grading
        and return "done" — the email was sent, nothing to grade.

        Returns:
            "generate" if documents are relevant.
            "rewrite" if documents are not relevant.
            "done" if the tool result is not a retrieval (e.g., email sent).
        """
        logger.info("  [grader] Checking document relevance...")

        messages = state["messages"]

        # Check if the last ToolMessage is from a non-retrieval tool (email, etc.)
        # If so, skip grading — the action is already complete.
        last_tool_name = ""
        last_tool_content = ""
        for msg in reversed(messages):
            if hasattr(msg, "type") and msg.type == "tool":
                last_tool_name = getattr(msg, "name", "")
                last_tool_content = msg.content
                break

        if last_tool_name == "send_email":
            logger.info("  [grader] Email tool result — skipping grading -> done")
            return "done"

        # Find the LAST HumanMessage — that's the current question
        question = ""
        for msg in reversed(messages):
            if hasattr(msg, "type") and msg.type == "human":
                question = msg.content
                break
        if not question and messages:
            question = messages[0].content

        docs = last_tool_content or ""

        try:
            result = grading_chain.invoke(
                {"question": question, "document": docs}
            )
            score = result.binary_score
        except Exception as e:
            # If grading fails (Groq structured output can be flaky),
            # default to "yes" to avoid blocking the pipeline.
            logger.warning(
                "  [grader] Grading failed: %s. Defaulting to 'yes'.",
                str(e)[:80],
            )
            score = "yes"

        if score == "yes":
            logger.info("  [grader] Documents are RELEVANT -> generate")
            return "generate"
        else:
            logger.info("  [grader] Documents NOT relevant -> rewrite query")
            return "rewrite"

    return grade_documents
