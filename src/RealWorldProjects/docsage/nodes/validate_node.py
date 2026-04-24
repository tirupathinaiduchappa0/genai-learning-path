"""
DocSage Validate Node — Post-generation quality checks.

WHY THIS NODE EXISTS:
    Even with relevant documents, the LLM can still:
    1. HALLUCINATE — generate info not in the documents.
    2. MISS THE POINT — produce a grounded answer that doesn't
       actually address the user's question.

    This node performs TWO checks after generation:
    - Hallucination check: "Is the answer grounded in the documents?"
    - Answer relevance check: "Does the answer address the question?"

    This is the ADAPTIVE RAG pattern — post-generation validation
    with a self-correction loop.

INTERVIEW POINT:
    "How do you validate the quality of generated answers?"
    "After generation, I run two checks using structured output.
    First, a hallucination grader checks if the answer is grounded
    in the retrieved documents. Second, an answer grader checks if
    the answer actually addresses the user's question. If either
    check fails, the query is rewritten and the full pipeline retries.
    This double validation is what makes Adaptive RAG more robust
    than basic RAG."

WHERE THIS NODE SITS IN THE GRAPH:
    [generate] --> [validate] --> "useful" (END) or "not supported" (retry generate)
                               or "not useful" (rewrite + full retry)
"""

import logging
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from docsage.state.state import AgentState

logger = logging.getLogger(__name__)


def create_validate_node(llm: ChatGroq) -> callable:
    """
    Create the post-generation validation node (conditional edge function).

    Performs hallucination check + answer relevance check.
    Returns a routing decision for the graph.

    Args:
        llm: The ChatGroq LLM for validation (from GroqLLMFactory.create_grading_llm).

    Returns:
        A conditional edge function: (AgentState) -> Literal["useful", "not supported", "not useful"]
    """

    # -- Hallucination Grader --------------------------------------------------
    # Checks if the generated answer is GROUNDED in the retrieved documents.
    class GradeHallucinations(BaseModel):
        """Binary score: is the answer grounded in the facts?"""
        binary_score: Literal["yes", "no"] = Field(
            description="Is the answer grounded in the retrieved documents? 'yes' or 'no'"
        )

    hallucination_grader = llm.with_structured_output(GradeHallucinations)

    hallucination_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a grader assessing whether an LLM generation is grounded in "
            "a set of retrieved facts. Give a binary score 'yes' or 'no'. "
            "'Yes' means the answer is supported by the facts.",
        ),
        (
            "human",
            "Retrieved facts: \n\n {documents} \n\n LLM generation: {generation}",
        ),
    ])

    hallucination_chain = hallucination_prompt | hallucination_grader

    # -- Answer Relevance Grader -----------------------------------------------
    # Checks if the answer actually ADDRESSES the user's question.
    class GradeAnswer(BaseModel):
        """Binary score: does the answer address the question?"""
        binary_score: Literal["yes", "no"] = Field(
            description="Does the answer address the question? 'yes' or 'no'"
        )

    answer_grader = llm.with_structured_output(GradeAnswer)

    answer_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a grader assessing whether an answer addresses a question. "
            "Give a binary score 'yes' or 'no'. "
            "'Yes' means the answer resolves the question.",
        ),
        (
            "human",
            "User question: \n\n {question} \n\n LLM generation: {generation}",
        ),
    ])

    answer_chain = answer_prompt | answer_grader

    def validate_node(state: AgentState) -> Literal["useful", "not supported", "not useful"]:
        """
        Validate the generated answer with two quality checks.

        Check 1 — Hallucination: Is the answer grounded in the documents?
            If NO -> return "not supported" (regenerate with same docs).

        Check 2 — Answer Relevance: Does the answer address the question?
            If YES -> return "useful" (done, go to END).
            If NO  -> return "not useful" (rewrite query, full retry).

        Args:
            state: Current graph state with generation and messages.

        Returns:
            "useful" — answer is good, proceed to END.
            "not supported" — hallucinated, retry generation.
            "not useful" — doesn't answer the question, rewrite and retry.
        """
        logger.info("  [validate] Checking generation quality...")

        messages = state["messages"]

        # Find the LAST HumanMessage — that's the current question
        question = ""
        for msg in reversed(messages):
            if hasattr(msg, "type") and msg.type == "human":
                question = msg.content
                break
        if not question and messages:
            question = messages[0].content

        generation = state["generation"]

        # Get the retrieved docs from the LAST ToolMessage
        docs_content = ""
        for msg in reversed(messages):
            if hasattr(msg, "type") and msg.type == "tool":
                docs_content = msg.content
                break

        # -- Check 1: Hallucination --------------------------------------------
        try:
            h_result = hallucination_chain.invoke({
                "documents": docs_content,
                "generation": generation,
            })
            grounded = h_result.binary_score
        except Exception as e:
            logger.warning("  [validate] Hallucination check failed: %s. Defaulting to 'yes'.", str(e)[:60])
            grounded = "yes"

        if grounded != "yes":
            logger.info("  [validate] HALLUCINATION detected -> regenerate")
            return "not supported"

        # -- Check 2: Answer Relevance -----------------------------------------
        logger.info("  [validate] Grounded in docs. Checking answer relevance...")
        try:
            a_result = answer_chain.invoke({
                "question": question,
                "generation": generation,
            })
            useful = a_result.binary_score
        except Exception as e:
            logger.warning("  [validate] Answer check failed: %s. Defaulting to 'yes'.", str(e)[:60])
            useful = "yes"

        if useful == "yes":
            logger.info("  [validate] Answer is USEFUL -> done")
            return "useful"
        else:
            logger.info("  [validate] Answer NOT USEFUL -> rewrite and retry")
            return "not useful"

    return validate_node
