"""
DocSage LLM Factory — Creates and configures Groq LLM instances.

WHY THIS FILE EXISTS:
    Different parts of the pipeline need different LLM configurations:
    - The agent needs tool-calling capability (bind_tools).
    - The grader needs structured output (with_structured_output).
    - The generator needs good text quality with moderate temperature.
    - The rewriter needs deterministic output (temperature=0).

    Instead of creating LLMs inline everywhere, this factory centralizes
    LLM creation. Change a model name in ONE place, not in 5 nodes.

INTERVIEW POINT:
    "How do you manage LLM instances in your application?"
    "I use a factory class that creates LLM instances with task-specific
    configurations. The agent gets a model that supports tool calling,
    the grader gets a model optimized for structured output, and the
    generator gets a model tuned for text quality. This separation lets
    me swap models per task without touching any node logic."

WHY GROQ (not OpenAI):
    Groq is FREE with generous rate limits. For a portfolio project,
    this means anyone can clone and run it without paying for API keys.
    In production, you'd swap to OpenAI, Anthropic, or Azure OpenAI
    by changing this one file.
"""

import logging
from typing import Optional

from langchain_groq import ChatGroq

from docsage.config import settings

logger = logging.getLogger(__name__)


class GroqLLMFactory:
    """
    Factory class for creating Groq LLM instances with task-specific configs.

    Usage:
        factory = GroqLLMFactory(api_key="gsk_...")
        agent_llm = factory.create_agent_llm()
        grading_llm = factory.create_grading_llm()
        generation_llm = factory.create_generation_llm()
    """

    def __init__(self, api_key: str, model_name: Optional[str] = None) -> None:
        """
        Initialize the factory with a Groq API key.

        Args:
            api_key: Groq API key for authentication.
            model_name: Optional override for the default model.
                        If provided, ALL LLMs will use this model.
                        If not provided, each task uses its default model.
        """
        if not api_key:
            raise ValueError(
                "Groq API key is required. Get one at https://console.groq.com/keys"
            )

        self.api_key = api_key
        self.model_override = model_name
        logger.info("GroqLLMFactory initialized (model_override=%s)", model_name)

    def _create_llm(self, model: str, temperature: float, max_tokens: int) -> ChatGroq:
        """
        Internal helper to create a ChatGroq instance.

        Args:
            model: The Groq model name.
            temperature: Sampling temperature (0=deterministic, 1=creative).
            max_tokens: Maximum tokens in the response.

        Returns:
            A configured ChatGroq instance.
        """
        actual_model = self.model_override or model

        return ChatGroq(
            api_key=self.api_key,
            model=actual_model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def create_agent_llm(self) -> ChatGroq:
        """
        Create an LLM for the agent node (tool selection).

        Uses temperature=0 for deterministic tool selection.
        The agent needs to reliably pick the right retriever tool,
        so we want no randomness in the decision.
        """
        llm = self._create_llm(
            model=settings.DEFAULT_AGENT_MODEL,
            temperature=0,
            max_tokens=settings.AGENT_MAX_TOKENS,
        )
        logger.info(
            "Agent LLM created (model=%s)",
            self.model_override or settings.DEFAULT_AGENT_MODEL,
        )
        return llm

    def create_grading_llm(self) -> ChatGroq:
        """
        Create an LLM for grading and validation nodes (structured output).

        Uses the larger 70b model because structured output (Pydantic)
        requires more capable reasoning to consistently return valid JSON.
        Temperature=0 for deterministic grading.
        """
        llm = self._create_llm(
            model=settings.DEFAULT_GRADING_MODEL,
            temperature=0,
            max_tokens=settings.GRADING_MAX_TOKENS,
        )
        logger.info(
            "Grading LLM created (model=%s)",
            self.model_override or settings.DEFAULT_GRADING_MODEL,
        )
        return llm

    def create_generation_llm(self) -> ChatGroq:
        """
        Create an LLM for the generation node (answer production).

        Uses temperature=0.3 for slightly creative but grounded answers.
        Higher max_tokens because answers can be longer than grading responses.
        """
        llm = self._create_llm(
            model=settings.DEFAULT_GENERATION_MODEL,
            temperature=0.3,
            max_tokens=settings.GENERATION_MAX_TOKENS,
        )
        logger.info(
            "Generation LLM created (model=%s)",
            self.model_override or settings.DEFAULT_GENERATION_MODEL,
        )
        return llm

    def create_rewrite_llm(self) -> ChatGroq:
        """
        Create an LLM for the query rewrite node.

        Uses temperature=0 for deterministic rewrites.
        Lower max_tokens because rewrites are short (just a better question).
        """
        llm = self._create_llm(
            model=settings.DEFAULT_GENERATION_MODEL,
            temperature=0,
            max_tokens=settings.REWRITE_MAX_TOKENS,
        )
        logger.info(
            "Rewrite LLM created (model=%s)",
            self.model_override or settings.DEFAULT_GENERATION_MODEL,
        )
        return llm
