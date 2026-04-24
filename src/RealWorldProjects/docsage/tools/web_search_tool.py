"""
DocSage Web Search Tool — Tavily web search as a fallback knowledge source.

WHY THIS FILE EXISTS:
    When the local PDF knowledge bases don't have the answer, the agent
    needs an EXTERNAL fallback. Tavily provides high-quality web search
    results optimized for LLM consumption.

    This is the "Corrective" part of our hybrid RAG pattern:
    - Local PDFs are the PRIMARY source (fast, controlled, private).
    - Web search is the FALLBACK (when local docs are irrelevant).

INTERVIEW POINT:
    "How do you handle questions that your knowledge base can't answer?"
    "The agent has web search as one of its tools alongside the PDF
    retrievers. If the agent determines that none of the PDF knowledge
    bases are relevant to the query, it calls the Tavily web search
    tool instead. This gives the system access to current, real-time
    information without requiring the user to update their documents."

WHY TAVILY (not Google Search or Bing):
    Tavily is purpose-built for LLM applications. It returns clean,
    structured content (not raw HTML). Free tier gives 1000 searches/month.
    In production, you'd use Tavily Pro or a custom search pipeline.
"""

import logging

from langchain_community.tools.tavily_search import TavilySearchResults

from docsage.config import settings

logger = logging.getLogger(__name__)


def create_web_search_tool() -> TavilySearchResults:
    """
    Create a Tavily web search tool for the agent.

    The tool is configured with max_results from settings.
    It requires TAVILY_API_KEY to be set in the environment.

    Returns:
        A TavilySearchResults tool instance that the agent can call.

    Note:
        The TAVILY_API_KEY must be set in os.environ before calling this.
        In the Streamlit UI, the user enters it in the sidebar.
    """
    tool = TavilySearchResults(max_results=settings.WEB_SEARCH_MAX_RESULTS)
    logger.info(
        "Web search tool created (max_results=%d)",
        settings.WEB_SEARCH_MAX_RESULTS,
    )
    return tool
