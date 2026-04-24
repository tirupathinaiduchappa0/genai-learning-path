"""
DocSage Agent Node — The LLM agent that decides which tool to call.

WHY THIS NODE EXISTS:
    This is the BRAIN of the Agentic RAG pattern. The agent receives
    the user's question, looks at the available tools (PDF retrievers
    + web search), and DECIDES which one to call. It reads the tool
    descriptions to make this decision.

    If the question is about a topic covered by an uploaded PDF, the
    agent calls that PDF's retriever tool. If no PDF is relevant, it
    calls the web search tool. If the question is simple enough to
    answer directly (e.g., "Hello"), it responds without any tool.

INTERVIEW POINT:
    "How does your agent decide which knowledge base to query?"
    "The agent LLM has all retriever tools bound via bind_tools().
    Each tool has a name and description that the LLM reads. For
    example, if a user uploaded a legal contract, the tool description
    says 'Search the legal contract for partnership terms.' When the
    user asks about partnership terms, the LLM matches the query to
    that tool's description and calls it. This is the same pattern
    as a ReAct agent — the LLM reasons about which tool to use."

HOW bind_tools WORKS:
    llm.bind_tools(tools) wraps the LLM so that its responses can
    include tool_calls. The LLM doesn't execute the tools — it just
    DECIDES which tool to call and with what arguments. The ToolNode
    (in the graph) actually executes the tool call.
"""

import logging
from typing import Any

from langchain_core.messages import SystemMessage
from langchain_groq import ChatGroq

from docsage.state.state import AgentState

logger = logging.getLogger(__name__)

# System prompt that instructs the agent on its role and tool usage.
# This prompt is critical — it tells the LLM HOW to decide between tools.
AGENT_SYSTEM_PROMPT = (
    "You are DocSage, an intelligent document assistant. "
    "You have access to these tools:\n"
    "- Document search tools for uploaded files\n"
    "- Web search tool (tavily) for general questions\n"
    "- send_email tool for emailing answers to people\n\n"
    "RULES (follow strictly):\n"
    "1. If the user asks to EMAIL, SEND, or SHARE something to an email address, "
    "you MUST use the send_email tool. Extract the recipient email from the message. "
    "Use the previous conversation answer as the email body. Create a clear subject.\n"
    "2. If the question is about a topic in the uploaded documents, use the document search tool.\n"
    "3. If no document tool is relevant, use the web search tool.\n"
    "4. If the question is a simple greeting, respond directly without tools.\n"
    "5. IMPORTANT: When you see an email address in the user's message AND words like "
    "'send', 'email', 'mail', 'share', 'forward' — ALWAYS use send_email tool."
)


def create_agent_node(llm: ChatGroq, tools: list) -> callable:
    """
    Create the agent node function with tools bound to the LLM.

    This is a FACTORY FUNCTION — it creates and returns the node function
    with the LLM and tools baked in via closure. This pattern keeps the
    node function signature clean (just state in, dict out) while still
    having access to the configured LLM.

    Args:
        llm: The ChatGroq LLM instance for the agent (from GroqLLMFactory).
        tools: List of LangChain tools (retriever tools + web search tool).

    Returns:
        A node function: (AgentState) -> dict
    """
    # Bind tools to the LLM so it can decide which to call.
    # After binding, the LLM's responses may include tool_calls.
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: AgentState) -> dict:
        """
        Agent node: LLM decides which tool to call based on the question.

        Builds a compact conversation context by:
        1. Collecting ALL human questions and AI answers (skipping tool messages).
        2. Keeping the last 10 relevant messages for broader context.
        3. This ensures follow-up questions like "his age" or "his hometown"
           always have the reference person in context.

        Args:
            state: Current graph state with messages.

        Returns:
            Dict with the agent's response appended to messages.
        """
        logger.info("  [agent] Deciding tool for query...")
        messages = state["messages"]

        # Build compact context: human questions + AI text answers only.
        # Skip ToolMessages (large raw text) and AI tool-call-only messages.
        # This keeps token count manageable while preserving conversation flow.
        compact = []
        for msg in messages:
            msg_type = getattr(msg, "type", "")
            if msg_type == "human":
                compact.append(msg)
            elif msg_type == "ai" and hasattr(msg, "content") and msg.content:
                # Only keep AI messages that have actual text content
                # (skip ones that are just tool_calls with empty content)
                if msg.content.strip():
                    compact.append(msg)

        # Keep last 10 messages (5 Q&A turns) — enough for follow-up context
        # without hitting token limits. 10 is safe for llama-3.1-8b (8K context).
        compact = compact[-10:]

        response = llm_with_tools.invoke(
            [SystemMessage(content=AGENT_SYSTEM_PROMPT)] + compact
        )

        if response.tool_calls:
            tool_names = [tc["name"] for tc in response.tool_calls]
            logger.info("  [agent] Tool selected: %s", tool_names)
        else:
            logger.info("  [agent] No tool needed, answering directly.")

        return {"messages": [response]}

    return agent_node
