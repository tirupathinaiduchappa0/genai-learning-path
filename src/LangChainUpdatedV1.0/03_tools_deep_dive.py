"""
🛠️ Lesson 10.3 — Tools Deep Dive: Giving LLMs Hands to Interact with the World

═══════════════════════════════════════════════════════════════════
WHAT ARE TOOLS IN LANGCHAIN?
═══════════════════════════════════════════════════════════════════

A Tool is a PAIRING of two things:
    1. A SCHEMA  — describes the tool's name, description, and argument types
    2. A FUNCTION — the actual Python code that runs when the tool is called

Think of it this way:
    - The SCHEMA is the menu at a restaurant (what's available, what it does)
    - The FUNCTION is the kitchen (actually makes the food)

The LLM reads the SCHEMA to decide WHICH tool to use and WHAT arguments to pass.
Then YOUR CODE executes the FUNCTION with those arguments.

═══════════════════════════════════════════════════════════════════
WHY DO TOOLS MATTER?
═══════════════════════════════════════════════════════════════════

LLMs are brains without hands. They can THINK but they can't DO:
    ❌ Can't check the weather          → Need a weather API tool
    ❌ Can't query a database            → Need a database tool
    ❌ Can't search the internet         → Need a search tool
    ❌ Can't send emails                 → Need an email tool
    ❌ Can't read files                  → Need a file reader tool

Tools are the BRIDGE between the LLM's intelligence and the real world.
Without tools, an agent is just a chatbot. WITH tools, it becomes an agent
that can actually get things done.

═══════════════════════════════════════════════════════════════════
HOW TOOLS CONNECT TO WHAT YOU ALREADY KNOW
═══════════════════════════════════════════════════════════════════

In Lesson 10.1 (Agents), you saw the ReAct loop:
    Think → Tool Call → Observe → Answer

Now we go DEEPER into the "Tool Call" part:
    1. HOW to create tools (@tool decorator, StructuredTool, BaseTool)
    2. HOW to connect tools to an LLM (bind_tools)
    3. HOW the tool execution loop works (the 3-step pattern)
    4. HOW to combine multiple tools into a mini agent

TOPICS COVERED:
    1. @tool Decorator — The primary way to create tools
    2. bind_tools() — Connecting tools to an LLM
    3. Tool Execution Loop — The complete 3-step pattern
    4. Built-in Tool — DuckDuckGo Web Search
    5. Production Pattern — Multi-tool agent



    Lesson 10.3 — Tools Deep Dive (03_tools_deep_dive.py) — all 5 demos ran successfully:

@tool Decorator — created a simple weather tool AND a production-level product catalog search tool with multiple params (query, category, max_results). Explained that the docstring is the MOST CRITICAL part (LLM reads it to decide when to use the tool). Mentioned alternatives: StructuredTool.from_function() for custom schemas, BaseTool subclass for stateful tools.

bind_tools() — showed how the LLM REQUESTS tool calls (doesn't execute them). Demonstrated the response structure: empty content + populated tool_calls with name, args, and id. Handled the edge case where eager models try to call tools for non-tool questions.

Tool Execution Loop — the CRITICAL 3-step pattern with full message flow traced: HumanMessage → AIMessage(tool_calls: get_weather) → ToolMessage(content: "Light rain, 19°C", tool_call_id: "e0ktddk5z") → AIMessage(final answer). Showed all 4 messages in the flow.

DuckDuckGo Built-in Tool — standalone search worked (found LangChain v1.1 release info), and LLM-bound search worked (searched for latest Python version).

Multi-Tool Agent — combined weather + calculator + DuckDuckGo. Agent correctly routed weather question to get_weather and math question to calculate.




HOW TO RUN:
    $ python 03_tools_deep_dive.py

Author: GenAI Learner
Date: 2026-04-14
"""

import logging
import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 1️⃣  @tool Decorator — The Primary Way to Create Tools
# ═══════════════════════════════════════════════════════════════════════════════
#
# The @tool decorator from langchain.tools converts a regular Python function
# into a LangChain Tool object. Here's what happens under the hood:
#
#   FUNCTION NAME    → becomes the tool's name (what the LLM sees)
#   DOCSTRING        → becomes the tool's description (CRITICAL — the LLM reads
#                      this to decide WHEN to use the tool and HOW)
#   TYPE HINTS       → become the tool's argument schema (tells the LLM what
#                      parameters to pass and their types)
#
# The docstring is the MOST IMPORTANT part. If it's vague, the LLM won't know
# when to use the tool. If it's wrong, the LLM will use it at the wrong time.
#
# OTHER WAYS TO CREATE TOOLS (for reference):
#   - StructuredTool.from_function() — when you need more control over the
#     schema without subclassing (e.g., custom arg descriptions)
#   - BaseTool subclass — for complex tools that need STATE (e.g., a database
#     connection pool, API client with auth, retry logic)
#
# For 90% of use cases, @tool is all you need.

def demo_tool_decorator() -> None:
    """Demonstrate creating tools with the @tool decorator."""
    from langchain.tools import tool

    # ── Simple Tool: Weather Lookup ──────────────────────────────────
    # Notice: name = "get_weather", description = the docstring,
    # args schema = {"city": str}
    @tool
    def get_weather(city: str) -> str:
        """Get the current weather for a city. Use this when the user asks
        about weather conditions, temperature, or forecast for a specific city."""
        weather_data: dict[str, str] = {
            "New York": "Sunny, 24°C, humidity 45%",
            "London": "Overcast, 14°C, humidity 78%",
            "Tokyo": "Light rain, 19°C, humidity 82%",
            "Paris": "Clear skies, 21°C, humidity 50%",
        }
        return weather_data.get(city, f"No weather data available for {city}")

    # ── Production Tool: Product Catalog Search ──────────────────────
    # A more realistic tool with multiple parameters and type hints.
    # The LLM uses the docstring + param types to construct the call.
    @tool
    def search_product_catalog(
        query: str,
        category: Optional[str] = None,
        max_results: int = 5,
    ) -> str:
        """Search the product catalog database. Use this when the user asks
        about products, pricing, availability, or wants to find items.

        Args:
            query: Search keywords (e.g., 'wireless headphones', 'laptop stand')
            category: Optional product category filter (e.g., 'electronics', 'furniture')
            max_results: Maximum number of results to return (default: 5)
        """
        # In production, this would query a real database (PostgreSQL, Elasticsearch, etc.)
        mock_products: list[dict[str, str]] = [
            {"name": "Wireless Headphones Pro", "price": "$79.99", "category": "electronics"},
            {"name": "Ergonomic Laptop Stand", "price": "$45.00", "category": "accessories"},
            {"name": "Mechanical Keyboard RGB", "price": "$129.99", "category": "electronics"},
        ]
        results = [
            p for p in mock_products
            if query.lower() in p["name"].lower()
            or (category and p["category"] == category.lower())
        ][:max_results]

        if not results:
            return f"No products found for '{query}'" + (f" in category '{category}'" if category else "")
        return "\n".join(f"- {p['name']}: {p['price']} ({p['category']})" for p in results)

    # ── Inspect the tool objects ─────────────────────────────────────
    logger.info("--- @tool Decorator ---")
    logger.info("Tool 1 — name: %s", get_weather.name)
    logger.info("Tool 1 — description: %s", get_weather.description[:80])
    logger.info("Tool 1 — args schema: %s", get_weather.args)

    logger.info("Tool 2 — name: %s", search_product_catalog.name)
    logger.info("Tool 2 — description: %s", search_product_catalog.description[:80])
    logger.info("Tool 2 — args schema: %s", search_product_catalog.args)

    # ── Invoke tools directly (without LLM) ──────────────────────────
    weather_result: str = get_weather.invoke({"city": "Tokyo"})
    logger.info("Direct invoke — get_weather('Tokyo'): %s", weather_result)

    product_result: str = search_product_catalog.invoke({
        "query": "headphones",
        "category": "electronics",
        "max_results": 3,
    })
    logger.info("Direct invoke — search_product_catalog('headphones'):\n%s", product_result)


# ═══════════════════════════════════════════════════════════════════════════════
# 2️⃣  bind_tools() — Connecting Tools to an LLM
# ═══════════════════════════════════════════════════════════════════════════════
#
# Creating tools is step 1. Step 2 is TELLING the LLM about them.
#
# model.bind_tools([tool1, tool2]) does this:
#   - Converts each tool's schema into the format the LLM provider expects
#   - Returns a NEW model instance that "knows" about these tools
#   - The LLM can now REQUEST tool calls in its responses
#
# CRITICAL DISTINCTION:
#   The LLM does NOT call tools. It REQUESTS tool calls.
#   It says: "I want to call get_weather with city='Tokyo'"
#   YOUR CODE is responsible for actually executing the tool.
#
# RESPONSE STRUCTURE:
#   response = model_with_tools.invoke("What's the weather in Tokyo?")
#
#   response.content     → "" (EMPTY when the LLM wants to call a tool)
#   response.tool_calls  → [{"name": "get_weather", "args": {"city": "Tokyo"}, "id": "call_abc123"}]
#
#   It's EITHER text OR tool calls — never both.
#   - If the LLM can answer directly → content has text, tool_calls is empty
#   - If the LLM needs a tool       → content is empty, tool_calls has the request

def demo_bind_tools() -> None:
    """Demonstrate binding tools to an LLM and inspecting tool call responses."""
    from langchain.tools import tool
    from langchain_groq import ChatGroq

    @tool
    def get_weather(city: str) -> str:
        """Get the current weather for a city. Use this when the user asks
        about weather conditions, temperature, or forecast for a specific city."""
        weather_data: dict[str, str] = {
            "New York": "Sunny, 24°C, humidity 45%",
            "London": "Overcast, 14°C, humidity 78%",
            "Tokyo": "Light rain, 19°C, humidity 82%",
        }
        return weather_data.get(city, f"No weather data available for {city}")

    @tool
    def calculate(expression: str) -> str:
        """Evaluate a mathematical expression. Use this when the user asks
        a math question or needs a calculation done."""
        try:
            result = eval(expression)  # noqa: S307 — demo only; use safe parser in production
            return str(result)
        except Exception as e:
            return f"Calculation error: {e}"

    # ── Bind tools to the LLM ───────────────────────────────────────
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_tokens=256)
    llm_with_tools = llm.bind_tools([get_weather, calculate])

    logger.info("--- bind_tools() ---")

    # ── Case 1: LLM wants to call a tool ─────────────────────────────
    response = llm_with_tools.invoke("What is the weather in London?")
    logger.info("Query: 'What is the weather in London?'")
    logger.info("content (empty when tool call): '%s'", response.content[:50] if response.content else "")
    logger.info("tool_calls: %s", response.tool_calls)

    if response.tool_calls:
        tc = response.tool_calls[0]
        logger.info("  → Tool name: %s", tc["name"])
        logger.info("  → Tool args: %s", tc["args"])
        logger.info("  → Tool call ID: %s", tc["id"])

    # ── Case 2: LLM answers directly (no tool needed) ────────────────
    # NOTE: Some models may still attempt tool calls even for general questions.
    # In production, handle this gracefully.
    try:
        response_direct = llm_with_tools.invoke("What is LangChain?")
        logger.info("Query: 'What is LangChain?'")
        logger.info("content: '%s'", response_direct.content[:100])
        logger.info("tool_calls: %s", response_direct.tool_calls)
    except Exception as e:
        logger.info("Query: 'What is LangChain?' — Model attempted tool call for non-tool question")
        logger.info("This is normal — some models are eager to use tools. Handle with try/except.")


# ═══════════════════════════════════════════════════════════════════════════════
# 3️⃣  Tool Execution Loop — The Complete 3-Step Pattern (CRITICAL)
# ═══════════════════════════════════════════════════════════════════════════════
#
# This is THE most important pattern in tool-calling. Every agent framework
# (LangChain, LangGraph, CrewAI) uses this loop under the hood.
#
# THE 3-STEP LOOP:
#
#   Step 1: User sends message → LLM responds with tool_calls
#           [HumanMessage("What's the weather in Tokyo?")]
#           → AIMessage(content="", tool_calls=[{name: "get_weather", args: {city: "Tokyo"}}])
#
#   Step 2: Execute each tool → Get ToolMessage
#           tool.invoke(tool_call) → ToolMessage(content="Light rain, 19°C", tool_call_id="call_abc")
#
#   Step 3: Append ToolMessage to messages → Send back to LLM → Get final answer
#           [HumanMessage, AIMessage(tool_calls), ToolMessage]
#           → AIMessage(content="The weather in Tokyo is light rain at 19°C.")
#
# THE FULL MESSAGE FLOW:
#   messages = [
#       HumanMessage("What's the weather in Tokyo?"),       ← User's question
#       AIMessage(tool_calls=[...]),                         ← LLM requests tool
#       ToolMessage(content="Light rain, 19°C", ...),       ← Tool result
#       AIMessage("It's light rain at 19°C in Tokyo."),     ← LLM's final answer
#   ]
#
# ToolMessage has 3 key fields:
#   - content:      The tool's output (string)
#   - name:         Which tool produced this result
#   - tool_call_id: Links this result back to the specific tool_call request

def demo_tool_execution_loop() -> None:
    """Demonstrate the complete tool execution loop — the core agent pattern."""
    from langchain.tools import tool
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from langchain_groq import ChatGroq

    @tool
    def get_weather(city: str) -> str:
        """Get the current weather for a city. Use this when the user asks
        about weather conditions, temperature, or forecast for a specific city."""
        weather_data: dict[str, str] = {
            "New York": "Sunny, 24°C, humidity 45%",
            "London": "Overcast, 14°C, humidity 78%",
            "Tokyo": "Light rain, 19°C, humidity 82%",
        }
        return weather_data.get(city, f"No weather data available for {city}")

    # ── Setup ────────────────────────────────────────────────────────
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_tokens=256)
    tools: list = [get_weather]
    tool_map: dict = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)

    logger.info("--- Tool Execution Loop (3-Step Pattern) ---")

    # ── STEP 1: User sends message → LLM responds with tool_calls ───
    user_message = HumanMessage(content="What is the weather like in Tokyo right now?")
    messages: list = [user_message]
    logger.info("STEP 1 — User: %s", user_message.content)

    ai_response: AIMessage = llm_with_tools.invoke(messages)
    messages.append(ai_response)
    logger.info("STEP 1 — LLM response: tool_calls=%s, content='%s'",
                ai_response.tool_calls, ai_response.content[:50] if ai_response.content else "")

    # ── STEP 2: Execute each tool → Get ToolMessage ──────────────────
    if ai_response.tool_calls:
        for tc in ai_response.tool_calls:
            logger.info("STEP 2 — Executing tool: %s(%s)", tc["name"], tc["args"])
            selected_tool = tool_map[tc["name"]]
            tool_result: ToolMessage = selected_tool.invoke(tc)
            messages.append(tool_result)
            logger.info("STEP 2 — ToolMessage: content='%s', tool_call_id='%s'",
                        tool_result.content, tool_result.tool_call_id)

    # ── STEP 3: Send back to LLM → Get final answer ─────────────────
    final_response: AIMessage = llm_with_tools.invoke(messages)
    messages.append(final_response)
    logger.info("STEP 3 — Final answer: %s", final_response.content)

    # ── Show the complete message flow ───────────────────────────────
    logger.info("Complete message flow (%d messages):", len(messages))
    for i, msg in enumerate(messages):
        msg_type: str = type(msg).__name__
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            logger.info("  [%d] %s → tool_calls: %s", i, msg_type,
                        [tc["name"] for tc in msg.tool_calls])
        elif isinstance(msg, ToolMessage):
            logger.info("  [%d] %s → content='%s', tool_call_id='%s'",
                        i, msg_type, str(msg.content)[:60], msg.tool_call_id)
        else:
            logger.info("  [%d] %s → '%s'", i, msg_type, str(msg.content)[:80])


# ═══════════════════════════════════════════════════════════════════════════════
# 4️⃣  Built-in Tool — DuckDuckGo Web Search
# ═══════════════════════════════════════════════════════════════════════════════
#
# LangChain has many built-in tools in langchain_community.tools.
# DuckDuckGo is a great example — free, no API key needed.
#
# Install: pip install duckduckgo-search
#
# You can use built-in tools in two ways:
#   1. STANDALONE — call tool.invoke("query") directly
#   2. BOUND TO LLM — let the LLM decide when to search

def demo_duckduckgo_search() -> None:
    """Demonstrate DuckDuckGo web search as a built-in LangChain tool."""
    try:
        from langchain_community.tools import DuckDuckGoSearchRun
    except ImportError:
        logger.warning("duckduckgo-search not installed. Run: pip install duckduckgo-search")
        return

    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from langchain_groq import ChatGroq

    search = DuckDuckGoSearchRun()

    logger.info("--- DuckDuckGo Web Search (Built-in Tool) ---")

    # ── Standalone usage ─────────────────────────────────────────────
    logger.info("Tool name: %s", search.name)
    logger.info("Tool description: %s", search.description[:80])

    standalone_result: str = search.invoke("LangChain v1 release date 2025")
    logger.info("Standalone search result: %s", standalone_result[:200])

    # ── Bound to LLM — let the model decide when to search ──────────
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_tokens=256)
    llm_with_search = llm.bind_tools([search])

    messages: list = [HumanMessage(content="Search the web: What is the latest version of Python?")]
    ai_response: AIMessage = llm_with_search.invoke(messages)

    if ai_response.tool_calls:
        logger.info("LLM requested search: %s", ai_response.tool_calls[0]["args"])
        # Execute the search
        tool_result: ToolMessage = search.invoke(ai_response.tool_calls[0])
        messages.append(ai_response)
        messages.append(tool_result)
        # Get final answer
        final: AIMessage = llm_with_search.invoke(messages)
        logger.info("Final answer: %s", final.content[:200])
    else:
        logger.info("LLM answered directly: %s", ai_response.content[:200])


# ═══════════════════════════════════════════════════════════════════════════════
# 5️⃣  Production Pattern — Multi-Tool Agent
# ═══════════════════════════════════════════════════════════════════════════════
#
# In production, agents have MULTIPLE tools. The LLM picks the right one
# based on the user's question and each tool's description.
#
# This demo combines:
#   - Custom tool: get_weather (our weather lookup)
#   - Custom tool: calculate (math evaluator)
#   - Built-in tool: DuckDuckGoSearchRun (web search)
#
# The tool execution loop is the same — just with more tools in the map.

def demo_multi_tool_agent() -> None:
    """Demonstrate a mini agent with multiple tools using the execution loop."""
    from langchain.tools import tool
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from langchain_groq import ChatGroq

    # ── Define custom tools ──────────────────────────────────────────
    @tool
    def get_weather(city: str) -> str:
        """Get the current weather for a city. Use this when the user asks
        about weather conditions, temperature, or forecast for a specific city."""
        weather_data: dict[str, str] = {
            "New York": "Sunny, 24°C, humidity 45%",
            "London": "Overcast, 14°C, humidity 78%",
            "Tokyo": "Light rain, 19°C, humidity 82%",
            "San Francisco": "Foggy, 16°C, humidity 88%",
        }
        return weather_data.get(city, f"No weather data available for {city}")

    @tool
    def calculate(expression: str) -> str:
        """Evaluate a mathematical expression and return the result. Use this
        when the user asks a math question, needs arithmetic, or wants a
        calculation performed."""
        try:
            result = eval(expression)  # noqa: S307 — demo only; use safe parser in production
            return f"{expression} = {result}"
        except Exception as e:
            return f"Calculation error: {e}"

    # ── Gather all tools (custom + built-in) ─────────────────────────
    all_tools: list = [get_weather, calculate]

    try:
        from langchain_community.tools import DuckDuckGoSearchRun
        web_search = DuckDuckGoSearchRun()
        all_tools.append(web_search)
        logger.info("DuckDuckGo search tool loaded.")
    except ImportError:
        logger.warning("duckduckgo-search not installed — skipping web search tool.")

    tool_map: dict = {t.name: t for t in all_tools}

    # ── Bind all tools to the LLM ────────────────────────────────────
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_tokens=256)
    llm_with_tools = llm.bind_tools(all_tools)

    logger.info("--- Multi-Tool Agent ---")
    logger.info("Available tools: %s", list(tool_map.keys()))

    # ── Helper: run one query through the tool execution loop ────────
    def run_agent_query(query: str) -> str:
        """Run a single query through the tool execution loop and return the final answer."""
        messages: list = [HumanMessage(content=query)]
        ai_response: AIMessage = llm_with_tools.invoke(messages)
        messages.append(ai_response)

        # Execute tool calls if any
        if ai_response.tool_calls:
            for tc in ai_response.tool_calls:
                logger.info("  🔧 Calling tool: %s(%s)", tc["name"], tc["args"])
                selected_tool = tool_map[tc["name"]]
                tool_result: ToolMessage = selected_tool.invoke(tc)
                messages.append(tool_result)

            # Get final answer after tool execution
            final_response: AIMessage = llm_with_tools.invoke(messages)
            return final_response.content
        else:
            return ai_response.content

    # ── Test with different queries ──────────────────────────────────
    queries: list[str] = [
        "What is the weather in San Francisco?",
        "What is 1234 * 5678 + 91011?",
    ]

    for query in queries:
        logger.info("Q: %s", query)
        answer: str = run_agent_query(query)
        logger.info("A: %s", answer[:200])


# ═══════════════════════════════════════════════════════════════════════════════
# 🏭 INTERVIEW QUESTIONS — Tools in LangChain
# ═══════════════════════════════════════════════════════════════════════════════
#
# Q1: What is a Tool in LangChain and what are its two components?
# A:  A Tool is a pairing of a SCHEMA (name, description, argument types) and
#     a FUNCTION (the actual Python code). The LLM reads the schema to decide
#     when to use the tool, and your code executes the function with the
#     arguments the LLM provides.
#
# Q2: Why is the docstring so important when creating a tool with @tool?
# A:  The docstring becomes the tool's DESCRIPTION, which is what the LLM reads
#     to decide WHEN and HOW to use the tool. A vague or missing docstring means
#     the LLM won't know when to call it. It's the most critical part of tool
#     definition — think of it as the tool's instruction manual for the LLM.
#
# Q3: What is the difference between bind_tools() and actually executing a tool?
# A:  bind_tools() tells the LLM about available tools (their schemas). The LLM
#     then REQUESTS tool calls via response.tool_calls — it does NOT execute them.
#     Your code is responsible for executing the tool with tool.invoke(tool_call)
#     and feeding the ToolMessage result back to the LLM.
#
# Q4: Describe the 3-step tool execution loop.
# A:  Step 1: Send user message to LLM → LLM responds with tool_calls (name,
#     args, id). Step 2: Execute each tool with tool.invoke(tool_call) → get
#     ToolMessage (content, name, tool_call_id). Step 3: Append ToolMessage to
#     messages, send back to LLM → LLM produces the final text answer.
#
# Q5: When would you use BaseTool subclass instead of the @tool decorator?
# A:  Use BaseTool when your tool needs STATE — e.g., a database connection pool,
#     an authenticated API client, retry logic, or caching. BaseTool lets you
#     define __init__ for setup and _run/_arun for execution. For stateless tools
#     (90% of cases), @tool is simpler and preferred.


# ═══════════════════════════════════════════════════════════════════════════════
# 🚀 Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("🛠️ TOOLS DEEP DIVE — Giving LLMs Hands to Interact with the World")
    logger.info("=" * 70)

    logger.info("\n🔹 1. @tool Decorator — Creating Tools")
    demo_tool_decorator()

    logger.info("\n🔹 2. bind_tools() — Connecting Tools to an LLM")
    demo_bind_tools()

    logger.info("\n🔹 3. Tool Execution Loop — The Complete 3-Step Pattern")
    demo_tool_execution_loop()

    logger.info("\n🔹 4. Built-in Tool — DuckDuckGo Web Search")
    demo_duckduckgo_search()

    logger.info("\n🔹 5. Production Pattern — Multi-Tool Agent")
    demo_multi_tool_agent()

    logger.info("\n" + "=" * 70)
    logger.info("✅ Tools Deep Dive lesson complete!")
    logger.info("Next: Messages deep dive (HumanMessage, AIMessage, SystemMessage, ToolMessage)")
    logger.info("=" * 70)
