"""
🤖 Lesson 9.1 — AI Agents: What They Are, How They Work, Why They Matter

═══════════════════════════════════════════════════════════════════
WHAT IS AN AI AGENT?
═══════════════════════════════════════════════════════════════════

An AI Agent is a software program powered by an LLM that can:
    1. UNDERSTAND a task (via natural language)
    2. REASON about what steps are needed
    3. USE TOOLS to accomplish those steps
    4. OBSERVE the results
    5. DECIDE what to do next

The key difference from a regular LLM call:
    Regular LLM:  You ask → It answers (one-shot, no actions)
    AI Agent:     You ask → It THINKS → Uses tools → Observes → Thinks again → Answers

Think of it like this:
    Regular LLM  = A very smart person who can only TALK
    AI Agent     = A very smart person who can TALK + USE A COMPUTER + BROWSE THE WEB

═══════════════════════════════════════════════════════════════════
THE BRAIN + TOOLS ARCHITECTURE
═══════════════════════════════════════════════════════════════════

Every AI Agent has TWO parts:

    ┌─────────────────────────────────────────────────────────────┐
    │                        AI AGENT                             │
    │                                                             │
    │   ┌───────────────┐         ┌───────────────────────────┐  │
    │   │   LLM (Brain) │ ──────→ │   Tools (Hands)           │  │
    │   │               │         │                           │  │
    │   │ - Understands  │         │ - Search the web          │  │
    │   │ - Reasons      │ ←────── │ - Query a database        │  │
    │   │ - Plans        │         │ - Run code                │  │
    │   │ - Decides      │         │ - Call APIs               │  │
    │   │               │         │ - Read/write files        │  │
    │   └───────────────┘         └───────────────────────────┘  │
    │                                                             │
    │   LLM = Brain (decides WHAT to do)                         │
    │   Tools = Hands (actually DOES it)                         │
    └─────────────────────────────────────────────────────────────┘

The LLM DECIDES which tool to use, with what parameters.
The tool EXECUTES and returns the result.
The LLM READS the result and decides the next step.

═══════════════════════════════════════════════════════════════════
THE ReAct PATTERN (Reason + Act)
═══════════════════════════════════════════════════════════════════

Most AI Agents follow the ReAct loop:

    User: "What's the weather in Tokyo and should I bring an umbrella?"

    THOUGHT 1: I need to check the weather in Tokyo. I'll use the weather tool.
    ACTION 1:  weather_tool("Tokyo")
    OBSERVATION 1: Tokyo: 22°C, Rain expected, 80% humidity

    THOUGHT 2: It's going to rain. I should recommend an umbrella.
    ACTION 2:  (no tool needed — I can answer directly)
    FINAL ANSWER: "Tokyo is 22°C with rain expected. Yes, bring an umbrella!"

    The agent LOOPS: Think → Act → Observe → Think → Act → ... → Answer

This is fundamentally different from a regular LLM which would just
guess the weather based on training data (possibly outdated/wrong).

═══════════════════════════════════════════════════════════════════
WHY ARE AI AGENTS TRENDING NOW?
═══════════════════════════════════════════════════════════════════

1. LLMs got SMART ENOUGH to reason about tool usage
   - GPT-4, Llama 3, Gemma 2 can reliably decide which tool to call
   - Earlier models (GPT-3) were too unreliable for this

2. Tool-calling became STANDARDIZED
   - OpenAI, Groq, Anthropic all support "function calling" natively
   - The LLM outputs structured JSON: {"tool": "search", "args": {"query": "..."}}

3. Frameworks made it EASY
   - LangChain, LangGraph, CrewAI, AutoGen — build agents in 10 lines
   - Before these, you'd write hundreds of lines of orchestration code

4. Real business VALUE
   - Agents can automate complex workflows that previously needed humans
   - Customer support, code review, research, data analysis — all automatable

═══════════════════════════════════════════════════════════════════
AI AGENTS IN A SOFTWARE ENGINEER'S DAILY LIFE
═══════════════════════════════════════════════════════════════════

You're ALREADY using AI agents (or will be soon):

1. CODE ASSISTANTS (Kiro, GitHub Copilot, Cursor)
   - Agent reads your code → understands context → suggests/writes code
   - Uses tools: file system, terminal, search, diagnostics

2. CI/CD AGENTS
   - Agent monitors PR → runs tests → reviews code → suggests fixes
   - Tools: GitHub API, test runners, linters

3. DEBUGGING AGENTS
   - Agent reads error log → searches docs → suggests fix → applies it
   - Tools: log reader, documentation search, code editor

4. DATABASE AGENTS
   - "Show me all users who signed up last week" → Agent writes SQL → runs it
   - Tools: SQL executor, schema reader

5. RESEARCH AGENTS
   - "Find the best Python library for PDF parsing" → searches → compares → recommends
   - Tools: web search, GitHub API, package registry

═══════════════════════════════════════════════════════════════════
ARE AGENTS USED IN PRODUCTION TODAY?
═══════════════════════════════════════════════════════════════════

YES, but with guardrails:

    ┌──────────────────────┬──────────────────────────────────────┐
    │ Company              │ How they use agents                  │
    ├──────────────────────┼──────────────────────────────────────┤
    │ Customer Support     │ Agent handles L1 tickets, escalates  │
    │                      │ complex ones to humans               │
    │ E-commerce           │ Agent searches products, compares    │
    │                      │ prices, places orders                │
    │ Healthcare           │ Agent reads patient data, suggests   │
    │                      │ preliminary diagnosis (human reviews)│
    │ Finance              │ Agent monitors transactions, flags   │
    │                      │ suspicious activity                  │
    │ Software Dev         │ Agent reviews PRs, runs tests,       │
    │                      │ generates documentation              │
    └──────────────────────┴──────────────────────────────────────┘

    IMPORTANT: Production agents ALWAYS have human-in-the-loop
    for critical decisions. Full autonomy is rare and risky.

═══════════════════════════════════════════════════════════════════
TECH STACK FOR BUILDING AI AGENTS
═══════════════════════════════════════════════════════════════════

    ┌──────────────────────┬──────────────────────────────────────┐
    │ Component            │ Options                              │
    ├──────────────────────┼──────────────────────────────────────┤
    │ LLM (Brain)          │ GPT-4, Llama 3, Gemma 2, Mistral    │
    │ LLM Provider         │ OpenAI, Groq (free!), Ollama (local)│
    │ Agent Framework      │ LangGraph (recommended), LangChain,  │
    │                      │ CrewAI, AutoGen                      │
    │ Tools                │ Tavily (search), custom Python funcs │
    │ Memory               │ Redis, PostgreSQL, in-memory         │
    │ Deployment           │ FastAPI + LangServe, Docker          │
    │ Frontend             │ Streamlit, React                     │
    │ Observability        │ LangSmith (tracing + debugging)      │
    │ Tool Protocol        │ MCP (Model Context Protocol)         │
    └──────────────────────┴──────────────────────────────────────┘

HOW TO BUILD AN AGENT (high-level steps):
    1. Choose an LLM (Groq/OpenAI/Ollama)
    2. Define tools (Python functions the agent can call)
    3. Create the agent (LangGraph's create_react_agent)
    4. Give it a system prompt (role, rules, constraints)
    5. Run it with a user query
    6. The agent loops: Think → Tool → Observe → Think → Answer

EXAMPLE (from Important-Rules.md):

    from langgraph.prebuilt import create_react_agent

    agent = create_react_agent(
        model=llm,
        tools=[search_tool, calculator_tool],
        state_modifier="You are a research assistant."
    )
    result = agent.invoke({"messages": [("human", "What is 25 * 47?")]})

═══════════════════════════════════════════════════════════════════
TYPES OF AI AGENTS
═══════════════════════════════════════════════════════════════════

    ┌──────────────────────┬──────────────────────────────────────┐
    │ Type                 │ Description                          │
    ├──────────────────────┼──────────────────────────────────────┤
    │ ReAct Agent          │ Reason → Act → Observe loop          │
    │                      │ Most common type. Uses tools.        │
    ├──────────────────────┼──────────────────────────────────────┤
    │ Tool-Calling Agent   │ LLM outputs structured tool calls    │
    │                      │ (JSON with function name + args)     │
    ├──────────────────────┼──────────────────────────────────────┤
    │ Conversational Agent │ ReAct + conversation memory          │
    │                      │ Remembers previous interactions      │
    ├──────────────────────┼──────────────────────────────────────┤
    │ Plan-and-Execute     │ Plans ALL steps first, then executes │
    │                      │ Good for complex multi-step tasks    │
    └──────────────────────┴──────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════
INTERVIEW QUESTIONS — AI Agents
═══════════════════════════════════════════════════════════════════

Q1: What is an AI Agent?
A:  A software program where an LLM acts as the "brain" that can
    reason about tasks, decide which tools to use, execute them,
    observe results, and iterate until the task is complete.

Q2: What is the ReAct pattern?
A:  Reason + Act. The agent thinks about what to do (Reason),
    takes an action using a tool (Act), observes the result,
    and repeats until it has a final answer.

Q3: What is the difference between an LLM and an AI Agent?
A:  An LLM can only generate text. An AI Agent uses an LLM as its
    brain but can also USE TOOLS (search, code execution, APIs)
    to interact with the real world.

Q4: What framework would you use to build an AI Agent?
A:  LangGraph (recommended by LangChain team) for complex agents
    with state management. LangChain's create_react_agent for
    simple agents. CrewAI for multi-agent systems.

Q5: Are AI Agents reliable enough for production?
A:  With guardrails, yes. Production agents use: human-in-the-loop
    for critical decisions, output validation, rate limiting,
    fallback mechanisms, and comprehensive logging via LangSmith.

Author: GenAI Learner
Date: 2026-04-14
"""

# This lesson is conceptual/theoretical — no runnable code.
# Hands-on agent building will happen when you learn LangGraph.

if __name__ == "__main__":
    print("=" * 70)
    print("🤖 Lesson 9.1 — AI Agents Explained")
    print("=" * 70)
    print()
    print("This is a conceptual lesson. Read the file for full content.")
    print()
    print("Key takeaways:")
    print("  - AI Agent = LLM (brain) + Tools (hands)")
    print("  - ReAct loop: Think → Act → Observe → Repeat")
    print("  - LangGraph is the recommended framework for building agents")
    print("  - Production agents always have human-in-the-loop")
    print("  - You're already using agents: Kiro, Copilot, CI/CD bots")
    print()
    print("Next: Lesson 9.2 — Agentic AI (multi-agent systems)")
