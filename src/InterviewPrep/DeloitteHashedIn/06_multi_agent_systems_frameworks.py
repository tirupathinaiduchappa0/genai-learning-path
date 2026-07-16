"""
===================================================================================
PHASE 6 — MULTI-AGENT SYSTEMS & FRAMEWORKS (HashedIn by Deloitte — Lead)
===================================================================================

WHY THIS IS PHASE 6:
    The JD demands: "Deep experience in Multi-Agent System workflow and design,
    including orchestration, communication protocols, and agent lifecycle
    management using LangChain, AutoGen, CrewAI, or similar" + "Familiarity
    with MCP and the Agent2Agent (A2A) protocol" + "design, build, and
    orchestrate autonomous agents that can reason, plan, and execute tasks."

    This is YOUR strength area (DocSage, MCP work) — so Phase 6 is about
    SHARPENING and adding the framework breadth + protocols (A2A) you may
    not have hands-on yet.

DEPTH LEVEL: Technical Lead. Architecture, trade-offs, lifecycle, protocols.

SECTIONS:
    1.  What Makes a System "Agentic" (agent anatomy, autonomy levels)
    2.  Single-Agent Patterns (ReAct, Plan-Execute, Reflection)
    3.  Multi-Agent Orchestration Patterns (the 4 topologies)
    4.  Agent Lifecycle Management
    5.  Inter-Agent Communication & Protocols (MCP, A2A)
    6.  Framework Deep Comparison (LangGraph / AutoGen / CrewAI / ADK)
    7.  Agent Memory & State Management
    8.  Debugging Agentic Loops (the JD calls this out)
    9.  Production Concerns (cost, latency, reliability, observability)
    10. Real-World Design Walkthrough (end-to-end)
    11. Interview Q&A (lead-level)
    12. GOLDEN LESSONS
===================================================================================
"""


# =================================================================================
# SECTION 1: WHAT MAKES A SYSTEM "AGENTIC" (agent anatomy, autonomy levels)
# =================================================================================
'''
A lead must define agentic precisely — interviewers test if you know the line
between a "chatbot," a "workflow," and an "agent."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE DEFINITION:
    An AGENT is an LLM that, given a goal, can REASON about what to do, CHOOSE
    and CALL TOOLS, observe the results, and ITERATE until the goal is met —
    deciding its own next step rather than following a fixed script.

    The 4 capabilities that make it agentic:
    1. REASONING/PLANNING — break a goal into steps.
    2. TOOL USE — call functions/APIs/retrievers to act on the world.
    3. MEMORY — retain context across steps (short + long term).
    4. AUTONOMY — decide the next action based on observations (a loop).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WORKFLOW vs AGENT (the distinction interviewers love):
    WORKFLOW: a FIXED, predefined path. The developer hardcodes the steps;
        the LLM fills in content at each step. Predictable, controllable.
        (e.g., always: retrieve -> grade -> generate.)
    AGENT: the LLM DECIDES the path dynamically — which tool, whether to loop,
        when to stop. Flexible, less predictable.

    Anthropic's framing: "workflows" orchestrate LLMs through predefined code
    paths; "agents" dynamically direct their own process and tool usage.

    THE LEAD INSIGHT: Don't reach for full autonomy by default. Workflows are
    more reliable, debuggable, and cheaper. Use an agent ONLY when the task
    genuinely needs dynamic decision-making. "Use the least autonomy that
    solves the problem."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

AUTONOMY SPECTRUM (low -> high):
    1. Single LLM call (no tools)            — not agentic.
    2. LLM + tools, one shot                 — tool-using, minimal autonomy.
    3. Fixed workflow (chain/graph)          — orchestrated, controlled.
    4. Single agent with a tool loop (ReAct) — autonomous within bounds.
    5. Multi-agent system                    — agents collaborating.
    6. Fully autonomous open-ended agent     — highest risk, least predictable.

INTERVIEW ANSWER:
    "An agent is an LLM that, given a goal, reasons about the steps, chooses
    and calls tools, observes results, and iterates until done — it decides its
    own next action. The four pillars are reasoning, tool use, memory, and
    autonomy. The key distinction is workflow versus agent: a workflow follows
    a fixed developer-defined path with the LLM filling in content, while an
    agent dynamically directs its own process. As a lead I default to the least
    autonomy that solves the problem — workflows are more reliable, debuggable,
    and cheaper, and I only introduce true agent autonomy when the task
    genuinely needs dynamic decision-making."
'''


# =================================================================================
# SECTION 2: SINGLE-AGENT PATTERNS (ReAct, Plan-Execute, Reflection)
# =================================================================================
'''
Before multi-agent, master the single-agent reasoning patterns.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PATTERN 1 — ReAct (Reason + Act) — the foundational loop:
    The agent alternates THOUGHT -> ACTION -> OBSERVATION until it can answer.
        Thought: "I need the user's order status. I'll call the orders tool."
        Action: get_order(id=123)
        Observation: {"status": "shipped"}
        Thought: "I have it. I can answer now."
        Answer: "Your order shipped."
    Most tool-using agents (incl. your DocSage agent node) are ReAct-style.
    The LLM reasons about WHICH tool, calls it, reads the result, repeats.

PATTERN 2 — PLAN-AND-EXECUTE:
    A PLANNER LLM first writes a multi-step plan; an EXECUTOR carries out each
    step (often calling tools), optionally re-planning if a step fails.
    Better for COMPLEX multi-step tasks than pure ReAct (which can lose the
    thread on long tasks). More tokens, more structure.

PATTERN 3 — REFLECTION / SELF-CRITIQUE:
    The agent GENERATES, then CRITIQUES its own output and REVISES.
    (Your DocSage validate node = reflection: check hallucination + relevance,
    regenerate/rewrite if it fails.) Improves quality at the cost of extra calls.

PATTERN 4 — ROUTER / DISPATCHER:
    A classifier/LLM routes the query to the right handler/tool/sub-agent.
    (Adaptive RAG routing.) Cheap, effective for mixed query types.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CONNECT TO YOUR PROJECT (say this in the interview):
    "My DocSage agent is ReAct-style for tool selection, with a reflection loop
    for self-correction — the grade and validate nodes critique retrieval and
    generation, and rewrite re-plans the query. So it combines ReAct, routing,
    and reflection in one LangGraph workflow."

INTERVIEW ANSWER:
    "The foundational pattern is ReAct — the agent loops through thought,
    action, observation: it reasons about which tool to call, calls it, reads
    the result, and repeats until it can answer. For complex multi-step tasks I
    use plan-and-execute, where a planner drafts the steps and an executor runs
    them with re-planning on failure. Reflection adds a self-critique step —
    generate, critique, revise — which is exactly my DocSage validate loop.
    And a router pattern dispatches different query types to the right handler.
    Real systems combine these — DocSage is ReAct plus routing plus reflection."
'''


# =================================================================================
# SECTION 3: MULTI-AGENT ORCHESTRATION PATTERNS (the 4 topologies)
# =================================================================================
'''
The JD: "Multi-Agent System workflow and design, including orchestration."
Know the topologies and when each fits.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHY MULTIPLE AGENTS AT ALL?
    - SPECIALIZATION — each agent has a focused role, prompt, and tool set
      (a "researcher", a "coder", a "reviewer") -> better than one mega-prompt.
    - SEPARATION OF CONCERNS — modular, testable, swappable.
    - PARALLELISM — independent sub-tasks run concurrently.
    Cost: more LLM calls (latency + $), coordination complexity, error
    propagation. Don't multi-agent if one agent suffices.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE 4 ORCHESTRATION TOPOLOGIES:

    1. ORCHESTRATOR-WORKER (Supervisor) — MOST COMMON in production:
        A SUPERVISOR agent receives the goal, decomposes it, and DELEGATES
        sub-tasks to specialist WORKER agents, then aggregates their results.
            [Supervisor]
              /    |    \
        [Research][Code][Review]
        Pros: clear control, easy to reason about, supervisor enforces flow.
        Use: most business workflows. (LangGraph supervisor pattern.)

    2. HIERARCHICAL (agents within agents):
        Supervisors of supervisors — a tree. A top orchestrator delegates to
        mid-level managers who delegate to workers. Scales to complex org-like
        task trees. More overhead.

    3. PEER-TO-PEER / NETWORK (agents talk directly):
        Agents communicate with each other without a central boss; any agent
        can hand off to any other. Flexible but harder to control/debug; risk
        of loops. (AutoGen group chat leans this way.)

    4. HUB-AND-SPOKE / ROUTER:
        A central router classifies the request and sends it to ONE specialist
        agent (not decomposition — selection). Simple, cheap for "pick the
        right expert" cases.

    Also: SEQUENTIAL PIPELINE (agent A -> B -> C, assembly line) and
    PARALLEL (fan out to several agents, then aggregate / vote).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CHOOSING (the lead's decision):
    Clear task decomposition + control needed   -> Orchestrator-Worker.
    Deep, nested task trees                      -> Hierarchical.
    Open-ended collaboration/brainstorming       -> Peer-to-peer (carefully).
    "Pick the right expert"                      -> Hub-and-spoke router.
    Linear stages                                -> Sequential pipeline.
    Independent subtasks                         -> Parallel + aggregate.

    DEFAULT: Orchestrator-Worker — it's controllable, debuggable, and maps to
    how you'd assign work to a human team.

INTERVIEW ANSWER:
    "Multiple agents help when you need specialization, separation of concerns,
    or parallelism — but they add cost and coordination complexity, so I only
    use them when one agent isn't enough. The topologies: orchestrator-worker,
    where a supervisor decomposes the goal and delegates to specialists, is my
    default because it's controllable and debuggable. Hierarchical extends that
    into nested supervisor trees for complex tasks. Peer-to-peer lets agents
    hand off directly — flexible but harder to control and prone to loops.
    Hub-and-spoke routes to a single best expert. I also use sequential
    pipelines for linear stages and parallel fan-out with aggregation for
    independent subtasks. I pick based on whether the work needs decomposition,
    selection, or collaboration."
'''


# =================================================================================
# SECTION 4: AGENT LIFECYCLE MANAGEMENT
# =================================================================================
'''
The JD explicitly says "agent lifecycle management." Most candidates have NO
answer for this — it's a differentiator.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHAT "LIFECYCLE" MEANS:
    The stages an agent (and an agent RUN) goes through, and how you manage
    each. Two senses:
    A) The RUN lifecycle (one task execution).
    B) The OPERATIONAL lifecycle (build -> deploy -> monitor -> improve).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

A) THE RUN LIFECYCLE (one execution):
    1. INITIALIZATION — load the agent: system prompt/persona, tool registry,
       memory/context, model config.
    2. PERCEPTION/INPUT — receive the goal + relevant context.
    3. REASONING/PLANNING — decide the next action.
    4. ACTION — call a tool / sub-agent.
    5. OBSERVATION — ingest the result into state.
    6. LOOP — repeat 3-5 until a TERMINATION condition.
    7. TERMINATION — goal met, OR max-iterations/recursion-limit hit, OR an
       unrecoverable error, OR human stop. (Termination is critical — see S8.)
    8. CLEANUP/RESPONSE — finalize output, release resources, persist memory.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

B) OPERATIONAL LIFECYCLE (the lead's responsibility):
    - VERSIONING — agent definitions, prompts (Lesson 25), tools are versioned.
    - DEPLOYMENT — promote through dev/staging/prod; canary new agent versions.
    - MONITORING — track success rate, steps-per-task, cost/task, latency,
      tool-error rates, loop frequency.
    - EVALUATION — golden task set; regression-test agent behavior on changes.
    - GOVERNANCE — guardrails, permissions per tool, human-in-the-loop for
      high-risk actions, audit logs.
    - IMPROVEMENT — analyze failures, refine prompts/tools, re-deploy.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

KEY LIFECYCLE CONCERNS:
    - STATE PERSISTENCE — checkpoint state so a run can pause/resume/recover
      (LangGraph MemorySaver + thread_id). Survives crashes, enables HITL.
    - RESOURCE MANAGEMENT — close tool connections, bound concurrent agents.
    - TIMEOUTS & LIMITS — max iterations, token budget, wall-clock timeout per run.
    - ERROR RECOVERY — retry a failed tool, fall back, or escalate to a human.

INTERVIEW ANSWER:
    "Agent lifecycle has two senses. The run lifecycle: initialize with prompt,
    tools, memory, and config; then loop through reason, act, observe until a
    termination condition — goal met, max iterations, unrecoverable error, or
    human stop — then finalize and persist memory. I checkpoint state with
    something like LangGraph's MemorySaver and a thread ID so a run can pause,
    resume, or recover from a crash, and I always set iteration, token, and
    wall-clock limits. The operational lifecycle is the lead's job: version
    agent definitions, prompts, and tools; deploy through environments with
    canaries; monitor success rate, steps per task, cost, and tool errors;
    regression-test against a golden task set; and govern with guardrails,
    per-tool permissions, human-in-the-loop for risky actions, and audit logs."
'''


# =================================================================================
# SECTION 5: INTER-AGENT COMMUNICATION & PROTOCOLS (MCP, A2A)
# =================================================================================
'''
The JD calls out MCP AND A2A by name. MCP is your strength; A2A may be new —
learn it. This section is a likely differentiator.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HOW AGENTS COMMUNICATE (the basics first):
    - SHARED STATE — agents read/write a common state object (LangGraph). Simple,
      in-process.
    - MESSAGE PASSING — agents send messages to each other (AutoGen chat).
    - HANDOFFS — one agent transfers control + context to another.
    - BLACKBOARD — a shared store agents post to and read from.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MCP — MODEL CONTEXT PROTOCOL (Anthropic, 2024):
    An open standard for how an LLM/agent connects to TOOLS and DATA SOURCES.
    "USB-C for AI" — one standard interface instead of custom glue per tool.

    Architecture:
    - MCP HOST/CLIENT — the AI app (the agent) that wants to use tools.
    - MCP SERVER — exposes tools, resources, and prompts over the protocol.
    - The client discovers what a server offers and calls it in a standard way.

    Why it matters (the value — your past interview point, sharpened):
    - N+M instead of N*M integrations: write a tool server ONCE, any
      MCP-compatible agent can use it. No rewriting connectors per model/app.
    - Standard discovery + invocation + auth for tools.
    YOUR EDGE: "I built an MCP server exposing ERP endpoints as tools" — that's
    direct, hands-on MCP experience the JD asks for.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

A2A — AGENT2AGENT PROTOCOL (Google, 2025):
    An open standard for how AUTONOMOUS AGENTS talk to OTHER AGENTS (peer
    collaboration), especially across different vendors/frameworks/orgs.

    THE KEY DISTINCTION (interviewers will test this):
    - MCP = agent <-> TOOLS/DATA (vertical: an agent reaching down to capabilities).
    - A2A = agent <-> AGENT (horizontal: agents collaborating as peers).
    They're COMPLEMENTARY, not competing. An agent uses MCP to access tools AND
    A2A to delegate to other agents.

    A2A core concepts:
    - AGENT CARD — a JSON descriptor advertising an agent's identity, skills,
      and endpoint (so other agents can DISCOVER what it can do).
    - TASK — a unit of work one agent sends another, with a lifecycle (states).
    - MESSAGES/ARTIFACTS — the content exchanged + the results produced.
    - Built on web standards (HTTP, JSON-RPC, SSE) for interoperability,
      supports long-running tasks and streaming updates.

INTERVIEW ANSWER:
    "Agents communicate via shared state, message passing, or handoffs within a
    framework. Across the ecosystem there are two emerging open standards. MCP,
    the Model Context Protocol, standardizes how an agent connects to tools and
    data — think USB-C for AI — so you write a tool server once and any
    MCP-compatible agent can use it, turning N-times-M custom integrations into
    N-plus-M. I've built an MCP server exposing our ERP endpoints as tools. A2A,
    the Agent2Agent protocol, standardizes how autonomous agents talk to each
    other as peers, even across vendors — each agent publishes an Agent Card
    advertising its skills and endpoint, and they exchange tasks over web
    standards. The mental model: MCP is vertical, agent-to-tools; A2A is
    horizontal, agent-to-agent. They're complementary — an agent uses MCP to
    act and A2A to collaborate."
'''


# =================================================================================
# SECTION 6: FRAMEWORK DEEP COMPARISON (LangGraph / AutoGen / CrewAI / ADK)
# =================================================================================
'''
The JD names LangChain, AutoGen, CrewAI, Google ADK. You know LangGraph — be
able to COMPARE all four and justify a choice. This is a lead-level question.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LANGGRAPH (LangChain) — your home framework:
    Model: a STATE GRAPH — nodes (functions) + edges (incl. conditional) +
    shared state, with checkpointing. You explicitly design the control flow.
    Strengths: precise control, loops + branching, durable state/checkpoints,
               human-in-the-loop, streaming, great for COMPLEX controlled
               workflows. Production-grade.
    Trade-off: more code/explicit wiring than higher-level frameworks.
    Use: when you need fine control over the flow (like DocSage's grade ->
         generate -> validate -> rewrite loop).

AUTOGEN (Microsoft):
    Model: CONVERSATIONAL multi-agent — agents talk to each other (and to a
    user proxy) in a chat. Strong for code-generation + tool-use agents that
    converse to solve a task; "group chat" with a manager.
    Strengths: flexible agent conversations, good for research/experimentation,
               strong code-execution agents.
    Trade-off: conversational flow can be less predictable/controllable;
               can loop. (AutoGen 0.4+ moved to an event-driven actor model.)
    Use: collaborative/conversational multi-agent, code tasks.

CREWAI:
    Model: ROLE-BASED crews — you define Agents (role, goal, backstory), Tasks,
    and a Process (sequential or hierarchical). High-level + opinionated.
    Strengths: FAST to build, intuitive "team of roles" mental model, readable.
    Trade-off: less low-level control than LangGraph; abstraction can limit
               complex custom flows.
    Use: quick role-based pipelines ("researcher + writer + editor").

GOOGLE ADK (Agent Development Kit, 2025):
    Model: Google's open-source framework for building + deploying agents,
    integrates with Vertex AI and the A2A protocol; supports multi-agent
    hierarchies and tool use.
    Strengths: Google Cloud / Gemini integration, A2A-native, production
               deployment story on Vertex.
    Trade-off: newer, Google-ecosystem-leaning.
    Use: GCP/Vertex shops, A2A-based multi-agent.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE COMPARISON TABLE:

    FRAMEWORK   ABSTRACTION  CONTROL    BEST FOR
    LangGraph   Low          High       Complex controlled workflows, prod
    AutoGen     Medium       Medium     Conversational/code multi-agent
    CrewAI      High         Low-Med    Fast role-based crews, simple pipelines
    Google ADK  Medium       Medium     GCP/Vertex + A2A multi-agent

    RULE OF THUMB: control vs speed-to-build. LangGraph = max control,
    CrewAI = max speed, AutoGen = conversational, ADK = Google/A2A.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

INTERVIEW ANSWER:
    "I use LangGraph most — it models the workflow as a state graph with nodes,
    conditional edges, and durable checkpoints, so I get precise control over
    loops, branching, and human-in-the-loop; that's how I built DocSage's
    grade-generate-validate-rewrite flow. AutoGen is conversational multi-agent —
    agents chat to solve a task, strong for code generation, though the chat
    flow is less predictable. CrewAI is high-level and role-based — you define
    agents with roles and tasks and a sequential or hierarchical process; it's
    the fastest to build but gives less low-level control. Google's ADK is
    newer, integrates with Vertex AI and is A2A-native, good for GCP shops. The
    trade-off axis is control versus speed-to-build: LangGraph for complex
    controlled production flows, CrewAI for quick role-based pipelines, AutoGen
    for conversational collaboration, ADK for the Google ecosystem."
'''


# =================================================================================
# SECTION 7: AGENT MEMORY & STATE MANAGEMENT
# =================================================================================
'''
The JD: "agent state management" and "long-term memory and knowledge bases."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TYPES OF MEMORY:
    1. SHORT-TERM / WORKING MEMORY — the current conversation/task context
       (the messages in state). Bounded by the context window.
    2. LONG-TERM MEMORY — persists across sessions:
       - SEMANTIC: facts/knowledge (often a VECTOR DB — RAG knowledge base).
       - EPISODIC: past interactions/experiences (what happened before).
       - PROCEDURAL: learned skills/instructions (often in the prompt/tools).
    3. STATE — the structured working data flowing through the graph
       (documents, intermediate results, flags) — your AgentState TypedDict.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MANAGING THE CONTEXT WINDOW (a real constraint):
    Conversations/tasks grow past the context limit. Strategies:
    - SLIDING WINDOW — keep the last N messages (DocSage keeps last ~10).
    - SUMMARIZATION — periodically summarize old turns into a compact memory.
    - RETRIEVAL — store history in a vector DB; retrieve only relevant past
      context per turn (memory as RAG).
    - STRUCTURED STATE — keep key facts in fields, not raw transcript.

PERSISTENCE & CHECKPOINTING:
    - LangGraph MemorySaver + thread_id = per-conversation persistent state;
      enables resume, multi-turn memory, and human-in-the-loop pauses.
    - Production: a durable checkpointer (Postgres/Redis) instead of in-memory.

THE add_messages REDUCER (your DocSage detail — a great concrete point):
    State messages use an append reducer so nodes ADD to history instead of
    OVERWRITING it — without it, the agent loses context between steps.

INTERVIEW ANSWER:
    "I distinguish short-term working memory — the current context window — from
    long-term memory, which can be semantic facts in a vector DB, episodic past
    interactions, or procedural instructions. The structured graph state holds
    the working data. Since context windows are bounded, I manage growth with a
    sliding window, periodic summarization, or retrieval — storing history in a
    vector store and pulling only relevant past context, essentially memory as
    RAG. For persistence I use a checkpointer keyed by a thread ID — in-memory
    for dev, Postgres or Redis in production — which gives multi-turn memory,
    resumability, and human-in-the-loop pauses. One concrete detail: the state's
    messages use an append reducer so each node adds to history rather than
    overwriting it."
'''


# =================================================================================
# SECTION 8: DEBUGGING AGENTIC LOOPS (the JD calls this out)
# =================================================================================
'''
The JD literally says "debugging agentic loops." Most candidates can't answer
this well — own it.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHY AGENTS ARE HARD TO DEBUG:
    Non-deterministic (LLM outputs vary), multi-step (failure can be anywhere
    in a long chain), and emergent (multi-agent interactions). A bug might be
    in the prompt, tool, routing logic, or the model's reasoning.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

COMMON AGENTIC FAILURE MODES:
    1. INFINITE / RUNAWAY LOOPS — agent keeps calling tools without converging
       (e.g., rewrite -> retrieve -> rewrite forever).
       FIX: recursion/iteration LIMIT (DocSage uses recursion_limit=20),
            loop detection, max-token budget.
    2. WRONG TOOL SELECTION — agent picks the wrong tool / bad arguments.
       FIX: clearer tool descriptions, few-shot examples, structured args
            (Pydantic), a routing classifier.
    3. HALLUCINATED TOOL CALLS / ARGS — calls a non-existent tool or malformed args.
       FIX: validate tool calls, constrain with the tool schema, retry.
    4. CONTEXT LOSS — forgets earlier info (overwritten state / window overflow).
       FIX: add_messages reducer, summarization, retrieval memory.
    5. ERROR PROPAGATION (multi-agent) — one agent's bad output poisons the next.
       FIX: validation between agents, guardrails, a reviewer agent.
    6. GETTING STUCK / NO PROGRESS — repeats the same failing action.
       FIX: detect repetition, change strategy, escalate to human.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE DEBUGGING TOOLKIT:
    - TRACING/OBSERVABILITY — LangSmith / LangFuse: see every step, prompt,
      tool call, token count, and decision. This is the #1 tool — you can't
      debug what you can't see.
    - STEP-LEVEL LOGGING — log each node's input/output (DocSage logs each node).
    - REPLAY — checkpointing lets you replay a run from a step.
    - EVAL SETS — reproduce failures on a golden set after a fix.
    - GUARDRAILS — limits, validation, timeouts as safety nets.

INTERVIEW ANSWER:
    "Agents are hard to debug because they're non-deterministic, multi-step,
    and emergent. The common failures: runaway loops, which I bound with a
    recursion limit and loop detection; wrong tool selection, fixed with clearer
    tool descriptions and structured arguments; context loss, fixed with an
    append reducer, summarization, or retrieval memory; and in multi-agent
    systems, error propagation, where I add validation between agents. The
    number-one tool is tracing — LangSmith or LangFuse — so I can see every
    prompt, tool call, and decision in a run; you can't debug what you can't
    see. I add step-level logging, use checkpoint replay to reproduce a failing
    step, and put guardrails — iteration limits, schema validation, timeouts —
    as safety nets. Then I lock the fix in with a golden eval set."
'''


# =================================================================================
# SECTION 9: PRODUCTION CONCERNS (cost, latency, reliability, observability)
# =================================================================================
'''
A LEAD owns production. Multi-agent systems multiply cost/latency/failure
surface — show you think about this.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

COST:
    Every agent step = an LLM call = tokens = money. Multi-agent multiplies it.
    Controls: use SMALLER models for simple steps (routing, grading) and big
    models only where needed; cap iterations; cache repeated calls/prompts;
    prune context (don't resend the whole history); batch where possible;
    track tokens-per-task as a core metric.

LATENCY:
    Multi-step + multi-agent = slow (each LLM call is seconds).
    Controls: PARALLELIZE independent steps (asyncio.gather — Phase 2);
    STREAM partial results so users see progress; use faster models for
    latency-critical steps; cache; set per-step timeouts.

RELIABILITY:
    LLM/tool calls fail and throttle. Controls: timeouts + retries with backoff
    (Phase 3), circuit breakers, fallbacks (cheaper model / cached / human),
    graceful degradation, idempotency for repeated actions.

GOVERNANCE & SAFETY:
    Agents take ACTIONS (send email, write DB) — risk is real.
    Controls: PER-TOOL PERMISSIONS (least privilege), HUMAN-IN-THE-LOOP
    approval for high-risk actions, input/output guardrails, sandboxing tool
    execution, full AUDIT LOGS of what each agent did and why.

OBSERVABILITY:
    Trace every run (LangSmith), dashboards for success rate, steps/task,
    cost/task, latency p95, tool-error rate, loop frequency; alert on drift.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

INTERVIEW ANSWER:
    "Multi-agent systems multiply cost, latency, and failure surface, so I
    engineer for all three. Cost: small models for simple steps like routing
    and grading, big models only where needed, cap iterations, cache, prune
    context, and track tokens per task. Latency: parallelize independent steps
    with asyncio.gather, stream partial results, and set per-step timeouts.
    Reliability: timeouts, retries with backoff, circuit breakers, and
    fallbacks to a cheaper model, cache, or a human. Because agents take real
    actions, governance matters — per-tool least-privilege permissions,
    human-in-the-loop approval for risky actions, guardrails, and audit logs.
    And I trace every run with LangSmith and dashboard success rate, cost, and
    latency."
'''


# =================================================================================
# SECTION 10: REAL-WORLD DESIGN WALKTHROUGH (end-to-end)
# =================================================================================
'''
A lead must DESIGN a multi-agent system on the spot. Here's a worked example
you can adapt to any "design an agentic system for X" prompt.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PROMPT: "Design a multi-agent system that handles enterprise customer support
tickets end to end."

STEP 1 — CLARIFY: volume, channels, what actions are allowed (refunds?),
    latency target, escalation rules, data sources.

STEP 2 — TOPOLOGY: Orchestrator-Worker (a supervisor routes + delegates).

STEP 3 — THE AGENTS (specialists):
    - SUPERVISOR/ROUTER — classifies the ticket, routes, aggregates, decides
      escalation.
    - KNOWLEDGE AGENT (RAG) — answers from docs/KB via a vector DB retriever.
    - ACCOUNT AGENT — reads order/account data via tools (MCP server to ERP).
    - ACTION AGENT — performs allowed actions (refund, reset) — HITL-gated.
    - ESCALATION AGENT — drafts a human handoff with summary + context.

STEP 4 — FLOW:
    Ticket -> Supervisor classifies -> route to Knowledge / Account / Action
    -> agent uses tools (RAG retriever, ERP via MCP) -> reflection/validation
    -> Supervisor checks confidence -> answer OR escalate to human (HITL).

STEP 5 — DATA/MEMORY: vector DB for the KB (RAG), conversation state per
    ticket (checkpointer + ticket_id), customer history as long-term memory.

STEP 6 — PROTOCOLS: MCP servers expose ERP/CRM tools uniformly; if other
    teams' agents are involved, A2A for cross-agent delegation.

STEP 7 — GUARDRAILS: action agent needs human approval for refunds over a
    threshold; per-tool permissions; PII redaction; audit log.

STEP 8 — PRODUCTION: async fan-out where possible, smaller model for routing,
    timeouts/retries on tools, LangSmith tracing, eval set of past tickets,
    success-rate + cost + escalation-rate dashboards.

STEP 9 — TRADE-OFFS: "Start with a single RAG agent + escalation; add
    specialist agents only when ticket types justify the added cost and
    complexity. Least autonomy that solves it."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE REUSABLE DESIGN FRAMEWORK (for ANY agentic design question):
    Clarify -> Topology -> Agents/roles -> Flow -> Data/memory -> Protocols
    -> Guardrails -> Production (cost/latency/reliability/observability)
    -> Trade-offs (start simple, scale autonomy with need).

INTERVIEW POINT:
    "I'd start by clarifying scope and allowed actions, pick an
    orchestrator-worker topology with a supervisor routing to specialist
    agents — knowledge, account, action, escalation — where the action agent
    is human-in-the-loop gated for risky operations. RAG over a vector DB for
    knowledge, an MCP server exposing ERP tools, conversation state checkpointed
    per ticket. Guardrails with per-tool permissions and audit logs, async
    parallelism and small models for cost, and LangSmith tracing. Critically,
    I'd start with a single RAG agent plus escalation and add specialist agents
    only when the ticket mix justifies the cost — least autonomy that works."
'''


# =================================================================================
# SECTION 11: INTERVIEW Q&A (lead-level)
# =================================================================================
'''
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q1. Workflow vs agent?
A:  "Workflow follows a fixed developer-defined path; an agent dynamically
    decides its own next step. I use the least autonomy that solves the problem."

Q2. What is the ReAct pattern?
A:  "Thought-Action-Observation loop: the agent reasons about a tool, calls it,
    observes the result, repeats until it can answer."

Q3. When multiple agents vs one?
A:  "Multiple for specialization, separation of concerns, or parallelism — but
    they add cost and coordination overhead, so only when one agent isn't
    enough."

Q4. Orchestration topologies?
A:  "Orchestrator-worker (supervisor delegates) is my default; hierarchical for
    nested trees; peer-to-peer for open collaboration; hub-and-spoke router to
    pick one expert."

Q5. MCP vs A2A?
A:  "MCP is agent-to-tools — vertical, standardizes tool/data access. A2A is
    agent-to-agent — horizontal, standardizes peer collaboration across vendors.
    Complementary: MCP to act, A2A to delegate."

Q6. LangGraph vs CrewAI vs AutoGen?
A:  "LangGraph = low-level state graph, max control, production workflows.
    CrewAI = high-level role-based crews, fast to build. AutoGen =
    conversational multi-agent. Control vs speed-to-build."

Q7. How do you manage agent memory?
A:  "Short-term context window with sliding window/summarization; long-term in
    a vector DB (semantic), episodic history, structured state; checkpointer +
    thread_id for persistence."

Q8. How do you stop infinite agent loops?
A:  "Recursion/iteration limit, loop/repetition detection, token budget, and a
    termination condition; escalate to human if no progress."

Q9. How do you debug an agent?
A:  "Tracing first (LangSmith) to see every step; step-level logging,
    checkpoint replay to reproduce, guardrails as safety nets, and a golden
    eval set to lock the fix."

Q10. How do you handle agent reliability in production?
A:  "Timeouts, retries with backoff, circuit breakers, fallbacks to a cheaper
    model or human, idempotency for actions, and per-tool permissions with HITL
    for risky operations."

Q11. What is agent lifecycle management?
A:  "Run lifecycle: init -> reason/act/observe loop -> terminate -> persist.
    Operational: version, deploy with canaries, monitor, eval, govern, improve."

Q12. How do you control multi-agent cost?
A:  "Smaller models for simple steps, cap iterations, cache, prune context,
    parallelize, and track tokens-per-task as a core metric."
'''


# =================================================================================
# SECTION 12: GOLDEN LESSONS
# =================================================================================
'''
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GOLDEN LESSON 1 — USE THE LEAST AUTONOMY THAT SOLVES THE PROBLEM.
    Workflows > agents when possible: reliable, debuggable, cheaper. This is
    THE senior framing for agentic design.

GOLDEN LESSON 2 — WORKFLOW (fixed path) vs AGENT (decides path). Know the line.

GOLDEN LESSON 3 — ReAct IS THE FOUNDATION; reflection + routing layer on top.
    Connect to DocSage: ReAct + routing + reflection in one graph.

GOLDEN LESSON 4 — ORCHESTRATOR-WORKER IS THE DEFAULT TOPOLOGY.
    Controllable, debuggable, maps to assigning work to a human team.

GOLDEN LESSON 5 — MCP (agent<->tools, vertical) vs A2A (agent<->agent, horizontal).
    Complementary, not competing. You have hands-on MCP — lead with it.

GOLDEN LESSON 6 — CONTROL vs SPEED IS THE FRAMEWORK AXIS.
    LangGraph (control) / CrewAI (speed) / AutoGen (conversation) / ADK (Google+A2A).

GOLDEN LESSON 7 — MEMORY: short-term window + long-term vector DB + checkpoint.
    Manage the context window (window/summarize/retrieve). add_messages reducer.

GOLDEN LESSON 8 — ALWAYS BOUND THE LOOP.
    Recursion limit + termination conditions. Runaway loops are the #1 agent bug.

GOLDEN LESSON 9 — TRACING IS HOW YOU DEBUG AGENTS.
    LangSmith/LangFuse. You can't debug what you can't see.

GOLDEN LESSON 10 — LEAD = OWN COST, LATENCY, RELIABILITY, GOVERNANCE.
    Small models for simple steps, parallelize, retries/fallbacks, HITL +
    per-tool permissions + audit logs for actions.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ONE-LINE CHEAT SHEET:
    Agent = reason + tools + memory + autonomy (decides next step).
    Workflow (fixed) vs Agent (dynamic); use least autonomy that works.
    Patterns: ReAct, Plan-Execute, Reflection, Router.
    Topologies: Orchestrator-Worker (default), Hierarchical, Peer-to-Peer, Hub-Spoke.
    Protocols: MCP = agent<->tools (vertical); A2A = agent<->agent (horizontal).
    Frameworks: LangGraph(control) / CrewAI(speed) / AutoGen(chat) / ADK(Google+A2A).
    Memory: window + summarize + retrieve; checkpoint + thread_id.
    Debug: bound loops + tracing (LangSmith) + replay + guardrails + evals.
    Prod: small models, parallelize, retries/fallbacks, HITL, audit, observe.
    Design: Clarify->Topology->Agents->Flow->Data->Protocols->Guardrails->Prod->Trade-offs.
'''


# =================================================================================
# RUN SUMMARY
# =================================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 6 — MULTI-AGENT SYSTEMS & FRAMEWORKS")
    print("=" * 70)
    print()
    print("AGENT = reason + tool use + memory + autonomy (decides its next step)")
    print("WORKFLOW (fixed path) vs AGENT (dynamic) -> use LEAST autonomy that works")
    print()
    print("PATTERNS: ReAct | Plan-Execute | Reflection | Router")
    print("TOPOLOGIES: Orchestrator-Worker (default) | Hierarchical | P2P | Hub-Spoke")
    print()
    print("PROTOCOLS (the JD names both):")
    print("  MCP = agent <-> TOOLS   (vertical)   <- you have hands-on experience")
    print("  A2A = agent <-> AGENT   (horizontal) <- complementary, not competing")
    print()
    print("FRAMEWORKS (control vs speed):")
    print("  LangGraph(control) | CrewAI(speed) | AutoGen(chat) | Google ADK(A2A)")
    print()
    print("DEBUG AGENTS: bound loops + tracing(LangSmith) + replay + guardrails + evals")
    print("DESIGN FRAMEWORK: Clarify->Topology->Agents->Flow->Data->Protocols")
    print("                  ->Guardrails->Production->Trade-offs")
    print()
    print("=" * 70)
    print("Next: Phase 7 — System Design for AI/Agentic Systems")
    print("=" * 70)
