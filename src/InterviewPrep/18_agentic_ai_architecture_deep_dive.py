"""
===================================================================================
AGENTIC AI ARCHITECTURE DEEP DIVE — Senior AI Architect Mastery
===================================================================================

This lesson covers AGENTIC AI ARCHITECTURE at the depth a 20+ year experienced
AI architect would expect. Required for: PwC Round 2, JPMC architect rounds,
and any senior GenAI/AI Engineering role.

WHY THIS MATTERS:
    Round 2 architect interviews don't test syntax or library knowledge.
    They test SYSTEM DESIGN — can you reason about layers, flows, patterns,
    and tradeoffs? Can you draw it on a whiteboard?

SECTIONS:
    1.  What is Agentic AI? (vs LLMs vs Workflows vs Agents)
    2.  The 5 Architectural Layers (Perception, Reasoning, Memory, Action, Learning)
    3.  Block Diagram of a Production Agentic System (Master Diagram)
    4.  Data Flow Diagram — Request to Response (End-to-End)
    5.  Multi-Agent Patterns — Orchestrator/Peer/Hierarchical/Hub-Spoke
    6.  Component Deep Dive — Orchestrator (LangGraph)
    7.  Component Deep Dive — Tool Registry
    8.  Component Deep Dive — Memory Layer (Short, Long, Vector)
    9.  Component Deep Dive — LLM Gateway (Multi-Provider)
    10. Component Deep Dive — Observability (Logging, Tracing, Metrics)
    11. Component Deep Dive — Safety & Governance (Guardrails, HITL)
    12. Production Deployment Architecture (K8s, Queues, Microservices)
    13. How to Draw These on a Whiteboard (Step by Step)
    14. 25+ Architect-Level Interview Q&A
    15. GOLDEN LESSONS

This is what a senior architect would draw and explain.
===================================================================================
"""


# =================================================================================
# SECTION 1: WHAT IS AGENTIC AI? (Definitions Matter)
# =================================================================================
"""
Senior interviewers test if you can articulate the DIFFERENCE between these:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LEVEL 1: PURE LLM (no agency)
    User sends a prompt → LLM generates a response.
    No tool calls, no loops, no decisions beyond text generation.

    Example: ChatGPT answering "What's the capital of France?"
    Limitation: Can't do anything beyond what's in its training data.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LEVEL 2: WORKFLOW (deterministic LLM chain)
    Predefined sequence of LLM calls. Same flow every time.

    Example: prompt → translate → summarize → format → output
    The DEVELOPER decides the steps. LLM just executes them.
    Tools: LangChain chains (LCEL pipe operator)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LEVEL 3: AGENT (LLM with tool use)
    LLM decides what tools to call and when.

    Example: User asks "What's the weather in Paris?"
    Agent: "I should call the weather API."
    Agent: Calls weather_tool("Paris") → gets data → generates response.

    The LLM ITSELF decides which tools to use.
    Tools: LangChain agents, OpenAI function calling

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LEVEL 4: AGENTIC AI (autonomous, multi-step, goal-directed)
    Agent (or multiple agents) plans, reasons, acts, observes, replans.
    Can run for extended periods. Has memory. Adapts to outcomes.

    Example: "Plan my Europe trip"
    Agent: Plans → searches flights → checks hotels → coordinates dates →
           handles failures → adjusts plan → returns complete itinerary.

    KEY CHARACTERISTICS:
    1. AUTONOMY: Agent decides what to do next
    2. PLANNING: Breaks goals into sub-tasks
    3. TOOL USE: Calls APIs, databases, other agents
    4. MEMORY: Remembers past interactions
    5. ADAPTATION: Changes plan based on results
    6. REASONING: Chain-of-thought decision making

    Tools: LangGraph, AutoGen, CrewAI, MCP

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE EVOLUTION:

    LLM:        "I generate text."
    Workflow:   "I execute steps you defined."
    Agent:      "I pick tools to use."
    Agentic:    "I plan, act, reflect, and adapt to achieve goals."

INTERVIEW ANSWER:
    "Agentic AI is autonomous, goal-directed AI that can plan, reason, act,
    and adapt. It's built on LLMs but adds 5 key capabilities: autonomy,
    planning, tool use, memory, and adaptation. Unlike a workflow which
    follows a fixed sequence, an agentic system DECIDES what to do next at
    each step. LangGraph and AutoGen are the dominant frameworks for building
    these systems in 2026."
"""


# =================================================================================
# SECTION 2: THE 5 ARCHITECTURAL LAYERS
# =================================================================================
"""
Every Agentic AI system has 5 conceptual LAYERS. This is the FOUNDATION that
senior interviewers expect you to know.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE 5 LAYERS (memorize this — it's THE master framework):

    ┌─────────────────────────────────────────────────────────────────┐
    │  LAYER 5: LEARNING (improve over time)                           │
    │  - Feedback collection                                           │
    │  - Model fine-tuning / prompt optimization                       │
    │  - A/B testing                                                   │
    └──────────────────────────┬──────────────────────────────────────┘
                               ↑
    ┌─────────────────────────────────────────────────────────────────┐
    │  LAYER 4: ACTION (execute decisions in the world)                │
    │  - Tool calls (APIs, DBs, services)                              │
    │  - File operations                                               │
    │  - Communication (email, slack, etc.)                            │
    └──────────────────────────┬──────────────────────────────────────┘
                               ↑
    ┌─────────────────────────────────────────────────────────────────┐
    │  LAYER 3: MEMORY (store and recall)                              │
    │  - Short-term (current conversation)                             │
    │  - Long-term (user history)                                      │
    │  - Vector store (semantic memory)                                │
    │  - Knowledge graph (relational memory)                           │
    └──────────────────────────┬──────────────────────────────────────┘
                               ↑
    ┌─────────────────────────────────────────────────────────────────┐
    │  LAYER 2: REASONING (plan and decide)                            │
    │  - LLM-based reasoning                                           │
    │  - Chain-of-thought                                              │
    │  - Tool selection                                                │
    │  - Plan generation and decomposition                             │
    └──────────────────────────┬──────────────────────────────────────┘
                               ↑
    ┌─────────────────────────────────────────────────────────────────┐
    │  LAYER 1: PERCEPTION (sense the input)                           │
    │  - User input (text, voice, images)                              │
    │  - Document parsing (PDFs, web pages)                            │
    │  - Sensor data (in robotics)                                     │
    └─────────────────────────────────────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LAYER 1: PERCEPTION (Input Layer)
    What it does: Receives and processes input from the user/environment.

    Components:
    - Multimodal input handlers (text, voice, image, video)
    - Document parsers (PDF, DOCX, web scraping)
    - Speech-to-text (for voice agents)
    - Computer vision (for visual agents)

    Engineering concerns:
    - Input validation (prompt injection detection)
    - Format normalization (convert all input to text/structured data)
    - PII detection and redaction

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LAYER 2: REASONING (Brain Layer)
    What it does: Thinks, plans, decides what to do next.

    Components:
    - LLM (the core reasoning engine)
    - Prompt engineering layer (system prompts, templates)
    - Plan generator (breaks goals into steps)
    - Tool selector (decides which tool to call)
    - Reflection / self-correction logic

    Engineering concerns:
    - LLM provider abstraction (multi-provider)
    - Cost optimization (model selection per task)
    - Reasoning patterns: ReAct, Plan-and-Execute, Tree of Thoughts

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LAYER 3: MEMORY (Knowledge Layer)
    What it does: Stores and retrieves context, history, and knowledge.

    Components:
    - Short-term memory (conversation buffer)
    - Long-term memory (user profile, preferences)
    - Vector store (semantic search of documents)
    - Knowledge graph (entity relationships)
    - Episodic memory (past interactions)

    Engineering concerns:
    - Memory compression (summarization for token limits)
    - Multi-tenant isolation (per-user memory)
    - Persistence (PostgreSQL, Redis, vector DBs)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LAYER 4: ACTION (Execution Layer)
    What it does: Executes decisions through tools, APIs, or external systems.

    Components:
    - Tool registry (catalog of available tools)
    - API gateway (calls external services)
    - File system operations
    - Communication channels (email, slack)
    - Database operations (read/write)

    Engineering concerns:
    - Authorization (RBAC: which tools can which agents call?)
    - Rate limiting (prevent runaway agents)
    - Error handling (retries, fallbacks)
    - Audit logging (every action recorded)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LAYER 5: LEARNING (Improvement Layer)
    What it does: Continuously improves the system over time.

    Components:
    - User feedback collection (👍/👎)
    - Quality metrics tracking (RAGAS metrics)
    - A/B testing infrastructure
    - Prompt optimization loops
    - Model fine-tuning pipelines

    Engineering concerns:
    - Eval datasets (golden Q&A pairs)
    - Drift detection
    - CI/CD for prompts and models

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

INTERVIEW ANSWER:
    "Every agentic AI system has 5 layers: Perception (handles input), Reasoning
    (LLM-based planning and decisions), Memory (short-term, long-term, vector,
    graph), Action (tool calls and execution), and Learning (continuous
    improvement). When designing a system, I think about each layer separately —
    what components, what interfaces, what concerns. This separation of
    concerns is what makes systems maintainable and scalable."
"""


# =================================================================================
# SECTION 3: BLOCK DIAGRAM OF A PRODUCTION AGENTIC SYSTEM
# =================================================================================
"""
THIS IS THE MASTER DIAGRAM. Senior interviewers will ask you to draw something
like this on the whiteboard. Memorize the components and how they connect.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                          USER INTERFACE LAYER                            │
    │              (Web App / Mobile / Slack / Voice / API)                    │
    └────────────────────────────────┬────────────────────────────────────────┘
                                     │ (REST / WebSocket / gRPC)
                                     ↓
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                       API GATEWAY / LOAD BALANCER                        │
    │     (Authentication, Rate Limiting, Request Routing, Caching)           │
    └────────────────────────────────┬────────────────────────────────────────┘
                                     │
                                     ↓
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                     INPUT VALIDATION & GUARDRAILS (Layer 1)              │
    │  - PII detection & redaction                                             │
    │  - Prompt injection detection                                            │
    │  - Topic / scope filtering                                               │
    │  - Input sanitization                                                    │
    └────────────────────────────────┬────────────────────────────────────────┘
                                     │
                                     ↓
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                  ORCHESTRATOR / AGENT FRAMEWORK (Layer 2)                │
    │           [LangGraph StateGraph / AutoGen / CrewAI]                      │
    │                                                                          │
    │   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐         │
    │   │ Planner  │←→  │ Executor │ ←→ │ Critic   │ ←→ │ Reflector│         │
    │   │ Agent    │    │ Agent    │    │ Agent    │    │ Agent    │         │
    │   └──────────┘    └──────────┘    └──────────┘    └──────────┘         │
    │   (decides    (calls tools,   (validates       (improves                │
    │    plan)       executes)       output)         next time)               │
    └────────────────────────────────┬────────────────────────────────────────┘
                                     │
                ┌────────────────────┼─────────────────────┐
                ↓                    ↓                     ↓
    ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
    │  TOOL REGISTRY   │  │  MEMORY LAYER    │  │  LLM GATEWAY     │
    │  (Layer 4)       │  │  (Layer 3)       │  │  (Layer 2)       │
    │                  │  │                  │  │                  │
    │  - REST APIs     │  │  - Short-term    │  │  - OpenAI        │
    │  - Databases     │  │  - Long-term     │  │  - Anthropic     │
    │  - File systems  │  │  - Vector store  │  │  - Bedrock       │
    │  - MCP servers   │  │  - Knowledge     │  │  - Groq, Gemini  │
    │  - Functions     │  │    graph         │  │  - Local Ollama  │
    └──────────────────┘  └──────────────────┘  └──────────────────┘
                ↓                    ↓                     ↓
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                  OUTPUT VALIDATION & GUARDRAILS                          │
    │  - PII scanning in output                                                │
    │  - Hallucination check (cross-ref with retrieved docs)                   │
    │  - Format validation                                                     │
    │  - Content moderation                                                    │
    └────────────────────────────────┬────────────────────────────────────────┘
                                     │
                                     ↓
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    OBSERVABILITY & GOVERNANCE                            │
    │  - Logging (structured, with traces)                                     │
    │  - Metrics (latency, cost, quality)                                      │
    │  - Audit trail (every action logged)                                     │
    │  - HITL approval gates (for sensitive actions)                           │
    │  - Cost tracking per request                                             │
    └────────────────────────────────┬────────────────────────────────────────┘
                                     │
                                     ↓
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                            RESPONSE                                      │
    │           (back to user via API gateway / streaming)                    │
    └─────────────────────────────────────────────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

KEY DESIGN PRINCIPLES:

    1. SEPARATION OF CONCERNS — each block does ONE thing well
    2. LOOSELY COUPLED — components communicate via defined interfaces
    3. FAULT ISOLATION — failure in one component doesn't crash others
    4. SCALABILITY — each component can scale independently
    5. OBSERVABILITY — everything is logged, traced, monitored
    6. SECURITY — guardrails at multiple layers (defense in depth)

INTERVIEW ANSWER:
    "A production agentic system has these key blocks: API gateway with
    authentication and rate limiting, input guardrails for PII and prompt
    injection, the orchestrator (LangGraph) that runs agents, three core
    services — tool registry, memory layer, LLM gateway — that the orchestrator
    coordinates, output guardrails, and observability/governance underneath
    everything. Each block has a single responsibility and communicates via
    defined interfaces. This separation makes it scalable, observable, and
    secure."
"""


# =================================================================================
# SECTION 4: DATA FLOW DIAGRAM — Request to Response
# =================================================================================
"""
EXAMPLE: User asks "Cancel my last order and refund me"

This is the END-TO-END FLOW through every layer:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    STEP 1: USER REQUEST
        User sends: "Cancel my last order and refund me"
        (via web chat, mobile, or voice)

    STEP 2: API GATEWAY
        - Authenticate (JWT token validation)
        - Rate limit check (per user/IP)
        - Route to backend service

    STEP 3: INPUT GUARDRAILS
        - Detect PII (no credit cards in message)
        - Detect prompt injection (no "ignore previous instructions")
        - Validate scope (is this within agent's domain?)
        - Result: ALLOW

    STEP 4: ORCHESTRATOR (LangGraph)
        Receives: user_message + conversation_history + user_context
        Loads: short-term memory (this conversation)
        Loads: long-term memory (user_id → preferences)

    STEP 5: PLANNER AGENT
        LLM thinks: "User wants to: (a) find last order (b) cancel it
                    (c) initiate refund. I need 3 tool calls."
        Generates plan:
            1. Call get_last_order(user_id)
            2. Call cancel_order(order_id)
            3. Call initiate_refund(order_id)

    STEP 6: TOOL REGISTRY → TOOL EXECUTION
        Step 6a: Calls get_last_order tool
            → Returns: order_id=12345, items=[...], total=$150
        Step 6b: Calls cancel_order(12345)
            → Returns: success, status="cancelled"

        Step 6c: PAUSE — initiate_refund requires HITL!
            → Sends event: "Refund of $150 pending approval"
            → Notifies admin via Slack
            → Waits for approval

        (Admin approves)

        Step 6d: Calls initiate_refund(12345)
            → Returns: success, refund_id=R789

    STEP 7: MEMORY UPDATE
        - Save to long-term: "User cancelled order 12345 on 2026-05-20"
        - Update conversation history
        - Add to user's interaction log

    STEP 8: GENERATION
        LLM combines all results into a natural response:
        "I've cancelled your order #12345 ($150) and initiated a refund.
        You'll receive it in 5-7 business days. Refund ID: R789."

    STEP 9: OUTPUT GUARDRAILS
        - Scan for accidental PII leakage
        - Verify no hallucinated numbers (check against tool results)
        - Format validation (proper response structure)

    STEP 10: OBSERVABILITY
        - Log: full request trace with all tool calls
        - Metrics: latency=4.2s, tokens=850, cost=$0.012
        - Audit: refund_R789 approved by admin_A

    STEP 11: RESPONSE TO USER
        Streamed back via WebSocket
        Total time: 4.2 seconds (3.5s for HITL approval)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

KEY OBSERVATIONS:

    1. EVERY INTERACTION goes through the SAME flow
    2. HITL is built INTO the flow (not added later)
    3. EVERYTHING is logged for audit
    4. ERRORS at any step have FALLBACKS
    5. STATE is preserved (can resume after crash)

INTERVIEW ANSWER:
    "Let me trace a request — say 'cancel my last order and refund me.' First,
    API gateway authenticates and rate-limits. Input guardrails check for PII
    and prompt injection. Orchestrator loads conversation context and plans
    the workflow: get last order → cancel it → initiate refund. Tools execute
    in sequence. Refund triggers HITL — pauses for admin approval. Memory
    updates with the action. LLM generates natural response. Output guardrails
    verify no PII leakage. Everything logged for audit. Total flow: ~4 seconds.
    Each step has clear responsibilities and graceful failure handling."
"""


# =================================================================================
# SECTION 5: MULTI-AGENT PATTERNS
# =================================================================================
"""
When ONE agent isn't enough, you use MULTIPLE agents. There are 4 standard
patterns. Senior interviewers EXPECT you to know all of them.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PATTERN 1: ORCHESTRATOR-WORKER (Most Common)

    One SUPERVISOR agent delegates to specialized WORKER agents.

                    ┌──────────────────┐
                    │   ORCHESTRATOR   │
                    │   (Supervisor)   │
                    └────────┬─────────┘
                             │
            ┌────────────────┼────────────────┐
            ↓                ↓                ↓
    ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
    │   WORKER 1   │ │   WORKER 2   │ │   WORKER 3   │
    │  (Planner)   │ │ (Researcher) │ │   (Writer)   │
    └──────────────┘ └──────────────┘ └──────────────┘

    HOW IT WORKS:
        1. User request → Orchestrator
        2. Orchestrator analyzes: "This needs research + writing."
        3. Orchestrator → Researcher: "Find facts about X"
        4. Orchestrator → Writer: "Write summary using these facts"
        5. Orchestrator combines results → User

    WHEN TO USE:
        - Tasks have clear sub-tasks
        - You have specialists for different jobs
        - Need centralized control

    EXAMPLE: Customer support
        Orchestrator → Triage agent → Refund agent → Notification agent

    PROS:
        - Clear control flow
        - Easy to debug
        - Specialists are reusable
    CONS:
        - Bottleneck at orchestrator
        - Sequential execution by default

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PATTERN 2: PEER-TO-PEER (Collaborative)

    Agents communicate DIRECTLY without a supervisor. Like a team meeting.

    ┌──────────────┐ ←──────→ ┌──────────────┐
    │   AGENT A    │            │   AGENT B    │
    │  (Coder)     │            │  (Reviewer)  │
    └──────────────┘            └──────────────┘
            ↑                            ↑
            ↓                            ↓
    ┌──────────────┐            ┌──────────────┐
    │   AGENT C    │            │   AGENT D    │
    │  (Tester)    │ ←────────→ │   (Doc)      │
    └──────────────┘            └──────────────┘

    HOW IT WORKS:
        Agents send messages to each other based on need.
        No central coordinator.
        Termination: when consensus or done condition is met.

    WHEN TO USE:
        - Iterative refinement (code review, debate)
        - Equal partners (no clear "leader")
        - Emergent behavior expected

    EXAMPLE: Code generation
        Coder writes code → Reviewer critiques →
        Coder revises → Tester runs tests → Reviewer signs off

    FRAMEWORK: AutoGen excels at this pattern.

    PROS:
        - Flexible, emergent behavior
        - No bottleneck
        - Good for iterative work
    CONS:
        - Hard to debug
        - Can loop forever without proper termination
        - Less predictable

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PATTERN 3: HIERARCHICAL (Tree Structure)

    Agents organized like a corporate hierarchy. Levels of control.

                       ┌──────────────┐
                       │     CEO      │
                       │   (Master)   │
                       └──────┬───────┘
                              │
              ┌───────────────┼───────────────┐
              ↓               ↓               ↓
        ┌──────────┐    ┌──────────┐    ┌──────────┐
        │   VP1    │    │   VP2    │    │   VP3    │
        │  (Mid)   │    │  (Mid)   │    │  (Mid)   │
        └─────┬────┘    └─────┬────┘    └─────┬────┘
              │               │               │
        ┌─────┼─────┐    ┌────┼────┐    ┌─────┼─────┐
        ↓     ↓     ↓    ↓    ↓    ↓    ↓     ↓     ↓
       (workers)   (workers)         (workers)

    HOW IT WORKS:
        Top-level master breaks task into sub-goals.
        Mid-level managers further decompose each sub-goal.
        Workers execute leaf tasks.
        Results flow UP through the hierarchy.

    WHEN TO USE:
        - Complex tasks with many levels
        - Large-scale automation
        - Need for governance at each level

    EXAMPLE: Research project
        Master: "Research climate change impact"
        VP1: "Energy sector" → workers research individually
        VP2: "Agriculture sector" → workers research individually
        VP3: "Healthcare sector" → workers research individually
        Results aggregate up to Master

    PROS:
        - Scales to many agents
        - Clear governance
        - Localized failures
    CONS:
        - Slow for simple tasks
        - Communication overhead

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PATTERN 4: HUB-AND-SPOKE (Centralized Routing)

    Central HUB routes to specialized SPOKES based on request type.

                    ┌──────────────────┐
                    │      HUB         │
                    │   (Router)       │
                    └────────┬─────────┘
                             │
        ┌──────┬─────────────┼─────────────┬──────┐
        ↓      ↓             ↓             ↓      ↓
    ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
    │SPOKE1│ │SPOKE2│ │SPOKE3│ │SPOKE4│ │SPOKE5│
    │ HR   │ │Sales │ │Tech  │ │Legal │ │ Fin  │
    └──────┘ └──────┘ └──────┘ └──────┘ └──────┘

    HOW IT WORKS:
        1. User request → Hub
        2. Hub classifies: "This is HR-related"
        3. Hub routes to HR spoke
        4. Spoke handles request entirely
        5. Returns answer through hub

    WHEN TO USE:
        - Domain specialists (each spoke is an expert)
        - Need clean separation
        - Predictable routing

    EXAMPLE: Enterprise chatbot
        Hub: "Is this HR / Sales / Tech / Legal / Finance?"
        Routes to the right spoke
        Spoke uses its specialized tools

    DIFFERENCE FROM ORCHESTRATOR-WORKER:
        - Orchestrator-Worker: orchestrator may delegate to MULTIPLE workers
        - Hub-Spoke: hub routes to ONE spoke, that spoke handles entire task

    PROS:
        - Clean specialization
        - Easy to add new spokes
        - Independent failure domains
    CONS:
        - Requires good classifier
        - Cross-domain queries are awkward

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DECISION TABLE — When to use which:

    SCENARIO                              BEST PATTERN
    Single task, multiple sub-steps       Orchestrator-Worker
    Iterative refinement (code review)    Peer-to-Peer
    Large complex task with many levels   Hierarchical
    Domain routing (HR/Sales/Tech)        Hub-and-Spoke
    Customer support with specialists     Hub-and-Spoke
    Research with parallel investigation  Hierarchical
    Data analysis pipeline                Orchestrator-Worker
    Open-ended creative work              Peer-to-Peer

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

INTERVIEW ANSWER:
    "There are 4 standard multi-agent patterns. Orchestrator-Worker has a
    supervisor delegating to specialists — most common, easy to debug. Peer-to-Peer
    has agents communicating directly — best for iterative work like code
    review with AutoGen. Hierarchical organizes agents like a corporate tree
    for complex tasks with multiple levels. Hub-and-Spoke routes to domain
    specialists — perfect for enterprise chatbots covering HR/Sales/Tech.
    The choice depends on the task structure: clear sub-tasks → Orchestrator,
    iterative → Peer-to-Peer, multi-level → Hierarchical, domain routing → Hub."
"""


# =================================================================================
# SECTION 6: COMPONENT DEEP DIVE — ORCHESTRATOR (LangGraph)
# =================================================================================
"""
The orchestrator is the BRAIN of the system. It coordinates everything.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHAT THE ORCHESTRATOR DOES:

    1. STATE MANAGEMENT — tracks the agent's current state
    2. NODE EXECUTION — runs each step (agent, tool, validator)
    3. ROUTING — decides what runs next based on current state
    4. ERROR HANDLING — retries, fallbacks, graceful failure
    5. STREAMING — emits intermediate updates to the user
    6. CHECKPOINTING — persists state for crash recovery

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LANGGRAPH ARCHITECTURE (Most Popular):

    KEY ABSTRACTIONS:
    - StateGraph: the workflow definition
    - State: TypedDict shared across nodes
    - Nodes: functions that update state
    - Edges: connections between nodes (fixed or conditional)
    - Checkpointer: persists state to memory/DB
    - Reducer: how state fields are merged (e.g., add_messages)

    Example state:
        class AgentState(TypedDict):
            messages: Annotated[list, add_messages]
            documents: list
            tool_calls: list
            user_id: str

    Example graph:
        builder = StateGraph(AgentState)
        builder.add_node("agent", agent_node)
        builder.add_node("tools", ToolNode(tools))
        builder.add_node("validate", validator_node)
        builder.add_edge(START, "agent")
        builder.add_conditional_edges("agent", should_use_tools)
        builder.add_edge("tools", "validate")
        builder.add_conditional_edges("validate", validation_router)
        graph = builder.compile(checkpointer=PostgresSaver(...))

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PRODUCTION FEATURES:

    - PERSISTENCE: PostgresSaver for crash recovery
    - HUMAN-IN-THE-LOOP: interrupt_before for approval gates
    - STREAMING: graph.stream() yields intermediate states
    - REPLAY: re-run from any checkpoint for debugging
    - TIME TRAVEL: roll back state to any prior step

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

INTERVIEW ANSWER:
    "I use LangGraph for orchestration. It's a state machine where each node
    is a function that updates shared state. Conditional edges enable routing
    based on state. Compiled with PostgresSaver checkpointer for production
    persistence. Key features: streaming for real-time updates, HITL via
    interrupt_before for approval gates, and replay/time-travel for debugging.
    The graph compiles to a runnable that handles state, routing, and error
    handling automatically."
"""


# =================================================================================
# SECTION 7: COMPONENT DEEP DIVE — TOOL REGISTRY
# =================================================================================
"""
The TOOL REGISTRY is the interface between agents and the outside world.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHAT IS A TOOL?

    A function the agent can call to interact with external systems.
    Each tool has:
    - NAME: unique identifier
    - DESCRIPTION: what it does (LLM uses this to decide)
    - PARAMETERS: typed inputs (Pydantic model)
    - RETURN TYPE: structured output

    Example:
        @tool
        def search_database(query: str, limit: int = 10) -> list[dict]:
            # Search the customer database for records matching the query.
            return db.search(query, limit)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TOOL CATEGORIES:

    1. RETRIEVAL TOOLS — search, fetch data (DB, vector store, web)
    2. ACTION TOOLS — modify data (create, update, delete)
    3. COMPUTATION TOOLS — calculate, analyze (math, code execution)
    4. COMMUNICATION TOOLS — email, slack, SMS
    5. INTEGRATION TOOLS — call external APIs, MCP servers

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

REGISTRY DESIGN PATTERNS:

    PATTERN 1: Static registry (simple)
        Tools defined at startup. All agents share the same toolset.

    PATTERN 2: Dynamic registry (advanced)
        Tools loaded based on user permissions, context, or RBAC.
        RBAC stands for Role-Based Access Control.
        Example: Admin sees admin_tools; user sees user_tools.

    PATTERN 3: MCP-based registry (modern)
        Tools provided via MCP servers (Model Context Protocol).
        Decoupled — tools live in separate processes/services.
        Easy to add new tool sources without changing agent code.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PRODUCTION CONCERNS:

    AUTHORIZATION: Which agents/users can call which tools?
        - RBAC checks before tool execution
        - Audit log of all tool calls

    RATE LIMITING: Prevent runaway agents
        - Max calls per tool per session
        - Cool-down periods after errors

    ERROR HANDLING: What if a tool fails?
        - Retries with exponential backoff
        - Fallbacks to alternative tools
        - Graceful degradation

    SECURITY: Tools = attack surface
        - Input validation (SQL injection, prompt injection)
        - Output sanitization (prevent data leakage)
        - Sandboxing (especially for code execution tools)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

INTERVIEW ANSWER:
    "Tool registry is the interface between agents and external systems. Each
    tool has a name, description (the LLM reads this to decide), typed
    parameters, and structured output. I categorize tools into retrieval,
    action, computation, communication, and integration. For production,
    I add RBAC for authorization, rate limiting to prevent runaway agents,
    retry logic for failures, and input/output sanitization for security.
    For modern systems, I use MCP servers for decoupled tool registries —
    tools live in separate services that any agent can use."
"""


# =================================================================================
# SECTION 8: COMPONENT DEEP DIVE — MEMORY LAYER
# =================================================================================
"""
Memory is what makes agents AGENTIC instead of just stateless functions.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE 4 TYPES OF MEMORY:

    SHORT-TERM (Working Memory):
        - Current conversation buffer
        - Recent messages (last N turns)
        - Storage: in-memory (RAM) or session cache (Redis)
        - TTL: minutes to hours
        - Use: immediate context for current interaction

    LONG-TERM (Persistent Memory):
        - User profile, preferences, history
        - Past interactions, facts learned
        - Storage: PostgreSQL, MongoDB
        - TTL: persistent
        - Use: personalization across sessions

    SEMANTIC (Vector Memory):
        - Embeddings of past conversations, documents
        - Storage: vector DB (Pinecone, Weaviate, FAISS)
        - Use: semantic search, RAG, "remember when..."

    EPISODIC (Knowledge Graph):
        - Entity relationships and events
        - Storage: graph DB (Neo4j) or PostgreSQL
        - Use: "What did the user do last week?" "Connections between concepts"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PRODUCTION DESIGN:

    Conversation buffer (short-term)
        ↓ [overflow]
    Summary memory (compress old messages)
        ↓ [save]
    Long-term store (PostgreSQL)
        ↓ [embed]
    Vector memory (Pinecone — for semantic recall)

    Example flow:
    - User starts conversation → empty buffer
    - 20 messages exchanged → buffer full
    - Older messages summarized → keep recent + summary in buffer
    - Conversation ends → save full transcript to DB
    - Embed transcript → store in vector DB for future recall

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

KEY DESIGN CONCERNS:

    1. CONTEXT WINDOW LIMITS
        LLM can only handle X tokens. As conversation grows, you must compress.
        Strategy: summarize older messages, keep recent ones verbatim.

    2. MULTI-TENANT ISOLATION
        User A's memory must NEVER leak to User B.
        Enforce at infrastructure level (separate namespaces).

    3. PRIVACY & GDPR
        User can request their memory be deleted.
        Implement "right to be forgotten" — actual deletion from all stores.

    4. PERSISTENCE STRATEGY
        Hot data (active conversations): Redis (fast)
        Warm data (recent users): PostgreSQL
        Cold data (long-term): S3 + vector DB

INTERVIEW ANSWER:
    "Memory has 4 types. Short-term is the conversation buffer in Redis.
    Long-term is user profile in PostgreSQL. Semantic memory is embeddings
    in a vector DB for semantic recall. Episodic is a knowledge graph for
    relationships. The key challenge is context window limits — I solve with
    sliding window + summary memory: keep recent verbatim, compress older.
    Multi-tenant isolation is enforced at infrastructure level — separate
    namespaces per user. GDPR requires hard deletion across all stores."
"""


# =================================================================================
# SECTION 9: COMPONENT DEEP DIVE — LLM GATEWAY
# =================================================================================
"""
LLM Gateway = abstraction layer for multiple LLM providers.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHY YOU NEED A GATEWAY:

    Without it: Your code is hard-coded to one provider (e.g., OpenAI).
    With it: Switch providers by changing config, not code.

    Why switch providers?
    - Cost optimization (use cheaper model when possible)
    - Failover (if OpenAI is down, use Anthropic)
    - Compliance (use Bedrock for HIPAA, OpenAI for general)
    - Performance (Groq for speed, GPT-4 for quality)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GATEWAY ARCHITECTURE:

    Application code calls: gateway.invoke(prompt, model_id)
        ↓
    Gateway maps model_id to provider:
        "gpt-4o" → OpenAI
        "claude-3.5" → Anthropic
        "llama-3.3" → Groq or Bedrock
        ↓
    Gateway adds:
        - Authentication (API keys from secrets manager)
        - Retries (exponential backoff)
        - Fallback (if primary fails, try secondary)
        - Rate limiting (per provider quotas)
        - Cost tracking (log tokens + cost per call)
        - Caching (semantic cache for repeated prompts)
        ↓
    Returns: standardized response

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

KEY PATTERNS:

    PATTERN 1: Strategy Pattern
        Each provider implements the same interface (invoke, stream, embed).

    PATTERN 2: Factory Pattern
        Factory creates the right provider instance based on config.
        (You already use this — GroqLLMFactory in DocSage!)

    PATTERN 3: Decorator Pattern
        Wrap providers with cross-cutting concerns:
        - LoggingDecorator (logs every call)
        - RetryDecorator (retries failures)
        - CacheDecorator (caches responses)

    PATTERN 4: Circuit Breaker
        If provider fails N times → "open" the circuit (skip it).
        Try again after cooldown. Auto-failover.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EXAMPLE PRODUCTION GATEWAY:

    class LLMGateway:
        def __init__(self):
            self.providers = {
                "primary": OpenAIProvider(),
                "fallback": AnthropicProvider(),
                "cheap": GroqProvider(),
            }
            self.cache = SemanticCache()
            self.cost_tracker = CostTracker()

        def invoke(self, prompt, task_complexity="medium"):
            # Cache check
            cached = self.cache.get(prompt)
            if cached:
                return cached

            # Choose provider based on task
            provider = self._select_provider(task_complexity)

            # Call with retries + fallback
            try:
                response = self._with_retries(provider.invoke, prompt)
            except Exception:
                response = self.providers["fallback"].invoke(prompt)

            # Track cost
            self.cost_tracker.log(provider, response.tokens)

            # Cache
            self.cache.set(prompt, response)

            return response

INTERVIEW ANSWER:
    "I build an LLM Gateway as an abstraction over multiple providers (OpenAI,
    Anthropic, Bedrock, Groq). It uses Strategy pattern — each provider
    implements the same interface. Factory creates the right one based on
    config. Decorators add retries, caching, logging. Circuit Breaker handles
    failures with auto-failover. Routing logic picks providers based on task
    complexity (cheap for simple, premium for complex) and cost tracking
    happens at every call. This lets me swap providers without changing
    application code."
"""


# =================================================================================
# SECTION 10: COMPONENT DEEP DIVE — OBSERVABILITY
# =================================================================================
"""
You can't operate what you can't see. Observability = visibility into
your AI system's behavior.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE 3 PILLARS OF OBSERVABILITY:

    1. LOGS — what happened
        Structured logs with request IDs, timestamps, context.
        "User X asked question Y, agent called tool Z, returned result W."

    2. METRICS — measurements over time
        Counters, gauges, histograms.
        Latency p50/p95/p99, request rate, error rate, cost per request.

    3. TRACES — flow of a single request
        Distributed tracing across all services.
        "Request 1234 took 4.2s — gateway 100ms, agent 2.5s, tool 1.5s, DB 100ms."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHAT TO LOG IN AGENTIC SYSTEMS:

    Per request:
    - User ID, session ID, request ID (correlation IDs)
    - Input prompt (with PII redacted)
    - Agent's plan / decisions
    - Each tool call (name, args, result, latency)
    - LLM calls (provider, model, tokens, cost)
    - Final response
    - Errors and warnings

    Aggregated metrics:
    - Requests per minute
    - p95 latency
    - Error rate
    - LLM cost per request
    - Tool call frequency
    - Quality metrics (if available — RAGAS scores)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TOOLS:

    LangSmith — Specialized for LLM apps (LangChain ecosystem)
        - Traces every node, tool call, LLM call
        - Quality scoring with custom evaluators
        - Cost tracking per chain

    OpenTelemetry — Industry standard for distributed tracing
        - Vendor-neutral
        - Works across all services

    Datadog / New Relic — General APM (Application Performance Monitoring)
        - Logs, metrics, traces
        - Alerting and dashboards

    Prometheus + Grafana — Open-source metrics + dashboards
        - Pull-based metrics collection
        - Highly customizable

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ALERTING STRATEGY:

    Alert on:
    - Error rate > 1% for 5 minutes
    - p95 latency > 5 seconds
    - LLM cost spike > 2x average
    - Faithfulness score drops > 5% in 24h
    - Specific error patterns (OOM, rate limits)

    Don't alert on (avoid noise):
    - Single failures (might be transient)
    - Metrics that fluctuate normally
    - Things you can't act on

INTERVIEW ANSWER:
    "Observability has 3 pillars: logs (what happened), metrics (measurements),
    traces (request flow). For agentic systems, I log every agent decision,
    tool call, and LLM call with correlation IDs. Metrics: latency p95, error
    rate, LLM cost. Traces: full request lifecycle across services. Tools:
    LangSmith for LLM-specific tracing, OpenTelemetry for distributed tracing,
    Prometheus + Grafana for metrics. Alerts on error rate, latency spikes,
    cost anomalies, and quality degradation."
"""


# =================================================================================
# SECTION 11: COMPONENT DEEP DIVE — SAFETY & GOVERNANCE
# =================================================================================
"""
Safety = preventing AI from causing harm. Governance = ensuring proper use.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GUARDRAILS — Defense in depth:

    INPUT GUARDRAILS (before LLM processes):
    - Prompt injection detection ("ignore previous instructions")
    - PII detection and redaction
    - Topic classification (block off-scope queries)
    - Toxicity / harm detection

    OUTPUT GUARDRAILS (before user sees):
    - PII scanning (no leaked SSNs, emails)
    - Hallucination check (grounded in retrieved context?)
    - Format validation (matches expected schema)
    - Toxicity / brand safety

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HUMAN-IN-THE-LOOP (HITL):

    Critical actions require human approval before execution.
    Examples:
    - Refunds over $X
    - Sending emails to customers
    - Making database changes
    - Escalating to legal

    Implementation: LangGraph interrupt_before for the action node.
    State persists. Human reviews. Approves or rejects.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GOVERNANCE FRAMEWORK:

    1. AUDIT LOGS
        Every agent decision logged.
        Who, what, when, why.
        Immutable storage (compliance).

    2. MODEL CARDS
        Document each LLM and tool used:
        - Capabilities, limitations
        - Bias, fairness considerations
        - Approved use cases

    3. POLICY ENFORCEMENT
        Agents check policies before actions:
        - Can this user access this data?
        - Is this action allowed in this region?
        - Are we within rate limits?

    4. REGULATORY COMPLIANCE
        Frameworks: HIPAA (healthcare), GDPR (EU privacy),
        SOC 2, PCI DSS, FedRAMP (US government).
        Each has specific requirements.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TOOLS:

    NeMo Guardrails (Nvidia) — programmable AI safety
    Guardrails AI — output validation framework
    Llama Guard — Meta's safety classifier
    OpenAI Moderation API — content moderation

INTERVIEW ANSWER:
    "Safety and governance is multi-layered. Input guardrails: prompt injection
    detection, PII redaction, topic filtering. Output guardrails: PII scanning,
    hallucination check, format validation. HITL for critical actions —
    refunds, emails, DB changes pause for human approval via LangGraph
    interrupt_before. Governance includes audit logs, model cards, policy
    enforcement, and regulatory compliance (HIPAA, GDPR, SOC 2 depending on
    domain). Tools: NeMo Guardrails, Guardrails AI, Llama Guard."
"""


# =================================================================================
# SECTION 12: PRODUCTION DEPLOYMENT ARCHITECTURE
# =================================================================================
"""
HOW TO DEPLOY AN AGENTIC AI SYSTEM IN PRODUCTION:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE PRODUCTION DEPLOYMENT STACK:

    ┌────────────────────────────────────────────────────────────────────┐
    │                      USER (browser/mobile)                          │
    └─────────────────────────────┬──────────────────────────────────────┘
                                  ↓
    ┌────────────────────────────────────────────────────────────────────┐
    │              CDN (CloudFlare, CloudFront) — static assets           │
    └─────────────────────────────┬──────────────────────────────────────┘
                                  ↓
    ┌────────────────────────────────────────────────────────────────────┐
    │            LOAD BALANCER (ALB, ELB) — distributes traffic           │
    └─────────────────────────────┬──────────────────────────────────────┘
                                  ↓
    ┌────────────────────────────────────────────────────────────────────┐
    │         API GATEWAY (Auth, Rate Limit, Request Validation)          │
    └─────────────────────────────┬──────────────────────────────────────┘
                                  ↓
    ┌────────────────────────────────────────────────────────────────────┐
    │                    APPLICATION LAYER (Microservices)                │
    │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
    │  │ FastAPI      │  │ FastAPI      │  │ FastAPI      │              │
    │  │ Agent Svc 1  │  │ Agent Svc 2  │  │ Agent Svc 3  │              │
    │  └──────────────┘  └──────────────┘  └──────────────┘              │
    │  (Kubernetes pods, auto-scaling, multiple replicas)                  │
    └─────────────────────────────┬──────────────────────────────────────┘
                                  ↓
    ┌────────────────────────────────────────────────────────────────────┐
    │                       MESSAGE QUEUE LAYER                            │
    │   (Redis Streams / Kafka / SQS — for async tasks like ingestion)    │
    └─────────────────────────────┬──────────────────────────────────────┘
                                  ↓
    ┌────────────────────────────────────────────────────────────────────┐
    │                       BACKGROUND WORKERS                             │
    │      (Celery workers / Lambda for long-running tasks)               │
    └─────────────────────────────┬──────────────────────────────────────┘
                                  ↓
    ┌────────────────────────────────────────────────────────────────────┐
    │                       PERSISTENCE LAYER                              │
    │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐  │
    │  │ PostgreSQL │  │   Redis    │  │  Pinecone  │  │     S3     │  │
    │  │ (relations)│  │  (cache)   │  │ (vectors)  │  │  (files)   │  │
    │  └────────────┘  └────────────┘  └────────────┘  └────────────┘  │
    └────────────────────────────────────────────────────────────────────┘
                                  ↓
    ┌────────────────────────────────────────────────────────────────────┐
    │                      EXTERNAL SERVICES                               │
    │  - LLM APIs (OpenAI, Bedrock, Anthropic)                            │
    │  - Tool APIs (third-party integrations)                              │
    │  - MCP Servers                                                       │
    └────────────────────────────────────────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DEPLOYMENT PATTERNS:

    KUBERNETES (Most Common):
        - Each microservice = Kubernetes Deployment
        - Auto-scaling based on CPU/memory/requests
        - Rolling updates with health checks
        - Service mesh (Istio) for inter-service comm
        - Helm charts for configuration management

    SERVERLESS (For Bursty Workloads):
        - AWS Lambda / Google Cloud Functions
        - Pay per execution (no idle cost)
        - Limited to 15min execution (good for short tasks)

    CONTAINER (Docker):
        - One container per service
        - Easy local dev, testing
        - Foundation for K8s

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

KEY PRODUCTION CONCERNS:

    1. SCALABILITY
       - Horizontal: more pods/replicas
       - Vertical: bigger machines
       - Auto-scaling: HPA (Horizontal Pod Autoscaler)

    2. RELIABILITY
       - Health checks (readiness, liveness probes)
       - Circuit breakers (prevent cascading failures)
       - Graceful degradation (fallback responses)

    3. COST OPTIMIZATION
       - Right-size pods (don't over-allocate)
       - Spot instances for non-critical workers
       - LLM cost monitoring (the big expense)

    4. SECURITY
       - Network policies (which services can talk?)
       - Secrets management (Vault, AWS Secrets Manager)
       - RBAC at every layer

    5. CI/CD
       - Automated tests on every PR
       - Eval suite runs before deployment
       - Canary deploys (5% traffic → monitor → 100%)
       - Easy rollback if issues

INTERVIEW ANSWER:
    "Production deployment uses Kubernetes for orchestration. CDN + Load
    Balancer + API Gateway in front. Multiple FastAPI microservices as
    pods with auto-scaling. Message queue (Redis/Kafka) for async work.
    Persistence: PostgreSQL for relational, Redis for cache, Pinecone
    for vectors, S3 for files. External: LLM APIs, tool services. Critical
    concerns: health checks, circuit breakers, cost monitoring (LLM costs
    are the big variable), and CI/CD with eval suite gates and canary
    deploys for safe rollouts."
"""


# =================================================================================
# SECTION 13: HOW TO DRAW THESE ON A WHITEBOARD (Step by Step)
# =================================================================================
"""
WHEN AN ARCHITECT INTERVIEWER SAYS "Draw the architecture":

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DRAW IN THIS ORDER (5-7 minutes):

    STEP 1: User and Entry Point (top of board)
        Box: "User" → Arrow → Box: "API Gateway"
        Say: "Users hit our API gateway through web/mobile/voice"

    STEP 2: Authentication & Validation (next layer)
        Box: "Auth" + "Rate Limit" + "Input Guardrails"
        Say: "Gateway authenticates with JWT, rate limits per user,
              input guardrails detect PII and prompt injection"

    STEP 3: Orchestrator (the core)
        BIG BOX: "Orchestrator (LangGraph)"
        Inside: smaller boxes for "Planner", "Executor", "Critic"
        Say: "Core is LangGraph orchestrator with state graph,
              conditional routing, checkpointing for crash recovery"

    STEP 4: Three Pillars Below Orchestrator
        Three boxes side by side:
        "Tool Registry" | "Memory Layer" | "LLM Gateway"

        Say:
        "Tool Registry: catalog of tools with RBAC and rate limits"
        "Memory: short-term buffer + long-term DB + vector store"
        "LLM Gateway: multi-provider abstraction (OpenAI, Bedrock, Groq)"

    STEP 5: Output Validation
        Box: "Output Guardrails" before sending to user
        Say: "PII scanning, hallucination check, format validation"

    STEP 6: Observability (across the bottom)
        Long box across whole diagram: "Observability (Logging, Tracing, Metrics)"
        Say: "Everything logged with request IDs, traces in LangSmith,
              metrics in Prometheus/Grafana"

    STEP 7: Persistence Layer (bottom)
        Boxes: "PostgreSQL", "Redis", "Pinecone", "S3"
        Say: "Relational data in Postgres, cache in Redis,
              vectors in Pinecone, files in S3"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHILE DRAWING, NARRATE:

    1. "Each block has ONE responsibility."
    2. "Components communicate via defined interfaces."
    3. "Failure in one block doesn't crash others."
    4. "Each block can scale independently."
    5. "Observability spans everything."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ANTICIPATE FOLLOW-UP QUESTIONS:

    Q: "How do you handle scale?"
    A: Point to K8s, HPA, queue layer, multi-replica services.

    Q: "What if LLM provider is down?"
    A: Point to LLM Gateway, explain fallback to secondary provider.

    Q: "How do you debug failures?"
    A: Point to Observability layer, explain LangSmith tracing.

    Q: "How do you handle long-running tasks?"
    A: Point to Message Queue + Workers (Celery, Lambda).

    Q: "What about cost?"
    A: Point to LLM Gateway with cost tracking, model routing
       (cheap models for simple tasks).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PRO TIPS FOR WHITEBOARD INTERVIEWS:

    1. Draw TOP-DOWN (user at top, infra at bottom).
    2. Use CLEAR boxes, not blob shapes.
    3. Label EVERY arrow with what it carries (REST, gRPC, Queue, etc.)
    4. Don't draw too small — leave room for follow-ups.
    5. Use COLOR (if available): user-facing in one color, infra in another.
    6. After drawing, RECAP the flow in 30 seconds.

INTERVIEW ANSWER:
    "I draw architecture top-down. Start with user → API gateway. Auth and
    input guardrails. Then the orchestrator (LangGraph) as the core box.
    Below it, three pillars: Tool Registry, Memory Layer, LLM Gateway.
    Output guardrails before response. Observability spans everything at
    the bottom. Persistence layer at the foundation. I narrate as I draw —
    explaining each block's responsibility and how they communicate. I
    leave room for follow-up questions and use clear labels on arrows."
"""


# =================================================================================
# SECTION 14: 25+ ARCHITECT-LEVEL INTERVIEW Q&A
# =================================================================================
"""
TIER 1 — FOUNDATION (must know):

Q1: "Define agentic AI."
A: "Autonomous, goal-directed AI that plans, reasons, acts, and adapts.
   Built on LLMs but adds 5 capabilities: autonomy, planning, tool use,
   memory, and adaptation. Unlike workflows (fixed sequences) or simple
   agents (one-step tool use), agentic AI can run multi-step goal-directed
   tasks with self-correction."

Q2: "What are the architectural layers of an agentic system?"
A: "5 layers: Perception (input), Reasoning (LLM-based planning), Memory
   (short/long/vector/graph), Action (tool execution), Learning (feedback
   loop). Each layer has its own components and concerns."

Q3: "Explain orchestrator-worker pattern."
A: "Supervisor agent decomposes the user request and delegates to specialized
   worker agents. Workers do their specific tasks. Orchestrator combines
   results. Most common pattern. Good for clear sub-task structure."

Q4: "When would you use peer-to-peer over orchestrator-worker?"
A: "Peer-to-peer for iterative refinement work where agents are equal
   collaborators (e.g., code review with coder + reviewer + tester).
   Orchestrator-worker for tasks with clear delegation hierarchy."

Q5: "What is HITL and when is it needed?"
A: "Human-in-the-Loop. Critical actions pause for human approval before
   execution. Needed for: financial actions, irreversible operations,
   compliance-required reviews. Implementation: LangGraph interrupt_before."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TIER 2 — DESIGN AND TRADEOFFS:

Q6: "How do you handle context window overflow in long conversations?"
A: "Sliding window + summary memory. Keep last 5-10 messages verbatim,
   summarize older messages into a compressed memory. For very long
   conversations, archive older summaries to vector DB for semantic recall."

Q7: "Walk through what happens when an LLM provider goes down."
A: "Circuit breaker in LLM Gateway opens after N failures. Requests route
   to fallback provider (Anthropic if OpenAI fails). User experience:
   maybe slower or different quality, but no outage. After cooldown, gateway
   periodically tests primary, restores when healthy."

Q8: "How do you ensure multi-tenant isolation?"
A: "Multi-layered. Vector DB: separate namespaces per tenant. Postgres:
   row-level security with tenant_id. Memory caches: scoped by user_id.
   API: every request validates user can access requested tenant.
   Audit: every cross-tenant access attempt logged for security review."

Q9: "How do you reduce LLM costs at scale?"
A: "Multiple strategies. Model routing: cheap model (GPT-4o-mini) for
   simple tasks, premium (GPT-4o) for complex. Prompt optimization: shorter
   context, fewer few-shot examples. Caching: semantic cache for repeated
   queries. Field filtering: send only relevant data to LLM (60-70% token
   savings in my MCP work)."

Q10: "Difference between LangGraph and AutoGen?"
A: "LangGraph: state machine — explicit nodes, edges, deterministic flow.
   Best for production where control matters. AutoGen: peer-to-peer agent
   conversations — emergent behavior, agents 'talk' to solve problems.
   Best for iterative refinement (code review). Use case driven choice."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TIER 3 — PRODUCTION:

Q11: "How do you handle a 30-minute document ingestion process?"
A: "Async with message queue. API endpoint enqueues job → returns job_id
   immediately. Background worker (Celery) processes: load → chunk → embed
   → store. WebSocket or polling for status. On completion, push notification."

Q12: "How do you debug an agent that's giving wrong answers?"
A: "Use LangSmith traces to see EVERY step. Check: (1) what was retrieved?
   Was it relevant? (2) What did the LLM see in its prompt? (3) What did
   it generate? Often reveals retrieval failed (wrong docs) or prompt was
   ambiguous. Replay specific traces with different parameters to test fixes."

Q13: "Design a system that handles 10K queries/sec."
A: "K8s with auto-scaling. CDN for static. Multiple stateless API replicas.
   LLM responses cached aggressively (semantic cache). Read replicas for
   PostgreSQL. Pinecone for vectors (handles QPS natively). Async processing
   for any heavy work. Circuit breakers and rate limits at every layer."

Q14: "What if your eval shows quality dropped 10%?"
A: "Roll back recent changes immediately. Then diagnose: was it a code change,
   prompt change, model update, or data drift? Use eval breakdown by category
   to localize. Fix → re-evaluate → deploy. Add this failure to test set
   to prevent regression."

Q15: "How do you implement HITL without blocking the entire system?"
A: "Pending approvals are async. Agent pauses, state persisted, request
   returns 'pending'. Notification sent to approver via Slack/email. When
   approved, callback triggers agent resumption. Other requests continue
   processing. UX: user sees 'pending approval' status."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TIER 4 — ARCHITECT-LEVEL:

Q16: "Tell me about the worst architectural mistake you've seen in agentic AI."
A: "Coupling agent logic with infrastructure. Like: 'agent_service' code
   directly calls vector DB, LLM API, email service. When provider changed,
   refactoring took weeks. Fix: layered architecture — agents only see
   abstract interfaces (LLMGateway, ToolRegistry), implementations swap
   without touching business logic."

Q17: "How would you architect for compliance (HIPAA/GDPR)?"
A: "Data residency: keep data in approved regions. PII detection at all
   boundaries. Encryption at rest and in transit. Audit logs immutable
   for required retention. Right to be forgotten: actual deletion across
   all stores. Use private LLMs (Bedrock) or on-prem (Ollama) for sensitive
   data — no third-party API calls with patient data."

Q18: "How do you handle long-running agentic workflows (hours/days)?"
A: "Decompose into restart-able tasks. Each task: (1) reads input from
   queue, (2) does work, (3) saves state, (4) enqueues next task. Crash
   recovery: pick up from last completed task. Use durable workflow
   engines (Temporal, Airflow) for orchestration."

Q19: "What's wrong with this architecture? [shows diagram with single LLM call]"
A: "Single point of failure. No fallback. No cost control. No observability.
   Add: LLM Gateway with multi-provider, retry/circuit breaker, caching,
   cost tracking, request tracing."

Q20: "How do you handle agent memory privacy?"
A: "Per-tenant isolation at infrastructure level (separate namespaces).
   Encryption with per-tenant keys. Audit access logs. Implement deletion
   API for GDPR. Don't leak via embeddings — use private embedding models
   for sensitive data, not public APIs."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TIER 5 — PWC-SPECIFIC:

Q21: "How do you balance quality vs speed in agentic systems?"
A: "Multi-tier approach. Fast path: simple queries → cheap model → fast
   response (< 1s). Slow path: complex queries → premium model + RAG +
   validation → 5-10s. Router classifies the query and picks the path.
   Most queries take fast path; complex ones get more thorough treatment."

Q22: "Explain the request flow in a multi-agent customer support system."
A: "Hub-and-Spoke pattern. Hub (router) classifies: 'Is this billing,
   technical, or general?' Routes to specialist spoke. Spoke uses its
   own tools, RAG, memory. Fallback: if spoke can't handle, escalates
   back to hub which can route to human or another spoke. All actions
   logged."

Q23: "How do you handle agent failures in production?"
A: "Multiple layers. Tool failures: retry with backoff. LLM failures:
   fallback provider via gateway. State corruption: checkpoint restore.
   Infinite loops: recursion limit. Total agent failure: graceful
   degradation — return safe canned response or escalate to human.
   All failures logged for analysis."

Q24: "How would you design an agentic system for legal contract review?"
A: "Components: (1) Document loader handles PDFs with layout-aware
   parser. (2) Multi-tenant RAG for client isolation. (3) Specialized
   agents: clause extractor, risk identifier, comparison analyst.
   (4) HITL for all flagged risks (lawyer reviews). (5) Strict audit
   trail. (6) Compliance: data stays in approved regions, encryption
   end-to-end. Strict no-hallucination policy with citation requirements."

Q25: "What would you change about a system that has 90% accuracy?"
A: "10% failure rate is unacceptable for production. Diagnose: which
   component failing? Run RAGAS — if Recall low, fix retrieval (better
   chunking, hybrid search, reranking). If Faithfulness low, fix
   generation (better prompts, stronger model). Add HITL for high-risk
   queries. Build feedback loop to learn from failures. Target: 99%+
   for critical use cases."
"""


# =================================================================================
# SECTION 15: GOLDEN LESSONS
# =================================================================================
"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GOLDEN LESSON 1: "Architecture is about LAYERS, not features."
    Senior engineers think in layers (Perception/Reasoning/Memory/Action/Learning).
    Junior engineers think in features. Layers make systems maintainable.

GOLDEN LESSON 2: "Every component should have ONE responsibility."
    Single Responsibility Principle. If your 'agent service' does retrieval,
    LLM calls, AND DB writes — split it.

GOLDEN LESSON 3: "Loose coupling is non-negotiable."
    Components should communicate via defined interfaces. If you can't swap
    LLM provider in 1 hour, your architecture is wrong.

GOLDEN LESSON 4: "Observability is not optional in production."
    You can't operate what you can't see. Logs + metrics + traces from day 1.

GOLDEN LESSON 5: "HITL is your safety net for irreversible actions."
    Refunds, deletes, emails — anything you can't undo MUST go through human approval.

GOLDEN LESSON 6: "Patterns matter — know all 4 multi-agent patterns."
    Orchestrator-Worker, Peer-to-Peer, Hierarchical, Hub-and-Spoke. Different
    problems need different patterns. Pick consciously, not randomly.

GOLDEN LESSON 7: "Memory is what makes agents AGENTIC."
    Without memory, you have a stateless function. With memory (short, long,
    semantic, episodic), you have an agent that learns and adapts.

GOLDEN LESSON 8: "Production is about FAILURE MODES, not happy paths."
    Anyone can build for the happy path. Senior engineers design for: API down,
    network slow, LLM hallucinates, user injects prompts, costs spike.

GOLDEN LESSON 9: "Cost is a first-class concern."
    LLM calls add up FAST. Cost tracking, model routing, caching, prompt
    optimization — these aren't optimizations, they're requirements.

GOLDEN LESSON 10: "Senior architects DRAW. Juniors DESCRIBE."
    Practice drawing architecture diagrams. Top-down. Clear boxes. Labeled
    arrows. Walk through the flow as you draw. This is how you DEMONSTRATE
    senior thinking in interviews.
"""

print("=" * 60)
print("Agentic AI Architecture Deep Dive — Complete")
print("=" * 60)
print()
print("15 Sections:")
print("  1.  What is Agentic AI? (vs LLMs/Workflows/Agents)")
print("  2.  The 5 Architectural Layers")
print("  3.  Block Diagram of Production System (Master)")
print("  4.  Data Flow Diagram (Request to Response)")
print("  5.  Multi-Agent Patterns (4 patterns explained)")
print("  6.  Orchestrator (LangGraph)")
print("  7.  Tool Registry")
print("  8.  Memory Layer (Short/Long/Vector/Graph)")
print("  9.  LLM Gateway (Multi-Provider)")
print("  10. Observability (Logs/Metrics/Traces)")
print("  11. Safety & Governance (Guardrails/HITL)")
print("  12. Production Deployment (K8s, Queues)")
print("  13. How to Draw on Whiteboard (Step by Step)")
print("  14. 25 Architect-Level Interview Q&A")
print("  15. GOLDEN LESSONS (10 lessons)")
print()
print("This is what 20+ year architects expect to hear.")
print("=" * 60)
