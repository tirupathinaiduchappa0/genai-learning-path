"""
===================================================================================
JPMC AGENTIC DEVELOPMENT ASSOCIATE — COMPLETE INTERVIEW PREPARATION
===================================================================================

Company: JPMorgan Chase & Co. (Consumer & Community Banking — DART Team)
Role: Agentic Development Associate
Interviewer: Kathan Joshi (Vice President)

THIS IS A VP-LEVEL INTERVIEW. Expect:
    - Sharp, focused questions (no fluff, no warm-up)
    - Architecture-level thinking (not just code syntax)
    - "How would you design..." questions
    - "Walk me through..." your agentic system
    - Behavioral: leadership, ownership, stakeholder communication

THIS LESSON COVERS:
    1.  Company & Team Overview (DART Team, Consumer Banking)
    2.  What is an Agentic System (definition the VP wants to hear)
    3.  Agentic Frameworks — LangGraph, AutoGen, Google ADK
    4.  Agentic Design Patterns (Modular Components)
    5.  Your Agentic System — DocSage Architecture (your main story)
    6.  Your MCP Server — 17+ Tools (your second story)
    7.  AWS Deployment & Containerization (Docker, EKS)
    8.  Testing Agentic Systems
    9.  Security & Compliance in Financial AI
    10. Agile, Jira, Team Leadership
    11. Your Experience Mapped to JD
    12. 30+ Interview Q&A (VP-level)

EXISTING LESSONS TO REVISE BEFORE THIS:
    - 06_docsage_project_deep_dive.py (your main agentic system story)
    - 07_rag_complete_guide.py (Sections 8, 10 — patterns + production checklist)
===================================================================================
"""


# =================================================================================
# SECTION 1: COMPANY & TEAM OVERVIEW
# =================================================================================
"""
JPMORGAN CHASE:
    - Largest bank in the US, one of the oldest financial institutions (200+ years)
    - Leader in investment banking, consumer banking, asset management
    - Global presence: US, India, Philippines, UK, Singapore
    - Hyderabad office: MAGMA building, Kondapur (where you'd work)

THE TEAM — DART (Data, Analytics and Reporting Team):
    - Part of Consumer & Community Banking Operations
    - Provides data analytics and automation solutions
    - Global team: US, India, Philippines
    - Building "future-ready solutions" — that means AGENTIC AI

WHAT THEY'RE BUILDING:
    - Intelligent automation for banking operations
    - Agentic systems that can handle customer queries, process documents,
      automate reporting, and orchestrate workflows
    - Think: AI agents that handle loan processing, fraud detection alerts,
      customer complaint resolution, regulatory reporting

WHY THIS ROLE EXISTS:
    - Banks are moving from rule-based automation (RPA) to intelligent automation (AI agents)
    - They need people who can BUILD agentic systems, not just use ChatGPT
    - The VP wants someone who can design, code, deploy, and lead

YOUR INTERVIEWER — Kathan Joshi (Vice President):
    - VP at JPMC = senior technical leader (not just management)
    - He likely leads the agentic AI initiative for DART
    - He'll ask architecture questions, not trivia
    - He wants to see: can you THINK at a systems level?
    - 30 minutes = he'll be direct and fast. No time for rambling.
"""


# =================================================================================
# SECTION 2: WHAT IS AN AGENTIC SYSTEM
# =================================================================================
"""
DEFINITION (what the VP wants to hear):

    "An agentic system is an AI system that can autonomously decide what
    actions to take, execute those actions using tools, observe the results,
    and iterate until a goal is achieved — without human intervention at
    every step."(an intelligent system that understands the goals and make decissions
    autonomously i.e ReAct Pattern)

KEY CHARACTERISTICS OF AGENTIC SYSTEMS:
    1. AUTONOMY — decides what to do next (not hardcoded sequence)
    2. TOOL USE — calls external tools/APIs to take actions
    3. REASONING — thinks about which tool to use and why
    4. MEMORY — remembers context across multiple steps
    5. SELF-CORRECTION — detects failures and retries with different approach
    6. GOAL-ORIENTED — works toward a defined objective, not just responds

AGENTIC vs NON-AGENTIC:

    NON-AGENTIC (simple chain):
        User asks → Retrieve docs → Generate answer → Return
        Fixed sequence. No decisions. No loops. No tool selection.

    AGENTIC:
        User asks → Agent DECIDES: "Do I need to search docs? Call an API?
        Run code? Ask for clarification?" → Executes chosen action →
        Observes result → DECIDES next step → Loops until goal is met.

    The KEY difference: the LLM makes DECISIONS at runtime about what to do.

INTERVIEW ANSWER:
    "An agentic system uses an LLM as a reasoning engine that autonomously
    decides which tools to call, in what order, and when to stop. Unlike
    a fixed pipeline, the agent adapts its behavior based on intermediate
    results. In my DocSage project, the agent decides which knowledge base
    to query, the grader decides if results are good enough, and if not,
    the system self-corrects by rewriting the query and retrying. That
    autonomous decision-making loop is what makes it agentic."
"""


# =================================================================================
# SECTION 3: AGENTIC FRAMEWORKS — LangGraph, AutoGen, Google ADK
# =================================================================================
"""
The JD specifically mentions: LangGraph, Google ADK, AutoGen

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LANGGRAPH (Your primary experience — LEAD WITH THIS)

    What: Framework for building stateful, multi-step agent workflows as graphs.
    By: LangChain team
    Key concepts:
        - StateGraph: define nodes (functions) and edges (transitions)
        - Conditional edges: dynamic routing based on node output
        - Checkpointer: conversation memory (MemorySaver, PostgresSaver)
        - ToolNode: executes tool calls from the agent
        - create_react_agent: one-line agent creation for simple cases

    When to use: Complex workflows with conditional routing, self-correction,
    multi-agent collaboration, human-in-the-loop.

    Your experience: DocSage (5 nodes, 3 conditional edges, self-correction loops)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

AUTOGEN (Microsoft)

    What: Framework for multi-agent conversations. Agents talk to each other.
    By: Microsoft Research
    Key concepts:
        - ConversableAgent: base class for all agents
        - AssistantAgent: LLM-powered agent that can use tools
        - UserProxyAgent: represents the human, can execute code
        - GroupChat: multiple agents collaborating in a conversation
        - Code execution: agents can write and run Python code

    When to use: Multi-agent collaboration where agents need to discuss,
    debate, or review each other's work. Good for code generation + review.

    Example use case at JPMC: One agent generates a report, another agent
    reviews it for compliance, a third agent formats it for stakeholders.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GOOGLE ADK (Agent Development Kit)

    What: Google's framework for building AI agents with Gemini models.
    By: Google DeepMind
    Key concepts:
        - Agent class with tools and instructions
        - Built-in Google Search, Code Execution tools
        - Multi-agent orchestration
        - Evaluation framework built-in

    When to use: When using Google Cloud / Vertex AI infrastructure.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

COMPARISON TABLE:

    FRAMEWORK    BEST FOR                    YOUR EXPERIENCE
    LangGraph    Complex workflows, RAG      Built DocSage, 10 agents project
    AutoGen      Multi-agent collaboration   Know the concepts, can explain
    Google ADK   Google Cloud ecosystem      Aware of it, haven't used

INTERVIEW ANSWER:
    "I primarily work with LangGraph because it gives fine-grained control
    over agent workflows — conditional routing, state management, and
    self-correction loops. I've built a 5-node graph with 3 conditional
    edges for my DocSage project. I'm also familiar with AutoGen for
    multi-agent collaboration scenarios and Google ADK for Gemini-based
    agents. The choice depends on the use case — LangGraph for complex
    orchestration, AutoGen for agent-to-agent collaboration."
"""


# =================================================================================
# SECTION 4: AGENTIC DESIGN PATTERNS (Modular Components)
# =================================================================================
"""
The JD says: "Design, test, and maintain MODULAR COMPONENTS for agentic systems"

WHAT ARE MODULAR COMPONENTS?
    Each piece of the agent system is independent, testable, and swappable.
    Like LEGO blocks — you can add, remove, or replace any piece without
    breaking the whole system.

THE 5 CORE MODULAR COMPONENTS:

    1. AGENT (Brain)
        - The LLM that makes decisions
        - Reads tool descriptions, picks the right one
        - Swappable: change from GPT-4 to Claude without touching other code

    2. TOOLS (Hands)
        - Functions the agent can call (@tool decorator)
        - Each tool does ONE thing well
        - Swappable: add/remove tools without changing agent logic

    3. STATE (Memory)
        - TypedDict that holds all data flowing through the graph
        - Messages, documents, generation, metadata
        - Swappable: change state schema without changing node logic

    4. NODES (Steps)
        - Individual processing steps (grade, generate, rewrite, validate)
        - Each node reads state, does work, writes to state
        - Swappable: replace a node's implementation without changing the graph

    5. GRAPH (Orchestrator)
        - Defines HOW nodes connect (edges, conditional routing)
        - Compiles into a runnable workflow
        - Swappable: change the flow without changing node implementations

DESIGN PRINCIPLES FOR MODULAR AGENTS:

    PRINCIPLE 1: SEPARATION OF CONCERNS
        Each file has ONE responsibility.
        - config/settings.py → all configuration
        - llms/groq_llm.py → LLM creation only
        - tools/retriever_tool.py → retrieval only
        - nodes/grade_node.py → grading only
        - graph/graph_builder.py → wiring only

    PRINCIPLE 2: FACTORY PATTERN
        Create components through factories, not inline.
        factory = GroqLLMFactory(api_key)
        agent_llm = factory.create_agent_llm()
        grading_llm = factory.create_grading_llm()

    PRINCIPLE 3: DEPENDENCY INJECTION
        Nodes receive their dependencies (LLM, tools) from outside.
        They don't create their own. This makes them testable.
        grade_fn = create_grade_node(grading_llm)  # LLM injected

    PRINCIPLE 4: CONFIGURATION OVER CODE
        Model names, chunk sizes, max tokens — all in settings.py.
        Change behavior without touching business logic.

INTERVIEW ANSWER:
    "I design agentic systems with 5 modular components: the agent (LLM brain),
    tools (callable functions), state (shared data), nodes (processing steps),
    and graph (orchestration). Each component is in its own file with one
    responsibility. I use factory patterns for LLM creation, dependency injection
    for testability, and centralized configuration. This means I can swap the
    LLM, add tools, or change the workflow without touching other components."
"""


# =================================================================================
# SECTION 5: AWS, DOCKER & DEPLOYMENT (What JPMC expects)
# =================================================================================
"""
The JD says: "Containerize and deploy agentic system components on cloud platforms
(preferably AWS), adhering to DevOps and infrastructure-as-code practices."

YOU DON'T NEED TO BE AN AWS EXPERT. But you need to know the concepts.

HOW AN AGENTIC SYSTEM GETS DEPLOYED:

    ┌─────────────────────────────────────────────────────────────────┐
    │  LOCAL DEVELOPMENT                                               │
    │  Python code → test locally → works on your machine              │
    └──────────────────────────┬──────────────────────────────────────┘
                               ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │  CONTAINERIZE (Docker)                                           │
    │  Dockerfile → builds an IMAGE with your code + dependencies      │
    │  docker build -t my-agent .                                      │
    │  docker run my-agent                                             │
    │  Now it runs the SAME way on any machine                         │
    └──────────────────────────┬──────────────────────────────────────┘
                               ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │  PUSH TO REGISTRY                                                │
    │  docker push to AWS ECR (Elastic Container Registry)             │
    │  Your image is now stored in the cloud                           │
    └──────────────────────────┬──────────────────────────────────────┘
                               ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │  DEPLOY ON AWS                                                   │
    │  Option 1: ECS (Elastic Container Service) — simpler             │
    │  Option 2: EKS (Elastic Kubernetes Service) — more control       │
    │  Your agent runs as a container in the cloud                     │
    └──────────────────────────┬──────────────────────────────────────┘
                               ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │  INFRASTRUCTURE AS CODE (Terraform / CloudFormation)             │
    │  Define ALL infrastructure in code files                         │
    │  terraform apply → creates everything automatically              │
    │  Reproducible, version-controlled, reviewable                    │
    └─────────────────────────────────────────────────────────────────┘

DOCKER BASICS (know these):
    Dockerfile — recipe to build your container image
    Image — a snapshot of your app + all dependencies
    Container — a running instance of an image
    docker build — creates the image
    docker run — starts a container from the image
    docker push — uploads image to a registry

    Example Dockerfile for an agent:
        FROM python:3.11-slim
        WORKDIR /app
        COPY requirements.txt .
        RUN pip install -r requirements.txt
        COPY . .
        CMD ["python", "main.py"]

AWS SERVICES TO KNOW:
    ECR — stores Docker images (like Docker Hub but private)
    ECS — runs containers (managed, simpler)
    EKS — runs Kubernetes (more control, more complex)
    S3 — stores files (documents, models, data)
    RDS — managed databases (PostgreSQL, MySQL)
    Lambda — serverless functions (for lightweight tasks)
    Bedrock — AWS's LLM service (Claude, Llama on AWS)
    SageMaker — ML model training and deployment

INTERVIEW ANSWER:
    "For deployment, I'd containerize the agentic system with Docker —
    each component (agent service, retrieval service, LLM gateway) as a
    separate container. Push images to ECR, deploy on ECS or EKS depending
    on scale requirements. Infrastructure defined in Terraform for
    reproducibility. For the LLM layer, I'd use AWS Bedrock which gives
    access to Claude and Llama without managing GPU infrastructure.
    Vector store on a managed service like OpenSearch or Pinecone."
"""


# =================================================================================
# SECTION 6: TESTING AGENTIC SYSTEMS
# =================================================================================
"""
The JD says: "Ensure code quality through rigorous testing, peer reviews,
and adherence to best practices."

HOW DO YOU TEST AN AGENT? (This is a common interview question)

LEVEL 1: UNIT TESTS (test individual tools)
    Each @tool function is a regular Python function.
    Test it independently with known inputs and expected outputs.

    def test_search_products():
        result = search_products("tap extension")
        assert "tap" in result.lower()
        assert len(result) > 0

LEVEL 2: INTEGRATION TESTS (test tool + LLM together)
    Give the agent a known question, verify it calls the right tool.

    def test_agent_routes_correctly():
        result = agent.invoke({"messages": [HumanMessage("search for tap extensions")]})
        # Verify the agent called search_products tool
        tool_calls = [m for m in result["messages"] if hasattr(m, "tool_calls")]
        assert any("search" in tc["name"] for tc in tool_calls[0].tool_calls)

LEVEL 3: END-TO-END TESTS (test full workflow)
    Give a real question, verify the final answer is correct.

    def test_full_rag_pipeline():
        answer = rag_chain.invoke("What RAG patterns does DocSage use?")
        assert "Agentic" in answer
        assert "Corrective" in answer

LEVEL 4: EVALUATION (measure quality at scale)
    Run 100+ test questions, measure:
    - Faithfulness (is answer grounded in docs?)
    - Answer relevance (does it address the question?)
    - Tool selection accuracy (did it pick the right tool?)

CHALLENGES IN TESTING AGENTS:
    - Non-deterministic: same input can give different outputs
    - Solution: test for PROPERTIES not exact strings
      (e.g., "answer mentions revenue" not "answer equals 'Revenue was $10M'")
    - Use temperature=0 for deterministic testing
    - Mock LLM calls in unit tests to avoid API costs

INTERVIEW ANSWER:
    "I test agentic systems at four levels: unit tests for individual tools,
    integration tests to verify the agent routes to the correct tool,
    end-to-end tests for full workflow validation, and evaluation at scale
    using metrics like faithfulness and tool selection accuracy. Since agents
    are non-deterministic, I test for properties rather than exact outputs,
    and use temperature=0 during testing for reproducibility."
"""


# =================================================================================
# SECTION 7: RELIABILITY & ERROR HANDLING IN AGENTS
# =================================================================================
"""
The JD says: "Design, test, and maintain modular components for agentic systems
to enable intelligent automation and orchestration."

WHAT CAN GO WRONG IN AN AGENT:
    1. LLM API goes down (rate limit, timeout, 500 error)
    2. Agent picks the wrong tool
    3. Tool returns an error
    4. Agent gets stuck in an infinite loop
    5. Token limit exceeded (context too long)
    6. Hallucination (agent makes up information)

HOW TO HANDLE EACH:

    PROBLEM 1: API failures
    SOLUTION: Retry with exponential backoff + fallback to secondary model
        try:
            response = llm.invoke(messages)
        except RateLimitError:
            time.sleep(2)
            response = fallback_llm.invoke(messages)

    PROBLEM 2: Wrong tool selection
    SOLUTION: Corrective RAG — grade the result, rewrite query if irrelevant

    PROBLEM 3: Tool errors
    SOLUTION: Try/except in every tool, return error message to agent
        @tool
        def search_products(query: str) -> str:
            try:
                results = api.search(query)
                return str(results)
            except Exception as e:
                return f"Error searching: {str(e)}. Try a different query."

    PROBLEM 4: Infinite loops
    SOLUTION: Recursion limit (we use 20 in DocSage)
        graph.invoke(input, {"recursion_limit": 20})

    PROBLEM 5: Token overflow
    SOLUTION: Compact context window (keep last 10 messages, skip ToolMessages)

    PROBLEM 6: Hallucination
    SOLUTION: Validate node checks if answer is grounded in retrieved docs

INTERVIEW ANSWER:
    "I build reliability into agents at multiple levels: retry with fallback
    for API failures, recursion limits to prevent infinite loops, error handling
    in every tool that returns informative messages to the agent, compact context
    windows to prevent token overflow, and post-generation validation to catch
    hallucination. In DocSage, if the validate node detects hallucination, it
    triggers a self-correction loop — rewrite the query and retry."
"""


# =================================================================================
# SECTION 8: YOUR EXPERIENCE MAPPED TO JPMC JD
# =================================================================================
"""
JD REQUIREMENT                              YOUR EXPERIENCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Design modular agentic components           DocSage: 8 packages, 15+ files, each with
                                            one responsibility. Factory pattern for LLMs.

Agentic frameworks (LangGraph)              Built DocSage with LangGraph StateGraph,
                                            5 nodes, 3 conditional edges, MemorySaver.
                                            Also building 10 agents hands-on.

Strong Python programming                   1.5 years daily Python. Built DocSage,
                                            MCP server, 10 agent projects.

Cloud platforms (AWS)                       Know concepts: ECR, ECS, S3, Bedrock.
                                            Deployed DocSage on HuggingFace Spaces.
                                            MCP server deployed on Kubernetes.

Docker/containerization                     MCP server containerized and deployed.
                                            Know Dockerfile, build, run, push.

LLMs and GenAI                              Daily use: Groq (Llama), Claude (via Bedrock).
                                            Built production systems with LLMs.

RAG                                         DocSage: Agentic + Corrective + Adaptive RAG.
                                            FAISS, HuggingFace embeddings, chunking.

Agile/Jira                                  4 years in agile teams. Sprint planning,
                                            story estimation, daily standups.

Documentation                               Comprehensive docs for all projects.
                                            Can explain complex concepts simply.

Lead a team                                 Collaborated with cross-functional teams.
                                            Mentored junior developers on GenAI concepts.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

YOUR INTRO FOR JPMC (tailored):
    "Hi, I'm Tirupathi Naidu. I have 4 years of experience in software
    development, with the last 1.5 years focused on building production
    agentic AI systems.

    I built DocSage — an enterprise document intelligence agent using
    LangGraph with a 5-node StateGraph implementing Agentic, Corrective,
    and Adaptive RAG patterns. The architecture is fully modular — 8
    packages, factory patterns for LLMs, separate tools, and conditional
    routing with self-correction loops.

    I also built an MCP server with 17+ LLM-callable tools for ERP
    integration, deployed on Kubernetes with OAuth authentication.
    The agent orchestrates product pricing, quote lifecycle, and order
    management through tool calling.

    I work daily with Python, LangGraph, LangChain, FAISS, and Groq API.
    I'm experienced in designing modular, testable agent architectures
    with error handling, streaming, and conversation memory."
"""


# =================================================================================
# SECTION 9: FINANCIAL SERVICES CONTEXT (What JPMC cares about)
# =================================================================================
"""
JPMC's DART team (Data, Analytics and Reporting) in Consumer Banking.
They handle: personal banking, credit cards, mortgages, auto financing.

WHERE AGENTS FIT IN BANKING:
    - Customer support automation (answer account queries)
    - Fraud detection agents (analyze transactions, flag suspicious activity)
    - Document processing (extract data from loan applications, contracts)
    - Compliance checking (verify documents against regulations)
    - Report generation (automated analytics reports)
    - Internal knowledge assistants (search policies, procedures)

BANKING-SPECIFIC CONCERNS: (Personal Identity Info-PII)(Role Based authentication)
    - SECURITY: All data is sensitive (PII, financial data). Agents must
      never leak customer information. RBAC, encryption, audit logging.
    - COMPLIANCE: Regulated industry. Every AI decision must be explainable
      and auditable. No black-box decisions.
    - RELIABILITY: Downtime costs millions. Agents must have fallbacks,
      retries, and graceful degradation.
    - DATA PRIVACY: Customer data cannot leave the organization's boundary.
      Use private LLM deployments (AWS Bedrock, not public OpenAI).

INTERVIEW ANSWER (if asked about financial services):
    "While I haven't worked directly in financial services, the principles
    are the same — security, reliability, and compliance. In my MCP server,
    I implemented OAuth authentication and session management for secure
    API access. In DocSage, I built error handling with fallbacks, recursion
    limits, and validation to ensure reliability. For a banking context,
    I'd add RBAC for data access control, audit logging for every agent
    action, and use AWS Bedrock for private LLM deployment to keep data
    within the organization's boundary."
"""


# =================================================================================
# SECTION 10: 30+ INTERVIEW Q&A FOR JPMC VP
# =================================================================================
"""
The VP (Kathan Joshi) will likely ask high-level architecture + depth questions.
30-minute interview = focused, no fluff. Be concise.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AGENTIC SYSTEMS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q1: "What is an agentic system?"
A: "An AI system where an LLM autonomously decides what actions to take —
   which tools to call, in what order, and when to stop. Unlike a fixed
   chain, an agent reasons about the best next step at each point."

Q2: "What's the difference between a chain and an agent?"
A: "A chain is a fixed sequence: A → B → C, always the same path.
   An agent is dynamic: it decides at each step whether to call tool A,
   tool B, or stop. The LLM is the decision-maker, not the developer."

Q3: "Walk me through an agentic system you built."
A: "DocSage — a document intelligence agent with LangGraph. 5 nodes:
   agent (decides which tool), tools (executes retrieval), grade (checks
   relevance), generate (produces answer), validate (checks hallucination).
   3 conditional edges for dynamic routing. Self-correction loops if
   quality checks fail. MemorySaver for multi-turn conversations."

Q4: "What is LangGraph and why did you choose it?"
A: "LangGraph is a framework for building stateful agent workflows as
   directed graphs. I chose it over AgentExecutor because I needed
   conditional routing (grade → generate OR rewrite), post-generation
   validation, and state persistence. LangGraph gives full control over
   the workflow while AgentExecutor is just a simple loop."

Q5: "What are conditional edges in LangGraph?"
A: "Edges where the next node depends on the output of the current node.
   For example, after grading: if docs are relevant → generate. If not →
   rewrite. The grade function returns a string and the graph routes
   accordingly. This enables dynamic, intelligent workflows."

Q6: "How do you handle state in LangGraph?"
A: "TypedDict with an add_messages reducer. The reducer ensures messages
   APPEND instead of overwrite. All nodes read from and write to this
   shared state. MemorySaver persists state across turns using thread_id."

Q7: "What is tool calling?"
A: "The LLM decides which function to call and with what arguments.
   bind_tools() gives the LLM tool schemas (name, description, parameters).
   The LLM returns tool_calls in its response. The ToolNode executes them.
   The LLM is the brain (decides), ToolNode is the hands (executes)."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SYSTEM DESIGN & ARCHITECTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q8: "How do you design modular agentic components?"
A: "Five principles: (1) Single responsibility — each file does one thing.
   (2) Factory pattern — LLM creation centralized in one factory class.
   (3) Dependency injection — nodes receive LLMs from outside, not create them.
   (4) Configuration over code — all constants in settings.py.
   (5) Separation of concerns — nodes define WHAT, graph defines HOW they connect."

Q9: "How would you deploy this on AWS?"
A: "Containerize each service with Docker. Push to ECR. Deploy on ECS for
   simpler management or EKS for Kubernetes orchestration. LLM calls go to
   AWS Bedrock (private, no data leaves AWS). Vector store on OpenSearch or
   managed Pinecone. Infrastructure defined in Terraform for reproducibility."

Q10: "How do you ensure reliability?"
A: "Multiple layers: retry with exponential backoff for API failures,
    recursion limit (20) to prevent infinite loops, error handling in every
    tool, fallback to secondary model if primary fails, validation node
    catches hallucination, and streaming gives real-time feedback."

Q11: "How do you handle scaling?"
A: "Horizontal scaling: multiple container instances behind a load balancer.
    Async processing for batch operations. Caching (Redis) for repeated
    queries. Rate limiting to protect LLM APIs. Queue-based processing
    (SQS) for high-volume workloads."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RAG & LLM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q12: "What RAG patterns have you implemented?"
A: "Three: Agentic RAG (agent decides which knowledge base to query),
    Corrective RAG (grade documents, rewrite if irrelevant), and Adaptive
    RAG (validate answer for hallucination and relevance after generation)."

Q13: "How do you prevent hallucination?"
A: "Three layers: (1) RAG prompt says 'answer ONLY from context'.
    (2) Grade node filters irrelevant docs before generation.
    (3) Validate node checks if answer is grounded in docs after generation.
    If hallucination detected, system self-corrects."

Q14: "What is Agentic RAG vs basic RAG?"
A: "Basic RAG: one retriever, every question searches the same index.
    Agentic RAG: agent has multiple retrievers and DECIDES which to call
    based on the question. The LLM reads tool descriptions and routes
    intelligently."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PYTHON & CODE QUALITY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q15: "How do you write maintainable Python code?"
A: "Type hints on every function, docstrings for every class and function,
    modular structure (one file = one responsibility), centralized config,
    logging (not print), error handling with meaningful messages, and
    comprehensive comments explaining WHY not WHAT."

Q16: "How do you test agentic systems?"
A: "Four levels: unit tests for tools, integration tests for tool routing,
    end-to-end tests for full workflows, and evaluation at scale with
    metrics like faithfulness and tool selection accuracy. Temperature=0
    for deterministic testing."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BEHAVIORAL / TEAM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q17: "Tell me about a challenging problem you solved."
A: "In DocSage, the agent was losing context in multi-turn conversations.
    ToolMessages (large document chunks) were overflowing the 8K token limit.
    I solved it by building a compact context window — keeping only the last
    10 human/AI messages and skipping ToolMessages. This preserved follow-up
    context while staying within token limits."

Q18: "How do you work in agile teams?"
A: "I've worked in agile for 4 years. Sprint planning, daily standups,
    story estimation, retrospectives. I break large features into small
    Jira stories, deliver incrementally, and communicate blockers early.
    I use Git with feature branches and PR reviews."

Q19: "How do you stay current with AI advancements?"
A: "I actively learn and build. I've implemented 10 different RAG patterns,
    built agents with LangGraph, and stay updated through AI communities,
    research papers, and hands-on experimentation. I apply new techniques
    to my projects within days of learning them."

Q20: "Why JPMC?"
A: "JPMC is building agentic AI at enterprise scale in a regulated industry.
    That's the hardest and most impactful application of this technology.
    I want to work on systems where reliability, security, and compliance
    are non-negotiable — it pushes you to build truly production-grade
    solutions, not just demos."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ADVANCED / FOLLOW-UP
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q21: "What is AutoGen?"
A: "Microsoft's framework for multi-agent conversations. Multiple agents
    talk to each other to solve a task. Different from LangGraph which is
    a single-agent graph. AutoGen is better for scenarios where you need
    a 'coder' agent and a 'reviewer' agent collaborating."

Q22: "What is Google ADK?"
A: "Agent Development Kit — Google's framework for building agents.
    Similar to LangGraph but integrated with Google's ecosystem (Gemini,
    Vertex AI). I'm familiar with the concepts but primarily use LangGraph."

Q23: "How do you handle secrets/API keys in production?"
A: "Never in code. Use environment variables loaded from .env locally,
    and AWS Secrets Manager or Parameter Store in production. Docker
    containers receive secrets via environment injection at runtime."

Q24: "What is infrastructure as code?"
A: "Defining all infrastructure (servers, databases, networks) in code files
    (Terraform or CloudFormation). Benefits: version controlled, reproducible,
    reviewable in PRs, and automated deployment. No manual clicking in AWS console."

Q25: "How would you monitor an agentic system in production?"
A: "Three layers: (1) Application logs — every agent decision, tool call, and
    error logged with structured logging. (2) Tracing — LangSmith or similar
    to trace each query end-to-end through all nodes. (3) Metrics — latency,
    error rate, token usage, tool selection accuracy. Alerts on anomalies."
"""

print("=" * 60)
print("JPMC Agentic Development — Interview Preparation")
print("=" * 60)
print()
print("10 Sections:")
print("  1.  Company & Team Overview (DART, Consumer Banking)")
print("  2.  What is an Agentic System")
print("  3.  Agentic Frameworks (LangGraph, AutoGen, Google ADK)")
print("  4.  Agentic Design Patterns (Modular Components)")
print("  5.  AWS, Docker & Deployment")
print("  6.  Testing Agentic Systems (4 levels)")
print("  7.  Reliability & Error Handling")
print("  8.  Your Experience Mapped to JD")
print("  9.  Financial Services Context")
print("  10. 25 Interview Q&A (VP-level)")
print()
print("Interview: Thursday May 7, 5:00-5:30 PM IST")
print("Interviewer: Kathan Joshi (Vice President)")
print("Format: 30-min Zoom with VP")
print("=" * 60)
