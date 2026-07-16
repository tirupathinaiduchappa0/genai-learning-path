"""
===================================================================================
PHASE 8 — ARCHITECTURE TRADE-OFFS & LEADERSHIP (HashedIn by Deloitte — Lead)
===================================================================================

WHY THIS IS PHASE 8:
    The JD mentions "trade-offs" and "architectural decision-making" repeatedly,
    AND it's a LEAD role: "Lead code reviews, enforce best practices, mentor
    team members," "effort estimations, resource planning, risk management,"
    "break down work into actionable tasks," "present to technical and business
    audiences." This phase covers the two things that make you a LEAD, not just
    a senior engineer: (1) articulating trade-offs, (2) the leadership skills.

DEPTH LEVEL: Technical Lead. Decision frameworks + leadership behaviors.

SECTIONS:
    1.  The Trade-off Mindset (how to articulate ANY decision)
    2.  The Major Architecture Trade-offs (the decision catalog)
    3.  GenAI-Specific Trade-offs (RAG vs fine-tune, build vs buy, etc.)
    4.  Effort Estimation & Risk Management
    5.  Breaking Down Work (requirements -> tasks)
    6.  Code Review & Enforcing Best Practices
    7.  Mentoring & Driving Code Quality
    8.  Presenting to Technical & Business Stakeholders (POCs)
    9.  Leadership Behavioral Questions (STAR answers)
    10. Interview Q&A + GOLDEN LESSONS
===================================================================================
"""


# =================================================================================
# SECTION 1: THE TRADE-OFF MINDSET (how to articulate ANY decision)
# =================================================================================
'''
The single most important LEAD skill the JD tests. A senior knows tech; a lead
makes and DEFENDS decisions with trade-offs. This mindset runs through every
answer in the interview.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE GOLDEN SENTENCE STRUCTURE (use it on EVERY decision):
    "I chose X because Y. The trade-off is Z. Given our context/scale W,
     it's the right call — and here's when I'd choose differently."

    This 4-part shape proves you:
    - Made a deliberate choice (X because Y).
    - Understand the cost (trade-off Z).
    - Tied it to context (W) — not dogma.
    - Know the alternatives and their conditions (when I'd switch).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE CORE PRINCIPLE: "IT DEPENDS" — BUT SAY ON WHAT.
    The wrong answer to "X or Y?" is a dogmatic "always X." The right answer
    names the DECIDING FACTORS, then makes a call.
    "It depends on [scale / latency / team size / consistency needs / budget].
     For most cases I'd start with X because... but if [factor] then Y."

THE DIMENSIONS YOU WEIGH (your checklist for any decision):
    - Scale & performance (now and projected)
    - Latency requirements
    - Consistency vs availability
    - Cost (infra + LLM tokens + dev time + maintenance)
    - Complexity / maintainability / team familiarity
    - Time to market
    - Reliability / risk
    - Security / compliance
    - Flexibility / lock-in

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

KEY LEAD PRINCIPLES TO VOICE:
    - "There's no perfect architecture, only the right one for the context."
    - "Start simple; add complexity only when a real requirement demands it."
      (YAGNI — You Aren't Gonna Need It. Avoid premature optimization.)
    - "Reversible decisions: decide fast. Irreversible: deliberate carefully."
      (Amazon's one-way vs two-way doors.)
    - "Optimize for the team's ability to maintain it, not for cleverness."

INTERVIEW ANSWER:
    "My rule for any architectural decision is to make it explicit: I chose X
    because Y, the trade-off is Z, and given our scale and constraints it's the
    right call — and I'll name when I'd choose differently. I avoid dogmatic
    answers; I name the deciding factors — scale, latency, consistency, cost,
    team familiarity, time to market — and decide from there. I start simple
    and add complexity only when a requirement demands it, decide fast on
    reversible choices and deliberate on irreversible ones, and optimize for
    what the team can maintain, not for cleverness."
'''


# =================================================================================
# SECTION 2: THE MAJOR ARCHITECTURE TRADE-OFFS (the decision catalog)
# =================================================================================
'''
The classic decisions you must be ready to discuss. Each: when X, when Y.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MONOLITH vs MICROSERVICES:
    Monolith: one deployable. Simpler to build/test/deploy/debug; fast early;
        no network/distributed complexity. Scales as one unit; can become a
        big ball of mud.
    Microservices: independent services. Independent scaling/deploy, team
        autonomy, fault isolation; BUT distributed complexity (network, data
        consistency, observability, ops overhead).
    DECISION: "Start with a (well-modularized) monolith — most teams don't
        have the scale or org size to justify microservices early. Split out
        services when you have clear bounded contexts, independent scaling
        needs, or large teams stepping on each other. Don't pay the distributed
        tax prematurely." (The "monolith first" stance is the senior consensus.)

SYNC vs ASYNC (request handling — ties to Phases 1-3):
    Sync: simple, client waits, immediate result. Async: non-blocking, handles
    long work + high concurrency, but more complexity (queues, callbacks).
    DECISION: sync for fast interactive; async for long-running or high-
    concurrency I/O. (LLM apps lean async.)

SQL vs NoSQL (Phase 4):
    SQL: relationships, ACID, complex queries. NoSQL: scale, flexible schema,
    specific shapes. DECISION: SQL default; NoSQL for specific needs; often
    polyglot.

CONSISTENCY vs AVAILABILITY (CAP — Phase 4):
    Strong consistency (reject during partition) vs high availability (serve
    possibly-stale). DECISION: by domain — banking needs consistency, a feed
    can tolerate eventual consistency.

CACHING (performance vs freshness/complexity):
    Cache = speed + cost savings, BUT staleness + invalidation complexity
    ("cache invalidation is one of the two hard problems"). DECISION: cache
    read-heavy, tolerant-of-slight-staleness data; TTL + event invalidation.

NORMALIZATION vs DENORMALIZATION (Phase 4):
    Normalize for integrity, denormalize for read speed. DECISION: normalize
    first, denormalize measured hot paths.

BUILD vs BUY (and OSS vs managed):
    Build: control + fit, BUT time + maintenance. Buy/managed: speed + less ops,
    BUT cost + lock-in + less control. DECISION: buy/managed for
    non-differentiating infra (auth, vector DB, queue); build your
    differentiator. "Don't build what isn't your core value."

VERTICAL vs HORIZONTAL SCALING:
    Vertical: bigger machine, simple, has a ceiling. Horizontal: more machines,
    scales further, needs statelessness + LB. DECISION: vertical first (simple),
    horizontal when you hit limits / need HA.

REST vs GraphQL vs gRPC (Phase 3):
    REST public/CRUD; GraphQL flexible client queries; gRPC fast internal.

STRONG TYPING / VALIDATION vs FLEXIBILITY:
    Strict (Pydantic, schemas) = safety + clarity; flexible = speed but bugs.
    DECISION: strict at boundaries (API, DB, LLM I/O); pragmatic internally.

INTERVIEW ANSWER (template for any "X vs Y"):
    "Both have a place. X gives [benefit] at the cost of [cost]; Y gives
    [benefit] at the cost of [cost]. The deciding factors are [scale / team /
    consistency / time]. For our context I'd choose [X], and I'd switch to [Y]
    when [condition]. For example, monolith-first because most teams don't have
    the scale to justify microservices, splitting out services only at clear
    bounded contexts or independent scaling needs."
'''


# =================================================================================
# SECTION 3: GenAI-SPECIFIC TRADE-OFFS (RAG vs fine-tune, build vs buy, etc.)
# =================================================================================
'''
The AI decisions a GenAI lead must own. These are high-probability for this role.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RAG vs FINE-TUNING vs PROMPT ENGINEERING vs LONG-CONTEXT:
    PROMPT ENGINEERING: cheapest, fastest; first thing to try. Limited by what
        you can fit/instruct.
    RAG: inject fresh, updatable, attributable KNOWLEDGE at query time. Best
        for facts that change, large corpora, citations. Doesn't change model
        behavior/style.
    FINE-TUNING: bake in STYLE, FORMAT, BEHAVIOR, domain tone, or a narrow task.
        Costly, slow to update, needs data; NOT for fresh facts (they go stale).
    LONG-CONTEXT: just stuff everything in the prompt if the data is small
        enough. Simple but costly per call + "lost in the middle".
    DECISION: "Prompt-engineer first, RAG for knowledge, fine-tune for behavior/
        format, long-context when the data is small. They COMBINE — RAG for
        facts + fine-tune for tone. Most problems are solved by RAG +
        good prompting, not fine-tuning."

AGENT vs WORKFLOW (Phase 6):
    Workflow (fixed, reliable, cheap) vs agent (dynamic, flexible, costly).
    DECISION: least autonomy that solves it.

BIG MODEL vs SMALL MODEL (per task):
    Big (GPT-4-class): better reasoning, costlier/slower. Small (8B/mini):
    cheaper/faster, fine for routing, grading, simple extraction.
    DECISION: route — small for simple steps, big for hard reasoning. Major
    cost lever.

MANAGED LLM API vs SELF-HOSTED OPEN MODEL:
    Managed (OpenAI/Anthropic): best quality, zero ops, BUT cost + data leaves
    infra + lock-in. Self-hosted (Llama/Mistral): data control + no per-token
    cost + customizable, BUT infra/ops + GPU cost + usually lower quality.
    DECISION: managed for quality/speed-to-market; self-hosted for data
    residency/compliance/high-volume cost. Often hybrid (Azure OpenAI bridges).

VECTOR DB: pgvector vs DEDICATED (Phase 4):
    pgvector (one system, metadata joins, moderate scale) vs dedicated
    (Pinecone/Qdrant, scale + ANN features). DECISION: pgvector until scale
    demands a dedicated store.

BUILD AGENT FRAMEWORK vs USE ONE (Phase 6):
    Build: full control, BUT reinventing + maintenance. Use (LangGraph/CrewAI):
    faster, battle-tested. DECISION: use a framework; build only if you have
    unusual needs they can't meet.

INTERVIEW ANSWER:
    "The big GenAI trade-off is RAG versus fine-tuning versus prompting. I
    prompt-engineer first, use RAG for fresh, attributable knowledge, and
    fine-tune only for style, format, or a narrow behavior — never for facts
    that go stale. They combine: RAG for knowledge, fine-tune for tone. For
    cost I route models — a small model for routing and grading, a big model
    only for hard reasoning. Managed APIs for quality and speed to market,
    self-hosted open models for data residency or high-volume cost, often a
    hybrid like Azure OpenAI. And I use an existing agent framework rather than
    building one unless we have needs it genuinely can't meet — don't build
    what isn't your core value."
'''


# =================================================================================
# SECTION 4: EFFORT ESTIMATION & RISK MANAGEMENT
# =================================================================================
'''
JD: "Strong skills in effort estimation, resource planning, and risk
assessment, ensuring timely and predictable delivery." A lead OWNS delivery.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HOW TO ESTIMATE EFFORT (give a real method, not "I guess"):
    1. DECOMPOSE — break the feature into small tasks (Section 5). You can only
       estimate what you can see.
    2. ESTIMATE EACH TASK — story points (relative) or time; use historical
       velocity, not gut. Estimate in RANGES (best/likely/worst), not single
       numbers.
    3. ACCOUNT FOR THE HIDDEN — testing, code review, integration, deployment,
       docs, meetings, bug-fixing. The "code" is ~40-60% of real effort.
    4. ADD A BUFFER for uncertainty/unknowns (and say WHY).
    5. CALIBRATE with the team — estimates from those doing the work (planning
       poker), not top-down.

    AI-SPECIFIC ESTIMATION NUANCE: GenAI work has HIGH uncertainty (will the
    model perform? prompt iteration is unpredictable). So: TIME-BOX a POC/spike
    first to reduce unknowns, THEN estimate the build. "I'd spike the risky
    part for 3 days, then give a confident estimate."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RISK MANAGEMENT (identify -> assess -> mitigate -> monitor):
    - IDENTIFY risks: technical (does the model work?), dependency (3rd-party
      API), resource (key person), scope creep, integration, data quality.
    - ASSESS: likelihood x impact -> prioritize.
    - MITIGATE: spikes/POCs for technical risk, fallbacks for dependency risk,
      buffers for schedule risk, early integration for integration risk.
    - MONITOR: track risks through delivery; re-assess at each milestone.

    THE HIGHEST-LEVERAGE MOVE: tackle the RISKIEST/most-uncertain part FIRST
    (a spike), so you fail fast and learn early — not at the deadline.

INTERVIEW ANSWER:
    "I estimate by decomposing the work into small tasks, estimating each in
    ranges using team velocity rather than gut, and explicitly accounting for
    the hidden effort — testing, review, integration, deployment — which is
    often half the real work, plus a buffer for uncertainty. For GenAI
    specifically there's high uncertainty in whether the model performs and how
    much prompt iteration it'll take, so I time-box a spike on the risky part
    first to shrink the unknowns, then give a confident estimate. On risk I
    identify by likelihood and impact, mitigate technical risk with early
    POCs, dependency risk with fallbacks, schedule risk with buffers, and I
    always tackle the riskiest piece first so we fail fast, not at the
    deadline."
'''


# =================================================================================
# SECTION 5: BREAKING DOWN WORK (requirements -> tasks)
# =================================================================================
'''
JD: "breaking down complex requirements into actionable tasks, creating
detailed work breakdown structures, aligning deliverables with objectives."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE WORK BREAKDOWN PROCESS:
    1. UNDERSTAND THE GOAL — the business objective + success criteria. Ask
       clarifying questions; don't build the wrong thing.
    2. DEFINE REQUIREMENTS — functional + non-functional; acceptance criteria.
    3. DECOMPOSE — epic -> features/stories -> tasks. Each task should be:
       - SMALL (ideally < 1-2 days), INDEPENDENT where possible, TESTABLE,
         with a clear "done" (Definition of Done).
    4. IDENTIFY DEPENDENCIES — what must come before what; sequence + parallelize.
    5. PRIORITIZE — MoSCoW (Must/Should/Could/Won't) or value vs effort. Deliver
       a thin VERTICAL SLICE (end-to-end working feature) first, not horizontal
       layers — get value + feedback early.
    6. ALIGN to business objectives — every task traces to a goal; cut work
       that doesn't.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GOOD TASK PRINCIPLES (INVEST for stories):
    Independent, Negotiable, Valuable, Estimable, Small, Testable.

VERTICAL SLICE (key lead concept):
    Build a thin end-to-end working path first (UI -> API -> DB for ONE
    feature) rather than "all the DB, then all the API, then all the UI."
    Delivers value early, surfaces integration risk early, enables feedback.

INTERVIEW ANSWER:
    "I start from the business goal and success criteria, then define
    functional and non-functional requirements with acceptance criteria. I
    decompose epic to story to task, keeping each task small, testable, and
    with a clear definition of done, and I map dependencies so I can sequence
    and parallelize. I prioritize with MoSCoW or value-versus-effort and
    deliver a thin vertical slice end-to-end first — one feature working through
    the whole stack — rather than building horizontal layers, because that
    delivers value early and surfaces integration risk before the deadline.
    Every task traces back to a business objective; if it doesn't, I question
    it."
'''


# =================================================================================
# SECTION 6: CODE REVIEW & ENFORCING BEST PRACTICES
# =================================================================================
'''
JD: "Lead code reviews, enforce best practices... drive continuous improvement
in code quality and maintainability." This is core lead behavior.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHAT I LOOK FOR IN A REVIEW (in priority order):
    1. CORRECTNESS — does it do the right thing? edge cases? does the logic hold?
    2. SECURITY — injection, authz, secrets, input validation, PII.
    3. DESIGN/ARCHITECTURE — right abstractions? fits the system? not
       over-engineered? separation of concerns?
    4. TESTS — adequate coverage of the change + edge cases.
    5. READABILITY/MAINTAINABILITY — clear names, simple, the next dev can
       understand it.
    6. PERFORMANCE — obvious issues (N+1, blocking the loop, unbounded queries).
    7. STYLE/CONVENTIONS — last; automate with linters/formatters so reviews
       focus on substance, not style nits.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HOW TO REVIEW WELL (the human side — a lead differentiator):
    - BE KIND + SPECIFIC: critique the code, not the person; explain WHY, suggest
      how. "What if input is empty here?" beats "this is wrong."
    - DISTINGUISH blocking issues from nits ("nit:" prefix for optional).
    - ASK, don't dictate: "Could we extract this?" invites discussion.
    - PRAISE good code too — reviews aren't only criticism.
    - TIMELY: review fast; a blocked PR blocks the team.
    - TEACH: reviews are a mentoring channel — explain the reasoning so people
      learn, not just comply.

ENFORCING BEST PRACTICES SYSTEMICALLY (not just per-PR):
    - AUTOMATE: linters (ruff/flake8), formatters (black), type checks (mypy),
      pre-commit hooks, CI gates (tests must pass, coverage threshold).
    - STANDARDS: a documented style guide, PR templates, definition of done.
    - ARCHITECTURE guardrails: design reviews for big changes, ADRs
      (Architecture Decision Records) to document why.
    "Automate the mechanical stuff so human review focuses on logic, design,
    and security."

INTERVIEW ANSWER:
    "In a review I prioritize correctness and security first, then design and
    tests, then readability, and I leave style to automated linters and
    formatters so reviews focus on substance. The human side matters as much:
    I critique the code not the person, explain the why and suggest a fix, mark
    nits as optional versus blocking, ask rather than dictate, and review
    quickly because a blocked PR blocks the team. I treat reviews as a mentoring
    channel — explaining reasoning so people learn. Systemically I enforce
    standards with linters, type checks, pre-commit hooks, and CI gates, plus
    design reviews and ADRs for big decisions, so the mechanical stuff is
    automated and humans focus on logic and architecture."
'''


# =================================================================================
# SECTION 7: MENTORING & DRIVING CODE QUALITY
# =================================================================================
'''
JD: "mentoring team members, and driving continuous improvement." Leads grow
people, not just code.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MENTORING APPROACH:
    - MEET PEOPLE WHERE THEY ARE — junior needs guidance + safety to ask;
      mid-level needs stretch + ownership; both need feedback.
    - TEACH THE WHY, not just the what — explain reasoning so they generalize.
    - GUIDE, DON'T SOLVE — ask questions that lead them to the answer ("what
      happens if this fails?") so they build judgment, not dependence.
    - PAIR PROGRAMMING / shadowing for hard problems.
    - DELEGATE with support — give ownership of meaningful work + a safety net.
    - GIVE FEEDBACK continuously, specific + actionable, both praise + growth.
    - CREATE PSYCHOLOGICAL SAFETY — people must feel safe to ask, fail, learn.

DRIVING CONTINUOUS IMPROVEMENT (team-level):
    - Retrospectives -> concrete action items.
    - Tech-debt budget (e.g., 20% time) — pay it down deliberately.
    - Knowledge sharing: brown-bags, design reviews, documentation, ADRs.
    - Metrics: cycle time, defect rate, review turnaround — improve the system.
    - Lead by example: your own code/reviews set the bar.

INTERVIEW ANSWER:
    "I mentor by meeting people where they are — juniors need guidance and a
    safe space to ask, mid-levels need stretch and ownership. I teach the why,
    not just the what, and I guide with questions rather than handing over the
    answer so they build judgment. I pair on hard problems, delegate
    meaningful work with a safety net, and give continuous, specific feedback —
    both praise and growth areas. Psychological safety is essential; people
    have to feel safe to ask and to fail. At the team level I drive improvement
    through retrospectives with real action items, a tech-debt budget,
    knowledge sharing and ADRs, and leading by example in my own code and
    reviews."
'''


# =================================================================================
# SECTION 8: PRESENTING TO TECHNICAL & BUSINESS STAKEHOLDERS (POCs)
# =================================================================================
'''
JD: "present architectural decisions, technical findings, and recommendations
to both technical and business audiences" + "POC findings... performance
benchmarks, risks, strategic recommendations... translate POCs into production."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ADAPT THE MESSAGE TO THE AUDIENCE:
    - BUSINESS audience: lead with OUTCOMES + value + cost + risk + timeline.
      No jargon. "This cuts ticket handling time 40%, costs ~$X/month, ships in
      6 weeks, main risk is data quality which we'll de-risk with a 2-week POC."
    - TECHNICAL audience: architecture, trade-offs, benchmarks, how it works.
    - The skill: SAME project, DIFFERENT framing. Translate tech <-> business value.

PRESENTING A POC (structure):
    1. PROBLEM + GOAL — what we set out to validate.
    2. APPROACH — what we built (briefly).
    3. FINDINGS — does it work? BENCHMARKS (latency, accuracy, cost) — numbers.
    4. RISKS + LIMITATIONS — honest. (Builds trust; shows maturity.)
    5. RECOMMENDATION — go / no-go / iterate, with reasoning.
    6. NEXT STEPS + what it takes to PRODUCTIONIZE (effort, cost, timeline).

    POC -> PRODUCTION (the JD stresses this): a POC proves feasibility but
    isn't production. The gap: scalability, reliability, security, monitoring,
    testing, error handling, cost optimization. State the gap explicitly and
    plan for it. "The POC validated the approach; productionizing needs X weeks
    for scale, security, observability, and eval."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

INTERVIEW ANSWER:
    "I adapt the framing to the audience. For business stakeholders I lead with
    outcomes, value, cost, risk, and timeline in plain language; for technical
    audiences I go into architecture, trade-offs, and benchmarks. Same project,
    different framing. When presenting a POC I cover the problem and goal, what
    we built, findings with hard benchmarks on latency, accuracy, and cost, the
    risks and limitations honestly, and a clear go/no-go recommendation with
    next steps. And I'm explicit that a POC proves feasibility but isn't
    production — productionizing it needs scalability, security, observability,
    testing, and cost work, which I estimate so the business decides with eyes
    open."
'''


# =================================================================================
# SECTION 9: LEADERSHIP BEHAVIORAL QUESTIONS (STAR answers)
# =================================================================================
'''
Lead interviews include behavioral questions. Use STAR: Situation, Task, Action,
Result. Prepare YOUR real stories (DocSage, MCP server, attribute pipeline,
mentoring). Below are frameworks — fill with your specifics.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE STAR METHOD:
    SITUATION — context (brief). TASK — your responsibility. ACTION — what YOU
    did (the bulk; use "I"). RESULT — outcome, ideally quantified + a learning.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

COMMON QUESTIONS + WHAT THEY PROBE (prep a story for each):

    Q: "Tell me about a hard technical decision you made."
       -> A trade-off decision. Use the X-because-Y-trade-off-Z structure.
       (e.g., choosing LangGraph for DocSage's controlled self-correction flow
       over a simpler chain — control vs simplicity.)

    Q: "A time you disagreed with a teammate / on architecture."
       -> Show you listen, use data/POC to decide, and disagree-and-commit.

    Q: "A time you mentored someone / helped the team."
       -> Section 7 behaviors with a concrete person + outcome.

    Q: "A project that failed / a mistake you made."
       -> Honesty + ownership + the LEARNING + what you changed. (Maybe: the
       prompt-versioning gap you learned, or an interview learning turned into
       a process.)

    Q: "How do you handle tight deadlines / scope pressure?"
       -> Prioritize (MoSCoW), vertical slice, negotiate scope, communicate
       early, de-risk first.

    Q: "A time you took ownership beyond your role."
       -> "Out of my own interest I built DocSage..." — initiative story.

    Q: "How do you handle a teammate who isn't performing / conflict?"
       -> Empathy first (understand why), specific feedback, support, clear
       expectations, escalate only if needed.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

INTERVIEW POINT:
    "I structure behavioral answers with STAR — situation, task, action,
    result — keeping the focus on what I did and ending with a quantified
    outcome and a learning. For a hard decision I'd talk through choosing
    LangGraph for DocSage's controlled self-correction loop over a simpler
    chain — trading some simplicity for the control the use case needed — and
    for initiative, building DocSage end to end out of my own interest."
'''


# =================================================================================
# SECTION 10: INTERVIEW Q&A + GOLDEN LESSONS
# =================================================================================
'''
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Q&A
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q1. Monolith or microservices?
A:  "Monolith-first, well-modularized — most teams lack the scale/org to justify
    microservices early. Split out services at clear bounded contexts or
    independent scaling needs. Don't pay the distributed tax prematurely."

Q2. RAG or fine-tuning?
A:  "RAG for fresh, attributable knowledge; fine-tuning for style/format/
    behavior, not facts. Prompt first; they combine. Most problems need RAG +
    good prompting, not fine-tuning."

Q3. How do you decide between two technologies?
A:  "Name the deciding factors — scale, latency, consistency, cost, team
    familiarity, time to market — then choose, state the trade-off, and say
    when I'd switch. No dogma."

Q4. How do you estimate a project?
A:  "Decompose into small tasks, estimate in ranges using velocity, account for
    hidden effort like testing and review, add a buffer. For GenAI I spike the
    risky part first to cut uncertainty, then estimate."

Q5. How do you break down a complex requirement?
A:  "Goal and success criteria, then epic->story->task, each small and
    testable with a definition of done, mapped dependencies, prioritized by
    value, delivered as a vertical slice end-to-end first."

Q6. What do you look for in a code review?
A:  "Correctness and security first, then design and tests, then readability;
    style is automated. I critique code not people, explain the why, and treat
    it as mentoring."

Q7. How do you mentor a struggling engineer?
A:  "Understand why first, give specific actionable feedback, pair on hard
    problems, set clear expectations with support, and create safety to ask and
    fail."

Q8. How do you present a POC to executives?
A:  "Outcomes, value, cost, risk, timeline in plain language; honest
    limitations; a clear go/no-go; and the gap to productionize — scale,
    security, observability — with an estimate."

Q9. How do you handle disagreement on architecture?
A:  "Listen to understand, bring data or a quick POC to decide objectively, and
    disagree-and-commit once we choose."

Q10. How do you manage technical debt?
A:  "Make it visible, budget time to pay it down (e.g., 20%), prioritize debt
    that slows delivery or risks reliability, and prevent new debt with reviews
    and standards."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GOLDEN LESSONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. EVERY DECISION = "X because Y, trade-off Z, switch when W." The lead mindset.
2. NO DOGMA — "it depends on [factors]" then DECIDE. Name the deciding factors.
3. START SIMPLE; ADD COMPLEXITY ONLY WHEN A REQUIREMENT DEMANDS IT (YAGNI).
4. RAG for knowledge, fine-tune for behavior, prompt first; they combine.
5. ESTIMATE BY DECOMPOSING + RANGES + HIDDEN WORK + BUFFER; spike GenAI risk first.
6. DELIVER VERTICAL SLICES; tie every task to a business objective.
7. REVIEW: correctness+security first, automate style, critique code not people.
8. MENTOR BY TEACHING THE WHY + GUIDING WITH QUESTIONS + PSYCHOLOGICAL SAFETY.
9. PRESENT TO THE AUDIENCE — business: outcomes/cost/risk; tech: design/trade-offs.
10. A POC ISN'T PRODUCTION — name the gap (scale, security, observability) + plan it.

ONE-LINE CHEAT SHEET:
    Trade-off sentence: "X because Y; trade-off Z; switch when W."
    Decisions: monolith-first, SQL-default, sync-for-fast/async-for-long,
      RAG-for-facts/fine-tune-for-behavior, small-model-route, buy-non-core.
    Estimate: decompose + ranges + hidden work + buffer; spike risk first.
    Breakdown: goal -> epic/story/task (INVEST) -> vertical slice -> trace to value.
    Review: correctness>security>design>tests>readability; automate style; be kind.
    Mentor: teach why, guide with questions, delegate w/ safety, feedback, safety.
    Stakeholders: adapt framing; POC = problem/findings/benchmarks/risks/reco/next.
    Always: start simple, decide with trade-offs, optimize for maintainability.
'''


# =================================================================================
# RUN SUMMARY
# =================================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 8 — ARCHITECTURE TRADE-OFFS & LEADERSHIP")
    print("=" * 70)
    print()
    print("THE LEAD MINDSET (use on EVERY decision):")
    print('  "I chose X because Y. Trade-off is Z. Given context W it fits —')
    print('   and here is when I would choose differently."')
    print()
    print("KEY ARCHITECTURE CALLS:")
    print("  monolith-first | SQL-default | sync-fast/async-long")
    print("  RAG-for-facts / fine-tune-for-behavior / prompt-first")
    print("  small-model-route | buy-non-core / build-your-differentiator")
    print()
    print("LEADERSHIP (the JD stresses these):")
    print("  Estimate: decompose + ranges + hidden work + buffer; spike risk first")
    print("  Breakdown: goal -> story (INVEST) -> vertical slice -> trace to value")
    print("  Review: correctness>security>design>tests; automate style; be kind")
    print("  Mentor: teach the WHY, guide with questions, psychological safety")
    print("  Stakeholders: business=outcomes/cost/risk | tech=design/trade-offs")
    print("  POC != production -> name the gap (scale/security/observability)")
    print()
    print("=" * 70)
    print("MEDIUM-PRIORITY BLOCK COMPLETE (Phases 6, 7, 8).")
    print("Remaining (lower priority): Phase 5 (Testing/Tooling), 9 (Cloud), 10 (Mock).")
    print("=" * 70)
