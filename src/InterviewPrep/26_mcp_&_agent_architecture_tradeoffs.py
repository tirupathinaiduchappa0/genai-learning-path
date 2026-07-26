"""
===================================================================================
MCP ARCHITECTURE, AGENT ORCHESTRATION & MODEL TRADE-OFFS — Interview Debrief
===================================================================================

THE INTERVIEW YOU JUST FINISHED (July 2026):
    Three questions, three different failure modes:
    1. "Why one MCP server instead of one per domain (HR, sales, IT, ERP)?"
       -> Answered "USB device" analogy. Interviewer: 50% satisfied.
    2. "What if you put every sub-agent into ONE super-agent instead of
       an orchestrator + sub-agents pattern?"
       -> Answered "context overlap, gets bulky." Correct, but missed the
          exact term he wanted: ATTENTION DILUTION.
    3. "Model A: 90% accuracy, 30 min. Model B: 99.99% accuracy, 3 days.
       Which do you recommend?"
       -> Recovered well by bringing in prediction FREQUENCY (monthly/
          biweekly forecast doesn't need a 30-min SLA).

WHY THIS FILE EXISTS:
    You were 50-70% correct on all three. That's a "pass" but not a
    "strong hire" signal. The gap in each case was the same: you had the
    right INSTINCT but not the precise VOCABULARY or the FULL DECISION
    FRAMEWORK an interviewer wants to hear at 5+ years level. This file
    closes that gap with the exact terms, the trade-off tables, and the
    answer structure to use next time.

SECTIONS:
    1.  Question 1 — Single MCP Server vs Multiple MCP Servers
    2.  Question 2 — Single Super-Agent vs Orchestrator + Sub-Agents (Attention Dilution)
    3.  Question 3 — Accuracy vs Latency/Inference-Time Trade-off
    4.  How a 5+ YOE Answer Differs From a 2-3 YOE Answer (meta-pattern)
    5.  Bonus/Related Questions Found for Future Interviews
    6.  GOLDEN LESSONS
===================================================================================
"""


# =================================================================================
# SECTION 1: SINGLE MCP SERVER vs MULTIPLE MCP SERVERS
# =================================================================================
"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE FULL INTERVIEW EXCHANGE (verbatim context — read this if you're
revisiting this file after 1-3 years and need the WHOLE scenario back,
not just the one-line summary):

    SETUP: This came up while discussing an agentic architecture you'd
    built/described, where an MCP server exposes tools to an LLM agent.

    INTERVIEWER'S QUESTION (paraphrased close to verbatim):
    "Why are you creating only ONE MCP server instead of MULTIPLE MCP
    servers, one for each domain? For example, if you have multiple
    domains like HR, Sales, IT, ERP, etc. — why not create a SEPARATE
    MCP server for each department, instead of ONE MCP server that
    integrates all the departmental tools together?"

    So the concrete scenario he painted: imagine a company with 4+
    departments (HR, Sales, IT, ERP), each with its own set of tools
    (HR: leave requests, payroll lookups; Sales: CRM queries, quote
    generation; IT: ticket creation; ERP: purchase orders, inventory).
    You had proposed ONE MCP server exposing ALL of these tools to the
    agent. He was asking you to justify that design choice against the
    alternative of FOUR separate MCP servers, one per department.

    YOUR ANSWER (as given in the interview):
    "An MCP server is like a USB device — it can connect anything, all
    applications work through it, and it acts as a centralized point.
    That's the purpose of the design."

    INTERVIEWER'S REACTION:
    He said he was about 50% satisfied with that answer, not 100% happy.
    He then added a hint: "Sometimes it's also necessary to create
    MULTIPLE MCP servers depending on the use case" — but did not fully
    explain which use cases, leaving you unsure what he meant. He moved
    on to the next question without you fully resolving this one.

    WHAT THE QUESTION WAS REALLY PROBING:
    Not "what is an MCP server" (you clearly knew that) but "can you
    reason about WHEN a single shared server is the right call versus
    when splitting by domain is the right call." He wanted a trade-off
    analysis, not a definition restated as a design justification.

WHAT YOU SAID (short form):
    "MCP server is like a USB device — it connects anything, acts as a
    centralized point, that's the purpose of the design."

WHY IT WAS ONLY 50% RIGHT:
    The USB analogy explains WHAT an MCP server does (a standard connector
    between an LLM app and tools/data). It does NOT answer WHY you'd choose
    ONE server over MANY. You defended the single-server design without
    acknowledging the real engineering trade-offs that push people toward
    multiple servers. The interviewer explicitly said "sometimes it's
    necessary to create multiple MCP servers depending on the use case" —
    he wanted you to name the use cases.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE REAL ANSWER: IT'S A TRADE-OFF, NOT A RULE.

    ONE MCP SERVER (monolithic / gateway style):
        Pros:
        - Single connection to manage from the client/agent side.
        - Shared auth, shared rate limiting, shared logging/observability
          in one place.
        - Less operational overhead for a small team — one deployment,
          one set of credentials, one place to patch.
        - Easier for the LLM to discover tools if the tool COUNT is small
          (under ~20-30 tools) — fewer moving parts to reason about.
        Cons:
        - Becomes a single point of failure — if HR tools break the
          server process, sales and IT tools go down too (blast radius).
        - Tool count grows unbounded -> the LLM has to pick the right tool
          out of 50+ candidates -> tool-selection accuracy drops.
        - One team/owner has to review every change across every domain.
        - Can't scale or deploy domains independently (a sales tool fix
          forces a redeploy that also touches HR and IT tools).
        - Security/compliance blast radius: HR data (PII) and ERP data
          (financial) sit behind the same auth boundary as everything else.

    MULTIPLE MCP SERVERS (domain-per-server):
        Pros:
        - BLAST RADIUS ISOLATION: an outage or bug in the "sales" server
          doesn't take down HR or ERP.
        - INDEPENDENT OWNERSHIP: the HR team owns/deploys/patches the HR
          MCP server without coordinating with the ERP team.
        - INDEPENDENT SCALING: ERP tools might be called constantly
          (reporting), HR tools rarely — scale each server to its own load.
        - SECURITY BOUNDARIES: HR server can enforce PII-specific access
          control; finance/ERP server enforces SOX-style controls. Blast
          radius of a compromised credential is contained to one domain.
        - TOOL-SELECTION ACCURACY: smaller, focused toolsets per server
          (~10-15 tools) mean the agent picks the right tool more reliably
          instead of confusing "create_ticket" (IT) with "raise_PO" (ERP).
        - VERSIONING/DEPLOYMENT: each domain iterates and ships on its
          own release cadence.
        Cons:
        - More infrastructure to run, monitor, and secure (N servers
          instead of 1).
        - Need a gateway/orchestration layer in front so the agent has
          ONE logical entry point even though there are many servers
          behind it (auth, routing, discovery).
        - Cross-domain workflows (e.g., "onboard new employee" touches
          HR + IT + ERP) now require the orchestrator to call multiple
          servers and stitch results together.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE DECISION FRAMEWORK (what actually separates domains into servers):

    Split into separate MCP servers when ANY of these are true:
    1. DIFFERENT OWNERSHIP/TEAMS — HR team and ERP team should not have
       to coordinate deploys or review each other's tool code.
    2. DIFFERENT SECURITY BOUNDARIES — PII, financial data, and general
       IT tickets have different compliance requirements and access
       policies. Mixing them behind one auth boundary is a risk.
    3. TOOL COUNT IS LARGE — once combined tools exceed ~20-30, LLM tool
       selection accuracy degrades. Split by domain to keep each
       server's toolset focused.
    4. DIFFERENT SCALING/RELIABILITY NEEDS — one domain is high-traffic
       and needs to scale independently, or one domain's outage should
       never affect another's.
    5. DIFFERENT RELEASE CADENCE — domains ship changes at different
       speeds and shouldn't block each other.

    Keep it as ONE MCP server when:
    - Small team, few tools total (well under the 20-30 threshold).
    - All domains share the same auth/security boundary anyway.
    - Speed of iteration matters more than isolation (prototyping, POC,
      internal tooling with a single on-call owner).
    - You're just getting started — start simple, split later when a
      real pain point (an outage, a security review, a scaling limit)
      forces the split. Don't pre-optimize.

    THE PATTERN INDUSTRY IS CONVERGING ON (as of 2026):
    Neither "one server for everything" nor "one server per tool." The
    common production pattern is a GATEWAY/ORCHESTRATOR layer in front
    of a handful of DOMAIN-SCOPED MCP servers (HR server, Sales server,
    ERP server) — giving the agent one logical connection point while
    keeping blast radius, ownership, and scaling separated behind it.
    Source: enterprise MCP architecture patterns from markaicode.com,
    getknit.dev, and Kong's MCP architecture guide (2026).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE 5+ YOE ANSWER (what to say next time):

    "An MCP server standardizes how an agent connects to tools and data —
    that part I got right with the USB analogy. But whether it's ONE
    server or MANY is a trade-off, not a fixed design. I'd split into
    separate MCP servers per domain when: the domains have different
    owners/teams, different security boundaries — HR touches PII, ERP
    touches financial data, those shouldn't share an auth boundary —
    when the combined tool count gets large enough to hurt the agent's
    tool-selection accuracy, typically past 20-30 tools, or when domains
    need to scale or deploy independently. I'd keep it as ONE server for
    a small team, a small tool count, or a prototype, where the
    coordination overhead of multiple servers isn't justified yet. In
    production, the common pattern is a gateway/orchestrator in front of
    a few domain-scoped MCP servers — one logical entry point for the
    agent, but isolated blast radius, ownership, and scaling behind it."

    ONE-LINER IF PRESSED FOR THE SHORT VERSION:
    "Split by domain when ownership, security boundary, or tool count
    diverges. Keep it unified when the team, security model, and scale
    are shared — don't split just because you technically can."
"""


# =================================================================================
# SECTION 2: SINGLE SUPER-AGENT vs ORCHESTRATOR + SUB-AGENTS (ATTENTION DILUTION)
# =================================================================================
"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE FULL INTERVIEW EXCHANGE (verbatim context — read this if you're
revisiting this file after 1-3 years and need the WHOLE scenario back,
not just the one-line summary):

    SETUP: You had described your project's architecture as using a
    SUPERVISOR/ORCHESTRATOR AGENT that routes incoming tasks to multiple
    SPECIALIZED SUB-AGENTS depending on what the task requires (e.g., one
    sub-agent for retrieval, one for grading/validation, one for
    generation — the classic orchestration pattern, similar to what you
    use in your LangGraph-based projects like DocSage).

    INTERVIEWER'S QUESTION (paraphrased close to verbatim):
    "You're saying you have a supervisor agent that routes to multiple
    sub-agents based on the task — that's the orchestration pattern
    you're using. But what happens if, INSTEAD, I don't create any
    sub-agents at all, and I put EVERYTHING into ONE main agent — one
    'super agent' — and I add all the sub-agents' responsibilities and
    logic into that SINGLE agent? What happens if I do that?"

    So the concrete alternative he was asking you to evaluate: take your
    orchestrator + N specialized sub-agents design, and COLLAPSE it into
    ONE agent that has ALL the tools, ALL the system prompt instructions,
    and ALL the context that used to be spread across the sub-agents —
    a monolithic single-agent design instead of a multi-agent one.

    YOUR ANSWER (as given in the interview):
    "If we do that, it could cause issues because the context would
    overlap and everything would be in one place, making it bulky."

    INTERVIEWER'S REACTION:
    He said: "Yeah, you're correct, but there's one word I'm expecting —
    tell me what that concept is called." You said you couldn't
    remember the term. He then told you the correct term is
    "ATTENTION DILUTION" — that's the specific concept/vocabulary he
    was fishing for the entire time. Your reasoning (context overlap,
    bulkiness) was substantively correct; you were just missing the
    precise industry term for the phenomenon you'd already described.

    WHAT THE QUESTION WAS REALLY PROBING:
    He wasn't testing whether multi-agent is "better" in some absolute
    sense — he was testing whether you know the PRECISE MECHANISM by
    which cramming everything into one agent degrades quality, and
    whether you have the vocabulary to name it crisply instead of
    describing it in your own words. This is a common interview pattern:
    they let you arrive at the right idea, then ask for the "textbook"
    term to check if you've engaged with the concept beyond just
    building it once.

WHAT YOU SAID (short form):
    "Context would overlap, everything in one place, bulky." — correct
    instinct, imprecise vocabulary. The term he wanted was ATTENTION
    DILUTION (sometimes also called context pollution / context dilution).

WHAT ATTENTION DILUTION ACTUALLY MEANS:
    Transformer attention is finite and gets DISTRIBUTED across every
    token in context. When you cram one super-agent with the system
    prompt, tool definitions, and conversation history of HR + sales +
    IT + ERP all at once, the model's attention has to spread across all
    of that irrelevant material for every single response — even for a
    query that only concerns one domain. The signal-to-noise ratio drops.
    Relevant tokens get "diluted" among irrelevant ones, and response
    quality/accuracy degrades — NOT because the model ran out of context
    window space, but because attention itself is a limited, distributed
    resource. This is distinct from simply "running out of tokens."
    Source: Anthropic engineering blog, "Building multi-agent systems:
    when and how to use them" (claude.com/blog, Jan 2026) — calls this
    "context pollution": irrelevant information "dilutes attention and
    reduces response quality."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHY A SINGLE SUPER-AGENT WITH EVERYTHING BREAKS DOWN — THE FULL LIST:

    1. ATTENTION DILUTION / CONTEXT POLLUTION (the term he wanted):
       Irrelevant tool defs + irrelevant history dilute the model's
       attention on what actually matters for the current turn.

    2. TOOL-SELECTION DEGRADATION:
       Anthropic's own testing: an agent with 20+ tools spanning
       unrelated domains starts confusing which tool applies — e.g.
       confusing "create_ticket" (IT) with "raise_PO" (ERP). More tools
       in one agent's toolbox = worse selection accuracy, not better
       capability.

    3. CONFLICTING SYSTEM PROMPTS / PERSONAS:
       An HR agent should be empathetic; an IT agent should be terse and
       procedural; a compliance agent should be rigid. Merging all of
       these into one system prompt produces a diluted, inconsistent
       persona that's mediocre at all of them instead of excellent at one.

    4. NO ISOLATION / BLAST RADIUS:
       A bug or bad tool call in one domain's logic sits in the SAME
       context and CAN influence reasoning in a completely unrelated
       domain within the same turn or session.

    5. NO PARALLELISM:
       A single agent is sequential — it processes HR, then sales, then
       IT one after another. An orchestrator can dispatch independent
       sub-agents to work in PARALLEL when the sub-tasks don't depend on
       each other (e.g., research across domains).

    6. MAINTENANCE / BLAME:
       When the super-agent gives a bad answer, you can't tell if the
       HR instructions, the ERP instructions, or the interaction between
       them caused it. Debugging a monolith prompt is much harder than
       debugging one focused sub-agent.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE IMPORTANT NUANCE (don't over-correct into "always use multi-agent"):

    Anthropic's own guidance (and this is worth quoting in an interview
    to sound current): teams often build elaborate multi-agent systems
    only to find a WELL-PROMPTED SINGLE AGENT does just as well, at
    3-10x LOWER token cost. Multi-agent systems consume 3-10x more
    tokens than single-agent for equivalent tasks (duplicated context
    per agent + coordination messages + summarization on handoff).

    So the honest, senior-level answer is NOT "multi-agent is always
    better" — it's "split only when a genuine constraint exists":
    - CONTEXT PROTECTION: a subtask generates high-volume, irrelevant-
      to-the-rest context (order lookup returning 2000+ tokens the main
      task doesn't need) -> isolate it in a sub-agent, return a summary.
    - PARALLELIZATION: independent subtasks (research facet A vs facet
      B) that can run concurrently.
    - SPECIALIZATION: distinct toolsets, conflicting personas, or deep
      domain expertise that would overwhelm a generalist.

    And when you DO split, split along CONTEXT boundaries, not PROBLEM
    TYPE. Splitting "planner / implementer / tester / reviewer" for the
    SAME feature is a classic anti-pattern — it creates a "telephone
    game" where each handoff loses fidelity, and teams have measured
    sub-agents spending MORE tokens coordinating than actually working.
    Splitting HR / Sales / IT / ERP is a GOOD context boundary because
    each domain genuinely needs different tools, different data, and
    different security context — that's exactly the isolation an
    orchestrator pattern is for.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE 5+ YOE ANSWER (what to say next time):

    "If I collapsed everything into one super-agent, the main failure
    mode is attention dilution — attention is a finite, distributed
    resource, and stuffing irrelevant tool definitions and history from
    every domain into one context means the model's attention gets
    spread thin even on queries that only touch one domain. Practically,
    this shows up as worse tool selection once you cross roughly 20-30
    tools, conflicting personas — HR needs to be empathetic, IT needs to
    be terse — and no blast-radius isolation, so a bad interaction in
    one domain can leak into reasoning for another domain in the same
    turn. That said, I wouldn't split reflexively — multi-agent setups
    cost 3-10x more tokens due to duplicated context and coordination
    overhead, so I'd only split when there's a real constraint: context
    volume that's irrelevant to other tasks, genuinely parallelizable
    work, or specialization that a single generalist agent can't sustain.
    And I'd split along CONTEXT boundaries — by domain, like HR vs ERP —
    not by problem type like planner vs implementer, since splitting by
    problem type just creates a telephone game where each handoff loses
    context."

    THE ONE WORD TO NEVER FORGET AGAIN: ATTENTION DILUTION
    (aka context pollution / context dilution — interchangeable terms,
    same underlying mechanism: irrelevant tokens compete for finite
    attention and degrade response quality on the relevant ones.)
"""


# =================================================================================
# SECTION 3: ACCURACY vs LATENCY / INFERENCE-TIME TRADE-OFF
# =================================================================================
"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE FULL INTERVIEW EXCHANGE (verbatim context — read this if you're
revisiting this file after 1-3 years and need the WHOLE scenario back,
not just the one-line summary):

    SETUP / THE SCENARIO HE GAVE (paraphrased close to verbatim):
    "I have TWO models trained for one customer who has a couple of
    years of statistical sales data. Using THREE years of that data, he
    wants to predict NEXT MONTH's sales based on the trained data. He's
    fine-tuned everything, used random forest and other techniques, and
    BOTH models are ready.
        - The FIRST model gives an answer in HALF AN HOUR with 90%
          accuracy.
        - The SECOND model gives 99.99% accuracy but takes THREE DAYS
          to run.
    Which model would you recommend to the customer?"

    So the concrete numbers to remember: 3 years of historical sales
    data, forecasting target = NEXT MONTH's sales, Model A = 30 min /
    90% accuracy, Model B = 3 days / 99.99% accuracy. Both are random
    forest-based (or similar classical ML), both are already trained
    and fine-tuned — the question is purely about which one to ship to
    the customer for production use.

    YOUR FIRST ANSWER (as given in the interview):
    "I'd recommend the second model, since accuracy is the main
    priority."

    INTERVIEWER'S FOLLOW-UP PUSH:
    "What about the first model — why aren't you recommending that?"
    (He was pushing you to justify dismissing the 30-min/90% model
    instead of just repeating that accuracy matters.)

    YOUR RECOVERY (as given in the interview):
    You said accuracy matters so you'd still go with the second model,
    BUT then you added: regarding inference speed/latency, even if it
    takes three days, it occurred to you that THIS KIND OF PREDICTION
    isn't generated every minute or hour — it's usually generated ONCE
    A DAY, ONCE A WEEK, BIWEEKLY, or MONTHLY, not continuously. So the
    3-day runtime isn't actually a practical problem for this use case.

    INTERVIEWER'S REACTION:
    He agreed and confirmed: since the prediction isn't generated
    frequently, if it takes 3 days, it would typically be run once a
    month, biweekly, or quarterly anyway — so the 3-day runtime is fine
    given that it produces accurate data. This was your strongest
    moment in the interview — you found the correct resolving factor
    (prediction CADENCE/frequency) without being told the term, unlike
    Q2 where you needed the term supplied to you.

    WHAT THE QUESTION WAS REALLY PROBING:
    Whether you'd default to "always pick the higher accuracy number"
    (a junior mistake) or actually reason about whether the LATENCY
    number matters at all given how often the prediction is actually
    needed. He was testing judgment about tying a technical metric
    (inference time) to actual business/operational cadence, not
    testing whether you know that "accuracy is good."

WHAT YOU SAID (short form):
    Initially: "Go with the 99.99% model, accuracy is the priority."
    Then, when pushed on the 30-min model: recovered by pointing out
    that sales forecasting isn't generated every minute — it's typically
    run monthly/biweekly/quarterly, so a 3-day inference time is
    acceptable. Interviewer agreed — this was the RIGHT recovery.

WHY THIS WAS A GOOD RECOVERY BUT INCOMPLETE:
    You landed on the correct axis — PREDICTION FREQUENCY / SLA — but
    didn't frame it as the general trade-off it is, and didn't mention
    the other angles a senior candidate is expected to raise: cost,
    overfitting risk, marginal accuracy gain, and reproducibility.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE FULL DECISION FRAMEWORK (name these axes explicitly):

    1. LATENCY / SLA FIT (the one you correctly found):
       What's the actual business cadence? Monthly sales forecasting has
       a loose SLA — a 3-day run fits comfortably inside a monthly cycle
       with room for review before the report is due. This is DIFFERENT
       from, say, real-time fraud detection where even 200ms matters.
       Rule of thumb from industry interviews: match model choice to how
       OFTEN a fresh prediction is actually needed, not to accuracy alone.

    2. MARGINAL ACCURACY GAIN vs COST OF THAT GAIN:
       90% -> 99.99% is a ~10-point jump but ask: is that gain REAL or is
       it a sign of overfitting? Three years of statistical sales data is
       a fairly small dataset for a model to hit 99.99% — that number
       should raise suspicion, not excitement. A senior candidate
       questions the number instead of taking it at face value.

    3. OVERFITTING / GENERALIZATION RISK:
       A model that takes 3 days and reports near-perfect accuracy on
       historical data may be memorizing noise rather than learning a
       generalizable pattern — especially with limited (3 years) data.
       Ask: what's validated on a HOLDOUT set, not just training data?
       Next month's actual sales is the real test, not backtested
       accuracy on data it may have seen patterns from.

    4. COMPUTE COST / INFRASTRUCTURE COST:
       3 days of compute for one forecast has a real dollar cost (cloud
       compute, or a blocked on-prem resource for 3 days). If this needs
       to run for MANY customers/products/regions, 3 days each doesn't
       scale — that's a serious operational constraint even if latency
       is "acceptable" for one customer.

    5. REPRODUCIBILITY / DEBUGGABILITY:
       If the 3-day model gives a bad forecast, how fast can you
       re-run it, tweak a hyperparameter, and get a corrected result
       before the business decision window closes? A 30-min model gives
       you iteration speed; a 3-day model gives you almost none.

    6. INTERPRETABILITY (bonus axis, worth mentioning for a business
       stakeholder like "a customer"):
       Random forest models ARE reasonably interpretable (feature
       importance), which matters when a business customer wants to
       know WHY the forecast says what it says, not just the number.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE STRUCTURED ANSWER PATTERN FOR "WHICH MODEL WOULD YOU RECOMMEND":

    Never answer with a single model name first. Answer with the
    QUESTION YOU'D ASK BEFORE CHOOSING:
    1. "What's the actual prediction cadence the business needs?"
       (monthly forecast vs real-time -> completely different answer)
    2. "Is the 99.99% accuracy validated on a true holdout/out-of-time
       set, or just on training data?" (overfitting check)
    3. "What does 3 days of compute cost, and does it need to scale
       across many customers/SKUs/regions?"
    4. "How much does the accuracy gap from 90% to 99.99% actually cost
       the business if we're wrong?" (tie accuracy to business impact,
       not accuracy for its own sake)

    THEN give a conditional recommendation:
    "If this is a monthly or quarterly forecast for a single customer,
    and the 99.99% is validated out-of-time (not overfit), I'd recommend
    the higher-accuracy model — the 3-day runtime easily fits inside a
    monthly cycle, and at that cadence, latency isn't the bottleneck,
    accuracy is. If this needs to scale to hundreds of customers/SKUs,
    or if that accuracy number isn't holding up on true holdout data, I'd
    default to the faster model and invest in improving it iteratively,
    since I can validate and re-run it 100+ times in the time the other
    model runs once."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE 5+ YOE ANSWER (what to say next time, from the start — not just as a
recovery when pushed):

    "I wouldn't pick a model on accuracy alone — I'd first ask what
    cadence the prediction actually needs. If it's a monthly or
    quarterly sales forecast, a 3-day run comfortably fits the business
    cycle, so the extra accuracy from the second model is basically free
    — latency isn't the bottleneck at that cadence. I'd also sanity-check
    that 99.99% number — with only three years of data, that level of
    accuracy is more likely a sign of overfitting to training data than
    a real generalization gain, so I'd want to see it validated on a
    true out-of-time holdout before trusting it over the 90% model. And
    I'd weigh the compute cost of 3 days if this needs to run across many
    customers or SKUs, not just once. So my answer is conditional: given
    a single customer, monthly cadence, and a properly validated accuracy
    number, I'd recommend the higher-accuracy model — but I'd validate the
    number and the scaling cost first rather than assuming bigger accuracy
    is automatically the right choice."

    THIS IS THE KEY UPGRADE FROM YOUR ACTUAL ANSWER:
    You correctly used CADENCE to justify latency. A 5+ YOE answer ALSO
    questions the accuracy number itself (overfitting) and ties the
    decision to compute cost and scale — showing you don't take metrics
    at face value.
"""


# =================================================================================
# SECTION 4: HOW A 5+ YOE ANSWER DIFFERS FROM A 2-3 YOE ANSWER (meta-pattern)
# =================================================================================
"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Looking across all three of your answers, there's ONE repeating gap —
worth internalizing as a pattern, not just three separate fixes:

    YOUR PATTERN: Give a correct, single-track answer that defends a
    specific choice (one MCP server / avoid the super-agent / pick the
    accurate model).

    THE 5+ YOE PATTERN: Frame the answer as A TRADE-OFF WITH NAMED AXES,
    THEN give a conditional recommendation ("it depends on X, Y, Z —
    given the constraints as I understand them, I'd choose A").

    Interviewers at senior level are usually not testing "did you get
    the right answer" — the "right answer" is almost always "it depends."
    They're testing WHETHER YOU KNOW WHAT IT DEPENDS ON, and whether you
    can name the trade-off axes without being prompted. Every follow-up
    question you got ("what about the other model," "what if you didn't
    split," "is there a term for that") was the interviewer probing for
    the framework you hadn't stated yet — not testing whether your
    initial answer was wrong.

    ACTIONABLE HABIT FOR NEXT INTERVIEW:
    Whenever asked "why did you choose X over Y," structure your answer as:
    1. Name the 2-4 axes the decision depends on (explicitly, out loud).
    2. State where THIS use case lands on each axis.
    3. Give the conditional recommendation.
    4. Name the concept/term if one exists (attention dilution, context
       pollution, blast radius, overfitting, etc.) — interviewers reward
       precise vocabulary because it signals you've read/discussed this
       beyond just building it once.
"""


# =================================================================================
# SECTION 5: BONUS/RELATED QUESTIONS FOUND FOR FUTURE INTERVIEWS
# =================================================================================
"""
Sourced from current (2026) interview-prep material on MCP architecture,
multi-agent design, and ML model-selection trade-offs. Use these to
pre-load answers using the SAME framework as Section 4.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MCP ARCHITECTURE:

Q: "If you have 5 domains and split into 5 MCP servers, how does the
   agent know which server to call?"
A: "A gateway/orchestrator layer sits in front — it exposes ONE logical
   endpoint to the agent, handles auth and routing, and forwards the
   call to the correct domain server. The agent doesn't need to know
   there are 5 servers behind it."

Q: "How do you handle a workflow that spans multiple domains, like
   'onboard a new employee' touching HR, IT, and ERP?"
A: "The orchestrator/lead agent calls each domain server in sequence or
   parallel depending on dependencies, and stitches results together.
   This is the same context-centric decomposition principle — each
   domain server keeps its own focused context; the orchestrator holds
   only the compact summary needed to sequence the workflow."

Q: "What's the security risk of one giant MCP server vs many?"
A: "Blast radius. One compromised credential or vulnerable tool in a
   monolithic server exposes every domain behind it. Domain-scoped
   servers let you apply different access policies — e.g., PII controls
   on HR, SOX-style controls on finance — and contain a breach to one
   domain."

Q: "At what tool count would you actually split a server?"
A: "There's no hard number, but tool-selection accuracy measurably
   degrades once an agent has to choose among 20-30+ tools spanning
   unrelated domains. That's the practical trigger, not a fixed rule."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MULTI-AGENT / ORCHESTRATION:

Q: "When would you NOT use a multi-agent architecture even though you
   could?"
A: "When the coordination overhead outweighs the benefit — multi-agent
   setups often use 3-10x more tokens than a single well-prompted agent
   for the same task. If there's no context-volume problem, no
   parallelizable work, and no conflicting specialization, I'd start
   single-agent and only split when a measured pain point appears."

Q: "What's the difference between splitting agents by 'problem type'
   vs by 'context'?"
A: "Problem-type splitting — planner, implementer, tester, reviewer for
   the SAME feature — creates a telephone game where each handoff loses
   fidelity, and teams have measured sub-agents spending more tokens
   coordinating than working. Context splitting — an agent keeps
   everything it needs for a self-contained unit of work, like HR vs
   ERP — is the pattern that actually reduces overhead."

Q: "What's a 'verification subagent' and when would you use one?"
A: "A dedicated agent whose only job is testing/validating another
   agent's output, with explicit, comprehensive success criteria. Useful
   when the main orchestrator model is less capable, when verification
   needs specialized tools, or when you want an explicit checkpoint.
   The key failure mode to guard against is the 'early victory problem'
   — a verifier that runs one test, sees it pass, and declares success
   without full coverage."

Q: "How do you decide the boundary between sub-agents?"
A: "By context isolation, not by role. Good boundaries: independent
   research paths, components with a clean interface contract, and
   blackbox verification. Bad boundaries: sequential phases of the same
   work, tightly coupled components, or anything requiring frequent
   shared-state sync."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MODEL SELECTION / ACCURACY-LATENCY TRADE-OFFS:

Q: "A 95% accurate model returns in 10ms; a 98% accurate model takes
   500ms. Which do you ship?"
A: "Depends on the use case's latency SLA and what the 3-point accuracy
   gap costs the business. For most consumer-facing, low-stakes
   predictions, the 10ms model wins — user experience often outweighs a
   marginal accuracy gain. For high-stakes, infrequent decisions (credit
   risk, medical), the accuracy gap may justify the latency."

Q: "What's the difference between batch and real-time inference, and
   how does that change your model choice?"
A: "Batch predictions process on a schedule (hours/days) and are
   cost-efficient at scale — fits a model with high latency but strong
   accuracy. Real-time predictions score events as they arrive
   (ms-seconds) and are needed when decisions are immediate, like fraud
   detection — that forces you toward a faster, possibly less accurate,
   model. My sales-forecast answer used this exact distinction: it's a
   batch, not real-time, prediction problem."

Q: "How would you validate that 99.99% accuracy number before trusting
   it?"
A: "Check it's measured on a true out-of-time holdout set, not just
   cross-validation on historical data that may share seasonal patterns
   with training data. With only 3 years of data, I'd also check for
   overfitting via learning curves and out-of-sample error, and
   consider whether the metric itself (e.g., MAPE vs R²) is appropriate
   for sales forecasting."

Q: "The customer wants both speed AND accuracy — what do you tell them?"
A: "I'd reframe it: you don't need both simultaneously if the prediction
   isn't needed continuously. Run the accurate model on the batch
   cadence the business actually needs, and if a faster, rough estimate
   is ever needed in between full runs, the 90% model can serve that as
   a stopgap. This gets both without forcing a single model to satisfy
   two different requirements at once."
"""


# =================================================================================
# SECTION 6: GOLDEN LESSONS
# =================================================================================
"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GOLDEN LESSON 1 — "IT DEPENDS" IS THE RIGHT ANSWER; NAME WHAT IT DEPENDS ON.
    Senior interviews rarely have a single correct choice. They test
    whether you can name the trade-off axes (ownership, security
    boundary, tool count, cadence, cost, overfitting risk) before giving
    a conditional recommendation.

GOLDEN LESSON 2 — LEARN THE VOCABULARY, NOT JUST THE INTUITION.
    You had the right intuition on all three questions. What was missing
    was precise terminology: attention dilution / context pollution,
    blast radius, context-centric vs problem-centric decomposition,
    out-of-time holdout validation. Correct instinct + correct term =
    "strong hire" signal. Correct instinct alone = "hire, needs polish."

GOLDEN LESSON 3 — MCP SERVER COUNT IS A BLAST-RADIUS/OWNERSHIP DECISION.
    Split by domain when ownership, security boundary, or tool count
    diverges enough to hurt tool-selection accuracy (~20-30+ tools).
    Otherwise keep it unified — don't split just because you can.

GOLDEN LESSON 4 — MULTI-AGENT IS NOT FREE; IT'S 3-10x MORE EXPENSIVE.
    Only split into sub-agents for context protection, real
    parallelization, or genuine specialization. Split by CONTEXT
    boundary, never by PROBLEM TYPE (planner/implementer/tester on the
    same feature is a classic anti-pattern — a "telephone game").

GOLDEN LESSON 5 — NEVER TRUST A METRIC WITHOUT ASKING HOW IT WAS VALIDATED.
    99.99% accuracy on 3 years of data is a red flag, not a win. Always
    ask "validated on what holdout?" before treating an accuracy number
    as ground truth.

GOLDEN LESSON 6 — TIE TECHNICAL TRADE-OFFS TO BUSINESS CADENCE.
    Your best moment in the interview was recognizing that a monthly
    forecast doesn't need a 30-minute SLA. Generalize this: always ask
    "how often does this actually need to run?" before optimizing for
    latency.

GOLDEN LESSON 7 — THE ANSWER STRUCTURE THAT SIGNALS SENIORITY:
    (1) Name the axes -> (2) place this use case on each axis ->
    (3) give a conditional recommendation -> (4) name the concept/term
    if one exists. Practice saying this structure out loud for every
    "why did you choose X" question, not just the three above.
"""
