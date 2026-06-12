"""
===================================================================================
VIDAXL CASE STUDY — PPT CONTENT (Round 2 Presentation)
===================================================================================

This file contains ALL content for your 12-slide presentation.
Use this as reference while building your PowerPoint/Google Slides.

For each slide:
    - TITLE: What goes on the slide title
    - CONTENT: Bullet points / visuals to put on the slide
    - TALKING POINTS: What you SAY while presenting (don't put all of this on slide)

REMEMBER: Slides should be VISUAL and MINIMAL text.
Put keywords on slides, explain details verbally.
===================================================================================
"""


# =================================================================================
# SLIDE 1: TITLE SLIDE
# =================================================================================
"""
TITLE: AI-Powered Research & Data Analysis Workflow for vidaXL

CONTENT ON SLIDE:
    - Your name: Tirupathi Naidu
    - Role: AI Prompter — Case Study Presentation
    - Date: May 2026
    - One-liner: "Transforming repetitive research into intelligent,
                  scalable AI workflows"

TALKING POINTS (what you say):
    "Thank you for the opportunity. I'm Tirupathi Naidu, and today I'll
    present my solution for the AI Prompter case study. I've chosen to
    solve Problem 3 — Research and Data Analysis — in depth, and I'll
    walk you through my prioritization reasoning, the AI workflow design,
    prompt strategy with cost calculations, and the team adoption plan."
"""


# =================================================================================
# SLIDE 2: PROBLEM PRIORITIZATION
# =================================================================================
"""
TITLE: Why I Chose Problem 3 — Research & Data Analysis

CONTENT ON SLIDE (table or comparison):

    FACTOR              PROBLEM 1        PROBLEM 2        PROBLEM 3
                        (Project Mgmt)   (Design)         (Research) ★
    ─────────────────────────────────────────────────────────────────
    Business Impact     Medium           Medium           HIGH
    Feasibility         Medium           Medium           HIGH
    Scalability         PMs only         Designers only   ALL TEAMS
    Speed to implement  Weeks            Weeks            Days
    AI fit (text-based) Low              Low              HIGH

    ★ SELECTED: Problem 3

TALKING POINTS:
    "I evaluated all three problems against five factors: business impact,
    feasibility with current AI tools, scalability across teams, speed of
    implementation, and how well AI fits the task type.

    Problem 3 wins on every factor because:

    1. BUSINESS IMPACT — Research drives business decisions. Better competitor
       analysis, better pricing insights, better trend identification directly
       impacts revenue. All teams benefit, not just one.

    2. FEASIBILITY — Research tasks are TEXT-HEAVY. LLMs excel at text
       processing, summarization, and analysis. No complex image generation
       or Jira integrations needed.

    3. SCALABILITY — Every team at vidaXL does research — product content,
       SEO, category management, marketing. One solution serves all teams.

    4. SPEED — We can implement prompt templates and workflows in days,
       not weeks. No new tool integrations required.

    5. AI FIT — Summarizing, analyzing, comparing, extracting insights —
       these are exactly what LLMs are built for.

    Problem 1 (Project Management) would be my second priority — it's
    valuable but requires Jira API integrations which take longer.
    Problem 2 (Design) is third because image AI tools are still maturing
    and require more specialized setup."
"""


# =================================================================================
# SLIDE 3: CURRENT PROCESS (BEFORE) — Pain Points
# =================================================================================
"""
TITLE: Current Research Process — Pain Points

CONTENT ON SLIDE (show the manual workflow):

    CURRENT MANUAL PROCESS:
    ┌──────────────────────────────────────────────────────┐
    │  1. Manager assigns research task                     │
    │  2. Researcher manually visits 10-20 competitor sites │
    │  3. Copy-pastes data into spreadsheets                │
    │  4. Manually compares features and pricing            │
    │  5. Writes summary document (2-4 hours)               │
    │  6. Shares via email/Slack                            │
    │  7. Repeat every week/month                           │
    └──────────────────────────────────────────────────────┘

    PAIN POINTS:
    ❌ Time-consuming (4-8 hours per research task)
    ❌ Inconsistent quality (depends on who does it)
    ❌ No standardized format
    ❌ Findings get lost in emails
    ❌ Repetitive (same competitors researched monthly)
    ❌ Hard to scale (more products = more research needed)

TALKING POINTS:
    "Currently, research at vidaXL is largely manual. A team member visits
    competitor websites, copies data into spreadsheets, manually compares
    features and pricing, and writes a summary. This takes 4-8 hours per
    task, the quality varies by person, there's no standard format, and
    findings often get buried in emails. With 600,000 products across
    multiple categories, this doesn't scale."
"""


# =================================================================================
# SLIDE 4: AI-ASSISTED WORKFLOW (AFTER) — Solution Overview
# =================================================================================
"""
TITLE: AI-Powered Research Workflow — The Solution

CONTENT ON SLIDE (show the improved workflow):

    AI-ASSISTED PROCESS:
    ┌──────────────────────────────────────────────────────┐
    │  1. Team member inputs research brief (2 min)         │
    │  2. AI generates research plan (30 sec)               │
    │  3. AI analyzes competitor data (2 min)               │
    │  4. AI compares features & pricing (1 min)            │
    │  5. AI generates structured report (1 min)            │
    │  6. Human reviews & validates (10-15 min)             │
    │  7. Auto-shared to team in standard format            │
    └──────────────────────────────────────────────────────┘

    BEFORE vs AFTER:
    ─────────────────────────────────────────────────
    Manual research:     4-8 hours
    AI-assisted:         20-30 minutes (including human review)
    Time saved:          85-90% reduction
    Quality:             Consistent, structured, standardized
    ─────────────────────────────────────────────────

TALKING POINTS:
    "With AI, the same research task takes 20-30 minutes instead of 4-8 hours.
    The team member provides a brief, AI generates a research plan, analyzes
    competitor data, compares features and pricing, and produces a structured
    report. The human reviews and validates — they're the quality gate, not
    the data collector. This is an 85-90% time reduction while improving
    consistency and quality."
"""


# =================================================================================
# SLIDE 5: WORKFLOW DIAGRAM — Visual Step-by-Step
# =================================================================================
"""
TITLE: AI Research Workflow — Detailed Steps

CONTENT ON SLIDE (visual diagram — use boxes and arrows):

    ┌─────────────┐     ┌──────────────┐     ┌──────────────┐
    │  STEP 1     │     │  STEP 2      │     │  STEP 3      │
    │  Research    │────▶│  AI Research │────▶│  AI Data     │
    │  Brief      │     │  Plan        │     │  Collection  │
    │  (Human)    │     │  (AI)        │     │  (AI)        │
    └─────────────┘     └──────────────┘     └──────────────┘
                                                     │
                                                     ▼
    ┌─────────────┐     ┌──────────────┐     ┌──────────────┐
    │  STEP 6     │     │  STEP 5      │     │  STEP 4      │
    │  Share &    │◀────│  Human       │◀────│  AI Report   │
    │  Archive    │     │  Review      │     │  Generation  │
    │  (Auto)     │     │  (Human)     │     │  (AI)        │
    └─────────────┘     └──────────────┘     └──────────────┘

    LEGEND:
    🤖 AI handles: Steps 2, 3, 4, 6 (plan, collect, analyze, share)
    👤 Human handles: Steps 1, 5 (brief input, quality review)

TALKING POINTS:
    "The workflow has 6 steps. The human does only 2 things: provide the
    research brief at the start, and review the output at the end. AI
    handles everything in between — generating the research plan, collecting
    and analyzing data, and producing the structured report. The final
    sharing and archiving is automated."
"""


# =================================================================================
# SLIDE 6: AI TOOLS SELECTION
# =================================================================================
"""
TITLE: AI Tools & Platforms

CONTENT ON SLIDE (table):

    TASK                    TOOL                    WHY
    ─────────────────────────────────────────────────────────────────
    Research planning       ChatGPT / Claude        Best for structured reasoning
    Competitor analysis     Perplexity AI           Real-time web search + summary
    Data extraction         ChatGPT + web browsing  Can read competitor pages
    Feature comparison      Claude                  Best for long-document analysis
    Report generation       ChatGPT / Claude        Structured output (JSON/Markdown)
    Automation/workflow     n8n / Make.com          No-code workflow automation
    Customer email analysis ChatGPT API             Batch processing via API
    Trend identification    Perplexity + ChatGPT    Web search + synthesis

    WHY THESE TOOLS:
    ✅ No coding required (business teams can use directly)
    ✅ Available today (no development needed)
    ✅ Cost-effective (ChatGPT Team: $25/user/month)
    ✅ Scalable (API access for batch processing)

TALKING POINTS:
    "I selected tools based on three criteria: no coding required so business
    teams can use them directly, available today with no development needed,
    and cost-effective at scale.

    For real-time competitor research, Perplexity AI is ideal because it
    searches the web and summarizes findings in one step. For analysis and
    report generation, ChatGPT or Claude with structured prompts. For
    automation — connecting steps together without coding — n8n or Make.com.

    The key principle: use the RIGHT tool for each step, not one tool for
    everything."
"""


# =================================================================================
# SLIDE 7: PROMPT STRATEGY — Real Examples
# =================================================================================
"""
TITLE: Prompt Strategy — Real Examples

CONTENT ON SLIDE (show 2-3 actual prompts):

    PROMPT 1 — RESEARCH PLAN GENERATION:
    ┌──────────────────────────────────────────────────────────────┐
    │  Role: You are a senior market research analyst for a home   │
    │  & garden eCommerce company with 600,000 products.           │
    │                                                              │
    │  Task: Create a research plan for analyzing competitors      │
    │  in the {category} category.                                 │
    │                                                              │
    │  Include:                                                    │
    │  - 5 key competitors to analyze                              │
    │  - Data points to collect (pricing, features, reviews)       │
    │  - Sources to check                                          │
    │  - Timeline estimate                                         │
    │                                                              │
    │  Format: Structured markdown with headers.                   │
    │  Category: {outdoor_furniture}                                │
    └──────────────────────────────────────────────────────────────┘

    PROMPT 2 — COMPETITOR FEATURE COMPARISON:
    ┌──────────────────────────────────────────────────────────────┐
    │  You are analyzing competitor products for vidaXL.            │
    │                                                              │
    │  Compare these products on: price, material, dimensions,     │
    │  customer rating, shipping options, and unique features.     │
    │                                                              │
    │  Products:                                                   │
    │  - vidaXL Garden Bench (teak, 150cm, €89)                    │
    │  - IKEA APPLARO (acacia, 128cm, €79)                         │
    │  - Wayfair Elsmere (eucalyptus, 157cm, €120)                 │
    │                                                              │
    │  Output: Comparison table in markdown format.                │
    │  Highlight where vidaXL has advantages and disadvantages.    │
    │  Max 300 words analysis after the table.                     │
    └──────────────────────────────────────────────────────────────┘

    PROMPT 3 — CUSTOMER EMAIL ANALYSIS:
    ┌──────────────────────────────────────────────────────────────┐
    │  Analyze these customer emails and identify:                  │
    │  1. Top 5 recurring complaints (with frequency)              │
    │  2. Top 5 positive feedback themes                           │
    │  3. Product improvement suggestions                          │
    │  4. Delivery/logistics issues                                │
    │                                                              │
    │  Format: JSON with categories and counts.                    │
    │  Emails: {batch_of_50_emails}                                │
    │                                                              │
    │  Rules:                                                      │
    │  - Do NOT include any customer PII in your output            │
    │  - Categorize by theme, not by individual email              │
    │  - Include direct quotes (anonymized) as evidence            │
    └──────────────────────────────────────────────────────────────┘

TALKING POINTS:
    "Here are three real prompt examples for different research tasks.
    Notice the pattern in each: Role (who the AI is), Task (what to do),
    Context (specific data), Format (how to structure output), and
    Constraints (rules and limits). This structure ensures consistent,
    high-quality outputs every time.

    The prompts use variables like {category} and {batch_of_50_emails}
    so they're reusable templates — the team fills in the specific data
    each time without rewriting the prompt."
"""


# =================================================================================
# SLIDE 8: QUALITY CONTROL — Hallucination Prevention
# =================================================================================
"""
TITLE: Quality Control & Hallucination Prevention

CONTENT ON SLIDE:

    HOW WE ENSURE QUALITY:

    1. STRUCTURED PROMPTS
       - Explicit format requirements (JSON, markdown tables)
       - Constraints: "Use ONLY the provided data"
       - Max word limits to prevent verbose, unfocused output

    2. HALLUCINATION PREVENTION
       - "Do NOT invent data. If information is unavailable, say 'Not found'"
       - Cross-reference: AI output checked against source data
       - Human review as final quality gate

    3. CONSISTENCY
       - Reusable prompt TEMPLATES (same structure every time)
       - Standard output format across all research reports
       - Version-controlled prompts (track changes over time)

    4. INPUT QUALITY
       - Clear research briefs with specific questions
       - Provide context (category, competitors, time period)
       - The better the input, the better the output

    QUALITY CHECKLIST (for human reviewer):
       ✅ Are all data points sourced? (no invented numbers)
       ✅ Is the format consistent with template?
       ✅ Are competitor names and prices accurate?
       ✅ Is the analysis balanced (pros AND cons)?
       ✅ Is it actionable (clear recommendations)?

TALKING POINTS:
    "Quality control happens at three levels. First, the prompt itself
    prevents hallucination with explicit constraints like 'use ONLY provided
    data' and 'if unavailable, say Not Found.' Second, we use structured
    output formats so the AI can't go off-track. Third, every output goes
    through human review before being shared — the human is the final
    quality gate. We also maintain a quality checklist so reviewers know
    exactly what to verify."
"""


# =================================================================================
# SLIDE 9: COST OPTIMIZATION — Token Calculations at Scale
# =================================================================================
"""
TITLE: Cost Optimization & Token Calculations

CONTENT ON SLIDE (show real numbers):

    MODEL SELECTION STRATEGY:
    ┌────────────────────────────────────────────────────────────────┐
    │  TASK                  MODEL              COST/1M tokens       │
    │  ──────────────────────────────────────────────────────────── │
    │  Simple summaries      GPT-4o-mini        $0.15 input/$0.60 out│
    │  Complex analysis      GPT-4o             $2.50 input/$10 out  │
    │  Real-time research    Perplexity Pro     $20/month flat       │
    │  Batch email analysis  Claude Haiku       $0.25 input/$1.25 out│
    └────────────────────────────────────────────────────────────────┘

    COST CALCULATION — MONTHLY RESEARCH (example):

    Scenario: 50 research reports/month, each ~2000 tokens input + 1500 output

    WITH GPT-4o (expensive):
        Input:  50 × 2000 = 100,000 tokens × $2.50/1M = $0.25
        Output: 50 × 1500 = 75,000 tokens × $10/1M = $0.75
        Total: $1.00/month ← Very affordable!

    WITH GPT-4o-mini (for simpler tasks):
        Input:  50 × 2000 = 100,000 tokens × $0.15/1M = $0.015
        Output: 50 × 1500 = 75,000 tokens × $0.60/1M = $0.045
        Total: $0.06/month ← Almost free!

    BATCH EMAIL ANALYSIS (500 emails/month):
        Input:  500 × 500 = 250,000 tokens × $0.25/1M = $0.06
        Output: 500 × 200 = 100,000 tokens × $1.25/1M = $0.13
        Total: $0.19/month

    TOTAL MONTHLY AI COST: ~$5-20/month (depending on volume)
    vs. MANUAL COST: 1 researcher × 40 hours × €30/hr = €1,200/month

    ROI: 98% cost reduction + 85% time savings

    TOKEN OPTIMIZATION TECHNIQUES:
    • Use GPT-4o-mini for simple tasks (10x cheaper than GPT-4o)
    • Batch multiple items per prompt (reduce system prompt repetition)
    • Constrain output length ("max 300 words")
    • Use structured output (JSON) — more compact than prose
    • Cache repeated research (same competitors monthly)

TALKING POINTS:
    "Let me show you the actual cost calculations. For 50 research reports
    per month using GPT-4o, the total API cost is about $1. Even with batch
    email analysis of 500 emails, we're looking at under $1 per month.
    Compare that to a researcher spending 40 hours at €30/hour — that's
    €1,200 per month for the same work.

    The ROI is clear: 98% cost reduction with 85% time savings. And we
    optimize further by using cheaper models for simple tasks, batching
    multiple items per prompt, and constraining output length.

    The key insight: AI research costs are negligible compared to human
    labor costs. The real value is in time saved and consistency gained."
"""


# =================================================================================
# SLIDE 10: IMPLEMENTATION & TEAM ADOPTION
# =================================================================================
"""
TITLE: Implementation & Team Adoption Plan

CONTENT ON SLIDE (phased rollout):

    PHASE 1 — PILOT (Week 1-2):
    ┌──────────────────────────────────────────────────────┐
    │  • Select 1 team (e.g., Category Management)          │
    │  • Train 3-5 people on prompt templates               │
    │  • Run 10 research tasks with AI assistance           │
    │  • Collect feedback on quality and time saved          │
    └──────────────────────────────────────────────────────┘

    PHASE 2 — REFINE (Week 3-4):
    ┌──────────────────────────────────────────────────────┐
    │  • Improve prompts based on pilot feedback            │
    │  • Create prompt library (10-15 templates)            │
    │  • Document best practices and common mistakes        │
    │  • Measure: time saved, quality score, adoption rate  │
    └──────────────────────────────────────────────────────┘

    PHASE 3 — SCALE (Week 5-8):
    ┌──────────────────────────────────────────────────────┐
    │  • Roll out to all teams (SEO, Marketing, Product)    │
    │  • Team-specific prompt templates per use case        │
    │  • Monthly prompt optimization reviews                │
    │  • Set up automation (n8n) for recurring research     │
    └──────────────────────────────────────────────────────┘

    HOW TO ENSURE ADOPTION:
    ✅ Make it EASIER than the old way (not harder)
    ✅ Show time saved with real numbers from pilot
    ✅ Provide ready-to-use templates (copy-paste)
    ✅ Weekly office hours for questions
    ✅ Celebrate wins (share success stories)
    ✅ Feedback loop: monthly survey → improve prompts

TALKING POINTS:
    "Adoption is the hardest part — great tools fail if teams don't use them.
    My approach is phased: start small with one team, prove the value with
    real numbers, then scale.

    In the pilot, I'd work directly with 3-5 people, train them on the
    prompt templates, and measure time saved. Once we have proof — say,
    '4 hours reduced to 30 minutes' — that story sells itself to other teams.

    The key to adoption: make it EASIER than the old way. If the AI workflow
    is more complex than manual research, nobody will use it. That's why I
    provide copy-paste templates and keep the human's role simple — just
    input the brief and review the output."
"""


# =================================================================================
# SLIDE 11: BEFORE/AFTER COMPARISON
# =================================================================================
"""
TITLE: Impact Summary — Before vs After

CONTENT ON SLIDE (visual comparison):

    METRIC              BEFORE (Manual)         AFTER (AI-Assisted)
    ─────────────────────────────────────────────────────────────────
    Time per task       4-8 hours               20-30 minutes
    Quality             Inconsistent            Standardized, structured
    Format              Varies by person        Consistent templates
    Scalability         Limited by headcount    Unlimited (AI scales)
    Cost per report     €120-240 (labor)        €0.02-0.05 (API)
    Knowledge sharing   Lost in emails          Archived, searchable
    Frequency           Monthly (too slow)      Weekly or on-demand
    Coverage            3-5 competitors         10-20 competitors

    KEY WINS:
    🎯 85-90% time reduction
    🎯 98% cost reduction per report
    🎯 Consistent quality across all teams
    🎯 Research available on-demand (not just monthly)
    🎯 All findings archived and searchable

TALKING POINTS:
    "The impact is dramatic across every metric. Time drops from hours to
    minutes. Cost drops from hundreds of euros to cents. Quality becomes
    consistent because we use standardized templates. And research becomes
    available on-demand instead of waiting for the monthly cycle.

    But the biggest win is scalability — with manual research, you're limited
    by headcount. With AI, you can analyze 20 competitors instead of 5,
    cover more categories, and do it weekly instead of monthly. The team
    focuses on ACTING on insights instead of COLLECTING data."
"""


# =================================================================================
# SLIDE 12: SUMMARY & NEXT STEPS
# =================================================================================
"""
TITLE: Summary & Next Steps

CONTENT ON SLIDE:

    WHAT I PROPOSED:
    ✅ Prioritized Problem 3 (Research) — highest impact, all teams benefit
    ✅ Designed 6-step AI workflow — human inputs brief, AI does the work
    ✅ Selected practical tools — ChatGPT, Perplexity, Claude, n8n
    ✅ Provided real prompts — reusable templates with variables
    ✅ Calculated costs — €5-20/month vs €1,200/month manual
    ✅ Planned phased adoption — pilot → refine → scale in 8 weeks

    NEXT STEPS (if approved):
    1. Week 1: Select pilot team + create first 5 prompt templates
    2. Week 2: Run pilot with 10 research tasks, measure results
    3. Week 3-4: Refine based on feedback, build prompt library
    4. Week 5+: Scale to all teams with training sessions

    MY APPROACH AS AI PROMPTER:
    "I don't just write prompts — I design workflows, measure results,
    optimize costs, and ensure teams actually adopt the solutions."

TALKING POINTS:
    "To summarize: I chose Problem 3 because it has the highest business
    impact and serves all teams. The AI workflow reduces research time by
    85-90% at negligible cost. The implementation is phased — we prove
    value with a pilot before scaling.

    As an AI Prompter, my role goes beyond writing prompts. I analyze
    problems, design end-to-end workflows, select the right tools, optimize
    for cost and quality, and drive team adoption. I'm excited about the
    opportunity to bring this approach to vidaXL's 600,000-product catalog
    and help every team work smarter with AI.

    Thank you. I'm happy to answer any questions."
"""


# =================================================================================
# BONUS: POTENTIAL QUESTIONS THEY'LL ASK AFTER YOUR PRESENTATION
# =================================================================================
"""
Q1: "Why didn't you choose Problem 1 or 2?"
A: "Problem 1 (Project Management) requires Jira API integrations which take
   longer to implement. Problem 2 (Design) relies on image AI tools which are
   still maturing. Problem 3 is text-based, serves ALL teams, and can be
   implemented in days with existing tools. I'd tackle Problems 1 and 2 next."

Q2: "How do you handle sensitive competitor data?"
A: "We use ChatGPT Team or Enterprise plans where data isn't used for training.
   For highly sensitive analysis, we can use Azure OpenAI which keeps data
   within our own cloud environment. Prompts explicitly say 'do not store
   or share this data.'"

Q3: "What if the AI gives wrong competitor prices?"
A: "That's why human review is mandatory. The AI collects and structures data,
   but a human verifies key numbers before sharing. We also add the constraint
   'if price is unavailable, say Not Found' to prevent hallucination."

Q4: "How do you measure success?"
A: "Four metrics: (1) Time saved per task (target: 80%+ reduction).
   (2) Quality score from human reviewers (target: 4/5 average).
   (3) Adoption rate (target: 80% of team using it weekly).
   (4) Cost per report (target: under €1)."

Q5: "Can this work for 600,000 products?"
A: "Yes. The prompt templates use variables — {category}, {competitor},
   {product_type}. The same template works for any product category.
   For batch analysis (like customer emails), we use the API to process
   hundreds of items automatically. The workflow scales linearly."

Q6: "What about teams that resist using AI?"
A: "I focus on making it EASIER than the old way. If someone can get a
   research report in 20 minutes instead of 4 hours, they'll adopt it.
   I also start with early adopters (enthusiastic team members), let them
   show results, and peer influence drives the rest."

Q7: "How do you keep prompts up to date?"
A: "Monthly prompt review sessions. I track which prompts produce the best
   results, which ones need refinement, and update the template library.
   Version control so we can always roll back if a change makes things worse."

Q8: "What's your experience with n8n or Make.com?"
A: "I'm familiar with workflow automation tools. They connect triggers
   (like a new Jira ticket or a scheduled time) to actions (like running
   a prompt and sending the result via email). For recurring research tasks,
   I'd set up automated workflows that run weekly without human intervention."
"""

print("=" * 60)
print("VidaXL Case Study — PPT Content Guide")
print("=" * 60)
print()
print("12 Slides:")
print("  1.  Title Slide")
print("  2.  Problem Prioritization (Why Problem 3)")
print("  3.  Current Process — Pain Points")
print("  4.  AI-Assisted Workflow — Solution Overview")
print("  5.  Workflow Diagram — Visual Steps")
print("  6.  AI Tools Selection")
print("  7.  Prompt Strategy — 3 Real Examples")
print("  8.  Quality Control & Hallucination Prevention")
print("  9.  Cost Optimization — Token Calculations")
print("  10. Implementation & Team Adoption (3 phases)")
print("  11. Before/After Impact Comparison")
print("  12. Summary & Next Steps")
print()
print("+ 8 Potential Follow-up Questions with Answers")
print("=" * 60)
