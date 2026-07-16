"""
===================================================================================
PROMPT MANAGEMENT & VERSIONING — Production Prompt Engineering at Scale
===================================================================================

THE INTERVIEW QUESTION YOU FACED:
    "How do you manage prompts? How do you version them? Prompts change over
    time — how do you update, track, test, and roll back prompt changes in
    production? How do you handle prompt libraries?"

WHY THIS MATTERS:
    Prompts are the most-changed artifact in GenAI applications. They change
    more often than code. Without versioning, you lose track of what was live
    when, can't reproduce a bug, can't roll back a bad change, and can't
    A/B test improvements. Senior engineers treat prompts like production code.

SECTIONS:
    1.  The Problem — Why Prompt Management Exists
    2.  Prompt Storage — Where Prompts Live (5 approaches)
    3.  Prompt Versioning — How to Track Changes Over Time
    4.  Prompt Testing — How to Validate Before Going Live
    5.  Prompt Deployment — How to Roll Out and Roll Back
    6.  Prompt Libraries & Organization — Structure at Scale
    7.  Tools & Platforms for Prompt Management
    8.  Common Interview Q&A (with crisp answers)
    9.  Code Examples (prompt registry pattern)
    10. GOLDEN LESSONS
===================================================================================
"""


# =================================================================================
# SECTION 1: THE PROBLEM — Why Prompt Management Exists
# =================================================================================
"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WITHOUT PROMPT MANAGEMENT:

    - Prompts are hardcoded strings scattered across 20 files.
    - Someone changes a prompt on Friday -> quality drops on Monday.
    - Nobody knows what the prompt was last week (no history).
    - Can't roll back to the version that worked.
    - Can't A/B test: "is the new prompt actually better?"
    - Can't audit: "what prompt generated this wrong answer on March 5?"
    - Different environments (dev/staging/prod) run different prompts
      with no clear tracking.

WITH PROMPT MANAGEMENT:

    - Prompts live in ONE place (library/registry), versioned like code.
    - Every change is tracked: who changed what, when, why.
    - New versions pass a regression test before going live.
    - Roll back to previous version in seconds.
    - A/B test prompts on a traffic subset.
    - Audit trail: any past answer can be traced to the exact prompt version.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE ANALOGY THAT WINS IN INTERVIEWS:

    "We treat prompts like database migrations or feature flags —
    versioned, tested, deployed through environments, and rollbackable.
    Because a bad prompt change can be as impactful as a bad code change."

INTERVIEW ANSWER (the opener):
    "Prompt management is critical because prompts change more often than
    code. Without versioning, you can't reproduce bugs, roll back a bad
    change, or A/B test improvements. I treat prompts as a first-class
    artifact with its own version history, testing pipeline, and
    deployment process — similar to how you'd manage database migrations."
"""


# =================================================================================
# SECTION 2: PROMPT STORAGE — Where Prompts Live (5 approaches)
# =================================================================================
"""
From simplest to most sophisticated. Know all 5; recommend based on team size.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

APPROACH 1 — CONSTANTS IN CODE (simplest, small team)

    # prompts/agent_prompts.py
    AGENT_SYSTEM_PROMPT = (
        "You are DocSage, an intelligent document assistant. "
        "Use ONLY the retrieved context to answer..."
    )

    Pros: simple, versioned by Git, code-reviewed in PRs.
    Cons: changing a prompt requires a code deploy.
    Use when: small team, few prompts, deploy is fast.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

APPROACH 2 — YAML/JSON FILES IN CODE (structured, still Git-versioned)

    # prompts/v2/agent.yaml
    name: agent_system_prompt
    version: "2.1"
    model: llama-3.1-8b-instant
    temperature: 0
    content: |
      You are DocSage, an intelligent document assistant.
      Use ONLY the retrieved context to answer...

    Pros: structured metadata (version, model, temp), still Git-versioned,
          easy to load at runtime, non-devs can edit YAML.
    Cons: still requires deploy to change.
    Use when: growing team, want structured prompt metadata.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

APPROACH 3 — DATABASE REGISTRY (runtime changes, no deploy)

    Table: prompt_registry
    | id | name                 | version | content     | is_active | created_by | created_at |
    |----|----------------------|---------|-------------|-----------|------------|------------|
    | 1  | agent_system_prompt  | 1.0     | "You are..."| false     | tiru       | 2026-01-01 |
    | 2  | agent_system_prompt  | 2.0     | "You are..."| true      | tiru       | 2026-03-15 |

    Code reads: SELECT content FROM prompt_registry WHERE name=? AND is_active=true

    Pros: change prompts WITHOUT redeploying; instant rollback (flip is_active).
    Cons: need a DB, need access control, need caching, harder to code-review.
    Use when: prompts change frequently, can't redeploy every time.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

APPROACH 4 — FEATURE FLAG / CONFIG SERVICE (A/B testing, gradual rollout)

    Use a feature-flag system (LaunchDarkly, Unleash, or custom):
    - Flag: "agent_prompt_version" -> values: "v1" | "v2" | "v3"
    - Route 90% traffic to v2, 10% to v3 (A/B test).
    - If v3 wins on metrics, roll to 100%.

    Pros: A/B testing, gradual rollout, instant kill-switch.
    Cons: more infrastructure, complex.
    Use when: at scale, optimizing prompts continuously.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

APPROACH 5 — DEDICATED PROMPT MANAGEMENT PLATFORM (full lifecycle)

    Tools: LangSmith Hub, PromptLayer, Humanloop, Weights & Biases Prompts.
    Features: version history, diff view, A/B test, metrics tracking,
              team collaboration, approval workflows, audit trail.

    Pros: all-in-one; designed for prompt lifecycle.
    Cons: cost, vendor dependency.
    Use when: enterprise team, many prompts, need governance.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE PROGRESSION (what to say in an interview):

    "For a small project I start with prompts as constants in a dedicated
    module, versioned through Git. As the team scales, I move to a database
    registry for runtime changes without redeployment. At scale I add
    feature flags for A/B testing and gradual rollout. The principle is the
    same as config management — prompts are config that needs versioning,
    review, and rollback."

INTERVIEW ANSWER:
    "I store prompts in a dedicated module — not scattered across code.
    For my current project, they're Python constants versioned via Git
    with PR review. For production at scale, I'd use a database prompt
    registry where each prompt has a name, version, content, and an
    is_active flag — so I can update without redeploying and roll back
    instantly. At enterprise scale, tools like LangSmith Hub or
    PromptLayer add A/B testing, metrics, and approval workflows."
"""


# =================================================================================
# SECTION 3: PROMPT VERSIONING — How to Track Changes Over Time
# =================================================================================
"""
The interviewer's real question: "prompts aren't fixed forever — how do you
manage the lifecycle?"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE VERSION LIFECYCLE:

    Draft -> Review -> Test -> Active -> Deprecated -> Archived

    1. DRAFT: someone writes/edits a new prompt version.
    2. REVIEW: peer reviews the change (PR for Git, approval for DB).
    3. TEST: run against a golden test set — does quality hold?
    4. ACTIVE: promoted to production (is_active = true, or deployed).
    5. DEPRECATED: a newer version is now active; this one still exists for rollback.
    6. ARCHIVED: old version, kept for audit trail, never used.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

VERSIONING STRATEGIES:

    STRATEGY A — SEMANTIC VERSIONING (major.minor):
        v1.0 -> v1.1 (minor tweak, same intent)
        v1.0 -> v2.0 (major rewrite, different structure/behavior)

    STRATEGY B — SEQUENTIAL (v1, v2, v3...):
        Simple counter. Less info but simple to manage.

    STRATEGY C — GIT COMMIT HASH:
        Each commit that touches the prompt file IS the version.
        Use git blame / git log to trace history.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHAT TO STORE PER VERSION:

    - prompt_name (e.g., "agent_system_prompt")
    - version (e.g., "2.1")
    - content (the actual prompt text)
    - model (which LLM it's designed for)
    - temperature / max_tokens (paired config)
    - created_by (who made the change)
    - created_at (when)
    - change_reason (WHY — "improved grounding instruction per eval results")
    - is_active (currently in production?)
    - test_results_link (pointer to eval scores)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ROLLBACK:

    Git-based: `git revert <commit>` or restore the file from a previous commit.
    DB-based: UPDATE prompt_registry SET is_active=false WHERE version='2.1';
              UPDATE prompt_registry SET is_active=true WHERE version='2.0';
    Platform-based: one-click in the UI.

    GOLDEN RULE: Always keep the previous version intact. Never DELETE a version;
    mark it deprecated. Deletion destroys your audit trail and rollback path.

INTERVIEW ANSWER:
    "I version prompts with a name + version number, and store metadata:
    who changed it, when, why, which model it targets, and the eval score
    it achieved. I never delete old versions — they're marked deprecated,
    so I can roll back in seconds. For Git-based: the commit history IS the
    version trail. For DB-based: I flip the is_active flag. The key is that
    every prompt change is traceable, reviewable, and reversible."
"""


# =================================================================================
# SECTION 4: PROMPT TESTING — How to Validate Before Going Live
# =================================================================================
"""
This is what the interviewer REALLY wanted to hear — you don't just update
a prompt and push to prod. You TEST it first, like code CI.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE PROMPT REGRESSION TEST PIPELINE:

    1. GOLDEN TEST SET:
       A set of 50-200 (input, expected_output) pairs that represent
       critical behaviors. Examples:
       - Input: "What's the refund policy?" -> Expected: mentions 14 days.
       - Input: "Hi" -> Expected: no tool call, polite greeting.
       - Input: "What is X?" (not in docs) -> Expected: "I don't have info."

    2. RUN THE NEW PROMPT against the golden set.
       For each test case, run the chain with the new prompt and collect outputs.

    3. EVALUATE with metrics:
       - Faithfulness (RAGAS) — did hallucination increase?
       - Answer relevance — are answers still on-topic?
       - Format compliance — does the output still match expected structure?
       - Regression count — how many previously-passing cases now fail?

    4. COMPARE old vs new:
       If new prompt DROPS quality on any critical metric -> REJECT.
       If new prompt IMPROVES or maintains -> APPROVE for production.

    5. HUMAN SPOT-CHECK:
       For high-stakes prompts, a human reviews a sample of outputs
       from the new version before promotion.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

AUTOMATION:

    # In CI/CD pipeline (GitHub Actions, etc.):
    def test_prompt(new_prompt, golden_set):
        results = []
        for input_text, expected in golden_set:
            output = run_chain(prompt=new_prompt, input=input_text)
            score = evaluate(output, expected)  # faithfulness, relevance
            results.append(score)
        avg_score = sum(results) / len(results)
        if avg_score < THRESHOLD:
            raise PromptRegressionError(f"Score {avg_score} < {THRESHOLD}")
        return "PASS"

    This runs automatically on every PR that touches a prompt file.

INTERVIEW ANSWER:
    "Before any prompt goes live, I run it against a golden test set —
    real inputs with expected behaviors. I measure faithfulness, relevance,
    and format compliance, then compare against the current production
    prompt. If the new version drops quality on any critical metric, it's
    rejected. This is essentially CI/CD for prompts — automated regression
    testing on every prompt change, same principle as unit tests for code."
"""


# =================================================================================
# SECTION 5: PROMPT DEPLOYMENT — How to Roll Out and Roll Back
# =================================================================================
"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DEPLOYMENT PATTERNS:

    PATTERN A — CODE DEPLOY (Git-based, simple):
        PR merges -> CI runs tests -> deploy -> new prompt is live.
        Rollback: revert commit, redeploy.
        Downside: requires full deploy cycle for a prompt tweak.

    PATTERN B — RUNTIME SWAP (DB-based, fast):
        Update the active version in the registry -> app reads new version
        on next request (with caching + TTL refresh).
        Rollback: flip the active flag back. Instant, no deploy.

    PATTERN C — GRADUAL ROLLOUT (feature flags):
        Route 5% traffic to new prompt, monitor metrics.
        If good -> 25% -> 50% -> 100%.
        If bad -> kill to 0% instantly.
        This is canary deployment for prompts.

    PATTERN D — BLUE-GREEN (full swap with instant fallback):
        "Blue" = current prompt set. "Green" = new prompt set.
        Switch all traffic from blue to green.
        If metrics drop -> switch back to blue. Zero downtime.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ENVIRONMENT PROMOTION:

    dev -> staging -> prod

    - Dev: experimental prompts, fast iteration, no review needed.
    - Staging: runs against full golden set, team review.
    - Prod: only promoted from staging after tests pass + approval.

    Same prompt, different environment -> different version allowed.
    "In dev we're testing v3.0; prod still runs v2.1 until v3.0 passes."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CACHING CONSIDERATION:

    If prompts live in a DB, don't hit the DB on every request.
    Cache the active prompt in memory with a TTL (e.g., 5 minutes).
    On update: either invalidate the cache or wait for TTL expiry.
    Trade-off: shorter TTL = faster rollout; longer TTL = fewer DB reads.

INTERVIEW ANSWER:
    "For deployment, I promote prompts through environments — dev, staging,
    prod. In staging, the golden test set validates quality. For production,
    I use either a code deploy (simple) or a runtime DB swap (fast, no
    redeploy). For high-risk changes I do a gradual rollout — route 5%
    traffic to the new prompt, monitor metrics, then scale up. Rollback
    is instant: revert the commit or flip the active flag. I cache the
    active prompt in memory with a short TTL so DB reads don't become
    a bottleneck."
"""


# =================================================================================
# SECTION 6: PROMPT LIBRARIES & ORGANIZATION — Structure at Scale
# =================================================================================
"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FOLDER STRUCTURE (Git-based):

    prompts/
        agent/
            system_prompt.py          # or .yaml / .jinja2
            tool_selection_prompt.py
        grading/
            relevance_grader.py
            hallucination_grader.py
        generation/
            rag_answer.py
            summary.py
        rewriting/
            query_rewrite.py
        shared/
            persona.py                # reusable persona blocks
            format_instructions.py    # reusable format specs

    Each file contains ONE named prompt with metadata (version, model, description).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PROMPT TEMPLATE COMPOSITION (DRY — Don't Repeat Yourself):

    Instead of copying the same persona/format instructions into every prompt,
    COMPOSE them:

    PERSONA = "You are DocSage, an intelligent document assistant."
    FORMAT = "Keep the answer concise (3-6 sentences). Use bullet points."

    AGENT_PROMPT = f"{PERSONA}\n\n{AGENT_RULES}"
    GENERATE_PROMPT = f"{PERSONA}\n\n{FORMAT}\n\n{RAG_INSTRUCTIONS}"

    This way, changing the persona updates ALL prompts at once.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

NAMING CONVENTIONS:

    {domain}_{task}_{version}
    Examples:
    - agent_system_v2.1
    - grading_relevance_v1.0
    - generation_rag_answer_v3.0

    Clear naming makes it obvious what each prompt does and which version is current.

INTERVIEW ANSWER:
    "I organize prompts in a dedicated module by domain — agent, grading,
    generation, rewriting — with shared building blocks like persona and
    format instructions composed in. Each prompt is named with domain,
    task, and version. Shared blocks mean changing the persona updates
    all prompts that use it. This gives a single source of truth and
    prevents the 'prompts scattered across 20 files' problem."
"""


# =================================================================================
# SECTION 7: TOOLS & PLATFORMS FOR PROMPT MANAGEMENT
# =================================================================================
"""
Name-drop these when asked "what tools do you use/know?" Picks are 2026-current.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    TOOL               TYPE           KEY FEATURES
    LangSmith Hub      Platform       Version, share, pull prompts; integrated with LangChain
    PromptLayer        Platform       Version, log, A/B test, analytics, REST API
    Humanloop          Platform       Versioning, eval, deployment, team collab
    Weights & Biases   ML Platform    Prompt tracking alongside experiment metrics
    Portkey            Gateway        Prompt management + LLM gateway + fallbacks
    Custom DB table    DIY            Full control; prompt_registry pattern (see code below)
    Git + YAML         DIY            Simple, free, works with any CI/CD

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHEN ASKED "WHICH DO YOU USE?":

    For my projects: "Prompts as Python constants in a dedicated module,
    versioned via Git with PR review. For a larger team I'd add a
    LangSmith Hub or a custom DB registry for runtime changes."

    Never say "I just hardcode strings inline" — that's the anti-pattern.
"""


# =================================================================================
# SECTION 8: COMMON INTERVIEW Q&A (with crisp answers)
# =================================================================================
"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q: "How do you manage prompts in your project?"
A: "Dedicated module, one file per prompt, versioned via Git with PR review.
    Each prompt has a name, version, target model, and description. I never
    scatter prompts inline across business logic."

Q: "How do you version prompts?"
A: "Same as code — Git commit history is the version trail. For runtime
    flexibility, a DB registry with name + version + is_active flag. I
    never delete old versions — deprecated, not deleted."

Q: "How do you update a prompt in production?"
A: "Write new version -> run against golden test set -> if quality holds or
    improves, promote to active. For Git-based: merge PR + deploy. For
    DB-based: flip is_active flag — no deploy needed, instant rollback."

Q: "How do you test prompt changes?"
A: "Automated regression testing. I have a golden set of 50-200 test cases.
    Every prompt change is evaluated on this set — faithfulness, relevance,
    format compliance. If metrics drop, the change is rejected. Same
    principle as code CI."

Q: "How do you roll back a bad prompt?"
A: "Git: revert the commit. DB: flip is_active to the previous version.
    Feature flag: set percentage to 0. Always instant, always safe."

Q: "What if the prompt works on GPT-4 but breaks on a different model?"
A: "Prompts are model-sensitive. I version per model, or at minimum re-run
    the golden test set when switching models. The registry can have a
    'model' column to pair prompts with their target model."

Q: "How do you handle multiple prompts across an agentic system?"
A: "Prompt library organized by domain (agent, grading, generation, rewriting)
    with shared composable blocks (persona, format instructions). Each node
    in my LangGraph reads its prompt from the library, not inline."

Q: "How do you A/B test prompts?"
A: "Feature flags route a percentage of traffic to the new version. Compare
    metrics (quality, latency, user satisfaction) against baseline. Promote
    only if it wins. LangSmith or a custom split handles the routing."

Q: "How do you handle secrets / dynamic values in prompts?"
A: "Prompts are TEMPLATES with placeholders ({context}, {question}, {user_name}).
    The template is versioned; dynamic values are injected at runtime.
    Secrets never go into prompts — they're env vars / vault."

Q: "What's the difference between a prompt template and a prompt version?"
A: "A template has static structure + placeholders. A version is a specific
    iteration of that template — same placeholders but different instructions,
    tone, or format. Version 2.0 might have stricter grounding rules than 1.0."

Q: "How do you prevent prompt injection when prompts are configurable?"
A: "Separate system/user messages strictly. Prompts in the registry are
    system-level (trusted); user input is always the user message (untrusted).
    Never interpolate user input into system prompts."

Q: "At your current company, how do you manage prompts?"
A: "We have a dedicated prompts folder. Prompts are structured as XML-format
    agent instructions — some for data retrieval, some for actions. They're
    versioned through Git, reviewed in PRs, and deployed with the service.
    When a prompt needs a change, it goes through the same PR + review
    process as code. For fast iteration during development, we test locally
    against sample inputs before merging."
"""


# =================================================================================
# SECTION 9: CODE EXAMPLES (prompt registry pattern)
# =================================================================================
"""
A minimal, runnable prompt registry to understand the pattern.
"""

from datetime import datetime


class PromptRegistry:
    """
    Simple in-memory prompt registry (production would use a DB).
    Demonstrates: versioning, active selection, rollback, history.
    """

    def __init__(self):
        self._store = {}  # {name: [versions]}

    def register(self, name: str, content: str, model: str = "",
                 created_by: str = "system", reason: str = ""):
        """Register a new version of a prompt."""
        if name not in self._store:
            self._store[name] = []

        version = len(self._store[name]) + 1
        entry = {
            "version": version,
            "content": content,
            "model": model,
            "is_active": True,
            "created_by": created_by,
            "created_at": datetime.now().isoformat(),
            "reason": reason,
        }

        # Deactivate all previous versions
        for prev in self._store[name]:
            prev["is_active"] = False

        self._store[name].append(entry)
        return version

    def get_active(self, name: str) -> str:
        """Get the currently active prompt content."""
        for entry in reversed(self._store.get(name, [])):
            if entry["is_active"]:
                return entry["content"]
        return ""

    def rollback(self, name: str):
        """Roll back to the previous version."""
        versions = self._store.get(name, [])
        if len(versions) < 2:
            return "Nothing to roll back to."

        # Deactivate current
        versions[-1]["is_active"] = False
        # Activate previous
        versions[-2]["is_active"] = True
        return f"Rolled back to v{versions[-2]['version']}"

    def history(self, name: str):
        """Get full version history."""
        return self._store.get(name, [])


# DEMO
registry = PromptRegistry()

# Register v1
registry.register(
    "agent_system",
    "You are DocSage. Answer using only the context.",
    model="llama-3.1-8b",
    created_by="tiru",
    reason="initial version"
)

# Register v2 (improved)
registry.register(
    "agent_system",
    "You are DocSage. Answer ONLY using the context. If no info, say 'I don't know'.",
    model="llama-3.1-8b",
    created_by="tiru",
    reason="added explicit I-dont-know instruction per eval results"
)

print("Active prompt:", registry.get_active("agent_system"))
print("History:", [(h["version"], h["is_active"], h["reason"]) for h in registry.history("agent_system")])
print(registry.rollback("agent_system"))
print("After rollback:", registry.get_active("agent_system"))


# =================================================================================
# YAML-BASED PATTERN (common in production)
# =================================================================================

SAMPLE_YAML = """
# prompts/agent/system_v2.yaml
name: agent_system_prompt
version: "2.0"
model: llama-3.1-8b-instant
temperature: 0
description: "Agent system prompt with strict grounding + I-dont-know fallback"
content: |
  You are DocSage, an intelligent document assistant.
  Rules:
  1. Answer ONLY from the provided context.
  2. If the context doesn't contain the answer, say "I don't have this information."
  3. Cite [Source N] for every claim.
  4. Keep answers concise (3-6 sentences).
"""
print("\nSample YAML prompt file:")
print(SAMPLE_YAML)


# =================================================================================
# SECTION 10: GOLDEN LESSONS
# =================================================================================
"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GOLDEN LESSON 1 — PROMPTS ARE CONFIG, NOT CODE.
    They change more often than code. Treat them accordingly: versioned,
    tested, deployed through environments, and rollbackable.

GOLDEN LESSON 2 — NEVER SCATTER PROMPTS INLINE.
    Centralize in a dedicated module/library. One source of truth.
    Finding "where is that prompt?" should take seconds, not a grep hunt.

GOLDEN LESSON 3 — EVERY CHANGE GOES THROUGH A TEST.
    Golden test set + automated evaluation = prompt CI. A bad prompt is as
    impactful as a bad code change — treat it with the same rigor.

GOLDEN LESSON 4 — NEVER DELETE, ALWAYS DEPRECATE.
    Keep the full version history. You need rollback AND audit trail.
    "What prompt generated that wrong answer on March 5?" — you can answer
    this only if you kept the history.

GOLDEN LESSON 5 — SEPARATE TEMPLATE FROM DYNAMIC VALUES.
    The template (structure, instructions) is versioned.
    The dynamic values (context, user query) are injected at runtime.
    Never hardcode runtime values into the template.

GOLDEN LESSON 6 — VERSION PER MODEL.
    A prompt optimized for GPT-4 may not work on Llama-3. When you
    switch models, re-test all prompts against the golden set.

GOLDEN LESSON 7 — USE THE PROGRESSION STORY IN INTERVIEWS.
    "For small projects: constants + Git. For medium teams: YAML/DB registry.
    For enterprise: platforms with A/B testing + approval workflows."
    This shows you know the spectrum and scale appropriately.

GOLDEN LESSON 8 — COMPOSE, DON'T COPY.
    Shared blocks (persona, format instructions) composed into prompts.
    Change the persona once, ALL prompts update. DRY applies to prompts too.

GOLDEN LESSON 9 — KNOW WHAT YOUR ANSWER WAS MISSING.
    When you said "we update it like code," you were correct but too vague.
    The winning answer adds: "with a golden test set for regression, a
    version trail for rollback, and environment promotion for safety."
    Same WHAT, richer HOW.

GOLDEN LESSON 10 — THE ANALOGY THAT LANDS.
    "We treat prompts like database migrations — versioned, tested, promoted
    through environments, and rollbackable." This one sentence makes you
    sound production-experienced.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ONE-LINE CHEAT SHEET:

    Storage:    dedicated module, not scattered inline.
    Versioning: Git (simple) | DB registry with is_active flag (scalable).
    Testing:    golden test set + automated eval on every change (prompt CI).
    Deployment: code deploy (Git) | runtime swap (DB) | gradual rollout (flags).
    Rollback:   revert commit | flip is_active | set flag to 0%. Always instant.
    Organization: by domain (agent/grading/generation), with shared composable blocks.
    Tools:      LangSmith Hub, PromptLayer, Humanloop, or custom DB table.
    Principle:  prompts are config — versioned, tested, deployed, and rollbackable.
"""


# =================================================================================
# RUN SUMMARY
# =================================================================================

if __name__ == "__main__":
    print()
    print("=" * 70)
    print("LESSON 25: PROMPT MANAGEMENT & VERSIONING")
    print("=" * 70)
    print()
    print("THE INTERVIEW ANSWER IN ONE BREATH:")
    print("  'Prompts in a dedicated module, versioned via Git or DB registry,")
    print("   tested against a golden set before promotion, deployed through")
    print("   environments, rollbackable in seconds. Same discipline as code.'")
    print()
    print("THE PROGRESSION:")
    print("  Small project -> constants + Git PR review")
    print("  Medium team   -> DB registry (is_active flag, no-deploy updates)")
    print("  Enterprise    -> Platform (LangSmith/PromptLayer) + A/B + approval")
    print()
    print("THE KILLER ANALOGY:")
    print("  'We treat prompts like database migrations.'")
    print()
    print("=" * 70)
