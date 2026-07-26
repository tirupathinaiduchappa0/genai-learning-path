"""
===================================================================================
LESSON 6 — AGENT DESIGN PATTERNS & EVALUATION  (Senior DS / GenAI Real-Time Deep-Dive)
===================================================================================

COVERS (Section F of the question bank, part 1 of 2 — F4/F6/F7 are in Lesson 7):
  F1. Chain vs Agent (LangChain) — in depth
  F2. ReAct vs Plan-and-Execute — in depth
  F3. How do you evaluate an agent?
  F5. Debugging an agent failure in production

HOW TO READ:  THEORY -> RUNNABLE CODE -> INTERVIEW ANSWER -> RELATED CONCEPTS.

Run:  & "C:\\Projects\\Gen AI Udemy\\LangCGS\\.venv\\Scripts\\python.exe" "06_agent_design_patterns_evaluation.py"

No LLM calls are made — every "reasoning" step below is a deterministic mock so the
CONTROL FLOW (the thing interviewers actually probe) is 100% reproducible without
needing an API key. That control flow is identical to what a real LLM-backed
agent does; only the "brain" behind each decision is swapped for a rule.
===================================================================================
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass


def banner(title: str) -> None:
    line = "=" * 83
    print("\n" + line + "\n" + title + "\n" + line)


def sub(title: str) -> None:
    print("\n" + "━" * 79 + "\n" + title + "\n" + "━" * 79)


# ===================================================================================
# F1. CHAIN vs AGENT (LangChain) — IN DEPTH
# ===================================================================================
#
# DEFINITIONS
#   Chain: a FIXED, predetermined sequence of steps — step 2 always runs after step 1,
#          regardless of what step 1 produced (beyond feeding it forward as input).
#          You, the developer, hard-code the control flow at build time.
#   Agent: the LLM itself DECIDES the next action at each step, based on the current
#          state/observation — the control flow is determined at RUN time by the
#          model's own reasoning, not hard-coded by you.
#
# CONTROL FLOW
#   Chain:  A -> B -> C. Always. Even if B's output suggests C is pointless, C runs.
#   Agent:  observe -> LLM decides {call tool X, call tool Y, or finish} -> observe
#           result -> LLT decides again -> ... loop until the LLM decides it's done.
#           The SEQUENCE and the TOOLS USED are not fixed in advance.
#
# THE LLM'S ROLE
#   Chain: the LLM (if used at all) is just an ONE OF the pipeline's fixed stages —
#          e.g., "summarize this text" — it does NOT choose what happens next.
#   Agent: the LLM IS the reasoning/decision engine controlling the loop itself —
#          "given this state, what should I do next?"
#
# WHEN TO CHOOSE WHICH (the tradeoffs, in one breath each)
#   Choose a CHAIN when: the steps are known and always the same (extract -> summarize
#   -> translate) — predictable, cheap, deterministic, easy to test and debug, no risk
#   of the model wandering into an unplanned tool call.
#   Choose an AGENT when: the right sequence of actions genuinely depends on the
#   input and can't be known in advance (e.g., "answer this question" might need 0,
#   1, or 5 tool calls depending on the question) — flexible, but less predictable,
#   costs more tokens (multiple LLM calls in the loop), and harder to guarantee
#   correctness/safety.
#   RULE OF THUMB: use the SIMPLEST thing that solves the problem. Most production
#   "agents" are actually a chain with ONE conditional branch — reach for a full
#   agent loop only when the branching genuinely can't be enumerated up front.


@dataclass
class ChainStep:
    name: str
    fn: callable


def run_chain(steps: list[ChainStep], input_data: str) -> str:
    """A chain: FIXED sequence, always all steps, in this order, no matter what."""
    data = input_data
    for step in steps:
        data = step.fn(data)
        print(f"    [chain] {step.name} -> {data!r}")
    return data


def mock_extract(text: str) -> str:
    return text.split(".")[0]           # pretend: extract the first sentence


def mock_summarize(text: str) -> str:
    return f"SUMMARY({text[:20]}...)"


def mock_translate(text: str) -> str:
    return f"TRANSLATED({text})"


class SimpleAgent:
    """An agent: at EACH step the 'LLM' (mocked decision fn) decides what to do next,
    based on current state — not a fixed developer-hardcoded sequence."""

    def __init__(self, tools: dict[str, callable], decide_fn: callable, max_steps: int = 6):
        self.tools = tools
        self.decide_fn = decide_fn   # stands in for "ask the LLM what to do next"
        self.max_steps = max_steps

    def run(self, goal: str) -> str:
        state = {"goal": goal, "observations": []}
        for step_num in range(1, self.max_steps + 1):
            action = self.decide_fn(state)   # the "LLM" decides, given CURRENT state
            print(f"    [agent step {step_num}] decision: {action['type']} "
                  f"{action.get('tool', '')}")
            if action["type"] == "finish":
                return action["answer"]
            result = self.tools[action["tool"]](state)
            state["observations"].append((action["tool"], result))
            print(f"                     observation: {result!r}")
        return "max steps reached without finishing"


def _demo_chain_vs_agent() -> None:
    sub("F1. CHAIN (fixed) vs AGENT (dynamic) — runnable side-by-side")
    print("A) CHAIN — same 3 steps run EVERY time, regardless of content:")
    steps = [ChainStep("extract", mock_extract),
             ChainStep("summarize", mock_summarize),
             ChainStep("translate", mock_translate)]
    run_chain(steps, "The refund window is 30 days. Items must be unused.")

    print("\nB) AGENT — the 'LLM' decides what to do next based on the goal AND what")
    print("   it has already observed; a DIFFERENT goal takes a DIFFERENT path:")

    def decide(state: dict) -> dict:
        # Mocked "LLM reasoning": decide based on the goal + how many observations exist.
        goal, obs_count = state["goal"], len(state["observations"])
        if "weather" in goal and obs_count == 0:
            return {"type": "tool_call", "tool": "get_weather"}
        if "weather" in goal and obs_count == 1:
            return {"type": "finish", "answer": f"Weather lookup done: {state['observations'][-1][1]}"}
        if "calculate" in goal and obs_count == 0:
            return {"type": "tool_call", "tool": "calculator"}
        if "calculate" in goal and obs_count == 1:
            return {"type": "finish", "answer": f"Calc result: {state['observations'][-1][1]}"}
        return {"type": "finish", "answer": "no tool needed for this goal"}

    tools = {
        "get_weather": lambda state: "72F and sunny",
        "calculator": lambda state: 42,
    }
    agent = SimpleAgent(tools, decide)
    print("\n  Goal A: 'what's the weather'")
    print("  ->", agent.run("what's the weather"))
    print("\n  Goal B: 'calculate 6*7' (a DIFFERENT tool gets picked — the sequence")
    print("           was never hard-coded, the agent chose it from the goal):")
    print("  ->", agent.run("calculate 6*7"))
    print("\n  Goal C: 'just say hi' (the agent decides NO tool is needed at all):")
    print("  ->", agent.run("just say hi"))


# INTERVIEW ANSWER (F1):
#   "A chain is a fixed, developer-defined sequence of steps that always runs the
#    same way; the LLM, if present, is just one stage in that pipeline. An agent
#    flips the control: the LLM itself decides the next action at each step based on
#    current state, so the sequence of tool calls isn't known until runtime. I reach
#    for a chain whenever the steps are enumerable in advance — cheaper, more
#    predictable, easier to test — and only reach for a full agent loop when the
#    right sequence of actions genuinely depends on the input and can't be
#    hard-coded."
#
# RELATED CONCEPTS: LCEL pipelines as chains; create_react_agent as the agent
#   primitive; 'agent' is really "chain + a loop + a decision function"; router
#   chains (a chain with ONE conditional branch) as the middle ground.


# ===================================================================================
# F2. ReAct vs PLAN-AND-EXECUTE — IN DEPTH
# ===================================================================================
#
# ReAct (Reason + Act, interleaved)
#   Loop: THOUGHT (reason about what to do next) -> ACTION (call a tool) ->
#         OBSERVATION (tool result) -> THOUGHT (reason again, now with new info) ->
#         ... -> FINISH.
#   The plan is IMPLICIT and re-derived at every single step, using the latest
#   observation. This is what create_react_agent implements.
#   + Naturally adapts mid-task: a surprising tool result immediately changes the
#     next decision — no separate re-planning step needed.
#   - Can be inefficient/myopic on multi-step tasks: it only reasons ONE step ahead
#     each time, so it may take a locally-sensible but globally suboptimal path, and
#     it re-pays the "what should I do" reasoning cost on every single step.
#
# PLAN-AND-EXECUTE (plan up front, then execute)
#   Two phases: (1) PLANNER — an LLM call that lays out the FULL multi-step plan
#   up front (e.g., "1. search for X, 2. compute Y from X, 3. summarize"). (2)
#   EXECUTOR — carries out each planned step (often with its OWN ReAct-style loop per
#   step), optionally RE-PLANNING if a step's result invalidates the rest of the plan.
#   + More efficient/coherent on complex, multi-step tasks — the model reasons about
#     the WHOLE task once instead of one step at a time; easier to inspect/approve
#     the plan before execution (a natural human-in-the-loop checkpoint).
#   - The upfront plan can become STALE/WRONG once execution reveals new information
#     — you need an explicit re-planning step to recover, adding complexity.
#
# FAILURE MODES
#   ReAct:  can loop or "wander" on tasks needing genuine lookahead; every step costs
#           an LLM call (more total tokens spent on plumbing vs. one big Planner call).
#   Plan-and-Execute: brittle when the world doesn't match the plan (a step fails and
#           the rest of the plan silently no longer makes sense) unless re-planning is
#           built in explicitly.
#
# WHEN TO CHOOSE WHICH
#   ReAct: simpler tasks, or tasks where each step's outcome genuinely changes what
#          should happen next in an unpredictable way (exploratory, reactive tasks).
#   Plan-and-Execute: complex, multi-step tasks with mostly-predictable structure
#          where you WANT visibility/approval of the plan before it runs, or where
#          the cost of many small LLM calls (ReAct) is worse than one planning call.


def mock_llm_react_step(goal: str, observations: list, step: int) -> dict:
    """Mocked ReAct 'thought': decide the NEXT single action given only what's been
    observed SO FAR — no full plan exists ahead of time."""
    if "temperature" in goal and step == 0:
        return {"thought": "I need the current weather first.",
                "action": "get_weather"}
    if "temperature" in goal and step == 1:
        temp = observations[-1][1]
        if temp > 80:
            return {"thought": f"{temp}F is hot, I should also check the AC status.",
                    "action": "check_ac"}
        return {"thought": f"{temp}F is fine, no need to check anything else.",
                "action": "finish", "answer": f"It's {temp}F, no AC needed."}
    if "temperature" in goal and step == 2:
        ac = observations[-1][1]
        return {"thought": "Got AC status, now I can answer.", "action": "finish",
                "answer": f"It's hot and the AC is {ac}."}
    return {"thought": "nothing more to do", "action": "finish", "answer": "done"}


def run_react(goal: str, tools: dict, max_steps: int = 5) -> str:
    """ReAct loop: Thought -> Action -> Observation, re-deciding every step."""
    observations = []
    for step in range(max_steps):
        decision = mock_llm_react_step(goal, observations, step)
        print(f"    [ReAct] Thought: {decision['thought']}")
        if decision["action"] == "finish":
            return decision["answer"]
        result = tools[decision["action"]]()
        observations.append((decision["action"], result))
        print(f"            Action: {decision['action']}() -> Observation: {result!r}")
    return "max steps reached"


def mock_llm_planner(goal: str) -> list[str]:
    """Plan-and-Execute PLANNER: one LLM call lays out the WHOLE plan up front."""
    if "temperature" in goal:
        return ["get_weather", "check_ac_if_hot", "compose_final_answer"]
    return ["compose_final_answer"]


def run_plan_and_execute(goal: str, tools: dict) -> str:
    """Plan-and-Execute: plan ALL steps first (visible/inspectable), then execute."""
    plan = mock_llm_planner(goal)
    print(f"    [Plan] full upfront plan: {plan}")
    observations = {}
    for i, planned_step in enumerate(plan, 1):
        if planned_step == "compose_final_answer":
            temp = observations.get("get_weather")
            ac = observations.get("check_ac_if_hot")
            answer = f"It's {temp}F" + (f" and the AC is {ac}." if ac else ", no AC needed.")
            print(f"    [Execute step {i}] {planned_step} -> {answer!r}")
            return answer
        if planned_step == "check_ac_if_hot":
            temp = observations.get("get_weather", 0)
            if temp <= 80:
                print(f"    [Execute step {i}] {planned_step} -> SKIPPED "
                      f"(re-plan check: {temp}F doesn't need AC check)")
                continue
        result = tools[planned_step]()
        observations[planned_step] = result
        print(f"    [Execute step {i}] {planned_step}() -> {result!r}")
    return "plan executed without a final answer"


def _demo_react_vs_plan_execute() -> None:
    sub("F2. ReAct (interleaved) vs PLAN-AND-EXECUTE (upfront plan) — runnable")
    tools = {"get_weather": lambda: 92, "check_ac": lambda: "running"}

    print("A) ReAct — reasons ONE step at a time, decides mid-flight:")
    answer = run_react("what's the temperature", tools)
    print(f"   final answer: {answer!r}")

    tools2 = {"get_weather": lambda: 92, "check_ac_if_hot": lambda: "running"}
    print("\nB) Plan-and-Execute — lays out the FULL plan BEFORE executing anything,")
    print("   and can skip a planned step once execution reveals it's unnecessary:")
    answer2 = run_plan_and_execute("what's the temperature", tools2)
    print(f"   final answer: {answer2!r}")


# INTERVIEW ANSWER (F2):
#   "ReAct interleaves reasoning and acting — thought, action, observation, repeat —
#    re-deriving the next step from the latest observation every time, which adapts
#    well but only reasons one step ahead and pays an LLM call per step. Plan-and-
#    Execute separates a planner, which lays out the full multi-step plan up front in
#    one call, from an executor that carries it out — more efficient and inspectable
#    for complex tasks, but the plan can go stale if execution reveals something
#    unexpected, so you need an explicit re-planning path. I use ReAct for simpler
#    or genuinely reactive tasks, and Plan-and-Execute when I want a visible,
#    approvable plan for a complex multi-step task."
#
# RELATED CONCEPTS: LangGraph's plan-and-execute reference architecture; ReWOO
#   (reasoning without observation — plan tool calls without executing them yet,
#   then run them, cutting LLM calls); tree-of-thought as a lookahead variant of
#   ReAct; human-in-the-loop approval gates fit naturally into Plan-and-Execute.


# ===================================================================================
# F3. HOW DO YOU EVALUATE AN AGENT?
# ===================================================================================
#
# WHY THIS IS HARDER THAN EVALUATING A PLAIN LLM OR RAG ANSWER
#   A plain LLM answer or a RAG answer is a single input -> single output you can
#   score directly. An agent produces a TRAJECTORY — a sequence of decisions, tool
#   calls, and observations — so you must evaluate the PROCESS, not just the final
#   answer. Two agents can reach the same correct answer through very different
#   quality of reasoning (one efficient and safe, one wasteful or lucky).
#
# DIMENSIONS TO EVALUATE
#   - TASK SUCCESS / COMPLETION: did it actually achieve the goal? (binary or scored)
#   - CORRECTNESS: is the final answer/action factually and logically right?
#   - TOOL-USE ACCURACY: did it call the RIGHT tools, with the RIGHT arguments? Did
#     it avoid calling tools it didn't need (wasteful/unsafe calls)?
#   - REASONING QUALITY: are the intermediate thoughts coherent and justified, or
#     is it "right for the wrong reason" (fragile — will fail on a slight variation)?
#   - LATENCY: how many steps / how much wall-clock time to finish?
#   - COST: how many LLM calls / tokens / tool-API costs consumed?
#   - ERROR RECOVERY: when a tool call fails or returns unexpected data, does the
#     agent notice and adapt, or does it plow ahead incorrectly?
#
# METHODS
#   - TRAJECTORY / STEP-LEVEL EVALUATION: score not just the final answer but EACH
#     step — "was this tool call justified given the state at that point?" Requires
#     either a labeled reference trajectory or an LLM-as-judge looking at each step.
#   - LLM-AS-JUDGE: a separate (often stronger) LLM scores the trajectory and/or final
#     answer against a rubric — scalable but itself imperfect and needs calibration.
#   - HUMAN EVALUATION: for high-stakes or ambiguous tasks, the ground truth for
#     "was this the right sequence of actions" often still needs a human rater,
#     especially early on before you trust an automated judge.
#   - AUTOMATED / RULE-BASED CHECKS: for tasks with a verifiable outcome (a SQL query
#     that must return matching rows, a booking that must exist in the system after
#     the run), just check the outcome programmatically — cheap and unambiguous
#     wherever the domain allows it.
#
# EVALUATING MULTI-STEP BEHAVIOR SPECIFICALLY
#   Check: (1) were the RIGHT tools chosen (not extra, not missing)? (2) was the
#   ORDER sensible (dependencies respected — e.g., don't book before checking
#   availability)? (3) on a tool ERROR or unexpected result, did the agent adapt its
#   plan, retry sensibly, or fail gracefully — versus hallucinating success or
#   looping?
#
# SETUP IN PRACTICE
#   - Build/curate a DATASET of representative tasks (with expected tool sequences
#     and/or expected outcomes where possible).
#   - OFFLINE evaluation: run the agent against this dataset in CI/pre-release,
#     compare to a baseline or to expected trajectories/outcomes — catches
#     regressions before shipping.
#   - ONLINE/production evaluation: sample real production trajectories, monitor
#     task-success rate, tool-error rate, average steps/cost, and flag/alert on
#     drift or spikes in failure rate; route a sample to human review.


@dataclass
class TrajectoryStep:
    tool: str
    args: dict
    result: str
    was_necessary: bool = True   # ground-truth label for eval: should this call have happened?


@dataclass
class AgentTrajectory:
    goal: str
    steps: list[TrajectoryStep] = field(default_factory=list)
    final_answer: str = ""
    expected_tools: list[str] = field(default_factory=list)
    expected_answer: str = ""


def evaluate_trajectory(traj: AgentTrajectory) -> dict:
    """A concrete, runnable agent-evaluation scorecard covering the dimensions above."""
    used_tools = [s.tool for s in traj.steps]

    # Tool-use accuracy: precision/recall vs the expected tool set (reuses the exact
    # TP/FP/FN framing from Lesson 6's evaluation section — same idea, applied to
    # tool calls instead of retrieved documents).
    used_set, expected_set = set(used_tools), set(traj.expected_tools)
    tp = len(used_set & expected_set)
    fp = len(used_set - expected_set)          # called but NOT needed — wasteful/unsafe
    fn = len(expected_set - used_set)          # needed but NEVER called — incomplete
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0

    unnecessary_calls = [s.tool for s in traj.steps if not s.was_necessary]
    task_success = traj.final_answer.strip().lower() == traj.expected_answer.strip().lower()

    return {
        "task_success": task_success,
        "tool_precision": round(precision, 2),   # were the tools it called the RIGHT ones?
        "tool_recall": round(recall, 2),          # did it call ALL the tools it needed?
        "num_steps": len(traj.steps),             # proxy for latency/cost
        "unnecessary_calls": unnecessary_calls,   # wasteful/unsafe tool-use flags
    }


def _demo_agent_evaluation() -> None:
    sub("F3. AGENT EVALUATION SCORECARD (runnable)")

    good_traj = AgentTrajectory(
        goal="book a flight if the weather is clear",
        steps=[
            TrajectoryStep("get_weather", {}, "clear", was_necessary=True),
            TrajectoryStep("book_flight", {}, "confirmed", was_necessary=True),
        ],
        final_answer="Flight booked.",
        expected_tools=["get_weather", "book_flight"],
        expected_answer="Flight booked.",
    )

    wasteful_traj = AgentTrajectory(
        goal="book a flight if the weather is clear",
        steps=[
            TrajectoryStep("get_weather", {}, "clear", was_necessary=True),
            TrajectoryStep("check_ac", {}, "n/a", was_necessary=False),  # irrelevant call
            TrajectoryStep("book_flight", {}, "confirmed", was_necessary=True),
        ],
        final_answer="Flight booked.",
        expected_tools=["get_weather", "book_flight"],
        expected_answer="Flight booked.",
    )

    incomplete_traj = AgentTrajectory(
        goal="book a flight if the weather is clear",
        steps=[TrajectoryStep("book_flight", {}, "confirmed", was_necessary=True)],
        final_answer="Flight booked.",   # right ANSWER, but skipped the weather check!
        expected_tools=["get_weather", "book_flight"],
        expected_answer="Flight booked.",
    )

    for label, traj in [("efficient/correct", good_traj), ("wasteful call", wasteful_traj),
                        ("skipped a required check", incomplete_traj)]:
        scorecard = evaluate_trajectory(traj)
        print(f"  [{label}]")
        for k, v in scorecard.items():
            print(f"      {k}: {v}")
    print("\n  -> 'incomplete_traj' gets task_success=True (same final answer!) but")
    print("     tool_recall < 1.0 flags that it skipped a required step — this is")
    print("     EXACTLY why trajectory-level eval catches things outcome-only eval")
    print("     would miss: it got lucky/right for the WRONG process.")


# INTERVIEW ANSWER (F3):
#   "I evaluate the trajectory, not just the final answer, because an agent can reach
#    a correct answer through a wrong or unsafe process. I track task success,
#    tool-use precision/recall against an expected tool set — did it call the right
#    tools, and did it skip any required ones — reasoning quality, step count as a
#    latency/cost proxy, and how it handles tool errors. I use LLM-as-judge or
#    rule-based checks for automated scoring at scale, human review for ambiguous or
#    high-stakes cases, and I run this offline against a curated task dataset before
#    release plus online against sampled production trajectories to catch drift."
#
# RELATED CONCEPTS: AgentBench / ToolBench-style benchmarks; reward hacking (an agent
#   optimizing the metric, not the goal); LangSmith trace-based evaluation; A/B
#   testing two agent versions on live traffic with guardrails.


# ===================================================================================
# F5. DEBUGGING AN AGENT FAILURE IN PRODUCTION
# ===================================================================================
#
# THE VERY FIRST THING TO CHECK
#   Before touching code: WHAT actually failed, from the USER'S perspective — wrong
#   answer? No answer/timeout? Crashed? Then immediately pull the TRACE for that
#   specific request (see observability below) — you need the exact trajectory,
#   not a guess, before doing anything else.
#
# STEP-BY-STEP PROCESS
#   1) REPRODUCE: run the same input (same prompt, same tool state if possible)
#      against the agent in a controlled environment. If it reproduces reliably,
#      you have a real bug; if it's intermittent, suspect nondeterminism (LLM
#      sampling, race conditions, flaky external API, timing).
#   2) INSPECT THE TRACE: walk the full step-by-step trajectory — every thought,
#      every tool call with its exact arguments, every observation/result, every
#      LLM prompt and response, in order. Tracing/observability tooling (LangSmith,
#      or custom structured logging) is what makes this possible at all; without it
#      you're debugging blind.
#   3) ISOLATE THE FAILING COMPONENT: at which STEP did things go wrong?
#        - LLM reasoning itself (bad decision despite correct inputs)?
#        - A tool call (wrong arguments, or the tool itself errored/returned bad data)?
#        - Retrieval (wrong or missing context fed to the LLM)?
#        - The prompt/template (missing instruction, ambiguous wording)?
#      Bisect: fix everything upstream of the suspect step and re-run just from
#      there to confirm.
#   4) ROOT CAUSE vs SYMPTOM: don't patch the symptom (e.g., "add a retry") without
#      understanding WHY — was it a genuinely flaky dependency (retry IS the fix),
#      or a logic bug that a retry will just paper over and hide until it recurs
#      worse?
#   5) VALIDATE THE FIX: re-run the exact failing case AND a broader regression
#      suite (Lesson 6-style trajectory eval) — a fix for one case must not silently
#      break other cases; watch metrics after deploy, not just at fix-time.
#
# THE ROLE OF OBSERVABILITY / TRACING
#   Without structured tracing, "debugging an agent" means re-reading opaque LLM
#   outputs with no visibility into intermediate tool calls — nearly impossible past
#   trivial cases. Tracing tools capture, per request: the full trajectory, latency
#   per step, token/cost per LLM call, and errors — turning "why did this fail"
#   from guesswork into a lookup.


@dataclass
class TraceEvent:
    step: int
    component: str          # "llm_reasoning" | "tool_call" | "retrieval" | "prompt"
    detail: str
    status: str              # "ok" | "error" | "unexpected"


def inspect_trace(trace: list[TraceEvent]) -> dict:
    """A minimal 'debugger': walk the trace in order and isolate the FIRST failing
    component — the systematic version of steps 2-3 above."""
    for event in trace:
        if event.status != "ok":
            return {
                "failing_step": event.step,
                "failing_component": event.component,
                "detail": event.detail,
                "upstream_steps_ok": event.step - 1,
            }
    return {"result": "no failure found in trace"}


def _demo_debugging_agent_failure() -> None:
    sub("F5. DEBUGGING AN AGENT FAILURE — trace inspection (runnable)")

    # Scenario: user reports "the agent gave a wrong refund amount."
    trace = [
        TraceEvent(1, "prompt", "built prompt with refund policy context", "ok"),
        TraceEvent(2, "retrieval", "retrieved chunk: '30-day refund window'", "ok"),
        TraceEvent(3, "tool_call", "get_order_total(order_id=4471) -> $89.99", "ok"),
        TraceEvent(4, "llm_reasoning",
                   "computed refund as $89.99 * 1.5 = $134.99 (invented a 'bonus' "
                   "multiplier not present anywhere in context)", "unexpected"),
        TraceEvent(5, "tool_call", "issue_refund(amount=134.99)", "ok"),  # tool worked fine;
        # it faithfully executed a BAD decision — the bug is upstream, at step 4.
    ]

    print("  Full trace for the failing request:")
    for e in trace:
        marker = "OK " if e.status == "ok" else "!! "
        print(f"    [{marker}step {e.step}] {e.component:<14} {e.detail}")

    diagnosis = inspect_trace(trace)
    print(f"\n  Diagnosis: {diagnosis}")
    print("\n  Root cause: NOT the tool (it executed correctly what it was told).")
    print("  NOT retrieval (context was correct, no '1.5x bonus' anywhere in it).")
    print("  It's the LLM REASONING step — it hallucinated a multiplier. The fix is")
    print("  a stricter, more constrained prompt/output schema at step 4 (e.g., force")
    print("  the refund amount to be copied verbatim from a tool result, not computed")
    print("  freely by the LLM) — NOT a retry, and NOT a change to the tool at step 5.")


# INTERVIEW ANSWER (F5):
#   "First I get the exact failing trace for that request — trajectory, tool calls
#    and arguments, retrieved context, and every LLM prompt/response — because
#    guessing without it is unreliable. I try to reproduce it; reproducible points to
#    a real logic bug, intermittent points to nondeterminism or a flaky dependency.
#    Then I walk the trace step by step to isolate WHICH component failed — LLM
#    reasoning, a tool call, retrieval, or the prompt itself — and fix the root
#    cause there, not just the symptom. Observability/tracing tooling is what makes
#    this tractable at all; without it you're debugging an agent blind. Finally I
#    validate the fix against the exact failing case plus a broader regression suite
#    before considering it resolved."
#
# RELATED CONCEPTS: LangSmith trace UI; structured logging with correlation/request
#   IDs; canary/shadow deployments to catch regressions before full rollout; the
#   'was this a logic bug or a flaky dependency' triage question as the very next
#   thing after isolating the component.


# ===================================================================================
# GOLDEN LESSONS
# ===================================================================================
GOLDEN_LESSONS = """
  1. Chain = developer-fixed sequence; Agent = LLM decides the next action at RUNTIME.
  2. Use a chain whenever steps are enumerable in advance — cheaper, predictable, testable.
  3. ReAct = reason+act interleaved, re-derived every step; adapts fast, but myopic + one LLM call/step.
  4. Plan-and-Execute = plan the whole task once, then execute; efficient + inspectable, but can go stale.
  5. Agent eval = score the TRAJECTORY, not just the final answer — right answer can hide a wrong process.
  6. Tool-use precision/recall = TP/FP/FN on tool calls, same framing as retrieval metrics, different subject.
  7. 'Task success' alone can mask a skipped required step — trajectory eval catches that, outcome eval doesn't.
  8. Debugging order: get the trace -> reproduce -> isolate the failing COMPONENT -> fix root cause -> validate.
  9. Isolate WHERE it broke: LLM reasoning vs tool call vs retrieval vs prompt — each has a different fix.
 10. A tool executing correctly on a BAD decision is still a failure — don't blame the tool for the reasoning bug.
 11. No tracing/observability = debugging an agent blind. It's not optional infrastructure, it's the debugger.
"""


if __name__ == "__main__":
    banner("LESSON 6 — AGENT DESIGN PATTERNS & EVALUATION")
    _demo_chain_vs_agent()
    _demo_react_vs_plan_execute()
    _demo_agent_evaluation()
    _demo_debugging_agent_failure()
    banner("GOLDEN LESSONS")
    print(GOLDEN_LESSONS)
