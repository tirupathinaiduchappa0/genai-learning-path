"""
===================================================================================
LESSON 9 — LLM APPLICATION ENGINEERING  (Senior DS / GenAI Real-Time Deep-Dive)
===================================================================================

COVERS (Section H of the question bank):
  H1. Structured JSON extraction prompt (support ticket -> validated JSON)
  H2. Knowledge-base AI agent (query -> retrieve -> grounded prompt -> answer)
  H3. Python function to call an LLM API (error handling, keys, config)
  H4. Fine-tuning in depth (full vs PEFT/LoRA/QLoRA, adapters, prefix tuning)
  H5. Temperature and top_p

HOW TO READ:  THEORY -> RUNNABLE CODE -> INTERVIEW ANSWER -> RELATED CONCEPTS.

Run:  & "C:\\Projects\\Gen AI Udemy\\LangCGS\\.venv\\Scripts\\python.exe" "09_llm_application_engineering.py"

NOTE ON LLM CALLS: this lesson is about the ENGINEERING around LLM calls (prompt
design, agent wiring, robust API-call code, sampling params). It does NOT require a
live API key — the LLM call itself is a mock so the STRUCTURE (retry logic, JSON
enforcement, agent flow, sampling math) is fully runnable and reproducible. Every
function is written so you could drop in a real client (openai/groq) unchanged.
===================================================================================
"""

from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

np.random.seed(9)
np.set_printoptions(precision=3, suppress=True)


def banner(title: str) -> None:
    line = "=" * 83
    print("\n" + line + "\n" + title + "\n" + line)


def sub(title: str) -> None:
    print("\n" + "━" * 79 + "\n" + title + "\n" + "━" * 79)


# ===================================================================================
# H1. STRUCTURED JSON EXTRACTION PROMPT  (support ticket -> validated JSON)
# ===================================================================================
#
# THE TASK
#   Extract structured fields from a free-text customer support ticket and return
#   VALID JSON — reliably, every time (downstream code will json.loads() it, so a
#   single stray word outside the JSON breaks the pipeline).
#
# HOW TO MAKE JSON OUTPUT RELIABLE (the techniques, in order of strength)
#   1) NATIVE STRUCTURED OUTPUT / FUNCTION CALLING: the strongest option — modern
#      APIs support a "response_format=json_object" or a JSON schema / tool-call
#      spec that the model is CONSTRAINED to fill. Prefer this over prompt-only
#      tricks whenever the provider supports it (OpenAI, Groq, Anthropic all do).
#   2) SCHEMA IN THE PROMPT: show the EXACT JSON shape you want with field names,
#      types, and allowed enum values — models follow a concrete template far better
#      than a prose description.
#   3) FEW-SHOT EXAMPLES: one or two input->output pairs pin down format and edge-case
#      handling (what to do with a missing field, etc.).
#   4) EXPLICIT INSTRUCTIONS: "Respond with ONLY valid JSON, no markdown, no prose,
#      no code fences." — reduces the classic ```json ... ``` wrapper problem.
#   5) VALIDATE + RETRY (belt and suspenders): parse the output; if it's not valid
#      JSON or fails schema validation, re-prompt with the error. NEVER assume the
#      model's output is valid — production JSON extraction ALWAYS validates.
#
# THE FIELDS (the question specified ticket_id + issue_category; a complete, senior
# answer ADDS the fields that make the output actually useful downstream):
#   ticket_id, issue_category (enum), priority (enum), customer_sentiment (enum),
#   summary (short free text), requested_action (short free text).


SUPPORT_TICKET_PROMPT = """You are a support-ticket triage system. Extract structured
data from the ticket below and respond with ONLY valid JSON — no markdown, no code
fences, no prose before or after.

Use EXACTLY this schema:
{
  "ticket_id": "<string, copy from input>",
  "issue_category": "<one of: billing | technical | account | shipping | other>",
  "priority": "<one of: low | medium | high | urgent>",
  "customer_sentiment": "<one of: positive | neutral | negative | angry>",
  "summary": "<one sentence summarizing the issue>",
  "requested_action": "<what the customer wants done>"
}

Rules:
- If a field cannot be determined, use "unknown" (never omit a field).
- Infer priority and sentiment from the tone and content.
- Respond with the JSON object ONLY.

Ticket:
{ticket_text}
"""


def extract_json_from_response(raw: str) -> dict:
    """Robustly pull a JSON object out of an LLM response — strips code fences and
    any accidental prose, then parses. This is the defensive layer step 5 above."""
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()   # strip leading ```json
    cleaned = re.sub(r"```$", "", cleaned).strip()             # strip trailing ```
    # Grab the outermost {...} in case the model added stray text around it.
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError("no JSON object found in response")
    return json.loads(match.group(0))


TICKET_SCHEMA = ["ticket_id", "issue_category", "priority",
                 "customer_sentiment", "summary", "requested_action"]


def validate_ticket(parsed: dict, schema: list[str] = TICKET_SCHEMA) -> dict:
    missing = [f for f in schema if f not in parsed]
    return {"valid": not missing, "missing_fields": missing}


def mock_llm_ticket_extraction(prompt: str) -> str:
    """Mock LLM: returns a JSON response (wrapped in a code fence, as models often
    do — to prove our extractor handles that). Swap for a real client unchanged."""
    return """```json
{
  "ticket_id": "TICK-88213",
  "issue_category": "billing",
  "priority": "high",
  "customer_sentiment": "angry",
  "summary": "Customer was charged twice for the same monthly subscription.",
  "requested_action": "Refund the duplicate charge and confirm by email."
}
```"""


def _demo_json_extraction() -> None:
    sub("H1. STRUCTURED JSON EXTRACTION — prompt + validate + parse (runnable)")
    ticket = ("Ticket TICK-88213: I've been charged TWICE for my subscription this "
              "month and I'm furious. Fix this now and refund the extra charge.")
    prompt = SUPPORT_TICKET_PROMPT.replace("{ticket_text}", ticket)
    print("  prompt sent to the LLM (schema + rules + ticket) — first 3 lines:")
    for line in prompt.split("\n")[:3]:
        print(f"    {line}")

    raw = mock_llm_ticket_extraction(prompt)
    print(f"\n  raw LLM response (note the ```json fence the model added):")
    print(f"    {raw.splitlines()[0]} ... {raw.splitlines()[-1]}")

    parsed = extract_json_from_response(raw)      # defensive parse (handles the fence)
    validation = validate_ticket(parsed)          # schema validation before trusting it
    print(f"\n  parsed JSON: {json.dumps(parsed, indent=None)}")
    print(f"  schema validation: {validation}")
    print("  -> code-fence stripped, JSON parsed, all required fields present. Safe to")
    print("     hand downstream. In prod I'd also retry on a validation failure.")


# INTERVIEW ANSWER (H1):
#   "I make JSON reliable in layers. Strongest is native structured output or
#    function calling, where the API constrains the model to a JSON schema — I prefer
#    that whenever the provider supports it. On top I put the exact schema in the
#    prompt with field names, types, and enum values, a few-shot example or two, and
#    an explicit 'respond with only valid JSON, no code fences' instruction. Then —
#    always — I parse defensively, stripping any code fences or stray prose, validate
#    against the schema, and retry with the error if it fails. I never assume the
#    model's output is valid JSON; production extraction validates every time."
#
# RELATED CONCEPTS: Pydantic models + instructor/LangChain structured output;
#   JSON mode vs function/tool calling; enums to constrain categorical fields;
#   handling partial/streaming JSON; the ```json code-fence problem specifically.


# ===================================================================================
# H2. KNOWLEDGE-BASE AI AGENT  (query -> retrieve -> grounded prompt -> answer)
# ===================================================================================
#
# THE FLOW (this is a minimal RAG agent — it ties Lessons 3/4/5 together into one
# working loop):
#   1) ACCEPT the user's query.
#   2) EMBED the query and RETRIEVE the most relevant chunks from the knowledge base
#      (vector similarity — Lesson 3).
#   3) CONSTRUCT a grounded prompt: inject the retrieved chunks as context, with an
#      explicit instruction to answer ONLY from that context and say "I don't know"
#      otherwise (grounding — Lesson 5's hallucination reduction).
#   4) GENERATE the answer with the LLM.
#   5) (Senior touch) return CITATIONS — which chunks backed the answer — so it's
#      verifiable, and so a downstream faithfulness check (Lesson 8) has something to
#      check against.
#
# WHY THIS IS AN "AGENT" (vs a plain chain): a fuller version lets the LLM DECIDE
# whether to retrieve at all, retrieve again with a rewritten query if the first
# retrieval was weak, or call other tools — that decision loop (Lesson 6) is what
# makes it agentic. Here we show the core retrieve-then-ground skeleton clearly.


def _stable_hash(s: str) -> int:
    """Deterministic hash (Python's built-in hash() is salted per-process, which would
    make the toy embedder non-reproducible across runs — this fixes that)."""
    import hashlib
    return int.from_bytes(hashlib.md5(s.encode()).digest()[:4], "little")


_STOPWORDS = {
    "how", "do", "i", "my", "is", "the", "a", "an", "to", "of", "for", "what",
    "your", "are", "within", "then", "go", "me", "on", "you", "and", "in", "it",
    "current", "get",
}


def toy_embed(text: str, d: int = 4096) -> np.ndarray:
    """Deterministic WORD-level bag-of-words embedder with stopword removal + crude
    stemming (trailing 's'). Far cleaner separation than char-trigrams for these toy
    KB demos, and fully reproducible. Real systems use a trained embedding model."""
    v = np.zeros(d)
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        stem = word[:-1] if word.endswith("s") and len(word) > 3 else word  # refunds->refund
        if stem in _STOPWORDS:
            continue
        v[_stable_hash(stem) % d] += 1.0
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


@dataclass
class KBChunk:
    chunk_id: str
    text: str


class KnowledgeBaseAgent:
    """Minimal KB agent: retrieve relevant chunks, build a grounded prompt, answer
    (with citations). The LLM call is a mock; the retrieval + grounding is real."""

    def __init__(self, knowledge_base: list[KBChunk], llm_fn, top_k: int = 2,
                 min_similarity: float = 0.05) -> None:
        self.kb = knowledge_base
        self.kb_vectors = [(c, toy_embed(c.text)) for c in knowledge_base]
        self.llm_fn = llm_fn
        self.top_k = top_k
        self.min_similarity = min_similarity

    def retrieve(self, query: str) -> list[tuple[KBChunk, float]]:
        qv = toy_embed(query)
        scored = [(c, cosine(qv, v)) for c, v in self.kb_vectors]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:self.top_k]

    def build_grounded_prompt(self, query: str, chunks: list[KBChunk]) -> str:
        context = "\n".join(f"[{c.chunk_id}] {c.text}" for c in chunks)
        return (
            "Answer the question using ONLY the context below. If the answer is not "
            "in the context, say 'I don't know based on the available information.' "
            "Cite the [chunk_id] you used.\n\n"
            f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        )

    def answer(self, query: str) -> dict:
        retrieved = self.retrieve(query)
        # Guard: if the best match is too weak, don't even generate — avoids
        # hallucinating on an out-of-scope question (retrieval-confidence gate).
        if not retrieved or retrieved[0][1] < self.min_similarity:
            return {"answer": "I don't know based on the available information.",
                    "citations": [], "grounded": False}
        chunks = [c for c, _ in retrieved]
        prompt = self.build_grounded_prompt(query, chunks)
        response = self.llm_fn(prompt)
        return {"answer": response, "citations": [c.chunk_id for c in chunks],
                "grounded": True}


def mock_kb_llm(prompt: str) -> str:
    """Mock generation: EXTRACTS the answer from the TOP retrieved chunk in the
    prompt's context (a real grounded LLM answers from the most relevant context).
    We echo the first '[KB-x] text' line — deterministic and truly grounded, so the
    answer always reflects what retrieval actually ranked highest."""
    match = re.search(r"\[(KB-\d+)\]\s*(.+)", prompt)
    if match:
        return f"{match.group(2).strip()} [{match.group(1)}]"
    return "I don't know based on the available information."


def _demo_kb_agent() -> None:
    sub("H2. KNOWLEDGE-BASE AI AGENT — retrieve + ground + answer (runnable)")
    kb = [
        KBChunk("KB-1", "Refunds are accepted within 30 days for unused items."),
        KBChunk("KB-2", "To reset your password go to Settings then Security then Reset Password."),
        KBChunk("KB-3", "Enterprise accounts have a dedicated account manager."),
    ]
    # Threshold chosen so genuine in-scope queries pass and out-of-scope is declined
    # (see the printed top-similarity for each query below to see WHY).
    agent = KnowledgeBaseAgent(kb, llm_fn=mock_kb_llm, min_similarity=0.10)

    for query in ["how do I reset my password?",
                  "what is your refund policy?",
                  "what is the current stock price of Tesla?"]:  # out-of-scope -> decline
        top_sim = agent.retrieve(query)[0][1]
        result = agent.answer(query)
        print(f"  Q: {query}")
        print(f"     top similarity: {top_sim:.3f} (gate={agent.min_similarity})")
        print(f"     answer: {result['answer']}")
        print(f"     citations: {result['citations']}, grounded: {result['grounded']}")
    print("\n  -> in-scope questions clear the gate and get grounded, cited answers;")
    print("     the out-of-scope question falls below the retrieval-confidence gate and")
    print("     is declined BEFORE generation (no weak-context hallucination).")


# INTERVIEW ANSWER (H2):
#   "A knowledge-base agent accepts the query, embeds it and retrieves the most
#    similar chunks from the vector store, builds a prompt that injects those chunks
#    as context with an explicit instruction to answer only from them and say 'I
#    don't know' otherwise, then generates the answer and returns citations to the
#    chunks it used. I add a retrieval-confidence gate — if the best match is too
#    weak, I decline before even calling the LLM, which stops out-of-scope
#    hallucination. Making it fully agentic means letting the LLM decide whether to
#    retrieve, re-retrieve with a rewritten query, or call other tools in a loop."
#
# RELATED CONCEPTS: citations for verifiability + faithfulness eval (Lesson 8);
#   query rewriting on weak retrieval (Lesson 5); the agent loop that decides WHEN
#   to retrieve (Lesson 6); conversation memory via a checkpointer (Lesson 7).


# ===================================================================================
# H3. PYTHON FUNCTION TO CALL AN LLM API  (production-grade)
# ===================================================================================
#
# WHAT A SENIOR ANSWER INCLUDES (not just "call the endpoint" — the ROBUSTNESS):
#   - API KEY MANAGEMENT: read from an environment variable, NEVER hard-code it in
#     source (it'd leak into git). Fail fast with a clear error if it's missing.
#   - CONFIGURABLE PARAMETERS: model, temperature, max_tokens, timeout — passed in,
#     not buried as magic numbers.
#   - ERROR HANDLING: catch transient failures (rate limits, 5xx, timeouts) and
#     RETRY WITH EXPONENTIAL BACKOFF; distinguish those from permanent errors (bad
#     request, auth failure) that should fail immediately, not retry.
#   - TIMEOUTS: never let a call hang forever — bound it.
#   - RETURN a clean result (the text) and let errors surface as exceptions the
#     caller can handle, rather than returning None silently.


def call_llm_api(prompt: str, *, model: str = "llama-3.1-8b-instant",
                 temperature: float = 0.7, max_tokens: int = 512,
                 timeout: float = 30.0, max_retries: int = 3,
                 _client=None) -> str:
    """Production-grade LLM API call: env-based key, configurable params, timeout,
    and exponential-backoff retry on transient errors.

    In real code, `_client` would be an openai/groq client; here it defaults to a
    mock so this function is runnable and testable without a key or network. The
    control flow (key check, retry/backoff, error classification) is real.
    """
    import os

    # 1) API key management — read from env, fail fast with a clear message.
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
    if _client is None and not api_key:
        # In this dependency-free lesson we fall back to a mock instead of raising,
        # but the REAL behavior (documented) is: raise a clear config error here.
        _client = _mock_llm_client

    client = _client or _mock_llm_client

    # 2) Retry with exponential backoff on TRANSIENT errors only.
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            return client(prompt=prompt, model=model, temperature=temperature,
                          max_tokens=max_tokens, timeout=timeout)
        except PermanentAPIError:
            raise                        # bad request / auth — retrying won't help
        except TransientAPIError as e:
            last_error = e
            backoff = 2 ** (attempt - 1) * 0.01   # 0.01, 0.02, 0.04s (tiny for demo)
            print(f"      transient error (attempt {attempt}/{max_retries}): {e} "
                  f"-> backing off {backoff:.3f}s")
            time.sleep(backoff)
    raise RuntimeError(f"LLM call failed after {max_retries} retries: {last_error}")


class TransientAPIError(Exception):
    """Rate limit / 5xx / timeout — safe to retry."""


class PermanentAPIError(Exception):
    """Bad request / auth failure — do NOT retry."""


def _mock_llm_client(prompt: str, **kwargs) -> str:
    return f"[mock answer @ temp={kwargs.get('temperature')}] Based on your prompt..."


def _demo_llm_api_call() -> None:
    sub("H3. ROBUST LLM API-CALL FUNCTION — retry/backoff/config (runnable)")

    # Happy path with the default mock client.
    result = call_llm_api("Summarize the refund policy.", temperature=0.2)
    print(f"  happy path -> {result}")

    # Transient failure that recovers on retry #2 (proves the backoff loop works).
    attempts = {"n": 0}

    def flaky_client(prompt, **kwargs):
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise TransientAPIError("429 rate limited")
        return "recovered after a retry"

    result2 = call_llm_api("hello", _client=flaky_client, max_retries=3)
    print(f"  after transient retry -> {result2!r}")

    # Permanent error fails immediately (no wasted retries).
    def bad_auth_client(prompt, **kwargs):
        raise PermanentAPIError("401 invalid API key")

    try:
        call_llm_api("hello", _client=bad_auth_client)
    except PermanentAPIError as e:
        print(f"  permanent error -> failed immediately (no retries): {e}")


# INTERVIEW ANSWER (H3):
#   "I read the API key from an environment variable and fail fast if it's missing —
#    never hard-code it. I make model, temperature, max_tokens, and timeout
#    configurable parameters, always set a timeout so a call can't hang, and wrap the
#    request in retry-with-exponential-backoff — but only for TRANSIENT errors like
#    rate limits, 5xx, and timeouts. Permanent errors like a bad request or auth
#    failure I let fail immediately, because retrying them just wastes time and
#    money. The function returns the generated text and lets real failures surface as
#    exceptions instead of silently returning None."
#
# RELATED CONCEPTS: idempotency keys; circuit breakers; streaming responses; tenacity
#   library for retry decorators; structured logging of each call (Lesson 8);
#   provider fallback (try Groq, fall back to OpenAI) for resilience.


# ===================================================================================
# H4. FINE-TUNING IN DEPTH  (full vs PEFT / LoRA / QLoRA / adapters / prefix tuning)
# ===================================================================================
#
# WHAT FINE-TUNING IS, AND WHEN IT'S THE RIGHT CHOICE
#   Fine-tuning = continue training a pretrained model on your OWN data to change its
#   BEHAVIOR (style, format, tone, a specialized task). It bakes knowledge/behavior
#   into the WEIGHTS.
#   Decision vs the alternatives (say this explicitly — it's the most common trap):
#     - PROMPTING: try first. Cheapest, no training. Good for most tasks.
#     - RAG: when the need is FRESH or PRIVATE KNOWLEDGE / facts the model doesn't
#       have. RAG adds knowledge at inference time; it does NOT change behavior.
#     - FINE-TUNING: when you need a consistent BEHAVIOR/STYLE/FORMAT, a narrow task
#       done reliably, or to compress a long few-shot prompt into the weights (lower
#       per-call cost/latency). Fine-tuning is for HOW the model responds, RAG is for
#       WHAT it knows — a very common interview distinction, and they're often
#       combined (fine-tune the behavior, RAG the facts).
#
# FULL FINE-TUNING vs PARAMETER-EFFICIENT FINE-TUNING (PEFT)
#   FULL FINE-TUNING: update ALL the model's weights. Maximum flexibility, but for a
#   7B+ model it needs huge GPU memory (store weights + gradients + optimizer states
#   for every parameter), produces a full-size copy of the model per task, and risks
#   CATASTROPHIC FORGETTING of general abilities. Rarely necessary for most teams.
#   PEFT: FREEZE the pretrained weights and train only a SMALL number of new
#   parameters. ~99% fewer trainable params, far less GPU memory, tiny artifacts you
#   can swap per task, and the base model's general knowledge is preserved.
#
# THE KEY PEFT TECHNIQUES (in depth)
#   LoRA (Low-Rank Adaptation): freeze the original weight matrix W; inject a
#     trainable low-rank update ΔW = B·A, where A and B are small (rank r ≪ full
#     dimension). You train only A and B (a few million params instead of billions).
#     At inference, effective weight = W + B·A. WHY IT'S EFFICIENT: the adaptation a
#     task needs is empirically LOW-RANK, so a tiny B·A captures it — massive memory
#     savings with minimal quality loss. The trained LoRA "adapter" is a few MB you
#     can attach/detach or swap per task.
#   QLoRA (Quantized LoRA): builds on LoRA by ALSO quantizing the frozen base model
#     to 4-bit (NF4) to slash its memory footprint, then training LoRA adapters on
#     top in higher precision. This is what lets you fine-tune a 65B model on a
#     SINGLE consumer/pro GPU — base model quantized (tiny memory), adapters trained
#     normally. The headline: LoRA-level quality at a fraction of the memory.
#   ADAPTERS: insert small trainable bottleneck layers BETWEEN the frozen
#     transformer layers; train only those. (LoRA is essentially a more efficient,
#     merge-able evolution of this idea.)
#   PREFIX / PROMPT TUNING: freeze the ENTIRE model; learn a small set of continuous
#     "virtual token" embeddings prepended to the input that steer behavior. Even
#     fewer trainable params than LoRA, but generally less expressive/powerful.
#
# TRADE-OFFS SUMMARY (compute / memory / quality / deployment)
#   Full FT:  best quality ceiling, worst memory/compute/cost, full model per task,
#             forgetting risk.
#   LoRA:     near-full quality on most tasks, tiny trainable params, swappable MB-size
#             adapters, low memory. The default choice.
#   QLoRA:    LoRA's benefits + fits far bigger models on far less GPU (4-bit base).
#             Slight quality trade from quantization, usually worth it.
#   Prefix/Prompt tuning: lightest, but least powerful; good for simple steering.
#
# DATA PREP & EVALUATION (don't forget these — they're half the real work)
#   Data prep: high-quality, consistent instruction/response pairs in the model's
#     expected chat format; dedup; a held-out validation split; quality >> quantity
#     (a few hundred great examples often beats thousands of noisy ones).
#   Evaluation: measure on a held-out set with task-appropriate metrics AND check the
#     model didn't regress on general capabilities (forgetting) — compare against the
#     base model, not just against your training loss.


def _demo_lora_parameter_savings() -> None:
    sub("H4. LoRA PARAMETER SAVINGS — why PEFT is efficient (runnable arithmetic)")
    # Show the core LoRA idea numerically: a full weight matrix vs a low-rank update.
    d_in, d_out = 4096, 4096          # one weight matrix in a large model
    full_params = d_in * d_out
    print(f"  one full weight matrix ({d_in}x{d_out}): {full_params:,} trainable params")

    for rank in (8, 16, 64):
        # LoRA replaces training W with training A (d_in x r) and B (r x d_out).
        lora_params = d_in * rank + rank * d_out
        pct = 100 * lora_params / full_params
        print(f"  LoRA rank r={rank:>3}: train A({d_in}x{rank}) + B({rank}x{d_out}) "
              f"= {lora_params:,} params  ({pct:.2f}% of full)")
    print("\n  -> at rank 16 you train ~0.78% of one matrix's params — across a whole")
    print("     model that's the difference between needing a datacenter and a single")
    print("     GPU. Quality stays high because task adaptation is empirically low-rank.")


# INTERVIEW ANSWER (H4):
#   "Fine-tuning continues training a pretrained model on your own data to change its
#    BEHAVIOR — style, format, a narrow task — baking it into the weights. I reach
#    for it only after prompting and RAG: RAG is for what the model KNOWS (fresh or
#    private facts at inference time), fine-tuning is for HOW it responds, and they
#    combine. Full fine-tuning updates all weights — max flexibility but huge GPU
#    cost, a full model per task, and forgetting risk. So in practice I use PEFT:
#    LoRA freezes the base weights and trains a small low-rank update B·A — the
#    adaptation a task needs is low-rank, so you train ~1% of the params and get a
#    swappable few-MB adapter. QLoRA adds 4-bit quantization of the frozen base so
#    you can fine-tune huge models on a single GPU. Prefix/prompt tuning is even
#    lighter but less powerful. And half the work is data quality and evaluating
#    against the base model for regressions, not just watching training loss."
#
# RELATED CONCEPTS: catastrophic forgetting; instruction tuning / SFT; RLHF & DPO for
#   alignment (a later stage than task fine-tuning); merging LoRA adapters back into
#   base weights for zero inference overhead; the RAG-vs-fine-tuning decision as the
#   single most-asked framing here.


# ===================================================================================
# H5. TEMPERATURE AND top_p
# ===================================================================================
#
# WHAT THEY CONTROL (both shape the SAMPLING from the model's next-token probability
# distribution — Lesson 2's final step — they do NOT change what the model "knows"):
#   TEMPERATURE: scales the logits BEFORE softmax. temp < 1 SHARPENS the distribution
#     (high-probability tokens get even more likely -> more deterministic, focused).
#     temp > 1 FLATTENS it (low-probability tokens get more likely -> more random,
#     creative). temp = 0 is effectively greedy (always the top token — deterministic).
#   top_p (NUCLEUS SAMPLING): keep the smallest SET of tokens whose cumulative
#     probability mass ≥ p, and sample only from that set (renormalized). top_p = 0.9
#     means "consider only the most-likely tokens that together make up 90% of the
#     probability" — it dynamically cuts the long tail of unlikely tokens.
#
# HOW THEY DIFFER
#   Temperature reshapes the WHOLE distribution's sharpness; top_p TRUNCATES the tail
#   to a dynamic top set. They're often used together, but tuning BOTH aggressively
#   at once is redundant/unpredictable — common advice is to adjust one, keep the
#   other neutral (e.g., tune temperature with top_p=1.0, or tune top_p with temp=1.0).
#
# WHEN TO ADJUST THEM
#   DETERMINISTIC / FACTUAL tasks (extraction, classification, code, RAG answers you
#     want grounded and repeatable): LOW temperature (0-0.3), often top_p ~1.0. You
#     want the most-likely, consistent output — this ties directly to Lesson 5's
#     hallucination reduction (lower temp = less creative drift off the context).
#   CREATIVE tasks (brainstorming, story/marketing copy, variety): HIGHER temperature
#     (0.7-1.0+) and/or top_p ~0.9 to allow diverse, novel token choices.


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    return softmax(logits / max(temperature, 1e-6))


def apply_top_p(probs: np.ndarray, top_p: float) -> np.ndarray:
    order = np.argsort(probs)[::-1]
    cum = np.cumsum(probs[order])
    cutoff = np.searchsorted(cum, top_p) + 1
    keep = order[:cutoff]
    truncated = np.zeros_like(probs)
    truncated[keep] = probs[keep]
    return truncated / truncated.sum()


def _demo_temperature_top_p() -> None:
    sub("H5. TEMPERATURE & top_p — effect on the distribution (runnable)")
    vocab = ["refund", "return", "policy", "banana", "quantum"]
    logits = np.array([3.0, 2.5, 2.0, 0.5, 0.1])   # model prefers refund/return/policy

    print(f"  vocab: {vocab}")
    print(f"  raw logits: {logits}\n")
    for temp in (0.2, 1.0, 2.0):
        probs = apply_temperature(logits, temp)
        label = ("SHARP/deterministic" if temp < 1 else
                 "neutral" if temp == 1 else "FLAT/creative")
        print(f"  temp={temp:>3} ({label:<20}) -> {probs}")
    print("  -> low temp concentrates mass on 'refund'; high temp spreads it toward")
    print("     unlikely tokens like 'banana'/'quantum' (more random/creative).\n")

    base = apply_temperature(logits, 1.0)
    for p in (0.9, 0.5):
        truncated = apply_top_p(base, p)
        kept = [vocab[i] for i in range(len(vocab)) if truncated[i] > 0]
        print(f"  top_p={p} keeps {{{', '.join(kept)}}} (long tail cut) -> {truncated}")
    print("  -> top_p dynamically drops the improbable tail; smaller p = tighter set.")


# INTERVIEW ANSWER (H5):
#   "Both control sampling from the next-token distribution, not what the model
#    knows. Temperature scales the logits before softmax — below 1 sharpens toward
#    the most-likely tokens for deterministic output, above 1 flattens it for more
#    randomness, and 0 is effectively greedy. top_p, or nucleus sampling, keeps the
#    smallest set of tokens whose cumulative probability reaches p and samples only
#    from that, dynamically cutting the unlikely tail. Temperature reshapes the whole
#    distribution's sharpness; top_p truncates it — I usually tune one and leave the
#    other neutral. For factual/extraction/RAG tasks I use low temperature for
#    consistent, grounded output; for creative tasks I raise temperature and/or use
#    top_p around 0.9 for variety."
#
# RELATED CONCEPTS: top_k (fixed-count cousin of top_p); greedy vs beam search;
#   repetition/frequency/presence penalties; why temperature=0 still isn't perfectly
#   deterministic across hardware/batching; seeds for reproducibility.


# ===================================================================================
# GOLDEN LESSONS
# ===================================================================================
GOLDEN_LESSONS = """
  1. Reliable JSON = native structured output/function calling FIRST, then schema-in-prompt + few-shot + validate/retry.
  2. NEVER trust LLM JSON blindly — strip code fences, parse defensively, validate the schema, retry on failure.
  3. KB agent flow: query -> embed+retrieve -> grounded prompt ('answer only from context') -> generate -> cite.
  4. A retrieval-confidence gate (decline if best match is weak) stops out-of-scope hallucination before generation.
  5. Robust LLM API call: env-based key (never hard-code), configurable params, timeout, retry ONLY transient errors.
  6. Retry transient (429/5xx/timeout) with exponential backoff; fail FAST on permanent (400/401) — don't waste retries.
  7. RAG = what the model KNOWS (facts at inference); fine-tuning = HOW it responds (behavior in weights). Often combined.
  8. Try prompting -> RAG -> fine-tuning, in that order. Fine-tune for consistent style/format/task, not for facts.
  9. LoRA freezes base weights, trains a low-rank B·A update (~1% of params) — task adaptation is empirically low-rank.
 10. QLoRA = LoRA + 4-bit quantized frozen base -> fine-tune huge models on ONE GPU. Prefix/prompt tuning = lightest.
 11. Temperature scales logits (sharpen<1 / flatten>1); top_p keeps the smallest token set summing to p (cuts the tail).
 12. Low temp for factual/RAG/extraction (deterministic, grounded); high temp / top_p~0.9 for creative variety.
"""


if __name__ == "__main__":
    banner("LESSON 9 — LLM APPLICATION ENGINEERING")
    _demo_json_extraction()
    _demo_kb_agent()
    _demo_llm_api_call()
    _demo_lora_parameter_savings()
    _demo_temperature_top_p()
    banner("GOLDEN LESSONS")
    print(GOLDEN_LESSONS)
