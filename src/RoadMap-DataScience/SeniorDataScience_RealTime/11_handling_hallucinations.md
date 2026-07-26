# Lesson 11 — Handling Hallucinations in LLM Responses

## 1. The Question

> "How do you detect and reduce hallucinations in LLM responses? Cover grounding, prompting, verification, and guardrails."

---

## 2. Theory — what is a hallucination and why it happens

A **hallucination** is output that is fluent and confident but **factually wrong or unsupported**. Types:

- **Factual** — states something false ("the CEO is X" when it isn't).
- **Faithfulness / contextual** — contradicts or invents beyond the provided context (the RAG-specific kind).
- **Citation** — fabricates sources, URLs, references.

**Why it happens:** an LLM is a next-token probability model, not a database. It optimizes for *plausible* continuations, not *true* ones. When it lacks knowledge, the highest-probability continuation is still a fluent guess. Contributing factors: knowledge gaps, stale training data, ambiguous prompts, high temperature, and long-context "lost in the middle."

---

## 3. Mitigation ladder (defense in depth)

No single fix; layer them.

### Layer 1 — Grounding (most effective)
Use **RAG** (Lesson 08) to supply real context, so the model paraphrases retrieved facts instead of inventing them. Quality of retrieval directly bounds faithfulness.

### Layer 2 — Prompt engineering
- **"Answer ONLY from the context. If it's not there, say 'I don't know.'"** — the single highest-value instruction.
- Ask for **citations** to specific chunks — forces the model to point at evidence.
- Request **step-by-step reasoning** for complex questions.
- Give the model an explicit **escape hatch** so "I don't know" is an acceptable answer (removes the pressure to fabricate).

### Layer 3 — Decoding controls
- **Low temperature** (0–0.3) for factual tasks → less random, more grounded.
- Constrained / structured output to limit free-form drift.

### Layer 4 — Verification (post-generation)
- **Faithfulness check** (Lesson 10): decompose the answer into claims, verify each against the source; drop/flag unsupported claims.
- **Self-consistency:** sample multiple answers; if they disagree, confidence is low.
- **Chain-of-verification (CoVe):** the model drafts, generates verification questions, answers them against context, then revises.
- **NLI entailment:** check that context *entails* each answer sentence.

### Layer 5 — Guardrails & product design
- Validate outputs (schema, allowed values, regex, moderation) — e.g., NeMo Guardrails, Guardrails AI.
- Show **sources/citations** in the UI so users can verify.
- Human-in-the-loop for high-stakes domains (medical, legal, finance).
- Abstain / route to human when confidence is low.

---

## 4. Hands-on

### Grounded, escape-hatch prompt

```python
SYSTEM = """You are a support assistant. Follow these rules strictly:
1. Answer ONLY using the provided context.
2. If the answer is not in the context, reply exactly: "I don't have that information."
3. Cite the source number [n] after each claim.
Never use outside knowledge."""

def grounded_answer(query, chunks):
    context = "\n".join(f"[{i}] {c}" for i, c in enumerate(chunks))
    return client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
        temperature=0.1,
    ).choices[0].message.content
```

### Post-hoc faithfulness gate

```python
def safe_answer(query, chunks):
    answer = grounded_answer(query, chunks)
    score = faithfulness_score(answer, "\n".join(chunks))   # from Lesson 10
    if score < 0.8:
        return "I'm not confident enough to answer that reliably."
    return answer
```

### Self-consistency check

```python
def self_consistency(query, chunks, n=3):
    answers = [grounded_answer(query, chunks) for _ in range(n)]  # temp>0
    # if answers diverge semantically, flag low confidence
    return answers
```

---

## 5. Detection in practice

- **Automated:** faithfulness/NLI scoring, uncertainty via self-consistency, citation validation (does cited chunk actually contain the claim?).
- **Reference-based:** compare to gold answers on an eval set.
- **Online:** user thumbs-down, "report incorrect," correction/edit rate.
- **Log everything** so you can audit and reproduce hallucinated outputs.

---

## 6. Real-time / production notes

- **You can't fully eliminate it — you manage it.** Set expectations accordingly.
- **Match rigor to stakes:** a brainstorming tool tolerates more than a medical assistant.
- **Retrieval quality is the ceiling:** garbage context → confident garbage answers. Fix retrieval first.
- **Latency/cost trade-off:** verification passes and self-consistency multiply cost; reserve for high-stakes paths.
- **Prompt injection from retrieved docs** can *induce* hallucination/misbehavior — sanitize untrusted context (Lesson 18).

---

## 7. Interview script

"Hallucination happens because an LLM predicts plausible tokens, not verified facts. I manage it in layers. First, grounding with RAG so it paraphrases real retrieved context. Second, prompting: 'answer only from context, otherwise say I don't know,' plus required citations and an explicit escape hatch so it isn't pressured to guess. Third, low temperature for factual tasks. Fourth, post-generation verification — decompose into claims and check each against the source, and use self-consistency or chain-of-verification for hard questions. Fifth, product guardrails: show sources, validate outputs, and route low-confidence or high-stakes cases to a human. You can't eliminate hallucination, but grounding plus a faithfulness gate plus citations gets you most of the way, and retrieval quality sets the ceiling."
