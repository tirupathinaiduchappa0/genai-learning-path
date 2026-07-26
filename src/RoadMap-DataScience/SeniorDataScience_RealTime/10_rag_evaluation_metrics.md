# Lesson 10 — RAG Evaluation: Faithfulness, Relevancy (in code)

## 1. The Question

> "How do you calculate scores for faithfulness, answer relevancy, and context relevancy in a RAG system? Explain in coding terms — how would you actually compute them?"

---

## 2. Theory — the RAG triad

RAG has two moving parts (retrieval + generation), so evaluation splits into metrics that isolate each. The **RAGAS triad**:

```
                Question
               /         \
    context relevancy   answer relevancy
             |               |
          Context ——————— Answer
               faithfulness
```

| Metric | Question it answers | Isolates |
|---|---|---|
| **Context Precision/Relevancy** | Are the retrieved chunks actually relevant to the question? | Retriever |
| **Context Recall** | Did we retrieve *all* the info needed to answer? | Retriever |
| **Faithfulness** | Is the answer supported by the retrieved context (no hallucination)? | Generator |
| **Answer Relevancy** | Does the answer actually address the question? | Generator |

Key insight: **faithfulness ≠ correctness.** An answer can be faithful to wrong context. And it can be relevant but unfaithful (made up). You need both.

---

## 3. How each metric is computed

### Faithfulness
1. Break the generated answer into atomic **claims**.
2. For each claim, ask: "Is this supported by the context?" (yes/no) — via an LLM judge or NLI model.
3. `faithfulness = supported_claims / total_claims`.

### Answer Relevancy
1. From the answer, generate N questions it *would* answer.
2. Embed those questions and the original question.
3. `answer_relevancy = mean(cosine(original_q, generated_q_i))`.
High = the answer is on-topic; low = it rambled or dodged.

### Context Precision
Of the retrieved chunks, what fraction are relevant, weighted toward the top ranks (relevant chunks should rank high).

### Context Recall
Break the ground-truth answer into claims; check what fraction can be attributed to the retrieved context.

---

## 4. Hands-on

### 4.1 Using RAGAS (the standard framework)

```python
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness, answer_relevancy,
    context_precision, context_recall,
)

data = {
    "question":     ["What is the API rate limit?"],
    "answer":       ["The API allows 100 requests per minute per key."],
    "contexts":     [["The API rate limit is 100 requests per minute per key."]],
    "ground_truth": ["100 requests per minute per key."],
}

result = evaluate(
    Dataset.from_dict(data),
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
)
print(result)   # {'faithfulness': 1.0, 'answer_relevancy': 0.98, ...}
```

### 4.2 Faithfulness from scratch (LLM-as-judge) — shows you understand it

```python
from openai import OpenAI
import json
client = OpenAI()

def extract_claims(answer):
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content":
            f"Break this answer into a JSON list of atomic factual claims.\n\n{answer}"}],
        response_format={"type": "json_object"},
    )
    return json.loads(r.choices[0].message.content)["claims"]

def claim_supported(claim, context):
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content":
            f"Context:\n{context}\n\nClaim: {claim}\n"
            f"Is the claim fully supported by the context? Answer only YES or NO."}],
        temperature=0,
    )
    return r.choices[0].message.content.strip().upper().startswith("YES")

def faithfulness_score(answer, context):
    claims = extract_claims(answer)
    if not claims:
        return 1.0
    supported = sum(claim_supported(c, context) for c in claims)
    return supported / len(claims)
```

### 4.3 Answer relevancy from scratch (embedding-based)

```python
import numpy as np
from sentence_transformers import SentenceTransformer
emb = SentenceTransformer("all-MiniLM-L6-v2")

def answer_relevancy(question, generated_questions):
    # generated_questions: N questions an LLM derived from the answer
    qv = emb.encode([question], normalize_embeddings=True)[0]
    gv = emb.encode(generated_questions, normalize_embeddings=True)
    return float(np.mean(gv @ qv))
```

---

## 5. Other evaluation approaches

- **Retrieval metrics (no LLM judge):** Hit Rate, MRR, nDCG@k — classic IR metrics when you have labeled relevant docs.
- **Reference-based generation:** BLEU/ROUGE (weak for open-ended), **BERTScore** (semantic), or LLM-judged correctness vs a gold answer.
- **LLM-as-judge pairwise:** ask a strong model which of two answers is better (used in arena-style eval).
- **Human eval:** still the gold standard for a final sanity check.

---

## 6. Real-time / production concerns

- **Judge cost & bias:** LLM-as-judge is expensive and can be biased (position bias, self-preference). Use a strong judge, randomize order, spot-check with humans.
- **Put it in CI:** run the eval set on every prompt/index/model change; block regressions (this is "LLM ops"/eval-driven development).
- **Online signals:** thumbs up/down, edit rate, deflection rate as production proxies.
- **Golden dataset:** curate and version a representative Q/context/ground-truth set; it's your regression test.
- **Component isolation:** low context recall → fix the retriever/chunking; high recall but low faithfulness → fix the prompt/generation.

---

## 7. Interview script

"I evaluate the two halves separately. For retrieval: context precision and recall — are the right chunks retrieved and are they enough. For generation: faithfulness — every claim in the answer must be grounded in the context — and answer relevancy — does it actually address the question. Faithfulness is computed by decomposing the answer into atomic claims and checking each against the context with an LLM or NLI judge, then taking the supported ratio. Answer relevancy reverse-generates questions from the answer and measures embedding similarity to the original question. I automate this with RAGAS over a versioned golden dataset in CI so I catch regressions, and I complement it with online thumbs-up/down signals. Faithfulness and correctness are different — an answer can be faithful to wrong context — so I track both."
