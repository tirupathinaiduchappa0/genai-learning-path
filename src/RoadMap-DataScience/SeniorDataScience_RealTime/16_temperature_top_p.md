# Lesson 16 — Temperature and top_p

## 1. The Question

> "What are temperature and top_p in OpenAI (and similar) models? How do they differ, and when would you adjust them?"

---

## 2. Theory — how an LLM picks the next token

At each step the model produces **logits** over the whole vocabulary, turned into a probability distribution via **softmax**. Decoding parameters reshape *how* the next token is sampled from that distribution.

### Temperature — reshape the whole distribution

Temperature `T` divides the logits before softmax:

```
p_i = softmax(logit_i / T)
```

- **T → 0:** distribution becomes peaky → always pick the highest-probability token → **deterministic, focused** (greedy).
- **T = 1:** the model's natural distribution.
- **T > 1:** distribution flattens → low-probability tokens get more chance → **more random, creative, diverse** (and more error-prone).

Intuition: temperature is a "creativity/randomness dial" applied to the *entire* vocabulary.

### top_p (nucleus sampling) — truncate the tail

Instead of scaling, top_p **restricts the candidate set**: sort tokens by probability, keep the smallest set whose cumulative probability ≥ `p`, sample only from that "nucleus."

- **top_p = 1.0:** consider all tokens.
- **top_p = 0.9:** consider only the top tokens making up 90% of the mass — cuts off the unlikely long tail.
- **top_p = 0.1:** very restrictive, near-deterministic.

Intuition: top_p is a "how much of the tail do I allow" dial. It adapts to context — when the model is confident, the nucleus is tiny; when uncertain, it's larger.

### top_k (bonus)
Keep only the `k` highest-probability tokens (fixed count). top_p is usually preferred because it adapts to the distribution's shape.

---

## 3. Temperature vs top_p — the key difference

| | Temperature | top_p |
|---|---|---|
| Mechanism | Rescales all logits | Truncates to a cumulative-probability set |
| Effect | Changes *relative* odds everywhere | Removes the unlikely tail entirely |
| Feel | Global randomness dial | Adaptive vocabulary cutoff |

**Best practice: tune ONE, not both.** Adjusting both interacts unpredictably. Most people fix `top_p=1.0` and adjust temperature (or fix temperature and adjust top_p).

---

## 4. Hands-on — see the effect

```python
from openai import OpenAI
client = OpenAI()

def generate(prompt, temperature=1.0, top_p=1.0):
    return client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
    ).choices[0].message.content

prompt = "Write a tagline for a coffee shop."

# Deterministic / factual — same-ish every run
print(generate(prompt, temperature=0.0))

# Balanced
print(generate(prompt, temperature=0.7))

# Creative / diverse — varied, surprising, riskier
print(generate(prompt, temperature=1.3))
```

Run each a few times: `T=0` barely changes; `T=1.3` produces wild variety.

---

## 5. When to adjust — task-based settings

| Task | Temperature | Why |
|---|---|---|
| Data extraction / parsing (Lesson 12) | 0 – 0.2 | need deterministic, repeatable output |
| Code generation | 0 – 0.3 | correctness over creativity |
| Factual Q&A / RAG | 0.1 – 0.3 | grounded, reduces hallucination (Lesson 11) |
| Summarization | 0.3 – 0.5 | faithful but readable |
| Chat assistant | 0.6 – 0.8 | natural, varied but coherent |
| Brainstorming / marketing copy | 0.9 – 1.2 | diversity and novelty |
| Creative fiction | 1.0 – 1.5 | maximum variety |

Rule: **low temperature when there's a right answer; high temperature when you want options.**

---

## 6. Real-time / production notes

- **Reproducibility:** for tests/evals set `temperature=0` (plus a `seed` where supported) so outputs are stable.
- **Determinism caveat:** even at `T=0`, outputs aren't 100% guaranteed identical across time due to backend nondeterminism — use `seed` and pin the model version.
- **Hallucination link:** higher temperature raises hallucination risk; keep it low for factual/grounded tasks.
- **Diversity via sampling:** for self-consistency (Lesson 11) you deliberately raise temperature to get varied samples, then vote.
- **Don't fight a bad prompt with temperature** — if output is wrong, fix the prompt/context first.

---

## 7. Interview script

"Both control sampling from the model's next-token distribution, but differently. Temperature rescales all the logits before softmax — low makes it peaky and deterministic, high flattens it so unlikely tokens get picked, which reads as creativity. top_p, or nucleus sampling, instead truncates: it keeps the smallest set of tokens whose cumulative probability hits p and samples only from that, adapting to how confident the model is. Best practice is to tune one, not both. I use low temperature — 0 to 0.3 — for extraction, code, and factual RAG where I want deterministic grounded output, and higher — around 1 — for brainstorming or creative copy. For evals I pin temperature 0 and a seed for reproducibility."
