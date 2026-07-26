# Lesson 13 — Design a Simple Knowledge-Base AI Agent

## 1. The Question

> "Design a simple AI agent that answers questions using a knowledge base. Show (explanation or pseudocode) how it accepts the user's query, retrieves relevant context, and produces an answer."

---

## 2. Theory — agent vs plain RAG

At its simplest, a knowledge-base "agent" is a **retrieve-then-generate** loop (RAG). What makes it an **agent** rather than a one-shot pipeline is the ability to **decide and act**:

- decide *whether* it needs to retrieve,
- decide *what* to search for (maybe rewrite the query),
- optionally retrieve **multiple times** or use **tools**,
- decide when it has enough to answer.

Core components of any such agent:

| Component | Role |
|---|---|
| **Query understanding** | clean/rewrite the user query |
| **Retriever** | fetch relevant context from the KB (vector search) |
| **Memory** | conversation history for follow-ups |
| **Reasoning/LLM** | decide actions, synthesize the answer |
| **Tools** (optional) | search, calculator, API calls |
| **Response builder** | grounded answer + citations |

---

## 3. Pseudocode — the simple agent

```
function answer_question(user_query, history):
    # 1. Understand / rewrite the query (resolve pronouns using history)
    query = rewrite_with_context(user_query, history)

    # 2. Decide: do we even need the knowledge base?
    if is_smalltalk(query):
        return llm(query)                      # no retrieval needed

    # 3. Retrieve relevant context
    q_vec   = embed(query)
    context = vector_store.search(q_vec, top_k=5, filter=user_permissions)

    # 4. (optional) check sufficiency; retrieve again if weak
    if max_similarity(context) < THRESHOLD:
        query   = expand_query(query)          # broaden / rephrase
        context = vector_store.search(embed(query), top_k=5)

    # 5. Augment prompt and generate a grounded answer
    prompt = build_prompt(system_rules, context, query, history)
    answer = llm(prompt, temperature=0.2)

    # 6. Return answer with sources; update memory
    history.append((user_query, answer))
    return answer, sources(context)
```

---

## 4. Hands-on — a working minimal agent

```python
from sentence_transformers import SentenceTransformer
import numpy as np
from openai import OpenAI

embedder = SentenceTransformer("all-MiniLM-L6-v2")
llm = OpenAI()

class KnowledgeBaseAgent:
    def __init__(self, documents):
        self.docs = documents
        self.vecs = embedder.encode(documents, normalize_embeddings=True)
        self.history = []                      # simple memory

    def retrieve(self, query, k=3, threshold=0.25):
        qv = embedder.encode([query], normalize_embeddings=True)[0]
        scores = self.vecs @ qv
        top = np.argsort(scores)[::-1][:k]
        # sufficiency check: only keep confidently relevant chunks
        return [(self.docs[i], float(scores[i])) for i in top
                if scores[i] >= threshold]

    def answer(self, query):
        hits = self.retrieve(query)
        if not hits:
            return "I don't have information about that in my knowledge base."

        context = "\n".join(f"[{i}] {doc}" for i, (doc, _) in enumerate(hits))
        history = "\n".join(f"Q: {q}\nA: {a}" for q, a in self.history[-3:])

        prompt = f"""Answer using ONLY the context. Cite sources like [0].
If the answer isn't in the context, say you don't know.

Conversation so far:
{history}

Context:
{context}

Question: {query}
Answer:"""
        resp = llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        answer = resp.choices[0].message.content
        self.history.append((query, answer))
        return answer


agent = KnowledgeBaseAgent([
    "Refunds are available within 30 days of purchase.",
    "Enterprise plans include 24/7 priority support.",
    "The free tier allows up to 1,000 API calls per month.",
])
print(agent.answer("How long do I have to get a refund?"))
print(agent.answer("And what about support on the enterprise plan?"))  # uses memory
```

---

## 5. Making it "more agentic" (senior bonus)

- **Tool use / function calling:** let the LLM choose to call `search_kb()`, `calculator()`, or an external API. This is the ReAct pattern (Reason → Act → Observe → repeat).
- **Query planning:** decompose a complex question into sub-questions, retrieve for each, then synthesize.
- **Self-reflection:** after answering, verify faithfulness (Lesson 10/11) and retry if weak.
- **Frameworks:** LangGraph, LlamaIndex agents, or the MCP protocol itself (like this repo) to expose KB search as a tool an LLM can call.

### ReAct sketch

```
loop:
    thought = llm("What do I need to answer this?")
    if thought.needs_tool:
        result = call_tool(thought.tool, thought.args)   # e.g., search_kb
        observations.append(result)
    else:
        return llm.final_answer(observations)
```

---

## 6. Real-time / production notes

- **Access control in retrieval:** filter by the user's permissions so the agent never surfaces docs they can't see (multi-tenant safety).
- **Grounding + citations:** always show sources; enforce "say I don't know."
- **Memory limits:** truncate/summarize history to fit the context window.
- **Latency:** cache embeddings and frequent answers; retrieve only when needed.
- **Observability:** log query → retrieved chunks → answer for debugging and eval.

---

## 7. Interview script

"At its core it's a retrieve-then-generate loop: rewrite the query using conversation memory, embed it, do a vector search filtered by the user's permissions, check the retrieved context is actually relevant, then build a grounded prompt that cites sources and is told to say 'I don't know' when the answer isn't there. It becomes a true agent when the LLM can decide whether and what to retrieve, rewrite queries, call tools, or retrieve iteratively — the ReAct reason-act-observe loop. In production I add per-tenant access control on retrieval, citations, memory summarization, caching, and logging of the full query-to-answer trace for evaluation."
