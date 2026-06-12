# PwC R2 — Architect Round Battle Plan

**Role:** Generative AI Software Engineer — Senior Associate
**When:** Jun 4, 2026, 3:00–3:45 PM IST (45 min, MS Teams)
**Interviewer:** Naga Venkata Lakshmi Narayana Posimsetti — **Sr. AI Manager (Architect)**
**R1 hint:** architect-level → agentic flowcharts, block diagrams, internal framework flow.

> No GenAI tools allowed in the interview. This is about YOUR thinking, communication, adaptability. That's preparable. Breathe.

---

## TONIGHT — prioritized, time-boxed (don't cram everything)

| Order | Read | Why | Time |
|-------|------|-----|------|
| 1 | **Lesson 22 + cheat card** (DocSage walkthrough) | They WILL ask "walk me through your project". Own this cold. | 30m |
| 2 | **Lesson 18** (Agentic AI architecture) | The core of this round. 5 layers, multi-agent patterns, components, diagrams. | 40m |
| 3 | **Lesson 21 §5** (Advanced RAG architectures) | CRAG/Self-RAG/Agentic/GraphRAG — name-drop + connect to DocSage. | 20m |
| 4 | **Lesson 17** (RAG evaluation) | The gap from R1. Be ready for "how do you evaluate?" | 20m |
| 5 | **Lesson 20** (skim) | Grounding + production confidence soundbites. | 15m |

**Then STOP and sleep.** Tired brain ≠ sharp architect. 7 hours beats 1 more lesson.

---

## PREDICTED 45-MIN STRUCTURE (so nothing surprises you)

1. **Intro / rapport** (~5m) — your background, current work.
2. **Project deep dive** (~12m) — DocSage architecture, draw the flow. *(Lesson 22)*
3. **Agentic internals** (~10m) — "how does an agent framework work inside?" *(Lesson 18)*
4. **A design scenario** (~10m) — "design an agentic system for X." *(framework below)*
5. **Fundamentals / trade-offs** (~5m) — RAG eval, scaling, failure modes.
6. **Your questions** (~3m) — have 2 ready (below).

---

## THE ARCHITECT MINDSET SHIFT (this is what he's scoring)

He is NOT testing if you can code. He's testing:
- **Structured thinking** — do you decompose a problem cleanly?
- **Trade-offs** — every choice has a "why" + a cost. *"Chose X because Y, trade-off Z, at scale W."*
- **Failure modes** — what breaks, how you detect and recover.
- **Production realism** — observability, cost, latency, security, scale.
- **Communication** — can you draw a diagram and narrate it calmly?

> Junior says WHAT. Senior says WHY + TRADE-OFF + AT SCALE. Live in that sentence shape all interview.

---

## 5 DIAGRAMS TO HAVE READY TO DRAW (practice each in <60s)

1. **DocSage graph** — START→agent→tools→grade→generate→validate→END + 2 self-correction loops. *(cheat card)*
2. **5-layer agentic stack** — Perception → Reasoning/Planning → Memory → Action/Tools → Learning. *(Lesson 18)*
3. **Orchestrator–Worker multi-agent** — one supervisor routing to specialist agents. *(Lesson 18)*
4. **RAG pipeline** — ingest (load→chunk→embed→store) + query (embed→retrieve→rerank→generate). *(Lesson 21)*
5. **Production agentic system** — UI → API/gateway → orchestrator(LangGraph) → tools/LLM gateway → vector DB + memory → observability + guardrails. *(Lesson 18 §production)*

> On Teams: ask "may I share my screen / use a whiteboard?" or describe spatially: "at the top… below that… feeding into…". Drawing while talking = instant senior signal.

---

## THE DESIGN-SCENARIO FRAMEWORK (for "design an agentic system for X")

Never freeze. Walk this 6-step script out loud:

1. **Clarify** — "Let me confirm scope: inputs, users, scale, latency, accuracy needs?"
2. **High-level blocks** — UI → orchestrator → agents/tools → data/memory → observability.
3. **Agentic flow** — how the agent decides, which tools, the reasoning loop, termination.
4. **Data/RAG layer** — ingestion, vector store, retrieval + reranking, grounding.
5. **Failure + quality** — hallucination guards, retries, eval (RAGAS), human-in-the-loop.
6. **Production** — scaling, cost, latency, security/governance, monitoring (LangSmith).

> Start broad (blocks), then drill where he probes. Always end with "trade-offs and what I'd watch in production."

---

## LIKELY QUESTIONS → ONE-LINE ANCHORS

- **"Walk me through your project"** → Lesson 22 spine: agent→tools→grade→generate→validate + self-correction.
- **"How does an agent framework work internally?"** → state + nodes + conditional edges + tool-calling loop + termination; LangGraph StateGraph, ReAct reasoning.
- **"Difference: workflow vs agent?"** → workflow = fixed path; agent = LLM decides next step dynamically (tool selection + loop).
- **"Multi-agent patterns?"** → Orchestrator-Worker, Hierarchical, Peer-to-Peer, Hub-and-Spoke *(Lesson 18)*.
- **"How do you evaluate a RAG/agent system?"** → retrieval (Recall@K, MRR) + generation (RAGAS: faithfulness, answer relevance) + LLM-as-Judge + human eval *(Lesson 17/21)*.
- **"Prevent hallucination?"** → grounding + grade docs + strict prompt + post-gen faithfulness check + retrieval gate *(Lesson 20)*.
- **"Scale to millions of docs / many users?"** → Qdrant/Pinecone + namespaces, async ingestion + queue, caching, autoscaling, observability.
- **"Memory in agents?"** → short-term (state/messages), long-term (vector store), checkpointer (MemorySaver + thread_id).
- **"Vector DB in prod?"** → Qdrant/Pinecone, NOT FAISS (in-memory); FAISS was for the portfolio.
- **"Cost/latency control?"** → smaller model per task (factory), caching, batching, streaming, reranker only when borderline.

---

## YOUR 2 QUESTIONS TO ASK HIM (shows architect-level curiosity)

1. "What does the agentic AI stack look like at PwC AC — are you standardizing on LangGraph/LangChain, or a custom orchestration layer?"
2. "For client GenAI engagements, what's the biggest architecture challenge your team is solving right now — evaluation, governance, or scale?"

---

## VIDEO-ROUND SOFT SKILLS (easy points)

- Camera on, eye-level, decent light, quiet room, **stable internet** (test the Teams link early).
- **Pause before answering** big questions — 2s of thinking beats rushing. "Good question, let me structure that…"
- Think out loud: narrate your reasoning, don't go silent.
- If you don't know something: "I haven't used that directly, but my approach would be…" — never bluff a fact.
- Slow down. Nervousness = fast talking. Breathe between points.

---

## MORNING-OF / 30-MIN-BEFORE ROUTINE

- Re-read this card + Lesson 22 cheat card only. No new material.
- Practice drawing the DocSage spine + 5-layer stack once each.
- Join 5 min early. Meeting ID **258134749974704**, passcode **mU2tk6nt**.
- Water nearby, resume open, a pen + paper for diagrams.

---

## IF YOU BLANK OUT (the reset)

Breathe → "Let me structure that." → start with **high-level blocks**, then drill down.
You know this material — you BUILT DocSage and wrote these lessons. Trust it.

> R1 went well. He already expects a strong candidate. Walk in calm, structured, and curious — you've got this.
