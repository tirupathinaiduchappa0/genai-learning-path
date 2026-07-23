# Insurance Support Agent — LangGraph + Streamlit

A customer support agent for an insurance company that handles three request
types end to end: checking policy status, filing a vehicle claim, and
requesting a policy cancellation.

## Requirements demonstrated
- LangGraph state management (`state.py`)
- Conditional routing based on intent and slot completeness (`graph.py`)
- Deterministic tool calling (`tools.py`)
- Multi-turn conversations with slot-filling across turns (checkpointed state)
- Error handling (invalid/lapsed policy numbers, mock API failures)
- Human-in-the-loop approval before policy cancellation (`interrupt()` / `Command(resume=...)`)
- Production design decisions — see comments in `nodes.py`

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Create a `.env` file in this folder (or the working directory you run from)
with:

```
GROQ_API_KEY=your_key_here
```

Get a free Groq API key at https://console.groq.com/keys.

## Run

```bash
streamlit run app.py
```

Opens at http://localhost:8501.

## Try it

- "What's the status of policy POL-1001?"
- "I want to report a claim for POL-1002, my car was hit on 2026-07-15 near
  MG Road, a side mirror was damaged"
- "Please cancel policy POL-1003, I sold the car" → approve/reject via the
  on-screen buttons to see the human-in-the-loop gate in action

Mock policies: `POL-1001` (active), `POL-1002` (active, premium due),
`POL-1003` (lapsed). All data is in-memory (`mock_data.py`) — no real
backend or database required.

## Architecture notes

The LLM is used only to understand the user's request (intent + slot
extraction via structured output). Tool dispatch and the final confirmation
message are deterministic Python, not LLM-generated — this keeps behavior
predictable and auditable, and reduces LLM calls in the critical path.
Policy cancellation is the only action gated behind human approval, since
it's the one action with real, hard-to-reverse business consequences.
