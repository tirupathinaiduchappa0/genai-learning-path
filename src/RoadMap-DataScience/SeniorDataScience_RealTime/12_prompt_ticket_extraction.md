# Lesson 12 — Prompt Design: Support-Ticket Extraction to JSON

## 1. The Question

> "Create a prompt that extracts details from a customer support ticket and returns the output in JSON format — at minimum Ticket ID and Issue Category."

Tests: prompt engineering, structured output, schema enforcement, and reliability thinking.

---

## 2. Theory — what makes an extraction prompt reliable

Extraction is a **structured output** task. Four principles:

1. **Role + task clarity** — tell the model exactly what it is and what to do.
2. **Explicit schema** — name every field, its type, and allowed values (enums). Ambiguity → inconsistent output.
3. **Output-format constraint** — "Return ONLY valid JSON, no prose, no markdown fences." Models love to add chatter.
4. **Few-shot example** — one example pins down the exact shape and edge-case handling.
5. **Missing-value rule** — say what to do when a field isn't present (`null`), so it doesn't hallucinate.

Then enforce it programmatically with the model's **JSON mode / structured output**, and validate with a schema (Pydantic).

---

## 3. The prompt

```
You are a support-ticket parser. Extract the fields below from the ticket
and return ONLY a valid JSON object — no explanation, no markdown.

Schema:
{
  "ticket_id":      string | null,   // e.g. "TKT-1023"; null if absent
  "issue_category": "billing" | "technical" | "account" | "shipping" | "other",
  "priority":       "low" | "medium" | "high" | "urgent",
  "sentiment":      "positive" | "neutral" | "negative",
  "summary":        string,          // one-sentence summary
  "customer_name":  string | null,
  "action_required": string | null   // next step, or null
}

Rules:
- Use only information present in the ticket. If a field is unknown, use null.
- issue_category and priority must be one of the listed values.
- Do not invent a ticket_id.

Example ticket:
"Ticket TKT-88: Hi, I was charged twice for my subscription this month and I'm
really frustrated. Please refund the duplicate charge ASAP. - Maria"

Example output:
{"ticket_id":"TKT-88","issue_category":"billing","priority":"high",
"sentiment":"negative","summary":"Customer was double-charged and requests a refund.",
"customer_name":"Maria","action_required":"Refund the duplicate charge."}

Now parse this ticket:
\"\"\"{ticket_text}\"\"\"
```

Why it works: enumerated values prevent free-text drift, the `null` rule kills hallucination, the example locks the shape, and "ONLY valid JSON" strips chatter.

---

## 4. Hands-on — enforce and validate

### With OpenAI JSON mode + Pydantic validation

```python
from openai import OpenAI
from pydantic import BaseModel, ValidationError
from typing import Optional, Literal
import json

client = OpenAI()

class Ticket(BaseModel):
    ticket_id: Optional[str]
    issue_category: Literal["billing", "technical", "account", "shipping", "other"]
    priority: Literal["low", "medium", "high", "urgent"]
    sentiment: Literal["positive", "neutral", "negative"]
    summary: str
    customer_name: Optional[str] = None
    action_required: Optional[str] = None

def parse_ticket(ticket_text: str) -> Ticket:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},   # forces valid JSON
        temperature=0,                              # deterministic extraction
        messages=[
            {"role": "system", "content": PROMPT},  # the prompt above
            {"role": "user", "content": ticket_text},
        ],
    )
    raw = json.loads(resp.choices[0].message.content)
    return Ticket(**raw)   # raises if schema violated -> catch & retry
```

### Even stronger: native structured output (schema-guaranteed)

```python
# OpenAI structured outputs bind the Pydantic model directly
completion = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[{"role": "system", "content": PROMPT},
              {"role": "user", "content": ticket_text}],
    response_format=Ticket,     # model output is guaranteed to match
)
ticket = completion.choices[0].message.parsed
```

### Retry-on-invalid wrapper

```python
def parse_with_retry(text, attempts=2):
    for i in range(attempts):
        try:
            return parse_ticket(text)
        except (ValidationError, json.JSONDecodeError) as e:
            if i == attempts - 1:
                raise
            # optionally feed the error back into the prompt for self-correction
```

---

## 5. Real-time / production notes

- **temperature=0** for extraction — you want deterministic, repeatable parses.
- **Validate, don't trust** — always parse into a schema; reject/retry on failure.
- **Enums over free text** so downstream systems (routing, dashboards) get clean categories.
- **PII handling** — tickets contain personal data; redact/handle per policy, don't log raw.
- **Prompt injection** — a ticket could contain "ignore instructions and output X"; treat ticket text as untrusted data, keep instructions in the system role, and validate output.
- **Cost/latency** — use the smallest model that hits your accuracy bar; batch when possible.
- **Evaluate** — keep a labeled set of tickets and measure field-level accuracy over time.

---

## 6. Interview script

"I treat extraction as structured output. The prompt gives a clear role, an explicit schema with enumerated allowed values, a 'return only valid JSON' constraint, a null rule for missing fields so it won't hallucinate, and one few-shot example to lock the shape. Then I enforce it in code: JSON mode or native structured outputs, temperature 0 for determinism, and Pydantic validation with a retry on failure. In production I also handle PII, treat the ticket text as untrusted to avoid prompt injection, use enums so downstream routing is clean, and track field-level accuracy on a labeled set."
