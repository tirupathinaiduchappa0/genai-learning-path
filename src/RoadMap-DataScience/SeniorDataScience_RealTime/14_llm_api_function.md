# Lesson 14 — Python Function to Call an LLM API

## 1. The Question

> "Write a Python function that accepts a user's query and sends it to an LLM API (OpenAI, Azure OpenAI, or equivalent). Accept the prompt as input; handle the response."

Tests: API integration hygiene — error handling, retries, secrets, configurability, streaming.

---

## 2. Theory — what a *production* wrapper needs

A one-liner works in a notebook. In an interview, show you know the difference:

- **Secrets from environment**, never hardcoded.
- **Configurable** model, temperature, max tokens, system prompt.
- **Error handling** for rate limits, timeouts, transient 5xx.
- **Retries with exponential backoff** on transient failures.
- **Timeouts** so a hung request doesn't block forever.
- **Structured return** (text, or text + usage/metadata).
- **Async** version if you're serving many concurrent requests (this repo's pattern).
- **Streaming** for responsive UIs.

---

## 3. Hands-on

### 3.1 Clean synchronous version

```python
import os
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])   # secret from env

def ask_llm(
    prompt: str,
    *,
    system: str = "You are a helpful assistant.",
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
    max_tokens: int = 1000,
) -> str:
    """Send a prompt to the LLM and return the text response."""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=30,
    )
    return resp.choices[0].message.content
```

### 3.2 Production version: retries, error handling, usage

```python
import os, time, logging
from openai import OpenAI, APITimeoutError, RateLimitError, APIError

log = logging.getLogger(__name__)
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def ask_llm(prompt, *, model="gpt-4o-mini", temperature=0.2,
            max_tokens=1000, max_retries=3):
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=30,
            )
            return {
                "text": resp.choices[0].message.content,
                "tokens": resp.usage.total_tokens,
                "model": resp.model,
            }
        except (RateLimitError, APITimeoutError, APIError) as e:
            wait = 2 ** attempt                       # exponential backoff
            log.warning("LLM call failed (%s), retry in %ss", e, wait)
            time.sleep(wait)
    raise RuntimeError("LLM call failed after retries")
```

### 3.3 Async version (fits this repo's httpx/async stack)

```python
from openai import AsyncOpenAI
client = AsyncOpenAI()

async def ask_llm_async(prompt: str, model="gpt-4o-mini") -> str:
    resp = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content
```

### 3.4 Streaming (responsive UI)

```python
def ask_llm_stream(prompt, model="gpt-4o-mini"):
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
```

### 3.5 Azure OpenAI variant (just different client config)

```python
from openai import AzureOpenAI
client = AzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_KEY"],
    api_version="2024-06-01",
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
)
# then use your *deployment name* as `model=`
```

---

## 4. Real-time / production notes

- **Secrets:** env vars / secret manager (AWS Secrets Manager, KMS — like this repo). Never commit keys.
- **Retries:** only on *transient* errors (429, timeout, 5xx); don't retry a 400 bad request.
- **Backoff + jitter:** avoid thundering-herd on rate limits.
- **Timeouts:** always set one; pair with circuit breakers for downstream protection.
- **Cost control:** cap `max_tokens`, log `usage`, cache identical prompts, pick the cheapest capable model.
- **Observability:** log latency, tokens, model, and errors; trace requests.
- **Idempotency & PII:** don't log raw sensitive prompts; scrub before logging.

---

## 5. Interview script

"The function takes the prompt plus optional system/model/temperature/max_tokens, pulls the API key from the environment, and calls chat.completions. Beyond the happy path I add a timeout, retries with exponential backoff on transient errors like 429s and 5xx — but not on 400s — and I return usage so I can track cost. For a service I'd use the async client to handle concurrency, and streaming for a responsive UI. In production the key lives in a secret manager, I cap max_tokens, cache identical prompts, scrub PII from logs, and emit latency/token metrics."
