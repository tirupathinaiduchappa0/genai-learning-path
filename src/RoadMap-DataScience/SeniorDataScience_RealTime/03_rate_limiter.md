# Lesson 03 — Implement a Rate Limiter

## 1. The Question

> "Implement a rate limiter: allow 100 requests/minute per user, reject beyond the limit, auto-reset after one minute. Discuss algorithms and concurrency."

---

## 2. Theory — the four classic algorithms

### 2.1 Fixed Window Counter
Count requests in the current minute bucket; reset at the boundary.
- **Pro:** trivial, O(1) memory per user.
- **Con:** burst at boundary — a user can send 100 at 00:59 and 100 at 01:00 = 200 in 2 seconds.

### 2.2 Sliding Window Log
Keep timestamps of each request; drop those older than the window; count what's left.
- **Pro:** exact, no boundary burst.
- **Con:** O(requests) memory per user.

### 2.3 Sliding Window Counter
Weighted blend of current + previous fixed windows. Approximation of the log with O(1) memory. What Cloudflare uses.

### 2.4 Token Bucket
A bucket refills at a steady rate (e.g., 100 tokens/60s). Each request consumes a token; empty bucket → reject. Allows controlled **bursts** up to bucket capacity.
- Used by AWS API Gateway, NGINX, Stripe.

### 2.5 Leaky Bucket
Requests queue and drain at a fixed rate — smooths bursts into a constant outflow.

---

## 3. Hands-on code

### 3.1 Sliding Window Log (exact, matches the question)

```python
import time
from collections import deque, defaultdict
import threading

class SlidingWindowLimiter:
    def __init__(self, limit=100, window_seconds=60):
        self.limit = limit
        self.window = window_seconds
        self.hits = defaultdict(deque)
        self.lock = threading.Lock()

    def allow(self, user_id: str) -> bool:
        now = time.time()
        with self.lock:                      # thread-safe
            q = self.hits[user_id]
            cutoff = now - self.window
            while q and q[0] <= cutoff:       # evict expired timestamps
                q.popleft()
            if len(q) < self.limit:
                q.append(now)
                return True
            return False
```

### 3.2 Token Bucket (allows bursts)

```python
import time, threading

class TokenBucket:
    def __init__(self, capacity=100, refill_per_sec=100/60):
        self.capacity = capacity
        self.tokens = capacity
        self.refill = refill_per_sec
        self.last = time.monotonic()
        self.lock = threading.Lock()

    def allow(self, cost=1) -> bool:
        with self.lock:
            now = time.monotonic()
            # refill based on elapsed time
            self.tokens = min(self.capacity,
                              self.tokens + (now - self.last) * self.refill)
            self.last = now
            if self.tokens >= cost:
                self.tokens -= cost
                return True
            return False
```

### 3.3 Distributed rate limiter with Redis (production answer)

In-memory limiters break when you run **multiple server instances** (each has its own counter). Centralize in Redis:

```python
import redis, time

r = redis.Redis()

def allow(user_id: str, limit=100, window=60) -> bool:
    """Sliding window using a Redis sorted set of timestamps."""
    key = f"rl:{user_id}"
    now = time.time()
    pipe = r.pipeline()
    pipe.zremrangebyscore(key, 0, now - window)  # drop expired
    pipe.zadd(key, {str(now): now})              # record this request
    pipe.zcard(key)                              # count in window
    pipe.expire(key, window)                     # auto-reset / cleanup
    _, _, count, _ = pipe.execute()
    return count <= limit
```

The `expire` gives automatic reset and prevents memory leaks for idle users. The pipeline makes it near-atomic; for strict atomicity use a Lua script.

### 3.4 As a FastAPI/Starlette middleware (ties into this repo's stack)

```python
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, limiter):
        super().__init__(app)
        self.limiter = limiter

    async def dispatch(self, request, call_next):
        user = request.headers.get("X-User-Id", request.client.host)
        if not self.limiter.allow(user):
            return JSONResponse(
                {"error": "rate limit exceeded"},
                status_code=429,
                headers={"Retry-After": "60"},
            )
        return await call_next(request)
```

Returning **HTTP 429** with a `Retry-After` header is the correct, standards-compliant behavior.

---

## 4. Real-time / production considerations

- **Per-user key:** API key, user ID, or IP. IP alone is weak (NAT, proxies).
- **Distributed:** always Redis (or a gateway like Kong/NGINX/AWS API GW) once you scale past one instance.
- **Response contract:** 429 + `Retry-After` + `X-RateLimit-Remaining` headers so clients back off gracefully.
- **Fail-open vs fail-closed:** if Redis is down, do you allow or block? Usually fail-open for availability, but log/alert.
- **Tiered limits:** free vs paid users get different limits — same algorithm, different config.

---

## 5. Choosing the algorithm

```
Need exactness, low volume        → sliding window log
Need O(1) memory, huge scale      → sliding window counter
Want to allow controlled bursts   → token bucket
Want perfectly smooth output rate → leaky bucket
Just need simple + acceptable     → fixed window
```

---

## 6. Interview script

"I'd use a sliding window to avoid the fixed-window boundary burst, or a token bucket if the product wants to allow short bursts. In-memory with a lock works for a single instance, but the moment you have multiple servers you must centralize the counter — I'd put it in Redis using a sorted set of timestamps with an EXPIRE for automatic reset. I return 429 with Retry-After so clients back off. I'd also decide fail-open vs fail-closed if Redis is unavailable."
