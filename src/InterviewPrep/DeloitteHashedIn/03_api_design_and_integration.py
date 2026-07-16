"""
===================================================================================
PHASE 3 — API DESIGN & INTEGRATION (HashedIn by Deloitte — Lead Python/GenAI)
===================================================================================

WHY THIS IS PHASE 3:
    The JD explicitly demands: "Expertise in API design, including RESTful,
    asynchronous, and event-driven APIs, ensuring security, scalability, and
    maintainability" and "API development and testing — HTTP, RESTful services,
    Postman, API Gateway." For a LEAD, API design is a core competency — you
    OWN the API contracts other teams depend on.

DEPTH LEVEL: Senior/Lead. Designing contracts + trade-offs, not just calling APIs.

SECTIONS:
    1.  HTTP Fundamentals (methods, status codes, idempotency)
    2.  REST Principles & Richardson Maturity Model
    3.  API Design Best Practices (resource naming, versioning, pagination)
    4.  Request/Response Design & Error Contracts
    5.  Synchronous vs Asynchronous vs Event-Driven APIs
    6.  API Security (auth, rate limiting, input, CORS)
    7.  Making API Calls (httpx, retries, timeouts, circuit breakers)
    8.  API Gateway & the API Ecosystem
    9.  REST vs GraphQL vs gRPC vs WebSockets (trade-offs)
    10. API Testing (Postman, contract testing, pytest)
    11. Designing APIs for AI/LLM Services
    12. Interview Q&A (senior-level)
    13. GOLDEN LESSONS
===================================================================================
"""


# =================================================================================
# SECTION 1: HTTP FUNDAMENTALS (methods, status codes, idempotency)
# =================================================================================
'''
A lead must know HTTP cold — it's the substrate of every REST API.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HTTP METHODS (verbs) — and their PROPERTIES:

    METHOD   PURPOSE              SAFE?  IDEMPOTENT?  HAS BODY?
    GET      Read a resource      Yes    Yes          No
    POST     Create / action      No     No           Yes
    PUT      Replace (full)       No     Yes          Yes
    PATCH    Partial update       No     No*          Yes
    DELETE   Remove               No     Yes          Maybe

    SAFE = doesn't modify state (GET). IDEMPOTENT = same result if called N times.
    *PATCH is not guaranteed idempotent (depends on the patch semantics).

WHY IDEMPOTENCY MATTERS (senior point):
    Networks are unreliable. A client may retry a request it isn't sure
    completed. Idempotent methods (GET, PUT, DELETE) are SAFE to retry —
    calling DELETE twice still leaves the resource deleted. POST is NOT
    idempotent — retrying "create order" could create TWO orders. Fix:
    idempotency keys (client sends a unique key; server dedupes).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STATUS CODES (know the families + the specific ones):

    2xx SUCCESS
        200 OK            — general success (GET, PATCH)
        201 Created       — resource created (POST) + Location header
        202 Accepted      — async work accepted, not yet done (queued job)
        204 No Content    — success, no body (DELETE)

    3xx REDIRECTION
        301/302           — moved; 304 Not Modified (caching)

    4xx CLIENT ERROR (the caller did something wrong)
        400 Bad Request   — malformed
        401 Unauthorized  — not authenticated (no/invalid credentials)
        403 Forbidden     — authenticated but not allowed (no permission)
        404 Not Found
        409 Conflict      — state conflict (duplicate, version clash)
        422 Unprocessable — validation failed (FastAPI default)
        429 Too Many Requests — rate limited

    5xx SERVER ERROR (we did something wrong)
        500 Internal Server Error
        502 Bad Gateway   — upstream returned junk
        503 Service Unavailable — overloaded / down
        504 Gateway Timeout — upstream too slow

THE 401 vs 403 DISTINCTION (commonly tested):
    401 = "I don't know who you are" (authenticate first).
    403 = "I know who you are, but you can't do this" (authorization fail).

INTERVIEW ANSWER:
    "HTTP methods carry semantic properties: GET is safe and idempotent, PUT
    and DELETE are idempotent, POST is neither. Idempotency matters because
    clients retry on flaky networks — retrying a POST could create duplicates,
    so I use idempotency keys for create operations. On status codes I'm
    precise: 201 for create, 202 for accepted-but-async, 400 for malformed,
    401 for not-authenticated vs 403 for not-authorized, 409 for conflicts,
    422 for validation, 429 for rate limits, and 5xx only for genuine
    server-side failures."
'''


# =================================================================================
# SECTION 2: REST PRINCIPLES & RICHARDSON MATURITY MODEL
# =================================================================================
'''
"What makes an API RESTful?" — a classic. Know the constraints + maturity levels.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

REST = Representational State Transfer. Core CONSTRAINTS:
    1. CLIENT-SERVER — separation of concerns (UI vs data).
    2. STATELESS — each request carries everything needed; server stores no
       client session between requests. (Enables scaling — any server can
       handle any request.)
    3. CACHEABLE — responses declare cacheability (improves performance).
    4. UNIFORM INTERFACE — resources identified by URIs, manipulated via
       standard methods, with representations (JSON).
    5. LAYERED SYSTEM — client can't tell if it's talking to the server or a
       proxy/gateway/load balancer in between.
    6. CODE ON DEMAND (optional) — server can send executable code.

    STATELESS is the one interviewers stress — it's WHY REST scales
    horizontally and pairs with JWT (token carries the state, not the server).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RICHARDSON MATURITY MODEL (levels of "RESTfulness"):
    Level 0 — One URI, one method (RPC-style, e.g., POST /api for everything).
    Level 1 — RESOURCES: multiple URIs (/users, /orders) but still one method.
    Level 2 — HTTP VERBS: proper use of GET/POST/PUT/DELETE + status codes.
              (Most "REST" APIs in industry are here — and that's fine.)
    Level 3 — HATEOAS: responses include links to related actions/resources
              (hypermedia). Rare in practice; mostly academic.

    Practical target: Level 2. Mention HATEOAS exists but say most production
    APIs stop at Level 2 for pragmatism.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

INTERVIEW ANSWER:
    "REST is defined by constraints — client-server, statelessness,
    cacheability, a uniform interface, and a layered system. The one I
    emphasize is statelessness: each request carries all the context it needs,
    so the server keeps no session between calls, which is exactly what lets
    you scale horizontally and pairs naturally with JWT. On maturity, the
    Richardson model goes from RPC-style level 0 up to HATEOAS at level 3;
    in practice I target level 2 — proper resources, HTTP verbs, and status
    codes — which is where most solid production APIs live."
'''


# =================================================================================
# SECTION 3: API DESIGN BEST PRACTICES (naming, versioning, pagination)
# =================================================================================
'''
This is where a LEAD shows craft. Designing a clean, durable contract.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RESOURCE NAMING (nouns, not verbs):
    GOOD: GET /users/123/orders        (resources, hierarchy)
    BAD:  GET /getUserOrders?id=123    (verb in URL — RPC smell)

    - Use plural nouns: /users, /orders, /products.
    - Nest for relationships: /users/123/orders.
    - Verbs live in the HTTP method, not the path.
    - Use kebab-case or lowercase: /order-items.
    - Actions that aren't CRUD: POST /orders/123/cancel (pragmatic exception).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

VERSIONING (so you don't break existing clients):
    1. URI path:    /api/v1/users   (most common, explicit, easy to route)
    2. Header:      Accept: application/vnd.myapi.v2+json (cleaner URLs)
    3. Query param: /users?version=2 (simple but less clean)

    Best practice: version from day one (/v1). Never break a published
    contract — add new fields (backward-compatible) rather than changing
    existing ones. Deprecate with timelines, don't delete abruptly.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PAGINATION (never return unbounded lists):
    - Offset/limit:  /users?limit=20&offset=40 (simple; slow on deep pages).
    - Cursor-based:  /users?limit=20&cursor=abc123 (stable, scales; best for
      large/real-time data — no "page drift" when rows are inserted).
    - Page-based:    /users?page=3&size=20 (user-friendly).

    Always return metadata: total count (if cheap), next/prev cursor, has_more.
    Senior point: cursor-based for large datasets, offset for small/simple ones.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FILTERING, SORTING, FIELD SELECTION:
    /products?category=electronics&sort=-price&fields=id,name,price
    - filter via query params, sort with +/- prefix, sparse fieldsets to
      reduce payload. Keep it consistent across endpoints.

INTERVIEW ANSWER:
    "I design resources as plural nouns with verbs in the HTTP method —
    GET /users/123/orders, not /getUserOrders. I version from day one, usually
    in the URI path as /v1, and I evolve contracts backward-compatibly by
    adding fields rather than changing existing ones, with deprecation
    timelines. I never return unbounded lists — pagination always, cursor-based
    for large or real-time datasets to avoid page drift, offset for simple
    cases, with next-cursor and has_more metadata. Filtering, sorting, and
    sparse fieldsets go through consistent query params."
'''


# =================================================================================
# SECTION 4: REQUEST/RESPONSE DESIGN & ERROR CONTRACTS
# =================================================================================
'''
Consistency is what makes an API maintainable — a LEAD-level concern.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CONSISTENT RESPONSE ENVELOPE:
    Pick ONE shape and use it everywhere. Example:
    {
        "data": { ... },              # the payload
        "meta": {"page": 1, "total": 100},  # pagination/extra
        "errors": null
    }
    Or return the resource directly for simple APIs. The KEY is consistency —
    every endpoint behaves the same way so clients can rely on it.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ERROR CONTRACT (consistent, machine-readable):
    {
        "error": {
            "code": "INSUFFICIENT_FUNDS",     # stable machine code
            "message": "Balance too low",      # human message
            "details": [{"field": "amount", "issue": "exceeds balance"}],
            "request_id": "req_abc123"         # for tracing/support
        }
    }
    - Stable error CODES (not just messages) so clients can branch on them.
    - request_id correlates with server logs for debugging.
    - NEVER leak stack traces or internal details to clients.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DESIGN PRINCIPLES:
    - Use the RIGHT status code (Section 1) AND a structured error body.
    - Validate input at the boundary (Pydantic) -> 422 with field details.
    - Return created resources with 201 + Location header.
    - Be consistent with date formats (ISO 8601, UTC), naming (snake_case
      or camelCase — pick one), and nulls vs omitted fields.
    - Document with OpenAPI (FastAPI auto-generates it).

INTERVIEW POINT:
    "I keep request and response shapes consistent across the whole API and
    use a structured error contract with a stable machine-readable code, a
    human message, field-level details, and a request_id that correlates to
    server logs. Stable codes let clients branch on errors programmatically,
    and I never leak stack traces — those stay in logs behind the request_id."
'''


# =================================================================================
# SECTION 5: SYNCHRONOUS vs ASYNCHRONOUS vs EVENT-DRIVEN APIs
# =================================================================================
'''
The JD names all three: "RESTful, asynchronous, and event-driven APIs."
Know the distinction precisely — these are different architectural styles.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SYNCHRONOUS (request-response):
    Client calls, WAITS, gets the result in the same response.
    GET /users/123 -> 200 with the user.
    Good for: fast operations where the client needs the answer immediately.
    Limitation: client blocks; bad for long-running work.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ASYNCHRONOUS API (long-running work, polling/callback):
    For work that takes too long for one request (report generation, video
    transcode, a long LLM pipeline):
    1. Client: POST /reports -> 202 Accepted + {"job_id": "j123", "status_url": "/jobs/j123"}
    2. Server processes in the background (queue + worker).
    3. Client POLLS GET /jobs/j123 -> {"status": "processing"} ... then "done" + result.
       OR server calls a WEBHOOK (callback URL) when done.

    NOTE: "asynchronous API" (this architectural pattern) is DIFFERENT from
    "async def" (Python concurrency). Don't conflate them in the interview.
    - async def = how ONE server handles concurrency internally.
    - asynchronous API = a request/response PATTERN where the result comes later.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EVENT-DRIVEN API (publish/subscribe, no direct request):
    Services communicate by emitting and reacting to EVENTS via a message
    broker (Kafka, RabbitMQ, SNS/SQS, Redis Streams), not direct HTTP calls.
    - Producer publishes an event ("OrderPlaced") to a topic.
    - Consumers subscribe and react independently (inventory, email, analytics).
    Benefits: DECOUPLING (producer doesn't know/wait for consumers),
              scalability, resilience (events buffered if a consumer is down),
              extensibility (add consumers without touching the producer).
    Cost: eventual consistency, harder to debug/trace, ordering/delivery
          guarantees to manage.
    Patterns: pub/sub, event sourcing, CQRS, choreography vs orchestration.

    Push to clients: WebSockets / Server-Sent Events (real-time updates).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHEN TO USE WHICH:
    Fast read/write, client needs answer now      -> Synchronous REST.
    Long-running job                              -> Async API (202 + poll/webhook).
    Multiple services reacting to state changes,
    decoupling, scale                             -> Event-driven (broker).
    Real-time server->client updates              -> WebSockets / SSE.

INTERVIEW ANSWER:
    "Synchronous REST is request-response — the client waits for the answer,
    good for fast operations. For long-running work I use an asynchronous API
    pattern: return 202 with a job ID, process in a queue, and the client
    polls a status endpoint or I call their webhook — and I'm careful that
    this is different from Python's async def, which is just in-process
    concurrency. Event-driven APIs are pub/sub through a broker like Kafka —
    a producer emits 'OrderPlaced' and independent consumers react. That gives
    decoupling, scalability, and resilience at the cost of eventual
    consistency and harder tracing. For real-time push to clients I use
    WebSockets or SSE."
'''


# =================================================================================
# SECTION 6: API SECURITY (auth, rate limiting, input, CORS)
# =================================================================================
'''
The JD stresses "secure APIs." Security is a lead's responsibility.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

AUTHENTICATION vs AUTHORIZATION:
    AuthN = WHO are you? (login, token verification) -> 401 if it fails.
    AuthZ = WHAT can you do? (roles, permissions, scopes) -> 403 if it fails.

AUTH MECHANISMS:
    - API keys: simple, service-to-service, send in a header.
    - JWT bearer: stateless, signed token with claims; verify per request.
    - OAuth2 / OIDC: delegated auth (login with Google), scopes, refresh tokens.
    - mTLS: mutual TLS for high-trust service-to-service.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE API SECURITY CHECKLIST (OWASP API Top 10 themes):
    1. AUTH on every protected endpoint (no "forgot to protect it" routes).
    2. AUTHORIZATION checks — object-level (can THIS user access THIS record?).
       (Broken object-level authorization is the #1 API vulnerability.)
    3. INPUT VALIDATION — Pydantic; reject malformed -> prevents injection.
    4. PARAMETERIZED QUERIES / ORM — never string-concatenate SQL.
    5. RATE LIMITING — per user/IP/key; prevents abuse, brute force, DoS (429).
    6. HTTPS/TLS everywhere — never tokens over plain HTTP.
    7. CORS — allow only specific trusted origins, not "*", for browser clients.
    8. SECRETS — env vars / vault, never in code or logs.
    9. DON'T OVER-EXPOSE — response_model filters fields; no internal data leak.
    10. SECURITY HEADERS, request size limits, dependency scanning.

RATE LIMITING STRATEGIES:
    - Fixed window, sliding window, token bucket, leaky bucket.
    - Enforce at API Gateway or with slowapi/Redis in-app.
    - Return 429 + Retry-After header.

INTERVIEW ANSWER:
    "I separate authentication — verifying who you are, 401 on failure — from
    authorization — what you're allowed to do, 403 on failure. For APIs I
    typically use JWT bearer tokens verified per request, with scopes/roles in
    the claims. The biggest real-world API vulnerability is broken
    object-level authorization, so I always check that THIS user can access
    THIS specific record, not just that they're logged in. Beyond that:
    Pydantic input validation and parameterized queries to stop injection,
    rate limiting with 429s, HTTPS everywhere, CORS locked to trusted origins,
    secrets in a vault, and response models that prevent over-exposing data."
'''


# =================================================================================
# SECTION 7: MAKING API CALLS (httpx, retries, timeouts, circuit breakers)
# =================================================================================
'''
The JD asks about "how API calls are made." Consuming APIs robustly is as
important as designing them. This is where resilience lives.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE CLIENT LIBRARIES:
    - requests: the classic SYNC HTTP library. Simple, blocking.
    - httpx: modern; supports BOTH sync and ASYNC (httpx.AsyncClient).
             Use httpx in async code (FastAPI) — requests would block the loop.
    - aiohttp: async-only HTTP client/server.

    BASIC ASYNC CALL:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get("https://api.example.com/data")
            resp.raise_for_status()        # raise on 4xx/5xx
            data = resp.json()

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RESILIENCE PATTERNS (this is what separates senior from junior):

    1. TIMEOUTS (always set them):
        Never make a call without a timeout — a hung upstream would hang YOU.
        connect timeout + read timeout. Default httpx has one; set it explicitly.

    2. RETRIES with EXPONENTIAL BACKOFF + JITTER:
        Transient failures (429, 503, network blips) should be retried, but
        with increasing delays (1s, 2s, 4s) + random jitter to avoid a
        thundering herd. Retry ONLY idempotent ops or use idempotency keys.
        Libraries: tenacity (Python), or httpx + custom transport.

    3. CIRCUIT BREAKER:
        If an upstream keeps failing, STOP calling it for a cooldown period
        (the breaker "opens"), failing fast instead of piling up requests.
        After cooldown, allow a trial request (half-open). Prevents cascading
        failures. Libraries: pybreaker.

    4. CONNECTION POOLING:
        Reuse a single AsyncClient (created once, e.g., on app startup) instead
        of creating one per request — reuses TCP connections, far faster.
        Anti-pattern: 'async with httpx.AsyncClient()' inside every call.

    5. RATE-LIMIT HANDLING:
        Respect 429 + Retry-After. Cap your own concurrency (Semaphore from
        Phase 2) to stay under the provider's limit.

    6. IDEMPOTENCY for retried writes:
        Send an idempotency key so a retried POST doesn't duplicate.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE ROBUST CALL (conceptual):
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(),
           retry=retry_if_exception_type((httpx.TimeoutException, RateLimitError)))
    async def call_upstream(client, payload):
        resp = await client.post(URL, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()

INTERVIEW ANSWER:
    "I use httpx in async code — requests would block the event loop. The
    resilience patterns matter most: always set explicit timeouts so a hung
    upstream can't hang me; retry transient failures like 429 or 503 with
    exponential backoff plus jitter, but only for idempotent calls or with an
    idempotency key; a circuit breaker that opens after repeated failures so I
    fail fast instead of cascading; and connection pooling — one shared
    AsyncClient created at startup, not a new one per request. I respect
    Retry-After on 429s and cap my own concurrency with a semaphore to stay
    under provider limits."
'''


# =================================================================================
# SECTION 8: API GATEWAY & THE API ECOSYSTEM
# =================================================================================
'''
The JD names "API Gateway" explicitly. Know what it does and why.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHAT AN API GATEWAY IS:
    A single entry point sitting IN FRONT of your backend services. Clients
    talk to the gateway; it routes to the right service. (AWS API Gateway,
    Kong, Apigee, nginx, Azure API Management.)

WHAT IT HANDLES (cross-cutting concerns, centralized):
    - ROUTING — maps paths to backend services (/users -> user-service).
    - AUTH — validates tokens/keys before traffic reaches services.
    - RATE LIMITING & THROTTLING — protects backends from overload.
    - REQUEST/RESPONSE TRANSFORMATION — reshape payloads, add headers.
    - CACHING — cache responses at the edge.
    - LOGGING/METRICS/TRACING — central observability.
    - TLS TERMINATION — handles HTTPS so services don't have to.
    - LOAD BALANCING — distribute across service instances.

WHY USE ONE (the value):
    Centralizes cross-cutting concerns so individual services stay focused on
    business logic. In microservices, the gateway is the front door — one
    place for auth, rate limiting, and routing instead of duplicating them in
    every service.

GATEWAY vs LOAD BALANCER (common confusion):
    - Load balancer: distributes traffic across identical instances (L4/L7).
    - API gateway: application-aware routing + auth + rate limit + transform.
      A gateway often includes load balancing but does much more.

INTERVIEW POINT:
    "An API gateway is the single front door to backend services. It
    centralizes cross-cutting concerns — routing, authentication, rate
    limiting, TLS termination, caching, and observability — so each service
    focuses on business logic instead of re-implementing all that. In a
    microservices setup it's where I enforce auth and throttling once rather
    than in every service. It's more than a load balancer — it's
    application-aware and handles auth and transformation, not just traffic
    distribution."
'''


# =================================================================================
# SECTION 9: REST vs GraphQL vs gRPC vs WebSockets (trade-offs)
# =================================================================================
'''
A lead must choose the right API style. Trade-offs again — the JD's #1 theme.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    STYLE      BEST FOR                       TRADE-OFFS
    REST       General CRUD, public APIs,     Simple, cacheable, ubiquitous;
               broad compatibility            over/under-fetching, many round-trips
    GraphQL    Flexible client queries,       Client asks for exactly the fields
               complex/nested data, mobile    it needs (no over-fetch); but
                                              caching is harder, complexity,
                                              N+1 risk on the resolver side
    gRPC       Internal service-to-service,   Binary (protobuf), fast, strongly
               low latency, streaming         typed, HTTP/2; not browser-native,
                                              less human-readable
    WebSockets Real-time bidirectional        Live chat, notifications, streaming;
               (chat, live updates)           stateful connections, scaling harder
    SSE        Server->client one-way stream  Simpler than WebSockets for
                                              streaming (e.g., LLM tokens)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHEN TO CHOOSE WHICH:
    - Public API, broad clients, caching matters     -> REST.
    - Rich front-end needing flexible data shapes    -> GraphQL.
    - Internal microservices, performance-critical   -> gRPC.
    - Real-time bidirectional (chat)                 -> WebSockets.
    - Streaming LLM tokens to a browser              -> SSE (or WebSockets).

INTERVIEW ANSWER:
    "REST is my default for public and CRUD APIs — simple, cacheable,
    universally supported, though it can over- or under-fetch. GraphQL when
    the client needs flexible, nested data and wants to avoid multiple
    round-trips, accepting harder caching and resolver complexity. gRPC for
    internal service-to-service where latency matters — it's binary protobuf
    over HTTP/2, fast and strongly typed but not browser-native. WebSockets
    for real-time bidirectional like chat, and SSE for one-way server streaming
    such as pushing LLM tokens to a browser. The choice is driven by client
    type, data shape, latency, and real-time needs."
'''


# =================================================================================
# SECTION 10: API TESTING (Postman, contract testing, pytest)
# =================================================================================
'''
The JD names Postman explicitly and stresses testing. Know the layers.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

POSTMAN (manual + automated API testing):
    - Build requests, organize into COLLECTIONS, use ENVIRONMENTS (dev/staging/
      prod variables), write test scripts (pm.test) to assert status/body.
    - Newman runs Postman collections in CI/CD.
    - Good for: exploratory testing, sharing API examples, smoke tests.

LEVELS OF API TESTING:
    1. UNIT — individual functions/handlers with mocked dependencies (pytest).
    2. INTEGRATION — endpoint + real DB/services (FastAPI TestClient + test DB).
    3. CONTRACT — verify the API matches the agreed schema/contract so a
       change doesn't break consumers (Pact, schema validation against OpenAPI).
    4. END-TO-END — full user flow across services.
    5. LOAD/PERFORMANCE — throughput, latency under load (Locust, k6, JMeter).
    6. SECURITY — auth bypass, injection, fuzzing (OWASP ZAP).

CONTRACT TESTING (senior concept):
    In microservices, a provider's API change can silently break consumers.
    Contract tests pin the agreed request/response shape so CI catches a
    breaking change before deploy. The OpenAPI spec is the contract; validate
    against it.

pytest FOR APIs (from Phase 1):
    client = TestClient(app)
    resp = client.post("/orders", json={...})
    assert resp.status_code == 201
    Use dependency_overrides to inject a test DB / mock upstream.

INTERVIEW POINT:
    "I test APIs in layers: pytest unit tests with mocked dependencies,
    integration tests with the FastAPI TestClient against a test DB, and
    contract tests so a provider change can't silently break consumers — the
    OpenAPI spec is the contract. Postman collections with environments cover
    exploratory and smoke testing, and Newman runs them in CI. For
    non-functional, Locust or k6 for load and OWASP ZAP for security."
'''


# =================================================================================
# SECTION 11: DESIGNING APIs FOR AI/LLM SERVICES
# =================================================================================
'''
Connect API design to the GenAI use case — your differentiator for this role.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SPECIAL CONSIDERATIONS FOR LLM/AGENT APIs:

    1. LATENCY — LLM calls take seconds. Design for it:
       - Stream responses (SSE/WebSocket) so users see tokens immediately.
       - Or async-API pattern (202 + job + poll) for long agentic pipelines.

    2. STREAMING ENDPOINT:
       POST /chat -> StreamingResponse yielding tokens as they generate.
       Far better UX than waiting 10s for a full answer.

    3. TIMEOUTS & RETRIES — model providers fail/throttle; wrap with timeout +
       backoff + circuit breaker (Section 7). Cap concurrency (Semaphore).

    4. COST CONTROL — token usage = money. Track tokens per request, set
       max_tokens, cache frequent prompts/responses, use smaller models for
       simple tasks (routing).

    5. STATELESS + CONVERSATION STATE — REST is stateless, but chat needs
       history. Pass a conversation_id; store history server-side (DB/Redis)
       keyed by it, or have the client send the history. (Maps to LangGraph
       checkpointer + thread_id.)

    6. INPUT/OUTPUT VALIDATION — Pydantic models for request (prompt, params)
       and structured output validation for the model's response (it may
       return malformed JSON — validate + retry).

    7. RATE LIMITING per user — protect your provider quota and your costs.

    8. IDEMPOTENCY — a retried "generate" shouldn't double-charge; idempotency keys.

    9. OBSERVABILITY — log prompts, token counts, latency, model version;
       trace agent steps (LangSmith) for debugging agentic loops.

INTERVIEW ANSWER:
    "Designing APIs for LLM services has specific concerns. Latency is the big
    one — calls take seconds — so I stream tokens with SSE for chat UX, or use
    the async-API pattern with a job ID for long agentic pipelines. I wrap
    provider calls with timeouts, backoff, and a circuit breaker, cap
    concurrency with a semaphore to respect quotas, and track token usage for
    cost control. REST is stateless, so for conversation I pass a
    conversation_id and store history server-side keyed by it — which maps to
    a LangGraph checkpointer with a thread_id. I validate both the request and
    the model's structured output, since LLMs can return malformed JSON, and
    I log prompts, tokens, latency, and trace agent steps for observability."
'''


# =================================================================================
# SECTION 12: INTERVIEW Q&A (senior-level)
# =================================================================================
'''
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q1. What makes an API RESTful?
A:  "Stateless client-server with a uniform interface — resources via URIs,
    HTTP verbs, status codes, cacheable, layered. Statelessness is what lets
    it scale horizontally."

Q2. Idempotency — what and why?
A:  "Same result if called N times. GET/PUT/DELETE are idempotent, POST isn't.
    It matters for safe retries on flaky networks; for creates I use
    idempotency keys to avoid duplicates."

Q3. 401 vs 403?
A:  "401 = not authenticated (who are you?). 403 = authenticated but not
    authorized (you can't do this)."

Q4. How do you version an API?
A:  "URI path /v1 usually; evolve backward-compatibly by adding fields, never
    breaking published contracts; deprecate with timelines."

Q5. Synchronous vs asynchronous vs event-driven API?
A:  "Sync = wait for response. Async API = 202 + job ID + poll/webhook for
    long work. Event-driven = pub/sub via a broker for decoupled services."

Q6. How do you make a resilient external API call?
A:  "httpx async client, explicit timeouts, retries with exponential backoff +
    jitter for transient errors, a circuit breaker, connection pooling, and
    respect for 429 Retry-After."

Q7. What does an API gateway do?
A:  "Single front door — routing, auth, rate limiting, TLS termination,
    caching, observability — centralizing cross-cutting concerns so services
    stay focused on business logic."

Q8. REST vs GraphQL vs gRPC?
A:  "REST for public/CRUD and caching; GraphQL for flexible nested client
    queries; gRPC for fast internal service-to-service over HTTP/2."

Q9. How do you paginate a large dataset?
A:  "Cursor-based for large/real-time data to avoid page drift; offset for
    simple cases; always return next-cursor and has_more."

Q10. How do you secure an API?
A:  "AuthN via JWT, AuthZ with object-level checks (the top API vuln), input
    validation, parameterized queries, rate limiting, HTTPS, scoped CORS,
    secrets in a vault, response models to avoid over-exposure."

Q11. How do you design an API for a chatbot/LLM?
A:  "Streaming endpoint with SSE for token-by-token UX, conversation_id for
    stateless history, timeouts/retries/circuit breaker on provider calls,
    token tracking for cost, and rate limiting per user."

Q12. How do you stop a slow upstream from taking down your service?
A:  "Timeouts so calls can't hang, a circuit breaker to fail fast after
    repeated failures, bulkheads/concurrency caps to isolate it, and
    graceful degradation — return a fallback or cached response."
'''


# =================================================================================
# SECTION 13: GOLDEN LESSONS
# =================================================================================
'''
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GOLDEN LESSON 1 — STATELESSNESS IS WHY REST SCALES.
    Each request self-contained; any server handles any request. Pairs with JWT.

GOLDEN LESSON 2 — IDEMPOTENCY ENABLES SAFE RETRIES.
    GET/PUT/DELETE idempotent; POST needs idempotency keys.

GOLDEN LESSON 3 — KNOW STATUS CODES PRECISELY.
    201 create, 202 async-accepted, 401 vs 403, 409 conflict, 422 validation,
    429 rate limit. 4xx = client, 5xx = server.

GOLDEN LESSON 4 — SYNC vs ASYNC API vs EVENT-DRIVEN ARE DIFFERENT STYLES.
    And "async API" != Python "async def". Don't conflate them.

GOLDEN LESSON 5 — RESILIENCE IS THE SENIOR DIFFERENTIATOR.
    Timeouts + retries with backoff + circuit breaker + connection pooling.
    Always. A call without a timeout is a bug.

GOLDEN LESSON 6 — BROKEN OBJECT-LEVEL AUTHZ IS THE #1 API VULN.
    Always check THIS user can access THIS record, not just "logged in".

GOLDEN LESSON 7 — DESIGN CONSISTENT CONTRACTS.
    Consistent envelopes, error codes, naming, dates. Consistency = maintainability.

GOLDEN LESSON 8 — VERSION FROM DAY ONE; EVOLVE BACKWARD-COMPATIBLY.
    Add fields, don't break. Deprecate with timelines.

GOLDEN LESSON 9 — API GATEWAY CENTRALIZES CROSS-CUTTING CONCERNS.
    Auth, rate limit, routing, TLS once at the front door, not per service.

GOLDEN LESSON 10 — FOR LLM APIs: STREAM, TIMEOUT, TRACK TOKENS, MANAGE STATE.
    Stream tokens (SSE), wrap provider calls in resilience, conversation_id for
    history, token accounting for cost. Ties API design to the JD's GenAI focus.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ONE-LINE CHEAT SHEET:
    REST = stateless + uniform interface; statelessness -> scales.
    Methods: GET/PUT/DELETE idempotent, POST not (idempotency keys).
    Codes: 201/202/400/401/403/404/409/422/429/5xx.
    Styles: sync (wait) | async-API (202+poll/webhook) | event-driven (broker).
    Consume: httpx + timeout + backoff retry + circuit breaker + pooling.
    Gateway: routing + auth + rate limit + TLS + observability at the front door.
    Security: AuthN(JWT) + object-level AuthZ + validation + rate limit + HTTPS.
    Choose: REST(public) | GraphQL(flexible) | gRPC(internal) | WS/SSE(realtime).
    LLM APIs: stream tokens, conversation_id, resilience, token cost tracking.
'''


# =================================================================================
# RUN SUMMARY
# =================================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 3 — API DESIGN & INTEGRATION")
    print("=" * 70)
    print()
    print("REST CORE: stateless + uniform interface -> horizontal scaling")
    print("METHODS: GET/PUT/DELETE idempotent | POST not (use idempotency keys)")
    print("CODES: 201 create | 202 async | 401 vs 403 | 409 | 422 | 429 | 5xx")
    print()
    print("3 API STYLES (the JD names all three):")
    print("  Synchronous  -> wait for response")
    print("  Asynchronous -> 202 + job id + poll/webhook (NOT Python async def)")
    print("  Event-driven -> pub/sub via broker (Kafka), decoupled services")
    print()
    print("RESILIENT CALLS (senior differentiator):")
    print("  httpx + timeouts + backoff retries + circuit breaker + pooling")
    print()
    print("SECURITY: AuthN(JWT) + object-level AuthZ (#1 vuln) + validation")
    print("          + rate limit + HTTPS + scoped CORS + secrets in vault")
    print()
    print("API GATEWAY: routing + auth + rate limit + TLS + observability")
    print()
    print("LLM APIs: stream tokens (SSE) + conversation_id + resilience + cost")
    print()
    print("=" * 70)
    print("Next: Phase 4 — Databases: Schema Design & Optimization")
    print("=" * 70)
