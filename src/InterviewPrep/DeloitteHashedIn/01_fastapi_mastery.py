"""
===================================================================================
PHASE 1 — FASTAPI MASTERY (HashedIn by Deloitte — Lead Python/GenAI Expert)
===================================================================================

WHY THIS LESSON IS #1:
    The JD names FastAPI FIRST and mentions async/asyncio TWICE. For a Lead
    Python role at HashedIn (Deloitte), FastAPI + async is the highest-
    probability deep-dive area. Interviewers will ask: how a request flows,
    dependency injection, Pydantic validation, async vs sync, middleware,
    background tasks, and how you'd design production APIs.

DEPTH LEVEL: Senior/Lead. Not "what is a decorator" — but "why async,
    what's the event loop doing, when does it hurt, how do you structure
    a production FastAPI service, what are the trade-offs."

SECTIONS:
    1.  Why FastAPI — and trade-offs vs Flask / Django REST
    2.  The Request Lifecycle (end-to-end — the #1 interview question)
    3.  Path, Query, Body Params + Pydantic Models
    4.  Async Endpoints — When async Helps and When It HURTS
    5.  Dependency Injection (deep — DI is heavily tested)
    6.  Pydantic v2 — Validation, Response Models, Settings
    7.  Middleware — Cross-cutting Concerns
    8.  Background Tasks vs Real Task Queues
    9.  Error Handling & Exception Handlers
    10. Authentication & Security
    11. Testing FastAPI (pytest + TestClient/httpx)
    12. Production Deployment (ASGI, uvicorn, gunicorn, workers)
    13. Interview Q&A (senior-level, with crisp answers)
    14. GOLDEN LESSONS
===================================================================================
"""


# =================================================================================
# SECTION 1: WHY FASTAPI — and trade-offs vs Flask / Django REST
# =================================================================================
"""
A lead must justify framework choice with TRADE-OFFS, not "it's popular."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHAT FASTAPI IS:
    A modern, high-performance Python web framework built on:
    - Starlette (the ASGI toolkit — handles the async HTTP layer)
    - Pydantic (data validation + serialization via Python type hints)
    - ASGI (Asynchronous Server Gateway Interface — the async successor to WSGI)

    Three pillars: ASGI (async), type hints (validation), auto docs (OpenAPI).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHY IT WINS (the selling points):
    1. NATIVE ASYNC — built on ASGI, so it handles thousands of concurrent
       I/O-bound requests (DB calls, LLM API calls) on a single worker.
    2. AUTOMATIC VALIDATION — type hints + Pydantic validate and parse
       requests/responses automatically. Less boilerplate, fewer bugs.
    3. AUTO-GENERATED DOCS — OpenAPI/Swagger UI + ReDoc out of the box.
    4. PERFORMANCE — among the fastest Python frameworks (close to Node/Go
       for I/O-bound workloads) because of async + Starlette.
    5. DEVELOPER SPEED — type hints give editor autocomplete + catch errors early.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TRADE-OFFS TABLE (this is what a LEAD is expected to articulate):

    DIMENSION          FASTAPI            FLASK              DJANGO REST (DRF)
    Async support      Native (ASGI)      Add-on (WSGI)      Limited (improving)
    Validation         Built-in (Pydantic) Manual/Marshmallow Serializers
    Performance (I/O)  Very high          Moderate           Moderate
    Batteries included Minimal            Minimal            Full (ORM, admin, auth)
    Learning curve     Low-medium         Low                Medium-high
    Best for           Async APIs, ML/AI  Small apps, simple Full web apps, CRUD
                       microservices      services           with ORM + admin

WHEN TO CHOOSE WHICH (the senior answer):
    - FastAPI: async I/O-bound services, ML/LLM APIs, microservices, when
      you need high concurrency and automatic validation. (Our GenAI use case.)
    - Django REST: when you need a full batteries-included framework — ORM,
      admin panel, built-in auth, mature ecosystem — for a large CRUD app.
    - Flask: small, simple synchronous services or when you want maximal
      control with minimal framework opinion.

INTERVIEW ANSWER:
    "FastAPI is built on ASGI, Starlette, and Pydantic, so it gives native
    async, automatic validation from type hints, and auto-generated OpenAPI
    docs. I choose it for I/O-bound services — especially LLM/AI APIs where
    a single request waits on a model call — because async lets one worker
    handle thousands of concurrent waits. I'd choose Django REST instead when
    I need a batteries-included framework with ORM, admin, and built-in auth
    for a large CRUD application, and Flask for small simple synchronous
    services. The deciding factor is concurrency needs and how much framework
    I want versus control."
"""


# =================================================================================
# SECTION 2: THE REQUEST LIFECYCLE (end-to-end — the #1 interview question)
# =================================================================================
"""
"Walk me through what happens when a request hits your FastAPI endpoint."
This is the most common deep-dive. Know every step.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE FULL REQUEST FLOW:

    1. CLIENT sends HTTP request -> hits the ASGI server (uvicorn/gunicorn).
    2. ASGI SERVER receives it on the event loop, passes the ASGI 'scope'
       (request metadata) to the FastAPI app.
    3. MIDDLEWARE chain runs (outermost first) — CORS, auth, logging, timing.
       Each middleware can short-circuit or pass to the next.
    4. ROUTING — FastAPI matches the URL path + HTTP method to a route handler.
    5. DEPENDENCY RESOLUTION — FastAPI resolves all Depends() for that route
       (DB sessions, auth, shared logic). Dependencies run BEFORE the handler.
    6. REQUEST PARSING + VALIDATION — path/query/body are extracted and
       validated against type hints + Pydantic models. Invalid -> 422 auto.
    7. HANDLER EXECUTION — your endpoint function runs.
       - If 'async def': runs ON the event loop (can await I/O).
       - If 'def': runs in a THREAD POOL (so it doesn't block the loop).
    8. RESPONSE MODEL — return value is validated/serialized via response_model.
    9. MIDDLEWARE (unwinding) — response passes back through middleware (innermost
       first) for response headers, timing, etc.
    10. ASGI SERVER sends the HTTP response back to the client.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE CRITICAL DETAIL (step 7 — sync vs async routing):

    FastAPI is smart about this:
    - 'async def' endpoint  -> runs directly on the event loop.
      MUST use 'await' for I/O (httpx, async DB). NEVER call blocking code.
    - 'def' (sync) endpoint -> FastAPI runs it in an external THREAD POOL
      so a blocking call doesn't freeze the event loop.

    This is THE most important FastAPI internal to understand. (See Section 4.)

INTERVIEW ANSWER:
    "A request hits the ASGI server — uvicorn — which puts it on the event
    loop. It passes through the middleware chain, then routing matches the
    path and method. FastAPI resolves the route's dependencies, then parses
    and validates path/query/body against the type hints and Pydantic models —
    invalid input returns a 422 automatically. The handler runs: async
    handlers run on the event loop, sync handlers run in a thread pool so
    they don't block it. The return value is serialized through the response
    model, unwinds back through middleware, and uvicorn sends the response."
"""


# =================================================================================
# SECTION 3: PATH, QUERY, BODY PARAMS + PYDANTIC MODELS
# =================================================================================
'''
How FastAPI knows where each parameter comes from — by type + declaration.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE RULES FASTAPI USES TO LOCATE A PARAMETER:
    - If the param name is in the PATH ("/items/{item_id}") -> path param.
    - If it's a singular type (int, str, bool) NOT in the path -> query param.
    - If it's a Pydantic model -> request BODY (parsed from JSON).
    - Special types (Header, Cookie, Form, File, Depends) -> declared explicitly.

EXAMPLE:

    from fastapi import FastAPI, Query, Path
    from pydantic import BaseModel

    app = FastAPI()

    class Item(BaseModel):
        name: str
        price: float
        tags: list[str] = []

    @app.post("/items/{item_id}")
    async def create_item(
        item_id: int = Path(..., gt=0),          # PATH param, must be > 0
        q: str | None = Query(None, max_length=50),  # QUERY param, optional
        item: Item = ...,                          # BODY (Pydantic model)
    ):
        return {"item_id": item_id, "q": q, "item": item}

    Request: POST /items/5?q=search   body: {"name": "Widget", "price": 9.99}
    - item_id = 5 (from path, validated > 0)
    - q = "search" (from query)
    - item = Item(name="Widget", price=9.99) (from body, validated)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

VALIDATION CONSTRAINTS (Query/Path/Field):
    gt, ge, lt, le        -> numeric bounds
    min_length, max_length -> string/list length
    pattern               -> regex
    default / ...         -> ... (Ellipsis) means REQUIRED; a value = default.

WHY THIS MATTERS:
    Invalid input is rejected with a 422 + a clear error BEFORE your code
    runs. You never write manual "if not item_id: return error" boilerplate.

INTERVIEW POINT:
    "FastAPI infers parameter source from declaration: path params from the
    URL template, scalar types as query params, and Pydantic models as the
    JSON body. Validation constraints live in the type declaration, so bad
    input is rejected with a 422 before the handler runs — no manual checks."
'''


# =================================================================================
# SECTION 4: ASYNC ENDPOINTS — When async Helps and When It HURTS
# =================================================================================
'''
THE most important section. The JD calls out asyncio TWICE. This is where
seniors get separated from juniors.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE MENTAL MODEL:
    FastAPI runs on a SINGLE-THREADED event loop (per worker). The loop can
    handle thousands of requests CONCURRENTLY — but only because, while one
    request is WAITING on I/O (DB, HTTP, LLM call), the loop switches to
    another request. The key word is AWAIT — it yields control back to the loop.

    CONCURRENCY (async) != PARALLELISM (multiple cores).
    Async gives concurrency on ONE thread by interleaving waits.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHEN async HELPS (I/O-bound — the 90% case for AI APIs):
    Calls that WAIT on something external:
    - Database queries (async driver: asyncpg, databases)
    - HTTP/LLM API calls (httpx async, openai async client)
    - File/network I/O

    async def get_data():
        async with httpx.AsyncClient() as client:
            resp = await client.get("https://api...")   # loop frees up here
        return resp.json()

    While awaiting, the worker serves OTHER requests. 1000 concurrent LLM
    calls on one worker — that's the async win.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHEN async HURTS (the trap that catches juniors):

    MISTAKE 1 — Blocking call inside async def:
        @app.get("/bad")
        async def bad():
            time.sleep(5)            # BLOCKS the entire event loop!
            requests.get(url)        # BLOCKS — sync library in async fn
        -> ALL other requests freeze for 5s. Catastrophic under load.

        FIX: use async libs (asyncio.sleep, httpx), OR offload:
            await asyncio.to_thread(blocking_function)

    MISTAKE 2 — CPU-bound work in async def:
        async def crunch():
            heavy_computation()      # CPU-bound blocks the loop too
        -> Async does NOT help CPU-bound work. Use multiprocessing or a
           separate worker/queue (Celery) for heavy CPU tasks.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE DECISION RULE:
    I/O-bound + async library available  -> async def + await.
    I/O-bound + only sync library        -> def (FastAPI runs it in thread pool)
                                            OR await asyncio.to_thread(...).
    CPU-bound                            -> def + offload to process pool / queue.

    KEY: a plain 'def' endpoint is SAFE — FastAPI runs it in a thread pool so
    it won't block the loop. The danger is a BLOCKING call inside 'async def'.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RUNNING MULTIPLE I/O CALLS CONCURRENTLY (gather):
    async def aggregate():
        results = await asyncio.gather(
            fetch_user(),       # these three run
            fetch_orders(),     # CONCURRENTLY, not
            fetch_recommendations(),  # sequentially
        )
        return results
    -> 3 calls that each take 200ms finish in ~200ms total, not 600ms.

INTERVIEW ANSWER:
    "FastAPI runs on a single-threaded event loop per worker. async gives
    concurrency by yielding control on 'await' during I/O waits — so one
    worker can handle thousands of concurrent DB or LLM calls. The critical
    rule: never put a BLOCKING call inside an async def — time.sleep or the
    sync requests library freezes the whole loop and every request stalls.
    For blocking work I either use an async library, offload with
    asyncio.to_thread, or use a plain def endpoint which FastAPI safely runs
    in a thread pool. Async does NOT help CPU-bound work — that needs
    multiprocessing or a task queue. And I use asyncio.gather to run
    independent I/O calls concurrently."
'''


# =================================================================================
# SECTION 5: DEPENDENCY INJECTION (deep — DI is heavily tested)
# =================================================================================
'''
DI is FastAPI's signature feature and a favorite deep-dive. Master it.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHAT IT IS:
    A dependency is a function/callable whose RESULT is "injected" into your
    route. FastAPI RESOLVES it before the handler runs and passes the value in.
    Declared with Depends().

    def get_db():
        db = SessionLocal()
        try:
            yield db                 # yield = provide the resource
        finally:
            db.close()               # cleanup runs AFTER the response

    @app.get("/users")
    async def list_users(db = Depends(get_db)):
        return db.query(User).all()

    FastAPI calls get_db(), injects the session, and after the response,
    runs the finally block to close it. This is RESOURCE MANAGEMENT via DI.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHY DI MATTERS (the senior framing):
    1. REUSE — shared logic (DB session, auth, pagination) declared once,
       used in many routes.
    2. TESTABILITY — override dependencies in tests (app.dependency_overrides)
       to inject mocks/fakes without touching route code. HUGE for testing.
    3. SEPARATION OF CONCERNS — routes stay thin; cross-cutting logic lives
       in dependencies.
    4. RESOURCE LIFECYCLE — yield-based dependencies handle setup + teardown
       (open/close DB, acquire/release locks).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

NESTED / CHAINED DEPENDENCIES:
    Dependencies can depend on other dependencies — FastAPI resolves the graph.

    def get_db(): ...
    def get_current_user(db = Depends(get_db), token = Depends(oauth2_scheme)):
        return decode_user(token, db)
    def require_admin(user = Depends(get_current_user)):
        if not user.is_admin:
            raise HTTPException(403)
        return user

    @app.delete("/users/{id}")
    async def delete_user(id: int, admin = Depends(require_admin)):
        ...   # only runs if require_admin passed

    FastAPI resolves get_db -> get_current_user -> require_admin in order.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DEPENDENCY SCOPES & CACHING:
    Within ONE request, if the same dependency is declared multiple times,
    FastAPI CACHES the result (calls it once). Disable with
    Depends(get_x, use_cache=False) if you need a fresh value each time.

    Class-based dependencies (callable classes) and parameterized dependencies
    (a class with __call__) are used for configurable shared logic.

INTERVIEW ANSWER:
    "Dependency injection lets me declare reusable logic — DB sessions, auth,
    pagination — as functions resolved before the handler runs. yield-based
    dependencies manage resource lifecycle: the code before yield is setup,
    after yield is teardown that runs post-response, like closing a DB
    session. Dependencies can be nested — get_db feeds get_current_user feeds
    require_admin — and FastAPI resolves the whole graph. The biggest wins are
    testability, because I can override any dependency with a mock via
    dependency_overrides, and separation of concerns, keeping routes thin.
    FastAPI also caches a dependency's result within a single request."
'''


# =================================================================================
# SECTION 6: PYDANTIC v2 — Validation, Response Models, Settings
# =================================================================================
'''
Pydantic is the validation engine. The JD lists Pydantic-style validation
across agent I/O. Know v2 specifics.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

REQUEST vs RESPONSE MODELS:
    Use SEPARATE models for input and output. Critical for security.

    class UserCreate(BaseModel):       # request — what client sends
        username: str
        password: str                  # accept password IN

    class UserOut(BaseModel):          # response — what we return
        id: int
        username: str                  # NO password OUT (never leak it)

    @app.post("/users", response_model=UserOut)
    async def create(user: UserCreate):
        ...
        return saved_user
    -> response_model=UserOut FILTERS the output — even if saved_user has a
       password field, it's stripped. This prevents accidental data leaks.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

VALIDATION (Pydantic v2):
    from pydantic import BaseModel, Field, field_validator

    class Order(BaseModel):
        quantity: int = Field(gt=0, le=1000)
        email: str
        temperature: float = Field(ge=0.0, le=2.0, default=0.7)

        @field_validator("email")
        @classmethod
        def must_have_at(cls, v):
            if "@" not in v:
                raise ValueError("invalid email")
            return v

    KEY v2 facts:
    - Coerces types by default ("5" -> 5, "0.7" -> 0.7). Use strict mode to forbid.
    - field_validator (v2) replaces v1's validator.
    - model_validator for cross-field checks.
    - model_dump() / model_dump_json() (v2) replace .dict() / .json().
    - Field(default_factory=list) for mutable defaults — NEVER Field(default=[]).
    - v2 is rewritten in Rust (pydantic-core) — much faster than v1.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SETTINGS MANAGEMENT (12-factor config):
    from pydantic_settings import BaseSettings

    class Settings(BaseSettings):
        database_url: str
        openai_api_key: str
        debug: bool = False
        model_config = {"env_file": ".env"}

    settings = Settings()   # auto-loads + validates from env vars / .env
    -> Type-safe config from environment. Missing/invalid -> error at startup,
       not at runtime. This is how you manage secrets/config cleanly.

INTERVIEW ANSWER:
    "I always use separate request and response models. The response_model
    filters output, so sensitive fields like passwords never leak even if the
    DB object contains them. Pydantic v2 validates and coerces from type hints —
    I use Field constraints for bounds, field_validator for custom rules, and
    model_validator for cross-field checks. For mutable defaults I use
    default_factory, never a literal list. For config I use pydantic-settings
    BaseSettings, which loads and validates environment variables at startup,
    so a missing secret fails fast instead of at runtime."
'''


# =================================================================================
# SECTION 7: MIDDLEWARE — Cross-cutting Concerns
# =================================================================================
'''
Middleware wraps EVERY request/response — for logging, timing, CORS, auth.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HOW IT WORKS (onion model):
    Request flows IN through middleware (outer->inner), handler runs,
    response flows OUT (inner->outer). Each middleware sees both directions.

    @app.middleware("http")
    async def add_timing(request: Request, call_next):
        start = time.time()
        response = await call_next(request)   # pass to next layer/handler
        response.headers["X-Process-Time"] = str(time.time() - start)
        return response

    Code BEFORE call_next = runs on the way in.
    Code AFTER call_next  = runs on the way out.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

COMMON MIDDLEWARE:
    - CORSMiddleware — allow cross-origin requests (front-end on another domain).
    - GZipMiddleware — compress responses.
    - Custom: request logging, timing, request-ID injection, rate limiting,
      auth pre-checks.

    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(CORSMiddleware, allow_origins=["https://myapp.com"],
                       allow_methods=["*"], allow_headers=["*"])

MIDDLEWARE vs DEPENDENCY (common interview distinction):
    - Middleware: runs on EVERY request globally; good for cross-cutting
      concerns (logging, CORS, timing). Operates at the raw request/response level.
    - Dependency: runs per-ROUTE (or router); good for route-specific logic
      (auth for protected routes, DB session). Can return values into the handler.
    Rule: global + every request -> middleware. Route-specific + needs a value
    in the handler -> dependency.

INTERVIEW POINT:
    "Middleware wraps every request in an onion model — code before call_next
    runs inbound, after runs outbound. I use it for global cross-cutting
    concerns: CORS, gzip, request logging, timing, request IDs. For
    route-specific logic that returns a value into the handler — like auth or
    a DB session — I use a dependency instead. Global vs per-route is the
    deciding line."
'''


# =================================================================================
# SECTION 8: BACKGROUND TASKS vs REAL TASK QUEUES
# =================================================================================
'''
"How do you handle work that shouldn't block the response?" — common question.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FASTAPI BackgroundTasks (lightweight, in-process):
    from fastapi import BackgroundTasks

    @app.post("/signup")
    async def signup(email: str, bg: BackgroundTasks):
        create_user(email)
        bg.add_task(send_welcome_email, email)  # runs AFTER response sent
        return {"status": "ok"}                  # client gets response immediately

    - Runs in the SAME process AFTER the response is returned.
    - Good for: quick, fire-and-forget tasks (send an email, log, cleanup).
    - BAD for: long/heavy tasks, anything that must survive a crash, retries,
      or scaling across machines.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

REAL TASK QUEUE (Celery / RQ / Arq + Redis/RabbitMQ):
    For serious async work:
    - Producer (FastAPI) pushes a job to a broker (Redis/RabbitMQ).
    - Worker processes (separate) pick up and execute jobs.
    - Benefits: survives API restarts, retries, scheduling, scales horizontally,
      monitoring, doesn't consume API worker resources.
    - Use for: document ingestion/embedding, long LLM pipelines, report
      generation, anything > a few seconds or that needs reliability.

THE DECISION:
    Quick + fire-and-forget + okay to lose on crash -> BackgroundTasks.
    Long + reliable + retryable + scalable           -> Celery/Arq + broker.

INTERVIEW ANSWER:
    "For quick fire-and-forget work like sending a welcome email, I use
    FastAPI's BackgroundTasks — it runs after the response in the same
    process, so the client isn't blocked. But for heavy or critical work —
    document embedding, long LLM pipelines, anything that needs retries or
    must survive a crash — I use a real task queue like Celery or Arq with
    Redis. The API just enqueues the job; separate workers process it, which
    keeps API workers free and lets the work scale and retry independently."
'''


# =================================================================================
# SECTION 9: ERROR HANDLING & EXCEPTION HANDLERS
# =================================================================================
'''
Production APIs need consistent, safe error responses.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RAISING HTTP ERRORS:
    from fastapi import HTTPException

    @app.get("/items/{id}")
    async def get_item(id: int):
        item = db.get(id)
        if not item:
            raise HTTPException(status_code=404, detail="Item not found")
        return item

CUSTOM EXCEPTION HANDLERS (global, consistent error shape):
    from fastapi.responses import JSONResponse

    class InsufficientCreditError(Exception):
        def __init__(self, needed): self.needed = needed

    @app.exception_handler(InsufficientCreditError)
    async def credit_handler(request, exc):
        return JSONResponse(status_code=402,
                            content={"error": "insufficient_credit", "needed": exc.needed})

    -> Any route can 'raise InsufficientCreditError(50)' and get a consistent
       402 response. Domain errors map to HTTP responses in ONE place.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BEST PRACTICES (senior):
    - Consistent error envelope: {"error": code, "message": ..., "detail": ...}.
    - NEVER leak stack traces / internal details to the client. Log them
      server-side, return a generic message + a correlation/request ID.
    - Validation errors -> 422 (FastAPI handles automatically).
    - Map domain exceptions to HTTP codes via exception handlers, not in every route.
    - 4xx = client error, 5xx = server error. Use the right code.

INTERVIEW POINT:
    "I raise HTTPException for expected errors like 404, and register global
    exception handlers to map domain exceptions to consistent HTTP responses
    in one place. I never leak stack traces to clients — those are logged
    server-side with a request ID, and the client gets a clean message. A
    consistent error envelope across the API makes it predictable for consumers."
'''


# =================================================================================
# SECTION 10: AUTHENTICATION & SECURITY
# =================================================================================
'''
A lead must secure APIs. Common in HashedIn rounds (the JD stresses secure APIs).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

AUTH APPROACHES:
    1. API KEY — simple; client sends a key in a header. Good for service-to-service.
    2. JWT (JSON Web Token) — stateless; token holds claims (user id, roles),
       signed by the server. Client sends it in Authorization: Bearer <token>.
    3. OAuth2 — delegated auth (login with Google, etc.); FastAPI has
       OAuth2PasswordBearer built in.

JWT FLOW (most common for APIs):
    1. User logs in (username/password) -> server verifies -> issues a signed JWT.
    2. Client stores it, sends it on every request in the Authorization header.
    3. Server VERIFIES the signature + expiry on each request (no DB lookup needed
       — that's the "stateless" benefit), extracts user/roles from claims.

    Implement as a dependency:
        oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
        async def get_current_user(token: str = Depends(oauth2_scheme)):
            payload = jwt.decode(token, SECRET, algorithms=["HS256"])  # verifies
            return get_user(payload["sub"])
    -> Protect any route with: user = Depends(get_current_user)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SECURITY CHECKLIST (mention these — shows production maturity):
    - HTTPS only (TLS); never send tokens over HTTP.
    - Hash passwords with bcrypt/argon2 — NEVER store plaintext.
    - Validate ALL input (Pydantic does this) — prevents injection.
    - Use parameterized queries / ORM — prevents SQL injection.
    - Rate limiting (slowapi, API Gateway) — prevents abuse/DoS.
    - CORS configured to specific origins — not "*".
    - Secrets in env vars / vault — never in code or git.
    - Principle of least privilege for tokens/scopes.
    - Short-lived access tokens + refresh tokens.

INTERVIEW ANSWER:
    "For API auth I typically use JWT — the server issues a signed token on
    login, the client sends it as a Bearer token, and I verify the signature
    and expiry on each request via a dependency, which is stateless so it
    needs no DB lookup. Roles live in the token claims for authorization. On
    security I enforce HTTPS, hash passwords with bcrypt, rely on Pydantic for
    input validation and parameterized queries to prevent injection, add rate
    limiting, lock CORS to specific origins, and keep secrets in env vars or a
    vault. Access tokens are short-lived with refresh tokens."
'''


# =================================================================================
# SECTION 11: TESTING FASTAPI (pytest + TestClient/httpx)
# =================================================================================
'''
The JD explicitly wants pytest + unit/functional/integration tests.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TESTING AN ENDPOINT:
    from fastapi.testclient import TestClient
    from myapp import app

    client = TestClient(app)

    def test_create_item():
        resp = client.post("/items/5", json={"name": "Widget", "price": 9.99})
        assert resp.status_code == 200
        assert resp.json()["item"]["name"] == "Widget"

    For ASYNC tests: use httpx.AsyncClient + pytest-asyncio.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

OVERRIDING DEPENDENCIES IN TESTS (the killer feature):
    app.dependency_overrides[get_db] = lambda: fake_test_db
    -> Now tests use an in-memory/fake DB without touching route code.
       This is WHY dependency injection matters for testability (Section 5).

THE TEST PYRAMID:
    - UNIT tests: individual functions/logic in isolation (mock dependencies). Many.
    - INTEGRATION tests: endpoint + real DB/services together. Fewer.
    - FUNCTIONAL/E2E tests: full flow through the API. Fewest.
    Most tests should be fast unit tests; fewer slow integration/E2E.

PYTEST ESSENTIALS:
    - Fixtures (@pytest.fixture) for setup/teardown (test DB, client).
    - @pytest.mark.parametrize for table-driven tests.
    - Mocking with unittest.mock / pytest-mock for external calls (LLM APIs).
    - conftest.py for shared fixtures across test files.
    - pytest-cov for coverage.

INTERVIEW POINT:
    "I test FastAPI with pytest and the TestClient, and httpx AsyncClient with
    pytest-asyncio for async paths. The key enabler is dependency_overrides —
    I swap the real DB or LLM client for a fake in tests without changing route
    code. I follow the test pyramid: many fast unit tests with mocked
    dependencies, fewer integration tests against a real test DB, and a few
    end-to-end tests. Fixtures handle setup/teardown and parametrize gives
    table-driven coverage."
'''


# =================================================================================
# SECTION 12: PRODUCTION DEPLOYMENT (ASGI, uvicorn, gunicorn, workers)
# =================================================================================
'''
"How do you run FastAPI in production?" — leads must answer this.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE STACK:
    - ASGI server: uvicorn (the async server that runs the app).
    - Process manager: gunicorn with uvicorn workers, OR uvicorn --workers.
      gunicorn -k uvicorn.workers.UvicornWorker -w 4 myapp:app
    - WHY multiple workers: each worker = one process = one event loop = one
      core. To use all CPU cores, run N workers (rule of thumb: 2*cores+1,
      but tune for your workload). Async handles concurrency WITHIN a worker;
      workers give PARALLELISM across cores.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CONCURRENCY vs PARALLELISM IN DEPLOYMENT (clarify this — common confusion):
    - WITHIN a worker: async event loop = concurrency (many waits interleaved).
    - ACROSS workers: multiple processes = parallelism (multiple cores).
    Production = multiple async workers = concurrency * parallelism.

PRODUCTION CHECKLIST:
    - Containerize (Docker), orchestrate (Kubernetes) for autoscaling.
    - Put behind a reverse proxy / load balancer (nginx, ALB).
    - Health check endpoint (/health) for readiness/liveness probes.
    - Structured logging + request IDs + tracing (OpenTelemetry).
    - Graceful shutdown (finish in-flight requests on deploy).
    - Connection pooling for DB; limits on concurrent LLM calls.
    - Env-based config (pydantic-settings), secrets from vault.

INTERVIEW ANSWER:
    "In production I run FastAPI under uvicorn workers managed by gunicorn —
    each worker is its own process with its own event loop pinned to a core.
    Async gives concurrency within a worker; multiple workers give parallelism
    across cores, so I size workers to the CPU and workload. I containerize
    with Docker, run on Kubernetes for autoscaling, put it behind a load
    balancer, expose a health endpoint for probes, add structured logging with
    request IDs and tracing, use DB connection pooling, and handle graceful
    shutdown so in-flight requests complete during deploys."
'''


# =================================================================================
# SECTION 13: INTERVIEW Q&A (senior-level, with crisp answers)
# =================================================================================
'''
Rapid-fire. These are the high-probability HashedIn/Deloitte FastAPI questions.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q1. Why FastAPI over Flask/Django REST?
A:  "Native async on ASGI, automatic Pydantic validation, auto OpenAPI docs.
    Best for I/O-bound and AI APIs. DRF when I need batteries-included ORM/admin."

Q2. What is ASGI and how does it differ from WSGI?
A:  "WSGI is the synchronous Python web standard — one request per worker at a
    time. ASGI is the async successor — it supports concurrent requests,
    WebSockets, and long-lived connections on an event loop. FastAPI is ASGI."

Q3. async def vs def in a FastAPI endpoint?
A:  "async def runs on the event loop — use await for I/O. def runs in a thread
    pool so blocking code doesn't freeze the loop. The danger is a blocking
    call INSIDE async def — that stalls every request."

Q4. How does FastAPI prevent the event loop from blocking on sync code?
A:  "It runs plain def endpoints and sync dependencies in an external thread
    pool. For blocking calls inside async code I use asyncio.to_thread."

Q5. What is dependency injection used for?
A:  "Reusable shared logic — DB sessions, auth, pagination — resolved before
    the handler. yield dependencies manage resource lifecycle. Biggest win is
    testability via dependency_overrides."

Q6. How do you handle DB sessions per request?
A:  "A yield-based dependency: create the session before yield, close it in
    finally after the response. Injected into routes via Depends."

Q7. Request model vs response model — why separate?
A:  "Security and clarity. response_model filters output so fields like
    passwords never leak even if the DB object has them."

Q8. How do you run long-running work without blocking the response?
A:  "BackgroundTasks for quick fire-and-forget; a real queue like Celery/Arq
    for heavy, retryable, scalable work."

Q9. How do you scale FastAPI?
A:  "Multiple uvicorn workers under gunicorn for cross-core parallelism, async
    for in-worker concurrency, containerized on K8s with autoscaling behind a
    load balancer, DB connection pooling."

Q10. How do you secure a FastAPI service?
A:  "JWT bearer auth verified in a dependency, HTTPS, bcrypt password hashing,
    Pydantic input validation, parameterized queries, rate limiting, scoped
    CORS, secrets in a vault."

Q11. How do you test it?
A:  "pytest + TestClient (httpx AsyncClient for async), dependency_overrides to
    inject fakes, test pyramid: many unit, fewer integration, few E2E."

Q12. A request is slow under load — how do you debug?
A:  "First check for a blocking call inside an async endpoint — the usual
    culprit. Then DB query performance (N+1, missing index), external API
    latency, worker count, and connection pool exhaustion. Add timing
    middleware and tracing to localize it."

Q13. How would you design a FastAPI service that calls an LLM?
A:  "Async endpoint, async LLM client (await the call), Pydantic request/
    response models, a dependency for the LLM client, timeouts + retries,
    rate-limit concurrent calls, stream the response with StreamingResponse,
    and offload heavy pre/post-processing to a queue."

Q14. What is StreamingResponse used for?
A:  "Streaming data to the client incrementally — e.g., streaming LLM tokens
    as they're generated (Server-Sent Events) instead of waiting for the full
    response. Big UX win for chat."
'''


# =================================================================================
# SECTION 14: GOLDEN LESSONS
# =================================================================================
'''
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GOLDEN LESSON 1 — ASYNC IS CONCURRENCY, NOT PARALLELISM.
    One event loop interleaves I/O waits. Workers give parallelism across
    cores. Production = multiple async workers. Say this precisely.

GOLDEN LESSON 2 — NEVER BLOCK THE EVENT LOOP.
    A blocking call (time.sleep, sync requests, heavy CPU) inside async def
    freezes ALL requests. This is THE FastAPI trap. Use async libs,
    asyncio.to_thread, or a plain def endpoint.

GOLDEN LESSON 3 — DI EXISTS FOR REUSE + TESTABILITY + LIFECYCLE.
    yield dependencies for setup/teardown; dependency_overrides for testing.
    This is FastAPI's most-tested concept.

GOLDEN LESSON 4 — SEPARATE REQUEST AND RESPONSE MODELS.
    response_model filters output and prevents data leaks. Always.

GOLDEN LESSON 5 — VALIDATION IS FREE — USE IT.
    Type hints + Pydantic reject bad input with a 422 before your code runs.
    No manual checks. Use Field constraints and validators.

GOLDEN LESSON 6 — MIDDLEWARE = GLOBAL, DEPENDENCY = PER-ROUTE.
    Cross-cutting (logging, CORS) -> middleware. Route logic returning a value
    -> dependency.

GOLDEN LESSON 7 — KNOW THE REQUEST LIFECYCLE COLD.
    Server -> middleware -> routing -> dependencies -> validation -> handler
    (async on loop / sync in thread pool) -> response model -> middleware -> out.

GOLDEN LESSON 8 — BACKGROUNDTASKS FOR QUICK, QUEUE FOR SERIOUS.
    Fire-and-forget vs reliable/retryable/scalable. Know the line.

GOLDEN LESSON 9 — EVERY CHOICE IS A TRADE-OFF (the JD's #1 theme).
    Frame answers as "X because Y, trade-off Z, at scale W." This is a LEAD role.

GOLDEN LESSON 10 — TIE IT TO AI.
    FastAPI + async shines for LLM APIs: one worker handles thousands of
    concurrent model calls. Stream tokens with StreamingResponse. Connect
    every FastAPI answer back to the AI use case in this JD.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ONE-LINE CHEAT SHEET:
    FastAPI = ASGI + Starlette + Pydantic (async + validation + docs).
    async def = event loop; def = thread pool; never block the loop.
    DI = reuse + testability + lifecycle (yield + dependency_overrides).
    Models: separate request/response; response_model filters output.
    Background: BackgroundTasks (quick) vs Celery/Arq (serious).
    Prod: uvicorn workers + gunicorn; concurrency (async) x parallelism (workers).
    Security: JWT + HTTPS + bcrypt + validation + rate limit + scoped CORS.
    Test: pytest + TestClient + dependency_overrides; test pyramid.
'''


# =================================================================================
# RUN SUMMARY
# =================================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 1 — FASTAPI MASTERY")
    print("=" * 70)
    print()
    print("THE 3 PILLARS: ASGI (async) + Pydantic (validation) + OpenAPI (docs)")
    print()
    print("MUST-KNOW COLD:")
    print("  1. Request lifecycle: server->middleware->routing->deps->")
    print("     validation->handler->response_model->middleware->out")
    print("  2. async def = event loop | def = thread pool | NEVER block the loop")
    print("  3. async = concurrency (1 loop); workers = parallelism (N cores)")
    print("  4. DI = reuse + testability (dependency_overrides) + lifecycle (yield)")
    print("  5. Separate request/response models; response_model filters output")
    print()
    print("THE LEAD-ROLE HABIT: every answer = 'X because Y, trade-off Z, at scale W'")
    print()
    print("THE TRAP TO AVOID: a blocking call inside async def freezes ALL requests.")
    print()
    print("=" * 70)
    print("Next: Phase 2 — Async Python & Concurrency Deep Dive")
    print("=" * 70)
