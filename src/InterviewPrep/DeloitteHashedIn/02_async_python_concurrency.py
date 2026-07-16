"""
===================================================================================
PHASE 2 — ASYNC PYTHON & CONCURRENCY DEEP DIVE (HashedIn by Deloitte)
===================================================================================

WHY THIS IS PHASE 2:
    The JD calls out asyncio EXPLICITLY: "hands-on experience using asyncio to
    build scalable, I/O-bound services." Phase 1 (FastAPI) sits ON TOP of this.
    This lesson is the FOUNDATION — the event loop, coroutines, and the
    concurrency model that makes your FastAPI answers bulletproof.

    Interviewers drill here because it's where most candidates have shallow,
    memorized knowledge. A lead must explain HOW the event loop works, not
    just "async makes it faster."

DEPTH LEVEL: Senior/Lead. The "why" and "how it works internally."

SECTIONS:
    1.  Concurrency vs Parallelism vs Async (the foundational distinction)
    2.  The GIL — What It Is and Why It Shapes Everything
    3.  Threading vs Multiprocessing vs Asyncio — When to Use Which
    4.  Coroutines, async/await — How They Actually Work
    5.  The Event Loop — The Engine Underneath
    6.  Tasks, gather, and Concurrent Execution
    7.  Blocking the Loop — The #1 Production Bug
    8.  Mixing Sync and Async (to_thread, run_in_executor)
    9.  Async Patterns for AI/LLM Workloads
    10. Common Async Gotchas & Pitfalls
    11. Interview Q&A (senior-level)
    12. GOLDEN LESSONS
===================================================================================
"""


# =================================================================================
# SECTION 1: CONCURRENCY vs PARALLELISM vs ASYNC (the foundational distinction)
# =================================================================================
'''
If you confuse these, every async answer falls apart. Nail this first.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CONCURRENCY:
    Dealing with MANY things at once by INTERLEAVING them. Tasks make progress
    by taking turns. On a single core, only one runs at any instant, but they
    switch so fast it looks simultaneous.
    Analogy: one chef juggling 3 dishes — stir one, while it simmers start
    another. One person, overlapping work.

PARALLELISM:
    Doing MANY things at the SAME literal instant on MULTIPLE cores.
    Analogy: 3 chefs, 3 dishes, all cooking simultaneously.

ASYNC (asyncio):
    A SPECIFIC FORM of concurrency — COOPERATIVE multitasking on a SINGLE
    thread. Tasks voluntarily YIELD control (at 'await') when they hit I/O,
    letting other tasks run during the wait. No OS thread switching.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE KEY TABLE:

    MODEL            HOW                       CORES   BEST FOR
    Asyncio          1 thread, cooperative     1       I/O-bound, high concurrency
                     yield at await
    Threading        N OS threads,             1*      I/O-bound (GIL released
                     preemptive switching              during I/O)
    Multiprocessing  N processes, true         N       CPU-bound (separate GILs)
                     parallel

    *Threading uses multiple cores for I/O waits but NOT for Python bytecode
     (GIL — see Section 2).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

COOPERATIVE vs PREEMPTIVE (a senior distinction):
    - Asyncio = COOPERATIVE: a task runs until IT decides to yield (await).
      If a task never awaits (e.g., a CPU loop), it HOGS the loop — nothing
      else runs. You are responsible for yielding.
    - Threading = PREEMPTIVE: the OS forcibly switches threads on a timer.
      A thread can't hog the CPU forever — but you get race conditions.

INTERVIEW ANSWER:
    "Concurrency is interleaving many tasks so they make progress by taking
    turns; parallelism is literally running them at the same instant on
    multiple cores. Asyncio is a specific kind of concurrency — cooperative
    multitasking on a single thread where tasks yield control at await during
    I/O. That's different from threading, which is preemptive — the OS
    switches threads — and from multiprocessing, which is true parallelism
    across cores. For I/O-bound work asyncio is ideal; for CPU-bound work you
    need multiprocessing."
'''


# =================================================================================
# SECTION 2: THE GIL — What It Is and Why It Shapes Everything
# =================================================================================
'''
The GIL is the reason Python concurrency works the way it does. Lead-level
interviewers ALWAYS probe this. Be precise — most people get it half-right.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHAT THE GIL IS:
    Global Interpreter Lock — a mutex in CPython that allows only ONE thread
    to execute Python BYTECODE at a time, even on a multi-core machine.
    It exists because CPython's memory management (reference counting) isn't
    thread-safe; the GIL protects it cheaply.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE CONSEQUENCE (this is what matters):
    - CPU-BOUND multithreading gets NO speedup — threads can't run Python
      bytecode in parallel. 4 threads doing math = same speed as 1 (often
      slower due to lock contention).
    - I/O-BOUND multithreading DOES help — the GIL is RELEASED during I/O
      waits (network, disk, sleep). So while thread A waits on a socket,
      thread B runs Python. This is why threading helps I/O.
    - Asyncio sidesteps the GIL debate entirely — it's single-threaded, so
      there's no contention; it just interleaves I/O waits cooperatively.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HOW TO GET TRUE CPU PARALLELISM:
    - multiprocessing — each process has its OWN interpreter + GIL. True
      parallel CPU. Cost: process spawn overhead + IPC (pickling data).
    - C extensions (NumPy, Pandas) — release the GIL during heavy native
      computation, so they parallelize internally.
    - Subinterpreters / no-GIL builds — Python 3.13 has an EXPERIMENTAL
      free-threaded (no-GIL) build, but assume the GIL exists in interviews
      unless told otherwise.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE COMMON MISCONCEPTION TO AVOID:
    WRONG: "The GIL makes Python single-threaded / Python can't do threads."
    RIGHT: "Python CAN run many threads, but only one executes Python
    bytecode at a time. Threads still help I/O-bound work because the GIL is
    released during I/O. It only blocks CPU-bound parallelism."

INTERVIEW ANSWER:
    "The GIL is a mutex in CPython that lets only one thread run Python
    bytecode at a time, because reference-counting memory management isn't
    thread-safe. The consequence: threading gives no speedup for CPU-bound
    work, but it DOES help I/O-bound work because the GIL is released during
    I/O waits. For true CPU parallelism I use multiprocessing — separate
    processes with separate GILs — or rely on NumPy/Pandas which release the
    GIL in native code. Asyncio avoids the whole issue by being
    single-threaded and cooperative."
'''


# =================================================================================
# SECTION 3: THREADING vs MULTIPROCESSING vs ASYNCIO — When to Use Which
# =================================================================================
'''
The decision question. A lead must pick the right model and justify it.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE DECISION MATRIX:

    WORKLOAD TYPE          BEST CHOICE        WHY
    I/O-bound, high count   asyncio            1 thread, thousands of concurrent
    (1000s of API/DB calls)                    awaits, lowest overhead
    I/O-bound, moderate     threading          simpler if libraries are sync-only;
    (using sync libraries)                     GIL released during I/O
    CPU-bound               multiprocessing    true parallelism, bypasses GIL
    (math, image, parsing)
    Mixed                   asyncio + process   async for I/O, offload CPU to
                            pool                a ProcessPoolExecutor

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

OVERHEAD COMPARISON:
    asyncio        — cheapest: no thread/process, just coroutine objects.
                     Can run tens of thousands of concurrent tasks.
    threading      — moderate: each OS thread ~8MB stack; context-switch cost;
                     practical limit hundreds-to-low-thousands.
    multiprocessing— heaviest: each process is a full interpreter; spawn cost;
                     data must be pickled across the IPC boundary.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CONCRETE EXAMPLES:
    - Calling 500 LLM/REST APIs        -> asyncio (await all with gather).
    - Scraping 1000 URLs               -> asyncio (httpx) or threads.
    - Resizing 10000 images            -> multiprocessing (CPU-bound).
    - Training/number-crunching        -> multiprocessing / NumPy / GPU.
    - A FastAPI service calling a DB    -> asyncio (async driver).

THE NUANCE FOR LEADS:
    "asyncio requires async-compatible libraries end-to-end. If a critical
    library is sync-only (no async driver), threading may be simpler than
    rewriting everything. The model is only as async as its slowest sync call."

INTERVIEW ANSWER:
    "I match the model to the bottleneck. I/O-bound with many concurrent calls —
    like fanning out to LLM or REST APIs — asyncio, because one thread handles
    thousands of awaits with the least overhead. I/O-bound but the libraries
    are sync-only — threading, since the GIL releases during I/O and I don't
    have to rewrite for async. CPU-bound — multiprocessing for true parallelism
    across cores. For a mixed workload I run async for I/O and offload CPU work
    to a process pool executor. The caveat is asyncio needs async libraries all
    the way down."
'''


# =================================================================================
# SECTION 4: COROUTINES, async/await — How They Actually Work
# =================================================================================
'''
Don't just say "async makes it faster." Explain the MECHANISM.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHAT A COROUTINE IS:
    'async def' defines a COROUTINE FUNCTION. CALLING it does NOT run it —
    it returns a COROUTINE OBJECT (a pausable computation). It only runs when
    driven by the event loop (await it, or schedule it as a task).

    async def fetch():
        return 42

    coro = fetch()      # NOT run yet — coro is a coroutine object
    result = await coro # NOW it runs (inside an async context)

    THE #1 BEGINNER BUG: calling an async function and forgetting to await it.
    You get a coroutine object (and a "coroutine was never awaited" warning),
    not the result.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHAT 'await' DOES (the core mechanism):
    'await' means: "pause THIS coroutine here, hand control back to the event
    loop, and let other tasks run until this awaitable is ready."
    When the awaited I/O completes, the loop RESUMES this coroutine right
    where it paused.

    async def handler():
        data = await db.fetch()    # PAUSE here; loop runs other tasks
        # ... resumes here when db.fetch() completes
        return process(data)

    This pause/resume is what enables concurrency on ONE thread: while one
    coroutine waits, others use the freed-up loop.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

YOU CAN ONLY await AN AWAITABLE:
    Awaitables = coroutines, Tasks, Futures. You await things that represent
    "a value that will be ready later." You CANNOT await a normal function.

ENTERING THE ASYNC WORLD:
    asyncio.run(main())   # creates the loop, runs main() coroutine, closes loop.
    'await' is only valid INSIDE an 'async def'. Top-level code uses
    asyncio.run() to bootstrap.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

UNDER THE HOOD (for the deep probe):
    Coroutines are built on GENERATORS — the pause/resume is the same machinery
    as yield. 'await' suspends the coroutine and returns control up to the loop,
    which holds a reference to resume it later. This is why it's "cooperative" —
    the coroutine chooses its pause points (the awaits).

INTERVIEW ANSWER:
    "An async def is a coroutine function — calling it returns a coroutine
    object that doesn't run until the event loop drives it. await is the key:
    it pauses the current coroutine, returns control to the loop so other
    tasks run, and resumes this one when the awaited I/O completes. That
    pause-and-resume on a single thread is what gives concurrency. Under the
    hood it's the same suspend/resume machinery as generators. The classic bug
    is calling an async function without awaiting it — you get a coroutine
    object, not the result."
'''


# =================================================================================
# SECTION 5: THE EVENT LOOP — The Engine Underneath
# =================================================================================
'''
"What is the event loop actually doing?" — the deep-dive that earns respect.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHAT THE EVENT LOOP IS:
    A single-threaded scheduler that runs in a loop:
    1. Picks a ready task and runs it until it hits 'await' (yields).
    2. Registers what that task is waiting on (a socket, timer, etc.) with the
       OS (via selectors — epoll/kqueue/IOCP).
    3. Runs the next ready task. Repeat.
    4. When the OS signals an awaited resource is ready, the loop marks that
       task ready again and resumes it on a future iteration.

    It's a "while True" that juggles tasks by their readiness, never blocking
    on any single one (as long as they cooperate by awaiting).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE CRITICAL PROPERTY:
    The loop runs on ONE thread. So if a task does NOT yield (a long CPU
    computation or a blocking sync call), the loop is STUCK — no other task
    can run. The loop's power AND its danger both come from being single-threaded.

    This is the mechanistic reason behind "never block the event loop"
    (Section 7).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

KEY APIs:
    asyncio.run(coro)        -> create loop, run coro to completion, close loop.
    asyncio.get_event_loop() -> get the current loop (rarely needed directly now).
    loop.run_in_executor()   -> offload blocking work to a thread/process pool.
    One loop PER thread. FastAPI/uvicorn manage the loop for you.

INTERVIEW ANSWER:
    "The event loop is a single-threaded scheduler. It runs a task until it
    awaits, registers what it's waiting on with the OS through selectors like
    epoll, then runs the next ready task. When the OS signals an awaited
    resource is ready, the loop resumes that task. Because it's one thread,
    everything depends on tasks cooperating by awaiting — if one task blocks or
    never yields, the whole loop stalls. That single-threaded design is both
    why it's efficient for I/O and why blocking it is catastrophic."
'''


# =================================================================================
# SECTION 6: TASKS, gather, and Concurrent Execution
# =================================================================================
'''
The difference between SEQUENTIAL awaits and CONCURRENT execution — a top
interview trap.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SEQUENTIAL (slow — a common mistake):
    async def slow():
        a = await fetch_user()      # wait 200ms
        b = await fetch_orders()    # THEN wait 200ms
        c = await fetch_recs()      # THEN wait 200ms
        return a, b, c              # total ~600ms

    Each await completes before the next STARTS. They don't overlap.
    This is correct only if b depends on a, etc.

CONCURRENT (fast — gather):
    async def fast():
        a, b, c = await asyncio.gather(
            fetch_user(), fetch_orders(), fetch_recs()
        )
        return a, b, c              # total ~200ms (all overlap)

    gather schedules all three coroutines, they run concurrently (interleaving
    their I/O waits), and gather returns when ALL complete. Results come back
    IN ARGUMENT ORDER regardless of which finished first.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TASKS (fire-and-track):
    asyncio.create_task(coro) schedules a coroutine to run on the loop
    IMMEDIATELY (concurrently) and returns a Task handle you can await later.

    task = asyncio.create_task(background_work())   # starts running now
    do_other_stuff()
    result = await task                              # collect when needed

    gather vs create_task:
    - create_task: schedule one coroutine concurrently, get a handle.
    - gather: schedule MANY and wait for all (it uses tasks internally).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ERROR HANDLING IN gather:
    - Default: if any coroutine raises, gather raises that exception (others
      keep running but their results are lost).
    - return_exceptions=True: gather returns exceptions as results instead of
      raising, so one failure doesn't kill the batch.
        results = await asyncio.gather(*tasks, return_exceptions=True)
        -> inspect each result; some may be Exception objects.

OTHER USEFUL PRIMITIVES:
    - asyncio.wait_for(coro, timeout) -> enforce a timeout (raises TimeoutError).
    - asyncio.as_completed(tasks)     -> process results as they finish.
    - asyncio.Semaphore(n)            -> limit concurrency (e.g., max 10
      simultaneous LLM calls to respect rate limits).
    - asyncio.TaskGroup (3.11+)       -> structured concurrency; cleaner than
      gather, auto-cancels siblings on failure.

INTERVIEW ANSWER:
    "Sequential awaits run one after another — three 200ms calls take 600ms.
    To run them concurrently I use asyncio.gather, which schedules them all
    and returns when every one finishes, so it's ~200ms total, with results in
    argument order. create_task schedules a single coroutine to run
    immediately and hands me a Task to await later. For resilience I use
    gather with return_exceptions=True so one failure doesn't kill the batch,
    wait_for for timeouts, and a Semaphore to cap concurrency — for example,
    limiting simultaneous LLM calls to respect rate limits."
'''


# =================================================================================
# SECTION 7: BLOCKING THE LOOP — The #1 Production Bug
# =================================================================================
'''
This is the single most important practical async lesson. It connects directly
to FastAPI (Phase 1).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHAT "BLOCKING THE LOOP" MEANS:
    The event loop is ONE thread. If a coroutine runs code that does NOT yield
    (await) — either a blocking sync call or a long CPU computation — the loop
    cannot switch to other tasks. EVERY other request/task freezes until it
    returns.

THE THREE WAYS PEOPLE BLOCK THE LOOP:

    1. BLOCKING SYNC I/O inside async:
        async def bad():
            time.sleep(5)            # BLOCKS loop 5s
            r = requests.get(url)    # BLOCKS loop (sync HTTP lib)
        FIX: await asyncio.sleep(5); use httpx.AsyncClient.

    2. CPU-BOUND work inside async:
        async def bad():
            result = sum(i*i for i in range(10**8))  # BLOCKS loop
        FIX: offload -> await asyncio.to_thread(...) or a ProcessPoolExecutor.

    3. A SYNC DATABASE DRIVER inside async:
        async def bad():
            rows = sync_db.query(...)   # BLOCKS loop
        FIX: use an async driver (asyncpg) or run in a thread.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HOW TO DETECT IT:
    - Symptom: under load, ALL requests get slow/time out together (not just
      the heavy one) — the tell-tale sign of a blocked loop.
    - asyncio debug mode (asyncio.run(main(), debug=True)) logs slow callbacks
      that block the loop too long.
    - Profiling / tracing shows one coroutine holding the loop.

THE GOLDEN RULE:
    Inside async code, EVERYTHING must either be fast-and-non-blocking, or be
    awaited (async), or be offloaded to a thread/process pool. No exceptions.

INTERVIEW ANSWER:
    "Blocking the loop is the number one async production bug. The event loop
    is single-threaded, so any non-yielding code — a sync sleep, the sync
    requests library, a CPU-heavy loop, or a sync DB driver inside an async
    function — freezes every other task until it returns. The tell-tale
    symptom is that under load ALL requests slow down together, not just the
    heavy one. The fix is to use async libraries, await asyncio.sleep, or
    offload blocking and CPU work with asyncio.to_thread or a process pool.
    The rule: inside async code, everything is non-blocking, awaited, or
    offloaded."
'''


# =================================================================================
# SECTION 8: MIXING SYNC AND ASYNC (to_thread, run_in_executor)
# =================================================================================
'''
Real systems have sync libraries. A lead must bridge the two worlds cleanly.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RUNNING SYNC CODE FROM ASYNC (don't block the loop):
    # asyncio.to_thread (3.9+) — simplest, runs sync fn in a thread pool:
    result = await asyncio.to_thread(blocking_function, arg1, arg2)

    # loop.run_in_executor — more control (custom pool):
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(thread_pool, blocking_function, arg)
    # Use a ProcessPoolExecutor for CPU-bound to get true parallelism.

    The loop stays free while the blocking call runs in the pool; you await
    its completion. This is how you call a sync library from async safely.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RUNNING ASYNC CODE FROM SYNC:
    asyncio.run(async_function())   # from a sync context, bootstraps a loop.
    Caveat: you can't call asyncio.run if a loop is already running (e.g.,
    inside Jupyter or an async handler) — use the existing loop or nest_asyncio.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE DECISION:
    Blocking I/O (sync lib you can't replace)  -> await asyncio.to_thread(...)
    CPU-bound work                              -> run_in_executor(ProcessPool, ...)
    Calling async from a script/sync entrypoint -> asyncio.run(...)

INTERVIEW POINT:
    "To call a blocking sync library from async without freezing the loop, I
    use asyncio.to_thread, which runs it in a thread pool and lets me await
    the result. For CPU-bound work I use run_in_executor with a
    ProcessPoolExecutor to get real parallelism. To go the other way — call
    async from a sync entrypoint — I use asyncio.run to bootstrap a loop."
'''


# =================================================================================
# SECTION 9: ASYNC PATTERNS FOR AI/LLM WORKLOADS
# =================================================================================
'''
Tie async directly to the GenAI use case in the JD — this is your edge.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHY ASYNC IS PERFECT FOR LLM APPS:
    LLM calls are I/O-bound — you send a request and WAIT seconds for the
    model to respond. During that wait, async frees the worker to handle other
    requests. One worker can have hundreds of LLM calls in flight concurrently.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PATTERN 1 — FAN-OUT (parallel LLM/embedding calls):
    # Embed 100 documents concurrently instead of one-by-one:
    async def embed_all(docs):
        tasks = [embed_async(doc) for doc in docs]
        return await asyncio.gather(*tasks)
    -> 100 sequential calls at 200ms = 20s; concurrent = ~200ms-2s.

PATTERN 2 — CONCURRENCY LIMIT (respect rate limits):
    sem = asyncio.Semaphore(10)            # max 10 in flight
    async def embed_limited(doc):
        async with sem:
            return await embed_async(doc)
    tasks = [embed_limited(d) for d in docs]
    results = await asyncio.gather(*tasks)
    -> Prevents hammering the API past its rate limit / 429s.

PATTERN 3 — TIMEOUT + RETRY (resilience):
    async def call_llm_safe(prompt):
        for attempt in range(3):
            try:
                return await asyncio.wait_for(call_llm(prompt), timeout=30)
            except (asyncio.TimeoutError, RateLimitError):
                await asyncio.sleep(2 ** attempt)   # backoff
        raise RuntimeError("LLM failed after retries")

PATTERN 4 — STREAMING TOKENS (async generator -> SSE):
    async def stream_llm(prompt):
        async for chunk in llm.astream(prompt):     # async generator
            yield chunk.content
    # In FastAPI: return StreamingResponse(stream_llm(prompt))
    -> User sees tokens appear live instead of waiting for the full answer.

PATTERN 5 — PARALLEL AGENT TOOLS:
    # An agent that needs results from retriever + web search + a DB:
    docs, web, data = await asyncio.gather(
        retriever.aretrieve(q), web_search.arun(q), db.afetch(q)
    )
    -> Gather independent tool calls; combine for the agent's context.

INTERVIEW ANSWER:
    "Async is ideal for LLM apps because model calls are I/O-bound — you wait
    seconds per call, and async lets one worker keep hundreds in flight. I
    fan out embedding and LLM calls with gather, cap concurrency with a
    Semaphore to respect rate limits, wrap calls in wait_for timeouts with
    exponential-backoff retries for resilience, and stream tokens with an
    async generator into a StreamingResponse so users see output live. For
    agents, I gather independent tool calls — retriever, web search, DB — to
    build context concurrently instead of sequentially."
'''


# =================================================================================
# SECTION 10: COMMON ASYNC GOTCHAS & PITFALLS
# =================================================================================
'''
The traps interviewers probe to test real experience.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GOTCHA 1 — Forgetting to await:
    result = fetch()         # BUG: result is a coroutine object, not the value
    result = await fetch()   # correct
    Python warns: "coroutine 'fetch' was never awaited."

GOTCHA 2 — Sequential awaits when you wanted concurrency:
    a = await x(); b = await y()   # 2x time — only do this if b needs a
    a, b = await asyncio.gather(x(), y())   # concurrent

GOTCHA 3 — Blocking call inside async (Section 7) — freezes the loop.

GOTCHA 4 — Fire-and-forget tasks getting garbage-collected:
    asyncio.create_task(work())   # if you don't keep a reference, it may be
                                  # GC'd before finishing
    FIX: keep a reference (store the task), or await it / use a TaskGroup.

GOTCHA 5 — Unhandled exceptions in tasks vanish:
    A task that raises but is never awaited can swallow the error silently.
    FIX: await tasks, use gather (raises), or TaskGroup (propagates).

GOTCHA 6 — Mixing sync and async DB sessions / clients:
    Using a sync SQLAlchemy session in async code blocks the loop. Use the
    async engine/session (SQLAlchemy 2.0 async) or run in a thread.

GOTCHA 7 — Shared mutable state across tasks:
    Even single-threaded async can have race conditions across await points —
    state can change while a coroutine is suspended. Use asyncio.Lock for
    critical sections that span awaits.

GOTCHA 8 — Calling asyncio.run() when a loop is already running:
    Raises "asyncio.run() cannot be called from a running event loop"
    (common in Jupyter). Use the existing loop / await directly.

INTERVIEW POINT:
    "The classics: forgetting to await (you get a coroutine, not a value),
    sequential awaits when gather would parallelize, and blocking the loop.
    Subtler ones: fire-and-forget tasks getting garbage-collected if you don't
    keep a reference, exceptions in un-awaited tasks vanishing silently, and
    race conditions across await points even though it's single-threaded —
    state can change while a coroutine is suspended, so I use an asyncio.Lock
    for critical sections that span awaits."
'''


# =================================================================================
# SECTION 11: INTERVIEW Q&A (senior-level)
# =================================================================================
'''
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q1. Difference between concurrency and parallelism?
A:  "Concurrency = interleaving tasks so they take turns (one core, overlapping).
    Parallelism = literally simultaneous on multiple cores. Asyncio is
    concurrency on one thread; multiprocessing is parallelism."

Q2. What is the GIL and how does it affect threading?
A:  "A mutex letting one thread run Python bytecode at a time. CPU-bound
    threading gets no speedup; I/O-bound does, because the GIL releases during
    I/O. CPU parallelism needs multiprocessing."

Q3. When asyncio vs threading vs multiprocessing?
A:  "I/O-bound, many calls -> asyncio. I/O-bound with sync libs -> threading.
    CPU-bound -> multiprocessing."

Q4. What does await actually do?
A:  "Pauses the coroutine, returns control to the event loop so other tasks
    run, and resumes when the awaited I/O completes."

Q5. What happens if I put time.sleep(5) in an async endpoint?
A:  "It blocks the single-threaded event loop for 5 seconds — every other
    request stalls. Use await asyncio.sleep instead."

Q6. Sequential awaits vs gather?
A:  "Sequential runs one after another; gather runs them concurrently and
    returns when all finish. Use gather for independent I/O calls."

Q7. How do you limit concurrent API calls?
A:  "asyncio.Semaphore(n) — acquire before the call, release after, capping
    in-flight calls to respect rate limits."

Q8. How do you call a blocking library from async?
A:  "await asyncio.to_thread(fn) for I/O; run_in_executor with a process pool
    for CPU-bound."

Q9. How do you run independent agent tool calls?
A:  "asyncio.gather of the async tool calls — retriever, web search, DB —
    concurrently, then combine the results."

Q10. How do you handle errors across many concurrent tasks?
A:  "gather(..., return_exceptions=True) so one failure doesn't kill the
    batch, then inspect each result; or a TaskGroup for structured handling."

Q11. Is asyncio faster than threading?
A:  "For high-concurrency I/O, yes — lower overhead, no thread context
    switching, scales to tens of thousands of tasks. For CPU work, neither
    helps; use multiprocessing."

Q12. Can you have race conditions in single-threaded async?
A:  "Yes — state can change across await points while a coroutine is
    suspended. Use asyncio.Lock for critical sections spanning awaits."
'''


# =================================================================================
# SECTION 12: GOLDEN LESSONS
# =================================================================================
'''
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GOLDEN LESSON 1 — CONCURRENCY != PARALLELISM.
    Asyncio = concurrency on one thread (interleaved I/O waits).
    Multiprocessing = parallelism across cores. Never blur these.

GOLDEN LESSON 2 — MATCH THE MODEL TO THE BOTTLENECK.
    I/O-bound + many -> asyncio. I/O + sync libs -> threading.
    CPU-bound -> multiprocessing. State this as a decision, with reasons.

GOLDEN LESSON 3 — THE GIL ONLY BLOCKS CPU-BOUND THREADING.
    Threads still help I/O (GIL released during I/O). Don't say "Python can't
    do threads" — say "one thread runs bytecode at a time."

GOLDEN LESSON 4 — await = PAUSE + YIELD TO LOOP + RESUME LATER.
    That mechanism, on one thread, is the entire async win. Explain it, don't
    just say "async is faster."

GOLDEN LESSON 5 — THE EVENT LOOP IS SINGLE-THREADED.
    Its power and its danger. One non-yielding task stalls everything.

GOLDEN LESSON 6 — NEVER BLOCK THE LOOP.
    Sync sleep, sync requests, CPU loops, sync DB drivers inside async = all
    requests freeze. Use async libs / to_thread / process pool.

GOLDEN LESSON 7 — gather FOR CONCURRENCY, Semaphore FOR LIMITS.
    Independent I/O -> gather. Rate limits -> Semaphore. Timeouts -> wait_for.

GOLDEN LESSON 8 — FORGETTING await IS THE #1 BUG.
    Calling an async fn returns a coroutine object, not the result.

GOLDEN LESSON 9 — RACE CONDITIONS EXIST IN ASYNC TOO.
    State can change across await points. Use asyncio.Lock for spans.

GOLDEN LESSON 10 — TIE ASYNC TO THE AI USE CASE.
    LLM calls are I/O-bound; async lets one worker keep hundreds in flight.
    Fan-out with gather, cap with Semaphore, stream with async generators.
    This connects Phase 2 to the JD's GenAI focus.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ONE-LINE CHEAT SHEET:
    Concurrency (interleave) vs parallelism (simultaneous). Asyncio = former.
    GIL: 1 thread runs bytecode; releases on I/O; CPU parallel -> multiprocessing.
    async def -> coroutine object; runs only when awaited / scheduled.
    await -> pause, yield to loop, resume when I/O ready.
    Event loop -> single-threaded scheduler; never block it.
    gather -> concurrent; create_task -> schedule now; Semaphore -> cap;
    wait_for -> timeout; to_thread/run_in_executor -> offload blocking/CPU.
    LLM apps: fan-out + concurrency cap + timeout/retry + token streaming.
'''


# =================================================================================
# RUN SUMMARY
# =================================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 2 — ASYNC PYTHON & CONCURRENCY DEEP DIVE")
    print("=" * 70)
    print()
    print("THE FOUNDATIONAL DISTINCTION:")
    print("  Concurrency = interleave tasks (1 thread) | Parallelism = N cores")
    print("  Asyncio = cooperative concurrency on a single-threaded event loop")
    print()
    print("THE DECISION RULE:")
    print("  I/O-bound + many calls -> asyncio")
    print("  I/O-bound + sync libs  -> threading")
    print("  CPU-bound              -> multiprocessing")
    print()
    print("THE GIL: one thread runs Python bytecode at a time;")
    print("  released during I/O (threads help I/O, not CPU).")
    print()
    print("await = pause coroutine, yield to loop, resume when I/O ready.")
    print("NEVER block the single-threaded loop (sync sleep/requests/CPU/sync-DB).")
    print()
    print("LLM PATTERNS: gather (fan-out) + Semaphore (rate limit)")
    print("  + wait_for (timeout/retry) + async generator (token streaming).")
    print()
    print("=" * 70)
    print("Next: Phase 3 — API Design & Integration")
    print("=" * 70)
