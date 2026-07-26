"""
===================================================================================
LESSON 1 — PYTHON & SYSTEMS FUNDAMENTALS  (Senior DS / GenAI Real-Time Deep-Dive)
===================================================================================

COVERS (Section A of the question bank):
  A1. Detecting duplicate file content  (hashing, chunked reads, size-bucketing)
  A2. Multiprocessing vs Multithreading  (GIL, CPU-bound vs I/O-bound)
  A3. Rate limiter  (fixed-window, sliding-window, token-bucket, thread-safe)

HOW TO READ THIS FILE:
  - Each topic = THEORY (why it works) -> RUNNABLE CODE (proof) -> INTERVIEW ANSWER
    (the soundbite to say) -> RELATED CONCEPTS (what they follow up with).
  - Run it:  & "C:\\Projects\\Gen AI Udemy\\LangCGS\\.venv\\Scripts\\python.exe" "01_python_systems_fundamentals.py"
  - Pure standard library only. No external installs. Safe on Windows (spawn-guarded).

===================================================================================
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import threading
import time
from collections import defaultdict, deque

# Force UTF-8 stdout so box-drawing chars (━ =) render even when the Windows
# console/redirect defaults to cp1252. Safe no-op where already UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass


def banner(title: str) -> None:
    line = "=" * 83
    print("\n" + line + "\n" + title + "\n" + line)


def sub(title: str) -> None:
    print("\n" + "━" * 79 + "\n" + title + "\n" + "━" * 79)


# ===================================================================================
# A1. DETECTING DUPLICATE FILE CONTENT
# ===================================================================================
#
# THE PROBLEM
#   Find files with IDENTICAL CONTENT across one or more directories — regardless of
#   filename. "report.pdf" and "report_copy.pdf" with the same bytes are duplicates;
#   two different files that happen to share a name are NOT.
#
# WHY HASHING (not name/size/byte-by-byte)?
#   - Names lie: same name != same content; different name can be same content.
#   - Size alone is a weak signal: many different files share a size. BUT it is a
#     perfect NEGATIVE filter — different sizes => definitely not duplicates.
#   - Byte-by-byte across ALL pairs is O(n^2) comparisons and re-reads files many
#     times. A hash reduces each file to one fixed fingerprint (O(n) reads), and
#     equal content => equal hash. You compare short hashes, not whole files.
#
# THE KILLER OPTIMIZATION — group by SIZE first, then hash only collisions
#   Hashing every file is wasteful. Files of different sizes can never be equal, so:
#     1) Bucket files by size (cheap: os.path.getsize, no read).
#     2) Only size-buckets with >=2 files can contain duplicates.
#     3) Hash ONLY those candidates. Most unique files are eliminated for free.
#   (Advanced: hash a small header first, then full hash — a two-tier filter.)
#
# LARGE FILES — read in CHUNKS
#   Never do f.read() on a multi-GB file (loads it all into RAM). Read fixed-size
#   blocks (e.g., 64 KB) and feed each to hasher.update(). Constant memory.
#
# HASH COLLISIONS
#   MD5/SHA are deterministic: same bytes -> same digest, ALWAYS. Risk is two
#   DIFFERENT files sharing a digest.
#     - MD5: broken for security (collisions can be crafted), but fine for
#       de-dup of non-adversarial data; still, prefer SHA-256 for safety.
#     - For 100% certainty, after hashes match, do a final byte-by-byte compare.
#       That's cheap because collisions are astronomically rare — you almost never
#       reach the byte-compare, so it costs nothing in practice but removes all doubt.


def hash_file(path: str, algo: str = "sha256", chunk_size: int = 65536) -> str:
    """Return the hex digest of a file's content, reading in fixed-size chunks.

    Constant memory regardless of file size — the core trick for large files.
    """
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk_size), b""):
            h.update(block)
    return h.hexdigest()


def files_identical(path_a: str, path_b: str, chunk_size: int = 65536) -> bool:
    """Definitive byte-by-byte comparison (used only to defeat hash collisions)."""
    if os.path.getsize(path_a) != os.path.getsize(path_b):
        return False
    with open(path_a, "rb") as fa, open(path_b, "rb") as fb:
        while True:
            ba, bb = fa.read(chunk_size), fb.read(chunk_size)
            if ba != bb:
                return False
            if not ba:  # both reached EOF simultaneously
                return True


def find_duplicate_files(paths: list[str], algo: str = "sha256",
                         verify_bytes: bool = True) -> list[list[str]]:
    """Find groups of duplicate files by content.

    Strategy: size-bucket -> hash only same-size candidates -> optional byte verify.
    Returns a list of groups; each group is a list of paths with identical content.
    """
    # 1) Bucket by size (cheap — no file reads).
    by_size: dict[int, list[str]] = defaultdict(list)
    for p in paths:
        try:
            by_size[os.path.getsize(p)].append(p)
        except OSError:
            continue  # unreadable / vanished file — skip gracefully

    duplicate_groups: list[list[str]] = []

    # 2) Only size-buckets with >= 2 files can hold duplicates.
    for size, candidates in by_size.items():
        if len(candidates) < 2:
            continue

        # 3) Hash only the candidates, bucket by digest.
        by_hash: dict[str, list[str]] = defaultdict(list)
        for p in candidates:
            try:
                by_hash[hash_file(p, algo)].append(p)
            except OSError:
                continue

        # 4) Each hash-bucket with >= 2 files is a duplicate group.
        for digest, group in by_hash.items():
            if len(group) < 2:
                continue
            if verify_bytes:
                # Defeat (astronomically rare) collisions: confirm against group[0].
                confirmed = [group[0]] + [
                    p for p in group[1:] if files_identical(group[0], p)
                ]
                if len(confirmed) >= 2:
                    duplicate_groups.append(confirmed)
            else:
                duplicate_groups.append(group)

    return duplicate_groups


def _demo_duplicate_detection() -> None:
    sub("A1. DUPLICATE FILE CONTENT DETECTION (runnable)")
    tmp = tempfile.mkdtemp(prefix="dedup_")
    # Create files: 3 identical, 1 unique-same-size, 1 unique-different-size.
    contents = {
        "a.txt": b"the quick brown fox" * 100,
        "b_copy.txt": b"the quick brown fox" * 100,   # dup of a.txt
        "c_backup.txt": b"the quick brown fox" * 100,  # dup of a.txt
        "d_unique.txt": b"the lazy brown dog" * 100,    # same SIZE, different content
        "e_other.txt": b"short unique content",         # different size
    }
    paths = []
    for name, data in contents.items():
        fp = os.path.join(tmp, name)
        with open(fp, "wb") as f:
            f.write(data)
        paths.append(fp)

    groups = find_duplicate_files(paths)
    print(f"Scanned {len(paths)} files in {tmp}")
    print(f"Found {len(groups)} duplicate group(s):")
    for i, g in enumerate(groups, 1):
        print(f"  Group {i}: {[os.path.basename(p) for p in g]}")

    # Cleanup
    for p in paths:
        os.remove(p)
    os.rmdir(tmp)
    print("(temp files cleaned up)")


# INTERVIEW ANSWER (A1):
#   "I fingerprint each file's CONTENT with a streaming SHA-256 read in 64KB chunks,
#    so memory stays flat even for huge files. Before hashing I bucket by file size —
#    different sizes can't be equal, so I only hash same-size candidates. Equal hash
#    means equal content; for zero-risk certainty I do a final byte compare, which I
#    almost never actually reach because collisions are astronomically rare."
#
# RELATED CONCEPTS they may probe:
#   - Idempotency & content-addressable storage (Git blobs, Docker layers, S3 ETags).
#   - Why security-broken MD5 is still OK for non-adversarial de-dup (speed).
#   - Rolling hashes / fuzzy hashing (ssdeep) for NEAR-duplicates, not exact.
#   - Bloom filters to probabilistically skip "definitely new" content at scale.


# ===================================================================================
# A2. MULTIPROCESSING vs MULTITHREADING
# ===================================================================================
#
# ONE-LINE MENTAL MODEL
#   Threads share ONE process's memory but fight over ONE GIL -> great for waiting
#   (I/O), useless for computing (CPU). Processes each get their OWN interpreter and
#   memory -> true parallel CPU, but pay for isolation (spawn cost + IPC).
#
# THE GIL (Global Interpreter Lock) — the whole reason this question exists
#   CPython protects its internals with a single lock: only ONE thread executes
#   Python bytecode at a time. So N threads do NOT run Python code in parallel.
#   BUT the GIL is RELEASED during blocking I/O (disk, network, sleep) and inside
#   many C extensions (NumPy, hashlib). That's the key:
#     - CPU-bound pure-Python  -> threads give ~no speedup (GIL serializes them).
#     - I/O-bound              -> threads overlap the WAITS -> big speedup.
#   (Note: free-threaded / no-GIL builds exist from Python 3.13+ experimentally,
#    but assume the GIL in interviews unless told otherwise.)
#
#   MEMORY MODEL
#     Threads:   shared address space. Fast data sharing, but you must guard shared
#                mutable state with Locks/Queues (race conditions, deadlocks).
#     Processes: separate address space. No accidental sharing; data crosses via
#                pickling through pipes/queues (serialization cost). Safer, heavier.
#
#   OVERHEAD / COMPLEXITY
#     Threads:   cheap to create, cheap to switch, shared memory = easy sharing/hard
#                correctness.
#     Processes: expensive to spawn (esp. Windows 'spawn' re-imports your module),
#                IPC costs, but crash isolation and real parallelism.
#
# WHEN TO CHOOSE WHICH  (say this crisply)
#   - I/O-bound (API calls, DB, file/network, web scraping)  -> threading or asyncio.
#   - CPU-bound (math, parsing, image/vector crunching)       -> multiprocessing
#                                                                 (or C-ext that frees GIL).
#   - Massive concurrency of I/O (10k sockets)                -> asyncio (event loop),
#                                                                 not thousands of threads.


def _cpu_task(n: int) -> int:
    """Pure-Python CPU-bound work (GIL-bound) — sum of squares."""
    total = 0
    for i in range(n):
        total += i * i
    return total


def _io_task(seconds: float) -> None:
    """Simulated I/O wait — time.sleep RELEASES the GIL (like a real network call)."""
    time.sleep(seconds)


def _demo_threads_vs_processes() -> None:
    import multiprocessing as mp

    sub("A2. MULTIPROCESSING vs MULTITHREADING (runnable benchmark)")
    N = 1_000_000          # per-task workload (kept small so the demo is quick)
    TASKS = 4

    # --- CPU-bound: sequential vs threads vs processes -----------------------------
    t0 = time.perf_counter()
    for _ in range(TASKS):
        _cpu_task(N)
    seq = time.perf_counter() - t0

    t0 = time.perf_counter()
    threads = [threading.Thread(target=_cpu_task, args=(N,)) for _ in range(TASKS)]
    for t in threads: t.start()
    for t in threads: t.join()
    thr = time.perf_counter() - t0

    t0 = time.perf_counter()
    with mp.Pool(processes=min(TASKS, os.cpu_count() or 2)) as pool:
        pool.map(_cpu_task, [N] * TASKS)
    proc = time.perf_counter() - t0

    print(f"CPU-bound ({TASKS} tasks x {N:,} iters):")
    print(f"  sequential      : {seq:.3f}s")
    print(f"  threads (GIL)   : {thr:.3f}s   <- ~no speedup, GIL serializes CPU work")
    print(f"  processes       : {proc:.3f}s   <- real parallelism (beats threads on CPU)")

    # --- I/O-bound: sequential vs threads ------------------------------------------
    SLEEP, IO_TASKS = 0.15, 4
    t0 = time.perf_counter()
    for _ in range(IO_TASKS):
        _io_task(SLEEP)
    seq_io = time.perf_counter() - t0

    t0 = time.perf_counter()
    ths = [threading.Thread(target=_io_task, args=(SLEEP,)) for _ in range(IO_TASKS)]
    for t in ths: t.start()
    for t in ths: t.join()
    thr_io = time.perf_counter() - t0

    print(f"\nI/O-bound ({IO_TASKS} tasks x {SLEEP}s sleep):")
    print(f"  sequential      : {seq_io:.3f}s")
    print(f"  threads         : {thr_io:.3f}s   <- overlaps the WAITS (GIL released on I/O)")


# INTERVIEW ANSWER (A2):
#   "CPython's GIL lets only one thread run Python bytecode at a time, so threads
#    don't parallelize CPU work — but the GIL is released during I/O, so threads DO
#    overlap network/disk waits. Rule of thumb: threads/asyncio for I/O-bound,
#    multiprocessing for CPU-bound because each process has its own interpreter and
#    memory and runs truly in parallel — at the cost of spawn overhead and pickling
#    data across processes."
#
# RELATED CONCEPTS they may probe:
#   - asyncio vs threads: one event loop, cooperative, best for huge I/O concurrency.
#   - concurrent.futures: ThreadPoolExecutor vs ProcessPoolExecutor (same API).
#   - Race conditions, Lock/RLock, deadlock, and why Queue is the safe way to share.
#   - Why NumPy/pandas/hashlib can parallelize with THREADS (they release the GIL in C).
#   - Windows 'spawn' vs Linux 'fork' start methods (why the __main__ guard matters).


# ===================================================================================
# A3. RATE LIMITER  (max 100 requests / minute / user)
# ===================================================================================
#
# THE THREE CLASSIC ALGORITHMS (know the tradeoffs cold)
#
#   1) FIXED WINDOW
#      Count requests per user in the current [minute] bucket; reset the count when
#      the minute rolls over.
#      + Trivial, O(1) memory per user (just a counter + window start).
#      - BURST AT EDGES: a user can send 100 at 00:59 and 100 at 01:00 -> 200 in ~1s,
#        because the boundary resets the counter. This is the classic flaw.
#
#   2) SLIDING WINDOW (log of timestamps)
#      Keep timestamps of recent requests; on each call, drop those older than 60s
#      and allow only if the remaining count < limit.
#      + Accurate — true "N per rolling 60s", no edge burst.
#      - O(k) memory per user (k = requests in window). Sliding-window-COUNTER is a
#        common approximation that keeps just two bucket counts to bound memory.
#
#   3) TOKEN BUCKET
#      A bucket holds up to `capacity` tokens; tokens refill at a steady rate. Each
#      request consumes one token; empty bucket -> reject.
#      + Allows controlled BURSTS up to capacity while enforcing an average rate.
#        This is what most API gateways use (AWS, Stripe, NGINX).
#      - Slightly more moving parts (refill math). Leaky bucket is the smoothing cousin.
#
# THREAD-SAFETY
#   A rate limiter is shared mutable state hit by many concurrent requests. Every
#   check-and-update must be atomic -> guard with a threading.Lock. Otherwise two
#   threads both read count=99 and both proceed -> limit breached (race condition).
#   (At multi-server scale you centralize this in Redis with atomic INCR/Lua scripts.)


class FixedWindowRateLimiter:
    """Per-user fixed-window limiter. Simple, O(1), but allows edge bursts."""

    def __init__(self, limit: int = 100, window_seconds: float = 60.0) -> None:
        self.limit = limit
        self.window = window_seconds
        self._state: dict[str, list[float]] = {}  # user -> [window_start, count]
        self._lock = threading.Lock()

    def allow(self, user: str, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        with self._lock:
            start, count = self._state.get(user, [now, 0])
            if now - start >= self.window:   # window expired -> reset
                start, count = now, 0
            if count < self.limit:
                self._state[user] = [start, count + 1]
                return True
            self._state[user] = [start, count]
            return False


class SlidingWindowRateLimiter:
    """Per-user sliding-window (timestamp log). Accurate, no edge burst."""

    def __init__(self, limit: int = 100, window_seconds: float = 60.0) -> None:
        self.limit = limit
        self.window = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, user: str, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        with self._lock:
            dq = self._hits[user]
            cutoff = now - self.window
            while dq and dq[0] <= cutoff:   # evict timestamps older than the window
                dq.popleft()
            if len(dq) < self.limit:
                dq.append(now)
                return True
            return False


class TokenBucketRateLimiter:
    """Per-user token bucket. Enforces average rate but permits bursts to capacity."""

    def __init__(self, capacity: int = 100, refill_per_second: float = 100 / 60) -> None:
        self.capacity = capacity
        self.refill_rate = refill_per_second  # 100 tokens per 60s
        self._state: dict[str, list[float]] = {}  # user -> [tokens, last_refill_ts]
        self._lock = threading.Lock()

    def allow(self, user: str, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        with self._lock:
            tokens, last = self._state.get(user, [float(self.capacity), now])
            # Refill based on elapsed time, capped at capacity.
            tokens = min(self.capacity, tokens + (now - last) * self.refill_rate)
            if tokens >= 1.0:
                self._state[user] = [tokens - 1.0, now]
                return True
            self._state[user] = [tokens, now]
            return False


def _demo_rate_limiter() -> None:
    sub("A3. RATE LIMITER (runnable — 3 algorithms)")
    # Use a small limit + simulated clock so the demo is instant and deterministic.
    limit = 5
    t = 1000.0  # fake "now"

    fw = FixedWindowRateLimiter(limit=limit, window_seconds=60)
    results = [fw.allow("alice", now=t + i * 0.1) for i in range(8)]
    print(f"Fixed window (limit={limit}): 8 rapid requests -> {results}")
    print(f"  allowed={sum(results)}, rejected={results.count(False)}")

    sw = SlidingWindowRateLimiter(limit=limit, window_seconds=60)
    r1 = [sw.allow("bob", now=t + i * 0.1) for i in range(8)]
    r2 = sw.allow("bob", now=t + 61)  # after the window slides, allowed again
    print(f"Sliding window (limit={limit}): 8 rapid -> allowed={sum(r1)}; "
          f"one more after 61s -> {r2}")

    tb = TokenBucketRateLimiter(capacity=limit, refill_per_second=limit / 60)
    burst = [tb.allow("carol", now=t) for _ in range(8)]  # all at same instant
    refilled = tb.allow("carol", now=t + 30)  # 30s later ~2.5 tokens refilled
    print(f"Token bucket (cap={limit}): burst of 8 at t=0 -> allowed={sum(burst)}; "
          f"one more at t=30s -> {refilled}")
    print("  (thread-safety: every check-and-update is guarded by a Lock)")


# INTERVIEW ANSWER (A3):
#   "I track requests per user and pick the algorithm by requirement. Fixed-window is
#    a simple per-minute counter but allows double-rate bursts at the boundary.
#    Sliding-window keeps a timestamp log for exact 'N per rolling 60s'. Token bucket
#    refills tokens at a steady rate and allows controlled bursts up to capacity —
#    that's what most API gateways use. All check-and-update logic sits under a Lock
#    to avoid races; at multi-server scale I move the counter into Redis with atomic
#    INCR/Lua so the limit is shared across instances."
#
# RELATED CONCEPTS they may probe:
#   - Leaky bucket (smooths output) vs token bucket (allows bursts).
#   - Distributed rate limiting with Redis (INCR + EXPIRE, or Lua for atomicity).
#   - 429 Too Many Requests, Retry-After header, exponential backoff on the client.
#   - Where to enforce: API gateway / middleware vs in-app; per-user vs per-IP vs global.


# ===================================================================================
# GOLDEN LESSONS  (60-second recall before you walk in)
# ===================================================================================
GOLDEN_LESSONS = """
  1. De-dup: size-bucket FIRST (free), hash SAME-SIZE only, stream in chunks, byte-verify last.
  2. Equal content => equal hash, always. The risk is the reverse (collision) — so verify.
  3. MD5 is broken for security, fine for non-adversarial de-dup; prefer SHA-256 anyway.
  4. GIL = one thread runs Python bytecode at a time; released during I/O and in C extensions.
  5. Threads/asyncio for I/O-bound; multiprocessing for CPU-bound. Say it in one breath.
  6. Processes = own memory + real parallelism, but spawn cost + pickling. Guard with __main__.
  7. Rate limiter: fixed-window (edge burst), sliding-window (exact), token-bucket (bursty avg).
  8. Any shared counter under concurrency MUST be atomic — Lock locally, Redis across servers.
  9. Always answer: THEORY -> tradeoff -> the ONE line an interviewer remembers you for.
"""


if __name__ == "__main__":
    banner("LESSON 1 — PYTHON & SYSTEMS FUNDAMENTALS")
    _demo_duplicate_detection()
    _demo_threads_vs_processes()
    _demo_rate_limiter()
    banner("GOLDEN LESSONS")
    print(GOLDEN_LESSONS)
