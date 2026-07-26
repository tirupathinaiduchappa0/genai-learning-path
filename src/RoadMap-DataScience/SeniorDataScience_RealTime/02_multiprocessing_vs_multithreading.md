# Lesson 02 — Multiprocessing vs Multithreading (and async)

## 1. The Question

> "Explain multiprocessing vs multithreading in Python. Compare their memory model, the GIL, and when to use each. Give a concrete example."

---

## 2. Theory — the three concurrency models in Python

| | Threads | Processes | Async (asyncio) |
|---|---|---|---|
| Memory | Shared | Separate (copied) | Shared (single thread) |
| Parallelism | No (GIL) for CPU | Yes, true parallel | No (cooperative) |
| Best for | I/O-bound | CPU-bound | High-concurrency I/O |
| Overhead | Low | High (spawn + IPC) | Very low |
| Data sharing | Easy (risk: races) | Hard (pickle/IPC) | Easy |

### The GIL (Global Interpreter Lock)

CPython has one lock that allows **only one thread to execute Python bytecode at a time**. So multiple threads doing pure-Python CPU work do **not** run in parallel — they take turns. The GIL is released during blocking I/O (network, disk) and inside some C extensions (NumPy), which is why threads still help for I/O.

Consequence:
- **CPU-bound Python** → threads give ~no speedup → use **processes** (each has its own interpreter and GIL).
- **I/O-bound** → threads (or async) give big speedup because the GIL is released while waiting.

> Note: Python 3.13 introduces an experimental **free-threaded (no-GIL)** build. Worth mentioning as forward-looking, but the default runtime still has the GIL.

---

## 3. Hands-on code — proving the difference

### 3.1 CPU-bound: processes win

```python
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

def cpu_task(n):
    total = 0
    for i in range(n):
        total += i * i
    return total

def benchmark(executor_cls, workers=4, tasks=(10_000_000,) * 4):
    start = time.perf_counter()
    with executor_cls(max_workers=workers) as ex:
        list(ex.map(cpu_task, tasks))
    return time.perf_counter() - start

if __name__ == "__main__":
    print(f"Threads:   {benchmark(ThreadPoolExecutor):.2f}s")   # slow (GIL)
    print(f"Processes: {benchmark(ProcessPoolExecutor):.2f}s")  # ~Ncores faster
```

Threads barely improve on serial time; processes scale with cores.

### 3.2 I/O-bound: threads win (and async wins more)

```python
import time, threading, requests
from concurrent.futures import ThreadPoolExecutor

URLS = ["https://httpbin.org/delay/1"] * 8

def fetch(url):
    return requests.get(url).status_code

# Serial ~8s; threaded ~1-2s because GIL releases during network wait
with ThreadPoolExecutor(max_workers=8) as ex:
    list(ex.map(fetch, URLS))
```

Async version (best for thousands of concurrent connections):

```python
import asyncio, httpx

async def fetch(client, url):
    r = await client.get(url)
    return r.status_code

async def main():
    async with httpx.AsyncClient() as client:
        await asyncio.gather(*(fetch(client, u) for u in URLS))

asyncio.run(main())
```

### 3.3 The classic race condition (why shared memory needs locks)

```python
import threading

counter = 0
def increment():
    global counter
    for _ in range(100_000):
        counter += 1   # NOT atomic: read, add, write

threads = [threading.Thread(target=increment) for _ in range(4)]
[t.start() for t in threads]; [t.join() for t in threads]
print(counter)  # often < 400_000 due to lost updates
```

Fix with a `threading.Lock()`. Processes avoid this because memory isn't shared, but then you pay IPC cost to share results.

---

## 4. Real-time / production scenarios

- **Web scraper hitting 1000s of URLs** → I/O-bound → `asyncio` + `httpx` (this MCP server itself uses async httpx for exactly this reason).
- **Image resizing / ML preprocessing on CPU** → CPU-bound → `ProcessPoolExecutor` or `multiprocessing.Pool`.
- **A web server (Uvicorn/Gunicorn)** → multiple **worker processes** (to use all cores, side-stepping the GIL) each running an **async** event loop (to handle many concurrent I/O-bound requests). Best of both.
- **NumPy/Pandas heavy math** → threads can help because BLAS releases the GIL in C.

---

## 5. Decision cheat-sheet

```
Is the work waiting on I/O (network, disk, DB)?
   → Yes: many connections?  → asyncio
          moderate + simple? → threads
Is the work crunching numbers in pure Python?
   → Yes: multiprocessing / ProcessPoolExecutor
Is it heavy NumPy/C math?
   → threads are fine (GIL released in C)
```

---

## 6. Interview script

"The GIL means only one thread runs Python bytecode at a time, so threads don't parallelize CPU-bound work — for that I use processes, which each get their own interpreter. Threads and asyncio shine for I/O-bound work because the GIL is released while waiting on the network or disk. Async scales to far more concurrent connections than threads because there's no per-thread stack overhead. In production a typical setup is multiple worker processes to use all cores, each running an async loop for concurrency."
