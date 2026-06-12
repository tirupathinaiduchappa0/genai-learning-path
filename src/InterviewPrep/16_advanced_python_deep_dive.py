"""
===================================================================================
ADVANCED PYTHON DEEP DIVE — Interview Mastery
===================================================================================

This lesson covers EVERYTHING senior interviewers test on Python.
Required for: PwC, JPMC, ETech, and any senior GenAI/backend role.

SECTIONS:
    1. OOP Fundamentals — Classes, Inheritance, Polymorphism, Encapsulation
    2. Advanced OOP — Abstract Classes, Mixins, MRO, Metaclasses
    3. Asyncio — Async/Await, Event Loop, Coroutines (DEEP)
    4. Concurrency — Threading vs Multiprocessing vs Asyncio
    5. GIL (Global Interpreter Lock) — The #1 Interview Question
    6. Design Patterns in Python (Singleton, Factory, Observer, Decorator)
    7. SOLID Principles with Real Code Examples
    8. Memory Management — References, Garbage Collection, Memory Leaks
    9. Performance Optimization (Profiling, Caching, Generators)
    10. GOLDEN LESSONS

This is the depth senior interviewers test for. Read carefully.
===================================================================================
"""


# =================================================================================
# SECTION 1: OOP FUNDAMENTALS
# =================================================================================
"""
THE 4 PILLARS OF OOP (you must know these):

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PILLAR 1: ENCAPSULATION (Hiding internal details)

    Bundling data and methods together, hiding internal state.
    In Python, we use naming conventions:
        public:    self.name        (anyone can access)
        protected: self._name       (single underscore — convention only)
        private:   self.__name      (double underscore — name mangling)

    class BankAccount:
        def __init__(self, balance):
            self.__balance = balance  # private — hidden from outside

        def deposit(self, amount):    # public method to access private data
            if amount > 0:
                self.__balance += amount

        def get_balance(self):
            return self.__balance

    account = BankAccount(1000)
    account.deposit(500)             # OK
    print(account.get_balance())     # 1500
    # account.__balance              # AttributeError (mangled to _BankAccount__balance)

    WHY: Prevents external code from breaking internal state.
    Example: Can't set balance to negative without going through deposit().

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PILLAR 2: INHERITANCE (Reuse code from parent class)

    A child class inherits attributes and methods from a parent class.

    class Animal:
        def __init__(self, name):
            self.name = name
        def speak(self):
            return "Some sound"

    class Dog(Animal):           # Dog inherits from Animal
        def speak(self):         # OVERRIDE parent method
            return "Woof!"

    class Cat(Animal):
        def speak(self):
            return "Meow!"

    dog = Dog("Rex")
    print(dog.name, dog.speak())   # Rex Woof!

    SUPER() — Call parent's method:
        class GoldenRetriever(Dog):
            def speak(self):
                parent_sound = super().speak()  # calls Dog.speak()
                return f"{parent_sound} (friendly)"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PILLAR 3: POLYMORPHISM (Same interface, different implementations)

    Different classes can implement the same method differently.

    animals = [Dog("Rex"), Cat("Whiskers")]
    for animal in animals:
        print(animal.speak())  # Rex says "Woof!", Whiskers says "Meow!"

    The SAME method call (.speak()) produces DIFFERENT behavior.
    This is the foundation of clean, extensible code.

    DUCK TYPING (Python's flavor of polymorphism):
        "If it walks like a duck and quacks like a duck, it's a duck."
        Python doesn't care about the type — it cares about the BEHAVIOR.

        def make_it_speak(thing):
            return thing.speak()  # works for ANY object with .speak() method

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PILLAR 4: ABSTRACTION (Hide complexity, show only essentials)

    Show WHAT the object does, hide HOW it does it.

    from abc import ABC, abstractmethod

    class Shape(ABC):                       # Abstract base class
        @abstractmethod
        def area(self):                     # Must be implemented by children
            pass

    class Circle(Shape):
        def __init__(self, radius):
            self.radius = radius
        def area(self):
            return 3.14 * self.radius ** 2

    # Shape("test")                          # Error — can't instantiate abstract class
    c = Circle(5)
    print(c.area())                         # 78.5

    WHY: Forces all child classes to implement essential methods.
    Without it, you might forget to implement .area() in a new shape class.

INTERVIEW ANSWER:
    "The 4 pillars of OOP are encapsulation (hiding internal state), inheritance
    (reusing code from parent classes), polymorphism (same interface different
    implementations), and abstraction (hiding complexity behind a simple interface).
    Python uses naming conventions for encapsulation (single underscore for
    protected, double for private), supports multiple inheritance, uses duck
    typing for polymorphism, and provides ABC module for abstract base classes."
"""


# =================================================================================
# SECTION 2: ADVANCED OOP — Abstract Classes, Mixins, MRO, Metaclasses
# =================================================================================
"""
ABSTRACT BASE CLASSES (ABC):

    Force subclasses to implement specific methods.

    from abc import ABC, abstractmethod

    class LLMProvider(ABC):
        @abstractmethod
        def invoke(self, prompt: str) -> str:
            # Every LLM provider must implement this.
            pass

    class OpenAIProvider(LLMProvider):
        def invoke(self, prompt):
            return openai.complete(prompt)  # Implementation

    class GroqProvider(LLMProvider):
        def invoke(self, prompt):
            return groq.complete(prompt)

    # LLMProvider()  # ERROR — can't instantiate abstract class
    # If OpenAIProvider didn't implement invoke(), it would also error.

    WHY: Ensures consistent interface across implementations.
    Used in: Strategy pattern, Factory pattern.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MIXINS (Multiple Inheritance for adding capabilities):

    A "mixin" is a class designed to ADD functionality to other classes.
    It's not meant to be instantiated alone.

    class LoggingMixin:
        def log(self, message):
            print(f"[{self.__class__.__name__}] {message}")

    class CachingMixin:
        def __init__(self):
            self._cache = {}
        def get_cached(self, key):
            return self._cache.get(key)

    class AIService(LoggingMixin, CachingMixin):
        def __init__(self):
            CachingMixin.__init__(self)
        def query(self, prompt):
            self.log(f"Querying: {prompt}")  # from LoggingMixin
            cached = self.get_cached(prompt)  # from CachingMixin
            if cached:
                return cached
            # ... actual logic

    AIService now has logging AND caching without copying code.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MRO (Method Resolution Order) — THE DIAMOND PROBLEM:

    What happens with multiple inheritance when classes share a parent?

    class A:
        def hello(self): return "A"
    class B(A):
        def hello(self): return "B" + super().hello()
    class C(A):
        def hello(self): return "C" + super().hello()
    class D(B, C):
        pass

    print(D().hello())  # What does this print?

    Python uses C3 LINEARIZATION to determine the order:
    MRO of D: D → B → C → A → object

    print(D.mro())  # Shows the resolution order

    Trace:
    D.hello() → B.hello() → "B" + super().hello()
        super() in B → C (next in MRO!)
        C.hello() → "C" + super().hello()
            super() in C → A (next in MRO)
            A.hello() → "A"
        Returns "CA"
    Returns "BCA"

    OUTPUT: "BCA"

    KEY INSIGHT: super() doesn't always mean PARENT.
    It means NEXT in the MRO chain.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CLASS METHODS vs STATIC METHODS vs INSTANCE METHODS:

    class MyClass:
        count = 0  # class variable

        def instance_method(self):
            "Has access to instance (self) and class (cls)"
            return self.count

        @classmethod
        def class_method(cls):
            "Has access to class (cls) only — not instance"
            return cls.count

        @staticmethod
        def static_method():
            "No access to self or cls — like a regular function"
            return "Just a utility"

    obj = MyClass()
    obj.instance_method()    # Needs an instance
    MyClass.class_method()   # Can be called on class
    MyClass.static_method()  # Can be called on class

    WHEN TO USE EACH:
        instance_method → needs to access/modify instance state
        classmethod → factory methods, alternate constructors
        staticmethod → utility functions related to the class

    EXAMPLE — Factory method pattern:
        class User:
            def __init__(self, name, email):
                self.name = name
                self.email = email

            @classmethod
            def from_dict(cls, data):
                return cls(data["name"], data["email"])

            @classmethod
            def from_json(cls, json_str):
                import json
                return cls.from_dict(json.loads(json_str))

        user = User.from_json('{"name": "Tirupathi", "email": "..."}')
"""


# =================================================================================
# SECTION 3: ASYNCIO — Async/Await, Event Loop, Coroutines (DEEP)
# =================================================================================
"""
THE PROBLEM ASYNCIO SOLVES:

    Traditional code is SYNCHRONOUS — it does one thing at a time:

    def fetch_data():
        response1 = requests.get("api1.com")  # waits 1 second
        response2 = requests.get("api2.com")  # waits 1 second
        response3 = requests.get("api3.com")  # waits 1 second
        # Total: 3 seconds

    With ASYNCIO:
    async def fetch_data():
        results = await asyncio.gather(
            fetch("api1.com"),
            fetch("api2.com"),
            fetch("api3.com")
        )  # All 3 run CONCURRENTLY
        # Total: 1 second (the slowest one)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CORE CONCEPTS:

    COROUTINE: A function defined with `async def`. It can be PAUSED and RESUMED.
        async def my_coroutine():
            print("Step 1")
            await asyncio.sleep(1)  # PAUSE here, let other coroutines run
            print("Step 2")

    EVENT LOOP: The "scheduler" that manages all coroutines.
        It picks a coroutine, runs it until it hits `await`, then switches
        to another coroutine. Like a single chef cooking 5 dishes — switches
        between them while one is simmering.

    AWAIT: Pause execution and let the event loop run other coroutines.
        Only allowed inside `async def` functions.

    asyncio.gather(): Run multiple coroutines CONCURRENTLY.
    asyncio.run(): Start the event loop and run a coroutine.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

REAL EXAMPLE — Concurrent API calls:

    import asyncio
    import aiohttp

    async def fetch_url(url):
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                return await response.text()

    async def main():
        urls = ["https://api1.com", "https://api2.com", "https://api3.com"]

        # SEQUENTIAL (slow):
        # for url in urls:
        #     result = await fetch_url(url)
        # Total: sum of all response times

        # CONCURRENT (fast):
        results = await asyncio.gather(*[fetch_url(url) for url in urls])
        # Total: time of the SLOWEST request

    asyncio.run(main())

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHY ASYNCIO IS PERFECT FOR LLM APPS:

    LLM API calls take 5-30 seconds (waiting for the model to generate).
    During that wait, the CPU is idle. Asyncio lets you do OTHER work while waiting.

    Example: You need to call 10 LLMs in parallel for an ensemble approach.
    Synchronous: 10 × 10 sec = 100 seconds total
    Asyncio:     10 in parallel = ~10 seconds total

    THIS IS WHY FastAPI USES ASYNC.
    Every API endpoint can handle MANY concurrent requests because most time
    is spent WAITING (for DB, LLM, external APIs) — async lets the server
    handle other requests during the wait.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CRITICAL RULES OF ASYNCIO:

    RULE 1: Can't await inside sync function
        def regular():
            await fetch()  # SyntaxError!

    RULE 2: Must call async functions with await (or asyncio.run)
        result = my_async_func()       # Returns a coroutine, NOT the result!
        result = await my_async_func() # Returns the actual result

    RULE 3: Don't mix sync and async badly
        async def good():
            await asyncio.sleep(1)         # async-friendly
            time.sleep(1)                  # BLOCKS the event loop! BAD!

    RULE 4: Event loop only runs coroutines, not regular threads
        Use loop.run_in_executor() for CPU-bound or sync code

INTERVIEW ANSWER:
    "Asyncio enables concurrent execution of I/O-bound tasks in a single thread
    using cooperative multitasking. Coroutines (async def functions) yield
    control at await points, letting the event loop run other coroutines.
    This is perfect for LLM applications where most time is spent waiting for
    API responses — instead of blocking, the server handles other requests
    during the wait. asyncio.gather() runs multiple coroutines concurrently,
    reducing total time from sum-of-all to max-of-all."
"""


# =================================================================================
# SECTION 4: CONCURRENCY — Threading vs Multiprocessing vs Asyncio
# =================================================================================
"""
THREE WAYS TO DO "PARALLEL" IN PYTHON. They are NOT the same.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THREADING (Multiple threads in one process):

    from threading import Thread

    def task(n):
        time.sleep(2)
        print(f"Task {n} done")

    threads = [Thread(target=task, args=(i,)) for i in range(5)]
    for t in threads: t.start()
    for t in threads: t.join()  # Wait for all to finish

    BEST FOR: I/O-bound tasks (waiting for network, disk, DB)
    LIMITATION: GIL prevents true parallelism for CPU-bound tasks (more on GIL below)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MULTIPROCESSING (Multiple processes, one per CPU core):

    from multiprocessing import Process, Pool

    def cpu_heavy_task(n):
        return sum(i*i for i in range(10_000_000))

    with Pool(4) as pool:  # 4 processes
        results = pool.map(cpu_heavy_task, [1,2,3,4])

    BEST FOR: CPU-bound tasks (heavy computation, ML training)
    BYPASSES THE GIL because each process has its own Python interpreter.
    DOWNSIDE: Higher memory usage (each process duplicates data).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ASYNCIO (Single thread, cooperative multitasking):

    See Section 3. Single thread, but switches between coroutines.

    BEST FOR: Many I/O-bound tasks (LLM calls, web requests, DB queries)
    NOT good for CPU-bound work (single thread can't compute in parallel).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DECISION TABLE — Which to use?

    SCENARIO                              BEST CHOICE
    Many LLM API calls                    Asyncio (concurrent waits)
    Heavy CPU computation (ML training)   Multiprocessing (true parallelism)
    Reading 1000 files from disk          Threading or Asyncio
    Image processing on multiple images   Multiprocessing
    Web scraping (network-bound)          Asyncio
    Database queries (network I/O)        Asyncio
    Encoding video files                  Multiprocessing
    Building a web server                 Asyncio (FastAPI, aiohttp)

INTERVIEW ANSWER:
    "Python has three concurrency models: threading (best for I/O-bound, limited
    by GIL for CPU work), multiprocessing (true parallelism, best for CPU-bound),
    and asyncio (single-thread cooperative multitasking, best for many I/O-bound
    tasks like LLM calls). For LLM applications, asyncio is ideal because most
    time is spent waiting for API responses — async lets us handle hundreds of
    concurrent requests in one thread. For ML training or heavy data processing,
    multiprocessing is needed to bypass the GIL and use multiple CPU cores."
"""


# =================================================================================
# SECTION 5: GIL (Global Interpreter Lock) — THE #1 INTERVIEW QUESTION
# =================================================================================
"""
WHAT IS THE GIL?

    A LOCK that allows only ONE thread to execute Python bytecode at a time.
    Even on a 16-core CPU, Python threads don't run truly in parallel.

    Why does Python have it?
    Memory management. Python uses reference counting for garbage collection.
    Without a lock, two threads modifying the same object's reference count
    could corrupt memory. The GIL prevents this by allowing only one thread
    to run Python code at a time.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE PARADOX:
    Threads exist in Python — but only ONE runs Python code at a time.
    So why use threading at all?

    BECAUSE THE GIL IS RELEASED DURING I/O:
    - Network call → GIL released → other threads run
    - Disk read → GIL released → other threads run
    - C extensions (NumPy) → can release GIL

    So threading IS useful for I/O-bound work, NOT for CPU-bound work.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PROOF (CPU-bound — threading DOESN'T help):

    import time
    from threading import Thread

    def cpu_task():
        sum(i*i for i in range(50_000_000))

    # Sequential: 4 seconds
    start = time.time()
    cpu_task(); cpu_task()
    print(f"Sequential: {time.time() - start}")  # ~4 sec

    # Threading: STILL 4 seconds (GIL prevents parallelism)
    start = time.time()
    t1 = Thread(target=cpu_task)
    t2 = Thread(target=cpu_task)
    t1.start(); t2.start()
    t1.join(); t2.join()
    print(f"Threading: {time.time() - start}")  # ~4 sec (NOT 2!)

    # Multiprocessing: 2 seconds (true parallelism)
    from multiprocessing import Process
    start = time.time()
    p1 = Process(target=cpu_task)
    p2 = Process(target=cpu_task)
    p1.start(); p2.start()
    p1.join(); p2.join()
    print(f"Multiprocessing: {time.time() - start}")  # ~2 sec ✅

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HOW TO BYPASS THE GIL:

    1. Use multiprocessing (each process has its own GIL)
    2. Use C extensions (NumPy, Pandas — release GIL for heavy ops)
    3. Use asyncio (single-thread but very efficient for I/O)
    4. Use Python 3.13+ with --disable-gil (experimental, no-GIL Python!)

INTERVIEW ANSWER:
    "The GIL is a mutex that allows only one thread to execute Python bytecode
    at a time. It exists for memory management safety. This means threading in
    Python doesn't give true parallelism for CPU-bound tasks. However, the GIL
    is RELEASED during I/O operations, so threading IS useful for I/O-bound work.
    For CPU-bound parallelism, use multiprocessing which spawns separate processes
    with their own GIL. For most modern Python work, asyncio is preferred for
    I/O-bound tasks because it's lighter weight than threading."
"""


# =================================================================================
# SECTION 6: DESIGN PATTERNS IN PYTHON
# =================================================================================
"""
DESIGN PATTERNS = Reusable solutions to common problems.
These come up CONSTANTLY in interviews. Know at least the top 5.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PATTERN 1: SINGLETON (Only one instance allowed)

    Use case: Database connection pool, configuration manager, logger.

    class Database:
        _instance = None

        def __new__(cls):
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance.connection = "Connected to DB"
            return cls._instance

    db1 = Database()
    db2 = Database()
    print(db1 is db2)  # True — same instance

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PATTERN 2: FACTORY (Create objects without specifying exact class)

    Use case: Creating LLM instances based on configuration.

    class LLMFactory:
        @staticmethod
        def create(provider, model):
            if provider == "openai":
                return OpenAI(model=model)
            elif provider == "groq":
                return Groq(model=model)
            elif provider == "anthropic":
                return Anthropic(model=model)
            raise ValueError(f"Unknown provider: {provider}")

    llm = LLMFactory.create("groq", "llama-3.3-70b")

    YOU ALREADY USE THIS in DocSage: GroqLLMFactory!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PATTERN 3: STRATEGY (Swap algorithms at runtime)

    Use case: Different retrieval strategies (vector, keyword, hybrid).

    from abc import ABC, abstractmethod

    class RetrievalStrategy(ABC):
        @abstractmethod
        def retrieve(self, query): pass

    class VectorRetrieval(RetrievalStrategy):
        def retrieve(self, query):
            return vector_search(query)

    class KeywordRetrieval(RetrievalStrategy):
        def retrieve(self, query):
            return bm25_search(query)

    class RAGSystem:
        def __init__(self, strategy: RetrievalStrategy):
            self.strategy = strategy
        def query(self, q):
            return self.strategy.retrieve(q)

    # Easy to swap strategies:
    rag = RAGSystem(VectorRetrieval())
    rag = RAGSystem(KeywordRetrieval())  # Different behavior, same interface

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PATTERN 4: OBSERVER (Notify subscribers when something happens)

    Use case: Event-driven systems, pub/sub, real-time updates.

    class EventEmitter:
        def __init__(self):
            self.subscribers = []

        def subscribe(self, callback):
            self.subscribers.append(callback)

        def emit(self, data):
            for callback in self.subscribers:
                callback(data)

    emitter = EventEmitter()
    emitter.subscribe(lambda data: print(f"Logger: {data}"))
    emitter.subscribe(lambda data: send_email(data))
    emitter.emit("Document uploaded")
    # Both subscribers are notified

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PATTERN 5: DECORATOR (Add behavior without modifying the original)

    Python has FIRST-CLASS decorators (@syntax). Most powerful Python feature.

    def cache_results(func):
        cache = {}
        def wrapper(*args):
            if args not in cache:
                cache[args] = func(*args)
            return cache[args]
        return wrapper

    @cache_results
    def expensive_function(x):
        time.sleep(2)  # simulating slow operation
        return x * x

    expensive_function(5)  # Takes 2 seconds (computes)
    expensive_function(5)  # Instant (cached!)

    YOU USE DECORATORS DAILY:
    - @tool (LangChain)
    - @app.route (Flask)
    - @app.get (FastAPI)
    - @property, @staticmethod, @classmethod

INTERVIEW ANSWER:
    "Python design patterns I use frequently: Factory pattern for creating LLM
    instances based on config (my GroqLLMFactory in DocSage), Strategy pattern
    for swappable retrieval methods (vector vs keyword vs hybrid), Singleton for
    shared resources like DB connections, Observer for event-driven notifications,
    and Decorator (which is built into Python with @ syntax) for adding behavior
    like caching, logging, or authentication without modifying the original function."
"""


# =================================================================================
# SECTION 7: SOLID PRINCIPLES
# =================================================================================
"""
SOLID = 5 principles for clean, maintainable OOP code.
Senior interviewers WILL ask about these.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

S — SINGLE RESPONSIBILITY PRINCIPLE
    A class should have ONE reason to change.

    BAD:
        class User:
            def __init__(self, data):
                self.data = data
            def save_to_database(self): ...    # DB responsibility
            def send_email(self): ...           # Email responsibility
            def generate_report(self): ...      # Report responsibility

    GOOD:
        class User: ...              # Just user data
        class UserRepository: ...    # DB operations
        class EmailService: ...      # Email sending
        class ReportGenerator: ...   # Report creation

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

O — OPEN/CLOSED PRINCIPLE
    Classes should be OPEN for extension but CLOSED for modification.

    BAD:
        class PaymentProcessor:
            def process(self, payment_type, amount):
                if payment_type == "credit_card": ...
                elif payment_type == "paypal": ...
                # Adding new type requires MODIFYING this class

    GOOD:
        class PaymentMethod(ABC):
            @abstractmethod
            def process(self, amount): pass

        class CreditCard(PaymentMethod): ...
        class PayPal(PaymentMethod): ...
        class Crypto(PaymentMethod): ...   # Add new type WITHOUT modifying existing

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

L — LISKOV SUBSTITUTION PRINCIPLE
    Subclasses must be SUBSTITUTABLE for their parent class without breaking code.

    BAD:
        class Bird:
            def fly(self): print("Flying")

        class Penguin(Bird):
            def fly(self):
                raise Exception("Penguins can't fly!")  # BREAKS LSP

    GOOD:
        class Bird: ...
        class FlyingBird(Bird):
            def fly(self): print("Flying")
        class Penguin(Bird): ...  # Doesn't inherit fly()

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

I — INTERFACE SEGREGATION PRINCIPLE
    Don't force classes to depend on methods they don't use.

    BAD:
        class Worker(ABC):
            @abstractmethod
            def work(self): pass
            @abstractmethod
            def eat(self): pass

        class RobotWorker(Worker):
            def work(self): ...
            def eat(self): pass  # Robots don't eat — forced empty implementation

    GOOD:
        class Workable(ABC):
            @abstractmethod
            def work(self): pass

        class Eatable(ABC):
            @abstractmethod
            def eat(self): pass

        class Human(Workable, Eatable): ...
        class Robot(Workable): ...  # Only implements what it needs

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

D — DEPENDENCY INVERSION PRINCIPLE
    Depend on ABSTRACTIONS (interfaces), not CONCRETIONS (specific classes).

    BAD:
        class EmailService:
            def __init__(self):
                self.smtp = GmailSMTP()  # Hardcoded dependency

    GOOD:
        class EmailProvider(ABC):
            @abstractmethod
            def send(self, message): pass

        class EmailService:
            def __init__(self, provider: EmailProvider):  # Depends on abstraction
                self.provider = provider

        # Now you can inject ANY email provider:
        service = EmailService(GmailSMTP())
        service = EmailService(SendGrid())
        service = EmailService(MockProvider())  # Great for testing!

INTERVIEW ANSWER:
    "SOLID principles guide clean OOP design. Single Responsibility — one class,
    one reason to change. Open/Closed — extend behavior through new classes, not
    by modifying existing ones. Liskov Substitution — subclasses must work where
    their parents do. Interface Segregation — don't force classes to implement
    methods they don't need. Dependency Inversion — depend on abstractions
    (interfaces) so you can swap implementations easily. In DocSage, I follow
    these — separate concerns into modules, use abstract LLMProvider for
    pluggable backends, and inject dependencies through constructors."
"""


# =================================================================================
# SECTION 8: MEMORY MANAGEMENT — References, Garbage Collection, Memory Leaks
# =================================================================================
"""
HOW PYTHON MANAGES MEMORY:

    Python uses two mechanisms:
    1. REFERENCE COUNTING (primary)
    2. GARBAGE COLLECTOR (handles cycles)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

REFERENCE COUNTING:

    Every Python object has a counter — how many references point to it.
    When the counter reaches 0, the object is destroyed.

    import sys

    a = [1, 2, 3]
    print(sys.getrefcount(a))  # 2 (one for 'a', one for getrefcount arg)

    b = a                       # Now 2 references
    print(sys.getrefcount(a))  # 3

    del b                       # Reference count drops to 2
    a = None                    # Reference count drops to 1, then 0 → destroyed

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE PROBLEM — CIRCULAR REFERENCES:

    class Node:
        def __init__(self):
            self.parent = None
            self.children = []

    a = Node()
    b = Node()
    a.children.append(b)
    b.parent = a   # CIRCULAR! a → b → a → b...

    # Even if we set a = None and b = None, the references between them
    # keep the count > 0 forever. MEMORY LEAK!

    SOLUTION: Garbage collector (gc module) detects cycles and breaks them.
        import gc
        gc.collect()  # Force garbage collection

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WEAK REFERENCES (Avoid memory leaks in caches):

    import weakref

    class HeavyObject:
        pass

    cache = weakref.WeakValueDictionary()
    obj = HeavyObject()
    cache["key"] = obj

    del obj  # Object is destroyed even though cache still has weak ref
    print(cache.get("key"))  # None — weakref doesn't prevent destruction

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

COMMON MEMORY LEAKS IN PYTHON:

    1. Global lists/dicts that grow forever:
        cached_data = []  # global
        def process(x):
            cached_data.append(x)  # never cleaned!

    2. Forgotten event listeners:
        emitter.subscribe(my_callback)
        # If you never unsubscribe, callback (and its captured variables) live forever

    3. __del__ methods preventing collection:
        class Leaky:
            def __del__(self):
                pass  # Custom __del__ prevents some cycle collection

INTERVIEW ANSWER:
    "Python uses reference counting plus a garbage collector for cycles. Every
    object has a refcount — when it reaches 0, the object is destroyed. The GC
    handles circular references that refcounting can't. Memory leaks happen with
    global containers that grow forever, forgotten event listeners, and circular
    references. To debug, I use sys.getsizeof, the gc module, and tools like
    memory_profiler. For caches, I use weakref to avoid keeping objects alive."
"""


# =================================================================================
# SECTION 9: PERFORMANCE OPTIMIZATION
# =================================================================================
"""
TECHNIQUES TO MAKE PYTHON CODE FASTER:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. PROFILING (Find the slow parts FIRST):

    import cProfile
    cProfile.run("my_slow_function()")
    # Shows which functions take the most time

    Or line-by-line:
        # pip install line_profiler
        @profile
        def my_func():
            ...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

2. GENERATORS (Lazy evaluation, save memory):

    # BAD — loads everything into memory:
    def get_squares(n):
        return [i*i for i in range(n)]    # 1M items in memory

    # GOOD — generates one at a time:
    def get_squares(n):
        for i in range(n):
            yield i*i                     # Only one item in memory at a time

    for sq in get_squares(1_000_000):
        process(sq)  # Memory-efficient!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

3. CACHING (functools.lru_cache):

    from functools import lru_cache

    @lru_cache(maxsize=128)
    def expensive_function(x, y):
        time.sleep(2)  # heavy computation
        return x + y

    expensive_function(1, 2)  # Takes 2 sec
    expensive_function(1, 2)  # Instant — cached!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

4. USE BUILT-INS (they're written in C, very fast):

    # SLOW (Python loop):
    total = 0
    for i in range(1_000_000):
        total += i

    # FAST (C implementation):
    total = sum(range(1_000_000))

    # Other fast built-ins: map, filter, sorted, set operations

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

5. APPROPRIATE DATA STRUCTURES:

    OPERATION              LIST     DICT/SET    DEQUE
    Append at end         O(1)      —          O(1)
    Append at start       O(n)      —          O(1)
    Lookup by index       O(1)      —          O(1)
    Lookup by value       O(n)      O(1)       O(n)
    Membership check (in) O(n)      O(1)       O(n)

    if x in my_list:    # O(n) — slow for large lists
    if x in my_set:     # O(1) — fast!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

6. NUMPY FOR NUMBERS:

    # SLOW Python loop:
    result = [x*2 for x in big_list]  # 100ms for 1M items

    # FAST NumPy (vectorized in C):
    import numpy as np
    arr = np.array(big_list)
    result = arr * 2  # 1ms for 1M items — 100x faster!

INTERVIEW ANSWER:
    "For Python performance: profile first to find bottlenecks (cProfile),
    use generators for memory efficiency, lru_cache for memoization, prefer
    built-ins (sum, map, filter) which are implemented in C. Choose the right
    data structure — sets/dicts for O(1) lookup vs lists O(n). For numeric
    work, NumPy is 100x faster than pure Python loops because of vectorization
    in C. For LLM apps specifically, asyncio for concurrent I/O is the biggest
    speedup."
"""


# =================================================================================
# SECTION 10: GOLDEN LESSONS
# =================================================================================
"""
GOLDEN LESSON 1: "OOP is about RESPONSIBILITY, not just classes."
    Don't create a 1000-line class. Each class = ONE responsibility.
    SOLID principles aren't optional — they're how senior developers think.

GOLDEN LESSON 2: "Asyncio is for I/O-bound. Multiprocessing is for CPU-bound."
    Don't confuse them. Threading is also for I/O but limited by GIL.
    For LLM apps (mostly waiting on APIs), asyncio wins.

GOLDEN LESSON 3: "The GIL exists. Plan around it."
    Threading won't speed up your CPU-heavy code. Multiprocessing will.
    Modern Python (3.13+) is moving toward optional no-GIL mode, but for now,
    know the limits.

GOLDEN LESSON 4: "Decorators are Python's superpower."
    @tool, @app.get, @lru_cache, @property, @classmethod —
    these are decorators. Master writing your own. They're elegant.

GOLDEN LESSON 5: "Memory leaks are usually circular refs or forgotten globals."
    Use weakref for caches. Be careful with global mutable state.
    The garbage collector helps, but it's not magic.

GOLDEN LESSON 6: "Profile before optimizing."
    Premature optimization is the root of all evil.
    Find the actual bottleneck with cProfile, then optimize THAT.

GOLDEN LESSON 7: "Built-ins are FAST because they're written in C."
    Replace Python loops with sum(), map(), filter(), sorted() when possible.
    Replace nested loops with set operations or NumPy.

GOLDEN LESSON 8: "Design Patterns aren't a Java thing — Python uses them too."
    You probably already use Factory (LLMFactory), Strategy (different retrievers),
    and Decorator (@tool). Just learn the NAMES so you can articulate them in interviews.
"""

print("=" * 60)
print("Advanced Python Deep Dive — Complete")
print("=" * 60)
print()
print("10 Sections:")
print("  1.  OOP Fundamentals (4 pillars)")
print("  2.  Advanced OOP (ABC, Mixins, MRO, Diamond)")
print("  3.  Asyncio Deep Dive (event loop, coroutines, gather)")
print("  4.  Concurrency (Threading vs Multiprocessing vs Asyncio)")
print("  5.  GIL — The #1 interview question")
print("  6.  Design Patterns (Singleton, Factory, Strategy, Observer, Decorator)")
print("  7.  SOLID Principles with code examples")
print("  8.  Memory Management (refcounts, GC, leaks)")
print("  9.  Performance Optimization (profiling, generators, caching)")
print("  10. GOLDEN LESSONS (8 lessons)")
print()
print("This is the depth senior interviewers test for.")
print("=" * 60)
