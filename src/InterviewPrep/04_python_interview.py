"""
GenAI Interview Prep - Lesson 4: Python Interview Questions & Answers

WHY THIS LESSON EXISTS:
    As a GenAI developer, Python is your primary language. Interviewers
    will ask Python questions to check if you can actually CODE, not just
    use frameworks. They test: core concepts, OOP, data structures,
    tricky behaviors, and small coding problems.

    Your profile: 1.5 years Python + GenAI. They expect you to explain
    decorators, generators, list comprehensions, and OOP confidently.
    They do NOT expect you to solve LeetCode hard problems.

HOW INTERVIEWERS ASK PYTHON QUESTIONS:
    1. "Explain X concept" (decorators, generators, GIL)
    2. "What's the difference between X and Y?" (list vs tuple, == vs is)
    3. "What's the output of this code?" (tricky behavior questions)
    4. "Write a small program" (reverse string, find duplicates, etc.)

TABLE OF CONTENTS:
    PART A: Core Python Concepts (15 topics)
    PART B: Tricky Python Behaviors (8 gotchas)
    PART C: Small Coding Problems (10 programs)
    PART D: Python for GenAI (5 topics)
    PART E: Interview Q&A (25 questions)

Author: GenAI Learner
"""


# ██████████████████████████████████████████████████████████████████████████████
# PART A: CORE PYTHON CONCEPTS (The 15 Most Asked Topics)
# ██████████████████████████████████████████████████████████████████████████████


# ==============================================================================
# 1. LIST vs TUPLE vs SET vs DICTIONARY
# ==============================================================================
#
# List []:   Ordered, mutable, allows duplicates. Most used.
#   fruits = ["apple", "banana", "apple"]  # duplicates OK
#   fruits.append("cherry")                # mutable
#
# Tuple ():  Ordered, IMMUTABLE, allows duplicates. Faster than list.
#   point = (10, 20)  # can't change after creation
#   Use when: data shouldn't change (coordinates, DB records, dict keys)
#
# Set {}:    Unordered, mutable, NO duplicates. Fast membership check.
#   unique = {1, 2, 3, 2, 1}  # becomes {1, 2, 3}
#   3 in unique                # O(1) lookup
#
# Dictionary {k:v}: Key-value pairs, ordered (3.7+), mutable.
#   person = {"name": "Tiru", "age": 28}
#
# INTERVIEW ANSWER:
#   "List is ordered and mutable. Tuple is ordered but immutable. Set has
#   no duplicates with O(1) lookup. Dictionary maps keys to values. In my
#   GenAI project, I use dicts for state, lists for messages, sets for dedup."


# ==============================================================================
# 2. MUTABLE vs IMMUTABLE
# ==============================================================================
#
# Mutable (can change): list, dict, set, custom objects
# Immutable (cannot change): int, float, str, tuple, frozenset, bool
#
# THE TRAP (mutable default argument):
#   def add_item(lst=[]):    # DEFAULT MUTABLE ARGUMENT!
#       lst.append(1)
#       return lst
#   add_item()  # [1]
#   add_item()  # [1, 1]  <- NOT [1]! Same list reused!
#   FIX: def add_item(lst=None):
#            if lst is None: lst = []
#
# In Python, default arguments are evaluated only once, at the time the function is  # defined — NOT every time the function is called.creates one single list object []
# that same list is reused for every call. 
#
# Default mutable arguments (like list, dict, set) are shared across function calls.
# Never use mutable objects (list, dict, set) as default arguments.
#
# INTERVIEW ANSWER:
# In Python, default arguments are evaluated once at definition time. If the default # is mutable, the same object is reused across calls, causing unintended side  effects.”


# ==============================================================================
# 3. DECORATORS (Asked in 90% of Python interviews)
# ==============================================================================
#
# A decorator WRAPS a function to add behavior WITHOUT modifying it.
#
#   def my_decorator(func):
#       def wrapper(*args, **kwargs):
#           print("Before")
#           result = func(*args, **kwargs)
#           print("After")
#           return result
#       return wrapper
#
#   @my_decorator        # syntactic sugar for: say_hello = my_decorator(say_hello)
#   def say_hello(name):
#       print(f"Hello, {name}!")
#
# REAL-WORLD USES IN GENAI:
#   @tool        -> LangChain tool (our email tool!)
#   @traceable   -> LangSmith tracing
#   @st.cache_resource -> Streamlit caching
#   @lru_cache   -> function result caching
#
# INTERVIEW ANSWER:
#   "A decorator wraps a function to add behavior. @decorator is syntactic
#   sugar for func = decorator(func). I use @tool for LangChain tools,
#   @traceable for LangSmith tracing, @st.cache_resource for Streamlit."


# ==============================================================================
# 4. GENERATORS and YIELD
# ==============================================================================
#
# A generator produces values ONE AT A TIME using yield (not return).
# Doesn't store all values in memory — memory efficient.
#
# REGULAR FUNCTION:
#   def get_squares(n):
#       return [i**2 for i in range(n)]  # ALL in memory
#
# GENERATOR:
#   def get_squares(n):
#       for i in range(n):
#           yield i**2  # one at a time, pauses, resumes on next()
#
# WHY IT MATTERS FOR GENAI:
#   LLM streaming uses generators! graph.stream() yields node updates
#   one at a time. Our DocSage streaming shows step-by-step progress.
#
# INTERVIEW ANSWER:
#   "A generator uses yield to produce values lazily — one at a time.
#   Memory efficient for large data. In GenAI, LLM streaming uses
#   generators — tokens are yielded one at a time."


# ==============================================================================
# 5. LIST COMPREHENSION vs GENERATOR EXPRESSION
# ==============================================================================
#
# List Comprehension []: entire list in memory NOW.
#   squares = [x**2 for x in range(1000)]  # list, all in memory
#
# Generator Expression (): lazy, one at a time.
#   squares = (x**2 for x in range(1000))  # generator, nothing computed yet
#
# INTERVIEW ANSWER:
#   "List comprehension creates everything in memory. Generator expression
#   is lazy — computes one at a time. Use generators for large data."


# ==============================================================================
# 6. *args and **kwargs
# ==============================================================================
#
# *args: extra POSITIONAL arguments as a TUPLE.
#   def add(*args): return sum(args)
#   add(1, 2, 3)  # args = (1, 2, 3), returns 6
#
# **kwargs: extra KEYWORD arguments as a DICTIONARY.
#   def greet(**kwargs):
#       for k, v in kwargs.items(): print(f"{k}: {v}")
#   greet(name="Tiru", age=28)
#
# INTERVIEW ANSWER:
#   "*args collects positional args as tuple. **kwargs collects keyword
#   args as dict. Together they make functions flexible."


# ==============================================================================
# 7. LAMBDA, MAP, FILTER
# ==============================================================================
#
# Lambda: anonymous one-line function.
#   square = lambda x: x ** 2
#   square(5)  # 25
#   Same as: def square(x): return x ** 2
#
# Map: apply a function to every item in an iterable.
#   nums = [1, 2, 3, 4]
#   squared = list(map(lambda x: x**2, nums))  # [1, 4, 9, 16]
#
# Filter: keep items that pass a condition.
#   nums = [1, 2, 3, 4, 5, 6]
#   evens = list(filter(lambda x: x % 2 == 0, nums))  # [2, 4, 6]
#
# MODERN PYTHON PREFERS LIST COMPREHENSIONS:
#   squared = [x**2 for x in nums]           # cleaner than map
#   evens = [x for x in nums if x % 2 == 0]  # cleaner than filter
#
# INTERVIEW ANSWER:
#   "Lambda creates anonymous one-line functions. Map applies a function
#   to all items. Filter keeps items matching a condition. In modern Python,
#   list comprehensions are preferred for readability."


# ==============================================================================
# 8. OOP: CLASSES, INHERITANCE, POLYMORPHISM
# ==============================================================================
#
# CLASS: blueprint for creating objects.
#   class Dog:
#       def __init__(self, name, breed):
#           self.name = name      # instance attribute
#           self.breed = breed
#       def bark(self):
#           return f"{self.name} says Woof!"
#   dog = Dog("Rex", "Labrador")
#   dog.bark()  # "Rex says Woof!"
#
# INHERITANCE: child class inherits from parent class.
#   class Animal:
#       def speak(self): return "..."
#   class Dog(Animal):
#       def speak(self): return "Woof!"
#   class Cat(Animal):
#       def speak(self): return "Meow!"
#
# POLYMORPHISM: same method name, different behavior.
#   for animal in [Dog(), Cat()]:
#       print(animal.speak())  # "Woof!" then "Meow!"
#
# ENCAPSULATION: hiding internal details.
#   self._private = "convention only"   # single underscore = "please don't touch"
#   self.__private = "name mangled"     # double underscore = harder to access
#
# IN OUR PROJECT:
#   GroqLLMFactory is a CLASS with methods (create_agent_llm, etc.)
#   Pydantic BaseModel classes for structured output (GradeDocuments, RouteQuery)
#   TypedDict for state definition (AgentState)
#
# INTERVIEW ANSWER:
#   "OOP in Python: classes define blueprints, __init__ initializes objects,
#   inheritance enables code reuse, polymorphism lets different classes share
#   method names with different behavior. In my project, I use classes for
#   the LLM factory, Pydantic models for structured output, and TypedDict
#   for graph state."


# ==============================================================================
# 9. EXCEPTION HANDLING (try/except/finally)
# ==============================================================================
#
# try:
#     result = 10 / 0
# except ZeroDivisionError as e:
#     print(f"Error: {e}")        # handles specific error
# except Exception as e:
#     print(f"Unexpected: {e}")   # catches everything else
# else:
#     print("Success!")           # runs only if NO exception
# finally:
#     print("Always runs")       # cleanup, runs no matter what
#
# CUSTOM EXCEPTIONS:
#   class DocumentLoadError(Exception):
#       pass
#   raise DocumentLoadError("PDF is corrupted")
#
# IN OUR PROJECT:
#   Every node has try/except for Groq API failures.
#   Grade node defaults to "yes" if grading fails.
#   Email tool catches SMTPAuthenticationError specifically.
#   Retriever tool uses try/finally to clean up temp files.
#
# INTERVIEW ANSWER:
#   "try/except handles errors gracefully. I catch specific exceptions first,
#   then general Exception as fallback. finally ensures cleanup runs always.
#   In my project, every LLM call is wrapped in try/except with meaningful
#   fallbacks — grading defaults to 'yes', email shows auth error messages."


# ==============================================================================
# 10. GIL (Global Interpreter Lock)
# ==============================================================================
#
# WHAT: The GIL is a mutex in CPython that allows only ONE thread to
# execute Python bytecode at a time, even on multi-core CPUs.
#
# WHY IT EXISTS: CPython's memory management (reference counting) is not
# thread-safe. The GIL prevents race conditions on reference counts.
#
# IMPACT:
#   CPU-bound tasks: GIL is a bottleneck. Threads don't help.
#     Use multiprocessing (separate processes) instead.
#   I/O-bound tasks: GIL is released during I/O waits.
#     Threads work fine for network calls, file reads, API calls.
#
# WHY IT MATTERS FOR GENAI:
#   LLM API calls are I/O-bound (waiting for network response).
#   The GIL is NOT a problem for our use case.
#   LangGraph's parallel node execution works because API calls release the GIL.
#
# INTERVIEW ANSWER:
#   "The GIL allows only one thread to execute Python code at a time.
#   It's a bottleneck for CPU-bound tasks but not for I/O-bound tasks
#   like API calls. In GenAI, LLM calls are I/O-bound, so the GIL
#   doesn't affect performance. For CPU-bound work, use multiprocessing."


# ==============================================================================
# 11. ASYNC/AWAIT
# ==============================================================================
#
# WHAT: Asynchronous programming lets you run multiple I/O operations
# concurrently without threads. Uses async/await syntax.
#
#   import asyncio
#
#   async def fetch_data(url):
#       # await pauses this function, lets others run
#       response = await some_async_http_call(url)
#       return response
#
#   async def main():
#       # Run 3 API calls concurrently (not sequentially!)
#       results = await asyncio.gather(
#           fetch_data("url1"),
#           fetch_data("url2"),
#           fetch_data("url3"),
#       )
#
# WHY IT MATTERS FOR GENAI:
#   Multiple LLM calls can run concurrently with async.
#   LangGraph supports async: graph.ainvoke(), graph.astream().
#   FastAPI (for deploying) is async-native.
#
# INTERVIEW ANSWER:
#   "async/await enables concurrent I/O without threads. Multiple API calls
#   run concurrently — while one waits for a response, others execute.
#   LangGraph supports async with ainvoke() and astream(). FastAPI is
#   async-native, making it ideal for deploying GenAI APIs."


# ==============================================================================
# 12. TYPE HINTS
# ==============================================================================
#
# Python is dynamically typed, but type hints add documentation and IDE support.
#
#   def greet(name: str) -> str:
#       return f"Hello, {name}"
#
#   from typing import Optional, List, Dict, Annotated
#   def process(items: List[str], limit: Optional[int] = None) -> Dict[str, int]:
#       ...
#
# IN OUR PROJECT:
#   Every function has type hints (as per Important-Rules.md).
#   TypedDict for state: class AgentState(TypedDict)
#   Pydantic models: class GradeDocuments(BaseModel)
#   Literal types: Literal["yes", "no"] for grading
#
# INTERVIEW ANSWER:
#   "Type hints add documentation and enable IDE autocompletion and error
#   checking. I use them on every function signature. In my project,
#   TypedDict defines graph state, Pydantic BaseModel defines structured
#   output schemas, and Literal constrains grading to yes/no."


# ==============================================================================
# 13. DUNDER METHODS (__init__, __str__, __repr__, __len__)
# ==============================================================================
#
# Dunder = "double underscore" methods. They define how objects behave.
#
#   __init__: constructor, called when creating an object
#   __str__:  human-readable string (print(obj) calls this)
#   __repr__: developer-readable string (for debugging)
#   __len__:  called by len(obj)
#   __eq__:   called by obj1 == obj2
#   __iter__: makes object iterable (for loops)
#   __getitem__: called by obj[key]
#
# INTERVIEW ANSWER:
#   "Dunder methods define object behavior. __init__ is the constructor,
#   __str__ is for print(), __repr__ is for debugging, __len__ for len().
#   They let you make custom classes work with Python's built-in operations."


# ==============================================================================
# 14. CONTEXT MANAGERS (with statement)
# ==============================================================================
#
# WHAT: Ensures resources are properly opened AND closed.
#
#   with open("file.txt", "r") as f:
#       content = f.read()
#   # File is automatically closed here, even if an error occurred
#
# CUSTOM CONTEXT MANAGER:
#   from contextlib import contextmanager
#
#   @contextmanager
#   def timer():
#       import time
#       start = time.time()
#       yield
#       print(f"Took {time.time() - start:.2f}s")
#
#   with timer():
#       do_something()
#
# IN OUR PROJECT:
#   smtplib.SMTP uses context manager: with smtplib.SMTP(...) as server:
#   Temp file cleanup: with tempfile.NamedTemporaryFile(...) as tmp:
#
# INTERVIEW ANSWER:
#   "Context managers (with statement) ensure resources are properly cleaned
#   up — files closed, connections released, locks freed. They use __enter__
#   and __exit__ dunder methods. I use them for SMTP connections and temp
#   file handling in my project."


# ==============================================================================
# 15. VIRTUAL ENVIRONMENTS and DEPENDENCY MANAGEMENT
# ==============================================================================
#
# WHY: Different projects need different package versions.
#   Project A needs langchain==0.2  |  Project B needs langchain==0.3
#   Without venvs, they conflict. With venvs, each project is isolated.
#
# COMMANDS:
#   python -m venv .venv           # create virtual environment
#   .venv/Scripts/activate         # activate (Windows)
#   source .venv/bin/activate      # activate (Mac/Linux)
#   pip install -r requirements.txt # install dependencies
#   pip freeze > requirements.txt   # save current dependencies
#
# INTERVIEW ANSWER:
#   "Virtual environments isolate project dependencies. Each project has
#   its own .venv with specific package versions. requirements.txt pins
#   versions for reproducibility. In my project, I use a dedicated venv
#   with pinned versions of langchain, langgraph, streamlit, and groq."


# ██████████████████████████████████████████████████████████████████████████████
# PART B: TRICKY PYTHON BEHAVIORS (Interview Gotchas)
# ██████████████████████████████████████████████████████████████████████████████
#
# These are the "What's the output?" questions that catch people off guard.


# GOTCHA 1: Mutable Default Arguments (most common trap!)
#   def append_to(item, lst=[]):
#       lst.append(item)
#       return lst
#   print(append_to(1))  # [1]
#   print(append_to(2))  # [1, 2]  <- NOT [2]! Same list reused!
#   FIX: Use None as default, create new list inside function.


# GOTCHA 2: == vs is
#   == checks VALUE equality:  [1,2] == [1,2]  -> True
#   is checks IDENTITY (same object in memory): [1,2] is [1,2] -> False
#   a = [1,2]; b = a; a is b -> True (same object)
#   EXCEPTION: Python caches small integers (-5 to 256) and short strings.
#   a = 256; b = 256; a is b -> True (cached)
#   a = 257; b = 257; a is b -> False (not cached)
# is checks whether two variables reference the same object in memory (identity), while == checks whether their values are equal.”


# GOTCHA 3: String Immutability
#   s = "hello"
#   s[0] = "H"  # TypeError! Strings are immutable.
#   s = "Hello"  # This creates a NEW string, doesn't modify the old one.


# GOTCHA 4: List Copy Trap (Shallow vs Deep)
#   a = [[1, 2], [3, 4]]
#   b = a.copy()        # SHALLOW copy
#   b[0].append(99)
#   print(a)  # [[1, 2, 99], [3, 4]]  <- a is ALSO modified!
#   FIX: import copy; b = copy.deepcopy(a)  # DEEP copy


# GOTCHA 5: Variable Scope (LEGB Rule)
#   Python looks up variables in this order:
#   L: Local (inside current function)
#   E: Enclosing (outer function, for nested functions)
#   G: Global (module level)
#   B: Built-in (Python built-ins like print, len)
#
#   x = "global"
#   def outer():
#       x = "enclosing"
#       def inner():
#           x = "local"
#           print(x)  # "local"
#       inner()
#   outer()


# GOTCHA 6: for Loop Variable Leak
#   for i in range(5):
#       pass
#   print(i)  # 4  <- i still exists after the loop!
#   In Python, loop variables leak into the enclosing scope.


# GOTCHA 7: Dictionary Key Types
#   Only IMMUTABLE types can be dict keys: str, int, tuple, frozenset.
#   d = {[1,2]: "value"}  # TypeError! Lists are mutable, can't be keys.
#   d = {(1,2): "value"}  # OK! Tuples are immutable.


# GOTCHA 8: Integer Division
#   10 / 3   # 3.3333 (true division, returns float)
#   10 // 3  # 3 (floor division, returns int)
#   10 % 3   # 1 (modulo, remainder)


# ██████████████████████████████████████████████████████████████████████████████
# PART C: SMALL CODING PROBLEMS (10 Most Asked)
# ██████████████████████████████████████████████████████████████████████████████


# PROBLEM 1: Reverse a string
#   def reverse_string(s: str) -> str:
#       return s[::-1]
#   reverse_string("hello")  # "olleh"


# PROBLEM 2: Check if string is palindrome
#   def is_palindrome(s: str) -> bool:
#       s = s.lower().replace(" ", "")
#       return s == s[::-1]
#   is_palindrome("racecar")  # True
#   is_palindrome("A man a plan a canal Panama")  # True


# PROBLEM 3: Find duplicates in a list
#   def find_duplicates(lst: list) -> list:
#       seen = set()
#       duplicates = set()
#       for item in lst:
#           if item in seen:
#               duplicates.add(item)
#           seen.add(item)
#       return list(duplicates)
#   find_duplicates([1, 2, 3, 2, 4, 3])  # [2, 3]


# PROBLEM 4: Count character frequency
#   def char_frequency(s: str) -> dict:
#       freq = {}
#       for char in s:
#           freq[char] = freq.get(char, 0) + 1
#       return freq
#   # OR: from collections import Counter; Counter(s)


# PROBLEM 5: FizzBuzz (classic!)
#   def fizzbuzz(n: int) -> list:
#       result = []
#       for i in range(1, n + 1):
#           if i % 15 == 0: result.append("FizzBuzz")
#           elif i % 3 == 0: result.append("Fizz")
#           elif i % 5 == 0: result.append("Buzz")
#           else: result.append(str(i))
#       return result


# PROBLEM 6: Flatten a nested list
#   def flatten(lst: list) -> list:
#       result = []
#       for item in lst:
#           if isinstance(item, list):
#               result.extend(flatten(item))  # recursion
#           else:
#               result.append(item)
#       return result
#   flatten([1, [2, [3, 4]], 5])  # [1, 2, 3, 4, 5]


# PROBLEM 7: Two Sum (find two numbers that add to target)
#   def two_sum(nums: list, target: int) -> list:
#       seen = {}
#       for i, num in enumerate(nums):
#           complement = target - num
#           if complement in seen:
#               return [seen[complement], i]
#           seen[num] = i
#       return []
#   two_sum([2, 7, 11, 15], 9)  # [0, 1] (2 + 7 = 9)


# PROBLEM 8: Remove duplicates preserving order
#   def remove_duplicates(lst: list) -> list:
#       seen = set()
#       result = []
#       for item in lst:
#           if item not in seen:
#               seen.add(item)
#               result.append(item)
#       return result
#   # OR: list(dict.fromkeys(lst))  # Python 3.7+ preserves order


# PROBLEM 9: Check if two strings are anagrams
#   def is_anagram(s1: str, s2: str) -> bool:
#       return sorted(s1.lower()) == sorted(s2.lower())
#   is_anagram("listen", "silent")  # True


# PROBLEM 10: Find the second largest number
#   def second_largest(nums: list) -> int:
#       unique = list(set(nums))
#       unique.sort()
#       return unique[-2] if len(unique) >= 2 else None
#   second_largest([5, 2, 8, 1, 8])  # 5


# ██████████████████████████████████████████████████████████████████████████████
# PART D: PYTHON FOR GENAI (How Python Connects to Your Work)
# ██████████████████████████████████████████████████████████████████████████████


# 1. PYDANTIC (used everywhere in GenAI)
#    Data validation library. Defines schemas with type checking.
#    class GradeDocuments(BaseModel):
#        binary_score: Literal["yes", "no"] = Field(description="...")
#    Used for: structured LLM output, API validation, config management.
#    In DocSage: grading, routing, hallucination check all use Pydantic.

# 2. TYPING MODULE (TypedDict, Annotated, Literal)
#    TypedDict: typed dictionary for LangGraph state.
#    Annotated: adds metadata (reducers in LangGraph).
#    Literal: constrains values ("yes" | "no", "vectorstore" | "web_search").
#    These are the building blocks of LangGraph state definitions.

# 3. DOTENV (environment variables)
#    from dotenv import load_dotenv
#    load_dotenv()  # loads .env file into os.environ
#    api_key = os.getenv("GROQ_API_KEY", "")
#    NEVER hardcode API keys. Always use .env files.

# 4. LOGGING (not print!)
#    import logging
#    logger = logging.getLogger(__name__)
#    logger.info("Processing document: %s", filename)
#    Production code uses logging, not print(). Configurable levels,
#    timestamps, and output destinations.

# 5. DATACLASSES vs PYDANTIC
#    dataclass: lightweight, no validation. Good for simple data containers.
#    Pydantic BaseModel: validation, serialization, JSON schema.
#    For GenAI: always use Pydantic (LangChain/LangGraph expect it).


# ██████████████████████████████████████████████████████████████████████████████
# PART E: INTERVIEW Q&A - 25 Questions
# ██████████████████████████████████████████████████████████████████████████████
#
# Q1: What is the difference between a list and a tuple?
# A: List is mutable (can change), tuple is immutable (cannot change).
#    Tuples are faster and can be used as dict keys. Use tuples for data
#    that shouldn't change (coordinates, DB records).
#
# Q2: What are decorators? Give an example.
# A: A decorator wraps a function to add behavior. @decorator is syntactic
#    sugar for func = decorator(func). Example: @tool in LangChain makes
#    a function callable by an agent. @lru_cache caches function results.
#
# Q3: What is a generator? How is it different from a list?
# A: A generator uses yield to produce values lazily, one at a time.
#    A list stores all values in memory. Generators are memory-efficient
#    for large data. LLM streaming uses generators.
#
# Q4: Explain the GIL.
# A: The Global Interpreter Lock allows only one thread to execute Python
#    code at a time. It's a bottleneck for CPU-bound tasks but not for
#    I/O-bound tasks like API calls. GenAI work is mostly I/O-bound.
#
# Q5: What is the difference between == and is?
# A: == checks value equality. is checks identity (same object in memory).
#    [1,2] == [1,2] is True. [1,2] is [1,2] is False (different objects).
#
# Q6: What is a mutable default argument trap?
# A: def f(lst=[]): lst.append(1); return lst. The default list is created
#    once and shared across calls. f() returns [1], f() returns [1,1].
#    Fix: use None as default, create new list inside.
#
# Q7: What is *args and **kwargs?
# A: *args collects extra positional arguments as a tuple. **kwargs collects
#    extra keyword arguments as a dict. Together they make functions flexible.
#
# Q8: Explain list comprehension vs generator expression.
# A: [x**2 for x in range(10)] creates a list in memory immediately.
#    (x**2 for x in range(10)) creates a generator that computes lazily.
#    Use generators for large data, lists when you need all values.
#
# Q9: What is the difference between shallow copy and deep copy?
# A: Shallow copy copies the outer object but shares inner objects.
#    Deep copy copies everything recursively. For nested structures,
#    shallow copy can cause unexpected mutations.
#
# Q10: What are dunder methods?
# A: Double underscore methods like __init__, __str__, __len__. They define
#     how objects behave with Python operations. __init__ is the constructor,
#     __str__ is for print(), __iter__ makes objects iterable.
#
# Q11: What is the LEGB rule?
# A: Variable lookup order: Local, Enclosing, Global, Built-in. Python
#     searches for variables in this order. If not found in any, NameError.
#
# Q12: How do you handle exceptions in Python?
# A: try/except/else/finally. Catch specific exceptions first, then general.
#     else runs on success, finally always runs for cleanup. I wrap every
#     LLM call in try/except with meaningful fallbacks.
#
# Q13: What is async/await?
# A: Asynchronous programming for concurrent I/O. await pauses the current
#     function and lets others run. asyncio.gather runs multiple coroutines
#     concurrently. LangGraph supports async with ainvoke/astream.
#
# Q14: What is Pydantic and why is it used in GenAI?
# A: Pydantic validates data using Python type hints. In GenAI, it's used
#     for structured LLM output (with_structured_output forces the LLM to
#     return a valid Pydantic object), API validation, and config management.
#
# Q15: What is the difference between a module and a package?
# A: A module is a single .py file. A package is a directory with __init__.py
#     containing multiple modules. In my project, docsage is a package with
#     sub-packages: config, llms, nodes, tools, graph, ui.
#
# Q16: How do you manage dependencies in Python?
# A: Virtual environments (.venv) isolate project dependencies.
#     requirements.txt pins versions. pip install -r requirements.txt
#     installs everything. Never install globally.
#
# Q17: What is the difference between @staticmethod and @classmethod?
# A: @staticmethod doesn't receive self or cls. It's just a function inside
#     a class. @classmethod receives cls (the class itself) as first argument.
#     Use classmethod for factory methods, staticmethod for utility functions.
#
# Q18: What is a context manager?
# A: The with statement ensures resources are cleaned up. Uses __enter__
#     and __exit__. Common: file handling, DB connections, SMTP connections.
#     In my project: with smtplib.SMTP(...) as server for email sending.
#
# Q19: What is the walrus operator (:=)?
# A: Assignment expression. Assigns and returns a value in one expression.
#     if (n := len(data)) > 10: print(f"Too long: {n}")
#     Useful in while loops and comprehensions.
#
# Q20: Write a function to check if a string is a palindrome.
# A: def is_palindrome(s): s = s.lower().replace(" ", ""); return s == s[::-1]
#
# Q21: Write a function to find duplicates in a list.
# A: def find_dupes(lst): seen = set(); return [x for x in lst if x in seen or seen.add(x)]
#    Or cleaner: from collections import Counter; [k for k,v in Counter(lst).items() if v > 1]
#
# Q22: What is the difference between append() and extend()?
# A: append adds ONE item: [1,2].append([3,4]) -> [1,2,[3,4]]
#    extend adds EACH item: [1,2].extend([3,4]) -> [1,2,3,4]
#
# Q23: How does Python memory management work?
# A: Python uses reference counting + garbage collection. Each object has
#     a reference count. When it reaches 0, memory is freed. The garbage
#     collector handles circular references. The GIL protects reference counts.
#
# Q24: What is monkey patching?
# A: Modifying a class or module at runtime. Example: replacing a method
#     during testing. Powerful but dangerous — can break code unexpectedly.
#     Use sparingly, prefer dependency injection instead.
#
# Q25: How does Python relate to your GenAI work?
# A: "Python is the primary language for GenAI development. I use it daily
#     with LangChain, LangGraph, and Streamlit. Key Python features I rely
#     on: decorators (@tool for LangChain tools), type hints (TypedDict for
#     state, Pydantic for structured output), generators (LLM streaming),
#     async (concurrent API calls), context managers (resource cleanup),
#     and the rich ecosystem of AI/ML libraries."
