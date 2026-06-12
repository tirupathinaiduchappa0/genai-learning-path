"""
================================================================================
PYTHON CODING QUESTIONS WALKTHROUGH — Screening Test Mastery
================================================================================

Built for fast revision before walk-in / online Python screening tests
(Cognizant, Capgemini, TCS, Infosys, Accenture, GenAI product companies).

HOW THIS FILE IS ORGANIZED:
    PART 0 — THE 10 COGNIZANT QUESTIONS (quick 10-15 min revision)
             Read these first. Each has: snippet, simple answer, traps,
             variations, and a brief explanation.
    PART 1 — BASIC (Q1-Q25)
    PART 2 — INTERMEDIATE (Q26-Q65)        [added in later batches]
    PART 3 — MEDIUM-ADVANCED (Q66-Q115)    [added in later batches]
    PART 4 — PRODUCTION & GENAI (Q116-Q160)[added in later batches]

HOW TO USE:
    - Try to predict the output BEFORE reading the answer.
    - Run this file to see real outputs:
          python 06_coding_questions_walkthrough.py
    - For each question, internalize the TRAP, not just the answer.

THE 5 TRAP CATEGORIES THAT COST MARKS (memorize these):
    1. Slicing arithmetic     -> stop index is EXCLUSIVE; s[1:3] = idx 1,2
    2. Late-binding closures  -> loop var captured by reference, not value
    3. Mutable aliasing       -> b = a shares object; += mutates in place
    4. MRO / super() chains   -> super follows MRO, not the literal parent
    5. Algorithm tracing      -> sliding window, two-pointer, stack, binary search
================================================================================
"""


# ##############################################################################
# TRAP INDEX — 5-STAR QUESTIONS GROUPED BY CATEGORY (10-MIN PRE-INTERVIEW SKIM)
# ##############################################################################
"""
Read ONLY this block in the final 10 minutes before you walk in. Every entry
is a 5-star (must-focus) trap. If you can answer all of these cold, you pass.

────────────────────────────────────────────────────────────────────────────
A. MUTABILITY / ALIASING / COPY  (your Cognizant Q1 + Q6 family)
────────────────────────────────────────────────────────────────────────────
  CogQ1  b=a aliases; a+=[4] mutates in place; copy() is independent. is vs ==.
  CogQ6  def f(x, acc=[]) -> default list is SHARED across calls. Use None.
  Q22    b = a does NOT copy; mutating b changes a.
  Q52    mutable CLASS attribute is shared by ALL instances. Init in __init__.
  Q60    mutable default arg created ONCE. Use None sentinel.
  Q65    default arg binds to the OUTER object once (same root cause).
  Q66    .copy()/[:] is SHALLOW; nested inner lists are still shared.
  Q68    [[0]*3]*2 repeats the SAME row reference. Use a comprehension.
  Q108   never add/remove from a list while iterating it; iterate a copy.
  Q146   mutable default history=[] LEAKS one user's data into another's call.
  Q158   passing a dict/list to a function -> mutating it affects the caller.
  RULE:  '=' aliases | '+=' mutates lists in place | use None for mutable defaults
         | deepcopy for nested independence | 'is' only for None/True/False.

────────────────────────────────────────────────────────────────────────────
B. CLOSURES / SCOPE  (your Cognizant Q4)
────────────────────────────────────────────────────────────────────────────
  CogQ4  [lambda: i for i in range(3)] -> all return 2 (late binding). Fix: i=i.
  Q36    same late-binding trap; bind value with default arg lambda i=i.
  Q112   assigning a name anywhere in a func makes it LOCAL -> UnboundLocalError.
  RULE:  closures read the variable LATE (at call); default args bind EARLY (at def).

────────────────────────────────────────────────────────────────────────────
C. IDENTITY / CONTROL-FLOW TRAPS
────────────────────────────────────────────────────────────────────────────
  Q13    == is value, is is identity; small-int caching makes 'is' lie. Use ==.
  Q17    for-else: else runs ONLY if the loop did NOT break.
  Q129   a return inside finally OVERRIDES try's return and swallows errors.

────────────────────────────────────────────────────────────────────────────
D. ADVANCED OOP  (your Cognizant Q5)
────────────────────────────────────────────────────────────────────────────
  CogQ5  diamond MRO: D(B,C) -> super() follows MRO (B->C->A) = "BCA".
  Q73    same MRO/super diamond. Check with Class.__mro__.
  Q76    descriptors (__get__/__set__) power @property, ORM fields, validators.
  Q77    metaclass __new__ runs at CLASS creation (Django/Pydantic use this).

────────────────────────────────────────────────────────────────────────────
E. CONCURRENCY / ASYNC / GIL
────────────────────────────────────────────────────────────────────────────
  Q86    GIL = one thread runs Python bytecode at a time. Threads help I/O, not CPU.
  Q88    'await coro' returns its value; bare fetch() returns a coroutine object.
  Q89    asyncio.gather runs coroutines CONCURRENTLY, results in argument order.
  RULE:  I/O-bound -> threads/asyncio | CPU-bound -> multiprocessing.

────────────────────────────────────────────────────────────────────────────
F. ALGORITHM TRACING  (your Cognizant Q7-Q10)
────────────────────────────────────────────────────────────────────────────
  CogQ7 / Q92   set toggle / XOR -> element with ODD count survives.
  CogQ8 / Q93   sliding window: add nums[i], drop nums[i-k]. O(n).
  CogQ9 / Q94   balanced brackets -> stack (LIFO); remember 'return not st'.
  CogQ10/ Q95   binary search on SORTED array; 'while lo <= hi'. O(log n).
  Q97           two-sum on UNSORTED -> hash map in one pass.

────────────────────────────────────────────────────────────────────────────
G. DECORATORS
────────────────────────────────────────────────────────────────────────────
  Q50    wrapper(*args, **kwargs) lets a decorator wrap ANY signature.
  Q126   retry decorator: parametrized decorator + try/except loop (+backoff).

────────────────────────────────────────────────────────────────────────────
H. GENAI / PYDANTIC / JSON / EMBEDDINGS  (your daily work — own these)
────────────────────────────────────────────────────────────────────────────
  Q115/131/132  Pydantic COERCES "0.7"->0.7, "3"->3; bad type -> ValidationError.
  Q117          LLMs wrap JSON in prose/fences -> slice {..} + try/except.
  Q142          literal braces in a prompt MUST be doubled {{ }} or KeyError.
  Q145          reassemble streamed tokens with ''.join.
  Q136/138      cosine = dot/(||a||*||b||); top-k = score, gate, sort desc, slice.
  Q160          RAG retriever = embed -> similarity -> threshold gate -> top-k.

────────────────────────────────────────────────────────────────────────────
I. NUMERIC / DATETIME / SETS
────────────────────────────────────────────────────────────────────────────
  Q121   can't subtract NAIVE from AWARE datetime -> TypeError. Store UTC-aware.
  Q147   never compare floats with ==; use round() or math.isclose.
  Q150   required <= candidate tests subset (all required tags present).
  Q153   sort items by (-value, key) for value-desc, key-asc tie-break.
  Q155   use Decimal (not float) for money/token-cost accounting.

THE ONE HABIT: for EVERY snippet ask "is there a trap?" -> mutable default? in-place
op? closure in a loop? 'in' checking keys not values? slice stop exclusive? is vs ==?
That single question moves you from 4/10 to 8/10.
"""


# ##############################################################################
# PART 0 — THE 10 COGNIZANT QUESTIONS (QUICK REVISION)
# ##############################################################################

print("=" * 70)
print("PART 0 — THE 10 COGNIZANT QUESTIONS")
print("=" * 70)


# ------------------------------------------------------------------------------
# COGNIZANT Q1 — Reference, copy, in-place add, is vs ==
# ------------------------------------------------------------------------------
a = [1, 2, 3]
b = a
c = a.copy()
a += [4]
print("CogQ1:", b == a, c == a, b is a, c is a)   # True False True False
"""
SIMPLE ANSWER: True False True False

WHY:
    b = a        -> b and a are the SAME object (alias).
    c = a.copy() -> c is a NEW independent list [1, 2, 3].
    a += [4]     -> in-place extend; mutates the object a (and b) point to.
                    Now a and b are [1,2,3,4]; c stays [1,2,3].
    b == a  -> True  (same object, equal values)
    c == a  -> False ([1,2,3] vs [1,2,3,4])
    b is a  -> True  (identical object)
    c is a  -> False (different objects)

TRAPS / PITFALLS:
    - Thinking b = a makes a copy. It does NOT — it makes an alias.
    - Forgetting that += on a list is IN-PLACE (calls __iadd__ / extend),
      so it changes the shared object rather than rebinding a new one.
    - Mixing up == (value equality) with is (identity / same object).

VARIATIONS THEY ASK:
    - Replace a += [4] with a = a + [4]. Now a is REBOUND to a NEW list,
      so b stays [1,2,3]:  b == a -> False, b is a -> False.
    - Use b = a[:] or list(a) instead of copy() (same effect: independent copy).
    - Nested lists + copy(): copy() is SHALLOW, inner lists are still shared.

ONE-LINE RULE: 'b = a' aliases; '+=' mutates in place; use 'is' only for None.
"""


# ------------------------------------------------------------------------------
# COGNIZANT Q2 — set symmetric difference (^) then sorted
# ------------------------------------------------------------------------------
t = (1, 2, 2, 3, 4)
l = [2, 4, 4, 5]
out = sorted((set(t) ^ set(l)))
print("CogQ2:", out)                                # [1, 3, 5]
"""
SIMPLE ANSWER: [1, 3, 5]

WHY:
    set(t) = {1, 2, 3, 4}     (duplicates dropped)
    set(l) = {2, 4, 5}
    ^  = symmetric difference = elements in EXACTLY ONE set
       = {1, 3, 5}   (2 and 4 are in both, so excluded)
    sorted(...) -> [1, 3, 5]

TRAPS / PITFALLS:
    - Confusing ^ (symmetric difference) with & (intersection -> {2,4})
      or | (union -> {1,2,3,4,5}) or - (difference).
    - Forgetting set() removes duplicates first.
    - sets are UNORDERED; you must sort to get a predictable list.

THE 4 SET OPERATORS (memorize):
    a & b -> intersection      (in BOTH)
    a | b -> union             (in EITHER)
    a - b -> difference        (in a, NOT in b)
    a ^ b -> symmetric diff    (in exactly ONE, i.e. (a|b) - (a&b))

VARIATIONS THEY ASK:
    - Ask for a & b -> {2, 4} -> [2, 4]
    - Ask for a - b -> {1, 3} -> [1, 3]   (order matters: b - a -> {5})
"""


# ------------------------------------------------------------------------------
# COGNIZANT Q3 — list comprehension filter + slice (YOU MISSED THIS)
# ------------------------------------------------------------------------------
arr = [9, 1, 8, 2, 7, 3]
res = sorted([x for x in arr if x % 2 == 1])[1:3]
print("CogQ3:", res)                                # [3, 7]
"""
SIMPLE ANSWER: [3, 7]   (NOT [1, 3])

WHY:
    Odd numbers from arr: [9, 1, 7, 3]
    sorted(...)         : [1, 3, 7, 9]
    [1:3]               : indexes 1 and 2  -> 3 and 7  -> [3, 7]

TRAPS / PITFALLS (this is the one that cost you):
    - Slice STOP is EXCLUSIVE. [1:3] = positions 1,2 — NOT 1,2,3.
    - Slice START is INCLUSIVE and 0-based, so it skips index 0 (the '1').
    - People stop after sorting and forget to apply the slice carefully.

SLICE RULE: s[start:stop] gives indexes start .. stop-1. Count = stop - start.

VARIATIONS THEY ASK:
    - [::2]  -> every other element -> [1, 7]
    - [-2:]  -> last two -> [7, 9]
    - even filter (x % 2 == 0) -> [8, 2] -> sorted [2, 8] -> [1:3] -> [8]
    - sorted(..., reverse=True)[:2] -> top 2 -> [9, 7]
"""


# ------------------------------------------------------------------------------
# COGNIZANT Q4 — late-binding closures (YOU MISSED THIS — THE classic)
# ------------------------------------------------------------------------------
funcs = []
for i in range(3):
    funcs.append(lambda: i)
print("CogQ4:", [f() for f in funcs])    # [2, 2, 2]


funcs = []

for i in range(3):
    def f():
        return i
    funcs.append(f)

print([fn() for fn in funcs])            # [2, 2, 2]

#Interviewers often replace lambda with a function.Same reason.
#Each function closes over the same variable i.
"""
SIMPLE ANSWER: [2, 2, 2]   (NOT [0, 1, 2], NOT [1, 2])

WHY:
    Each lambda captures the VARIABLE i, not its value at creation time.
    By the time the lambdas are CALLED, the loop has finished and i == 2.
    So all three return 2.

TRAPS / PITFALLS:
    - Assuming each lambda 'remembers' the i from its iteration. It doesn't.
    - Closures capture variables by REFERENCE (late binding), evaluated
      only when the function is actually called.

THE FIX (default-argument binding — capture value NOW):
    funcs.append(lambda i=i: i)   ->  [0, 1, 2]
    Because default args are evaluated at definition time.

VARIATIONS THEY ASK:
    - Same idea with def inside a loop, or with functools/map.
    - With list of multipliers: [lambda x: x*i ...] -> all use final i.
    - 'How do you fix it?' -> answer: default arg, or functools.partial.

ONE-LINE RULE: loop var + lambda = all share the LAST value. Bind with i=i.
"""


# ------------------------------------------------------------------------------
# COGNIZANT Q5 — MRO / super() diamond inheritance
# ------------------------------------------------------------------------------
class A:
    def who(self):
        return "A"

class B(A):
    def who(self):
        return "B" + super().who()

class C(A):
    def who(self):
        return "C" + super().who()

class D(B, C):
    pass

print("CogQ5:", D().who())                           # BCA
"""
SIMPLE ANSWER: "BCA"

WHY:
    MRO (Method Resolution Order) of D is: D -> B -> C -> A -> object
    (C3 linearization). super() follows the MRO, NOT the literal parent.
    D().who() -> B.who(): "B" + super() (next in MRO is C, not A!)
              -> C.who(): "C" + super() (next in MRO is A)
              -> A.who(): "A"
    Concatenate: "B" + "C" + "A" = "BCA"

TRAPS / PITFALLS:
    - Assuming B's super() goes straight to A. In a diamond, B's super()
      resolves to C because of D's MRO. This is the whole point of the question.
    - Thinking the answer is "BA" (forgetting C is in the chain).

HOW TO CHECK MRO:
    print(D.__mro__)   or   print(D.mro())
    -> [D, B, C, A, object]

VARIATIONS THEY ASK:
    - Swap class D(C, B) -> MRO becomes D->C->B->A -> output "CBA".
    - Add print statements / counters to trace call order.
    - Ask 'what is the MRO?' directly, or 'what algorithm computes it?' (C3).
"""


# ------------------------------------------------------------------------------
# COGNIZANT Q6 — mutable default argument (THE most famous Python trap)
# ------------------------------------------------------------------------------
def f(x, acc=[]):
    acc.append(x)
    return acc

print("CogQ6:", f(10))                               # [10]
print("CogQ6:", f(20))                               # [10, 20]  (!!)
"""
SIMPLE ANSWER: first call [10], second call [10, 20]

WHY:
    The default list acc=[] is created ONCE, when the function is DEFINED,
    not on each call. Every call without acc reuses that SAME list, so
    values accumulate across calls.

TRAPS / PITFALLS:
    - Expecting a fresh [] each call. Default values are evaluated once.
    - This bug is silent and dangerous in real code (shared state).

THE FIX (use None sentinel):
    def f(x, acc=None):
        if acc is None:
            acc = []          # fresh list every call
        acc.append(x)
        return acc

VARIATIONS THEY ASK:
    - Same with acc={} (dict) or acc=set().
    - 'Why does this happen / how do you fix it?' -> None sentinel.
    - Mix with a third call f(30, [99]) -> uses the passed list -> [99, 30].

ONE-LINE RULE: NEVER use mutable defaults ([], {}, set()). Use None + create inside.
"""


# ------------------------------------------------------------------------------
# COGNIZANT Q7 — set toggle (find the element appearing an odd number of times)
# ------------------------------------------------------------------------------
nums = [4, 1, 2, 1, 2]
seen = set()
ans = None
for n in nums:
    if n in seen:
        seen.remove(n)
    else:
        seen.add(n)
ans = seen.pop()
print("CogQ7:", ans)                                 # 4
"""
SIMPLE ANSWER: 4

WHY:
    The loop TOGGLES membership: add if absent, remove if present.
    An element added an EVEN number of times cancels out (ends up removed).
    An element added an ODD number of times remains in the set.
    Trace: 4->{4}, 1->{4,1}, 2->{4,1,2}, 1->{4,2}, 2->{4}
    Only 4 (appears once = odd) remains. seen.pop() returns 4.

TRAPS / PITFALLS:
    - Misreading the toggle logic as 'remove duplicates'.
    - set.pop() removes an ARBITRARY element — fine here because only one
      element remains, but unreliable if the set had several.

VARIATIONS THEY ASK:
    - This is the XOR 'single number' pattern. Same result with:
          ans = 0
          for n in nums: ans ^= n     # XOR cancels pairs -> 4
    - 'Find the element appearing an odd number of times' -> XOR is the trick.
"""


# ------------------------------------------------------------------------------
# COGNIZANT Q8 — sliding window maximum sum of k consecutive elements
# ------------------------------------------------------------------------------
def max_sum_k(nums, k):
    w = sum(nums[:k])          # sum of first window
    best = w
    for i in range(k, len(nums)):
        w += nums[i] - nums[i - k]   # slide: add new, drop oldest
        best = max(best, w)
    return best

print("CogQ8:", max_sum_k([2, 1, 5, 1, 3, 2], 3))    # 9
"""
SIMPLE ANSWER: 9

WHY (sliding window):
    First window [2,1,5] = 8.
    Slide: add nums[3]=1, drop nums[0]=2 -> [1,5,1] = 7
    Slide: add nums[4]=3, drop nums[1]=1 -> [5,1,3] = 9   <- best
    Slide: add nums[5]=2, drop nums[2]=5 -> [1,3,2] = 6
    Max = 9.

TRAPS / PITFALLS:
    - In the original screenshot, 'best = W' had a capital W (typo / OCR).
      In real Python that is a NameError. The intended code is best = w.
    - Recomputing sum each window is O(n*k); the slide trick is O(n).
    - Off-by-one in the window: w += nums[i] - nums[i-k].

VARIATIONS THEY ASK:
    - Minimum sum window (use min instead of max).
    - Maximum average (best / k).
    - Return the window itself, not just the sum.

PATTERN TO RECOGNIZE: 'k consecutive elements' => sliding window, O(n).
"""


# ------------------------------------------------------------------------------
# COGNIZANT Q9 — balanced brackets using a stack (deque)
# ------------------------------------------------------------------------------
from collections import deque

def valid(s):
    st = deque()
    mp = {')': '(', ']': '[', '}': '{'}
    for ch in s:
        if ch in "([{":
            st.append(ch)
        else:
            if not st or st.pop() != mp[ch]:
                return False
    return not st            # True only if stack is empty at the end

print("CogQ9:", valid("([{}])"), valid("([)]"))      # True False
"""
SIMPLE ANSWER: True False

WHY (stack matching):
    "([{}])" -> push (, [, { ; then }] match {, [, ( in reverse -> empty -> True
    "([)]"   -> push (, [ ; ')' expects matching '(' but pop() gives '[' -> False

TRAPS / PITFALLS:
    - Forgetting the FINAL 'return not st': "(((" has no mismatch but leaves
      a non-empty stack -> must return False. 'not st' handles that.
    - 'not st or st.pop() != mp[ch]': the 'not st' guard prevents popping an
      empty stack (a stray closing bracket like ")" alone).
    - Order matters: brackets must close in LIFO (last-opened-first-closed).

VARIATIONS THEY ASK:
    - Single bracket type only (just count, no stack needed).
    - Return the index of the first mismatch.
    - Use a plain list as the stack (append / pop) instead of deque.

PATTERN TO RECOGNIZE: 'matching / nesting / balanced' => stack (LIFO).
"""


# ------------------------------------------------------------------------------
# COGNIZANT Q10 — binary search (note: original 'l' looked like '1' via OCR)
# ------------------------------------------------------------------------------
def bs(arr, target):
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        if arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1

print("CogQ10:", bs([1, 3, 5, 7, 9], 7), bs([1, 3, 5, 7, 9], 6))   # 3 -1
"""
SIMPLE ANSWER: 3 -1

WHY (binary search on a SORTED array):
    Search 7 in [1,3,5,7,9]:
        lo=0 hi=4 mid=2 arr[2]=5 < 7 -> lo=3
        lo=3 hi=4 mid=3 arr[3]=7 == 7 -> return 3
    Search 6 in [1,3,5,7,9]:
        narrows down, never finds 6 -> loop ends -> return -1

TRAPS / PITFALLS:
    - The original used the variable name 'l' (lowercase L) which OCR shows
      as '1'. '1, r = 0, ...' would be a SyntaxError; it's really 'l, r'.
      I renamed to lo/hi for clarity — never name a variable 'l'.
    - Binary search REQUIRES a sorted array. On unsorted input it's wrong.
    - 'while lo <= hi' (not <) — using < misses the last element.
    - mid = (lo + hi) // 2 — integer division.

VARIATIONS THEY ASK:
    - Return True/False instead of index.
    - Find first/last occurrence of a duplicate (lower/upper bound).
    - Recursive version of binary search.

PATTERN TO RECOGNIZE: 'sorted array + find' => binary search, O(log n).
"""

print()
print("=" * 70)
print("PART 0 COMPLETE — revise the 5 trap categories above before any test.")
print("=" * 70)

# The 7 to drill before any test (these are the mark-stealers): Q7 tuple-comma,
#  Q8 in=keys, Q12 bool=int, Q13 is-vs-==, Q17 for-else, Q19 return-vs-print, Q22 aliasing.
#  Note Q3 (slicing) and Q22 (aliasing) directly map to your Cognizant Q3 and Q1 misses.



# tricky ones worth a second read: Q28 (ternary in comprehension), Q31 (nested loop order),
#  Q34 (reduce initializer), Q42 (zip(*) transpose), Q45 (except ordering), Q56 (static vs classmethod),
#   Q59 (slice copy), Q62/Q63 (JSON parsing).aliasing (Q22, Q59) and mutable/late-binding (Q36, Q60, Q65).



# Q117 (LLM JSON), Q121 (naive/aware datetime), Q142 (escaped braces — a very common LangChain bug), Q146 + Q158 (mutable-state leaks across users — a genuine security issue in multi-user GenAI services), and Q160 (the retrieval-gate pipeline, which ties straight back to your RAG lessons).

# ##############################################################################
# PART 1 — BASIC (Q1-Q25)
# ##############################################################################
# Star ratings:  (1) trivial  (2) easy  (3) medium  (4) tricky  (5) must-focus trap
# Mix in this batch: ~15 easy, ~7 medium, ~3 trap. Difficulty rises toward Q25.
# ##############################################################################

print()
print("=" * 70)
print("PART 1 — BASIC (Q1-Q25)")
print("=" * 70)


# ------------------------------------------------------------------------------
# Q1. (star 1) [Output Prediction] — arithmetic operators
# ------------------------------------------------------------------------------
print("Q1:", 7 / 2, 7 // 2, 7 % 2, 7 ** 2)            # 3.5 3 1 49
"""
RATING: * (trivial)
SIMPLE ANSWER: 3.5 3 1 49
TRICK/PATTERN: 4 operators — /=true div (float), //=floor, %=remainder, **=power.
PITFALL: '/' ALWAYS returns a float in Python 3 (7/2 -> 3.5, not 3).
EXPLAIN: // floors toward negative infinity; -7//2 = -4, not -3 (watch negatives).
"""


# ------------------------------------------------------------------------------
# Q2. (star 1) [Output Prediction] — string reverse slice
# ------------------------------------------------------------------------------
print("Q2:", "GenAI"[::-1])                            # IAneG
"""
RATING: * (trivial)
SIMPLE ANSWER: IAneG
TRICK/PATTERN: s[::-1] = reverse. Step -1 walks the string backwards.
PITFALL: Case is preserved; only ORDER reverses (G-e-n-A-I -> I-A-n-e-G).
EXPLAIN: Slicing syntax is s[start:stop:step]; empty start/stop + step -1 = full reverse.
"""


# ------------------------------------------------------------------------------
# Q3. (star 3) [Output Prediction] — slicing indexes (inclusive/exclusive)
# ------------------------------------------------------------------------------
s = "LangChain"
print("Q3:", s[0:4], s[4:], s[-5:])                    # Lang Chain Chain
"""
RATING: *** (medium — slicing arithmetic)
SIMPLE ANSWER: Lang Chain Chain
TRICK/PATTERN: s[a:b] = indexes a..b-1. STOP is EXCLUSIVE.
PITFALL: s[0:4] is 'Lang' (idx 0,1,2,3) NOT 'LangC'. Negative -5 counts from end.
EXPLAIN: L0 a1 n2 g3 C4 h5 a6 i7 n8. s[4:]='Chain', s[-5:]= last 5 = 'Chain'.
"""


# ------------------------------------------------------------------------------
# Q4. (star 2) [Short Answer] — string immutability
# ------------------------------------------------------------------------------
# s = "hello"; s[0] = "H"  ->  TypeError
print("Q4:", "H" + "hello"[1:])                        # Hello (the correct way)
"""
RATING: ** (easy concept, common mistake)
SIMPLE ANSWER: s[0]='H' raises TypeError; strings are IMMUTABLE.
TRICK/PATTERN: To 'edit' a string, build a NEW one: 'H' + s[1:].
PITFALL: Assuming strings behave like lists. Lists allow item assignment, strings don't.
EXPLAIN: Immutability is why repeated += in a loop is slow (new object each time);
         use ''.join(list) for building big strings.
"""


# ------------------------------------------------------------------------------
# Q5. (star 2) [Output Prediction] — append adds ONE element
# ------------------------------------------------------------------------------
nums = [1, 2, 3]
nums.append([4, 5])
print("Q5:", len(nums), nums)                          # 4 [1, 2, 3, [4, 5]]
"""
RATING: ** (easy, append vs extend trap)
SIMPLE ANSWER: 4  and  [1, 2, 3, [4, 5]]
TRICK/PATTERN: append(x) adds x as a SINGLE element (even if x is a list).
PITFALL: Expecting length 5. append never unpacks; extend() does.
EXPLAIN: nums[3] is the nested list [4,5]; nums[3][0] == 4.
"""


# ------------------------------------------------------------------------------
# Q6. (star 2) [Output Prediction] — extend vs append
# ------------------------------------------------------------------------------
a = [1, 2]
a.extend([3, 4])
a.append([5, 6])
print("Q6:", a)                                        # [1, 2, 3, 4, [5, 6]]
"""
RATING: ** (easy, reinforces Q5)
SIMPLE ANSWER: [1, 2, 3, 4, [5, 6]]
TRICK/PATTERN: extend UNPACKS the iterable; append adds it whole.
PITFALL: Mixing them up. extend([3,4]) -> 3 and 4 separately; append([5,6]) -> one element.
EXPLAIN: a += [3,4] is equivalent to a.extend([3,4]).
"""


# ------------------------------------------------------------------------------
# Q7. (star 4) [Output Prediction] — single-element tuple needs a comma
# ------------------------------------------------------------------------------
t = (1,)
t2 = (1)
print("Q7:", type(t).__name__, type(t2).__name__)     # tuple int
"""
RATING: **** (tricky — classic misconception)
SIMPLE ANSWER: tuple  int
TRICK/PATTERN: A tuple is made by the COMMA, not the parentheses.
PITFALL: (1) is just int 1 in grouping parens. (1,) is a 1-element tuple.
EXPLAIN: Even without parens: x = 1, 2 is a tuple. And t = 1, is a 1-tuple.
"""


# ------------------------------------------------------------------------------
# Q8. (star 4) [Output Prediction] — 'in' checks KEYS for a dict
# ------------------------------------------------------------------------------
d = {"a": 1, "b": 2}
d["c"] = 3
print("Q8:", len(d), "a" in d, 1 in d)                 # 3 True False
"""
RATING: **** (tricky — very common trap)
SIMPLE ANSWER: 3 True False
TRICK/PATTERN: 'x in dict' tests KEYS only, never values.
PITFALL: '1 in d' is False because 1 is a VALUE, not a key.
EXPLAIN: To test values: 1 in d.values(). To test pairs: ('a',1) in d.items().
"""


# ------------------------------------------------------------------------------
# Q9. (star 2) [Short Answer / GenAI] — dict.get with default
# ------------------------------------------------------------------------------
config = {"model": "gpt-4"}
print("Q9:", config.get("temperature", 0.7))          # 0.7
"""
RATING: ** (easy, GenAI-relevant)
SIMPLE ANSWER: 0.7
TRICK/PATTERN: .get(key, default) returns default if key missing (no crash).
PITFALL: config['temperature'] would raise KeyError. .get is the safe read.
EXPLAIN: Everywhere in GenAI config: reading optional LLM params
         (temperature, max_tokens, top_p) without KeyError.
"""


# ------------------------------------------------------------------------------
# Q10. (star 1) [Output Prediction] — set drops duplicates
# ------------------------------------------------------------------------------
print("Q10:", len({1, 2, 2, 3, 3, 3}))                 # 3
"""
RATING: * (trivial)
SIMPLE ANSWER: 3
TRICK/PATTERN: A set stores UNIQUE elements only.
PITFALL: Counting the literal items (6). Duplicates collapse to {1,2,3}.
EXPLAIN: set() is the idiomatic way to dedupe: len(set(arr)) = unique count.
"""


# ------------------------------------------------------------------------------
# Q11. (star 3) [Output Prediction] — set operators
# ------------------------------------------------------------------------------
a = {1, 2, 3}
b = {2, 3, 4}
print("Q11:", a & b, a | b, a - b, a ^ b)              # {2,3} {1,2,3,4} {1} {1,4}
"""
RATING: *** (medium — must know all 4)
SIMPLE ANSWER: {2, 3}  {1, 2, 3, 4}  {1}  {1, 4}
TRICK/PATTERN: & intersection, | union, - difference, ^ symmetric difference.
PITFALL: Confusing ^ (in exactly one) with & (in both). This was Cognizant Q2.
EXPLAIN: a-b is asymmetric: b-a would be {4}. a^b = (a|b) - (a&b).
"""


# ------------------------------------------------------------------------------
# Q12. (star 4) [Output Prediction] — bool is a subclass of int
# ------------------------------------------------------------------------------
print("Q12:", True + True + False, True * 3)           # 2 3
"""
RATING: **** (tricky — surprises most candidates)
SIMPLE ANSWER: 2  3
TRICK/PATTERN: In Python, True == 1 and False == 0 (bool subclasses int).
PITFALL: Thinking you can't do arithmetic on booleans. You can.
EXPLAIN: sum([True, False, True]) == 2 is a handy way to COUNT True values
         (e.g., count how many items pass a condition).
"""


# ------------------------------------------------------------------------------
# Q13. (star 5) [Output Prediction] — is vs == and integer caching
# ------------------------------------------------------------------------------
a = 100
b = 100
print("Q13:", a == b, a is b)                          # True True
"""
RATING: ***** (must-focus trap — is vs ==)
SIMPLE ANSWER: True True   (but the 'is' part is an implementation detail!)
TRICK/PATTERN: == compares VALUE; is compares IDENTITY (same object).
PITFALL: CPython caches small ints (-5..256), so 100 is 100 -> True.
         For 257: a=257; b=257; a is b may be FALSE. Never rely on 'is' for ints.
EXPLAIN: RULE: use == for values; reserve 'is' for None/True/False singletons.
"""


# ------------------------------------------------------------------------------
# Q14. (star 2) [Output Prediction / GenAI] — strip() whitespace
# ------------------------------------------------------------------------------
raw = "  GPT-4  "
clean = raw.strip()
print("Q14:", repr(clean), len(clean))                 # 'GPT-4' 5
"""
RATING: ** (easy, GenAI-relevant)
SIMPLE ANSWER: 'GPT-4'  and  5
TRICK/PATTERN: strip() removes leading/trailing whitespace (incl. \n, \t).
PITFALL: strip() does NOT remove internal spaces. "a b".strip() -> "a b".
EXPLAIN: Essential when parsing LLM text output / model names that arrive
         with stray spaces or trailing newlines. lstrip()/rstrip() for one side.
"""


# ------------------------------------------------------------------------------
# Q15. (star 3) [Output Prediction / GenAI] — f-string format spec
# ------------------------------------------------------------------------------
model = "claude"
temp = 0.5
print("Q15:", f"{model.upper()} @ {temp:.2f}")        # CLAUDE @ 0.50
"""
RATING: *** (medium — format specs)
SIMPLE ANSWER: CLAUDE @ 0.50
TRICK/PATTERN: f"{value:.2f}" formats a float to 2 decimals; methods work inside {}.
PITFALL: Forgetting :.2f keeps full float (0.5). The spec pads to 0.50.
EXPLAIN: Common in logging LLM calls: f"{model} cost ${cost:.4f}".
         Other specs: :, (thousands), :>10 (right-align), :.0% (percent).
"""


# ------------------------------------------------------------------------------
# Q16. (star 2) [Output Prediction] — range with step
# ------------------------------------------------------------------------------
print("Q16:", list(range(2, 10, 3)))                   # [2, 5, 8]
"""
RATING: ** (easy)
SIMPLE ANSWER: [2, 5, 8]
TRICK/PATTERN: range(start, stop, step); stop is EXCLUSIVE.
PITFALL: Expecting 11 to appear. 8+3=11 > 10, so it stops at 8.
EXPLAIN: range(10) -> 0..9. range(1,5) -> 1..4. Always stop-1 is the last possible.
"""


# ------------------------------------------------------------------------------
# Q17. (star 5) [Output Prediction] — for-else (no break)
# ------------------------------------------------------------------------------
for i in range(3):
    if i == 5:
        break
else:
    print("Q17:", "done")                              # done
"""
RATING: ***** (must-focus trap — for-else)
SIMPLE ANSWER: done
TRICK/PATTERN: loop's else runs ONLY IF the loop finished WITHOUT a break.
PITFALL: Thinking else runs when the loop body is skipped. It's tied to break.
EXPLAIN: i never hits 5, so no break -> else runs. If 'if i == 1: break',
         nothing prints. Used for search loops: 'else: not found'.
"""


# ------------------------------------------------------------------------------
# Q18. (star 2) [Output Prediction] — enumerate with start
# ------------------------------------------------------------------------------
for idx, ch in enumerate("AI", start=1):
    print("Q18:", idx, ch)                             # 1 A  /  2 I
"""
RATING: ** (easy)
SIMPLE ANSWER: 1 A   then   2 I
TRICK/PATTERN: enumerate(iterable, start=N) yields (index, item), counter from N.
PITFALL: Forgetting default start is 0, not 1.
EXPLAIN: Cleaner than manual counters. Common: for i, doc in enumerate(chunks):
"""


# ------------------------------------------------------------------------------
# Q19. (star 4) [Output Prediction] — return vs print (implicit None)
# ------------------------------------------------------------------------------
def greet(name):
    print("Q19: Hi", name)

result = greet("Tiru")
print("Q19:", result)                                  # Hi Tiru  /  None
"""
RATING: **** (tricky — top beginner confusion)
SIMPLE ANSWER: prints 'Hi Tiru', then prints None
TRICK/PATTERN: A function with no return statement returns None implicitly.
PITFALL: Confusing print (shows on screen) with return (gives back a value).
EXPLAIN: Matters in GenAI helpers: a function that 'displays' output but you
         expected a value back will silently give None downstream.
"""


# ------------------------------------------------------------------------------
# Q20. (star 3) [Output Prediction] — *args collects into a tuple
# ------------------------------------------------------------------------------
def total(*args):
    return sum(args)

print("Q20:", total(1, 2, 3, 4))                       # 10
"""
RATING: *** (medium)
SIMPLE ANSWER: 10
TRICK/PATTERN: *args packs all positional args into a TUPLE.
PITFALL: Thinking you must pass a list. You pass loose args; Python packs them.
EXPLAIN: total() -> sum(()) -> 0 (sum of empty is 0). Unpack with total(*[1,2,3]).
"""


# ------------------------------------------------------------------------------
# Q21. (star 3) [Output Prediction / GenAI] — **kwargs collects into a dict
# ------------------------------------------------------------------------------
def build_config(**kwargs):
    return list(kwargs.keys())

print("Q21:", build_config(model="gpt-4", temperature=0.7))   # ['model', 'temperature']
"""
RATING: *** (medium, GenAI-relevant)
SIMPLE ANSWER: ['model', 'temperature']
TRICK/PATTERN: **kwargs packs keyword args into a DICT.
PITFALL: Order — since Python 3.7 dicts keep insertion order, so keys come as passed.
EXPLAIN: Exact pattern to forward arbitrary params to an LLM client:
         client.chat(**kwargs). *args = tuple, **kwargs = dict.
"""


# ------------------------------------------------------------------------------
# Q22. (star 5) [Output Prediction] — reference aliasing (b = a)
# ------------------------------------------------------------------------------
a = [1, 2, 3]
b = a
b.append(4)
print("Q22:", a)                                       # [1, 2, 3, 4]
"""
RATING: ***** (must-focus trap — aliasing; root of Cognizant Q1)
SIMPLE ANSWER: [1, 2, 3, 4]
TRICK/PATTERN: b = a does NOT copy; both names point to the SAME list object.
PITFALL: Expecting a to stay [1,2,3]. Mutating via b is visible via a.
EXPLAIN: Independent copy: b = a.copy() / a[:] / list(a). For nested lists those
         are SHALLOW; use copy.deepcopy for full independence.
"""


# ------------------------------------------------------------------------------
# Q23. (star 3) [Output Prediction] — join and split
# ------------------------------------------------------------------------------
print("Q23a:", "-".join(["I", "love", "GenAI"]))       # I-love-GenAI
print("Q23b:", "a,b,c".split(","))                     # ['a', 'b', 'c']
"""
RATING: *** (medium, text-processing staple)
SIMPLE ANSWER: I-love-GenAI   and   ['a', 'b', 'c']
TRICK/PATTERN: 'sep'.join(list_of_strings); 'string'.split(sep) -> list.
PITFALL: join is called on the SEPARATOR, not the list: "-".join(words).
         join fails if list has non-strings: "-".join([1,2]) -> TypeError.
EXPLAIN: Bread-and-butter of token/text preprocessing. "".split() splits on
         any whitespace and drops empties; "a,,b".split(",") keeps empties.
"""


# ------------------------------------------------------------------------------
# Q24. (star 3) [Fix the Bug] — accumulation vs overwrite
# ------------------------------------------------------------------------------
def average(nums):
    total = 0
    for n in nums:
        total += n            # FIX: was 'total = n' (overwrites instead of sums)
    return total / len(nums)

print("Q24:", average([10, 20, 30]))                   # 20.0
"""
RATING: *** (medium — common logic bug)
SIMPLE ANSWER: 20.0  (bug version printed 10.0 — only the last value)
TRICK/PATTERN: Accumulators must use += , not = , inside the loop.
PITFALL: total = n overwrites each pass, leaving only the last element.
EXPLAIN: Pythonic: return sum(nums) / len(nums). Guard len==0 to avoid ZeroDivisionError.
"""


# ------------------------------------------------------------------------------
# Q25. (star 3) [Code Completion] — palindrome check in one line
# ------------------------------------------------------------------------------
def is_palindrome(s):
    return s == s[::-1]       # COMPLETED

print("Q25:", is_palindrome("level"), is_palindrome("genai"))   # True False
"""
RATING: *** (medium)
SIMPLE ANSWER: True  False
TRICK/PATTERN: Compare the string to its reverse s[::-1].
PITFALL: Case/space sensitivity: "Level" != "leveL". For robust check first do
         s = s.lower().replace(" ", "").
EXPLAIN: Same idea works for lists/numbers (convert to str). O(n) time, O(n) space.
"""

print()
print("=" * 70)
print("PART 1 COMPLETE (Q1-Q25).")
print("FOCUS (4-5 star): Q7 tuple-comma, Q8 'in'=keys, Q12 bool=int,")
print("Q13 is-vs-==, Q17 for-else, Q19 return-vs-print, Q22 aliasing.")
print("=" * 70)


# ##############################################################################
# PART 2 — INTERMEDIATE (Q26-Q65)
# ##############################################################################
# Topics: comprehensions, lambda/map/filter/reduce, iterators, generators,
#         decorators, closures, exceptions, file handling, modules, basic OOP.
# Mix: ~8 easy, ~17 medium, ~10 tricky, ~5 must-focus traps. Difficulty rises.
# ##############################################################################

print()
print("=" * 70)
print("PART 2 — INTERMEDIATE (Q26-Q65)")
print("=" * 70)


# ------------------------------------------------------------------------------
# Q26. (star 2) [Output Prediction] — basic list comprehension
# ------------------------------------------------------------------------------
print("Q26:", [x * x for x in range(5)])              # [0, 1, 4, 9, 16]
"""
RATING: ** (easy)
SIMPLE ANSWER: [0, 1, 4, 9, 16]
TRICK/PATTERN: [expr for item in iterable] builds a list in one line.
PITFALL: range(5) is 0..4, so squares are 0,1,4,9,16 (no 25).
EXPLAIN: Comprehensions are faster and cleaner than append loops.
"""


# ------------------------------------------------------------------------------
# Q27. (star 3) [Output Prediction] — comprehension with filter (if)
# ------------------------------------------------------------------------------
print("Q27:", [x for x in range(10) if x % 2 == 0])   # [0, 2, 4, 6, 8]
"""
RATING: *** (medium)
SIMPLE ANSWER: [0, 2, 4, 6, 8]
TRICK/PATTERN: 'if' AFTER the loop = FILTER (keep when true).
PITFALL: A filtering if goes at the END: [x for x in it if cond].
EXPLAIN: Different from a conditional EXPRESSION (see Q28) which goes BEFORE 'for'.
"""


# ------------------------------------------------------------------------------
# Q28. (star 4) [Output Prediction] — ternary inside comprehension
# ------------------------------------------------------------------------------
print("Q28:", [x if x % 2 == 0 else -x for x in range(5)])   # [0, -1, 2, -3, 4]
"""
RATING: **** (tricky — placement confusion)
SIMPLE ANSWER: [0, -1, 2, -3, 4]
TRICK/PATTERN: 'A if cond else B' is a value EXPRESSION; it goes BEFORE 'for'.
PITFALL: Mixing the two forms. Filtering 'if' goes AFTER the loop and has NO else.
EXPLAIN: Rule -> "if-else BEFORE for" (transform); "if ONLY AFTER for" (filter).
"""


# ------------------------------------------------------------------------------
# Q29. (star 3) [Output Prediction] — dict comprehension
# ------------------------------------------------------------------------------
print("Q29:", {x: x ** 2 for x in range(3)})          # {0: 0, 1: 1, 2: 4}
"""
RATING: *** (medium)
SIMPLE ANSWER: {0: 0, 1: 1, 2: 4}
TRICK/PATTERN: {key_expr: val_expr for item in iterable} builds a dict.
PITFALL: Duplicate keys overwrite: {k: v for k, v in [(1,'a'),(1,'b')]} -> {1:'b'}.
EXPLAIN: Common for GenAI: {doc.id: doc.embedding for doc in docs}.
"""


# ------------------------------------------------------------------------------
# Q30. (star 3) [Output Prediction] — set comprehension dedupes
# ------------------------------------------------------------------------------
print("Q30:", {x % 3 for x in range(10)})             # {0, 1, 2}
"""
RATING: *** (medium)
SIMPLE ANSWER: {0, 1, 2}
TRICK/PATTERN: {expr for ...} with braces (no key:val) = SET comprehension.
PITFALL: {} alone is an empty DICT, not a set. Use set() for empty set.
EXPLAIN: x%3 over 0..9 yields 0,1,2,0,1,2,... deduped to {0,1,2}.
"""


# ------------------------------------------------------------------------------
# Q31. (star 4) [Output Prediction] — nested comprehension / flatten
# ------------------------------------------------------------------------------
matrix = [[1, 2], [3, 4], [5, 6]]
print("Q31:", [x for row in matrix for x in row])     # [1, 2, 3, 4, 5, 6]
"""
RATING: **** (tricky — loop order)
SIMPLE ANSWER: [1, 2, 3, 4, 5, 6]
TRICK/PATTERN: Nested loops read LEFT-TO-RIGHT, same order as normal for loops.
PITFALL: Reversing the order. 'for row in matrix' is OUTER, 'for x in row' INNER.
EXPLAIN: Equivalent to: for row in matrix: for x in row: result.append(x).
"""


# ------------------------------------------------------------------------------
# Q32. (star 3) [Output Prediction] — map returns an iterator
# ------------------------------------------------------------------------------
result = map(lambda x: x * 2, [1, 2, 3])
print("Q32:", list(result), list(result))             # [2, 4, 6]  []

result = list(map(lambda x: x * 2, [1, 2, 3]))

print(result)                                        #  [2, 4, 6]
print(result)                                        # [2, 4, 6]
"""
RATING: *** (medium — iterator exhaustion)
SIMPLE ANSWER: [2, 4, 6]   then   []
map() returns an iterator. Iterators are consumed as they are traversed, so after the first list(result) exhausts the iterator, the second list(result) returns an empty list.
TRICK/PATTERN: map/filter return LAZY ITERATORS, consumed ONCE.
PITFALL: Second list(result) is empty — the iterator is already exhausted.
EXPLAIN: Wrap in list() once and reuse that list if you need it multiple times.
"""


# ------------------------------------------------------------------------------
# Q33. (star 3) [Output Prediction] — filter
# ------------------------------------------------------------------------------
print("Q33:", list(filter(lambda x: x > 2, [1, 2, 3, 4])))   # [3, 4]
"""
RATING: *** (medium)
SIMPLE ANSWER: [3, 4]
TRICK/PATTERN: filter(func, it) keeps items where func returns truthy.
PITFALL: filter(None, it) removes FALSY items (0, '', None, []).
EXPLAIN: Equivalent comprehension: [x for x in it if x > 2].
"""


# ------------------------------------------------------------------------------
# Q34. (star 4) [Output Prediction] — reduce
# ------------------------------------------------------------------------------
from functools import reduce
print("Q34:", reduce(lambda a, b: a + b, [1, 2, 3, 4], 100))   # 110
"""
RATING: **** (tricky — initializer)
SIMPLE ANSWER: 110
TRICK/PATTERN: reduce(func, iterable, initializer) folds left-to-right.
PITFALL: The 3rd arg (100) is the STARTING accumulator: 100+1+2+3+4 = 110.
EXPLAIN: Without initializer: reduce(add, [1,2,3,4]) = 10. reduce lives in functools.
"""


# ------------------------------------------------------------------------------
# Q35. (star 3) [Output Prediction] — sorted with key=lambda
# ------------------------------------------------------------------------------
data = [("a", 3), ("b", 1), ("c", 2)]
print("Q35:", sorted(data, key=lambda t: t[1]))       # [('b',1),('c',2),('a',3)]
"""
RATING: *** (medium — very common in screening)
SIMPLE ANSWER: [('b', 1), ('c', 2), ('a', 3)]
TRICK/PATTERN: key=lambda decides WHAT to sort by; here the 2nd tuple element.
PITFALL: key returns the sort VALUE, it does not transform the output items.
EXPLAIN: reverse=True for descending. key=lambda t: (-t[1], t[0]) = multi-key sort.
"""


# ------------------------------------------------------------------------------
# Q36. (star 5) [Output Prediction] — late-binding closure (Cognizant Q4!)
# ------------------------------------------------------------------------------
funcs = [lambda: i for i in range(3)]
print("Q36:", [f() for f in funcs])                   # [2, 2, 2]
"""
RATING: ***** (must-focus trap — the one that cost you marks)
SIMPLE ANSWER: [2, 2, 2]   (NOT [0, 1, 2])
TRICK/PATTERN: Closures capture the VARIABLE by reference (late binding).
PITFALL: All lambdas read i AFTER the loop ends, when i == 2.
FIX: [lambda i=i: i for i in range(3)] -> [0, 1, 2] (default arg binds value now).
EXPLAIN: Same trap with def in a loop, or callbacks. Bind with default arg or partial.
"""


# ------------------------------------------------------------------------------
# Q37. (star 4) [Output Prediction] — closure that DOES work (factory)
# ------------------------------------------------------------------------------
def multiplier(n):
    return lambda x: x * n

double = multiplier(2)
triple = multiplier(3)
print("Q37:", double(5), triple(5))                   # 10 15
"""
RATING: **** (tricky — contrast with Q36)
SIMPLE ANSWER: 10  15
TRICK/PATTERN: Here n is a PARAMETER, fixed per call, so each closure keeps its own n.
PITFALL: This works (unlike Q36) because n is bound by the function call, not a loop var.
EXPLAIN: This is the closure 'factory' pattern — partial application by hand.
"""


# ------------------------------------------------------------------------------
# Q38. (star 3) [Output Prediction] — generator with yield
# ------------------------------------------------------------------------------
def gen():
    yield 1
    yield 2
    yield 3

g = gen()
print("Q38:", next(g), next(g), list(g))              # 1 2 [3]


def gen1():
    yield 1
    yield 2
    yield 3

gn = gen1()

print(list(gn))                      # [1,2,3]
print(list(gn))                      # []
"""
RATING: *** (medium)
SIMPLE ANSWER: 1 2 [3]
TRICK/PATTERN: yield makes a generator; next() pulls one value, state is REMEMBERED.
PITFALL: After two next() calls, only 3 remains; list(g) drains the rest -> [3].
EXPLAIN: Generators are lazy and memory-efficient (one value at a time).
"""


# ------------------------------------------------------------------------------
# Q39. (star 4) [Output Prediction] — generator expression vs list
# ------------------------------------------------------------------------------
gen_exp = (x * x for x in range(3))
print("Q39:", type(gen_exp).__name__, sum(gen_exp))   # generator 5
"""
RATING: **** (tricky)
SIMPLE ANSWER: generator  5
TRICK/PATTERN: (expr for ...) with PARENS = generator (lazy); [..] = list (eager).
PITFALL: A genexp is consumed once. sum() drains it: 0+1+4 = 5.
EXPLAIN: Use genexps for large data: sum(x*x for x in huge) needs no list in memory.
"""


# ------------------------------------------------------------------------------
# Q40. (star 4) [Short Answer] — yield vs return
# ------------------------------------------------------------------------------
def count_up(n):
    i = 0
    while i < n:
        yield i
        i += 1

print("Q40:", list(count_up(4)))                      # [0, 1, 2, 3]
"""
RATING: **** (tricky concept)
SIMPLE ANSWER: [0, 1, 2, 3]
TRICK/PATTERN: yield PAUSES and resumes; return ENDS the function.
PITFALL: A function with any yield is a generator — calling it runs NO code until
         you iterate. 'return' inside a generator just stops iteration (StopIteration).
EXPLAIN: Great for streaming LLM tokens: yield each chunk as it arrives.
"""


# ------------------------------------------------------------------------------
# Q41. (star 3) [Output Prediction] — zip
# ------------------------------------------------------------------------------
print("Q41:", list(zip([1, 2, 3], ["a", "b"])))      # [(1,'a'), (2,'b')]
"""
RATING: *** (medium)
SIMPLE ANSWER: [(1, 'a'), (2, 'b')]
TRICK/PATTERN: zip pairs items; STOPS at the SHORTEST iterable.
PITFALL: The 3 has no partner, so it's dropped (no error). zip is also a one-shot iterator.
EXPLAIN: dict(zip(keys, values)) builds a dict. itertools.zip_longest fills gaps.
"""


# ------------------------------------------------------------------------------
# Q42. (star 4) [Output Prediction] — unzip with zip(*...)
# ------------------------------------------------------------------------------
pairs = [(1, "a"), (2, "b"), (3, "c")]
nums, letters = zip(*pairs)
print("Q42:", nums, letters)                          # (1, 2, 3) ('a', 'b', 'c')
"""
RATING: **** (tricky — the star unpacking trick)
SIMPLE ANSWER: (1, 2, 3)  ('a', 'b', 'c')
TRICK/PATTERN: zip(*pairs) transposes: unzips a list of tuples into columns.
PITFALL: Results are TUPLES, not lists. Wrap in list() if you need lists.
EXPLAIN: zip(*matrix) transposes rows<->columns. Very common interview trick.
"""


# ------------------------------------------------------------------------------
# Q43. (star 2) [Output Prediction] — enumerate to build dict
# ------------------------------------------------------------------------------
print("Q43:", {ch: i for i, ch in enumerate("abc")})  # {'a':0,'b':1,'c':2}
"""
RATING: ** (easy)
SIMPLE ANSWER: {'a': 0, 'b': 1, 'c': 2}
TRICK/PATTERN: enumerate + dict comprehension = char/item -> index map.
PITFALL: Order of (i, ch) in enumerate is (index, item) — don't swap.
EXPLAIN: Common for building vocab/token-id maps in NLP.
"""


# ------------------------------------------------------------------------------
# Q44. (star 3) [Output Prediction] — try/except/else/finally order
# ------------------------------------------------------------------------------
def divide(a, b):
    try:
        r = a / b
    except ZeroDivisionError:
        return "error"
    else:
        return r
    finally:
        print("Q44: finally always runs")

print("Q44:", divide(10, 2))                          # finally...  then 5.0
"""
RATING: *** (medium)
SIMPLE ANSWER: prints 'finally always runs', then returns 5.0
TRICK/PATTERN: else runs if NO exception; finally runs ALWAYS (even on return).
PITFALL: finally executes even when try/except returns — it runs BEFORE the return completes.
EXPLAIN: Order: try -> (except OR else) -> finally. Use finally for cleanup (close files).
"""


# ------------------------------------------------------------------------------
# Q45. (star 4) [Output Prediction] — exception type matching order
# ------------------------------------------------------------------------------
try:
    raise ValueError("bad")
except Exception:
    print("Q45: caught as Exception")
except ValueError:
    print("Q45: caught as ValueError")
"""
RATING: **** (tricky — except ordering)
SIMPLE ANSWER: caught as Exception
TRICK/PATTERN: except clauses are checked TOP-DOWN; first match wins.
PITFALL: Broad 'except Exception' BEFORE 'except ValueError' shadows the specific one.
EXPLAIN: Always order from MOST specific to MOST general. ValueError is a subclass of Exception.
"""


# ------------------------------------------------------------------------------
# Q46. (star 4) [Output Prediction] — exception in for-else with break
# ------------------------------------------------------------------------------
def find_first_even(nums):
    for n in nums:
        if n % 2 == 0:
            return n
    return None

print("Q46:", find_first_even([1, 3, 5, 8, 9]))       # 8
"""
RATING: **** (tricky tracing)
SIMPLE ANSWER: 8
TRICK/PATTERN: return inside a loop EXITS the function immediately on first match.
PITFALL: Thinking it keeps scanning. Once 8 is found and returned, loop stops.
EXPLAIN: next((n for n in nums if n%2==0), None) is the one-line equivalent.
"""


# ------------------------------------------------------------------------------
# Q47. (star 3) [Output Prediction] — custom exception
# ------------------------------------------------------------------------------
class ConfigError(Exception):
    pass

try:
    raise ConfigError("missing API key")
except ConfigError as e:
    print("Q47:", str(e))                             # missing API key
"""
RATING: *** (medium, production-relevant)
SIMPLE ANSWER: missing API key
TRICK/PATTERN: Subclass Exception to make domain-specific errors.
PITFALL: str(e) gives the MESSAGE; repr(e) gives ConfigError('missing API key').
EXPLAIN: Common in GenAI: raise a clear ConfigError when an env var / key is absent.
"""


# ------------------------------------------------------------------------------
# Q48. (star 4) [Output Prediction] — raise from / chaining (concept)
# ------------------------------------------------------------------------------
def parse_int(s):
    try:
        return int(s)
    except ValueError:
        return -1

print("Q48:", parse_int("42"), parse_int("abc"))      # 42 -1
"""
RATING: **** (tricky — defensive parsing)
SIMPLE ANSWER: 42  -1
TRICK/PATTERN: int('abc') raises ValueError; catch and return a sentinel.
PITFALL: int(' 42 ') works (strips spaces) but int('4.2') raises ValueError.
         int('0x10', 16) -> 16 (base arg). float('abc') also raises ValueError.
EXPLAIN: Robust parsing of LLM/API string outputs needs this try/except guard.
"""


# ------------------------------------------------------------------------------
# Q49. (star 3) [Output Prediction] — basic decorator
# ------------------------------------------------------------------------------

# A decorator is a function that takes another function as input, adds some behavior before or after its execution, and returns a new function without modifying the original function's code.
def shout(func):
    def wrapper():
        return func().upper()
    return wrapper

@shout
def greet():
    return "hello"

print("Q49:", greet())                                # HELLO

def my_decorator(func):
    def wrapper(*args, **kwargs):
        print("Before function call")
        result = func(*args, **kwargs)
        print("After function call")
        return result
    return wrapper

@my_decorator
def greet(name):
    print(f"Hello {name}")

greet("Tirupathi")  #Before function call, Hello Tirupathi,After function call

"""
RATING: *** (medium)
SIMPLE ANSWER: HELLO
TRICK/PATTERN: @shout means greet = shout(greet); wrapper runs instead.
PITFALL: Without @functools.wraps, greet.__name__ becomes 'wrapper'.
EXPLAIN: Decorators wrap behavior (logging, timing, auth, retries) around a function.
"""


# ------------------------------------------------------------------------------
# Q50. (star 5) [Output Prediction] — decorator with args (*args, **kwargs)
# ------------------------------------------------------------------------------
def logged(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return f"{func.__name__} -> {result}"
    return wrapper

@logged
def add(a, b):
    return a + b

print("Q50:", add(3, 4))                              # add -> 7
"""
RATING: ***** (must-focus — the real-world decorator pattern)
SIMPLE ANSWER: add -> 7
TRICK/PATTERN: wrapper(*args, **kwargs) lets a decorator wrap ANY function signature.
PITFALL: Forgetting *args/**kwargs makes the decorator break functions that take args.
EXPLAIN: This is THE template for retry/timing/logging decorators in production code.
         Use @functools.wraps(func) to preserve name/docstring.
"""


# ------------------------------------------------------------------------------
# Q51. (star 3) [Output Prediction] — basic class & __init__
# ------------------------------------------------------------------------------
class Dog:
    def __init__(self, name):
        self.name = name
    def speak(self):
        return f"{self.name} says woof"

print("Q51:", Dog("Rex").speak())                     # Rex says woof
"""
RATING: *** (medium)
SIMPLE ANSWER: Rex says woof
TRICK/PATTERN: __init__ is the constructor; self is the instance, passed automatically.
PITFALL: Forgetting self in method signature, or forgetting self.name = name.
EXPLAIN: Dog("Rex") calls __init__(self, "Rex"); you don't pass self explicitly.
"""


# ------------------------------------------------------------------------------
# Q52. (star 5) [Output Prediction] — class vs instance attribute (shared mutable)
# ------------------------------------------------------------------------------
class Counter:
    items = []                  # CLASS attribute (shared by ALL instances!)
    def add(self, x):
        self.items.append(x)

a = Counter()
b = Counter()
a.add(1)
b.add(2)
print("Q52:", a.items, b.items)                       # [1, 2] [1, 2]
"""
RATING: ***** (must-focus trap — shared class attribute)
SIMPLE ANSWER: [1, 2]  [1, 2]   (both share the SAME list!)
TRICK/PATTERN: A mutable class attribute is SHARED across all instances.
PITFALL: Expecting separate lists. items lives on the class, not the instance.
FIX: initialize per-instance in __init__: self.items = [].
EXPLAIN: Same family as the mutable-default-arg trap. Per-instance state -> __init__.
"""


# ------------------------------------------------------------------------------
# Q53. (star 4) [Output Prediction] — inheritance & method override
# ------------------------------------------------------------------------------
class Animal:
    def sound(self):
        return "..."
class Cat(Animal):
    def sound(self):
        return "meow"

print("Q53:", Cat().sound(), Animal().sound())        # meow ...
"""
RATING: **** (tricky)
SIMPLE ANSWER: meow  ...
TRICK/PATTERN: Subclass method OVERRIDES parent's; Python picks the most-derived.
PITFALL: Cat() uses Cat.sound; Animal() uses Animal.sound. No mixing.
EXPLAIN: Use super().sound() inside Cat to also call the parent's version.
"""


# ------------------------------------------------------------------------------
# Q54. (star 4) [Output Prediction] — super() in __init__
# ------------------------------------------------------------------------------
class Base:
    def __init__(self):
        self.x = 1
class Child(Base):
    def __init__(self):
        super().__init__()
        self.y = 2

c = Child()
print("Q54:", c.x, c.y)                               # 1 2
"""
RATING: **** (tricky)
SIMPLE ANSWER: 1 2
TRICK/PATTERN: super().__init__() runs the parent constructor (sets self.x).
PITFALL: Forgetting super().__init__() -> self.x never set -> AttributeError.
EXPLAIN: Always call super().__init__() when the parent sets up needed state.
"""


# ------------------------------------------------------------------------------
# Q55. (star 3) [Output Prediction] — __str__ vs __repr__
# ------------------------------------------------------------------------------
class Point:
    def __init__(self, x):
        self.x = x
    def __repr__(self):
        return f"Point({self.x})"

p = Point(5)
print("Q55:", str(p), [p])                            # Point(5) [Point(5)]
"""
RATING: *** (medium)
SIMPLE ANSWER: Point(5)   [Point(5)]
TRICK/PATTERN: If __str__ is absent, str() falls back to __repr__.
         Containers (lists) ALWAYS use __repr__ for their elements.
PITFALL: Defining only __str__ -> printing a list of objects shows <object at 0x...>.
EXPLAIN: Define __repr__ for debugging; __str__ for user-facing text.
"""


# ------------------------------------------------------------------------------
# Q56. (star 4) [Output Prediction] — @staticmethod vs @classmethod
# ------------------------------------------------------------------------------
class Math:
    factor = 10
    @staticmethod
    def add(a, b):
        return a + b
    @classmethod
    def scaled(cls, x):
        return x * cls.factor

print("Q56:", Math.add(2, 3), Math.scaled(5))         # 5 50
"""
RATING: **** (tricky)
SIMPLE ANSWER: 5  50
TRICK/PATTERN: staticmethod = no self/cls; classmethod gets cls (the class).
PITFALL: staticmethod can't access class state; classmethod can (cls.factor).
EXPLAIN: classmethod is common for alternative constructors: cls(...) factory methods.
"""


# ------------------------------------------------------------------------------
# Q57. (star 3) [Output Prediction] — property
# ------------------------------------------------------------------------------
class Circle:
    def __init__(self, r):
        self.r = r
    @property
    def area(self):
        return round(3.14159 * self.r ** 2, 2)

print("Q57:", Circle(2).area)                         # 12.57
"""
RATING: *** (medium)
SIMPLE ANSWER: 12.57
TRICK/PATTERN: @property makes a method accessible like an ATTRIBUTE (no parens).
PITFALL: Calling Circle(2).area() -> TypeError (it's not callable, it's a value).
EXPLAIN: Properties give computed attributes with optional setters/validation.
"""


# ------------------------------------------------------------------------------
# Q58. (star 3) [Output Prediction] — list sort vs sorted
# ------------------------------------------------------------------------------
nums = [3, 1, 2]
s = sorted(nums)
nums.sort()
print("Q58:", s, nums, nums.sort())                   # [1,2,3] [1,2,3] None
"""
RATING: *** (medium — returns None trap)
SIMPLE ANSWER: [1, 2, 3]  [1, 2, 3]  None
TRICK/PATTERN: sorted() returns a NEW list; list.sort() sorts IN PLACE, returns None.
PITFALL: x = nums.sort() sets x to None (a classic bug). Same for .append, .reverse.
EXPLAIN: Methods that mutate in place return None. Use sorted() if you need a value.
"""


# ------------------------------------------------------------------------------
# Q59. (star 4) [Output Prediction] — slicing makes a shallow copy
# ------------------------------------------------------------------------------
a = [1, 2, 3]
b = a[:]
b.append(4)
print("Q59:", a, b, a is b)                           # [1,2,3] [1,2,3,4] False
"""
RATING: **** (tricky — contrast with Q22 aliasing)
SIMPLE ANSWER: [1, 2, 3]  [1, 2, 3, 4]  False
TRICK/PATTERN: a[:] creates a NEW (shallow) copy; b is independent.
PITFALL: For NESTED lists, a[:] copies only the outer list; inner lists are shared.
EXPLAIN: copy.deepcopy(a) for full independence. Compare to Q22 where b=a aliased.
"""


# ------------------------------------------------------------------------------
# Q60. (star 5) [Output Prediction] — mutable default argument (Cognizant Q6!)
# ------------------------------------------------------------------------------
def append_item(x, lst=[]):
    lst.append(x)
    return lst

print("Q60:", append_item(1), append_item(2))         # [1] [1, 2]
"""
RATING: ***** (must-focus trap — the famous one)
SIMPLE ANSWER: [1]   then   [1, 2]
TRICK/PATTERN: Default value is created ONCE at definition; reused every call.
PITFALL: Expecting a fresh [] each call. State accumulates across calls.
FIX: def append_item(x, lst=None): if lst is None: lst = [].
EXPLAIN: NEVER use mutable defaults ([], {}, set()). Use None sentinel.
"""


# ------------------------------------------------------------------------------
# Q61. (star 3) [Output Prediction] — string methods chain (GenAI cleanup)
# ------------------------------------------------------------------------------
raw = "  Hello, World!  "
print("Q61:", raw.strip().lower().replace(",", ""))   # hello world!
"""
RATING: *** (medium, text-processing)
SIMPLE ANSWER: hello world!
TRICK/PATTERN: String methods chain left-to-right, each returns a NEW string.
PITFALL: strip() only trims ENDS; the comma is removed by replace, not strip.
EXPLAIN: Order: trim -> lowercase -> remove comma. Classic LLM-output normalization.
"""


# ------------------------------------------------------------------------------
# Q62. (star 4) [Output Prediction / GenAI] — JSON parse from a string
# ------------------------------------------------------------------------------
import json
raw = '{"model": "gpt-4", "temp": 0.7, "tools": ["search"]}'
data = json.loads(raw)
print("Q62:", data["temp"], type(data["tools"]).__name__)   # 0.7 list
"""
RATING: **** (tricky, GenAI-relevant)
SIMPLE ANSWER: 0.7  list
TRICK/PATTERN: json.loads(str) -> Python dict; JSON arrays become lists.
PITFALL: json.loads parses a STRING; json.load reads a FILE object. Don't mix.
         JSON true/false/null become Python True/False/None.
EXPLAIN: Core skill for parsing LLM/API responses. Wrap in try/except
         json.JSONDecodeError because LLMs sometimes emit malformed JSON.
"""


# ------------------------------------------------------------------------------
# Q63. (star 4) [Output Prediction / GenAI] — json.dumps formatting
# ------------------------------------------------------------------------------
import json
obj = {"b": 2, "a": 1}
print("Q63:", json.dumps(obj, sort_keys=True))        # {"a": 1, "b": 2}
"""
RATING: **** (tricky)
SIMPLE ANSWER: {"a": 1, "b": 2}
TRICK/PATTERN: json.dumps(obj) -> JSON STRING; sort_keys orders keys alphabetically.
PITFALL: dumps gives a string with DOUBLE quotes (valid JSON), not Python single quotes.
         Non-serializable types (datetime, set) raise TypeError — need a custom encoder.
EXPLAIN: indent=2 for pretty printing. ensure_ascii=False to keep unicode readable.
"""


# ------------------------------------------------------------------------------
# Q64. (star 3) [Fix the Bug] — file handling with context manager
# ------------------------------------------------------------------------------
# BUG: file never closed; data may not flush. FIX: use 'with'.
import io
def write_then_read():
    buf = io.StringIO()
    with buf:                       # context manager auto-closes
        buf.write("line1\n")
        return buf.getvalue()

print("Q64:", repr(write_then_read()))                # 'line1\n'
"""
RATING: *** (medium, best practice)
SIMPLE ANSWER: 'line1\n'
TRICK/PATTERN: 'with open(...) as f:' guarantees the file closes even on error.
PITFALL: f = open(...) without close() leaks handles / may not flush writes.
EXPLAIN: Real file pattern: with open('f.txt','w') as f: f.write(...). The with
         block calls __enter__/__exit__ — same idea powers DB sessions, locks.
"""


# ------------------------------------------------------------------------------
# Q65. (star 5) [Output Prediction] — default arg evaluated once (time-like trap)
# ------------------------------------------------------------------------------
calls = []
def register(name, log=calls):     # default bound to the SAME list object
    log.append(name)
    return len(log)

print("Q65:", register("a"), register("b"))           # 1 2
"""
RATING: ***** (must-focus trap — default binds to outer mutable)
SIMPLE ANSWER: 1  2
TRICK/PATTERN: Default arg is evaluated ONCE and bound to the 'calls' object.
PITFALL: Both calls share 'calls'; it grows -> 1 then 2. Same root cause as Q60.
EXPLAIN: Defaults capture the OBJECT at def-time, not a fresh value per call.
         For per-call fresh state, use None + create inside the function.
"""

print()
print("=" * 70)
print("PART 2 COMPLETE (Q26-Q65).")
print("FOCUS (5 star): Q36 late-binding closure, Q50 decorator *args,")
print("Q52 shared class attr, Q60 mutable default, Q65 default binds object.")
print("FOCUS (4 star): Q28 ternary-in-comp, Q31 nested comp, Q34 reduce,")
print("Q42 zip(*), Q45 except order, Q56 static/classmethod, Q59 slice copy, Q62/63 JSON.")
print("=" * 70)


# ##############################################################################
# PART 3 — MEDIUM-ADVANCED (Q66-Q115)
# ##############################################################################
# Topics: memory model (copy/mutability/id), advanced OOP (MRO, metaclass,
#         descriptors, __slots__, dunders), context managers, concurrency
#         (GIL, threading, multiprocessing, asyncio), algorithm tracing
#         (sliding window, two-pointer, stack, binary search).
# Mix: ~5 medium, ~28 hard, ~17 very tricky. Maps to Cognizant Q5/Q7/Q8/Q9/Q10.
# ##############################################################################

print()
print("=" * 70)
print("PART 3 — MEDIUM-ADVANCED (Q66-Q115)")
print("=" * 70)


# ============================ MEMORY MODEL & COPY =============================

# ------------------------------------------------------------------------------
# Q66. (star 5) [Output Prediction] — shallow copy shares NESTED objects
# ------------------------------------------------------------------------------
import copy
a = [[1, 2], [3, 4]]
b = a.copy()                 # shallow: new outer list, SAME inner lists
b[0].append(99)
print("Q66:", a, b)          # [[1,2,99],[3,4]] [[1,2,99],[3,4]]
"""
RATING: ***** (must-focus trap — shallow copy)
SIMPLE ANSWER: [[1, 2, 99], [3, 4]]  [[1, 2, 99], [3, 4]]
TRICK/PATTERN: .copy()/[:]/list() copy ONLY the outer level; inner objects are shared.
PITFALL: Mutating b[0] also changes a[0] — they point to the same inner list.
FIX: copy.deepcopy(a) for fully independent nested structures.
EXPLAIN: Rebinding b[0]=[...] would NOT affect a; mutating b[0] in place does.
"""


# ------------------------------------------------------------------------------
# Q67. (star 4) [Output Prediction] — deepcopy fully independent
# ------------------------------------------------------------------------------
a = [[1, 2], [3, 4]]
b = copy.deepcopy(a)
b[0].append(99)
print("Q67:", a, b)          # [[1,2],[3,4]] [[1,2,99],[3,4]]
"""
RATING: **** (tricky — contrast with Q66)
SIMPLE ANSWER: [[1, 2], [3, 4]]  [[1, 2, 99], [3, 4]]
TRICK/PATTERN: deepcopy recursively clones every nested object.
PITFALL: deepcopy is slower and can break on un-copyable objects (sockets, locks).
EXPLAIN: Use deepcopy only when you truly need nested independence.
"""


# ------------------------------------------------------------------------------
# Q68. (star 4) [Output Prediction] — list multiplication aliases rows
# ------------------------------------------------------------------------------
grid = [[0] * 3] * 2         # outer multiply REPEATS THE SAME inner list ref
grid[0][0] = 1
print("Q68:", grid)          # [[1, 0, 0], [1, 0, 0]]
"""
RATING: ***** (must-focus trap — [[]]*n)
SIMPLE ANSWER: [[1, 0, 0], [1, 0, 0]]   (both rows changed!)
TRICK/PATTERN: [x]*n repeats the SAME reference n times for mutable x.
PITFALL: Expecting independent rows. Both rows are the SAME list object.
FIX: [[0]*3 for _ in range(2)] -> independent rows.
EXPLAIN: [0]*3 is fine (ints immutable); the OUTER multiply of a list is the trap.
"""


# ------------------------------------------------------------------------------
# Q69. (star 4) [Output Prediction] — tuple is immutable but holds mutables
# ------------------------------------------------------------------------------
t = (1, [2, 3])
t[1].append(4)
print("Q69:", t)            # (1, [2, 3, 4])
"""
RATING: **** (tricky)
SIMPLE ANSWER: (1, [2, 3, 4])
TRICK/PATTERN: Tuple immutability means you can't REBIND slots, not that contents freeze.
PITFALL: t[1] = [...] raises TypeError, but t[1].append(4) works (mutating the list).
EXPLAIN: A tuple with a mutable element is unhashable: hash(t) -> TypeError.
"""


# ------------------------------------------------------------------------------
# Q70. (star 3) [Output Prediction] — id() identity after rebinding
# ------------------------------------------------------------------------------
x = [1, 2]
before = id(x)
x = x + [3]                 # rebinds x to a NEW list
print("Q70:", id(x) == before)   # False
"""
RATING: *** (medium)
SIMPLE ANSWER: False
TRICK/PATTERN: x = x + [3] creates a NEW object (new id); x += [3] mutates in place (same id).
PITFALL: Confusing + (new object) with += (in-place for lists).
EXPLAIN: This is the core of the Cognizant Q1 aliasing behavior, viewed via id().
"""


# ------------------------------------------------------------------------------
# Q71. (star 4) [Output Prediction] — augmented assign on tuple member
# ------------------------------------------------------------------------------
t = (1, [10])
try:
    t[1] += [20]            # mutates the list AND tries to rebind -> partial!
except TypeError:
    pass
print("Q71:", t)           # (1, [10, 20])
"""
RATING: **** (tricky — famous gotcha)
SIMPLE ANSWER: (1, [10, 20])  (the list IS extended, even though a TypeError is raised!)
TRICK/PATTERN: += on a list does extend (in place) THEN attempts to store back into
               the tuple slot -> the store fails (TypeError) but the extend already happened.
PITFALL: Believing nothing changed because of the error. The mutation persists.
EXPLAIN: This is the classic 'why does it both fail AND succeed' interview puzzle.
"""


# ------------------------------------------------------------------------------
# Q72. (star 3) [Output Prediction] — is None vs == None
# ------------------------------------------------------------------------------
x = None
print("Q72:", x is None, x == None)   # True True
"""
RATING: *** (medium, best practice)
SIMPLE ANSWER: True True
TRICK/PATTERN: Always test 'x is None' (identity); None is a singleton.
PITFALL: == can be overridden by a class's __eq__ and give surprising results;
         'is' cannot be faked. Linters flag '== None'.
EXPLAIN: Same rule from Q13: 'is' for None/True/False, '==' for value comparison.
"""


# ============================ ADVANCED OOP ===================================

# ------------------------------------------------------------------------------
# Q73. (star 5) [Output Prediction] — MRO / super() diamond (Cognizant Q5!)
# ------------------------------------------------------------------------------
class A:
    def who(self): return "A"
class B(A):
    def who(self): return "B" + super().who()
class C(A):
    def who(self): return "C" + super().who()
class D(B, C):
    pass
print("Q73:", D().who())              # BCA
"""
RATING: ***** (must-focus — exact Cognizant Q5)
SIMPLE ANSWER: BCA
TRICK/PATTERN: super() follows the MRO (D->B->C->A), NOT the literal parent.
PITFALL: Thinking B's super() goes to A. In the diamond it goes to C.
EXPLAIN: D.__mro__ = [D, B, C, A, object] via C3 linearization. Check with D.mro().
"""


# ------------------------------------------------------------------------------
# Q74. (star 4) [Output Prediction] — read the MRO directly
# ------------------------------------------------------------------------------
print("Q74:", [c.__name__ for c in D.__mro__])    # ['D','B','C','A','object']
"""
RATING: **** (tricky)
SIMPLE ANSWER: ['D', 'B', 'C', 'A', 'object']
TRICK/PATTERN: __mro__ lists the lookup order; every class ends at object.
PITFALL: Forgetting 'object' is always last.
EXPLAIN: C3 keeps subclasses before parents and preserves left-to-right base order.
"""


# ------------------------------------------------------------------------------
# Q75. (star 4) [Output Prediction] — __slots__ blocks new attributes
# ------------------------------------------------------------------------------
class P:
    __slots__ = ("x", "y")
    def __init__(self, x, y):
        self.x, self.y = x, y

p = P(1, 2)
try:
    p.z = 3
    print("Q75:", "no error")
except AttributeError:
    print("Q75:", "AttributeError")    # AttributeError
"""
RATING: **** (tricky)
SIMPLE ANSWER: AttributeError
TRICK/PATTERN: __slots__ fixes allowed attributes, removes per-instance __dict__.
PITFALL: Assigning an attribute not in __slots__ raises AttributeError.
EXPLAIN: __slots__ saves memory for millions of small objects; trade-off is no
         dynamic attributes and trickier multiple inheritance.
"""


# ------------------------------------------------------------------------------
# Q76. (star 5) [Output Prediction] — descriptor (__get__/__set__)
# ------------------------------------------------------------------------------
class Positive:
    def __set_name__(self, owner, name):
        self.name = "_" + name
    def __get__(self, obj, owner):
        return getattr(obj, self.name, 0)
    def __set__(self, obj, value):
        if value < 0:
            raise ValueError("must be >= 0")
        setattr(obj, self.name, value)

class Account:
    balance = Positive()

acc = Account()
acc.balance = 50
print("Q76:", acc.balance)             # 50
"""
RATING: ***** (must-focus — descriptors)
SIMPLE ANSWER: 50
TRICK/PATTERN: A descriptor defines __get__/__set__; it manages attribute access on
               the OWNER class. acc.balance routes through Positive.__get__/__set__.
PITFALL: Descriptor must live on the CLASS, not the instance, to fire.
EXPLAIN: This is the machinery behind @property, ORM fields, and validators.
         acc.balance = -1 would raise ValueError.
"""


# ------------------------------------------------------------------------------
# Q77. (star 5) [Output Prediction] — metaclass intercepts class creation
# ------------------------------------------------------------------------------
class Meta(type):
    def __new__(mcs, name, bases, ns):
        ns["created_by"] = "Meta"
        return super().__new__(mcs, name, bases, ns)

class Widget(metaclass=Meta):
    pass

print("Q77:", Widget.created_by)       # Meta
"""
RATING: ***** (must-focus — metaclasses)
SIMPLE ANSWER: Meta
TRICK/PATTERN: A metaclass is the 'class of a class'; __new__ runs at class CREATION.
PITFALL: Confusing metaclass __new__ (builds the class) with instance __new__ (builds objects).
EXPLAIN: type is the default metaclass. Frameworks (Django models, ABCs, Pydantic)
         use metaclasses to inject behavior at class definition time.
"""


# ------------------------------------------------------------------------------
# Q78. (star 4) [Output Prediction] — __call__ makes instances callable
# ------------------------------------------------------------------------------
class Adder:
    def __init__(self, n): self.n = n
    def __call__(self, x): return x + self.n

add5 = Adder(5)
print("Q78:", add5(10), callable(add5))   # 15 True
"""
RATING: **** (tricky)
SIMPLE ANSWER: 15  True
TRICK/PATTERN: Defining __call__ lets you call an instance like a function.
PITFALL: add5(10) invokes __call__, not __init__ (that ran at Adder(5)).
EXPLAIN: Used for stateful callables, function objects, LangChain Runnables.
"""


# ------------------------------------------------------------------------------
# Q79. (star 4) [Output Prediction] — __eq__ without __hash__ -> unhashable
# ------------------------------------------------------------------------------
class Pt:
    def __init__(self, x): self.x = x
    def __eq__(self, other): return self.x == other.x

try:
    s = {Pt(1)}
    print("Q79:", "hashable")
except TypeError:
    print("Q79:", "unhashable")        # unhashable
"""
RATING: **** (tricky)
SIMPLE ANSWER: unhashable
TRICK/PATTERN: Defining __eq__ sets __hash__ to None -> object becomes unhashable.
PITFALL: You can no longer put it in a set or use it as a dict key.
EXPLAIN: Add __hash__ = lambda self: hash(self.x), or use @dataclass(frozen=True).
"""


# ------------------------------------------------------------------------------
# Q80. (star 3) [Output Prediction] — isinstance with inheritance
# ------------------------------------------------------------------------------
class Base: pass
class Sub(Base): pass
s = Sub()
print("Q80:", isinstance(s, Base), type(s) is Base)   # True False
"""
RATING: *** (medium)
SIMPLE ANSWER: True  False
TRICK/PATTERN: isinstance respects inheritance; 'type(x) is C' checks the EXACT class.
PITFALL: Using 'type(x) == Base' for a subclass instance returns False.
EXPLAIN: Prefer isinstance for polymorphism; reserve exact type checks for special cases.
"""


# ------------------------------------------------------------------------------
# Q81. (star 4) [Output Prediction] — class attribute shadowed by instance
# ------------------------------------------------------------------------------
class Cfg:
    debug = False

c = Cfg()
c.debug = True
print("Q81:", c.debug, Cfg.debug)      # True False
"""
RATING: **** (tricky)
SIMPLE ANSWER: True  False
TRICK/PATTERN: Assigning c.debug creates an INSTANCE attr that shadows the class attr.
PITFALL: The class attribute is unchanged; only this instance sees True.
EXPLAIN: Reads fall back to the class; writes create an instance-level attribute.
         (Contrast Q52 where a MUTATED shared list affected all instances.)
"""


# ------------------------------------------------------------------------------
# Q82. (star 4) [Output Prediction] — dataclass auto __init__/__eq__
# ------------------------------------------------------------------------------
from dataclasses import dataclass
@dataclass
class Point:
    x: int
    y: int

print("Q82:", Point(1, 2) == Point(1, 2))   # True
"""
RATING: **** (tricky, modern Python)
SIMPLE ANSWER: True
TRICK/PATTERN: @dataclass auto-generates __init__, __repr__, __eq__ (value equality).
PITFALL: Without @dataclass, two objects compare by identity -> would be False.
         NEVER use a mutable default in a dataclass field; use field(default_factory=list).
EXPLAIN: Great for config/DTO objects; frozen=True makes them immutable + hashable.
"""


# ============================ CONTEXT MANAGERS ===============================

# ------------------------------------------------------------------------------
# Q83. (star 4) [Output Prediction] — context manager __enter__/__exit__ order
# ------------------------------------------------------------------------------
class CM:
    def __enter__(self):
        print("Q83: enter")
        return self
    def __exit__(self, *a):
        print("Q83: exit")
        return False

with CM():
    print("Q83: body")
# enter / body / exit
"""
RATING: **** (tricky)
SIMPLE ANSWER: enter, then body, then exit (in that order)
TRICK/PATTERN: with calls __enter__ (its return goes to 'as'), runs body, then __exit__.
PITFALL: __exit__ runs even if the body raises. Returning True SUPPRESSES the exception.
EXPLAIN: Powers 'with open(...)', DB sessions, locks, timers. 'as x' = __enter__'s return.
"""


# ------------------------------------------------------------------------------
# Q84. (star 4) [Output Prediction] — __exit__ returning True swallows error
# ------------------------------------------------------------------------------
class Suppress:
    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb):
        return exc_type is ValueError   # swallow only ValueError

with Suppress():
    raise ValueError("boom")
print("Q84:", "survived")              # survived
"""
RATING: **** (tricky)
SIMPLE ANSWER: survived
TRICK/PATTERN: __exit__ returning truthy SUPPRESSES the exception.
PITFALL: Returning True for ALL exceptions hides real bugs. Be specific.
EXPLAIN: This is exactly how contextlib.suppress(ValueError) works internally.
"""


# ------------------------------------------------------------------------------
# Q85. (star 4) [Output Prediction] — contextlib.contextmanager
# ------------------------------------------------------------------------------
from contextlib import contextmanager
@contextmanager
def tag(name):
    print(f"Q85: <{name}>")
    yield
    print(f"Q85: </{name}>")

with tag("b"):
    print("Q85: hi")
# <b> / hi / </b>
"""
RATING: **** (tricky)
SIMPLE ANSWER: <b>, then hi, then </b>
TRICK/PATTERN: Code BEFORE yield = __enter__; AFTER yield = __exit__. yield's value -> 'as'.
PITFALL: If the body raises, the code after yield is skipped unless you wrap in try/finally.
EXPLAIN: Cleaner than a class for simple setup/teardown (timers, temp dirs, DB tx).
"""


# ============================ GIL & CONCURRENCY ==============================

# ------------------------------------------------------------------------------
# Q86. (star 5) [Short Answer] — what the GIL actually does
# ------------------------------------------------------------------------------
print("Q86:", "GIL = one thread runs Python bytecode at a time")
"""
RATING: ***** (must-focus — the GIL question)
SIMPLE ANSWER: The Global Interpreter Lock lets only ONE thread execute Python
               bytecode at a time in CPython.
TRICK/PATTERN: Threads help I/O-bound work (the GIL releases during I/O waits);
               they do NOT speed up CPU-bound work.
PITFALL: Believing threading gives parallel CPU speedup in CPython. It doesn't.
EXPLAIN: CPU-bound -> multiprocessing (separate processes, separate GILs) or
         native extensions. I/O-bound -> threading or asyncio. (Py3.13 has an
         experimental no-GIL build, but assume GIL for interviews.)
"""


# ------------------------------------------------------------------------------
# Q87. (star 4) [Short Answer] — threading vs multiprocessing vs asyncio
# ------------------------------------------------------------------------------
print("Q87:", "IO-bound: threads/asyncio | CPU-bound: multiprocessing")
"""
RATING: **** (tricky — pick the right tool)
SIMPLE ANSWER: threading & asyncio for I/O-bound; multiprocessing for CPU-bound.
TRICK/PATTERN: Match concurrency model to the bottleneck.
PITFALL: Using multiprocessing for tiny tasks (process spawn + pickling overhead).
EXPLAIN: asyncio = single thread, cooperative (one event loop); threading = OS
         threads (GIL-limited for CPU); multiprocessing = true parallel CPU.
         For LLM apps, calls are I/O-bound -> asyncio/threads scale concurrent requests.
"""


# ------------------------------------------------------------------------------
# Q88. (star 5) [Output Prediction] — await returns the value (asyncio)
# ------------------------------------------------------------------------------
import asyncio
async def fetch(x):
    await asyncio.sleep(0)     # yield control; 0 = no real delay
    return x * 2

async def main():
    return await fetch(21)

print("Q88:", asyncio.run(main()))   # 42
"""
RATING: ***** (must-focus — async basics)
SIMPLE ANSWER: 42
TRICK/PATTERN: 'await coro' runs it and gives its RETURN value. asyncio.run starts the loop.
PITFALL: Calling fetch(21) WITHOUT await returns a coroutine object, not 42.
         Also 'await' is only valid inside an 'async def'.
EXPLAIN: Core pattern for concurrent LLM/API calls. async funcs return coroutines.
"""


# ------------------------------------------------------------------------------
# Q89. (star 5) [Output Prediction] — asyncio.gather runs concurrently
# ------------------------------------------------------------------------------
import asyncio
async def work(n):
    await asyncio.sleep(0)
    return n * n

async def run_all():
    return await asyncio.gather(work(1), work(2), work(3))

print("Q89:", asyncio.run(run_all()))   # [1, 4, 9]
"""
RATING: ***** (must-focus — concurrency the right way)
SIMPLE ANSWER: [1, 4, 9]
TRICK/PATTERN: gather schedules coroutines CONCURRENTLY, returns results IN ORDER.
PITFALL: 'await work(1); await work(2)' runs them SEQUENTIALLY (slower).
         Order of results follows argument order, not completion order.
EXPLAIN: This is how you fan out many LLM/embedding calls and wait for all.
"""


# ------------------------------------------------------------------------------
# Q90. (star 4) [Output Prediction / GenAI] — async generator (streaming)
# ------------------------------------------------------------------------------
import asyncio
async def stream():
    for tok in ["Hel", "lo", "!"]:
        await asyncio.sleep(0)
        yield tok

async def collect():
    out = ""
    async for tok in stream():
        out += tok
    return out

print("Q90:", asyncio.run(collect()))   # Hello!
"""
RATING: **** (tricky, GenAI-relevant)
SIMPLE ANSWER: Hello!
TRICK/PATTERN: 'async def' + yield = async generator; consume with 'async for'.
PITFALL: Using a plain 'for' on an async generator -> TypeError. Need 'async for'.
EXPLAIN: This is the exact shape of streaming LLM token output token-by-token.
"""


# ------------------------------------------------------------------------------
# Q91. (star 4) [Short Answer] — common async gotcha: blocking the loop
# ------------------------------------------------------------------------------
print("Q91:", "time.sleep blocks the loop; use await asyncio.sleep")
"""
RATING: **** (tricky — top async mistake)
SIMPLE ANSWER: Calling a BLOCKING function (time.sleep, requests.get, heavy CPU)
               inside async code freezes the WHOLE event loop.
TRICK/PATTERN: In async code use async libs (httpx/aiohttp, asyncio.sleep) or
               run blocking calls via asyncio.to_thread / run_in_executor.
PITFALL: 'async def' does NOT make blocking code non-blocking by itself.
EXPLAIN: One blocking call stalls every other coroutine on that loop.
"""


# ============================ ALGORITHM TRACING ==============================
# (Maps to Cognizant Q7 set-toggle, Q8 sliding window, Q9 stack, Q10 binary search)

# ------------------------------------------------------------------------------
# Q92. (star 5) [Output Prediction] — set toggle / XOR single number (Cognizant Q7)
# ------------------------------------------------------------------------------
def single(nums):
    seen = set()
    for n in nums:
        if n in seen: seen.remove(n)
        else: seen.add(n)
    return seen.pop()

print("Q92:", single([4, 1, 2, 1, 2]))     # 4
"""
RATING: ***** (must-focus — exact Cognizant Q7)
SIMPLE ANSWER: 4
TRICK/PATTERN: Toggle membership; the element appearing an ODD count survives.
PITFALL: set.pop() is fine only because exactly one element remains.
EXPLAIN: XOR equivalent: ans=0; for n: ans ^= n -> 4. Pairs cancel out.
"""


# ------------------------------------------------------------------------------
# Q93. (star 5) [Output Prediction] — sliding window max sum (Cognizant Q8)
# ------------------------------------------------------------------------------
def max_sum_k(nums, k):
    w = sum(nums[:k]); best = w
    for i in range(k, len(nums)):
        w += nums[i] - nums[i - k]
        best = max(best, w)
    return best

print("Q93:", max_sum_k([2, 1, 5, 1, 3, 2], 3))   # 9
"""
RATING: ***** (must-focus — exact Cognizant Q8)
SIMPLE ANSWER: 9
TRICK/PATTERN: Slide window: add nums[i], drop nums[i-k]. O(n) not O(n*k).
PITFALL: Off-by-one in 'nums[i] - nums[i-k]'. Windows: 8,7,9,6 -> max 9.
EXPLAIN: 'k consecutive elements' => sliding window. Min variant uses min().
"""


# ------------------------------------------------------------------------------
# Q94. (star 5) [Output Prediction] — balanced brackets stack (Cognizant Q9)
# ------------------------------------------------------------------------------
def valid(s):
    st = []
    pairs = {")": "(", "]": "[", "}": "{"}
    for ch in s:
        if ch in "([{": st.append(ch)
        else:
            if not st or st.pop() != pairs[ch]: return False
    return not st

print("Q94:", valid("([{}])"), valid("([)]"), valid("((("))  # True False False
"""
RATING: ***** (must-focus — exact Cognizant Q9)
SIMPLE ANSWER: True False False
TRICK/PATTERN: Push openers; on a closer, the popped opener must match.
PITFALL: Forgetting 'return not st' — "(((" never mismatches but leaves stack non-empty.
EXPLAIN: 'matching / nesting / balanced' => stack (LIFO). 'not st' guards empty pops.
"""


# ------------------------------------------------------------------------------
# Q95. (star 5) [Output Prediction] — binary search (Cognizant Q10)
# ------------------------------------------------------------------------------
def bsearch(arr, target):
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target: return mid
        if arr[mid] < target: lo = mid + 1
        else: hi = mid - 1
    return -1

print("Q95:", bsearch([1, 3, 5, 7, 9], 7), bsearch([1, 3, 5, 7, 9], 6))  # 3 -1
"""
RATING: ***** (must-focus — exact Cognizant Q10)
SIMPLE ANSWER: 3 -1
TRICK/PATTERN: Halve the range each step on a SORTED array. O(log n).
PITFALL: 'while lo <= hi' (not <) or you miss the last element. Never name a var 'l'.
EXPLAIN: 'sorted array + find' => binary search. Returns index or -1.
"""


# ------------------------------------------------------------------------------
# Q96. (star 4) [Output Prediction] — two-pointer pair sum (sorted)
# ------------------------------------------------------------------------------
def two_sum_sorted(arr, target):
    i, j = 0, len(arr) - 1
    while i < j:
        s = arr[i] + arr[j]
        if s == target: return (arr[i], arr[j])
        if s < target: i += 1
        else: j -= 1
    return None

print("Q96:", two_sum_sorted([1, 2, 4, 7, 11], 9))   # (2, 7)
"""
RATING: **** (tricky)
SIMPLE ANSWER: (2, 7)
TRICK/PATTERN: Two pointers from both ends; move based on sum vs target. O(n).
PITFALL: Only valid on a SORTED array. For unsorted, use a hash set instead.
EXPLAIN: 'pair that sums to X in a sorted array' => two-pointer.
"""


# ------------------------------------------------------------------------------
# Q97. (star 4) [Output Prediction] — two_sum with dict (unsorted)
# ------------------------------------------------------------------------------
def two_sum(nums, target):
    seen = {}
    for i, n in enumerate(nums):
        if target - n in seen:
            return (seen[target - n], i)
        seen[n] = i
    return None

print("Q97:", two_sum([3, 2, 4], 6))     # (1, 2)
"""
RATING: **** (tricky — the #1 interview question)
SIMPLE ANSWER: (1, 2)
TRICK/PATTERN: Store value->index; check if complement (target-n) was seen. O(n).
PITFALL: Storing AFTER the check avoids reusing the same element twice.
EXPLAIN: 'find a pair' on UNSORTED data => hash map in one pass.
"""


# ------------------------------------------------------------------------------
# Q98. (star 4) [Output Prediction] — Counter / most_common
# ------------------------------------------------------------------------------
from collections import Counter
print("Q98:", Counter("banana").most_common(1))    # [('a', 3)]
"""
RATING: **** (tricky, very common)
SIMPLE ANSWER: [('a', 3)]
TRICK/PATTERN: Counter tallies frequencies; most_common(k) returns top-k (value, count).
PITFALL: most_common(1) returns a LIST of one tuple, not the tuple alone.
EXPLAIN: Counter(text) for char/word frequency. Counter(a) - Counter(b) subtracts.
"""


# ------------------------------------------------------------------------------
# Q99. (star 4) [Output Prediction] — defaultdict grouping
# ------------------------------------------------------------------------------
from collections import defaultdict
d = defaultdict(list)
for word in ["apple", "ant", "bee"]:
    d[word[0]].append(word)
print("Q99:", dict(d))     # {'a': ['apple', 'ant'], 'b': ['bee']}
"""
RATING: **** (tricky)
SIMPLE ANSWER: {'a': ['apple', 'ant'], 'b': ['bee']}
TRICK/PATTERN: defaultdict(list) auto-creates an empty list on first access to a key.
PITFALL: A plain dict raises KeyError on d[k].append without initializing the key.
EXPLAIN: Standard pattern for grouping (anagrams, buckets, adjacency lists).
"""


# ------------------------------------------------------------------------------
# Q100. (star 4) [Output Prediction] — sort by multiple keys
# ------------------------------------------------------------------------------
people = [("Tom", 30), ("Ann", 25), ("Bob", 30)]
print("Q100:", sorted(people, key=lambda p: (-p[1], p[0])))
# [('Bob', 30), ('Tom', 30), ('Ann', 25)]
"""
RATING: **** (tricky — multi-key sort)
SIMPLE ANSWER: [('Bob', 30), ('Tom', 30), ('Ann', 25)]
TRICK/PATTERN: key returns a TUPLE; sort by age DESC (-p[1]) then name ASC (p[0]).
PITFALL: You can't mix reverse=True per-field; negate numeric keys for DESC instead.
EXPLAIN: Python sort is STABLE, so equal keys keep their relative order.
"""


# ------------------------------------------------------------------------------
# Q101. (star 3) [Output Prediction] — string * int and list * int
# ------------------------------------------------------------------------------
print("Q101:", "ab" * 3, [0] * 4)        # ababab [0, 0, 0, 0]
"""
RATING: *** (medium)
SIMPLE ANSWER: ababab  [0, 0, 0, 0]
TRICK/PATTERN: * repeats sequences. Safe for immutables (str, int elements).
PITFALL: For nested mutables [[0]]*n shares refs (see Q68). "ab"*0 -> '' (empty).
EXPLAIN: Quick init of fixed-size flat lists/strings.
"""


# ------------------------------------------------------------------------------
# Q102. (star 4) [Output Prediction] — chained comparison
# ------------------------------------------------------------------------------
x = 5
print("Q102:", 1 < x < 10, 10 < x < 20)   # True False
"""
RATING: **** (tricky)
SIMPLE ANSWER: True  False
TRICK/PATTERN: Python chains comparisons: 1 < x < 10 means (1<x) and (x<10).
PITFALL: Other languages evaluate (1<x)<10 = True<10. Python does NOT; it chains.
EXPLAIN: a == b == c checks all equal. Each operand is evaluated once.
"""


# ------------------------------------------------------------------------------
# Q103. (star 4) [Output Prediction] — and/or return operands (truthiness)
# ------------------------------------------------------------------------------
print("Q103:", 0 or "x", "a" and "b", None or [])   # x b []
"""
RATING: **** (tricky)
SIMPLE ANSWER: x  b  []
TRICK/PATTERN: 'or' returns the first TRUTHY (or last); 'and' returns first FALSY (or last).
PITFALL: They return an OPERAND, not a bool. 0 or "x" -> "x"; "a" and "b" -> "b".
EXPLAIN: Common idiom: name = user_name or "default". Beware 0/""/[]  are falsy.
"""


# ------------------------------------------------------------------------------
# Q104. (star 4) [Output Prediction] — default arg captures variable value
# ------------------------------------------------------------------------------
n = 10
def f(x=n):
    return x
n = 20
print("Q104:", f())     # 10
"""
RATING: **** (tricky)
SIMPLE ANSWER: 10
TRICK/PATTERN: Default values are evaluated ONCE at def time (binds n's value = 10 then).
PITFALL: Reassigning n later does NOT change the captured default.
EXPLAIN: Contrast with closures (Q36) which read the variable LATE. Defaults bind EARLY.
"""


# ------------------------------------------------------------------------------
# Q105. (star 3) [Output Prediction] — *unpacking in calls and assignment
# ------------------------------------------------------------------------------
a, *rest = [1, 2, 3, 4]
print("Q105:", a, rest)     # 1 [2, 3, 4]
"""
RATING: *** (medium)
SIMPLE ANSWER: 1  [2, 3, 4]
TRICK/PATTERN: Starred target captures the 'rest' as a LIST.
PITFALL: *rest is always a list (even if empty: a,*rest=[1] -> rest==[]).
EXPLAIN: first, *mid, last = seq also works. Great for head/tail splits.
"""


# ------------------------------------------------------------------------------
# Q106. (star 4) [Output Prediction] — dict merge with | and update precedence
# ------------------------------------------------------------------------------
d1 = {"a": 1, "b": 2}
d2 = {"b": 3, "c": 4}
print("Q106:", d1 | d2)     # {'a': 1, 'b': 3, 'c': 4}
"""
RATING: **** (tricky)
SIMPLE ANSWER: {'a': 1, 'b': 3, 'c': 4}
TRICK/PATTERN: dict1 | dict2 merges; the RIGHT dict wins on key conflicts (b -> 3).
PITFALL: Order matters: d2 | d1 would give b -> 2. (| for dicts is Python 3.9+.)
EXPLAIN: Same precedence as {**d1, **d2}. Used to layer config defaults + overrides.
"""


# ------------------------------------------------------------------------------
# Q107. (star 4) [Output Prediction] — generator is single-use
# ------------------------------------------------------------------------------
g = (x for x in range(3))
print("Q107:", sum(g), sum(g))    # 3 0
"""
RATING: **** (tricky)
SIMPLE ANSWER: 3  0
TRICK/PATTERN: A generator is exhausted after one full pass; second sum sees nothing.
PITFALL: Reusing a generator silently yields 0/empty, not an error.
EXPLAIN: Materialize once (data = list(g)) if you must iterate multiple times.
"""


# ------------------------------------------------------------------------------
# Q108. (star 5) [Output Prediction] — modifying a list while iterating
# ------------------------------------------------------------------------------
nums = [1, 2, 3, 4, 5]
for n in nums[:]:           # iterate a COPY to delete safely
    if n % 2 == 0:
        nums.remove(n)
print("Q108:", nums)        # [1, 3, 5]
"""
RATING: ***** (must-focus trap — mutate-while-iterating)
SIMPLE ANSWER: [1, 3, 5]
TRICK/PATTERN: Iterating nums[:] (a copy) while removing from nums avoids skips.
PITFALL: Looping the LIVE list and removing skips elements (index shifts) -> wrong result.
FIX: iterate a copy, or build a new list: nums = [n for n in nums if n%2].
EXPLAIN: Never add/remove from a collection you're directly iterating.
"""


# ------------------------------------------------------------------------------
# Q109. (star 4) [Output Prediction] — any / all on empty and mixed
# ------------------------------------------------------------------------------
print("Q109:", all([]), any([]), all([1, 2, 0]), any([0, "", 3]))  # True False False True
"""
RATING: **** (tricky)
SIMPLE ANSWER: True  False  False  True
TRICK/PATTERN: all([]) is True (vacuous truth); any([]) is False.
PITFALL: all() False because 0 is falsy; any() True because 3 is truthy.
EXPLAIN: all/any short-circuit. all = 'no falsy found'; any = 'at least one truthy'.
"""


# ------------------------------------------------------------------------------
# Q110. (star 4) [Output Prediction] — sorted stability + key
# ------------------------------------------------------------------------------
words = ["bb", "a", "ccc", "dd"]
print("Q110:", sorted(words, key=len))   # ['a', 'bb', 'dd', 'ccc']
"""
RATING: **** (tricky)
SIMPLE ANSWER: ['a', 'bb', 'dd', 'ccc']
TRICK/PATTERN: key=len sorts by length; equal lengths keep input order (stable).
PITFALL: 'bb' and 'dd' tie at length 2 -> original order preserved (bb before dd).
EXPLAIN: Stability lets you sort by secondary key first, then primary.
"""


# ------------------------------------------------------------------------------
# Q111. (star 4) [Output Prediction] — nonlocal in nested function
# ------------------------------------------------------------------------------
def outer():
    count = 0
    def inc():
        nonlocal count
        count += 1
        return count
    return inc

c = outer()
print("Q111:", c(), c(), c())    # 1 2 3
"""
RATING: **** (tricky)
SIMPLE ANSWER: 1 2 3
TRICK/PATTERN: nonlocal lets the inner function REBIND the enclosing variable.
PITFALL: Without nonlocal, count += 1 raises UnboundLocalError (treated as local).
EXPLAIN: This is a closure with mutable state (a counter). Contrast 'global' (module scope).
"""


# ------------------------------------------------------------------------------
# Q112. (star 5) [Output Prediction] — UnboundLocalError (assignment makes local)
# ------------------------------------------------------------------------------
x = 10
def show():
    try:
        print(x)        # references x...
        y = x + 1
    except UnboundLocalError:
        return "UnboundLocalError"
# A version that actually triggers it:
def buggy():
    try:
        x += 1          # x is local (because assigned) but used before assignment
        return x
    except UnboundLocalError:
        return "UnboundLocalError"

print("Q112:", buggy())     # UnboundLocalError
"""
RATING: ***** (must-focus trap — scope rule)
SIMPLE ANSWER: UnboundLocalError
TRICK/PATTERN: ANY assignment to a name in a function makes it LOCAL for the whole body.
PITFALL: 'x += 1' reads x first, but x is local & unassigned yet -> UnboundLocalError.
FIX: declare 'global x' (module var) or 'nonlocal x' (enclosing), or pass x in.
EXPLAIN: Reading-only is fine (uses outer x); assigning flips the whole name to local.
"""


# ------------------------------------------------------------------------------
# Q113. (star 4) [Output Prediction] — set/dict ordering & frozenset
# ------------------------------------------------------------------------------
print("Q113:", list({"b": 1, "a": 2}.keys()), 2 in frozenset([1, 2, 3]))  # ['b','a'] True
"""
RATING: **** (tricky)
SIMPLE ANSWER: ['b', 'a']  True
TRICK/PATTERN: dicts preserve INSERTION order (3.7+); sets/frozensets do NOT guarantee order.
PITFALL: Assuming sets are ordered. dict keys keep insertion order; sets don't.
EXPLAIN: frozenset is an immutable, hashable set — usable as a dict key / set element.
"""


# ------------------------------------------------------------------------------
# Q114. (star 4) [Fix the Bug] — recursion base case / depth
# ------------------------------------------------------------------------------
def factorial(n):
    if n <= 1:              # FIX: base case prevents infinite recursion
        return 1
    return n * factorial(n - 1)

print("Q114:", factorial(5))    # 120
"""
RATING: **** (tricky)
SIMPLE ANSWER: 120
TRICK/PATTERN: Every recursion needs a BASE CASE and progress toward it.
PITFALL: Missing/incorrect base case -> RecursionError (default limit ~1000).
EXPLAIN: Python has no tail-call optimization; deep recursion -> convert to a loop.
"""


# ------------------------------------------------------------------------------
# Q115. (star 5) [Output Prediction / GenAI] — Pydantic-style validation + type coercion
# ------------------------------------------------------------------------------
# Simulating Pydantic v2 behavior without importing it (deterministic).
def validate_temperature(value):
    # Pydantic coerces "0.7" (str) -> 0.7 (float) for a float field
    coerced = float(value)
    if not (0.0 <= coerced <= 2.0):
        raise ValueError("temperature out of range")
    return coerced

print("Q115:", validate_temperature("0.7"), validate_temperature(1))   # 0.7 1.0
"""
RATING: ***** (must-focus, GenAI-relevant)
SIMPLE ANSWER: 0.7  1.0
TRICK/PATTERN: Pydantic validates AND coerces types ("0.7" -> 0.7, 1 -> 1.0 for float fields).
PITFALL: Assuming Pydantic rejects a string for a float field — by default it COERCES it.
         (Use strict mode to forbid coercion.) Out-of-range raises ValidationError.
EXPLAIN: Core of agent/tool I/O: define a BaseModel, let Pydantic validate LLM JSON args.
         A 'temperature: float' field accepts "0.7" and stores 0.7.
"""

print()
print("=" * 70)
print("PART 3 COMPLETE (Q66-Q115).")
print("FOCUS (5 star): Q66 shallow copy, Q68 [[]]*n, Q73 MRO/super,")
print("Q76 descriptors, Q77 metaclass, Q86 GIL, Q88/89 async, Q92-95 Cognizant")
print("Q7-Q10 algos, Q108 mutate-while-iterate, Q112 UnboundLocalError, Q115 Pydantic.")
print("=" * 70)


# ##############################################################################
# PART 4 — PRODUCTION-READY & GENAI-FOCUSED (Q116-Q160)
# ##############################################################################
# Topics: real-world debugging, edge cases, logging, prod error handling,
#         JSON/datetime/timezone, Pydantic-style validation, LLM output parsing,
#         streaming, embeddings/cosine similarity, LangChain-style patterns,
#         common GenAI Python mistakes.
# Mix: ~5 medium, ~22 hard, ~18 trap/advanced. Highest difficulty band.
# NOTE: stdlib only — Pydantic/embeddings/LLM behavior is SIMULATED so the
#       file runs anywhere without external packages.
# ##############################################################################

print()
print("=" * 70)
print("PART 4 — PRODUCTION-READY & GENAI-FOCUSED (Q116-Q160)")
print("=" * 70)


# ============================ JSON / LLM OUTPUT PARSING ======================

# ------------------------------------------------------------------------------
# Q116. (star 4) [Output Prediction / GenAI] — json round-trip type changes
# ------------------------------------------------------------------------------
import json
data = {"ok": True, "n": None, "vals": (1, 2)}
restored = json.loads(json.dumps(data))
print("Q116:", restored, type(restored["vals"]).__name__)   # {...} list
"""
RATING: **** (tricky, GenAI-relevant)
SIMPLE ANSWER: {'ok': True, 'n': None, 'vals': [1, 2]}  and  list
TRICK/PATTERN: JSON has no tuple — tuples become LISTS on round-trip.
PITFALL: True/None survive (JSON true/null), but a tuple comes back as a list.
         Dict int keys become STRINGS: json.loads(json.dumps({1:'a'})) -> {'1':'a'}.
EXPLAIN: Critical when caching/serializing LLM tool args — types can silently shift.
"""


# ------------------------------------------------------------------------------
# Q117. (star 5) [Fix the Bug / GenAI] — safely parse messy LLM JSON
# ------------------------------------------------------------------------------
import json
def parse_llm_json(text):
    # LLMs often wrap JSON in prose or ```json fences. Extract the object.
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None

msg = 'Sure! Here is the result:\n```json\n{"action": "search", "k": 3}\n```'
print("Q117:", parse_llm_json(msg))     # {'action': 'search', 'k': 3}
"""
RATING: ***** (must-focus, GenAI-relevant)
SIMPLE ANSWER: {'action': 'search', 'k': 3}
TRICK/PATTERN: LLMs rarely return pure JSON; slice between first '{' and last '}'.
PITFALL: json.loads on the raw text raises JSONDecodeError (prose + ``` fences).
         ALWAYS wrap in try/except — never trust LLM output to be valid JSON.
EXPLAIN: In real code prefer structured-output / function-calling, but this
         salvage parser is the classic fallback. rfind('}') handles trailing text.
"""


# ------------------------------------------------------------------------------
# Q118. (star 4) [Output Prediction] — json.dumps non-serializable -> default
# ------------------------------------------------------------------------------
import json
from datetime import datetime
dt = datetime(2026, 6, 6, 15, 30)
print("Q118:", json.dumps({"when": dt}, default=str))   # {"when": "2026-06-06 15:30:00"}
"""
RATING: **** (tricky)
SIMPLE ANSWER: {"when": "2026-06-06 15:30:00"}
TRICK/PATTERN: json.dumps(..., default=str) stringifies objects it can't serialize.
PITFALL: Without default=, datetime raises TypeError: not JSON serializable.
EXPLAIN: For APIs prefer ISO: default=lambda o: o.isoformat(). Same issue with set/Decimal.
"""


# ------------------------------------------------------------------------------
# Q119. (star 4) [Output Prediction / GenAI] — get nested dict safely
# ------------------------------------------------------------------------------
resp = {"choices": [{"message": {"content": "hi"}}]}
content = resp.get("choices", [{}])[0].get("message", {}).get("content")
print("Q119:", content)     # hi
"""
RATING: **** (tricky, GenAI-relevant)
SIMPLE ANSWER: hi
TRICK/PATTERN: Chain .get with safe defaults to walk an LLM API response.
PITFALL: resp["choices"][0]["message"]["content"] raises KeyError/IndexError if any
         level is missing. Defaults ([{}], {}) keep it from crashing.
EXPLAIN: Exactly how you'd dig into an OpenAI-style chat completion response safely.
"""


# ------------------------------------------------------------------------------
# Q120. (star 3) [Output Prediction] — dict.setdefault
# ------------------------------------------------------------------------------
d = {}
d.setdefault("tags", []).append("genai")
d.setdefault("tags", []).append("python")
print("Q120:", d)           # {'tags': ['genai', 'python']}
"""
RATING: *** (medium)
SIMPLE ANSWER: {'tags': ['genai', 'python']}
TRICK/PATTERN: setdefault(k, default) returns existing value, else sets+returns default.
PITFALL: The second setdefault does NOT overwrite — it returns the existing list.
EXPLAIN: One-liner grouping without defaultdict. Default is created each call (minor cost).
"""


# ============================ DATETIME / TIMEZONE ============================

# ------------------------------------------------------------------------------
# Q121. (star 5) [Output Prediction] — naive vs aware datetime subtraction
# ------------------------------------------------------------------------------
from datetime import datetime, timezone, timedelta
aware = datetime(2026, 1, 1, tzinfo=timezone.utc)
naive = datetime(2026, 1, 1)
try:
    diff = aware - naive
    print("Q121:", diff)
except TypeError:
    print("Q121:", "TypeError: naive vs aware")    # TypeError: naive vs aware
"""
RATING: ***** (must-focus trap — datetime)
SIMPLE ANSWER: TypeError: naive vs aware
TRICK/PATTERN: You cannot subtract a NAIVE datetime from an AWARE one (or compare them).
PITFALL: Mixing tz-aware (has tzinfo) and naive (no tzinfo) datetimes raises TypeError.
EXPLAIN: Pick one policy: store everything in UTC-aware. datetime.now(timezone.utc).
         datetime.utcnow() returns a NAIVE datetime — a classic bug source.
"""


# ------------------------------------------------------------------------------
# Q122. (star 4) [Output Prediction] — timedelta arithmetic
# ------------------------------------------------------------------------------
from datetime import datetime, timedelta
start = datetime(2026, 6, 6, 10, 0, 0)
end = start + timedelta(hours=2, minutes=30)
print("Q122:", (end - start).total_seconds())     # 9000.0
"""
RATING: **** (tricky)
SIMPLE ANSWER: 9000.0
TRICK/PATTERN: datetime - datetime = timedelta; .total_seconds() gives a float.
PITFALL: timedelta.seconds (NOT total_seconds) drops the days part — use total_seconds().
         2h30m = 9000s.
EXPLAIN: Use total_seconds() for durations/latency. .seconds is only the sub-day remainder.
"""


# ------------------------------------------------------------------------------
# Q123. (star 4) [Output Prediction] — ISO format parse/format
# ------------------------------------------------------------------------------
from datetime import datetime
dt = datetime.fromisoformat("2026-06-06T15:30:00")
print("Q123:", dt.year, dt.strftime("%Y-%m-%d %H:%M"))   # 2026 2026-06-06 15:30
"""
RATING: **** (tricky)
SIMPLE ANSWER: 2026  2026-06-06 15:30
TRICK/PATTERN: fromisoformat parses ISO strings; strftime formats with directives.
PITFALL: strptime needs an exact format string; a mismatch raises ValueError.
         %Y=4-digit year, %m=month, %d=day, %H=24h, %M=min.
EXPLAIN: Standard for timestamping logs / LLM trace records.
"""


# ============================ LOGGING / PROD ERROR HANDLING ==================

# ------------------------------------------------------------------------------
# Q124. (star 4) [Fix the Bug] — don't swallow exceptions silently
# ------------------------------------------------------------------------------
def safe_divide(a, b):
    try:
        return a / b
    except ZeroDivisionError as e:
        # FIX: handle specifically + signal failure, don't 'except: pass'
        return None

print("Q124:", safe_divide(10, 2), safe_divide(1, 0))   # 5.0 None
"""
RATING: **** (tricky, best practice)
SIMPLE ANSWER: 5.0  None
TRICK/PATTERN: Catch SPECIFIC exceptions; return a clear signal (None / raise).
PITFALL: 'except: pass' or bare 'except Exception' hides bugs (even KeyboardInterrupt).
EXPLAIN: In prod: log the error (logging.exception) and fail loud or return a typed result.
"""


# ------------------------------------------------------------------------------
# Q125. (star 4) [Output Prediction] — exception chaining (raise from)
# ------------------------------------------------------------------------------
def load(cfg):
    try:
        return cfg["key"]
    except KeyError as e:
        raise ValueError("config missing 'key'") from e

try:
    load({})
except ValueError as e:
    print("Q125:", str(e), "| cause:", type(e.__cause__).__name__)
# config missing 'key' | cause: KeyError
"""
RATING: **** (tricky)
SIMPLE ANSWER: config missing 'key' | cause: KeyError
TRICK/PATTERN: 'raise NewError(...) from e' preserves the original as __cause__.
PITFALL: Re-raising without 'from' loses the root cause context in tracebacks.
EXPLAIN: Wrap low-level errors in domain errors but keep the chain for debugging.
"""


# ------------------------------------------------------------------------------
# Q126. (star 5) [Output Prediction / GenAI] — retry with backoff (decorator)
# ------------------------------------------------------------------------------
import functools
def retry(times):
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            last = None
            for attempt in range(times):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    last = e
            raise last
        return wrapper
    return deco

calls = {"n": 0}
@retry(3)
def flaky():
    calls["n"] += 1
    if calls["n"] < 3:
        raise ConnectionError("transient")
    return "ok"

print("Q126:", flaky(), "after", calls["n"], "tries")   # ok after 3 tries
"""
RATING: ***** (must-focus, GenAI-relevant)
SIMPLE ANSWER: ok after 3 tries
TRICK/PATTERN: Parametrized decorator (retry(3)) wraps a flaky call and retries.
PITFALL: Retrying NON-transient errors (ValueError) wastes time — filter exception types.
         Real impl adds exponential backoff + jitter (time.sleep(2**attempt)).
EXPLAIN: Standard around LLM/vector-DB API calls (rate limits, timeouts). Use tenacity in prod.
"""


# ------------------------------------------------------------------------------
# Q127. (star 4) [Output Prediction] — logging level filtering
# ------------------------------------------------------------------------------
import logging
logger = logging.getLogger("q127")
logger.setLevel(logging.WARNING)
captured = []
class Grab(logging.Handler):
    def emit(self, record): captured.append(record.levelname)
logger.addHandler(Grab())
logger.info("ignored"); logger.warning("shown"); logger.error("shown")
print("Q127:", captured)    # ['WARNING', 'ERROR']
"""
RATING: **** (tricky)
SIMPLE ANSWER: ['WARNING', 'ERROR']
TRICK/PATTERN: Messages below the logger level are dropped. INFO < WARNING -> filtered.
PITFALL: Levels: DEBUG<INFO<WARNING<ERROR<CRITICAL. Default root level is WARNING.
EXPLAIN: Use logging (not print) in prod; control verbosity via level + handlers.
"""


# ------------------------------------------------------------------------------
# Q128. (star 4) [Output Prediction] — assert is for invariants, not validation
# ------------------------------------------------------------------------------
def f(x):
    assert x > 0, "x must be positive"
    return x

try:
    print("Q128:", f(5))
    print("Q128:", f(-1))
except AssertionError as e:
    print("Q128:", "AssertionError:", e)   # 5 ; AssertionError: x must be positive
"""
RATING: **** (tricky, best practice)
SIMPLE ANSWER: 5 ; then AssertionError: x must be positive
TRICK/PATTERN: assert checks an invariant and raises AssertionError if false.
PITFALL: asserts are STRIPPED when Python runs with -O. NEVER use them for input
         validation or security checks — raise real exceptions instead.
EXPLAIN: Use assert for 'should never happen' internal checks; ValueError for bad input.
"""


# ------------------------------------------------------------------------------
# Q129. (star 3) [Output Prediction] — finally overrides return
# ------------------------------------------------------------------------------
def tricky():
    try:
        return "try"
    finally:
        return "finally"

print("Q129:", tricky())    # finally
"""
RATING: *** (medium — nasty trap)
SIMPLE ANSWER: finally
TRICK/PATTERN: A return in finally OVERRIDES a return/exception from try.
PITFALL: The 'try' return value is discarded; finally's return wins (and swallows errors).
EXPLAIN: Avoid returning from finally — it silently hides results and exceptions.
"""


# ------------------------------------------------------------------------------
# Q130. (star 4) [Output Prediction] — walrus operator in a loop
# ------------------------------------------------------------------------------
data = [1, 2, 3, 4, 5]
result = []
it = iter(data)
while (n := next(it, None)) is not None:
    if n % 2 == 0:
        result.append(n)
print("Q130:", result)      # [2, 4]
"""
RATING: **** (tricky, modern Python 3.8+)
SIMPLE ANSWER: [2, 4]
TRICK/PATTERN: ':=' (walrus) assigns AND returns inside an expression.
PITFALL: Watch sentinel: if data can contain None, use a unique sentinel object instead.
EXPLAIN: Common in 'while (chunk := stream.read())' loops and comprehensions.
"""


# ============================ PYDANTIC-STYLE VALIDATION ======================
# (Simulated to run without the pydantic package; behavior mirrors Pydantic v2.)

# ------------------------------------------------------------------------------
# Q131. (star 5) [Output Prediction / GenAI] — type coercion for tool args
# ------------------------------------------------------------------------------
def coerce_field(value, target_type):
    # Pydantic v2 (non-strict) coerces compatible types.
    return target_type(value)

print("Q131:", coerce_field("3", int), coerce_field("0.7", float), coerce_field(1, bool))
# 3 0.7 True
"""
RATING: ***** (must-focus, GenAI-relevant)
SIMPLE ANSWER: 3  0.7  True
TRICK/PATTERN: Pydantic (non-strict) COERCES "3"->3, "0.7"->0.7, 1->True.
PITFALL: Assuming a string is rejected for an int field — by default it's coerced.
         bool("False") is True (non-empty string)! Pydantic parses "false"->False, but
         Python's bool() does NOT. Know which layer you're in.
EXPLAIN: LLMs emit args as strings; Pydantic coercion turns them into typed values.
         Use strict=True to forbid coercion when you need exact types.
"""


# ------------------------------------------------------------------------------
# Q132. (star 5) [Output Prediction / GenAI] — validation error on bad type
# ------------------------------------------------------------------------------
def validate_int(value):
    try:
        return int(value)
    except (ValueError, TypeError):
        return "ValidationError"

print("Q132:", validate_int("42"), validate_int("abc"), validate_int(None))
# 42 ValidationError ValidationError
"""
RATING: ***** (must-focus, GenAI-relevant)
SIMPLE ANSWER: 42  ValidationError  ValidationError
TRICK/PATTERN: Coercion succeeds for "42" but fails for "abc"/None -> Pydantic raises ValidationError.
PITFALL: int(None) raises TypeError (not ValueError) — catch BOTH.
EXPLAIN: In agents, wrap tool-arg parsing so a bad LLM arg becomes a handled
         validation error you can feed back to the model, not a crash.
"""


# ------------------------------------------------------------------------------
# Q133. (star 4) [Output Prediction / GenAI] — default_factory for model lists
# ------------------------------------------------------------------------------
def make_model():
    # Simulates: class M(BaseModel): tags: list = Field(default_factory=list)
    return {"tags": []}

a = make_model(); a["tags"].append("x")
b = make_model()
print("Q133:", a["tags"], b["tags"])    # ['x'] []
"""
RATING: **** (tricky, GenAI-relevant)
SIMPLE ANSWER: ['x']  []
TRICK/PATTERN: default_factory=list creates a FRESH list per instance.
PITFALL: Using a plain mutable default (tags: list = []) would SHARE one list across
         all models (Pydantic actually forbids this; dataclasses error too). Same as Q60.
EXPLAIN: Always default_factory for mutable model fields (list/dict/set).
"""


# ------------------------------------------------------------------------------
# Q134. (star 4) [Output Prediction] — model dump excludes None (concept)
# ------------------------------------------------------------------------------
def model_dump(d, exclude_none=False):
    if exclude_none:
        return {k: v for k, v in d.items() if v is not None}
    return dict(d)

m = {"model": "gpt-4", "temp": None, "top_p": 0.9}
print("Q134:", model_dump(m, exclude_none=True))   # {'model': 'gpt-4', 'top_p': 0.9}
"""
RATING: **** (tricky, GenAI-relevant)
SIMPLE ANSWER: {'model': 'gpt-4', 'top_p': 0.9}
TRICK/PATTERN: Pydantic's model_dump(exclude_none=True) drops None fields.
PITFALL: Sending None params to an LLM API can override server defaults — strip them.
EXPLAIN: Build clean request payloads by excluding unset/None optional params.
"""


# ------------------------------------------------------------------------------
# Q135. (star 4) [Output Prediction / GenAI] — enum-style choice validation
# ------------------------------------------------------------------------------
VALID_ROLES = {"system", "user", "assistant"}
def validate_role(role):
    if role not in VALID_ROLES:
        raise ValueError(f"invalid role: {role}")
    return role

try:
    print("Q135:", validate_role("user"), validate_role("admin"))
except ValueError as e:
    print("Q135:", "user", "|", e)     # user | invalid role: admin
"""
RATING: **** (tricky, GenAI-relevant)
SIMPLE ANSWER: user | invalid role: admin
TRICK/PATTERN: Constrain to an allowed set (Pydantic uses Enum / Literal).
PITFALL: 'role in VALID_ROLES' checks set membership (fast O(1)); a list would be O(n).
EXPLAIN: Chat message roles must be one of system/user/assistant — validate before the API call.
"""


# ============================ EMBEDDINGS / VECTOR UTILS ======================

# ------------------------------------------------------------------------------
# Q136. (star 5) [Output Prediction / GenAI] — cosine similarity
# ------------------------------------------------------------------------------
import math
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb)

print("Q136:", cosine([1, 0], [1, 0]), cosine([1, 0], [0, 1]))   # 1.0 0.0
"""
RATING: ***** (must-focus, GenAI-relevant)
SIMPLE ANSWER: 1.0  0.0
TRICK/PATTERN: cosine = dot(a,b) / (||a|| * ||b||). 1.0 = identical direction, 0 = orthogonal.
PITFALL: Division by zero if a vector is all zeros — guard it. Cosine ignores magnitude.
EXPLAIN: The core similarity metric for embeddings/semantic search. Range -1..1
         (0..1 for typical non-negative-ish text embeddings).
"""


# ------------------------------------------------------------------------------
# Q137. (star 4) [Output Prediction / GenAI] — normalize a vector
# ------------------------------------------------------------------------------
import math
def normalize(v):
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]

print("Q137:", normalize([3, 4]))    # [0.6, 0.8]
"""
RATING: **** (tricky, GenAI-relevant)
SIMPLE ANSWER: [0.6, 0.8]
TRICK/PATTERN: Divide each component by the L2 norm -> unit length (magnitude 1).
PITFALL: norm of [3,4] = 5, so [3/5, 4/5] = [0.6, 0.8]. Guard against zero vector.
EXPLAIN: When embeddings are normalized, DOT PRODUCT == cosine similarity (faster).
"""


# ------------------------------------------------------------------------------
# Q138. (star 5) [Output Prediction / GenAI] — top-k nearest by similarity
# ------------------------------------------------------------------------------
def top_k(query, docs, k):
    scored = [(name, sum(q * d for q, d in zip(query, vec))) for name, vec in docs]
    return [name for name, _ in sorted(scored, key=lambda t: t[1], reverse=True)[:k]]

docs = [("a", [1, 0]), ("b", [0.9, 0.1]), ("c", [0, 1])]
print("Q138:", top_k([1, 0], docs, 2))    # ['a', 'b']
"""
RATING: ***** (must-focus, GenAI-relevant)
SIMPLE ANSWER: ['a', 'b']
TRICK/PATTERN: Score each doc (dot product), sort DESC, take first k. This is retrieval.
PITFALL: reverse=True for similarity (higher = closer). For DISTANCE you'd sort ASC.
EXPLAIN: Mirrors a vector DB top-k search. Real DBs use ANN (HNSW/IVF) for speed.
"""


# ------------------------------------------------------------------------------
# Q139. (star 4) [Output Prediction] — batching a list into chunks
# ------------------------------------------------------------------------------
def batch(items, size):
    return [items[i:i + size] for i in range(0, len(items), size)]

print("Q139:", batch([1, 2, 3, 4, 5], 2))   # [[1, 2], [3, 4], [5]]
"""
RATING: **** (tricky, GenAI-relevant)
SIMPLE ANSWER: [[1, 2], [3, 4], [5]]
TRICK/PATTERN: range(0, n, size) + slicing creates fixed-size batches; last may be short.
PITFALL: Slicing past the end is SAFE in Python (no IndexError) -> last batch [5].
EXPLAIN: Batch embedding/API calls to respect rate limits and reduce round-trips.
"""


# ------------------------------------------------------------------------------
# Q140. (star 4) [Output Prediction / GenAI] — naive token estimate
# ------------------------------------------------------------------------------
def est_tokens(text):
    # rough heuristic: ~1 token per 4 chars
    return max(1, len(text) // 4)

print("Q140:", est_tokens("Hello, world!"))   # 3
"""
RATING: **** (tricky, GenAI-relevant)
SIMPLE ANSWER: 3
TRICK/PATTERN: Rough token count ~= chars/4 for English (real tokenizers differ).
PITFALL: This is an ESTIMATE — use tiktoken for exact counts before hitting limits.
         len("Hello, world!") = 13; 13//4 = 3.
EXPLAIN: Used to pre-check prompt size / budget context window before an API call.
"""


# ============================ LANGCHAIN-STYLE PATTERNS =======================

# ------------------------------------------------------------------------------
# Q141. (star 4) [Output Prediction / GenAI] — prompt template .format
# ------------------------------------------------------------------------------
template = "Answer about {topic} in {n} words."
print("Q141:", template.format(topic="RAG", n=5))   # Answer about RAG in 5 words.
"""
RATING: **** (tricky, GenAI-relevant)
SIMPLE ANSWER: Answer about RAG in 5 words.
TRICK/PATTERN: str.format fills {placeholders} by name; LangChain PromptTemplate wraps this.
PITFALL: A literal { in your prompt must be escaped as {{ or format raises KeyError.
         Missing a key (.format(topic="RAG")) also raises KeyError.
EXPLAIN: This is the core of prompt templating. f-strings interpolate immediately;
         templates defer until .format/.invoke is called with variables.
"""


# ------------------------------------------------------------------------------
# Q142. (star 5) [Output Prediction / GenAI] — escaped braces in prompt (JSON example)
# ------------------------------------------------------------------------------
tmpl = 'Return JSON like {{"k": "v"}} for input {x}.'
print("Q142:", tmpl.format(x="hi"))   # Return JSON like {"k": "v"} for input hi.
"""
RATING: ***** (must-focus, GenAI-relevant)
SIMPLE ANSWER: Return JSON like {"k": "v"} for input hi.
TRICK/PATTERN: {{ }} produce LITERAL braces; {x} is the real placeholder.
PITFALL: Forgetting to double braces around JSON examples -> KeyError: 'k'.
         This is a VERY common LangChain bug when prompts contain JSON.
EXPLAIN: Any literal brace in a format/template string must be doubled.
"""


# ------------------------------------------------------------------------------
# Q143. (star 4) [Output Prediction / GenAI] — pipe-style chaining (Runnable sim)
# ------------------------------------------------------------------------------
class Step:
    def __init__(self, fn): self.fn = fn
    def __or__(self, other): return Step(lambda x: other.fn(self.fn(x)))
    def invoke(self, x): return self.fn(x)

clean = Step(lambda s: s.strip())
upper = Step(lambda s: s.upper())
chain = clean | upper
print("Q143:", chain.invoke("  hi  "))    # HI
"""
RATING: **** (tricky, GenAI-relevant)
SIMPLE ANSWER: HI
TRICK/PATTERN: Overloading __or__ enables 'a | b' pipelines (LangChain LCEL style).
PITFALL: Composition order: clean runs FIRST, then upper (left-to-right).
EXPLAIN: This is how 'prompt | llm | parser' chains work under the hood.
"""


# ------------------------------------------------------------------------------
# Q144. (star 4) [Output Prediction / GenAI] — message list construction
# ------------------------------------------------------------------------------
def build_messages(system, history, user):
    msgs = [{"role": "system", "content": system}]
    msgs += history
    msgs.append({"role": "user", "content": user})
    return [m["role"] for m in msgs]

hist = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
print("Q144:", build_messages("You are helpful", hist, "bye"))
# ['system', 'user', 'assistant', 'user']
"""
RATING: **** (tricky, GenAI-relevant)
SIMPLE ANSWER: ['system', 'user', 'assistant', 'user']
TRICK/PATTERN: Chat APIs take an ordered list: system first, then alternating history, then new user.
PITFALL: 'msgs += history' extends; 'msgs.append(history)' would nest a list (bug).
EXPLAIN: Exact shape of an OpenAI/Anthropic chat 'messages' array.
"""


# ------------------------------------------------------------------------------
# Q145. (star 5) [Output Prediction / GenAI] — streaming accumulation
# ------------------------------------------------------------------------------
def stream_tokens():
    for t in ["The", " quick", " fox"]:
        yield t

full = "".join(tok for tok in stream_tokens())
print("Q145:", full)    # The quick fox
"""
RATING: ***** (must-focus, GenAI-relevant)
SIMPLE ANSWER: The quick fox
TRICK/PATTERN: Accumulate streamed chunks with ''.join over the generator.
PITFALL: Using += in a loop works but is slower; join is idiomatic. Each chunk
         is a partial string — don't assume word boundaries.
EXPLAIN: How you reassemble a streamed LLM response token-by-token for the final text.
"""


# ============================ COMMON GENAI / PROD MISTAKES ===================

# ------------------------------------------------------------------------------
# Q146. (star 5) [Fix the Bug / GenAI] — mutable default in agent state
# ------------------------------------------------------------------------------
def add_message(msg, history=None):
    if history is None:        # FIX: was history=[]
        history = []
    history.append(msg)
    return history

print("Q146:", add_message("a"), add_message("b"))   # ['a'] ['b']
"""
RATING: ***** (must-focus trap, GenAI-relevant)
SIMPLE ANSWER: ['a']  ['b']   (independent — the bug version gives ['a'] then ['a','b'])
TRICK/PATTERN: None sentinel -> fresh history each call (no shared state).
PITFALL: history=[] would persist conversation across unrelated calls (data leak between users!).
EXPLAIN: In multi-user GenAI services this exact bug leaks one user's history into another's.
"""


# ------------------------------------------------------------------------------
# Q147. (star 4) [Output Prediction] — float equality / rounding
# ------------------------------------------------------------------------------
print("Q147:", 0.1 + 0.2 == 0.3, round(0.1 + 0.2, 2) == 0.3)   # False True
"""
RATING: **** (tricky)
SIMPLE ANSWER: False  True
TRICK/PATTERN: Floats are binary approximations: 0.1+0.2 = 0.30000000000000004.
PITFALL: NEVER compare floats with ==. Use round(), or math.isclose(a, b).
EXPLAIN: math.isclose(0.1+0.2, 0.3) is the correct way. Matters for scores/thresholds.
"""


# ------------------------------------------------------------------------------
# Q148. (star 4) [Output Prediction / GenAI] — dedup preserving order
# ------------------------------------------------------------------------------
def dedup(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

print("Q148:", dedup([3, 1, 3, 2, 1]))   # [3, 1, 2]
"""
RATING: **** (tricky)
SIMPLE ANSWER: [3, 1, 2]
TRICK/PATTERN: Track seen in a set; append only first occurrence -> order preserved.
PITFALL: list(set(seq)) dedups but LOSES order. dict.fromkeys(seq) also dedups in order.
EXPLAIN: dict.fromkeys([3,1,3,2,1]) -> keys [3,1,2]; one-line ordered dedup.
"""


# ------------------------------------------------------------------------------
# Q149. (star 4) [Output Prediction] — environment-config fallback chain
# ------------------------------------------------------------------------------
import os
def get_config(key, default):
    return os.environ.get(key) or default

os.environ.pop("MODEL_NAME", None)         # ensure not set
print("Q149:", get_config("MODEL_NAME", "gpt-4o-mini"))   # gpt-4o-mini
"""
RATING: **** (tricky, prod-relevant)
SIMPLE ANSWER: gpt-4o-mini
TRICK/PATTERN: os.environ.get returns None if unset; 'or default' supplies a fallback.
PITFALL: 'or' treats EMPTY STRING as falsy too -> "" env var falls back to default.
         If "" is a valid value, use os.environ.get(key, default) instead of 'or'.
EXPLAIN: Standard 12-factor config: read API keys / model names from env with a default.
"""


# ------------------------------------------------------------------------------
# Q150. (star 5) [Output Prediction] — set operations for tag filtering
# ------------------------------------------------------------------------------
required = {"python", "genai"}
candidate = {"python", "genai", "aws", "react"}
print("Q150:", required <= candidate, required & candidate == required)  # True True
"""
RATING: ***** (must-focus)
SIMPLE ANSWER: True  True
TRICK/PATTERN: required <= candidate tests SUBSET (all required present).
PITFALL: <= is subset, < is PROPER subset. required & candidate == required is an
         equivalent subset check via intersection.
EXPLAIN: Clean way to check 'does this doc/candidate have all required tags/skills'.
"""


# ------------------------------------------------------------------------------
# Q151. (star 4) [Output Prediction] — string startswith/endswith with tuple
# ------------------------------------------------------------------------------
files = ["a.pdf", "b.txt", "c.PDF", "d.md"]
pdfs = [f for f in files if f.lower().endswith((".pdf", ".md"))]
print("Q151:", pdfs)    # ['a.pdf', 'c.PDF', 'd.md']
"""
RATING: **** (tricky)
SIMPLE ANSWER: ['a.pdf', 'c.PDF', 'd.md']
TRICK/PATTERN: endswith accepts a TUPLE of suffixes (OR match). lower() handles case.
PITFALL: Without .lower(), 'c.PDF' is missed. endswith(['.pdf']) (list) raises TypeError — must be tuple.
EXPLAIN: Common when filtering document types in a RAG ingestion pipeline.
"""


# ------------------------------------------------------------------------------
# Q152. (star 4) [Output Prediction / GenAI] — truncate text to budget
# ------------------------------------------------------------------------------
def truncate(text, max_chars):
    return text if len(text) <= max_chars else text[:max_chars - 3] + "..."

print("Q152:", truncate("Hello World", 8))   # Hello...
"""
RATING: **** (tricky, GenAI-relevant)
SIMPLE ANSWER: Hello...
TRICK/PATTERN: Reserve room for the ellipsis: text[:max_chars-3] + '...'.
PITFALL: text[:max_chars] + '...' would EXCEED max_chars. Off-by-three errors are common.
EXPLAIN: Trim context chunks/prompts to fit a char/token budget before the API call.
"""


# ------------------------------------------------------------------------------
# Q153. (star 5) [Output Prediction] — sorting dicts by value, then key
# ------------------------------------------------------------------------------
scores = {"rag": 3, "agent": 5, "llm": 3}
ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
print("Q153:", ranked)    # [('agent', 5), ('llm', 3), ('rag', 3)]
"""
RATING: ***** (must-focus)
SIMPLE ANSWER: [('agent', 5), ('llm', 3), ('rag', 3)]
TRICK/PATTERN: Sort items by value DESC (-kv[1]) then key ASC (kv[0]) for ties.
PITFALL: -kv[1] for descending numeric; can't negate strings, so tie-break ascends naturally.
EXPLAIN: Ranking retrieval results / leaderboards where ties need a stable secondary order.
"""


# ------------------------------------------------------------------------------
# Q154. (star 4) [Output Prediction] — any() with generator (short-circuit)
# ------------------------------------------------------------------------------
def contains_banned(text, banned):
    return any(word in text for word in banned)

print("Q154:", contains_banned("safe message", ["spam", "scam"]))   # False
"""
RATING: **** (tricky, GenAI-relevant)
SIMPLE ANSWER: False
TRICK/PATTERN: any() over a genexp short-circuits at the first True (efficient).
PITFALL: 'word in text' is a SUBSTRING check — 'scam' would match 'scammer'. Use word
         boundaries / tokenization for precise moderation.
EXPLAIN: Basic content-filter / guardrail pattern before sending to an LLM.
"""


# ------------------------------------------------------------------------------
# Q155. (star 5) [Fix the Bug / GenAI] — accumulate cost without float drift
# ------------------------------------------------------------------------------
from decimal import Decimal
def total_cost(token_counts, price_per_1k):
    total = Decimal("0")
    p = Decimal(str(price_per_1k))
    for n in token_counts:
        total += (Decimal(n) / Decimal(1000)) * p
    return float(total)

print("Q155:", total_cost([1000, 1000, 1000], 0.002))   # 0.006
"""
RATING: ***** (must-focus, prod-relevant)
SIMPLE ANSWER: 0.006
TRICK/PATTERN: Use Decimal for MONEY to avoid float drift (0.002*3 in float != clean 0.006).
PITFALL: Summing floats for billing accumulates rounding error over millions of calls.
         Decimal(str(x)) — pass a STRING, not the float, to avoid inheriting float error.
EXPLAIN: Token-cost accounting / billing must be exact -> Decimal, not float.
"""


# ------------------------------------------------------------------------------
# Q156. (star 4) [Output Prediction] — itertools.islice on a generator
# ------------------------------------------------------------------------------
import itertools
def infinite():
    i = 0
    while True:
        yield i
        i += 1

print("Q156:", list(itertools.islice(infinite(), 5)))   # [0, 1, 2, 3, 4]
"""
RATING: **** (tricky)
SIMPLE ANSWER: [0, 1, 2, 3, 4]
TRICK/PATTERN: islice takes the first N from an iterator WITHOUT materializing it.
PITFALL: list(infinite()) would hang forever. islice bounds an infinite/lazy stream.
EXPLAIN: Safely cap streaming/paginated sources (e.g., first N results from a cursor).
"""


# ------------------------------------------------------------------------------
# Q157. (star 4) [Output Prediction] — functools.lru_cache memoization
# ------------------------------------------------------------------------------
import functools
calls = {"n": 0}
@functools.lru_cache(maxsize=None)
def square(x):
    calls["n"] += 1
    return x * x

print("Q157:", square(4), square(4), calls["n"])   # 16 16 1
"""
RATING: **** (tricky)
SIMPLE ANSWER: 16  16  1
TRICK/PATTERN: lru_cache memoizes results by args; the 2nd square(4) is cached (no recompute).
PITFALL: Args must be HASHABLE (no lists/dicts). Cache can grow unbounded with maxsize=None.
EXPLAIN: Cache pure, expensive, repeatable calls (e.g., embedding the same text twice).
"""


# ------------------------------------------------------------------------------
# Q158. (star 5) [Output Prediction] — mutating arg inside function (aliasing)
# ------------------------------------------------------------------------------
def add_tag(config, tag):
    config["tags"].append(tag)     # mutates the CALLER's dict (passed by reference)

cfg = {"tags": ["base"]}
add_tag(cfg, "extra")
print("Q158:", cfg)    # {'tags': ['base', 'extra']}
"""
RATING: ***** (must-focus trap)
SIMPLE ANSWER: {'tags': ['base', 'extra']}
TRICK/PATTERN: Python passes object REFERENCES; mutating a dict/list arg affects the caller.
PITFALL: Assuming the function gets a copy. It doesn't — side effects leak out.
FIX: copy.deepcopy inside, or treat inputs as immutable and return a new object.
EXPLAIN: A top source of 'spooky action at a distance' bugs in shared config/state.
"""


# ------------------------------------------------------------------------------
# Q159. (star 4) [Output Prediction] — dict.get vs [] for counting
# ------------------------------------------------------------------------------
text = "aba"
freq = {}
for ch in text:
    freq[ch] = freq.get(ch, 0) + 1
print("Q159:", freq)    # {'a': 2, 'b': 1}
"""
RATING: **** (tricky)
SIMPLE ANSWER: {'a': 2, 'b': 1}
TRICK/PATTERN: freq.get(ch, 0) + 1 safely initializes missing keys to 0.
PITFALL: freq[ch] += 1 on a missing key raises KeyError. (Counter(text) is the shortcut.)
EXPLAIN: The manual frequency-count idiom — appears constantly in screening tests.
"""


# ------------------------------------------------------------------------------
# Q160. (star 5) [Output Prediction / GenAI] — full mini RAG-style pipeline trace
# ------------------------------------------------------------------------------
import math
def cos(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a)); nb = math.sqrt(sum(y*y for y in b))
    return dot / (na * nb) if na and nb else 0.0

def retrieve(query_vec, store, k, threshold):
    scored = [(doc, cos(query_vec, vec)) for doc, vec in store]
    scored = [(d, s) for d, s in scored if s >= threshold]      # retrieval gate
    scored.sort(key=lambda t: t[1], reverse=True)
    return [d for d, _ in scored[:k]]

store = [("doc_python", [1, 0, 0]), ("doc_rag", [0, 1, 0]), ("doc_mix", [0.7, 0.7, 0])]
hits = retrieve([1, 0, 0], store, k=2, threshold=0.5)
print("Q160:", hits)    # ['doc_python', 'doc_mix']
"""
RATING: ***** (must-focus, GenAI-relevant — the capstone)
SIMPLE ANSWER: ['doc_python', 'doc_mix']
TRICK/PATTERN: Score by cosine -> apply a THRESHOLD gate -> sort DESC -> take top-k.
PITFALL: doc_rag (similarity 0.0) is filtered by the 0.5 gate; doc_mix (~0.707) passes.
         The gate is what prevents low-relevance hits feeding the LLM (anti-hallucination).
EXPLAIN: This is the heart of a RAG retriever: embed query, similarity search,
         confidence gate, top-k. Real systems add reranking + metadata filters.
"""

print()
print("=" * 70)
print("PART 4 COMPLETE (Q116-Q160). ALL 160 QUESTIONS DONE.")
print("FOCUS (5 star): Q117 LLM-JSON parse, Q121 naive/aware datetime,")
print("Q126 retry decorator, Q131/132 Pydantic coercion, Q136/138 cosine+top-k,")
print("Q142 escaped braces, Q145 streaming, Q146/158 mutable-state leaks,")
print("Q155 Decimal cost, Q160 full RAG retriever trace.")
print("=" * 70)


# ##############################################################################
# MOCK TEST — TIMED 10-QUESTION COGNIZANT-STYLE ROUND
# ##############################################################################
"""
HOW TO USE:
    - Give yourself 15 MINUTES, NO IDE, pen and paper. Predict each output.
    - Mix mirrors a real screening: 2 easy, 5 medium, 3 advanced/trap.
    - Write your 10 answers down FIRST.
    - Then scroll to the ANSWER KEY block (or run reveal_answers()) to score.
    - Passing = 7/10. You scored 4/10 at Cognizant; target 7-8 here.

DO NOT read the answer key until you've attempted all 10.

────────────────────────────────────────────────────────────────────────────
M1. (easy)  What is the output?
        s = "data"
        print(s[1:3], s[-1], len(s))                      

M2. (easy)  What is the output?
        d = {"a": 1, "b": 2}
        print("a" in d, 1 in d, list(d.values()))       

M3. (medium)  What is the output?
        nums = [5, 3, 8, 1, 9, 2]
        print(sorted([x for x in nums if x > 3])[1:3])    

M4. (medium)  What is the output?
        def f(x, acc=[]):
            acc.append(x)
            return acc
        print(f(1), f(2), f(3))         

M5. (medium)  What is the output?
        funcs = [lambda: i for i in range(3)]
        print([g() for g in funcs])    

M6. (medium)  What is the output?
        a = [1, 2, 3]
        b = a
        c = a[:]
        a += [4]
        print(b is a, c == a, len(b), len(c))  

M7. (medium)  What is the output?
        class A:
            def f(self): return "A"
        class B(A):
            def f(self): return "B" + super().f()
        class C(A):
            def f(self): return "C" + super().f()
        class D(B, C): pass
        print(D().f())                  

M8. (advanced)  What is the output?
        def max_window(nums, k):
            w = sum(nums[:k]); best = w
            for i in range(k, len(nums)):
                w += nums[i] - nums[i - k]
                best = max(best, w)
            return best
        print(max_window([1, 4, 2, 10, 2, 3], 3))          

M9. (advanced)  What is the output?
        nums = [2, 3, 2, 4, 3]
        seen = set()
        for n in nums:
            seen.symmetric_difference_update({n})
        print(seen.pop())                                 

M10. (advanced/trap)  What is the output?
        def g():
            try:
                return "try"
            finally:
                return "finally"
        print(g(), 0.1 + 0.2 == 0.3)
────────────────────────────────────────────────────────────────────────────
"""


def reveal_answers():
    """Call this AFTER you've written your 10 answers. Prints the key + scoring."""
    key = [
        ("M1", "at a 4",
         "s[1:3]='at' (idx 1,2; stop exclusive); s[-1]='a'; len=4."),
        ("M2", "True False [1, 2]",
         "'in' checks KEYS -> 'a' True, 1 False (it's a value). .values() -> [1,2]."),
        ("M3", "[8, 9]",
         "filter>3 -> [5,8,9]; sorted -> [5,8,9]; [1:3] -> idx 1,2 -> [8,9]."),
        ("M4", "[1] [1, 2] [1, 2, 3]",
         "Mutable default acc=[] is SHARED across calls -> accumulates."),
        ("M5", "[2, 2, 2]",
         "Late-binding closure: all lambdas read final i==2. Fix: lambda i=i: i."),
        ("M6", "True False 4 3",
         "b is a (alias)=True. c=a[:] is a copy; a+=[4] mutates a&b in place, "
         "so a=[1,2,3,4], c stays [1,2,3] -> c==a is False. len(b)=4, len(c)=3."),
        ("M7", "BCA",
         "Diamond MRO D->B->C->A; super() follows MRO, not literal parent."),
        ("M8", "16",
         "Windows: [1,4,2]=7, [4,2,10]=16, [2,10,2]=14, [10,2,3]=15 -> max 16."),
        ("M9", "4",
         "symmetric_difference toggles membership; odd-count element (4) survives."),
        ("M10", "finally False",
         "return in finally overrides try; 0.1+0.2 != 0.3 (float drift)."),
    ]
    print("\n" + "=" * 70)
    print("MOCK TEST — ANSWER KEY")
    print("=" * 70)
    for qid, ans, why in key:
        print(f"{qid}: {ans}")
        print(f"     {why}")
    print("=" * 70)
    print("Passing = 7/10. Count ONLY exact-match outputs as correct.")
    print("=" * 70)


# Self-check: compute the tricky answers live so the key is never wrong.
def _verify_mock():
    # M6
    a = [1, 2, 3]; b = a; c = a[:]; a += [4]
    m6 = (b is a, c == a, len(b), len(c))
    # M8
    def max_window(nums, k):
        w = sum(nums[:k]); best = w
        for i in range(k, len(nums)):
            w += nums[i] - nums[i - k]; best = max(best, w)
        return best
    m8 = max_window([1, 4, 2, 10, 2, 3], 3)
    return m6, m8


print()
print("=" * 70)
print("MOCK TEST READY (10 questions). Attempt on paper, 15 min, no IDE.")
print("Then call reveal_answers() to score. Passing = 7/10.")
print("Verified tricky outputs -> M6:", _verify_mock()[0], "| M8:", _verify_mock()[1])
print("=" * 70)
