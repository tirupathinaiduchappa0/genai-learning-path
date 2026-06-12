"""
Python Coding Prep — Advanced Patterns (JS → Python)

These are the remaining unique problems from your Advanced JS Coding folder.
Focus on: Currying, Patterns, Math, Partition, Anagrams, and more.
"""


# ==============================================================================
# 1. CURRYING — INFINITE ADD (with sentinel)
# ==============================================================================
# JS: const add = (a) => (b) => b ? add(a+b) : a;
# Python doesn't auto-coerce, so we use a sentinel (empty call returns result)

def add(a):
    def inner(b=None):
        if b is None:
            return a
        return add(a + b)
    return inner

print("Currying:", add(1)(2)(3)())       # 6
print("Currying:", add(5)())              # 5
print("Currying:", add(1)(2)())           # 3
print("Currying:", add(10)(20)(30)(40)()) # 100


# ==============================================================================
# 2. STAR PATTERN — FULL SQUARE (n x n)
# ==============================================================================
# JS: for(i=0;i<n;i++) { row=""; for(j=0;j<n;j++) row+="*"; console.log(row) }

def star_square(n):
    for i in range(n):
        row = ""
        for j in range(n):
            row += "* "
        print(row)

print("--- Pattern 1: Full Square ---")
star_square(5)


# ==============================================================================
# 3. STAR PATTERN — RIGHT TRIANGLE
# ==============================================================================
# JS: for(i=0;i<n;i++) { row=""; for(j=0;j<i+1;j++) row+="* "; console.log(row) }

def star_right_triangle(n):
    for i in range(n):
        row = ""
        for j in range(i + 1):
            row += "* "
        print(row)

print("--- Pattern 2: Right Triangle ---")
star_right_triangle(5)


# ==============================================================================
# 4. STAR PATTERN — NUMBER TRIANGLE (1 2 3...)
# ==============================================================================
# JS: for(i=0;i<n;i++) { row=""; for(j=0;j<i+1;j++) row+=(j+1)+" "; }

def number_triangle(n):
    for i in range(n):
        row = ""
        for j in range(i + 1):
            row += str(j + 1) + " "
        print(row)

print("--- Pattern 3: Number Triangle ---")
number_triangle(5)


# ==============================================================================
# 5. STAR PATTERN — REPEATED ROW NUMBER
# ==============================================================================
# JS: for(i=0;i<n;i++) { row=""; for(j=0;j<i+1;j++) row+=(i+1)+" "; }

def repeated_row_number(n):
    for i in range(n):
        row = ""
        for j in range(i + 1):
            row += str(i + 1) + " "
        print(row)

print("--- Pattern 4: Repeated Row Number ---")
repeated_row_number(5)


# ==============================================================================
# 6. STAR PATTERN — INVERTED NUMBER TRIANGLE
# ==============================================================================
# JS: for(i=0;i<n;i++) { row=""; for(j=0;j<n-i;j++) row+=(j+1)+" "; }

def inverted_number_triangle(n):
    for i in range(n):
        row = ""
        for j in range(n - i):
            row += str(j + 1) + " "
        print(row)

print("--- Pattern 5: Inverted Number Triangle ---")
inverted_number_triangle(5)


# ==============================================================================
# 7. COUNT DIGITS IN A NUMBER
# ==============================================================================
# JS: while(num > 10) { num /= 10; count++ }

def count_digits(n):
    n = abs(n)
    count = 1
    while n >= 10:       # JS uses > 10, but >= 10 is correct
        n //= 10         # integer division in Python
        count += 1
    return count

# Pythonic: len(str(abs(n)))

print("Count Digits:", count_digits(12345))   # 5
print("Count Digits:", count_digits(-987))    # 3
print("Count Digits:", count_digits(0))       # 1


# ==============================================================================
# 8. COUNT SMALLER NUMBERS AFTER SELF
# ==============================================================================
# JS: for each i, count how many j > i have arr[j] < arr[i]

def count_smaller_after_self(arr):
    result = []
    for i in range(len(arr)):
        count = 0
        for j in range(i + 1, len(arr)):
            if arr[j] < arr[i]:
                count += 1
        result.append(count)
    return result

print("Smaller After Self:", count_smaller_after_self([5, 2, 6, 1]))  # [2, 1, 1, 0]


# ==============================================================================
# 9. EMAIL NAME EXTRACTION
# ==============================================================================
# JS: eachMail.split("@")[0].replace(".", " ")

def extract_names(emails):
    result = []
    for email in emails:
        name_part = email.split("@")[0]
        result.append(name_part.replace(".", " "))
    return result

# Pythonic one-liner:
def extract_names_pythonic(emails):
    return [email.split("@")[0].replace(".", " ") for email in emails]

emails = ["tiru.naidu@gmail.com", "chappa.naidu@gmail.com"]
print("Names:", extract_names(emails))  # ['tiru naidu', 'chappa naidu']


# ==============================================================================
# 10. MAX SUM OF 2 ELEMENTS — NOT CONSECUTIVE INTEGERS
# ==============================================================================
# JS: skip if Math.abs(arr[i] - arr[j]) === 1

def max_sum_non_consecutive_values(arr):
    max_sum = float('-inf')
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if abs(arr[i] - arr[j]) == 1:  # skip consecutive integers
                continue
            max_sum = max(max_sum, arr[i] + arr[j])
    return max_sum

print("Max Sum (non-consec values):", max_sum_non_consecutive_values([4, 7, 10, 3, 8]))  # 18 (10+8)


# ==============================================================================
# 11. MAX SUM OF 3 ELEMENTS — NOT ALL 3 CONSECUTIVE INTEGERS
# ==============================================================================
def max_sum_3_non_consecutive(arr):
    max_sum = float('-inf')
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            for k in range(j + 1, len(arr)):
                vals = sorted([arr[i], arr[j], arr[k]])
                # skip if all 3 form consecutive integers (like 5,6,7)
                if vals[1] == vals[0] + 1 and vals[2] == vals[1] + 1:
                    continue
                max_sum = max(max_sum, arr[i] + arr[j] + arr[k])
    return max_sum

print("Max Sum 3:", max_sum_3_non_consecutive([5, 1, 3, 7, 2, 8]))  # 20 (5+7+8)


# ==============================================================================
# 12. MAX SUM OF 3 ELEMENTS — NO TWO CHOSEN FROM ADJACENT INDICES
# ==============================================================================
def max_sum_3_non_adjacent_index(arr):
    max_sum = float('-inf')
    for i in range(len(arr)):
        for j in range(i + 2, len(arr)):        # skip adjacent index
            for k in range(j + 2, len(arr)):     # skip adjacent index
                max_sum = max(max_sum, arr[i] + arr[j] + arr[k])
    return max_sum

print("Max Sum 3 (non-adj idx):", max_sum_3_non_adjacent_index([5, 1, 3, 7, 2, 8]))  # 16 (5+3+8)


# ==============================================================================
# 13. MEDIAN OF TWO SORTED ARRAYS
# ==============================================================================
# JS: concat + sort, then find middle

def find_median(arr1, arr2):
    merged = sorted(arr1 + arr2)
    n = len(merged)
    if n % 2 == 1:
        return merged[n // 2]
    else:
        return (merged[n // 2] + merged[n // 2 - 1]) / 2

print("Median:", find_median([1, 3], [2]))      # 2
print("Median:", find_median([1, 2], [3, 4]))    # 2.5


# ==============================================================================
# 14. SECOND LARGEST ELEMENT (O(n) single pass)
# ==============================================================================
# JS: track firstLargest and secondLargest, skip duplicates

def second_largest(arr):
    first = second = float('-inf')
    for num in arr:
        if num > first:
            second = first
            first = num
        elif num > second and num < first:  # skip duplicates
            second = num
    return second if second != float('-inf') else None

print("2nd Largest:", second_largest([0, 3, 5, 2, 7, 12, 12]))  # 7
print("2nd Largest:", second_largest([10, 20]))                   # 10


# ==============================================================================
# 15. PRODUCT OF ALL ELEMENTS EXCEPT CURRENT
# ==============================================================================
# JS: for each i, multiply all j where j != i

def product_except_self(arr):
    result = []
    for i in range(len(arr)):
        product = 1
        for j in range(len(arr)):
            if i != j:
                product *= arr[j]
        result.append(product)
    return result

print("Product Except Self:", product_except_self([1, 2, 3, 4, 5]))
# [120, 60, 40, 30, 24]

# ==============================================================================
# 16. CHECK SQUARE ELEMENTS OF ANOTHER ARRAY
# ==============================================================================
# JS: for each element in arr1, check if arr2 includes its square

def check_square_elements(arr1, arr2):
    if len(arr1) != len(arr2):
        return False
    for num in arr1:
        if num * num not in arr2:
            return False
    return True

# Pythonic:
def check_square_pythonic(arr1, arr2):
    return sorted(x * x for x in arr1) == sorted(arr2)

print("Square Check:", check_square_elements([1, 2, 3, 4], [1, 4, 9, 16]))  # True
print("Square Check:", check_square_elements([1, 2, 3], [1, 4, 10]))        # False


# ==============================================================================
# 17. FRUIT MAP — FIRST LETTER → STARS
# ==============================================================================
# JS: result[key[0]] = "*".repeat(map[key])

def fruit_star_map(fruits):
    result = {}
    for key, value in fruits.items():
        result[key[0]] = "*" * value    # Python: "*" * n  ==  JS: "*".repeat(n)
    # Sort by key
    sorted_entries = sorted(result.items())
    return result, sorted_entries

fruits = {"mango": 5, "banana": 6, "guava": 3, "avocado": 7}
result, sorted_result = fruit_star_map(fruits)
print("Fruit Map:", result)
print("Sorted:", sorted_result)
# {'m': '*****', 'b': '******', 'g': '***', 'a': '*******'}
# [('a', '*******'), ('b', '******'), ('g', '***'), ('m', '*****')]


# ==============================================================================
# 18. GROUP ANAGRAMS
# ==============================================================================
# JS: sort each string, use as key in freq object

def group_anagrams(words):
    groups = {}
    for word in words:
        key = "".join(sorted(word))   # JS: word.split("").sort().join("")
        if key not in groups:
            groups[key] = []
        groups[key].append(word)
    return list(groups.values())

# Pythonic with defaultdict:
from collections import defaultdict

def group_anagrams_pythonic(words):
    groups = defaultdict(list)
    for word in words:
        groups["".join(sorted(word))].append(word)
    return list(groups.values())

print("Anagrams:", group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"]))
# [['eat', 'tea', 'ate'], ['tan', 'nat'], ['bat']]


# ==============================================================================
# 19. FIBONACCI — ITERATIVE + RECURSIVE
# ==============================================================================
# JS: arr.push(arr[i-2] + arr[i-1])

def fibonacci_iterative(n):
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[i - 2] + fib[i - 1])
    return fib

def fibonacci_recursive(n):
    if n <= 1:
        return n
    return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)

print("Fib Series:", fibonacci_iterative(10))
# [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
print("Fib(9):", fibonacci_recursive(9))  # 34


# ==============================================================================
# 20. BOARD PAINTERS — PARTITION PROBLEM (Min/Max time)
# ==============================================================================
# JS: split boards into 2 groups, find min of max(left, right)

def board_painters(boards):
    total = sum(boards)
    left_sum = 0
    min_time = float('inf')
    max_time = 0

    for i in range(len(boards)):
        left_sum += boards[i]
        right_sum = total - left_sum
        time_required = max(left_sum, right_sum)
        min_time = min(min_time, time_required)
        max_time = max(max_time, time_required)

    return min_time, max_time

min_t, max_t = board_painters([10, 20, 30, 40])
print(f"Painters — Min: {min_t}, Max: {max_t}")  # Min: 60, Max: 100


# ==============================================================================
# 21. FILTER PRODUCTS + ADD DISCOUNT (Map/Filter/Reduce)
# ==============================================================================
# JS: products.filter(p => p.price > 20).map(p => ({...p, discountedPrice: p.price * 0.9}))

products = [
    {"id": 1, "name": "Product A", "price": 20, "category": "Electronics"},
    {"id": 2, "name": "Product B", "price": 30, "category": "Clothing"},
    {"id": 3, "name": "Product C", "price": 15, "category": "Electronics"},
    {"id": 4, "name": "Product D", "price": 25, "category": "Clothing"},
    {"id": 5, "name": "Product E", "price": 50, "category": "Electronics"},
]

discounted = [
    {**p, "discountedPrice": round(p["price"] * 0.9, 2)}
    for p in products
    if p["price"] > 20
]
print("Discounted:", discounted)


# ==============================================================================
# 22. COUNT NEGATIVE NUMBERS + FIND LARGEST + FIND SMALLEST
# ==============================================================================
def count_negatives(arr):
    count = 0
    for num in arr:
        if num < 0:
            count += 1
    return count

def find_largest(arr):
    largest = arr[0]
    for num in arr[1:]:
        if num > largest:
            largest = num
    return largest

def find_smallest(arr):
    smallest = arr[0]
    for num in arr[1:]:
        if num < smallest:
            smallest = num
    return smallest

test = [2, -6, 4, 8, 1, -9]
print("Negatives:", count_negatives(test))   # 2
print("Largest:", find_largest(test))         # 8
print("Smallest:", find_smallest(test))       # -9
# Pythonic: len([x for x in arr if x < 0]), max(arr), min(arr)


# ==============================================================================
# 23. FIRST NON-REPEATING + FIRST REPEATING (in array)
# ==============================================================================
def first_non_repeating_and_repeating(arr):
    # Build frequency map
    freq = {}
    for num in arr:
        freq[num] = freq.get(num, 0) + 1

    first_non_rep = None
    first_rep = None

    for num in arr:
        if freq[num] == 1 and first_non_rep is None:
            first_non_rep = num
        if freq[num] > 1 and first_rep is None:
            first_rep = num
        if first_non_rep is not None and first_rep is not None:
            break

    return first_non_rep, first_rep

print("Non-Rep & Rep:", first_non_repeating_and_repeating([4, 5, 1, 2, 1, 5, 3]))
# (4, 5) — 4 is first non-repeating, 5 is first repeating


# ==============================================================================
# 24. VALID ANAGRAM
# ==============================================================================
# JS: s.split("").sort().join("") === t.split("").sort().join("")

def is_anagram(s, t):
    return sorted(s) == sorted(t)

# Using frequency count (O(n)):
def is_anagram_freq(s, t):
    if len(s) != len(t):
        return False
    freq = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    for c in t:
        freq[c] = freq.get(c, 0) - 1
    return all(v == 0 for v in freq.values())

print("Anagram:", is_anagram("anagram", "nagaram"))       # True
print("Anagram:", is_anagram("rat", "car"))                # False
print("Anagram (freq):", is_anagram_freq("listen", "silent"))  # True


# ==============================================================================
# 25. UNIQUE ELEMENTS COUNT
# ==============================================================================
# JS: use freq object, count keys with freq[i] = true

def count_unique(arr):
    freq = {}
    for num in arr:
        freq[num] = True
    return len(freq)

# Pythonic: len(set(arr))

print("Unique Count:", count_unique([1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9, 10, 9]))  # 10
print("Unique (set):", len(set([1, 1, 2, 2, 3, 3, 4, 5])))  # 5
