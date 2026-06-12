"""
Python Coding Prep — Arrays/Lists (JS → Python)

KEY SYNTAX DIFFERENCES (memorize these):
    JS: array.push(x)          Python: lst.append(x)
    JS: array.length            Python: len(lst)
    JS: array.includes(x)      Python: x in lst
    JS: array.indexOf(x)       Python: lst.index(x)
    JS: array.filter(fn)       Python: [x for x in lst if condition]
    JS: array.map(fn)          Python: [fn(x) for x in lst]
    JS: array.reduce(fn, init) Python: from functools import reduce
    JS: [...new Set(arr)]      Python: list(set(arr))
    JS: array.sort((a,b)=>a-b) Python: lst.sort() or sorted(lst)
    JS: array.splice(i, 1)     Python: lst.pop(i) or del lst[i]
    JS: Math.max(...arr)        Python: max(arr)
    JS: Math.min(...arr)        Python: min(arr)
    JS: for (let x of arr)     Python: for x in lst:
    JS: arr.forEach(fn)        Python: for x in lst: fn(x)
    JS: console.log()          Python: print()
"""


# ==============================================================================
# 1. FIND DUPLICATES IN ARRAY
# ==============================================================================
# JS: const findDuplicates = (arr) => { const count = {}; ... }


for n in range(2, 101):
    is_prime = True
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            is_prime = False
            break
    if is_prime:
        print(n, end=" ")


#print 1st 10 prime numbers
import math

primes = []
num = 2

while len(primes) < 10:

    is_prime = True

    for i in range(2, int(math.sqrt(num)) + 1):
        if num % i == 0:
            is_prime = False
            break

    if is_prime:
        primes.append(num)

    num += 1

print(primes)

#check given num is prime or not
import math

n = 5

for i in range(2, int(math.sqrt(n)) + 1):
    if n % i == 0:
        print("Not Prime")
        break
else:
    print("Prime")

# Method 1: Using dictionary (same logic as your JS)
def find_duplicates(arr):
    count = {}
    duplicates = []
    for num in arr:
        count[num] = count.get(num, 0) + 1
    for key, val in count.items():
        if val > 1:
            duplicates.append(key)
    return duplicates

# Method 2: Using set (Pythonic)
def find_duplicates_set(arr):
    seen = set()
    dupes = set()
    for x in arr:
        if x in seen:
            dupes.add(x)
        seen.add(x)
    return list(dupes)

print("Duplicates:", find_duplicates([1, 2, 3, 2, 4, 5, 3]))
# [2, 3]


# ==============================================================================
# 2. COMMON ELEMENTS BETWEEN TWO ARRAYS
# ==============================================================================
# JS: const set1 = [...new Set(array1)]; set2 = new Set(array2);
#     const dupds = set1.filter(e => set2.has(e));

def common_elements(arr1, arr2):
    set1 = set(arr1)
    set2 = set(arr2)
    return list(set1 & set2)  # & is set intersection

# OR one-liner:
# common = list(set(arr1) & set(arr2))

print("Common:", common_elements([1, 2, 8, 12, 9], [11, 2, 18, 12, 9]))
# [2, 9, 12]


# ==============================================================================
# 3. FIND MAX AND MIN
# ==============================================================================
# JS: Math.max(...array), Math.min(...array)

def find_min_max(arr):
    return {"min": min(arr), "max": max(arr)}

# Manual way (same as your JS):
def find_min_max_manual(arr):
    mn, mx = arr[0], arr[0]
    for num in arr:
        if num > mx: mx = num
        if num < mn: mn = num
    return {"min": mn, "max": mx}

print("MinMax:", find_min_max([5, 2, 9, 1, 7]))
# {'min': 1, 'max': 9}


# ==============================================================================
# 4. SECOND MAXIMUM ELEMENT
# ==============================================================================
# JS: [...new Set(arr)].sort((a,b) => b-a)[1]

def second_max(arr):
    unique = sorted(set(arr), reverse=True)
    return unique[1] if len(unique) >= 2 else None

# Manual way (same as your JS):
def second_max_manual(arr):
    mx = float('-inf')       # JS: -Infinity
    second = float('-inf')
    for num in arr:
        if num > mx:
            second = mx
            mx = num
        elif num > second and num < mx:
            second = num
    return second

print("2nd Max:", second_max([100, 100, 50, 10, 200, 400, 300, 400, 500, 600, 600]))
# 500


# ==============================================================================
# 5. MISSING NUMBERS IN ARRAY
# ==============================================================================
# JS: for (let i = Math.min(...arr); i < Math.max(...arr); i++)
#       if (!arr.includes(i)) missing.push(i);

def find_missing(arr):
    arr = sorted(set(arr))
    missing = []
    for i in range(min(arr), max(arr)):
        if i not in arr:
            missing.append(i)
    return missing

print("Missing:", find_missing([3, 5, 4, 7, 9, 10]))
# [6, 8]


# ==============================================================================
# 6. SEPARATE EVEN AND ODD, SORT DIFFERENTLY
# ==============================================================================
# JS: arr.forEach(x => x%2===0 ? even.push(x) : odd.push(x))

def separate_even_odd(arr):
    unique = list(set(arr))
    evens = sorted([x for x in unique if x % 2 == 0])
    odds = sorted([x for x in unique if x % 2 != 0], reverse=True)
    return evens, odds

print("Even/Odd:", separate_even_odd([3, 4, 2, 5, 7, 13, 2, 5, 78, 9, 23, 28, 4, 12, 3, 7]))


# ==============================================================================
# 7. MAXIMUM REPEATED ELEMENT
# ==============================================================================
# JS: const count = {}; for (let num of arr) count[num] = (count[num]||0)+1;

def max_repeated(arr):
    count = {}
    for num in arr:
        count[num] = count.get(num, 0) + 1
    return max(count, key=count.get)

# OR using Counter (Pythonic):
from collections import Counter
def max_repeated_counter(arr):
    return Counter(arr).most_common(1)[0][0]

print("Max Repeated:", max_repeated([1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5]))
# 5


# ==============================================================================
# 8. PRODUCT OF EVEN NUMBERS
# ==============================================================================
# JS: nums.filter(a => a%2==0).reduce((a,b) => a*b, 1)

from functools import reduce
def product_of_even(arr):
    evens = [x for x in arr if x % 2 == 0]
    return reduce(lambda a, b: a * b, evens, 1)

print("Product Even:", product_of_even([1, 2, 3, 4, 5, 6, 7, 8]))
# 384


# ==============================================================================
# 9. TWO SUM (find pairs that add to target)
# ==============================================================================
# JS: const seen = new Map(); for (let i...) complement = target - arr[i];

def two_sum(arr, target):
    result = []
    seen = set()
    
    for ele in arr:
        complement = target - ele
        if complement in seen:
            result.append([ele, complement])
        seen.add(ele)
    
    return result

def two_sum_pairs(arr, target):
    seen = set()
    result = []
    for num in arr:
        complement = target - num
        if complement in seen:
            result.append([num, complement])
        seen.add(num)
    return result

print("Two Sum Indices:", two_sum([2, 7, 11, 15], 9))
# [[0, 1]]
print("Two Sum Pairs:", two_sum_pairs([10, 20, 30, 40, 50, 60, 70, 80, 90], 100))


# ==============================================================================
# 10. SUM OF ALL EXCEPT CURRENT ELEMENT
# ==============================================================================
# JS: arr.map(i => arr.filter(j => j!=i).reduce((a,b) => a+b, 0))

def sum_except_current(arr):
    total = sum(arr)
    return [total - x for x in arr]

print("Sum Except:", sum_except_current([1, 2, 3, 4, 5, 6, 7]))
# [27, 26, 25, 24, 23, 22, 21]


# ==============================================================================
# 11. PRODUCT OF ALL EXCEPT CURRENT
# ==============================================================================
def product_except_current(arr):
    result = []
    for i in range(len(arr)):
        product = 1
        for j in range(len(arr)):
            if i != j:
                product *= arr[j]
        result.append(product)
    return result

print("Product Except:", product_except_current([1, 2, 3, 4, 5]))
# [120, 60, 40, 30, 24]


# ==============================================================================
# 12. FLATTEN NESTED ARRAY
# ==============================================================================
# JS: arr.flat(Infinity)

def flatten(arr):
    result = []
    for item in arr:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result

print("Flatten:", flatten([1, [1, 2, 3], [1, 2, [7], 0], 9, 8, [5]]))
# [1, 1, 2, 3, 1, 2, 7, 0, 9, 8, 5]


# ==============================================================================
# 13. BUBBLE SORT (without built-in)
# ==============================================================================
# JS: for i... for j... if arr[j] > arr[j+1] swap

def bubble_sort(arr):
    arr = arr.copy()
    for i in range(len(arr)):
        for j in range(len(arr) - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]  # Python swap!
    return arr

print("Bubble Sort:", bubble_sort([5, 2, 67, 1, 36, 4, 8, 3]))


# ==============================================================================
# 14. FIBONACCI SERIES
# ==============================================================================
# JS: for (let i=2; i<n; i++) seq.push(seq[i-1]+seq[i-2])

def fibonacci(n):
    seq = [0, 1]
    for i in range(2, n):
        seq.append(seq[i-1] + seq[i-2])
    return seq

print("Fibonacci:", fibonacci(10))
# [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]


# ==============================================================================
# 15. CATEGORIZE NUMBERS INTO RANGES
# ==============================================================================
# JS: reduce with acc[0], acc[1], acc[2]

def categorize(arr):
    result = [[], [], []]
    for num in arr:
        if 0 < num < 100:
            result[0].append(num)
        elif 100 <= num < 200:
            result[1].append(num)
        else:
            result[2].append(num)
    return result

print("Categorize:", categorize([100, 1, 200, 2, 4, 5, 102, 201, 105]))


# ==============================================================================
# 16. GROUP ANAGRAMS
# ==============================================================================
# JS: let sorted = each.split("").sort().join("")

def group_anagrams(arr):
    groups = {}
    for word in arr:
        key = "".join(sorted(word))
        if key not in groups:
            groups[key] = []
        groups[key].append(word)
    return list(groups.values())

print("Anagrams:", group_anagrams(["eat", "tea", "ate", "max", "axm", "box", "xob"]))


# ==============================================================================
# 17. REVERSE ARRAY IN PLACE
# ==============================================================================
# JS: for (let i=0; i<arr.length/2; i++) swap arr[i] and arr[len-1-i]

def reverse_in_place(arr):
    arr = arr.copy()
    for i in range(len(arr) // 2):
        arr[i], arr[len(arr) - 1 - i] = arr[len(arr) - 1 - i], arr[i]
    return arr

# Pythonic: arr[::-1]

print("Reverse:", reverse_in_place([1, 2, 3, 4]))
# [4, 3, 2, 1]


# ==============================================================================
# 18. FILTER NUMERIC STRINGS FROM MIXED ARRAY
# ==============================================================================
# JS: if (!isNaN(+i)) result.push(+i)

def filter_numeric(arr):
    result = []
    for item in arr:
        if item.isdigit():
            result.append(int(item))
    return result

# Pythonic:
# [int(x) for x in arr if x.isdigit()]

print("Numeric:", filter_numeric(["1", "2", "seetha", "3", "geetha", "4"]))
# [1, 2, 3, 4]


# ==============================================================================
# 19. SUM OF EVEN AND ODD (1 to 100)
# ==============================================================================
def sum_even_odd(n):
    even_sum = sum(i for i in range(1, n+1) if i % 2 == 0)
    odd_sum = sum(i for i in range(1, n+1) if i % 2 != 0)
    return even_sum, odd_sum

print("Even/Odd Sum:", sum_even_odd(100))
# (2550, 2500)


# ==============================================================================
# 20. FACTORIAL
# ==============================================================================
def factorial(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

print("Factorial:", factorial(6))
# 720
