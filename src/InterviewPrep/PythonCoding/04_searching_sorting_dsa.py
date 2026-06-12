"""
Python Coding Prep — Searching, Sorting & DSA (JS → Python)

These are the DSA-level problems from your Advanced JS Coding folder.
Focus on: Binary Search, Kadane's, Sliding Window, Two Pointers.
"""


# ==============================================================================
# 1. LINEAR SEARCH
# ==============================================================================
def linear_search(arr, target):
    for i, val in enumerate(arr):
        if val == target:
            return i
    return -1

print("Linear Search:", linear_search([10, 20, 30, 40, 50], 30))  # 2


# ==============================================================================
# 2. BINARY SEARCH (sorted array, O(log n))
# ==============================================================================
# JS: let left=0, right=arr.length-1; while(left<=right) mid=Math.floor((left+right)/2)

def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2    # JS: Math.floor((left+right)/2)
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

print("Binary Search:", binary_search([1, 3, 5, 7, 9, 11], 7))  # 3


# ==============================================================================
# 3. SEARCH INSERT POSITION (Binary Search variant)
# ==============================================================================
def search_insert(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return left  # insertion point

print("Insert Pos:", search_insert([1, 3, 5, 6], 5))  # 2
print("Insert Pos:", search_insert([1, 3, 5, 6], 2))  # 1


# ==============================================================================
# 4. KADANE'S ALGORITHM (Maximum Subarray Sum)
# ==============================================================================
# JS: let maxSum = arr[0], currSum = arr[0]; for(let i=1...)

def max_subarray_sum(arr):
    max_sum = curr_sum = arr[0]
    for num in arr[1:]:
        curr_sum = max(num, curr_sum + num)
        max_sum = max(max_sum, curr_sum)
    return max_sum

print("Kadane's:", max_subarray_sum([-2, 1, -3, 4, -1, 2, 1, -5, 4]))  # 6


# ==============================================================================
# 5. MAX CONSECUTIVE ONES
# ==============================================================================
def max_consecutive_ones(arr):
    max_count = count = 0
    for num in arr:
        if num == 1:
            count += 1
            max_count = max(max_count, count)
        else:
            count = 0
    return max_count

print("Max Ones:", max_consecutive_ones([1, 1, 0, 1, 1, 1]))  # 3


# ==============================================================================
# 6. MOVE ZEROS TO END
# ==============================================================================
# JS: let pos=0; for(let i...) if(arr[i]!==0) arr[pos++]=arr[i]; fill rest with 0

def move_zeros(arr):
    arr = arr.copy()
    pos = 0
    for num in arr:
        if num != 0:
            arr[pos] = num
            pos += 1
    while pos < len(arr):
        arr[pos] = 0
        pos += 1
    return arr

# Pythonic:
def move_zeros_pythonic(arr):
    non_zeros = [x for x in arr if x != 0]
    return non_zeros + [0] * (len(arr) - len(non_zeros))

print("Move Zeros:", move_zeros([0, 1, 0, 3, 12]))  # [1, 3, 12, 0, 0]


# ==============================================================================
# 7. MERGE TWO SORTED ARRAYS
# ==============================================================================
def merge_sorted(arr1, arr2):
    result = []
    i = j = 0
    while i < len(arr1) and j < len(arr2):
        if arr1[i] <= arr2[j]:
            result.append(arr1[i])
            i += 1
        else:
            result.append(arr2[j])
            j += 1
    result.extend(arr1[i:])
    result.extend(arr2[j:])
    return result

# Pythonic: sorted(arr1 + arr2)

print("Merge Sorted:", merge_sorted([1, 3, 5], [2, 4, 6]))  # [1,2,3,4,5,6]


# ==============================================================================
# 8. SINGLE NUMBER (all appear twice except one)
# ==============================================================================
# JS: arr.reduce((a,b) => a^b, 0)  — XOR trick

def single_number(arr):
    result = 0
    for num in arr:
        result ^= num  # XOR: same numbers cancel out
    return result

print("Single:", single_number([4, 1, 2, 1, 2]))  # 4


# ==============================================================================
# 9. BEST TIME TO BUY AND SELL STOCK
# ==============================================================================
def max_profit(prices):
    min_price = float('inf')
    max_profit = 0
    for price in prices:
        min_price = min(min_price, price)
        max_profit = max(max_profit, price - min_price)
    return max_profit

print("Stock Profit:", max_profit([7, 1, 5, 3, 6, 4]))  # 5 (buy@1, sell@6)


# ==============================================================================
# 10. ROTATE ARRAY LEFT BY K POSITIONS
# ==============================================================================
def rotate_left(arr, k):
    k = k % len(arr)
    return arr[k:] + arr[:k]

print("Rotate Left:", rotate_left([1, 2, 3, 4, 5], 2))  # [3, 4, 5, 1, 2]


# ==============================================================================
# 11. MISSING NUMBER (0 to n, one missing)
# ==============================================================================
# JS: n*(n+1)/2 - sum(arr)

def missing_number(arr):
    n = len(arr)
    expected = n * (n + 1) // 2
    return expected - sum(arr)

print("Missing:", missing_number([3, 0, 1]))  # 2


# ==============================================================================
# 12. REMOVE ELEMENT IN-PLACE
# ==============================================================================
def remove_element(arr, val):
    return [x for x in arr if x != val]

print("Remove:", remove_element([3, 2, 2, 3], 3))  # [2, 2]


# ==============================================================================
# 13. KTH LARGEST ELEMENT
# ==============================================================================
def kth_largest(arr, k):
    return sorted(arr, reverse=True)[k - 1]

print("3rd Largest:", kth_largest([3, 2, 1, 5, 6, 4], 3))  # 4


# ==============================================================================
# 14. SLIDING WINDOW — MAX SUM OF K CONSECUTIVE
# ==============================================================================
def max_sum_k(arr, k):
    window_sum = sum(arr[:k])
    max_sum = window_sum
    for i in range(k, len(arr)):
        window_sum += arr[i] - arr[i - k]
        max_sum = max(max_sum, window_sum)
    return max_sum

print("Sliding Window:", max_sum_k([2, 1, 5, 1, 3, 2], 3))  # 9


# ==============================================================================
# 15. FIND PIVOT IN ROTATED SORTED ARRAY
# ==============================================================================
def find_pivot(arr):
    for i in range(len(arr) - 1):
        if arr[i] > arr[i + 1]:
            return i + 1
    return 0

print("Pivot:", find_pivot([4, 5, 6, 7, 0, 1, 2]))  # 4


# ==============================================================================
# 16. MERGE INTERVALS
# ==============================================================================
def merge_intervals(intervals):
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged

print("Merge Intervals:", merge_intervals([[1,3],[2,6],[8,10],[15,18]]))
# [[1,6],[8,10],[15,18]]


# ==============================================================================
# 17. CONTAINER WITH MOST WATER (Two Pointers)
# ==============================================================================
def max_area(heights):
    left, right = 0, len(heights) - 1
    max_water = 0
    while left < right:
        width = right - left
        height = min(heights[left], heights[right])
        max_water = max(max_water, width * height)
        if heights[left] < heights[right]:
            left += 1
        else:
            right -= 1
    return max_water

print("Max Water:", max_area([1, 8, 6, 2, 5, 4, 8, 3, 7]))  # 49


# ==============================================================================
# 18. FIRST NON-REPEATING CHARACTER
# ==============================================================================
def first_non_repeating(s):
    count = {}
    for c in s:
        count[c] = count.get(c, 0) + 1
    for c in s:
        if count[c] == 1:
            return c
    return None

print("First Non-Rep:", first_non_repeating("aabbcdd"))  # c


# ==============================================================================
# 19. PALINDROME NUMBER (without string conversion)
# ==============================================================================
def is_palindrome_num(n):
    if n < 0:
        return False
    original = n
    reversed_num = 0
    while n > 0:
        reversed_num = reversed_num * 10 + n % 10
        n //= 10
    return original == reversed_num

print("Palindrome Num:", is_palindrome_num(121))   # True
print("Palindrome Num:", is_palindrome_num(-121))  # False


# ==============================================================================
# 20. STRING COMPRESSION
# ==============================================================================
def compress(s):
    result = ""
    count = 1
    for i in range(len(s)):
        if s[i] == s[i + 1]:
            count += 1
        else:
            result += s[i] + str(count)
            count = 1
    return result

print("Compress:", compress("aabcccccaaa"))  # a2b1c5a3
