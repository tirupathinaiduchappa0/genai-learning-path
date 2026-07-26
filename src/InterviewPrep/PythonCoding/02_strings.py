"""
Python Coding Prep — Strings (JS → Python)

KEY SYNTAX DIFFERENCES:
    JS: str.split("")           Python: list(s) or s.split("")  (but list(s) is cleaner)
    JS: str.split(" ")          Python: s.split(" ") or s.split()
    JS: arr.join("")            Python: "".join(lst)
    JS: str.substring(0, 3)     Python: s[0:3] or s[:3]
    JS: str.charAt(0)           Python: s[0]
    JS: str.toUpperCase()       Python: s.upper()
    JS: str.toLowerCase()       Python: s.lower()
    JS: str.includes("x")      Python: "x" in s
    JS: str.replace("a","b")   Python: s.replace("a","b")
    JS: str.length              Python: len(s)
    JS: str.trim()              Python: s.strip()
    JS: str.padStart(2,"0")    Python: s.zfill(2) or s.rjust(2,"0")
    JS: str.repeat(3)          Python: s * 3
    JS: `template ${var}`      Python: f"template {var}"
"""


# ==============================================================================
# 1. REVERSE A STRING
# ==============================================================================
# JS: str.split("").reverse().join("")

def reverse_string(s):
    return s[::-1]  # Python slicing — the most Pythonic way

# Manual way (same as your JS):
def reverse_string_manual(s):
    result = ""
    for i in range(len(s) - 1, -1, -1):
        result += s[i]
    return result

print("Reverse:", reverse_string("hello"))
# olleh


# ==============================================================================
# 2. CHECK PALINDROME
# ==============================================================================
# JS: s === s.split("").reverse().join("")

def is_palindrome(s):
    s = s.lower().replace(" ", "")
    return s == s[::-1]

print("Palindrome:", is_palindrome("racecar"))       # True
print("Palindrome:", is_palindrome("A man a plan a canal Panama"))  # True


# ==============================================================================
# 3. COUNT VOWELS IN STRING
# ==============================================================================
# JS: strArray.reduce((count, char) => vowels.includes(char) ? count+1 : count, 0)

def count_vowels(s):
    vowels = "aeiou"
    return sum(1 for c in s.lower() if c in vowels)

print("Vowels:", count_vowels("I love My country"))
# 5


# ==============================================================================
# 4. COUNT EACH VOWEL SEPARATELY
# ==============================================================================
# JS: strArray.reduce((acc, char) => { if (vowels.includes(char)) acc[char]=(acc[char]||0)+1; return acc; }, {})

def count_each_vowel(s):
    vowels = "aeiou"
    count = {}
    for c in s.lower():
        if c in vowels:
            count[c] = count.get(c, 0) + 1
    return count

print("Each Vowel:", count_each_vowel("I love My country"))


# ==============================================================================
# 5. COUNT EACH CHARACTER FREQUENCY
# ==============================================================================
# JS: s.reduce((acc, curr) => { acc[curr]=(acc[curr]||0)+1; return acc; }, {})

def char_frequency(s):
    s = s.replace(" ", "")
    count = {}
    for c in s:
        count[c] = count.get(c, 0) + 1
    return count

# Pythonic: from collections import Counter; Counter(s)

print("Char Freq:", char_frequency("This is java code"))


# ==============================================================================
# 6. MOST REPEATED CHARACTER
# ==============================================================================
# JS: Object.keys(count).reduce((a,b) => count[a]>count[b] ? a : b)

def most_repeated_char(s):
    count = char_frequency(s)
    return max(count, key=count.get)

print("Most Repeated:", most_repeated_char("This is java code"))


# ==============================================================================
# 7. REVERSE EACH WORD IN SENTENCE
# ==============================================================================
# JS: sent.split(" ").map(word => word.split("").reverse().join("")).join(" ")

def reverse_each_word(s):
    return " ".join(word[::-1] for word in s.split(" "))

print("Reverse Words:", reverse_each_word("my name is tirupathi naidu"))
# ym eman si ihtapurit udian


# ==============================================================================
# 8. REVERSE FULL SENTENCE AND EACH WORD
# ==============================================================================
# JS: str.split(" ").reverse().map(x => x.split("").reverse().join("")).join(" ")

def reverse_sentence_and_words(s):
    return " ".join(word[::-1] for word in s.split(" ")[::-1])

print("Rev Sent+Words:", reverse_sentence_and_words("i am not string"))
# gnirts ton ma i


# ==============================================================================
# 9. CAPITALIZE FIRST LETTER OF EACH WORD
# ==============================================================================
# JS: str.split(" ").map(x => x[0].toUpperCase() + x.slice(1).toLowerCase()).join(" ")

def capitalize_words(s):
    return " ".join(word[0].upper() + word[1:].lower() for word in s.split(" "))

# Pythonic: s.title()

print("Capitalize:", capitalize_words("i am good boy"))
# I Am Good Boy


# ==============================================================================
# 10. REPLACE UNDERSCORE WITH SPACE AND CAPITALIZE
# ==============================================================================
# JS: str.split("_").map(x => x[0].toUpperCase() + x.slice(1)).join(" ")

def underscore_to_title(s):
    return " ".join(word.capitalize() for word in s.split("_"))

print("Underscore:", underscore_to_title("i_am_good_boy"))
# I Am Good Boy


# ==============================================================================
# 11. RUN-LENGTH ENCODING (compress string)
# ==============================================================================
# JS: "abbbcccadeff" → "a1b3c3a1d1e1f2"

def run_length_encode(s):
    result = ""
    count = 1
    for i in range(len(s)):
        if s[i] == s[i + 1]:
            count += 1
        else:
            result += s[i] + str(count)
            count = 1
    return result

print("RLE:", run_length_encode("abbbcccadeff"))
# a1b3c3a1d1e1f2


# ==============================================================================
# 12. SPLIT ON CAPITAL LETTERS AND ADD SPACES
# ==============================================================================
# JS: for (let c of str) if (c >= "A" && c <= "Z") result += " " + c.toLowerCase()

def split_camel_case(s):
    result = ""
    for c in s:
        if c.isupper():
            result += " " + c.lower()
        else:
            result += c
    return result.strip()

print("CamelCase:", split_camel_case("naiduChappaTirupathiNaidu"))
# naidu chappa tirupathi naidu


# ==============================================================================
# 13. MAXIMUM LENGTH WORD IN SENTENCE
# ==============================================================================
# JS: str.split(" ").reduce((a,b) => a.length > b.length ? a : b)

def max_length_word(s):
    return max(s.split(" "), key=len)

print("Max Word:", max_length_word("I love my India"))
# India


# ==============================================================================
# 14. SWAP TWO STRINGS WITHOUT THIRD VARIABLE
# ==============================================================================
# JS: x = x+y; y = x.substring(0, x.length-y.length); x = x.substring(y.length)

def swap_strings():
    a, b = "Ramu", "Beemu"
    a, b = b, a  # Python makes this trivial!
    return a, b

print("Swap:", swap_strings())
# ('Beemu', 'Ramu')


# ==============================================================================
# 15. CHECK IF TWO STRINGS ARE ANAGRAMS
# ==============================================================================
# JS: sorted(s1) === sorted(s2)

def is_anagram(s1, s2):
    return sorted(s1.lower()) == sorted(s2.lower())

print("Anagram:", is_anagram("listen", "silent"))
# True


# ==============================================================================
# 16. ALPHABETICAL ORDER OF STRING
# ==============================================================================
# JS: str.split("").sort().join("")

def alphabetical_order(s):
    return "".join(sorted(s))

print("Alpha:", alphabetical_order("alphabet"))
# aabehlt


# ==============================================================================
# 17. COUNT SPECIFIC CHARACTER
# ==============================================================================
# JS: for (let c of str) if (c === "l") count++

def count_char(s, char):
    return s.lower().count(char.lower())

print("Count 'l':", count_char("Hello world", "l"))
# 3


# ==============================================================================
# 18. REVERSE STRING PRESERVING SPACES
# ==============================================================================
# JS: complex logic with index tracking

def reverse_preserve_spaces(s):
    chars = list(s.replace(" ", ""))[::-1]
    result = list(s)
    idx = 0
    for i in range(len(result)):
        if result[i] != " ":
            result[i] = chars[idx]
            idx += 1
    return "".join(result)

print("Rev Preserve:", reverse_preserve_spaces("i am good"))
# d oo gmai


# ==============================================================================
# 19. LONGEST SUBSTRING WITHOUT REPEATING CHARACTERS
# ==============================================================================
def longest_unique_substring(s):
    result = ""
    for i in range(len(s)):
        seen = set()
        current = ""
        for j in range(i, len(s)):
            if s[j] in seen:
                break
            seen.add(s[j])
            current += s[j]
        if len(current) > len(result):
            result = current
    return result

print("Longest Unique:", longest_unique_substring("tirupathinaidu"))


# ==============================================================================
# 20. GROUP STRINGS BY FIRST CHARACTER
# ==============================================================================
# JS: arr.reduce((acc, curr) => { const f=curr[0]; if(!acc[f]) acc[f]=[]; acc[f].push(curr); return acc; }, {})

def group_by_first_char(arr):
    groups = {}
    for word in arr:
        key = word[0]
        if key not in groups:
            groups[key] = []
        groups[key].append(word)
    return groups

print("Group:", group_by_first_char(["Rajini", "Ramu", "Suhash", "Soundarya", "Aman"]))

#Group anagrams from a list of strings.

input_data = ["eat", "tea", "tan", "ate", "nat", "bat"]

def groupAnagrams(data):
    freq = {}
    for word in data:
        key = "".join(sorted(word))  # sort chars -> string key
        if key not in freq:
            freq[key] = []
        freq[key].append(word)       # append, don't overwrite
    print(freq)
    return list(freq.values())

print(groupAnagrams(input_data))
