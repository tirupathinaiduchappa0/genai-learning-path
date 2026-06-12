"""
Python Coding Prep — Dictionaries/Objects (JS → Python)

KEY SYNTAX DIFFERENCES:
    JS: const obj = {}           Python: d = {}
    JS: obj[key] = val           Python: d[key] = val
    JS: obj.hasOwnProperty(k)    Python: k in d
    JS: Object.keys(obj)         Python: d.keys() or list(d.keys())
    JS: Object.values(obj)       Python: d.values() or list(d.values())
    JS: Object.entries(obj)      Python: d.items() or list(d.items())
    JS: for (let key in obj)     Python: for key in d:
    JS: {...obj1, ...obj2}       Python: {**d1, **d2}
    JS: delete obj[key]          Python: del d[key]
    JS: obj[key] || 0            Python: d.get(key, 0)
"""


# ==============================================================================
# 1. COMMON ELEMENTS BETWEEN TWO OBJECTS
# ==============================================================================
# JS: Object.keys(obj1).reduce((acc, curr) => { if (obj2[curr]===obj1[curr]) acc[curr]=obj1[curr]; return acc; }, {})

def common_objects(obj1, obj2):
    result = {}
    for key in obj1:
        if key in obj2 and obj1[key] == obj2[key]:
            result[key] = obj1[key]
    return result

print("Common:", common_objects({"a": 1, "b": 2, "c": 3, "d": 4}, {"d": 4, "e": 5}))
# {'d': 4}


# ==============================================================================
# 2. MERGE OBJECTS BY SUMMING VALUES
# ==============================================================================
# JS: let keys = [...new Set([...Object.keys(a1), ...Object.keys(a2)])]

def merge_sum(d1, d2):
    keys = set(list(d1.keys()) + list(d2.keys()))
    return {k: d1.get(k, 0) + d2.get(k, 0) for k in keys}

print("Merge Sum:", merge_sum({"a": 1, "b": 2}, {"a": 4, "b": 6, "c": 3}))
# {'a': 5, 'b': 8, 'c': 3}


# ==============================================================================
# 3. OBJECT TO ARRAY OF ENTRIES
# ==============================================================================
# JS: Object.entries(obj) → [[key, val], ...]

def obj_to_entries(d):
    return list(d.items())

# OR: [[k, v] for k, v in d.items()]

print("Entries:", obj_to_entries({"name": "Ramu", "age": 30, "gender": "male"}))
# [('name', 'Ramu'), ('age', 30), ('gender', 'male')]


# ==============================================================================
# 4. CONVERT ARRAY TO OBJECT (index as key)
# ==============================================================================
# JS: arr.reduce((acc, curr, i) => { acc[i]=curr; return acc; }, {})

def array_to_obj(arr):
    return {i: val for i, val in enumerate(arr)}

print("Arr→Obj:", array_to_obj(["ramu", "teena", "beemu", "seetha"]))
# {0: 'ramu', 1: 'teena', 2: 'beemu', 3: 'seetha'}


# ==============================================================================
# 5. GROUP ARRAY BY FIRST LETTER
# ==============================================================================
# JS: arr.reduce((acc, curr) => { const f=curr[0]; if(!acc[f]) acc[f]=[]; acc[f].push(curr); return acc; }, {})

def group_by_first_letter(arr):
    groups = {}
    for word in arr:
        key = word[0]
        if key not in groups:
            groups[key] = []
        groups[key].append(word)
    return groups

# Pythonic with defaultdict:
from collections import defaultdict
def group_by_first_letter_dd(arr):
    groups = defaultdict(list)
    for word in arr:
        groups[word[0]].append(word)
    return dict(groups)

print("Group:", group_by_first_letter(["Rajini", "Ramu", "Reethu", "Suhash", "Aman"]))


# ==============================================================================
# 6. COUNT GENDER FROM ARRAY OF OBJECTS
# ==============================================================================
# JS: arr.reduce((acc, curr) => { acc[curr.gen]=(acc[curr.gen]||0)+1; return acc; }, {})

def count_gender(people):
    count = {}
    for person in people:
        g = person["gen"]
        count[g] = count.get(g, 0) + 1
    return count

people = [
    {"name": "ramu", "gen": "M", "age": 30},
    {"name": "seetha", "gen": "F", "age": 23},
    {"name": "geetha", "gen": "F", "age": 33},
    {"name": "sam", "gen": "M", "age": 13},
]
print("Gender Count:", count_gender(people))
# {'M': 2, 'F': 2}


# ==============================================================================
# 7. GROUP BY GENDER
# ==============================================================================
def group_by_gender(people):
    groups = defaultdict(list)
    for person in people:
        groups[person["gen"]].append(person)
    return dict(groups)

print("Group Gender:", {k: len(v) for k, v in group_by_gender(people).items()})


# ==============================================================================
# 8. GROUP ACTORS BY NAME, LIST MOVIES
# ==============================================================================
# JS: actors.reduce((acc, curr) => { if(!acc[curr.name]) acc[curr.name]=[]; acc[curr.name].push(curr.movies); return acc; }, {})

actors = [
    {"name": "prabas", "movies": "chatrapathi"},
    {"name": "prabas", "movies": "bahu1"},
    {"name": "prabas", "movies": "bahu2"},
    {"name": "ntr", "movies": "student 1"},
    {"name": "ntr", "movies": "rrr"},
    {"name": "charan", "movies": "rrr"},
    {"name": "charan", "movies": "magadeera"},
]

def group_movies(actors):
    groups = defaultdict(list)
    for actor in actors:
        groups[actor["name"]].append(actor["movies"])
    return dict(groups)

print("Movies:", group_movies(actors))


# ==============================================================================
# 9. COUNT MOVIES PER ACTOR
# ==============================================================================
def count_movies(actors):
    count = {}
    for actor in actors:
        name = actor["name"]
        count[name] = count.get(name, 0) + 1
    return count

print("Movie Count:", count_movies(actors))


# ==============================================================================
# 10. MULTIPLY NUMERIC VALUES IN OBJECT BY 2
# ==============================================================================
# JS: for (key in user) if (typeof user[key]==="number") user[key]*=2

def multiply_numeric(d, factor=2):
    return {k: v * factor if isinstance(v, (int, float)) else v for k, v in d.items()}

print("Multiply:", multiply_numeric({"name": "ramu", "salary": 1000, "age": 20}))
# {'name': 'ramu', 'salary': 2000, 'age': 40}


# ==============================================================================
# 11. SORT ARRAY OF OBJECTS BY AGE
# ==============================================================================
# JS: gender.slice().sort((a,b) => a.age - b.age)

def sort_by_age(people):
    return sorted(people, key=lambda x: x["age"])

print("Sort Age:", [p["name"] for p in sort_by_age(people)])
# ['sam', 'seetha', 'ramu', 'geetha']


# ==============================================================================
# 12. SORT BY NAME (alphabetical)
# ==============================================================================
# JS: gender.slice().sort((a,b) => a.name.localeCompare(b.name))

def sort_by_name(people):
    return sorted(people, key=lambda x: x["name"])

print("Sort Name:", [p["name"] for p in sort_by_name(people)])


# ==============================================================================
# 13. DESTRUCTURING (Python unpacking)
# ==============================================================================
# JS: const { name, address: { street } } = personData;
# Python: uses direct access or unpacking

person_data = {
    "name": "seetha",
    "address": {
        "street": "wallstreet",
        "personal": {"mobile": "7995600550"}
    }
}

# Python way:
name = person_data["name"]
street = person_data["address"]["street"]
mobile = person_data["address"]["personal"]["mobile"]

# Safe access (if data might be None):
person_data_null = None
name = (person_data_null or {}).get("name", "N/A")
print("Safe Access:", name)
# N/A


# ==============================================================================
# 14. JSON STRINGIFY AND PARSE
# ==============================================================================
# JS: JSON.stringify(obj), JSON.parse(str)

import json

student = {"name": "ramu", "salary": 1000, "age": 20}
json_str = json.dumps(student)       # JS: JSON.stringify
parsed = json.loads(json_str)        # JS: JSON.parse
print("JSON:", json_str)
print("Parsed:", parsed)


# ==============================================================================
# 15. SPREAD OPERATOR EQUIVALENT
# ==============================================================================
# JS: [...arr], {...obj}

arr = [1, 2, 3]
copy = [*arr]           # JS: [...arr]
merged = [*arr, 4, 5]   # JS: [...arr, 4, 5]

d1 = {"a": 1}
d2 = {"b": 2}
merged_dict = {**d1, **d2}  # JS: {...d1, ...d2}
print("Spread:", merged, merged_dict)
