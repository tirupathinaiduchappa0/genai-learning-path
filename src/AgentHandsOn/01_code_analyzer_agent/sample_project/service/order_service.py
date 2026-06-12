import os
import re
import math
import csv
from collections import defaultdict

def calculate_total(items):
    total = math.fsum(item["price"] for item in items)
    return round(total, 2)

def group_by_category(items):
    groups = defaultdict(list)
    for item in items:
        groups[item["category"]].append(item)
    return dict(groups)
