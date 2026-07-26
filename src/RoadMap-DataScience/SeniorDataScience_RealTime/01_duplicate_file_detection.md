# Lesson 01 — Detecting Duplicate File Content (hashlib)

## 1. The Question (interview framing)

> "Write a program that finds duplicate files in a directory tree based on their **content**, not their names. Use hashing. Make it work for large files and large numbers of files."

This tests: hashing fundamentals, memory-aware I/O, algorithmic optimization, and edge-case thinking (collisions, symlinks, empty files).

---

## 2. Theory — Why hashing?

A **hash function** maps arbitrary bytes to a fixed-size digest. Key properties we exploit:

- **Deterministic:** same bytes → same digest, always.
- **Avalanche effect:** one flipped bit → completely different digest.
- **Fixed size:** a 10 GB file and a 1 KB file both produce, say, a 256-bit SHA-256 digest.

So instead of comparing every file against every other file (O(n²) full-content comparisons), we compute one digest per file (O(n)) and group by digest. Files with identical digests are (almost certainly) identical.

### MD5 vs SHA-256

- **MD5** (128-bit) is faster and fine for *dedup* (we're not defending against attackers), but it's cryptographically broken (collisions can be crafted).
- **SHA-256** (256-bit) is the safe default. Collision probability by accident is astronomically small.

For pure deduplication where no adversary is crafting files, MD5 is acceptable and faster. If files could be adversarial, use SHA-256.

---

## 3. The optimization ladder

Naive: hash every file fully, group by hash. Works, but hashes files that are obviously unique.

Better pipeline (each stage cheaper than the next filters candidates):

1. **Group by size.** Files with different sizes cannot be identical. Sizes come from the filesystem metadata (`os.path.getsize`) — essentially free.
2. **Group by a partial hash** (first few KB). Cheap; separates most same-size-but-different files.
3. **Full hash** only for the survivors.
4. **(Optional) byte-for-byte compare** on full-hash matches for absolute certainty (defeats even hash collisions).

This is exactly how tools like `fdupes`/`rdfind` work.

---

## 4. Hands-on code

### 4.1 Core: chunked hashing (memory-safe)

```python
import hashlib

def file_hash(path, algo="sha256", chunk_size=8192):
    """Hash a file's content without loading it fully into memory."""
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk_size), b""):
            h.update(block)
    return h.hexdigest()
```

`iter(callable, sentinel)` repeatedly calls `f.read(chunk_size)` until it returns `b""` (EOF). This streams the file so RAM stays flat even for multi-GB files.

### 4.2 Naive grouping

```python
from collections import defaultdict

def find_duplicates_simple(paths):
    by_hash = defaultdict(list)
    for p in paths:
        by_hash[file_hash(p)].append(p)
    return {h: ps for h, ps in by_hash.items() if len(ps) > 1}
```

### 4.3 Optimized pipeline (size → partial → full)

```python
import os
from collections import defaultdict

def partial_hash(path, algo="sha256", peek=4096):
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        h.update(f.read(peek))
    return h.hexdigest()

def find_duplicates(root):
    # Stage 1: group by size
    by_size = defaultdict(list)
    for dirpath, _, files in os.walk(root):
        for name in files:
            p = os.path.join(dirpath, name)
            try:
                by_size[os.path.getsize(p)].append(p)
            except OSError:
                continue  # broken symlink, permission error, etc.

    duplicates = defaultdict(list)
    for size, group in by_size.items():
        if len(group) < 2:
            continue  # unique size => unique file, skip hashing

        # Stage 2: group by partial hash
        by_partial = defaultdict(list)
        for p in group:
            by_partial[partial_hash(p)].append(p)

        # Stage 3: full hash only for partial-hash collisions
        for partial_group in by_partial.values():
            if len(partial_group) < 2:
                continue
            by_full = defaultdict(list)
            for p in partial_group:
                by_full[file_hash(p)].append(p)
            for full_hash, matches in by_full.items():
                if len(matches) > 1:
                    duplicates[full_hash].extend(matches)

    return dict(duplicates)


if __name__ == "__main__":
    for h, paths in find_duplicates(".").items():
        print(f"\nDuplicate set ({h[:12]}...):")
        for p in paths:
            print(f"  {p}")
```

### 4.4 Collision-proof variant

```python
import filecmp

def confirm_identical(paths):
    """Byte-compare a group that shares a hash, to defeat hash collisions."""
    reference = paths[0]
    return [p for p in paths[1:] if filecmp.cmp(reference, p, shallow=False)]
```

---

## 5. Real-time / production scenarios

- **Photo/asset de-duplication** in a storage backend before upload to S3 — save cost by storing one copy and referencing it (content-addressable storage, exactly how Git stores blobs by SHA).
- **Data pipeline dedup:** incoming files land in a bucket; hash-on-arrival prevents reprocessing the same payload.
- **ML dataset hygiene:** duplicate training images/rows inflate metrics and cause train/test leakage; content-hash dedup is a standard cleaning step.
- **Scale:** for millions of files, parallelize hashing across processes (CPU/IO bound mix), and persist hashes in a DB so re-scans are incremental (only hash files whose size+mtime changed).

---

## 6. Edge cases to mention

- **Empty files** all hash the same → they're "duplicates," decide if that matters.
- **Symlinks / hardlinks** → you may double-count the same inode; check `os.stat().st_ino`.
- **Permission / IO errors** → wrap in try/except, don't crash the whole scan.
- **Very large file counts** → store `(size, mtime, hash)` cache to avoid rehashing unchanged files.

---

## 7. What to say in the interview (script)

"I hash content, not names, because names lie. I stream the file in chunks so memory stays constant. To avoid hashing files that are obviously unique, I filter by size first, then a cheap partial hash, then a full hash only on survivors — that's the fdupes strategy. SHA-256 by default; if I need absolute certainty I do a final byte compare, since hashing alone is probabilistic. At scale I cache hashes keyed on size+mtime so re-scans are incremental, and I parallelize across processes."
