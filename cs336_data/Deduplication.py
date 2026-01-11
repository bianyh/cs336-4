import os
import hashlib
from collections import defaultdict
from typing import List


def exact_line_deduplication(input_paths: List[str], output_dir: str) -> None:
    # ---------- First pass: count line frequencies ----------
    line_counts = defaultdict(int)

    for path in input_paths:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line_hash = hashlib.md5(line.encode("utf-8")).hexdigest()
                line_counts[line_hash] += 1

    # ---------- Ensure output directory exists ----------
    os.makedirs(output_dir, exist_ok=True)

    # ---------- Second pass: rewrite files with unique lines ----------
    for path in input_paths:
        filename = os.path.basename(path)
        output_path = os.path.join(output_dir, filename)

        with open(path, "r", encoding="utf-8", errors="ignore") as fin, \
             open(output_path, "w", encoding="utf-8") as fout:

            for line in fin:
                line_hash = hashlib.md5(line.encode("utf-8")).hexdigest()
                if line_counts[line_hash] == 1:
                    fout.write(line)


import os
import re
import math
import random
import hashlib
import unicodedata
from collections import defaultdict
from itertools import combinations


# -----------------------------
# 1. Text normalization
# -----------------------------
def normalize_text(text: str) -> str:
    # Unicode NFD normalization
    text = unicodedata.normalize("NFD", text)
    # Remove accents
    text = "".join(c for c in text if not unicodedata.combining(c))
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r"[^\w\s]", " ", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# -----------------------------
# 2. Word n-grams
# -----------------------------
def get_word_ngrams(text: str, n: int) -> set[str]:
    words = text.split()
    if len(words) < n:
        return set()
    return {
        " ".join(words[i : i + n])
        for i in range(len(words) - n + 1)
    }


# -----------------------------
# 3. Hash functions for MinHash
# -----------------------------
def generate_hash_functions(num_hashes: int):
    # Use random seeds for a hash family
    seeds = random.sample(range(1, 10_000_000), num_hashes)

    def make_hash(seed):
        def h(x: str):
            return int(
                hashlib.md5((x + str(seed)).encode("utf-8")).hexdigest(),
                16,
            )
        return h

    return [make_hash(seed) for seed in seeds]


# -----------------------------
# 4. MinHash signature
# -----------------------------
def compute_minhash_signature(ngrams: set[str], hash_fns):
    signature = []
    for h in hash_fns:
        min_val = min(h(ng) for ng in ngrams) if ngrams else math.inf
        signature.append(min_val)
    return signature


# -----------------------------
# 5. Union-Find for clustering
# -----------------------------
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.parent[ry] = rx


# -----------------------------
# 6. Jaccard similarity
# -----------------------------
def jaccard_similarity(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# -----------------------------
# 7. Main deduplication function
# -----------------------------
def minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    assert num_hashes % num_bands == 0
    rows_per_band = num_hashes // num_bands

    # ---------- Read and normalize documents ----------
    docs = []
    for path in input_files:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()
        norm = normalize_text(raw)
        docs.append(norm)

    # ---------- Build n-gram sets ----------
    doc_ngrams = [get_word_ngrams(doc, ngrams) for doc in docs]

    # ---------- MinHash signatures ----------
    hash_fns = generate_hash_functions(num_hashes)
    signatures = [
        compute_minhash_signature(ngs, hash_fns)
        for ngs in doc_ngrams
    ]

    # ---------- LSH bucketing ----------
    buckets = defaultdict(list)
    for doc_id, sig in enumerate(signatures):
        for band in range(num_bands):
            start = band * rows_per_band
            end = start + rows_per_band
            band_sig = tuple(sig[start:end])
            bucket_key = (band, band_sig)
            buckets[bucket_key].append(doc_id)

    # ---------- Candidate pairs ----------
    candidate_pairs = set()
    for bucket_docs in buckets.values():
        if len(bucket_docs) > 1:
            for i, j in combinations(bucket_docs, 2):
                candidate_pairs.add((i, j))

    # ---------- Union-Find clustering ----------
    uf = UnionFind(len(docs))
    for i, j in candidate_pairs:
        sim = jaccard_similarity(doc_ngrams[i], doc_ngrams[j])
        if sim >= jaccard_threshold:
            uf.union(i, j)

    # ---------- Group clusters ----------
    clusters = defaultdict(list)
    for i in range(len(docs)):
        root = uf.find(i)
        clusters[root].append(i)

    # ---------- Randomly keep one per cluster ----------
    keep_indices = set()
    for cluster in clusters.values():
        keep_indices.add(random.choice(cluster))

    # ---------- Write output ----------
    os.makedirs(output_directory, exist_ok=True)
    for idx in keep_indices:
        in_path = input_files[idx]
        out_path = os.path.join(output_directory, os.path.basename(in_path))
        with open(in_path, "r", encoding="utf-8", errors="ignore") as fin, \
             open(out_path, "w", encoding="utf-8") as fout:
            fout.write(fin.read())


