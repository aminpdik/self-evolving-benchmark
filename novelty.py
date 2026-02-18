"""Heuristic novelty detection utilities.

Implements simple normalization, shingling, and Jaccard similarity to detect near-duplicate
questions and retrieve the most similar existing examples."""


from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple


def normalize_text(s: str) -> str:
    """Normalize text for similarity comparisons.

Args:
    s: Raw input text.

Returns:
    A normalized string (lowercased, punctuation removed, and whitespace-collapsed).

How it works:
    Uses regex to replace non-alphanumeric characters with spaces and collapses repeated spaces."""

    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def shingles(s: str, k: int = 5) -> List[str]:
    """Create word-level k-shingles for a text.

Args:
    s: Input text.
    k: Number of consecutive words per shingle.

Returns:
    List of shingles (each shingle is a 'k words joined by spaces' string).

How it works:
    Normalizes the text, splits into tokens, then slides a window of size k over tokens.
    For short texts (<k tokens), returns a single shingle containing the whole text."""
    s = normalize_text(s)
    tokens = s.split()
    if len(tokens) < k:
        return [" ".join(tokens)] if tokens else [""]
    return [" ".join(tokens[i:i+k]) for i in range(len(tokens) - k + 1)]


def jaccard(a: List[str], b: List[str]) -> float:
    """Compute Jaccard similarity between two shingle lists.

Args:
    a: Shingles from text A.
    b: Shingles from text B.

Returns:
    Similarity in [0, 1], where 1 means identical sets.

How it works:
    Converts both lists to sets and returns |A∩B| / |A∪B| (with edge-case handling)."""

    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


@dataclass
class SimilarityHit:
    """A similarity search result.

Attributes:
    idx: Index of the matched question in the internal list.
    score: Jaccard similarity score in [0, 1].
    text: The matched question text."""

    idx: int
    score: float
    text: str


class NoveltyFilter:
    """Heuristic novelty filter + retrieval of similar examples for the judge."""

    def __init__(self, k_shingle: int = 5):
        """Initialize the novelty filter.

Args:
    k_shingle: Shingle size (in words) used for Jaccard similarity.

Returns:
    None.

How it works:
    Keeps an in-memory list of previously seen question texts and a cached list of their shingles
    for faster similarity checks."""

        self.k_shingle = k_shingle
        self.questions: List[str] = []
        self.shingles_cache: List[List[str]] = []

    def add(self, question_text: str) -> None:
        """Add a question text to the novelty index.

Args:
    question_text: The full text to index (often instruction + '\n' + user_input).

Returns:
    None.

How it works:
    Stores the raw text and its precomputed shingles in parallel arrays."""

        self.questions.append(question_text)
        self.shingles_cache.append(shingles(question_text, self.k_shingle))

    def top_k_similar(self, candidate: str, k: int = 5) -> List[SimilarityHit]:
        """Retrieve the top-k most similar existing questions to a candidate.

Args:
    candidate: Candidate question text to compare.
    k: Number of most similar items to return.

Returns:
    A list of SimilarityHit objects sorted by similarity descending.

How it works:
    Computes shingles for candidate, then computes Jaccard similarity against each cached shingle set,
    sorts, and returns the top k results."""

        cand_sh = shingles(candidate, self.k_shingle)
        hits: List[SimilarityHit] = []
        for i, qsh in enumerate(self.shingles_cache):
            sim = jaccard(cand_sh, qsh)
            hits.append(SimilarityHit(idx=i, score=sim, text=self.questions[i]))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]

    def is_novel_heuristic(self, candidate: str, threshold: float = 0.70) -> Tuple[bool, float]:
        """Heuristically decide if a candidate question is novel.

Args:
    candidate: Candidate question text.
    threshold: Maximum allowed similarity to consider the candidate novel.

Returns:
    (is_novel, max_similarity) where:
      - is_novel is True if the best match similarity is below threshold.
      - max_similarity is the highest observed Jaccard similarity.

How it works:
    Compares the candidate to the most similar existing question and checks if the similarity
    crosses the threshold."""

        if not self.questions:
            return True, 0.0
        hits = self.top_k_similar(candidate, k=1)
        max_sim = hits[0].score if hits else 0.0
        return (max_sim < threshold), max_sim