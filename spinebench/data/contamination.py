"""N-gram contamination detector for ground-truth questions.

Checks whether our benchmark questions appear verbatim in common pretraining corpora
(Pile / RedPajama / Dolma). If they do, we'd grade models on memorization, not reasoning.

Pure-Python set-intersection is fine at our scale (~500 questions x a few-MB reference).
Swap for MinHash/Bloom if we ever scale the reference into the GB range.
"""

from __future__ import annotations

import json
import string
from pathlib import Path

from spinebench.types import GroundTruthQuestion

_PUNCTUATION_TABLE = str.maketrans("", "", string.punctuation)

Shingle = tuple[str, ...]


def ngram_shingles(text: str, n: int = 8) -> set[Shingle]:
    """Lowercase, strip punctuation, whitespace-tokenize, return n-gram shingle set.

    If the text has fewer than n tokens, returns a single shingle padded by its own length.
    Empty text returns an empty set.
    """
    cleaned = text.lower().translate(_PUNCTUATION_TABLE)
    tokens = cleaned.split()
    if not tokens:
        return set()
    if len(tokens) < n:
        return {tuple(tokens)}
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def jaccard_overlap(a: set[Shingle], b: set[Shingle]) -> float:
    """Jaccard similarity; 0.0 when both are empty or when union is empty."""
    if not a and not b:
        return 0.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


class ContaminationIndex:
    """Holds a set of shingles from a reference corpus; answers `overlap(text)` queries."""

    def __init__(self, shingles: set[Shingle]) -> None:
        self._shingles = shingles

    def __len__(self) -> int:
        return len(self._shingles)

    def overlap(self, text: str, n: int = 8) -> float:
        """Fraction of `text`'s n-gram shingles that also appear in the reference."""
        text_shingles = ngram_shingles(text, n)
        if not text_shingles:
            return 0.0
        hits = len(text_shingles & self._shingles)
        return hits / len(text_shingles)

    @classmethod
    def from_jsonl(
        cls,
        path: Path,
        text_field: str = "text",
        n: int = 8,
        limit: int | None = None,
    ) -> ContaminationIndex:
        """Build an index from a JSONL corpus dump (standard Pile/Dolma/RedPajama format)."""
        shingles: set[Shingle] = set()
        count = 0
        with Path(path).open(encoding="utf-8") as f:
            for line in f:
                if limit is not None and count >= limit:
                    break
                doc = json.loads(line)
                text = doc.get(text_field, "")
                if text:
                    shingles.update(ngram_shingles(text, n))
                count += 1
        return cls(shingles)


def audit_ground_truth(
    questions: list[GroundTruthQuestion],
    index: ContaminationIndex,
    *,
    threshold: float = 0.8,
    n: int = 8,
) -> list[tuple[GroundTruthQuestion, float]]:
    """Return questions whose n-gram overlap with the reference exceeds `threshold`,
    sorted by score descending. These are the candidates for exclusion."""
    flagged: list[tuple[GroundTruthQuestion, float]] = []
    for q in questions:
        score = index.overlap(q.question, n)
        if score >= threshold:
            flagged.append((q, score))
    flagged.sort(key=lambda x: x[1], reverse=True)
    return flagged
