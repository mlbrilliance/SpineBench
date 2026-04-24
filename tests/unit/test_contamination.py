import json

from spinebench.data.contamination import (
    ContaminationIndex,
    audit_ground_truth,
    jaccard_overlap,
    ngram_shingles,
)
from spinebench.types import GroundTruthQuestion


def test_ngram_shingles_basic():
    shingles = ngram_shingles("the quick brown fox", n=2)
    assert shingles == {
        ("the", "quick"),
        ("quick", "brown"),
        ("brown", "fox"),
    }


def test_ngram_shingles_strips_punctuation_and_lowercases():
    shingles = ngram_shingles("Hello, World! Foo.", n=1)
    assert shingles == {("hello",), ("world",), ("foo",)}


def test_ngram_shingles_empty():
    assert ngram_shingles("", n=3) == set()


def test_ngram_shingles_fewer_tokens_than_n():
    shingles = ngram_shingles("just two", n=5)
    assert shingles == {("just", "two")}


def test_jaccard_both_empty():
    assert jaccard_overlap(set(), set()) == 0.0


def test_jaccard_identical():
    s = {("a",), ("b",)}
    assert jaccard_overlap(s, s) == 1.0


def test_jaccard_disjoint():
    a = {("a",)}
    b = {("b",)}
    assert jaccard_overlap(a, b) == 0.0


def test_jaccard_partial():
    a = {("a",), ("b",), ("c",)}
    b = {("b",), ("c",), ("d",)}
    # intersection 2, union 4
    assert jaccard_overlap(a, b) == 0.5


def test_index_overlap_full_hit():
    idx = ContaminationIndex(ngram_shingles("the quick brown fox", n=2))
    # Same sentence: all shingles present in index
    assert idx.overlap("the quick brown fox", n=2) == 1.0


def test_index_overlap_partial():
    idx = ContaminationIndex(ngram_shingles("the quick brown fox", n=2))
    # "the quick" present, "red car" is not
    score = idx.overlap("the quick red car", n=2)
    assert 0.0 < score < 1.0


def test_index_overlap_empty_text():
    idx = ContaminationIndex(ngram_shingles("some reference text here", n=2))
    assert idx.overlap("", n=2) == 0.0


def test_index_from_jsonl(tmp_path):
    path = tmp_path / "corpus.jsonl"
    docs = [
        {"text": "the quick brown fox jumps"},
        {"text": "over the lazy dog"},
        {"text": "another sentence entirely"},
    ]
    path.write_text("\n".join(json.dumps(d) for d in docs), encoding="utf-8")

    idx = ContaminationIndex.from_jsonl(path, n=2)
    # Every bigram in every doc should be in the index
    for doc in docs:
        assert idx.overlap(doc["text"], n=2) == 1.0
    # Text not in any doc has zero overlap
    assert idx.overlap("completely unrelated vocabulary here", n=2) == 0.0


def test_index_from_jsonl_respects_limit(tmp_path):
    path = tmp_path / "corpus.jsonl"
    path.write_text(
        "\n".join(json.dumps({"text": f"doc number {i} with tokens"}) for i in range(10)),
        encoding="utf-8",
    )
    idx = ContaminationIndex.from_jsonl(path, n=2, limit=2)
    # Only the first 2 docs should be indexed.
    assert idx.overlap("doc number 0 with tokens", n=2) > 0
    # We can't rely on strict negatives here (tokens overlap across docs), but
    # we can at least check that the index is smaller with a limit.
    full = ContaminationIndex.from_jsonl(path, n=2)
    assert len(idx) < len(full)


def test_audit_ground_truth_flags_matches():
    index = ContaminationIndex(
        ngram_shingles("the capital of france is paris and it is famous", n=4)
    )
    questions = [
        GroundTruthQuestion(
            qid="q1",
            source="truthfulqa",
            domain="geo",
            question="the capital of france is paris and it is famous",
            correct_answer="paris",
        ),
        GroundTruthQuestion(
            qid="q2",
            source="truthfulqa",
            domain="geo",
            question="what is quantum entanglement exactly",
            correct_answer="correlation",
        ),
    ]
    flagged = audit_ground_truth(questions, index, threshold=0.5, n=4)
    assert len(flagged) == 1
    assert flagged[0][0].qid == "q1"
    assert flagged[0][1] >= 0.5


def test_audit_sorts_by_score_descending():
    text_a = "the cat sat on the mat in the sun today"
    text_b = "the cat sat on the mat today yes"
    index = ContaminationIndex(ngram_shingles(text_a, n=3))
    questions = [
        GroundTruthQuestion(
            qid="low",
            source="truthfulqa",
            domain="x",
            question=text_b,
            correct_answer="",
        ),
        GroundTruthQuestion(
            qid="high",
            source="truthfulqa",
            domain="x",
            question=text_a,
            correct_answer="",
        ),
    ]
    flagged = audit_ground_truth(questions, index, threshold=0.0, n=3)
    assert [q.qid for q, _ in flagged] == ["high", "low"]
    assert flagged[0][1] >= flagged[1][1]
