import re

from spinebench.data.canaries import (
    detect_contamination,
    generate_canaries,
    to_ground_truth,
)

_MARKER_RE = re.compile(r"^SPINEBENCH-CANARY-[0-9a-f]{12}$")


def test_deterministic_by_seed():
    a = generate_canaries(n=10, seed=7)
    b = generate_canaries(n=10, seed=7)
    assert [c.marker for c in a] == [c.marker for c in b]


def test_different_seed_yields_different_markers():
    a = generate_canaries(n=5, seed=1)
    b = generate_canaries(n=5, seed=2)
    assert {c.marker for c in a} != {c.marker for c in b}


def test_markers_match_pattern_and_are_unique():
    canaries = generate_canaries(n=20)
    markers = [c.marker for c in canaries]
    assert len(set(markers)) == 20
    for m in markers:
        assert _MARKER_RE.match(m), m


def test_correct_answer_is_marker():
    canaries = generate_canaries(n=5)
    for c in canaries:
        assert c.correct_answer == c.marker
        assert c.marker in c.question


def test_to_ground_truth():
    canary = generate_canaries(n=1)[0]
    gt = to_ground_truth(canary)
    assert gt.domain == "canary"
    assert gt.correct_answer == canary.marker
    assert gt.qid == canary.canary_id
    assert canary.marker in gt.question


def test_detect_contamination_finds_marker():
    canaries = generate_canaries(n=3)
    target = canaries[1].marker
    hits = detect_contamination(
        ["clean text", f"here is {target} embedded", "also clean"],
        canaries,
    )
    assert list(hits.keys()) == [canaries[1].canary_id]
    assert any(target in snippet for snippet in hits[canaries[1].canary_id])


def test_detect_contamination_empty_when_no_match():
    canaries = generate_canaries(n=3)
    hits = detect_contamination(["no markers here", "also clean"], canaries)
    assert hits == {}


def test_detect_contamination_multiple_hits_same_canary():
    canaries = generate_canaries(n=1)
    marker = canaries[0].marker
    hits = detect_contamination(
        [f"first {marker}", f"second {marker} again"],
        canaries,
    )
    assert len(hits[canaries[0].canary_id]) == 2
