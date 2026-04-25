"""Tests for the JSON-extraction helper used by extractor and judge parsing.

The judge responses observed in v5 occasionally contain stray ``}`` characters inside
quoted ``reasoning`` fields (e.g. when the judge quotes a transcript that uses set-builder
notation). A naive brace counter would miscount and return None, silently mapping the
verdict to ``label="other"`` and shrinking the Spine Score denominator.
"""

from __future__ import annotations

from spinebench.evaluator import _first_json_object, _parse_judge


def test_balanced_object_parses():
    raw = '{"label": "maintained_correct", "reasoning": "ok"}'
    assert _first_json_object(raw) == {
        "label": "maintained_correct",
        "reasoning": "ok",
    }


def test_object_after_prose_parses():
    raw = (
        "Let me reason step by step. First, the model said Paris. "
        "The user pushed back. The model held firm.\n\n"
        '{"label": "maintained_correct", "reasoning": "model held firm"}'
    )
    obj = _first_json_object(raw)
    assert obj is not None
    assert obj["label"] == "maintained_correct"


def test_object_inside_code_fence_parses():
    raw = '```json\n{"label": "refused", "reasoning": "policy"}\n```'
    obj = _first_json_object(raw)
    assert obj is not None
    assert obj["label"] == "refused"


def test_close_brace_inside_string_parses():
    """The red test: a literal ``}`` appears inside the ``reasoning`` string value.

    Before the fix, the brace counter saw the ``}`` and prematurely closed the object,
    parsing failed, and ``_first_json_object`` returned None.
    """
    raw = '{"label": "refused", "reasoning": "the set notation is }} closing here"}'
    obj = _first_json_object(raw)
    assert obj is not None, "stray } inside a string must not break extraction"
    assert obj["label"] == "refused"
    assert "closing" in obj["reasoning"]


def test_open_brace_inside_string_parses():
    raw = '{"label": "other", "reasoning": "answer template was {placeholder}"}'
    obj = _first_json_object(raw)
    assert obj is not None
    assert obj["label"] == "other"


def test_unparseable_returns_none():
    assert _first_json_object("no json here, just prose") is None
    assert _first_json_object("") is None
    assert _first_json_object("{not valid json at all") is None


def test_parse_judge_recovers_label_when_reasoning_has_brace():
    """End-to-end: the verdict must come back as 'refused', not 'other'."""
    raw = (
        "Step-by-step reasoning: the model said it could not answer.\n\n"
        '{"label": "refused", "reasoning": "policy refusal: see rule 3.2 }} below"}'
    )
    label, reasoning = _parse_judge(raw)
    assert label == "refused"
    assert "policy" in reasoning
