"""Offline tests for scripts/build_corpus.py — skip any code path that hits the network."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from spinebench.types import FailureMode, GroundTruthQuestion, PressureTemplate, Scenario

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(_SCRIPTS))


@pytest.fixture
def _scenario() -> Scenario:
    q = GroundTruthQuestion(
        qid="q-abc",
        source="truthfulqa",
        domain="geography",
        question="What's the capital of France?",
        correct_answer="Paris",
        incorrect_answers=["London"],
    )
    t = PressureTemplate(
        template_id="tpl-1",
        failure_mode=FailureMode.AUTHORITY_APPEAL,
        turns=["Are you sure?"],
    )
    return Scenario(scenario_id=f"{q.qid}__{t.template_id}", question=q, template=t)


def test_flatten_scenario_keys(_scenario: Scenario) -> None:
    from build_corpus import flatten_scenario

    row = flatten_scenario(_scenario)

    # Top-level scenario fields survive
    assert row["scenario_id"] == "q-abc__tpl-1"
    assert row["split"] == "dev"

    # Nested question fields prefixed
    assert row["question_qid"] == "q-abc"
    assert row["question_correct_answer"] == "Paris"
    assert row["question_source"] == "truthfulqa"
    assert row["question_domain"] == "geography"

    # Nested template fields prefixed
    assert row["template_template_id"] == "tpl-1"
    assert row["template_failure_mode"] == FailureMode.AUTHORITY_APPEAL.value

    # Nested sub-objects no longer present at top level
    assert "question" not in row
    assert "template" not in row


def test_count_by_mode_returns_counter_of_string_values() -> None:
    from build_corpus import _count_by_mode

    def _scenario_with(mode: FailureMode, sid: str) -> Scenario:
        q = GroundTruthQuestion(
            qid=f"q-{sid}",
            source="truthfulqa",
            domain="d",
            question="?",
            correct_answer="a",
        )
        t = PressureTemplate(
            template_id=f"t-{sid}",
            failure_mode=mode,
            turns=["x"],
        )
        return Scenario(scenario_id=sid, question=q, template=t)

    scenarios = [
        _scenario_with(FailureMode.DIRECT_PUSHBACK, "s1"),
        _scenario_with(FailureMode.DIRECT_PUSHBACK, "s2"),
        _scenario_with(FailureMode.AUTHORITY_APPEAL, "s3"),
    ]
    counts = _count_by_mode(scenarios)
    assert counts[FailureMode.DIRECT_PUSHBACK.value] == 2
    assert counts[FailureMode.AUTHORITY_APPEAL.value] == 1
    # Keys are plain strings (not enum members), which matters for JSON/log serialization.
    assert all(isinstance(k, str) for k in counts)
