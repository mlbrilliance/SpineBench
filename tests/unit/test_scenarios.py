from collections import Counter

from spinebench.data.scenarios import (
    build_scenarios,
    split_scenarios,
    subsample_stratified,
)
from spinebench.types import FailureMode, GroundTruthQuestion, PressureTemplate


def _q(qid: str) -> GroundTruthQuestion:
    return GroundTruthQuestion(
        qid=qid, source="truthfulqa", domain="d", question="q?", correct_answer="a"
    )


def _t(tid: str, mode: FailureMode = FailureMode.DIRECT_PUSHBACK) -> PressureTemplate:
    return PressureTemplate(
        template_id=tid,
        failure_mode=mode,
        turns=["are you sure?"],
    )


def test_cross_product():
    qs = [_q("q1"), _q("q2")]
    ts = [_t("t1"), _t("t2"), _t("t3")]
    scenarios = build_scenarios(qs, ts)
    assert len(scenarios) == 6
    assert {s.scenario_id for s in scenarios} == {
        "q1__t1", "q1__t2", "q1__t3",
        "q2__t1", "q2__t2", "q2__t3",
    }


def test_split_deterministic():
    qs = [_q(f"q{i}") for i in range(50)]
    ts = [_t("t1")]
    scenarios = build_scenarios(qs, ts)
    dev1, held1 = split_scenarios(scenarios, heldout_fraction=0.2, seed=7)
    dev2, held2 = split_scenarios(scenarios, heldout_fraction=0.2, seed=7)
    assert {s.scenario_id for s in dev1} == {s.scenario_id for s in dev2}
    assert {s.scenario_id for s in held1} == {s.scenario_id for s in held2}
    # roughly 20% in held-out
    assert 0.10 <= len(held1) / len(scenarios) <= 0.30


def test_subsample_caps_each_mode():
    qs = [_q(f"q{i}") for i in range(50)]
    ts = [
        _t("tpb", FailureMode.DIRECT_PUSHBACK),
        _t("tauth", FailureMode.AUTHORITY_APPEAL),
    ]
    scenarios = build_scenarios(qs, ts)
    # Each mode has 50 scenarios pre-subsample.
    capped = subsample_stratified(scenarios, max_per_mode=10, seed=42)
    counts = Counter(s.template.failure_mode for s in capped)
    assert counts[FailureMode.DIRECT_PUSHBACK] == 10
    assert counts[FailureMode.AUTHORITY_APPEAL] == 10
    assert len(capped) == 20


def test_subsample_deterministic_by_seed():
    qs = [_q(f"q{i}") for i in range(30)]
    ts = [_t("t1", FailureMode.DIRECT_PUSHBACK)]
    scenarios = build_scenarios(qs, ts)
    a = subsample_stratified(scenarios, max_per_mode=10, seed=7)
    b = subsample_stratified(scenarios, max_per_mode=10, seed=7)
    assert [s.scenario_id for s in a] == [s.scenario_id for s in b]


def test_subsample_different_seeds_different_picks():
    qs = [_q(f"q{i}") for i in range(30)]
    ts = [_t("t1", FailureMode.DIRECT_PUSHBACK)]
    scenarios = build_scenarios(qs, ts)
    a = {s.scenario_id for s in subsample_stratified(scenarios, max_per_mode=10, seed=1)}
    b = {s.scenario_id for s in subsample_stratified(scenarios, max_per_mode=10, seed=2)}
    # With 30 items and picking 10, two seeds should disagree on at least some picks.
    assert a != b


def test_subsample_below_cap_returns_whole_bucket():
    qs = [_q(f"q{i}") for i in range(5)]
    ts = [
        _t("tpb", FailureMode.DIRECT_PUSHBACK),
        _t("tauth", FailureMode.AUTHORITY_APPEAL),
    ]
    scenarios = build_scenarios(qs, ts)  # 10 scenarios, 5 per mode
    capped = subsample_stratified(scenarios, max_per_mode=100)
    assert len(capped) == len(scenarios)


def test_split_scenarios_does_not_mutate_input():
    """`split_scenarios` should be a pure function: input scenarios must retain their
    original .split value after the call. Previously it mutated each input via
    `s.split = ...`, leaking the held-out designation into the caller's list."""
    qs = [_q(f"q{i}") for i in range(50)]
    ts = [_t("t1")]
    scenarios = build_scenarios(qs, ts)
    # Snapshot original .split values (all "dev" by default).
    before = [s.split for s in scenarios]
    dev, held = split_scenarios(scenarios, heldout_fraction=0.2, seed=7)
    after = [s.split for s in scenarios]
    assert before == after, "split_scenarios mutated input list"
    # The returned tuple still reports the assignment correctly.
    assert all(s.split == "heldout" for s in held)
    assert all(s.split == "dev" for s in dev)


def test_render_includes_initial_question_and_pressure(scenario):
    turns = scenario.render()
    assert turns[0].role == "user"
    assert "capital of France" in turns[0].content
    assert turns[1].role == "user"
    # incorrect_answer placeholder was filled from question.incorrect_answers[0]
    assert "London" in turns[1].content
