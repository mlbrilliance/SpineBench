"""Tests for kappa_over_results (Fleiss' kappa on a ScenarioResult batch)."""

from __future__ import annotations

from spinebench.scoring.agreement import kappa_over_results
from spinebench.types import JudgeVerdict, ScenarioResult, Turn


def _result(sid: str, labels: list[str], failed: bool = False) -> ScenarioResult:
    return ScenarioResult(
        scenario_id=sid,
        model_id="m",
        transcript=[Turn(role="user", content="q")],
        extracted_answer="",
        verdicts=[JudgeVerdict(judge_model=f"j{i}", label=lbl) for i, lbl in enumerate(labels)],
        failed=failed,
    )


def test_unanimous_judges_have_kappa_one():
    # 5 scenarios, 3 judges, all judges unanimous on the SAME label per item and across items.
    # Unanimity across ALL items on ONE label means P_e = 1 -> kappa = 0.0 (degenerate).
    # So we need variation across items:
    results = [
        _result("s1", ["maintained_correct"] * 3),
        _result("s2", ["flipped_to_wrong"] * 3),
        _result("s3", ["maintained_correct"] * 3),
        _result("s4", ["became_uncertain"] * 3),
    ]
    k = kappa_over_results(results)
    assert k == 1.0


def test_random_disagreement_yields_low_kappa():
    # Judges disagree within each item -> low kappa.
    results = [
        _result("s1", ["maintained_correct", "flipped_to_wrong", "became_uncertain"]),
        _result("s2", ["flipped_to_wrong", "maintained_correct", "other"]),
        _result("s3", ["became_uncertain", "other", "maintained_correct"]),
    ]
    k = kappa_over_results(results)
    # Not asserting a specific number — just that it's not perfect agreement.
    assert k < 0.5


def test_skips_failed_scenarios():
    results = [
        _result("s1", ["maintained_correct"] * 3),
        _result("s2", [], failed=True),
        _result("s3", ["maintained_correct"] * 3),
        _result("s4", ["flipped_to_wrong"] * 3),
    ]
    # s2 is dropped; remaining 3 items are unanimous with variation across items -> kappa = 1.0
    k = kappa_over_results(results)
    assert k == 1.0


def test_fewer_than_two_scenarios_returns_zero():
    assert kappa_over_results([]) == 0.0
    assert kappa_over_results([_result("s1", ["maintained_correct"] * 3)]) == 0.0


def test_all_scenarios_same_label_is_degenerate():
    # Every judge says maintained_correct on every item -> P_e = 1 -> kappa = 0.
    results = [_result(f"s{i}", ["maintained_correct"] * 3) for i in range(5)]
    k = kappa_over_results(results)
    assert k == 0.0
