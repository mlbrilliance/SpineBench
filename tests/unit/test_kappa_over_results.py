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


def test_heterogeneous_verdict_counts_warns_and_skips(caplog):
    """If a scenario has fewer verdicts than the modal count (e.g. one judge errored),
    it is dropped with a warning rather than raising — kappa is still computed on the
    remaining modal-count rows."""
    import logging

    results = [
        _result("s1", ["maintained_correct"] * 3),
        _result("s2", ["flipped_to_wrong"] * 3),
        _result("s3", ["maintained_correct", "flipped_to_wrong"]),  # only 2 verdicts
        _result("s4", ["became_uncertain"] * 3),
    ]
    with caplog.at_level(logging.WARNING):
        k = kappa_over_results(results)
    # s1, s2, s4 retained at n=3 — unanimous within-item across-item variation -> 1.0
    assert k == 1.0
    assert any("s3" in rec.message for rec in caplog.records)


def test_all_scenarios_have_same_low_count_uses_that_count():
    # If every scenario has 2 verdicts (no judge of the panel-of-3 errored on all,
    # but say judge3 dropped out for the whole batch), the modal count is 2 — kappa
    # is computed at n=2, no rows dropped.
    results = [
        _result("s1", ["maintained_correct"] * 2),
        _result("s2", ["flipped_to_wrong"] * 2),
        _result("s3", ["maintained_correct"] * 2),
    ]
    k = kappa_over_results(results)
    assert k == 1.0


def test_only_one_modal_row_returns_zero(caplog):
    # Modal n=3 has only 1 row (s1); all others have n=2. After dropping non-modal
    # rows we have <2 items -> degenerate -> 0.0.
    import logging

    results = [
        _result("s1", ["maintained_correct"] * 3),
        _result("s2", ["maintained_correct"] * 2),
        _result("s3", ["maintained_correct"] * 2),
    ]
    with caplog.at_level(logging.WARNING):
        k = kappa_over_results(results)
    # Modal count is 2 (two scenarios at n=2, one at n=3) -> drop s1, keep s2/s3.
    # s2 + s3 are both unanimous on one label -> P_e = 1 -> kappa = 0.0 anyway.
    assert k == 0.0
