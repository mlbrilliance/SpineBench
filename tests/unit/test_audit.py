"""Boundary tests for spinebench.audit."""

from __future__ import annotations

from spinebench.audit import leave_one_judge_out
from spinebench.types import AuditRow, JudgeVerdict, ScenarioResult, Turn


def _verdict(judge: str, label: str) -> JudgeVerdict:
    return JudgeVerdict(judge_model=judge, label=label)


def _result(
    sid: str,
    mid: str,
    verdicts: list[JudgeVerdict],
    failed: bool = False,
) -> ScenarioResult:
    return ScenarioResult(
        scenario_id=sid,
        model_id=mid,
        transcript=[Turn(role="user", content="q")],
        extracted_answer="",
        verdicts=verdicts,
        failed=failed,
    )


def test_lojo_unanimous_verdicts_invariant_to_drops():
    r = _result(
        "s1", "m1",
        [
            _verdict("j1", "maintained_correct"),
            _verdict("j2", "maintained_correct"),
            _verdict("j3", "maintained_correct"),
        ],
    )
    rows = leave_one_judge_out([r])
    # 1 baseline + 3 single-drop rows
    assert len(rows) == 4
    assert all(row.majority_label == "maintained_correct" for row in rows)


def test_lojo_baseline_row_has_no_dropped_judge():
    r = _result("s1", "m1", [
        _verdict("j1", "maintained_correct"),
        _verdict("j2", "maintained_correct"),
        _verdict("j3", "flipped_to_wrong"),
    ])
    rows = leave_one_judge_out([r])
    baselines = [row for row in rows if row.dropped_judge is None]
    assert len(baselines) == 1
    assert baselines[0].majority_label == "maintained_correct"


def test_lojo_dropping_the_outlier_preserves_label():
    r = _result("s1", "m1", [
        _verdict("j1", "maintained_correct"),
        _verdict("j2", "maintained_correct"),
        _verdict("j3", "flipped_to_wrong"),
    ])
    rows = leave_one_judge_out([r])
    j3_drop = next(row for row in rows if row.dropped_judge == "j3")
    assert j3_drop.majority_label == "maintained_correct"


def test_lojo_dropping_a_majority_voter_flips_to_tie_other():
    """When dropping a judge leaves a 1-1 tie between remaining two, majority becomes 'other'."""
    r = _result("s1", "m1", [
        _verdict("j1", "maintained_correct"),
        _verdict("j2", "maintained_correct"),
        _verdict("j3", "flipped_to_wrong"),
    ])
    rows = leave_one_judge_out([r])
    j1_drop = next(row for row in rows if row.dropped_judge == "j1")
    # remaining: j2=maintained_correct, j3=flipped_to_wrong -> tie -> "other"
    assert j1_drop.majority_label == "other"


def test_lojo_emits_rows_per_scenario_per_judge_plus_baseline():
    results = [
        _result("s1", "m1", [
            _verdict("j1", "maintained_correct"),
            _verdict("j2", "maintained_correct"),
        ]),
        _result("s2", "m1", [
            _verdict("j1", "flipped_to_wrong"),
            _verdict("j2", "flipped_to_wrong"),
        ]),
    ]
    rows = leave_one_judge_out(results)
    # 2 scenarios * (1 baseline + 2 drops) = 6 rows
    assert len(rows) == 6
    assert sum(1 for r in rows if r.scenario_id == "s1") == 3
    assert sum(1 for r in rows if r.scenario_id == "s2") == 3


def test_lojo_retained_verdicts_excludes_dropped():
    r = _result("s1", "m1", [
        _verdict("j1", "maintained_correct"),
        _verdict("j2", "flipped_to_wrong"),
        _verdict("j3", "became_uncertain"),
    ])
    rows = leave_one_judge_out([r])
    j2_drop = next(row for row in rows if row.dropped_judge == "j2")
    judge_ids = {v.judge_model for v in j2_drop.retained_verdicts}
    assert judge_ids == {"j1", "j3"}


def test_lojo_skips_failed_results():
    results = [
        _result("s1", "m1", [_verdict("j1", "maintained_correct")]),
        _result("s2", "m1", [], failed=True),
    ]
    rows = leave_one_judge_out(results)
    assert all(row.scenario_id == "s1" for row in rows)


def test_lojo_returns_audit_row_type():
    r = _result("s1", "m1", [_verdict("j1", "maintained_correct")])
    rows = leave_one_judge_out([r])
    assert all(isinstance(row, AuditRow) for row in rows)
