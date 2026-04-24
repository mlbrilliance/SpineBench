"""Boundary tests for spinebench.reporting."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from spinebench.reporting import audit_to_parquet, results_to_parquet
from spinebench.types import AuditRow, JudgeVerdict, ScenarioResult, Turn


def _result(sid: str, mid: str, label: str, failed: bool = False) -> ScenarioResult:
    return ScenarioResult(
        scenario_id=sid,
        model_id=mid,
        transcript=[
            Turn(role="user", content="q"),
            Turn(role="assistant", content="Paris"),
        ],
        extracted_answer="Paris",
        verdicts=[
            JudgeVerdict(judge_model="j1", label=label),
            JudgeVerdict(judge_model="j2", label=label),
            JudgeVerdict(judge_model="j3", label=label),
        ],
        failed=failed,
    )


def test_results_to_parquet_round_trip(tmp_path: Path):
    results = [
        _result("s1", "m1", "maintained_correct"),
        _result("s2", "m1", "flipped_to_wrong"),
    ]
    path = tmp_path / "results.parquet"
    results_to_parquet(results, path)
    df = pd.read_parquet(path)
    assert len(df) == 2
    assert set(["scenario_id", "model_id", "extracted_answer", "failed"]).issubset(df.columns)
    assert df["scenario_id"].tolist() == ["s1", "s2"]


def test_results_to_parquet_failed_column(tmp_path: Path):
    results = [
        _result("s1", "m1", "maintained_correct"),
        _result("s2", "m1", "flipped_to_wrong", failed=True),
    ]
    path = tmp_path / "results.parquet"
    results_to_parquet(results, path)
    df = pd.read_parquet(path)
    assert df["failed"].tolist() == [False, True]


def test_audit_to_parquet_round_trip(tmp_path: Path):
    rows = [
        AuditRow(
            scenario_id="s1",
            model_id="m1",
            dropped_judge=None,
            majority_label="maintained_correct",
            retained_verdicts=[JudgeVerdict(judge_model="j1", label="maintained_correct")],
        ),
        AuditRow(
            scenario_id="s1",
            model_id="m1",
            dropped_judge="j1",
            majority_label="other",
            retained_verdicts=[],
        ),
    ]
    path = tmp_path / "audit.parquet"
    audit_to_parquet(rows, path)
    df = pd.read_parquet(path)
    assert len(df) == 2
    assert set(["scenario_id", "dropped_judge", "majority_label"]).issubset(df.columns)
    # pd nullable/NaN handling: the None dropped_judge should not equal "j1"
    assert df["dropped_judge"].tolist()[1] == "j1"


def test_results_to_parquet_creates_parent_dirs(tmp_path: Path):
    path = tmp_path / "nested" / "dir" / "results.parquet"
    results_to_parquet([_result("s1", "m1", "maintained_correct")], path)
    assert path.exists()


def test_empty_results_writes_empty_parquet(tmp_path: Path):
    path = tmp_path / "empty.parquet"
    results_to_parquet([], path)
    df = pd.read_parquet(path)
    assert len(df) == 0
