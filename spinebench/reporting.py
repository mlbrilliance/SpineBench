"""Parquet serialization for results and audit rows.

Nested lists (transcript, verdicts, retained_verdicts) are stored as object columns.
Pandas + pyarrow round-trip them cleanly as long as each row has a consistent schema.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from spinebench.types import AuditRow, ScenarioResult

_RESULT_COLUMNS = [
    "scenario_id",
    "model_id",
    "transcript",
    "extracted_answer",
    "verdicts",
    "failed",
    "error",
]

_AUDIT_COLUMNS = [
    "scenario_id",
    "model_id",
    "dropped_judge",
    "majority_label",
    "retained_verdicts",
]


def results_to_parquet(results: list[ScenarioResult], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if results:
        df = pd.DataFrame([r.model_dump() for r in results])
    else:
        df = pd.DataFrame({col: pd.Series([], dtype="object") for col in _RESULT_COLUMNS})
    df.to_parquet(path, index=False)


def audit_to_parquet(rows: list[AuditRow], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        df = pd.DataFrame([r.model_dump() for r in rows])
    else:
        df = pd.DataFrame({col: pd.Series([], dtype="object") for col in _AUDIT_COLUMNS})
    df.to_parquet(path, index=False)
