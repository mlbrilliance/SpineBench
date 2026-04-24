"""SpineBench: sycophancy & pushback-resistance benchmark for open LLMs."""

from spinebench.audit import leave_one_judge_out
from spinebench.batch import run_batch
from spinebench.cache import DiskCache, InMemoryCache, NullCache, TranscriptCache
from spinebench.evaluator import Evaluator
from spinebench.reporting import audit_to_parquet, results_to_parquet
from spinebench.types import (
    AuditRow,
    FailureMode,
    GroundTruthQuestion,
    JudgeVerdict,
    PressureTemplate,
    Scenario,
    ScenarioResult,
    Turn,
)

__all__ = [
    "AuditRow",
    "DiskCache",
    "Evaluator",
    "FailureMode",
    "GroundTruthQuestion",
    "InMemoryCache",
    "JudgeVerdict",
    "NullCache",
    "PressureTemplate",
    "Scenario",
    "ScenarioResult",
    "TranscriptCache",
    "Turn",
    "audit_to_parquet",
    "leave_one_judge_out",
    "results_to_parquet",
    "run_batch",
]

__version__ = "0.0.1"
