# RFC 0001: Deepen the evaluation pipeline into a single `Evaluator`

**Status**: Accepted
**Author**: nick
**Date**: 2026-04-24
**Priority**: high — Week 3 blocker for CoT judges, leave-one-judge-out audit, and adversarial probe set

## Problem

The scenario-execution pipeline is currently split across three shallow files that are always called together in the same order:

- `spinebench/runner.py` (58 LOC) — multi-turn rollout + top-level error handling
- `spinebench/judges/extractor.py` (82 LOC) — calls a small LLM to parse the transcript into `{final_answer, refused, expressed_uncertainty}`
- `spinebench/judges/ensemble.py` (108 LOC) — runs N judges, collects per-judge verdicts

Three pieces of evidence that this cluster is shallow:

1. **Always called together.** Every real caller (`runner.run_scenario`, future batch runner) invokes rollout → extract → judge in exactly the same order. No call-site uses extractor or ensemble independently.
2. **Duplicated concerns.** Error handling, JSON-parse tolerance, and `provider.generate()` dispatch are reimplemented in each file with slight drift.
3. **Week-3 pressure.** Three new features land in this cluster simultaneously: CoT judge prompts (mutates `ensemble.py`), leave-one-judge-out variance audit (mutates `runner.py` *and* `ensemble.py`), adversarial probe set (mutates `runner.py` test fixtures). Without deepening, these features will worsen the drift.

The integration risk lives in the seams: today every new feature that touches two-of-three of these files risks breaking the third, and the test suite only tests each in isolation.

## Proposed interface

A single `Evaluator` that owns rollout + extraction + judging for one scenario at a time, plus four small companion modules that handle concerns the evaluator deliberately does not own.

### Core interface

```python
# spinebench/evaluator.py
from dataclasses import dataclass
from spinebench.cache import TranscriptCache, NullCache
from spinebench.providers.base import ChatProvider
from spinebench.types import Scenario, ScenarioResult

@dataclass
class Evaluator:
    subject: ChatProvider
    extractor: ChatProvider
    judges: list[ChatProvider]
    cache: TranscriptCache = NullCache()

    def evaluate(self, scenario: Scenario) -> ScenarioResult:
        """Execute one scenario end-to-end: rollout -> extract -> judge.

        On subject ProviderError, returns a ScenarioResult with failed=True and
        partial transcript preserved. Judge errors are isolated per judge and
        surface as label='other' verdicts with error reasoning.
        """
```

### Companion modules

```python
# spinebench/cache.py
class TranscriptCache(Protocol):
    def get(self, model_id: str, scenario_id: str) -> list[Turn] | None: ...
    def put(self, model_id: str, scenario_id: str, transcript: list[Turn]) -> None: ...

class NullCache:  # default
    ...
class InMemoryCache:  # process-local dict
    ...
class DiskCache:  # one JSON per (model, scenario); atomic via os.replace
    def __init__(self, root: Path): ...
```

```python
# spinebench/audit.py
def leave_one_judge_out(
    results: list[ScenarioResult],
) -> list[AuditRow]:
    """Recompute majority labels with each judge dropped. No LLM calls — purely
    reaggregates the verdicts already attached to each ScenarioResult."""
```

```python
# spinebench/batch.py
def run_batch(
    pairs: Iterable[tuple[str, Evaluator]],   # (model_id, evaluator)
    scenarios: list[Scenario],
    *, max_workers: int = 8,
) -> list[ScenarioResult]:
    """Thread-pool orchestrator. The only entry point for full-run sweeps."""
```

```python
# spinebench/reporting.py
def results_to_parquet(results: list[ScenarioResult], path: Path) -> None: ...
def audit_to_parquet(audit: list[AuditRow], path: Path) -> None: ...
```

### Usage example

```python
from spinebench.evaluator import Evaluator
from spinebench.cache import DiskCache
from spinebench.batch import run_batch
from spinebench.audit import leave_one_judge_out
from spinebench.reporting import results_to_parquet, audit_to_parquet

cache = DiskCache(Path("runs/week3/cache"))

pairs = []
for subject in subject_models:
    ev = Evaluator(subject, extractor, judges=[j1, j2, j3], cache=cache)
    pairs.append((subject.model_id, ev))

results = run_batch(pairs, scenarios, max_workers=8)
audit = leave_one_judge_out(results)

results_to_parquet(results, Path("runs/week3/results.parquet"))
audit_to_parquet(audit, Path("runs/week3/audit.parquet"))
```

### What complexity it hides

Inside `Evaluator.evaluate`:

- Multi-turn rollout loop (interleaving user/assistant turns).
- `ProviderError` handling with partial-transcript preservation.
- Extractor prompt construction, JSON parsing with fenced-block tolerance, fallback to empty answer.
- CoT judge prompt construction (module constant, no runtime toggle).
- Per-judge try/except so one broken judge doesn't fail the scenario.
- Cache lookup keyed on `(subject.model_id, scenario_id)`, consulted before rollout only. Judge changes do not invalidate the cache.

Inside `batch.run_batch`:

- Thread-pool fan-out over (model, scenario) pairs with bounded concurrency.
- Progress logging, per-pair error capture.
- Resumability via cache — re-running the batch skips already-rolled-out pairs.

## Dependency strategy

**Category: In-process (constructor-injected).**

All `ChatProvider` instances are passed in by the caller. The evaluator never constructs a provider, reads environment variables, or touches the network directly. Tests use the existing `FakeProvider` in `conftest.py` — no mocking library needed.

The `TranscriptCache` is the one remote-adjacent dependency (it may touch disk for `DiskCache`). Treated as a Protocol with a `NullCache` default and an in-memory implementation for tests, so the Evaluator's core test suite stays fully in-process.

## Testing strategy

### New boundary tests on `Evaluator.evaluate`

Test the observable contract, not internals. ~6 tests:

1. Full pipeline success with stubbed subject + extractor + judges → correct `ScenarioResult` with transcript, extracted answer, and N verdicts.
2. Subject `ProviderError` → returns `ScenarioResult(failed=True, error=...)` with partial transcript preserved.
3. Judge error → that judge's verdict has `label="other"`, other judges' verdicts unaffected.
4. Cache hit → rollout short-circuited; extractor+judges still run; cache's `get` called; `put` not called.
5. Cache miss → full pipeline runs; cache's `put` called exactly once with the produced transcript.
6. CoT prompt is in the judge call (assert via FakeProvider call-recording).

### New tests on companion modules (~8 total)

- `cache.py`: `NullCache` no-op semantics; `InMemoryCache` round-trip; `DiskCache` atomic write + reload across process restart.
- `audit.py`: `leave_one_judge_out` reproduces full majority when no judge dropped; drops change labels predictably when judges disagree.
- `batch.py`: `run_batch` schedules N × M pairs; respects `max_workers`; per-pair error isolation.
- `reporting.py`: `results_to_parquet` round-trips via pandas; nested verdicts flattened to sidecar table.

### Old tests to delete

- `tests/unit/test_runner.py` — the 2 tests here are redundant with Evaluator boundary tests #1 and #2.
- `tests/unit/test_extractor.py` — all 4 tests, superseded by boundary test #1 (extractor parse paths become internal, tested via observable output).
- `tests/unit/test_ensemble.py` — all 3 tests, superseded by boundary tests #1, #3, #6.

Net: 9 shallow-module tests deleted → 14 deeper tests added, with better coverage of the real integration risk.

## Implementation recommendations

The module should own:

- The invariant ordering rollout → extract → judge.
- CoT judge prompts as module constants.
- Majority-vote tie handling (ties → "other"), isolated judge failures.
- Cache lookup/store choreography with rollout-only caching semantics.

The module should hide:

- Multi-turn rollout mechanics.
- Extractor JSON schema.
- Per-judge error handling.
- Cache key construction.

The module should expose:

- The constructor signature (subject, extractor, judges, cache).
- The single `evaluate(scenario) -> ScenarioResult` method.
- Nothing else.

Caller migration:

- Replace `run_scenario(model, scenario, extractor=..., judges=...)` calls with `Evaluator(subject=model, extractor=..., judges=...).evaluate(scenario)`.
- Replace `JudgeEnsemble` / `AnswerExtractor` direct construction with Evaluator construction.
- Batch runs move from "for scenario in scenarios: run_scenario(...)" to `run_batch(pairs, scenarios)`.

The keepable parts of today's code (`AnswerExtractor._parse`, `JudgeEnsemble._parse`, etc.) become private module-level functions in the new `evaluator.py`. No behavior changes in the rollout loop or judge classification — this is a structural refactor.

## Build order

1. Write companion modules first (cache, audit, reporting) — smallest, independent.
2. Write `Evaluator` with boundary tests.
3. Write `run_batch` against the new Evaluator.
4. Delete superseded tests and old files (`runner.py`, `judges/*.py`).
5. Update `spinebench/__init__.py` re-exports.

TDD: red → green → refactor on each step. Route implementation through multi-model-router (GLM 5.1) per the standing rule.
