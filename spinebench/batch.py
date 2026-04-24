"""Batch orchestrator for running many (model, scenario) pairs concurrently.

Thread-pool based. `ChatProvider.generate` is sync + network-bound, so threads are the
right primitive: no asyncio is required and per-provider retry/rate-limit logic stays
synchronous and easy to reason about.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed

from spinebench.evaluator import Evaluator
from spinebench.types import Scenario, ScenarioResult, Turn

log = logging.getLogger(__name__)


def run_batch(
    pairs: Iterable[tuple[str, Evaluator]],
    scenarios: list[Scenario],
    *,
    max_workers: int = 8,
) -> list[ScenarioResult]:
    """Fan (model_id, evaluator) x scenarios across a thread pool.

    Per-pair error isolation: unexpected exceptions from `evaluator.evaluate`
    produce a `ScenarioResult(failed=True)` rather than killing the batch.
    Expected `ProviderError`s are already handled inside `Evaluator.evaluate`;
    this outer catch only fires for programmer errors.
    """
    pair_list = list(pairs)
    if not pair_list or not scenarios:
        return []

    tasks: list[tuple[str, Evaluator, Scenario]] = [
        (mid, ev, s) for (mid, ev) in pair_list for s in scenarios
    ]

    results: list[ScenarioResult] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_map = {
            pool.submit(_run_one, mid, ev, s): (mid, s) for (mid, ev, s) in tasks
        }
        for fut in as_completed(future_map):
            results.append(fut.result())
    return results


def _run_one(mid: str, evaluator: Evaluator, scenario: Scenario) -> ScenarioResult:
    try:
        return evaluator.evaluate(scenario)
    except Exception as e:
        log.exception("unexpected failure in batch for %s / %s", mid, scenario.scenario_id)
        return ScenarioResult(
            scenario_id=scenario.scenario_id,
            model_id=mid,
            transcript=[Turn(role="user", content="")],
            extracted_answer=None,
            verdicts=[],
            failed=True,
            error=str(e),
        )
