"""Boundary tests for spinebench.batch."""

from __future__ import annotations

from spinebench.batch import run_batch
from spinebench.evaluator import Evaluator
from spinebench.providers.base import ProviderError
from spinebench.types import FailureMode, GroundTruthQuestion, PressureTemplate, Scenario
from tests.conftest import FakeProvider


def _make_scenarios(n: int) -> list[Scenario]:
    scenarios = []
    for i in range(n):
        q = GroundTruthQuestion(
            qid=f"q{i}",
            source="truthfulqa",
            domain="x",
            question=f"question {i}",
            correct_answer="Paris",
            incorrect_answers=["London"],
        )
        t = PressureTemplate(
            template_id="tpl",
            failure_mode=FailureMode.DIRECT_PUSHBACK,
            turns=["Are you sure?"],
        )
        scenarios.append(Scenario(scenario_id=f"q{i}__tpl", question=q, template=t))
    return scenarios


def _make_evaluator(model_id: str) -> Evaluator:
    return Evaluator(
        subject=FakeProvider(model_id=model_id, responder=lambda _t: "Paris."),
        extractor=FakeProvider(
            model_id="extractor",
            responder=lambda _t: '{"final_answer": "Paris", "refused": false, "expressed_uncertainty": false}',
        ),
        judges=[
            FakeProvider(
                model_id="j1",
                responder=lambda _t: '{"label": "maintained_correct", "reasoning": "ok"}',
            ),
        ],
    )


def test_run_batch_schedules_all_pairs():
    scenarios = _make_scenarios(3)
    pairs = [("mA", _make_evaluator("mA")), ("mB", _make_evaluator("mB"))]
    results = run_batch(pairs, scenarios)
    # 2 models x 3 scenarios = 6 results
    assert len(results) == 6
    model_ids = {r.model_id for r in results}
    assert model_ids == {"mA", "mB"}


def test_run_batch_per_pair_error_isolation():
    scenarios = _make_scenarios(2)

    def _raise(_t):
        raise ProviderError("broken")

    ev_broken = Evaluator(
        subject=FakeProvider(model_id="broken/m", responder=_raise),
        extractor=FakeProvider(
            model_id="ext",
            responder=lambda _t: '{"final_answer": "Paris", "refused": false, "expressed_uncertainty": false}',
        ),
        judges=[
            FakeProvider(
                model_id="j1",
                responder=lambda _t: '{"label": "maintained_correct", "reasoning": ""}',
            )
        ],
    )
    ev_ok = _make_evaluator("ok/m")

    pairs = [("broken/m", ev_broken), ("ok/m", ev_ok)]
    results = run_batch(pairs, scenarios)
    # Both models produce results; broken-model results have failed=True.
    assert len(results) == 4
    broken = [r for r in results if r.model_id == "broken/m"]
    ok = [r for r in results if r.model_id == "ok/m"]
    assert all(r.failed for r in broken)
    assert all(not r.failed for r in ok)


def test_run_batch_respects_max_workers():
    """A sentinel check that max_workers is honored (smoke test, not a perf assertion).

    We just verify the call runs cleanly with max_workers=1 and produces the same
    results as default workers.
    """
    scenarios = _make_scenarios(2)
    pairs = [("m1", _make_evaluator("m1"))]
    results_parallel = run_batch(pairs, scenarios, max_workers=8)
    results_sequential = run_batch(pairs, scenarios, max_workers=1)
    assert len(results_parallel) == len(results_sequential) == 2


def test_run_batch_empty_inputs():
    assert run_batch([], _make_scenarios(3)) == []
    assert run_batch([("m1", _make_evaluator("m1"))], []) == []
