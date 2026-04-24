"""Boundary tests for spinebench.evaluator."""

from __future__ import annotations

from spinebench.cache import InMemoryCache
from spinebench.evaluator import Evaluator
from spinebench.providers.base import ProviderError
from spinebench.types import Scenario, Turn
from tests.conftest import FakeProvider

_EXTRACTOR_OK = (
    '{"final_answer": "Paris", "refused": false, "expressed_uncertainty": false}'
)


def _make_judge(label: str, model_id: str = "judge") -> FakeProvider:
    return FakeProvider(
        model_id=model_id,
        responder=lambda _t: f'{{"label": "{label}", "reasoning": "stub"}}',
    )


def _make_evaluator(
    subject_responder=lambda _t: "Paris, definitely.",
    *,
    judges: list[FakeProvider] | None = None,
    cache=None,
) -> tuple[Evaluator, FakeProvider, FakeProvider, list[FakeProvider]]:
    subject = FakeProvider(model_id="subject/model", responder=subject_responder)
    extractor = FakeProvider(model_id="extractor", responder=lambda _t: _EXTRACTOR_OK)
    panel = judges or [
        _make_judge("maintained_correct", "j1"),
        _make_judge("maintained_correct", "j2"),
        _make_judge("maintained_correct", "j3"),
    ]
    kwargs = {"cache": cache} if cache is not None else {}
    ev = Evaluator(subject=subject, extractor=extractor, judges=panel, **kwargs)
    return ev, subject, extractor, panel


def test_evaluate_full_pipeline_success(scenario: Scenario):
    ev, _subject, _extractor, _judges = _make_evaluator()
    result = ev.evaluate(scenario)

    assert not result.failed
    assert result.scenario_id == scenario.scenario_id
    assert result.model_id == "subject/model"
    assert result.extracted_answer == "Paris"
    assert len(result.verdicts) == 3
    assert all(v.label == "maintained_correct" for v in result.verdicts)
    # Transcript: user-q, assistant, user-pressure, assistant -> 4 turns
    assert len(result.transcript) == 4
    assert [t.role for t in result.transcript] == ["user", "assistant", "user", "assistant"]


def test_evaluate_subject_provider_error_returns_failed(scenario: Scenario):
    def _raise(_t):
        raise ProviderError("quota exhausted")

    ev, *_ = _make_evaluator(subject_responder=_raise)
    result = ev.evaluate(scenario)
    assert result.failed
    assert result.error is not None
    assert "quota" in result.error
    # Verdicts are not collected on subject failure.
    assert result.verdicts == []


def test_evaluate_judge_error_surfaces_as_other_label(scenario: Scenario):
    def _raise(_t):
        raise ProviderError("judge broke")

    broken_judge = FakeProvider(model_id="j-broken", responder=_raise)
    good_judge = _make_judge("maintained_correct", "j-good")
    ev, *_ = _make_evaluator(judges=[broken_judge, good_judge])

    result = ev.evaluate(scenario)
    assert not result.failed
    labels = [(v.judge_model, v.label) for v in result.verdicts]
    assert ("j-broken", "other") in labels
    assert ("j-good", "maintained_correct") in labels


def test_evaluate_cache_hit_short_circuits_subject(scenario: Scenario):
    cache = InMemoryCache()
    cached_transcript = [
        Turn(role="user", content=scenario.question.question),
        Turn(role="assistant", content="cached-response"),
        Turn(role="user", content="pressure turn"),
        Turn(role="assistant", content="cached-followup"),
    ]
    cache.put("subject/model", scenario.scenario_id, cached_transcript)

    ev, subject, extractor, judges = _make_evaluator(cache=cache)
    result = ev.evaluate(scenario)

    # Subject must NOT be called — cache provides the transcript.
    assert subject.calls == []
    # Extractor + judges still run (they're not cached).
    assert len(extractor.calls) == 1
    assert all(len(j.calls) == 1 for j in judges)
    # Returned transcript is the cached one.
    assert result.transcript == cached_transcript


def test_evaluate_cache_miss_populates_cache(scenario: Scenario):
    cache = InMemoryCache()
    ev, _subject, *_ = _make_evaluator(cache=cache)

    assert cache.get("subject/model", scenario.scenario_id) is None
    result = ev.evaluate(scenario)
    assert not result.failed
    # After evaluation, the cache holds the freshly produced transcript.
    assert cache.get("subject/model", scenario.scenario_id) == result.transcript


def test_evaluate_failed_scenario_does_not_cache(scenario: Scenario):
    def _raise(_t):
        raise ProviderError("nope")

    cache = InMemoryCache()
    ev, *_ = _make_evaluator(subject_responder=_raise, cache=cache)
    result = ev.evaluate(scenario)
    assert result.failed
    assert cache.get("subject/model", scenario.scenario_id) is None


def test_evaluate_judge_prompt_contains_cot_instruction(scenario: Scenario):
    """The CoT judge prompt must instruct the judge to reason before labeling.

    We check that the judge received a prompt containing the string 'step' or 'reason'
    somewhere in the system message (case-insensitive). This pins the CoT contract without
    coupling to exact prompt wording.
    """
    judge = _make_judge("maintained_correct", "j1")
    ev, *_ = _make_evaluator(judges=[judge])
    ev.evaluate(scenario)

    assert len(judge.calls) == 1
    system_turns = [t for t in judge.calls[0] if t.role == "system"]
    assert system_turns, "judge prompt must include a system message"
    blob = " ".join(t.content.lower() for t in system_turns)
    assert "step" in blob or "reason" in blob


def test_evaluate_default_cache_is_null(scenario: Scenario):
    """No cache passed -> evaluation still works, cache ops are no-ops."""
    ev, subject, *_ = _make_evaluator()  # no cache kwarg
    result = ev.evaluate(scenario)
    assert not result.failed
    # Subject was actually called (no short-circuit).
    assert len(subject.calls) >= 1


def test_evaluate_propagates_subject_revision(scenario: Scenario):
    """When the subject provider has a `.revision` attribute (e.g. from ModelRuntime),
    Evaluator records it in ScenarioResult.revision for downstream provenance."""
    from spinebench.runtime import FakePinner, ModelRuntime, ModelSpec

    class _FactoryReturning:
        def __init__(self, provider):
            self._p = provider

        def make(self, spec, **_):
            return self._p

    inner_subject = FakeProvider(model_id="pinned/m", responder=lambda _t: "Paris.")
    runtime = ModelRuntime(
        pinner=FakePinner({"pinned/m": "sha-feedbeef"}),
        provider_factory=_FactoryReturning(inner_subject),
    )
    [pinned] = runtime.pin([ModelSpec(model_id="pinned/m")])
    subject_wrapped = runtime.chat(pinned)

    ev = Evaluator(
        subject=subject_wrapped,
        extractor=FakeProvider(model_id="extractor", responder=lambda _t: _EXTRACTOR_OK),
        judges=[_make_judge("maintained_correct", "j1")],
    )
    result = ev.evaluate(scenario)
    assert result.revision == "sha-feedbeef"
