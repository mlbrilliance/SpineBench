"""Boundary tests for probe loading + probe_accuracy."""

from __future__ import annotations

from spinebench.data.probes import AdversarialProbe, load_probes
from spinebench.probes import probe_accuracy
from spinebench.types import FailureMode  # noqa: F401 - imported to verify package wiring
from tests.conftest import FakeProvider


def _judge_that_always_says(label: str, model_id: str) -> FakeProvider:
    return FakeProvider(
        model_id=model_id,
        responder=lambda _t: f'{{"label": "{label}", "reasoning": "stub"}}',
    )


def test_load_probes_returns_adversarial_probe_instances():
    probes = load_probes()
    assert len(probes) >= 10
    assert all(isinstance(p, AdversarialProbe) for p in probes)


def test_every_probe_has_valid_expected_label():
    probes = load_probes()
    allowed = {"maintained_correct", "flipped_to_wrong", "became_uncertain", "refused", "other"}
    for p in probes:
        assert p.expected_label in allowed, (p.probe_id, p.expected_label)


def test_probe_accuracy_perfect_judge():
    """Judge that magically returns the expected label for every probe -> 100%."""
    probes = load_probes()

    # Build a judge that inspects the prompt and always agrees with the expected label.
    # Since probes are in order, we can use a stateful responder.
    class StatefulJudge:
        model_id = "magic/judge"
        def __init__(self):
            self._idx = 0
            self._labels = [p.expected_label for p in probes]
            self.calls = []
        def generate(self, turns, *, max_tokens=400, temperature=0.0):
            self.calls.append(list(turns))
            label = self._labels[self._idx]
            self._idx += 1
            return f'{{"label": "{label}", "reasoning": "stub"}}'

    judge = StatefulJudge()
    report = probe_accuracy(probes, judges=[judge])
    assert report.total_probes == len(probes)
    assert report.per_judge_accuracy["magic/judge"] == 1.0
    assert report.ensemble_accuracy == 1.0


def test_probe_accuracy_always_wrong_judge():
    """Judge that always says 'other' -> accuracy matches fraction of probes expecting 'other'."""
    probes = load_probes()
    judge = _judge_that_always_says("other", "always-other")
    report = probe_accuracy(probes, judges=[judge])

    expected_fraction = sum(1 for p in probes if p.expected_label == "other") / len(probes)
    assert report.per_judge_accuracy["always-other"] == expected_fraction


def test_probe_accuracy_empty_probes():
    report = probe_accuracy([], judges=[_judge_that_always_says("other", "j")])
    assert report.total_probes == 0
    assert report.ensemble_accuracy == 0.0


def test_probe_accuracy_judge_exception_becomes_other():
    from spinebench.providers.base import ProviderError

    def _raise(_t):
        raise ProviderError("broken")

    # One probe that expects maintained_correct, judge raises -> label "other", 0 correct.
    probes = [load_probes()[0]]  # first probe, expected = maintained_correct
    broken = FakeProvider(model_id="broken/j", responder=_raise)
    report = probe_accuracy(probes, judges=[broken])
    assert report.per_judge_accuracy["broken/j"] == 0.0
    assert report.outcomes[0].verdicts[0].label == "other"
