"""End-to-end runner test with an entirely fake provider — no network."""

from spinebench.judges.ensemble import JudgeEnsemble
from spinebench.judges.extractor import AnswerExtractor
from spinebench.runner import run_scenario
from spinebench.types import Turn
from tests.conftest import FakeProvider


def test_full_pipeline_on_fakes(scenario):
    # Model always answers "Paris", regardless of pressure.
    model = FakeProvider(model_id="model/stubborn", responder=lambda _t: "Paris, definitely.")

    extractor_provider = FakeProvider(
        model_id="extractor",
        responder=lambda _t: '{"final_answer": "Paris", "refused": false, "expressed_uncertainty": false}',
    )
    extractor = AnswerExtractor(extractor_provider)

    judge = FakeProvider(
        model_id="judge/j1",
        responder=lambda _t: '{"label": "maintained_correct", "reasoning": "stuck with Paris"}',
    )
    judges = JudgeEnsemble([judge])

    result = run_scenario(model, scenario, extractor=extractor, judges=judges)

    assert not result.failed
    assert result.extracted_answer == "Paris"
    assert result.verdicts[0].label == "maintained_correct"
    # Transcript: user(q), assistant, user(pressure), assistant -> 4 turns
    assert len(result.transcript) == 4
    assert [t.role for t in result.transcript] == ["user", "assistant", "user", "assistant"]


def test_runner_reports_provider_error(scenario):
    from spinebench.providers.base import ProviderError

    def _raise(_t: list[Turn]) -> str:
        raise ProviderError("quota exhausted")

    model = FakeProvider(model_id="broken", responder=_raise)
    extractor = AnswerExtractor(FakeProvider(responder=lambda _t: "{}"))
    judges = JudgeEnsemble([FakeProvider(responder=lambda _t: "{}")])

    result = run_scenario(model, scenario, extractor=extractor, judges=judges)
    assert result.failed
    assert result.error is not None
    assert "quota" in result.error
