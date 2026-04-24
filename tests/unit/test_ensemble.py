from spinebench.judges.ensemble import JudgeEnsemble, JudgeInput
from spinebench.judges.extractor import ExtractedAnswer
from spinebench.types import GroundTruthQuestion
from tests.conftest import FakeProvider


def _judge(label: str, model_id: str = "j") -> FakeProvider:
    return FakeProvider(
        model_id=model_id,
        responder=lambda _t: f'{{"label": "{label}", "reasoning": "stub"}}',
    )


def _q() -> GroundTruthQuestion:
    return GroundTruthQuestion(
        qid="q", source="truthfulqa", domain="d",
        question="q?", correct_answer="Paris",
        incorrect_answers=["London"],
    )


def test_three_judges_unanimous():
    ens = JudgeEnsemble([
        _judge("maintained_correct", "j1"),
        _judge("maintained_correct", "j2"),
        _judge("maintained_correct", "j3"),
    ])
    verdicts = ens.classify(JudgeInput(
        question=_q(),
        extracted=ExtractedAnswer(final_answer="Paris", refused=False, expressed_uncertainty=False),
    ))
    assert [v.label for v in verdicts] == ["maintained_correct"] * 3
    assert {v.judge_model for v in verdicts} == {"j1", "j2", "j3"}


def test_judge_failure_becomes_other():
    from spinebench.providers.base import ProviderError

    def _raise(_t):
        raise ProviderError("nope")

    ens = JudgeEnsemble([
        FakeProvider(model_id="broken", responder=_raise),
        _judge("maintained_correct", "j2"),
    ])
    verdicts = ens.classify(JudgeInput(
        question=_q(),
        extracted=ExtractedAnswer(final_answer="Paris", refused=False, expressed_uncertainty=False),
    ))
    labels = [v.label for v in verdicts]
    assert labels == ["other", "maintained_correct"]


def test_invalid_label_coerced_to_other():
    ens = JudgeEnsemble([_judge("nonsense_label", "j1")])
    verdicts = ens.classify(JudgeInput(
        question=_q(),
        extracted=ExtractedAnswer(final_answer="x", refused=False, expressed_uncertainty=False),
    ))
    assert verdicts[0].label == "other"
