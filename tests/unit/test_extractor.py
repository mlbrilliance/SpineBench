from spinebench.judges.extractor import AnswerExtractor
from spinebench.types import Turn
from tests.conftest import FakeProvider


def _fake_returning(raw: str) -> FakeProvider:
    return FakeProvider(model_id="fake/judge", responder=lambda _t: raw)


def test_extracts_clean_json():
    provider = _fake_returning(
        '{"final_answer": "Paris", "refused": false, "expressed_uncertainty": false}'
    )
    out = AnswerExtractor(provider).extract(
        [Turn(role="user", content="q"), Turn(role="assistant", content="Paris")]
    )
    assert out.final_answer == "Paris"
    assert out.refused is False


def test_strips_fenced_json():
    provider = _fake_returning(
        '```json\n{"final_answer": "Paris", "refused": false, "expressed_uncertainty": false}\n```'
    )
    out = AnswerExtractor(provider).extract([Turn(role="user", content="q")])
    assert out.final_answer == "Paris"


def test_tolerates_leading_prose():
    provider = _fake_returning(
        'Here is the JSON: {"final_answer": "Paris", "refused": false, "expressed_uncertainty": true} done.'
    )
    out = AnswerExtractor(provider).extract([Turn(role="user", content="q")])
    assert out.expressed_uncertainty is True


def test_falls_back_on_garbage():
    provider = _fake_returning("literally nothing parseable")
    out = AnswerExtractor(provider).extract([Turn(role="user", content="q")])
    assert out.final_answer == ""
    assert out.refused is False
