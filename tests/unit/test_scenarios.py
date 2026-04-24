from spinebench.data.scenarios import build_scenarios, split_scenarios
from spinebench.types import FailureMode, GroundTruthQuestion, PressureTemplate


def _q(qid: str) -> GroundTruthQuestion:
    return GroundTruthQuestion(
        qid=qid, source="truthfulqa", domain="d", question="q?", correct_answer="a"
    )


def _t(tid: str) -> PressureTemplate:
    return PressureTemplate(
        template_id=tid,
        failure_mode=FailureMode.DIRECT_PUSHBACK,
        turns=["are you sure?"],
    )


def test_cross_product():
    qs = [_q("q1"), _q("q2")]
    ts = [_t("t1"), _t("t2"), _t("t3")]
    scenarios = build_scenarios(qs, ts)
    assert len(scenarios) == 6
    assert {s.scenario_id for s in scenarios} == {
        "q1__t1", "q1__t2", "q1__t3",
        "q2__t1", "q2__t2", "q2__t3",
    }


def test_split_deterministic():
    qs = [_q(f"q{i}") for i in range(50)]
    ts = [_t("t1")]
    scenarios = build_scenarios(qs, ts)
    dev1, held1 = split_scenarios(scenarios, heldout_fraction=0.2, seed=7)
    dev2, held2 = split_scenarios(scenarios, heldout_fraction=0.2, seed=7)
    assert {s.scenario_id for s in dev1} == {s.scenario_id for s in dev2}
    assert {s.scenario_id for s in held1} == {s.scenario_id for s in held2}
    # roughly 20% in held-out
    assert 0.10 <= len(held1) / len(scenarios) <= 0.30


def test_render_includes_initial_question_and_pressure(scenario):
    turns = scenario.render()
    assert turns[0].role == "user"
    assert "capital of France" in turns[0].content
    assert turns[1].role == "user"
    # incorrect_answer placeholder was filled from question.incorrect_answers[0]
    assert "London" in turns[1].content
