"""Smoke test: bundled pressure templates YAML loads and covers all 10 failure modes."""

from spinebench.data.template_loader import load_pressure_templates
from spinebench.types import FailureMode


def test_load_bundled_templates():
    templates = load_pressure_templates()
    assert len(templates) >= 10

    modes_covered = {t.failure_mode for t in templates}
    expected = set(FailureMode)
    missing = expected - modes_covered
    assert not missing, f"failure modes missing from v0 templates: {missing}"


def test_templates_render_placeholders_without_error(question, template):
    templates = load_pressure_templates()
    for t in templates:
        for turn in t.turns:
            turn.format(
                question=question.question,
                correct_answer=question.correct_answer,
                incorrect_answer=question.incorrect_answers[0],
            )
