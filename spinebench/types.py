"""Core data types for SpineBench.

Everything flows through these: ground truth -> scenario -> turns -> judge verdicts -> score.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class FailureMode(StrEnum):
    """The 10 failure modes cross-producted with ground-truth questions."""

    DIRECT_PUSHBACK = "direct_pushback"
    FALSE_PREMISE = "false_premise"
    AUTHORITY_APPEAL = "authority_appeal"
    SOCIAL_PRESSURE = "social_pressure"
    EMOTIONAL_PRESSURE = "emotional_pressure"
    ESCALATING_INSISTENCE = "escalating_insistence"
    FABRICATED_CITATION = "fabricated_citation"
    SELF_CONTRADICTION = "self_contradiction"
    CONFIDENT_WRONGNESS = "confident_wrongness"
    PERSONA_DRIFT = "persona_drift"


class GroundTruthQuestion(BaseModel):
    """A question with a known-correct answer pulled from TruthfulQA / SimpleQA / MMLU-Pro."""

    qid: str
    source: Literal["truthfulqa", "simpleqa", "mmlu_pro"]
    domain: str
    question: str
    correct_answer: str
    incorrect_answers: list[str] = Field(default_factory=list)


class PressureTemplate(BaseModel):
    """A templated user-side pressure applied across one or more turns.

    `turns` is a list of user messages. Placeholders:
      {question}         -> the ground-truth question
      {correct_answer}   -> the correct answer (for constructing false-premise setups)
      {incorrect_answer} -> a plausible wrong answer (for false-premise setups)
    """

    template_id: str
    failure_mode: FailureMode
    turns: list[str]
    weight: float = 1.0


class Turn(BaseModel):
    """A single turn in a multi-turn chat transcript."""

    role: Literal["system", "user", "assistant"]
    content: str


class Scenario(BaseModel):
    """A concrete prompt instance: ground-truth question + pressure template applied."""

    scenario_id: str
    question: GroundTruthQuestion
    template: PressureTemplate
    split: Literal["dev", "heldout"] = "dev"

    def render(self) -> list[Turn]:
        """Render this scenario as a multi-turn conversation up to but excluding the model's
        final response."""
        turns: list[Turn] = []
        # Initial user turn is always the ground-truth question itself.
        turns.append(Turn(role="user", content=self.question.question))
        # Subsequent user turns are pressure-template turns, rendered with placeholders.
        incorrect = (
            self.question.incorrect_answers[0] if self.question.incorrect_answers else ""
        )
        for raw in self.template.turns:
            rendered = raw.format(
                question=self.question.question,
                correct_answer=self.question.correct_answer,
                incorrect_answer=incorrect,
            )
            turns.append(Turn(role="user", content=rendered))
        return turns


class JudgeVerdict(BaseModel):
    """One judge's classification of a model's final answer."""

    judge_model: str
    label: Literal[
        "maintained_correct",
        "flipped_to_wrong",
        "became_uncertain",
        "refused",
        "other",
    ]
    reasoning: str | None = None


class ScenarioResult(BaseModel):
    """A single (model, scenario) evaluation outcome."""

    scenario_id: str
    model_id: str
    transcript: list[Turn]
    extracted_answer: str | None
    verdicts: list[JudgeVerdict]
    failed: bool = False
    error: str | None = None


class AuditRow(BaseModel):
    """One row from a leave-one-judge-out variance audit.

    Produced by :func:`spinebench.audit.leave_one_judge_out`. Each row describes what
    the majority label would be if `dropped_judge` were excluded from the ensemble.
    `dropped_judge=None` is the baseline row with no judge dropped.
    """

    scenario_id: str
    model_id: str
    dropped_judge: str | None
    majority_label: str
    retained_verdicts: list[JudgeVerdict]
