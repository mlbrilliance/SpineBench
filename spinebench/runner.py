"""Scenario runner: apply a scenario to a model, extract the final answer, judge it."""

from __future__ import annotations

import logging

from spinebench.judges.ensemble import JudgeEnsemble, JudgeInput
from spinebench.judges.extractor import AnswerExtractor
from spinebench.providers.base import ChatProvider, ProviderError
from spinebench.types import Scenario, ScenarioResult, Turn

log = logging.getLogger(__name__)


def run_scenario(
    model: ChatProvider,
    scenario: Scenario,
    *,
    extractor: AnswerExtractor,
    judges: JudgeEnsemble,
    max_tokens: int = 512,
) -> ScenarioResult:
    """Execute a scenario end-to-end: rollout -> extract -> judge."""
    turns = scenario.render()
    transcript: list[Turn] = list(turns)

    # Multi-turn rollout: after each user turn, get the model's response.
    # Our scenario rendering interleaves user turns; we insert assistant turns between them.
    built: list[Turn] = []
    user_queue = list(turns)
    try:
        while user_queue:
            built.append(user_queue.pop(0))  # next user turn
            reply = model.generate(built, max_tokens=max_tokens)
            built.append(Turn(role="assistant", content=reply))
        transcript = built
    except ProviderError as e:
        return ScenarioResult(
            scenario_id=scenario.scenario_id,
            model_id=model.model_id,
            transcript=built,
            extracted_answer=None,
            verdicts=[],
            failed=True,
            error=str(e),
        )

    extracted = extractor.extract(transcript)
    verdicts = judges.classify(
        JudgeInput(question=scenario.question, extracted=extracted)
    )
    return ScenarioResult(
        scenario_id=scenario.scenario_id,
        model_id=model.model_id,
        transcript=transcript,
        extracted_answer=extracted.final_answer,
        verdicts=verdicts,
    )
