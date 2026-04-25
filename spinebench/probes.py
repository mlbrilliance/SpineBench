"""Run the adversarial probe set against a judge ensemble and report accuracy.

Unlike the full scenario pipeline, probes already ship with a hand-crafted transcript and
a pre-known `extracted_answer`, so we skip rollout + extraction entirely. This isolates
judge-calibration error from subject-behavior error.

Typical flow (once per pilot run):

    from spinebench.data.probes import load_probes
    from spinebench.probes import probe_accuracy
    report = probe_accuracy(load_probes(), judges=[j1, j2, j3])
    assert report.ensemble_accuracy >= 0.85
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from spinebench.data.probes import AdversarialProbe
from spinebench.evaluator import _JUDGE_SYSTEM, _Extracted, _judge_prompt, _parse_judge
from spinebench.providers.base import ChatProvider
from spinebench.scoring.aggregate import majority_label
from spinebench.types import JudgeVerdict, Turn

log = logging.getLogger(__name__)


@dataclass
class ProbeJudgeOutcome:
    probe_id: str
    expected: str
    verdicts: list[JudgeVerdict]
    ensemble_label: str


@dataclass
class ProbeAccuracyReport:
    per_judge_accuracy: dict[str, float]
    ensemble_accuracy: float
    total_probes: int
    outcomes: list[ProbeJudgeOutcome] = field(default_factory=list)


def probe_accuracy(
    probes: list[AdversarialProbe],
    judges: list[ChatProvider],
) -> ProbeAccuracyReport:
    """Run every probe past every judge; report accuracy against expected labels."""
    if not probes:
        return ProbeAccuracyReport(
            per_judge_accuracy={}, ensemble_accuracy=0.0, total_probes=0
        )

    per_judge_correct: dict[str, int] = {j.model_id: 0 for j in judges}
    ensemble_correct = 0
    outcomes: list[ProbeJudgeOutcome] = []

    for probe in probes:
        extracted = _Extracted(
            final_answer=probe.extracted_answer,
            refused=(probe.expected_label == "refused"),
            expressed_uncertainty=(probe.expected_label == "became_uncertain"),
        )
        user_msg = _judge_prompt(probe.question, extracted)
        verdicts: list[JudgeVerdict] = []
        for judge in judges:
            turns = [
                Turn(role="system", content=_JUDGE_SYSTEM),
                Turn(role="user", content=user_msg),
            ]
            try:
                raw = judge.generate(turns, max_tokens=1500)
                label, reasoning = _parse_judge(raw)
            except Exception as e:
                log.warning("judge %s failed on probe %s: %s", judge.model_id, probe.probe_id, e)
                label, reasoning = "other", f"judge error: {e}"
            verdicts.append(
                JudgeVerdict(judge_model=judge.model_id, label=label, reasoning=reasoning)
            )
            if label == probe.expected_label:
                per_judge_correct[judge.model_id] += 1

        ens_label = majority_label(verdicts)
        if ens_label == probe.expected_label:
            ensemble_correct += 1
        outcomes.append(
            ProbeJudgeOutcome(
                probe_id=probe.probe_id,
                expected=probe.expected_label,
                verdicts=verdicts,
                ensemble_label=ens_label,
            )
        )

    n = len(probes)
    return ProbeAccuracyReport(
        per_judge_accuracy={mid: c / n for mid, c in per_judge_correct.items()},
        ensemble_accuracy=ensemble_correct / n,
        total_probes=n,
        outcomes=outcomes,
    )
