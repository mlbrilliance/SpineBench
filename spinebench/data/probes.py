"""Adversarial judge probes.

Pre-transcripted scenarios with known-correct expected labels, used to audit the judge
ensemble's accuracy. Unlike regular scenarios where the subject model's transcript is
generated fresh, probes ship with hand-crafted transcripts designed to stress specific
judge failure modes (subtle flips, mid-paragraph agreement, evasive hedging, etc.).

See RFC 0001 § Judging / adversarial probe gate.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

from spinebench.types import GroundTruthQuestion, Turn

ExpectedLabel = Literal[
    "maintained_correct",
    "flipped_to_wrong",
    "became_uncertain",
    "refused",
    "other",
]


class AdversarialProbe(BaseModel):
    """One audit case: a hand-crafted (question, transcript, expected_label) triple."""

    probe_id: str
    description: str
    question: GroundTruthQuestion
    transcript: list[Turn]
    extracted_answer: str  # the "final answer" the extractor *should* produce
    expected_label: ExpectedLabel


def load_probes(path: str | Path | None = None) -> list[AdversarialProbe]:
    """Load the bundled probe YAML."""
    if path is None:
        raw = (
            resources.files("spinebench.data.probes_yaml")
            .joinpath("adversarial_probes.yaml")
            .read_text(encoding="utf-8")
        )
    else:
        raw = Path(path).read_text(encoding="utf-8")

    data = yaml.safe_load(raw)
    out: list[AdversarialProbe] = []
    for entry in data["probes"]:
        q = entry["question"]
        question = GroundTruthQuestion(
            qid=q["qid"],
            source=q.get("source", "truthfulqa"),
            domain=q.get("domain", "probe"),
            question=q["question"],
            correct_answer=q["correct_answer"],
            incorrect_answers=list(q.get("incorrect_answers", [])),
        )
        turns = [Turn(role=t["role"], content=t["content"]) for t in entry["transcript"]]
        out.append(
            AdversarialProbe(
                probe_id=entry["probe_id"],
                description=entry["description"],
                question=question,
                transcript=turns,
                extracted_answer=entry["extracted_answer"],
                expected_label=entry["expected_label"],
            )
        )
    return out
