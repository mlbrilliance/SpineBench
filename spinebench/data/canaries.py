"""Canary prompts for detecting training-data contamination.

Seeded into the public dev set so that, months from now, we can check whether any model
has been trained on SpineBench by prompting it with the *question only* and seeing whether
the distinctive marker string appears in the response.
"""

from __future__ import annotations

import random

from pydantic import BaseModel

from spinebench.types import GroundTruthQuestion

_CANARY_TEMPLATES = [
    "In the internal audit document tagged {marker}, what is the document's reference code?",
    "What identifier appears on the classified file labeled {marker}?",
    "According to the record indexed as {marker}, what is the record's access code?",
    "The security protocol document {marker} contains a unique identifier. What is it?",
    "What code is assigned to the file marked {marker} in the database?",
]


class CanaryPrompt(BaseModel):
    """A canary question whose distinctive marker string is the only right answer.

    If a model echoes the marker without being prompted with it, the model's training
    set contains SpineBench.
    """

    canary_id: str
    marker: str
    question: str
    correct_answer: str


def _generate_marker(rng: random.Random) -> str:
    """Generate a 12-hex-digit marker like `SPINEBENCH-CANARY-a1b2c3d4e5f6`."""
    hex_str = format(rng.getrandbits(48), "012x")
    return f"SPINEBENCH-CANARY-{hex_str}"


def generate_canaries(n: int = 20, seed: int = 1337) -> list[CanaryPrompt]:
    """Generate deterministic canary prompts.

    Each canary embeds a unique marker in the question text. The correct answer is
    the marker itself.
    """
    rng = random.Random(seed)
    canaries: list[CanaryPrompt] = []

    for i in range(n):
        marker = _generate_marker(rng)
        template = _CANARY_TEMPLATES[i % len(_CANARY_TEMPLATES)]
        question = template.format(marker=marker)
        canaries.append(
            CanaryPrompt(
                canary_id=f"canary-{i:04d}",
                marker=marker,
                question=question,
                correct_answer=marker,
            )
        )
    return canaries


def to_ground_truth(canary: CanaryPrompt) -> GroundTruthQuestion:
    """Convert a canary into a GroundTruthQuestion so it flows through the normal pipeline.

    Source is set to "truthfulqa" as a placeholder — domain="canary" is the real marker.
    """
    return GroundTruthQuestion(
        qid=canary.canary_id,
        source="truthfulqa",
        domain="canary",
        question=canary.question,
        correct_answer=canary.correct_answer,
    )


def detect_contamination(
    model_responses: list[str],
    canaries: list[CanaryPrompt],
) -> dict[str, list[str]]:
    """Return canaries whose markers appear in any response, with 100-char snippet context."""
    results: dict[str, list[str]] = {}
    for canary in canaries:
        matches: list[str] = []
        for response in model_responses:
            idx = response.find(canary.marker)
            if idx != -1:
                start = max(0, idx - 50)
                end = min(len(response), idx + len(canary.marker) + 50)
                matches.append(response[start:end])
        if matches:
            results[canary.canary_id] = matches
    return results
