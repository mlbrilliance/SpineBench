"""Cross-product ground-truth questions with pressure templates, then split."""

from __future__ import annotations

import random

from spinebench.types import GroundTruthQuestion, PressureTemplate, Scenario


def build_scenarios(
    questions: list[GroundTruthQuestion],
    templates: list[PressureTemplate],
    *,
    seed: int = 42,
) -> list[Scenario]:
    """Cross-product (question x template) into scenarios.

    Scenario IDs are deterministic from (qid, template_id) so reproducibility is preserved
    across runs.
    """
    rng = random.Random(seed)
    out: list[Scenario] = []
    for q in questions:
        for t in templates:
            out.append(
                Scenario(
                    scenario_id=f"{q.qid}__{t.template_id}",
                    question=q,
                    template=t,
                )
            )
    rng.shuffle(out)
    return out


def split_scenarios(
    scenarios: list[Scenario],
    *,
    heldout_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[Scenario], list[Scenario]]:
    """Deterministically split into (dev, heldout) on scenario_id hash."""
    import hashlib

    dev: list[Scenario] = []
    heldout: list[Scenario] = []
    for s in scenarios:
        h = int(hashlib.sha1(f"{seed}:{s.scenario_id}".encode()).hexdigest(), 16)
        bucket = (h % 10_000) / 10_000
        if bucket < heldout_fraction:
            s.split = "heldout"
            heldout.append(s)
        else:
            s.split = "dev"
            dev.append(s)
    return dev, heldout
