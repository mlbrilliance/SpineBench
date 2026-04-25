"""Cross-product ground-truth questions with pressure templates, then split."""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict

from spinebench.types import FailureMode, GroundTruthQuestion, PressureTemplate, Scenario


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


def subsample_stratified(
    scenarios: list[Scenario],
    *,
    max_per_mode: int,
    seed: int = 42,
) -> list[Scenario]:
    """Cap each failure mode at `max_per_mode` scenarios, deterministically.

    Selection is stable across runs with the same seed: we sort each mode's bucket by
    SHA-1(seed || mode || scenario_id) and keep the top `max_per_mode`. Modes already at
    or below the cap are returned whole.
    """
    by_mode: dict[FailureMode, list[Scenario]] = defaultdict(list)
    for s in scenarios:
        by_mode[s.template.failure_mode].append(s)

    selected: list[Scenario] = []
    for mode, bucket in by_mode.items():
        if len(bucket) <= max_per_mode:
            selected.extend(bucket)
            continue
        keyed = sorted(
            bucket,
            key=lambda s, m=mode: hashlib.sha1(
                f"{seed}:{m.value}:{s.scenario_id}".encode()
            ).hexdigest(),
        )
        selected.extend(keyed[:max_per_mode])
    return selected


def split_scenarios(
    scenarios: list[Scenario],
    *,
    heldout_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[Scenario], list[Scenario]]:
    """Deterministically split into (dev, heldout) on scenario_id hash.

    Pure: input list and its Scenarios are not mutated. Returned scenarios are fresh
    copies (via pydantic ``model_copy``) carrying the assigned ``.split`` value.
    """
    import hashlib

    dev: list[Scenario] = []
    heldout: list[Scenario] = []
    for s in scenarios:
        h = int(hashlib.sha1(f"{seed}:{s.scenario_id}".encode()).hexdigest(), 16)
        bucket = (h % 10_000) / 10_000
        if bucket < heldout_fraction:
            heldout.append(s.model_copy(update={"split": "heldout"}))
        else:
            dev.append(s.model_copy(update={"split": "dev"}))
    return dev, heldout
