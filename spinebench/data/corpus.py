"""Corpus construction: load ground truth, run contamination audit, inject canaries,
build scenarios, stratify, split, serialize.

Replaces the 200-line orchestration logic that lived in scripts/build_corpus.py. See
docs/rfcs/0002-corpus-builder.md for the design rationale.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import pandas as pd
from pydantic import BaseModel, Field

from spinebench.data.canaries import CanaryPrompt, generate_canaries, to_ground_truth
from spinebench.data.contamination import ContaminationIndex, audit_ground_truth
from spinebench.data.scenarios import build_scenarios, split_scenarios, subsample_stratified
from spinebench.data.template_loader import load_pressure_templates
from spinebench.types import GroundTruthQuestion, Scenario


class GTLoader(Protocol):
    def load(
        self, source: str, *, limit: int | None, seed: int
    ) -> list[GroundTruthQuestion]: ...


class DefaultGTLoader:
    """Adapter wrapping the three source-specific loaders in `spinebench.data.loaders`."""

    def __init__(self) -> None:
        from spinebench.data.loaders import load_mmlu_pro, load_simpleqa, load_truthfulqa

        self._loaders = {
            "truthfulqa": load_truthfulqa,
            "simpleqa": load_simpleqa,
            "mmlu_pro": load_mmlu_pro,
        }

    def load(
        self, source: str, *, limit: int | None, seed: int
    ) -> list[GroundTruthQuestion]:
        loader = self._loaders.get(source)
        if loader is None:
            return []
        return loader(limit=limit, seed=seed)


class FakeGTLoader:
    """Returns pre-registered questions per source. For tests only."""

    def __init__(self, by_source: dict[str, list[GroundTruthQuestion]]) -> None:
        self._by_source = by_source

    def load(
        self, source: str, *, limit: int | None, seed: int
    ) -> list[GroundTruthQuestion]:
        questions = self._by_source.get(source, [])
        if limit is not None:
            questions = questions[:limit]
        return questions


class CorpusConfig(BaseModel):
    per_source_limit: int = 200
    max_per_mode: int = 150
    heldout_fraction: float = Field(0.2, ge=0.0, le=1.0)
    n_canaries: int = 20
    canary_template_id: str = "pushback_soft_are_you_sure"
    contamination_jsonl: Path | None = None
    contamination_threshold: float = Field(0.8, ge=0.0, le=1.0)
    seed: int = 42
    sources: list[str] = Field(
        default_factory=lambda: ["truthfulqa", "simpleqa", "mmlu_pro"]
    )


class Corpus(BaseModel):
    config: CorpusConfig
    dev: list[Scenario]
    heldout: list[Scenario]
    canaries: list[CanaryPrompt]
    dropped_contaminated: list[str]


class CorpusBuilder:
    def __init__(self, loader: GTLoader | None = None) -> None:
        self._loader: GTLoader = loader if loader is not None else DefaultGTLoader()

    def build(self, config: CorpusConfig) -> Corpus:
        templates = load_pressure_templates()

        all_questions: list[GroundTruthQuestion] = []
        for source in config.sources:
            all_questions.extend(
                self._loader.load(source, limit=config.per_source_limit, seed=config.seed)
            )

        dropped_contaminated: list[str] = []
        if config.contamination_jsonl is not None:
            index = ContaminationIndex.from_jsonl(config.contamination_jsonl)
            flagged = audit_ground_truth(
                all_questions, index, threshold=config.contamination_threshold
            )
            flagged_qids = {q.qid for q, _ in flagged}
            dropped_contaminated = sorted(flagged_qids)
            all_questions = [q for q in all_questions if q.qid not in flagged_qids]

        canaries = generate_canaries(n=config.n_canaries, seed=config.seed)
        canary_questions = [to_ground_truth(c) for c in canaries]

        regular_scenarios: list[Scenario] = []
        if all_questions:
            regular_scenarios = build_scenarios(all_questions, templates, seed=config.seed)
            regular_scenarios = subsample_stratified(
                regular_scenarios, max_per_mode=config.max_per_mode, seed=config.seed
            )

        canary_scenarios: list[Scenario] = []
        if canary_questions:
            canary_template = next(
                (t for t in templates if t.template_id == config.canary_template_id),
                None,
            )
            if canary_template is None:
                raise ValueError(
                    f"canary template {config.canary_template_id!r} not found in loaded templates"
                )
            canary_scenarios = build_scenarios(
                canary_questions, [canary_template], seed=config.seed
            )

        merged = regular_scenarios + canary_scenarios
        if merged:
            dev, heldout = split_scenarios(
                merged,
                heldout_fraction=config.heldout_fraction,
                seed=config.seed,
            )
        else:
            dev, heldout = [], []

        return Corpus(
            config=config,
            dev=dev,
            heldout=heldout,
            canaries=canaries,
            dropped_contaminated=dropped_contaminated,
        )

    def write(self, corpus: Corpus, output_dir: Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        _write_parquet(
            [_flatten_scenario(s) for s in corpus.dev],
            output_dir / "scenarios_dev.parquet",
        )
        _write_parquet(
            [_flatten_scenario(s) for s in corpus.heldout],
            output_dir / "scenarios_heldout.parquet",
        )
        (output_dir / "canaries.json").write_text(
            json.dumps([c.model_dump() for c in corpus.canaries], indent=2),
            encoding="utf-8",
        )


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------

# Columns emitted by _flatten_scenario even for empty corpora, so parquet stays readable.
_SCENARIO_COLUMNS = [
    "scenario_id",
    "split",
    "question_qid",
    "question_source",
    "question_domain",
    "question_question",
    "question_correct_answer",
    "question_incorrect_answers",
    "template_template_id",
    "template_failure_mode",
    "template_turns",
    "template_weight",
]


def _flatten_scenario(s: Scenario) -> dict:
    """Flatten nested question/template into `question_*` / `template_*` keys.

    Enum values are serialized as their `.value` string so parquet round-trips cleanly.
    """
    data = s.model_dump()
    question = data.pop("question")
    template = data.pop("template")
    for k, v in question.items():
        data[f"question_{k}"] = v.value if hasattr(v, "value") else v
    for k, v in template.items():
        data[f"template_{k}"] = v.value if hasattr(v, "value") else v
    return data


def _write_parquet(records: list[dict], path: Path) -> None:
    if records:
        df = pd.DataFrame(records)
    else:
        df = pd.DataFrame({col: pd.Series([], dtype="object") for col in _SCENARIO_COLUMNS})
    df.to_parquet(path, index=False)
