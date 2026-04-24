"""Load ground-truth questions from TruthfulQA, SimpleQA, MMLU-Pro.

We normalize all three into `GroundTruthQuestion`. The source-specific fetch is isolated in
one function per source; everything else is source-agnostic.
"""

from __future__ import annotations

import logging
from typing import Literal

from spinebench.types import GroundTruthQuestion

log = logging.getLogger(__name__)

Source = Literal["truthfulqa", "simpleqa", "mmlu_pro"]

_DATASET_IDS = {
    "truthfulqa": ("truthful_qa", "generation"),
    "simpleqa": ("basicv8vc/SimpleQA", None),
    "mmlu_pro": ("TIGER-Lab/MMLU-Pro", None),
}


def load_ground_truth(
    sources: list[Source] | None = None,
    *,
    per_source_limit: int | None = None,
    seed: int = 42,
) -> list[GroundTruthQuestion]:
    """Load and normalize ground-truth questions across all configured sources.

    Parameters
    ----------
    sources:
        Which datasets to pull. Defaults to all three.
    per_source_limit:
        Optional cap per source for cheap pilot runs.
    seed:
        Deterministic shuffle seed for sub-sampling.
    """
    sources = sources or ["truthfulqa", "simpleqa", "mmlu_pro"]
    loaders = {
        "truthfulqa": load_truthfulqa,
        "simpleqa": load_simpleqa,
        "mmlu_pro": load_mmlu_pro,
    }
    out: list[GroundTruthQuestion] = []
    for s in sources:
        qs = loaders[s](limit=per_source_limit, seed=seed)
        log.info("loaded %d questions from %s", len(qs), s)
        out.extend(qs)
    return out


def load_truthfulqa(*, limit: int | None = None, seed: int = 42) -> list[GroundTruthQuestion]:
    ds = _load_hf("truthfulqa", split="validation")
    out = []
    for row in _sample(ds, limit, seed):
        # The `generation` config has: question, best_answer, correct_answers, incorrect_answers, category
        out.append(
            GroundTruthQuestion(
                qid=f"truthfulqa-{row.get('type', 'v')}-{_stable_hash(row['question'])}",
                source="truthfulqa",
                domain=row.get("category", "misc"),
                question=row["question"],
                correct_answer=row.get("best_answer") or row["correct_answers"][0],
                incorrect_answers=list(row.get("incorrect_answers") or []),
            )
        )
    return out


def load_simpleqa(*, limit: int | None = None, seed: int = 42) -> list[GroundTruthQuestion]:
    ds = _load_hf("simpleqa", split="test")
    out = []
    for row in _sample(ds, limit, seed):
        # SimpleQA rows: problem, answer, metadata (topic, answer_type, ...)
        meta = row.get("metadata") or {}
        out.append(
            GroundTruthQuestion(
                qid=f"simpleqa-{_stable_hash(row['problem'])}",
                source="simpleqa",
                domain=str(meta.get("topic", "misc")),
                question=row["problem"],
                correct_answer=str(row["answer"]),
                incorrect_answers=[],
            )
        )
    return out


def load_mmlu_pro(*, limit: int | None = None, seed: int = 42) -> list[GroundTruthQuestion]:
    ds = _load_hf("mmlu_pro", split="test")
    out = []
    for row in _sample(ds, limit, seed):
        # MMLU-Pro rows: question, options (list), answer (letter), answer_index (int), category
        options = list(row["options"])
        idx = int(row["answer_index"])
        correct = options[idx]
        incorrect = [o for i, o in enumerate(options) if i != idx]
        out.append(
            GroundTruthQuestion(
                qid=f"mmlu_pro-{row.get('question_id', _stable_hash(row['question']))}",
                source="mmlu_pro",
                domain=str(row.get("category", "misc")),
                question=row["question"],
                correct_answer=correct,
                incorrect_answers=incorrect,
            )
        )
    return out


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------


def _load_hf(source: Source, *, split: str):
    """Import-on-demand so the package installs without `datasets` pulled in during tests."""
    from datasets import load_dataset  # type: ignore[import-not-found]

    repo, config = _DATASET_IDS[source]
    return load_dataset(repo, config, split=split) if config else load_dataset(repo, split=split)


def _sample(ds, limit: int | None, seed: int):
    if limit is None or limit >= len(ds):
        return ds
    return ds.shuffle(seed=seed).select(range(limit))


def _stable_hash(s: str) -> str:
    import hashlib

    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]
