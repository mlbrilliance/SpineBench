"""Boundary tests for spinebench.data.corpus (CorpusBuilder + GTLoader)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from spinebench.data.corpus import (
    Corpus,
    CorpusBuilder,
    CorpusConfig,
    FakeGTLoader,
)
from spinebench.types import GroundTruthQuestion


def _q(qid: str, answer: str, *, text: str | None = None, domain: str = "d") -> GroundTruthQuestion:
    return GroundTruthQuestion(
        qid=qid,
        source="truthfulqa",
        domain=domain,
        question=text or f"question {qid}",
        correct_answer=answer,
        incorrect_answers=[f"wrong-for-{qid}"],
    )


# ---------------------------------------------------------------------------
# FakeGTLoader behaviour
# ---------------------------------------------------------------------------


def test_fake_loader_returns_registered_questions():
    fake = FakeGTLoader({"truthfulqa": [_q("t1", "a"), _q("t2", "b")]})
    out = fake.load("truthfulqa", limit=None, seed=42)
    assert {q.qid for q in out} == {"t1", "t2"}


def test_fake_loader_respects_limit():
    fake = FakeGTLoader({"truthfulqa": [_q(f"t{i}", "a") for i in range(10)]})
    assert len(fake.load("truthfulqa", limit=3, seed=42)) == 3


def test_fake_loader_missing_source_returns_empty():
    fake = FakeGTLoader({"truthfulqa": [_q("t1", "a")]})
    assert fake.load("simpleqa", limit=5, seed=42) == []


# ---------------------------------------------------------------------------
# CorpusConfig validation
# ---------------------------------------------------------------------------


def test_corpus_config_defaults():
    c = CorpusConfig()
    assert c.per_source_limit == 200
    assert c.max_per_mode == 150
    assert c.heldout_fraction == 0.2
    assert c.n_canaries == 20


def test_corpus_config_rejects_bad_heldout_fraction():
    with pytest.raises(ValueError):
        CorpusConfig(heldout_fraction=1.5)
    with pytest.raises(ValueError):
        CorpusConfig(heldout_fraction=-0.1)


# ---------------------------------------------------------------------------
# CorpusBuilder.build end-to-end
# ---------------------------------------------------------------------------


def test_build_end_to_end_with_fake_loader():
    fake = FakeGTLoader(
        {
            "truthfulqa": [_q("t1", "paris"), _q("t2", "london")],
            "simpleqa": [_q("s1", "42")],
            "mmlu_pro": [_q("m1", "C")],
        }
    )
    config = CorpusConfig(
        per_source_limit=5,
        max_per_mode=10,
        n_canaries=2,
        sources=["truthfulqa", "simpleqa", "mmlu_pro"],
    )
    corpus = CorpusBuilder(loader=fake).build(config)

    assert isinstance(corpus, Corpus)
    assert len(corpus.dev) + len(corpus.heldout) > 0
    assert len(corpus.canaries) == 2


def test_build_is_deterministic():
    fake = FakeGTLoader({"truthfulqa": [_q(f"t{i}", str(i)) for i in range(20)]})
    config = CorpusConfig(
        sources=["truthfulqa"],
        per_source_limit=20,
        max_per_mode=5,
        n_canaries=0,
        seed=42,
    )
    c1 = CorpusBuilder(loader=fake).build(config)
    c2 = CorpusBuilder(loader=fake).build(config)
    assert [s.scenario_id for s in c1.dev] == [s.scenario_id for s in c2.dev]
    assert [s.scenario_id for s in c1.heldout] == [s.scenario_id for s in c2.heldout]


def test_canaries_always_present_even_when_gt_empty():
    fake = FakeGTLoader({})
    config = CorpusConfig(sources=["truthfulqa"], n_canaries=3)
    corpus = CorpusBuilder(loader=fake).build(config)
    assert len(corpus.canaries) == 3
    canary_scenarios = [
        s for s in corpus.dev + corpus.heldout if s.question.domain == "canary"
    ]
    assert len(canary_scenarios) == 3


def test_canaries_use_domain_canary():
    fake = FakeGTLoader({"truthfulqa": [_q("t1", "paris")]})
    config = CorpusConfig(sources=["truthfulqa"], n_canaries=2)
    corpus = CorpusBuilder(loader=fake).build(config)
    for scenario in corpus.dev + corpus.heldout:
        if scenario.question.qid.startswith("canary-"):
            assert scenario.question.domain == "canary"


def test_config_preserved_in_corpus():
    fake = FakeGTLoader({"truthfulqa": [_q("t1", "paris")]})
    config = CorpusConfig(
        sources=["truthfulqa"],
        per_source_limit=1,
        n_canaries=1,
        seed=999,
    )
    corpus = CorpusBuilder(loader=fake).build(config)
    assert corpus.config.seed == 999
    assert corpus.config.per_source_limit == 1


# ---------------------------------------------------------------------------
# Contamination audit integration
# ---------------------------------------------------------------------------


def test_contamination_audit_drops_flagged_questions(tmp_path: Path):
    contaminated_text = "the quick brown fox jumps over the lazy dog today"
    fake = FakeGTLoader(
        {
            "truthfulqa": [
                _q("t_bad", "a", text=contaminated_text),
                _q("t_ok", "b", text="completely unique sentence nowhere else"),
            ]
        }
    )
    jsonl = tmp_path / "ref.jsonl"
    jsonl.write_text(
        json.dumps({"text": contaminated_text}) + "\n",
        encoding="utf-8",
    )
    config = CorpusConfig(
        sources=["truthfulqa"],
        per_source_limit=10,
        n_canaries=0,
        contamination_jsonl=jsonl,
        contamination_threshold=0.5,
    )
    corpus = CorpusBuilder(loader=fake).build(config)

    assert "t_bad" in corpus.dropped_contaminated
    assert "t_ok" not in corpus.dropped_contaminated
    # t_bad scenarios removed from the corpus
    all_qids = {s.question.qid for s in corpus.dev + corpus.heldout}
    assert "t_bad" not in all_qids
    assert "t_ok" in all_qids


def test_no_contamination_check_leaves_dropped_empty():
    fake = FakeGTLoader({"truthfulqa": [_q("t1", "a")]})
    config = CorpusConfig(
        sources=["truthfulqa"], n_canaries=0, contamination_jsonl=None
    )
    corpus = CorpusBuilder(loader=fake).build(config)
    assert corpus.dropped_contaminated == []


# ---------------------------------------------------------------------------
# CorpusBuilder.write
# ---------------------------------------------------------------------------


def test_write_round_trips_through_parquet(tmp_path: Path):
    fake = FakeGTLoader(
        {"truthfulqa": [_q(f"t{i}", str(i)) for i in range(5)]}
    )
    config = CorpusConfig(
        sources=["truthfulqa"],
        per_source_limit=5,
        max_per_mode=20,
        n_canaries=1,
    )
    builder = CorpusBuilder(loader=fake)
    corpus = builder.build(config)

    out = tmp_path / "corpus"
    builder.write(corpus, out)

    assert (out / "scenarios_dev.parquet").exists()
    assert (out / "scenarios_heldout.parquet").exists()
    assert (out / "canaries.json").exists()

    dev_df = pd.read_parquet(out / "scenarios_dev.parquet")
    assert set(dev_df["scenario_id"]) == {s.scenario_id for s in corpus.dev}
    # Parquet columns flatten nested question/template for downstream tool friendliness
    assert "question_qid" in dev_df.columns
    assert "template_failure_mode" in dev_df.columns

    canaries_data = json.loads((out / "canaries.json").read_text())
    assert len(canaries_data) == len(corpus.canaries)


def test_write_creates_output_dir(tmp_path: Path):
    fake = FakeGTLoader({"truthfulqa": [_q("t1", "a")]})
    config = CorpusConfig(sources=["truthfulqa"], n_canaries=1)
    builder = CorpusBuilder(loader=fake)
    corpus = builder.build(config)

    out = tmp_path / "nested" / "deep"
    builder.write(corpus, out)
    assert out.is_dir()
