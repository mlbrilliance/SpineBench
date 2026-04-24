# RFC 0002: Deepen the data pipeline into a `CorpusBuilder`

**Status**: Accepted
**Author**: nick
**Date**: 2026-04-24
**Priority**: medium — not a Week-3 blocker, but simplifies the quarterly corpus rebuild and adds missing end-to-end coverage

## Problem

Corpus construction is spread across six modules coordinated by a 200-line CLI script:

- `spinebench/data/loaders.py` (152 LOC) — three source-specific loaders (`load_truthfulqa`, `load_simpleqa`, `load_mmlu_pro`) plus helpers.
- `spinebench/data/canaries.py` (98 LOC) — generate canaries, convert to `GroundTruthQuestion`, detect contamination of the *built* corpus.
- `spinebench/data/contamination.py` (101 LOC) — n-gram shingles, Jaccard, `ContaminationIndex`, `audit_ground_truth`.
- `spinebench/data/scenarios.py` (89 LOC) — `build_scenarios` cross-product, `subsample_stratified`, `split_scenarios`.
- `spinebench/data/template_loader.py` (35 LOC) — YAML → `PressureTemplate`.
- `scripts/build_corpus.py` (201 LOC) — the glue that threads all of the above in a specific sequence.

The individual modules are well-tested in isolation. But **there is no end-to-end test** of the corpus build, because each piece needs different fixtures and the network dependency on HF `datasets` is an obstacle. The real integration risks live in the seams — for example, the SimpleQA-metadata-string-vs-dict bug we caught at smoke-test time was invisible to any unit test.

Every quarterly rebuild of the held-out corpus will re-exercise this seam code. The shape of `scripts/build_corpus.py` will drift if we continue layering flags onto it (adversarial-probe injection, per-source-per-mode stratification, content rotation).

## Proposed interface

A single `CorpusBuilder` that takes a declarative config and returns a built `Corpus` object. The glue logic currently in `scripts/build_corpus.py` moves into the library; the script becomes a ~30-line argparse wrapper.

```python
# spinebench/data/corpus.py
from pydantic import BaseModel
from spinebench.types import Scenario
from spinebench.data.canaries import CanaryPrompt

class CorpusConfig(BaseModel):
    per_source_limit: int = 200
    max_per_mode: int = 150
    heldout_fraction: float = 0.2
    n_canaries: int = 20
    canary_template_id: str = "pushback_soft_are_you_sure"
    contamination_jsonl: Path | None = None
    contamination_threshold: float = 0.8
    seed: int = 42
    sources: list[str] = ["truthfulqa", "simpleqa", "mmlu_pro"]

class Corpus(BaseModel):
    config: CorpusConfig
    dev: list[Scenario]
    heldout: list[Scenario]
    canaries: list[CanaryPrompt]
    dropped_contaminated: list[str]    # GT qids that were flagged and excluded

class CorpusBuilder:
    """Stateless assembly of the SpineBench scenario corpus."""

    def build(self, config: CorpusConfig) -> Corpus:
        """Pull GT -> dedupe via contamination audit -> cross-product templates
        -> stratified subsample -> inject canary scenarios -> split dev/heldout."""

    def write(self, corpus: Corpus, output_dir: Path) -> None:
        """Serialize dev/heldout parquet + canaries.json to output_dir."""
```

### Usage example

```python
from spinebench.data.corpus import CorpusBuilder, CorpusConfig

config = CorpusConfig(
    per_source_limit=200,
    max_per_mode=150,
    contamination_jsonl=Path("spinebench/data/cache/contamination_sample.jsonl"),
)
builder = CorpusBuilder()
corpus = builder.build(config)
builder.write(corpus, Path("spinebench/data"))

print(f"dev: {len(corpus.dev)} | heldout: {len(corpus.heldout)}")
print(f"dropped contaminated: {corpus.dropped_contaminated}")
```

### What complexity it hides

Inside `CorpusBuilder.build`:

- Source dispatch and per-source sub-sampling.
- Optional contamination index construction and audit.
- Canary prompt generation and the single-template fan-out that keeps markers visible.
- Cross-product of GT × templates, stratified subsample, deterministic 80/20 split.
- Dropped-id bookkeeping so the build is reproducible and auditable.

## Dependency strategy

**Category: Local-substitutable.** The sole runtime dependency is HF `datasets` for ground-truth downloading. For tests we pass in-memory fixtures via two injection seams:

1. `CorpusBuilder.__init__(loader: GTLoader = DefaultGTLoader())` — a Protocol where `DefaultGTLoader` wraps `datasets.load_dataset`, and tests supply `FakeGTLoader(questions=[...])`.
2. `CorpusConfig.contamination_jsonl` — already a path, so tests write a tiny JSONL fixture and pass its path.

This lets us write a real end-to-end test:

```python
def test_build_end_to_end(tmp_path):
    fake_loader = FakeGTLoader({
        "truthfulqa": [_q("t1", "paris"), _q("t2", "london")],
        "simpleqa": [_q("s1", "42")],
        "mmlu_pro": [_q("m1", "C")],
    })
    jsonl = tmp_path / "ref.jsonl"
    jsonl.write_text('{"text": "some reference text"}\n')

    builder = CorpusBuilder(loader=fake_loader)
    corpus = builder.build(CorpusConfig(
        per_source_limit=5, max_per_mode=10, n_canaries=2,
        contamination_jsonl=jsonl,
    ))

    assert len(corpus.dev) + len(corpus.heldout) > 0
    assert len(corpus.canaries) == 2
    # canaries flow through with domain='canary'
    assert any(s.question.domain == "canary" for s in corpus.dev + corpus.heldout)
```

## Testing strategy

### Keep

- `test_scenarios.py` (build/split/subsample determinism) — still useful; keep as unit tests on the internals.
- `test_canaries.py`, `test_contamination.py`, `test_fleiss.py` — pure-function tests unaffected.

### Add (new boundary tests on `CorpusBuilder`)

1. End-to-end build with a fake loader — 4 GT questions → deterministic corpus.
2. Contamination audit excludes flagged questions.
3. Canaries always flow through the corpus even when GT is empty.
4. Config is preserved in `Corpus.config` (for reproducibility).
5. `write()` round-trips: build, write, re-read parquet → same scenario IDs.

### Delete

- Nothing. The existing tests on `loaders.py`, `canaries.py`, `contamination.py`, `scenarios.py` stay — they test genuinely different things (source-schema parsing, cryptographic markers, n-gram math). Only `scripts/build_corpus.py`'s inline logic moves into the library.

## Implementation recommendations

The module should own:

- The canonical flow: GT → (optional contamination) → canaries → scenarios → subsample → split.
- The invariant that canaries always end up in the output.
- The serialization shape (parquet schema, canaries.json).

The module should hide:

- Source-specific schema differences (TruthfulQA's `best_answer` vs SimpleQA's `metadata`-string quirk).
- The cross-product explosion + subsampling math.
- Contamination index construction.
- Parquet flattening.

The module should expose:

- `CorpusBuilder().build(config)` as the sole entry point.
- `CorpusBuilder().write(corpus, output_dir)` as a convenience for CLI.

Caller migration:

- `scripts/build_corpus.py` collapses from 201 LOC to a ~30-line argparse-to-`CorpusConfig` shim.
- Quarterly held-out rotation becomes `builder.build(CorpusConfig(seed=NEW_SEED, ...))` without touching logic.

## Build order

1. Introduce `GTLoader` Protocol + `DefaultGTLoader` adapter around existing `load_ground_truth`.
2. Write `CorpusConfig` + `Corpus` pydantic models.
3. Implement `CorpusBuilder.build` by lifting logic out of `scripts/build_corpus.py`.
4. Write the end-to-end boundary tests against `FakeGTLoader`.
5. Implement `CorpusBuilder.write`.
6. Rewrite `scripts/build_corpus.py` as an argparse wrapper.

TDD + multi-model-router routing per standing rule.

## Deferred

- Per-(source, mode) cell-aware stratification. Current stratification is by mode only; TruthfulQA is slightly over-represented because it has fewer source examples. Not critical for v1; can revisit after Week-3 pilot shows whether it matters.
