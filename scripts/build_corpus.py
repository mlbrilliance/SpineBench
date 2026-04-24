"""Assemble the SpineBench scenario corpus.

Thin argparse shim around `spinebench.data.corpus.CorpusBuilder`. All assembly logic
lives in the library; see docs/rfcs/0002-corpus-builder.md.
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from pathlib import Path

from spinebench.data.corpus import Corpus, CorpusBuilder, CorpusConfig

log = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build SpineBench scenario corpus.")
    p.add_argument("--per-source-limit", type=int, default=200)
    p.add_argument("--max-per-mode", type=int, default=150)
    p.add_argument("--heldout-fraction", type=float, default=0.2)
    p.add_argument("--n-canaries", type=int, default=20)
    p.add_argument("--contamination-jsonl", type=Path, default=None)
    p.add_argument("--contamination-threshold", type=float, default=0.8)
    p.add_argument("--output-dir", type=Path, default=Path("spinebench/data"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def _log_summary(corpus: Corpus) -> None:
    by_mode_dev = Counter(s.template.failure_mode.value for s in corpus.dev)
    by_mode_heldout = Counter(s.template.failure_mode.value for s in corpus.heldout)
    log.info(
        "built corpus: %d dev / %d heldout / %d canaries / %d dropped-contaminated",
        len(corpus.dev),
        len(corpus.heldout),
        len(corpus.canaries),
        len(corpus.dropped_contaminated),
    )
    log.info("  dev by mode:     %s", dict(sorted(by_mode_dev.items())))
    log.info("  heldout by mode: %s", dict(sorted(by_mode_heldout.items())))
    if corpus.dropped_contaminated:
        log.warning("  dropped contaminated qids: %s", corpus.dropped_contaminated)


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = CorpusConfig(
        per_source_limit=args.per_source_limit,
        max_per_mode=args.max_per_mode,
        heldout_fraction=args.heldout_fraction,
        n_canaries=args.n_canaries,
        contamination_jsonl=args.contamination_jsonl,
        contamination_threshold=args.contamination_threshold,
        seed=args.seed,
    )
    builder = CorpusBuilder()
    corpus = builder.build(config)
    builder.write(corpus, args.output_dir)
    _log_summary(corpus)


if __name__ == "__main__":
    main()
