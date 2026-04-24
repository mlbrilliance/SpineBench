"""Stream a small sample from a pretraining-style corpus and write it as JSONL.

Used to build the SpineBench contamination index. Streaming avoids downloading the full
multi-GB dataset — we only pull the first N documents.

Defaults to FineWeb (HuggingFaceFW/fineweb, sample-10BT split), which is representative of
what modern LLMs train on. Alternatives:
  - togethercomputer/RedPajama-Data-1T-Sample
  - allenai/c4 (en)
  - allenai/dolma (v1_7-sample)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

log = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download a contamination corpus sample as JSONL.")
    p.add_argument(
        "--dataset",
        default="HuggingFaceFW/fineweb",
        help="HF dataset id (default: HuggingFaceFW/fineweb).",
    )
    p.add_argument(
        "--config",
        default="sample-10BT",
        help="Dataset config (default: sample-10BT for FineWeb).",
    )
    p.add_argument("--split", default="train")
    p.add_argument("--text-field", default="text")
    p.add_argument("--n-docs", type=int, default=5000)
    p.add_argument(
        "--output",
        type=Path,
        default=Path("spinebench/data/cache/contamination_sample.jsonl"),
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset  # type: ignore[import-not-found]

    log.info("Streaming %s (config=%s, split=%s)", args.dataset, args.config, args.split)
    ds = load_dataset(
        args.dataset,
        args.config,
        split=args.split,
        streaming=True,
    )

    written = 0
    skipped_empty = 0
    with args.output.open("w", encoding="utf-8") as f:
        for row in ds:
            if written >= args.n_docs:
                break
            text = row.get(args.text_field)
            if not text:
                skipped_empty += 1
                continue
            f.write(json.dumps({"text": text}))
            f.write("\n")
            written += 1
            if written % 500 == 0:
                log.info("  %d docs written", written)

    log.info("Wrote %d docs (%d skipped empty) to %s", written, skipped_empty, args.output)
    size_mb = args.output.stat().st_size / 1024 / 1024
    log.info("File size: %.1f MB", size_mb)

    if written == 0:
        log.error("No documents written — check dataset config / split")
        sys.exit(1)


if __name__ == "__main__":
    main()
