"""CLI entry points. Minimal stubs for Week 1 — wiring real batch runs comes Week 3."""

from __future__ import annotations

import argparse
import logging


def run() -> None:
    parser = argparse.ArgumentParser(description="Run SpineBench against a model.")
    parser.add_argument("--model", required=True, help="HF model id, e.g. Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--scenarios", type=int, default=10, help="How many scenarios to run")
    parser.add_argument("--split", choices=["dev", "heldout"], default="dev")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # TODO(week-3): wire up HFInferenceProvider -> build_scenarios -> run_scenario loop.
    print(
        f"[stub] would run {args.scenarios} {args.split} scenarios against {args.model} "
        "— implementation lands in week 3"
    )


def aggregate() -> None:
    parser = argparse.ArgumentParser(description="Aggregate raw results into a Spine Score.")
    parser.add_argument("results", help="Path to raw results parquet")
    parser.add_argument("-o", "--output", default="leaderboard.parquet")
    args = parser.parse_args()
    # TODO(week-5): pandas-driven aggregation over ScenarioResult parquet.
    print(f"[stub] would aggregate {args.results} -> {args.output}")
