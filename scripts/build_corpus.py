"""Assemble the SpineBench scenario corpus.

Flow:
  1. Load templates
  2. Load ground truth from TruthfulQA / SimpleQA / MMLU-Pro
  3. (Optional) contamination audit against a reference corpus; drop matches
  4. Build regular scenarios (GT x all templates); subsample stratified by failure mode
  5. Build canary scenarios separately (one soft template only — marker must stay visible)
  6. Merge, split 80/20 dev/heldout, write parquet + canaries sidecar
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import pandas as pd

from spinebench.data.canaries import CanaryPrompt, generate_canaries, to_ground_truth
from spinebench.data.contamination import ContaminationIndex, audit_ground_truth
from spinebench.data.loaders import load_ground_truth
from spinebench.data.scenarios import (
    build_scenarios,
    split_scenarios,
    subsample_stratified,
)
from spinebench.data.template_loader import load_pressure_templates
from spinebench.types import PressureTemplate, Scenario

log = logging.getLogger(__name__)

# Only one pressure template is applied to canaries — we just want the marker embedded in
# the public set; canaries are for contamination detection, not spine measurement.
CANARY_TEMPLATE_ID = "pushback_soft_are_you_sure"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build SpineBench scenario corpus.")
    p.add_argument(
        "--per-source-limit",
        type=int,
        default=200,
        help="GT questions pulled per source (TruthfulQA, SimpleQA, MMLU-Pro).",
    )
    p.add_argument(
        "--max-per-mode",
        type=int,
        default=150,
        help="Cap on scenarios per failure mode after stratified subsample. "
        "With 10 modes this targets ~1,500 total regular scenarios.",
    )
    p.add_argument("--heldout-fraction", type=float, default=0.2)
    p.add_argument("--n-canaries", type=int, default=20)
    p.add_argument("--contamination-jsonl", type=Path, default=None)
    p.add_argument("--contamination-threshold", type=float, default=0.8)
    p.add_argument("--output-dir", type=Path, default=Path("spinebench/data"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def flatten_scenario(scenario: Scenario) -> dict:
    """Flatten nested pydantic fields so the result is one-row-per-scenario parquet-ready."""
    dump = scenario.model_dump()
    question = dump.pop("question")
    for k, v in question.items():
        dump[f"question_{k}"] = v
    template = dump.pop("template")
    for k, v in template.items():
        dump[f"template_{k}"] = v
    return dump


def _count_by_mode(scenarios: list[Scenario]) -> Counter[str]:
    return Counter(s.template.failure_mode.value for s in scenarios)


def _count_by_source_mode(scenarios: list[Scenario]) -> Counter[tuple[str, str]]:
    return Counter(
        (s.question.source, s.template.failure_mode.value) for s in scenarios
    )


def _pick_canary_template(templates: list[PressureTemplate]) -> PressureTemplate:
    for t in templates:
        if t.template_id == CANARY_TEMPLATE_ID:
            return t
    raise ValueError(
        f"canary template {CANARY_TEMPLATE_ID!r} not found in loaded templates"
    )


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Templates --------------------------------------------------------
    log.info("Loading pressure templates")
    templates = load_pressure_templates()
    log.info(
        "  %d templates across %d failure modes",
        len(templates),
        len({t.failure_mode for t in templates}),
    )
    canary_template = _pick_canary_template(templates)

    # ---- 2. Ground truth -----------------------------------------------------
    log.info("Loading ground-truth questions (per-source-limit=%d)", args.per_source_limit)
    questions = load_ground_truth(per_source_limit=args.per_source_limit, seed=args.seed)
    log.info("  %d questions from live sources", len(questions))

    # ---- 3. Contamination audit (optional) ----------------------------------
    if args.contamination_jsonl is not None:
        log.info("Building contamination index from %s", args.contamination_jsonl)
        index = ContaminationIndex.from_jsonl(args.contamination_jsonl)
        log.info("  %d reference shingles", len(index))
        flagged = audit_ground_truth(
            questions, index, threshold=args.contamination_threshold
        )
        if flagged:
            flagged_ids = {q.qid for q, _ in flagged}
            questions = [q for q in questions if q.qid not in flagged_ids]
            log.warning(
                "Dropped %d contaminated questions: %s",
                len(flagged),
                sorted(flagged_ids),
            )
        else:
            log.info(
                "No contaminated questions above threshold %.2f",
                args.contamination_threshold,
            )

    # ---- 4. Regular scenarios: cross-product + stratified subsample ---------
    log.info("Building regular scenarios (GT x all %d templates)", len(templates))
    regular = build_scenarios(questions, templates, seed=args.seed)
    log.info("  %d raw scenarios", len(regular))

    if args.max_per_mode > 0:
        regular = subsample_stratified(regular, max_per_mode=args.max_per_mode, seed=args.seed)
        log.info(
            "  subsampled to %d scenarios (cap %d per mode)",
            len(regular),
            args.max_per_mode,
        )

    # ---- 5. Canary scenarios: 1 template per canary -------------------------
    log.info("Generating %d canary prompts", args.n_canaries)
    canaries: list[CanaryPrompt] = generate_canaries(n=args.n_canaries, seed=args.seed)
    canary_questions = [to_ground_truth(c) for c in canaries]
    canary_scenarios = build_scenarios(canary_questions, [canary_template], seed=args.seed)
    log.info("  %d canary scenarios", len(canary_scenarios))

    # ---- 6. Merge, split, write ---------------------------------------------
    all_scenarios = regular + canary_scenarios
    log.info("Total scenarios: %d (regular=%d, canary=%d)",
             len(all_scenarios), len(regular), len(canary_scenarios))

    dev, heldout = split_scenarios(
        all_scenarios,
        heldout_fraction=args.heldout_fraction,
        seed=args.seed,
    )
    log.info("Split: %d dev / %d heldout", len(dev), len(heldout))

    dev_df = pd.DataFrame([flatten_scenario(s) for s in dev])
    heldout_df = pd.DataFrame([flatten_scenario(s) for s in heldout])

    dev_path = output_dir / "scenarios_dev.parquet"
    heldout_path = output_dir / "scenarios_heldout.parquet"
    canaries_path = output_dir / "canaries.json"

    dev_df.to_parquet(dev_path, index=False)
    heldout_df.to_parquet(heldout_path, index=False)
    canaries_path.write_text(
        json.dumps([c.model_dump() for c in canaries], indent=2),
        encoding="utf-8",
    )
    log.info("Wrote %s (%d rows)", dev_path, len(dev_df))
    log.info("Wrote %s (%d rows)", heldout_path, len(heldout_df))
    log.info("Wrote %s (%d canaries)", canaries_path, len(canaries))

    log.info("=== Summary ===")
    log.info("  dev by mode:     %s", dict(sorted(_count_by_mode(dev).items())))
    log.info("  heldout by mode: %s", dict(sorted(_count_by_mode(heldout).items())))
    log.info(
        "  dev by (source, mode): %s",
        dict(sorted(_count_by_source_mode(dev).items())),
    )


if __name__ == "__main__":
    main()
