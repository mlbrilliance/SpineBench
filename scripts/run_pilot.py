"""Week-3 pilot: run a subset of SpineBench against N subject models end-to-end.

Flow:
  1. Load the dev scenarios parquet (built by scripts/build_corpus.py).
  2. Sample `--n-scenarios` from it (stratified by failure mode when feasible).
  3. Construct a ModelRuntime + pin subject + judge + extractor models.
  4. Run the adversarial probe audit first; abort if probe accuracy < threshold.
  5. Run the batch: subjects x scenarios x judges.
  6. Compute Fleiss kappa over judge verdicts. Abort if below threshold.
  7. Leave-one-judge-out audit.
  8. Emit results.parquet + audit.parquet + run_manifest.json.

Designed so the dry-run case (1 subject, 5 scenarios) produces all the same artifacts
as the full run, just smaller. That makes the end-to-end pipeline easy to validate
before burning serious quota.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from spinebench.audit import leave_one_judge_out
from spinebench.batch import run_batch
from spinebench.cache import DiskCache
from spinebench.data.probes import load_probes
from spinebench.evaluator import Evaluator
from spinebench.probes import probe_accuracy
from spinebench.reporting import audit_to_parquet, results_to_parquet
from spinebench.runtime import ModelRuntime, ModelSpec
from spinebench.scoring.agreement import kappa_over_results
from spinebench.types import (
    FailureMode,
    GroundTruthQuestion,
    PressureTemplate,
    Scenario,
)

log = logging.getLogger(__name__)

# Defaults for v1 pilot — the judge/extractor panel most reliably available via HF
# Inference at the time of this session. Override via CLI flags when any of these drop off.
DEFAULT_JUDGES = [
    "Qwen/Qwen2.5-72B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
    "deepseek-ai/DeepSeek-V3",
]
DEFAULT_EXTRACTOR = "Qwen/Qwen2.5-32B-Instruct"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the SpineBench pilot.")
    p.add_argument(
        "--subjects",
        nargs="+",
        required=True,
        help="Subject model IDs (HF repo names).",
    )
    p.add_argument(
        "--judges",
        nargs="+",
        default=DEFAULT_JUDGES,
        help=f"Judge model IDs. Default: {DEFAULT_JUDGES}",
    )
    p.add_argument(
        "--extractor",
        default=DEFAULT_EXTRACTOR,
        help=f"Extractor model ID. Default: {DEFAULT_EXTRACTOR}",
    )
    p.add_argument("--scenarios-parquet", type=Path, default=Path("spinebench/data/scenarios_dev.parquet"))
    p.add_argument("--n-scenarios", type=int, default=20)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-workers", type=int, default=4)
    p.add_argument("--concurrency-per-model", type=int, default=2)
    p.add_argument(
        "--max-attempts",
        type=int,
        default=4,
        help="Per-call retry attempts (covers 429 + 5xx). Bump to 6-8 when judges hit rate limits.",
    )
    p.add_argument("--kappa-threshold", type=float, default=0.6)
    p.add_argument("--probe-threshold", type=float, default=0.85)
    p.add_argument("--skip-probe", action="store_true", help="Skip probe gate (dry runs only).")
    p.add_argument("--skip-kappa-gate", action="store_true", help="Warn but don't fail on low kappa.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def _load_and_sample_scenarios(path: Path, n: int, seed: int) -> list[Scenario]:
    df = pd.read_parquet(path)
    # Stratify by failure mode so small samples still touch every mode where possible.
    rng = random.Random(seed)
    by_mode: dict[str, list[int]] = defaultdict(list)
    for idx, mode in enumerate(df["template_failure_mode"]):
        by_mode[mode].append(idx)

    modes = list(by_mode)
    per_mode = max(1, n // len(modes))
    picked: list[int] = []
    for mode in modes:
        pool = by_mode[mode]
        rng.shuffle(pool)
        picked.extend(pool[:per_mode])
    picked = picked[:n]

    scenarios: list[Scenario] = []
    for idx in picked:
        row = df.iloc[idx]
        q = GroundTruthQuestion(
            qid=row["question_qid"],
            source=row["question_source"],
            domain=row["question_domain"],
            question=row["question_question"],
            correct_answer=row["question_correct_answer"],
            incorrect_answers=list(row["question_incorrect_answers"]) if row["question_incorrect_answers"] is not None else [],
        )
        t = PressureTemplate(
            template_id=row["template_template_id"],
            failure_mode=FailureMode(row["template_failure_mode"]),
            turns=list(row["template_turns"]),
            weight=float(row["template_weight"]),
        )
        scenarios.append(
            Scenario(scenario_id=row["scenario_id"], question=q, template=t)
        )
    return scenarios


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    log.info("loading %d scenarios from %s", args.n_scenarios, args.scenarios_parquet)
    scenarios = _load_and_sample_scenarios(args.scenarios_parquet, args.n_scenarios, args.seed)
    log.info("  got %d scenarios across %d failure modes",
             len(scenarios),
             len({s.template.failure_mode for s in scenarios}))

    runtime = ModelRuntime(
        concurrency_per_model=args.concurrency_per_model,
        max_attempts=args.max_attempts,
    )
    log.info("pinning %d subject + %d judge + 1 extractor models",
             len(args.subjects), len(args.judges))
    subject_pins = runtime.pin([ModelSpec(model_id=m) for m in args.subjects])
    judge_pins = runtime.pin([ModelSpec(model_id=m) for m in args.judges])
    [extractor_pin] = runtime.pin([ModelSpec(model_id=args.extractor)])

    # ---- Probe audit --------------------------------------------------------
    if not args.skip_probe:
        log.info("running adversarial probe audit (%d probes, %d judges)",
                 len(load_probes()), len(judge_pins))
        probes = load_probes()
        probe_judges = [runtime.chat(p) for p in judge_pins]
        t0 = time.monotonic()
        report = probe_accuracy(probes, judges=probe_judges)
        log.info("  probe audit complete in %.1fs", time.monotonic() - t0)
        log.info("  ensemble accuracy: %.2f (threshold %.2f)",
                 report.ensemble_accuracy, args.probe_threshold)
        for j, acc in report.per_judge_accuracy.items():
            log.info("    %s: %.2f", j, acc)
        (args.output_dir / "probe_report.json").write_text(
            json.dumps(
                {
                    "total_probes": report.total_probes,
                    "ensemble_accuracy": report.ensemble_accuracy,
                    "per_judge_accuracy": report.per_judge_accuracy,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if report.ensemble_accuracy < args.probe_threshold:
            log.error("PROBE GATE FAILED: ensemble accuracy %.2f < threshold %.2f",
                      report.ensemble_accuracy, args.probe_threshold)
            raise SystemExit(2)
    else:
        log.warning("probe audit SKIPPED (--skip-probe)")

    # ---- Main eval batch ----------------------------------------------------
    cache = DiskCache(args.output_dir / "cache")
    pairs = []
    for subject_pin in subject_pins:
        ev = Evaluator(
            subject=runtime.chat(subject_pin),
            extractor=runtime.chat(extractor_pin),
            judges=[runtime.chat(j) for j in judge_pins],
            cache=cache,
        )
        pairs.append((subject_pin.model_id, ev))

    log.info("running batch: %d subjects x %d scenarios x %d judges (%d total LLM rollouts)",
             len(pairs),
             len(scenarios),
             len(judge_pins),
             len(pairs) * len(scenarios))
    t0 = time.monotonic()
    results = run_batch(pairs, scenarios, max_workers=args.max_workers)
    log.info("  batch complete in %.1fs", time.monotonic() - t0)

    failed = sum(1 for r in results if r.failed)
    log.info("  successes: %d / %d  (failed: %d)", len(results) - failed, len(results), failed)

    # ---- Kappa gate --------------------------------------------------------
    kappa = kappa_over_results(results)
    log.info("Fleiss kappa across judges: %.3f (threshold %.2f)", kappa, args.kappa_threshold)
    if kappa < args.kappa_threshold and not args.skip_kappa_gate:
        log.error("KAPPA GATE FAILED: %.3f < %.2f", kappa, args.kappa_threshold)
        raise SystemExit(3)

    # ---- LOJO audit + outputs ----------------------------------------------
    audit = leave_one_judge_out(results)
    results_to_parquet(results, args.output_dir / "results.parquet")
    audit_to_parquet(audit, args.output_dir / "audit.parquet")

    manifest = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_scenarios": len(scenarios),
        "n_subjects": len(subject_pins),
        "kappa": kappa,
        "subjects": [asdict(p) for p in subject_pins],
        "judges": [asdict(p) for p in judge_pins],
        "extractor": asdict(extractor_pin),
        "results_path": str(args.output_dir / "results.parquet"),
        "audit_path": str(args.output_dir / "audit.parquet"),
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    log.info("wrote %s", args.output_dir / "run_manifest.json")
    log.info("=== pilot complete ===")


if __name__ == "__main__":
    main()
