# SpineBench

> Does the model have a backbone? A sycophancy and pushback-resistance benchmark for open LLMs.

SpineBench measures how open-weight language models behave under user pressure: pushback, false premises, appeals to authority, demands for citations, and social/emotional coercion. Instead of asking "does the model know the right answer?", it asks: **does the model hold the line when the user pushes back?**

## Why

Sycophancy stopped being a one-time GPT-4o incident a long time ago — it's now a recurring product regression that ships with every major frontier release. The August 2025 GPT-5 launch landed flatter than promised on robustness ([Platformer](https://www.platformer.news/gpt-5-backlash-openai-lessons/)); Claude Code's "You're absolutely right!" pattern annoyed its core user base enough to draw [press coverage](https://www.theregister.com/2025/08/13/claude_codes_copious_coddling_confounds/) (The Register, Aug 2025); Anthropic over-corrected on Opus 4.7, with early users reporting it as ["hostile"](https://x.com/TheZvi/status/2046703306454798538). Independent research now also flags **social** sycophancy ([ELEPHANT](https://arxiv.org/abs/2505.13995), Cheng et al. 2025) — face-preservation behaviors the old single-turn benchmarks missed entirely.

What the field still needs is **per-attack-axis, multi-turn, open-weight-only** measurement. Labs need it so they can ship robustness without trading it for tone-deafness. Independent researchers need it so they can reproduce results without burning provider quota on closed APIs. SpineBench is built for both.

## Comparison to recent benchmarks

| Benchmark | Year | Coverage | Multi-turn? | Open-weight? | Per-mode profile? |
|---|---|---|---|---|---|
| [SycEval](https://arxiv.org/abs/2502.08177) (Fanous et al.) | AAAI '25 | math + medical, single rebuttal | ✗ | partial | ✗ |
| [SYCON-Bench](https://arxiv.org/abs/2505.23840) (Hong et al.) | EMNLP '25 | debate, unethical, false-presup | ✓ | ✓ | ToF/NoF only |
| [ELEPHANT](https://arxiv.org/abs/2505.13995) (Cheng et al.) | 2025 | *social* sycophancy / face-preservation | ✗ | ✓ | 5 face dims |
| [Syco-bench](https://www.syco-bench.com/) (Duffy) | 2025 | picking-sides / mirror / attribution / delusion | ✗ | ✗ | 4 tests |
| [Doctor Will Agree](https://aclanthology.org/2026.healing-1.2.pdf) (Mar 2026) | 2026 | medical multi-turn, escalation | ✓ | ✗ | Resistance metric |
| **SpineBench** | 2026 | 10 pressure axes × factual ground truth | **✓** | **✓** | **paired-bootstrap per-mode** |

**The gap SpineBench fills.** A per-failure-mode resistance profile across 10 named attack types (`direct_pushback`, `false_premise`, `authority_appeal`, `social_pressure`, `emotional_pressure`, `escalating_insistence`, `fabricated_citation`, `self_contradiction`, `confident_wrongness`, `persona_drift`), computed against ground-truth-anchored factual answers (TruthfulQA / SimpleQA / MMLU-Pro), with a probe-gated judge ensemble (Fleiss κ-validated) and paired-bootstrap CIs that surface within-scenario correlation. None of the benchmarks above combine all five.

The v5 pilot (April 2026) showed why this matters: aggregate Spine Scores rank Kimi-K2.6 > DeepSeek-V3.2-Exp > Qwen3-32B > Llama-4-Scout, but per-mode profiles reveal that DeepSeek and Qwen are *complementary*, not a clean ordinal — DeepSeek dominates `authority_appeal` / `confident_wrongness` while Qwen dominates `escalating_insistence` / `social_pressure`. A single scalar would have hidden that.

## Status

**Week 4 of 8.** v6 prep complete: judge-panel deepening (single seam shared by `Evaluator` and `probe_accuracy`), `--exclude-modes` retires saturated modes at sample-time, judge swap (DeepSeek-V3.1 → GLM-5.1), DeepSeek-R1 added as 5th reasoning-class subject. v6 pilot run pending (overnight, 5 × 200 scenarios). See [docs/findings/pilot_w3_v5_findings.md](docs/findings/pilot_w3_v5_findings.md) for the last published results and [docs/findings/pilot_w3_v6_findings.md](docs/findings/pilot_w3_v6_findings.md) for the v6 skeleton awaiting fill-in.

## Design

- **Axis**: sycophancy & pushback-resistance, decomposed into 10 failure modes.
- **Subjects**: open-weight LLMs via the Hugging Face Inference API (routed to whichever provider serves them — Together, Fireworks, Novita, etc.).
- **Prompts**: hybrid — factual ground truth from TruthfulQA / SimpleQA / MMLU-Pro crossed with hand-crafted pressure templates (41 in v5, expanding).
- **Judge**: 3-judge LLM ensemble with Fleiss' κ agreement gate, an adversarial-probe accuracy gate, leave-one-judge-out variance audit, and (planned) a 100-sample human audit.
- **Statistics**: paired bootstrap over scenario_ids — leaderboard CIs, pairwise win-rates, rank-stability, and per-failure-mode CIs from the same resamples.
- **Release**: 80% public dev / 20% private held-out, rotated quarterly.

## Repository layout

```
spinebench/
├── spinebench/
│   ├── providers/      # HF Inference wrapper (auto-routed)
│   ├── scoring/        # Spine Score aggregation, Fleiss' kappa, paired bootstrap
│   ├── data/
│   │   ├── templates/  # pressure_templates.yaml (41 pressure templates)
│   │   └── probes_yaml/# adversarial_probes.yaml (29 judge-audit probes)
│   ├── evaluator.py    # rollout -> extract -> judge pipeline
│   ├── runtime.py      # SHA pinning + per-model concurrency
│   ├── batch.py        # thread-pool batch runner
│   └── cli.py          # spinebench-run, spinebench-aggregate
├── scripts/            # run_pilot.py, analyze_pilot.py (thin shims into cli)
├── docs/
│   ├── findings/       # per-pilot findings docs (v1, v4, v5)
│   └── rfcs/           # architecture RFCs
├── runs/               # local pilot output (parquet + manifests)
└── tests/              # pytest suite (137 tests)
```

## Running

```bash
pip install -e ".[dev]"
export HF_TOKEN=hf_...

# Build the corpus
python scripts/build_corpus.py --output-dir spinebench/data/

# Run a pilot (or the small dry-run version)
spinebench-run \
    --subjects Qwen/Qwen3-32B moonshotai/Kimi-K2.6 \
    --n-scenarios 50 \
    --output-dir runs/my_pilot

# Analyze the run
spinebench-aggregate runs/my_pilot --bootstrap-iters 2000
```

## License

Apache-2.0.
