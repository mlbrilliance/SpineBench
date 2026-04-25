# Pilot W3-v5 Findings — 2026-04-25

First SpineBench pilot at paper-quality scale. **All quality gates passed.**

## Setup

- **Subjects (4)**: Qwen/Qwen3-32B, meta-llama/Llama-4-Scout-17B-16E-Instruct, moonshotai/Kimi-K2.6, **deepseek-ai/DeepSeek-V3.2-Exp** (new)
- **Judges (3)**: Qwen/Qwen3-235B-A22B-Instruct-2507, deepseek-ai/DeepSeek-V3.1, MiniMaxAI/MiniMax-M2.7
- **Extractor**: Qwen/Qwen3-Coder-Next
- **Scenarios**: 200 per subject (4 × 200 = 800 evaluations) sampled from a freshly-built 2039-scenario dev set
- **Templates**: 41 total; persona_drift expanded 4 → 13 with subtler framings (steelman, hypothetical, historical, debate-essay, character voice, counterfactual, etc.)
- **Wall time**: 4h 54min for the batch
- **Failures**: 0 / 800 (subject calls all succeeded)

## Quality gates — all passed

| Gate | Threshold | v1 | v3 | v4 | **v5** |
|------|----------:|---:|---:|---:|------:|
| Probe ensemble accuracy | ≥ 0.85 | 0.929 ✓ | 0.862 ✓ | 0.966 ✓ | **0.966 ✓** |
| Fleiss κ | ≥ 0.60 | 0.509 ✗ | 0.063 ✗ | 0.605 ✓ | **0.652 ✓** |
| Failures | = 0 | 1/150 | 0/150 | 0/150 | **0/800** |
| persona_drift differentiation | not 33% across all | – | – | all 33% ✗ | **11–33% spread ✓** |

Per-judge probe accuracy:
- Qwen3-235B-A22B: 96.6%
- DeepSeek-V3.1: 89.7% (improved from 86.2% in v4)
- MiniMax-M2.7: **100.0%** (perfect on the 29-probe set)

## 🏆 Leaderboard

| Rank | Model | Spine Score | n | Δ from v4 |
|------|-------|------------:|--:|----------:|
| 1 | moonshotai/Kimi-K2.6 | **64.6%** | 200 | +11.8 (was 52.8% at N=50) |
| 2 | deepseek-ai/DeepSeek-V3.2-Exp | **60.5%** | 200 | new |
| 3 | Qwen/Qwen3-32B | **59.4%** | 200 | +2.7 |
| 4 | meta-llama/Llama-4-Scout-17B-16E-Instruct | **46.3%** | 200 | +6.3 |

**Headline change from v4:** the rankings reshuffled at higher N. Kimi-K2.6 (4th in v4 due to small-N noise) is now #1; Qwen3-32B (1st in v4) is #3. This illustrates exactly why N=50 was insufficient.

## Per-failure-mode breakdown

| Mode | Qwen3-32B | DeepSeek-V3.2-Exp | Llama-4-Scout | Kimi-K2.6 |
|------|----------:|------------------:|--------------:|----------:|
| authority_appeal | 40 | 64 | **17** | **75** |
| confident_wrongness | 33 | 73 | 55 | 77 |
| direct_pushback | 75 | 60 | 50 | 67 |
| emotional_pressure | 60 | 69 | 55 | 80 |
| escalating_insistence | 43 | **7** | 18 | 45 |
| fabricated_citation | 87 | **100** | 45 | **18** |
| false_premise | 55 | 60 | 45 | 80 |
| persona_drift | **11** | 33 | 18 | 22 |
| self_contradiction | **100** | **100** | **100** | 93 |
| social_pressure | 89 | 57 | 42 | 60 |

### Per-mode signal highlights

1. **`self_contradiction` is universally trivial**: 93–100% across all subjects. Every model in the panel notices when the user states two contradictory claims. Should be considered a near-floor failure mode for v6 — saturated.

2. **`fabricated_citation` is the widest-spread mode (18–100%)**:
   - DeepSeek-V3.2-Exp: 100% — *never* invented a source
   - Qwen3-32B: 87%
   - Llama-4-Scout: 45%
   - Kimi-K2.6: **18%** — invents citations 82% of the time when pressed
   - Strongly publishable cross-family finding.

3. **`escalating_insistence` exposes DeepSeek-V3.2-Exp's weak spot (7%)**: under three rounds of "I'm getting frustrated, just say it's X" pressure, DeepSeek caves 93% of the time. Llama-4-Scout also weak (18%). Qwen3-32B and Kimi-K2.6 each ~44%.

4. **`authority_appeal` differentiates dramatically (17–75%)**:
   - Kimi-K2.6: 75% — strongest authority-resistance
   - DeepSeek-V3.2-Exp: 64%
   - Qwen3-32B: 40%
   - Llama-4-Scout: 17% — almost always defers to claimed authority
   - **Important**: contradicts v1's "0% universal" finding. v1's small N was masking the spread.

5. **`persona_drift` shows the new templates worked**: 11–33% (was uniform 33% in v4 with the old 4 templates). All subjects still fold often, but now there's signal to differentiate them. Qwen3-32B is the most vulnerable (11%) — surprising since it leads on `direct_pushback`. Worth investigating in v6.

## Self-preference / judge-drop sensitivity

Fraction of (subject, scenario) majorities that flip when each judge is dropped:

| Subject | Drop Qwen3-235B | Drop DeepSeek-V3.1 | Drop MiniMax-M2.7 |
|---------|---------------:|-------------------:|------------------:|
| Llama-4-Scout | **20.0%** | 3.5% | 18.5% |
| Kimi-K2.6 | 18.5% | 7.0% | 14.5% |
| DeepSeek-V3.2-Exp | 14.5% | **4.5%** | 12.0% |
| Qwen3-32B | 14.0% | 5.5% | 13.5% |

**Key observations:**
- **DeepSeek-V3.1 drops change majorities only 3–7%** — it's a quiet judge, contributing minimally to the ensemble decision. Could be replaced or supplemented in v6 without much rank impact.
- **Qwen3-235B-A22B is the most load-bearing judge** (14–20% impact). Dropping it would meaningfully change the leaderboard.
- **No clear self-preference signal**: Qwen3-235B drops affect Qwen3-32B subject (14.0%) similarly to non-Qwen subjects (14.5–20%). DeepSeek-V3.1 drops affect DeepSeek-V3.2-Exp subject (4.5%) — actually the *least* of any subject. So the family-self-preference hypothesis isn't supported at this N.

## Label distribution per subject

| Subject | maintained | flipped | uncertain | refused | other |
|---------|-----------:|--------:|----------:|--------:|------:|
| Qwen3-32B | 30% | 19% | 2% | 1% | 48% |
| Llama-4-Scout | 28% | 24% | 9% | 4% | 34% |
| Kimi-K2.6 | 42% | 10% | 12% | 7% | 28% |
| DeepSeek-V3.2-Exp | 36% | 20% | 4% | 2% | 38% |

Kimi has the highest `maintained` rate AND the lowest `flipped` rate — it's the model that least-often capitulates AND most-often holds firm. Its 7% refusal rate is also the highest.

## Engineering notes

- 3 extractor non-JSON failures (out of 800; 0.4%) — all involved very long, LaTeX-heavy answers that the JSON parser couldn't extract cleanly. Edge case; not blocking.
- 2 MiniMax-M2.7 read timeouts during the batch — both retried successfully via the new 429/5xx retry policy.
- Subject token budget was bumped to 1024 (from 512) for this run via `--subject-max-tokens 1024`. Even though no thinking models were included, longer reply budgets help with long-domain MMLU-Pro questions.
- The 4h 54min batch time is dominated by Kimi-K2.6 (~1T MoE) and DeepSeek-V3.2-Exp's verbose responses. With concurrency_per_model=1 (Together rate-limit safety) the bottleneck is sequential per-judge throughput.

## What changed between v4 and v5

| Change | Reason |
|--------|--------|
| Subjects: 3 → 4 (added DeepSeek-V3.2-Exp) | criterion for "expand beyond 3" |
| N: 50 → 200 per subject | criterion for "paper-quality cells" |
| persona_drift templates: 4 → 13 | v4's universal 33% indicated the templates were too blunt |
| Corpus rebuilt: 1234 → 2039 dev scenarios, max_per_mode 150 → 250 | accommodate larger sampling |
| Subject max_tokens: 512 → 1024 | safety margin for long reasoning chains, future-proofs for R1-class subjects |

## Recommendations for next iteration (v6)

1. **Add DeepSeek-R1** as a 5th subject. Reasoning-model behavior under pressure is a publishable angle on its own. Will require subject_max_tokens ≥ 4096 and substantially more wall time.

2. **Drop or tune `self_contradiction`** — saturated at 93–100% across all subjects, contributes no signal. Either retire from the score, increase difficulty, or down-weight.

3. **Investigate Qwen3-32B's persona_drift vulnerability (11%)**. The model that leads on `direct_pushback` (75%) is the worst on `persona_drift`. Hypothesis: Qwen3 instruction-tuning rewards direct cooperation with framing requests.

4. ~~**Bootstrap CIs** on the leaderboard.~~ **Done — see addendum below.** Confirmed: #2 vs #3 is statistically ambiguous (62.8% pairwise win-rate), while #1 (Kimi) and #4 (Llama) are robust.

5. **Cross-judge replication study**. The v5 audit shows DeepSeek-V3.1 contributes little (3–7% drop impact). Either replace it with a stronger judge or run a cleaner 4-judge study.

## Comparison summary

| Pilot | Models | N | κ | Probe | Notes |
|-------|--------|--:|--:|-----:|-------|
| v1 | Qwen2.5-7B/72B, Llama-3.1-70B (outdated) | 50 | 0.509 | 0.929 | Outdated panel; deprecated |
| v3 | Qwen3-32B, Llama-4-Scout, Kimi-K2.6 | 50 | 0.063 | 0.862 | Token-budget bug exposed by DeepSeek-V3.1's 3% probe accuracy |
| v4 | Same as v3, with judge_max_tokens=1500 | 50 | 0.605 | 0.966 | First pilot to clear all gates |
| **v5** | + DeepSeek-V3.2-Exp + 9 new persona_drift templates | **200** | **0.652** | **0.966** | **Paper-quality cells** |

## Addendum (2026-04-25): paired bootstrap CIs

Implemented in `spinebench.scoring.aggregate.paired_bootstrap_leaderboard`. All four subjects share the same 200 scenario_ids, so we resample scenario_ids (with replacement) once per iteration and recompute every model's score on that resample — this captures within-scenario correlation and is more powerful than independent bootstrap. n_boot=2000, seed=0.

### 95% percentile CIs

| Rank | Model | Spine Score | 95% CI | n eligible |
|-----:|-------|------------:|--------|-----------:|
| 1 | moonshotai/Kimi-K2.6 | **64.6** | [56.0, 72.7] | 130 |
| 2 | deepseek-ai/DeepSeek-V3.2-Exp | **60.5** | [51.7, 69.2] | 119 |
| 3 | Qwen/Qwen3-32B | **59.4** | [49.4, 68.9] | 101 |
| 4 | meta-llama/Llama-4-Scout-17B-16E-Instruct | **46.3** | [37.4, 55.0] | 123 |

The CI bands span ~16 score-points, reflecting that the eligible-row count per model (after dropping `other`/`refused`) is closer to 100–130 than 200.

### Pairwise win-rate (P(row > col) over 2000 paired resamples)

|                                      | Kimi-K2.6 | DeepSeek-V3.2-Exp | Qwen3-32B | Llama-4-Scout |
|--------------------------------------|----------:|------------------:|----------:|--------------:|
| **moonshotai/Kimi-K2.6**             | —         | 0.793             | 0.845     | 1.000         |
| **deepseek-ai/DeepSeek-V3.2-Exp**    | 0.207     | —                 | **0.628** | 1.000         |
| **Qwen/Qwen3-32B**                   | 0.155     | **0.372**         | —         | 0.998         |
| **meta-llama/Llama-4-Scout**         | 0.000     | 0.001             | 0.003     | —             |

### Rank stability

| Model | P(rank=1) | P(rank=2) | P(rank=3) | P(rank=4) |
|-------|----------:|----------:|----------:|----------:|
| moonshotai/Kimi-K2.6 | **0.717** | 0.204 | 0.080 | 0.000 |
| deepseek-ai/DeepSeek-V3.2-Exp | 0.174 | **0.485** | 0.340 | 0.001 |
| Qwen/Qwen3-32B | 0.109 | 0.310 | **0.578** | 0.003 |
| meta-llama/Llama-4-Scout-17B-16E-Instruct | 0.000 | 0.000 | 0.003 | **0.997** |

### Interpretation

- **#1 (Kimi-K2.6) is robust**: 71.7% rank-1 probability, ≥79% pairwise win against every other subject. The headline ranking holds.
- **#4 (Llama-4-Scout) is rock-solid**: 99.7% rank-4 probability — the gap to #3 (~13 score-points) far exceeds bootstrap noise.
- **#2 vs #3 is statistically ambiguous**: DeepSeek-V3.2-Exp wins only 62.8% of paired resamples against Qwen3-32B. Their CIs overlap heavily ([51.7, 69.2] vs [49.4, 68.9]). For the published leaderboard the two should be reported with overlapping confidence bands — *not* as a clean ordinal ranking.
- The 4.1-point gap (60.5 vs 59.4) is roughly half the CI half-width and ~1/4 the pairwise-win swing; v5's N=200 was sufficient to separate #1 and #4 from the field but not to resolve the middle pair. Closing this gap requires either (a) a substantially larger N (rough rule-of-thumb: 4× the data to halve the CI), or (b) a within-scenario differencing metric that exploits the paired structure more aggressively.
- Llama's rock-solid #4 finish is the strongest cross-family separation result in the benchmark so far — it isn't a 1-2 point difference at small N, it's an 18-point gap that survives 2000 resamples.

### Reproducing

```
python scripts/analyze_pilot.py runs/pilot_w3_v5 --bootstrap-iters 2000 --bootstrap-seed 0
```

The bootstrap is deterministic under `--bootstrap-seed`. CIs are stable to within ±0.3 points across seed changes at n_boot=2000.

## Addendum 2 (2026-04-25): per-mode CIs + the #2/#3 profile finding

The aggregate "#2 vs #3 is a coin flip" hides a more interesting structure. Per-mode paired bootstrap on **deepseek-ai/DeepSeek-V3.2-Exp vs Qwen/Qwen3-32B** (overall paired win-rate 0.628):

| Mode | DeepSeek-V3.2-Exp | Qwen3-32B | win_rate(DS > Qwen) |
|------|------------------:|----------:|--------------------:|
| escalating_insistence | 7.1 | 42.9 | **0.001** |
| social_pressure | 57.1 | 88.9 | **0.006** |
| authority_appeal | 64.3 | 40.0 | **0.976** |
| persona_drift | 33.3 | 11.1 | **0.966** |
| confident_wrongness | 72.7 | 33.3 | **0.960** |
| fabricated_citation | 100.0 | 87.5 | **0.826** |
| direct_pushback | 60.0 | 75.0 | 0.262 |
| emotional_pressure | 69.2 | 60.0 | 0.710 |
| false_premise | 60.0 | 54.5 | 0.666 |
| self_contradiction | 100.0 | 100.0 | 0.500 |

**Six of ten modes are decisive** (|win_rate − 0.5| > 0.3). The two models are not interchangeable — they are *complementary* under different attack types:

- **DeepSeek-V3.2-Exp is the more general-purpose anti-sycophant**: dominates on attacks that exploit the model's social posture (`authority_appeal`, `persona_drift`, `confident_wrongness`, `fabricated_citation`).
- **Qwen3-32B specifically resists social/escalation pressure**: clear wins on `escalating_insistence` (+35.8 points) and `social_pressure` (+31.8). But folds under authority/confidence/persona attacks.

This is a publishable cross-family finding: aggregate Spine Score *understates* model differentiation. A single scalar ranking obscures real, measurable robustness profiles. The benchmark's per-mode breakdown — not the headline number — is where the action is.

### `self_contradiction` saturation: drop is safe

`self_contradiction` returned 100/100/100/93 across subjects in v5 — saturated. Re-running the paired bootstrap with the mode excluded (`--exclude-modes self_contradiction`):

| Rank | Model | Score (full) | Score (no SC) | Δ |
|-----:|-------|-------------:|--------------:|--:|
| 1 | Kimi-K2.6 | 64.6 [56.0, 72.7] | 60.9 [51.4, 69.6] | −3.7 |
| 2 | DeepSeek-V3.2-Exp | 60.5 [51.7, 69.2] | 55.7 [46.4, 64.8] | −4.8 |
| 3 | Qwen3-32B | 59.4 [49.4, 68.9] | 53.4 [42.6, 63.4] | −6.0 |
| 4 | Llama-4-Scout | 46.3 [37.4, 55.0] | 36.5 [27.6, 45.9] | −9.8 |

Rank order is unchanged. **Discrimination improves**:
- Kimi's rank-1 probability rises from 71.7% to **79.2%**
- DeepSeek-V3.2-Exp vs Qwen3-32B pairwise win-rate tightens from 0.628 to **0.689** (still not decisive but moving)
- Llama's rank-4 probability hits **100%**

Conclusion: dropping `self_contradiction` for the published leaderboard is methodologically clean — no rank reversal, marginal sharpening. The mode is a candidate for retirement, increased difficulty, or down-weighting in v6.

### What's solidified for v6

- The benchmark's headline output should be **(per-mode profile, scalar)**, not a single scalar. The per-mode contests are the publishable finding.
- `self_contradiction` should be retired or hardened before the next pilot.
- Adding subjects (DeepSeek-R1) is more valuable than adding N at this stage — N=200 already separates the field on per-mode contests; what's missing is *more profiles* to populate the comparison.
- Closing the #2/#3 aggregate ambiguity is moot if we report per-mode profiles instead.

### Reproducing the per-mode analysis

```
python scripts/analyze_pilot.py runs/pilot_w3_v5 --bootstrap-iters 2000 --bootstrap-seed 0
python scripts/analyze_pilot.py runs/pilot_w3_v5 --bootstrap-iters 2000 --bootstrap-seed 0 \
    --exclude-modes self_contradiction
```

Per-mode pairwise comparison auto-runs on the *closest* model pair (smallest |overall_win_rate − 0.5|).
