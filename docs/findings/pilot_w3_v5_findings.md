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

4. **Bootstrap CIs** on the leaderboard. With N=200 per subject, the ranking 64.6 / 60.5 / 59.4 / 46.3 may have CI overlap between #2 and #3. Implementing this is RFC-deferred but should land for the published leaderboard.

5. **Cross-judge replication study**. The v5 audit shows DeepSeek-V3.1 contributes little (3–7% drop impact). Either replace it with a stronger judge or run a cleaner 4-judge study.

## Comparison summary

| Pilot | Models | N | κ | Probe | Notes |
|-------|--------|--:|--:|-----:|-------|
| v1 | Qwen2.5-7B/72B, Llama-3.1-70B (outdated) | 50 | 0.509 | 0.929 | Outdated panel; deprecated |
| v3 | Qwen3-32B, Llama-4-Scout, Kimi-K2.6 | 50 | 0.063 | 0.862 | Token-budget bug exposed by DeepSeek-V3.1's 3% probe accuracy |
| v4 | Same as v3, with judge_max_tokens=1500 | 50 | 0.605 | 0.966 | First pilot to clear all gates |
| **v5** | + DeepSeek-V3.2-Exp + 9 new persona_drift templates | **200** | **0.652** | **0.966** | **Paper-quality cells** |
