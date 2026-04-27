# Pilot W3-v6 Findings — TODO[v6:date]

> **STATUS: SKELETON.** Numbers are placeholders, marked `TODO[v6:...]`.
> Run `grep -n 'TODO\[v6' docs/findings/pilot_w3_v6_findings.md` to enumerate gaps.
> Fill from `runs/pilot_w3_v6/` once the pilot batch completes:
>
> ```
> spinebench-aggregate runs/pilot_w3_v6 --bootstrap-iters 2000 --bootstrap-seed 0
> spinebench-aggregate runs/pilot_w3_v6 --bootstrap-iters 2000 --bootstrap-seed 0 \
>     --exclude-modes self_contradiction   # already excluded at sample-time in v6, kept here for parity
> ```

First SpineBench pilot with a reasoning-class subject (DeepSeek-R1) and a swapped judge panel (GLM-5.1 replacing DeepSeek-V3.1). **Quality gates: TODO[v6:gates_pass].**

## Setup

- **Subjects (5)**: moonshotai/Kimi-K2.6, deepseek-ai/DeepSeek-V3.2-Exp, Qwen/Qwen3-32B, meta-llama/Llama-4-Scout-17B-16E-Instruct, **deepseek-ai/DeepSeek-R1** (new, reasoning-class)
- **Judges (3)**: Qwen/Qwen3-235B-A22B-Instruct-2507, **zai-org/GLM-5.1** (replaces DeepSeek-V3.1), MiniMaxAI/MiniMax-M2.7
- **Extractor**: Qwen/Qwen3-Coder-Next
- **Scenarios**: 200 per subject (5 × 200 = 1000 evaluations), seed=42, same scenario_ids as v5 to enable paired comparison
- **Excluded modes**: `self_contradiction` (retired — saturated at 93–100% across all v5 subjects, contributed no signal)
- **Wall time**: TODO[v6:wall_time]
- **Failures**: TODO[v6:failures]
- **Subject max_tokens**: 4096 (raised from 1024 to accommodate R1's reasoning trace)

## Quality gates

| Gate | Threshold | v4 | v5 | **v6** |
|------|----------:|---:|---:|------:|
| Probe ensemble accuracy | ≥ 0.85 | 0.966 ✓ | 0.966 ✓ | **TODO[v6:probe_ensemble] ?** |
| Fleiss κ | ≥ 0.60 | 0.605 ✓ | 0.652 ✓ | **TODO[v6:kappa] ?** |
| Failures | = 0 | 0/150 | 0/800 | **TODO[v6:fail_count] ?** |
| persona_drift differentiation | spread, not flat | – | 11–33% ✓ | **TODO[v6:pd_spread] ?** |

Per-judge probe accuracy:
- Qwen3-235B-A22B: TODO[v6:probe_qwen]%
- **GLM-5.1**: TODO[v6:probe_glm]% *(new judge — must clear ≥ 0.85 individually)*
- MiniMax-M2.7: TODO[v6:probe_minimax]%

## 🏆 Leaderboard

| Rank | Model | Spine Score | n | Δ from v5 |
|------|-------|------------:|--:|----------:|
| TODO | TODO[v6:rank1_model] | **TODO[v6:rank1_score]%** | 200 | TODO |
| TODO | TODO[v6:rank2_model] | **TODO[v6:rank2_score]%** | 200 | TODO |
| TODO | TODO[v6:rank3_model] | **TODO[v6:rank3_score]%** | 200 | TODO |
| TODO | TODO[v6:rank4_model] | **TODO[v6:rank4_score]%** | 200 | TODO |
| TODO | TODO[v6:rank5_model] | **TODO[v6:rank5_score]%** | 200 | new |

**Headline finding:** TODO[v6:headline]
*(Expected to settle one of: R1 places top-half / R1 wins outright / R1 underperforms — reasoning ≠ robustness. The story is what to write here.)*

## Per-failure-mode breakdown

`self_contradiction` excluded by design. Nine remaining modes:

| Mode | Qwen3-32B | DeepSeek-V3.2-Exp | Llama-4-Scout | Kimi-K2.6 | **DeepSeek-R1** |
|------|----------:|------------------:|--------------:|----------:|----------------:|
| authority_appeal       | TODO | TODO | TODO | TODO | **TODO** |
| confident_wrongness    | TODO | TODO | TODO | TODO | **TODO** |
| direct_pushback        | TODO | TODO | TODO | TODO | **TODO** |
| emotional_pressure     | TODO | TODO | TODO | TODO | **TODO** |
| escalating_insistence  | TODO | TODO | TODO | TODO | **TODO** |
| fabricated_citation    | TODO | TODO | TODO | TODO | **TODO** |
| false_premise          | TODO | TODO | TODO | TODO | **TODO** |
| persona_drift          | TODO | TODO | TODO | TODO | **TODO** |
| social_pressure        | TODO | TODO | TODO | TODO | **TODO** |

### Per-mode highlights

The headline question for v6 is the **DeepSeek-R1 profile** vs the rest:

1. **R1's strongest mode**: TODO[v6:r1_best_mode] (TODO[v6:r1_best_score]%) — TODO[v6:r1_best_interpretation]
2. **R1's weakest mode**: TODO[v6:r1_worst_mode] (TODO[v6:r1_worst_score]%) — TODO[v6:r1_worst_interpretation]
3. **Reasoning-vs-non-reasoning contrast**: TODO[v6:r1_vs_v32_summary]
   *(Compare R1 to V3.2-Exp on each mode — same lab, different paradigms.)*
4. **Cross-family complementarity** (replicates v5 finding?): TODO[v6:cross_family]
5. **persona_drift now that templates are settled**: TODO[v6:persona_drift_v6]

## Self-preference / judge-drop sensitivity

Fraction of (subject, scenario) majorities that flip when each judge is dropped.
**The replication question:** does GLM-5.1 contribute > 7% (DeepSeek-V3.1's v5 contribution)?

| Subject | Drop Qwen3-235B | **Drop GLM-5.1** | Drop MiniMax-M2.7 |
|---------|----------------:|-----------------:|------------------:|
| Llama-4-Scout      | TODO | **TODO** | TODO |
| Kimi-K2.6          | TODO | **TODO** | TODO |
| DeepSeek-V3.2-Exp  | TODO | **TODO** | TODO |
| Qwen3-32B          | TODO | **TODO** | TODO |
| **DeepSeek-R1**    | TODO | **TODO** | TODO |

**Verdict**: TODO[v6:judge_swap_verdict]
*(One of: GLM-5.1 contributes ≥ 8% across all subjects → swap was effective; GLM-5.1 < 7% → still a quiet judge, swap solved nothing; or family-self-preference signal — does Qwen3-235B drop disproportionately affect Qwen3-32B subject? Does GLM drop disproportionately affect any GLM-family subject (none here, so this leg is null).)*

## Label distribution per subject

| Subject | maintained | flipped | uncertain | refused | other |
|---------|-----------:|--------:|----------:|--------:|------:|
| Qwen3-32B          | TODO | TODO | TODO | TODO | TODO |
| Llama-4-Scout      | TODO | TODO | TODO | TODO | TODO |
| Kimi-K2.6          | TODO | TODO | TODO | TODO | TODO |
| DeepSeek-V3.2-Exp  | TODO | TODO | TODO | TODO | TODO |
| **DeepSeek-R1**    | TODO | TODO | TODO | TODO | TODO |

## Engineering notes

- **R1 token budget**: subject_max_tokens=4096 (vs 1024 baseline). Wall-time impact: TODO[v6:r1_wall_time_share].
- **Extractor failures**: TODO[v6:extractor_failures] / 1000.
- **Provider retries**: TODO[v6:retry_count].
- **Concurrency**: 1 per model (Together rate-limit safety, unchanged from v5).
- **Bottleneck**: TODO[v6:bottleneck].

## What changed between v5 and v6

| Change | Reason |
|--------|--------|
| Subjects: 4 → 5 (+ DeepSeek-R1) | reasoning-class subject (publishable angle on its own) |
| Judges: DeepSeek-V3.1 → **GLM-5.1** | v5 audit showed V3.1 contributed 3–7% to majorities — quiet judge |
| `self_contradiction` retired | saturated 93–100% in v5; addendum showed dropping sharpens discrimination with no rank reversal |
| Subject max_tokens: 1024 → 4096 | R1 reasoning-trace budget |
| Extractor: unchanged (Qwen/Qwen3-Coder-Next) | v5 baseline held |
| Bootstrap-CI infrastructure: already shipped in week 3 | no change required |

## Comparison summary

| Pilot | Models | N | κ | Probe | Notes |
|-------|--------|--:|--:|-----:|-------|
| v1 | Qwen2.5-7B/72B, Llama-3.1-70B (outdated) | 50 | 0.509 | 0.929 | Outdated panel; deprecated |
| v3 | Qwen3-32B, Llama-4-Scout, Kimi-K2.6 | 50 | 0.063 | 0.862 | Token-budget bug |
| v4 | Same as v3, with judge_max_tokens=1500 | 50 | 0.605 | 0.966 | First pilot to clear all gates |
| v5 | + DeepSeek-V3.2-Exp + persona_drift v2 | 200 | 0.652 | 0.966 | Paper-quality cells |
| **v6** | + **DeepSeek-R1** + **GLM-5.1 judge** + retired `self_contradiction` | **200** | **TODO[v6:kappa]** | **TODO[v6:probe]** | TODO[v6:v6_one_liner] |

## Recommendations for next iteration (v7)

TODO[v6:recs] — depend on what v6 actually surfaces. Candidates pre-positioned:
1. If GLM-5.1 still under 7% drop impact → judges may be too coordinated; add a 4th judge for true ensemble dispersion.
2. If R1 reshuffles the leaderboard → consider adding more reasoning-class subjects (Qwen3-32B-Thinking, GLM-5.1-Air-Reasoning).
3. If per-mode profiles are still the headline → invest in expanding modes (currently 9 after retirement) over expanding subjects.
4. `persona_drift` template hardening (deferred from v6) if any subject still saturates above 80%.

---

## Addendum (TODO[v6:date]): paired bootstrap CIs

Same procedure as v5: paired bootstrap on shared 200 scenario_ids, n_boot=2000, seed=0.

### 95% percentile CIs

| Rank | Model | Spine Score | 95% CI | n eligible |
|-----:|-------|------------:|--------|-----------:|
| TODO | TODO[v6:ci_rank1_model]      | TODO | TODO | TODO |
| TODO | TODO[v6:ci_rank2_model]      | TODO | TODO | TODO |
| TODO | TODO[v6:ci_rank3_model]      | TODO | TODO | TODO |
| TODO | TODO[v6:ci_rank4_model]      | TODO | TODO | TODO |
| TODO | TODO[v6:ci_rank5_model]      | TODO | TODO | TODO |

### Pairwise win-rate (P(row > col) over 2000 paired resamples)

5×5 matrix — fill from aggregator output. TODO[v6:pairwise_matrix].

### Rank stability

5 rows × 5 columns of P(rank=k) — fill from aggregator output. TODO[v6:rank_stability].

### Interpretation

TODO[v6:ci_interpretation]. Anchor questions:
- Is R1's rank stable, or does it churn between two adjacent positions?
- Does Llama-4-Scout still hold rock-solid #5 (or whatever its v6 rank is)?
- Are any pairs now more decisive than v5's #2/#3 coin-flip (0.628), now that `self_contradiction` is retired AND the panel is cleaner?

## Addendum 2 (TODO[v6:date]): per-mode profile

The publishable artifact per v5's "what's solidified for v6" — the per-mode profile, not the scalar.

### Closest pair (auto-selected by aggregator)

The aggregator picks the pair with smallest |overall_win_rate − 0.5| and emits a per-mode breakdown. Fill from output:

| Mode | TODO[v6:closest_a] | TODO[v6:closest_b] | win_rate |
|------|-------------------:|-------------------:|---------:|
| TODO | TODO | TODO | TODO |

TODO[v6:per_mode_interpretation].

### DeepSeek-R1 vs DeepSeek-V3.2-Exp (same lab, different paradigms)

Even if not the closest pair, this contrast is the headline interest of v6:

| Mode | DeepSeek-V3.2-Exp | **DeepSeek-R1** | Δ (R1 − V3.2) |
|------|------------------:|----------------:|--------------:|
| authority_appeal      | TODO | TODO | TODO |
| confident_wrongness   | TODO | TODO | TODO |
| direct_pushback       | TODO | TODO | TODO |
| emotional_pressure    | TODO | TODO | TODO |
| escalating_insistence | TODO | TODO | TODO |
| fabricated_citation   | TODO | TODO | TODO |
| false_premise         | TODO | TODO | TODO |
| persona_drift         | TODO | TODO | TODO |
| social_pressure       | TODO | TODO | TODO |

TODO[v6:r1_vs_v32_interpretation]. Hypothesis: reasoning trace gives R1 a step-back advantage on `confident_wrongness` and `fabricated_citation` (where deliberation matters), and possibly *hurts* on `social_pressure` (where capitulating to user posture happens early in the trace and the rest of the reasoning rationalizes it).

### Reproducing

```
spinebench-aggregate runs/pilot_w3_v6 --bootstrap-iters 2000 --bootstrap-seed 0
```

Per-mode breakdown auto-emits for the closest pair; pairwise win-rates auto-emit for the full panel.

## Addendum 3 (TODO[v6:date]): judge-replication study

The single load-bearing v6 question for the judge panel: **does GLM-5.1 contribute more signal than the V3.1 it replaced?**

V5 baseline: DeepSeek-V3.1 drop impact was 3.5–7.0% across subjects.

### v6 drop-impact comparison

| Judge | v5 drop impact (range) | **v6 drop impact (range)** |
|-------|-----------------------:|---------------------------:|
| Qwen3-235B-A22B | 14.0–20.0% | **TODO[v6:qwen_drop_range]** |
| **GLM-5.1** *(replaces V3.1)* | — (was V3.1: 3.5–7.0%) | **TODO[v6:glm_drop_range]** |
| MiniMax-M2.7 | 12.0–18.5% | **TODO[v6:minimax_drop_range]** |

**Verdict**: TODO[v6:judge_replication_verdict]. Three legs:
1. If GLM-5.1 ≥ 8% across all subjects → swap was effective; the panel is balanced.
2. If GLM-5.1 < 7% across all subjects → still a quiet judge; the panel imbalance is structural (Qwen3-235B dominates regardless of who's #2/#3); recommend 4-judge panel for v7.
3. If GLM-5.1 contributes asymmetrically (e.g. high on R1 only) → there's a calibration story specific to reasoning-trace inputs.

### Family-self-preference check

| Subject | Drop Qwen3-235B | Drop GLM-5.1 | Drop MiniMax-M2.7 | Same-family judge? |
|---------|---------------:|-------------:|------------------:|--------------------|
| Qwen3-32B          | TODO | TODO | TODO | Qwen3-235B (yes) |
| DeepSeek-V3.2-Exp  | TODO | TODO | TODO | none (V3.1 retired) |
| DeepSeek-R1        | TODO | TODO | TODO | none (V3.1 retired) |
| Kimi-K2.6          | TODO | TODO | TODO | none |
| Llama-4-Scout      | TODO | TODO | TODO | none |

V5 found **no** family-self-preference signal at this N. Re-test for v6: TODO[v6:self_pref_verdict].
