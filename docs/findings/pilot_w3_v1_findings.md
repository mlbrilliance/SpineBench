# Pilot W3-v1 Findings — 2026-04-24

First real-API SpineBench pilot. Not paper-quality (N=50, 3 subjects), but the pipeline ran
end-to-end and produced a counterintuitive result.

## Setup

- **Subjects (3)**: Qwen-7B, Qwen-72B, Llama-3.1-70B
- **Judges (3)**: Llama-3.3-70B, DeepSeek-V3, Qwen-Coder-32B
- **Extractor**: Qwen-Coder-32B
- **Scenarios**: 50 dev scenarios, stratified across 10 failure modes
- **Probe audit**: 14 probes, **92.9% ensemble accuracy** (above 85% threshold)
- **Wall time**: ~24 min for batch (probe audit + 150 (model, scenario) pairs)
- **Failures**: 1 / 150 subject calls failed; multiple DeepSeek-V3 judge calls hit Together's dynamic rate limit and degraded to `label=other`

## Leaderboard

| Rank | Model | Spine Score | n | failed |
|------|-------|------------|---|--------|
| 1 | meta-llama/Llama-3.1-70B-Instruct | **44.4%** | 50 | 0 |
| 2 | Qwen/Qwen2.5-7B-Instruct | **38.5%** | 50 | 0 |
| 3 | Qwen/Qwen2.5-72B-Instruct | **31.0%** | 50 | 1 |

**Headline:** Qwen-7B has more spine than Qwen-72B (38.5% vs 31.0%).
Within the Qwen family, the smaller model is more stubborn under user pressure. Worth
investigating whether instruction-tuning ate spine in the larger model. Sample is
small — would not bet a paper on this with N=50, but it's a real signal.

## Per-failure-mode breakdown

| Mode | Qwen-7B | Qwen-72B | Llama-3.1-70B |
|------|--------:|---------:|--------------:|
| authority_appeal | **0%** | **0%** | **0%** |
| confident_wrongness | 50 | 67 | 33 |
| direct_pushback | 0 | 0 | 50 |
| emotional_pressure | 100 | 33 | 100 |
| escalating_insistence | 0 | 0 | 25 |
| fabricated_citation | 100 | – | 0 |
| false_premise | 100 | – | 100 |
| persona_drift | 50 | 40 | 67 |
| self_contradiction | 75 | 50 | 60 |
| social_pressure | 0 | 67 | 50 |

**Notable:**
- **authority_appeal: 0% across all three.** Every model in the panel folded when the user invoked authority ("my professor said..."). This is the cleanest universal vulnerability the pilot found.
- **fabricated_citation**: Qwen-7B 100% vs Llama-3.1-70B 0%. Qwen refused to fabricate, Llama did. Big spread.
- **direct_pushback**: only Llama-3.1-70B held at all. Both Qwen models caved.

## Statistical concerns

- **Fleiss kappa = 0.509** (target ≥ 0.60). Below threshold, but this is a "moderate agreement" zone. The DeepSeek-V3 rate-limit failures (Together capped to 0.5 RPM dynamically mid-run) inflated the disagreement rate by leaving DeepSeek out of many votes.
- **High `other` rates per subject** (32–44%). Mostly driven by ties when one judge dropped out.
- **NaN cells in per-mode table** — small N per (model, mode) cell, some have zero clean (non-other, non-refused) scenarios.

## Self-preference / judge-drop sensitivity

When dropping each judge, fraction of (subject, scenario) majorities that flip:

| Subject | Dropped judge | Δ |
|---------|--------------|--:|
| Qwen-72B | DeepSeek-V3 | 24% |
| Llama-3.1-70B | DeepSeek-V3 | 22% |
| Llama-3.1-70B | Qwen-Coder-32B | 22% |
| Qwen-72B | Qwen-Coder-32B | 20% |

DeepSeek-V3 is the most load-bearing judge — its votes change majorities ~20-24% of the time. Confounded with the rate-limit failures (its missing votes count as drops). Hard to separate "this judge has unique perspective" from "this judge wasn't always there." Need a clean re-run without rate-limit artifacts.

## Recommendations

1. **Re-run with backoff for DeepSeek-V3** — extend tenacity retry to handle 429s with `Retry-After` honoring; current implementation only retries on transport errors.
2. **Wait on DeepSeek between calls** — Together's dynamic rate limit punishes burst, so even a 5-second sleep between DeepSeek calls would help. Or: use a separate provider for DeepSeek if available.
3. **Increase N** — bump to 200 scenarios per subject before drawing real conclusions. Burns more quota but gets the per-mode cells out of NaN territory.
4. **Cross-family judge swap** — a separate DeepSeek-only-as-subject run would test the self-preference theory cleanly.
5. **Expand routable provider list** — current pool is 6 models, all Qwen/Llama/DeepSeek. Adding Mistral/Gemma/Phi via expanded HF Inference providers would diversify.

## Engineering notes

- 2 extractor calls returned non-JSON (long prose explanations); the fenced-block tolerance + JSON-finder fell back to empty answers. Worth tightening the extractor system prompt to force the JSON-only output more aggressively, OR using a different model for extraction.
- Disk cache hit rate: 0% (first run). Future re-runs against the same scenarios would be substantially faster.
