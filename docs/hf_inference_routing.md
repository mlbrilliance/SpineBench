# HF Inference routing for SpineBench

This doc lists which open models are routable through your HF Inference account at the
time of this session, and what to do when a model returns `model_not_supported`.

**Last probed: 2026-04-24.** Re-run `scripts/probe_routable_models.py` after expanding
HF Inference providers (https://huggingface.co/settings/inference-providers).

## Confirmed routable — 2026-current models

These are the actively-recommended open-weight models for spring 2026:

| Model | Family | Size | Suggested role |
|---|---|---|---|
| `Qwen/Qwen3-235B-A22B-Instruct-2507` | Qwen 3 | 235B (22B active MoE) | judge / flagship subject |
| `Qwen/Qwen3-32B` | Qwen 3 | 32B dense | subject |
| `Qwen/Qwen3-Coder-Next` | Qwen 3 Coder | code-tuned | extractor |
| `meta-llama/Llama-4-Scout-17B-16E-Instruct` | Llama 4 Scout | 109B total / 17B active | subject (cross-family) |
| `deepseek-ai/DeepSeek-V3.1` | DeepSeek V3 | 671B (37B active) | judge |
| `deepseek-ai/DeepSeek-V3.2-Exp` | DeepSeek V3.2 (experimental) | 671B | judge — output can be verbose, prefer V3.1 |
| `deepseek-ai/DeepSeek-R1` | DeepSeek R1 | 671B reasoning | use cautiously — long reasoning traces |
| `moonshotai/Kimi-K2.6` | Kimi K2 | ~1T MoE | subject (cross-family flagship) |
| `MiniMaxAI/MiniMax-M2.7` | MiniMax M2 | varies | judge (different family) |

## Confirmed NOT routable

These either return `model_not_supported`/HTTP errors or produce empty completions
under your current provider config:

- `Qwen/Qwen3.6-27B`, `Qwen/Qwen3.6-35B-A3B` (latest Qwen 3.6 dense + small MoE — Apr 2026)
- `Qwen/Qwen3.5-397B-A17B`, `Qwen/Qwen3.5-122B`, `Qwen/Qwen3.5-9B`, `Qwen/Qwen3.5-4B`
- `meta-llama/Llama-4-Maverick-17B-128E-Instruct`
- `deepseek-ai/DeepSeek-V3.2`
- `google/gemma-4-27b-it`, `google/gemma-4-31b`, `google/gemma-4-9b-it`
- `zai-org/GLM-5.1`, `zai-org/GLM-4.6`, `THUDM/glm-5`
- `moonshotai/Kimi-K2.5`
- `MiniMaxAI/MiniMax-M2`

## Why this happens

HF Inference routes requests through upstream providers (Together, Fireworks, Novita,
SambaNova, HF-Inference, Featherless, etc.). Each model is hosted by a subset of
providers; your account has a subset of providers enabled. When `provider="auto"` and
none of your enabled providers host the requested model, you get `model_not_supported`.

## How to expand

1. Visit https://huggingface.co/settings/inference-providers
2. Enable additional providers (Together, Fireworks, Novita, Featherless cover most flagship models).
3. Re-run `scripts/probe_routable_models.py` to confirm.

## Pilot v3 (current) — model panel

With 9 routable 2026-current models:

- **3 subjects**: `Qwen/Qwen3-32B`, `meta-llama/Llama-4-Scout-17B-16E-Instruct`, `moonshotai/Kimi-K2.6`
- **3 judges**: `Qwen/Qwen3-235B-A22B-Instruct-2507`, `deepseek-ai/DeepSeek-V3.1`, `MiniMaxAI/MiniMax-M2.7`
- **Extractor**: `Qwen/Qwen3-Coder-Next`

Three different model families across the subjects, three different families across the
judges, and the extractor is a code-tuned variant separate from any judge or subject.
The only mild overlap is Qwen-32B subject vs Qwen-235B judge, but these are very
different scales and posture (instruct vs flagship).

## Pilot v1 (deprecated) — outdated panel

Earlier pilot used Qwen2.5-7B/72B, Llama-3.1-70B / 3.3-70B, DeepSeek-V3, Qwen-Coder-32B.
These were a generation behind the April 2026 frontier and have been replaced. See
`docs/findings/pilot_w3_v1_findings.md` for the v1 leaderboard, retained for posterity.
