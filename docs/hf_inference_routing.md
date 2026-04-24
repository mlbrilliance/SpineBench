# HF Inference routing for SpineBench

This doc lists which open models are actually routable through your HF Inference
account at the time of this session, and what to do when a model returns
`model_not_supported`.

## Confirmed routable (as of 2026-04-24)

These models successfully produced chat completions via
`InferenceClient(provider="auto")`:

| Model | Family | Size | Role in pilot |
|---|---|---|---|
| `Qwen/Qwen2.5-7B-Instruct` | Qwen 2.5 | 7B | subject (small tier) |
| `Qwen/Qwen2.5-72B-Instruct` | Qwen 2.5 | 72B | subject (flagship) |
| `Qwen/Qwen2.5-Coder-32B-Instruct` | Qwen 2.5 Coder | 32B | extractor + judge |
| `meta-llama/Llama-3.1-70B-Instruct` | Llama 3.1 | 70B | subject |
| `meta-llama/Llama-3.3-70B-Instruct` | Llama 3.3 | 70B | judge |
| `deepseek-ai/DeepSeek-V3` | DeepSeek V3 | 671B (MoE) | judge |

## Confirmed NOT routable (return `model_not_supported`)

Models that returned `{"code": "model_not_supported"}` or `"is not a chat model"`
from your HF account:

- `Qwen/Qwen2.5-14B-Instruct`
- `Qwen/Qwen2.5-Math-72B-Instruct`
- `Qwen/QwQ-32B-Preview`
- `meta-llama/Meta-Llama-3.1-8B-Instruct` (the 8B variant; the 70B works)
- `mistralai/Mistral-7B-Instruct-v0.3`
- `mistralai/Mistral-Nemo-Instruct-2407`
- `google/gemma-2-9b-it`
- `microsoft/Phi-3.5-mini-instruct`
- `HuggingFaceH4/zephyr-7b-beta`
- `CohereForAI/c4ai-command-r-plus-08-2024`

This isn't a bug in the code — it reflects the providers enabled on your account.

## Why this happens

HF Inference routes requests through upstream providers (Together, Fireworks,
Novita, SambaNova, HF-Inference itself, etc.). Each model is hosted by different
providers; your account has a subset of providers enabled via
https://huggingface.co/settings/inference-providers .

When `provider="auto"` and none of your enabled providers host the requested model,
you get `model_not_supported`.

## How to expand

1. Visit https://huggingface.co/settings/inference-providers
2. Enable additional providers (at minimum: Together, Fireworks, HF-Inference, Novita).
3. Re-run the one-shot probe in `scripts/probe_routable_models.py` (see below) to
   verify which candidates now route.

## One-shot probe utility

Minimal snippet to check which models are currently routable from your account:

```python
from spinebench.providers.hf_inference import HFInferenceProvider
from spinebench.providers.base import ProviderError
from spinebench.types import Turn

candidates = [
    "meta-llama/Llama-3.3-70B-Instruct",
    "google/gemma-2-9b-it",
    "mistralai/Mistral-Nemo-Instruct-2407",
    # ... add more
]

for m in candidates:
    try:
        p = HFInferenceProvider(model_id=m, timeout_s=15, max_attempts=1)
        p.generate([Turn(role="user", content="hi")], max_tokens=5)
        print(f"OK  {m}")
    except Exception as e:
        print(f"FAIL {m}: {str(e)[:100]}")
```

## Impact on the pilot design

With 6 routable models, the Week-3 pilot settled on:

- **3 subjects**: Qwen-7B, Qwen-72B, Llama-3.1-70B (spans size and family).
- **1 extractor**: Qwen-Coder-32B (code-tuned, cheap, capable at JSON).
- **3 judges**: Llama-3.3-70B, DeepSeek-V3, Qwen-Coder-32B.

**Known compromise**: Qwen-Coder-32B is both the extractor and one judge. A single
judge shouldn't tip the ensemble, but leave-one-judge-out audit will flag any
cases where that judge dominates majority votes.

If HF provider routing expands before Week 4's full 50-model run, we can shift to
a cleaner setup (distinct extractor, 3 non-overlapping judges from different
families).
