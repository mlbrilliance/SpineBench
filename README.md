# SpineBench

> Does the model have a backbone? A sycophancy and pushback-resistance benchmark for open LLMs.

SpineBench measures how open-weight language models behave under user pressure: pushback, false premises, appeals to authority, demands for citations, and social/emotional coercion. Instead of asking "does the model know the right answer?", it asks: **does the model hold the line when the user pushes back?**

## Why

After the GPT-4o sycophancy incident, the field has the intuition that "models fold" — but no cheap, reproducible, open-models-only benchmark quantifies *how much* and *along what axes*. SpineBench fills that gap.

## Status

Early scaffolding. Week 1 of 8. Nothing is shipped.

## Design

- **Axis**: sycophancy & pushback-resistance, decomposed into 10 failure modes.
- **Models**: 50+ open-weight LLMs via the Hugging Face Inference API (routed to whichever provider serves them).
- **Prompts**: hybrid — factual ground truth from TruthfulQA / SimpleQA / MMLU-Pro crossed with hand-crafted pressure templates.
- **Judge**: 3-judge LLM ensemble with Fleiss' kappa agreement reporting and a 100-sample human audit.
- **Release**: 80% public dev / 20% private held-out, rotated quarterly.

Full plan: [/home/claude/.claude/plans/i-have-hugging-face-foamy-cosmos.md](../.claude/plans/i-have-hugging-face-foamy-cosmos.md)

## Repository layout

```
spinebench/
├── spinebench/
│   ├── providers/      # HF Inference wrapper + ZeroGPU fallback
│   ├── judges/         # 3-judge ensemble + answer extractor
│   ├── scoring/        # Spine Score aggregation, Fleiss' kappa
│   ├── data/
│   │   ├── ground_truth/
│   │   └── templates/  # pressure_templates.yaml
│   └── cli.py
├── scripts/            # run_eval.py, build_dataset.py, etc.
├── space/              # Gradio leaderboard app
├── paper/              # arXiv preprint source
└── tests/              # pytest suite
```

## Running (stubs, nothing functional yet)

```bash
pip install -e ".[dev]"
export HF_TOKEN=hf_...
spinebench-run --model Qwen/Qwen2.5-7B-Instruct --scenarios 10
```

## License

Apache-2.0.
