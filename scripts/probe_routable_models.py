"""Quick check: which models are routable through this HF account right now.

Prints OK/FAIL per candidate. Run after expanding providers via
https://huggingface.co/settings/inference-providers to see which models unlocked.
"""

from __future__ import annotations

import argparse
import logging

from spinebench.providers.base import ProviderError
from spinebench.providers.hf_inference import HFInferenceProvider
from spinebench.types import Turn

_DEFAULT_CANDIDATES = [
    # Confirmed working in this session:
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "Qwen/Qwen2.5-Coder-32B-Instruct",
    "meta-llama/Llama-3.1-70B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
    "deepseek-ai/DeepSeek-V3",
    # Commonly-desired but not currently enabled in this session:
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/QwQ-32B-Preview",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "google/gemma-2-9b-it",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Mistral-Nemo-Instruct-2407",
    "microsoft/Phi-3.5-mini-instruct",
    "CohereForAI/c4ai-command-r-plus-08-2024",
    "HuggingFaceH4/zephyr-7b-beta",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe which models are routable.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=_DEFAULT_CANDIDATES,
        help="Model IDs to probe (default: common candidates).",
    )
    parser.add_argument("--timeout", type=float, default=15.0)
    args = parser.parse_args()

    # Suppress noisy HTTP logs; we only want our OK/FAIL lines.
    logging.basicConfig(level=logging.ERROR)

    ok: list[str] = []
    fail: list[tuple[str, str]] = []
    for model_id in args.models:
        try:
            p = HFInferenceProvider(model_id=model_id, timeout_s=args.timeout, max_attempts=1)
            p.generate([Turn(role="user", content="hi")], max_tokens=5)
            print(f"OK   {model_id}")
            ok.append(model_id)
        except ProviderError as e:
            msg = str(e).split("\n")[0][:100]
            print(f"FAIL {model_id}: {msg}")
            fail.append((model_id, msg))
        except Exception as e:  # noqa: BLE001
            msg = f"{type(e).__name__}: {str(e)[:100]}"
            print(f"ERR  {model_id}: {msg}")
            fail.append((model_id, msg))

    print()
    print(f"Routable: {len(ok)} / {len(args.models)}")
    if ok:
        print("  " + "\n  ".join(ok))


if __name__ == "__main__":
    main()
