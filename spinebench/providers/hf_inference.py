"""Hugging Face Inference provider — the only backend we ship v1 with.

Design:
- Narrow interface (ChatProvider.generate) over the full HF Inference + provider-routing surface.
- huggingface_hub>=1.2.0 already retries 429s using RateLimit headers; we add tenacity on top
  for transport / 5xx / InferenceTimeoutError.
- provider="auto" lets HF route to whichever backend (Together, Fireworks, Novita, HF-Inference,
  etc.) currently serves the model. Pro credits are charged against the HF account either way.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

from huggingface_hub import InferenceClient
from huggingface_hub.errors import InferenceTimeoutError
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from spinebench.providers.base import ProviderError
from spinebench.types import Turn

log = logging.getLogger(__name__)


@dataclass
class HFInferenceProvider:
    """Chat provider backed by the HF Inference API.

    Parameters
    ----------
    model_id:
        HF repo id, e.g. "Qwen/Qwen2.5-7B-Instruct".
    provider:
        Which backend to route through. "auto" lets HF pick.
    api_key:
        HF token. Falls back to $HF_TOKEN then to whatever huggingface_hub finds locally.
    timeout_s:
        Per-request timeout. HF default is effectively forever, which is wrong for a benchmark.
    max_attempts:
        Transport-level retries on top of the library's built-in 429 handling.
    """

    model_id: str
    provider: str = "auto"
    api_key: str | None = None
    timeout_s: float = 60.0
    max_attempts: int = 4
    _client: InferenceClient = field(init=False, repr=False)

    def __post_init__(self) -> None:
        key = self.api_key or os.environ.get("HF_TOKEN")
        self._client = InferenceClient(
            provider=self.provider,
            model=self.model_id,
            api_key=key,
            timeout=self.timeout_s,
        )

    def generate(
        self,
        turns: list[Turn],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        messages = [{"role": t.role, "content": t.content} for t in turns]
        try:
            return self._call(messages, max_tokens=max_tokens, temperature=temperature)
        except RetryError as e:
            cause = e.last_attempt.exception() if e.last_attempt else e
            raise ProviderError(
                f"HF inference failed after {self.max_attempts} attempts for {self.model_id}: {cause}"
            ) from cause

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _call(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
    ) -> str:
        @retry(
            stop=stop_after_attempt(self.max_attempts),
            wait=wait_exponential(multiplier=2, min=2, max=30),
            retry=retry_if_exception_type((InferenceTimeoutError, ConnectionError, TimeoutError)),
            reraise=True,
        )
        def _do() -> str:
            output = self._client.chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            content = output.choices[0].message.content if output.choices else None
            if not content:
                raise ProviderError(f"empty completion from {self.model_id}")
            return content

        return _do()
