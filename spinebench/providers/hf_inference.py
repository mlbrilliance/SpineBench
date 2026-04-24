"""Hugging Face Inference provider — the only backend we ship v1 with.

Design:
- Narrow interface (ChatProvider.generate) over the full HF Inference + provider-routing surface.
- huggingface_hub>=1.2.0 retries 429s on the resolver path; for chat completion through
  the inference router, dynamic-rate-limit 429s come back as `HfHubHTTPError` and we
  retry them ourselves below.
- We honor the `Retry-After` response header when present; otherwise we use exponential
  backoff capped at 60s.
- 5xx errors are retried with exponential backoff. 4xx errors *other than 429* are NOT
  retried — they're permanent (model not supported, malformed request, auth, etc.).
- provider="auto" lets HF route to whichever backend (Together, Fireworks, Novita,
  HF-Inference, etc.) currently serves the model. Pro credits are charged against the
  HF account either way.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError, InferenceTimeoutError
from tenacity import (
    RetryError,
    retry,
    retry_if_exception,
    stop_after_attempt,
)

from spinebench.providers.base import ProviderError
from spinebench.types import Turn

log = logging.getLogger(__name__)


def _is_retryable(exc: BaseException) -> bool:
    """Decide whether an exception is worth retrying.

    Retryable: transport errors (timeouts, broken connections), 429 rate limits,
    and any 5xx server error.
    Not retryable: 4xx other than 429 (model_not_supported, auth failure, bad request),
    ProviderError (empty completion is permanent for this attempt).
    """
    if isinstance(exc, (InferenceTimeoutError, ConnectionError, TimeoutError)):
        return True
    if isinstance(exc, HfHubHTTPError):
        resp = getattr(exc, "response", None)
        if resp is not None:
            status = resp.status_code
            if status == 429:
                return True
            if 500 <= status < 600:
                return True
    return False


def _wait_for_attempt(retry_state: Any) -> float:
    """Compute seconds to wait before the next attempt.

    For 429s with a `Retry-After` header, honor it (capped at 120s for sanity).
    Otherwise, exponential backoff: 2, 4, 8, 16, 32, 60 (capped at 60s).
    """
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    if isinstance(exc, HfHubHTTPError):
        resp = getattr(exc, "response", None)
        if resp is not None and resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After") if resp.headers else None
            if retry_after:
                try:
                    return min(120.0, float(retry_after))
                except (ValueError, TypeError):
                    pass
    return min(60.0, 2.0 ** retry_state.attempt_number)


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
        Total attempts (including the first call). Covers both transport-level retries
        and 429/5xx HTTP retries.
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
        except HfHubHTTPError as e:
            # Non-retryable HTTP error (4xx other than 429).
            raise ProviderError(
                f"HF inference HTTP error for {self.model_id}: {e}"
            ) from e

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
            wait=_wait_for_attempt,
            retry=retry_if_exception(_is_retryable),
            reraise=False,
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
