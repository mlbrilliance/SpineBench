"""The one interface every model provider has to satisfy."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from spinebench.types import Turn


class ProviderError(Exception):
    """Raised when the provider cannot produce a response (after retries exhausted)."""


@runtime_checkable
class ChatProvider(Protocol):
    """Minimal chat interface.

    The entire retry / rate-limit / routing / timeout story lives behind this one method.
    Callers get back a string or a ProviderError; nothing else.
    """

    model_id: str

    def generate(
        self,
        turns: list[Turn],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        """Produce the assistant's next turn given the conversation so far.

        Raises:
            ProviderError: if the provider fails permanently (quota, 4xx, timeout after retries).
        """
        ...
