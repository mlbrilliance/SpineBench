"""Unit tests for HFInferenceProvider retry behavior.

Mocks the underlying InferenceClient.chat_completion so no network is touched. Verifies
that 429 errors are retried with backoff, that the Retry-After header is honored when
present, and that permanent 4xx errors are NOT retried.
"""

from __future__ import annotations

import time
from unittest.mock import patch

import httpx
import pytest
from huggingface_hub.errors import BadRequestError, HfHubHTTPError

from spinebench.providers.base import ProviderError
from spinebench.providers.hf_inference import HFInferenceProvider
from spinebench.types import Turn


def _fake_response(status_code: int, headers: dict[str, str] | None = None) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        headers=headers or {},
        content=b'{"message": "boom"}',
        request=httpx.Request("POST", "https://router.huggingface.co/v1/chat/completions"),
    )


def _make_429(retry_after: str | None = None) -> HfHubHTTPError:
    headers = {"Retry-After": retry_after} if retry_after is not None else {}
    return HfHubHTTPError("429 Too Many Requests", response=_fake_response(429, headers))


def _make_400() -> BadRequestError:
    return BadRequestError("model_not_supported", response=_fake_response(400))


class _FakeOutput:
    """Minimal stand-in for ChatCompletionOutput."""
    def __init__(self, content: str = "ok"):
        from types import SimpleNamespace

        self.choices = [
            SimpleNamespace(message=SimpleNamespace(content=content))
        ]


def _provider(max_attempts: int = 4) -> HFInferenceProvider:
    return HFInferenceProvider(
        model_id="fake/model",
        api_key="hf_test",
        timeout_s=5.0,
        max_attempts=max_attempts,
    )


def test_429_then_success_retries(scenario):
    """First call hits 429; second call succeeds. Provider must retry and return."""
    p = _provider(max_attempts=3)
    call_count = {"n": 0}

    def _fake_call(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise _make_429(retry_after="0.01")  # tiny wait so test stays fast
        return _FakeOutput("paris")

    with patch.object(p._client, "chat_completion", side_effect=_fake_call):
        out = p.generate([Turn(role="user", content="q")], max_tokens=10)
    assert out == "paris"
    assert call_count["n"] == 2


def test_429_repeatedly_raises_provider_error_after_max_attempts():
    """N attempts of 429 in a row -> ProviderError."""
    p = _provider(max_attempts=3)
    call_count = {"n": 0}

    def _always_429(*args, **kwargs):
        call_count["n"] += 1
        raise _make_429(retry_after="0.01")

    with (
        patch.object(p._client, "chat_completion", side_effect=_always_429),
        pytest.raises(ProviderError),
    ):
        p.generate([Turn(role="user", content="q")], max_tokens=10)
    assert call_count["n"] == 3  # respected max_attempts


def test_400_does_not_retry():
    """A permanent 4xx (model_not_supported) must NOT be retried."""
    p = _provider(max_attempts=4)
    call_count = {"n": 0}

    def _bad_request(*args, **kwargs):
        call_count["n"] += 1
        raise _make_400()

    with (
        patch.object(p._client, "chat_completion", side_effect=_bad_request),
        pytest.raises(ProviderError),
    ):
        p.generate([Turn(role="user", content="q")], max_tokens=10)
    assert call_count["n"] == 1  # exactly once — no retries


def test_429_honors_retry_after_header():
    """When Retry-After is present, the wait should be at least that long."""
    p = _provider(max_attempts=2)
    call_count = {"n": 0}
    timestamps: list[float] = []

    def _fake_call(*args, **kwargs):
        call_count["n"] += 1
        timestamps.append(time.monotonic())
        if call_count["n"] == 1:
            raise _make_429(retry_after="0.5")
        return _FakeOutput("ok")

    with patch.object(p._client, "chat_completion", side_effect=_fake_call):
        p.generate([Turn(role="user", content="q")], max_tokens=10)

    assert call_count["n"] == 2
    # Retry waited at least the requested 0.5s
    elapsed = timestamps[1] - timestamps[0]
    assert elapsed >= 0.45, f"expected ~0.5s wait, got {elapsed:.2f}s"


def test_429_without_retry_after_falls_back_to_backoff():
    """No Retry-After header -> still retries (with exponential backoff)."""
    p = _provider(max_attempts=2)
    call_count = {"n": 0}

    def _fake_call(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise _make_429(retry_after=None)
        return _FakeOutput("ok")

    with patch.object(p._client, "chat_completion", side_effect=_fake_call):
        out = p.generate([Turn(role="user", content="q")], max_tokens=10)
    assert out == "ok"
    assert call_count["n"] == 2


def test_500_server_error_retries():
    """5xx is also retryable."""
    p = _provider(max_attempts=3)
    call_count = {"n": 0}

    def _fake_call(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] < 3:
            raise HfHubHTTPError("500 Internal", response=_fake_response(500))
        return _FakeOutput("ok")

    with patch.object(p._client, "chat_completion", side_effect=_fake_call):
        out = p.generate([Turn(role="user", content="q")], max_tokens=10)
    assert out == "ok"
    assert call_count["n"] == 3
