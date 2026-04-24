"""Model runtime: SHA pinning + per-model concurrency + provider construction.

Sits between callers (Evaluator, batch runner) and providers (HFInferenceProvider).
See docs/rfcs/0003-model-runtime.md for design rationale.

The cache is deliberately NOT owned here — it lives in the Evaluator per RFC 0001 to
avoid double-state. The runtime owns only: revision pinning, concurrency guard, provider
construction with consistent timeout/retry/api_key.
"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from typing import Protocol

from spinebench.providers.base import ChatProvider
from spinebench.types import Turn


@dataclass
class ModelSpec:
    model_id: str
    revision: str | None = None
    provider: str = "auto"


@dataclass(frozen=True)
class PinnedModel:
    model_id: str
    revision: str
    provider_name: str


class RevisionPinner(Protocol):
    def resolve(self, model_id: str, revision: str | None) -> str: ...


class ProviderFactory(Protocol):
    def make(
        self,
        spec: ModelSpec,
        *,
        api_key: str | None,
        timeout_s: float,
        max_attempts: int,
    ) -> ChatProvider: ...


class HFApiPinner:
    """Production pinner: calls huggingface_hub.HfApi.model_info to resolve latest SHA.

    Explicit revisions (caller-supplied) pass through unchanged without an API call — the
    user has already committed to a specific snapshot.
    """

    def __init__(self, api_key: str | None = None) -> None:
        from huggingface_hub import HfApi

        self._api = HfApi(token=api_key)

    def resolve(self, model_id: str, revision: str | None) -> str:
        if revision is not None:
            return revision
        return self._api.model_info(model_id).sha


class HFInferenceProviderFactory:
    """Production factory: constructs HFInferenceProvider instances."""

    def make(
        self,
        spec: ModelSpec,
        *,
        api_key: str | None,
        timeout_s: float,
        max_attempts: int,
    ) -> ChatProvider:
        from spinebench.providers.hf_inference import HFInferenceProvider

        return HFInferenceProvider(
            model_id=spec.model_id,
            provider=spec.provider,
            api_key=api_key,
            timeout_s=timeout_s,
            max_attempts=max_attempts,
        )


class FakePinner:
    """Deterministic pinner for tests.

    Looks up `model_id` in an optional dict; falls back to a stable hash of the model_id
    so tests that don't provide an explicit mapping still get a predictable SHA.
    """

    def __init__(self, shas: dict[str, str] | None = None) -> None:
        self._shas: dict[str, str] = shas if shas is not None else {}

    def resolve(self, model_id: str, revision: str | None) -> str:
        if revision is not None:
            return revision
        if model_id in self._shas:
            return self._shas[model_id]
        return f"fake-sha-{hashlib.sha256(model_id.encode()).hexdigest()[:12]}"


@dataclass
class _ConcurrencyGuardedProvider:
    """Wraps a ChatProvider with a threading.Semaphore capping per-model concurrency.

    `revision` is surfaced publicly so downstream code (scenario result writers) can
    record the exact pinned SHA in their output.
    """

    model_id: str
    revision: str
    _inner: ChatProvider
    _semaphore: threading.Semaphore

    def generate(
        self,
        turns: list[Turn],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        with self._semaphore:
            return self._inner.generate(
                turns, max_tokens=max_tokens, temperature=temperature
            )


class ModelRuntime:
    """Orchestrates model pinning, provider construction, and concurrency for a run.

    Flow:
      specs -> pin() -> list[PinnedModel] (SHAs resolved, recorded for run metadata)
      pinned -> chat() -> ChatProvider (semaphore-guarded, shared across callers)
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        timeout_s: float = 60.0,
        max_attempts: int = 4,
        concurrency_per_model: int = 4,
        pinner: RevisionPinner | None = None,
        provider_factory: ProviderFactory | None = None,
    ) -> None:
        self._api_key = api_key
        self._timeout_s = timeout_s
        self._max_attempts = max_attempts
        self._concurrency_per_model = concurrency_per_model
        self._pinner: RevisionPinner = (
            pinner if pinner is not None else HFApiPinner(api_key=api_key)
        )
        self._factory: ProviderFactory = (
            provider_factory
            if provider_factory is not None
            else HFInferenceProviderFactory()
        )
        self._pinned: list[PinnedModel] = []
        self._pinned_index: dict[tuple[str, str], PinnedModel] = {}
        self._semaphores: dict[str, threading.Semaphore] = {}
        self._providers: dict[PinnedModel, ChatProvider] = {}

    def pin(self, specs: list[ModelSpec]) -> list[PinnedModel]:
        result: list[PinnedModel] = []
        for spec in specs:
            key = (spec.model_id, spec.provider)
            if key in self._pinned_index:
                result.append(self._pinned_index[key])
                continue
            revision = self._pinner.resolve(spec.model_id, spec.revision)
            pinned = PinnedModel(
                model_id=spec.model_id,
                revision=revision,
                provider_name=spec.provider,
            )
            self._pinned_index[key] = pinned
            self._pinned.append(pinned)
            result.append(pinned)
        return result

    def chat(self, pinned: PinnedModel) -> ChatProvider:
        if pinned in self._providers:
            return self._providers[pinned]

        spec = ModelSpec(
            model_id=pinned.model_id,
            revision=pinned.revision,
            provider=pinned.provider_name,
        )
        inner = self._factory.make(
            spec,
            api_key=self._api_key,
            timeout_s=self._timeout_s,
            max_attempts=self._max_attempts,
        )

        if pinned.model_id not in self._semaphores:
            self._semaphores[pinned.model_id] = threading.Semaphore(
                self._concurrency_per_model
            )

        wrapped = _ConcurrencyGuardedProvider(
            model_id=pinned.model_id,
            revision=pinned.revision,
            _inner=inner,
            _semaphore=self._semaphores[pinned.model_id],
        )
        self._providers[pinned] = wrapped
        return wrapped

    def specs(self) -> list[PinnedModel]:
        return list(self._pinned)
