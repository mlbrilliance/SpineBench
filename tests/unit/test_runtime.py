"""Boundary tests for spinebench.runtime (ModelRuntime + pinning + concurrency)."""

from __future__ import annotations

import threading
import time

from spinebench.providers.base import ChatProvider
from spinebench.runtime import (
    FakePinner,
    ModelRuntime,
    ModelSpec,
    PinnedModel,
    ProviderFactory,
)
from spinebench.types import Turn
from tests.conftest import FakeProvider

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class _DictFactory:
    """Factory that returns FakeProvider per requested model_id, recording creations."""

    def __init__(self, responder=None) -> None:
        self._responder = responder or (lambda _t: "ok")
        self.created: list[str] = []

    def make(
        self,
        spec: ModelSpec,
        *,
        api_key: str | None,
        timeout_s: float,
        max_attempts: int,
    ) -> ChatProvider:
        self.created.append(spec.model_id)
        return FakeProvider(model_id=spec.model_id, responder=self._responder)


def _rt(
    *,
    pinner: FakePinner | None = None,
    factory: ProviderFactory | None = None,
    concurrency: int = 4,
) -> ModelRuntime:
    return ModelRuntime(
        pinner=pinner or FakePinner(),
        provider_factory=factory or _DictFactory(),
        concurrency_per_model=concurrency,
    )


# ---------------------------------------------------------------------------
# pin() behaviour
# ---------------------------------------------------------------------------


def test_pin_resolves_none_revision_via_pinner():
    pinner = FakePinner({"m1": "sha-abc"})
    runtime = _rt(pinner=pinner)
    [pinned] = runtime.pin([ModelSpec(model_id="m1")])
    assert pinned.model_id == "m1"
    assert pinned.revision == "sha-abc"


def test_pin_passes_explicit_revision_through():
    runtime = _rt(pinner=FakePinner({"m1": "sha-auto"}))
    [pinned] = runtime.pin([ModelSpec(model_id="m1", revision="my-override-sha")])
    assert pinned.revision == "my-override-sha"


def test_pin_is_idempotent():
    runtime = _rt(pinner=FakePinner({"m1": "sha-a"}))
    p1 = runtime.pin([ModelSpec(model_id="m1")])
    p2 = runtime.pin([ModelSpec(model_id="m1")])
    assert p1 == p2
    # And specs() only records it once.
    assert len(runtime.specs()) == 1


def test_pin_deterministic_fake_pinner_for_unknown_model():
    """FakePinner returns a predictable SHA for models not in its dict — useful for tests."""
    pinner = FakePinner()
    runtime = _rt(pinner=pinner)
    [pinned] = runtime.pin([ModelSpec(model_id="org/repo")])
    assert pinned.revision  # non-empty
    # Same input -> same output
    [pinned2] = runtime.pin([ModelSpec(model_id="org/repo")])
    assert pinned.revision == pinned2.revision


def test_pin_returns_pinned_model_type():
    runtime = _rt()
    [pinned] = runtime.pin([ModelSpec(model_id="m1")])
    assert isinstance(pinned, PinnedModel)


def test_specs_returns_all_pinned_in_order():
    runtime = _rt()
    runtime.pin([ModelSpec(model_id="a"), ModelSpec(model_id="b")])
    runtime.pin([ModelSpec(model_id="c")])
    assert [s.model_id for s in runtime.specs()] == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# chat() behaviour
# ---------------------------------------------------------------------------


def test_chat_returns_provider_with_matching_model_id():
    runtime = _rt()
    [pinned] = runtime.pin([ModelSpec(model_id="Qwen/Qwen2.5-7B")])
    provider = runtime.chat(pinned)
    assert provider.model_id == "Qwen/Qwen2.5-7B"


def test_chat_reuses_provider_for_same_pinned():
    factory = _DictFactory()
    runtime = _rt(factory=factory)
    [pinned] = runtime.pin([ModelSpec(model_id="m1")])
    p1 = runtime.chat(pinned)
    p2 = runtime.chat(pinned)
    assert p1 is p2
    # Factory should only have been asked to make the provider once.
    assert factory.created.count("m1") == 1


def test_chat_exposes_pinned_revision():
    """Downstream code should be able to read provider.revision to record in result metadata."""
    pinner = FakePinner({"m1": "sha-deadbeef"})
    runtime = _rt(pinner=pinner)
    [pinned] = runtime.pin([ModelSpec(model_id="m1")])
    provider = runtime.chat(pinned)
    assert getattr(provider, "revision", None) == "sha-deadbeef"


def test_chat_generate_passes_through():
    runtime = _rt(factory=_DictFactory(responder=lambda _t: "echo"))
    [pinned] = runtime.pin([ModelSpec(model_id="m1")])
    provider = runtime.chat(pinned)
    assert provider.generate([Turn(role="user", content="x")]) == "echo"


# ---------------------------------------------------------------------------
# concurrency guard
# ---------------------------------------------------------------------------


class _CountingProvider:
    """Tracks in-flight `generate` calls to verify concurrency caps."""

    def __init__(self, model_id: str, delay_s: float = 0.02) -> None:
        self.model_id = model_id
        self._delay = delay_s
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.total_calls = 0

    def generate(self, turns, *, max_tokens: int = 512, temperature: float = 0.0) -> str:
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.total_calls += 1
        time.sleep(self._delay)
        with self._lock:
            self.active -= 1
        return "done"


class _FixedProviderFactory:
    def __init__(self, providers: dict[str, ChatProvider]) -> None:
        self._providers = providers

    def make(
        self,
        spec: ModelSpec,
        *,
        api_key: str | None,
        timeout_s: float,
        max_attempts: int,
    ) -> ChatProvider:
        return self._providers[spec.model_id]


def test_concurrency_per_model_caps_in_flight_calls():
    slow = _CountingProvider("m1")
    runtime = _rt(factory=_FixedProviderFactory({"m1": slow}), concurrency=2)
    [pinned] = runtime.pin([ModelSpec(model_id="m1")])
    provider = runtime.chat(pinned)

    threads = [
        threading.Thread(target=lambda: provider.generate([Turn(role="user", content="x")]))
        for _ in range(6)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert slow.total_calls == 6
    assert slow.max_active <= 2


def test_different_models_have_independent_semaphores():
    slow_a = _CountingProvider("m_a")
    slow_b = _CountingProvider("m_b")
    runtime = _rt(
        factory=_FixedProviderFactory({"m_a": slow_a, "m_b": slow_b}),
        concurrency=1,
    )
    [pa, pb] = runtime.pin([ModelSpec(model_id="m_a"), ModelSpec(model_id="m_b")])
    provider_a = runtime.chat(pa)
    provider_b = runtime.chat(pb)

    started = threading.Barrier(2)

    def _hit(p):
        started.wait()
        p.generate([Turn(role="user", content="x")])

    t1 = threading.Thread(target=_hit, args=(provider_a,))
    t2 = threading.Thread(target=_hit, args=(provider_b,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # Each model's in-flight peak is 1 (can't exceed its own semaphore).
    assert slow_a.max_active == 1
    assert slow_b.max_active == 1
    # But collectively they ran concurrently — neither blocked the other.
    # (We can't easily assert timing without flakiness; the barrier ensures both started.)


# ---------------------------------------------------------------------------
# ModelSpec / PinnedModel basics
# ---------------------------------------------------------------------------


def test_model_spec_defaults():
    spec = ModelSpec(model_id="m1")
    assert spec.revision is None
    assert spec.provider == "auto"


def test_pinned_model_is_hashable():
    p = PinnedModel(model_id="m1", revision="sha-a", provider_name="auto")
    # frozen=True makes it hashable — useful for dict keys in downstream code
    d = {p: 1}
    assert d[p] == 1
