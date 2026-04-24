# RFC 0003: Introduce a `ModelRuntime` layer for provider orchestration

**Status**: Accepted
**Author**: nick
**Date**: 2026-04-24
**Priority**: medium — not a Week-3 blocker, but a Week-4 prerequisite for async batching and SHA pinning

## Problem

Today, every caller that wants to talk to a model constructs a `HFInferenceProvider` directly:

```python
provider = HFInferenceProvider(model_id="Qwen/Qwen2.5-7B-Instruct", ...)
```

This is fine for Week 3 (10 models, one-at-a-time pilot). Week 4 brings three new requirements:

1. **Model SHA pinning** — the plan requires every result row to record the exact `huggingface_hub.HfApi().model_info(..., revision=...)` commit SHA. Today, providers have no notion of a pinned SHA.
2. **Response caching** — a (model_id, scenario_id) → transcript cache that survives process restarts. RFC 0001 (Evaluator) introduces a `TranscriptCache` Protocol, but the cache key ignores model SHA, which is wrong if a model revision changes mid-run.
3. **Async concurrency** — 50 models × 1500 scenarios × 3 turns × 3 judges ≈ 675k inference calls. Sync sequential execution is cost-prohibitive in wall-clock time. We need controlled concurrency with per-provider rate-limit awareness.

Without a deepened runtime layer, each caller (Evaluator, batch runner, future eval scripts) will re-solve these concerns independently. The `ChatProvider` Protocol itself is correctly shallow — it is the **boundary**, not the orchestration layer. The missing piece is a thin runtime that sits between callers and providers.

## Proposed interface

A `ModelRuntime` that resolves model IDs to providers, pins revisions, and enforces concurrency and caching invariants. Callers get providers from the runtime; they don't construct them.

```python
# spinebench/runtime.py
from dataclasses import dataclass
from spinebench.providers.base import ChatProvider

@dataclass
class ModelSpec:
    model_id: str
    revision: str | None = None    # HF commit SHA; None = resolve latest at pin time
    provider: str = "auto"          # "auto", "together", "fireworks", etc.

@dataclass(frozen=True)
class PinnedModel:
    model_id: str
    revision: str                   # always resolved; never None after pinning
    provider_name: str

class ModelRuntime:
    def __init__(
        self,
        *,
        api_key: str | None = None,                   # defaults to $HF_TOKEN
        timeout_s: float = 60.0,
        max_attempts: int = 4,
        concurrency_per_model: int = 4,
        cache: TranscriptCache | None = None,
    ) -> None: ...

    def pin(self, specs: list[ModelSpec]) -> list[PinnedModel]:
        """Resolve each spec's revision to a concrete commit SHA and record it.
        Call once at the start of a run; the returned list is what goes into
        the result metadata."""

    def chat(self, pinned: PinnedModel) -> ChatProvider:
        """Return a ChatProvider for this pinned model. The provider enforces
        per-model concurrency limits and integrates with the shared cache."""

    def specs(self) -> list[PinnedModel]:
        """All models pinned so far in this runtime instance."""
```

### Usage example

```python
from spinebench.runtime import ModelRuntime, ModelSpec
from spinebench.cache import DiskCache
from spinebench.evaluator import Evaluator

cache = DiskCache(Path("runs/week4/cache"))
runtime = ModelRuntime(cache=cache, concurrency_per_model=8)

subjects = runtime.pin([
    ModelSpec("Qwen/Qwen2.5-7B-Instruct"),
    ModelSpec("meta-llama/Llama-3.3-70B-Instruct"),
    # 48 more...
])
judges = runtime.pin([
    ModelSpec("Qwen/Qwen3-72B-Instruct"),
    ModelSpec("meta-llama/Llama-3.3-70B-Instruct"),
    ModelSpec("deepseek-ai/DeepSeek-V3"),
])
extractor = runtime.pin([ModelSpec("Qwen/Qwen3-32B-Instruct")])[0]

for subject in subjects:
    ev = Evaluator(
        subject=runtime.chat(subject),
        extractor=runtime.chat(extractor),
        judges=[runtime.chat(j) for j in judges],
        cache=cache,
    )
    # ... evaluate scenarios ...

# Emit run metadata with pinned SHAs
write_run_manifest({
    "subjects": [dataclasses.asdict(s) for s in subjects],
    "judges": [dataclasses.asdict(j) for j in judges],
    "extractor": dataclasses.asdict(extractor),
})
```

### What complexity it hides

- Resolving `revision=None` to a concrete commit SHA at pin time via `huggingface_hub.HfApi().model_info`.
- Per-model concurrency semaphores to avoid hammering one provider when it has a rate limit.
- Sharing the cache across providers so swapping a judge doesn't invalidate subject rollouts.
- Uniform token/timeout/retry configuration.
- Future async/await support via `AsyncChatProvider` — when we swap the underlying HF client to `AsyncInferenceClient`, only `ModelRuntime` changes.

## Dependency strategy

**Category: Ports & adapters.** The `ChatProvider` Protocol is already the port; `HFInferenceProvider` is the production adapter; `FakeProvider` (in `conftest.py`) is the test adapter.

What `ModelRuntime` adds:

- A **pinning adapter** (`huggingface_hub.HfApi`) that production uses; tests supply `FakePinningAdapter(return_sha="test-sha")`.
- A **concurrency adapter** (`asyncio.Semaphore` or `threading.Semaphore`) that tests substitute with no-op semaphores.

All three adapter seams are injected via the `ModelRuntime` constructor:

```python
class ModelRuntime:
    def __init__(
        self,
        *,
        pinner: RevisionPinner = HFApiPinner(),
        provider_factory: ProviderFactory = HFInferenceProviderFactory(),
        cache: TranscriptCache | None = None,
        ...
    ) -> None: ...
```

Production: defaults. Tests: all three substituted.

## Testing strategy

### New boundary tests on `ModelRuntime`

1. `pin(specs)` resolves `revision=None` to the pinner's returned SHA; explicit revisions pass through unchanged.
2. `pin` is idempotent: pinning the same spec twice returns the same `PinnedModel`.
3. `chat(pinned)` returns a provider whose `model_id` matches.
4. Concurrency: two parallel `generate()` calls to the same model block once the semaphore is saturated; parallel calls to different models do not block each other.
5. Cache integration: a `generate()` call with a warm cache bypasses the underlying provider.
6. Revision is surfaced: the returned provider exposes `pinned.revision` so downstream code can record it.

### Keep

- `test_hf_inference.py` (integration tests, marked `network`) — still useful as adapter-level tests.
- `FakeProvider` stays as the test adapter; no changes.

### Delete

- Nothing. `ChatProvider` base stays; `HFInferenceProvider` stays. `ModelRuntime` is additive.

## Implementation recommendations

The module should own:

- The full lifecycle of a model in a run: spec → pin → provider → concurrency → cache.
- Revision pinning invariants (once pinned, never changes within a run).
- Concurrency allocation (semaphores per pinned model).

The module should hide:

- The HF Hub API call that resolves `model_info(...).sha`.
- The semaphore bookkeeping.
- Per-provider rate-limit strategies.

The module should expose:

- `ModelRuntime(...).pin(specs)` for resolution.
- `ModelRuntime(...).chat(pinned)` for getting a `ChatProvider`.
- `ModelRuntime(...).specs()` for recording run metadata.

Caller migration:

- Evaluator constructor is unchanged (it still takes `ChatProvider` instances; they now come from `runtime.chat(...)`).
- Batch runner constructs subjects/extractor/judges once via `runtime.pin` + `runtime.chat` at the top of each run.
- Run metadata JSON now includes pinned SHAs for every model touched.

## Build order

1. Introduce `RevisionPinner` Protocol + `HFApiPinner` default, with tests against a fake.
2. `ModelRuntime.pin`.
3. Wrap `HFInferenceProvider` with a thin "concurrency + cache" layer at `chat()` time.
4. Thread `runtime` through the batch orchestrator from RFC 0001.
5. Record `pinned.revision` in every `ScenarioResult` row (schema bump).

TDD + multi-model-router routing per standing rule.

## Deferred

- Async/await support. The sync interface is fine for Week 3. Week 4's scale (675k calls) will force the issue; that upgrade can swap the adapter without changing `ModelRuntime`'s external API.
- Cross-runtime cache sharing (e.g., Redis). `DiskCache` is good enough until we hit multi-machine orchestration.
- Budget/credit tracking. Useful for Week-4 cost accounting but out of scope for the runtime layer; belongs in `batch.py` or a dedicated observability module.
