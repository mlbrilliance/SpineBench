"""Transcript cache backends.

Keyed by (model_id, scenario_id). Only the subject-model rollout is cached; judges and
extractor always run fresh (so swapping judges doesn't invalidate expensive rollouts).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Protocol
from urllib.parse import quote

from spinebench.types import Turn


class TranscriptCache(Protocol):
    def get(self, model_id: str, scenario_id: str) -> list[Turn] | None: ...
    def put(self, model_id: str, scenario_id: str, transcript: list[Turn]) -> None: ...


class NullCache:
    """Always-miss cache. The default when no cache is configured."""

    def get(self, model_id: str, scenario_id: str) -> list[Turn] | None:
        return None

    def put(self, model_id: str, scenario_id: str, transcript: list[Turn]) -> None:
        return None


class InMemoryCache:
    """Process-local dict-backed cache. Wiped when the process exits."""

    def __init__(self) -> None:
        self._store: dict[tuple[str, str], list[Turn]] = {}

    def get(self, model_id: str, scenario_id: str) -> list[Turn] | None:
        return self._store.get((model_id, scenario_id))

    def put(self, model_id: str, scenario_id: str, transcript: list[Turn]) -> None:
        self._store[(model_id, scenario_id)] = list(transcript)


class DiskCache:
    """Persistent cache: one JSON file per (model_id, scenario_id).

    Atomic writes via os.replace. Model IDs are URL-quoted to survive the `/` characters
    common in HF repo names without creating unwanted subdirectories.
    """

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def _path(self, model_id: str, scenario_id: str) -> Path:
        safe_model = quote(model_id, safe="")
        safe_scenario = quote(scenario_id, safe="")
        path = self._root / f"{safe_model}__{safe_scenario}.json"
        # Defense-in-depth: assert the resolved path stays under the cache root.
        # Today the f-string scaffolding makes traversal impossible (the result is
        # always a single filename component), but this guards against silent
        # regressions if the path-build pattern ever changes.
        if not path.resolve().is_relative_to(self._root.resolve()):
            raise ValueError(
                f"cache path escaped root: {path} not under {self._root}"
            )
        return path

    def get(self, model_id: str, scenario_id: str) -> list[Turn] | None:
        path = self._path(model_id, scenario_id)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return [Turn(**t) for t in data]

    def put(self, model_id: str, scenario_id: str, transcript: list[Turn]) -> None:
        path = self._path(model_id, scenario_id)
        tmp = path.with_suffix(path.suffix + ".tmp")
        payload = json.dumps([t.model_dump() for t in transcript], ensure_ascii=False)
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, path)
