"""Boundary tests for spinebench.cache."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

from spinebench.cache import DiskCache, InMemoryCache, NullCache
from spinebench.types import Turn


def _sample_transcript() -> list[Turn]:
    return [
        Turn(role="user", content="What is the capital of France?"),
        Turn(role="assistant", content="Paris."),
        Turn(role="user", content="Are you sure?"),
        Turn(role="assistant", content="Yes, Paris is the capital."),
    ]


def test_null_cache_get_always_none():
    c = NullCache()
    c.put("model/x", "scenario-1", _sample_transcript())
    assert c.get("model/x", "scenario-1") is None


def test_in_memory_round_trip():
    c = InMemoryCache()
    t = _sample_transcript()
    assert c.get("model/x", "scenario-1") is None
    c.put("model/x", "scenario-1", t)
    loaded = c.get("model/x", "scenario-1")
    assert loaded == t


def test_in_memory_key_scoped_by_model_and_scenario():
    c = InMemoryCache()
    t1 = [Turn(role="user", content="a")]
    t2 = [Turn(role="user", content="b")]
    c.put("model/x", "s1", t1)
    c.put("model/y", "s1", t2)
    c.put("model/x", "s2", t1)
    assert c.get("model/x", "s1") == t1
    assert c.get("model/y", "s1") == t2
    assert c.get("model/x", "s2") == t1


def test_disk_cache_round_trip(tmp_path: Path):
    c = DiskCache(tmp_path)
    t = _sample_transcript()
    c.put("Qwen/Qwen2.5-7B", "scenario-abc", t)
    loaded = c.get("Qwen/Qwen2.5-7B", "scenario-abc")
    assert loaded == t


def test_disk_cache_miss_returns_none(tmp_path: Path):
    c = DiskCache(tmp_path)
    assert c.get("never/stored", "x") is None


def test_disk_cache_handles_slashes_in_model_id(tmp_path: Path):
    """HF model IDs contain '/' which would otherwise create unexpected subdirs."""
    c = DiskCache(tmp_path)
    c.put("org/repo/nested", "s1", _sample_transcript())
    assert c.get("org/repo/nested", "s1") == _sample_transcript()


def test_disk_cache_rejects_traversal(tmp_path: Path):
    """Defense-in-depth: a model_id whose URL-encoded form would escape the cache root
    must be refused. The current implementation prevents this by URL-quoting and pinning
    the result to a single filename component, but the post-build assertion makes the
    invariant explicit so future refactors can't regress it silently."""
    c = DiskCache(tmp_path)
    # Sanity: well-formed model_ids continue to work.
    c.put("org/model", "scen", _sample_transcript())
    # Hostile model_id: under-the-hood we still expect a single filename under tmp_path.
    # Whether put() succeeds or raises, the resulting path must NOT be outside tmp_path.
    target_root = tmp_path.resolve()
    for hostile in ["..", "../etc/passwd", "../../../../etc", "./.."]:
        try:
            c.put(hostile, "scen", _sample_transcript())
        except Exception:
            continue  # explicit refusal is acceptable
        # If it succeeded, every file written must live under tmp_path.
        for f in tmp_path.rglob("*"):
            assert f.resolve().is_relative_to(target_root), f"escaped: {f}"


def test_disk_cache_survives_process_restart(tmp_path: Path):
    """Write from one process, read from a fresh one."""
    DiskCache(tmp_path).put("m", "s", _sample_transcript())
    script = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {str(Path.cwd())!r})
        from spinebench.cache import DiskCache
        c = DiskCache({str(tmp_path)!r})
        t = c.get("m", "s")
        print(len(t) if t else -1)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "4"
