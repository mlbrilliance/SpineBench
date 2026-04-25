"""Thin shim: ``python scripts/analyze_pilot.py`` -> ``spinebench.cli.aggregate``.

The actual analysis lives in :mod:`spinebench.cli` so the installed console script
``spinebench-aggregate`` and this in-tree invocation execute identical code.
"""

from __future__ import annotations

from spinebench.cli import aggregate

if __name__ == "__main__":
    aggregate()
