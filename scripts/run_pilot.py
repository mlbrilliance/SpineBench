"""Thin shim: ``python scripts/run_pilot.py`` -> ``spinebench.cli.run``.

The actual pipeline lives in :mod:`spinebench.cli` so the installed console script
``spinebench-run`` and this in-tree invocation execute identical code.
"""

from __future__ import annotations

from spinebench.cli import run

if __name__ == "__main__":
    run()
