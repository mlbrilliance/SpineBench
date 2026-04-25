"""Smoke tests for the public CLI entry points.

These don't run a real pilot — they just confirm that the advertised commands
parse arguments and dispatch into the package, so ``pip install -e . && spinebench-run``
behaves like a command rather than a stub.
"""

from __future__ import annotations

import sys

import pytest

from spinebench import cli


def test_run_help_exits_clean(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    monkeypatch.setattr(sys, "argv", ["spinebench-run", "--help"])
    with pytest.raises(SystemExit) as exc_info:
        cli.run()
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "--subjects" in out
    assert "--n-scenarios" in out


def test_aggregate_help_exits_clean(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.setattr(sys, "argv", ["spinebench-aggregate", "--help"])
    with pytest.raises(SystemExit) as exc_info:
        cli.aggregate()
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "pilot_dir" in out
    assert "--bootstrap-iters" in out


def test_run_requires_subjects(monkeypatch: pytest.MonkeyPatch):
    """argparse should reject an invocation missing the required --subjects flag."""
    monkeypatch.setattr(sys, "argv", ["spinebench-run", "--output-dir", "/tmp/out"])
    with pytest.raises(SystemExit) as exc_info:
        cli.run()
    assert exc_info.value.code != 0


def test_aggregate_rejects_missing_results_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    """analyze_pilot should bail cleanly when the pilot dir has no results.parquet."""
    monkeypatch.setattr(sys, "argv", ["spinebench-aggregate", str(tmp_path)])
    with pytest.raises(SystemExit) as exc_info:
        cli.aggregate()
    assert exc_info.value.code != 0


def test_run_pilot_function_is_importable():
    """run_pilot and analyze_pilot are imported by scripts/*.py shims."""
    from spinebench.cli import analyze_pilot, run_pilot

    assert callable(run_pilot)
    assert callable(analyze_pilot)
