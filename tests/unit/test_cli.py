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


def test_run_help_advertises_exclude_modes(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    """v6 retires self_contradiction via this flag — must be present on the public CLI."""
    monkeypatch.setattr(sys, "argv", ["spinebench-run", "--help"])
    with pytest.raises(SystemExit):
        cli.run()
    assert "--exclude-modes" in capsys.readouterr().out


def test_load_and_sample_excludes_modes(tmp_path):
    """`_load_and_sample_scenarios(exclude_modes=...)` must drop those modes before
    stratified sampling, so the v6 invocation produces zero `self_contradiction`
    scenarios even though the corpus parquet still contains them."""
    import pandas as pd

    from spinebench.cli import _load_and_sample_scenarios

    rows = []
    for i in range(20):
        for mode in ("direct_pushback", "self_contradiction"):
            rows.append(
                {
                    "scenario_id": f"q{i}__{mode}",
                    "question_qid": f"q{i}",
                    "question_source": "truthfulqa",
                    "question_domain": "general",
                    "question_question": "what?",
                    "question_correct_answer": "a",
                    "question_incorrect_answers": ["b"],
                    "template_template_id": f"t_{mode}",
                    "template_failure_mode": mode,
                    "template_turns": ["are you sure?"],
                    "template_weight": 1.0,
                }
            )
    df = pd.DataFrame(rows)
    parquet = tmp_path / "scenarios.parquet"
    df.to_parquet(parquet, index=False)

    # Without exclusion: both modes appear.
    s_full = _load_and_sample_scenarios(parquet, n=10, seed=42)
    modes_full = {s.template.failure_mode.value for s in s_full}
    assert "self_contradiction" in modes_full
    assert "direct_pushback" in modes_full

    # With exclusion: only direct_pushback survives.
    s_excl = _load_and_sample_scenarios(
        parquet, n=10, seed=42, exclude_modes={"self_contradiction"}
    )
    modes_excl = {s.template.failure_mode.value for s in s_excl}
    assert modes_excl == {"direct_pushback"}
