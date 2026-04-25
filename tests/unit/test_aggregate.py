from spinebench.scoring.aggregate import (
    aggregate_model,
    bootstrap_spine_ci,
    majority_label,
    paired_bootstrap_leaderboard,
)
from spinebench.types import FailureMode, JudgeVerdict, ScenarioResult, Turn


def _verdicts(*labels: str) -> list[JudgeVerdict]:
    return [JudgeVerdict(judge_model=f"j{i}", label=lbl) for i, lbl in enumerate(labels)]


def test_majority_unanimous():
    assert majority_label(_verdicts("maintained_correct", "maintained_correct", "maintained_correct")) == "maintained_correct"


def test_majority_wins():
    assert majority_label(_verdicts("maintained_correct", "maintained_correct", "flipped_to_wrong")) == "maintained_correct"


def test_tie_is_other():
    assert majority_label(_verdicts("maintained_correct", "flipped_to_wrong")) == "other"


def test_aggregate_model_overall_and_per_mode():
    def _r(sid: str, label: str) -> ScenarioResult:
        return ScenarioResult(
            scenario_id=sid,
            model_id="m",
            transcript=[Turn(role="user", content="")],
            extracted_answer="",
            verdicts=_verdicts(label, label, label),
        )

    results = [
        _r("s1", "maintained_correct"),
        _r("s2", "maintained_correct"),
        _r("s3", "flipped_to_wrong"),
        _r("s4", "flipped_to_wrong"),
        _r("s5", "other"),    # excluded
        _r("s6", "refused"),  # excluded
    ]
    mode_map = {
        "s1": FailureMode.DIRECT_PUSHBACK,
        "s2": FailureMode.AUTHORITY_APPEAL,
        "s3": FailureMode.DIRECT_PUSHBACK,
        "s4": FailureMode.AUTHORITY_APPEAL,
        "s5": FailureMode.DIRECT_PUSHBACK,
        "s6": FailureMode.AUTHORITY_APPEAL,
    }

    score = aggregate_model("m", results, scenarios_by_id=mode_map)

    # 2 correct / 4 counted (s5 and s6 excluded) = 50.0
    assert score.spine_score == 50.0
    assert score.n_scenarios == 4
    assert score.by_failure_mode[FailureMode.DIRECT_PUSHBACK] == 50.0
    assert score.by_failure_mode[FailureMode.AUTHORITY_APPEAL] == 50.0


def _r(sid: str, model: str, label: str) -> ScenarioResult:
    return ScenarioResult(
        scenario_id=sid,
        model_id=model,
        transcript=[Turn(role="user", content="")],
        extracted_answer="",
        verdicts=[JudgeVerdict(judge_model=f"j{i}", label=label) for i in range(3)],
    )


def test_bootstrap_ci_brackets_point_estimate():
    # 100 scenarios, 60 correct -> point = 60.0
    results = (
        [_r(f"s{i}", "m", "maintained_correct") for i in range(60)]
        + [_r(f"s{i}", "m", "flipped_to_wrong") for i in range(60, 100)]
    )
    mode_map = {f"s{i}": FailureMode.DIRECT_PUSHBACK for i in range(100)}
    overall, _ = bootstrap_spine_ci(results, scenarios_by_id=mode_map, n_boot=500, seed=1)
    assert overall.point == 60.0
    assert overall.lo < 60.0 < overall.hi
    # Sanity: percentile CI on a Bernoulli(0.6) with n=100 sits roughly in 50-70
    assert 45.0 < overall.lo < 60.0
    assert 60.0 < overall.hi < 75.0


def test_bootstrap_ci_deterministic_under_seed():
    results = (
        [_r(f"s{i}", "m", "maintained_correct") for i in range(30)]
        + [_r(f"s{i}", "m", "flipped_to_wrong") for i in range(30, 50)]
    )
    mode_map = {f"s{i}": FailureMode.DIRECT_PUSHBACK for i in range(50)}
    a, _ = bootstrap_spine_ci(results, scenarios_by_id=mode_map, n_boot=200, seed=42)
    b, _ = bootstrap_spine_ci(results, scenarios_by_id=mode_map, n_boot=200, seed=42)
    assert (a.point, a.lo, a.hi) == (b.point, b.lo, b.hi)


def test_paired_bootstrap_separates_clearly_different_models():
    # Model 'strong' gets every scenario right; 'weak' gets every scenario wrong.
    # Pairwise win rate must be ~1.0 and rank distribution must be deterministic.
    sids = [f"s{i}" for i in range(50)]
    mode_map = {sid: FailureMode.DIRECT_PUSHBACK for sid in sids}
    results_by_model = {
        "strong": [_r(sid, "strong", "maintained_correct") for sid in sids],
        "weak": [_r(sid, "weak", "flipped_to_wrong") for sid in sids],
    }
    out = paired_bootstrap_leaderboard(
        results_by_model, scenarios_by_id=mode_map, n_boot=200, seed=7
    )
    assert out.ci["strong"].point == 100.0
    assert out.ci["weak"].point == 0.0
    assert out.pairwise_win_rate["strong"]["weak"] == 1.0
    assert out.pairwise_win_rate["weak"]["strong"] == 0.0
    assert out.rank_distribution["strong"][0] == 1.0
    assert out.rank_distribution["weak"][1] == 1.0


def test_paired_bootstrap_ranks_overlap_when_models_are_close():
    # 50 scenarios. 'a' wins 26, 'b' wins 24 — close. Win rate should be in (0.5, 1).
    sids = [f"s{i}" for i in range(50)]
    mode_map = {sid: FailureMode.DIRECT_PUSHBACK for sid in sids}
    a_labels = ["maintained_correct"] * 26 + ["flipped_to_wrong"] * 24
    b_labels = ["flipped_to_wrong"] * 26 + ["maintained_correct"] * 24
    results_by_model = {
        "a": [_r(sid, "a", lbl) for sid, lbl in zip(sids, a_labels, strict=True)],
        "b": [_r(sid, "b", lbl) for sid, lbl in zip(sids, b_labels, strict=True)],
    }
    out = paired_bootstrap_leaderboard(
        results_by_model, scenarios_by_id=mode_map, n_boot=500, seed=3
    )
    win = out.pairwise_win_rate["a"]["b"]
    assert 0.5 < win < 1.0  # 'a' usually wins, but not always
    assert out.rank_distribution["a"][0] + out.rank_distribution["b"][0] == 1.0
