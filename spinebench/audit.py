"""Leave-one-judge-out variance audit.

Pure data transform over ScenarioResult.verdicts — no LLM calls. For each scenario,
emits a baseline row (no judge dropped) plus one row per judge dropped, with the
majority label recomputed on the retained subset.
"""

from __future__ import annotations

from spinebench.scoring.aggregate import majority_label
from spinebench.types import AuditRow, ScenarioResult


def leave_one_judge_out(results: list[ScenarioResult]) -> list[AuditRow]:
    rows: list[AuditRow] = []
    for r in results:
        if r.failed:
            continue
        verdicts = r.verdicts
        rows.append(
            AuditRow(
                scenario_id=r.scenario_id,
                model_id=r.model_id,
                dropped_judge=None,
                majority_label=majority_label(verdicts),
                retained_verdicts=list(verdicts),
            )
        )
        for i, v in enumerate(verdicts):
            retained = verdicts[:i] + verdicts[i + 1 :]
            rows.append(
                AuditRow(
                    scenario_id=r.scenario_id,
                    model_id=r.model_id,
                    dropped_judge=v.judge_model,
                    majority_label=majority_label(retained),
                    retained_verdicts=retained,
                )
            )
    return rows
