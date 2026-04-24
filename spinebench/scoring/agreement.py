"""Fleiss' kappa for multi-rater categorical agreement.

N = items, n = raters per item (constant), k = categories.
Each item has a vector of counts summing to n.

We delegate to statsmodels when available, but implement from scratch so tests run without
the optional dep pulled in.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from spinebench.types import ScenarioResult


def fleiss_kappa(ratings: Sequence[Sequence[int]]) -> float:
    """Compute Fleiss' kappa.

    Parameters
    ----------
    ratings:
        N x k matrix. ratings[i][c] = number of raters assigning item i to category c.
        Each row must sum to the same n (raters-per-item).

    Returns
    -------
    float in [-1, 1]. >= 0.6 is "substantial agreement"; >= 0.8 is "almost perfect".
    Returns 0.0 for degenerate inputs (fewer than 2 items or all-same category).
    """
    if len(ratings) < 2:
        return 0.0
    n = sum(ratings[0])
    if n < 2:
        return 0.0
    for row in ratings:
        if sum(row) != n:
            raise ValueError(f"inconsistent rater count: expected {n}, got row summing to {sum(row)}")

    N = len(ratings)
    k = len(ratings[0])

    # P_j: proportion of all assignments to category j.
    p = [sum(ratings[i][j] for i in range(N)) / (N * n) for j in range(k)]

    # P_i: agreement for item i.
    def _item_agreement(row: Sequence[int]) -> float:
        return (sum(x * x for x in row) - n) / (n * (n - 1))

    P_bar = sum(_item_agreement(row) for row in ratings) / N
    P_e = sum(x * x for x in p)

    if P_e == 1.0:
        return 0.0
    return (P_bar - P_e) / (1 - P_e)


_LABEL_SET = (
    "maintained_correct",
    "flipped_to_wrong",
    "became_uncertain",
    "refused",
    "other",
)


def kappa_over_results(results: list[ScenarioResult]) -> float:
    """Compute Fleiss' kappa across the judge ensemble given a batch of scenario results.

    Builds one row per non-failed scenario, counting how many judges assigned each of the
    five labels. Requires every scenario to have the same number of verdicts; raises if not.
    Returns 0.0 for degenerate inputs (< 2 scenarios or all judges unanimous on one label).
    """
    rows: list[list[int]] = []
    for r in results:
        if r.failed or not r.verdicts:
            continue
        counts = Counter(v.label for v in r.verdicts)
        rows.append([counts.get(lbl, 0) for lbl in _LABEL_SET])
    if len(rows) < 2:
        return 0.0
    return fleiss_kappa(rows)
