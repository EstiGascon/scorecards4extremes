"""
Unit tests for the pure metric functions used by the diagnostics scripts.

These cover the numerical cores that are easy to get subtly wrong and hard to
eyeball in a plot:

  * ``twcrps_score``            (diagnostics/diagnose_twcrps_simple.py) — fair
                                 threshold-weighted CRPS.
  * ``calculate_skill_scores`` (diagnostics/diagnose_extremes.py) — POD/FAR/CSI/
                                 PSS/ETS from a contingency table.
  * ``threshold_conditional_mae`` / ``threshold_conditional_mse``
                                 (diagnostics/diagnose_extremes.py).

Run with:  .venv/bin/python -m pytest tests/ -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest

# The diagnostics scripts import ``case_studies.case_study_utils`` (needs the repo
# root on sys.path) and are themselves imported as top-level modules (needs the
# diagnostics/ directory on sys.path).
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "diagnostics"))

from diagnose_twcrps_simple import twcrps_score  # noqa: E402
from diagnose_extremes import (  # noqa: E402
    calculate_skill_scores,
    threshold_conditional_mae,
    threshold_conditional_mse,
)


# ---------------------------------------------------------------------------
# twcrps_score
# ---------------------------------------------------------------------------

def _ref_twcrps(ens, obs, T, event_type):
    """Independent reference for fair twCRPS using the O(M^2) pairwise form.

    fair_CRPS = mean_k |v_k - v_obs| - 1/(2 M (M-1)) * sum_{i,j} |v_i - v_j|
    where v is the threshold-weighted transform of the members.
    """
    ens = np.asarray(ens, dtype=float)
    obs = np.asarray(obs, dtype=float)
    T = np.asarray(T, dtype=float)
    M = ens.shape[1]
    if event_type == "above":
        v = np.maximum(ens - T[:, None], 0.0)
        vo = np.maximum(obs - T, 0.0)
    else:
        v = np.maximum(T[:, None] - ens, 0.0)
        vo = np.maximum(T - obs, 0.0)
    scores = []
    for i in range(len(obs)):
        t1 = np.mean(np.abs(v[i] - vo[i]))
        pair_sum = np.sum(np.abs(v[i][:, None] - v[i][None, :]))
        fair = pair_sum / (2.0 * M * (M - 1))
        scores.append(t1 - fair)
    return float(np.mean(scores))


@pytest.mark.parametrize("event_type", ["above", "below"])
def test_twcrps_matches_pairwise_reference(event_type):
    rng = np.random.default_rng(0)
    n, M = 40, 12
    ens = rng.normal(10.0, 3.0, size=(n, M))
    obs = rng.normal(10.0, 3.0, size=n)
    T = np.full(n, 11.0)
    got = twcrps_score(ens, obs, T, event_type)
    exp = _ref_twcrps(ens, obs, T, event_type)
    assert got == pytest.approx(exp, rel=1e-9, abs=1e-9)


def test_twcrps_perfect_below_threshold_is_zero():
    # All members equal the obs, and the obs is below T for an "above" event:
    # every threshold-weighted value is 0, so the score is exactly 0.
    obs = np.array([5.0, 6.0, 7.0])
    ens = np.repeat(obs[:, None], 8, axis=1)
    T = np.full(3, 20.0)
    assert twcrps_score(ens, obs, T, "above") == pytest.approx(0.0, abs=1e-12)


def test_twcrps_nonnegative_and_finite():
    rng = np.random.default_rng(7)
    ens = rng.normal(0, 1, size=(25, 10))
    obs = rng.normal(0, 1, size=25)
    T = np.zeros(25)
    val = twcrps_score(ens, obs, T, "above")
    assert np.isfinite(val)
    assert val >= 0.0


# ---------------------------------------------------------------------------
# calculate_skill_scores
# ---------------------------------------------------------------------------

def test_skill_scores_known_contingency():
    # Construct a table with hits=2, misses=2, false_alarms=1, correct_neg=5.
    obs = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=bool)
    fc  = np.array([1, 1, 0, 0, 1, 0, 0, 0, 0, 0], dtype=bool)
    s = calculate_skill_scores(fc, obs)

    assert int(s["hits"]) == 2
    assert int(s["misses"]) == 2
    assert int(s["false_alarms"]) == 1

    assert s["pod"] == pytest.approx(2 / 4)
    assert s["far"] == pytest.approx(1 / 3)
    assert s["csi"] == pytest.approx(2 / 5)
    assert s["pss"] == pytest.approx(0.5 - 1 / 6)

    hits_random = (4 * 3) / 10  # (hits+misses)*(hits+fa)/N = 1.2
    assert s["ets"] == pytest.approx((2 - hits_random) / (5 - hits_random))


def test_skill_scores_perfect_forecast():
    obs = np.array([1, 0, 1, 0, 1], dtype=bool)
    s = calculate_skill_scores(obs.copy(), obs.copy())
    assert s["pod"] == pytest.approx(1.0)
    assert s["far"] == pytest.approx(0.0)
    assert s["csi"] == pytest.approx(1.0)


def test_skill_scores_no_events_no_divzero():
    # No observed and no forecast extremes: guarded divisions must return 0.0,
    # not raise.
    obs = np.zeros(6, dtype=bool)
    fc = np.zeros(6, dtype=bool)
    s = calculate_skill_scores(fc, obs)
    assert s["pod"] == 0.0
    assert s["far"] == 0.0
    assert s["csi"] == 0.0


# ---------------------------------------------------------------------------
# threshold_conditional_mae / mse
# ---------------------------------------------------------------------------

def test_conditional_mae_above():
    obs = np.array([1.0, 2.0, 3.0, 10.0, 20.0])
    fc  = np.array([2.0, 2.0, 2.0, 12.0, 25.0])
    # var_short != '2t' → mask is obs >= threshold (10 and 20): |12-10|, |25-20|
    mae = threshold_conditional_mae(fc, obs, threshold=5.0, var_short="10ff")
    assert mae == pytest.approx((2.0 + 5.0) / 2)


def test_conditional_mse_above():
    obs = np.array([1.0, 2.0, 3.0, 10.0, 20.0])
    fc  = np.array([2.0, 2.0, 2.0, 12.0, 25.0])
    mse = threshold_conditional_mse(fc, obs, threshold=5.0, var_short="10ff")
    assert mse == pytest.approx((2.0**2 + 5.0**2) / 2)


def test_conditional_mae_cold_tail_2t():
    obs = np.array([1.0, 2.0, 3.0, 10.0, 20.0])
    fc  = np.array([2.0, 2.0, 2.0, 12.0, 25.0])
    # 2t with percentile < 50 → cold tail → mask is obs <= threshold (1,2,3):
    # |2-1|, |2-2|, |2-3| = 1, 0, 1
    mae = threshold_conditional_mae(
        fc, obs, threshold=5.0, threshold_percentile=1, var_short="2t"
    )
    assert mae == pytest.approx((1.0 + 0.0 + 1.0) / 3)


def test_conditional_mae_empty_mask_is_nan():
    obs = np.array([1.0, 2.0, 3.0])
    fc = np.array([1.0, 2.0, 3.0])
    # No obs >= 100 → mask empty → NaN (not a crash).
    assert np.isnan(threshold_conditional_mae(fc, obs, threshold=100.0, var_short="10ff"))
