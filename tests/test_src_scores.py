"""
Unit tests for the production scoring functions in ``src/`` — the functions that the
scorecard pipeline actually calls (previously only the ``diagnostics/`` copies were tested).

Covers the new scores (SEDI, EDI, SEEPS, CRPSS, reliability), the chaining-function
twMAE/twRMSE, the fair-CRPS / twCRPS numerics against the ``scores`` library, the
threshold-weighted decompositions, the date-block bootstrap helper and the signed-bias
scorecard transform.

Run with:  .venv/bin/python -m pytest tests/ -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

import det_scores as D          # noqa: E402
import ens_scores as E          # noqa: E402
import plot                     # noqa: E402
from scores.probability import crps_for_ensemble, tail_tw_crps_for_ensemble  # noqa: E402


# --------------------------------------------------------------------------- #
# SEDI / EDI
# --------------------------------------------------------------------------- #
def _ref_edi_sedi(H, F, kind):
    if kind == "SEDI":
        num = np.log(F) - np.log(H) - np.log(1 - F) + np.log(1 - H)
        den = np.log(F) + np.log(H) + np.log(1 - F) + np.log(1 - H)
    else:
        num = np.log(F) - np.log(H)
        den = np.log(F) + np.log(H)
    return num / den


def test_sedi_edi_match_reference_and_hit_false_rates():
    rng = np.random.default_rng(0)
    rho, N = 0.85, 400_000
    zc = rng.normal(size=N)
    zo = rho * zc + np.sqrt(1 - rho ** 2) * rng.normal(size=N)
    zf = rho * zc + np.sqrt(1 - rho ** 2) * rng.normal(size=N)
    T = np.percentile(zo, 99)
    H, F = D._hit_rate_false_alarm_rate(zf, zo, T, "above")
    assert D.calculate_sedi(zf, zo, T, "above") == pytest.approx(_ref_edi_sedi(H, F, "SEDI"), abs=1e-9)
    assert D.calculate_edi(zf, zo, T, "above") == pytest.approx(_ref_edi_sedi(H, F, "EDI"), abs=1e-9)


def test_sedi_non_degenerate_vs_ets():
    """ETS collapses toward 0 as the event gets rarer at fixed skill; SEDI stays bounded away."""
    rng = np.random.default_rng(1)
    rho, N = 0.85, 1_000_000
    zc = rng.normal(size=N)
    zo = rho * zc + np.sqrt(1 - rho ** 2) * rng.normal(size=N)
    zf = rho * zc + np.sqrt(1 - rho ** 2) * rng.normal(size=N)
    ets, sedi = {}, {}
    for p in (90, 99.9):
        T = np.percentile(zo, p)
        ets[p] = D.calculate_ets(zf, zo, T, "above")
        sedi[p] = D.calculate_sedi(zf, zo, T, "above")
    assert ets[99.9] < 0.6 * ets[90]            # ETS degrades substantially
    assert sedi[99.9] > 0.85 * sedi[90]         # SEDI roughly stable


def test_sedi_boundary_returns_nan():
    # perfect forecast -> H=1, F=0 -> undefined -> NaN (safe)
    obs = np.array([1.0, 2, 3, 10, 11, 12])
    assert np.isnan(D.calculate_sedi(obs, obs, 9.0, "above"))


# --------------------------------------------------------------------------- #
# twMAE / twRMSE  (chaining function)
# --------------------------------------------------------------------------- #
def test_twmae_is_chaining_not_indicator():
    f = np.array([33.0, 25, 28, 40])
    o = np.array([35.0, 35, 20, 42])
    T = 30.0
    chaining = float(np.mean(np.abs(np.maximum(f, T) - np.maximum(o, T))))
    assert D.calculate_twmae(f, o, T, "above") == pytest.approx(chaining)
    # and it differs from the (wrong) indicator-weighted version on the miss case
    indicator = float(np.mean((np.maximum(o, f) >= T) * np.abs(o - f)))
    assert not np.isclose(chaining, indicator)


def test_twmae_decomposition_sums():
    rng = np.random.default_rng(2)
    for et in ("above", "below"):
        f = rng.normal(20, 8, 500)
        o = rng.normal(20, 8, 500)
        T = 25.0
        comp = D.calculate_twmae_components(f, o, T, et)
        total = D.calculate_twmae(f, o, T, et)
        assert comp["twMAE_hits"] + comp["twMAE_misses"] + comp["twMAE_FA"] == pytest.approx(total, abs=1e-9)


def test_twrmse_handles_nan_threshold():
    rng = np.random.default_rng(4)
    f = rng.normal(20, 8, 200)
    o = rng.normal(20, 8, 200)
    thr = np.full(200, 25.0)
    thr[0] = np.nan
    val = D.calculate_twrmse(f, o, thr, "above")
    assert not np.isnan(val)   # NaN threshold must not poison the whole cell


# --------------------------------------------------------------------------- #
# Fair CRPS / twCRPS numerics vs the scores library
# --------------------------------------------------------------------------- #
def test_fair_crps_matches_library():
    rng = np.random.default_rng(5)
    fc = rng.normal(size=(2000, 51))
    obs = rng.normal(size=2000)
    mine = E._fair_crps_numpy(fc, obs)
    lib = float(crps_for_ensemble(xr.DataArray(fc, dims=["sample", "ensemble"]),
                                  xr.DataArray(obs, dims=["sample"]),
                                  ensemble_member_dim="ensemble", method="fair"))
    assert mine == pytest.approx(lib, abs=1e-8)


def test_twcrps_per_station_matches_library_for_constant_threshold():
    rng = np.random.default_rng(6)
    fc = rng.normal(size=(2000, 51))
    obs = rng.normal(size=2000)
    for et, tail in (("above", "upper"), ("below", "lower")):
        T = 0.5
        thr = np.full(2000, T)
        if et == "above":
            fcv, obv = np.maximum(fc, thr[:, None]), np.maximum(obs, thr)
        else:
            fcv, obv = np.minimum(fc, thr[:, None]), np.minimum(obs, thr)
        mine = E._fair_crps_numpy(fcv, obv)
        lib = float(tail_tw_crps_for_ensemble(xr.DataArray(fc, dims=["sample", "ensemble"]),
                                              xr.DataArray(obs, dims=["sample"]),
                                              ensemble_member_dim="ensemble",
                                              threshold=T, tail=tail, method="fair"))
        assert mine == pytest.approx(lib, abs=1e-8)


def test_twcrps_decomposition_sums():
    rng = np.random.default_rng(7)
    fc = rng.normal(size=(2000, 51))
    obs = rng.normal(size=2000)
    T = 0.5
    comp = E.calculate_twcrps_components(fc, obs, np.full(2000, T), "above")
    lib = float(tail_tw_crps_for_ensemble(xr.DataArray(fc, dims=["sample", "ensemble"]),
                                          xr.DataArray(obs, dims=["sample"]),
                                          ensemble_member_dim="ensemble",
                                          threshold=T, tail="upper", method="fair"))
    assert comp["twCRPS_hits"] + comp["twCRPS_misses"] + comp["twCRPS_FA"] == pytest.approx(lib, abs=1e-6)
    assert comp["twCRPS_accuracy"] - comp["twCRPS_spread"] == pytest.approx(lib, abs=1e-6)


# --------------------------------------------------------------------------- #
# CRPSS / reliability
# --------------------------------------------------------------------------- #
def _da(fc, obs):
    return xr.DataArray(fc, dims=["sample", "ensemble"]), xr.DataArray(obs, dims=["sample"])


def test_crpss_climatology_zero_perfect_one():
    rng = np.random.default_rng(8)
    n, m = 4000, 50
    obs = rng.normal(10, 3, n)
    f_clim, o = _da(rng.normal(10, 3, (n, m)), obs)
    f_perf, _ = _da(obs[:, None] + rng.normal(0, 0.2, (n, m)), obs)
    assert abs(E._compute_single_score("CRPSS", f_clim, o, np.full(n, 99.0), "above")) < 0.05
    assert E._compute_single_score("CRPSS", f_perf, o, np.full(n, 99.0), "above") > 0.9


def test_reliability_perfect_is_small_and_miscalibrated_positive():
    rng = np.random.default_rng(9)
    N = 20000
    # perfectly reliable: event occurs with prob exactly p_hat
    p = rng.uniform(0, 1, N)
    o_reliable = (rng.uniform(0, 1, N) < p).astype(float)
    assert E._brier_reliability(p, o_reliable) < 0.01
    # miscalibrated: forecast says p but outcome frequency is a distorted p**2
    o_bad = (rng.uniform(0, 1, N) < p ** 2).astype(float)
    assert E._brier_reliability(p, o_bad) > 0.02


# --------------------------------------------------------------------------- #
# SEEPS
# --------------------------------------------------------------------------- #
def test_seeps_perfect_zero_and_skill_ordering():
    rng = np.random.default_rng(10)
    nst, nd = 40, 200
    ob, st = [], []
    for s in range(nst):
        pdry = rng.uniform(0.3, 0.7)
        wet = rng.random(nd) > pdry
        ob.append(np.where(wet, rng.gamma(2.0, 3.0, nd), 0.0))
        st.append(np.full(nd, s))
    ob = np.concatenate(ob)
    st = np.concatenate(st)
    good = np.maximum(ob + rng.normal(0, 2.0, ob.size), 0.0)
    noskill = rng.permutation(ob)
    assert D.calculate_seeps(ob, ob, st) == pytest.approx(0.0, abs=1e-9)
    assert D.calculate_seeps(good, ob, st) < D.calculate_seeps(noskill, ob, st)
    assert D.calculate_seeps(noskill, ob, st) == pytest.approx(1.0, abs=0.15)  # equitability: climatology ~ 1


# --------------------------------------------------------------------------- #
# spread-skill ratio definition  (sqrt(mean var) / RMSE)
# --------------------------------------------------------------------------- #
def test_extreme_spread_skill_ratio_definition():
    rng = np.random.default_rng(11)
    n, m = 3000, 50
    T = 15.0
    obs = rng.normal(18, 2, n)          # ~93% above T; the rest must be excluded
    fc = obs[:, None] + rng.normal(0, 2.5, (n, m))
    f, o = _da(fc, obs)
    r = E._compute_single_score("extreme_spread_skill_ratio", f, o, T, "above")
    ext = obs >= T                      # score is conditioned on observed exceedance
    fe, oe = fc[ext], obs[ext]
    expected = np.sqrt(np.mean(np.var(fe, ddof=1, axis=1))) / np.sqrt(np.mean((fe.mean(axis=1) - oe) ** 2))
    assert r == pytest.approx(expected, rel=1e-6)


# --------------------------------------------------------------------------- #
# date-block bootstrap helper
# --------------------------------------------------------------------------- #
def test_date_blocks():
    assert E._date_blocks(None) is None
    assert E._date_blocks(np.array([1, 1, 2, 2]), min_dates=5) is None  # <5 distinct dates
    blocks = E._date_blocks(np.array([1, 1, 2, 3, 4, 5, 5]), min_dates=5)
    assert blocks is not None and len(blocks) == 5
    # every row index is covered exactly once across the blocks
    assert sorted(np.concatenate(blocks).tolist()) == list(range(7))


# --------------------------------------------------------------------------- #
# signed-bias scorecard transform
# --------------------------------------------------------------------------- #
def test_signed_bias_pct_bounded_and_oriented():
    sb = plot._signed_bias_pct
    assert float(sb(2.0, 1.0)) == pytest.approx(50.0)     # model2 halves |bias| -> +50%
    assert float(sb(2.0, 0.0)) == pytest.approx(100.0)    # model2 perfect
    assert float(sb(-2.0, 1.0)) == pytest.approx(50.0)    # orientation on |bias|, sign-agnostic
    assert float(sb(0.01, 2.0)) == pytest.approx(-99.5, abs=0.1)  # no explosion near zero baseline
    r1 = np.random.default_rng(12).uniform(-5, 5, 5000)
    r2 = np.random.default_rng(13).uniform(-5, 5, 5000)
    v = sb(r1, r2)
    assert np.nanmin(v) >= -100 - 1e-9 and np.nanmax(v) <= 100 + 1e-9


# --------------------------------------------------------------------------- #
# scorecard routing: FAR (perfect = 0) must not use the perfect = 1 skill branch
# --------------------------------------------------------------------------- #
def test_far_and_pod_scorecard_routing():
    assert "FAR" in plot._BOUNDED_ERROR_SCORES          # lower-is-better branch + error palette
    assert "POD" not in (plot._BOUNDED_ERROR_SCORES | plot._ERROR_SCORES
                         | plot._HIGHER_IS_BETTER_SCORES)  # perfect = 1 -> skill branch
    assert {"POD", "FAR"} <= set(plot._DETERMINISTIC_SCORES)  # detected as plottable


def test_seeps_skipped_for_non_precip_variable(capsys):
    cfg = {"variable": "2t", "scores": {"deterministic": ["SEEPS"]}}
    with pytest.raises(Exception):
        D.run_step6(cfg, None, 0.0, "above", {"fc1_name": "a", "fc2_name": "b"})
    assert "SEEPS is precipitation-only" in capsys.readouterr().out
