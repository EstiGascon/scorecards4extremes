#!/usr/bin/env python3
"""
Ensemble Extreme-Events Diagnostic Tool — Scorecards4Extremes
=============================================================
Probabilistic counterpart of ``diagnose_extremes.py`` (which is deterministic).
Reads the ensemble parquet produced by the pipeline (one column per member,
``fc1_member_*`` / ``fc2_member_*``), reproduces the pipeline's exact scoring
path (daily aggregation → per-station threshold → ``ens_scores`` functions) and
generates two diagnostic figures for one (day, season, orography) condition:

  FIGURE 1 — twCRPS <-> twQS score attribution
    Explains why the threshold-weighted CRPS (twCRPS) and the threshold-weighted
    quantile score (twQS) can rank two ensembles differently, even though both
    are computed from the same tail of the same forecast distributions. The
    identity twCRPS = 2 * integral_0^1 twQS_alpha d(alpha) (Gneiting & Ranjan,
    2011) means twCRPS is the continuous integral of the pinball loss over
    EVERY quantile level, while the reported twQS is the mean of only 3
    discrete levels (alpha = 0.90/0.95/0.99 by default). A model that performs
    differently across quantile levels can therefore win one score and lose
    the other; panel 3 makes this mechanism visible directly.
    1. Tail predictive CDFs vs observed CDF     (what twCRPS integrates)
    2. Quantile-coverage / reliability curve     (is each tail quantile calibrated?)
    3. twQS(alpha) across the whole tail, with the 3 reported levels marked
       — the bridge: twCRPS is the area under this whole curve; twQS reports
       only the 3 marked points
    4. twCRPS decomposition (hits / misses / false alarms)
    5. Head-to-head % difference for twCRPS vs each twQS level (who wins each)
    6. Auto-generated plain-language attribution summary

  FIGURE 2 — Ensemble calibration
    1. Rank (Talagrand) histogram — under/over-dispersion
    2. Spread–skill diagram
    3. Reliability diagram for P(exceed threshold)  (Brier / BSS context)

The scores are computed with the *same* functions the pipeline uses
(``src/ens_scores.calculate_ensemble_scores``), so every number matches the
heatmap CSV for the corresponding (season, orography, forecast_day) cell.

Usage
-----
  # 99th-percentile warm extremes, day 5, JJA, low terrain:
  python diagnostics/ensemble/diagnose_ens_extremes.py \
      --config configs/ensemble/config_2t_ens_local_p99obsclim_aifsvsifs.yaml \
      --day 5 --season JJA --orog low

  # cold extremes (event_type 'below' comes from the config), day 3, DJF:
  python diagnostics/ensemble/diagnose_ens_extremes.py \
      --config configs/ensemble/config_2t_ens_local_p1obsclim.yaml \
      --day 3 --season DJF --orog high
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — never allocates a display buffer
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xarray as xr
import yaml

# Repo root (parent of diagnostics/) first so `case_studies` and `src` resolve
# when the script is run as `python diagnostics/ensemble/diagnose_ens_extremes.py`.
_HERE = Path(__file__).resolve().parent            # diagnostics/ensemble
_DIAG = _HERE.parent                               # diagnostics/  (shared _style)
_REPO = _DIAG.parent                               # repository root
for _p in (_REPO, _REPO / "src", _DIAG, _HERE):
    sys.path.insert(0, str(_p))

import _style  # noqa: E402
from ens_scores import calculate_ensemble_scores, _TW_QS_ALPHAS, _alpha_label  # noqa: E402
from scores.continuous import tw_quantile_score  # noqa: E402
import threshold as _thr_module  # noqa: E402  (src/threshold.py)
from run import _aggregate_to_daily_mean  # noqa: E402  (exact pipeline daily aggregation)
import season_utils  # noqa: E402

warnings.filterwarnings("ignore")

# ============================================================================
# CONSTANTS
# ============================================================================

VARIABLE_LABELS = {
    "2t":     ("2m Temperature",    "°C"),
    "10ff":   ("10m Wind Speed",    "m/s"),
    "tp24":   ("24h Precipitation", "mm"),
    "aod500": ("AOD 500 nm",        ""),
    "go3":    ("Ozone (O3)",        "ppb"),
    "pm2p5":  ("PM2.5",             "µg/m³"),
}

SEASON_MONTHS = {
    "DJF": {12, 1, 2}, "MAM": {3, 4, 5},
    "JJA": {6, 7, 8},  "SON": {9, 10, 11},
    "ASO": {8, 9, 10},
}

OROG_FALLBACK = {"low": (0, 40), "flat": (0, 40),
                 "mid": (40, 120), "hilly": (40, 120),
                 "high": (120, 3000), "complex": (120, 3000)}

AREAS = {
    "europe":          [68, -15, 27, 50],
    "nh_extratropics": [90, -180, 20, 180],
    "tropics":         [20, -180, -20, 180],
}

C1, C2, COBS = _style.C_FC1, _style.C_FC2, _style.C_OBS
C_HIT, C_MISS, C_FA = _style.C_HIT, _style.C_MISS, _style.C_FA

# numpy 2.0 renamed trapz → trapezoid; support both.
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))


# ============================================================================
# DATA LOADING & PREPARATION
# ============================================================================

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def get_model_names(config):
    return (config["read_data"]["forecast_model1"]["name"],
            config["read_data"]["forecast_model2"]["name"])


def resolve_orog_range(config, orog):
    """Return (lo, hi) orography band, honouring config.filter.orography_ranges."""
    raw = config.get("filter", {}).get("orography_ranges", {})
    key = {"flat": "low", "hilly": "mid", "complex": "high"}.get(orog.lower(), orog.lower())
    if key in raw:
        return tuple(raw[key])
    if orog.lower() in OROG_FALLBACK:
        return OROG_FALLBACK[orog.lower()]
    raise SystemExit(f"ERROR: orography '{orog}' not in config orography_ranges "
                     f"({list(raw)}) nor built-in fallback ({list(OROG_FALLBACK)}).")


def find_parquet(config, day):
    base = Path(config["extract_points"]["output_path"])
    fp = next(base.glob(f"*_day{day}.parquet"), None)
    if fp is None:
        raise FileNotFoundError(
            f"No day-{day} parquet in {base}. Run the pipeline first: python run.py <config>")
    return fp


def filter_ensemble(df, config, season, orog_range):
    """Apply the pipeline's ENSEMBLE-path filters to one batch: physical-bounds
    QC (obs + every member) then season then orography.

    NB: the ensemble scoring loop in run.py deliberately does NOT apply the
    coastal (lsm) filter or a re-clip of area/date — those are extraction-time
    concerns. We match that exactly so the scores reproduce the heatmap CSV.
    """
    cfg = config.get("filter", {})
    var = config.get("variable", "")
    member_cols = [c for c in df.columns
                   if c.startswith("fc1_member_") or c.startswith("fc2_member_")]

    # Physical-bounds QC on the observation AND every member (matches run.py).
    if var == "2t":
        lo_t = cfg.get("min_valid_temperature", -60.0)
        hi_t = cfg.get("max_valid_temperature", 60.0)
        vm = (df["obs_value"] >= lo_t) & (df["obs_value"] <= hi_t)
        for c in member_cols:
            vm &= (df[c] >= lo_t) & (df[c] <= hi_t)
        df = df[vm]
    elif var == "tp24":
        mx = cfg.get("max_valid_precipitation")
        if mx is not None:
            df = df[df["obs_value"] <= mx]

    # Season
    _, months = season_utils.resolve_season(season)
    if months and "date" in df.columns:
        dm = pd.to_datetime(df["date"], format="%Y%m%d").dt.month
        df = df[dm.isin(months)]

    # Orography
    lo, hi = orog_range
    if "sdfor" in df.columns:
        df = df[(df["sdfor"] >= lo) & (df["sdfor"] < hi)]

    return df


def load_and_filter(parquet_path, config, season, orog_range, batch_size=100_000):
    """Chunk-read the (multi-GB) per-day parquet, filtering each batch.

    Chunked reading keeps memory bounded — the full per-day file is ~7 GB and a
    plain pd.read_parquet needs >20 GB, which OOM-kills on a login node.
    """
    pf = pq.ParquetFile(str(parquet_path))
    n_raw, chunks = 0, []
    for batch in pf.iter_batches(batch_size=batch_size):
        chunk = batch.to_pandas()
        # Downcast to float32 like run.py to halve the memory footprint.
        for col in chunk.select_dtypes(include="float64").columns:
            chunk[col] = chunk[col].astype("float32")
        n_raw += len(chunk)
        kept = filter_ensemble(chunk, config, season, orog_range)
        if not kept.empty:
            chunks.append(kept)
    if not chunks:
        return pd.DataFrame(), n_raw
    df = pd.concat(chunks, ignore_index=True)
    print(f"  Filter ({season}, sdfor∈[{orog_range[0]},{orog_range[1]})): "
          f"{n_raw:,} → {len(df):,} sub-daily rows")
    return df, n_raw


def prepare_daily(config, subdaily, model_names):
    """Reproduce run.py's local_obs_climatology order EXACTLY:
    Step 5 threshold on sub-daily → Step 5b daily aggregation → keep valid-threshold
    rows (the same subset _process_single_day scores).

    Returns (daily_df with `_obs_threshold` column, thr_arr, event_type).
    """
    thr, event_type = _thr_module.run_step5(config, subdaily)
    daily, thr2 = _aggregate_to_daily_mean(subdaily, thr, model_names, config)
    thr_series = thr2 if isinstance(thr2, pd.Series) else pd.Series(thr2, index=daily.index)
    valid = thr_series.notna().values
    daily = daily.loc[valid].reset_index(drop=True)
    thr_arr = thr_series[valid].to_numpy(dtype=float)
    daily["_obs_threshold"] = thr_arr
    print(f"  Daily aggregation + threshold: {len(daily):,} valid daily cases")
    return daily, thr_arr, event_type


# ============================================================================
# SCORING (exact pipeline reproduction) & COMPARISON QUANTITIES
# ============================================================================

def compute_official_scores(df, thr_arr, event_type, config):
    """Score both models with the pipeline's own function → exact heatmap match."""
    tw_qs_alphas = config.get("scores", {}).get("tw_qs_alphas")
    alphas = (tw_qs_alphas or _TW_QS_ALPHAS)[event_type]
    base = (["twCRPS", "fCRPS", "Brier", "BSS", "tw_quantile_score"]
            + [f"tw_quantile_score_{_alpha_label(a)}" for a in alphas]
            + ["extreme_spread_skill_ratio", "ens_spread",
               "ens_mean_bias", "ens_mean_rmse"])
    cfg_scores = list(config.get("scores", {}).get("ensemble", []))
    selected = list(dict.fromkeys(base + cfg_scores))  # union, order-preserving
    sc1 = calculate_ensemble_scores(df, thr_arr, event_type, selected, "fc1",
                                    tw_qs_alphas=tw_qs_alphas)
    sc2 = calculate_ensemble_scores(df, thr_arr, event_type, selected, "fc2",
                                    tw_qs_alphas=tw_qs_alphas)
    return sc1, sc2


def _member_matrix(df, prefix):
    cols = sorted([c for c in df.columns if c.startswith(prefix)],
                  key=lambda c: int(c.split("_")[-1]))
    return df[cols].values.astype(float)


def twqs_grid(members, obs, thr_arr, alphas, event_type):
    """Exact threshold-weighted quantile score at each α (pipeline formula).

    Uses scores.continuous.tw_quantile_score with the tail weight interval,
    identical to ens_scores.calculate_ensemble_scores.
    """
    fcst = xr.DataArray(members, dims=["sample", "ensemble"])
    obs_da = xr.DataArray(obs, dims=["sample"])
    thr_da = xr.DataArray(thr_arr, dims=["sample"])
    interval = (thr_da, np.inf) if event_type == "above" else (-np.inf, thr_da)
    out = []
    for a in alphas:
        q = fcst.quantile(a, dim="ensemble")
        try:
            out.append(float(tw_quantile_score(q, obs_da, a, interval_where_one=interval)))
        except Exception:
            out.append(np.nan)
    return np.array(out)


def coverage_curve(members, obs, alphas):
    """Empirical coverage P(obs ≤ ensemble α-quantile) for each α.

    A calibrated ensemble yields coverage ≈ α. Tail deviations explain why the
    quantile score at a given level rewards or penalises a model.
    """
    q = np.quantile(members, alphas, axis=1)          # (n_alpha, n_case)
    return (obs[None, :] <= q).mean(axis=1)           # (n_alpha,)


def rank_histogram(members, obs):
    """Talagrand rank of the observation among M members → counts over 0..M."""
    below = (members < obs[:, None]).sum(axis=1)
    ties = (members == obs[:, None]).sum(axis=1)
    # Randomise ties uniformly across the tied bins (standard treatment).
    rng = np.random.default_rng(0)
    rank = below + (rng.random(len(obs)) * (ties + 1)).astype(int)
    M = members.shape[1]
    counts, _ = np.histogram(rank, bins=np.arange(M + 2))
    return counts


def spread_skill(members, obs, n_bins=10):
    """Binned ensemble spread vs RMSE of the ensemble mean.

    Returns (bin_spread, bin_rmse). The 1:1 line is perfect reliability of the
    spread; points below it → under-dispersion.
    """
    spread = members.std(axis=1, ddof=1)
    err = members.mean(axis=1) - obs
    order = np.argsort(spread)
    spread, err = spread[order], err[order]
    edges = np.linspace(0, len(spread), n_bins + 1).astype(int)
    bs, br = [], []
    for i in range(n_bins):
        s, e = edges[i], edges[i + 1]
        if e - s < 5:
            continue
        bs.append(spread[s:e].mean())
        br.append(np.sqrt(np.mean(err[s:e] ** 2)))
    return np.array(bs), np.array(br)


def reliability(members, obs, thr_arr, event_type, n_bins=10):
    """Forecast exceedance probability vs observed exceedance frequency."""
    if event_type == "above":
        p = (members >= thr_arr[:, None]).mean(axis=1)
        o = (obs >= thr_arr).astype(float)
    else:
        p = (members <= thr_arr[:, None]).mean(axis=1)
        o = (obs <= thr_arr).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    fp, of, cnt = [], [], []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        fp.append(p[m].mean())
        of.append(o[m].mean())
        cnt.append(int(m.sum()))
    return np.array(fp), np.array(of), np.array(cnt)


def csv_crosscheck(config, var, season, orog, day):
    """Return {metric: value, metric_is_significant: bool} from the pipeline CSV."""
    rdir = config.get("save", {}).get("output_directory")
    if not rdir:
        return {}
    for name in (f"scores_by_leadtime_{var}_{season}_{orog.lower()}.csv",
                 f"scores_by_leadtime_{var}_{orog.lower()}.csv",
                 f"scores_by_leadtime_{var}_{season}.csv"):
        p = Path(rdir) / name
        if not p.exists():
            continue
        try:
            d = pd.read_csv(p)
            if "forecast_day" in d.columns:
                d = d[d["forecast_day"] == day]
            if not d.empty:
                return d.iloc[0].to_dict()
        except Exception:
            pass
    return {}


# ============================================================================
# NARRATIVE
# ============================================================================

def _pct(a, b):
    """Signed % difference of fc2 (b) relative to fc1 (a); negative → fc2 lower."""
    if a is None or b is None or not np.isfinite(a) or not np.isfinite(b) or abs(a) < 1e-12:
        return np.nan
    return (b - a) / abs(a) * 100.0


def build_narrative(sc1, sc2, m1, m2, event_type, unit, cov_alpha, cov1, cov2, sig):
    """Plain-language attribution of any twCRPS vs twQS disagreement."""
    L = []
    lower_is_better = True  # both twCRPS and twQS are negatively oriented

    def winner(a, b):
        if not (np.isfinite(a) and np.isfinite(b)):
            return "—"
        return m2 if (b < a) == lower_is_better else m1

    tw_w = winner(sc1.get("twCRPS", np.nan), sc2.get("twCRPS", np.nan))
    qs_w = winner(sc1.get("tw_quantile_score", np.nan), sc2.get("tw_quantile_score", np.nan))
    alphas = _TW_QS_ALPHAS[event_type]
    deepest = alphas[-1]
    deepest_key = f"tw_quantile_score_{_alpha_label(deepest)}"
    q99_w = winner(sc1.get(deepest_key, np.nan), sc2.get(deepest_key, np.nan))

    L.append("Identity:  twCRPS = 2 . integral_0^1 twQS(alpha) d(alpha)")
    L.append("(Gneiting & Ranjan, 2011) — both scores are built from the SAME")
    L.append("pinball loss on the SAME tail; they differ only in how much of the")
    L.append("alpha-axis (panel 3) is summarised: twCRPS = full integral,")
    L.append("twQS = mean of 3 discrete points (marked, dotted).")
    L.append("")
    L.append(f"twCRPS   winner (full-integral score): {tw_w}")
    L.append(f"twQS     winner (mean of {alphas}): {qs_w}")
    L.append(f"twQS α={deepest:.4g} winner (single deepest tail level): {q99_w}")
    L.append("")

    if tw_w != qs_w and tw_w != "—" and qs_w != "—":
        L.append(">> DISAGREEMENT — see panel 3: the twQS(alpha) curves for the two")
        L.append("   models must CROSS somewhere in the tail. One model is better on")
        L.append("   most of the alpha-axis (→ wins the integral, twCRPS); the other is")
        L.append("   better specifically near the discrete levels used for twQS (often")
        L.append("   dominated by the noisy, data-sparse alpha=0.99 point).")
        L.append("   Neither score is 'wrong' — they summarise different portions of")
        L.append("   the same alpha-axis. Use panel 3 to see where the crossover is.")
    else:
        L.append(">> AGREEMENT — the twQS(alpha) curves in panel 3 do not cross in the")
        L.append("   reported range, so the full-integral (twCRPS) and the 3-point")
        L.append("   summary (twQS) verdicts coincide.")
    L.append("")

    # Calibration cue from coverage at the highest reported level.
    a_ext = cov_alpha[-1] if event_type == "above" else cov_alpha[0]
    c1 = cov1[-1] if event_type == "above" else cov1[0]
    c2 = cov2[-1] if event_type == "above" else cov2[0]
    L.append(f"Coverage at α={a_ext:.2f}: target {a_ext:.2f} | {m1}={c1:.2f}  {m2}={c2:.2f}")
    for nm, c in ((m1, c1), (m2, c2)):
        if np.isfinite(c):
            if c < a_ext - 0.01:
                L.append(f"   {nm}: coverage < α → tail quantile too LOW (under-forecasts extremes).")
            elif c > a_ext + 0.01:
                L.append(f"   {nm}: coverage > α → tail quantile too HIGH (over-forecasts extremes).")
            else:
                L.append(f"   {nm}: well calibrated at this level.")
    L.append("")

    ssr1 = sc1.get("extreme_spread_skill_ratio", np.nan)
    ssr2 = sc2.get("extreme_spread_skill_ratio", np.nan)
    L.append(f"Extreme spread-skill ratio (1 = ideal): {m1}={ssr1:.2f}  {m2}={ssr2:.2f}")
    L.append("   <1 under-dispersive (over-confident), >1 over-dispersive.")

    if sig:
        flags = [k.replace("_is_significant", "") for k, v in sig.items()
                 if k.endswith("_is_significant") and bool(v)]
        if flags:
            L.append("")
            L.append("Bootstrap-significant differences: " + ", ".join(sorted(set(flags))[:8]))
    return "\n".join(L)


# ============================================================================
# FIGURES
# ============================================================================

def figure_score_attribution(members1, members2, obs, thr_arr, sc1, sc2, meta, out_path):
    """twCRPS <-> twQS attribution figure: explains score disagreements via the
    identity twCRPS = 2 * integral_0^1 twQS(alpha) d(alpha) (see module docstring).
    """
    m1, m2 = meta["m1"], meta["m2"]
    event_type, unit, var_disp = meta["event_type"], meta["unit"], meta["var_disp"]
    T = float(np.mean(thr_arr))

    # α grid spans (nearly) the full unit interval: the identity twCRPS =
    # 2∫₀¹twQS(α)dα requires integrating over ALL quantile levels of the
    # underlying distribution, not just the extreme ones — restricting the
    # grid to the tail alone breaks this panel's own validating identity
    # (the boxed "2·∫twQS dα" check would then read ≈0, not ≈twCRPS).
    # The OFFICIAL reported levels (now at-or-beyond the extreme threshold's
    # own percentile — see _TW_QS_ALPHAS) are simply marked on top as the
    # dotted reference lines.
    official = _TW_QS_ALPHAS[event_type]
    if event_type == "above":
        grid = np.unique(np.round(np.concatenate(
            [np.linspace(0.50, 0.999, 100), official]), 5))
    else:
        grid = np.unique(np.round(np.concatenate(
            [np.linspace(0.001, 0.50, 100), official]), 5))

    tw1 = twqs_grid(members1, obs, thr_arr, grid, event_type)
    tw2 = twqs_grid(members2, obs, thr_arr, grid, event_type)
    cov1 = coverage_curve(members1, obs, grid)
    cov2 = coverage_curve(members2, obs, grid)

    fig, axes = plt.subplots(2, 3, figsize=(21, 12))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        f"{var_disp}  ·  twCRPS ↔ twQS score attribution  ·  Day {meta['day']}  ·  "
        f"{meta['season']}  ·  {meta['orog'].upper()} terrain  ·  {m1} vs {m2}\n"
        f"twCRPS = 2·∫twQS(α)dα  →  same tail, same loss family, different summary of α"
        f"   (T̄ = {T:.2f} {unit}, {meta['n_case']:,} daily cases, "
        f"{meta['n_ext']:,} extremes = {100*meta['n_ext']/meta['n_case']:.1f}%)",
        fontsize=13.5, fontweight="bold")

    # ── Panel 1: tail predictive CDFs vs observed CDF ────────────────────────
    ax = axes[0, 0]
    all_vals = np.concatenate([obs, members1.ravel(), members2.ravel()])
    if event_type == "above":
        zlo, zhi = T - 0.5 * np.std(obs), np.nanpercentile(all_vals, 99.5)
    else:
        zlo, zhi = np.nanpercentile(all_vals, 0.5), T + 0.5 * np.std(obs)
    zs = np.linspace(zlo, zhi, 200)
    # Mean forecast CDF over cases = ECDF of the pooled member values (every case
    # contributes the same M members), computed cheaply via searchsorted.
    pooled1 = np.sort(members1.ravel())
    pooled2 = np.sort(members2.ravel())
    F1 = np.searchsorted(pooled1, zs, side="right") / pooled1.size
    F2 = np.searchsorted(pooled2, zs, side="right") / pooled2.size
    Gobs = np.searchsorted(np.sort(obs), zs, side="right") / obs.size
    ax.plot(zs, F1, color=C1, lw=2.2, label=f"{m1} mean CDF")
    ax.plot(zs, F2, color=C2, lw=2.2, label=f"{m2} mean CDF")
    ax.plot(zs, Gobs, color=COBS, lw=2.5, label="Observed CDF")
    ax.axvline(T, color=_style.C_THRESHOLD, ls="--", lw=1.5, label=f"Threshold T={T:.1f}")
    if event_type == "above":
        ax.axvspan(T, zhi, color="gold", alpha=0.08)
    else:
        ax.axvspan(zlo, T, color="gold", alpha=0.08)
    ax.set_xlim(zlo, zhi)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(f"Value z  ({unit})")
    ax.set_ylabel("Cumulative probability  F(z)")
    ax.set_title("1 · Forecast vs observed CDF in the tail\n"
                 "twCRPS = ∫[F(z) − 1{obs≤z}]² dz  over the shaded region (z ≥ T)",
                 fontsize=10.5)
    ax.legend(fontsize=8, loc="lower right")

    # ── Panel 2: quantile-coverage / reliability curve ───────────────────────
    ax = axes[0, 1]
    ax.plot(grid, grid, color=_style.C_REF, ls="--", lw=1.5, label="Perfect calibration")
    ax.plot(grid, cov1, color=C1, lw=2.2, marker="o", ms=3, label=m1)
    ax.plot(grid, cov2, color=C2, lw=2.2, marker="s", ms=3, label=m2)
    for a in official:
        ax.axvline(a, color="grey", ls=":", lw=0.8, alpha=0.6)
    ax.set_xlabel("Quantile level α")
    ax.set_ylabel("Empirical coverage  P(obs ≤ ensemble α-quantile)")
    ax.set_title("2 · Is each tail quantile calibrated?\n"
                 "Curve below the 1:1 line at α → that quantile is forecast too LOW",
                 fontsize=10.5)
    ax.legend(fontsize=8, loc="upper left")

    # ── Panel 3: twQS(α) across the tail — the bridge ────────────────────────
    ax = axes[0, 2]
    ax.plot(grid, tw1, color=C1, lw=2.2, label=f"{m1}  twQS(α)")
    ax.plot(grid, tw2, color=C2, lw=2.2, label=f"{m2}  twQS(α)")
    ax.fill_between(grid, np.minimum(tw1, tw2), alpha=0.06, color="grey")
    for a in official:
        ax.axvline(a, color="grey", ls=":", lw=0.9, alpha=0.7)
        ax.annotate(f"{a:.4g}", xy=(a, ax.get_ylim()[1]), fontsize=7,
                    color="grey", ha="center", va="top")
    # Area under twQS ≈ twCRPS/2 (illustrative bridge; exact for un-weighted CRPS)
    area1 = 2 * _trapz(tw1, grid)
    area2 = 2 * _trapz(tw2, grid)
    ax.set_xlabel("Quantile level α")
    ax.set_ylabel("twQS(α)  (pinball loss at level α, tail-weighted)")
    ax.set_title("3 · The shared quantity behind both scores\n"
                 "twCRPS = 2·∫₀¹twQS(α)dα (full domain); reported twQS = mean of the\n"
                 "3 dotted points only — a disagreement means the curves CROSS",
                 fontsize=10.5)
    ax.legend(fontsize=8, loc="best")
    same_order = (area2 >= area1) == (sc2.get('twCRPS', np.nan) >= sc1.get('twCRPS', np.nan))
    ax.text(0.02, 0.02,
            f"Partial-domain check (α∈[{grid.min():.2f},{grid.max():.3g}], NOT the full\n"
            f"identity \u2014 shown \u2192 winner {'agrees' if same_order else 'DISAGREES'} with exact twCRPS):\n"
            f"2·∫twQS(α)dα here:  {m1}={area1:.3f}  {m2}={area2:.3f}\n"
            f"twCRPS (pipeline, exact, full tail): {m1}={sc1.get('twCRPS', np.nan):.3f}  "
            f"{m2}={sc2.get('twCRPS', np.nan):.3f}",
            transform=ax.transAxes, fontsize=7.2, va="bottom",
            bbox=dict(boxstyle="round", fc="white", ec="grey", alpha=0.85))

    # ── Panel 4: twCRPS decomposition ────────────────────────────────────────
    ax = axes[1, 0]
    comps = ["twCRPS_hits", "twCRPS_misses", "twCRPS_FA"]
    labels = ["Hits", "Misses", "False alarms"]
    colors = [C_HIT, C_MISS, C_FA]
    x = np.arange(len(comps))
    w = 0.38
    v1 = [sc1.get(c, np.nan) for c in comps]
    v2 = [sc2.get(c, np.nan) for c in comps]
    b1 = ax.bar(x - w / 2, v1, w, color=colors, alpha=0.55, edgecolor=C1, lw=2, label=m1)
    b2 = ax.bar(x + w / 2, v2, w, color=colors, alpha=0.95, edgecolor=C2, lw=2, label=m2)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            if np.isfinite(h):
                ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(f"Contribution to mean twCRPS  ({unit})")
    ax.set_title("4 · twCRPS split by daily outcome at threshold T\n"
                 "hit = obs & forecast both extreme · miss = obs extreme only · "
                 "FA = forecast extreme only", fontsize=10.5)
    ax.text(0.03, 0.97,
            f"twCRPS total\n{m1} = {sc1.get('twCRPS', np.nan):.3f}\n"
            f"{m2} = {sc2.get('twCRPS', np.nan):.3f}",
            transform=ax.transAxes, va="top", fontsize=8,
            bbox=dict(boxstyle="round", fc="white", ec="#cccccc", alpha=0.9))
    leg = [mpatches.Patch(facecolor="0.7", edgecolor=C1, lw=2, label=f"{m1} (edge)"),
           mpatches.Patch(facecolor="0.7", edgecolor=C2, lw=2, label=f"{m2} (edge)")]
    ax.legend(handles=leg, fontsize=8, loc="upper right")

    # ── Panel 5: head-to-head % difference ───────────────────────────────────
    ax = axes[1, 1]
    metrics = [("twCRPS", "twCRPS (whole tail)"),
               ("tw_quantile_score", "twQS (mean of levels)"),
               (f"tw_quantile_score_{_alpha_label(official[0])}", f"twQS α={official[0]:.4g}"),
               (f"tw_quantile_score_{_alpha_label(official[1])}", f"twQS α={official[1]:.4g}"),
               (f"tw_quantile_score_{_alpha_label(official[2])}", f"twQS α={official[2]:.4g}")]
    diffs = [_pct(sc1.get(k), sc2.get(k)) for k, _ in metrics]
    ypos = np.arange(len(metrics))[::-1]
    bar_colors = [C2 if (d < 0) else C1 for d in diffs]  # negative → fc2 better
    ax.barh(ypos, diffs, color=bar_colors, alpha=0.85, edgecolor="white")
    ax.axvline(0, color="black", lw=1.2)
    finite = [d for d in diffs if np.isfinite(d)]
    span = max(abs(min(finite, default=-1)), abs(max(finite, default=1)), 1)
    ax.set_xlim(-span * 1.45, span * 1.45)
    for y, d in zip(ypos, diffs):
        if np.isfinite(d):
            ax.annotate(f"{d:+.1f}%", xy=(d, y),
                        xytext=(4 if d >= 0 else -4, 0), textcoords="offset points",
                        ha="left" if d >= 0 else "right", va="center", fontsize=8,
                        fontweight="bold")
    ax.set_yticks(ypos)
    ax.set_yticklabels([lbl for _, lbl in metrics], fontsize=9)
    ax.set_xlabel(f"% difference ({m2} − {m1}) / |{m1}|\n"
                  f"(blue bar → {m1} better · orange bar → {m2} better)",
                  fontsize=9)
    ax.set_title("5 · Same two models, one row per score\n"
                 "a sign flip down the list = the scores disagree", fontsize=10.5)

    # ── Panel 6: narrative ───────────────────────────────────────────────────
    ax = axes[1, 2]
    ax.axis("off")
    narrative = build_narrative(sc1, sc2, m1, m2, event_type, unit,
                                grid, cov1, cov2, meta.get("sig", {}))
    ax.text(0.0, 1.0, "6 · Attribution summary", fontsize=12, fontweight="bold",
            va="top", transform=ax.transAxes)
    ax.text(0.0, 0.94, narrative, fontsize=8.4, family="monospace",
            va="top", transform=ax.transAxes,
            bbox=dict(boxstyle="round", fc="#f6f6f6", ec="#cccccc"))

    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ wrote {out_path}")
    return cov1, cov2, grid


def figure_calibration(members1, members2, obs, thr_arr, sc1, sc2, meta, out_path):
    m1, m2 = meta["m1"], meta["m2"]
    event_type, unit, var_disp = meta["event_type"], meta["unit"], meta["var_disp"]

    fig, axes = plt.subplots(1, 3, figsize=(21, 6.2))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        f"{var_disp}  ·  Ensemble calibration  ·  Day {meta['day']}  ·  "
        f"{meta['season']}  ·  {meta['orog'].upper()} terrain  ·  {m1} vs {m2}",
        fontsize=14, fontweight="bold")

    # ── Rank histogram ───────────────────────────────────────────────────────
    ax = axes[0]
    M = members1.shape[1]
    rh1 = rank_histogram(members1, obs)
    rh2 = rank_histogram(members2, obs)
    ranks = np.arange(M + 1)
    flat = len(obs) / (M + 1)
    w = 0.42
    ax.bar(ranks - w / 2, rh1, w, color=C1, alpha=0.75, label=m1)
    ax.bar(ranks + w / 2, rh2, w, color=C2, alpha=0.75, label=m2)
    ax.axhline(flat, color=_style.C_REF, ls="--", lw=1.5, label="flat (calibrated)")
    ax.set_xlabel(f"Rank of observation among {M} members")
    ax.set_ylabel("Count")
    ax.set_title("Rank (Talagrand) histogram\nU-shape → under-dispersive · dome → over-dispersive",
                 fontsize=11)
    ax.legend(fontsize=8)

    # ── Spread–skill ─────────────────────────────────────────────────────────
    ax = axes[1]
    for members, c, nm in ((members1, C1, m1), (members2, C2, m2)):
        bs, br = spread_skill(members, obs)
        ax.plot(bs, br, color=c, lw=2, marker="o", ms=5, label=nm)
    lims = ax.get_xlim(), ax.get_ylim()
    hi = max(lims[0][1], lims[1][1])
    ax.plot([0, hi], [0, hi], color=_style.C_REF, ls="--", lw=1.5, label="1:1 (reliable spread)")
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_xlabel(f"Ensemble spread (std)  ({unit})")
    ax.set_ylabel(f"RMSE of ensemble mean  ({unit})")
    ax.set_title("Spread–skill diagram\nabove 1:1 → under-dispersive (over-confident)",
                 fontsize=11)
    ax.legend(fontsize=8)

    # ── Reliability diagram ──────────────────────────────────────────────────
    ax = axes[2]
    ax.plot([0, 1], [0, 1], color=_style.C_REF, ls="--", lw=1.5, label="perfect")
    for members, c, nm, sc in ((members1, C1, m1, sc1), (members2, C2, m2, sc2)):
        fp, of, cnt = reliability(members, obs, thr_arr, event_type)
        sizes = 20 + 180 * np.array(cnt) / max(1, max(cnt))
        ax.plot(fp, of, color=c, lw=1.8, label=f"{nm} (BSS={sc.get('BSS', np.nan):.2f})")
        ax.scatter(fp, of, s=sizes, color=c, alpha=0.6, edgecolor="white", zorder=5)
    obs_base = (obs >= thr_arr).mean() if event_type == "above" else (obs <= thr_arr).mean()
    ax.axhline(obs_base, color="grey", ls=":", lw=1, label=f"base rate = {obs_base:.3f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Forecast probability  P(exceed T)")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Reliability diagram (exceedance of T)\nmarker size ∝ #cases in bin",
                 fontsize=11)
    ax.legend(fontsize=8, loc="upper left")

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ wrote {out_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--day", type=int, default=5)
    ap.add_argument("--season", default="JJA")
    ap.add_argument("--orog", default="low")
    ap.add_argument("--output-dir", default=None,
                    help="Default: case_study_output/ens_diagnostics/<config stem>")
    args = ap.parse_args()

    _style.apply_style()
    config = load_config(args.config)
    var = config["variable"]
    m1, m2 = get_model_names(config)
    model_names = {"fc1_name": m1, "fc2_name": m2}
    var_disp, unit = VARIABLE_LABELS.get(var, (var, ""))

    orog_range = resolve_orog_range(config, args.orog)
    parquet = find_parquet(config, args.day)
    print(f"Config  : {args.config}")
    print(f"Variable: {var}  |  Day {args.day}  |  Season {args.season}  |  "
          f"Orog {args.orog} {orog_range}")
    print(f"Models  : {m1} vs {m2}")
    print(f"Parquet : {parquet.name}")

    subdaily, _ = load_and_filter(parquet, config, args.season, orog_range)
    if subdaily.empty:
        raise SystemExit("No rows after filtering — check season/orography.")
    df, thr_arr, event_type = prepare_daily(config, subdaily, model_names)
    del subdaily
    if len(df) < 30:
        raise SystemExit(f"Only {len(df)} valid cases — too few for diagnostics.")

    members1 = _member_matrix(df, "fc1_member_")
    members2 = _member_matrix(df, "fc2_member_")
    obs = df["obs_value"].values.astype(float)

    # Keep only fully-finite cases for the manual (numpy) diagnostic panels.
    good = (np.isfinite(obs) & np.isfinite(thr_arr)
            & np.isfinite(members1).all(axis=1) & np.isfinite(members2).all(axis=1))
    members1, members2, obs, thr_arr = members1[good], members2[good], obs[good], thr_arr[good]
    n_ext = int((obs >= thr_arr).sum() if event_type == "above" else (obs <= thr_arr).sum())
    print(f"  Scoring {len(obs):,} cases ({n_ext:,} extremes, "
          f"{100*n_ext/len(obs):.1f}%)")

    sc1, sc2 = compute_official_scores(df.loc[good].reset_index(drop=True),
                                       thr_arr, event_type, config)
    sig = csv_crosscheck(config, var, args.season, args.orog, args.day)
    if sig:
        for k in ("twCRPS", "tw_quantile_score"):
            if f"{k}_fc1" in sig:
                print(f"  CSV cross-check {k}: fc1={sig[f'{k}_fc1']:.4f} "
                      f"fc2={sig.get(f'{k}_fc2', float('nan')):.4f} | "
                      f"computed fc1={sc1.get(k, float('nan')):.4f} "
                      f"fc2={sc2.get(k, float('nan')):.4f}")

    out_dir = Path(args.output_dir) if args.output_dir else \
        _REPO / "case_study_output" / "ens_diagnostics" / Path(args.config).stem
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{var}_day{args.day}_{args.season}_{args.orog}"

    meta = dict(m1=m1, m2=m2, event_type=event_type, unit=unit, var_disp=var_disp,
                day=args.day, season=args.season, orog=args.orog,
                n_case=len(obs), n_ext=n_ext, sig=sig)

    figure_score_attribution(members1, members2, obs, thr_arr, sc1, sc2, meta,
                             out_dir / f"ens_score_attribution_{tag}.png")
    figure_calibration(members1, members2, obs, thr_arr, sc1, sc2, meta,
                       out_dir / f"ens_calibration_{tag}.png")

    print(f"\nDone. Figures in {out_dir}")


if __name__ == "__main__":
    main()
