"""
case_study_utils.py — shared utilities for case study identification and scoring.

Threshold loading, event classification, per-date metric computation, and
composite scoring used by find_case_studies.py and plot_case_study.py.
"""

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow importing from the parent scorecards4extremes package
sys.path.insert(0, str(Path(__file__).parent.parent))
import threshold as _thr_module


# ─── Constants ────────────────────────────────────────────────────────────────

EUROPE_REGIONS = {
    "NW": (52, 72,  -25, 10),   # lat_min, lat_max, lon_min, lon_max
    "NE": (52, 72,   10, 40),
    "CE": (42, 52,    5, 30),
    "SW": (35, 52,  -10,  5),
    "SE": (35, 52,    5, 40),
}


# ─── Threshold loading ────────────────────────────────────────────────────────

def load_per_station_thresholds(config: dict, df: pd.DataFrame) -> np.ndarray:
    """Return a float32 array of per-station thresholds aligned with df rows.

    Uses the threshold method defined in the config.  Falls back to a global
    pooled percentile if the per-station computation fails.

    Returns
    -------
    np.ndarray of shape (len(df),)
    """
    cfg_copy = copy.deepcopy(config)
    try:
        thr, _ = _thr_module.run_step5(cfg_copy, df)
        if isinstance(thr, (int, float, np.floating)):
            return np.full(len(df), float(thr), dtype=np.float32)
        thr_arr = np.asarray(thr, dtype=np.float32)
        if len(thr_arr) == len(df):
            return thr_arr
        # Scalar result broadcast
        return np.full(len(df), float(np.nanmean(thr_arr)), dtype=np.float32)
    except Exception as exc:
        pct = (config.get("threshold", {})
               .get("local_obs_climatology", {})
               .get("percentile", 99))
        event_type = config.get("threshold", {}).get("event_type", "above")
        fallback = float(np.nanpercentile(df["obs_value"].values, pct))
        print(f"  ⚠  Threshold fallback (pooled p{pct}={fallback:.3f}): {exc}")
        return np.full(len(df), fallback, dtype=np.float32)


def get_event_type(config: dict) -> str:
    """Return 'above' or 'below' from the config threshold section."""
    cfg = config.get("threshold", {})
    if cfg.get("method") == "fixed":
        return cfg.get("fixed", {}).get("event_type", "above")
    return cfg.get("event_type", "above")


# ─── Ensemble / deterministic value extraction ───────────────────────────────

def extract_forecast_values(df: pd.DataFrame, mode: str,
                             exceedance_pct_threshold: float = 0.5
                             ) -> tuple[np.ndarray, np.ndarray]:
    """Return (fc1_vals, fc2_vals) as 1-D float arrays for the given mode.

    deterministic : fc1_value / fc2_value columns
    ensemble      : ensemble mean of fc1_member_* / fc2_member_* columns.
                    For exceedance probability, call
                    ``extract_exceedance_probability`` instead.
    """
    if mode == "deterministic":
        fc1 = df["fc1_value"].values.astype(np.float32)
        fc2 = df["fc2_value"].values.astype(np.float32)
    else:
        fc1_cols = [c for c in df.columns if c.startswith("fc1_member_")]
        fc2_cols = [c for c in df.columns if c.startswith("fc2_member_")]
        fc1 = df[fc1_cols].mean(axis=1).values.astype(np.float32)
        fc2 = df[fc2_cols].mean(axis=1).values.astype(np.float32)
    return fc1, fc2


def extract_exceedance_probability(df: pd.DataFrame,
                                   T_arr: np.ndarray,
                                   event_type: str = "above"
                                   ) -> tuple[np.ndarray, np.ndarray]:
    """For ensemble data, return P(member exceeds T) per station for fc1 and fc2.

    Returns (prob1, prob2) as float arrays in [0, 1].
    """
    fc1_cols = [c for c in df.columns if c.startswith("fc1_member_")]
    fc2_cols = [c for c in df.columns if c.startswith("fc2_member_")]
    fc1_mat = df[fc1_cols].values.astype(np.float32)   # (N, n_members)
    fc2_mat = df[fc2_cols].values.astype(np.float32)
    T = T_arr[:, None]  # broadcast to (N, 1)
    if event_type == "above":
        prob1 = (fc1_mat > T).mean(axis=1)
        prob2 = (fc2_mat > T).mean(axis=1)
    else:
        prob1 = (fc1_mat < T).mean(axis=1)
        prob2 = (fc2_mat < T).mean(axis=1)
    return prob1.astype(np.float32), prob2.astype(np.float32)


# ─── Event classification ─────────────────────────────────────────────────────

def classify_events(obs: np.ndarray, fc: np.ndarray,
                    T: np.ndarray, event_type: str
                    ) -> dict[str, np.ndarray]:
    """Return boolean masks for hit / miss / false_alarm / correct_neg.

    For deterministic forecasts fc is the single-valued prediction.
    For ensemble, pass the ensemble mean or a probability array; pass
    use_probability=True to threshold on 0.5 exceedance probability.
    """
    if event_type == "above":
        obs_ext = obs > T
        fc_ext  = fc  > T
    else:
        obs_ext = obs < T
        fc_ext  = fc  < T
    return {
        "hit":      obs_ext & fc_ext,
        "miss":     obs_ext & ~fc_ext,
        "false_alarm": ~obs_ext & fc_ext,
        "correct_neg": ~obs_ext & ~fc_ext,
    }


def classify_events_probabilistic(obs: np.ndarray, prob: np.ndarray,
                                   T: np.ndarray, event_type: str,
                                   prob_threshold: float = 0.5
                                   ) -> dict[str, np.ndarray]:
    """Classify using ensemble exceedance probability instead of mean."""
    if event_type == "above":
        obs_ext = obs > T
    else:
        obs_ext = obs < T
    fc_ext = prob >= prob_threshold
    return {
        "hit":         obs_ext & fc_ext,
        "miss":        obs_ext & ~fc_ext,
        "false_alarm": ~obs_ext & fc_ext,
        "correct_neg": ~obs_ext & ~fc_ext,
    }


# ─── Per-date metrics ─────────────────────────────────────────────────────────

def compute_date_metrics(obs: np.ndarray, fc1: np.ndarray, fc2: np.ndarray,
                          T: np.ndarray, event_type: str,
                          lats: np.ndarray = None, lons: np.ndarray = None
                          ) -> dict:
    """Compute a comprehensive set of comparison metrics for one date×step slice.

    Parameters
    ----------
    obs, fc1, fc2 : 1-D arrays of length N (already filtered for valid T)
    T             : per-station threshold array of length N
    event_type    : 'above' or 'below'
    lats, lons    : optional station coordinates for region breakdown

    Returns
    -------
    dict with all metrics and a ``composite_score`` key.
    """
    N = len(obs)
    masks1 = classify_events(obs, fc1, T, event_type)
    masks2 = classify_events(obs, fc2, T, event_type)

    def _safe_mean(arr, mask):
        return float(np.mean(arr[mask])) if mask.sum() > 0 else np.nan

    def _sd(a, b):
        return float(a / b) if b > 0 else np.nan

    n_obs_ext = int(masks1["hit"].sum() + masks1["miss"].sum())
    n_hit1  = int(masks1["hit"].sum())
    n_miss1 = int(masks1["miss"].sum())
    n_fa1   = int(masks1["false_alarm"].sum())
    n_hit2  = int(masks2["hit"].sum())
    n_miss2 = int(masks2["miss"].sum())
    n_fa2   = int(masks2["false_alarm"].sum())

    # Severity: how far above/below the threshold are the FA / miss events?
    if event_type == "above":
        fa_exc1  = _safe_mean(fc1 - T, masks1["false_alarm"])
        fa_exc2  = _safe_mean(fc2 - T, masks2["false_alarm"])
        ms_exc1  = _safe_mean(obs - T, masks1["miss"])
        ms_exc2  = _safe_mean(obs - T, masks2["miss"])
    else:
        fa_exc1  = _safe_mean(T - fc1, masks1["false_alarm"])
        fa_exc2  = _safe_mean(T - fc2, masks2["false_alarm"])
        ms_exc1  = _safe_mean(T - obs, masks1["miss"])
        ms_exc2  = _safe_mean(T - obs, masks2["miss"])

    hit_err1 = _safe_mean(np.abs(fc1 - obs), masks1["hit"])
    hit_err2 = _safe_mean(np.abs(fc2 - obs), masks2["hit"])

    # twMAE (fraction of N)
    def _twmae(fc, masks, fc_vals, obs_vals, T_vals):
        hm = masks["hit"];  mm = masks["miss"];  fm = masks["false_alarm"]
        hit_c  = float(np.mean(np.abs(fc_vals[hm]  - obs_vals[hm])))  if hm.sum()  > 0 else 0.
        if event_type == "above":
            ms_c = float(np.mean(obs_vals[mm] - T_vals[mm])) if mm.sum() > 0 else 0.
            fa_c = float(np.mean(fc_vals[fm]  - T_vals[fm])) if fm.sum() > 0 else 0.
        else:
            ms_c = float(np.mean(T_vals[mm] - obs_vals[mm])) if mm.sum() > 0 else 0.
            fa_c = float(np.mean(T_vals[fm] - fc_vals[fm])) if fm.sum() > 0 else 0.
        return (hit_c * hm.sum() + ms_c * mm.sum() + fa_c * fm.sum()) / N

    twmae1 = _twmae(fc1, masks1, fc1, obs, T)
    twmae2 = _twmae(fc2, masks2, fc2, obs, T)

    pod1 = _sd(n_hit1, n_hit1 + n_miss1)
    pod2 = _sd(n_hit2, n_hit2 + n_miss2)
    far1 = _sd(n_fa1, n_hit1 + n_fa1)
    far2 = _sd(n_fa2, n_hit2 + n_fa2)

    # ETS for both models
    def _ets(nh, nm, nf, nn):
        hits_r = _sd((nh + nm) * (nh + nf), nh + nm + nf + nn)
        return _sd(nh - hits_r, nh + nm + nf - hits_r) if (nh + nm + nf - hits_r) else np.nan
    nn1 = int(masks1["correct_neg"].sum()); nn2 = int(masks2["correct_neg"].sum())
    ets1 = _ets(n_hit1, n_miss1, n_fa1, nn1)
    ets2 = _ets(n_hit2, n_miss2, n_fa2, nn2)

    # FA per-station exceedance distribution (useful for "storm severity")
    fa_max1 = float(np.max(fc1[masks1["false_alarm"]] - T[masks1["false_alarm"]])) \
              if masks1["false_alarm"].sum() > 0 else np.nan
    fa_max2 = float(np.max(fc2[masks2["false_alarm"]] - T[masks2["false_alarm"]])) \
              if masks2["false_alarm"].sum() > 0 else np.nan

    # Dominant region of FA/miss (where spatial coords provided)
    fa_region1 = _dominant_region(lats, lons, masks1["false_alarm"])
    fa_region2 = _dominant_region(lats, lons, masks2["false_alarm"])
    miss_region1 = _dominant_region(lats, lons, masks1["miss"])
    miss_region2 = _dominant_region(lats, lons, masks2["miss"])
    fa_conc1   = _spatial_concentration(lats, lons, masks1["false_alarm"])
    fa_conc2   = _spatial_concentration(lats, lons, masks2["false_alarm"])
    miss_conc1 = _spatial_concentration(lats, lons, masks1["miss"])
    miss_conc2 = _spatial_concentration(lats, lons, masks2["miss"])

    # ── Deltas (positive → m1 worse than m2) ──────────────────────────────────
    delta_twmae    = twmae1 - twmae2
    delta_fa_count = (n_fa1 - n_fa2) / max(N, 1)
    delta_fa_sev   = (fa_exc1 or 0) - (fa_exc2 or 0)
    delta_miss_cnt = (n_miss1 - n_miss2) / max(N, 1)
    delta_pod      = pod1 - pod2   # positive → m1 higher POD
    delta_far      = far1 - far2   # positive → m1 higher FAR (worse)

    # ── Case type classification ───────────────────────────────────────────────
    case_type = _classify_case(
        delta_fa_count, delta_fa_sev, delta_miss_cnt, delta_twmae,
        n_obs_ext, N
    )

    return {
        "n_stations": N,
        "n_obs_extreme": n_obs_ext,
        "n_hit1": n_hit1, "n_miss1": n_miss1, "n_fa1": n_fa1,
        "n_hit2": n_hit2, "n_miss2": n_miss2, "n_fa2": n_fa2,
        "pod1": pod1, "far1": far1, "ets1": ets1,
        "pod2": pod2, "far2": far2, "ets2": ets2,
        "fa_severity1": fa_exc1, "fa_severity2": fa_exc2,
        "fa_max1": fa_max1, "fa_max2": fa_max2,
        "miss_severity1": ms_exc1, "miss_severity2": ms_exc2,
        "hit_err1": hit_err1, "hit_err2": hit_err2,
        "twmae1": twmae1, "twmae2": twmae2,
        "delta_twmae": delta_twmae,
        "delta_fa_count": delta_fa_count,
        "delta_fa_severity": delta_fa_sev,
        "delta_miss_count": delta_miss_cnt,
        "delta_pod": delta_pod,
        "delta_far": delta_far,
        "fa_region1": fa_region1, "fa_region2": fa_region2,
        "miss_region1": miss_region1, "miss_region2": miss_region2,
        "fa_conc1": fa_conc1, "fa_conc2": fa_conc2,
        "miss_conc1": miss_conc1, "miss_conc2": miss_conc2,
        "case_type": case_type,
    }


def _dominant_region(lats, lons, mask):
    """Return the name of the Europe region with the most flagged stations."""
    if lats is None or lons is None or mask.sum() == 0:
        return "—"
    lat_m = lats[mask]; lon_m = lons[mask]
    counts = {}
    for name, (la0, la1, lo0, lo1) in EUROPE_REGIONS.items():
        counts[name] = int(((lat_m >= la0) & (lat_m <= la1) &
                            (lon_m >= lo0) & (lon_m <= lo1)).sum())
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else "—"


def _spatial_concentration(lats, lons, mask):
    """Return the fraction of flagged stations in their dominant region (0–1).

    High values (>0.6) indicate a geographically concentrated event;
    low values indicate scattered stations across Europe.
    """
    if lats is None or lons is None or mask.sum() == 0:
        return np.nan
    lat_m = lats[mask]; lon_m = lons[mask]
    counts = {}
    for name, (la0, la1, lo0, lo1) in EUROPE_REGIONS.items():
        counts[name] = int(((lat_m >= la0) & (lat_m <= la1) &
                            (lon_m >= lo0) & (lon_m <= lo1)).sum())
    total = sum(counts.values())
    return max(counts.values()) / total if total > 0 else np.nan


def _classify_case(delta_fa_count, delta_fa_sev, delta_miss_cnt, delta_twmae,
                   n_obs_ext, N):
    """Return a human-readable case type label.

    Positive deltas = model1 worse; negative deltas = model2 worse.
    """
    if N == 0:
        return "NO_DATA"
    if n_obs_ext == 0:
        # No observed extremes — any FA is spurious
        if delta_fa_count > 0.02:
            return "M1_FALSE_ALARM"
        if delta_fa_count < -0.02:
            return "M2_FALSE_ALARM"
        return "NO_EXTREMES"

    dom = max(
        ("FA_COUNT",   abs(delta_fa_count)),
        ("FA_SEV",     abs(delta_fa_sev) if not np.isnan(delta_fa_sev or 0) else 0),
        ("MISS_COUNT", abs(delta_miss_cnt)),
        ("TWMAE",      abs(delta_twmae)),
        key=lambda x: x[1],
    )
    label, _ = dom
    positive = {
        "FA_COUNT":   delta_fa_count > 0,
        "FA_SEV":     (delta_fa_sev or 0) > 0,
        "MISS_COUNT": delta_miss_cnt > 0,
        "TWMAE":      delta_twmae > 0,
    }[label]
    prefix = "M1" if positive else "M2"
    descriptions = {
        "FA_COUNT":   "FALSE_ALARM_COUNT",
        "FA_SEV":     "FALSE_ALARM_SEVERITY",
        "MISS_COUNT": "MISS_COUNT",
        "TWMAE":      "TWMAE",
    }
    return f"{prefix}_WORSE_{descriptions[label]}"


# ─── Composite ranking score ──────────────────────────────────────────────────

def add_composite_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Z-score normalised composite score and add it to df in-place.

    Composite score > 0  → Model 1 clearly worse than Model 2
    Composite score < 0  → Model 2 clearly worse than Model 1
    |score| close to 0   → Models perform similarly

    Weights are: 40% twMAE delta, 30% FA count delta, 30% FA severity delta.
    """
    result = df.copy()
    def _zscore(col):
        s = result[col].dropna()
        if s.std() < 1e-10:
            return pd.Series(0.0, index=result.index)
        return (result[col] - s.mean()) / s.std()

    z_twmae = _zscore("delta_twmae")
    z_fa    = _zscore("delta_fa_count")
    z_fasev = _zscore("delta_fa_severity").fillna(0)
    z_miss  = _zscore("delta_miss_count")

    result["composite_score"] = (
        0.35 * z_twmae +
        0.25 * z_fa    +
        0.25 * z_fasev +
        0.15 * z_miss
    ).round(4)

    result["abs_composite"] = result["composite_score"].abs()
    result["rank"] = result["abs_composite"].rank(ascending=False, method="first").astype(int)
    return result.sort_values("rank")
