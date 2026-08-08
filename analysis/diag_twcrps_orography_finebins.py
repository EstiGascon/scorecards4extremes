"""
Diagnostic: twCRPS vs orography (sdfor) at arbitrary bin resolution.

Purpose
-------
Test whether the "flat good / mid bad / mountainous good" U-shape seen in the
3-bin (low/mid/high) ensemble scorecards is a real signal or an artifact of the
coarse, internally-heterogeneous orography stratification.

It reuses the REAL pipeline pieces so the numbers match production:
  * per-station cold threshold = threshold._compute_local_obs_climatology_threshold
    (nearest lat/lon match to the 20-year q1 obs-climatology files, month from
    valid time) — NOT a sample percentile.
  * fair tail-weighted CRPS = identical formula to ens_scores._twcrps_per_case
    (below-threshold chaining v_T(x) = min(x, T)).
  * the same base QC as filter.run_step4 (coastal lsm cut, valid-temp range,
    member sentinel removal).

It does NOT bootstrap and does NOT plot — it is a fast point-estimate check.
Processes one forecast-day parquet at a time to stay within memory.

Usage
-----
    .venv/bin/python analysis/diag_twcrps_orography_finebins.py \
        configs/ensemble/config_2t_ens_local_p1obsclim_aifsvsifs_commonperiod.yaml
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import threshold as thr_mod  # reuse the real threshold logic


def fair_twcrps_below(members, obs, thr):
    """Fair tail-weighted CRPS per case for a 'below' event.

    Identical to ens_scores._per_case_score_diff / _twcrps_per_case ('below'):
        v_T(x) = min(x, T);  CRPS_fair(v_T(fc), v_T(obs)).
    members : (n, m) ensemble; obs, thr : (n,). Returns (n,).
    """
    m = members.shape[1]
    fv = np.minimum(members, thr[:, None])
    ov = np.minimum(obs, thr)
    term1 = np.abs(fv - ov[:, None]).mean(axis=1)
    s = np.sort(fv, axis=1)
    w = (2 * np.arange(m) - m + 1)[None, :]
    term2 = (s * w).sum(axis=1) / (m * (m - 1))
    return term1 - term2


def main(config_path):
    with open(config_path) as fh:
        config = yaml.safe_load(fh)

    rd = config['read_data']
    fc1_name = rd['forecast_model1']['name']
    fc2_name = rd['forecast_model2']['name']
    variable = config['variable']
    steps = set(config.get('steps') or [])

    ep = Path(config['extract_points']['output_path'])
    files = sorted(ep.glob(f"{variable}_{fc1_name}_vs_{fc2_name}_*day*.parquet"),
                   key=lambda p: int(p.name.split('day')[-1].split('.')[0]))
    if not files:
        raise SystemExit(f"No parquet files found in {ep}")

    fcfg = config.get('filter', {})
    lsm_cut = fcfg.get('coastal_lsm_threshold', 0.9)
    remove_coastal = fcfg.get('remove_coastal_stations', True)
    tmin = fcfg.get('min_valid_temperature', -60.0)
    tmax = fcfg.get('max_valid_temperature', 60.0)

    # Fine bin edges (a refinement of the production 40/120 cuts so the 3-bin
    # summary can be reconstructed by merging fine bins). Accumulate sums per bin
    # incrementally per row group so we never hold all rows in memory.
    edges = np.array([0, 20, 40, 60, 80, 120, 160, 220, 300, 3000.])
    nb = len(edges) - 1
    cnt = np.zeros(nb)
    s1 = np.zeros(nb)
    s2 = np.zeros(nb)
    sthr = np.zeros(nb)
    n_total = 0

    for f in files:
        pf = pq.ParquetFile(f)
        cols = pf.schema_arrow.names
        fc1c = [c for c in cols if c.startswith('fc1_member_')]
        fc2c = [c for c in cols if c.startswith('fc2_member_')]
        need = ['date', 'step', 'station_id', 'lat', 'lon', 'obs_value',
                'sdfor', 'lsm'] + fc1c + fc2c
        kept_file = 0
        for rg in range(pf.num_row_groups):
            t = pf.read_row_group(rg, columns=need)
            step = t.column('step').to_numpy()
            keep0 = np.isin(step, list(steps)) if steps else np.ones(len(step), bool)
            obs = t.column('obs_value').to_numpy()
            sd = t.column('sdfor').to_numpy()
            lsm = t.column('lsm').to_numpy()
            m1 = np.column_stack([t.column(c).to_numpy() for c in fc1c])
            m2 = np.column_stack([t.column(c).to_numpy() for c in fc2c])

            good = (keep0 & np.isfinite(obs) & (obs >= tmin) & (obs <= tmax) & np.isfinite(sd)
                    & (m1.min(1) >= tmin) & (m1.max(1) <= tmax)
                    & (m2.min(1) >= tmin) & (m2.max(1) <= tmax))
            if remove_coastal:
                good &= (lsm > lsm_cut)

            idxg = np.where(good)[0]
            if idxg.size == 0:
                continue
            # small DataFrame (only the columns the threshold matcher needs)
            sub = pd.DataFrame({
                'date': t.column('date').to_numpy()[idxg],
                'step': step[idxg],
                'station_id': t.column('station_id').to_numpy()[idxg],
                'lat': t.column('lat').to_numpy()[idxg],
                'lon': t.column('lon').to_numpy()[idxg],
            })
            thr = thr_mod._compute_local_obs_climatology_threshold(config, sub).to_numpy()
            matched = np.isfinite(thr)
            if not matched.any():
                continue
            sel = idxg[matched]
            thr = thr[matched]
            tw1 = fair_twcrps_below(m1[sel], obs[sel], thr)
            tw2 = fair_twcrps_below(m2[sel], obs[sel], thr)
            sds = sd[sel]

            bi = np.clip(np.digitize(sds, edges) - 1, 0, nb - 1)
            for k in range(nb):
                mk = bi == k
                if mk.any():
                    cnt[k] += mk.sum(); s1[k] += tw1[mk].sum(); s2[k] += tw2[mk].sum()
                    sthr[k] += thr[mk].sum()
            kept_file += len(sel); n_total += len(sel)
            del m1, m2
        print(f"  {f.name}: scored {kept_file:,} rows")

    print(f"\nTotal scored rows: {n_total:,}  (model1={fc1_name}, model2={fc2_name})")
    print("diff = twCRPS(model2) - twCRPS(model1);  negative = model2 better\n")

    a = np.divide(s1, cnt, out=np.full(nb, np.nan), where=cnt > 0)
    b = np.divide(s2, cnt, out=np.full(nb, np.nan), where=cnt > 0)

    def show(mask_edges, title):
        print(f"=== {title} ===")
        print(f"{'sdfor_bin':>13}{'n':>11}{'mean_thr':>10}{'twC_m1':>10}{'twC_m2':>10}{'diff':>11}{'better':>8}")
        for lo, hi in mask_edges:
            sel = [k for k in range(nb) if edges[k] >= lo and edges[k + 1] <= hi]
            c = cnt[sel].sum()
            if c == 0:
                continue
            am = s1[sel].sum() / c; bm = s2[sel].sum() / c; tm = sthr[sel].sum() / c
            lbl = f"[{lo:.0f},{hi:.0f})"
            print(f"{lbl:>13}{int(c):>11}{tm:>10.2f}{am:>10.4f}{bm:>10.4f}{bm-am:>+11.4f}"
                  f"{('m2' if bm < am else 'm1'):>8}")
        print()

    ranges = fcfg.get('orography_ranges', {'low': [0, 40], 'mid': [40, 120], 'high': [120, 3000]})
    lo0, lo1 = ranges['low']; mi1 = ranges['mid'][1]; hi1 = ranges['high'][1]
    show([(lo0, lo1), (lo1, mi1), (mi1, hi1)], "PRODUCTION 3-bin (low/mid/high) — validation")
    show([(edges[k], edges[k + 1]) for k in range(nb)], "FINER bins")


if __name__ == '__main__':
    cfg = sys.argv[1] if len(sys.argv) > 1 else \
        'configs/ensemble/config_2t_ens_local_p1obsclim_aifsvsifs_commonperiod.yaml'
    main(cfg)
