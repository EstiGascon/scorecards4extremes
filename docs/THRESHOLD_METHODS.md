# Threshold Methods

All methods are configured under `threshold:` in the YAML config.
All methods return either a **scalar** or a **per-row Series** (aligned to `data.index`).
The `event_type` key (`above` / `below`) controls the tail direction everywhere.

---

## 1. `fixed`

A single hard-coded value applied uniformly to all stations, lead times and dates.

```yaml
threshold:
  method: "fixed"
  fixed:
    value: 30.0       # e.g. 30 mm for heavy rain, -10°C for cold spell
    event_type: "above"
```

- **Returns:** scalar
- **Use when:** you have a physically meaningful universal threshold (e.g. WMO warning criteria)
- **Pitfall:** ignores climatological differences between stations and seasons

---

## 2. `dataset_climatology`

Percentile computed from the **observed values already in the extracted dataset**, pooled
within each season × orography stratum (i.e. the filtered `data` passed to Step 5).
When a `forecast_day` column exists, one threshold is computed per forecast day.

```yaml
threshold:
  method: "dataset_climatology"
  event_type: "above"
  dataset_climatology:
    percentile: 99
    use_filtered_data: true
```

- **Returns:** scalar (or per-forecast-day scalar broadcast to rows)
- **Use when:** you want a threshold defined purely by the score period and domain, with no
  external climatology file needed
- **Pitfall:** threshold changes if the dataset period or station list changes → not stable
  across experiments; does not account for seasonality or station geography

---

## 3. `station_climatology`  *(quaver backend only)*

Per-station percentile retrieved from **STVL** (the ECMWF observation climatology database,
1980–2009 period). Stations are matched by nearest lat/lon (tolerance 0.1°).

```yaml
threshold:
  method: "station_climatology"
  event_type: "above"
  station_climatology:
    percentile: 99
```

- **Returns:** per-row Series (each row gets the threshold of its station)
- **Use when:** you want an observationally-grounded, station-specific threshold based on a
  long climate reference period
- **Pitfall:** STVL uses the 1980–2009 climate period (fixed); not available for all
  parameters; falls back to `dataset_climatology` if STVL returns no data

---

## 4. `area_mean_climatology`  *(quaver backend only)*

Same as `station_climatology` but then takes the **mean across all matched stations**,
giving a single scalar threshold for the whole domain.

```yaml
threshold:
  method: "area_mean_climatology"
  event_type: "above"
  area_mean_climatology:
    percentile: 99
```

- **Returns:** scalar
- **Use when:** you want a single domain-representative threshold but still grounded in
  observed climatology rather than the evaluation dataset itself
- **Pitfall:** smooths out spatial gradients; warm/cold regions are averaged together

---

## 5. `model_percentile`

Per-station percentile computed from **one model's forecast distribution** over the
evaluation period. Each station gets a threshold derived from the model's own predicted
values at that location.

```yaml
threshold:
  method: "model_percentile"
  event_type: "above"
  model_percentile:
    percentile: 99
    model: "fc1"    # or "fc2"
```

- **Returns:** per-row Series (each row gets the threshold of its station × model)
- **Use when:** you want to score a model's ability to distinguish its own extreme events
  (self-referential; not recommended for model comparison)
- **Pitfall:** threshold depends on the model being scored → comparing fc1 vs fc2 is
  inconsistent unless both use the same model for the threshold

---

## 6. `local_obs_climatology`  *(recommended for extremes)*

Per-station, per-month percentile from **locally computed obs climatology files**
(GEO NCOLS format, produced by `obsclim.py`). One climatology file per calendar month.
Stations matched by nearest lat/lon (tolerance `max_match_dist`, default 0.1°).
The valid-date month (init date + step) is used, not the init-date month.

```yaml
threshold:
  method: "local_obs_climatology"
  event_type: "above"
  local_obs_climatology:
    path: "/path/to/scorecards4extremes/obs_clim_local"
    parameter: "2t"          # variable name used in filename
    percentile: 99
    window_days: 1
    n_years: 20
    first_year: 2005
    last_year: 2024
    min_availability_pct: 65
    max_match_dist: 0.1      # degrees lat/lon
```

**Filename pattern:** `clim_{param}_{window}_{MM}_{N}years_{Y1}_{Y2}_{pct}`
e.g. `clim_2t_1_10_20years_2005_2024_65` for October.

- **Returns:** per-row Series — each row gets `θ_s` = local p99 of its station for the
  valid month
- **How the score is computed:** `_compute_twcrps` is called on all rows in a
  season × orog × lead-time cell. Each row `i` uses `threshold[i] = θ_{s(i)}`. The score
  is the flat mean over all (station × date) pairs in the cell — **not** a mean of
  per-station means. A station with more dates weighs proportionally more.
- **Effective sample size per cell:** ~22k stations × 157 dates ÷ 4 seasons ÷ 3 orog
  types ≈ 290k rows → no sample size concern
- **Use when:** comparing models over a seasonal window where stations span different
  climatic regimes (e.g. Mediterranean vs Scandinavia in winter); each station's extreme
  is defined relative to its own local climate
- **Pitfall:** requires pre-computed clim files; stations not matched within 0.1° get
  `NaN` threshold and are excluded; match rate should be checked (see log output)

---

## Summary table

| Method | Threshold type | Varies by station? | Varies by month? | External data needed? |
|---|---|---|---|---|
| `fixed` | scalar | no | no | no |
| `dataset_climatology` | scalar (per day) | no | no | no |
| `station_climatology` | per-row | yes | no | STVL (1980–2009) |
| `area_mean_climatology` | scalar | no | no | STVL (1980–2009) |
| `model_percentile` | per-row | yes | no | no (uses forecast) |
| `local_obs_climatology` | per-row | yes | yes | local clim files |

> **CAMS atmospheric composition variables** (`aod500`, `pm2p5`, `pm10`, `go3`, `no2`,
> `so2`, `co`, `no`) have no per-station climatology files or STVL access — use
> `fixed`, `dataset_climatology`, or `model_percentile` only. `station_climatology`
> and `local_obs_climatology` are not applicable to these variables.
