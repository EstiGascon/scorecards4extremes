# CAMS ver0d thresholds per species and network

Fixed event thresholds (`threshold.method: fixed`, `event_type: above`) used by the
`configs/cams/config_<species>_ver0d_<network>_fixed<value>.yaml` configs, which
read the raw ver0d NetCDF archive through `src/cams_extract.py`.

Status: working values as of 2026-09-25. Thresholds were chosen from short test
runs (10–16 Jan and 10–16 Jul 2025) and should be revisited once full-year runs
are available.

## Threshold matrix

The threshold is always in the **obs units of that network** — the extractor
converts the model into those units (see [Units](#units)).

| Species | Airbase (Europe) | AirNow (North America) | China | AERONET |
|---|---|---|---|---|
| `go3`   | 120 µg/m³ | 70 ppb | 160 µg/m³ | – |
| `no2`   | 40 µg/m³ | 21 ppb | 80 µg/m³ | – |
| `so2`   | 10 µg/m³ | 4 ppb | 20 µg/m³ | – |
| `co`    | 1000 µg/m³ | 0.87 ppm | 1500 µg/m³ | – |
| `pm2p5` | 25 µg/m³ | 25 µg/m³ | 75 µg/m³ | – |
| `pm10`  | 45 µg/m³ | 45 µg/m³ | 150 µg/m³ | – |
| `aod500`| – | – | – | 0.5 (unitless) |

## Basis and observed event frequency

"Obs events" = share of 3-hourly station values above the threshold in the
Jan / Jul 2025 test weeks. Aim was roughly 5% in winter where no usable
regulatory value exists.

| Species | Network | Threshold | Basis | Obs events Jan / Jul |
|---|---|---|---|---|
| go3 | Airbase | 120 µg/m³ | EU 8h target value | – / 3.6% |
| go3 | AirNow | 70 ppb | US NAAQS 8h standard (≈137 µg/m³) | – / 0.9% |
| go3 | China | 160 µg/m³ | CN GB 3095-2012 Grade II 8h limit | – / 5.0% |
| no2 | Airbase | 40 µg/m³ | EU annual limit value | 6.3% / 0.2% |
| no2 | AirNow | 21 ppb | = 40 µg/m³ at 25 °C (matches Airbase) | 7.2% / 0.5% |
| no2 | China | 80 µg/m³ | CN Grade II 24h limit | 5.2% / 0.05% |
| so2 | Airbase | 10 µg/m³ | data-driven (regulatory limits never reached) | 7.1% / 5.7% |
| so2 | AirNow | 4 ppb | ≈ 10.5 µg/m³ at 25 °C (matches Airbase) | 4.9% / 1.6% |
| so2 | China | 20 µg/m³ | data-driven | 6.2% / 0.9% |
| co | Airbase | 1000 µg/m³ | data-driven (regulatory limits never reached) | 4.9% / 1.8% |
| co | AirNow | 0.87 ppm | = 1000 µg/m³ at 25 °C (matches Airbase) | 5.0% / 0.4% |
| co | China | 1500 µg/m³ | data-driven (CN 4 mg/m³ gives ~0.1%) | 4.9% / 0.4% |
| pm2p5 | Airbase | 25 µg/m³ | EU 2030 24h limit | not tested |
| pm2p5 | AirNow | 25 µg/m³ | EU 2030 24h limit (same as Airbase) | not tested |
| pm2p5 | China | 75 µg/m³ | CN Grade II 24h limit (25 µg/m³ gave 74% winter events) | 15.2% / 0.05% |
| pm10 | Airbase | 45 µg/m³ | EU 2030 24h limit (= WHO 2021 24h guideline) | 5.6% / 0.4% |
| pm10 | AirNow | 45 µg/m³ | same as Airbase (US 150 µg/m³ gives <0.5%) | 7.0% / 8.4% |
| pm10 | China | 150 µg/m³ | CN Grade II 24h limit | 5.9% / 0.2% |
| aod500 | AERONET | 0.5 | – | not tested |

Why not the regulatory values everywhere: most limits are 8h/24h/annual means,
but the pipeline scores native 3-hourly station values, and several limits are
never reached at all in 2025 (hourly NO2 200 µg/m³ / 100 ppb, all SO2 limits,
CO 4–10 mg/m³ / 9 ppm). Where that happened a lower, recognised value or a
data-driven value was used instead, and the AirNow value was chosen as the ppb/ppm
equivalent of the Airbase one so the two networks are comparable.

ppb/ppm ↔ µg/m³ conversions use 25 °C, 1013 hPa (molar volume 24.45 L):
µg/m³ = ppb × M / 24.45, with M = 48.00 (O3), 46.01 (NO2), 64.07 (SO2), 28.01 (CO).

## Units

Obs units, checked on the 1st/15th of each month in 2025:

| Network | Gases (O3, NO2, SO2) | CO | PM2.5 / PM10 | Model conversion (`conv`) |
|---|---|---|---|---|
| Airbase | µg/m³ | µg/m³ (see below) | µg/m³ | gases `density`, PM `conc` |
| AirNow | ppb | ppm | µg/m³ | gases `vmr`, PM `conc` |
| China | µg/m³ | µg/m³ | µg/m³ | gases `density`, PM `conc` |

Model fields: gases in kg/kg (`*_ml137`), PM in kg/m³. Conversions are defined per
network in `NETWORKS` in `src/cams_extract.py`.

## Known data issues

- **Airbase CO is stored ~1000× too large.** The files label it `ug/m3`, but the
  values (median ~2e5–4e5 at every station, 2024–2025) are in ng/m³. The registry
  entry sets `obs_scale=1e-3` for Airbase `co`, so the pipeline rescales on read and
  prints `Obs scale : x0.001` in step 3. ver0d itself would currently reject all of
  this data (its sane range for CO is 0–2000 µg/m³) — worth reporting upstream.
- **China CO** values are all multiples of 100 µg/m³, which suggests they come from
  mg/m³ reported to 0.1. They are correct in magnitude.

## Caveats for interpreting scores

- **Summer:** for NO2, SO2, CO and PM10 (Airbase, China) summer events are rare
  (<1%), and for CO the model often never exceeds the threshold, giving POD = 0 and
  undefined FAR. Consider dropping JJA for these species or reading those panels as
  "no model events".
- **CO:** model underestimates in all networks; correlation ~0.1–0.2.
- **SO2:** almost no skill on AirNow and China (correlation ≈ 0). Over China the
  model overestimates about threefold (Jan median 27 vs 8 µg/m³ obs), so the high
  POD there is not skill — ETS ≈ 0.
- **NO2:** the most informative gas (correlation 0.5–0.6, small bias).
- **PM2.5 China:** 75 µg/m³ still gives ~15% obs events in winter (Jan median
  40 µg/m³), more than the ~5% aimed for elsewhere; ~100 µg/m³ would give ~5%.
  25 µg/m³ (as for Airbase/AirNow) was tried first and rejected: 74% of winter
  values exceed it. In summer the model overestimates (median 24 vs 15 µg/m³ obs)
  and events at 75 are almost all false alarms (FAR ≈ 1).
- **PM10:** AirNow is the only case with more events in summer than winter
  (probably wildfire smoke and dust); China summer has spurious model peaks.

## Changing a threshold

Edit `threshold.fixed.value` in the config and keep the file name and the
`save.output_directory` suffix (`..._fixed<value>`) in step with it, so results
from different thresholds don't overwrite each other. Update this table too.
