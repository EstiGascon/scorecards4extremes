"""
One-off script to create SDFOR masks at O2560 resolution for tp24 DestinE jobs.

Input:  sdfor_tco2559.grib  (N=1280, 8.5M points)
Output: sdfor_o2560_flat/hilly/complex.grib  (O2560, 26.3M points)

Run via: sbatch submit_create_sdfor_masks.sh
"""
import metview as mv
import os

SDFOR_IN  = '/ec/vol/destine/continuous_evaluation/sdfor_tco2559.grib'
OROG_REF  = '/ec/vol/destine/continuous_evaluation/orog_tco2559.grib'   # O2560, 26.3M
OUT_DIR   = '/ec/vol/destine/continuous_evaluation'

RANGES = {
    'flat':    (0,   40),
    'hilly':   (40,  120),
    'complex': (120, 3000),
}

print("Loading SDFOR (N1280, 8.5M pts)...")
sdfor = mv.read(SDFOR_IN)
print(f"  SDFOR npts: {mv.grib_get_long(sdfor[0], 'numberOfDataPoints')}")

print("Loading orography template (O2560, 26.3M pts)...")
orog = mv.read(OROG_REF)
print(f"  OROG  npts: {mv.grib_get_long(orog[0], 'numberOfDataPoints')}")

print("Regridding SDFOR N1280 → O2560 via MIR...")
sdfor_o2560 = mv.regrid(
    data=sdfor,
    grid_definition_mode='template',
    template_data=orog[0],
)
print(f"  Regridded npts: {mv.grib_get_long(sdfor_o2560[0], 'numberOfDataPoints')}")

for name, (lo, hi) in RANGES.items():
    out_path = os.path.join(OUT_DIR, f'sdfor_o2560_{name}.grib')
    print(f"Creating {out_path}  ({lo}–{hi})...")
    mask = (sdfor_o2560 >= lo) * (sdfor_o2560 < hi)
    mv.write(out_path, mask)
    npts = mv.grib_get_long(mask[0], 'numberOfDataPoints')
    print(f"  Written ({npts} pts)")

print("\nDone. O2560 SDFOR masks created.")
