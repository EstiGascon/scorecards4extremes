"""
Direct data retrieval for scorecards4extremes
=============================================

Materialise the local files the extractors already read, so downstream code is
unchanged — retrieval is just a pre-step that populates the folders Step 3 reads.

Two entry points:
  - retrieve_forecast(...)  → forecast GRIB via the ``mars`` CLI
  - retrieve_obs(...)       → observation geopoints (.geo) via STVL (``vtb``)

Design notes
------------
* File naming matches what extract_points.py / extract_points_ensemble.py expect:
    forecast: ``{param}_{YYYYMMDD}.grib``  (param: 2t | 10u | 10v | tp)
    obs:      ``{variable}_obs_{YYYYMMDD}{HH}.geo``
* Storage folder is DERIVED from the MARS identity keys (class/stream/expver), so
  the folder can never disagree with the expver actually retrieved — this is the
  whole point of the feature.
* Never writes under $HOME (small quota); target must be on /ec/vol/... scratch/perm.
* mars CLI and the ECMWF-internal ``vtb`` package only exist on ECMWF compute nodes.
"""

import os
import json
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime, timedelta


# ---------------------------------------------------------------------------
# Parameter mapping: variable -> list of MARS/GRIB short names to retrieve
# ---------------------------------------------------------------------------
# 10ff (wind speed) is retrieved as its two components and combined at extraction.
_VARIABLE_TO_PARAMS = {
    '2t':   ['2t'],
    '10ff': ['10u', '10v'],
    'tp24': ['tp'],
    'tp':   ['tp'],
}


# IFS Cycle 50r1 (12 May 2026) discontinued the separate ENS control in the
# ``enfo`` stream: for the OPERATIONAL IFS (class=od), the ensemble control from
# this date is archived as class=od, stream=oper, type=fc (NOT stream=enfo,
# type=cf). Perturbed members stay stream=enfo, type=pf. AIFS (class=ai) and
# pre-upgrade od still accept enfo/cf. Verified against MARS on 2026-07-27.
# https://confluence.ecmwf.int/display/FCST/Implementation+of+IFS+Cycle+50r1
_IFS_50R1_DATE = datetime(2026, 5, 12)


def _params_for_variable(variable):
    if variable not in _VARIABLE_TO_PARAMS:
        raise ValueError(
            f"mars_retrieve: unsupported variable '{variable}'. "
            f"Supported: {sorted(_VARIABLE_TO_PARAMS)}")
    return _VARIABLE_TO_PARAMS[variable]


def _check_not_home(path):
    """Refuse to write large data under $HOME (strict per-user quota)."""
    resolved = str(Path(path).expanduser().resolve())
    home = str(Path(os.environ.get('HOME', '/home')).resolve())
    forbidden = (resolved == home or resolved.startswith(home + '/')
                 or resolved.startswith('/home/')
                 or '/dh1_home' in resolved)
    if forbidden:
        raise ValueError(
            f"mars_retrieve: refusing to store data under $HOME ('{resolved}'). "
            f"Use /ec/vol/... , $SCRATCH or $HPCPERM instead — $HOME has a small quota.")
    return resolved


def _norm_time(time_val):
    """Normalise a MARS time key to 'HH:00:00' (accepts '0', '00', 0, '00:00:00')."""
    s = str(time_val).strip()
    if ':' in s:
        return s
    return f"{int(s):02d}:00:00"


def _derive_subdir(mars_cfg):
    """
    Folder name derived from the MARS identity keys, so it always reflects what
    was retrieved. e.g. class=rd expver=j5vo -> 'rd_oper_j5vo';
    class=od stream=enfo expver=1 -> 'od_enfo_0001'.
    """
    cls = str(mars_cfg['class']).strip()
    stream = str(mars_cfg.get('stream', 'oper')).strip()
    expver = str(mars_cfg['expver']).strip()
    return f"{cls}_{stream}_{expver}"


def _resolve_target_dir(mars_cfg):
    """base_path + derived identity subdir; created if missing; never under $HOME."""
    base_path = mars_cfg.get('base_path')
    if not base_path:
        raise ValueError("mars_retrieve: 'base_path' is required in the mars config block")
    target_dir = Path(_check_not_home(base_path)) / _derive_subdir(mars_cfg)
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


def _derive_obs_subdir(stvl_cfg):
    """
    Folder name derived from the STVL identity keys (sources list), so a fresh
    STVL retrieval can NEVER collide with a pre-staged ``local_gpt`` directory
    that happens to share the same ``base_path`` (e.g. both configs pointing at
    the same '.../obs/raw' folder for a given variable). Without this, retrieve_obs
    used ``base_path`` directly with no subdir, so if that folder already contained
    pre-staged .geo files (put there for a local_gpt run), the skip-if-exists check
    would find every target file already present and silently skip ALL retrieval
    100% of the time — meaning 'method 2' (mars+stvl) ends up silently reading the
    exact same pre-staged obs as 'method 1' (local) instead of ever calling STVL.
    Confirmed 2026-08-04: this caused method1/method2 ensemble results for 10ff to
    be byte-for-byte identical (md5-identical extracted parquet files).
    """
    sources = stvl_cfg.get('sources', ['synop'])
    return "stvl_" + "_".join(sorted(str(s) for s in sources))


def _date_range(start_date, end_date):
    """Inclusive daily range; accepts 'YYYY-MM-DD' or 'YYYYMMDD'."""
    fmt_in = '%Y-%m-%d' if '-' in str(start_date) else '%Y%m%d'
    cur = datetime.strptime(str(start_date), fmt_in)
    end = datetime.strptime(str(end_date), fmt_in)
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


# ===========================================================================
# Forecast retrieval (MARS CLI)
# ===========================================================================
def _ens_control_stream_type(mars_cfg, date_str):
    """
    (stream, type) for the ENS control member on a given day.

    Operational IFS (class=od) from IFS Cycle 50r1 (12 May 2026) onwards: the
    control lives in stream=oper, type=fc. Everything else (pre-upgrade od, AIFS)
    keeps the classic stream=enfo, type=cf.
    """
    cls = str(mars_cfg['class']).strip()
    day = datetime.strptime(date_str, '%Y%m%d')
    if cls == 'od' and day >= _IFS_50R1_DATE:
        return 'oper', 'fc'
    return 'enfo', 'cf'


def _mars_block(cls, mars_type, stream, expver, date_str, time_str, step_str,
                 levtype, param, target, model=None, number=None,
                 database=None, grid=None, include_number=False):
    """Build one MARS ``retrieve,`` stanza as request text."""
    lines = [
        "retrieve,",
        f"  class    = {cls},",
        f"  type     = {mars_type},",
    ]
    # stream is optional: research (rd/fdb) data is archived without it, and
    # MARS defaults to oper for od/ai deterministic. Only emit if given.
    if stream:
        lines.append(f"  stream   = {stream},")
    lines += [
        f"  expver   = {expver},",
        f"  date     = {date_str},",
        f"  time     = {time_str},",
        f"  step     = {step_str},",
        f"  levtype  = {levtype},",
        f"  param    = {param},",
    ]
    # model is required for AIFS (class=ai): model=aifs-ens / aifs-single.
    if model:
        lines.append(f"  model    = {model},")
    if include_number:
        lines.append(f"  number   = {number},")
    if database:
        lines.append(f"  database = {database},")
    if grid:
        lines.append(f"  grid     = {grid},")
    lines.append(f'  target   = "{target}"')
    return "\n".join(lines) + "\n"


def _build_mars_request(mars_cfg, param, date_str, steps, mode, target):
    """Return MARS request text for one deterministic param/day (single block).

    Ensemble mode is handled separately by _build_ensemble_mars_requests —
    see that function's docstring for why control/perturbed are NOT combined
    into one multi-block request here.
    """
    step_str = '/'.join(str(s) for s in steps)
    levtype = mars_cfg.get('levtype', 'sfc')
    time_str = _norm_time(mars_cfg.get('time', '00'))
    cls = mars_cfg['class']
    expver = mars_cfg['expver']
    database = mars_cfg.get('database')       # e.g. 'fdb' for research (rd) data
    grid = mars_cfg.get('grid')               # optional regrid, e.g. '0.25/0.25'
    model = mars_cfg.get('model')             # AIFS requires model=aifs-ens (class=ai)
    # stream optional (omitted -> MARS default oper for od/ai; matches the
    # research scripts which never set stream for rd/fdb data).
    stream = mars_cfg.get('stream')
    mars_type = mars_cfg.get('type', 'fc')
    return _mars_block(cls, mars_type, stream, expver, date_str, time_str,
                        step_str, levtype, param, target,
                        model=model, database=database, grid=grid)


def _build_ensemble_mars_requests(mars_cfg, param, date_str, steps,
                                   ctrl_target, pert_target):
    """Return (control_request, perturbed_request) text, each targeting its
    OWN file — kept as two separate MARS requests rather than one combined
    multi-block request.

    Some research (rd) experiments only archive perturbed members (no
    control), and MARS fails an ENTIRE multi-block request the moment any
    one block returns 0 fields — which would otherwise discard perfectly
    good perturbed-member data just because the control member is missing.

    `mars_cfg['control_source']` (optional) controls where the control
    member comes from:
      - unset / {'mode': 'own'} (default): control uses the SAME class/expver
        as the perturbed members (era-aware stream/type via
        _ens_control_stream_type).
      - {'mode': 'omit'}: no control is retrieved at all — control_request
        is returned as None. Use this when an experiment has no control and
        you'd rather compare on perturbed members only than substitute one.
      - {'mode': 'external', 'class':..., 'expver':..., 'type':..., ...}: the
        control is retrieved from a DIFFERENT MARS identity (e.g. a
        deterministic sibling experiment j1l8 standing in for j1l9's missing
        control). Any key not given falls back to the perturbed members'
        own mars_cfg value (levtype/time/database/grid/model), except
        class/expver which MUST be given, and type which defaults to 'fc'.
    """
    step_str = '/'.join(str(s) for s in steps)
    levtype = mars_cfg.get('levtype', 'sfc')
    time_str = _norm_time(mars_cfg.get('time', '00'))
    cls = mars_cfg['class']
    expver = mars_cfg['expver']
    database = mars_cfg.get('database')
    grid = mars_cfg.get('grid')
    model = mars_cfg.get('model')
    number = mars_cfg.get('number', '1/to/50')

    control_source = mars_cfg.get('control_source') or {}
    ctrl_mode = control_source.get('mode', 'own')

    control = None
    if ctrl_mode == 'omit':
        control = None
    elif ctrl_mode == 'external':
        ext_cls = control_source['class']
        ext_expver = control_source['expver']
        ext_type = control_source.get('type', 'fc')
        ext_stream = control_source.get('stream')
        ext_database = control_source.get('database', database)
        ext_grid = control_source.get('grid', grid)
        ext_model = control_source.get('model', model)
        control = _mars_block(ext_cls, ext_type, ext_stream, ext_expver, date_str,
                               time_str, step_str, levtype, param, ctrl_target,
                               model=ext_model, database=ext_database, grid=ext_grid)
    else:
        ctrl_stream, ctrl_type = _ens_control_stream_type(mars_cfg, date_str)
        control = _mars_block(cls, ctrl_type, ctrl_stream, expver, date_str, time_str,
                               step_str, levtype, param, ctrl_target,
                               model=model, database=database, grid=grid)

    perturbed = _mars_block(cls, 'pf', 'enfo', expver, date_str, time_str,
                             step_str, levtype, param, pert_target,
                             model=model, number=number, database=database,
                             grid=grid, include_number=True)
    return control, perturbed


def _run_mars(request_text):
    """Write the request to a temp file and invoke the mars CLI. Returns exit code."""
    if shutil.which('mars') is None:
        raise EnvironmentError(
            "mars_retrieve: the 'mars' CLI is not available (ECMWF compute nodes only). "
            "Use source: local_grib with pre-retrieved files instead.")
    with tempfile.NamedTemporaryFile('w', suffix='.req', delete=False,
                                     dir=os.environ.get('TMPDIR', '/tmp')) as fh:
        fh.write(request_text)
        reqfile = fh.name
    try:
        proc = subprocess.run(['mars', reqfile], capture_output=True, text=True)
        for line in proc.stdout.splitlines():
            if any(k in line for k in ('retrieved', 'WARN', 'ERROR', 'No errors')):
                print(f"      {line.strip()}")
        if proc.returncode != 0 and proc.stderr:
            print(f"      [mars stderr] {proc.stderr.strip()[:500]}")
        return proc.returncode
    finally:
        try:
            os.remove(reqfile)
        except OSError:
            pass


def _retrieve_ensemble_param(mars_cfg, param, date_str, steps, target):
    """Retrieve one ensemble param/day as control + perturbed MARS requests,
    tolerating a missing control member (some research experiments only
    archive perturbed members). Concatenates whatever succeeded into
    `target` — GRIB messages are self-delimited, so plain concatenation of
    the two files is a valid multi-message GRIB file.

    Returns (ok, exit_code).
    """
    ctrl_tmp = target.with_name(target.name + '.ctrl.tmp')
    pert_tmp = target.with_name(target.name + '.pert.tmp')
    control_req, perturbed_req = _build_ensemble_mars_requests(
        mars_cfg, param, date_str, steps, str(ctrl_tmp), str(pert_tmp))

    have_ctrl = False
    if control_req is None:
        # control_source.mode == 'omit' — deliberately not requested.
        pass
    else:
        ctrl_rc = _run_mars(control_req)
        have_ctrl = ctrl_rc == 0 and ctrl_tmp.exists() and ctrl_tmp.stat().st_size > 0
        if not have_ctrl:
            print(f"      [WARN] control member unavailable for {param} {date_str} — using perturbed members only")
            if ctrl_tmp.exists():
                ctrl_tmp.unlink()

    pert_rc = _run_mars(perturbed_req)
    have_pert = pert_rc == 0 and pert_tmp.exists() and pert_tmp.stat().st_size > 0
    if not have_pert:
        if pert_tmp.exists():
            pert_tmp.unlink()
        if ctrl_tmp.exists():
            ctrl_tmp.unlink()
        return False, pert_rc

    with open(target, 'wb') as out:
        if have_ctrl:
            with open(ctrl_tmp, 'rb') as f:
                shutil.copyfileobj(f, out)
            ctrl_tmp.unlink()
        with open(pert_tmp, 'rb') as f:
            shutil.copyfileobj(f, out)
    pert_tmp.unlink()
    return True, 0


def retrieve_forecast(mars_cfg, variable, start_date, end_date, steps, mode='deterministic'):
    """
    Retrieve forecast GRIB for one model into a folder derived from the MARS keys.

    Writes one file per param per day named ``{param}_{YYYYMMDD}.grib`` — exactly
    what extract_points.py / extract_points_ensemble.py look for — and returns the
    resolved directory so the caller can treat the source as 'local_grib' downstream.

    Skip-if-exists (idempotent); partial files are deleted on failure.
    """
    params = _params_for_variable(variable)
    target_dir = _resolve_target_dir(mars_cfg)
    force = bool(mars_cfg.get('force', False))

    print(f"  [mars_retrieve] variable={variable} params={params} mode={mode}")
    print(f"  [mars_retrieve] target: {target_dir}")
    print(f"  [mars_retrieve] steps:  {steps}")

    n_ok = n_skip = n_err = 0
    for dt in _date_range(start_date, end_date):
        date_str = dt.strftime('%Y%m%d')
        for param in params:
            target = target_dir / f"{param}_{date_str}.grib"
            if target.exists() and not force:
                n_skip += 1
                continue
            print(f"    [{date_str}] retrieving {param} ...")
            if mode == 'ensemble':
                ok, rc = _retrieve_ensemble_param(mars_cfg, param, date_str, steps, target)
            else:
                req = _build_mars_request(mars_cfg, param, date_str, steps, mode, str(target))
                rc = _run_mars(req)
                ok = rc == 0 and target.exists()
            if ok:
                n_ok += 1
            else:
                n_err += 1
                if target.exists():
                    target.unlink()  # remove incomplete file
                print(f"    [{date_str}] {param} FAILED (exit {rc})")

    print(f"  [mars_retrieve] done — OK: {n_ok}  Skipped: {n_skip}  Failed: {n_err}")
    if n_ok == 0 and n_skip == 0:
        raise RuntimeError(
            f"mars_retrieve: no forecast files retrieved for {variable} into {target_dir}. "
            f"Check the MARS keys ({mars_cfg.get('class')}/{mars_cfg.get('stream')}/"
            f"expver={mars_cfg.get('expver')}) and date range.")
    return target_dir


# ===========================================================================
# Observation retrieval (STVL via vtb) — materialises per-cycle .geo geopoints
# ===========================================================================
def _import_vtb_mv():
    try:
        import vtb  # ECMWF-internal
        import metview as mv
    except ImportError as e:
        raise EnvironmentError(
            "mars_retrieve: observation retrieval needs the ECMWF-internal 'vtb' package "
            "(and metview), available on ECMWF compute nodes only. "
            f"Use observation_source: local_gpt with pre-retrieved .geo files instead. ({e})")
    return vtb, mv


def _valid_datetimes(start_date, end_date, steps, base_time):
    """Unique forecast valid datetimes = base_date@base_time + step, over the range."""
    base_hour = int(str(base_time).split(':')[0]) if ':' in str(base_time) else int(base_time)
    seen = set()
    out = []
    for dt in _date_range(start_date, end_date):
        base = dt.replace(hour=base_hour)
        for s in steps:
            vdt = base + timedelta(hours=int(s))
            if vdt not in seen:
                seen.add(vdt)
                out.append(vdt)
    return sorted(out)


def _vtb_available():
    """True if vtb is importable in the current interpreter."""
    try:
        import vtb  # noqa: F401
        return True
    except ImportError:
        return False


def _vtb_worker_cmd(arg_file):
    """
    Command (argv list) that runs this module's STVL worker under a vtb-capable
    Python. `vtb` is a compiled extension tied to a specific Python (the ECMWF
    `python3` module, currently 3.13) and is not importable from the project's
    3.12 .venv. Rather than force a single interpreter on the whole pipeline, we
    run just the STVL step in a subprocess under the right Python — the same way
    scripts/submit_extraction.sh already runs vtb code.

    Resolution order:
      1. $S4E_VTB_PYTHON  — explicit python executable (most robust/portable).
      2. `module load python3` in a login shell (standard ECMWF path).
    """
    module = os.path.abspath(__file__)
    override = os.environ.get('S4E_VTB_PYTHON')
    if override:
        return [override, module, '--stvl-worker', arg_file]
    inner = (f"module load python3 >/dev/null 2>&1 && "
             f"exec python3 {shlex.quote(module)} --stvl-worker {shlex.quote(arg_file)}")
    return ['bash', '-lc', inner]


def retrieve_obs(stvl_cfg, variable, start_date, end_date, steps, base_time='00'):
    """
    Retrieve observations from STVL and write one Metview geopoints file per valid
    cycle, named ``{variable}_obs_{YYYYMMDD}{HH}.geo`` into a folder that never sits
    under $HOME. Returns the resolved directory (treated as local_gpt downstream).

    Interpreter-agnostic: if `vtb` is importable here (e.g. the pipeline is running
    under the ECMWF python3 module) the retrieval runs in-process; otherwise it is
    run once in a subprocess under a vtb-capable Python (see _vtb_worker_cmd). This
    lets the main pipeline keep running in its .venv while the STVL step uses the
    interpreter that actually ships vtb.
    """
    if _vtb_available():
        return _retrieve_obs_inproc(stvl_cfg, variable, start_date, end_date, steps, base_time)

    # Subprocess fallback — run the whole retrieval once under a vtb-capable Python.
    base_path = stvl_cfg.get('base_path')
    if not base_path:
        raise ValueError("mars_retrieve: 'base_path' is required in the stvl config block")
    target_dir = Path(_check_not_home(base_path)) / _derive_obs_subdir(stvl_cfg)
    target_dir.mkdir(parents=True, exist_ok=True)

    payload = dict(stvl_cfg=stvl_cfg, variable=variable, start_date=start_date,
                   end_date=end_date, steps=list(steps), base_time=base_time)
    with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False,
                                     dir=os.environ.get('TMPDIR', '/tmp')) as fh:
        json.dump(payload, fh)
        arg_file = fh.name
    cmd = _vtb_worker_cmd(arg_file)
    print(f"  [stvl_retrieve] vtb not in current interpreter — running STVL step via subprocess:")
    print(f"                  {' '.join(cmd)}")
    try:
        proc = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    finally:
        try:
            os.remove(arg_file)
        except OSError:
            pass
    if proc.returncode != 0:
        raise RuntimeError(
            "mars_retrieve: STVL observation subprocess failed. Ensure the ECMWF "
            "'python3' module provides vtb, or set $S4E_VTB_PYTHON to a vtb-capable "
            "python. To skip retrieval, use observation_source: local_gpt with "
            "pre-retrieved .geo files.")
    return target_dir


def _retrieve_obs_inproc(stvl_cfg, variable, start_date, end_date, steps, base_time='00'):
    """Actual STVL retrieval loop. Must run under a vtb-capable interpreter."""
    import pandas as pd

    vtb, mv = _import_vtb_mv()

    base_path = stvl_cfg.get('base_path')
    if not base_path:
        raise ValueError("mars_retrieve: 'base_path' is required in the stvl config block")
    target_dir = Path(_check_not_home(base_path)) / _derive_obs_subdir(stvl_cfg)
    target_dir.mkdir(parents=True, exist_ok=True)

    sources = stvl_cfg.get('sources', ['synop'])
    force = bool(stvl_cfg.get('force', False))
    # STVL uses 'tp' with a 24h period for tp24 accumulation obs
    stvl_param = 'tp' if variable == 'tp24' else variable
    stvl_period = pd.to_timedelta('24h') if variable == 'tp24' else None

    vdts = _valid_datetimes(start_date, end_date, steps, base_time)
    print(f"  [stvl_retrieve] variable={variable} param={stvl_param} sources={sources}")
    print(f"  [stvl_retrieve] target: {target_dir}")
    print(f"  [stvl_retrieve] {len(vdts)} valid cycles to obtain")

    n_ok = n_skip = n_err = 0
    for vdt in vdts:
        vdate = vdt.strftime('%Y%m%d')
        vhh = vdt.strftime('%H')
        target = target_dir / f"{variable}_obs_{vdate}{vhh}.geo"
        if target.exists() and not force:
            n_skip += 1
            continue
        try:
            obs_kw = dict(table="observation", parameter=stvl_param,
                          date=[pd.to_datetime(vdt)], sources=sources)
            if stvl_period is not None:
                obs_kw['period'] = stvl_period
            obs_fs = vtb.media.stvl_retrieve(**obs_kw)
            if obs_fs is None or len(obs_fs) == 0:
                n_err += 1
                print(f"    [{vdate}{vhh}] no obs returned")
                continue
            # stvl_retrieve returns a vtb Fieldset, which mv.write cannot serialise
            # directly. Convert to a Metview geopointset and take the single field
            # (one valid datetime per call) -> plain Geopoints, matching the existing
            # obs files exactly (columns: stnid, latitude, longitude, level, date,
            # elevation, value_0), so extract_points.py reads it as before.
            gp = obs_fs.to_metview_geopointset()[0]
            mv.write(str(target), gp)
            n_ok += 1
        except Exception as e:  # noqa: BLE001 - report and continue per cycle
            n_err += 1
            if target.exists():
                target.unlink()
            print(f"    [{vdate}{vhh}] FAILED: {e}")

    print(f"  [stvl_retrieve] done — OK: {n_ok}  Skipped: {n_skip}  Failed: {n_err}")
    if n_ok == 0 and n_skip == 0:
        raise RuntimeError(
            f"mars_retrieve: no observations retrieved for {variable} into {target_dir}.")
    return target_dir


if __name__ == '__main__':
    # Subprocess entry point for the STVL worker (invoked by retrieve_obs when the
    # calling interpreter lacks vtb). Reads a JSON payload and runs the retrieval.
    import argparse
    _ap = argparse.ArgumentParser(description="scorecards4extremes retrieval worker")
    _ap.add_argument('--stvl-worker', dest='stvl_worker', metavar='ARGS_JSON',
                     help='path to a JSON file with the retrieve_obs arguments')
    _args = _ap.parse_args()
    if _args.stvl_worker:
        with open(_args.stvl_worker) as _fh:
            _p = json.load(_fh)
        _retrieve_obs_inproc(_p['stvl_cfg'], _p['variable'], _p['start_date'],
                             _p['end_date'], _p['steps'], _p.get('base_time', '00'))
    else:
        _ap.error("no action requested (expected --stvl-worker)")
