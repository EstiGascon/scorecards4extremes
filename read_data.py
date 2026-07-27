"""
STEP 1: READ DATA
=================
Options: 
  - Forecast Model 1 & 2: Local GRIB files or Quaver/VTB (MARS retrieve)
  - Observations: Local .gpt files or Quaver/VTB (STVL retrieve)
"""

from pathlib import Path
from datetime import datetime, timedelta

from utils import compute_steps as _compute_steps


def _resolve_forecast_source(cfg_model, model_label, config):
    """Resolve a single forecast model source, return (path_or_None, name, source)."""
    name = cfg_model['name']
    source = cfg_model['source']
    print(f"  Name: {name}")
    print(f"  Source: {source}")

    if source == 'local_grib':
        path = cfg_model['local_grib']['path']
        print(f"  Path: {path}")
        if not Path(path).exists():
            raise FileNotFoundError(f"{model_label} path not found: {path}")
        return path, name, source

    elif source == 'mars':
        # Retrieve forecast GRIB now, then treat as local_grib downstream.
        import mars_retrieve
        mars_cfg = cfg_model['mars']
        steps, _ = _compute_steps(config)
        print(f"  MARS class={mars_cfg.get('class')}, type={mars_cfg.get('type')}, "
              f"stream={mars_cfg.get('stream')}, expver={mars_cfg.get('expver')}")
        target_dir = mars_retrieve.retrieve_forecast(
            mars_cfg,
            variable=config['variable'],
            start_date=config['start_date'],
            end_date=config['end_date'],
            steps=steps,
            mode=config.get('mode', 'deterministic'),
        )
        # Return as local_grib so all downstream extraction is unchanged
        return str(target_dir), name, 'local_grib'

    elif source == 'quaver':
        q = cfg_model.get('quaver', {})
        print(f"  MARS class={q.get('class')}, expver={q.get('expver')}, "
              f"stream={q.get('stream')}, type={q.get('type')}")
        # No local path — data will be retrieved via VTB at extraction time
        return None, name, source

    else:
        raise ValueError(f"Unknown forecast source for {model_label}: {source}")


def run_step1(config):
    """
    Execute Step 1: Read Data
    Returns paths to both forecast models and observation data.

    When source='quaver', forecast paths are None (retrieved later via VTB).
    """
    print("\n" + "="*80)
    print("STEP 1: READ DATA")
    print("="*80)
    
    cfg = config['read_data']
    
    # Read forecast model 1
    print("\nForecast Model 1:")
    fc1_path, fc1_name, fc1_source = _resolve_forecast_source(cfg['forecast_model1'], "Forecast model 1", config)

    # Read forecast model 2
    print("\nForecast Model 2:")
    fc2_path, fc2_name, fc2_source = _resolve_forecast_source(cfg['forecast_model2'], "Forecast model 2", config)
    
    # Read observation data
    print("\nObservation data:")
    obs_source = cfg['observation_source']
    print(f"  Source: {obs_source}")
    
    if obs_source == 'local_gpt':
        obs_path = cfg['local_gpt']['path']
        print(f"  Path: {obs_path}")
        if not Path(obs_path).exists():
            raise FileNotFoundError(f"Observation path not found: {obs_path}")

    elif obs_source == 'stvl':
        # Retrieve observations from STVL now, then treat as local_gpt downstream.
        import mars_retrieve
        stvl_cfg = cfg['stvl']
        steps, _ = _compute_steps(config)
        base_time = cfg.get('forecast_model1', {}).get('mars', {}).get('time', '00')
        print(f"  STVL sources: {stvl_cfg.get('sources', ['synop'])}")
        obs_dir = mars_retrieve.retrieve_obs(
            stvl_cfg,
            variable=config['variable'],
            start_date=config['start_date'],
            end_date=config['end_date'],
            steps=steps,
            base_time=base_time,
        )
        obs_path = str(obs_dir)
        obs_source = 'local_gpt'  # downstream reads the materialised .geo files

    elif obs_source == 'quaver':
        obs_path = None  # Will be retrieved via STVL at extraction time
        q_obs = cfg.get('quaver_obs', {})
        print(f"  STVL sources: {q_obs.get('sources', ['synop'])}")

    else:
        raise ValueError(f"Unknown observation source: {obs_source}")
    
    print("\n✓ Step 1 complete")
    
    return {
        'fc1_path': fc1_path,
        'fc1_name': fc1_name,
        'fc1_source': fc1_source,
        'fc2_path': fc2_path,
        'fc2_name': fc2_name,
        'fc2_source': fc2_source,
        'obs_path': obs_path,
        'obs_source': obs_source,
    }
