"""
STEP 2: PRE-PROCESS DATA
=========================
Variable-specific preprocessing:
  - 10ff: Calculate wind speed from u/v components
  - 2t: Apply lapse-rate correction
  - tp24: Compute 24h accumulation
"""


def run_step2(config, paths):
    """
    Execute Step 2: Pre-process Data
    Returns preprocessing settings to use in extraction
    """
    print("\n" + "="*80)
    print("STEP 2: PRE-PROCESS DATA")
    print("="*80)
    
    variable = config['variable']
    cfg = config['preprocess']
    
    print(f"\nVariable: {variable}")
    
    preprocess_settings = {}
    
    if variable == '2t':
        lapse_rate = cfg.get('lapse_rate', -0.0065)
        print(f"  → Lapse-rate correction will be applied: {lapse_rate} K/m")
        preprocess_settings['lapse_rate_correction'] = True
        preprocess_settings['lapse_rate'] = lapse_rate
    
    elif variable == '10ff':
        if cfg.get('wind_speed_from_components', False):
            print(f"  → Wind speed will be calculated from u/v components")
            preprocess_settings['calculate_wind_speed'] = True
        else:
            print(f"  → Wind speed will be read directly")
            preprocess_settings['calculate_wind_speed'] = False
    
    elif variable == 'tp24':
        period = cfg.get('precipitation_accumulation_hours', 24)
        print(f"  → {period}h accumulation will be computed")
        preprocess_settings['accumulation_period'] = period
    
    print("\n✓ Step 2 complete")
    
    return preprocess_settings
