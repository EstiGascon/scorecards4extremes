
import pandas as pd
import numpy as np
from pathlib import Path

# --- Configuration ---
EXPERIMENT_NAME = "tp24_local_p99obsclim_ifs_oper_aifs1.0_oper_new"
FORECAST_DAY = 3
LEAD_TIME_HOURS = 72
SEASON = "JJA"
OROGRAPHY = "flat"
OROGRAPHY_RANGES = {"flat": [0, 40], "hilly": [40, 120], "complex": [120, 3000]}
EVENT_TYPE = "above"

# --- Paths ---
EXTRACTED_DATA_DIR = Path(f"./extracted_points/{EXPERIMENT_NAME}")
RESULTS_DIR = Path(f"./results/{EXPERIMENT_NAME}")
CLIMATOLOGY_DIR = Path("./obs_clim_local")

# --- Independent Score Functions ---

def get_contingency_table(obs, fc, threshold, event_type='above'):
    if event_type == 'above':
        obs_event = obs >= threshold
        fc_event = fc >= threshold
    else: # 'below'
        obs_event = obs <= threshold
        fc_event = fc <= threshold

    hits = np.sum(obs_event & fc_event)
    misses = np.sum(obs_event & ~fc_event)
    false_alarms = np.sum(~obs_event & fc_event)
    correct_negatives = np.sum(~obs_event & ~fc_event)
    
    return hits, misses, false_alarms, correct_negatives

def calculate_ets(h, m, fa, cn):
    if h + m + fa == 0: return 0.0
    h_random = ((h + m) * (h + fa)) / (h + m + fa + cn)
    return (h - h_random) / (h + m + fa - h_random)

def calculate_pss(h, m, fa, cn):
    pod = h / (h + m) if (h + m) > 0 else 0.0
    pofd = fa / (fa + cn) if (fa + cn) > 0 else 0.0
    return pod - pofd

def calculate_pod(h, m, fa, cn):
    return h / (h + m) if (h + m) > 0 else 0.0

def calculate_far(h, m, fa, cn):
    return fa / (h + fa) if (h + fa) > 0 else 0.0

def calculate_tw_mae(obs, fc, threshold, event_type='above'):
    if event_type == 'above':
        extreme_cases = obs >= threshold
    else:
        extreme_cases = obs <= threshold
    
    if np.sum(extreme_cases) == 0:
        return 0.0
    
    return np.mean(np.abs(fc[extreme_cases] - obs[extreme_cases]))

def calculate_tw_rmse(obs, fc, threshold, event_type='above'):
    if event_type == 'above':
        extreme_cases = obs >= threshold
    else:
        extreme_cases = obs <= threshold

    if np.sum(extreme_cases) == 0:
        return 0.0

    return np.sqrt(np.mean((fc[extreme_cases] - obs[extreme_cases])**2))

def calculate_bias(obs, fc):
    return np.mean(fc - obs)

def calculate_mae(obs, fc):
    return np.mean(np.abs(fc - obs))

def calculate_rmse(obs, fc):
    return np.sqrt(np.mean((fc - obs)**2))


def main():
    print("--- Independent Verification Test ---")
    
    # 1. Load Data
    data_file = EXTRACTED_DATA_DIR / f"tp24_ifs_oper_0.25degree_vs_aifs1.0_oper_0.25degree_day{FORECAST_DAY}.parquet"
    print(f"Loading data from: {data_file}")
    df = pd.read_parquet(data_file)

    # 2. Filter Data
    # Season
    season_months = {'JJA': [6, 7, 8]}
    df['month'] = pd.to_datetime(df['date'], format='%Y%m%d').dt.month
    df = df[df['month'].isin(season_months[SEASON])]
    # Orography
    orog_min, orog_max = OROGRAPHY_RANGES[OROGRAPHY]
    df = df[(df['sdfor'] >= orog_min) & (df['sdfor'] < orog_max)]
    print(f"Filtered data to {len(df)} rows for {SEASON} season and '{OROGRAPHY}' orography.")

    # 3. Load Threshold
    clim_files = [CLIMATOLOGY_DIR / f"clim_tp_1_{str(m).zfill(2)}_20years_2005_2024_65" for m in season_months[SEASON]]
    clim_df = pd.concat([pd.read_csv(f, sep=";") for f in clim_files])
    clim_df = clim_df.rename(columns={'lat': 'clim_lat', 'lon': 'clim_lon', 'p99': 'threshold'})
    
    # Simple nearest neighbor merge (for test purposes)
    df['lat_round'] = df['lat'].round(1)
    df['lon_round'] = df['lon'].round(1)
    clim_df['lat_round'] = clim_df['clim_lat'].round(1)
    clim_df['lon_round'] = clim_df['clim_lon'].round(1)
    
    # Use monthly threshold
    clim_df_monthly = clim_df.groupby(['lat_round', 'lon_round', 'month'])['threshold'].mean().reset_index()
    df = pd.merge(df, clim_df_monthly, on=['lat_round', 'lon_round', 'month'], how='inner')
    print(f"Successfully merged {len(df)} rows with climatology thresholds.")

    # 4. Calculate Scores Independently
    obs = df['obs_value'].values
    fc1 = df['fc1_value'].values
    fc2 = df['fc2_value'].values
    threshold = df['threshold'].values

    scores = {}
    for model_idx, fc in enumerate([fc1, fc2]):
        model_name = f"fc{model_idx+1}"
        h, m, fa, cn = get_contingency_table(obs, fc, threshold, EVENT_TYPE)
        scores[f'{model_name}_ETS'] = calculate_ets(h, m, fa, cn)
        scores[f'{model_name}_PSS'] = calculate_pss(h, m, fa, cn)
        scores[f'{model_name}_POD'] = calculate_pod(h, m, fa, cn)
        scores[f'{model_name}_FAR'] = calculate_far(h, m, fa, cn)
        scores[f'{model_name}_twMAE'] = calculate_tw_mae(obs, fc, threshold, EVENT_TYPE)
        scores[f'{model_name}_twRMSE'] = calculate_tw_rmse(obs, fc, threshold, EVENT_TYPE)
        scores[f'{model_name}_bias'] = calculate_bias(obs, fc)
        scores[f'{model_name}_mae'] = calculate_mae(obs, fc)
        scores[f'{model_name}_rmse'] = calculate_rmse(obs, fc)

    # 5. Load Official Results
    official_results_file = RESULTS_DIR / f"scores_{SEASON}_{OROGRAPHY}.csv"
    print(f"Loading official results from: {official_results_file}")
    official_df = pd.read_csv(official_results_file)
    official_scores = official_df[official_df['lead_time'] == LEAD_TIME_HOURS].iloc[0]

    # 6. Compare and Report
    print("\n--- Comparison Report ---")
    print(f"Condition: Day={FORECAST_DAY}, Season={SEASON}, Orography={OROGRAPHY}")
    print("-" * 50)
    print(f"{'Score':<10} | {'Independent':<15} | {'Official':<15} | {'Match':<5}")
    print("-" * 50)
    
    all_match = True
    for score_name in ['ETS', 'PSS', 'POD', 'FAR', 'twMAE', 'twRMSE', 'bias', 'mae', 'rmse']:
        for model in ['fc1', 'fc2']:
            key = f"{model}_{score_name}"
            ind_val = scores[key]
            off_val = official_scores[key]
            
            match = "✅" if np.isclose(ind_val, off_val, atol=1e-6) else "❌"
            if match == "❌":
                all_match = False
            
            print(f"{key:<10} | {ind_val:<15.6f} | {off_val:<15.6f} | {match}")
    
    print("-" * 50)
    if all_match:
        print("\n🎉 SUCCESS: All independently calculated scores match the official results.")
    else:
        print("\n🚨 FAILURE: One or more scores do not match.")

if __name__ == "__main__":
    main()
