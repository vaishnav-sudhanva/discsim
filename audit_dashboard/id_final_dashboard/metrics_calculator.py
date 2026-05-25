import pandas as pd
import numpy as np
import os
from datetime import datetime

def calculate_metrics(df, group_cols, col_meas, col_baseline):
    temp_df = df.copy()
    temp_df['_err'] = np.abs(temp_df[col_meas] - temp_df[col_baseline])
    temp_df['_sq_err'] = temp_df['_err'] ** 2
    return temp_df.groupby(group_cols).agg(
        MAE=('_err', 'mean'), RMSE=('_sq_err', lambda x: np.sqrt(np.mean(x))), P90=('_err', lambda x: np.quantile(x, 0.90))
    ).reset_index()

def get_top_k_overlap(ranked_df, god_set, rank_col, target_k, tie_breaker_col='L1_id'):
    ranked_df = ranked_df.sort_values(by=[rank_col, tie_breaker_col], ascending=[False, True]).reset_index(drop=True)
    caught_set = set(ranked_df.head(target_k)[tie_breaker_col])
    if target_k == 0: return 0.0
    return (len(caught_set & god_set) / target_k) * 100

def generate_dynamic_strategies(budget, max_c, max_k, target_qty=6):
    strats = []
    min_k = max(1, int(np.floor(budget / max_c)))
    max_possible_k = min(max_k, budget)
    if min_k > max_possible_k: return [(max_c, max_k)]
    raw_ks = np.unique(np.round(np.linspace(min_k, max_possible_k, target_qty * 2)).astype(int))
    for k in raw_ks:
        c = int(np.round(budget / k))
        if 1 <= c <= max_c and c * k <= max_c * max_k:
            if (c, k) not in strats: strats.append((c, k))
    strats = sorted(strats, key=lambda x: x[0], reverse=True)
    if len(strats) > target_qty:
        indices = np.round(np.linspace(0, len(strats)-1, target_qty)).astype(int)
        strats = [strats[i] for i in indices]
    while 0 < len(strats) < target_qty: strats.append(strats[-1])
    return strats

def run_engine(parquet_path, output_dir, scene_name):
    df_pop = pd.read_parquet(parquet_path)
    
    TARGET_L1_PERCENTILE = 0.30  
    TARGET_L0_PERCENTILE = 0.30 
    N_SIMULATIONS = 1           
    MAX_L1_CLINICS = df_pop['L0_id'].nunique()
    MAX_KIDS_PER_CLINIC = df_pop['child_id'].nunique() // MAX_L1_CLINICS // df_pop['L1_id'].nunique()
    TOTAL_L1_KIDS = MAX_L1_CLINICS * MAX_KIDS_PER_CLINIC  
    PERCENTAGES = [0.20, 0.40, 0.60, 0.80, 1.00]
    
    INDICATORS = {
        'Height': {'l0': 'L0_haz', 'l1': 'L1_haz', 'l2': 'L2_haz', 'real': 'real_haz'},
        'Weight': {'l0': 'L0_waz', 'l1': 'L1_waz', 'l2': 'L2_waz', 'real': 'real_waz'}
    }

    total_l1s = len(df_pop['L1_id'].unique())
    target_l1_count = max(1, int(np.floor(total_l1s * TARGET_L1_PERCENTILE)))
    
    master_database = []
    
    # ... [INSERT THE EXACT REST OF THE TRACER ENGINE SAMPLING LOOP HERE (Chunks 1, 2, and 3)] ...
    # (To save space here, you paste the exact loops we tested earlier using the `.sample(frac=1).head()` fix)
    
    # 5. Save the final CSV
    final_df = pd.DataFrame(master_database)
    final_df['Scenario'] = scene_name
    final_df['Rho_Model'] = 0.7  # Or pass this dynamically
    
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f"Tracer_Master_DB_Height_Weight_{timestamp}.csv")
    final_df.to_csv(csv_path, index=False)
    
    return csv_path