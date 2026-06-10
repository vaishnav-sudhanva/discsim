# import os
# import sys
# import numpy as np
# import pandas as pd

# # ==============================================================================
# # 1. DYNAMIC PATH RESOLUTION
# # ==============================================================================
# CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# API_DIR = os.path.dirname(CURRENT_DIR)
# WHO_DIR = os.path.join(CURRENT_DIR, "who")  # dynamically finds 'who' next to this script

# # We add API_DIR to path so Python sees the folder as a package
# if API_DIR not in sys.path: 
#     sys.path.append(API_DIR)

# # FIX: Also explicitly add the ecd_nested_simulation_functions directory so it can find disc_score.py
# ECD_DIR = os.path.join(API_DIR, "ecd_nested_simulation_functions")
# if ECD_DIR not in sys.path:
#     sys.path.append(ECD_DIR)

# from ecd_nested_simulation_functions import generate_ecd_dummy_data
import os
import sys
import numpy as np
import pandas as pd

# ==============================================================================
# 1. DYNAMIC PATH RESOLUTION
# ==============================================================================
# Safely get the directory where data_generator.py lives (the 'services' folder)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Safely point to the WHO data folder located next to this script
WHO_DIR = os.path.join(CURRENT_DIR, "who")

# 🟢 Clean, absolute import (Works perfectly because Uvicorn runs from 'api')
from ecd_nested_simulation_functions import generate_ecd_dummy_data

def build_universe(params, task_id, output_dir) -> tuple[str, pd.DataFrame]:
    print(f"   -> [Step 1] Building Synthetic Universe for task: {task_id}")
    
    # =========================================================
    # 1. SAFELY UNPACK KEYS (Strict Integer Casting for Pylance)
    # =========================================================
    if hasattr(params, "dict"):
        params = params.dict()
    elif hasattr(params, "model_dump"):
        params = params.model_dump()
    else:
        params = dict(params)

    n_l1 = int(params.get("n_L1s", 100))
    n_l0 = int(params.get("n_L0s_per_L1", 25))
    n_kids = int(params.get("n_children_per_L0", 15))
    
    # Cast to float first to handle strings like "5.0", then strict int for Pylance
    stunting_ur = int(float(params.get("mean_percent_under_reporting_stunting", 5.0)))
    underweight_ur = int(float(params.get("mean_percent_under_reporting_underweight", 5.0)))
    
    bunch_haz = float(params.get("mean_bunch_factor_haz", 0.05))
    bunch_waz = float(params.get("mean_bunch_factor_waz", 0.05))
    bunch_whz = float(params.get("mean_bunch_factor_whz", 0.05))
    
    copy_pct = int(float(params.get("mean_percent_copy", 5.0)))
    collusion = float(params.get("mean_collusion_index", 0.05))
    
    err_ht = int(float(params.get("error_sd_height_all_L0s", 0.0)))
    err_wt = int(float(params.get("error_sd_weight_all_L0s", 0.0)))
    
    rho_val = float(params.get("rho", 0.7))

    # =========================================================
    # 2. LOAD WHO FILES
    # =========================================================
    def get_dta(file):
        target_path = os.path.join(WHO_DIR, file)
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"CRITICAL ERROR: Could not find {file} precisely at {target_path}")
        return pd.read_stata(target_path)
    
    haz_p, waz_p = get_dta('lenanthro.dta'), get_dta('weianthro.dta')
    whz_ly, whz_st = get_dta('wflanthro.dta'), get_dta('wfhanthro.dta')

    common_params = {
        'girl_ratio': 0.5, 'min_age': 0, 'max_age': 1790, 'num_timepoints': 1, 'time_lags': [],
        'percent_stunting': float(params.get("real_percent_stunting", 35.0)), 
        'percent_underweight': float(params.get("real_percent_underweight", 33.0)), 
        'rho': rho_val
    }

    # =========================================================
    # 3. GENERATE BIOLOGY & FRAUD 
    # =========================================================
    print("      * Simulating field fraud physics...")
    L0_p, L1_p, L2_p = generate_ecd_dummy_data.generate_nested_distortion_parameters(
        n_L1s=n_l1, n_L0s_per_L1=n_l0, 
        real_percent_stunting=common_params['percent_stunting'], 
        real_percent_underweight=common_params['percent_underweight'],
        
        mean_percent_under_reporting_stunting=stunting_ur, 
        mean_percent_under_reporting_underweight=underweight_ur, 
        
        mean_bunch_factor_haz=bunch_haz, mean_bunch_factor_waz=bunch_waz, mean_bunch_factor_whz=bunch_whz,
        
        mean_percent_copy=copy_pct, 
        mean_collusion_index=collusion,
        
        sd_across_units_percent_under_reporting_stunting=int(float(params.get("sd_across_units_percent_under_reporting_stunting", 2))),
        sd_across_units_percent_under_reporting_underweight=int(float(params.get("sd_across_units_percent_under_reporting_underweight", 2))),
        sd_within_units_percent_under_reporting_stunting=int(float(params.get("sd_within_units_percent_under_reporting_stunting", 1))),
        sd_within_units_percent_under_reporting_underweight=int(float(params.get("sd_within_units_percent_under_reporting_underweight", 1))),
        
        sd_across_units_bunch_factor_haz=float(params.get("sd_across_units_bunch_factor_haz", 0.01)),
        sd_across_units_bunch_factor_waz=float(params.get("sd_across_units_bunch_factor_waz", 0.01)),
        sd_across_units_bunch_factor_whz=float(params.get("sd_across_units_bunch_factor_whz", 0.01)),
        sd_within_units_bunch_factor_haz=float(params.get("sd_within_units_bunch_factor_haz", 0.01)),
        sd_within_units_bunch_factor_waz=float(params.get("sd_within_units_bunch_factor_waz", 0.01)),
        sd_within_units_bunch_factor_whz=float(params.get("sd_within_units_bunch_factor_whz", 0.01)),
        
        sd_percent_copy=int(float(params.get("sd_percent_copy", 2))), 
        sd_collusion_index=float(params.get("sd_collusion_index", 0.02)),
        
        error_sd_height_all_L0s=err_ht, 
        error_sd_weight_all_L0s=err_wt,
        error_sd_height_L1=err_ht, 
        error_sd_weight_L1=err_wt, 
        error_sd_height_L2=err_ht, 
        error_sd_weight_L2=err_wt, 
        
        mean_time_lag_L1=int(params.get("mean_time_lag_L1", 15)), 
        mean_time_lag_L2=int(params.get("mean_time_lag_L2", 30))
    )

    np.random.seed(42)
    nested_measurements = generate_ecd_dummy_data.generate_nested_measurements(
        real_params=common_params, L0_params_list=L0_p, L1_params_list=L1_p, L2_params_dict=L2_p,
        n_L1s=n_l1, n_L0s_per_L1=n_l0, n_children_per_L0=n_kids, n_children_L1=n_kids, n_children_L2=n_kids,
        haz_params=haz_p, waz_params=waz_p, whz_params_lying=whz_ly, whz_params_standing=whz_st, make_plots=False
    )

    # =========================================================
    # 4. FLATTEN AND RESTORE RAW COLUMNS
    # =========================================================
    records = []
    for L1_id, L1_data in nested_measurements.items():
        if L1_id == 'metadata': continue
        for L0_id, L0_data in L1_data.items():
            if L0_id == 'L1_info': continue
            
            df_real = L0_data['real']['data'].copy().rename(columns={'haz': 'real_haz', 'waz': 'real_waz', 'whz': 'real_whz', 'height': 'real_height', 'weight': 'real_weight', 'age': 'real_age_days'})
            df_L0 = L0_data['L0']['data'].copy().rename(columns={'haz': 'L0_haz', 'waz': 'L0_waz', 'whz': 'L0_whz', 'height': 'L0_height', 'weight': 'L0_weight'})
            df_L1 = L0_data['L1']['data'].copy().rename(columns={'haz': 'L1_haz', 'waz': 'L1_waz', 'whz': 'L1_whz', 'height': 'L1_height', 'weight': 'L1_weight', 'age': 'L1_age_days'})
            
            # Dynamic matching: ONLY selects columns that actually exist to prevent KeyErrors
            real_cols = [c for c in ['child_id', 'gender', 'real_age_days', 'real_height', 'real_weight', 'real_haz', 'real_waz', 'real_whz', 'loh'] if c in df_real.columns]
            l0_cols = [c for c in ['child_id', 'L0_height', 'L0_weight', 'L0_haz', 'L0_waz', 'L0_whz'] if c in df_L0.columns]
            l1_cols = [c for c in ['child_id', 'L1_age_days', 'L1_height', 'L1_weight', 'L1_haz', 'L1_waz', 'L1_whz'] if c in df_L1.columns]
            
            merged = df_real[real_cols].merge(
                df_L0[l0_cols], on='child_id', how='left').merge(
                df_L1[l1_cols], on='child_id', how='left')
            
            # THE TRUE FIX: Ensure L2 is not None before trying to extract 'data'
            l2_node = L0_data.get('L2')
            if l2_node is not None:
                l2_raw = l2_node.get('data')
                if isinstance(l2_raw, pd.DataFrame) and not l2_raw.empty:
                    df_L2 = l2_raw.copy().rename(columns={'haz': 'L2_haz', 'waz': 'L2_waz', 'whz': 'L2_whz', 'height': 'L2_height', 'weight': 'L2_weight', 'age': 'L2_age_days'})
                    l2_cols = [c for c in ['child_id', 'L2_age_days', 'L2_height', 'L2_weight', 'L2_haz', 'L2_waz', 'L2_whz'] if c in df_L2.columns]
                    merged = merged.merge(df_L2[l2_cols], on='child_id', how='left')
            
            merged['L1_id'] = L1_id
            merged['L0_id'] = L0_id
            records.append(merged)

    df_pop = pd.concat(records, ignore_index=True)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    child_path = os.path.join(output_dir, f"child_data_{task_id}.parquet")
    df_pop.to_parquet(child_path, engine='pyarrow', index=False)
    
    return child_path, df_pop