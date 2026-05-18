import pandas as pd
import numpy as np
import os
import sys

# Ensure we can import the library
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from ecd_nested_simulation_functions import generate_ecd_dummy_data
    print("✅ Library imported successfully.")
except ImportError:
    # Fallback if file is in the same directory but not a package
    import generate_ecd_dummy_data
    print("✅ Library imported (local).")

# ==============================================================================
# 1. SETUP: WHO DATA LOADER
# ==============================================================================
# Update this path to your folder
WHO_DATA_DIR = r"C:\Users\CEGIS FOUNDATION\New folder\GitHub\discsim\igrowup_update"

def load_who_standards(base_dir):
    try:
        haz = pd.read_stata(os.path.join(base_dir, "lenanthro.dta"))
        waz = pd.read_stata(os.path.join(base_dir, "weianthro.dta"))
        whz_l = pd.read_stata(os.path.join(base_dir, "wflanthro.dta"))
        whz_s = pd.read_stata(os.path.join(base_dir, "wfhanthro.dta"))
        
        # Rename columns to match library expectations
        haz = haz.rename(columns={'age': '_agedays', 'sex': '__000001'})
        waz = waz.rename(columns={'age': '_agedays', 'sex': '__000001'})
        
        # Fix WHZ columns
        len_col = 'length' if 'length' in whz_l.columns else 'height'
        whz_l = whz_l.rename(columns={len_col: '__000002', 'sex': '__000001'})
        
        ht_col = 'height' if 'height' in whz_s.columns else 'length'
        whz_s = whz_s.rename(columns={ht_col: '__000003', 'sex': '__000001'})
        
        return haz, waz, whz_l, whz_s
    except Exception as e:
        print(f"❌ Error loading WHO standards: {e}")
        return None, None, None, None

# ==============================================================================
# 2. SIMULATION RUNNER
# ==============================================================================
def run_simulation_scenario(name, lag_l1, lag_l2, haz, waz, whz_l, whz_s, seed=42):
    print(f"\n--- Running Scenario: {name} (L1={lag_l1}d, L2={lag_l2}d) ---")
    
    # 1. Generate Parameters
    # We use the SAME seed to ensure the "Real" children are generated identically
    L0_params, L1_params, L2_params = generate_ecd_dummy_data.generate_nested_distortion_parameters(
        n_L1s=5, 
        n_L0s_per_L1=4,
        mean_time_lag_L1=lag_l1,
        mean_time_lag_L2=lag_l2,
        random_seed=seed 
    )
    
    # 2. Generate Data
    # Reset numpy seed explicitly before data generation to guarantee identical children
    np.random.seed(seed)
    
    nested_data = generate_ecd_dummy_data.generate_nested_measurements(
        real_params={
            'girl_ratio': 0.5, 'min_age': 0, 'max_age': 1800, 'num_timepoints': 1,
            'percent_stunting': 30.0, 'percent_underweight': 25.0, 'time_lags': []
        },
        L0_params_list=L0_params,
        L1_params_list=L1_params,
        L2_params_dict=L2_params,
        n_L1s=5, n_L0s_per_L1=4, n_children_per_L0=20, n_children_L1=10, n_children_L2=10,
        haz_params=haz, waz_params=waz, whz_params_lying=whz_l, whz_params_standing=whz_s,
        make_plots=False
    )
    return nested_data

def extract_l2_results(nested_data):
    """Flattens the nested dictionary to get all L2 (Auditor) measurements."""
    records = []
    for l1_key, l0_dict in nested_data.items():
        if l1_key == 'metadata': continue
        for l0_key, datasets in l0_dict.items():
            if l0_key == 'L1_info': continue
            
            # Get L2 DataFrame
            df = datasets['L2']['data'].copy()
            
            # We assume the index corresponds to child_id or is preserved
            # Ideally we'd have a 'child_id' column to merge on. 
            # If 'child_id' is in columns, use it.
            if 'child_id' not in df.columns:
                df['child_id'] = df.index # Fallback
                
            records.append(df)
            
    return pd.concat(records)

# ==============================================================================
# 3. MAIN COMPARISON LOGIC
# ==============================================================================
if __name__ == "__main__":
    # Load Standards
    haz, waz, whz_l, whz_s = load_who_standards(WHO_DATA_DIR)
    
    if haz is not None:
        # RUN SCENARIO 1: CONTROL (No Time Lag)
        data_control = run_simulation_scenario("CONTROL", 0, 0, haz, waz, whz_l, whz_s)
        df_control = extract_l2_results(data_control)
        
        # RUN SCENARIO 2: DRIFT (15d / 30d Lag)
        data_drift = run_simulation_scenario("DRIFT", 15, 30, haz, waz, whz_l, whz_s)
        df_drift = extract_l2_results(data_drift)
        
        # MERGE & COMPARE
        # We merge on 'child_id' to compare the SAME child across simulations
        # Since we used the same seed, child_0 in Control should match child_0 in Drift
        merged = pd.merge(
            df_control[['child_id', 'haz', 'waz', 'whz', 'height', 'weight']],
            df_drift[['child_id', 'haz', 'waz', 'whz', 'height', 'weight']],
            on='child_id',
            suffixes=('_control', '_drift')
        )
        
        # Calculate Differences (Drift - Control)
        merged['diff_haz'] = merged['haz_drift'] - merged['haz_control']
        merged['diff_waz'] = merged['waz_drift'] - merged['waz_control']
        merged['diff_height'] = merged['height_drift'] - merged['height_control']
        
        print("\n" + "="*60)
        print("COMPARISON RESULTS: SCENARIO 1 (0d) vs SCENARIO 2 (30d)")
        print("="*60)
        print(f"Total Children Compared: {len(merged)}")
        print("-" * 30)
        print(f"Mean Height Drift:   {merged['diff_height'].mean():.4f} cm")
        print(f"Mean HAZ Drift:      {merged['diff_haz'].mean():.4f} SD")
        print(f"Mean WAZ Drift:      {merged['diff_waz'].mean():.4f} SD")
        print("-" * 30)
        
        print("\nInterpretation:")
        print("If Mean Height Drift > 0, the Physics Engine is working correctly.")
        print("(Children grew during the 30-day lag)")