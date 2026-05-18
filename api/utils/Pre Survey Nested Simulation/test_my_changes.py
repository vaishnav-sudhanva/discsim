import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# ==============================================================================
# 1. SETUP & IMPORTS
# ==============================================================================
# Ensure we can import from the sibling package 'ecd_nested_simulation_functions'
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from ecd_nested_simulation_functions import generate_ecd_dummy_data
    from ecd_nested_simulation_functions import ecd_anthro_score_calc
    print("✅ Successfully imported simulation libraries.")
except ImportError as e:
    print("CRITICAL ERROR: Could not import simulation functions.")
    print(f"Make sure the folder 'ecd_nested_simulation_functions' is inside: {current_dir}")
    raise e

# ==============================================================================
# 2. CONFIGURATION
# ==============================================================================
# [USER ACTION REQUIRED] Update this path to your WHO standards folder
WHO_DATA_DIR = r"C:\Users\CEGIS FOUNDATION\New folder\GitHub\discsim\igrowup_update"

SIM_CONFIG = {
    'random_seed': 42,
    
    # Hierarchy
    'n_L1s': 5,              # 5 Supervisors
    'n_L0s_per_L1': 4,       # 4 Surveyors per Supervisor
    'n_children_per_L0': 20, # 20 Children per Surveyor
    
    # Sampling
    'n_children_L1': 10,     # Supervisor checks 10 kids
    'n_children_L2': 5,      # Auditor checks 5 kids
    
    # PHYSICS / TIME LAGS (The Core Experiment)
    'mean_time_lag_L1': 15,  # Supervisor measures 15 days later (Growth happens!)
    'sd_time_lag_L1': 2,     # +/- 2 days variability
    'mean_time_lag_L2': 30,  # Auditor measures 30 days later (Significant growth!)
}

REAL_POP_PARAMS = {
    # Study Design
    'girl_ratio': 0.5,
    'min_age': 0,
    'max_age': 1800,         # 0-5 years
    'num_timepoints': 1,     # Single survey
    'time_lags': [],         # Empty for single survey
    
    # Health Status
    'percent_stunting': 30.0,    # 30% Stunting
    'percent_underweight': 25.0, # 25% Underweight
}

# ==============================================================================
# 3. DATA LOADER (THE BRIDGE)
# ==============================================================================
def load_who_standards(base_dir):
    """
    Loads .dta files and renames columns to match the simulation engine's requirements.
    Expected Mapping:
    - age -> _agedays
    - sex -> __000001 (1=Male, 2=Female)
    - length/height -> __000002 / __000003
    """
    print(f"[1/5] Loading WHO Standards from: {base_dir}")
    
    try:
        # 1. Load HAZ (Length/Height for Age)
        # Usually 'lenanthro.dta' (0-24m) or combined. We assume lenanthro covers the need or matches format.
        haz_path = os.path.join(base_dir, "lenanthro.dta")
        haz = pd.read_stata(haz_path)
        # Rename columns to match ecd_anthro_score_calc expectations
        haz = haz.rename(columns={'age': '_agedays', 'sex': '__000001'})
        
        # 2. Load WAZ (Weight for Age)
        waz_path = os.path.join(base_dir, "weianthro.dta")
        waz = pd.read_stata(waz_path)
        waz = waz.rename(columns={'age': '_agedays', 'sex': '__000001'})
        
        # 3. Load WHZ (Weight for Length/Height)
        # Lying (0-2 years, Length) -> wflanthro.dta
        whz_l_path = os.path.join(base_dir, "wflanthro.dta")
        whz_lying = pd.read_stata(whz_l_path)
        # Note: 'length' column needs to be mapped to the internal variable name used by the calc script
        # Check if 'length' exists, else try 'height'
        len_col = 'length' if 'length' in whz_lying.columns else 'height'
        whz_lying = whz_lying.rename(columns={len_col: '__000002', 'sex': '__000001'})
        
        # Standing (2-5 years, Height) -> wfhanthro.dta
        whz_s_path = os.path.join(base_dir, "wfhanthro.dta")
        whz_standing = pd.read_stata(whz_s_path)
        ht_col = 'height' if 'height' in whz_standing.columns else 'length'
        whz_standing = whz_standing.rename(columns={ht_col: '__000003', 'sex': '__000001'})
        
        print("   -> Loaded: HAZ, WAZ, WHZ (Lying), WHZ (Standing)")
        return haz, waz, whz_lying, whz_standing
        
    except FileNotFoundError as e:
        print(f"ERROR: Missing WHO file. {e}")
        print("Please check the filenames in your 'igrowup_update' folder.")
        # FALLBACK: Create Dummy data if files are missing (Just so script doesn't crash during testing)
        print("   -> WARNING: Using Dummy Data Fallback due to missing files!")
        return create_dummy_fallback()

def create_dummy_fallback():
    # ... (Keep the dummy generator from previous solution as backup) ...
    days = np.arange(0, 2001)
    m_h = 50 + (days * 0.03) + np.log1p(days) * 5 
    m_w = 3 + (days * 0.008) + np.log1p(days) * 0.5
    haz = pd.DataFrame({'_agedays': days, '__000001': np.tile([1, 2], len(days)//2 + 1)[:len(days)], 'l': 1, 'm': m_h, 's': 0.05})
    waz = pd.DataFrame({'_agedays': days, '__000001': np.tile([1, 2], len(days)//2 + 1)[:len(days)], 'l': 1, 'm': m_w, 's': 0.10})
    whz_l = pd.DataFrame({'__000002': np.arange(45, 120, 0.1), '__000001': 1, 'l': 1, 'm': 10, 's': 0.1})
    whz_s = whz_l.rename(columns={'__000002': '__000003'})
    return haz, waz, whz_l, whz_s

# ==============================================================================
# 4. MAIN EXECUTION
# ==============================================================================
def run_simulation():
    print("\n--- STARTING FULL ANALYSIS ---")
    
    # A. Load Parameters
    haz, waz, whz_l, whz_s = load_who_standards(WHO_DATA_DIR)

    # B. Configure Surveyors
    print(f"[2/5] Configuring Surveyors... (L1 Lag={SIM_CONFIG['mean_time_lag_L1']}d, L2 Lag={SIM_CONFIG['mean_time_lag_L2']}d)")
    
    distortion_config = {k: v for k, v in SIM_CONFIG.items() if 'n_children' not in k}
    L0_params, L1_params, L2_params = generate_ecd_dummy_data.generate_nested_distortion_parameters(
        **distortion_config,
        real_percent_stunting=REAL_POP_PARAMS['percent_stunting'],
        mean_percent_under_reporting_stunting=20
    )

    # C. Run Simulation
    print("[3/5] Generating Nested Data (Physics Engine Active)...")
    nested_data = generate_ecd_dummy_data.generate_nested_measurements(
        real_params=REAL_POP_PARAMS, 
        L0_params_list=L0_params,
        L1_params_list=L1_params,
        L2_params_dict=L2_params,
        n_L1s=SIM_CONFIG['n_L1s'],
        n_L0s_per_L1=SIM_CONFIG['n_L0s_per_L1'],
        n_children_per_L0=SIM_CONFIG['n_children_per_L0'], 
        n_children_L1=SIM_CONFIG['n_children_L1'],
        n_children_L2=SIM_CONFIG['n_children_L2'],
        haz_params=haz,
        waz_params=waz,
        whz_params_lying=whz_l,
        whz_params_standing=whz_s,
        make_plots=False 
    )

    # D. Extract Data
    print("[4/5] Extracting Analysis Data...")
    pairwise_data = generate_ecd_dummy_data.get_L1_L2_pairwise_data(nested_data)
    
    return pairwise_data

# ==============================================================================
# 5. ANALYSIS & PLOTTING
# ==============================================================================
def analyze_results(pairwise_data):
    print("[5/5] Generating Dashboard...")
    
    rank_data = []

    # 1. Supervisor Ranking
    for l1_id, datasets in pairwise_data.items():
        l1_df = datasets['L1']
        l2_df = datasets['L2'] # The "Truth" (Auditor)
        
        # Calculate Error
        diff_height = l1_df['height'] - l2_df['height']
        abs_diff_height = np.abs(diff_height)
        
        rank_data.append({
            'Supervisor_ID': l1_id,
            'Mean_Abs_Error_Height': abs_diff_height.mean(),
            'Max_Error_Height': abs_diff_height.max(),
            'Sample_Size': len(l1_df)
        })

    rank_df = pd.DataFrame(rank_data)
    
    if rank_df.empty:
        print("No data available for analysis.")
        return

    rank_df.sort_values('Mean_Abs_Error_Height', inplace=True)

    # 2. Charts
    all_l1 = pd.concat([d['L1'] for d in pairwise_data.values()])
    all_l2 = pd.concat([d['L2'] for d in pairwise_data.values()])
    global_diff = all_l1['height'] - all_l2['height']
    
    plt.figure(figsize=(15, 10))
    
    # Plot A: Error Distribution
    plt.subplot(2, 2, 1)
    sns.histplot(global_diff, kde=True, bins=20, color='skyblue')
    plt.axvline(0, color='red', linestyle='--')
    plt.title("Height Discrepancy (L1 - L2)")
    plt.xlabel("Difference (cm)")
    
    # Plot B: Physics Trap Check
    # We plot Error vs True Height. If physics is active, we might see patterns based on age/size drift.
    plt.subplot(2, 2, 2)
    plt.scatter(all_l2['height'], global_diff, alpha=0.5, color='purple')
    plt.axhline(0, color='red', linestyle='--')
    plt.title("Physics Trap: Systematic Error vs Child Size")
    plt.xlabel("True Height (L2) [cm]")
    plt.ylabel("Error (L1 - L2) [cm]")

    # Plot C: Supervisor Ranking
    plt.subplot(2, 1, 2)
    sns.barplot(data=rank_df, x='Supervisor_ID', y='Mean_Abs_Error_Height', palette='viridis')
    plt.title("Supervisor Performance (Ranked by Accuracy)")
    plt.ylabel("Mean Absolute Error (cm)")
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

    print("\n" + "="*50)
    print(f"ANALYSIS COMPLETE: {len(all_l1)} Paired Measurements Processed")
    print(f"Time Lags Applied: L1={SIM_CONFIG['mean_time_lag_L1']}d, L2={SIM_CONFIG['mean_time_lag_L2']}d")
    print("="*50)
    print(rank_df.to_string(index=False))

if __name__ == "__main__":
    data = run_simulation()
    analyze_results(data)