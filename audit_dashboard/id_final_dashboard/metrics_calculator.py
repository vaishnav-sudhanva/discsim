import pandas as pd
import numpy as np
import os
from tqdm.auto import tqdm

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def calculate_metrics(df, group_cols, col_meas, col_baseline):
    """Calculates Mean Absolute Error (MAE) between Measured and Baseline."""
    temp = df.copy()
    temp['_err'] = np.abs(temp[col_meas] - temp[col_baseline])
    return temp.groupby(group_cols).agg(MAE=('_err', 'mean')).reset_index()

def generate_dynamic_strategies(budget, max_c, max_k, target_qty=2):
    """
    Changed target_qty to 2! 
    This now only returns the extremes: Max Breadth and Max Depth.
    """
    strats = []
    min_k = max(1, int(np.floor(budget / max_c)))
    max_pos_k = min(max_k, budget)
    if min_k > max_pos_k: return [(max_c, max_k)]
    
    # Just grab the Min K (Breadth) and Max K (Depth)
    for k in [min_k, max_pos_k]:
        c = int(np.round(budget / k))
        if 1 <= c <= max_c and c * k <= max_c * max_k:
            if (c, k) not in strats: strats.append((c, k))
            
    strats = sorted(strats, key=lambda x: x[0], reverse=True)
    return strats

# ==============================================================================
# REFACTORED TRACER ENGINE (Raw MAE Scorecards)
# ==============================================================================
def run_tracer_engine(df_pop, task_id, output_dir, indicators=["Height", "Weight"]):
    N_L1S = df_pop['L1_id'].nunique()
    N_L0S = df_pop['L0_id'].nunique() // N_L1S
    N_KIDS = len(df_pop) // N_L1S // N_L0S
    TOTAL_L1_KIDS = N_L0S * N_KIDS
    
    PERCENTAGES = [0.20, 0.40, 0.60, 0.80, 1.00] # Kept from your OG code
    N_SIMULATIONS = 50  # Hardcoded here to ensure we get variance for error bars!
    
    db_l1 = []  # Stores L1 and L2 tracking
    db_l0 = []  # Stores L0 tracking
    
    for ind in indicators:
        l0_col = 'L0_haz' if ind == 'Height' else 'L0_waz'
        l1_col = 'L1_haz' if ind == 'Height' else 'L1_waz'
        l2_col = 'L2_haz' if ind == 'Height' else 'L2_waz'
        real_col = 'real_haz' if ind == 'Height' else 'real_waz'

        # 1. GROUND TRUTH (Calculate True MAEs once per universe)
        god_l1 = calculate_metrics(df_pop, ['L1_id'], l0_col, real_col).rename(columns={'MAE': 'True_L1_MAE'})
        god_l0 = calculate_metrics(df_pop, ['L1_id', 'L0_id'], l0_col, real_col).rename(columns={'MAE': 'True_L0_MAE'})

        for l1_pct in PERCENTAGES:
            l1_budget = int(TOTAL_L1_KIDS * l1_pct)
            l1_strats = generate_dynamic_strategies(l1_budget, N_L0S, N_KIDS, target_qty=2)
            
            for l1_c, l1_k in l1_strats:
                # --------------------------------------------------------------
                # MONTE CARLO LOOP (Shuffling the data N times)
                # --------------------------------------------------------------
                for sim_id in range(N_SIMULATIONS):
                    
                    # A. Shuffle and Sample L1 Data
                    u_clinics = df_pop[['L1_id', 'L0_id']].drop_duplicates()
                    l1_clinics = u_clinics.sample(frac=1).groupby('L1_id').head(l1_c)
                    df_l1_sheet = df_pop.merge(l1_clinics, on=['L1_id', 'L0_id']).sample(frac=1).groupby(['L1_id', 'L0_id']).head(l1_k)
                    
                    # B. Calculate L1's Measured MAEs
                    l1_diag = calculate_metrics(df_l1_sheet, ['L1_id'], l1_col, l0_col).rename(columns={'MAE': 'Meas_L1_MAE'})
                    l1_vs_l0_clinic = calculate_metrics(df_l1_sheet, ['L1_id', 'L0_id'], l1_col, l0_col).rename(columns={'MAE': 'Meas_L0_MAE'})
                    
                    # C. Save L0 Scorecard (Notice this is outside the L2 loop to save space!)
                    l0_merged = god_l0.merge(l1_vs_l0_clinic, on=['L1_id', 'L0_id'], how='left').fillna(-1) # -1 means not sampled
                    for _, row in l0_merged.iterrows():
                        db_l0.append({
                            'Indicator': ind,
                            'Sim_ID': sim_id,
                            'L1_Budget_Pct': f"{int(round(l1_pct*100))}%",
                            'L1_Label': f"{l1_c}C x {l1_k}K",
                            'L1_id': row['L1_id'], 'L0_id': row['L0_id'],
                            'True_L0_MAE': row['True_L0_MAE'],
                            'Meas_L0_MAE': row['Meas_L0_MAE']
                        })
                    
                    # D. Sample L2 Data (Nested strictly inside the L1 Sheet)
                    for l2_pct in PERCENTAGES:
                        l2_budget = int((l1_c * l1_k) * l2_pct)
                        l2_strats = generate_dynamic_strategies(l2_budget, l1_c, l1_k, target_qty=2)
                        
                        for l2_c, l2_k in l2_strats:
                            # L2 ONLY samples from df_l1_sheet!
                            u_l2_clinics = df_l1_sheet[['L1_id', 'L0_id']].drop_duplicates()
                            l2_clinics = u_l2_clinics.sample(frac=1).groupby('L1_id').head(l2_c)
                            df_l2_audit = df_l1_sheet.merge(l2_clinics, on=['L1_id', 'L0_id']).sample(frac=1).groupby(['L1_id', 'L0_id']).head(l2_k)
                            
                            # E. Calculate L2's Measured MAE
                            l2_vs_l1 = calculate_metrics(df_l2_audit, ['L1_id'], l2_col, l1_col).rename(columns={'MAE': 'Meas_L2_MAE'})
                            
                            # F. Save L1 & L2 Scorecard
                            l1_merged = god_l1.merge(l1_diag, on='L1_id', how='left').merge(l2_vs_l1, on='L1_id', how='left').fillna(-1)
                            for _, row in l1_merged.iterrows():
                                db_l1.append({
                                    'Indicator': ind,
                                    'Sim_ID': sim_id,
                                    'L1_Budget_Pct': f"{int(round(l1_pct*100))}%",
                                    'L1_Label': f"{l1_c}C x {l1_k}K",
                                    'L2_Budget_Pct': f"{int(round(l2_pct*100))}%",
                                    'L2_Label': f"{l2_c}C x {l2_k}K",
                                    'L1_id': row['L1_id'],
                                    'True_L1_MAE': row['True_L1_MAE'],
                                    'Meas_L1_MAE': row['Meas_L1_MAE'],
                                    'Meas_L2_MAE': row['Meas_L2_MAE']
                                })
                        
    # Save the raw scorecard files
    df_l1_out = pd.DataFrame(db_l1)
    df_l0_out = pd.DataFrame(db_l0)
    
    export_path_l1 = os.path.join(output_dir, f"result_L1_{task_id}.csv")
    export_path_l0 = os.path.join(output_dir, f"result_L0_{task_id}.csv")
    
    df_l1_out.to_csv(export_path_l1, index=False)
    df_l0_out.to_csv(export_path_l0, index=False)
    
    return export_path_l1, export_path_l0