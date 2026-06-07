import os  # Used to construct file paths dynamically based on your operating system
import numpy as np  # High-performance math library used for strict threshold rounding
import pandas as pd  # Core dataframe engine to group, sort, and merge the metrics
from tqdm.auto import tqdm # Progress bar to track large Monte Carlo ranking loops

# ==============================================================================
# MAIN ENGINE: THE RANKING PROCESSOR (VECTORIZED FOR MAX SPEED)
# ==============================================================================
def process_ranking_analytics(l0_parquet_path, l1_parquet_path, output_dir, task_id, target_percentiles=[0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]):
    """
    Step 3 Analytical Ranker Engine. Slices the precalculated data matrices across multiple 
    user-defined thresholds, evaluates detection accuracies, and exports ultra-lightweight tables.
    """
    print(f"   -> [Step 3] Executing Analytics Ranking Engine for Task: {task_id}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print("      * Reading precalculated matrices from Step 2...")
    df_l0 = pd.read_parquet(l0_parquet_path)
    df_l1 = pd.read_parquet(l1_parquet_path)
    
    # Format target_percentiles for output strings mapping (e.g., 0.30 -> "30%")
    pct_strings = {pct: f"{int(pct*100)}%" for pct in target_percentiles}
    
    # ==========================================================================
    # EVALUATION BLOCK 1: CLINIC DETECTION RATES (V2: L1 catching L0)
    # ==========================================================================
    print("      * Evaluating Clinic (L0) catching accuracies (V2) [Vectorized]...")
    
    # 1. TRUTH V2: Pre-sort the global baseline instantly
    df_l0_truth = df_l0[['Indicator', 'Sim_ID', 'L1_id', 'L0_id', 'Pop_L0_Real_MAE']].drop_duplicates()
    df_l0_truth = df_l0_truth.sort_values(['Indicator', 'Sim_ID', 'L1_id', 'Pop_L0_Real_MAE', 'L0_id'], ascending=[True, True, True, False, True])
    truth_l0_dict = df_l0_truth.groupby(['Indicator', 'Sim_ID', 'L1_id'])['L0_id'].apply(list).to_dict()
    
    # 2. SAMPLE V2: Pre-sort the field worker guesses instantly
    df_l0_samp = df_l0.dropna(subset=['Samp_L1_L0_MAE'])[['Indicator', 'Sim_ID', 'L1_Budget_Pct', 'L1_Label', 'L1_id', 'L0_id', 'Samp_L1_L0_MAE']].drop_duplicates()
    df_l0_samp = df_l0_samp.sort_values(['Indicator', 'Sim_ID', 'L1_Budget_Pct', 'L1_Label', 'L1_id', 'Samp_L1_L0_MAE', 'L0_id'], ascending=[True, True, True, True, True, False, True])
    samp_l0_dict = df_l0_samp.groupby(['Indicator', 'Sim_ID', 'L1_Budget_Pct', 'L1_Label', 'L1_id'])['L0_id'].apply(list).to_dict()
    
    overlap_results_l0 = []
    
    # Instantly calculate overlaps via dictionary intersection
    for k_tuple, s_list in samp_l0_dict.items():
        ind, sim, l1_b, l1_lbl, l1_id = k_tuple
        t_list = truth_l0_dict.get((ind, sim, l1_id), [])
        
        n_tot = len(t_list)
        if n_tot == 0 or len(s_list) == 0:
            for pct in target_percentiles:
                overlap_results_l0.append((ind, sim, l1_b, l1_lbl, pct_strings[pct], 0.0))
            continue
            
        for pct in target_percentiles:
            k = max(1, int(np.round(n_tot * pct)))
            t_set = set(t_list[:k])
            s_set = set(s_list[:k])
            acc = len(t_set.intersection(s_set)) / k * 100.0
            overlap_results_l0.append((ind, sim, l1_b, l1_lbl, pct_strings[pct], acc))
            
    res_l0 = pd.DataFrame(overlap_results_l0, columns=['Indicator', 'Sim_ID', 'L1_Budget_Pct', 'L1_Label', 'Target_Percentile', 'V2_MAE_Acc'])
    
    # Mean across L1_id to get the final aggregated metric per slice
    analytics_l0_df = res_l0.groupby(['Indicator', 'Sim_ID', 'L1_Budget_Pct', 'L1_Label', 'Target_Percentile'])['V2_MAE_Acc'].mean().reset_index()
    
    # ==========================================================================
    # EVALUATION BLOCK 2: SUPERVISOR DETECTION RATES (V1 & V3: L2 catching L1)
    # ==========================================================================
    print("      * Evaluating Supervisor (L1) catching accuracies (V1, V3) [Vectorized]...")
    
    # 1. TRUTH V1 (God-Mode): Constant across budgets
    df_l1_truth_v1 = df_l1[['Indicator', 'Sim_ID', 'L1_id', 'Pop_Dist_L1_Real_MAE']].drop_duplicates()
    df_l1_truth_v1 = df_l1_truth_v1.sort_values(['Indicator', 'Sim_ID', 'Pop_Dist_L1_Real_MAE', 'L1_id'], ascending=[True, True, False, True])
    truth_v1_dict = df_l1_truth_v1.groupby(['Indicator', 'Sim_ID'])['L1_id'].apply(list).to_dict()
    
    # 2. TRUTH V3 (Native): Dynamic per L1 Strategy
    df_l1_truth_v3 = df_l1[['Indicator', 'Sim_ID', 'L1_Budget_Pct', 'L1_Label', 'L1_id', 'Samp_Dist_L1_Real_MAE']].drop_duplicates()
    df_l1_truth_v3 = df_l1_truth_v3.sort_values(['Indicator', 'Sim_ID', 'L1_Budget_Pct', 'L1_Label', 'Samp_Dist_L1_Real_MAE', 'L1_id'], ascending=[True, True, True, True, False, True])
    truth_v3_dict = df_l1_truth_v3.groupby(['Indicator', 'Sim_ID', 'L1_Budget_Pct', 'L1_Label'])['L1_id'].apply(list).to_dict()
    
    # 3. SAMPLE V1/V3 (The Auditor's Guess)
    df_l1_samp = df_l1.dropna(subset=['Samp_Dist_L2_L1_MAE'])[['Indicator', 'Sim_ID', 'L1_Budget_Pct', 'L1_Label', 'L2_Budget_Pct', 'L2_Label', 'L1_id', 'Samp_Dist_L2_L1_MAE']].drop_duplicates()
    df_l1_samp = df_l1_samp.sort_values(['Indicator', 'Sim_ID', 'L1_Budget_Pct', 'L1_Label', 'L2_Budget_Pct', 'L2_Label', 'Samp_Dist_L2_L1_MAE', 'L1_id'], ascending=[True, True, True, True, True, True, False, True])
    samp_dict = df_l1_samp.groupby(['Indicator', 'Sim_ID', 'L1_Budget_Pct', 'L1_Label', 'L2_Budget_Pct', 'L2_Label'])['L1_id'].apply(list).to_dict()
    
    overlap_results_l1 = []
    
    for k_tuple, s_list in samp_dict.items():
        ind, sim, l1_b, l1_lbl, l2_b, l2_lbl = k_tuple
        
        t1_list = truth_v1_dict.get((ind, sim), [])
        t3_list = truth_v3_dict.get((ind, sim, l1_b, l1_lbl), [])
        
        n_tot = len(t1_list) # Global baseline (total N_L1S)
        if n_tot == 0 or len(s_list) == 0:
            for pct in target_percentiles:
                overlap_results_l1.append((ind, sim, l1_b, l1_lbl, l2_b, l2_lbl, pct_strings[pct], 0.0, 0.0))
            continue
            
        for pct in target_percentiles:
            k = max(1, int(np.round(n_tot * pct)))
            
            s_set = set(s_list[:k])
            
            # V1 Acc
            t1_set = set(t1_list[:k])
            v1_acc = len(t1_set.intersection(s_set)) / k * 100.0
            
            # V3 Acc
            if len(t3_list) == 0:
                v3_acc = 0.0
            else:
                t3_set = set(t3_list[:k])
                v3_acc = len(t3_set.intersection(s_set)) / k * 100.0
                
            overlap_results_l1.append((ind, sim, l1_b, l1_lbl, l2_b, l2_lbl, pct_strings[pct], v1_acc, v3_acc))
            
    analytics_l1_df = pd.DataFrame(overlap_results_l1, columns=['Indicator', 'Sim_ID', 'L1_Budget_Pct', 'L1_Label', 'L2_Budget_Pct', 'L2_Label', 'Target_Percentile', 'V1_MAE_Acc', 'V3_MAE_Acc'])
    
    # ==========================================================================
    # CONSOLIDATION & STORAGE EXPORTS (PARQUET & CSV)
    # ==========================================================================
    # 1. Export Intermediate Files
    export_path_l0_summary_pq = os.path.join(output_dir, f"analytics_summary_L0_{task_id}.parquet")
    export_path_l1_summary_pq = os.path.join(output_dir, f"analytics_summary_L1_{task_id}.parquet")
    export_path_l0_summary_csv = os.path.join(output_dir, f"analytics_summary_L0_{task_id}.csv")
    export_path_l1_summary_csv = os.path.join(output_dir, f"analytics_summary_L1_{task_id}.csv")

    analytics_l0_df.to_parquet(export_path_l0_summary_pq, engine='pyarrow', index=False)
    analytics_l1_df.to_parquet(export_path_l1_summary_pq, engine='pyarrow', index=False)
    analytics_l0_df.to_csv(export_path_l0_summary_csv, index=False)
    analytics_l1_df.to_csv(export_path_l1_summary_csv, index=False)
    
    # 2. Merge L0 and L1 together to create the Unified "Master Database" for the Dashboard
    final_df = pd.merge(
        analytics_l0_df, analytics_l1_df, 
        on=['Indicator', 'Sim_ID', 'L1_Budget_Pct', 'L1_Label', 'Target_Percentile'],
        how='inner'
    )
    
    # Format labels/budgets for the UI rendering Engine
    final_df['L1_Budget_Pct'] = (final_df['L1_Budget_Pct'] * 100).round().astype(int).astype(str) + "%"
    final_df['L2_Budget_Pct'] = (final_df['L2_Budget_Pct'] * 100).round().astype(int).astype(str) + "%"
    final_df['L1_Label'] = final_df['L1_Label'].str.replace('_', ' ')
    final_df['L2_Label'] = final_df['L2_Label'].str.replace('_', ' ')
    
    # Extract strict integer counts so the UI plotting functions can sort mathematically
    final_df['L1_C'] = final_df['L1_Label'].str.split('C').str[0].astype(int)
    final_df['L1_K'] = final_df['L1_Label'].str.split(' x ').str[1].str.replace('K', '').astype(int)
    final_df['L2_C'] = final_df['L2_Label'].str.split('C').str[0].astype(int)
    final_df['L2_K'] = final_df['L2_Label'].str.split(' x ').str[1].str.replace('K', '').astype(int)

    # Inject legacy Dummy Data and Metadata so the old Streamlit plotting functions don't crash
    for col in ['V1_MAE_Std', 'V1_RMSE_Acc', 'V1_P90_Acc', 'V2_MAE_Std', 'V2_RMSE_Acc', 'V2_P90_Acc', 'V3_MAE_Std', 'V3_RMSE_Acc', 'V3_P90_Acc']:
        final_df[col] = 0.0
        
    # Extract the base scenario logic from the filename string 
    scenario_str = task_id.split('_202')[0] 
    final_df['Scenario'] = scenario_str
    final_df['Universe'] = f"{scenario_str} (rho=0.0)"
    final_df['Rho_Model'] = 0.0
    
    # Export the final Dashboard Engine file
    final_parquet_path = os.path.join(output_dir, f"Tracer_Master_DB_{task_id}.parquet")
    final_csv_path = os.path.join(output_dir, f"Tracer_Master_DB_{task_id}.csv")
    
    # Enforce strict column ordering to prevent pipeline mismatches downstream
    export_cols = [
        'Universe', 'Indicator', 'Sim_ID', 'L1_Budget_Pct', 'L1_C', 'L1_K', 'L1_Label', 
        'L2_Budget_Pct', 'L2_C', 'L2_K', 'L2_Label', 'V1_MAE_Acc', 'V1_MAE_Std', 'V1_RMSE_Acc', 
        'V1_P90_Acc', 'V2_MAE_Acc', 'V2_MAE_Std', 'V2_RMSE_Acc', 'V2_P90_Acc', 'V3_MAE_Acc', 
        'V3_MAE_Std', 'V3_RMSE_Acc', 'V3_P90_Acc', 'Scenario', 'Rho_Model', 'Target_Percentile'
    ]
    final_df = final_df[export_cols]

    final_df.to_parquet(final_parquet_path, index=False)
    final_df.to_csv(final_csv_path, index=False)
    
    print(f"   -> [Step 3] Complete. Unified Final DB Exported to:")
    print(f"      - {final_parquet_path}")
    print(f"      - {final_csv_path}")
    
    return final_csv_path, final_parquet_path