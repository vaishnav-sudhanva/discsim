import os  # Used to construct file paths dynamically based on your operating system
import numpy as np  # High-performance math library used for strict threshold rounding
import pandas as pd  # Core dataframe engine to group, sort, and merge the metrics
from tqdm.auto import tqdm # Progress bar to track large Monte Carlo ranking loops

# ==============================================================================
# CORE FUNCTION: THE OVERLAP CALCULATOR (THE CATCH RATE)
# ==============================================================================
# HOW: Added 'precalc_truth' and 'precalc_k' optional arguments to skip redundant math
def calculate_overlap_percentage(group_df, target_pct, id_cols, truth_sort_col, sample_sort_col, precalc_truth=None, precalc_k=None):
    """
    Calculates how successfully a Field Worker's physical sample caught the True Worst fraudsters.
    FIXED: Enforces the "Penalty of Ignorance" and accepts CPU Memory Caching to skip redundant math.
    """
    # 1. Prevent ID Collisions: If we are tracking nested IDs (L1 + L0), merge them into a single unique string
    group_df = group_df.copy() # Safe copy to prevent pandas warnings
    if isinstance(id_cols, list):
        group_df['_uid'] = group_df[id_cols].astype(str).agg('_'.join, axis=1)
        uid_col = '_uid'
    else:
        uid_col = id_cols
        
    # --------------------------------------------------------------------------
    # STEP A: THE ABSOLUTE TRUTH (Calculated on the GLOBAL Baseline)
    # --------------------------------------------------------------------------
    # WHAT: If the main loop passes us a cached answer key, bypass the sorting entirely!
    if precalc_truth is not None and precalc_k is not None:
        true_worst_set = precalc_truth
        k_threshold = precalc_k
    else:
        # Fallback to standard calculation if no cache is provided (Used natively by V3)
        total_units_global = group_df[uid_col].nunique()
        k_threshold = max(1, int(np.round(total_units_global * target_pct)))
        
        # Drop row duplicates for the same entity before running the head filter.
        dedup_global = group_df.drop_duplicates(subset=[uid_col])
        true_worst_df = dedup_global.sort_values(by=[truth_sort_col, uid_col], ascending=[False, True]).head(k_threshold)
        true_worst_set = set(true_worst_df[uid_col]) 
    
    # --------------------------------------------------------------------------
    # STEP B: THE FIELD WORKER's GUESS (Calculated strictly on the SAMPLE)
    # --------------------------------------------------------------------------
    # Now that we know the Truth, we drop the clinics/supervisors the field worker NEVER visited.
    clean_df = group_df.dropna(subset=[sample_sort_col]).copy()
    
    if clean_df.empty:
        return 0.0
        
    # WHAT: Clean duplicates from the field worker's dataset before truncating.
    dedup_sample = clean_df.drop_duplicates(subset=[uid_col])
        
    # Find what the Field Worker thinks are the worst unique units
    caught_worst_df = dedup_sample.sort_values(by=[sample_sort_col, uid_col], ascending=[False, True]).head(k_threshold)
    caught_worst_set = set(caught_worst_df[uid_col])
    
    # --------------------------------------------------------------------------
    # STEP C: THE GRADE
    # --------------------------------------------------------------------------
    # Calculate exactly how many of the Field Worker's suspects match the True Worst list
    matched_count = len(true_worst_set.intersection(caught_worst_set))
    
    # Return the accuracy as a flat percentage (0 to 100)
    return (matched_count / k_threshold) * 100.0


# ==============================================================================
# MAIN ENGINE: THE RANKING PROCESSOR
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
    
    clinic_summary_records = []
    supervisor_summary_records = []
    
    # WHAT: Create distinct slice keys for the decoupled datasets.
    # WHY: df_l0 was optimized in Step 2 and no longer carries redundant L2 columns.
    slice_cols_l0 = ['Indicator', 'Sim_ID', 'L1_Budget_Pct', 'L1_Label']
    slice_cols_l1 = ['Indicator', 'Sim_ID', 'L1_Budget_Pct', 'L1_Label', 'L2_Budget_Pct', 'L2_Label']
    
    # ==========================================================================
    # EVALUATION BLOCK 1: CLINIC DETECTION RATES (V2: L1 catching L0)
    # ==========================================================================
    print("      * Pre-calculating God-Mode Answer Keys for Clinics (V2)...")
    l0_truth_df = df_l0[df_l0['L1_Budget_Pct'] == 1.0].copy()
    god_l0_answers = {}
    
    # Pre-build the absolute truth dictionary to skip sorting during the loop
    for ind_t in df_l0['Indicator'].unique():
        god_l0_answers[ind_t] = {}
        ind_df = l0_truth_df[l0_truth_df['Indicator'] == ind_t]
        for pct_t in target_percentiles:
            god_l0_answers[ind_t][pct_t] = {}
            for sim_t, sim_df in ind_df.groupby('Sim_ID'):
                god_l0_answers[ind_t][pct_t][sim_t] = {}
                for l1_id, l1_group in sim_df.groupby('L1_id'):
                    dedup = l1_group.drop_duplicates(subset=['L0_id'])
                    k_targ = max(1, int(np.round(len(dedup) * pct_t)))
                    t_set = set(dedup.sort_values(by=['Pop_L0_Real_MAE', 'L0_id'], ascending=[False, True]).head(k_targ)['L0_id'])
                    god_l0_answers[ind_t][pct_t][sim_t][l1_id] = {'k': k_targ, 'truth_set': t_set}

    print("      * Evaluating Clinic (L0) catching accuracies (V2)...")
    
    for keys, group in tqdm(df_l0.groupby(slice_cols_l0), desc="L0 (Clinic) Analysis", leave=False):
        ind, sim, l1_b, l1_lbl = keys 
        
        for pct in target_percentiles:
            v2_acc_list = []
            
            # MEMORY FIX: Convert Pandas GroupBy to Python Dict for instant data fetching
            l1_groups_dict = dict(tuple(group.groupby('L1_id')))
            
            for l1_id, l1_group in l1_groups_dict.items():
                cached_ans = god_l0_answers[ind][pct][sim].get(l1_id, {})
                
                acc = calculate_overlap_percentage(
                    group_df=l1_group,         
                    target_pct=pct,
                    id_cols='L0_id',           
                    truth_sort_col='Pop_L0_Real_MAE', 
                    sample_sort_col='Samp_L1_L0_MAE',
                    precalc_truth=cached_ans.get('truth_set'), # INJECT V2 CACHE
                    precalc_k=cached_ans.get('k')              # INJECT V2 CACHE
                )
                v2_acc_list.append(acc)
                
            final_v2_acc = np.mean(v2_acc_list) if len(v2_acc_list) > 0 else 0.0
            
            clinic_summary_records.append({
                'Indicator': ind, 'Sim_ID': sim,
                'L1_Budget_Pct': l1_b, 'L1_Label': l1_lbl,
                'Target_Percentile': f"{int(pct*100)}%",
                'V2_MAE_Acc': final_v2_acc
            })
            
    # ==========================================================================
    # EVALUATION BLOCK 2: SUPERVISOR DETECTION RATES (V1 & V3: L2 catching L1)
    # ==========================================================================
    print("      * Pre-calculating God-Mode Answer Keys for Supervisors (V1)...")
    l1_truth_df = df_l1[df_l1['L1_Budget_Pct'] == 1.0].copy()
    god_l1_answers_v1 = {}
    
    # Pre-build the absolute truth dictionary for all Supervisors
    for ind_t in df_l1['Indicator'].unique():
        god_l1_answers_v1[ind_t] = {}
        ind_df = l1_truth_df[l1_truth_df['Indicator'] == ind_t]
        for pct_t in target_percentiles:
            god_l1_answers_v1[ind_t][pct_t] = {}
            for sim_t, sim_df in ind_df.groupby('Sim_ID'):
                dedup = sim_df.drop_duplicates(subset=['L1_id'])
                k_targ = max(1, int(np.round(len(dedup) * pct_t)))
                t_set = set(dedup.sort_values(by=['Pop_Dist_L1_Real_MAE', 'L1_id'], ascending=[False, True]).head(k_targ)['L1_id'])
                god_l1_answers_v1[ind_t][pct_t][sim_t] = {'k': k_targ, 'truth_set': t_set}

    print("      * Evaluating Supervisor (L1) catching accuracies (V1, V3)...")
    
    for keys, group in tqdm(df_l1.groupby(slice_cols_l1), desc="L1 (Supervisor) Analysis", leave=False):
        ind, sim, l1_b, l1_lbl, l2_b, l2_lbl = keys
        
        for pct in target_percentiles:
            
            cached_v1 = god_l1_answers_v1[ind][pct][sim]
            
            v1_acc = calculate_overlap_percentage(
                group_df=group,            
                target_pct=pct,
                id_cols='L1_id',
                truth_sort_col='Pop_Dist_L1_Real_MAE', 
                sample_sort_col='Samp_Dist_L2_L1_MAE',
                precalc_truth=cached_v1['truth_set'], # INJECT V1 CACHE
                precalc_k=cached_v1['k']              # INJECT V1 CACHE
            )
            
            # V3 native calculation (Cannot be cached, as its truth changes dynamically)
            v3_acc = calculate_overlap_percentage(
                group_df=group,
                target_pct=pct,
                id_cols='L1_id',
                truth_sort_col='Samp_Dist_L1_Real_MAE', 
                sample_sort_col='Samp_Dist_L2_L1_MAE'   
            )
            
            supervisor_summary_records.append({
                'Indicator': ind, 'Sim_ID': sim,
                'L1_Budget_Pct': l1_b, 'L1_Label': l1_lbl,
                'L2_Budget_Pct': l2_b, 'L2_Label': l2_lbl,
                'Target_Percentile': f"{int(pct*100)}%",
                'V1_MAE_Acc': v1_acc,
                'V3_MAE_Acc': v3_acc
            })


    # ==========================================================================
    # CONSOLIDATION & STORAGE EXPORTS (PARQUET & CSV)
    # ==========================================================================
    # Convert the raw dictionaries into Pandas Dataframes
    analytics_l0_df = pd.DataFrame(clinic_summary_records)
    analytics_l1_df = pd.DataFrame(supervisor_summary_records)
    
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
    # WHY: Blends the clean clinic records with the granular auditor rows.
    # HOW: Join solely on the mutual structural keys. Pandas automatically broadcasts the V2 values.
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
    scenario_str = task_id.split('_202')[0] # Strips the timestamp to leave just "1_Good_L0_Good_L1", etc.
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