import os  # Provides tools to create directories and resolve system file paths
import numpy as np  # Handles high-performance numerical array operations and absolute math
import pandas as pd  # Core data manipulation library used to group, sample, and merge datasets
from scipy.stats import sem  # Specifically imported to compute the Standard Error of the Mean (SE)


def compute_error_metrics(df, group_cols, col_a, col_b, prefix):
    """
    Helper function that isolates two columns, calculates their absolute errors,
    and aggregates them strictly into Mean Absolute Error (MAE) for max speed.
    """
    temp = df[group_cols + [col_a, col_b]].copy()
    temp['_err'] = np.abs(temp[col_a] - temp[col_b])
    
    # Run high-speed pandas group aggregation to extract ONLY the MAE dimension
    agg = temp.groupby(group_cols)['_err'].agg(
        mae='mean'    # Mean Absolute Error (Average size of the lie/discrepancy)
    ).reset_index()
    
    # Rename column dynamically
    agg.rename(columns={
        'mae': f'{prefix}_MAE'
    }, inplace=True)
    
    return agg
# def compute_error_metrics(df, group_cols, col_a, col_b, prefix):
#     """
#     Helper function that isolates two columns, calculates their absolute errors,
#     and aggregates them into Mean(MAE), Variance, SD, and SE grouped by target columns.
#     """
#     # Create a lightweight, isolated copy of the dataframe containing only what we need to save memory
#     temp = df[group_cols + [col_a, col_b]].copy()
    
#     # Calculate the element-wise absolute difference between the two data entries
#     temp['_err'] = np.abs(temp[col_a] - temp[col_b])
    
#     # Run high-speed pandas group aggregation to extract the 4 required statistical dimensions
#     agg = temp.groupby(group_cols)['_err'].agg(
#         mae='mean',    # Mean Absolute Error (Average size of the lie/discrepancy)
#         var='var',     # Variance (Measures the volatility/spread of data errors)
#         sd='std',      # Standard Deviation (Shows how much errors vary from the mean)
#         se=sem         # Standard Error (Quantifies the statistical confidence of our sample)
#     ).reset_index()
    
#     # Rename columns dynamically using the prefix to prevent overlap collisions during final merges
#     agg.rename(columns={
#         'mae': f'{prefix}_MAE',
#         'var': f'{prefix}_Var',
#         'sd': f'{prefix}_SD',
#         'se': f'{prefix}_SE'
#     }, inplace=True)
    
#     return agg  # Return the clean, aggregated statistical summary table
# import numpy as np

# import numpy as np

def generate_dynamic_strategies(budget, max_c, max_k, tolerance=0.05, max_qty=10):
    """
    Generates max 10 combinations of Clinics (c) and Kids (k).
    Prioritizes exact budget matches, falls back to a +/- 5% buffer.
    Sorted from high clinics/low kids to low clinics/high kids.
    """
    strats = []
    
    # 1. Calculate the 5% buffer and round UP to the nearest whole child
    # Example: 5% of 19 kids = 0.95 kids -> Rounds up to 1 child buffer
    buffer = int(np.ceil(budget * tolerance))
    
    min_budget = budget - buffer
    max_budget = budget + buffer
    
    # 2. Brute-force scan every possible combination
    for c in range(1, max_c + 1):
        for k in range(1, max_k + 1):
            total_kids = c * k
            
            # If it falls within our acceptable buffer range
            if min_budget <= total_kids <= max_budget:
                # Calculate how far off this is from the exact budget
                distance_from_target = abs(total_kids - budget)
                strats.append((distance_from_target, c, k))
                
    # 3. If NO combinations fit inside the buffer, return an empty list 
    # (No minimum criteria, we accept 0 if constraints are impossible)
    if not strats:
        return []
        
    # 4. PRIORITIZE exact matches: Sort primarily by distance from target budget
    strats.sort(key=lambda x: x[0]) 
    
    # 5. Cap at the Maximum allowed combinations (10)
    top_strats = strats[:max_qty]
    
    # 6. Strip away the distance metric, leaving only (Clinics, Kids)
    final_strats = [(x[1], x[2]) for x in top_strats]
    
    # 7. Sort Breadth to Depth: High Clinics -> Low Clinics
    final_strats.sort(key=lambda x: x[0], reverse=True)
    
    return final_strats

def run_tracer_engine(df_pop, task_id, output_dir, n_simulations=1, indicators=["Height", "Weight"]) -> tuple[str, str]:
    """
    Core Step 2 Math Engine. Sweeps through nested strategies, maps population parameters against 
    sampled budget allocations, and exports the final decoupled L0 and L1 scorecards.
    """
    print(f"   -> [Step 2] Executing Math Matrix Engine for Task: {task_id}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    N_L1S = df_pop['L1_id'].nunique()
    first_l1 = df_pop['L1_id'].iloc[0]
    N_L0S = df_pop[df_pop['L1_id'] == first_l1]['L0_id'].nunique()
    N_KIDS = len(df_pop) // (N_L1S * N_L0S)
    TOTAL_L1_KIDS = N_L0S * N_KIDS
    
    PERCENTAGES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    db_l0 = []
    db_l1 = []
    
    for ind in indicators:
        print(f"      * Processing Indicator: {ind}...")
        
        ind_db_l0 = []  
        ind_db_l1 = []  
        
        l0_col = 'L0_haz' if ind == 'Height' else ('L0_waz' if ind == 'Weight' else 'L0_whz')
        l1_col = 'L1_haz' if ind == 'Height' else ('L1_waz' if ind == 'Weight' else 'L1_whz')
        l2_col = 'L2_haz' if ind == 'Height' else ('L2_waz' if ind == 'Weight' else 'L2_whz')
        real_col = 'real_haz' if ind == 'Height' else ('real_waz' if ind == 'Weight' else 'real_whz')

        # --------------------------------------------------------------------------
        # SECTION A: MICRO LEVEL POPULATION TRUTHS (No Slicer - Entire Population)
        # --------------------------------------------------------------------------
        p_l0_r = compute_error_metrics(df_pop, ['L1_id', 'L0_id'], l0_col, real_col, 'Pop_L0_Real')
        p_l1_r = compute_error_metrics(df_pop, ['L1_id', 'L0_id'], l1_col, real_col, 'Pop_L1_Real')
        p_l2_r = compute_error_metrics(df_pop, ['L1_id', 'L0_id'], l2_col, real_col, 'Pop_L2_Real')
        p_l1_l0 = compute_error_metrics(df_pop, ['L1_id', 'L0_id'], l1_col, l0_col, 'Pop_L1_L0')
        p_l2_l1 = compute_error_metrics(df_pop, ['L1_id', 'L0_id'], l2_col, l1_col, 'Pop_L2_L1')
        p_l2_l0 = compute_error_metrics(df_pop, ['L1_id', 'L0_id'], l2_col, l0_col, 'Pop_L2_L0')
        
        pop_micro_base = p_l0_r.merge(p_l1_r).merge(p_l2_r).merge(p_l1_l0).merge(p_l2_l1).merge(p_l2_l0)

        # --------------------------------------------------------------------------
        # SECTION B: MACRO LEVEL POPULATION TRUTHS (No Slicer - District Pool)
        # --------------------------------------------------------------------------
        p_dist_l0_r = compute_error_metrics(df_pop, ['L1_id'], l0_col, real_col, 'Pop_Dist_L0_Real')
        p_dist_l1_r = compute_error_metrics(df_pop, ['L1_id'], l1_col, real_col, 'Pop_Dist_L1_Real')
        p_dist_l2_r = compute_error_metrics(df_pop, ['L1_id'], l2_col, real_col, 'Pop_Dist_L2_Real')
        p_dist_l1_l0 = compute_error_metrics(df_pop, ['L1_id'], l1_col, l0_col, 'Pop_Dist_L1_L0')
        p_dist_l2_l1 = compute_error_metrics(df_pop, ['L1_id'], l2_col, l1_col, 'Pop_Dist_L2_L1')
        p_dist_l2_l0 = compute_error_metrics(df_pop, ['L1_id'], l2_col, l0_col, 'Pop_Dist_L2_L0')
        
        pop_macro_base = p_dist_l0_r.merge(p_dist_l1_r).merge(p_dist_l2_r).merge(p_dist_l1_l0).merge(p_dist_l2_l1).merge(p_dist_l2_l0)

        # --------------------------------------------------------------------------
        # SECTION C: NESTED SAMPLING SIMULATION ENGINE
        # --------------------------------------------------------------------------
        for l1_pct in PERCENTAGES:
            l1_budget = int(np.round(TOTAL_L1_KIDS * l1_pct))
            l1_strats = generate_dynamic_strategies(l1_budget, N_L0S, N_KIDS, tolerance=0.05, max_qty=10)
            
            for l1_c, l1_k in l1_strats:
                for sim_id in range(n_simulations):
                    np.random.seed(sim_id)
                    
                    u_clinics = df_pop[['L1_id', 'L0_id']].drop_duplicates()
                    sampled_l1_clinics = u_clinics.groupby('L1_id').sample(n=l1_c, replace=False)
                    
                    df_l1_sampled = df_pop.merge(sampled_l1_clinics, on=['L1_id', 'L0_id'])
                    df_l1_sheet = df_l1_sampled.groupby(['L1_id', 'L0_id']).sample(n=l1_k, replace=False)
                    
                    s_l0_r = compute_error_metrics(df_l1_sheet, ['L1_id', 'L0_id'], l0_col, real_col, 'Samp_L0_Real')
                    s_l1_r = compute_error_metrics(df_l1_sheet, ['L1_id', 'L0_id'], l1_col, real_col, 'Samp_L1_Real')
                    s_l1_l0 = compute_error_metrics(df_l1_sheet, ['L1_id', 'L0_id'], l1_col, l0_col, 'Samp_L1_L0')
                    l0_samp_merged = s_l0_r.merge(s_l1_r).merge(s_l1_l0)
                    
                    s_dist_l0_r = compute_error_metrics(df_l1_sheet, ['L1_id'], l0_col, real_col, 'Samp_Dist_L0_Real')
                    s_dist_l1_r = compute_error_metrics(df_l1_sheet, ['L1_id'], l1_col, real_col, 'Samp_Dist_L1_Real')
                    s_dist_l1_l0 = compute_error_metrics(df_l1_sheet, ['L1_id'], l1_col, l0_col, 'Samp_Dist_L1_L0')
                    l1_samp_macro = s_dist_l0_r.merge(s_dist_l1_r).merge(s_dist_l1_l0)

                    for l2_pct in PERCENTAGES:
                        l2_budget = max(1, int(np.round((l1_c * l1_k) * l2_pct)))
                        l2_strats = generate_dynamic_strategies(l2_budget, l1_c, l1_k, tolerance=0.05, max_qty=10)
                        
                        for l2_c, l2_k in l2_strats:
                            np.random.seed(sim_id + 500)
                            
                            u_l2_clinics = df_l1_sheet[['L1_id', 'L0_id']].drop_duplicates()
                            sampled_l2_clinics = u_l2_clinics.groupby('L1_id').sample(n=l2_c, replace=False)
                            
                            df_l2_filtered = df_l1_sheet.merge(sampled_l2_clinics, on=['L1_id', 'L0_id'])
                            df_l2_audit = df_l2_filtered.groupby(['L1_id', 'L0_id']).sample(n=l2_k, replace=False)
                            
                            s_l2_r = compute_error_metrics(df_l2_audit, ['L1_id', 'L0_id'], l2_col, real_col, 'Samp_L2_Real')
                            s_l2_l1 = compute_error_metrics(df_l2_audit, ['L1_id', 'L0_id'], l2_col, l1_col, 'Samp_L2_L1')
                            s_l2_l0 = compute_error_metrics(df_l2_audit, ['L1_id', 'L0_id'], l2_col, l0_col, 'Samp_L2_L0')
                            l0_audit_merged = s_l2_r.merge(s_l2_l1).merge(s_l2_l0)
                            
                            s_dist_l2_r = compute_error_metrics(df_l2_audit, ['L1_id'], l2_col, real_col, 'Samp_Dist_L2_Real')
                            s_dist_l2_l1 = compute_error_metrics(df_l2_audit, ['L1_id'], l2_col, l1_col, 'Samp_Dist_L2_L1')
                            s_dist_l2_l0 = compute_error_metrics(df_l2_audit, ['L1_id'], l2_col, l0_col, 'Samp_Dist_L2_L0')
                            l1_audit_macro = s_dist_l2_r.merge(s_dist_l2_l1).merge(s_dist_l2_l0)

                            # ----------------------------------------------------------------------
                            # COMPILATION: BUILD FILE 1 (THE L0 CLINIC-LEVEL SCORECARD)
                            # ----------------------------------------------------------------------
                            # 1. Merge sampled data
                            l0_samp = l0_samp_merged.merge(l0_audit_merged, on=['L1_id', 'L0_id'], how='left')
                            # 2. FIX: Merge against full population so unvisited clinics are restored as blanks (NaN)
                            l0_block = pop_micro_base[['L1_id', 'L0_id']].merge(l0_samp, on=['L1_id', 'L0_id'], how='left')
                            
                            # 3. Assign labels to ALL rows in the block so grouping works perfectly in Step 3
                            l0_block['Indicator'] = ind
                            l0_block['Sim_ID'] = sim_id
                            l0_block['L1_Budget_Pct'] = l1_pct
                            l0_block['L1_Total_Kids_Sampled'] = l1_k
                            l0_block['L1_Label'] = f"{l1_c}C_x_{l1_k}K"
                            l0_block['L2_Budget_Pct'] = l2_pct
                            l0_block['L2_Total_Kids_Sampled'] = l2_k
                            l0_block['L2_Label'] = f"{l2_c}C_x_{l2_k}K"
                            
                            ind_db_l0.append(l0_block)

                            # ----------------------------------------------------------------------
                            # COMPILATION: BUILD FILE 2 (THE L1 SUPERVISOR-LEVEL SCORECARD)
                            # ----------------------------------------------------------------------
                            # 1. Merge sampled macro data
                            l1_samp = l1_samp_macro.merge(l1_audit_macro, on=['L1_id'], how='left')
                            # 2. FIX: Merge against full L1 list to restore unaudited supervisors as blanks
                            l1_block = pop_macro_base[['L1_id']].merge(l1_samp, on=['L1_id'], how='left')
                            
                            # 3. Assign labels to ALL rows
                            l1_block['Indicator'] = ind
                            l1_block['Sim_ID'] = sim_id
                            l1_block['L1_Budget_Pct'] = l1_pct
                            l1_block['L1_District_Total_Kids_Sampled'] = (l1_c * l1_k)
                            l1_block['L1_Label'] = f"{l1_c}C_x_{l1_k}K"
                            l1_block['L2_Budget_Pct'] = l2_pct
                            l1_block['L2_District_Total_Kids_Sampled'] = (l2_c * l2_k)
                            l1_block['L2_Label'] = f"{l2_c}C_x_{l2_k}K"
                            
                            ind_db_l1.append(l1_block)

        # ======================================================================
        # VECTORIZED POST-PROCESSING (Runs exactly ONCE per indicator)
        # ======================================================================
        if ind_db_l0:
            ind_df_l0 = pd.concat(ind_db_l0, ignore_index=True)
            ind_df_l0 = ind_df_l0.merge(pop_micro_base, on=['L1_id', 'L0_id'], how='left')
            ind_df_l0['Gap_Samp_vs_Pop_L0_Real'] = np.abs(ind_df_l0['Samp_L0_Real_MAE'] - ind_df_l0['Pop_L0_Real_MAE'])
            ind_df_l0['Gap_Samp_vs_Pop_L1_L0'] = np.abs(ind_df_l0['Samp_L1_L0_MAE'] - ind_df_l0['Pop_L1_L0_MAE'])
            db_l0.append(ind_df_l0)
            
        if ind_db_l1:
            ind_df_l1 = pd.concat(ind_db_l1, ignore_index=True)
            ind_df_l1 = ind_df_l1.merge(pop_macro_base, on=['L1_id'], how='left')
            db_l1.append(ind_df_l1)

    # --------------------------------------------------------------------------
    # SECTION D: FILE STORAGE EXPORTS (PARQUET SPECIFICATIONS)
    # --------------------------------------------------------------------------
    out_df_l0 = pd.concat(db_l0, ignore_index=True)
    out_df_l1 = pd.concat(db_l1, ignore_index=True)
    
    export_path_l0 = os.path.join(output_dir, f"calculated_metrics_L0_{task_id}.parquet")
    export_path_l1 = os.path.join(output_dir, f"calculated_metrics_L1_{task_id}.parquet")
    
    out_df_l0.to_parquet(export_path_l0, engine='pyarrow', index=False)
    out_df_l1.to_parquet(export_path_l1, engine='pyarrow', index=False)
    
    print(f"   -> [Step 2] Complete. Matrix Files Extracted to:\n      1. {export_path_l0}\n      2. {export_path_l1}")
    return export_path_l0, export_path_l1

