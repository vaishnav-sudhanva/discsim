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

def generate_dynamic_strategies(budget, max_c, max_k, target_qty=2):
    """
    Takes a child budget constraint and identifies the ideal whole-number 
    combinations of Clinics (C) and Kids (K) that fit perfectly inside it.
    """
    strats = []  # Initialize an empty array to house our valid strategy tuples
    
    # Determine the lower bound for kids per clinic so we don't exceed max allowed clinics
    min_k = max(1, int(np.floor(budget / max_c)))
    
    # Establish the absolute ceiling of kids we can pull based on the current budget
    max_pos_k = min(max_k, budget)
    
    # Safety check: if constraints are impossible, fall back to the absolute max capacity available
    if min_k > max_pos_k: 
        return [(max_c, max_k)]
        
    # Generate an evenly spaced linear grid of whole-number options for kids (K)
    raw_ks = np.unique(np.round(np.linspace(min_k, max_pos_k, target_qty * 2)).astype(int))
    
    # Loop through the candidate kid counts to solve for matching clinic counts (C)
    for k in raw_ks:
        c = int(np.round(budget / k))  # Round the clinic count to a strict whole-number integer
        # Validate that the solved strategy respects both local capacity and the global budget
        if 1 <= c <= max_c and c * k <= max_c * max_k:
            if (c, k) not in strats: 
                strats.append((c, k))  # Log the unique layout if it passes all validations
                
    # Sort strategies from maximum clinic coverage to deepest clinic sampling depth
    strats = sorted(strats, key=lambda x: x[0], reverse=True)
    
    # Downsample the strategy array if it contains more items than our target dashboard limit
    if len(strats) > target_qty: 
        strats = [strats[i] for i in np.round(np.linspace(0, len(strats)-1, target_qty)).astype(int)]
        
    # Pad the list using duplicates of the last element if we fall short of our desired layout options
    while 0 < len(strats) < target_qty: 
        strats.append(strats[-1])
        
    return strats  # Return the array of validated (Clinics, Kids) strategy pairs

#def run_tracer_engine(df_pop, task_id, output_dir, n_simulations=1, indicators=["Height", "Weight"]): 
def run_tracer_engine(df_pop, task_id, output_dir, n_simulations=1, indicators=["Height", "Weight"]) -> tuple[str, str]:
    """
    Core Step 2 Math Engine. Sweeps through nested strategies, maps population parameters against 
    sampled budget allocations, and exports the final decoupled L0 and L1 scorecards.
    """
    print(f"   -> [Step 2] Executing Math Matrix Engine for Task: {task_id}")
    
    # Ensure the target folder exists on the operating system to prevent write failures
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Extract total number of unique supervisor regions present in the synthetic universe
    N_L1S = df_pop['L1_id'].nunique()
    
    # Grab the first available supervisor key to determine layout properties safely
    first_l1 = df_pop['L1_id'].iloc[0]
    
    # Calculate the fixed count of clinics assigned underneath a single supervisor area
    N_L0S = df_pop[df_pop['L1_id'] == first_l1]['L0_id'].nunique()
    
    # Derive the baseline population count of children mapped to each individual clinic
    N_KIDS = len(df_pop) // (N_L1S * N_L0S)
    
    # Calculate the total pool of children registered inside an entire supervisor's district
    TOTAL_L1_KIDS = N_L0S * N_KIDS
    
    # 10-Step grid representing operational capacities from 10% to 100% workloads
    PERCENTAGES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    # Initialize empty master registries to aggregate data inside our execution loops
    db_l0 = []
    db_l1 = []
    
    # Begin iteration across the biological metrics (Height/HAZ, Weight/WAZ, Wasting/WHZ)
    for ind in indicators:
        print(f"      * Processing Indicator: {ind}...")
        
        ind_db_l0 = []  # Temp fast-storage for L0
        ind_db_l1 = []  # Temp fast-storage for L1
        
        
    #     # Maps user-facing flags to the precise column string configurations built in Step 1
        l0_col = 'L0_haz' if ind == 'Height' else ('L0_waz' if ind == 'Weight' else 'L0_whz')
        l1_col = 'L1_haz' if ind == 'Height' else ('L1_waz' if ind == 'Weight' else 'L1_whz')
        l2_col = 'L2_haz' if ind == 'Height' else ('L2_waz' if ind == 'Weight' else 'L2_whz')
        real_col = 'real_haz' if ind == 'Height' else ('real_waz' if ind == 'Weight' else 'real_whz')

        # --------------------------------------------------------------------------
        # SECTION A: MICRO LEVEL POPULATION TRUTHS (No Slicer - Entire Population)
        # --------------------------------------------------------------------------
        # Run baseline metrics for all six pairwise combinations utilizing our helper function
        p_l0_r = compute_error_metrics(df_pop, ['L1_id', 'L0_id'], l0_col, real_col, 'Pop_L0_Real')
        p_l1_r = compute_error_metrics(df_pop, ['L1_id', 'L0_id'], l1_col, real_col, 'Pop_L1_Real')
        p_l2_r = compute_error_metrics(df_pop, ['L1_id', 'L0_id'], l2_col, real_col, 'Pop_L2_Real')
        p_l1_l0 = compute_error_metrics(df_pop, ['L1_id', 'L0_id'], l1_col, l0_col, 'Pop_L1_L0')
        p_l2_l1 = compute_error_metrics(df_pop, ['L1_id', 'L0_id'], l2_col, l1_col, 'Pop_L2_L1')
        p_l2_l0 = compute_error_metrics(df_pop, ['L1_id', 'L0_id'], l2_col, l0_col, 'Pop_L2_L0')
        
        # Merge the local population arrays sequentially using consecutive inner joins
        pop_micro_base = p_l0_r.merge(p_l1_r).merge(p_l2_r).merge(p_l1_l0).merge(p_l2_l1).merge(p_l2_l0)

        # --------------------------------------------------------------------------
        # SECTION B: MACRO LEVEL POPULATION TRUTHS (No Slicer - District Pool)
        # --------------------------------------------------------------------------
        # Group strictly by L1_id to calculate macro district-wide baseline true errors
        p_dist_l0_r = compute_error_metrics(df_pop, ['L1_id'], l0_col, real_col, 'Pop_Dist_L0_Real')
        p_dist_l1_r = compute_error_metrics(df_pop, ['L1_id'], l1_col, real_col, 'Pop_Dist_L1_Real')
        p_dist_l2_r = compute_error_metrics(df_pop, ['L1_id'], l2_col, real_col, 'Pop_Dist_L2_Real')
        p_dist_l1_l0 = compute_error_metrics(df_pop, ['L1_id'], l1_col, l0_col, 'Pop_Dist_L1_L0')
        p_dist_l2_l1 = compute_error_metrics(df_pop, ['L1_id'], l2_col, l1_col, 'Pop_Dist_L2_L1')
        p_dist_l2_l0 = compute_error_metrics(df_pop, ['L1_id'], l2_col, l0_col, 'Pop_Dist_L2_L0')
        
        # Merge the macro district-wide metrics together into a single broadcastable master rows block
        pop_macro_base = p_dist_l0_r.merge(p_dist_l1_r).merge(p_dist_l2_r).merge(p_dist_l1_l0).merge(p_dist_l2_l1).merge(p_dist_l2_l0)

        # --------------------------------------------------------------------------
        # SECTION C: NESTED SAMPLING SIMULATION ENGINE
        # --------------------------------------------------------------------------
        for l1_pct in PERCENTAGES:
            # Solve exact whole integer counts for the L1 supervisor allocation budget
            l1_budget = int(np.round(TOTAL_L1_KIDS * l1_pct))
            l1_strats = generate_dynamic_strategies(l1_budget, N_L0S, N_KIDS, target_qty=2)
            
            for l1_c, l1_k in l1_strats:
                for sim_id in range(n_simulations):
                    # Set independent seeds matching the current loop index to isolate runs cleanly
                    np.random.seed(sim_id)
                    
                    # Randomly determine which clinics (L0) inside each supervisor district get visited
                    u_clinics = df_pop[['L1_id', 'L0_id']].drop_duplicates()
                    sampled_l1_clinics = u_clinics.groupby('L1_id').sample(n=l1_c, replace=False)
                    
                    # Randomly isolate K children out of the selected clinics to form the field spreadsheet
                    df_l1_sampled = df_pop.merge(sampled_l1_clinics, on=['L1_id', 'L0_id'])
                    df_l1_sheet = df_l1_sampled.groupby(['L1_id', 'L0_id']).sample(n=l1_k, replace=False)
                    
                    # Calculate micro (clinic) metrics inside L1's sampled data framework
                    s_l0_r = compute_error_metrics(df_l1_sheet, ['L1_id', 'L0_id'], l0_col, real_col, 'Samp_L0_Real')
                    s_l1_r = compute_error_metrics(df_l1_sheet, ['L1_id', 'L0_id'], l1_col, real_col, 'Samp_L1_Real')
                    s_l1_l0 = compute_error_metrics(df_l1_sheet, ['L1_id', 'L0_id'], l1_col, l0_col, 'Samp_L1_L0')
                    l0_samp_merged = s_l0_r.merge(s_l1_r).merge(s_l1_l0)
                    
                    # Calculate macro (district pooled) metrics across all kids this supervisor inspected
                    s_dist_l0_r = compute_error_metrics(df_l1_sheet, ['L1_id'], l0_col, real_col, 'Samp_Dist_L0_Real')
                    s_dist_l1_r = compute_error_metrics(df_l1_sheet, ['L1_id'], l1_col, real_col, 'Samp_Dist_L1_Real')
                    s_dist_l1_l0 = compute_error_metrics(df_l1_sheet, ['L1_id'], l1_col, l0_col, 'Samp_Dist_L1_L0')
                    l1_samp_macro = s_dist_l0_r.merge(s_dist_l1_r).merge(s_dist_l1_l0)

                    # Trigger the independent inner loop representing the L2 Audit level
                    for l2_pct in PERCENTAGES:
                        # Solve exact budget numbers relative to the total scale of L1's current field sheets
                        l2_budget = max(1, int(np.round((l1_c * l1_k) * l2_pct)))
                        l2_strats = generate_dynamic_strategies(l2_budget, l1_c, l1_k, target_qty=2)
                        
                        for l2_c, l2_k in l2_strats:
                            # Adjust seed space safely away from L1 parameters to maintain true independence
                            np.random.seed(sim_id + 500)
                            
                            # Auditor selects a subset of clinics from the list the supervisor visited
                            u_l2_clinics = df_l1_sheet[['L1_id', 'L0_id']].drop_duplicates()
                            sampled_l2_clinics = u_l2_clinics.groupby('L1_id').sample(n=l2_c, replace=False)
                            
                            # Auditor samples kids from within those audited clinics
                            df_l2_filtered = df_l1_sheet.merge(sampled_l2_clinics, on=['L1_id', 'L0_id'])
                            df_l2_audit = df_l2_filtered.groupby(['L1_id', 'L0_id']).sample(n=l2_k, replace=False)
                            
                            # Calculate micro audit metrics for the specific kids the auditor reached
                            s_l2_r = compute_error_metrics(df_l2_audit, ['L1_id', 'L0_id'], l2_col, real_col, 'Samp_L2_Real')
                            s_l2_l1 = compute_error_metrics(df_l2_audit, ['L1_id', 'L0_id'], l2_col, l1_col, 'Samp_L2_L1')
                            s_l2_l0 = compute_error_metrics(df_l2_audit, ['L1_id', 'L0_id'], l2_col, l0_col, 'Samp_L2_L0')
                            l0_audit_merged = s_l2_r.merge(s_l2_l1).merge(s_l2_l0)
                            
                            # Calculate macro pooled metrics across the auditor's entire district sample block
                            s_dist_l2_r = compute_error_metrics(df_l2_audit, ['L1_id'], l2_col, real_col, 'Samp_Dist_L2_Real')
                            s_dist_l2_l1 = compute_error_metrics(df_l2_audit, ['L1_id'], l2_col, l1_col, 'Samp_Dist_L2_L1')
                            s_dist_l2_l0 = compute_error_metrics(df_l2_audit, ['L1_id'], l2_col, l0_col, 'Samp_Dist_L2_L0')
                            l1_audit_macro = s_dist_l2_r.merge(s_dist_l2_l1).merge(s_dist_l2_l0)

                            # ----------------------------------------------------------------------
                            # COMPILATION: BUILD FILE 1 (THE L0 CLINIC-LEVEL SCORECARD)
                            # ----------------------------------------------------------------------
                            l0_block = l0_samp_merged.merge(l0_audit_merged, on=['L1_id', 'L0_id'], how='left')
                            
                            # Fast column assignment (bypasses slow memory reallocation of .insert)
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
                            l1_block = l1_samp_macro.merge(l1_audit_macro, on=['L1_id'], how='left')
                            
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
            # Concat all 1,600 iterations into one dataframe instantly
            ind_df_l0 = pd.concat(ind_db_l0, ignore_index=True)
            # Merge the population truth exactly ONCE
            ind_df_l0 = ind_df_l0.merge(pop_micro_base, on=['L1_id', 'L0_id'], how='left')
            # Calculate gaps vectorized (instantly across all rows)
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
    # --------------------------------------------------------------------------
    # SECTION D: FILE STORAGE EXPORTS (PARQUET SPECIFICATIONS)
    # --------------------------------------------------------------------------
    # Concatenate array streams into highly responsive individual flat dataframes
    out_df_l0 = pd.concat(db_l0, ignore_index=True)
    out_df_l1 = pd.concat(db_l1, ignore_index=True)
    
    # Establish absolute execution file names targeted at your precalculated system repository
    export_path_l0 = os.path.join(output_dir, f"calculated_metrics_L0_{task_id}.parquet")
    export_path_l1 = os.path.join(output_dir, f"calculated_metrics_L1_{task_id}.parquet")
    
    # Commit files to disk using binary column tracking via PyArrow (Locks down analytical types)
    out_df_l0.to_parquet(export_path_l0, engine='pyarrow', index=False)
    out_df_l1.to_parquet(export_path_l1, engine='pyarrow', index=False)
    
    print(f"   -> [Step 2] Complete. Matrix Files Extracted to:\n      1. {export_path_l0}\n      2. {export_path_l1}")
    return export_path_l0, export_path_l1  # Execution paths returned seamlessly to the Master script

