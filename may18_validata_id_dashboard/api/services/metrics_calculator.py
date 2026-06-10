import os  # Provides tools to create directories and resolve system file paths
import numpy as np  # Handles high-performance numerical array operations and absolute math
import pandas as pd  # Core data manipulation library used to group, sample, and merge datasets
from scipy.stats import sem  # Specifically imported to compute the Standard Error of the Mean (SE)

def compute_error_metrics(tensor_a, tensor_b, calc_axis):
    """
    V4 Tensor Helper: Calculates Mean Absolute Error (MAE) instantly across 3D array axes.
    Replaces the old Pandas .groupby() helper for maximum C-compiled speed.
    """
    return np.abs(tensor_a - tensor_b).mean(axis=calc_axis)

def generate_dynamic_strategies(budget, max_c, max_k, tolerance=0.01, max_qty=10):
    """
    Generates Breadth/Depth strategies based on mathematical constraints.
    """
    strats = []
    buffer = int(np.ceil(budget * tolerance))
    min_budget = budget - buffer
    max_budget = budget + buffer
    
    for c in range(1, max_c + 1):
        for k in range(1, max_k + 1):
            total_kids = c * k
            if min_budget <= total_kids <= max_budget:
                strats.append((c, k))
                
    if not strats: return []
        
    sorted_strats = sorted(strats, key=lambda x: (x[0], x[1]), reverse=True)
    
    if len(sorted_strats) > max_qty:
        idx = np.linspace(0, len(sorted_strats) - 1, max_qty).astype(int)
        final_strats = [sorted_strats[i] for i in idx]
    else:
        final_strats = sorted_strats
    return final_strats

def run_tracer_engine(df_pop, task_id, output_dir, n_simulations=1, indicators=["Height", "Weight"]) -> tuple[str, str]:
    """
    Core Step 2 Math Engine. Uses C-Optimized 3D Tensors to bypass Pandas loop overhead.
    """
    print(f"   -> [Step 2] Executing 3D Tensor Math Engine for Task: {task_id}")
    
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    # 1. Structural Sorting for Tensor Reshape (Absolutely Critical)
    df_pop = df_pop.sort_values(by=['L1_id', 'L0_id']).reset_index(drop=True)
    
    N_L1S = df_pop['L1_id'].nunique()
    first_l1 = df_pop['L1_id'].iloc[0]
    N_L0S = df_pop[df_pop['L1_id'] == first_l1]['L0_id'].nunique()
    N_KIDS = len(df_pop) // (N_L1S * N_L0S)
    TOTAL_L1_KIDS = N_L0S * N_KIDS
    
    PERCENTAGES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    # Map L1 and L0 IDs for final Pandas reconstruction
    l1_l0_map = df_pop[['L1_id', 'L0_id']].drop_duplicates().values.reshape(N_L1S, N_L0S, 2)
    l1_ids_flat = l1_l0_map[:, :, 0].flatten()
    l0_ids_flat = l1_l0_map[:, :, 1].flatten()
    l1_ids_unique = l1_l0_map[:, 0, 0]

    db_l0, db_l1 = [], []
    
    for ind in indicators:
        print(f"      * Processing Indicator: {ind} (Tensor Accelerated)...")
        
        l0_col = 'L0_haz' if ind == 'Height' else ('L0_waz' if ind == 'Weight' else 'L0_whz')
        l1_col = 'L1_haz' if ind == 'Height' else ('L1_waz' if ind == 'Weight' else 'L1_whz')
        l2_col = 'L2_haz' if ind == 'Height' else ('L2_waz' if ind == 'Weight' else 'L2_whz')
        real_col = 'real_haz' if ind == 'Height' else ('real_waz' if ind == 'Weight' else 'real_whz')
        
        # ----------------------------------------------------------------------
        # EXTRACT TO C-OPTIMIZED 3D TENSORS: Shape (N_L1S, N_L0S, N_KIDS)
        # ----------------------------------------------------------------------
        T_L0 = df_pop[l0_col].values.reshape(N_L1S, N_L0S, N_KIDS)
        T_L1 = df_pop[l1_col].values.reshape(N_L1S, N_L0S, N_KIDS)
        T_L2 = df_pop[l2_col].values.reshape(N_L1S, N_L0S, N_KIDS)
        T_R  = df_pop[real_col].values.reshape(N_L1S, N_L0S, N_KIDS)
        
        # POPULATION MACRO BASELINES (Shape: N_L1S)
        pop_dist_l0_r = compute_error_metrics(T_L0, T_R, (1, 2))
        pop_dist_l1_r = compute_error_metrics(T_L1, T_R, (1, 2))
        pop_dist_l2_r = compute_error_metrics(T_L2, T_R, (1, 2))
        pop_dist_l1_l0 = compute_error_metrics(T_L1, T_L0, (1, 2))
        pop_dist_l2_l1 = compute_error_metrics(T_L2, T_L1, (1, 2))
        pop_dist_l2_l0 = compute_error_metrics(T_L2, T_L0, (1, 2))
        
        # POPULATION MICRO BASELINES (Shape: N_L1S, N_L0S)
        pop_l0_r = compute_error_metrics(T_L0, T_R, 2)
        pop_l1_r = compute_error_metrics(T_L1, T_R, 2)
        pop_l2_r = compute_error_metrics(T_L2, T_R, 2)
        pop_l1_l0 = compute_error_metrics(T_L1, T_L0, 2)
        pop_l2_l1 = compute_error_metrics(T_L2, T_L1, 2)
        pop_l2_l0 = compute_error_metrics(T_L2, T_L0, 2)
        
        ind_db_l0, ind_db_l1 = [], []
        
        for l1_pct in PERCENTAGES:
            l1_budget = int(np.round(TOTAL_L1_KIDS * l1_pct))
            
            # 🟢 L1 STRATEGY LOCKED: L1 must visit all Anganwadi Centers (N_L0S)
            l1_c = N_L0S
            l1_k = max(1, int(np.round(l1_budget / l1_c)))
            
            # We assign exactly ONE valid strategy for L1 per budget
            l1_strats = [(l1_c, l1_k)]
            
            for l1_c, l1_k in l1_strats:
                for sim_id in range(n_simulations):

                    np.random.seed(sim_id)
                    
                    # --- L1 CLINIC SAMPLING (Instant Tensor Slice) ---
                    idx_c = np.random.rand(N_L1S, N_L0S).argsort(axis=1)[:, :l1_c]
                    T_L0_sc = np.take_along_axis(T_L0, idx_c[:, :, None], axis=1)
                    T_L1_sc = np.take_along_axis(T_L1, idx_c[:, :, None], axis=1)
                    T_L2_sc = np.take_along_axis(T_L2, idx_c[:, :, None], axis=1)
                    T_R_sc  = np.take_along_axis(T_R,  idx_c[:, :, None], axis=1)
                    
                    # --- L1 KID SAMPLING ---
                    idx_k = np.random.rand(N_L1S, l1_c, N_KIDS).argsort(axis=2)[:, :, :l1_k]
                    T_L0_s = np.take_along_axis(T_L0_sc, idx_k, axis=2)
                    T_L1_s = np.take_along_axis(T_L1_sc, idx_k, axis=2)
                    T_L2_s = np.take_along_axis(T_L2_sc, idx_k, axis=2)
                    T_R_s  = np.take_along_axis(T_R_sc,  idx_k, axis=2)
                    
                    # --- L1 MACRO ERRORS ---
                    s_dist_l0_r = compute_error_metrics(T_L0_s, T_R_s, (1, 2))
                    s_dist_l1_r = compute_error_metrics(T_L1_s, T_R_s, (1, 2))
                    s_dist_l1_l0 = compute_error_metrics(T_L1_s, T_L0_s, (1, 2))
                    
                    # --- L1 MICRO ERRORS ---
                    s_l0_r_raw = compute_error_metrics(T_L0_s, T_R_s, 2)
                    s_l1_r_raw = compute_error_metrics(T_L1_s, T_R_s, 2)
                    s_l1_l0_raw = compute_error_metrics(T_L1_s, T_L0_s, 2)
                    
                    # Restore to original (N_L1S, N_L0S) shape padded with NaNs
                    s_l0_r = np.full((N_L1S, N_L0S), np.nan)
                    s_l1_r = np.full((N_L1S, N_L0S), np.nan)
                    s_l1_l0 = np.full((N_L1S, N_L0S), np.nan)
                    
                    np.put_along_axis(s_l0_r, idx_c, s_l0_r_raw, axis=1)
                    np.put_along_axis(s_l1_r, idx_c, s_l1_r_raw, axis=1)
                    np.put_along_axis(s_l1_l0, idx_c, s_l1_l0_raw, axis=1)
                    
                    # BUILD L0 DATAFRAME BLOCK 
                    l0_block = pd.DataFrame({
                        'L1_id': l1_ids_flat, 'L0_id': l0_ids_flat, 'Indicator': ind, 'Sim_ID': sim_id,
                        'L1_Budget_Pct': l1_pct, 'L1_Total_Kids_Sampled': l1_c * l1_k, 'L1_Label': f"{l1_c}C_x_{l1_k}K",
                        'Pop_L0_Real_MAE': pop_l0_r.flatten(), 'Pop_L1_Real_MAE': pop_l1_r.flatten(),
                        'Pop_L2_Real_MAE': pop_l2_r.flatten(), 'Pop_L1_L0_MAE': pop_l1_l0.flatten(),
                        'Pop_L2_L1_MAE': pop_l2_l1.flatten(), 'Pop_L2_L0_MAE': pop_l2_l0.flatten(),
                        'Samp_L0_Real_MAE': s_l0_r.flatten(), 'Samp_L1_Real_MAE': s_l1_r.flatten(), 'Samp_L1_L0_MAE': s_l1_l0.flatten(),
                        'Gap_Samp_vs_Pop_L0_Real': np.abs(s_l0_r.flatten() - pop_l0_r.flatten()),
                        'Gap_Samp_vs_Pop_L1_L0': np.abs(s_l1_l0.flatten() - pop_l1_l0.flatten())
                    })
                    ind_db_l0.append(l0_block)

                    # --- L2 AUDITOR LOOP ---
                    for l2_pct in PERCENTAGES:
                        l2_budget = max(1, int(np.round((l1_c * l1_k) * l2_pct)))
                        l2_strats = generate_dynamic_strategies(l2_budget, l1_c, l1_k, tolerance=0.01, max_qty=10)
                        
                        for l2_c, l2_k in l2_strats:
                            np.random.seed(sim_id + 500)
                            
                            # L2 Clinics (Slice from L1's visited clinics)
                            idx_l2_c = np.random.rand(N_L1S, l1_c).argsort(axis=1)[:, :l2_c]
                            
                            T_L0_l2c = np.take_along_axis(T_L0_s, idx_l2_c[:, :, None], axis=1)
                            T_L1_l2c = np.take_along_axis(T_L1_s, idx_l2_c[:, :, None], axis=1)
                            T_L2_l2c = np.take_along_axis(T_L2_s, idx_l2_c[:, :, None], axis=1)
                            T_R_l2c  = np.take_along_axis(T_R_s,  idx_l2_c[:, :, None], axis=1)
                            
                            # L2 Kids (Slice from L1's measured kids)
                            idx_l2_k = np.random.rand(N_L1S, l2_c, l1_k).argsort(axis=2)[:, :, :l2_k]
                            
                            T_L0_audit = np.take_along_axis(T_L0_l2c, idx_l2_k, axis=2)
                            T_L1_audit = np.take_along_axis(T_L1_l2c, idx_l2_k, axis=2)
                            T_L2_audit = np.take_along_axis(T_L2_l2c, idx_l2_k, axis=2)
                            T_R_audit  = np.take_along_axis(T_R_l2c,  idx_l2_k, axis=2)
                            
                            # L2 MACRO ERRORS
                            s_dist_l2_r = compute_error_metrics(T_L2_audit, T_R_audit, (1, 2))
                            s_dist_l2_l1 = compute_error_metrics(T_L2_audit, T_L1_audit, (1, 2))
                            s_dist_l2_l0 = compute_error_metrics(T_L2_audit, T_L0_audit, (1, 2))
                            
                            # BUILD L1 DATAFRAME BLOCK
                            l1_block = pd.DataFrame({
                                'L1_id': l1_ids_unique, 'Indicator': ind, 'Sim_ID': sim_id,
                                'L1_Budget_Pct': l1_pct, 'L1_District_Total_Kids_Sampled': l1_c * l1_k, 'L1_Label': f"{l1_c}C_x_{l1_k}K",
                                'L2_Budget_Pct': l2_pct, 'L2_District_Total_Kids_Sampled': l2_c * l2_k, 'L2_Label': f"{l2_c}C_x_{l2_k}K",
                                'Pop_Dist_L0_Real_MAE': pop_dist_l0_r, 'Pop_Dist_L1_Real_MAE': pop_dist_l1_r, 'Pop_Dist_L2_Real_MAE': pop_dist_l2_r,
                                'Pop_Dist_L1_L0_MAE': pop_dist_l1_l0, 'Pop_Dist_L2_L1_MAE': pop_dist_l2_l1, 'Pop_Dist_L2_L0_MAE': pop_dist_l2_l0,
                                'Samp_Dist_L0_Real_MAE': s_dist_l0_r, 'Samp_Dist_L1_Real_MAE': s_dist_l1_r, 'Samp_Dist_L1_L0_MAE': s_dist_l1_l0,
                                'Samp_Dist_L2_Real_MAE': s_dist_l2_r, 'Samp_Dist_L2_L1_MAE': s_dist_l2_l1, 'Samp_Dist_L2_L0_MAE': s_dist_l2_l0
                            })
                            ind_db_l1.append(l1_block)
        
        # Compile the hyper-fast blocks back into Pandas for export
        if ind_db_l0: db_l0.append(pd.concat(ind_db_l0, ignore_index=True))
        if ind_db_l1: db_l1.append(pd.concat(ind_db_l1, ignore_index=True))
            
    out_df_l0 = pd.concat(db_l0, ignore_index=True)
    out_df_l1 = pd.concat(db_l1, ignore_index=True)
    
    export_path_l0 = os.path.join(output_dir, f"calculated_metrics_L0_{task_id}.parquet")
    export_path_l1 = os.path.join(output_dir, f"calculated_metrics_L1_{task_id}.parquet")
    
    out_df_l0.to_parquet(export_path_l0, engine='pyarrow', index=False)
    out_df_l1.to_parquet(export_path_l1, engine='pyarrow', index=False)
    
    print(f"   -> [Step 2] Complete. Matrix Files Extracted to:\n      1. {export_path_l0}\n      2. {export_path_l1}")
    return export_path_l0, export_path_l1