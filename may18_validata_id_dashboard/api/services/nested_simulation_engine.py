import os  # Structural file routing engine to resolve system paths safely
import uuid  # Universally Unique Identifier generator to assign distinct task tokens
import math  # Low-level mathematical checks to identify system NaN and Infinity elements
import pandas as pd  # Primary data frame tool used here to load completed Parquet files
import numpy as np  # Array structures used to catch float distortions in JSON outputs
from datetime import datetime  # Date timestamp tools to partition task folders cleanly
from fastapi import FastAPI, BackgroundTasks  # Core asynchronous web router frameworks
from pydantic import BaseModel  # Strict validation structures for typed input blocks
from typing import Dict, Any, List  # Explicit type annotations for clean system compilation

# Pull in our structural decoupled backend steps
import data_generator  # Step 1: Population Creator
import metrics_calculator  # Step 2: Combinatorial Matrix Engine
import analytics_ranker  # Step 3: Lightweight Rank Summary Engine

# Initialize the application instance once
app = FastAPI(title="Nested Simulation Engine API")

# Lightweight thread-safe storage simulation representing our tasks lookup table
tasks_db = {}

def clean_json_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Helper function that converts a dataframe to a dictionary array, safely 
    replacing illegal floating-point NaN/Infinity tokens with clean None values
    so the JSON serialization layer never crashes Streamlit.
    """
    raw_records = df.to_dict(orient="records")
    clean_records = []
    
    for row in raw_records:
        clean_row = {}
        for k, v in row.items():
            # Intercept system float anomalies before they break web encoders
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                clean_row[k] = None
            else:
                clean_row[k] = v
        clean_records.append(clean_row)
        
    return clean_records

def run_simulation_task(params: dict, task_id: str):
    """
    Asynchronous background worker execution pipeline. Sequences Steps 1, 2, and 3 
    safely on a isolated execution line without hanging web thread requests.
    """
    try:
        print(f"[{task_id}] Initializing full asynchronous simulation loop pipeline...")
        
        # =====================================================================
        # 1. SETUP UNIQUE DIRECTORY ISOLATION
        # =====================================================================
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_run_name = f"run_{timestamp}_{task_id[:8]}"
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "engine_outputs", unique_run_name)
        os.makedirs(output_dir, exist_ok=True)
        print(f"[{task_id}] Run artifacts will be deposited to: {output_dir}")
        
        # =====================================================================
        # 2. DYNAMIC INPUT PARAMETER RESOLUTION
        # =====================================================================
        # Parse standard parameter overrides, handling dynamic Monte Carlo scales safely
        n_simulations = int(params.get("n_simulations", 5))
        
        # Map target percentiles out to an operational array frame for Step 3 list evaluations
        raw_pct = params.get("target_percentile", 0.30)
        # Support either individual float inputs or extended listing sets from advanced configs
        target_percentiles = [raw_pct] if isinstance(raw_pct, float) else list(raw_pct)
        
        # Set clean metric switches and remove old indicators (no "Wasting")
        out_var = params.get("output_variable", "Both").lower()
        if out_var == "both": 
            target_inds = ["Height", "Weight"]
        elif out_var == "weight": 
            target_inds = ["Weight"]
        else: 
            target_inds = ["Height"]
        
        # =====================================================================
        # 3. CORE STEP PIPELINE INTEGRATION SEQUENCE
        # =====================================================================
        # EXECUTION STEP 1: Fabricate baseline biological universe
        print(f"[{task_id}] Triggering Step 1 Data Generator...")
        child_path, df_pop = data_generator.build_universe(params, task_id, output_dir)
        
        # EXECUTION STEP 2: Calculate combinatorial strategy matrices
        print(f"[{task_id}] Triggering Step 2 Metrics Matrix Calculator...")
        l0_matrix_path, l1_matrix_path = metrics_calculator.run_tracer_engine(
            df_pop=df_pop, 
            task_id=task_id, 
            output_dir=output_dir, 
            n_simulations=n_simulations,
            indicators=target_inds
        )
        
        # EXECUTION STEP 3: Compile lightweight overlap detection summary lists
        print(f"[{task_id}] Triggering Step 3 Analytics Overlap Ranker Summary...")
        final_csv_path, final_parquet_path = analytics_ranker.process_ranking_analytics(
            l0_parquet_path=l0_matrix_path,
            l1_parquet_path=l1_matrix_path,
            output_dir=output_dir,
            task_id=task_id,
            target_percentiles=target_percentiles
        )
        
        # =====================================================================
        # 4. WRITE TASK COMPLETE ENTRY FOR FRONTLINE POLLING LOOKUPS
        # =====================================================================
        tasks_db[task_id] = {
            "status": "Complete",
            "strategy_path": final_csv_path, # 🟢 NEW: Saving the exact CSV path
            "child_data_path": child_path,
            "run_folder": output_dir
        }
        print(f"[{task_id}] Full Pipeline Executed Successfully. Logs locked down.")
        
    except Exception as e:
        print(f"[{task_id}] PIPELINE EXCEPTION ERROR CRASH: {str(e)}")
        tasks_db[task_id] = {"status": "Failed", "error": str(e)}


@app.post("/start-nested-sim")
def start_nested_sim(params: Dict[str, Any], background_tasks: BackgroundTasks):
    """
    API Payload Entrypoint. Instantiates a unique transaction token registry,
    spins up the asynchronous processing task thread, and returns immediately.
    """
    task_id = str(uuid.uuid4())
    tasks_db[task_id] = {"status": "Processing"}
    
    # Hand off execution flow cleanly to background layer threads
    background_tasks.add_task(run_simulation_task, params, task_id)
    
    return {"task_id": task_id, "status": "Started"}


@app.get("/check-nested-sim/{task_id}")
def check_nested_sim(task_id: str):
    """
    Dynamic polling sync check endpoint utilized by Streamlit to monitor background jobs.
    When complete, it decodes the unified Tracer Master DB safely for UI rendering.
    """
    task = tasks_db.get(task_id, {"status": "Failed", "error": "Task identification token not found."})
    
    if task["status"] == "Complete":
        # 🟢 NEW: Read the single unified CSV file safely from Step 3
        df = pd.read_csv(task["strategy_path"])
        
        # Clean the dataframe to shield the web protocol from NaN/Infinity serialization issues
        clean_records = clean_json_records(df)

        # Return the unified operational summary down to the waiting user dashboard state
        return {
            "status": "Complete",
            "l1_summary": clean_records,          # 🟢 Matches what the UI file expects
            "final_csv_path": task["strategy_path"] # 🟢 Passes the physical path back
        }
        
    return {"status": task["status"]}



# import os
# import uuid
# import pandas as pd
# import numpy as np  # <-- Required for the JSON fix
# import math
# from datetime import datetime
# from fastapi import FastAPI, BackgroundTasks
# from pydantic import BaseModel
# from typing import Dict, Any

# import data_generator
# import metrics_calculator

# # Initialize the app exactly once
# app = FastAPI(title="Nested Simulation Engine API")

# # A simple dictionary to keep track of running tasks
# tasks_db = {}

# def run_simulation_task(params: dict, task_id: str):
#     try:
#         print(f"[{task_id}] Delegating to Data Generator...")
        
#         # =====================================================================
#         # 1. CREATE UNIQUE TIMESTAMPED FOLDER FOR THIS RUN
#         # =====================================================================
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         unique_run_name = f"run_{timestamp}_{task_id[:8]}"
#         output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "engine_outputs", unique_run_name)
#         os.makedirs(output_dir, exist_ok=True)
#         print(f"[{task_id}] All files will be saved to: {output_dir}")
        
#         # 2. Parse Parameters
#         target_pct = params.get("target_percentile", 0.30)
#         has_l2 = params.get("has_l2", "Yes")
#         out_var = params.get("output_variable", "Both").lower()
        
#         if out_var == "both": target_inds = ["Height", "Weight"]
#         elif out_var == "weight": target_inds = ["Weight"]
#         else: target_inds = ["Height"]
        
#         # 3. Generate the heavy child data
#         child_path, df_pop = data_generator.build_universe(params, task_id, output_dir)
        
#         # 4. Sweep the strategies
#         print(f"[{task_id}] Delegating to Metrics Calculator...")
#         strategy_path = metrics_calculator.run_tracer_engine(
#             df_pop, task_id, output_dir, 
#             target_percentile=target_pct, indicators=target_inds, has_l2=has_l2
#         )
        
#         # 5. Mark task as complete
#         tasks_db[task_id] = {
#             "status": "Complete",
#             "strategy_path": strategy_path,
#             "child_data_path": child_path,
#             "run_folder": output_dir
#         }
#         print(f"[{task_id}] Simulation Finished Successfully!")
        
#     except Exception as e:
#         print(f"[{task_id}] FAILED: {str(e)}")
#         tasks_db[task_id] = {"status": "Failed", "error": str(e)}


# @app.post("/start-nested-sim")
# def start_nested_sim(params: Dict[str, Any], background_tasks: BackgroundTasks):
#     """Endpoint to trigger the simulation and return a tracking ID."""
#     task_id = str(uuid.uuid4())
#     tasks_db[task_id] = {"status": "Processing"}
    
#     # Hand the heavy lifting off to a background thread
#     background_tasks.add_task(run_simulation_task, params, task_id)
    
#     return {"task_id": task_id, "status": "Started"}

# @app.get("/check-nested-sim/{task_id}")
# def check_nested_sim(task_id: str):
#     """Endpoint for Streamlit to constantly poll and check if the math is done."""
#     task = tasks_db.get(task_id, {"status": "Failed", "error": "Task not found"})
    
#     if task["status"] == "Complete":
#         # Read the calculated CSV
#         df = pd.read_csv(task["strategy_path"])
        
#         # =====================================================================
#         # BULLETPROOF FIX: Clean NaN/Infinity strictly at the Python level
#         # =====================================================================
#         raw_records = df.to_dict(orient="records")
#         clean_records = []
        
#         for row in raw_records:
#             clean_row = {}
#             for k, v in row.items():
#                 # If the value is a float and is either NaN or Infinity, convert to None
#                 if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
#                     clean_row[k] = None
#                 else:
#                     clean_row[k] = v
#             clean_records.append(clean_row)
        
#         return {
#             "status": "Complete",
#             "data": clean_records,
#             "child_data_path": task["child_data_path"]
#         }
        
#     return {"status": task["status"]}




# import os
# import sys
# import time
# import numpy as np
# import pandas as pd

# # DYNAMIC PATH REGISTRATION
# CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# API_DIR = os.path.dirname(CURRENT_DIR)
# ECD_DIR = os.path.join(API_DIR, "ecd_nested_simulation_functions")

# if API_DIR not in sys.path: sys.path.append(API_DIR)
# if ECD_DIR not in sys.path: sys.path.append(ECD_DIR)

# try:
#     from ecd_nested_simulation_functions import generate_ecd_dummy_data
#     from ecd_nested_simulation_functions import ecd_sampling_strategy
#     print("✅ ECD Simulation modules imported successfully.")
# except ImportError as e:
#     print(f"❌ IMPORT ERROR: {e}")

# def run_custom_simulation(params: dict, task_id: str):
#     print(f"[{task_id}] BOOTING UP: Grand Master Engine for Custom Scenario...")
#     start_time = time.time()

#     output_dir = os.path.join(CURRENT_DIR, "engine_outputs")
#     if not os.path.exists(output_dir): os.makedirs(output_dir)

#     copy_paste = params.get("copy_paste_pct", 5)
#     collusion = params.get("collusion_factor", 0.10)
#     eq_error = params.get("equipment_error", 0.1)
#     stunting_under_report = int(params.get("l1_corruption_pct", 0.05) * 60)
#     bunch_factor = params.get("l0_fraud_pct", 0.05) * 0.50 

#     N_L1S = int(params.get("n_L1s", 334))
#     N_L0S_PER_L1 = int(params.get("n_L0s_per_L1", 25))
#     N_CHILDREN = int(params.get("n_children_per_L0", 15)) 

#     def extract_data_from_dta(filename):
#         return pd.read_stata(os.path.join(CURRENT_DIR, filename))

#     print(f"[{task_id}] Loading WHO Standards...")
#     haz_params = extract_data_from_dta('lenanthro.dta')
#     waz_params = extract_data_from_dta('weianthro.dta')
#     whz_params_lying = extract_data_from_dta('wflanthro.dta')
#     whz_params_standing = extract_data_from_dta('wfhanthro.dta')

#     rho = 0.7 
#     common_real_params = {
#         'girl_ratio': 0.5, 'min_age': 0, 'max_age': 1790, 'num_timepoints': 1, 'time_lags': [],
#         'percent_stunting': params.get("real_percent_stunting", 35), 
#         'percent_underweight': params.get("real_percent_underweight", 33), 
#         'rho': rho
#     }
    
#     L0_params_list, L1_params_list, L2_params_dict = generate_ecd_dummy_data.generate_nested_distortion_parameters(
#         n_L1s=N_L1S, n_L0s_per_L1=N_L0S_PER_L1,
#         real_percent_stunting=params.get("real_percent_stunting", 35), 
#         real_percent_underweight=params.get("real_percent_underweight", 33),
#         mean_percent_under_reporting_stunting=stunting_under_report, 
#         mean_percent_under_reporting_underweight=stunting_under_report, 
#         mean_bunch_factor_haz=bunch_factor, mean_bunch_factor_waz=bunch_factor, mean_bunch_factor_whz=bunch_factor,
#         mean_percent_copy=copy_paste, mean_collusion_index=collusion,
#         sd_across_units_percent_under_reporting_stunting=params.get("sd_across_units_percent_under_reporting_stunting", 2.0), 
#         sd_across_units_percent_under_reporting_underweight=params.get("sd_across_units_percent_under_reporting_underweight", 2.0),
#         sd_within_units_percent_under_reporting_stunting=params.get("sd_within_units_percent_under_reporting_stunting", 1.0), 
#         sd_within_units_percent_under_reporting_underweight=params.get("sd_within_units_percent_under_reporting_underweight", 1.0),
#         sd_across_units_bunch_factor_haz=params.get("sd_across_units_bunch_factor_haz", 0.01), 
#         sd_across_units_bunch_factor_waz=params.get("sd_across_units_bunch_factor_waz", 0.01), 
#         sd_across_units_bunch_factor_whz=params.get("sd_across_units_bunch_factor_whz", 0.01),
#         sd_within_units_bunch_factor_haz=params.get("sd_within_units_bunch_factor_haz", 0.01), 
#         sd_within_units_bunch_factor_waz=params.get("sd_within_units_bunch_factor_waz", 0.01), 
#         sd_within_units_bunch_factor_whz=params.get("sd_within_units_bunch_factor_whz", 0.01),
#         sd_percent_copy=params.get("sd_percent_copy", 2.0), sd_collusion_index=params.get("sd_collusion_index", 0.02),
#         error_sd_height_all_L0s=eq_error, error_sd_weight_all_L0s=eq_error*0.1,
#         error_sd_height_L1=eq_error, error_sd_weight_L1=eq_error*0.1, 
#         error_sd_height_L2=eq_error, error_sd_weight_L2=eq_error*0.1, 
#         mean_time_lag_L1=params.get("mean_time_lag_L1", 15), mean_time_lag_L2=params.get("mean_time_lag_L2", 30)
#     )

#     np.random.seed(42) 
#     print(f"[{task_id}] Generating Synthetic Records...")
#     nested_measurements = generate_ecd_dummy_data.generate_nested_measurements(
#         real_params=common_real_params, L0_params_list=L0_params_list, L1_params_list=L1_params_list, L2_params_dict=L2_params_dict,
#         n_L1s=N_L1S, n_L0s_per_L1=N_L0S_PER_L1, 
#         n_children_per_L0=N_CHILDREN, n_children_L1=N_CHILDREN, n_children_L2=N_CHILDREN,
#         haz_params=haz_params, waz_params=waz_params, whz_params_lying=whz_params_lying, whz_params_standing=whz_params_standing, make_plots=False
#     )

#     all_children_records = []
#     for L1_id, L1_data in nested_measurements.items():
#         if L1_id == 'metadata': continue
#         for L0_id, L0_data in L1_data.items():
#             if L0_id == 'L1_info': continue
            
#             df_real = L0_data['real']['data'].copy().rename(columns={'haz': 'real_haz', 'waz': 'real_waz', 'height': 'real_height', 'weight': 'real_weight'})
#             df_L0 = L0_data['L0']['data'].copy().rename(columns={'haz': 'L0_haz', 'waz': 'L0_waz', 'height': 'L0_height', 'weight': 'L0_weight'})
#             df_L1 = L0_data['L1']['data'].copy().rename(columns={'haz': 'L1_haz', 'waz': 'L1_waz', 'height': 'L1_height', 'weight': 'L1_weight'})
#             df_L2 = L0_data['L2']['data'].copy().rename(columns={'haz': 'L2_haz', 'waz': 'L2_waz', 'height': 'L2_height', 'weight': 'L2_weight'})
            
#             merged_df = df_real[['child_id', 'real_height', 'real_weight', 'real_haz', 'real_waz']].merge(
#                 df_L0[['child_id', 'L0_height', 'L0_weight', 'L0_haz', 'L0_waz']], on='child_id', how='left').merge(
#                 df_L1[['child_id', 'L1_height', 'L1_weight', 'L1_haz', 'L1_waz']], on='child_id', how='left').merge(
#                 df_L2[['child_id', 'L2_height', 'L2_weight', 'L2_haz', 'L2_waz']], on='child_id', how='left')
                
#             merged_df['L1_id'], merged_df['L0_id'] = L1_id, L0_id
#             all_children_records.append(merged_df)

#     df_pop = pd.concat(all_children_records, ignore_index=True)
    
#     # Calculate Child Errors and Region MAEs
#     df_pop['L0_HAZ_Error (vs Real)'] = np.abs(df_pop['L0_haz'] - df_pop['real_haz'])
#     df_pop['L1_HAZ_Error (vs L0)'] = np.abs(df_pop['L1_haz'] - df_pop['L0_haz'])
#     df_pop['L2_HAZ_Error (vs L1)'] = np.abs(df_pop['L2_haz'] - df_pop['L1_haz'])
    
#     df_pop['L0_Center_Overall_MAE'] = df_pop.groupby('L0_id')['L0_HAZ_Error (vs Real)'].transform('mean')
#     df_pop['L1_Region_Overall_MAE'] = df_pop.groupby('L1_id')['L1_HAZ_Error (vs L0)'].transform('mean')
    
#     cols = ['L1_id', 'L0_id', 'child_id', 
#             'real_height', 'L0_height', 'L1_height', 'L2_height', 
#             'real_weight', 'L0_weight', 'L1_weight', 'L2_weight', 
#             'real_haz', 'L0_haz', 'L1_haz', 'L2_haz', 
#             'real_waz', 'L0_waz', 'L1_waz', 'L2_waz',
#             'L0_HAZ_Error (vs Real)', 'L1_HAZ_Error (vs L0)', 'L2_HAZ_Error (vs L1)',
#             'L0_Center_Overall_MAE', 'L1_Region_Overall_MAE']
#     df_pop = df_pop[cols]
    
#     child_export_path = os.path.join(output_dir, f"child_data_{task_id}.csv")
#     df_pop.to_csv(child_export_path, index=False)
    
#     print(f"[{task_id}] Calculating Optimal Strategies...")
#     TARGET_L1_COUNT = int(N_L1S * 0.30)
#     if TARGET_L1_COUNT < 1: TARGET_L1_COUNT = 1
    
#     TOTAL_L1_KIDS = N_L0S_PER_L1 * N_CHILDREN  
#     PERCENTAGES = [0.20, 0.40, 0.60, 0.80, 1.00] 
    
#     def calculate_metrics(df, group_cols, col_meas, col_baseline):
#         temp_df = df.copy()
#         temp_df['_err'] = np.abs(temp_df[col_meas] - temp_df[col_baseline])
#         return temp_df.groupby(group_cols).agg(MAE=('_err', 'mean')).reset_index()

#     master_database = []
#     god_l1 = calculate_metrics(df_pop, ['L1_id'], 'L0_haz', 'real_haz')
#     god_l1_mae = set(god_l1.sort_values(by=['MAE', 'L1_id'], ascending=[False, True]).head(TARGET_L1_COUNT)['L1_id'])

#     for l1_pct in PERCENTAGES:
#         l1_budget = int(TOTAL_L1_KIDS * l1_pct)
#         l1_c, l1_k = min(int(np.sqrt(l1_budget)), N_L0S_PER_L1), min(int(np.sqrt(l1_budget)), N_CHILDREN)
#         if l1_c == 0: l1_c = 1
#         if l1_k == 0: l1_k = 1
        
#         l1_clinics = df_pop[['L1_id', 'L0_id']].drop_duplicates().groupby('L1_id').apply(
#             lambda x: x.sample(n=min(len(x), l1_c), replace=False)
#         ).reset_index(drop=True)
        
#         df_l1_sheet = df_pop.merge(l1_clinics, on=['L1_id', 'L0_id']).groupby(['L1_id', 'L0_id']).apply(
#             lambda x: x.sample(n=min(len(x), l1_k), replace=False)
#         ).reset_index(drop=True)
        
#         sheet_truth = calculate_metrics(df_l1_sheet, ['L1_id'], 'L1_haz', 'real_haz')
#         sheet_mae = set(sheet_truth.sort_values(by=['MAE', 'L1_id'], ascending=[False, True]).head(TARGET_L1_COUNT)['L1_id'])
        
#         l1_diagnosis = calculate_metrics(df_l1_sheet, ['L1_id'], 'L1_haz', 'L0_haz')
#         l1_diag_caught = set(l1_diagnosis.sort_values(by=['MAE', 'L1_id'], ascending=[False, True]).head(TARGET_L1_COUNT)['L1_id'])
#         v1_acc = (len(l1_diag_caught & god_l1_mae) / TARGET_L1_COUNT) * 100

#         for l2_pct in PERCENTAGES:
#             l2_budget = int((l1_c * l1_k) * l2_pct)
#             l2_c, l2_k = min(l1_c, max(1, int(np.sqrt(l2_budget)))), min(l1_k, max(1, int(np.sqrt(l2_budget))))
            
#             df_l2_audit = df_l1_sheet.groupby('L1_id').apply(
#                 lambda x: x.sample(n=min(len(x), l2_c * l2_k), replace=False)
#             ).reset_index(drop=True)
            
#             l2_vs_l1 = calculate_metrics(df_l2_audit, ['L1_id'], 'L2_haz', 'L1_haz')
#             l2_caught = set(l2_vs_l1.sort_values(by=['MAE', 'L1_id'], ascending=[False, True]).head(TARGET_L1_COUNT)['L1_id'])
#             v3_acc = (len(l2_caught & sheet_mae) / TARGET_L1_COUNT) * 100
            
#             master_database.append({
#                 'L1_Budget_Pct': f"{int(l1_pct*100)}%",
#                 'L1_Strategy': f"{l1_c} Clinics x {l1_k} Kids",
#                 'L2_Budget_Pct': f"{int(l2_pct*100)}%",
#                 'L2_Strategy': f"{l2_c} Clinics x {l2_k} Kids",
#                 'L1_Accuracy_vs_L0': f"{round(v1_acc, 1)}%",
#                 'L2_Accuracy_vs_L1': f"{round(v3_acc, 1)}%"
#             })

#     final_df = pd.DataFrame(master_database)
#     export_path = os.path.join(output_dir, f"result_{task_id}.csv")
#     final_df.to_csv(export_path, index=False)
    
#     print(f"[{task_id}] SUCCESS! Data saved.")
#     return {"strategy_path": export_path, "child_path": child_export_path}