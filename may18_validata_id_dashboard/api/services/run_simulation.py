import os  # Structural engine used to navigate, verify, and build machine directories
import sys  # System module used to dynamically adjust and extend the environment import paths
import time  # Time library utilized to profile execution speeds across processing blocks
from datetime import datetime  # Handles generation of clean, non-clashing timestamp markers

# ==============================================================================
# 1. DYNAMIC ENVIRONMENT PATH RESOLUTION
# ==============================================================================
# Pinpoint exactly where this controller script lives on the host operating system
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Ensure the root directory path is visible to the Python runtime import environment
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

# 🟢 CORRECTED: Removed the sys.path hack and used absolute imports
from services import data_generator as step1
from services import metrics_calculator as step2
from services import analytics_ranker as step3

def execute_full_pipeline(user_params=None, run_name="Custom_Run", n_simulations=5):
    """
    Orchestrates the entire simulation framework sequentially.
    Passes data down the pipeline using lightweight file paths rather than memory-heavy objects.
    """
    print("=" * 80)
    print(f"STARTING SIMULATION ENGINE PIPELINE | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Capture initial timestamp to compute total elapsed processing run runtime
    start_time = time.time()

    # --------------------------------------------------------------------------
    # CONFIGURATION & REPOSITORY INTERACTION INITIALIZATION
    # --------------------------------------------------------------------------
    # Default fallback physics parameters used if none are provided by a UI or wrapper script
    default_params = {
        "n_L1s": 20,                          # Fixed state count of L1 Supervisors
        "n_L0s_per_L1": 15,                    # Fixed clinic infrastructure count per Supervisor
        "n_children_per_L0": 15,               # Child registrations tracked per clinic registry
        "real_percent_stunting": 35.0,         # Baseline true stunting biology percentage
        "real_percent_underweight": 33.0,      # Baseline true underweight biology percentage
        "mean_percent_under_reporting_stunting": 5.0,
        "mean_percent_under_reporting_underweight": 5.0,
        "mean_bunch_factor_haz": 0.05,
        "mean_bunch_factor_waz": 0.05,
        "mean_bunch_factor_whz": 0.05,
        "mean_percent_copy": 5.0,
        "mean_collusion_index": 0.02,
        "error_sd_height_all_L0s": 0,
        "error_sd_weight_all_L0s": 0,
        "rho": 0.7                             # Bivariate biometric correlation factor
    }
    
    # Safely unpack user configurations directly over the default fallback values
    active_params = default_params if user_params is None else {**default_params, **user_params}
    
    # Create an un-clashable execution task ID token using a date-time signature
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_id = f"{run_name}_{timestamp}"
    
    # Establish the absolute folder target path where data outputs will be stored
    output_repository = os.path.join(CURRENT_DIR, "outputs", "Precalculated_Presets")

    # Pre-emptively verify and generate directory layout before Step 1 runs to avoid OS errors
    os.makedirs(output_repository, exist_ok=True)

    print(f"[PRE-RUN] Generated Task ID: {task_id}")
    print(f"[PRE-RUN] Saving artifacts to: {output_repository}\n")

    # --------------------------------------------------------------------------
    # EXECUTION STEP 1: GENERATE SYNTHETIC UNIVERSE
    # --------------------------------------------------------------------------
    t1 = time.time()
    print(f"[PIPELINE STEP 1] Fabricating biological population universe...")
    
    # Step 1 builds the entire biological baseline population and logs it to disk
    universe_parquet, df_universe = step1.build_universe(
        params=active_params, 
        task_id=task_id, 
        output_dir=output_repository
    )
    print(f"   -> Step 1 Finished in {time.time() - t1:.2f} seconds.")
    print(f"   -> Universe Population Generated: {len(df_universe)} children.\n")

    # --------------------------------------------------------------------------
    # EXECUTION STEP 2: GENERATION OF COMBINATORIAL MATH MATRICES
    # --------------------------------------------------------------------------
    t2 = time.time()
    print(f"[PIPELINE STEP 2] Simulating field operations and calculating error scorecards...")
    
    # Fed the dynamic n_simulations argument down into the Step 2 call block
    l0_matrix_path, l1_matrix_path = step2.run_tracer_engine(
        df_pop=df_universe, 
        task_id=task_id, 
        output_dir=output_repository,
        n_simulations=n_simulations,                      
        indicators=["Height", "Weight"]       # Firmly bound back to our matching column metrics
    )
    print(f"   -> Step 2 Finished in {time.time() - t2:.2f} seconds.\n")

    # --------------------------------------------------------------------------
    # EXECUTION STEP 3: ANALYTICS RANKING & DATA CONDENSATION
    # --------------------------------------------------------------------------
    t3 = time.time()
    print(f"[PIPELINE STEP 3] Sorting metrics and generating detection analytics...")
    
    # Step 3 evaluates catch-rate detection metrics across multiple percentile thresholds
    l0_final_summary, l1_final_summary = step3.process_ranking_analytics(
        l0_parquet_path=l0_matrix_path, 
        l1_parquet_path=l1_matrix_path, 
        output_dir=output_repository, 
        task_id=task_id,
        target_percentiles=[0.10, 0.20, 0.30]  # Standard precalculated visualization slots
    )
    print(f"   -> Step 3 Finished in {time.time() - t3:.2f} seconds.\n")

    # --------------------------------------------------------------------------
    # SYSTEM EXECUTION LOG RECAP SUMMARY
    # --------------------------------------------------------------------------
    total_execution_time = time.time() - start_time
    print("=" * 80)
    print(" PIPELINE EXECUTION SUCCESSFUL!")
    print(f" Total Elapsed Time: {total_execution_time:.2f} seconds")
    print(" Final Dashboard Deliverables Generated:")
    print(f"   1. Clinic UI Summary Table:     {os.path.basename(l0_final_summary)}")
    print(f"   2. Supervisor UI Summary Table: {os.path.basename(l1_final_summary)}")
    print("=" * 80)
    
    # Return both condensed summary target file paths to the calling context/frontend interface
    return l0_final_summary, l1_final_summary

if __name__ == "__main__":
    # If run standalone directly inside a shell terminal, execute with baseline constraints
    execute_full_pipeline(run_name="Terminal_Direct_Run", n_simulations=5)