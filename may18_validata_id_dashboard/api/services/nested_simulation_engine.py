import os  # Structural file routing engine to resolve system paths safely
import uuid  # Universally Unique Identifier generator to assign distinct task tokens
import math  # Low-level mathematical checks to identify system NaN and Infinity elements
import pandas as pd  # Primary data frame tool used here to load completed Parquet files
import numpy as np  # Array structures used to catch float distortions in JSON outputs
from datetime import datetime  # Date timestamp tools to partition task folders cleanly
from fastapi import FastAPI, BackgroundTasks  # Core asynchronous web router frameworks
from pydantic import BaseModel  # Strict validation structures for typed input blocks
from typing import Dict, Any, List  # Explicit type annotations for clean system compilation

# 🟢 CORRECTED: Use absolute imports starting from 'services'
from services import data_generator  # Step 1: Population Creator
from services import metrics_calculator  # Step 2: Combinatorial Matrix Engine
from services import analytics_ranker  # Step 3: Lightweight Rank Summary Engine

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

