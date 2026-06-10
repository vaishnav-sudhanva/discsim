import uuid
import pandas as pd
from fastapi import APIRouter, BackgroundTasks, HTTPException
from typing import Dict, Any

# 🟢 1. No more path hacks!
# 🟢 2. Import the worker function AND the database directly from the engine
from services.nested_simulation_engine import run_simulation_task, tasks_db, clean_json_records

router = APIRouter()

@router.post("/start-nested-sim")
async def start_nested_sim(payload: Dict[str, Any], background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    
    # Put the initial processing status directly into the engine's database
    tasks_db[task_id] = {"status": "Processing"}
    
    # Start the engine's background worker
    background_tasks.add_task(run_simulation_task, payload, task_id)
    
    return {"task_id": task_id, "status": "Processing", "message": "Simulation started."}

@router.get("/check-nested-sim/{task_id}")
async def check_nested_sim(task_id: str):
    # Fetch the status directly from the engine's database
    status_info = tasks_db.get(task_id)
    
    if not status_info:
        raise HTTPException(status_code=404, detail="Task ID not found.")
    
    # If the engine finished the job successfully, read the files!
    if status_info["status"] == "Complete":
        try:
            # Read the CSV file the engine created
            df = pd.read_csv(status_info["strategy_path"])
            
            # Clean it so it doesn't crash the web browser (using the engine's function)
            clean_records = clean_json_records(df)
            
            # 🟢 THE SWISS ARMY KNIFE PAYLOAD: Guarantees the UI finds its data!
            return {
                "status": "Complete", 
                "data": clean_records,                                 # Primary data key
                "l1_summary": clean_records,                           # Fallback data key
                "child_data_path": status_info.get("child_data_path"), # Primary path key
                "result_path": status_info.get("strategy_path"),       # Fallback path key
                "final_csv_path": status_info.get("strategy_path")     # Fallback path key
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to read result file: {str(e)}")
            
    # If it is still 'Processing' or 'Failed', just return that status
    return {"status": status_info["status"]}