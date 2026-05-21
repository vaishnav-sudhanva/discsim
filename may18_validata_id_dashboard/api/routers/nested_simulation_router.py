import os
import sys
import uuid
import pandas as pd
from fastapi import APIRouter, BackgroundTasks, HTTPException
from typing import Dict, Any

# Dynamic Pathing to find the Engine
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
API_DIR = os.path.dirname(CURRENT_DIR)
if API_DIR not in sys.path:
    sys.path.append(API_DIR)

from services.nested_simulation_engine import run_custom_simulation

router = APIRouter()

# Temporary database to hold the status of background jobs
TASK_STATUS = {}

def run_and_update_status(params: dict, task_id: str):
    """Background worker function that runs the engine and updates the status."""
    try:
        paths = run_custom_simulation(params, task_id)
        
        TASK_STATUS[task_id] = {
            "status": "Complete", 
            "result_path": paths["strategy_path"],
            "child_path": paths["child_path"]
        }
    except Exception as e:
        print(f"[{task_id}] ERROR: {str(e)}")
        TASK_STATUS[task_id] = {"status": "Failed", "error": str(e)}

@router.post("/start-nested-sim")
async def start_nested_sim(payload: Dict[str, Any], background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    TASK_STATUS[task_id] = {"status": "Processing", "result_path": None, "child_path": None}
    background_tasks.add_task(run_and_update_status, payload, task_id)
    return {"task_id": task_id, "status": "Processing", "message": "Simulation started."}

@router.get("/check-nested-sim/{task_id}")
async def check_nested_sim(task_id: str):
    status_info = TASK_STATUS.get(task_id)
    if not status_info:
        raise HTTPException(status_code=404, detail="Task ID not found.")
    
    if status_info["status"] == "Complete":
        try:
            df = pd.read_csv(status_info["result_path"])
            return {
                "status": "Complete", 
                "data": df.to_dict(orient="records"),
                "child_data_path": status_info["child_path"]
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to read result file: {str(e)}")
            
    return {"status": status_info["status"]}