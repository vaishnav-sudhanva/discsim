import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import our new router
from routers.nested_simulation_router import router as nested_sim_router

app = FastAPI(title="VALIData Nested Simulation API")

# Add standard CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Attach the router to the app
app.include_router(nested_sim_router)

@app.get("/")
def read_root():
    return {"message": "VALIData Sandbox Backend is running perfectly on Port 8005!"}

if __name__ == "__main__":
    print("Starting Uvicorn Server on Port 8005...")
    uvicorn.run("main:app", host="0.0.0.0", port=8005, reload=True)