# Intervention Design Dashboard: Setup & Execution Guide

This guide provides the exact steps to download, install, and run the Nested Simulation Dashboard and its backend math engine locally. 

### Step 1: Download Only the Dashboard Folder (Sparse Checkout)
Because this dashboard is part of a larger repository, use Git's sparse-checkout to download *only* the required files. Open your terminal and run these commands one by one:

```bash
git clone --filter=blob:none --no-checkout https://github.com/vaishnav-sudhanva/discsim.git
cd discsim
git sparse-checkout set may18_validata_id_dashboard
git checkout main
cd may18_validata_id_dashboard
```

### Step 2: Create and Activate a Virtual Environment
Create an isolated environment to install the required mathematical and UI libraries.

**For Windows:**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**For Mac / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```
*(You should now see `(venv)` at the start of your terminal prompt).*

### Step 3: Install Dependencies
With the virtual environment active, install all required packages:
```bash
pip install -r requirements.txt
```

### Step 4: Start the Backend Math Engine (Terminal 1)
The dashboard relies on an asynchronous FastAPI backend to crunch the heavy simulations. You must start this server first and **leave this terminal open**.
```bash
cd api
uvicorn main:app --port 8005 --reload
```
*Wait until you see `Application startup complete.` before moving to the next step.*

### Step 5: Start the Streamlit Dashboard (Terminal 2)
Open a **new, second terminal window**. Navigate back to the project folder, activate the virtual environment again, and launch the user interface.

**For Windows:**
```powershell
cd C:\Users\CEGIS\discsim\may18_validata_id_dashboard
.\venv\Scripts\activate  
cd dashboard\src\utils\intervention_design
streamlit run nested_simulation_ui.py
```

**For Mac / Linux:**
```bash
cd path/to/discsim/may18_validata_id_dashboard
source venv/bin/activate
cd dashboard/src/utils/intervention_design
streamlit run nested_simulation_ui.py
```

---
**Success!** Streamlit will automatically open a browser tab at `http://localhost:8501`. Since your backend is running on port 8005, the dashboard is fully connected and ready to run simulations.
