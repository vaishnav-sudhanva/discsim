Markdown
# How to Run the Intervention Design Dashboard Locally

### 1. Clone the Repository
Download the code to your local machine and navigate directly to the dashboard folder.

```bash
git clone [https://github.com/vaishnav-sudhanva/discsim.git](https://github.com/vaishnav-sudhanva/discsim.git)
cd discsim/may18_validata_id_dashboard
2. Create and Activate a Virtual Environment
It is highly recommended to run this in a clean environment so it doesn't conflict with other Python projects on your computer.

Bash
# Create the environment
python -m venv venv

# Activate it (If you are on Windows)
venv\Scripts\activate

# Activate it (If you are on Mac/Linux)
source venv/bin/activate
3. Install Dependencies
Install all the required mathematical and UI libraries.

Bash
pip install -r requirements.txt
4. Start the Backend Math Engine (Terminal 1)
The dashboard relies on a heavy asynchronous calculation engine. You must start this server first and leave this terminal open.

Bash
cd api
uvicorn main:app --port 8005 --reload
5. Start the Streamlit Dashboard (Terminal 2)
Open a brand new terminal window, activate the virtual environment again, and launch the user interface.

Bash
# Make sure you are in the correct subfolder
cd discsim/may18_validata_id_dashboard

# Activate the environment again (Windows)
venv\Scripts\activate  
# (Or use 'source venv/bin/activate' for Mac/Linux)

# Navigate to the UI folder and run the app
cd dashboard/src/utils/intervention_design
streamlit run nested_simulation_ui.py
Streamlit will automatically open a browser tab at http://localhost:8501. Because you started the backend engine on port 8005, the dashboard will connect instantly and you are ready to go!
