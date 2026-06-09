import os
import time
from datetime import datetime
import streamlit as st
import pandas as pd
import requests
import plotting_engine
from ui_text_config import TOOLTIPS

API_BASE_URL = "http://localhost:8005"

# ==============================================================================
# 1. DIRECTORY SETUP & PRESET CONFIGURATIONS
# ==============================================================================
base_output_dir = r"C:\Users\CEGIS\Documents\GitHub\discsim\may18_validata_id_dashboard\outputs"
preset_output_dir = os.path.join(base_output_dir, "Precalculated_Presets")
MASTER_REGISTRY_PATH = os.path.join(base_output_dir, "simulation_master_registry.csv")

PRESET_MAPPING = {
    # "Select a Preset...": None,
    "Custom Scenario (Live Generation)": None,
    # "Good L1, Good L0": "Good_L0_Good_L1_20260526_063027_Calc_1sims",
    # "Good L1, Bad L0": "Good_L0_Bad_L1_20260526_063458_Calc_1sims",
    # "Bad L1, Good L0": "Bad_L0_Good_L1_20260526_063931_Calc_1sims",
    # "Bad L1, Bad L0": "Bad_L0_Bad_L1_20260526_064358_Calc_1sims",
    #  "Good L1 Good L0 V1": "Good_L0_Good_L1_20260603_173812_Step2_Iter_Eval_20260603_190640",
    #     "Bad L1 Good L0 V1": "Good_L0_Bad_L1_20260603_221306_Step2_Iter_Eval_20260604_075210",
    # "Good L1 Bad L0 V1": "Bad_L0_Good_L1_20260604_001321_Step2_Iter_Eval_20260604_082126",
      "Bad L1 Bad L0 V1": "Bad_L0_Bad_L1_20260608_213154_Step2_Iter_Eval_20260608_213214",
    #   "Bad L1 Bad L0 V2": "Bad_L0_Bad_L1_20260606_225327_Step2_Iter_Eval_20260606_225408"
}

def apply_preset():
    """Reads the Registry CSV and snaps the UI sliders to match the exact physical parameters."""
    selected_name = st.session_state.preset_dropdown
    task_id = PRESET_MAPPING.get(selected_name)
    
    if task_id is None:
        st.session_state["simulation_results"] = None
        return 
        
    try:
        registry = pd.read_csv(MASTER_REGISTRY_PATH)
        
        # Look for the exact calculated run row first
        matching_rows = registry[registry['Task_ID'] == task_id]
        
        # Fallback to base blueprint ID if needed
        if matching_rows.empty:
            base_id = task_id.split("_Calc_")[0]
            matching_rows = registry[registry['Task_ID'] == base_id]
            
        if matching_rows.empty:
            st.error(f"Could not find {task_id} inside your CSV Registry.")
            return
            
        row = matching_rows.iloc[-1]
        
        # Complete parameter mapping block
        ui_keys = [
            "n_L1s", "n_L0s_per_L1", "n_children_per_L0", 
            "real_percent_stunting", "real_percent_underweight",
            "mean_collusion_index", "mean_percent_copy",
            "error_sd_height_all_L0s", "error_sd_weight_all_L0s",
            "mean_percent_under_reporting_stunting", "mean_percent_under_reporting_underweight",
            "sd_across_units_percent_under_reporting_stunting", "sd_across_units_percent_under_reporting_underweight",
            "sd_within_units_percent_under_reporting_stunting", "sd_within_units_percent_under_reporting_underweight",
            "sd_across_units_bunch_factor_haz", "sd_across_units_bunch_factor_waz",
            "sd_within_units_bunch_factor_haz", "sd_within_units_bunch_factor_waz",
            "sd_percent_copy", "sd_collusion_index", 
            "mean_time_lag_L1", "mean_time_lag_L2",
            "n_simulations_used"
        ]
        
        int_keys = ["n_L1s", "n_L0s_per_L1", "n_children_per_L0", "mean_time_lag_L1", "mean_time_lag_L2", "n_simulations_used"]
        
        # Unified type-casting loop with a protective string safeguard filter
        for key in ui_keys:
            if key in row and pd.notna(row[key]):
                val = str(row[key]).strip()
                if val.lower() != "pending" and val != "":
                    if key in int_keys:
                        st.session_state[key] = int(data_type_parser(val))
                    else:
                        st.session_state[key] = float(val)
                
        # Load evaluated metrics table directly into frontend memory
        eval_path = os.path.join(preset_output_dir, f"Tracer_Master_DB_{task_id}.csv")
        if os.path.exists(eval_path):
            st.session_state["simulation_results"] = pd.read_csv(eval_path)
            st.session_state["sim_params_memory"] = {
                "has_l2": st.session_state.get("has_l2", "Yes"),
                "output_variable": st.session_state.get("output_variable", "Height"),
                "target_percentile": st.session_state.get("target_percentile", 0.30),
                "n_L0s_per_L1": int(st.session_state.get("n_L0s_per_L1", 25)),
                "n_children_per_L0": int(st.session_state.get("n_children_per_L0", 15))
            }
        else:
            st.error(f"Missing evaluated metrics file at: {eval_path}")
            
    except Exception as e:
        st.error(f"Failed to read CSV parameters: {e}")

def data_type_parser(val):
    """Safely converts string floats to float before integer casting to preserve formatting."""
    try:
        return float(val)
    except ValueError:
        return 0.0

def log_simulation_run(params_dict, universe_file, results_file, registry_path="simulation_run_registry.csv"):
    """Logs individual custom run metrics into a separate logging CSV."""
    new_row = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Universe_Data_File": universe_file,
        "Calculated_Results_File": results_file
    }
    new_row.update(params_dict)
    
    df_new = pd.DataFrame([new_row])
    if os.path.exists(registry_path):
        df_existing = pd.read_csv(registry_path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(registry_path, index=False)
    else:
        df_new.to_csv(registry_path, index=False)

# ==============================================================================
# 2. CORE RENDERING ENGINE
# ==============================================================================
def render_nested_simulation_ui():
    st.set_page_config(layout="wide", page_title="Intervention Design")
    st.markdown("<h1 style='text-align: center; color: #2c3e50;'>Intervention Design: Nested Simulation Model</h1>", unsafe_allow_html=True)
    
    # 🟢 NEW: Comprehensive Framework Description
    # 🟢 NEW: Comprehensive Framework Description
    with st.expander("Read the Framework Documentation (How this Module Works)", expanded=False):
        st.markdown("""
        ### Overview of the Intervention Design
        This module shows the results of a "nested supervision" model. In this system, Anganwadi workers (L0) collect data of each child in their Anganwadi Centre, Block / District Level supervisors (L1) check the L0 Anganwadi workers, and auditors (L2) check the L1 Supervisor. The main goal of this module is to test the quality of the administrative health data collected at both the L0 and L1 levels.

        ### Step 1: Simulating the Population (The Universe)
        To test how well the supervision model works, we first generate a synthetic universe of children based on your input parameters. This generation happens in three parts:
        * **Population Details:** We define the basic structure, including the total number of Children, L0 workers, and L1 supervisors, as well as the Height & Weight (in terms of Z score), Age, Gender of each child.
        * **Nutritional Reality:** We set the true health metrics of the population, such as the actual percentage of children who are underweight, stunted, or wasted.
        * **Behavior and Distortions:** We simulate real-world human behavior and mistakes. This includes normal measurement error, data drift, copying old data, and active collusion at the L0 and L1 levels.

        **The Output Data:** At the end of the simulation, we get a dataset that shows the *real* biological height, weight, and age of the kids, right next to the *measured* height and weight (calculated in Z-scores) recorded by the L0, L1, and L2 workers.

        ### Step 2: The Nested Supervision Process
        Once the fake population is created, the calculation and sampling phase begins. 
        * An **L1 supervisor** audits an equal number of samples from each of their L0 workers. 
        * If an **L2 auditor** is present, they will take an equal number of samples from the exact children already measured by the L1 supervisor. 

        This creates a nested chain of accountability. We use the L1's sample data to rank the performance of the L0 workers, and we use the L2's sample data to rank the performance of the L1 supervisors. This chain allows us to check if the data is being measured correctly at the ground level.

        ### Step 3: Understanding Ranking Accuracy
        The system grades the success of these audits using a metric called "Ranking Accuracy." Here is a simple example of how it works:

        Imagine an L1 supervisor manages **25 L0 workers**. We ask our model to identify the worst-performing 25% of those workers. Out of 25, the worst 25% equals **5 workers**.
        1. The L1 supervisor uses their sample data to give us a list of the 5 workers they *think* are the worst. 
        2. We check their list against the absolute truth from our simulation. 
        3. If 4 out of the 5 workers on the supervisor's list are *actually* the worst, the ranking accuracy is 4 out of 5, which equals **80%**.

        ### The Final Goal: The Optimal Strategy
        By testing these ranking accuracies across many different scenarios, this module provides you with the optimal sampling strategy. It tells you exactly how many clinics and children you need to sample to get the maximum ranking efficiency without wasting your budget.
        """)
    st.markdown("---")
    
    if "simulation_results" not in st.session_state:
        st.session_state["simulation_results"] = None
    if "sim_params_memory" not in st.session_state:
        st.session_state["sim_params_memory"] = {}
        
    # Baseline configuration fallbacks to keep UI memory fields populated on cold boot
    default_boots = {
        "n_L1s": 100, "n_L0s_per_L1": 25, "n_children_per_L0": 15,
        "real_percent_stunting": 36.0, "real_percent_underweight": 34.0,
        "sd_across_units_percent_under_reporting_stunting": 2.0, "sd_across_units_percent_under_reporting_underweight": 2.0,
        "sd_within_units_percent_under_reporting_stunting": 1.0, "sd_within_units_percent_under_reporting_underweight": 1.0,
        "sd_across_units_bunch_factor_haz": 0.01, "sd_across_units_bunch_factor_waz": 0.01,
        "sd_within_units_bunch_factor_haz": 0.01, "sd_within_units_bunch_factor_waz": 0.01,
        "sd_percent_copy": 2.0, "sd_collusion_index": 0.02,
        "mean_time_lag_L1": 15, "mean_time_lag_L2": 30, "n_simulations_used": 1
    }
    for k, v in default_boots.items():
        if k not in st.session_state:
            st.session_state[k] = v

    p = {}

    # ==============================================================================
    # 3. ARCHITECTURE & TOGGLES
    # ==============================================================================
# 🟢 HARDCODE Architecture settings that we removed from the UI
    p["has_l2"] = "Yes"
    p["target_percentile"] = 0.30

    # ==============================================================================
    # 4. SIMULATION PARAMETERS
    # ==============================================================================
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 2. Simulation Parameters")
    
    st.selectbox(
        "Load a Preset Scenario or Generate Data According to your Requirement", 
        list(PRESET_MAPPING.keys()),
        key="preset_dropdown", 
        on_change=apply_preset
    )

    st.markdown("---")
    
    # ==========================================
    # POPULATION PARAMETERS SUBSECTION
    # ==========================================
    st.markdown("#### Population Parameters")
    
    pop_col1, pop_col2 = st.columns(2)
    with pop_col1:
        p["n_L1s"] = st.number_input("Number of L1 Supervisor", min_value=1, max_value=500, key="n_L1s", step=1, help=TOOLTIPS.get("n_L1s", ""))
        p["n_L0s_per_L1"] = st.number_input("Number of L0 Anganwadi Center ", min_value=1, max_value=100, key="n_L0s_per_L1", step=1, help=TOOLTIPS.get("n_L0s_per_L1", ""))
        p["n_children_per_L0"] = st.number_input("Number of Children Per L0 (AWC)", min_value=1, max_value=500, key="n_children_per_L0", step=1, help=TOOLTIPS.get("n_children_per_L0", ""))
    with pop_col2:
        p["real_percent_stunting"] = st.number_input("Stunting Percentage of The Population", min_value=0.0, max_value=100.0, key="real_percent_stunting", step=1.0, help=TOOLTIPS.get("real_percent_stunting", ""))
        p["real_percent_underweight"] = st.number_input("Underweight Percentage of The population", min_value=0.0, max_value=100.0, key="real_percent_underweight", step=1.0, help=TOOLTIPS.get("real_percent_underweight", ""))

    st.markdown("<br>", unsafe_allow_html=True)

    # ==========================================
    # DISTORTION PARAMETERS SUBSECTION
    # ==========================================
    st.markdown("####  Distortion Parameters")
    
    dist_col1, dist_col2 = st.columns(2)
    with dist_col1:
        p["mean_collusion_index"] = st.slider("Collusion Between (L1 & L0)", 0.0, 1.0, key="mean_collusion_index", help=TOOLTIPS.get("mean_collusion_index", ""))
        p["mean_percent_copy"] = st.slider("Percentage of Data Copied by L1 ", 0.0, 100.0, key="mean_percent_copy", help=TOOLTIPS.get("mean_percent_copy", ""))
        p["error_sd_height_all_L0s"] = st.slider("Measurement Error (Cm/Kg)", 0.0, 3.0, key="error_sd_height_all_L0s", help=TOOLTIPS.get("error_sd_height_all_L0s", ""))
        p["error_sd_weight_all_L0s"] = p["error_sd_height_all_L0s"] * 0.1

    with dist_col2:
        p["mean_percent_under_reporting_stunting"] = st.slider("Percentage of Under Reporting Stunting (Height) in Each L0", 0.0, 100.0, key="mean_percent_under_reporting_stunting", help=TOOLTIPS.get("mean_percent_under_reporting_stunting", ""))
        p["mean_percent_under_reporting_underweight"] = st.slider("Percentage of Under reporting Underweight (Weight) in Each L0", 0.0, 100.0, key="mean_percent_under_reporting_underweight", help=TOOLTIPS.get("mean_percent_under_reporting_underweight", ""))
   
    # ==============================================================================
    # 5. ADVANCED SIMULATION EXPANSER PARAMETERS
    # ==============================================================================
    with st.expander("Expand Advanced Simulation Parameters (For Power Users)", expanded=False):
        exp_col1, exp_col2 = st.columns(2)
        
        with exp_col1:
            st.markdown("**Fraud Variances (SD)**")
            p["sd_across_units_percent_under_reporting_stunting"] = st.number_input("SD of Under Reporting Stunting (Height) Across L1", key="sd_across_units_percent_under_reporting_stunting", help=TOOLTIPS.get("sd_across", ""))
            p["sd_across_units_percent_under_reporting_underweight"] = st.number_input("SD of Under Reporting Underweight (Weight) Across L1", key="sd_across_units_percent_under_reporting_underweight", help=TOOLTIPS.get("sd_across", ""))
            p["sd_within_units_percent_under_reporting_stunting"] = st.number_input("SD of Under Reporting Stunting (Height) Within L1", key="sd_within_units_percent_under_reporting_stunting", help=TOOLTIPS.get("sd_within", ""))
            p["sd_within_units_percent_under_reporting_underweight"] = st.number_input("SD of Under Reporting Underweight (Weight) Within L1", key="sd_within_units_percent_under_reporting_underweight", help=TOOLTIPS.get("sd_within", ""))
            p["sd_across_units_bunch_factor_haz"] = st.number_input("SD Across L1 Bunch Factor (HAZ)", key="sd_across_units_bunch_factor_haz", help=TOOLTIPS.get("bunch_factor", ""))
            p["sd_across_units_bunch_factor_waz"] = st.number_input("SD Across L1 Bunch Factor (WAZ)", key="sd_across_units_bunch_factor_waz", help=TOOLTIPS.get("bunch_factor", ""))

        with exp_col2:
            st.markdown("**Misc Variances & Lags**")
            p["sd_within_units_bunch_factor_haz"] = st.number_input("SD Within L1 Bunch Factor (HAZ)", key="sd_within_units_bunch_factor_haz", help=TOOLTIPS.get("bunch_factor", ""))
            p["sd_within_units_bunch_factor_waz"] = st.number_input("SD Within L1 Bunch Factor (WAZ)", key="sd_within_units_bunch_factor_waz", help=TOOLTIPS.get("bunch_factor", ""))
            p["sd_percent_copy"] = st.number_input("SD Data Copy By L1", key="sd_percent_copy", help=TOOLTIPS.get("sd_copy", ""))
            p["sd_collusion_index"] = st.number_input("SD Collusion Between (L0 & L1)", key="sd_collusion_index", help=TOOLTIPS.get("sd_collusion", ""))
            p["mean_time_lag_L1"] = st.number_input("Number of Days Before L1 Samples", key="mean_time_lag_L1", step=1, help=TOOLTIPS.get("time_lag", ""))
            p["mean_time_lag_L2"] = st.number_input("Number of Days Before L2 Samples", key="mean_time_lag_L2", step=1, help=TOOLTIPS.get("time_lag", ""))
            
            st.markdown("---")
            st.markdown("**Engine Configuration**")
            p["n_simulations"] = st.number_input("Number of Simulations", min_value=1, max_value=30, key="n_simulations_used", step=1, help=TOOLTIPS.get("n_simulations", "")) 
        
    # ==============================================================================
    # 6. DUAL RUN INTERACTIVE PIPELINE
    # ==============================================================================
    
    # ==============================================================================
    # 6. DUAL RUN INTERACTIVE PIPELINE
    # ==============================================================================
    selected_preset = st.session_state.get("preset_dropdown", "Select a Preset...")
    
    if selected_preset == "Custom Scenario (Live Generation)":
        if st.button("Run Heavy Custom Simulation Locally", type="primary", use_container_width=True):
            status_text = st.empty()
            progress_bar = st.progress(0)
            
            try:
                status_text.info("Sending simulation parameters to the backend engine...")
                
                # 1. Trigger the asynchronous background task
                response = requests.post(f"{API_BASE_URL}/start-nested-sim", json=p)
                
                if response.status_code == 200:
                    task_data = response.json()
                    task_id = task_data["task_id"]
                    status_text.warning("Engine is crunching the simulation... This may take a few minutes depending on Universe size.")
                    progress_bar.progress(10)
                    
                    # 2. Polling Loop to check on the math
                    while True:
                        time.sleep(3) # Check every 3 seconds to not spam the server
                        check_resp = requests.get(f"{API_BASE_URL}/check-nested-sim/{task_id}")
                        
                        if check_resp.status_code == 200:
                            status_data = check_resp.json()
                            
                            if status_data["status"] == "Complete":
                                progress_bar.progress(100)
                                status_text.success("Simulation Complete! Loading data into the dashboard...")
                                
                                # 3. Extract the clean data from the API response
                                # Your Step 3 engine now outputs L1 data explicitly
                                clean_records = status_data.get("l1_summary", []) 
                                
                                if clean_records:
                                    new_df = pd.DataFrame(clean_records)
                                    
                                    # Ensure integers are cast properly for Matplotlib
                                    if 'L1_Budget_Pct' in new_df.columns:
                                        new_df['L1_Pct_Num'] = new_df['L1_Budget_Pct'].astype(str).str.replace('%', '').astype(int)
                                    if 'L2_Budget_Pct' in new_df.columns:
                                        new_df['L2_Pct_Num'] = new_df['L2_Budget_Pct'].astype(str).str.replace('%', '').astype(int)
                                        
                                    st.session_state["simulation_results"] = new_df
                                    st.session_state["sim_params_memory"] = p.copy()
                                    time.sleep(1) # Give the user a second to read the success message
                                    st.rerun() # Instantly reload the UI with the fresh data
                                    break
                                else:
                                    status_text.error("Simulation finished but returned empty data.")
                                    break
                                    
                            elif status_data["status"] == "Failed":
                                status_text.error(f"Engine Failed: {status_data.get('error', 'Unknown Error')}")
                                break
                            else:
                                # Keep spinning the bar to show it's alive
                                progress_bar.progress(50) 
                        else:
                            status_text.error("Lost connection to the backend engine during polling.")
                            break
                else:
                    status_text.error(f"Failed to start simulation. Error {response.status_code}: {response.text}")
                    
            except Exception as e:
                st.error(f"Could not connect to the Backend API. Ensure Uvicorn is running on port 8005. Error: {str(e)}")


    # ==============================================================================
    # 7. RENDER FILTERS AND PLOTS FROM SCENARIO RESULTS
    # ==============================================================================
    if st.session_state["simulation_results"] is not None:
        strategy_df = st.session_state["simulation_results"]
        
        # 🟢 Cast integers instantly before we render, right out of the session state
        if 'L1_Pct_Num' not in strategy_df.columns:
            strategy_df['L1_Pct_Num'] = strategy_df['L1_Budget_Pct'].str.replace('%', '').astype(int)
            strategy_df['L2_Pct_Num'] = strategy_df['L2_Budget_Pct'].str.replace('%', '').astype(int)
            st.session_state["simulation_results"] = strategy_df # Save it back so it doesn't recalculate

        mem_p = st.session_state.get("sim_params_memory", {})

        st.markdown("---")
        st.markdown("### Interactive Plot Filters")
        st.info("The data has been pre-calculated. Use these filters to instantly update the charts below.")

        # 🟢 Reduced to 4 columns since we removed the L1 Budget filter!
        filter_col1, filter_col2, filter_col3, filter_col4 = st.columns([2, 2, 2, 1])
        budget_options = [f"{i}0%" for i in range(1, 11)]
        
        if 'Target_Percentile' in strategy_df.columns:
            target_options = sorted(strategy_df['Target_Percentile'].unique().tolist(), key=lambda x: int(x.replace('%', '')))
            default_target = "30%" if "30%" in target_options else target_options[0]
        else:
            target_options = ["30%"]
            default_target = "30%"
        
        with filter_col1:
            ui_l2_budget = st.selectbox("L2 Budget Filter (Applies to Heatmap)", budget_options, index=3, help=TOOLTIPS.get("l2_budget", ""))
        with filter_col2:
            ui_target_pct = st.selectbox("Target Catch Rate", target_options, index=target_options.index(default_target))
        with filter_col3:
            ui_indicator = st.selectbox("Indicator", ["Height", "Weight", "Both"], index=0)
        with filter_col4:
            st.markdown("<br>", unsafe_allow_html=True)
            csv_data = strategy_df.to_csv(index=False).encode('utf-8')
            st.download_button(label="Download Data", data=csv_data, file_name='simulation_results.csv', mime='text/csv', use_container_width=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        # # 🟢 The Plot Selector Radio Button (Stops Streamlit from freezing)
        # plot_options = [
        #     "All Plots",
        #     # "1. L1 Sensitivity (God-Mode V1)",
        #     "L1 Sampling Strategy",
        #     # "3. Breadth/Depth Optimization",
        #     # "4. Auditor Robustness",
        #     # "5. L2 Breadth Grid",
        #     "L2 Sampling Strategy"
        # ]
        # selected_plot = st.radio("Select Plot to Display:", plot_options, horizontal=True)
        # st.markdown("---")

        # 🟢 The Plot Selector Multi-Select (Order matters!)
        plot_options = [
            # "0. L1 Budget Optimization (Elbow Curve)",
            # "1. L1 Sensitivity (God-Mode V1)",
            # "2. L0 Intra-Regional (Clinic Catch V2)",
            # "3. Breadth/Depth Optimization",
            # "4. Auditor Robustness",
            # "5. L2 Breadth Grid",
            # "6. Heatmap Strategy Match"
                "L1 Sampling Strategy",
            # "3. Breadth/Depth Optimization",
            # "4. Auditor Robustness",
            # "5. L2 Breadth Grid",
            "L2 Sampling Strategy"
        ]
        
        selected_plots = st.multiselect(
            "Select the plots you want to view (Order of selection determines display order):",
            options=plot_options,
            default=[ "L1 Sampling Strategy","L2 Sampling Strategy"]
        )
        st.markdown("---")

        # Map UI indicator selection to the loop
        if ui_indicator.lower() == "both": target_inds = ["Height", "Weight"]
        elif ui_indicator.lower() == "weight": target_inds = ["Weight"]
        else: target_inds = ["Height"]

        # # Math for descriptions
        # total_kids_per_l1 = mem_p.get("n_L0s_per_L1", 25) * mem_p.get("n_children_per_L0", 15)
        # l1_pct_float = int(ui_l1_budget.replace("%", "")) / 100.0
        # actual_l1_budget = int(total_kids_per_l1 * l1_pct_float)
        
        # 🟢 Filter the dataframe to a single target threshold BEFORE plotting
        if 'Target_Percentile' in strategy_df.columns:
            active_df = strategy_df[strategy_df['Target_Percentile'] == ui_target_pct].copy()
        else:
            active_df = strategy_df.copy()

        # Dynamically pull the actual universe size from memory
        current_uni = active_df['Universe'].unique().tolist()
        total_clinics = int(mem_p.get("n_L0s_per_L1", 25))
        total_kids = int(mem_p.get("n_children_per_L0", 15))

        # 🟢 START LOOP
        # 🟢 START LOOP
        for ind in target_inds:
            st.markdown(f"<h2 style='text-align: center; color: #2c3e50; margin-top: 20px;'>=== RESULTS FOR {ind.upper()} ===</h2>", unsafe_allow_html=True)
            df_ind = active_df[active_df['Indicator'] == ind].copy()

            if not selected_plots:
                st.info("👆 Please select at least one plot from the dropdown above.")
                continue

            for plot_name in selected_plots:
                
                if plot_name == "0. L1 Budget Optimization (Elbow Curve)":
                    st.markdown("#### Plot 0: L1 Budget Optimization (Find the Elbow)")
                    chart = plot_l1_budget_optimization(df_ind)
                    st.altair_chart(chart, use_container_width=True)
                    st.info("**What is this?** This shows the 'Elbow' where diminishing returns start. It plots the overall L1 Budget against the Clinic Catch Rate (V2 Accuracy). The mathematically optimal budget is the point where the curve starts to flatten out.")

                elif plot_name == "1. L1 Sensitivity (God-Mode V1)":
                    st.markdown(f"#### Plot 1: L1 Sensitivity (Targeting Top {ui_target_pct})")
                    fig1 = plotting_engine.plot_1_sensitivity(df_ind, 'V1_MAE_Acc', 'V1 MAE God-Mode', current_uni, ui_target_pct, total_kids)
                    if fig1: 
                        st.pyplot(fig1)
                        try:
                            max_acc = df_ind['V1_MAE_Acc'].max()
                            st.info(f"**What is this?** This plot displays the 'God-Mode' absolute baseline. It shows the maximum physical capability of a Supervisor (L1) to catch the worst **{ui_target_pct}** of fraud. As budgets increase, the variance (shaded area) shrinks. The absolute peak accuracy achieved in this simulation is **{max_acc:.1f}%**.")
                        except: pass
                    
                
                elif plot_name == "L1 Sampling Strategy":
                    st.markdown(f"#### Plot 1: L1 Sampling Strategy (Targeting Top {ui_target_pct})")
            
            # 1. Draw the Plot
                    fig2 = plotting_engine.plot_2_intra_regional(df_ind, 'V2_MAE_Acc', 'V2 MAE Clinic Score', current_uni, ui_target_pct, total_clinics, total_kids)
            
                    if fig2: 
                        st.pyplot(fig2)
                
                # 2. Dynamic Optimal Strategy Calculation
                # Find the lowest sample size that hits 85% accuracy, or max possible if it never hits 85%
                        try:
                            uni_data = df_ind[df_ind['Universe'].isin(current_uni)].groupby('L1_Pct_Num')['V2_MAE_Acc'].mean().reset_index()
                            uni_data = uni_data.sort_values('L1_Pct_Num')
                    
                            meets_target = uni_data[uni_data['V2_MAE_Acc'] >= 85.0]
                            if not meets_target.empty:
                                opt_pct = meets_target.iloc[0]['L1_Pct_Num']
                                opt_acc = meets_target.iloc[0]['V2_MAE_Acc']
                            else:
                                opt_pct = uni_data.loc[uni_data['V2_MAE_Acc'].idxmax()]['L1_Pct_Num']
                                opt_acc = uni_data['V2_MAE_Acc'].max()
                        
                            opt_kids = int(round((opt_pct / 100) * total_kids))
                    
                    # Print the Dynamic Recommendation
                            st.success(f"### Recommended L1 Strategy \n"
                                       f"To accurately rank L0 Anganwadi Centers, the L1 Supervisor should sample **{int(opt_pct)}% ({opt_kids} children)** per L0. "
                                       f"This guarantees an average accuracy of **{opt_acc:.1f}%** in catching the worst offenders. Sampling more children beyond this point yields diminishing returns.")
                        except Exception as e:
                            pass # Failsafe in case data shape changes

                # 3. Detailed Explanation & Calculation Steps
                        with st.expander("How to Interpret This Plot & Calculations", expanded=True):
                            st.markdown(f"""
                            This plot answers the critical operational question: *"How many children does a Supervisor (L1) need to sample per Anganwadi Center (L0) to accurately identify the worst-performing Anganwadi (The Anganwadi that produce very high false reports of child health) in their region?"*
                    
                            * **The X-Axis (Bottom):** Shows the L1 Supervisor's sampling strategy: The percentage (and exact number) of children sampled per L0.
                            * **The Y-Axis (Left):** Shows the Ranking Accuracy: The percentage of the truly "Worst" L0 Centers that the L1 Supervisor successfully caught at that sample size.
                    
                            **How We Calculated This (Step-by-Step):**
                            1. **Simulate the Audit:** We let the Supervisor (L1) sample a given percentage (X) of children from each Anganwadi Center (L0).
                            2. **Supervisor's Perceived Ranking:** For that specific sample, we calculate the Mean Absolute Error between the L0 Centers reported data and the supervisor's measurement `MAE(L0 - L1)`. A large MAE means the L0's data is highly distorted. We rank this list of L0 in descending order (worst offenders at the top).
                            3. **The "True" Ranking:** We calculate the population's true error `MAE(Real - L0)` using the Real Simulated biological data, and arrange the L0 in descending order. This is our absolute ground truth of who the worst offenders are.
                            4. **Evaluate Accuracy (Overlap):** We now have two lists. We apply the target filter (e.g., Top {ui_target_pct}). We look at the worst {ui_target_pct} of L0 from *both* lists and check how many overlap. 
                            5. **Final Score:** If 9 out of 10 targeted L0 match between the Supervisor's list and the True list, we determine the ranking accuracy for that sample size is 90%.
                            """)


                elif plot_name == "3. Breadth/Depth Optimization":
                    st.markdown(f"#### Plot 3: Breadth/Depth Optimization (L1 Budget: {ui_l1_budget})")
                    fig3 = plotting_engine.plot_3_bd_optimization(df_ind, 'V1_MAE_Acc', 'V1 MAE God-Mode', current_uni, ui_l1_budget, ui_target_pct, total_clinics)
                    if fig3: 
                        st.pyplot(fig3)
                        try:
                            subset = df_ind[df_ind['L1_Budget_Pct'] == ui_l1_budget]
                            best_row = subset.loc[subset['V1_MAE_Acc'].idxmax()]
                            st.info(f"**What is this?** The Breadth vs. Depth mathematical trade-off. You are forcing the supervisor to use exactly a **{ui_l1_budget}** budget ({actual_l1_budget} total kids). To maximize accuracy under these specific constraints, the optimal strategy is: visit **{int(best_row['L1_C'])} clinics** and measure **{int(best_row['L1_K'])} children** per clinic, yielding **{best_row['V1_MAE_Acc']:.1f}%** accuracy.")
                        except: pass

                elif plot_name == "4. Auditor Robustness":
                    st.markdown("#### Plot 4: Auditor Robustness (Single Universe)")
                    if len(current_uni) > 0:
                        fig4 = plotting_engine.plot_4_robustness(df_ind, 'V1_MAE_Acc', 'V1 MAE God-Mode', current_uni[0], ui_target_pct)
                        if fig4: 
                            st.pyplot(fig4)
                            st.info("**What is this?** This plot proves the 'Bottleneck' effect. It shows the accuracy of an Independent Auditor (L2) trying to catch bad Supervisors. Because the Auditor relies on the Supervisor's spreadsheet, their ultimate accuracy is hard-capped by whatever base budget the Supervisor originally used.")

                elif plot_name == "5. L2 Breadth Grid":
                    st.markdown(f"#### Plot 5: Auditor Strategy Grid (L1 Budget Fixed at {ui_l1_budget})")
                    fig5 = plotting_engine.plot_5_master_grid(df_ind, 'V1_MAE_Acc', 'V1 MAE God-Mode', ui_l1_budget, ui_target_pct, current_uni)
                    if fig5: 
                        st.pyplot(fig5)
                        st.info(f"**What is this?** A multi-verse grid view showing how the Independent Auditor (L2) should optimize their *own* breadth (how many clinics they spot-check) given that the Supervisor below them already used a fixed **{ui_l1_budget}** sampling strategy.")

                elif plot_name == "L2 Sampling Strategy":
                    st.markdown(f"#### Plot 2: L2 Strategy Heatmap (L2 Budget Fixed at: {ui_l2_budget})")
                    if len(current_uni) > 0:
                        fig6 = plotting_engine.plot_6_heatmap(df_ind, 'V3_MAE_Acc', 'God-Mode MAE Comparison', current_uni[0], ui_l2_budget)
                        if fig6: 
                            st.pyplot(fig6)
                            try:
                                # Scan the entire heatmap (all L1s) to find the absolute peak accuracy for this L2 budget
                                df_hm = df_ind[df_ind['L2_Budget_Pct'] == ui_l2_budget]
                                best_combo = df_hm.loc[df_hm['V3_MAE_Acc'].idxmax()]
                                
                                st.success(f"**What is this?** The Ultimate Synergy Matrix. The rows show the Supervisor's fixed baseline strategy. The columns show the Auditor's flexible options at **{ui_l2_budget}** budget. "
                                           f"Across all possible combinations shown above, the peak accuracy of **{best_combo['V3_MAE_Acc']:.1f}%** is achieved when the Supervisor uses a **{best_combo['L1_Budget_Pct']}** budget, "
                                           f"and the Auditor checks **{int(best_combo['L2_C'])} clinics** by measuring **{int(best_combo['L2_K'])} kids** per clinic.")
                            except Exception as e: 
                                pass

                st.markdown("<hr style='border: 2px dashed #ccc;'>", unsafe_allow_html=True)

if __name__ == "__main__":
    render_nested_simulation_ui()