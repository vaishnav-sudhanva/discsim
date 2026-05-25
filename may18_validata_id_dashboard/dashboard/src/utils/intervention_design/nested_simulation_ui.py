import streamlit as st
import requests
import time
import pandas as pd
import plotting_engine
from ui_text_config import TOOLTIPS

API_BASE_URL = "http://localhost:8005"

# ---> PASTE THIS EXACTLY HERE <---
# ==============================================================================
# PRESET CONFIGURATIONS
# ==============================================================================
PRESETS = {
    "Select a Preset...": {},
    "Good L1, Good L0": {"l1_corruption_pct": 0.05, "l0_fraud_pct": 0.05, "collusion_factor": 0.10, "copy_paste_pct": 5, "equipment_error": 0.1},
    "Good L1, Bad L0": {"l1_corruption_pct": 0.05, "l0_fraud_pct": 0.75, "collusion_factor": 0.10, "copy_paste_pct": 10, "equipment_error": 0.5},
    "Bad L1, Good L0": {"l1_corruption_pct": 0.80, "l0_fraud_pct": 0.05, "collusion_factor": 0.10, "copy_paste_pct": 75, "equipment_error": 0.5},
    "Bad L1, Bad L0": {"l1_corruption_pct": 0.80, "l0_fraud_pct": 0.80, "collusion_factor": 0.90, "copy_paste_pct": 85, "equipment_error": 1.5}
}

def apply_preset():
    """Callback function: When the dropdown changes, update the slider session states."""
    selected = st.session_state.preset_dropdown
    if selected in PRESETS and PRESETS[selected]:
        for key, val in PRESETS[selected].items():
            st.session_state[key] = val
# ---> END PASTE <---

def render_nested_simulation_ui():
    st.set_page_config(layout="wide", page_title="Nested Simulation Dashboard")
    
    st.markdown("<h1 style='text-align: center;'>Nested Simulation Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Initialize Session State to hold our data so it doesn't disappear when we use filters
    if "simulation_results" not in st.session_state:
        st.session_state["simulation_results"] = None
    if "sim_params_memory" not in st.session_state:
        st.session_state["sim_params_memory"] = {}
# ---> PASTE THIS EXACTLY HERE <---
    # Initialize Core Fraud keys so the sliders have a default state to read from
    default_vals = PRESETS["Good L1, Good L0"]
    for key in default_vals.keys():
        if key not in st.session_state:
            st.session_state[key] = default_vals[key]
    # ---> END PASTE <---
    p = {}

    # ==============================================================================
    # 1. ARCHITECTURE & TOGGLES
    # ==============================================================================
    st.markdown("### 1. System Architecture & Display")
    col1, col2, col3 = st.columns(3)
    with col1: p["has_l2"] = st.radio("Include L2 Auditor?", ["Yes", "No"], help=TOOLTIPS.get("has_l2", ""))
    with col2: p["output_variable"] = st.selectbox("Indicator to Display", ["Height", "Weight", "Both"], help=TOOLTIPS.get("output_variable", ""))
    with col3:
        target_ui_val = st.slider("Target Worst % to Catch", 5, 50, 30, help=TOOLTIPS.get("target_percentile", ""))
        p["target_percentile"] = target_ui_val / 100.0  

    # # ==============================================================================
    # # 2. CORE PARAMETERS
    # # ==============================================================================
    # st.markdown("<br>", unsafe_allow_html=True)
    # st.markdown("### 2. Core Fraud & Error Parameters")
    # f_col1, f_col2 = st.columns(2)
    # with f_col1:
    #     p["l1_corruption_pct"] = st.slider("L1 Corruption Rate", 0.0, 1.0, 0.05, help=TOOLTIPS.get("l1_corruption_pct", ""))
    #     p["collusion_factor"] = st.slider("Collusion Factor", 0.0, 1.0, 0.1, help=TOOLTIPS.get("collusion_factor", ""))
    #     p["equipment_error"] = st.slider("Equipment Error (cm/kg)", 0.0, 3.0, 0.1, help=TOOLTIPS.get("equipment_error", ""))
    # with f_col2:
    #     p["l0_fraud_pct"] = st.slider("L0 Fraud Rate", 0.0, 1.0, 0.05, help=TOOLTIPS.get("l0_fraud_pct", ""))
    #     p["copy_paste_pct"] = st.slider("Copy-Paste Rate (%)", 0, 100, 5, help=TOOLTIPS.get("copy_paste_pct", ""))

# ==============================================================================
    # 2. CORE FRAUD & ERROR PARAMETERS
    # ==============================================================================
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 2. Core Fraud & Error Parameters")
    
    # ---> THE NEW PRESET DROPDOWN ON THE UI <---
    st.selectbox(
        "Load a Scenario Preset:", 
        list(PRESETS.keys()), 
        key="preset_dropdown", 
        on_change=apply_preset
    )

    f_col1, f_col2 = st.columns(2)
    with f_col1:
        # Notice we added key="..." to every slider so they link to the presets!
        p["l1_corruption_pct"] = st.slider("L1 Corruption Rate", 0.0, 1.0, key="l1_corruption_pct", help=TOOLTIPS.get("l1_corruption_pct", ""))
        p["collusion_factor"] = st.slider("Collusion Factor", 0.0, 1.0, key="collusion_factor", help=TOOLTIPS.get("collusion_factor", ""))
        p["equipment_error"] = st.slider("Equipment Error (cm/kg)", 0.0, 3.0, key="equipment_error", help=TOOLTIPS.get("equipment_error", ""))
    with f_col2:
        p["l0_fraud_pct"] = st.slider("L0 Fraud Rate", 0.0, 1.0, key="l0_fraud_pct", help=TOOLTIPS.get("l0_fraud_pct", ""))
        p["copy_paste_pct"] = st.slider("Copy-Paste Rate (%)", 0, 100, key="copy_paste_pct", help=TOOLTIPS.get("copy_paste_pct", ""))

    # ==============================================================================
    # 3. ADVANCED PARAMETERS (NOW WITH TOOLTIPS)
    # ==============================================================================
    with st.expander("⚙️ Expand Advanced Simulation Parameters (For Power Users)", expanded=False):
        exp_col1, exp_col2, exp_col3 = st.columns(3)
        with exp_col1:
            st.markdown("**Population Structure**")
            p["n_L1s"] = st.number_input("Total L1 Supervisors", value=5, help=TOOLTIPS.get("n_L1s"))
            p["n_L0s_per_L1"] = st.number_input("L0 Centers per L1", value=25, help=TOOLTIPS.get("n_L0s_per_L1"))
            p["n_children_per_L0"] = st.number_input("Children per L0", value=15, help=TOOLTIPS.get("n_children_per_L0"))
            p["real_percent_stunting"] = st.number_input("Real % Stunting", value=35.0, help=TOOLTIPS.get("real_percent_stunting"))
            p["real_percent_underweight"] = st.number_input("Real % Underweight", value=33.0, help=TOOLTIPS.get("real_percent_underweight"))
        
        with exp_col2:
            st.markdown("**Fraud Variances (SD)**")
            p["sd_across_units_percent_under_reporting_stunting"] = st.number_input("Across Units Under-Report (Stunt)", value=2.0, help=TOOLTIPS.get("sd_across"))
            p["sd_across_units_percent_under_reporting_underweight"] = st.number_input("Across Units Under-Report (Weight)", value=2.0, help=TOOLTIPS.get("sd_across"))
            p["sd_within_units_percent_under_reporting_stunting"] = st.number_input("Within Units Under-Report (Stunt)", value=1.0, help=TOOLTIPS.get("sd_within"))
            p["sd_within_units_percent_under_reporting_underweight"] = st.number_input("Within Units Under-Report (Weight)", value=1.0, help=TOOLTIPS.get("sd_within"))
            p["sd_across_units_bunch_factor_haz"] = st.number_input("Across Units Bunch Factor (HAZ)", value=0.01, help=TOOLTIPS.get("bunch_factor"))
            p["sd_across_units_bunch_factor_waz"] = st.number_input("Across Units Bunch Factor (WAZ)", value=0.01, help=TOOLTIPS.get("bunch_factor"))

        with exp_col3:
            st.markdown("**Misc Standard Deviations & Lags**")
            p["sd_within_units_bunch_factor_haz"] = st.number_input("Within Units Bunch Factor (HAZ)", value=0.01, help=TOOLTIPS.get("bunch_factor"))
            p["sd_within_units_bunch_factor_waz"] = st.number_input("Within Units Bunch Factor (WAZ)", value=0.01, help=TOOLTIPS.get("bunch_factor"))
            p["sd_percent_copy"] = st.number_input("SD Copy-Paste (%)", value=2.0, help=TOOLTIPS.get("sd_copy"))
            p["sd_collusion_index"] = st.number_input("SD Collusion Index", value=0.02, help=TOOLTIPS.get("sd_collusion"))
            p["mean_time_lag_L1"] = st.number_input("Mean Time Lag L1 (Days)", value=15, help=TOOLTIPS.get("time_lag"))
            p["mean_time_lag_L2"] = st.number_input("Mean Time Lag L2 (Days)", value=30, help=TOOLTIPS.get("time_lag"))

    st.markdown("<br>", unsafe_allow_html=True)

    # ==============================================================================
    # 4. EXECUTE SIMULATION BUTTON
    # ==============================================================================
    if st.button("🚀 Generate Data & Run Simulation on Server", type="primary", use_container_width=True):
        with st.spinner("Server is crunching numbers (Calculating all 10% to 100% budget slices)..."):
            try:
                resp = requests.post(f"{API_BASE_URL}/start-nested-sim", json=p)
                if resp.status_code != 200:
                    st.error(f"Failed to start simulation. Server returned {resp.status_code}")
                else:
                    task_id = resp.json()["task_id"]
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    while True:
                        time.sleep(2)
                        check = requests.get(f"{API_BASE_URL}/check-nested-sim/{task_id}").json()
                        
                        if check["status"] in ["Processing", "Started"]:
                            status_text.info("Generating universe and calculating matrix...")
                            progress_bar.progress(50)
                            
                        elif check["status"] == "Complete":
                            progress_bar.progress(100)
                            status_text.success("Data successfully generated!")
                            
                            # SAVE DATA TO SESSION STATE!
                            st.session_state["simulation_results"] = pd.DataFrame(check["data"])
                            st.session_state["sim_params_memory"] = p.copy()
                            time.sleep(1) # Let the user see the success message
                            st.rerun() # Refresh the page to show the results below
                            break
                            
                        elif check["status"] == "Failed":
                            status_text.error(f"Server Error! Error: {check.get('error', '')}")
                            break
            except Exception as e:
                st.error(f"Could not connect to the Backend API. Ensure uvicorn is running. Error: {str(e)}")

    # ==============================================================================
    # 5. RENDER FILTERS AND PLOTS IF DATA EXISTS IN MEMORY
    # ==============================================================================
    if st.session_state["simulation_results"] is not None:
        strategy_df = st.session_state["simulation_results"]
        mem_p = st.session_state["sim_params_memory"]

        st.markdown("---")
        st.markdown("### 📊 Interactive Plot Filters")
        st.info("The data has been pre-calculated for all budget scenarios. Use these filters to instantly update the charts below without re-running the simulation.")
        
        filter_col1, filter_col2, filter_col3 = st.columns([2, 2, 1])
        budget_options = [f"{i}0%" for i in range(1, 11)]
        
        with filter_col1:
            ui_l1_budget = st.selectbox("L1 Budget Filter (Applies to all plots)", budget_options, index=5, help=TOOLTIPS.get("l1_budget", ""))
        
        with filter_col2:
            if mem_p["has_l2"] == "Yes":
                ui_l2_budget = st.selectbox("L2 Budget Filter (Applies to Heatmap)", budget_options, index=3, help=TOOLTIPS.get("l2_budget", ""))
            else:
                st.selectbox("L2 Budget Filter", ["N/A (L2 Disabled)"], disabled=True)
                ui_l2_budget = "N/A"

        # Download Button to save results locally
        with filter_col3:
            st.markdown("<br>", unsafe_allow_html=True) # vertical alignment
            csv_data = strategy_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Data (CSV)",
                data=csv_data,
                file_name='simulation_results.csv',
                mime='text/csv',
                use_container_width=True
            )

        st.markdown("<hr>", unsafe_allow_html=True)

        # Parse which indicators to show
        if mem_p["output_variable"].lower() == "both": target_inds = ["Height", "Weight"]
        elif mem_p["output_variable"].lower() == "weight": target_inds = ["Weight"]
        else: target_inds = ["Height"]

        # Calculate absolute measurement numbers for the text
        total_kids_per_l1 = mem_p["n_L0s_per_L1"] * mem_p["n_children_per_L0"]
        l1_pct_float = int(ui_l1_budget.replace("%", "")) / 100.0
        actual_l1_budget = int(total_kids_per_l1 * l1_pct_float)
        
        actual_l2_budget = 0
        if mem_p["has_l2"] == "Yes" and ui_l2_budget != "N/A":
            l2_pct_float = int(ui_l2_budget.replace("%", "")) / 100.0
            actual_l2_budget = int(actual_l1_budget * l2_pct_float)

        # Render the Plots (this is where your current loop starts)
        for ind in target_inds:
            st.markdown(f"<h2 style='text-align: center; color: #2c3e50; margin-top: 20px;'>=== RESULTS FOR {ind.upper()} ===</h2>", unsafe_allow_html=True)
            df_ind = strategy_df[strategy_df['Indicator'] == ind]
            
            # ... (the rest of your plotting code) ...
            
            # --- CASE 1: Heatmap (Only if L2 is Yes) ---
            if mem_p["has_l2"] == "Yes":
                df_filtered_p6 = df_ind[(df_ind['L1_Budget_Pct'] == ui_l1_budget) & (df_ind['L2_Budget_Pct'] == ui_l2_budget)]
                fig6 = plotting_engine.plot_6_heatmap(df_filtered_p6, l1_pct_str=ui_l1_budget, l2_pct_str=ui_l2_budget, indicator=ind)
                if fig6: 
                    st.pyplot(fig6)
                    with st.expander(f"📖 Dynamic Insight: L1 vs L2 Synergy ({ui_l1_budget} L1 | {ui_l2_budget} L2)"):
                        st.markdown(f"**How the {ui_l1_budget} L1 and {ui_l2_budget} L2 budgets interact:**")
                        st.write("The blue matrix on the left represents how accurately the L1 supervisor is catching the worst centers based purely on their own sample. The red/green matrix on the right evaluates the L2 independent auditor.")
                        
                        if not df_filtered_p6.empty:
                            best_l2 = df_filtered_p6.loc[df_filtered_p6['V3_MAE_Acc'].idxmax()]
                            st.success(f"**Optimal L2 Strategy:** To maximize accuracy with this specific budget combination, the L2 auditor should randomly re-measure **{best_l2['L2_K']} children** across **{best_l2['L2_C']} clinics**, which yields an execution accuracy of **{best_l2['V3_MAE_Acc']:.1f}%**.")

            # --- CASE 1 & 2: Breadth vs Depth ---
            df_filtered_p3 = df_ind[df_ind['L1_Budget_Pct'] == ui_l1_budget]
            if mem_p["has_l2"] == "No": 
                df_filtered_p3 = df_filtered_p3.drop_duplicates(subset=['L1_Label'])
            
            # UPDATED: Passing the new parameters to power the dynamic plot formatting
            fig3 = plotting_engine.plot_3_breadth_depth(
                df_filtered_p3, 
                l1_pct_str=ui_l1_budget, 
                indicator=ind,
                n_L0s_per_L1=mem_p["n_L0s_per_L1"],
                target_percentile=mem_p["target_percentile"]
            )
            
            if fig3: 
                st.pyplot(fig3)
                with st.expander(f"📖 Guide & Dynamic Insight: Breadth vs Depth at {ui_l1_budget} Budget ({actual_l1_budget} Total Measurements)"):
                    # 1. Show the static guide from ui_text_config.py
                    st.markdown(TOOLTIPS.get("plot_3_breadth_depth", ""))
                    st.markdown("---")
                    
                    # 2. Show the dynamic insight
                    st.markdown(f"**⚡ Dynamic Analysis for your selected {ui_l1_budget} L1 budget:**")
                    if not df_filtered_p3.empty:
                        best_l1 = df_filtered_p3.loc[df_filtered_p3['V1_MAE_Acc'].idxmax()]
                        st.info(f"**Optimal L1 Allocation:** Based on the synthetic biological universe and fraud parameters generated, the mathematically optimal strategy at a **{ui_l1_budget} ({actual_l1_budget} measurements)** budget is to audit **{best_l1['L1_C']} clinics** and measure **{best_l1['L1_K']} children** per clinic. This peak yields a baseline accuracy of **{best_l1['V1_MAE_Acc']:.1f}%**.")
            # --- CASE 1 & 2: Intra-Regional ---
            df_filtered_p2 = df_ind.drop_duplicates(subset=['L1_Budget_Pct'])
            
            # UPDATED: Passing the population structure data into the plotting engine
            fig2 = plotting_engine.plot_2_intra_regional(
                df_filtered_p2, 
                l1_pct_str=ui_l1_budget, 
                indicator=ind,
                n_L0s_per_L1=mem_p["n_L0s_per_L1"],
                n_children_per_L0=mem_p["n_children_per_L0"],
                target_percentile=mem_p["target_percentile"]
            )
            
            if fig2: 
                st.pyplot(fig2)
                with st.expander(f"📖 Guide & Dynamic Insight: Overall Return on Investment"):
                    # 1. Show the static guide from ui_text_config.py
                    st.markdown(TOOLTIPS.get("plot_2_intra_regional", ""))
                    st.markdown("---")
                    
                    # 2. Show the dynamic insight
                    st.markdown(f"**⚡ Dynamic Analysis for your selected {ui_l1_budget} L1 budget:**")
                    if not df_filtered_p2.empty:
                        current_acc = df_filtered_p2[df_filtered_p2['L1_Budget_Pct'] == ui_l1_budget]['V2_MAE_Acc'].values[0]
                        max_acc_row = df_filtered_p2.loc[df_filtered_p2['V2_MAE_Acc'].idxmax()]
                        
                        st.warning(f"**Current Selection vs Maximum Potential:** You have currently allocated a **{ui_l1_budget} ({actual_l1_budget} measurements)** budget, which achieves an overall intra-regional accuracy of **{current_acc:.1f}%** (marked by the red star). By comparison, increasing the budget to its maximum tested limit ({max_acc_row['L1_Budget_Pct']}) would yield **{max_acc_row['V2_MAE_Acc']:.1f}%** accuracy.")
            
            st.markdown("<hr style='border: 2px dashed #ccc;'>", unsafe_allow_html=True)
            
if __name__ == "__main__":
    render_nested_simulation_ui()


# import streamlit as st
# import requests
# import time
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import matplotlib.gridspec as gridspec

# # MUST MATCH THE MAIN.PY PORT (8005)
# API_BASE_URL = "http://localhost:8005"

# DEFAULT_PARAMS = {
#     "l1_corruption_pct": 0.05, "l0_fraud_pct": 0.05, 
#     "collusion_factor": 0.10, "copy_paste_pct": 5, "equipment_error": 0.1,
#     "n_L1s": 334, "n_L0s_per_L1": 25, "n_children_per_L0": 15,
#     "real_percent_stunting": 35, "real_percent_underweight": 33,
#     "sd_across_units_percent_under_reporting_stunting": 2.0,
#     "sd_across_units_percent_under_reporting_underweight": 2.0,
#     "sd_within_units_percent_under_reporting_stunting": 1.0,
#     "sd_within_units_percent_under_reporting_underweight": 1.0,
#     "sd_across_units_bunch_factor_haz": 0.01, "sd_across_units_bunch_factor_waz": 0.01, "sd_across_units_bunch_factor_whz": 0.01,
#     "sd_within_units_bunch_factor_haz": 0.01, "sd_within_units_bunch_factor_waz": 0.01, "sd_within_units_bunch_factor_whz": 0.01,
#     "sd_percent_copy": 2.0, "sd_collusion_index": 0.02,
#     "mean_time_lag_L1": 15, "mean_time_lag_L2": 30
# }

# PRESETS = {
#     "Select a Preset...": {},
#     "Honest State (Baseline)": {}, 
#     "Corrupt Collusion": {"l1_corruption_pct": 0.60, "l0_fraud_pct": 0.50, "collusion_factor": 0.90, "copy_paste_pct": 60, "equipment_error": 0.5},
#     "Lazy Supervisors (High Copy-Paste)": {"l1_corruption_pct": 0.80, "l0_fraud_pct": 0.20, "collusion_factor": 0.20, "copy_paste_pct": 85, "equipment_error": 0.2},
#     "Bad Equipment (High Noise)": {"l1_corruption_pct": 0.10, "l0_fraud_pct": 0.10, "collusion_factor": 0.10, "copy_paste_pct": 10, "equipment_error": 1.5}
# }

# def render_nested_simulation_ui():
#     st.markdown("<h2 style='text-align: center;'>Nested Simulation Strategy Predictor</h2>", unsafe_allow_html=True)
#     st.markdown("---")

#     col1, col2 = st.columns(2)
#     with col1:
#         st.markdown("### System Architecture")
#         has_l2 = st.radio("Is a Third-Party (L2) Auditor present?", options=["Yes", "No"])
#     with col2:
#         st.markdown("### Configuration Mode")
#         config_mode = st.radio("Select Input Mode:", options=["Preset Selection", "Advanced Selection"])
#     st.markdown("---")

#     if "sim_params" not in st.session_state:
#         st.session_state.sim_params = DEFAULT_PARAMS.copy()

#     if config_mode == "Preset Selection":
#         selected_preset = st.selectbox("Choose a Scenario Preset:", list(PRESETS.keys()))
#         if selected_preset != "Select a Preset...":
#             st.session_state.sim_params = DEFAULT_PARAMS.copy()
#             st.session_state.sim_params.update(PRESETS[selected_preset])
#             st.success(f"Loaded '{selected_preset}'.")
            
#             with st.expander("View Preset Parameters", expanded=False):
#                 st.write(f"**L1 Corruption:** {st.session_state.sim_params['l1_corruption_pct']*100}%")
#                 st.write(f"**Collusion:** {st.session_state.sim_params['collusion_factor']*100}%")
#                 st.write(f"**Equipment Error:** {st.session_state.sim_params['equipment_error']} cm")
#                 st.write(f"**Copy-Paste Rate:** {st.session_state.sim_params['copy_paste_pct']}%")
#     else: 
#         st.markdown("### Core Parameters")
#         slider_col1, slider_col2 = st.columns(2)
#         with slider_col1:
#             st.session_state.sim_params["l1_corruption_pct"] = st.slider("L1 Corruption Rate", 0.0, 1.0, st.session_state.sim_params["l1_corruption_pct"])
#             st.session_state.sim_params["collusion_factor"] = st.slider("Collusion Factor", 0.0, 1.0, st.session_state.sim_params["collusion_factor"])
#             st.session_state.sim_params["equipment_error"] = st.slider("Equipment Error (cm)", 0.0, 3.0, float(st.session_state.sim_params["equipment_error"]))
#         with slider_col2:
#             st.session_state.sim_params["l0_fraud_pct"] = st.slider("L0 Fraud Rate", 0.0, 1.0, st.session_state.sim_params["l0_fraud_pct"])
#             st.session_state.sim_params["copy_paste_pct"] = st.slider("Copy-Paste Rate (%)", 0, 100, int(st.session_state.sim_params["copy_paste_pct"]))
        
#         with st.expander("Expand all simulation parameters (Power Users)"):
#             exp_col1, exp_col2, exp_col3 = st.columns(3)
#             with exp_col1:
#                 st.markdown("**Population Structure**")
#                 st.session_state.sim_params["n_L1s"] = st.number_input("Total L1 Supervisors", value=st.session_state.sim_params["n_L1s"])
#                 st.session_state.sim_params["n_L0s_per_L1"] = st.number_input("L0 Centers per L1", value=st.session_state.sim_params["n_L0s_per_L1"])
#                 st.session_state.sim_params["n_children_per_L0"] = st.number_input("Children per L0", value=st.session_state.sim_params["n_children_per_L0"])
#                 st.session_state.sim_params["real_percent_stunting"] = st.number_input("Real % Stunting", value=st.session_state.sim_params["real_percent_stunting"])
#                 st.session_state.sim_params["real_percent_underweight"] = st.number_input("Real % Underweight", value=st.session_state.sim_params["real_percent_underweight"])
#             with exp_col2:
#                 st.markdown("**Fraud Variances**")
#                 st.session_state.sim_params["sd_across_units_percent_under_reporting_stunting"] = st.number_input("SD Across Under-Report (Stunt)", value=float(st.session_state.sim_params["sd_across_units_percent_under_reporting_stunting"]))
#                 st.session_state.sim_params["sd_within_units_percent_under_reporting_stunting"] = st.number_input("SD Within Under-Report (Stunt)", value=float(st.session_state.sim_params["sd_within_units_percent_under_reporting_stunting"]))
#                 st.session_state.sim_params["sd_across_units_bunch_factor_haz"] = st.number_input("SD Across Bunch Factor HAZ", value=float(st.session_state.sim_params["sd_across_units_bunch_factor_haz"]))
#                 st.session_state.sim_params["sd_within_units_bunch_factor_haz"] = st.number_input("SD Within Bunch Factor HAZ", value=float(st.session_state.sim_params["sd_within_units_bunch_factor_haz"]))
#             with exp_col3:
#                 st.markdown("**Misc Standard Deviations & Lags**")
#                 st.session_state.sim_params["sd_percent_copy"] = st.number_input("SD Copy-Paste", value=float(st.session_state.sim_params["sd_percent_copy"]))
#                 st.session_state.sim_params["sd_collusion_index"] = st.number_input("SD Collusion", value=float(st.session_state.sim_params["sd_collusion_index"]))
#                 st.session_state.sim_params["mean_time_lag_L1"] = st.number_input("Mean Time Lag L1 (Days)", value=int(st.session_state.sim_params["mean_time_lag_L1"]))
#                 st.session_state.sim_params["mean_time_lag_L2"] = st.number_input("Mean Time Lag L2 (Days)", value=int(st.session_state.sim_params["mean_time_lag_L2"]))

#     st.markdown("<br>", unsafe_allow_html=True) 

#     if st.button("Generate Strategy Recommendation", type="primary", use_container_width=True):
#         status_text = st.empty()
#         progress_bar = st.progress(0)
        
#         try:
#             status_text.info("Initiating Engine... Connecting to Backend.")
#             response = requests.post(f"{API_BASE_URL}/start-nested-sim", json=st.session_state.sim_params)
            
#             if response.status_code == 200:
#                 task_id = response.json()["task_id"]
#                 is_complete = False
                
#                 while not is_complete:
#                     time.sleep(3) 
#                     check_resp = requests.get(f"{API_BASE_URL}/check-nested-sim/{task_id}")
#                     if check_resp.status_code == 200:
#                         status_data = check_resp.json()
                        
#                         if status_data["status"] == "Complete":
#                             is_complete = True
#                             progress_bar.progress(100)
#                             status_text.success("Simulation Complete! Rendering Visuals...")
                            
#                             final_df = pd.DataFrame(status_data["data"])
#                             final_df['L1_Pct_Num'] = final_df['L1_Budget_Pct'].str.replace('%', '').astype(int)
#                             final_df['L2_Pct_Num'] = final_df['L2_Budget_Pct'].str.replace('%', '').astype(int)
#                             final_df['L1_Acc_Num'] = final_df['L1_Accuracy_vs_L0'].str.replace('%', '').astype(float)
#                             final_df['L2_Acc_Num'] = final_df['L2_Accuracy_vs_L1'].str.replace('%', '').astype(float)
                            
#                             st.markdown("---")
#                             st.markdown("### 📈 Custom Scenario Analysis")
                            
#                             fig1, ax1 = plt.subplots(figsize=(12, 6), dpi=100)
#                             l1_trend = final_df.drop_duplicates(subset=['L1_Pct_Num']).sort_values('L1_Pct_Num')
#                             ax1.plot(l1_trend['L1_Pct_Num'], l1_trend['L1_Acc_Num'], marker='o', linestyle='-', color='#27ae60', lw=3, ms=10, label="Custom L1 Accuracy")
#                             ax1.set_xlabel('L1 Base Budget / Children Sampled (%)', fontsize=14, fontweight='bold', labelpad=15)
#                             ax1.set_ylabel('Top Worst L0 Centers Caught (%)', fontsize=14, fontweight='bold', labelpad=15)
#                             ax1.set_xticks(sorted(l1_trend['L1_Pct_Num'].unique()))
#                             ax1.set_xticklabels([f"{x}%" for x in sorted(l1_trend['L1_Pct_Num'].unique())], fontsize=12)
#                             ax1.set_ylim(0, 105)
#                             ax1.grid(True, linestyle='--', alpha=0.5)
#                             for spine in ax1.spines.values():
#                                 spine.set_linewidth(2.0)
#                                 spine.set_color('black')
#                             ax1.legend(loc='lower right', fontsize=12, framealpha=0.9, shadow=True)
#                             plt.title("🎯 Intra-Regional: L1 Ranking Accuracy of L0 Centers", fontsize=16, fontweight='bold', pad=20)
#                             st.pyplot(fig1)

#                             if has_l2 == "Yes":
#                                 st.markdown("---")
#                                 st.markdown("### 🔥 L1 vs L2 Split-Pane Matrix")
                                
#                                 l1_order = sorted(final_df['L1_Pct_Num'].unique(), reverse=True)
#                                 fig2 = plt.figure(figsize=(14, max(6, len(l1_order) * 2)), dpi=100)
#                                 gs = gridspec.GridSpec(nrows=len(l1_order), ncols=2, width_ratios=[1, 5], wspace=0.1, hspace=0.6)
#                                 sns.set_theme(style="white")

#                                 for i, l1_pct in enumerate(l1_order):
#                                     ax_l1 = fig2.add_subplot(gs[i, 0])
#                                     ax_l2 = fig2.add_subplot(gs[i, 1])

#                                     subset = final_df[final_df['L1_Pct_Num'] == l1_pct].sort_values('L2_Pct_Num')

#                                     l1_acc_val = subset['L1_Acc_Num'].iloc[0] if not subset.empty else 0
#                                     sns.heatmap(np.array([[l1_acc_val]]), annot=True, fmt=".1f", cmap="Blues",
#                                                 cbar=False, linewidths=2, linecolor='white', vmin=0, vmax=100,
#                                                 ax=ax_l1, annot_kws={"size": 14, "weight": "bold"})
#                                     ax_l1.set_xticks([])
#                                     ax_l1.set_yticks([0.5])
#                                     l1_label_str = subset['L1_Strategy'].iloc[0] if not subset.empty else f"{l1_pct}%"
#                                     ax_l1.set_yticklabels([f"L1: {l1_pct}%\n({l1_label_str})"], rotation=0, fontsize=11, fontweight='bold')
#                                     if i == 0: ax_l1.set_title("L1 Baseline Accuracy (%)", fontsize=12, fontweight='bold', pad=10)

#                                     if not subset.empty:
#                                         heatmap_data_l2 = subset[['L2_Acc_Num']].T
#                                         l2_labels = [f"L2: {row['L2_Pct_Num']}%\n({row['L2_Strategy']})" for _, row in subset.iterrows()]
#                                         sns.heatmap(heatmap_data_l2, annot=True, fmt=".1f", cmap="RdYlGn",
#                                                     cbar=False, linewidths=2, linecolor='white', vmin=0, vmax=100,
#                                                     ax=ax_l2, annot_kws={"size": 13, "weight": "bold"})
#                                         ax_l2.set_yticks([])
#                                         ax_l2.set_xticks(np.arange(len(l2_labels)) + 0.5)
#                                         if i == len(l1_order) - 1:
#                                             ax_l2.set_xticklabels(l2_labels, rotation=0, fontsize=10, fontweight='bold')
#                                             ax_l2.set_xlabel(r"Increasing L2 Audit Depth $\longrightarrow$", fontsize=12, fontweight='bold', color='grey', labelpad=15)
#                                         else:
#                                             ax_l2.set_xticks([])
#                                         if i == 0: ax_l2.set_title("L2 Auditor Execution Accuracy (%)", fontsize=12, fontweight='bold', pad=10)

#                                     for ax in [ax_l1, ax_l2]:
#                                         for spine in ax.spines.values():
#                                             spine.set_visible(True)
#                                             spine.set_linewidth(2)
#                                             spine.set_color('black')

#                                 fig2.suptitle("L2 Matrix: Ranking Fraudulent L1 Supervisors", fontsize=18, fontweight='bold', y=1.02)
#                                 st.pyplot(fig2)

#                             st.markdown("### 📋 Recommended Sampling Strategies")
#                             display_df = final_df.drop(columns=['L1_Pct_Num', 'L2_Pct_Num', 'L1_Acc_Num', 'L2_Acc_Num'], errors='ignore')
#                             if has_l2 == "No":
#                                 display_df = display_df.drop(columns=["L2_Budget_Pct", "L2_Strategy", "L2_Accuracy_vs_L1"], errors='ignore')
#                             st.dataframe(display_df, use_container_width=True)
                            
#                             st.markdown("---")
#                             st.markdown("### 👶 Child-Level Biological Data & Fraud Metrics")
#                             st.info("Showing each child's Height, Weight, HAZ, WAZ at every tier, plus the localized MAE scores.")
                            
#                             child_path = status_data.get("child_data_path")
#                             if child_path:
#                                 child_df = pd.read_csv(child_path)
#                                 float_cols = child_df.select_dtypes(include=['float64']).columns
#                                 child_df[float_cols] = child_df[float_cols].round(2)
#                                 if has_l2 == "No":
#                                     child_df = child_df.drop(columns=[col for col in child_df.columns if "L2" in col], errors='ignore')
#                                 st.dataframe(child_df, use_container_width=True)
                            
#                         elif status_data["status"] == "Failed":
#                             status_text.error(f"Engine Failed: {status_data.get('error', 'Unknown Error')}")
#                             break
#                         else:
#                             status_text.warning("Crunching synthetic child records in the background... Please wait.")
#                             progress_bar.progress(25)
#                     else:
#                         status_text.error("Lost connection to the engine.")
#                         break
#             else:
#                 status_text.error(f"Failed to start simulation. Error {response.status_code}: {response.text}")
                
#         except Exception as e:
#             st.error(f"Could not connect to the Backend API. Ensure uvicorn is running. Error: {str(e)}")

# if __name__ == "__main__":
#     render_nested_simulation_ui()