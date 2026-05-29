import os
import time
from datetime import datetime
import streamlit as st
import pandas as pd
import plotting_engine
from ui_text_config import TOOLTIPS

# ==============================================================================
# 1. DIRECTORY SETUP & PRESET CONFIGURATIONS (CLOUD SAFE)
# ==============================================================================
# Dynamically find the path relative to wherever app.py is hosted
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "data")

# Fallback: if the 'data' folder isn't found, look in the current folder
if not os.path.exists(DATA_DIR):
    DATA_DIR = CURRENT_DIR

MASTER_REGISTRY_PATH = os.path.join(DATA_DIR, "simulation_master_registry.csv")

# ==============================================================================
# CHANGE MADE HERE: Removed "Tracer_Master_DB_" and ".csv" so it matches the registry
# ==============================================================================
PRESET_MAPPING = {
    "Select a Preset...": None,
    "Good L1 Good L0": "Good_L0_Good_L1_20260528_145250_Calc_1sims_Eval_20260529_003255",
    "Bad L1 Good L0": "Good_L0_Bad_L1_20260528_150009_Calc_1sims_Eval_20260529_014234",
    "Good L1 Bad L0": "Bad_L0_Good_L1_20260528_151136_Calc_1sims_Eval_20260529_015516",
    "Bad L1 Bad L0": "Bad_L0_Bad_L1_20260528_152028_Calc_1sims_Eval_20260529_021745"
    # Add your other presets here as you generate them!
    # "Bad L1, Bad L0": "Bad_L0_Bad_L1_..._Eval_...",
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
            st.error(f"Could not find {task_id} inside your CSV Registry at {MASTER_REGISTRY_PATH}.")
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
        # Dynamically checks for your new Tracer_Master_DB format or the old final_evaluation format
        eval_path = os.path.join(DATA_DIR, f"Tracer_Master_DB_{task_id}.csv")
        if not os.path.exists(eval_path):
            eval_path = os.path.join(DATA_DIR, f"final_evaluation_{task_id}.csv")
            
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

# ==============================================================================
# 2. CORE RENDERING ENGINE
# ==============================================================================
def render_nested_simulation_ui():
    st.set_page_config(layout="wide", page_title="Nested Simulation Dashboard")
    st.markdown("<h1 style='text-align: center;'>Nested Simulation Dashboard</h1>", unsafe_allow_html=True)
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
    st.markdown("### 1. System Architecture & Display")
    col1, col2, col3 = st.columns(3)
    with col1: p["has_l2"] = st.radio("Include L2 Auditor?", ["Yes", "No"], help=TOOLTIPS.get("has_l2", ""))
    with col2: p["output_variable"] = st.selectbox("Indicator to Display", ["Height", "Weight", "Both"], help=TOOLTIPS.get("output_variable", ""))
    with col3:
        target_ui_val = st.slider("Target Worst % to Catch", 5, 50, 30, help=TOOLTIPS.get("target_percentile", ""))
        p["target_percentile"] = target_ui_val / 100.0  

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
    selected_preset = st.session_state.get("preset_dropdown", "Select a Preset...")
    
    if selected_preset == "Custom Scenario (Live Generation)":
        if st.button("Run Heavy Custom Simulation Locally", type="primary", use_container_width=True):
            st.warning("Custom Local Generation logic is pending. We need your Step 1 Python script to plug the raw engine into this button.")

    # ==============================================================================
    # 7. RENDER FILTERS AND PLOTS FROM SCENARIO RESULTS
    # ==============================================================================
    if st.session_state["simulation_results"] is not None:
        strategy_df = st.session_state["simulation_results"]
        mem_p = st.session_state["sim_params_memory"]

        st.markdown("---")
        st.markdown("### Interactive Plot Filters")
        st.info("The data has been pre-calculated for all budget scenarios. Use these filters to instantly update the charts below without re-running the simulation.")
        
        # Changed to 4 columns
        filter_col1, filter_col2, filter_col3, filter_col4 = st.columns([2, 2, 2, 1])
        budget_options = [f"{i}0%" for i in range(1, 11)]
        
        # Extract available target percentiles directly from the CSV
        if 'Target_Percentile' in strategy_df.columns:
            target_options = sorted(strategy_df['Target_Percentile'].unique().tolist())
            default_target = "30%" if "30%" in target_options else target_options[0]
        else:
            target_options = ["30%"]
            default_target = "30%"
        
        with filter_col1:
            ui_l1_budget = st.selectbox("L1 Budget Filter (Applies to all plots)", budget_options, index=5, help=TOOLTIPS.get("l1_budget", ""))
        
        with filter_col2:
            if mem_p["has_l2"] == "Yes":
                ui_l2_budget = st.selectbox("L2 Budget Filter (Applies to Heatmap)", budget_options, index=3, help=TOOLTIPS.get("l2_budget", ""))
            else:
                st.selectbox("L2 Budget Filter", ["N/A (L2 Disabled)"], disabled=True)
                ui_l2_budget = "N/A"
                
        # Added the new Target Percentile Dropdown in Col 3
        with filter_col3:
            ui_target_pct = st.selectbox("Target Catch Bracket", target_options, index=target_options.index(default_target), help="Filters all plots to show accuracy for catching this specific % of worst offenders.")

        # Moved Download button to Col 4
        with filter_col4:
            st.markdown("<br>", unsafe_allow_html=True)
            csv_data = strategy_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Data",
                data=csv_data,
                file_name='simulation_results.csv',
                mime='text/csv',
                use_container_width=True
            )

        st.markdown("<hr>", unsafe_allow_html=True)

        if mem_p["output_variable"].lower() == "both": target_inds = ["Height", "Weight"]
        elif mem_p["output_variable"].lower() == "weight": target_inds = ["Weight"]
        else: target_inds = ["Height"]

        total_kids_per_l1 = mem_p["n_L0s_per_L1"] * mem_p["n_children_per_L0"]
        l1_pct_float = int(ui_l1_budget.replace("%", "")) / 100.0
        actual_l1_budget = int(total_kids_per_l1 * l1_pct_float)
        
        actual_l2_budget = 0
        if mem_p["has_l2"] == "Yes" and ui_l2_budget != "N/A":
            l2_pct_float = int(ui_l2_budget.replace("%", "")) / 100.0
            actual_l2_budget = int(actual_l1_budget * l2_pct_float)

        # Slice the master dataframe by the selected Target Percentile BEFORE plotting!
        if 'Target_Percentile' in strategy_df.columns:
            active_df = strategy_df[strategy_df['Target_Percentile'] == ui_target_pct].copy()
            current_target_float = float(ui_target_pct.replace('%', '')) / 100.0
        else:
            active_df = strategy_df.copy()
            current_target_float = mem_p["target_percentile"]

        for ind in target_inds:
            st.markdown(f"<h2 style='text-align: center; color: #2c3e50; margin-top: 20px;'>=== RESULTS FOR {ind.upper()} ===</h2>", unsafe_allow_html=True)
            
            # Use active_df instead of strategy_df so the plots only see ONE target percentile at a time
            df_ind = active_df[active_df['Indicator'] == ind]
            
            # --- CASE 1: Heatmap (Only if L2 is Yes) ---
            if mem_p["has_l2"] == "Yes":
                df_filtered_p6 = df_ind[(df_ind['L1_Budget_Pct'] == ui_l1_budget) & (df_ind['L2_Budget_Pct'] == ui_l2_budget)]
                try:
                    fig6 = plotting_engine.plot_6_heatmap(df_filtered_p6, l1_pct_str=ui_l1_budget, l2_pct_str=ui_l2_budget, indicator=ind)
                    if fig6: 
                        st.pyplot(fig6)
                        with st.expander(f"Dynamic Insight: L1 vs L2 Synergy ({ui_l1_budget} L1 | {ui_l2_budget} L2)"):
                            st.markdown(f"**How the {ui_l1_budget} L1 and {ui_l2_budget} L2 budgets interact:**")
                            st.write("The blue matrix on the left represents how accurately the L1 supervisor is catching the worst centers based purely on their own sample. The red/green matrix on the right evaluates the L2 independent auditor.")
                            
                            if not df_filtered_p6.empty:
                                best_l2 = df_filtered_p6.loc[df_filtered_p6['V3_MAE_Acc'].idxmax()]
                                st.success(f"**Optimal L2 Strategy:** To maximize accuracy with this specific budget combination, the L2 auditor should randomly re-measure **{best_l2['L2_K']} children** across **{best_l2['L2_C']} clinics**, which yields an execution accuracy of **{best_l2['V3_MAE_Acc']:.1f}%**.")
                except Exception as e:
                    st.error(f"Error plotting Heatmap: {e}")

            # --- CASE 2: Breadth vs Depth ---
            df_filtered_p3 = df_ind[df_ind['L1_Budget_Pct'] == ui_l1_budget]
            if mem_p["has_l2"] == "No": 
                df_filtered_p3 = df_filtered_p3.drop_duplicates(subset=['L1_Label'])
            
            try:
                fig3 = plotting_engine.plot_3_breadth_depth(
                    df_filtered_p3, 
                    l1_pct_str=ui_l1_budget, 
                    indicator=ind,
                    n_L0s_per_L1=mem_p["n_L0s_per_L1"],
                    target_percentile=current_target_float
                )
                
                if fig3: 
                    st.pyplot(fig3)
                    with st.expander(f"Guide & Dynamic Insight: Breadth vs Depth at {ui_l1_budget} Budget ({actual_l1_budget} Total Measurements)"):
                        st.markdown(TOOLTIPS.get("plot_3_breadth_depth", ""))
                        st.markdown("---")
                        st.markdown(f"**Dynamic Analysis for your selected {ui_l1_budget} L1 budget:**")
                        if not df_filtered_p3.empty:
                            best_l1 = df_filtered_p3.loc[df_filtered_p3['V1_MAE_Acc'].idxmax()]
                            st.info(f"**Optimal L1 Allocation:** Based on the synthetic biological universe and fraud parameters generated, the mathematically optimal strategy at a **{ui_l1_budget} ({actual_l1_budget} measurements)** budget is to audit **{best_l1['L1_C']} clinics** and measure **{best_l1['L1_K']} children** per clinic. This peak yields a baseline accuracy of **{best_l1['V1_MAE_Acc']:.1f}%**.")
            except Exception as e:
                st.error(f"Error plotting Breadth vs Depth: {e}")

            # --- CASE 3: Intra-Regional Return on Investment ---
            df_filtered_p2 = df_ind.drop_duplicates(subset=['L1_Budget_Pct'])
            try:
                fig2 = plotting_engine.plot_2_intra_regional(
                    df_filtered_p2, 
                    l1_pct_str=ui_l1_budget, 
                    indicator=ind,
                    n_L0s_per_L1=mem_p["n_L0s_per_L1"],
                    n_children_per_L0=mem_p["n_children_per_L0"],
                    target_percentile=current_target_float
                )
                
                if fig2: 
                    st.pyplot(fig2)
                    with st.expander(f"Guide & Dynamic Insight: Overall Return on Investment"):
                        st.markdown(TOOLTIPS.get("plot_2_intra_regional", ""))
                        st.markdown("---")
                        st.markdown(f"**Dynamic Analysis for your selected {ui_l1_budget} L1 budget:**")
                        if not df_filtered_p2.empty:
                            current_acc = df_filtered_p2[df_filtered_p2['L1_Budget_Pct'] == ui_l1_budget]['V2_MAE_Acc'].values[0]
                            max_acc_row = df_filtered_p2.loc[df_filtered_p2['V2_MAE_Acc'].idxmax()]
                            st.warning(f"**Current Selection vs Maximum Potential:** You have currently allocated a **{ui_l1_budget}** budget, which achieves an overall intra-regional accuracy of **{current_acc:.1f}%**. By comparison, increasing the budget to its maximum tested limit ({max_acc_row['L1_Budget_Pct']}) would yield **{max_acc_row['V2_MAE_Acc']:.1f}%** accuracy.")
            except Exception as e:
                st.error(f"Error plotting Intra-Regional chart: {e}")
            
            st.markdown("<hr style='border: 2px dashed #ccc;'>", unsafe_allow_html=True)
            
if __name__ == "__main__":
    render_nested_simulation_ui()