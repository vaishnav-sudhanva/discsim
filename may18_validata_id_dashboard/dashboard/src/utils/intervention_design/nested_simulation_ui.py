import streamlit as st
import requests
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec

# MUST MATCH THE MAIN.PY PORT (8005)
API_BASE_URL = "http://localhost:8005"

DEFAULT_PARAMS = {
    "l1_corruption_pct": 0.05, "l0_fraud_pct": 0.05, 
    "collusion_factor": 0.10, "copy_paste_pct": 5, "equipment_error": 0.1,
    "n_L1s": 334, "n_L0s_per_L1": 25, "n_children_per_L0": 15,
    "real_percent_stunting": 35, "real_percent_underweight": 33,
    "sd_across_units_percent_under_reporting_stunting": 2.0,
    "sd_across_units_percent_under_reporting_underweight": 2.0,
    "sd_within_units_percent_under_reporting_stunting": 1.0,
    "sd_within_units_percent_under_reporting_underweight": 1.0,
    "sd_across_units_bunch_factor_haz": 0.01, "sd_across_units_bunch_factor_waz": 0.01, "sd_across_units_bunch_factor_whz": 0.01,
    "sd_within_units_bunch_factor_haz": 0.01, "sd_within_units_bunch_factor_waz": 0.01, "sd_within_units_bunch_factor_whz": 0.01,
    "sd_percent_copy": 2.0, "sd_collusion_index": 0.02,
    "mean_time_lag_L1": 15, "mean_time_lag_L2": 30
}

PRESETS = {
    "Select a Preset...": {},
    "Honest State (Baseline)": {}, 
    "Corrupt Collusion": {"l1_corruption_pct": 0.60, "l0_fraud_pct": 0.50, "collusion_factor": 0.90, "copy_paste_pct": 60, "equipment_error": 0.5},
    "Lazy Supervisors (High Copy-Paste)": {"l1_corruption_pct": 0.80, "l0_fraud_pct": 0.20, "collusion_factor": 0.20, "copy_paste_pct": 85, "equipment_error": 0.2},
    "Bad Equipment (High Noise)": {"l1_corruption_pct": 0.10, "l0_fraud_pct": 0.10, "collusion_factor": 0.10, "copy_paste_pct": 10, "equipment_error": 1.5}
}

def render_nested_simulation_ui():
    st.markdown("<h2 style='text-align: center;'>Nested Simulation Strategy Predictor</h2>", unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### System Architecture")
        has_l2 = st.radio("Is a Third-Party (L2) Auditor present?", options=["Yes", "No"])
    with col2:
        st.markdown("### Configuration Mode")
        config_mode = st.radio("Select Input Mode:", options=["Preset Selection", "Advanced Selection"])
    st.markdown("---")

    if "sim_params" not in st.session_state:
        st.session_state.sim_params = DEFAULT_PARAMS.copy()

    if config_mode == "Preset Selection":
        selected_preset = st.selectbox("Choose a Scenario Preset:", list(PRESETS.keys()))
        if selected_preset != "Select a Preset...":
            st.session_state.sim_params = DEFAULT_PARAMS.copy()
            st.session_state.sim_params.update(PRESETS[selected_preset])
            st.success(f"Loaded '{selected_preset}'.")
            
            with st.expander("View Preset Parameters", expanded=False):
                st.write(f"**L1 Corruption:** {st.session_state.sim_params['l1_corruption_pct']*100}%")
                st.write(f"**Collusion:** {st.session_state.sim_params['collusion_factor']*100}%")
                st.write(f"**Equipment Error:** {st.session_state.sim_params['equipment_error']} cm")
                st.write(f"**Copy-Paste Rate:** {st.session_state.sim_params['copy_paste_pct']}%")
    else: 
        st.markdown("### Core Parameters")
        slider_col1, slider_col2 = st.columns(2)
        with slider_col1:
            st.session_state.sim_params["l1_corruption_pct"] = st.slider("L1 Corruption Rate", 0.0, 1.0, st.session_state.sim_params["l1_corruption_pct"])
            st.session_state.sim_params["collusion_factor"] = st.slider("Collusion Factor", 0.0, 1.0, st.session_state.sim_params["collusion_factor"])
            st.session_state.sim_params["equipment_error"] = st.slider("Equipment Error (cm)", 0.0, 3.0, float(st.session_state.sim_params["equipment_error"]))
        with slider_col2:
            st.session_state.sim_params["l0_fraud_pct"] = st.slider("L0 Fraud Rate", 0.0, 1.0, st.session_state.sim_params["l0_fraud_pct"])
            st.session_state.sim_params["copy_paste_pct"] = st.slider("Copy-Paste Rate (%)", 0, 100, int(st.session_state.sim_params["copy_paste_pct"]))
        
        with st.expander("Expand all simulation parameters (Power Users)"):
            exp_col1, exp_col2, exp_col3 = st.columns(3)
            with exp_col1:
                st.markdown("**Population Structure**")
                st.session_state.sim_params["n_L1s"] = st.number_input("Total L1 Supervisors", value=st.session_state.sim_params["n_L1s"])
                st.session_state.sim_params["n_L0s_per_L1"] = st.number_input("L0 Centers per L1", value=st.session_state.sim_params["n_L0s_per_L1"])
                st.session_state.sim_params["n_children_per_L0"] = st.number_input("Children per L0", value=st.session_state.sim_params["n_children_per_L0"])
                st.session_state.sim_params["real_percent_stunting"] = st.number_input("Real % Stunting", value=st.session_state.sim_params["real_percent_stunting"])
                st.session_state.sim_params["real_percent_underweight"] = st.number_input("Real % Underweight", value=st.session_state.sim_params["real_percent_underweight"])
            with exp_col2:
                st.markdown("**Fraud Variances**")
                st.session_state.sim_params["sd_across_units_percent_under_reporting_stunting"] = st.number_input("SD Across Under-Report (Stunt)", value=float(st.session_state.sim_params["sd_across_units_percent_under_reporting_stunting"]))
                st.session_state.sim_params["sd_within_units_percent_under_reporting_stunting"] = st.number_input("SD Within Under-Report (Stunt)", value=float(st.session_state.sim_params["sd_within_units_percent_under_reporting_stunting"]))
                st.session_state.sim_params["sd_across_units_bunch_factor_haz"] = st.number_input("SD Across Bunch Factor HAZ", value=float(st.session_state.sim_params["sd_across_units_bunch_factor_haz"]))
                st.session_state.sim_params["sd_within_units_bunch_factor_haz"] = st.number_input("SD Within Bunch Factor HAZ", value=float(st.session_state.sim_params["sd_within_units_bunch_factor_haz"]))
            with exp_col3:
                st.markdown("**Misc Standard Deviations & Lags**")
                st.session_state.sim_params["sd_percent_copy"] = st.number_input("SD Copy-Paste", value=float(st.session_state.sim_params["sd_percent_copy"]))
                st.session_state.sim_params["sd_collusion_index"] = st.number_input("SD Collusion", value=float(st.session_state.sim_params["sd_collusion_index"]))
                st.session_state.sim_params["mean_time_lag_L1"] = st.number_input("Mean Time Lag L1 (Days)", value=int(st.session_state.sim_params["mean_time_lag_L1"]))
                st.session_state.sim_params["mean_time_lag_L2"] = st.number_input("Mean Time Lag L2 (Days)", value=int(st.session_state.sim_params["mean_time_lag_L2"]))

    st.markdown("<br>", unsafe_allow_html=True) 

    if st.button("Generate Strategy Recommendation", type="primary", use_container_width=True):
        status_text = st.empty()
        progress_bar = st.progress(0)
        
        try:
            status_text.info("Initiating Engine... Connecting to Backend.")
            response = requests.post(f"{API_BASE_URL}/start-nested-sim", json=st.session_state.sim_params)
            
            if response.status_code == 200:
                task_id = response.json()["task_id"]
                is_complete = False
                
                while not is_complete:
                    time.sleep(3) 
                    check_resp = requests.get(f"{API_BASE_URL}/check-nested-sim/{task_id}")
                    if check_resp.status_code == 200:
                        status_data = check_resp.json()
                        
                        if status_data["status"] == "Complete":
                            is_complete = True
                            progress_bar.progress(100)
                            status_text.success("Simulation Complete! Rendering Visuals...")
                            
                            final_df = pd.DataFrame(status_data["data"])
                            final_df['L1_Pct_Num'] = final_df['L1_Budget_Pct'].str.replace('%', '').astype(int)
                            final_df['L2_Pct_Num'] = final_df['L2_Budget_Pct'].str.replace('%', '').astype(int)
                            final_df['L1_Acc_Num'] = final_df['L1_Accuracy_vs_L0'].str.replace('%', '').astype(float)
                            final_df['L2_Acc_Num'] = final_df['L2_Accuracy_vs_L1'].str.replace('%', '').astype(float)
                            
                            st.markdown("---")
                            st.markdown("### 📈 Custom Scenario Analysis")
                            
                            fig1, ax1 = plt.subplots(figsize=(12, 6), dpi=100)
                            l1_trend = final_df.drop_duplicates(subset=['L1_Pct_Num']).sort_values('L1_Pct_Num')
                            ax1.plot(l1_trend['L1_Pct_Num'], l1_trend['L1_Acc_Num'], marker='o', linestyle='-', color='#27ae60', lw=3, ms=10, label="Custom L1 Accuracy")
                            ax1.set_xlabel('L1 Base Budget / Children Sampled (%)', fontsize=14, fontweight='bold', labelpad=15)
                            ax1.set_ylabel('Top Worst L0 Centers Caught (%)', fontsize=14, fontweight='bold', labelpad=15)
                            ax1.set_xticks(sorted(l1_trend['L1_Pct_Num'].unique()))
                            ax1.set_xticklabels([f"{x}%" for x in sorted(l1_trend['L1_Pct_Num'].unique())], fontsize=12)
                            ax1.set_ylim(0, 105)
                            ax1.grid(True, linestyle='--', alpha=0.5)
                            for spine in ax1.spines.values():
                                spine.set_linewidth(2.0)
                                spine.set_color('black')
                            ax1.legend(loc='lower right', fontsize=12, framealpha=0.9, shadow=True)
                            plt.title("🎯 Intra-Regional: L1 Ranking Accuracy of L0 Centers", fontsize=16, fontweight='bold', pad=20)
                            st.pyplot(fig1)

                            if has_l2 == "Yes":
                                st.markdown("---")
                                st.markdown("### 🔥 L1 vs L2 Split-Pane Matrix")
                                
                                l1_order = sorted(final_df['L1_Pct_Num'].unique(), reverse=True)
                                fig2 = plt.figure(figsize=(14, max(6, len(l1_order) * 2)), dpi=100)
                                gs = gridspec.GridSpec(nrows=len(l1_order), ncols=2, width_ratios=[1, 5], wspace=0.1, hspace=0.6)
                                sns.set_theme(style="white")

                                for i, l1_pct in enumerate(l1_order):
                                    ax_l1 = fig2.add_subplot(gs[i, 0])
                                    ax_l2 = fig2.add_subplot(gs[i, 1])

                                    subset = final_df[final_df['L1_Pct_Num'] == l1_pct].sort_values('L2_Pct_Num')

                                    l1_acc_val = subset['L1_Acc_Num'].iloc[0] if not subset.empty else 0
                                    sns.heatmap(np.array([[l1_acc_val]]), annot=True, fmt=".1f", cmap="Blues",
                                                cbar=False, linewidths=2, linecolor='white', vmin=0, vmax=100,
                                                ax=ax_l1, annot_kws={"size": 14, "weight": "bold"})
                                    ax_l1.set_xticks([])
                                    ax_l1.set_yticks([0.5])
                                    l1_label_str = subset['L1_Strategy'].iloc[0] if not subset.empty else f"{l1_pct}%"
                                    ax_l1.set_yticklabels([f"L1: {l1_pct}%\n({l1_label_str})"], rotation=0, fontsize=11, fontweight='bold')
                                    if i == 0: ax_l1.set_title("L1 Baseline Accuracy (%)", fontsize=12, fontweight='bold', pad=10)

                                    if not subset.empty:
                                        heatmap_data_l2 = subset[['L2_Acc_Num']].T
                                        l2_labels = [f"L2: {row['L2_Pct_Num']}%\n({row['L2_Strategy']})" for _, row in subset.iterrows()]
                                        sns.heatmap(heatmap_data_l2, annot=True, fmt=".1f", cmap="RdYlGn",
                                                    cbar=False, linewidths=2, linecolor='white', vmin=0, vmax=100,
                                                    ax=ax_l2, annot_kws={"size": 13, "weight": "bold"})
                                        ax_l2.set_yticks([])
                                        ax_l2.set_xticks(np.arange(len(l2_labels)) + 0.5)
                                        if i == len(l1_order) - 1:
                                            ax_l2.set_xticklabels(l2_labels, rotation=0, fontsize=10, fontweight='bold')
                                            ax_l2.set_xlabel(r"Increasing L2 Audit Depth $\longrightarrow$", fontsize=12, fontweight='bold', color='grey', labelpad=15)
                                        else:
                                            ax_l2.set_xticks([])
                                        if i == 0: ax_l2.set_title("L2 Auditor Execution Accuracy (%)", fontsize=12, fontweight='bold', pad=10)

                                    for ax in [ax_l1, ax_l2]:
                                        for spine in ax.spines.values():
                                            spine.set_visible(True)
                                            spine.set_linewidth(2)
                                            spine.set_color('black')

                                fig2.suptitle("L2 Matrix: Ranking Fraudulent L1 Supervisors", fontsize=18, fontweight='bold', y=1.02)
                                st.pyplot(fig2)

                            st.markdown("### 📋 Recommended Sampling Strategies")
                            display_df = final_df.drop(columns=['L1_Pct_Num', 'L2_Pct_Num', 'L1_Acc_Num', 'L2_Acc_Num'], errors='ignore')
                            if has_l2 == "No":
                                display_df = display_df.drop(columns=["L2_Budget_Pct", "L2_Strategy", "L2_Accuracy_vs_L1"], errors='ignore')
                            st.dataframe(display_df, use_container_width=True)
                            
                            st.markdown("---")
                            st.markdown("### 👶 Child-Level Biological Data & Fraud Metrics")
                            st.info("Showing each child's Height, Weight, HAZ, WAZ at every tier, plus the localized MAE scores.")
                            
                            child_path = status_data.get("child_data_path")
                            if child_path:
                                child_df = pd.read_csv(child_path)
                                float_cols = child_df.select_dtypes(include=['float64']).columns
                                child_df[float_cols] = child_df[float_cols].round(2)
                                if has_l2 == "No":
                                    child_df = child_df.drop(columns=[col for col in child_df.columns if "L2" in col], errors='ignore')
                                st.dataframe(child_df, use_container_width=True)
                            
                        elif status_data["status"] == "Failed":
                            status_text.error(f"Engine Failed: {status_data.get('error', 'Unknown Error')}")
                            break
                        else:
                            status_text.warning("Crunching synthetic child records in the background... Please wait.")
                            progress_bar.progress(25)
                    else:
                        status_text.error("Lost connection to the engine.")
                        break
            else:
                status_text.error(f"Failed to start simulation. Error {response.status_code}: {response.text}")
                
        except Exception as e:
            st.error(f"Could not connect to the Backend API. Ensure uvicorn is running. Error: {str(e)}")

if __name__ == "__main__":
    render_nested_simulation_ui()