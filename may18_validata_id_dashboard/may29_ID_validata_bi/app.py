import streamlit as st
import pandas as pd
import os
import glob
from nested_simulation_ui import render_sidebar_filters
import plotting_engine

# ==============================================================================
# 0. STREAMLIT CONFIGURATION
# ==============================================================================
st.set_page_config(page_title="Validata BI Dashboard", layout="wide")
st.title("Validata Data Quality Audit Dashboard")
st.caption("Locally Hosted Precalculated Analytics")

# ==============================================================================
# 1. DATA LOADER
# ==============================================================================
# Hardcoded path to where Step 3 saves the Tracer_Master_DB CSVs

# Dynamically find the path relative to wherever app.py is hosted
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PRESETS_DIR = os.path.join(CURRENT_DIR, "data")

@st.cache_data
def get_available_presets():
    search_pattern = os.path.join(PRESETS_DIR, "Tracer_Master_DB_*.csv")
    files = glob.glob(search_pattern)
    # Create a clean dictionary mapping display names to actual file paths
    return {os.path.basename(f).replace("Tracer_Master_DB_", "").replace(".csv", ""): f for f in files}

presets = get_available_presets()

if not presets:
    st.error(f"No precalculated CSV datasets found in {PRESETS_DIR}. Please run Step 3 first.")
    st.stop()

# Let user pick which simulation run they want to view
col1, col2 = st.columns([1, 2])
with col1:
    selected_preset_name = st.selectbox("Select Simulation Scenario:", list(presets.keys()))
file_path = presets[selected_preset_name]

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    # Ensure 'L1_Pct_Num' exists for sorting purposes in the plotting engine
    if 'L1_Budget_Pct' in df.columns and 'L1_Pct_Num' not in df.columns:
        df['L1_Pct_Num'] = df['L1_Budget_Pct'].str.replace('%', '').astype(int)
    return df

df = load_data(file_path)

if df.empty:
    st.error("The selected dataset is empty.")
    st.stop()

# ==============================================================================
# 2. RENDER UI FILTERS
# ==============================================================================
df_filtered, indicator, selected_pct = render_sidebar_filters(df)

if df_filtered.empty:
    st.warning("No data matches the selected sidebar filters. Please expand your selection.")
    st.stop()

st.success(f"Successfully loaded `{selected_preset_name}` | Visualizing **{indicator}** at the **{selected_pct}** fraud bracket. | Total Data Points: {len(df_filtered):,}")
st.divider()

# ==============================================================================
# 3. RENDER PLOTS
# ==============================================================================
# Add a multi-select to let the user pick which charts to render (saves memory)
plot_options = [
    "Plot 1: V2 Accuracy (L1 catching L0)",
    "Plot 2: Heatmap (L2 catching L1)"
]
selected_plots = st.multiselect("Display Charts:", plot_options, default=plot_options)

# We use standard Streamlit columns to lay out the charts nicely
if "Plot 1: V2 Accuracy (L1 catching L0)" in selected_plots:
    st.subheader("1. L1 Supervisor Accuracy vs Budget")
    # Calling your existing matplotlib functions from plotting_engine
    try:
        # Note: You may need to adjust the arguments here based on what your plotting_engine.py actually requires
        fig1 = plotting_engine.plot_2_intra_regional(
            df=df_filtered, 
            l1_pct_str="100%", # Fallback or dynamic based on what your function expects
            indicator=indicator, 
            n_L0s_per_L1=25, 
            n_children_per_L0=15, 
            target_percentile=float(selected_pct.strip('%')) / 100
        )
        if fig1:
            st.pyplot(fig1)
    except Exception as e:
        st.error(f"Could not render Plot 1: {e}")

if "Plot 2: Heatmap (L2 catching L1)" in selected_plots:
    st.subheader("2. L2 Auditor Heatmap Matrix")
    try:
        # Assuming your heatmap requires specific subsetting.
        # Ensure plot_6_heatmap exists in your plotting_engine
        fig2 = plotting_engine.plot_6_heatmap(
            df=df_filtered, 
            metric_col_l1='V2_MAE_Acc', 
            metric_col_l2='V1_MAE_Acc', 
            metric_label="Accuracy", 
            selected_uni=df_filtered['Universe'].iloc[0], 
            l1_pct_str=df_filtered['L1_Budget_Pct'].iloc[0], 
            l2_pct_str=df_filtered['L2_Budget_Pct'].iloc[0]
        )
        if fig2:
            st.pyplot(fig2)
    except Exception as e:
        st.error(f"Could not render Heatmap: {e}")