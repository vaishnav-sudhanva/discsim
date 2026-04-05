
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import glob
import os

# ==============================================================================
# 0. STREAMLIT CONFIG & DATA LOADER
# ==============================================================================
st.set_page_config(page_title="Audit Optimization Dashboard", layout="wide")

@st.cache_data
def load_data():
    # 1. Find the exact folder where app.py lives
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Tell it to look for the CSV specifically in that folder
    search_pattern = os.path.join(current_dir, "Tracer_Master_DB_Percentages_*.csv")
    csv_files = glob.glob(search_pattern)
    
    if not csv_files:
        return None
    latest_csv = max(csv_files, key=os.path.getctime)
    df = pd.read_csv(latest_csv)
    
    # Clean percentages to integers for easier plotting
    df['L1_Pct_Num'] = df['L1_Budget_Pct'].str.replace('%', '').astype(int)
    df['L2_Pct_Num'] = df['L2_Budget_Pct'].str.replace('%', '').astype(int)
    
    # Rename universes to analytical names
    df['Universe'] = df['Universe'].replace({
        "Utopia": "Good L0 Good L1",
        "Blind Spot": "Good L0 Bad L1",
        "Whistleblowers": "Bad L0 Good L1",
        "Mafia": "Bad L0 Bad L1",
        "Normal": "Normal"
    })
    
    return df
    
df = load_data()
if df is None:
    st.error("❌ No data found. Please run the Grand Master Engine first to generate the CSV.")
    st.stop()

# ==============================================================================
# 1. GLOBAL UI & FILTERS (SIDEBAR)
# ==============================================================================
st.sidebar.title("⚙️ Global Dashboard Controls")

metric_options = {
    "MAE (Mean Absolute Error)": "MAE", 
    "RMSE (Root Mean Square Error)": "RMSE", 
    "90th Percentile Error": "P90"
}
selected_metric_label = st.sidebar.selectbox("📊 Target Metric", options=list(metric_options.keys()))
metric = metric_options[selected_metric_label]

st.sidebar.markdown("---")
st.sidebar.info("This dashboard dynamically renders your custom Matplotlib charts based on pre-calculated Monte Carlo simulations.")

# Determine dynamic column names based on selected metric
v1_col = f"V1_{metric}_Acc" # L1 vs Real World
v2_col = f"V2_{metric}_Acc" # L1 vs L0 (Intra-Regional)
v3_col = f"V3_{metric}_Acc" # L2 vs L1 (Audit)

target_count = 100
uni_colors = {
    "Good L0 Good L1": "#27ae60", 
    "Good L0 Bad L1": "#f1c40f", 
    "Normal": "#2980b9", 
    "Bad L0 Good L1": "#8e44ad", 
    "Bad L0 Bad L1": "#c0392b"
}

# ==============================================================================
# 2. PLOTTING FUNCTIONS 
# ==============================================================================

def plot_1_sensitivity(df, metric_col, metric_label, selected_unis):
    filtered_df = df[df['Universe'].isin(selected_unis)]

    agg_df = filtered_df.groupby(['Universe', 'L1_Pct_Num']).agg(
        mean_acc=(metric_col, 'mean'), std_acc=(metric_col, 'std'), count=(metric_col, 'count')
    ).reset_index()
    agg_df['std_acc'] = agg_df['std_acc'].fillna(0)
    agg_df['ci95'] = 1.96 * (agg_df['std_acc'] / np.sqrt(agg_df['count']))

    fig, ax = plt.subplots(figsize=(12, 7), dpi=100)

    for uni in [u for u in uni_colors.keys() if u in agg_df['Universe'].unique()]:
        data = agg_df[agg_df['Universe'] == uni].sort_values('L1_Pct_Num')
        ax.plot(data['L1_Pct_Num'], data['mean_acc'], marker='o', linestyle='-', 
                color=uni_colors[uni], label=uni, lw=2.5, ms=8)
        ax.fill_between(data['L1_Pct_Num'], data['mean_acc'] - data['ci95'], 
                        data['mean_acc'] + data['ci95'], color=uni_colors[uni], alpha=0.15)

    ax.set_xlabel('Percentage of Children Sampled per L0: Total 15 Children per L0', fontsize=14, fontweight='bold', labelpad=15)
    ax.set_ylabel(f'Top {target_count} Worst L1 Regions Caught (%)', fontsize=14, labelpad=15)

    x_ticks = sorted(agg_df['L1_Pct_Num'].unique())
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{x}%" for x in x_ticks], fontsize=12)
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle='--', alpha=0.5)

    for spine in ax.spines.values():
        spine.set_linewidth(2.0)
        spine.set_color('black')

    ax.legend(title='Simulated Universe', bbox_to_anchor=(0.5, 1.15), loc='upper center', 
              ncol=5, fontsize=11, framealpha=0.9, shadow=True)
    plt.title(f"L1 Ranking of L1 Regions: Height Using [{metric_label}]", 
              fontsize=16, fontweight='bold', pad=75)
    plt.subplots_adjust(top=0.82)
    return fig


def plot_2_intra_regional(df, metric_col, metric_label, selected_unis):
    filtered_df = df[df['Universe'].isin(selected_unis)]

    agg_df = filtered_df.groupby(['Universe', 'L1_Pct_Num']).agg(
        mean_acc=(metric_col, 'mean'), std_acc=(metric_col, 'std'), count=(metric_col, 'count')
    ).reset_index()
    agg_df['std_acc'] = agg_df['std_acc'].fillna(0)
    agg_df['ci95'] = 1.96 * (agg_df['std_acc'] / np.sqrt(agg_df['count']))

    TARGET_CLINICS = 8
    agg_df['mean_acc_count'] = agg_df['mean_acc'] * (TARGET_CLINICS / 100)
    agg_df['ci_95_count'] = agg_df['ci95'] * (TARGET_CLINICS / 100)

    fig, ax = plt.subplots(figsize=(12, 7), dpi=100)

    for uni in [u for u in uni_colors.keys() if u in agg_df['Universe'].unique()]:
        data = agg_df[agg_df['Universe'] == uni].sort_values('L1_Pct_Num')
        ax.plot(data['L1_Pct_Num'], data['mean_acc_count'], marker='o', linestyle='-', 
                color=uni_colors[uni], label=uni, lw=2.5, ms=8)
        ax.fill_between(data['L1_Pct_Num'], data['mean_acc_count'] - data['ci_95_count'], 
                        data['mean_acc_count'] + data['ci_95_count'], color=uni_colors[uni], alpha=0.15)

    ax.set_xlabel('Percentage of Children Sampled per L0: Total 15 Children per L0', fontsize=14, fontweight='bold', labelpad=15)
    ax.set_ylabel(f'Top 30% Worst L0 caught \n({TARGET_CLINICS}/25 Target L0 in L1 Region)', fontsize=14, labelpad=15)

    x_ticks = sorted(agg_df['L1_Pct_Num'].unique())
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{x}%" for x in x_ticks], fontsize=12)
    ax.set_ylim(0, 8.5)
    ax.grid(True, linestyle='--', alpha=0.5)

    for spine in ax.spines.values():
        spine.set_linewidth(2.0)
        spine.set_color('black')

    ax.legend(title='Simulated Universe', bbox_to_anchor=(0.5, 1.15), loc='upper center', 
              ncol=5, fontsize=11, framealpha=0.9, shadow=True)
    plt.title(f"L1 Ranking of L0 in each L1 Region: Height Using [{metric_label}]", 
              fontsize=16, fontweight='bold', pad=75)
    plt.subplots_adjust(top=0.82)
    return fig


def plot_3_bd_optimization(df, metric_col, metric_label, selected_unis, l1_budget_str):
    filtered_df = df[(df['Universe'].isin(selected_unis)) & (df['L1_Budget_Pct'] == l1_budget_str)].copy()
    if filtered_df.empty: return None

    agg_df = filtered_df.groupby(['Universe', 'L1_C', 'L1_K']).agg(
        mean_acc=(metric_col, 'mean'), std_acc=(metric_col, 'std'), count=(metric_col, 'count')
    ).reset_index()
    agg_df['std_acc'] = agg_df['std_acc'].fillna(0)
    agg_df['ci95'] = 1.96 * (agg_df['std_acc'] / np.sqrt(agg_df['count']))

    agg_df.sort_values(['Universe', 'L1_C'], ascending=[True, False], inplace=True)

    sample_uni = agg_df[agg_df['Universe'] == agg_df['Universe'].iloc[0]]
    if sample_uni.empty: return None

    x_indices = np.arange(len(sample_uni)) 
    x_breadth_labels = sample_uni['L1_C'].values.astype(int)
    x_depth_labels = sample_uni['L1_K'].values.astype(int)
    approx_kids = int(sample_uni['L1_C'].iloc[0] * sample_uni['L1_K'].iloc[0])

    fig, ax1 = plt.subplots(figsize=(14, 8), dpi=100)

    for uni in [u for u in uni_colors.keys() if u in agg_df['Universe'].unique()]:
        data = agg_df[agg_df['Universe'] == uni]
        ax1.errorbar(x_indices, data['mean_acc'], yerr=data['ci95'], 
                     fmt='o-', markersize=8, color=uni_colors[uni], linewidth=2.5, 
                     capsize=5, capthick=1.5, label=uni, zorder=5)

    ax1.axhline(y=target_count, color='black', linestyle=':', linewidth=2.5, alpha=0.8, zorder=3)

    ax1.set_xlim(min(x_indices) - 0.5, max(x_indices) + 0.5)
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(x_breadth_labels, fontsize=12) 
    ax1.set_xlabel('BREADTH: No. of L0 Visited by L1 (Out of 25)', fontsize=14, fontweight='bold', labelpad=15)
    ax1.set_ylabel(f'Top {target_count} Worst L1 Regions Caught (%)', fontsize=14, fontweight='bold')

    ax2 = ax1.twiny()
    ax2.xaxis.set_ticks_position('bottom')
    ax2.xaxis.set_label_position('bottom')
    ax2.spines['bottom'].set_position(('outward', 60)) 

    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(x_indices)
    ax2.set_xticklabels(x_depth_labels, fontsize=12)
    ax2.set_xlabel(f'DEPTH: No. of Kids Measured per L0 [Budget ≈ {approx_kids} Kids]', fontsize=14, fontweight='bold', color='#333333', labelpad=10)

    ax1.set_ylim(0, 105)
    ax1.grid(True, linestyle=':', alpha=0.6)

    for ax in [ax1, ax2]:
        for spine in ax.spines.values():
            spine.set_linewidth(2.0)
            spine.set_color('black')

    ax1.legend(title='Simulated Universe', bbox_to_anchor=(0.5, 1.15), loc='upper center', 
               ncol=5, fontsize=11, framealpha=0.9, shadow=True)

    plt.title(f"L1 Sampling Strategy in Ranking L1 Regions: [{metric_label}]\n(Fixed Budget: {l1_budget_str})", 
              fontsize=16, pad=75, fontweight='bold')

    plt.subplots_adjust(bottom=0.25, top=0.82) 
    return fig


def plot_4_robustness(df, metric_col, metric_label, selected_uni):
    df_uni = df[df['Universe'] == selected_uni].copy()
    if df_uni.empty: return None

    agg_df = df_uni.groupby(['L1_Pct_Num', 'L2_Pct_Num']).agg(
        mean_acc=(metric_col, 'mean'), std_acc=(metric_col, 'std'), count=(metric_col, 'count')
    ).reset_index()
    agg_df['std_acc'] = agg_df['std_acc'].fillna(0)
    agg_df['ci95'] = 1.96 * (agg_df['std_acc'] / np.sqrt(agg_df['count']))

    fig, ax = plt.subplots(figsize=(12, 7), dpi=100)
    colors = {20: '#e74c3c', 40: '#f39c12', 60: '#2ecc71', 80: '#3498db', 100: '#9b59b6'}

    for l1_pct in sorted(agg_df['L1_Pct_Num'].unique()):
        subset = agg_df[agg_df['L1_Pct_Num'] == l1_pct].sort_values('L2_Pct_Num')
        ax.errorbar(subset['L2_Pct_Num'], subset['mean_acc'], yerr=subset['ci95'], 
                    fmt='o-', color=colors.get(l1_pct, '#333'), linewidth=3, markersize=8, 
                    capsize=5, label=f"L1 Base Budget: {l1_pct}%")

    ax.axhline(y=target_count, color='black', linestyle=':', linewidth=2)
    ax.set_title(f'{selected_uni.upper()}: L2 Audit Accuracy vs L1 Spreadsheet [{metric_label}]', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("L2 Sample Size (% of Children Re-measured from L1's Spreadsheet)", 
                  fontsize=14, fontweight='bold', labelpad=15)
    ax.set_ylabel(f'Accuracy\n(Top {target_count} Corrupt Supervisors Caught)', 
                  fontsize=14, fontweight='bold', labelpad=15)

    x_ticks = sorted(agg_df['L2_Pct_Num'].unique())
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{x}%" for x in x_ticks], fontsize=12)
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(title='Supervisor (L1) Base Policy', loc='lower right', fontsize=12, framealpha=0.9, shadow=True)

    for spine in ax.spines.values():
        spine.set_linewidth(2.0)
        spine.set_color('black')
    plt.tight_layout()
    return fig


def plot_5_master_grid(df, metric_col, metric_label, l1_budget_str):
    df_filtered = df[df['L1_Budget_Pct'] == l1_budget_str].copy()
    if df_filtered.empty: return None

    agg_df = df_filtered.groupby(['Universe', 'L2_Budget_Pct', 'L2_C']).agg(
        mean_acc=(metric_col, 'mean'), std_acc=(metric_col, 'std'), count=(metric_col, 'count')
    ).reset_index()
    agg_df['std_acc'] = agg_df['std_acc'].fillna(0)
    agg_df['ci95'] = 1.96 * (agg_df['std_acc'] / np.sqrt(agg_df['count']))

    budget_styles = {
        '20%': {'color': '#e74c3c', 'label': 'L2 Budget: 20%'},
        '40%': {'color': '#f39c12', 'label': 'L2 Budget: 40%'},
        '60%': {'color': '#2ecc71', 'label': 'L2 Budget: 60%'},
        '80%': {'color': '#3498db', 'label': 'L2 Budget: 80%'},
        '100%': {'color': '#9b59b6', 'label': 'L2 Budget: 100%'}
    }

    fig, axes = plt.subplots(3, 2, figsize=(16, 16), dpi=100)
    axes = axes.flatten()

    universes = [u for u in uni_colors.keys() if u in agg_df['Universe'].unique()]

    for i, uni in enumerate(universes):
        ax = axes[i]
        uni_data = agg_df[agg_df['Universe'] == uni]

        for budget_name, style in budget_styles.items():
            subset = uni_data[uni_data['L2_Budget_Pct'] == budget_name].sort_values(by='L2_C')
            if subset.empty: continue
            ax.errorbar(subset['L2_C'], subset['mean_acc'], yerr=subset['ci95'], 
                        fmt='o-', color=style['color'], linewidth=2.5, markersize=7, 
                        capsize=4, capthick=1.5, label=style['label'])

        ax.axhline(y=target_count, color='black', linestyle=':', linewidth=2)
        ax.set_title(f"{uni.upper()}", fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel('L2 BREADTH: Clinics Audited', fontsize=11, fontweight='bold')
        ax.set_ylabel(f'Top {target_count} Worst Caught', fontsize=11, fontweight='bold')
        ax.set_xlim(left=0)
        ax.set_ylim(0, 105)
        ax.grid(True, linestyle='--', alpha=0.5)

        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color('black')

    ax_legend = axes[5]
    ax_legend.axis('off') 
    legend_elements = [Line2D([0], [0], marker='o', color=style['color'], label=style['label'], 
                              markersize=10, linewidth=3) for budget_name, style in budget_styles.items()]
    ax_legend.legend(handles=legend_elements, loc='center', fontsize=14, frameon=True, shadow=True, 
                     title=f"L2 Budgets (Base L1 = {l1_budget_str})", title_fontsize=16)

    plt.suptitle(f"L2 Breadth Optimization [{metric_label}]", fontsize=20, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0.02, 0.02, 1, 0.95]) 
    return fig


def plot_6_heatmap(df, metric_col_l1, metric_col_l2, metric_label, selected_uni, l1_pct_str, l2_pct_str):
    df_hm = df[(df['Universe'] == selected_uni) & (df['L1_Budget_Pct'] == l1_pct_str) & (df['L2_Budget_Pct'] == l2_pct_str)].copy()
    if df_hm.empty: return None

    agg_df = df_hm.groupby(['L1_Label', 'L2_Label', 'L2_K']).agg(
        L1_Acc=(metric_col_l1, 'mean'), L2_Acc=(metric_col_l2, 'mean')
    ).reset_index()

    l1_order = sorted(agg_df['L1_Label'].unique(), key=lambda x: int(x.split('C')[0]), reverse=True)
    vmin_l1, vmax_l1 = agg_df['L1_Acc'].min(), agg_df['L1_Acc'].max()
    vmin_l2, vmax_l2 = agg_df['L2_Acc'].min(), agg_df['L2_Acc'].max()

    fig = plt.figure(figsize=(16, 12), dpi=100)
    gs = gridspec.GridSpec(nrows=len(l1_order), ncols=2, width_ratios=[1, 6], wspace=0.1, hspace=0.8)

    sns.set_theme(style="white")

    for i, l1_lbl in enumerate(l1_order):
        ax_l1 = fig.add_subplot(gs[i, 0]) 
        ax_l2 = fig.add_subplot(gs[i, 1]) 

        subset = agg_df[agg_df['L1_Label'] == l1_lbl].sort_values(by='L2_K')

        l1_acc_value = subset['L1_Acc'].iloc[0] if not subset.empty else 0
        sns.heatmap(np.array([[l1_acc_value]]), annot=True, fmt=".1f", cmap="Blues", 
                    cbar=False, linewidths=2, linecolor='white', vmin=vmin_l1, vmax=vmax_l1, 
                    ax=ax_l1, annot_kws={"size": 14, "weight": "bold"})

        ax_l1.set_xticks([])
        ax_l1.set_yticks([0.5])
        ax_l1.set_yticklabels([l1_lbl], rotation=0, fontsize=14, fontweight='bold')
        if i == 0: ax_l1.set_title("L1 Baseline Accuracy", fontsize=12, fontweight='bold', pad=10)

        heatmap_data_l2 = subset[['L2_Acc']].T 
        l2_labels = subset['L2_Label'].tolist()

        sns.heatmap(heatmap_data_l2, annot=True, fmt=".1f", cmap="RdYlGn", 
                    cbar=False, linewidths=2, linecolor='white', vmin=vmin_l2, vmax=vmax_l2, 
                    ax=ax_l2, annot_kws={"size": 13, "weight": "bold"})

        ax_l2.set_yticks([])
        ax_l2.set_xticks(np.arange(len(l2_labels)) + 0.5)
        ax_l2.set_xticklabels(l2_labels, rotation=0, fontsize=12, fontweight='bold')
        ax_l2.set_xlabel(r"Increasing L2 Depth $\longrightarrow$", fontsize=11, fontweight='bold', color='grey')
        if i == 0: ax_l2.set_title("L2 Auditor Execution Options", fontsize=12, fontweight='bold', pad=10)

        for ax in [ax_l1, ax_l2]:
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(2)
                spine.set_color('black')

    fig.suptitle(f"{selected_uni.upper()} | {metric_label}\n(Budgets: L1={l1_pct_str}, L2={l2_pct_str})", 
                 fontsize=18, fontweight='bold', y=1.02)
    return fig


# ==============================================================================
# 3. DASHBOARD TABS
# ==============================================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 1. L1 Global Sensitivity", 
    "🎯 2. L1 Intra-Regional", 
    "⚖️ 3. L1 Breadth vs Depth", 
    "🛡️ 4. L2 Budget Robustness", 
    "📊 5. 3x2 L2 Breadth Matrix", 
    "🔥 6. Heatmap Matrix"
])

with tab1:
    st.markdown("### Standalone Sensitivity: L1 Global Accuracy")
    sel_unis_t1 = st.multiselect("Filter Universes for Plot 1:", options=list(uni_colors.keys()), default=list(uni_colors.keys()))
    if sel_unis_t1:
        fig1 = plot_1_sensitivity(df, v1_col, selected_metric_label, sel_unis_t1)
        st.pyplot(fig1)
        st.markdown("""
        <div style="background-color: #f8f9fa; border-left: 6px solid #2c3e50; padding: 20px; border-radius: 5px; margin-top: 20px;">
            <h4 style="margin-top: 0;">📊 Analytical Brief: L1 Baseline Accuracy</h4>
            <b>Parameters Used:</b> Varying L1 Overall Budget (X-axis).<br>
            <b>Hypothesis:</b> "Good L1s" track upward as budget increases. "Bad L1s" flatline because they are copying data.<br>
            <b>Conclusions:</b> Without L2, the system cannot distinguish between a highly accurate "Good L1" and a lazy "Bad L1" copying an honest clinic's homework.
        </div>
        """, unsafe_allow_html=True)

with tab2:
    st.markdown("### L1 Ranking of L0 Clinics (Intra-Regional Accuracy)")
    sel_unis_t2 = st.multiselect("Filter Universes for Plot 2:", options=list(uni_colors.keys()), default=list(uni_colors.keys()), key="ms_t2")
    if sel_unis_t2:
        fig2 = plot_2_intra_regional(df, v2_col, selected_metric_label, sel_unis_t2)
        st.pyplot(fig2)
        st.markdown("""
        <div style="background-color: #f8f9fa; border-left: 6px solid #2980b9; padding: 20px; border-radius: 5px; margin-top: 20px;">
            <h4 style="margin-top: 0;">📊 Analytical Brief: L1 Intra-Regional Accuracy (L1 Ranking L0)</h4>
            <b>Hypothesis:</b> If underlying clinics are honest, a lazy supervisor gets a high score because copied data is true.<br>
            <b>Conclusions:</b> Blind Spot (Yellow Line) performs perfectly. This proves intra-regional ranking validates the data, but fails to validate the supervisor's physical effort.
        </div>
        """, unsafe_allow_html=True)

with tab3:
    # 1. Streamlit Title placed ABOVE the filters
    st.markdown("### ⚖️ L1 Breadth vs Depth Optimization (Fixed Budget)")

    # 2. Filters placed BELOW the Streamlit Title
    col1, col2 = st.columns([1, 1])
    sel_unis_t3 = col1.multiselect("Filter Universes:", options=list(uni_colors.keys()), default=list(uni_colors.keys()), key="ms_t3")
    sel_l1_pct_t3 = col2.selectbox("Select Fixed L1 Budget:", options=sorted(df['L1_Budget_Pct'].unique(), key=lambda x: int(x.replace('%',''))), index=2)

    # 3. Render the Plot (Matplotlib generates its own internal title)
    if sel_unis_t3:
        fig3 = plot_3_bd_optimization(df, v1_col, selected_metric_label, sel_unis_t3, sel_l1_pct_t3)
        if fig3: 
            st.pyplot(fig3)
        else: 
            st.warning("No data available for this configuration.")

        # 4. Analytical Brief at the bottom
        st.markdown("""
        <div style="background-color: #f8f9fa; border-left: 6px solid #27ae60; padding: 20px; border-radius: 5px; margin-top: 20px;">
            <h4 style="margin-top: 0;">📊 Analytical Brief: L1 Breadth vs. Depth Trade-off</h4>
            <b>Parameters Used:</b> Budget is strictly frozen. Moves from maximum Breadth (Left) to maximum Depth (Right).<br>
            <b>Hypothesis:</b> Spreading measurements thinly across many clinics (Breadth) detects widespread/systemic fraud better than deeply auditing a few clinics.<br>
            <b>Conclusions & Implications:</b> Notice how the lines behave as you move left to right. In corrupt universes, extreme Depth (right side) causes detection rates to plummet because corrupt clinics that were left unvisited entirely drag down the whole region's ranking. Wide Breadth (left side) is almost always safer for fraud detection.
        </div>
        """, unsafe_allow_html=True)

with tab4:
    st.markdown("### L2 Robustness by L1 Budget")
    sel_uni_t4 = st.selectbox("Select Universe", options=df['Universe'].unique(), key='t4_uni', index=list(df['Universe'].unique()).index("Normal") if "Normal" in df['Universe'].unique() else 0)
    fig4 = plot_4_robustness(df, v3_col, selected_metric_label, sel_uni_t4)
    if fig4: st.pyplot(fig4)

    st.markdown("""
    <div style="background-color: #f8f9fa; border-left: 6px solid #2980b9; padding: 20px; border-radius: 5px; margin-top: 20px;">
        <h4 style="margin-top: 0;">📊 Analytical Brief: L2 Auditor Robustness</h4>
        <b>Parameters Used:</b> Universe isolated test. Compares L2 Audit Percentage (X-Axis) against Accuracy (Y-Axis) for different underlying L1 budgets.<br>
        <b>Hypothesis:</b> The L2 auditor's ability to catch fraud is directly constrained by how much data L1 originally collected. If L1 only collected a tiny sample (20%), even a 100% audit by L2 will hit a strict mathematical ceiling.<br>
        <b>Conclusions:</b> Notice how the lines flatten out. To catch the Top 100 corrupt regions, L2 requires L1 to have a sufficiently large base budget. A high L2 audit percentage cannot save a system where the L1 baseline data is severely lacking.
    </div>
    """, unsafe_allow_html=True)

with tab5:
    st.markdown("### L2 Breadth Optimization (Across 5 Universes)")
    sel_l1_pct_t5 = st.selectbox("Filter: Fixed L1 Base Budget", options=sorted(df['L1_Budget_Pct'].unique(), key=lambda x: int(x.replace('%',''))), index=4, key='t5_l1_sel')
    fig5 = plot_5_master_grid(df, v3_col, selected_metric_label, sel_l1_pct_t5)
    if fig5: st.pyplot(fig5)

    st.markdown("""
    <div style="background-color: #f8f9fa; border-left: 6px solid #2980b9; padding: 20px; border-radius: 5px; margin-top: 20px;">
        <h4 style="margin-top: 0;">📊 Analytical Brief: Breadth vs. Depth Matrix</h4>
        <b>Parameters Used:</b> L1 Budget is fixed. Evaluates L2 Breadth (number of clinics visited, X-axis) across 5 Universes and 5 L2 Budgets.<br>
        <b>Hypothesis:</b> In highly corrupt universes (Mafia), auditing a few kids across many clinics (High Breadth) is more effective than heavily auditing a few clinics (High Depth), because fraud is systemic rather than isolated.<br>
        <b>Conclusions:</b> In "Good L0 Good L1" universes, extreme breadth introduces statistical noise, occasionally causing accuracy to dip. However, in "Bad L0 Bad L1" (Mafia) scenarios, pushing the X-axis to the right (visiting more clinics) consistently yields higher detection rates for the same budget.
    </div>
    """, unsafe_allow_html=True)

with tab6:
    st.markdown("### L1 vs L2 Split-Pane Heatmap")
    c1, c2, c3 = st.columns(3)
    sel_uni_t6 = c1.selectbox("Select Universe", options=df['Universe'].unique(), key='t6_uni')
    sel_l1_pct_t6 = c2.selectbox("L1 Base Budget", options=sorted(df['L1_Budget_Pct'].unique(), key=lambda x: int(x.replace('%',''))), index=2, key='t6_l1')
    sel_l2_pct_t6 = c3.selectbox("L2 Audit Budget", options=sorted(df['L2_Budget_Pct'].unique(), key=lambda x: int(x.replace('%',''))), index=1, key='t6_l2')
    fig6 = plot_6_heatmap(df, v1_col, v3_col, selected_metric_label, sel_uni_t6, sel_l1_pct_t6, sel_l2_pct_t6)
    if fig6: st.pyplot(fig6)

    st.markdown("""
    <div style="background-color: #f8f9fa; border-left: 6px solid #2980b9; padding: 20px; border-radius: 5px; margin-top: 20px;">
        <h4 style="margin-top: 0;">📊 Analytical Brief: L2 Tactical Execution Menu</h4>
        <b>Parameters Used:</b> Isolates a specific Universe, L1 Budget, and L2 Budget. Displays L1 baseline accuracy (left) vs. L2 strategy execution (right).<br>
        <b>Hypothesis:</b> Even with fixed budgets, the specific deployment of L2 auditors (Depth vs. Breadth) dictates success. The "greenest" square on the right side represents the optimal tactical deployment for that specific scenario.<br>
        <b>Conclusions:</b> This heatmap serves as the operational menu for field deployment. If L1 deployed a specific strategy (left column), L2 leadership can scan the corresponding row to the right to find the strategy with the highest accuracy (darkest green) that fits their budget constraints.
    </div>
    """, unsafe_allow_html=True)
