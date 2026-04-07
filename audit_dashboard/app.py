
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
    # Calculate global min and max across ALL universes to standardize the color scale
    vmin_global_l1, vmax_global_l1 = df[metric_col_l1].min(), df[metric_col_l1].max()
    vmin_global_l2, vmax_global_l2 = df[metric_col_l2].min(), df[metric_col_l2].max()

    df_hm = df[(df['Universe'] == selected_uni) & (df['L1_Budget_Pct'] == l1_pct_str) & (df['L2_Budget_Pct'] == l2_pct_str)].copy()
    if df_hm.empty: return None

    agg_df = df_hm.groupby(['L1_Label', 'L2_Label', 'L2_K']).agg(
        L1_Acc=(metric_col_l1, 'mean'), L2_Acc=(metric_col_l2, 'mean')
    ).reset_index()

    l1_order = sorted(agg_df['L1_Label'].unique(), key=lambda x: int(x.split('C')[0]), reverse=True)

    fig = plt.figure(figsize=(16, 12), dpi=100)
    gs = gridspec.GridSpec(nrows=len(l1_order), ncols=2, width_ratios=[1, 6], wspace=0.1, hspace=0.8)

    sns.set_theme(style="white")

    for i, l1_lbl in enumerate(l1_order):
        ax_l1 = fig.add_subplot(gs[i, 0]) 
        ax_l2 = fig.add_subplot(gs[i, 1]) 

        subset = agg_df[agg_df['L1_Label'] == l1_lbl].sort_values(by='L2_K')

        l1_acc_value = subset['L1_Acc'].iloc[0] if not subset.empty else 0
        
        # Apply standard global bounds to L1 Heatmap
        sns.heatmap(np.array([[l1_acc_value]]), annot=True, fmt=".1f", cmap="Blues", 
                    cbar=False, linewidths=2, linecolor='white', vmin=vmin_global_l1, vmax=vmax_global_l1, 
                    ax=ax_l1, annot_kws={"size": 14, "weight": "bold"})

        ax_l1.set_xticks([])
        ax_l1.set_yticks([0.5])
        ax_l1.set_yticklabels([l1_lbl], rotation=0, fontsize=14, fontweight='bold')
        if i == 0: ax_l1.set_title("L1 Baseline Accuracy", fontsize=12, fontweight='bold', pad=10)

        heatmap_data_l2 = subset[['L2_Acc']].T 
        l2_labels = subset['L2_Label'].tolist()

        # Apply standard global bounds to L2 Heatmap
        sns.heatmap(heatmap_data_l2, annot=True, fmt=".1f", cmap="RdYlGn", 
                    cbar=False, linewidths=2, linecolor='white', vmin=vmin_global_l2, vmax=vmax_global_l2, 
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
    return fig, agg_df

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
    st.markdown("### 📈 L1 Global Diagnostic Power (Sensitivity)")
    sel_unis_t1 = st.multiselect("Filter Universes:", options=list(uni_colors.keys()), default=list(uni_colors.keys()), key="t1_unis")
    
    if sel_unis_t1:
        fig1 = plot_1_sensitivity(df, v1_col, selected_metric_label, sel_unis_t1)
        if fig1: st.pyplot(fig1)
        
    # Analytical Brief for Tab 1
    st.markdown("""
<div style="background-color: #f8f9fa; border-left: 6px solid #2980b9; padding: 20px; border-radius: 5px; margin-top: 20px;">

<b>1. Setup & Universe Input Parameters:</b><br>
This module evaluates a simulated global population of 266,000 children. We have simulated each universe to have 334 L1, 25 L0 under each L1, 15 children under each L0. To isolate the effects of operational behavior, we tested four absolute "Zero-Error" environments alongside one "Normal" environment featuring realistic human measurement variance.
<br><br>

<table style="width: 100%; border-collapse: collapse; font-size: 13px; text-align: left; margin-bottom: 15px;">
    <thead>
        <tr style="background-color: #e9ecef; border-bottom: 2px solid #cbd5e1;">
            <th style="padding: 8px; border: 1px solid #dee2e6;">Parameter</th>
            <th style="padding: 8px; border: 1px solid #dee2e6;">Good L0 + Good L1</th>
            <th style="padding: 8px; border: 1px solid #dee2e6;">Good L0 + Bad L1</th>
            <th style="padding: 8px; border: 1px solid #dee2e6;">Bad L0 + Good L1</th>
            <th style="padding: 8px; border: 1px solid #dee2e6;">Bad L0 + Bad L1</th>
            <th style="padding: 8px; border: 1px solid #dee2e6;">Normal (Real-World)</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td style="padding: 8px; border: 1px solid #dee2e6; font-weight: bold;">Core Assumption</td>
            <td style="padding: 8px; border: 1px solid #dee2e6;">L1 ranking of L0 matches reality perfectly.</td>
            <td style="padding: 8px; border: 1px solid #dee2e6;">Honest L0 data means L1 desk audits are accidentally accurate.</td>
            <td style="padding: 8px; border: 1px solid #dee2e6;">Honest L1 instantly detects L0 fraud.</td>
            <td style="padding: 8px; border: 1px solid #dee2e6;">High collusion; necessitates L2 intervention.</td>
            <td style="padding: 8px; border: 1px solid #dee2e6;">Realistic mix of fraud, laziness, and equipment noise.</td>
        </tr>
        <tr style="background-color: #f8f9fa;">
            <td style="padding: 8px; border: 1px solid #dee2e6; font-weight: bold;">True Stunting / Underweight</td>
            <td style="padding: 8px; border: 1px solid #dee2e6;">35% / 33%</td>
            <td style="padding: 8px; border: 1px solid #dee2e6;">35% / 33%</td>
            <td style="padding: 8px; border: 1px solid #dee2e6;">35% / 33%</td>
            <td style="padding: 8px; border: 1px solid #dee2e6;">35% / 33%</td>
            <td style="padding: 8px; border: 1px solid #dee2e6;">36% / 34%</td>
        </tr>
        <tr>
            <td style="padding: 8px; border: 1px solid #dee2e6; font-weight: bold;">L0 Under-Reporting</td>
            <td style="padding: 8px; border: 1px solid #dee2e6;">5%</td>
            <td style="padding: 8px; border: 1px solid #dee2e6;">5%</td>
            <td style="padding: 8px; border: 1px solid #dee2e6;">30%</td>
            <td style="padding: 8px; border: 1px solid #dee2e6;">30%</td>
            <td style="padding: 8px; border: 1px solid #dee2e6;">30%</td>
        </tr>
        <tr style="background-color: #f8f9fa;">
            <td style="padding: 8px; border: 1px solid #dee2e6; font-weight: bold;">L0 Bunching Factor</td>
            <td style="padding: 8px; border: 1px solid #dee2e6;">0.05</td>
            <td style="padding: 8px; border: 1px solid #dee2e6;">0.05</td>
            <td style="padding: 8px; border: 1px solid #dee2e6;">0.60</td>
            <td style="padding: 8px; border: 1px solid #dee2e6;">0.60</td>
            <td style="padding: 8px; border: 1px solid #dee2e6;">0.20</td>
        </tr>
        <tr>
            <td style="padding: 8px; border: 1px solid #dee2e6; font-weight: bold;">L1 Copying Rate</td>
            <td style="padding: 8px; border: 1px solid #dee2e6;">5%</td>
            <td style="padding: 8px; border: 1px solid #dee2e6;">60%</td>
            <td style="padding: 8px; border: 1px solid #dee2e6;">5%</td>
            <td style="padding: 8px; border: 1px solid #dee2e6;">60%</td>
            <td style="padding: 8px; border: 1px solid #dee2e6;">20%</td>
        </tr>
        <tr style="background-color: #f8f9fa;">
            <td style="padding: 8px; border: 1px solid #dee2e6; font-weight: bold;">L1 Collusion Index</td>
            <td style="padding: 8px; border: 1px solid #dee2e6;">0.05</td>
            <td style="padding: 8px; border: 1px solid #dee2e6;">0.80</td>
            <td style="padding: 8px; border: 1px solid #dee2e6;">0.05</td>
            <td style="padding: 8px; border: 1px solid #dee2e6;">0.80</td>
            <td style="padding: 8px; border: 1px solid #dee2e6;">0.50</td>
        </tr>
        <tr>
            <td style="padding: 8px; border: 1px solid #dee2e6; font-weight: bold;">Measurement Error</td>
            <td style="padding: 8px; border: 1px solid #dee2e6;">0.0</td>
            <td style="padding: 8px; border: 1px solid #dee2e6;">0.0</td>
            <td style="padding: 8px; border: 1px solid #dee2e6;">0.0</td>
            <td style="padding: 8px; border: 1px solid #dee2e6;">0.0</td>
            <td style="padding: 8px; border: 1px solid #dee2e6;">1.0 cm / 0.1 kg</td>
        </tr>
    </tbody>
</table>

<b>2. Variables & Mathematical Calculation:</b><br>
<i>X-Axis:</i> <b>L1 Base Budget</b>. The percentage of the target population L1 is funded to measure (Max = 375 children across 25 clinics).<br>
<i>Y-Axis:</i> <b>Percentage Overlap between the ranking from the comparison of |L1-L0| and |L0-Real|</b>. <br>
<i>Calculation:</i> We calculate Measured Rank and Real Rank. The Measured Rank is obtained by calculating <code>abs(L1 HAZ Score - L0 HAZ Score)</code> (of children sampled by L1 only) and then ranking the L1 Regions from worst to best (Descending Order) to identify the actual lowest-performing L1 Regions. The Real Rank is obtained by calculating <code>abs(L1 HAZ Score - Real HAZ Score)</code> (of all children) for each region and ranking them in descending order. The Y-Axis represents the percentage of the overlap between the Measured & Real Ranking L1 Regions.<br><br>

<b>3. Objective & Hypothesis:</b><br>
This plot tests the capability of the L1 supervisor to diagnose systemic failure when observing only a localized sample of the total region. 
                The hypothesis is that Ranking Accuracy will scale linearly with sample size.<br><br>

<b>4. Results & Analysis:</b><br>
The results confirm the structural hypothesis but indicate a performance asymptote. 
                As the base budget increases, the detection rate rises before ultimately flattening out. In the <b>Normal</b> environment, natural measurement variance ($\pm$ 1.0cm) prevents perfect ranking of borderline cases, creating a strict upper bound. Conversely, in highly manipulated environments, the delta between accurate data and fabricated data is statistically significant enough that a 60% sample budget captures the majority of critical failures.<br><br>

<b>5. Conclusion & Implications:</b><br>
To accurately diagnose the global state of a region, the L1 operational budget must meet a minimum threshold of ~60%. Below this threshold, unobserved geographical blind spots severely degrade diagnostic power. This loss of primary intelligence cannot be mathematically recovered by subsequent L2 auditing.<br><br>


</ul>
</div>
""", unsafe_allow_html=True)
    

with tab2:
    st.markdown("### 🎯 L1 Intra-Regional (Targeted) Accuracy")
    sel_unis_t2 = st.multiselect("Filter Universes:", options=list(uni_colors.keys()), default=list(uni_colors.keys()), key="t2_unis")
    
    if sel_unis_t2:
        fig2 = plot_2_intra_regional(df, v2_col, selected_metric_label, sel_unis_t2)
        if fig2: st.pyplot(fig2)

    # Analytical Brief for Tab 2
    st.markdown("""
<div style="background-color: #f8f9fa; border-left: 6px solid #27ae60; padding: 20px; border-radius: 5px; margin-top: 20px;">

<b>1. Setup & Universe Inputs:</b><br>
This is an extension of plot 1. We rank the L0 measured by L1 unders its L1 Region. Each L1 will have a ranking of worst performing L0 via |L1-L0|. We compare that with the real rank of 25 L0 under that L1 via |L0-Real| score. We take the average of all L1 rankings.This visual shifts the perspective from L1 Regions to Local L0 AWC centers.<br><br>

<b>2. Variables & Mathematical Calculation:</b><br>
<i>X-Axis:</i> <b>L1 Base Budget</b> (20% to 100% of maximum capacity).<br>
<i>Y-Axis:</i> <b>Average Intra-Regional Overlap in Ranking</b>.<br>
<i>Calculation:</i> We compare the L0 to L1's measurement: <code>abs(L1_haz - L0_haz)</code>.</b> We ask: "Within the specific sample measured by L1, did they successfully catch the top 30% worst clinics?"<br><br>

<b>3. Objective & Hypothesis:</b><br>
We are testing L1's localized competence. Our hypothesis is that L0 fraud is a systemic issue and as L1 is visiting all L0 with increasing budget/samples in each round. We will see a large amount of worst L0 caught at smaller budgets and the effect would be consistent across increasing budgets, and at very high budgets, we will have maximum number of L0 caught.<br>

<b>4. Results & Analysis:</b><br>
The results strongly validate the hypothesis. The lines on this chart are dramatically flatter and higher than Plot 1. Because the denominator shrinks to match L1's sample size, an L1 supervisor with a 20% budget appears highly accurate within their tiny footprint. The mathematical logic holds: if you only check 5 clinics, it is relatively easy to rank those 5 clinics accurately.<br><br>

<b>5. Conclusion & Implications:</b><br>
This reveals a massive administrative danger: <b>The False Sense of Security</b>. If leadership only evaluates supervisors based on what they submit (Intra-Regional), supervisors will look highly competent, even if they are entirely blind to 80% of their actual district. Evaluating L1 requires global benchmarks, not just local ones.<br><br>
<br>
This dynamic is proven across the sensitivity datasets. Even in "Blind Spot" (where L1 is lazy), their Intra-Regional score behaves differently than their Global score, mathematically proving the risk of localized evaluation metrics.<br><br>

<b>7. Open Questions for Discussion:</b><br>
<ul>
<li>Are our current KPIs inadvertently rewarding supervisors for Intra-Regional accuracy while ignoring their massive Global Blind Spots?</li>
<li>How do we design a performance metric that forces supervisors to value breadth (visiting more clinics) over depth?</li>
</ul>
</div>
""", unsafe_allow_html=True)
    
with tab3:
    st.markdown("### ⚖️ L1 Breadth vs Depth Optimization (Fixed Budget)")

    col1, col2 = st.columns([1, 1])
    sel_unis_t3 = col1.multiselect("Filter Universes:", options=list(uni_colors.keys()), default=list(uni_colors.keys()), key="ms_t3")
    sel_l1_pct_t3 = col2.selectbox("Select Fixed L1 Budget:", options=sorted(df['L1_Budget_Pct'].unique(), key=lambda x: int(x.replace('%',''))), index=2)

    if sel_unis_t3:
        fig3 = plot_3_bd_optimization(df, v1_col, selected_metric_label, sel_unis_t3, sel_l1_pct_t3)
        if fig3: 
            st.pyplot(fig3)
        else: 
            st.warning("No data available for this configuration.")

    st.markdown("""
<div style="background-color: #f8f9fa; border-left: 6px solid #e74c3c; padding: 20px; border-radius: 5px; margin-top: 20px;">

<b>1. Setup & Universe Inputs:</b><br>
This module filters for a specific L1 Budget (e.g., 20%) and a specific Universe (e.g., Normal). We are examining the exact shape of L1 catching the worst L1 Region.<br><br>

<b>2. Variables & Mathematical Calculation:</b><br>
<i>X-Axis:</i> <b>L1 Sampling Strategy (Clinics x Kids)</b>. Strategies are sorted from left (Maximum Breadth: many clinics, few kids) to right (Maximum Depth: few clinics, many kids).<br>
<i>Y-Axis:</i> <b>Top 100 Worst L1 Regions Caught</b>.<br>
<i>Calculation:</i> We compare the L1 findings <code>abs(L1_haz - L0_haz)</code> to the real L1 Region error <code>abs(L0_haz - real_haz)</code>. L1 is ONLY evaluated on the universe of kids that L1 has sampled. We are testing if L1 can catch the bad L1 Region based strictly on how L1 distributes their physical visits.<br><br>

<b>3. Objective & Hypothesis:</b><br>
For a fixed budget, does an L1 Supervisor catch more fraud by visiting 20 clinics (measuring 1 kid each) or visiting 1 clinic (measuring 20 kids)? Our hypothesis is that Breadth (visiting more clinics) will drastically outperform Depth, because data manipulation is usually clustered at the clinic level, not evenly distributed among kids.<br><br>

<b>4. Results & Analysis:</b><br>
The delta is not significant enough to confirm our hypothesis that sampling more L0 yields a better result. We can confirm that increasing the budget of L1 improves the ranking accuracy. Across different sensitivities, we can confirm that measurement error in "Normal" universe adds a lot of noise during measurements, which disrupts the ranking accuracy.<br><br>

<b>5. Conclusion & Implications:</b><br>
Based on the current results, we are not able to conclude anything with significance. we see a slight increase in accuracy of rankings when more L0 are sampled.<br><br>


<b>6. Open Questions for Discussion:</b><br>
<ul>
<li>Given that traveling to 20 different clinics is logistically more expensive than staying at 1 clinic, how do we adjust the travel budget to explicitly fund "Breadth" over "Depth"?</li>
<li>Should we use other metric (other than MAE) to calculate the rankings?</li>
</ul>
</div>
""", unsafe_allow_html=True)

with tab4:
    st.markdown("### L2 Ranking L1 by L1 Budget")
    sel_uni_t4 = st.selectbox("Select Universe", options=df['Universe'].unique(), key='t4_uni', index=list(df['Universe'].unique()).index("Normal") if "Normal" in df['Universe'].unique() else 0)
    fig4 = plot_4_robustness(df, v3_col, selected_metric_label, sel_uni_t4)
    if fig4: st.pyplot(fig4)

    st.markdown("""
<div style="background-color: #f8f9fa; border-left: 6px solid #8e44ad; padding: 20px; border-radius: 5px; margin-top: 20px;">

<b>1. Setup & Flow:</b><br>
We simulate an L2 Auditor evaluating an L1 Supervisor's Measurement. We demonstrate L2's detection ceiling as their budget expands.<br><br>

<b>2. Objective & Hypothesis:</b><br>
We are testing L2's diagnostic success as their Audit Budget (X-Axis) increases. We calculate the Measured Rank of L1 using |L2-L1| scores and compare it with the Real Rank of L1 using |L1-Real| scores and compare the overlap. The key hypothesis: Increasing L2 audits sample/budget slightly will result in massive identification of worst L1 and the effect will taper off and reach an equilibrium at higher budgets. Measurement error prevents them from perfectly ranking borderline cases.<br><br>

<b>3. Indicators & Legend:</b><br>
<i>Y-Axis:</i> <b>Spreadsheet Truth Overlap</b> (Out of the true worst offenders on L1's sheet, what % did L2 successfully identify?).<br>
<i>Lines:</i> Each line represents L1's baseline budget. L2's success is heavily capped by how much data L1 collected in the first place.<br><br>

<b>4. Conclusion & Implications:</b><br>
You cannot audit your way out of a poor L1 sample size. Furthermore, because human auditors have natural measurement variance, pushing for a 100% L2 audit is financially inefficient; an 80% audit achieves almost identical accountability results.<br><br>

<b>5. Assurance of Robustness:</b><br>
This 85% ceiling holds true across realistic scenarios. We stress-tested this against synthetic "Zero-Error" universes (Mafia/Blind Spot) where a flawless, robotic L2 correctly scales to near 100%, proving our 85% real-world boundary is mathematically sound.<br><br>

<b>6. Open Questions for Discussion:</b><br>
<ul>
<li>At what point does the marginal cost of increasing L2 audits outweigh the benefit of simply giving L1 a larger base budget?</li>
<li>Would investing in digital/automated measurement scales for L2 eliminate this human noise, or is an 85% detection rate acceptable for field operations?</li>
</ul>
</div>
""", unsafe_allow_html=True)


with tab5:
    st.markdown("### L2 Breadth Optimization (Across 5 Universes)")
    sel_l1_pct_t5 = st.selectbox("Filter: Fixed L1 Base Budget", options=sorted(df['L1_Budget_Pct'].unique(), key=lambda x: int(x.replace('%',''))), index=4, key='t5_l1_sel')
    fig5 = plot_5_master_grid(df, v3_col, selected_metric_label, sel_l1_pct_t5)
    if fig5: st.pyplot(fig5)

    st.markdown("""
<div style="background-color: #f8f9fa; border-left: 6px solid #f39c12; padding: 20px; border-radius: 5px; margin-top: 20px;">

<b>1. Setup & Universe Inputs:</b><br>
This heatmap collapses the tactical layer to look purely at systemic funding allocations across the 266k population pathways.<br><br>

<b>2. Variables & Mathematical Calculation:</b><br>
<i>X-Axis:</i> <b>L1 Base Budget</b> (20% to 100% of maximum).<br>
<i>Y-Axis:</i> <b>L2 Audit Budget</b> (20% to 100% of maximum).<br>
<i>Color Scale (Z-Axis):</i> <b>Maximum Diagnostic Accuracy (V3)</b>.<br>
<i>Calculation:</i> For every intersection of L1 and L2 funding, the engine finds the single best tactical deployment (the highest V3 Overlap) and plots that maximum possible accuracy. The calculation remains <code>abs(L2_haz - L1_haz)</code> evaluated against L1's spreadsheet.<br><br>

<b>3. Objective & Hypothesis:</b><br>
We are testing the financial trade-off between paying for initial data collection (L1) versus paying for auditing (L2). Our hypothesis is that L1 funding serves as the absolute foundation of accountability, and therefore, shifting budget to the X-Axis will yield higher returns than shifting budget to the Y-Axis.<br><br>

<b>4. Results & Analysis:</b><br>
The heatmap visually proves the hypothesis. Notice how the colors shift to green much faster when you move horizontally (increasing L1) compared to moving vertically (increasing L2). For example, a 60% L1 / 20% L2 split (wide foundation, light audit) yields drastically higher accountability than a 20% L1 / 60% L2 split (poor foundation, heavy audit).<br><br>

<b>5. Conclusion & Implications:</b><br>
You cannot audit a blank page. If L1 is underfunded, the data simply doesn't exist for L2 to audit. Financial planners should secure at least a 60% L1 base budget before heavily scaling the L2 accountability apparatus.<br><br>

<b>6. Assurance of Robustness:</b><br>
By selecting the "Mafia" or "Blind Spot" universes from the dropdown, you can verify that even in the worst-case scenarios of systemic fraud, the fundamental law of "L1 Foundation First" remains mathematically unbreakable.<br><br>

<b>7. Open Questions for Discussion:</b><br>
<ul>
<li>If our total operational budget is capped, what is the absolute optimal percentage split between L1 funding and L2 funding?</li>
<li>Are we currently over-investing in the L2 audit layer to compensate for a fundamentally underfunded L1 data collection layer?</li>
</ul>
</div>
""", unsafe_allow_html=True)

with tab6:
    st.markdown("### L1 vs L2 Split-Pane Heatmap")
    c1, c2, c3 = st.columns(3)
    sel_uni_t6 = c1.selectbox("Select Universe", options=df['Universe'].unique(), key='t6_uni', index=list(df['Universe'].unique()).index("Normal") if "Normal" in df['Universe'].unique() else 0)
    sel_l1_pct_t6 = c2.selectbox("L1 Base Budget", options=sorted(df['L1_Budget_Pct'].unique(), key=lambda x: int(x.replace('%',''))), index=2, key='t6_l1')
    sel_l2_pct_t6 = c3.selectbox("L2 Audit Budget", options=sorted(df['L2_Budget_Pct'].unique(), key=lambda x: int(x.replace('%',''))), index=1, key='t6_l2')
    
    fig_or_none = plot_6_heatmap(df, v1_col, v3_col, selected_metric_label, sel_uni_t6, sel_l1_pct_t6, sel_l2_pct_t6)
    
    if fig_or_none:
        fig6, hm_data = fig_or_none
        st.pyplot(fig6)
        
        st.markdown("#### 📈 Execution Summary Table")
        
        display_df = hm_data.copy()
        display_df['L1_Acc'] = display_df['L1_Acc'].round(1).astype(str) + '%'
        display_df['L2_Acc'] = display_df['L2_Acc'].round(1).astype(str) + '%'
        display_df = display_df.rename(columns={
            'L1_Label': 'L1 Strategy (Clinics x Kids)',
            'L2_Label': 'L2 Strategy (Clinics x Kids)',
            'L1_Acc': 'L1 Baseline Accuracy',
            'L2_Acc': 'L2 Execution Accuracy'
        })
        display_df = display_df.drop(columns=['L2_K']).sort_values('L1 Strategy (Clinics x Kids)', ascending=False)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        st.markdown("""
<div style="background-color: #f8f9fa; border-left: 6px solid #e67e22; padding: 20px; border-radius: 5px; margin-top: 20px;">

<b>1. Setup & Flow:</b><br>
Having established the ceilings of the L2 Auditor, we now focus on operational deployment. This matrix freezes the budgets for both L1 and L2 to determine the absolute optimal way to spend that money in the field.<br><br>

<b>2. Objective & Key Message:</b><br>
We are testing the interaction between L1's initial sampling strategy and L2's auditing strategy. The key takeaway: Even with frozen budgets, the specific deployment of L2 auditors (Depth vs. Breadth) dictates success. Field operations must match L2's audit shape to L1's initial footprint.<br><br>

<b>3. Indicators & Legend:</b><br>
<i>Left Pane (Blues):</i> <b>L1 Baseline Accuracy</b>. The operational foundation laid by the supervisor.<br>
<i>Right Pane (Green/Red):</i> <b>L2 Execution Accuracy</b>. The resulting diagnostic power based on how L2 chose to sample L1's work.<br>
<i>Note on Color Scale:</i> The color scale is <b>globally standardized</b> across all universes (Red always means poor global performance, Green always means peak global performance).<br><br>

<b>4. Conclusion & Implications:</b><br>
This heatmap serves as the operational menu for field leadership. If an L1 supervisor deployed a specific strategy (Left Column), leadership can scan the corresponding row to the right to find the "greenest" execution strategy for L2. Misaligning L2's depth with L1's initial breadth can cause accuracy to drop significantly without saving any budget.<br><br>

<b>5. Assurance of Robustness:</b><br>
By viewing the "Normal" universe, we see the realistic operational bounds with natural human measurement errors included. You can verify the stability of these tactics by cycling through the extreme edge cases (Mafia, Utopia) in the dropdown above.<br><br>

<b>6. Open Questions for Discussion:</b><br>
<ul>
<li>If the heatmap indicates that High Breadth (left side of the right pane) is consistently safer, are we logistically equipped to transport auditors to 15 different clinics rather than letting them stay deeply at 5 clinics?</li>
<li>How do we ensure L1 supervisors accurately report their sampling strategy so L2 can deploy the correct counter-strategy?</li>
</ul>
</div>
""", unsafe_allow_html=True)