import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec

# ==============================================================================
# PLOT 2: INTRA-REGIONAL (OVERALL RETURN ON INVESTMENT)
# ==============================================================================
# ==============================================================================
# PLOT 2: INTRA-REGIONAL (OVERALL RETURN ON INVESTMENT)
# ==============================================================================
# ==============================================================================
# PLOT 2: INTRA-REGIONAL (OVERALL RETURN ON INVESTMENT)
# ==============================================================================
def plot_2_intra_regional(df, l1_pct_str, indicator, n_L0s_per_L1, n_children_per_L0, target_percentile):
    df = df.copy()
    
    # Extract the integer from the budget string (e.g., "60%" -> 60) for sorting
    if 'L1_Pct_Num' not in df.columns:
        df['L1_Pct_Num'] = df['L1_Budget_Pct'].str.replace('%', '').astype(int)
    df = df.sort_values('L1_Pct_Num')

    # Calculate dynamic targets based on user inputs
    TARGET_CLINICS = round(n_L0s_per_L1 * target_percentile)
    target_pct_label = int(target_percentile * 100)

    fig, ax1 = plt.subplots(figsize=(12, 7), dpi=100)
    
    # Check for standard deviation/variance columns, fallback to 0 if missing
    
    v2_std = df.get('V2_MAE_Std', df.get('V2_MAE_Acc_Std', np.zeros(len(df))))

    # Plot the shaded confidence interval first
    ax1.fill_between(df['L1_Pct_Num'], df['V2_MAE_Acc'] - v2_std, df['V2_MAE_Acc'] + v2_std, 
                    color='#27ae60', alpha=0.15, label='±1 Standard Deviation')

    # Plot the main trendline (Percentages)
    ax1.plot(df['L1_Pct_Num'], df['V2_MAE_Acc'], marker='o', linestyle='-', color='#27ae60', lw=2.5, ms=8, label='Simulation Average')

    # Add a red star to highlight the currently selected budget filter
    selected_num = int(l1_pct_str.replace('%', ''))
    selected_row = df[df['L1_Pct_Num'] == selected_num]
    if not selected_row.empty:
        ax1.plot(selected_num, selected_row['V2_MAE_Acc'].values[0], marker='*', color='red', ms=18, zorder=5, label="Selected Budget")

    # Formatting Left Y-Axis (Percentages)
    ax1.set_ylim(0, 105)
    ax1.set_ylabel(f'Top {target_pct_label}% Worst L0 caught (%)\n({TARGET_CLINICS}/{n_L0s_per_L1} Target L0 in L1 Region)', fontsize=14, labelpad=15)
    
    # Formatting Secondary Right Y-Axis (Absolute Clinic Counts)
    ax2 = ax1.twinx()
    # Scale the right axis perfectly to the left axis (105% = TARGET_CLINICS * 1.05)
    ax2.set_ylim(0, TARGET_CLINICS * 1.05) 
    ax2.set_ylabel(f'Actual Number of Clinics Caught\n(Out of {TARGET_CLINICS})', fontsize=14, color='#333333', labelpad=15)
    
    # Formatting X-Axis
    ax1.set_xlabel(f'Percentage of Children Sampled per L0: Total {n_children_per_L0} Children per L0', fontsize=14, fontweight='bold', labelpad=15)
    x_ticks = sorted(df['L1_Pct_Num'].unique())
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels([f"{x}%" for x in x_ticks], fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Spines for both axes
    for ax in [ax1, ax2]:
        for spine in ax.spines.values():
            spine.set_linewidth(2.0)
            spine.set_color('black')

    # Legend & Title (Matching original aesthetics)
    ax1.legend(title='Simulated Universe', bbox_to_anchor=(0.5, 1.15), loc='upper center', ncol=5, fontsize=11, framealpha=0.9, shadow=True)
    plt.title(f"L1 Ranking of L0 in each L1 Region: {indicator} Using [MAE]", fontsize=16, fontweight='bold', pad=75)
    
    plt.subplots_adjust(top=0.82)
    return fig

# ==============================================================================
# PLOT 3: BREADTH VS DEPTH OPTIMIZATION
# ==============================================================================
# ==============================================================================
# PLOT 3: BREADTH VS DEPTH OPTIMIZATION
# ==============================================================================
def plot_3_breadth_depth(df, l1_pct_str, indicator, n_L0s_per_L1, target_percentile):
    df = df.copy()
    
    # Sort by Clinics (Breadth) descending
    df = df.sort_values(by=['L1_C'], ascending=False).reset_index(drop=True)
    if df.empty: return None

    # =========================================================
    # VISUAL FILTER: Restrict to max 10 points (evenly spaced)
    # =========================================================
    if len(df) > 10:
        idx = np.linspace(0, len(df) - 1, 10).astype(int)
        df = df.iloc[idx].reset_index(drop=True)

    # Calculate dynamic targets based on user inputs
    TARGET_CLINICS = round(n_L0s_per_L1 * target_percentile)
    approx_kids = int(df['L1_C'].iloc[0] * df['L1_K'].iloc[0])

    x_indices = np.arange(len(df)) 
    x_breadth_labels = df['L1_C'].astype(int).values
    x_depth_labels = df['L1_K'].astype(int).values

    # Check for standard deviation/variance columns, fallback to 0 if missing
    v1_std = df.get('V1_MAE_Std', df.get('V1_MAE_Acc_Std', np.zeros(len(df))))

    fig, ax1 = plt.subplots(figsize=(14, 8), dpi=100)

    # Plot with error bars
    ax1.errorbar(x_indices, df['V1_MAE_Acc'], yerr=v1_std, 
                 fmt='o-', markersize=8, color='#2980b9', linewidth=2.5, 
                 capsize=5, capthick=1.5, label='Simulation Average', zorder=5)

    ax1.set_xlim(min(x_indices) - 0.5, max(x_indices) + 0.5)
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(x_breadth_labels, fontsize=12) 
    ax1.set_xlabel(f'BREADTH: No. of L0 Visited by L1 (Out of {n_L0s_per_L1})', fontsize=14, fontweight='bold', labelpad=15)
    ax1.set_ylabel(f'Top {TARGET_CLINICS} Worst L0 Clinics Caught (%)', fontsize=14, fontweight='bold')

    # Add the secondary X-axis at the bottom for Depth
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

    plt.title(f"L1 Sampling Strategy in Ranking L0 Clinics: [{indicator}]\n(Fixed Budget: {l1_pct_str})", 
              fontsize=16, pad=75, fontweight='bold')

    plt.subplots_adjust(bottom=0.25, top=0.82) 
    return fig


# ==============================================================================
# PLOT 6: L1 VS L2 HEATMAP
# ==============================================================================
# ==============================================================================
# PLOT 6: L1 VS L2 HEATMAP
# ==============================================================================
def plot_6_heatmap(df, l1_pct_str, l2_pct_str, indicator):
    df = df.copy()
    if df.empty: return None

    # Sort L1 strategies by the number of clinics (e.g., 20C comes before 10C)
    l1_order = sorted(df['L1_Label'].unique(), key=lambda x: int(x.split('C')[0]), reverse=True)

    # =========================================================
    # VISUAL FILTER: Restrict L1 Rows to max 10
    # =========================================================
    if len(l1_order) > 10:
        idx = np.linspace(0, len(l1_order) - 1, 10).astype(int)
        l1_order = [l1_order[i] for i in idx]

    # Increased height scaling and hspace to allow room for X-axis labels on every row
    fig = plt.figure(figsize=(14, max(6, len(l1_order) * 2.0)), dpi=100)
    gs = gridspec.GridSpec(nrows=len(l1_order), ncols=2, width_ratios=[1, 5], wspace=0.1, hspace=0.8)
    sns.set_theme(style="white")

    for i, l1_lbl in enumerate(l1_order):
        ax_l1 = fig.add_subplot(gs[i, 0]) 
        ax_l2 = fig.add_subplot(gs[i, 1]) 

        # Get data for this specific L1 strategy
        subset = df[df['L1_Label'] == l1_lbl].sort_values(by='L2_K').reset_index(drop=True)

        # =========================================================
        # VISUAL FILTER: Restrict L2 Columns to max 10
        # =========================================================
        if len(subset) > 10:
            col_idx = np.linspace(0, len(subset) - 1, 10).astype(int)
            subset = subset.iloc[col_idx].reset_index(drop=True)

        # ----------------------------------------------------
        # Blue Heatmap (L1 Accuracy)
        # ----------------------------------------------------
        l1_acc_val = subset['V1_MAE_Acc'].iloc[0] if not subset.empty else 0
        sns.heatmap(np.array([[l1_acc_val]]), annot=True, fmt=".1f", cmap="Blues", 
                    cbar=False, linewidths=2, linecolor='white', vmin=0, vmax=100, 
                    ax=ax_l1, annot_kws={"size": 14, "weight": "bold"})
        ax_l1.set_xticks([])
        ax_l1.set_yticks([0.5])
        ax_l1.set_yticklabels([l1_lbl], rotation=0, fontsize=12, fontweight='bold')
        if i == 0: ax_l1.set_title("L1 Baseline Accuracy (%)", fontsize=12, fontweight='bold', pad=10)

        # ----------------------------------------------------
        # Green/Red Heatmap (L2 Accuracy)
        # ----------------------------------------------------
        if not subset.empty:
            heatmap_data_l2 = subset[['V3_MAE_Acc']].T 
            l2_labels = subset['L2_Label'].tolist()

            sns.heatmap(heatmap_data_l2, annot=True, fmt=".1f", cmap="RdYlGn", 
                        cbar=False, linewidths=2, linecolor='white', vmin=0, vmax=100, 
                        ax=ax_l2, annot_kws={"size": 13, "weight": "bold"})
            
            ax_l2.set_yticks([])
            ax_l2.set_xticks(np.arange(len(l2_labels)) + 0.5)
            
            # FIX: Show labels on EVERY row, because L2 strategies change based on the L1 footprint above them!
            ax_l2.set_xticklabels(l2_labels, rotation=0, fontsize=11, fontweight='bold')
            ax_l2.set_xlabel(r"Increasing L2 Audit Depth $\longrightarrow$", fontsize=10, fontweight='bold', color='grey', labelpad=5)
                
            if i == 0: ax_l2.set_title("L2 Auditor Execution Accuracy (%)", fontsize=12, fontweight='bold', pad=10)

        for ax in [ax_l1, ax_l2]:
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(2)
                spine.set_color('black')

    fig.suptitle(f"L1 vs L2 Synergy ({indicator})\nBudgets: L1={l1_pct_str}, L2={l2_pct_str}", fontsize=18, fontweight='bold', y=1.05)
    return fig