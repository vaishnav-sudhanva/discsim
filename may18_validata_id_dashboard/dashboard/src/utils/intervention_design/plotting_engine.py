import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# -------------------------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------------------------
def get_target_clinics(target_pct_str, total_clinics):
    """Dynamically calculates physical clinic count based on the percentile string."""
    pct_val = int(target_pct_str.replace('%', ''))
    return int(np.round(total_clinics * (pct_val / 100)))

def get_dynamic_colors(universes):
    """Dynamically assigns colors to ANY universe name passed from the UI."""
    # Standard Tableau 10 Color Palette
    base_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                   '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    return {u: base_colors[i % len(base_colors)] for i, u in enumerate(universes)}

# -------------------------------------------------------------------------
# PLOTTING FUNCTIONS
# -------------------------------------------------------------------------
def plot_1_sensitivity(df, metric_col, metric_label, selected_unis, target_pct_str, total_kids=15):
    filtered_df = df[df['Universe'].isin(selected_unis)]
    if filtered_df.empty: return None

    agg_df = filtered_df.groupby(['Universe', 'L1_Pct_Num']).agg(
        mean_acc=(metric_col, 'mean'), std_acc=(metric_col, 'std'), count=(metric_col, 'count')
    ).reset_index()
    agg_df['std_acc'] = agg_df['std_acc'].fillna(0)
    agg_df['ci95'] = 1.96 * (agg_df['std_acc'] / np.sqrt(agg_df['count']))

    uni_colors = get_dynamic_colors(selected_unis)
    fig, ax = plt.subplots(figsize=(12, 7), dpi=100)

    for uni in [u for u in selected_unis if u in agg_df['Universe'].unique()]:
        data = agg_df[agg_df['Universe'] == uni].sort_values('L1_Pct_Num')
        ax.plot(data['L1_Pct_Num'], data['mean_acc'], marker='o', linestyle='-', 
                color=uni_colors[uni], label=uni, lw=2.5, ms=8)
        ax.fill_between(data['L1_Pct_Num'], data['mean_acc'] - data['ci95'], 
                        data['mean_acc'] + data['ci95'], color=uni_colors[uni], alpha=0.15)

    # ax.set_xlabel('Percentage of Children Sampled per L0: Total 15 Children per L0', fontsize=14, fontweight='bold', labelpad=15)
    # ax.set_ylabel(f'Top {target_pct_str} Worst L1 Regions Caught (%)', fontsize=14, labelpad=15)
# 🟢 Inject total_kids into the xlabel
    ax.set_xlabel(f'Percentage of Children Sampled per L0: Total {total_kids} Children per L0', fontsize=14, fontweight='bold', labelpad=15)
    ax.set_ylabel(f'Top {target_pct_str} Worst L1 Regions Caught (%)', fontsize=14, labelpad=15)
    x_ticks = sorted(agg_df['L1_Pct_Num'].unique())
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{x}%" for x in x_ticks], fontsize=12)
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle='--', alpha=0.5)

    for spine in ax.spines.values():
        spine.set_linewidth(2.0)
        spine.set_color('black')

    ax.legend(title='Simulated Universe', bbox_to_anchor=(0.5, 1.15), loc='upper center', 
              ncol=min(5, len(selected_unis)), fontsize=11, framealpha=0.9, shadow=True)
    plt.title(f"L1 Ranking of L1 Regions: Height Using [{metric_label}]", 
              fontsize=16, fontweight='bold', pad=75)
    plt.subplots_adjust(top=0.82)
    return fig

# 🟢 Add total_clinics and total_kids arguments
def plot_2_intra_regional(df, metric_col, metric_label, selected_unis, target_pct_str, total_clinics=25, total_kids=15):
    # ... (keep existing code) ...
    filtered_df = df[df['Universe'].isin(selected_unis)]
    if filtered_df.empty: return None

    agg_df = filtered_df.groupby(['Universe', 'L1_Pct_Num']).agg(
        mean_acc=(metric_col, 'mean'), std_acc=(metric_col, 'std'), count=(metric_col, 'count')
    ).reset_index()
    agg_df['std_acc'] = agg_df['std_acc'].fillna(0)
    agg_df['ci95'] = 1.96 * (agg_df['std_acc'] / np.sqrt(agg_df['count']))

    # 🟢 Pass the dynamic total_clinics to the helper function
    target_clinics = get_target_clinics(target_pct_str, total_clinics)
    
    agg_df['mean_acc_count'] = agg_df['mean_acc'] * (target_clinics / 100)
    agg_df['ci_95_count'] = agg_df['ci95'] * (target_clinics / 100)

    uni_colors = get_dynamic_colors(selected_unis)
    fig, ax = plt.subplots(figsize=(12, 7), dpi=100)

    for uni in [u for u in selected_unis if u in agg_df['Universe'].unique()]:
        data = agg_df[agg_df['Universe'] == uni].sort_values('L1_Pct_Num')
        ax.plot(data['L1_Pct_Num'], data['mean_acc_count'], marker='o', linestyle='-', 
                color=uni_colors[uni], label=uni, lw=2.5, ms=8)
        ax.fill_between(data['L1_Pct_Num'], data['mean_acc_count'] - data['ci_95_count'], 
                        data['mean_acc_count'] + data['ci_95_count'], color=uni_colors[uni], alpha=0.15)

    # 🟢 Inject the dynamic variables into the exact labels you requested
    ax.set_xlabel(f'Percentage of Children Sampled per L0: Total {total_kids} Children per L0', fontsize=14, fontweight='bold', labelpad=15)
    ax.set_ylabel(f'Top {target_pct_str} Worst L0 Caught \n({target_clinics}/{total_clinics} Target L0 in L1 Region)', fontsize=14, labelpad=15)
    
    # 🟢 X-Axis: Show both % and exact number in brackets
    x_ticks = sorted(agg_df['L1_Pct_Num'].unique())
    ax.set_xticks(x_ticks)
    # ax.set_xticklabels([f"{int(x)}%\n({round((x/100)*total_kids)} Kids)" for x in x_ticks], fontsize=12)
    ax.set_xticklabels([f"{int(x)}%\n({int(round((x/100)*total_kids))} Kids)" for x in x_ticks], fontsize=12)
    # 🟢 Y-Axis: Show both % and exact number in brackets (calculated based on target_clinics limit)
    ax.set_ylim(0, target_clinics + 0.5)
    
    # Creates 5 perfectly spaced ticks from 0 to the target_clinics limit (e.g., 0, 2, 4, 6, 8)
    y_ticks = np.linspace(0, target_clinics, num=5) 
    ax.set_yticks(y_ticks)
    # ax.set_yticklabels([f"{int((y/target_clinics)*100)}%\n({int(y)} L0)" for y in y_ticks], fontsize=12)
    ax.set_yticklabels([f"{int(round((y/target_clinics)*100))}%\n({int(y)} L0)" for y in y_ticks], fontsize=12)

    ax.grid(True, linestyle='--', alpha=0.5)

    for spine in ax.spines.values():
        spine.set_linewidth(2.0)
        spine.set_color('black')

    # ax.legend(title='Simulated Universe', bbox_to_anchor=(0.5, 1.15), loc='upper center', 
    #           ncol=min(5, len(selected_unis)), fontsize=11, framealpha=0.9, shadow=True)
    
    # 🟢 Original Title name kept, but gap fixed with pad=15 and tight_layout
    plt.title(f"L1 Ranking of L0 in each L1 Region: Height", 
              fontsize=16, fontweight='bold', pad=15)
    
    plt.tight_layout()
    return fig


def plot_3_bd_optimization(df, metric_col, metric_label, selected_unis, l1_budget_str, target_pct_str, total_clinics=25):
    # ... (keep existing code until the axis labels) ...
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

    uni_colors = get_dynamic_colors(selected_unis)
    fig, ax1 = plt.subplots(figsize=(14, 8), dpi=100)

    for uni in [u for u in selected_unis if u in agg_df['Universe'].unique()]:
        data = agg_df[agg_df['Universe'] == uni]
        ax1.errorbar(x_indices, data['mean_acc'], yerr=data['ci95'], 
                     fmt='o-', markersize=8, color=uni_colors[uni], linewidth=2.5, 
                     capsize=5, capthick=1.5, label=uni, zorder=5)

    # ax1.set_xlim(min(x_indices) - 0.5, max(x_indices) + 0.5)
    # ax1.set_xticks(x_indices)
    # ax1.set_xticklabels(x_breadth_labels, fontsize=12) 
    ax1.set_xlim(min(x_indices) - 0.5, max(x_indices) + 0.5)
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(x_breadth_labels, fontsize=12)
    # 🟢 Inject total_clinics into the xlabel
    ax1.set_xlabel(f'BREADTH: No. of L0 Visited by L1 (Out of {total_clinics})', fontsize=14, fontweight='bold', labelpad=15)
    ax1.set_ylabel(f'Top {target_pct_str} Worst L1 Regions Caught (%)', fontsize=14, fontweight='bold')
    # ax1.set_xlabel('BREADTH: No. of L0 Visited by L1 (Out of 25)', fontsize=14, fontweight='bold', labelpad=15)
    # ax1.set_ylabel(f'Top {target_pct_str} Worst L1 Regions Caught (%)', fontsize=14, fontweight='bold')

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
               ncol=min(5, len(selected_unis)), fontsize=11, framealpha=0.9, shadow=True)

    plt.title(f"L1 Sampling Strategy in Ranking L1 Regions: [{metric_label}]\n(Fixed Budget: {l1_budget_str})", 
              fontsize=16, pad=75, fontweight='bold')

    plt.subplots_adjust(bottom=0.25, top=0.82) 
    return fig

def plot_4_robustness(df, metric_col, metric_label, selected_uni, target_pct_str):
    df_uni = df[df['Universe'] == selected_uni].copy()
    if df_uni.empty: return None

    agg_df = df_uni.groupby(['L1_Pct_Num', 'L2_Pct_Num']).agg(
        mean_acc=(metric_col, 'mean'), std_acc=(metric_col, 'std'), count=(metric_col, 'count')
    ).reset_index()
    agg_df['std_acc'] = agg_df['std_acc'].fillna(0)
    agg_df['ci95'] = 1.96 * (agg_df['std_acc'] / np.sqrt(agg_df['count']))

    fig, ax = plt.subplots(figsize=(12, 7), dpi=100)
    
    # Static colors for budget tiers (not universes)
    colors = {10: '#555555', 20: '#e74c3c', 30: '#f39c12', 40: '#2ecc71', 50: '#3498db', 
              60: '#9b59b6', 70: '#e67e22', 80: '#1abc9c', 90: '#34495e', 100: '#16a085'}

    for l1_pct in sorted(agg_df['L1_Pct_Num'].unique()):
        subset = agg_df[agg_df['L1_Pct_Num'] == l1_pct].sort_values('L2_Pct_Num')
        ax.errorbar(subset['L2_Pct_Num'], subset['mean_acc'], yerr=subset['ci95'], 
                    fmt='o-', color=colors.get(l1_pct, '#333'), linewidth=3, markersize=8, 
                    capsize=5, label=f"L1 Base Budget: {l1_pct}%")

    ax.set_title(f'{selected_uni.upper()}: L2 Audit Accuracy vs L1 Spreadsheet [{metric_label}]', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("L2 Sample Size (% of Children Re-measured from L1's Spreadsheet)", 
                  fontsize=14, fontweight='bold', labelpad=15)
    ax.set_ylabel(f'Accuracy\n(Top {target_pct_str} Corrupt Supervisors Caught)', 
                  fontsize=14, fontweight='bold', labelpad=15)

    x_ticks = sorted(agg_df['L2_Pct_Num'].unique())
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{x}%" for x in x_ticks], fontsize=12)
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    ax.legend(title='Supervisor (L1) Base Policy', loc='lower right', fontsize=10, ncol=2, framealpha=0.9, shadow=True)

    for spine in ax.spines.values():
        spine.set_linewidth(2.0)
        spine.set_color('black')
    plt.tight_layout()
    return fig

def plot_5_master_grid(df, metric_col, metric_label, l1_budget_str, target_pct_str, selected_unis):
    df_filtered = df[(df['L1_Budget_Pct'] == l1_budget_str) & (df['Universe'].isin(selected_unis))].copy()
    if df_filtered.empty: return None

    agg_df = df_filtered.groupby(['Universe', 'L2_Budget_Pct', 'L2_C']).agg(
        mean_acc=(metric_col, 'mean'), std_acc=(metric_col, 'std'), count=(metric_col, 'count')
    ).reset_index()
    agg_df['std_acc'] = agg_df['std_acc'].fillna(0)
    agg_df['ci95'] = 1.96 * (agg_df['std_acc'] / np.sqrt(agg_df['count']))

    target_l2_budgets = ['20%', '40%', '60%', '80%', '100%']
    budget_styles = {
        '20%': {'color': '#e74c3c', 'label': 'L2 Budget: 20%'},
        '40%': {'color': '#f39c12', 'label': 'L2 Budget: 40%'},
        '60%': {'color': '#2ecc71', 'label': 'L2 Budget: 60%'},
        '80%': {'color': '#3498db', 'label': 'L2 Budget: 80%'},
        '100%': {'color': '#9b59b6', 'label': 'L2 Budget: 100%'}
    }

    # Only create enough subplots for the selected universes
    num_unis = len(selected_unis)
    rows = max(1, (num_unis + 1) // 2) 
    fig, axes = plt.subplots(rows, 2, figsize=(16, 5 * rows), dpi=100)
    
    # Flatten axes for easy iteration, handle single row edge case
    if num_unis == 1:
        axes = [axes[0], axes[1]] if hasattr(axes, '__len__') else [axes]
    else:
        axes = axes.flatten()

    for i, uni in enumerate(selected_unis):
        if uni not in agg_df['Universe'].unique(): continue
        
        ax = axes[i]
        uni_data = agg_df[agg_df['Universe'] == uni]

        for budget_name in target_l2_budgets:
            if budget_name in uni_data['L2_Budget_Pct'].unique():
                style = budget_styles[budget_name]
                subset = uni_data[uni_data['L2_Budget_Pct'] == budget_name].sort_values(by='L2_C')
                ax.errorbar(subset['L2_C'], subset['mean_acc'], yerr=subset['ci95'], 
                            fmt='o-', color=style['color'], linewidth=2.5, markersize=7, 
                            capsize=4, capthick=1.5, label=style['label'])

        ax.set_title(f"{uni.upper()}", fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel('L2 BREADTH: Clinics Audited', fontsize=11, fontweight='bold')
        ax.set_ylabel(f'Top {target_pct_str} Worst Caught', fontsize=11, fontweight='bold')
        ax.set_xlim(left=0)
        ax.set_ylim(0, 105)
        ax.grid(True, linestyle='--', alpha=0.5)

        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color('black')

    # Hide any unused axes
    for j in range(num_unis, len(axes)):
        axes[j].axis('off')

    # Find the last axis to place the legend
    ax_legend = axes[min(num_unis, len(axes)-1)] if num_unis < len(axes) else axes[-1]
    if num_unis < len(axes):
        ax_legend.axis('off') 
        
    legend_elements = [Line2D([0], [0], marker='o', color=style['color'], label=style['label'], 
                              markersize=10, linewidth=3) for budget_name, style in budget_styles.items()]
    fig.legend(handles=legend_elements, loc='lower center', fontsize=14, frameon=True, shadow=True, 
                     title=f"L2 Budgets (Base L1 = {l1_budget_str})", title_fontsize=16, ncol=5, bbox_to_anchor=(0.5, -0.05))

    plt.suptitle(f"L2 Breadth Optimization [{metric_label}]", fontsize=20, fontweight='bold', y=1.05)
    plt.tight_layout() 
    return fig





# 🟢 Cleaned up signature: Removed metric_col_l1 and l1_pct_str entirely
def plot_6_heatmap(df, metric_col_l2, metric_label, selected_uni, l2_pct_str):
    
    df_hm = df[(df['Universe'] == selected_uni) & (df['L2_Budget_Pct'] == l2_pct_str)].copy()
    if df_hm.empty: return None

    agg_df = df_hm.groupby(['L1_Budget_Pct', 'L1_Label', 'L2_Label', 'L2_K']).agg(
        L2_Acc=(metric_col_l2, 'mean')
    ).reset_index()

    # Sort L1 Budgets logically (10% to 100%)
    agg_df['L1_Budget_Int'] = agg_df['L1_Budget_Pct'].astype(str).str.replace('%', '').astype(float).astype(int)
    
    # Map the labels to their percentages so we can print both on the Y-Axis
    sorted_l1 = agg_df[['L1_Budget_Int', 'L1_Budget_Pct', 'L1_Label']].drop_duplicates().sort_values('L1_Budget_Int')
    l1_order = sorted_l1['L1_Label'].tolist()
    l1_pct_map = dict(zip(sorted_l1['L1_Label'], sorted_l1['L1_Budget_Pct']))

    fig = plt.figure(figsize=(16, 12), dpi=100)
    gs = gridspec.GridSpec(nrows=len(l1_order), ncols=2, width_ratios=[15, 0.5], wspace=0.05, hspace=0.8)

    sns.set_theme(style="white")
    cbar_ax = fig.add_subplot(gs[:, 1])

    for i, l1_lbl in enumerate(l1_order):
        ax_l2 = fig.add_subplot(gs[i, 0]) 
        subset = agg_df[agg_df['L1_Label'] == l1_lbl].sort_values(by='L2_K')

        heatmap_data_l2 = subset[['L2_Acc']].T 
        l2_labels = subset['L2_Label'].tolist()

        sns.heatmap(heatmap_data_l2, annot=True, fmt=".1f", cmap="RdYlGn", 
                    cbar=(i == 0), cbar_ax=cbar_ax if i == 0 else None, 
                    linewidths=2, linecolor='white', vmin=0, vmax=100, 
                    ax=ax_l2, annot_kws={"size": 13, "weight": "bold"})

        # 🟢 Format the Y-Axis to show exactly what you asked for!
        pct_str = l1_pct_map[l1_lbl]
        display_label = f"{l1_lbl}\n({pct_str} L1 Budget)"
        
        ax_l2.set_yticks([0.5])
        ax_l2.set_yticklabels([display_label], rotation=0, fontsize=12, fontweight='bold')
        # ax_l2.set_ylabel("L1 SS", fontsize=12, fontweight='bold', rotation=0, labelpad=35, va='center')
        
        ax_l2.set_xticks(np.arange(len(l2_labels)) + 0.5)
        ax_l2.set_xticklabels(l2_labels, rotation=0, fontsize=12, fontweight='bold')
        
        if i == len(l1_order) - 1:
            ax_l2.set_xlabel(r"Increasing L2 Depth $\longrightarrow$", fontsize=11, fontweight='bold', color='grey')
        else:
            ax_l2.set_xlabel("")
            
        if i == 0: 
            ax_l2.set_title("L2 Sampling Strategy", fontsize=12, fontweight='bold', pad=10)

        for spine in ax_l2.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2)
            spine.set_color('black')

    for spine in cbar_ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)
        spine.set_color('black')

    fig.suptitle(f"L2 Ranking Accuracy\n(Selected L2 Budget: {l2_pct_str})", 
                 fontsize=18, fontweight='bold', y=1.02)
    return fig

# # 🟢 Maintained original signature so your UI code won't crash!
# def plot_6_heatmap(df, metric_col_l1, metric_col_l2, metric_label, selected_uni, l1_pct_str, l2_pct_str):
    
#     # 🟢 Dropped the l1_pct_str filter so we get ALL 10 L1 budgets as rows
#     df_hm = df[(df['Universe'] == selected_uni) & (df['L2_Budget_Pct'] == l2_pct_str)].copy()
#     if df_hm.empty: return None

#     agg_df = df_hm.groupby(['L1_Budget_Pct', 'L1_Label', 'L2_Label', 'L2_K']).agg(
#         L1_Acc=(metric_col_l1, 'mean'), L2_Acc=(metric_col_l2, 'mean')
#     ).reset_index()

#     # Sort L1 Budgets so they display correctly from 10% down to 100%
#     agg_df['L1_Budget_Int'] = agg_df['L1_Budget_Pct'].str.replace('%', '').astype(int)
#     l1_order = agg_df[['L1_Budget_Int', 'L1_Label']].drop_duplicates().sort_values('L1_Budget_Int')['L1_Label'].tolist()

#     fig = plt.figure(figsize=(16, 12), dpi=100)
    
#     # 🟢 Updated GridSpec: 1 Main Column for heatmaps, 1 Skinny Column for the Colorbar
#     gs = gridspec.GridSpec(nrows=len(l1_order), ncols=2, width_ratios=[15, 0.5], wspace=0.05, hspace=0.8)

#     sns.set_theme(style="white")
    
#     # 🟢 Create a single colorbar axis on the right side that spans all the rows
#     cbar_ax = fig.add_subplot(gs[:, 1])

#     for i, l1_lbl in enumerate(l1_order):
#         # 🟢 ax_l1 is completely gone. Only ax_l2 remains!
#         ax_l2 = fig.add_subplot(gs[i, 0]) 

#         subset = agg_df[agg_df['L1_Label'] == l1_lbl].sort_values(by='L2_K')

#         heatmap_data_l2 = subset[['L2_Acc']].T 
#         l2_labels = subset['L2_Label'].tolist()

#         # 🟢 Plot L2 Heatmap (Locked 0 to 100), linked to the colorbar on the right
#         sns.heatmap(heatmap_data_l2, annot=True, fmt=".1f", cmap="RdYlGn", 
#                     cbar=(i == 0), cbar_ax=cbar_ax if i == 0 else None, 
#                     linewidths=2, linecolor='white', vmin=0, vmax=100, 
#                     ax=ax_l2, annot_kws={"size": 13, "weight": "bold"})

#         # 🟢 Correctly label the Y-Axis with the L1 SS (e.g., "25C x 2K")
#         ax_l2.set_yticks([0.5])
#         ax_l2.set_yticklabels([l1_lbl], rotation=0, fontsize=12, fontweight='bold')
#         ax_l2.set_ylabel("L1 SS", fontsize=12, fontweight='bold', rotation=0, labelpad=25, va='center')
        
#         ax_l2.set_xticks(np.arange(len(l2_labels)) + 0.5)
#         ax_l2.set_xticklabels(l2_labels, rotation=0, fontsize=12, fontweight='bold')
        
#         # Only put the X-axis label on the very bottom row
#         if i == len(l1_order) - 1:
#             ax_l2.set_xlabel(r"Increasing L2 Depth $\longrightarrow$", fontsize=11, fontweight='bold', color='grey')
#         else:
#             ax_l2.set_xlabel("")
            
#         if i == 0: 
#             ax_l2.set_title("L2 Auditor Execution Options", fontsize=12, fontweight='bold', pad=10)

#         for spine in ax_l2.spines.values():
#             spine.set_visible(True)
#             spine.set_linewidth(2)
#             spine.set_color('black')

#     # Make the colorbar outline clean
#     for spine in cbar_ax.spines.values():
#         spine.set_visible(True)
#         spine.set_linewidth(1)
#         spine.set_color('black')

#     # 🟢 Simplified Title
#     fig.suptitle(f"L2 Ranking Accuracy\n(Selected L2 Budget: {l2_pct_str})", 
#                  fontsize=18, fontweight='bold', y=1.02)
#     return fig
# # def plot_6_heatmap(df, metric_col_l1, metric_col_l2, metric_label, selected_uni, l1_pct_str, l2_pct_str):
# #     vmin_global_l1, vmax_global_l1 = df[metric_col_l1].min(), df[metric_col_l1].max()
# #     vmin_global_l2, vmax_global_l2 = df[metric_col_l2].min(), df[metric_col_l2].max()

# #     df_hm = df[(df['Universe'] == selected_uni) & (df['L1_Budget_Pct'] == l1_pct_str) & (df['L2_Budget_Pct'] == l2_pct_str)].copy()
# #     if df_hm.empty: return None

# #     agg_df = df_hm.groupby(['L1_Label', 'L2_Label', 'L2_K']).agg(
# #         L1_Acc=(metric_col_l1, 'mean'), L2_Acc=(metric_col_l2, 'mean')
# #     ).reset_index()

# #     l1_order = sorted(agg_df['L1_Label'].unique(), key=lambda x: int(x.split('C')[0]), reverse=True)

# #     fig = plt.figure(figsize=(16, 12), dpi=100)
# #     gs = gridspec.GridSpec(nrows=len(l1_order), ncols=2, width_ratios=[1, 6], wspace=0.1, hspace=0.8)

# #     sns.set_theme(style="white")

# #     for i, l1_lbl in enumerate(l1_order):
# #         ax_l1 = fig.add_subplot(gs[i, 0]) 
# #         ax_l2 = fig.add_subplot(gs[i, 1]) 

# #         subset = agg_df[agg_df['L1_Label'] == l1_lbl].sort_values(by='L2_K')

# #         l1_acc_value = subset['L1_Acc'].iloc[0] if not subset.empty else 0
        
# #         sns.heatmap(np.array([[l1_acc_value]]), annot=True, fmt=".1f", cmap="Blues", 
# #                     cbar=False, linewidths=2, linecolor='white', vmin=0, vmax=100, 
# #                     ax=ax_l1, annot_kws={"size": 14, "weight": "bold"})

# #         ax_l1.set_xticks([])
#         ax_l1.set_yticks([0.5])
#         ax_l1.set_yticklabels([l1_lbl], rotation=0, fontsize=14, fontweight='bold')
#         if i == 0: ax_l1.set_title("L1 Baseline Accuracy", fontsize=12, fontweight='bold', pad=10)

#         heatmap_data_l2 = subset[['L2_Acc']].T 
#         l2_labels = subset['L2_Label'].tolist()

#         sns.heatmap(heatmap_data_l2, annot=True, fmt=".1f", cmap="RdYlGn", 
#                     cbar=False, linewidths=2, linecolor='white', vmin=vmin_global_l2, vmax=vmax_global_l2, 
#                     ax=ax_l2, annot_kws={"size": 13, "weight": "bold"})

#         ax_l2.set_yticks([])
#         ax_l2.set_xticks(np.arange(len(l2_labels)) + 0.5)
#         ax_l2.set_xticklabels(l2_labels, rotation=0, fontsize=12, fontweight='bold')
#         ax_l2.set_xlabel(r"Increasing L2 Depth $\longrightarrow$", fontsize=11, fontweight='bold', color='grey')
#         if i == 0: ax_l2.set_title("L2 Auditor Execution Options", fontsize=12, fontweight='bold', pad=10)

#         for ax in [ax_l1, ax_l2]:
#             for spine in ax.spines.values():
#                 spine.set_visible(True)
#                 spine.set_linewidth(2)
#                 spine.set_color('black')

#     fig.suptitle(f"L2 Ranking Accuracy\n(Budgets: L1={l1_pct_str}, L2={l2_pct_str})", 
#                  fontsize=18, fontweight='bold', y=1.02)
#     return fig