import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec

# 1. Plot 6: Heatmap
def plot_6_heatmap(df, metric_col_l1, metric_col_l2, metric_label, selected_uni, l1_pct_str, l2_pct_str):
    # [Paste the exact Plot 6 function we wrote earlier]
    pass

# 2. Plot 3: Breadth vs Depth
def plot_3_bd_optimization(df, metric_col, metric_label, selected_unis, l1_budget_str):
    # [Paste the exact Plot 3 function we wrote earlier]
    pass

# 3. Plot 2: Intra-Regional
def plot_2_intra_regional(df, metric_col, metric_label, selected_unis):
    # [Paste the exact Plot 2 function we wrote earlier]
    pass

# 4. The New Backward View Plot!
def plot_backward_view(parquet_df, target_percentile=0.30):
    indicator = 'haz'
    
    def calc_mae(df_temp, group_cols, col_meas, col_baseline):
        df_temp['_err'] = np.abs(df_temp[col_meas] - df_temp[col_baseline])
        return df_temp.groupby(group_cols)['_err'].mean().reset_index().rename(columns={'_err': 'MAE'})

    # Step 1: L2 finds good L1s
    l2_vs_l1 = calc_mae(parquet_df, ['L1_id'], f'L2_{indicator}', f'L1_{indicator}')
    n_good_l1s = max(1, int(len(l2_vs_l1) * target_percentile))
    good_l1_list = l2_vs_l1.sort_values(by='MAE', ascending=True).head(n_good_l1s)['L1_id'].tolist()

    # Step 2: Good L1s accuracy
    df_good_l1s = parquet_df[parquet_df['L1_id'].isin(good_l1_list)]
    god_l0 = calc_mae(df_good_l1s, ['L1_id', 'L0_id'], f'L0_{indicator}', f'real_{indicator}')
    measured_l0 = calc_mae(df_good_l1s, ['L1_id', 'L0_id'], f'L1_{indicator}', f'L0_{indicator}')

    accuracies = []
    for l1_id in good_l1_list:
        subset_god = god_l0[god_l0['L1_id'] == l1_id]
        subset_meas = measured_l0[measured_l0['L1_id'] == l1_id]
        n_worst_l0s = max(1, int(len(subset_god) * target_percentile))
        
        true_worst = set(subset_god.sort_values(by='MAE', ascending=False).head(n_worst_l0s)['L0_id'])
        meas_worst = set(subset_meas.sort_values(by='MAE', ascending=False).head(n_worst_l0s)['L0_id'])
        overlap = len(true_worst & meas_worst) / n_worst_l0s
        accuracies.append({'L1_id': l1_id, 'Accuracy': overlap * 100})

    acc_df = pd.DataFrame(accuracies)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    sns.barplot(data=acc_df, x='L1_id', y='Accuracy', palette='viridis', ax=ax)
    ax.axhline(y=acc_df['Accuracy'].mean(), color='red', linestyle='--', linewidth=2)
    ax.set_ylim(0, 105)
    ax.set_xlabel("The 'Good' L1 Supervisors identified by L2", fontweight='bold')
    ax.set_ylabel("Accuracy in catching Bad L0s (%)", fontweight='bold')
    ax.set_title("Backward View: How well did 'Good L1s' rank their L0s?", fontweight='bold', fontsize=14)
    
    return fig
