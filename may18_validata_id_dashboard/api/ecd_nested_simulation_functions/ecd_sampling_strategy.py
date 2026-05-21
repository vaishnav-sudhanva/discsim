import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .generate_ecd_dummy_data import generate_nested_distortion_parameters
from .generate_ecd_dummy_data import generate_nested_measurements
from .generate_ecd_dummy_data import get_L1_L2_pairwise_data




def calculate_discrepancy_scores(measurements1, measurements2, variable, method, 
                               make_plot=False, plot_title=None,
                               measurements1_name=None, measurements2_name=None,
                               discrepancy_unit=None):
    """Calculate discrepancy scores between two sets of measurements for a given variable.
    
    Args:
        measurements1 (dict): First set of measurements (e.g., L1)
        measurements2 (dict): Second set of measurements (e.g., L2 - The Baseline/Denominator)
        variable (str): Variable to compare ('height', 'weight', 'haz', 'waz' or 'whz')
        method (str): Method for calculating discrepancy:
            - 'percent_difference'
            - 'absolute_difference'
            - 'absolute_percent_difference'
            - 'simple_difference'
        make_plot (bool): Whether to create visualization plots
        plot_title (str): Title for the plots
        measurements1_name (str): Label for measurements1 on x-axis
        measurements2_name (str): Label for measurements2 on x-axis
        discrepancy_unit (str): Unit for discrepancy scores (e.g., 'kg' for weight)
    
    Returns:
        pd.Series or np.array: Array of discrepancy scores for each measurement
    """
    
    # Check if variable exists in both dictionaries
    if variable not in measurements1 or variable not in measurements2:
        raise KeyError(f"Variable '{variable}' must be present in both measurement dictionaries")
        
    # Extract values (Assume they are Pandas Series with aligned indices from previous steps)
    values1 = measurements1[variable]
    values2 = measurements2[variable]
    
    # Mathematical Safety: Warn user if trying to do percentage math on Z-scores
    if "percent" in method and variable in ['haz', 'waz', 'whz']:
        print(f"WARNING: Calculating percentage difference on Z-scores ({variable}) is mathematically unstable and not recommended. Using simple/absolute difference is advised.")

    # Calculate individual discrepancy scores
    if method == "percent_difference":
        # Safe division: Replace 0s with slightly above 0 to prevent Infinity errors, or handle natively
        safe_values2 = np.where(values2 == 0, 1e-9, values2)
        disc_scores = (values1 - values2) / safe_values2 * 100
        
    elif method == "absolute_difference":
        disc_scores = np.abs(values1 - values2)
        
    elif method == "absolute_percent_difference":
        safe_values2 = np.where(values2 == 0, 1e-9, values2)
        disc_scores = np.abs((values1 - values2) / safe_values2 * 100)
        
    elif method == "simple_difference":
        disc_scores = values1 - values2
        
    else:
        raise ValueError("Method must be one of: percent_difference, absolute_difference, absolute_percent_difference, simple_difference")
    
    if make_plot:
        # Set default labels if not provided
        if measurements1_name is None:
            measurements1_name = "Measurements 1"
        if measurements2_name is None:
            measurements2_name = "Measurements 2"
        if plot_title is None:
            plot_title = f"Discrepancy Analysis for {variable}"
            
        # Smart Y-Axis Labeling
        if "percent" in method:
            y_axis_label = "Discrepancy (%)"
            discrepancy_label = "Discrepancy (%)"
        else:
            unit_str = f" ({discrepancy_unit})" if discrepancy_unit else ""
            y_axis_label = f"Unit-wise discrepancy{unit_str}"
            discrepancy_label = f"Discrepancy{unit_str}"
            
        # Create figure with 1 row and 3 columns
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(plot_title, fontsize=14)
        
        # Histogram of discrepancy scores
        ax1.hist(disc_scores, bins=50, color='lightgray', edgecolor='black')
        ax1.set_xlabel(discrepancy_label, fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        
        # Scatter plot against measurements1
        ax2.scatter(values1, disc_scores, alpha=0.5, color='turquoise', s=20, linewidth=0)
        ax2.set_xlabel(f"{measurements1_name} {variable}", fontsize=12)
        ax2.set_ylabel(y_axis_label, fontsize=12)
        
        # Scatter plot against measurements2
        ax3.scatter(values2, disc_scores, alpha=0.5, color='salmon', s=20, linewidth=0)
        ax3.set_xlabel(f"{measurements2_name} {variable}", fontsize=12)
        ax3.set_ylabel(y_axis_label, fontsize=12)
        
        # Adjust layout
        plt.tight_layout()
        plt.show()
    
    return disc_scores









def calculate_ranks_L0_units(nested_measurements, measurement_var, method='simple_difference'):
    """
    Calculate real and measured ranks for units of L0s based on average discrepancy scores.
    """
    # Store average discrepancy scores for each L1 unit
    unit_real_discrepancies = {}
    unit_measured_discrepancies = {}
    
    # Calculate average discrepancy scores for each L1 unit
    for L1_id in nested_measurements:
        if L1_id == 'metadata':
            continue
            
        # Store discrepancy scores for all L0s in this L1 unit
        real_discrepancies = []
        measured_discrepancies = []
        
        # Calculate discrepancy scores for each L0 in this L1 unit
        for L0_id in nested_measurements[L1_id]:
            if L0_id == 'L1_info':
                continue
                
            # Get measurements
            real_meas = nested_measurements[L1_id][L0_id]['real']
            L0_meas = nested_measurements[L1_id][L0_id]['L0']
            L1_meas = nested_measurements[L1_id][L0_id]['L1']
            
            # Calculate real discrepancy (L0 vs real)
            real_disc = calculate_discrepancy_scores(
                L0_meas['data'],
                real_meas['data'],
                measurement_var,
                method,
                make_plot=False
            )
            
            # --- MATH FIX: Take Absolute Value FIRST, then Mean (Mean Absolute Error) ---
           # real_discrepancies.append(np.abs(real_disc).mean())
            real_discrepancies.append(np.nanmean(np.abs(real_disc)))
            # Calculate measured discrepancy (L0 vs L1)
            # Only use children measured by L1
            L1_indices = L1_meas['data'].index
            L0_subset = {
                'data': L0_meas['data'].loc[L1_indices].copy(),
                'metadata': L0_meas['metadata'].copy()
            }
            
            measured_disc = calculate_discrepancy_scores(
                L0_subset['data'],
                L1_meas['data'],
                measurement_var,
                method,
                make_plot=False
            )
            
            # --- MATH FIX: Take Absolute Value FIRST, then Mean ---
            #measured_discrepancies.append(np.abs(measured_disc).mean())
            measured_discrepancies.append(np.nanmean(np.abs(measured_disc)))
        # Calculate average discrepancy for this L1 unit
        unit_real_discrepancies[L1_id] = real_discrepancies
        unit_measured_discrepancies[L1_id] = measured_discrepancies

    # --- SORTING FIX: Extract the integer ID so L1_10 doesn't come before L1_2 ---
    L1_ids = sorted(unit_real_discrepancies.keys(), key=lambda x: int(x.split('_')[1]))
    
    mean_real_discrepancies = np.array([np.mean(unit_real_discrepancies[L1_id]) for L1_id in L1_ids])
    mean_measured_discrepancies = np.array([np.mean(unit_measured_discrepancies[L1_id]) for L1_id in L1_ids])

    # Calculate ranks (1-based ranking, ascending order of discrepancy)
    # The double argsort() is a genius way to get ranks, glad you used it!
    real_ranks = mean_real_discrepancies.argsort().argsort() + 1
    measured_ranks = mean_measured_discrepancies.argsort().argsort() + 1

    return real_ranks, measured_ranks, unit_real_discrepancies, unit_measured_discrepancies













def calculate_ranks_L0s(nested_measurements, measurement_var, method='simple_difference'):
    """
    Calculate real and measured ranks for L0s within each L1 unit based on discrepancy scores.
    """
    L0_ranks = {}
    
    # Calculate ranks for L0s within each L1 unit
    for L1_id in nested_measurements:
        if L1_id == 'metadata':
            continue
        
        real_discrepancies = []
        measured_discrepancies = []
        
        # --- SORTING FIX: Gather L0 keys and sort them numerically so 10 doesn't come before 2 ---
        raw_L0_keys = [k for k in nested_measurements[L1_id].keys() if k.startswith('L0_')]
        sorted_L0_ids = sorted(raw_L0_keys, key=lambda x: int(x.split('_')[1]))
        
        # Calculate discrepancy scores for each L0 in this L1 unit, using the sorted order
        for L0_id in sorted_L0_ids:
            
            # Get measurements
            real_meas = nested_measurements[L1_id][L0_id]['real']
            L0_meas = nested_measurements[L1_id][L0_id]['L0']
            L1_meas = nested_measurements[L1_id][L0_id]['L1']
            
            # Calculate real discrepancy (L0 vs real)
            real_disc = calculate_discrepancy_scores(
                L0_meas['data'],
                real_meas['data'],
                measurement_var,
                method,
                make_plot=False
            )
            
            # --- MATH FIX: Take Absolute Value FIRST, then Mean (Mean Absolute Error) ---
            #real_discrepancies.append(np.abs(real_disc).mean())
            real_discrepancies.append(np.nanmean(np.abs(real_disc)))
            # Calculate measured discrepancy (L0 vs L1)
            # Only use children measured by L1
            L1_indices = L1_meas['data'].index
            L0_subset = L0_meas['data'].loc[L1_indices].copy()
            
            measured_disc = calculate_discrepancy_scores(
                L0_subset,
                L1_meas['data'],
                measurement_var,
                method,
                make_plot=False
            )
            
            # --- MATH FIX: Take Absolute Value FIRST, then Mean ---
            measured_discrepancies.append(np.nanmean(np.abs(measured_disc)))
        
        # Convert to arrays
        real_discrepancies = np.array(real_discrepancies)
        measured_discrepancies = np.array(measured_discrepancies)
        
        # Calculate ranks (1-based ranking, ascending order of discrepancy)
        real_ranks = real_discrepancies.argsort().argsort() + 1
        measured_ranks = measured_discrepancies.argsort().argsort() + 1
        
        # Store ranks for this L1 unit
        L0_ranks[L1_id] = {
            'real_ranks': real_ranks,
            'measured_ranks': measured_ranks,
            'L0_ids': sorted_L0_ids  # Safely storing the numerically sorted list
        }
    
    return L0_ranks







import numpy as np
import matplotlib.pyplot as plt

def plot_nested_measurements_and_ranks(nested_measurements, measurement_var, measurement_unit, 
                                     real_discrepancies, measured_discrepancies,
                                     real_ranks, measured_ranks, figsize=(15, 6)):
    """
    Visualize nested measurements and ranks using scatter plots and violin plots.
    """
    # Get number of L1 units
    n_L1s = len(real_ranks)
    
    # Create figure
    fig, axs = plt.subplots(3, n_L1s, figsize=figsize, sharey='row', constrained_layout=True)
    
    # Font sizes
    TITLE_SIZE = 12
    LABEL_SIZE = 12
    TICK_SIZE = 10
    
    # Color scheme for L0s - safely find the maximum number of L0s
    max_L0s = max(len([k for k in nested_measurements[L1_id].keys() if k.startswith('L0_')]) 
                  for L1_id in nested_measurements if L1_id != 'metadata')
    cmap = plt.get_cmap('Dark2')
    L0_colors = cmap(np.linspace(0, 1, max_L0s))
    
    # Get order of L1 units based on real ranks
    L1_order = np.argsort(real_ranks)
    
    # Plot for each L1 unit
    for i, L1_idx in enumerate(L1_order):
        L1_id = f'L1_{L1_idx}'
        
        # Get measurement ranges for this unit to set axis limits
        all_measurements = []
        
        # --- FIX 1: Sort L0 keys numerically so colors never scramble! ---
        l0_keys = sorted([k for k in nested_measurements[L1_id].keys() if k.startswith('L0_')], 
                         key=lambda x: int(x.split('_')[1]))
        
        # Plot scatter plots for each L0
        for j, L0_id in enumerate(l0_keys):
            
            # Get measurements
            real_meas = nested_measurements[L1_id][L0_id]['real']['data'][measurement_var]
            L0_meas = nested_measurements[L1_id][L0_id]['L0']['data'][measurement_var]
            L1_meas = nested_measurements[L1_id][L0_id]['L1']['data'][measurement_var]
            
            # --- FIX 2: Safely strip Pandas index baggage to calculate clean min/max ---
            all_measurements.extend(real_meas.tolist())
            all_measurements.extend(L0_meas.tolist())
            all_measurements.extend(L1_meas.tolist())
            
            # First row: Real vs L0
            axs[0, i].scatter(real_meas, L0_meas, color=L0_colors[j], alpha=0.5, 
                            label=f'Worker {j}', s=20)
            
            # Second row: L0 vs L1 (only for children measured by L1)
            L1_indices = nested_measurements[L1_id][L0_id]['L1']['data'].index
            
            # --- FIX 3: Force explicit index alignment for both X and Y ---
            axs[1, i].scatter(L1_meas.loc[L1_indices], L0_meas.loc[L1_indices],
                            color=L0_colors[j], alpha=0.5, label=f'Worker {j}', s=20)
        
        # Set axis limits and add X=Y line for scatter plots
        axis_min = min(all_measurements)
        axis_max = max(all_measurements)
        line = np.linspace(axis_min, axis_max, 100)
        
        for row in [0, 1]:
            axs[row, i].plot(line, line, 'k--', alpha=0.5)
            axs[row, i].set_xlim(axis_min, axis_max)
            axs[row, i].set_ylim(axis_min, axis_max)
            if i == 0:
                axs[row, i].set_ylabel(f'L0 {measurement_var} ({measurement_unit})', fontsize=LABEL_SIZE)
            axs[row, i].tick_params(axis='both', labelsize=TICK_SIZE)
        
        # Set x-labels
        axs[0, i].set_xlabel(f'Real {measurement_var} ({measurement_unit})', fontsize=LABEL_SIZE)
        axs[1, i].set_xlabel(f'L1 {measurement_var} ({measurement_unit})', fontsize=LABEL_SIZE)
        





        # VS change 
        # --- FIX: Violin plot crash protection & line coloring ---
        clean_real = [x for x in real_discrepancies[L1_id] if not np.isnan(x)]
        clean_measured = [x for x in measured_discrepancies[L1_id] if not np.isnan(x)]
        
        if len(clean_real) > 0 and len(clean_measured) > 0:
            parts = axs[2, i].violinplot([clean_real, clean_measured],
                                       positions=[1, 2],
                                       showmeans=True)
            
            colors = ['#2c7bb6', '#d7191c']
            for pc, color in zip(parts['bodies'], colors):
                pc.set_facecolor(color)
                pc.set_alpha(0.7)
                
            for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans'):
                if partname in parts:
                    parts[partname].set_edgecolor('black')
        
        axs[2, i].set_xticks([1, 2])
        axs[2, i].set_xticklabels(['Real', 'Measured'], fontsize=TICK_SIZE)
        axs[2, i].tick_params(axis='y', labelsize=TICK_SIZE)
        if i == 0:
            axs[2, i].set_ylabel(f'{measurement_var.capitalize()} Disc. ({measurement_unit})', fontsize=LABEL_SIZE)
        
        axs[0, i].set_title(f'Real rank: {real_ranks[L1_idx]}\nMeasured rank: {measured_ranks[L1_idx]}', 
                           fontsize=TITLE_SIZE)
        
        if i == n_L1s - 1:
            axs[0, i].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=TICK_SIZE)

    return fig
    #     # Third row: Violin plots of discrepancies
    #     parts = axs[2, i].violinplot([real_discrepancies[L1_id], measured_discrepancies[L1_id]],
    #                                positions=[1, 2],
    #                                showmeans=True)
        
    #     # Color the violin plots
        
    #     colors = {'real': '#2c7bb6', 'measured': '#d7191c'}
    #     for pc, color in zip(parts['bodies'], colors.values()):
    #         pc.set_facecolor(color)
    #         pc.set_alpha(0.7)
    #     parts['cmeans'].set_color('black')
        
    #     # Set labels for violin plots
    #     axs[2, i].set_xticks([1, 2])
    #     axs[2, i].set_xticklabels(['Real', 'Measured'], fontsize=TICK_SIZE)
    #     axs[2, i].tick_params(axis='y', labelsize=TICK_SIZE)
    #     if i == 0:
    #         axs[2, i].set_ylabel(f'{measurement_var.capitalize()} Disc. ({measurement_unit})', fontsize=LABEL_SIZE)
        
    #     # Add title showing ranks
    #     axs[0, i].set_title(f'Real rank: {real_ranks[L1_idx]}\nMeasured rank: {measured_ranks[L1_idx]}', 
    #                        fontsize=TITLE_SIZE)
        
    #     # --- NEW FIX: Add the missing legend to the far right plot! ---
    #     if i == n_L1s - 1:
    #         axs[0, i].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=TICK_SIZE)

    # return fig






import numpy as np
import pandas as pd

def L0_unit_classification_confidence(
    # Real parameters
    real_params,
    n_L1s,
    n_L0s_per_L1,
    n_children_per_L0,
    n_children_L1,
    n_children_L2,
    # WHO parameters
    haz_params,
    waz_params,
    whz_params_lying,
    whz_params_standing,
    # Analysis parameters
    measurement_var,
    n_L1_units_rewarded,
    # Distortion parameters for L0s
    real_percent_stunting=40,
    real_percent_underweight=40,
    real_percent_wasting=None,
    mean_percent_under_reporting_stunting=20,
    mean_percent_under_reporting_underweight=20,
    mean_percent_under_reporting_wasting=None,
    mean_bunch_factor_haz=0.1,
    mean_bunch_factor_waz=0.1,
    mean_bunch_factor_whz=0.1,
    sd_across_units_percent_under_reporting_stunting=5,
    sd_across_units_percent_under_reporting_underweight=5,
    sd_across_units_percent_under_reporting_wasting=5,
    sd_across_units_bunch_factor_haz=0.02,
    sd_across_units_bunch_factor_waz=0.02,
    sd_across_units_bunch_factor_whz=0.02,
    sd_within_units_percent_under_reporting_stunting=2,
    sd_within_units_percent_under_reporting_underweight=2,
    sd_within_units_percent_under_reporting_wasting=2,
    sd_within_units_bunch_factor_haz=0.01,
    sd_within_units_bunch_factor_waz=0.01,
    sd_within_units_bunch_factor_whz=0.01,
    # Distortion parameters for L1s
    mean_percent_copy=10,
    mean_collusion_index=0.2,
    sd_percent_copy=2,
    sd_collusion_index=0.1,
    error_mean_height_L1 = 0,
    error_sd_height_L1 = 1,
    error_mean_weight_L1 = 0,
    error_sd_weight_L1 = 0.1,
    bunch_factor_haz_L1 = 0.05,
    bunch_factor_waz_L1 = 0.05,
    bunch_factor_whz_L1 = 0.05,
    # Distortion parameters for L2
    error_mean_height_L2=0,
    error_sd_height_L2=1,
    error_mean_weight_L2=0,
    error_sd_weight_L2=0.1,
    
    # --- NEW FIX: Removed old drift vars, added the proper Biological Time Lags ---
    mean_time_lag_L1=15,  
    sd_time_lag_L1=2,    
    mean_time_lag_L2=30,  
    # ------------------------------------------------------------------------------

    random_seed=None,
    n_simulations=100,  # Add n_simulations parameter
):
    """Analyze classification confidence for different parameter combinations."""
    
    # Convert inputs to lists if they're not already
    n_L0s_list = [n_L0s_per_L1] if isinstance(n_L0s_per_L1, int) else n_L0s_per_L1
    n_children_L0_list = [n_children_per_L0] if isinstance(n_children_per_L0, int) else n_children_per_L0
    n_children_L1_list = [n_children_L1] if isinstance(n_children_L1, int) else n_children_L1
    
    # Check if n_L1_units_rewarded is valid
    if n_L1_units_rewarded >= n_L1s:
        raise ValueError(f"n_L1_units_rewarded ({n_L1_units_rewarded}) must be less than n_L1s ({n_L1s})")
    
    results = []
    
    # Iterate over all parameter combinations
    for n_L0s in n_L0s_list:
        for n_children_L0 in n_children_L0_list:
            for n_children_L1 in n_children_L1_list:
                sim_real_ranks = []
                sim_overlaps = []
                warning_count = 0 
                
                # Run multiple simulations
                for sim in range(n_simulations):
                    
                    # Generate distortion parameters
                    L0_params_list, L1_params_list, L2_params_dict = generate_nested_distortion_parameters(
                        n_L1s=n_L1s,
                        n_L0s_per_L1=n_L0s,
                        real_percent_stunting=real_percent_stunting,
                        real_percent_underweight=real_percent_underweight,
                        real_percent_wasting=real_percent_wasting,
                        mean_percent_under_reporting_stunting=mean_percent_under_reporting_stunting,
                        mean_percent_under_reporting_underweight=mean_percent_under_reporting_underweight,
                        mean_percent_under_reporting_wasting=mean_percent_under_reporting_wasting,
                        mean_bunch_factor_haz=mean_bunch_factor_haz,
                        mean_bunch_factor_waz=mean_bunch_factor_waz,
                        mean_bunch_factor_whz=mean_bunch_factor_whz,
                        sd_across_units_percent_under_reporting_stunting=sd_across_units_percent_under_reporting_stunting,
                        sd_across_units_percent_under_reporting_underweight=sd_across_units_percent_under_reporting_underweight,
                        sd_across_units_percent_under_reporting_wasting=sd_across_units_percent_under_reporting_wasting,
                        sd_across_units_bunch_factor_haz=sd_across_units_bunch_factor_haz,
                        sd_across_units_bunch_factor_waz=sd_across_units_bunch_factor_waz,
                        sd_across_units_bunch_factor_whz=sd_across_units_bunch_factor_whz,
                        sd_within_units_percent_under_reporting_stunting=sd_within_units_percent_under_reporting_stunting,
                        sd_within_units_percent_under_reporting_underweight=sd_within_units_percent_under_reporting_underweight,
                        sd_within_units_percent_under_reporting_wasting=sd_within_units_percent_under_reporting_wasting,
                        sd_within_units_bunch_factor_haz=sd_within_units_bunch_factor_haz,
                        sd_within_units_bunch_factor_waz=sd_within_units_bunch_factor_waz,
                        sd_within_units_bunch_factor_whz=sd_within_units_bunch_factor_whz,
                        mean_percent_copy=mean_percent_copy,
                        mean_collusion_index=mean_collusion_index,
                        sd_percent_copy=sd_percent_copy,
                        sd_collusion_index=sd_collusion_index,
                        error_mean_height_L1=error_mean_height_L1,
                        error_sd_height_L1=error_sd_height_L1,
                        error_mean_weight_L1=error_mean_weight_L1,
                        error_sd_weight_L1=error_sd_weight_L1,
                        bunch_factor_haz_L1=bunch_factor_haz_L1,
                        bunch_factor_waz_L1=bunch_factor_waz_L1,
                        bunch_factor_whz_L1=bunch_factor_whz_L1,
                        error_mean_height_L2=error_mean_height_L2,
                        error_sd_height_L2=error_sd_height_L2,
                        error_mean_weight_L2=error_mean_weight_L2,
                        error_sd_weight_L2=error_sd_weight_L2,
                        
                        # --- NEW FIX: Replaced old drift vars with the time lags ---
                        mean_time_lag_L1=mean_time_lag_L1,
                        sd_time_lag_L1=sd_time_lag_L1,
                        mean_time_lag_L2=mean_time_lag_L2,
                        # -----------------------------------------------------------
                        
                        #random_seed=random_seed+sim if random_seed else None
                        random_seed=random_seed+sim if random_seed is not None else None # vs code change
                    )
                    
                    # Generate nested measurements
                    nested_measurements = generate_nested_measurements(
                        real_params=real_params,
                        L0_params_list=L0_params_list,
                        L1_params_list=L1_params_list,
                        L2_params_dict=L2_params_dict,
                        n_L1s=n_L1s,
                        n_L0s_per_L1=n_L0s,
                        n_children_per_L0=n_children_L0,
                        n_children_L1=n_children_L1,
                        n_children_L2=n_children_L2,
                        haz_params=haz_params,
                        waz_params=waz_params,
                        whz_params_lying=whz_params_lying,
                        whz_params_standing=whz_params_standing,
                        make_plots=False
                    )
                    
                    # Check for warnings in the measurements (Streamlined!)
                    warning_found = False
                    for L1_id, L1_data in nested_measurements.items():
                        if L1_id == 'metadata': continue
                        for L0_id, L0_data in L1_data.items():
                            if L0_id == 'L1_info': continue
                            
                            # Check both L0 and L1 metadata dicts for any True warnings
                            for level in ['L0', 'L1']:
                                meta = L0_data[level]['metadata']
                                if any(meta.get(f'bunching_warning_{var}', False) for var in ['haz', 'waz', 'whz']):
                                    warning_found = True
                                    break
                            if warning_found: break
                        if warning_found: break
                            
                    if warning_found:
                        warning_count += 1

                    # Calculate ranks
                    real_ranks, measured_ranks, _, _ = calculate_ranks_L0_units(
                        nested_measurements, 
                        measurement_var
                    )
                    
                    # Find real rank of L1 unit with best measured rank
                    best_measured_idx = np.argmin(measured_ranks)
                    sim_real_ranks.append(real_ranks[best_measured_idx])
                    
                    # Find overlap between top units
                    top_real = np.where(real_ranks <= n_L1_units_rewarded)[0]
                    top_measured = np.where(measured_ranks <= n_L1_units_rewarded)[0]
                    sim_overlaps.append(len(set(top_real) & set(top_measured)))
                
                # Print warning count for this parameter combination
                if warning_count > 0:
                    print(f"\nWarning: For parameter combination n_L0s={n_L0s}, n_child_L0={n_children_L0}, n_child_L1={n_children_L1}")
                    print(f"Percent shift was too high in {warning_count} out of {n_simulations} simulations")
                
                # Store results with mean and SEM across simulations
                results.append({
                    'n_L0s_per_L1': n_L0s,
                    'n_children_per_L0': n_children_L0,
                    'n_children_L1': n_children_L1,
                    'real_rank_mean': np.mean(sim_real_ranks),
                    'real_rank_sem': np.std(sim_real_ranks,ddof=1) / np.sqrt(n_simulations),
                    'n_overlap_mean': np.mean(sim_overlaps),
                    'n_overlap_sem': np.std(sim_overlaps,ddof=1) / np.sqrt(n_simulations)
                })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df









import numpy as np
import matplotlib.pyplot as plt

def plot_L0_unit_classification_confidence_vs_parameters(
    n_L0s_list,
    n_children_L0_list,
    n_children_L1_list,
    results_df,
    n_L1_units_rewarded
):
    """
    Plot classification confidence results from L0_unit_classification_confidence function.
    """

    # Create plots
    FONT_SIZE = 14
    fig_list = []
    
    # Determine which parameters are varying
    plot_vars = []
    if len(n_L0s_list) > 1:
        plot_vars.append('n_L0s_per_L1')
    if len(n_children_L0_list) > 1:
        plot_vars.append('n_children_per_L0')
    if len(n_children_L1_list) > 1:
        plot_vars.append('n_children_L1')
        
    # --- FIX 2: Prevent the 3D Graph Trap ---
    if len(plot_vars) > 2:
        raise ValueError("This plotting function only supports varying a maximum of TWO parameters at a time. "
                         "Please hold at least one parameter constant to generate readable 2D line plots.")
    
    # --- FIX 3: Updated Colormap Syntax ---
    try:
        color_map = plt.colormaps['Dark2']
    except AttributeError:
        color_map = plt.cm.get_cmap('Dark2') # Fallback for older Matplotlib versions
    

    # vs code change
    # --- FIX 4: Prevent Redundant Inverted Plots ---
    # Only iterate through the FIRST varying parameter to act as the X-axis. 
    # If there is a second varying parameter, it inherently becomes the colored lines.
    primary_x_var = plot_vars[0] if plot_vars else None
    
    if primary_x_var:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Find the other varying parameter (if it exists)
        other_vars = [v for v in plot_vars if v != primary_x_var]
        
        var = primary_x_var  # Reassign for the rest of your existing logic


    # for var in plot_vars:
    #     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
    #     # Find other varying parameters
    #     other_vars = [v for v in plot_vars if v != var]
        
        if other_vars:
            # Plot lines for each value of the other parameter
            other_var = other_vars[0]
            other_values = sorted(results_df[other_var].unique())
            colors = color_map(np.linspace(0, 1, len(other_values)))
            
            for i, val in enumerate(other_values):
                mask = results_df[other_var] == val
                
                # --- FIX 1: Sort by X-axis to prevent zig-zag lines ---
                plot_data = results_df[mask].sort_values(by=var)
                
                x = plot_data[var]
                
                # Plot real rank
                y = plot_data['real_rank_mean']
                yerr = plot_data['real_rank_sem']
                ax1.errorbar(x, y, yerr=yerr, fmt='o-', color=colors[i], 
                           label=f'{other_var}={val}', linewidth=2, markersize=8)
                
                # Plot overlap
                y = plot_data['n_overlap_mean']
                yerr = plot_data['n_overlap_sem']
                ax2.errorbar(x, y, yerr=yerr, fmt='o-', color=colors[i], 
                           label=f'{other_var}={val}', linewidth=2, markersize=8)
        else:
            # Single line if no other parameters vary
            # --- FIX 1: Sort by X-axis ---
            plot_data = results_df.sort_values(by=var)
            x = plot_data[var]
            
            # Plot real rank
            y = plot_data['real_rank_mean']
            yerr = plot_data['real_rank_sem']
            ax1.errorbar(x, y, yerr=yerr, fmt='ko-', linewidth=2, markersize=8)
            
            # Plot overlap
            y = plot_data['n_overlap_mean']
            yerr = plot_data['n_overlap_sem']
            ax2.errorbar(x, y, yerr=yerr, fmt='ko-', linewidth=2, markersize=8)
        
        # Set labels and formatting
        ax1.set_xlabel(var.replace('_', ' ').title(), fontsize=FONT_SIZE)
        ax1.set_ylabel('Real Rank of Best\nMeasured Unit', fontsize=FONT_SIZE)
        ax1.tick_params(axis='both', labelsize=FONT_SIZE-2)
        ax1.grid(True)
        if other_vars:
            ax1.legend(fontsize=FONT_SIZE-2)
        
        ax2.set_xlabel(var.replace('_', ' ').title(), fontsize=FONT_SIZE)
        ax2.set_ylabel(f'Number of True Top {n_L1_units_rewarded}\nUnits Identified', fontsize=FONT_SIZE)
        ax2.tick_params(axis='both', labelsize=FONT_SIZE-2)
        ax2.grid(True)
        if other_vars:
            ax2.legend(fontsize=FONT_SIZE-2)
        
        plt.tight_layout()
        fig_list.append(fig)

    return fig_list











import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def L0_classification_confidence_vs_L2_L1_discrepancy(
    # Real parameters
    real_params,
    n_L1s,
    n_L0s_per_L1,
    n_children_per_L0,
    n_children_L1,
    n_children_L2,
    # WHO parameters
    haz_params,
    waz_params,
    whz_params_lying,
    whz_params_standing,
    # Analysis parameters
    measurement_var,
    n_L0s_rewarded_per_L1,
    discrepancy_method='simple_difference',
    # Distortion parameters for L0s
    real_percent_stunting=40,
    real_percent_underweight=40,
    real_percent_wasting=None,
    mean_percent_under_reporting_stunting=20,
    mean_percent_under_reporting_underweight=20,
    mean_percent_under_reporting_wasting=None,
    mean_bunch_factor_haz=0.1,
    mean_bunch_factor_waz=0.1,
    mean_bunch_factor_whz=0.1,
    error_mean_height_all_L0s = 0,
    error_sd_height_all_L0s = 1,
    error_mean_weight_all_L0s = 0,
    error_sd_weight_all_L0s = 0.1,
    sd_across_units_percent_under_reporting_stunting=5,
    sd_across_units_percent_under_reporting_underweight=5,
    sd_across_units_percent_under_reporting_wasting=5,
    sd_across_units_bunch_factor_haz=0.02,
    sd_across_units_bunch_factor_waz=0.02,
    sd_across_units_bunch_factor_whz=0.02,
    sd_within_units_percent_under_reporting_stunting=2,
    sd_within_units_percent_under_reporting_underweight=2,
    sd_within_units_percent_under_reporting_wasting=2,
    sd_within_units_bunch_factor_haz=0.01,
    sd_within_units_bunch_factor_waz=0.01,
    sd_within_units_bunch_factor_whz=0.01,
    # Distortion parameters for L1s
    mean_percent_copy=10,
    mean_collusion_index=0.2,
    sd_percent_copy=2,
    sd_collusion_index=0.1,
    error_mean_height_L1=0,
    error_sd_height_L1=1,
    error_mean_weight_L1=0,
    error_sd_weight_L1=0.1,
    bunch_factor_haz_L1=0.05,
    bunch_factor_waz_L1=0.05,
    bunch_factor_whz_L1=0.05,
    # Distortion parameters for L2
    error_mean_height_L2=0,
    error_sd_height_L2=1,
    error_mean_weight_L2=0,
    error_sd_weight_L2=0.1,
    
    # --- NEW FIX: Removed old drift vars, added the Biological Time Lags ---
    mean_time_lag_L1=15,  
    sd_time_lag_L1=2,    
    mean_time_lag_L2=30,  
    # -----------------------------------------------------------------------

    random_seed=None,
    n_simulations=100,
):
    """
    Analyze L0 classification confidence versus L2-L1 discrepancy scores.
    """
    # Validation
    if n_L0s_rewarded_per_L1 > n_L0s_per_L1:
        raise ValueError(f"n_L0s_rewarded_per_L1 ({n_L0s_rewarded_per_L1}) must be <= n_L0s_per_L1 ({n_L0s_per_L1})")
    
    # Initialize lists
    n_real_L0s_rewarded = []
    L2_L1_discrepancies = []
    params = {
        'percent_copy': [],
        'collusion_index': []
    }
    warning_count = 0
    
    # Run simulations
    for sim in range(n_simulations):

        # Generate distortion parameters
        L0_params_list, L1_params_list, L2_params_dict = generate_nested_distortion_parameters(
            n_L1s=n_L1s,
            n_L0s_per_L1=n_L0s_per_L1,
            real_percent_stunting=real_percent_stunting,
            real_percent_underweight=real_percent_underweight,
            real_percent_wasting=real_percent_wasting,
            mean_percent_under_reporting_stunting=mean_percent_under_reporting_stunting,
            mean_percent_under_reporting_underweight=mean_percent_under_reporting_underweight,
            mean_percent_under_reporting_wasting=mean_percent_under_reporting_wasting,
            mean_bunch_factor_haz=mean_bunch_factor_haz,
            mean_bunch_factor_waz=mean_bunch_factor_waz,
            mean_bunch_factor_whz=mean_bunch_factor_whz,
            error_mean_height_all_L0s = error_mean_height_all_L0s,
            error_sd_height_all_L0s = error_sd_height_all_L0s,
            error_mean_weight_all_L0s = error_mean_weight_all_L0s,
            error_sd_weight_all_L0s = error_sd_weight_all_L0s,
            sd_across_units_percent_under_reporting_stunting=sd_across_units_percent_under_reporting_stunting,
            sd_across_units_percent_under_reporting_underweight=sd_across_units_percent_under_reporting_underweight,
            sd_across_units_percent_under_reporting_wasting=sd_across_units_percent_under_reporting_wasting,
            sd_across_units_bunch_factor_haz=sd_across_units_bunch_factor_haz,
            sd_across_units_bunch_factor_waz=sd_across_units_bunch_factor_waz,
            sd_across_units_bunch_factor_whz=sd_across_units_bunch_factor_whz,
            sd_within_units_percent_under_reporting_stunting=sd_within_units_percent_under_reporting_stunting,
            sd_within_units_percent_under_reporting_underweight=sd_within_units_percent_under_reporting_underweight,
            sd_within_units_percent_under_reporting_wasting=sd_within_units_percent_under_reporting_wasting,
            sd_within_units_bunch_factor_haz=sd_within_units_bunch_factor_haz,
            sd_within_units_bunch_factor_waz=sd_within_units_bunch_factor_waz,
            sd_within_units_bunch_factor_whz=sd_within_units_bunch_factor_whz,
            mean_percent_copy=mean_percent_copy,
            mean_collusion_index=mean_collusion_index,
            sd_percent_copy=sd_percent_copy,
            sd_collusion_index=sd_collusion_index,
            error_mean_height_L1=error_mean_height_L1,
            error_sd_height_L1=error_sd_height_L1,
            error_mean_weight_L1=error_mean_weight_L1,
            error_sd_weight_L1=error_sd_weight_L1,
            bunch_factor_haz_L1=bunch_factor_haz_L1,
            bunch_factor_waz_L1=bunch_factor_waz_L1,
            bunch_factor_whz_L1=bunch_factor_whz_L1,
            error_mean_height_L2=error_mean_height_L2,
            error_sd_height_L2=error_sd_height_L2,
            error_mean_weight_L2=error_mean_weight_L2,
            error_sd_weight_L2=error_sd_weight_L2,
            
            # --- NEW FIX: Injecting the Time Lags ---
            mean_time_lag_L1=mean_time_lag_L1,
            sd_time_lag_L1=sd_time_lag_L1,
            mean_time_lag_L2=mean_time_lag_L2,
            # ----------------------------------------
            
            #random_seed=random_seed+sim if random_seed else None
            random_seed=random_seed+sim if random_seed is not None else None# vs code change

        )
    
        # Generate nested measurements
        nested_measurements = generate_nested_measurements(
            real_params=real_params,
            L0_params_list=L0_params_list,
            L1_params_list=L1_params_list,
            L2_params_dict=L2_params_dict,
            n_L1s=n_L1s,
            n_L0s_per_L1=n_L0s_per_L1,
            n_children_per_L0=n_children_per_L0,
            n_children_L1=n_children_L1,
            n_children_L2=n_children_L2,
            haz_params=haz_params,
            waz_params=waz_params,
            whz_params_lying=whz_params_lying,
            whz_params_standing=whz_params_standing,
            make_plots=False
        )
        L1_L2_pairwise_data = get_L1_L2_pairwise_data(nested_measurements)

        # Check for warnings (Streamlined)
        warning_found = False
        for L1_id, L1_data in nested_measurements.items():
            if L1_id == 'metadata': continue
            for L0_id, L0_data in L1_data.items():
                if L0_id == 'L1_info': continue
                for level in ['L0', 'L1']:
                    meta = L0_data[level]['metadata']
                    if any(meta.get(f'bunching_warning_{var}', False) for var in ['haz', 'waz', 'whz']):
                        warning_found = True
                        break
                if warning_found: break
            if warning_found: break
                
        if warning_found:
            warning_count += 1
        
        # Calculate L0 ranks within each L1 unit
        L0_ranks = calculate_ranks_L0s(nested_measurements, measurement_var, method=discrepancy_method)
        
        # Calculate overlap and L2-L1 discrepancy for each L1 unit
        L1_no = 0
        for L1_id in sorted([k for k in L0_ranks.keys() if k.startswith('L1_')], key=lambda x: int(x.split('_')[1])):

            real_ranks = L0_ranks[L1_id]['real_ranks']
            measured_ranks = L0_ranks[L1_id]['measured_ranks']
            
            # Find overlap in top n_L0s_rewarded_per_L1
            top_real = np.where(real_ranks <= n_L0s_rewarded_per_L1)[0]
            top_measured = np.where(measured_ranks <= n_L0s_rewarded_per_L1)[0]
            overlap = len(set(top_real) & set(top_measured))
            n_real_L0s_rewarded.append(overlap)
            
            # Calculate L2-L1 discrepancy for this L1 unit            
            disc = calculate_discrepancy_scores(
                L1_L2_pairwise_data[L1_id]['L1'],
                L1_L2_pairwise_data[L1_id]['L2'],
                measurement_var,
                discrepancy_method,
                make_plot=False
            )
            
            # --- MATH FIX: Take Absolute Value FIRST, then Mean (Mean Absolute Error) ---
            #L2_L1_discrepancies.append(np.abs(disc).mean())
            #VS code change
            # --- MATH FIX: Take Absolute Value FIRST, then Mean (Mean Absolute Error) ---
            # --- NEW SAFETY FIX: Use nanmean to prevent missing data from wiping out the whole unit ---
            L2_L1_discrepancies.append(np.nanmean(np.abs(disc)))

            # Store L1 parameters for this L1 unit
            params['percent_copy'].append(L1_params_list[L1_no]['percent_copy'])
            params['collusion_index'].append(L1_params_list[L1_no]['collusion_index'])

            L1_no += 1
    
    # Print warning statistics
    if warning_count > 0:
        print(f"\nWarning: Percent shift was too high in {warning_count} out of {n_simulations} simulations")
    
    # Create scatter plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(L2_L1_discrepancies, n_real_L0s_rewarded, alpha=0.5, s=50, color='steelblue')
    ax.set_xlabel(f'L2-L1 Discrepancy ({measurement_var})', fontsize=14)
    ax.set_ylabel(f'Number of True Top {n_L0s_rewarded_per_L1} L0s Identified', fontsize=14)
    ax.set_title('L0 Classification Confidence vs L2-L1 Discrepancy', fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=12)
    plt.tight_layout()
    
    return n_real_L0s_rewarded, L2_L1_discrepancies, params, fig









    # VS code addition
    # Breadth vs Depth Trade off Graph. 
    # For a fixed No. of units measured, is it better to visit many L0 (Breadth) and measure few children or visit few L0 and measure many children (Depth)





import numpy as np
import matplotlib.pyplot as plt

def plot_dual_axis_tradeoff(results_df, x_col, depth_col, y_col, budget, title, y_label, higher_is_better=True):
    """
    Generates a high-quality Dual X-Axis plot using simulation data.
    """
    # ==============================================================================
    # 1. CRITICAL FIX: Sort data by the primary X-axis to prevent label mismatch!
    # ==============================================================================
    df_sorted = results_df.sort_values(by=x_col).copy()
    
    # Extract data safely
    breadth = df_sorted[x_col].values
    depth = df_sorted[depth_col].values
    scores = df_sorted[y_col].values

    # ==============================================================================
    # 2. CREATE THE PLOT
    # ==============================================================================
    fig, ax1 = plt.subplots(figsize=(12, 8), dpi=100)

    # Plot the Simulation Curve
    ax1.plot(breadth, scores, marker='o', markersize=8, color='#1a5276', 
             linewidth=2.5, label='Simulation Result')

    # ==============================================================================
    # 3. CONSTRUCT STACKED BOTTOM AXES
    # ==============================================================================

    # --- PRIMARY AXIS (Breadth) ---
    ax1.set_xticks(breadth)
    ax1.set_xlim(min(breadth) * 0.9, max(breadth) * 1.05) # Dynamic padding

    # Label 1: Placed BELOW the 1st row of ticks
    ax1.set_xlabel(f'Primary: {x_col.replace("_", " ").title()} (Breadth)', 
                   fontsize=12, fontweight='bold', labelpad=15)

    # --- SECONDARY AXIS (Depth) ---
    ax2 = ax1.twiny() # Create a twin axis

    # Force the secondary axis to the bottom
    ax2.xaxis.set_ticks_position('bottom')
    ax2.xaxis.set_label_position('bottom')

    # Move the 2nd axis "spine" down to create the "sandwich" gap
    ax2.spines['bottom'].set_position(('outward', 60)) # Reduced slightly for better fit

    # Align ticks perfectly with the primary axis
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(breadth)
    ax2.set_xticklabels(depth)

    # Label 2: Placed at the very bottom
    ax2.set_xlabel(f'Secondary: {depth_col.replace("_", " ").title()} (Depth)', 
                   fontsize=12, fontweight='bold', color='#d35400', labelpad=10)

    # ==============================================================================
    # 4. STYLING & HIGHLIGHTING
    # ==============================================================================

    ax1.set_ylabel(y_label, fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.7)
    
    # --- CRITICAL FIX 2: Safely find the optimal score based on the metric ---
    if higher_is_better:
        best_idx = np.argmax(scores)
    else:
        best_idx = np.argmin(scores)
        
    best_breadth = breadth[best_idx]
    best_score = scores[best_idx]
    best_depth = depth[best_idx]
    
    # Highlight the Peak (Optimal Strategy)
    ax1.scatter([best_breadth], [best_score], color='#1e8449', s=150, zorder=5, edgecolor='black')
    ax1.text(best_breadth, best_score, f'Optimal Strategy:\n{best_breadth} Units\n({best_depth} Kids/Unit)', 
             color='#1e8449', 
             bbox=dict(facecolor='white', edgecolor='#1e8449', boxstyle='round,pad=0.5', alpha=0.9),
             ha='center', weight='bold', va='bottom')

    plt.title(f'{title}\n(Fixed Budget: {budget})', fontsize=14, fontweight='bold', pad=20)
    
    # --- CRITICAL FIX 3: Explicit bottom margin to prevent cutting off the outer spine ---
    plt.subplots_adjust(bottom=0.25)
    
    return fig



















import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run_simulation_suite(
    mode='all',             
    total_budget=2000,      
    haz_params=None, 
    waz_params=None, 
    whz_params_lying=None, 
    whz_params_standing=None
):
    """
    Master function to run any or all audit simulation charts.
    """
    print(f"=== Running Simulation Suite: Mode = '{mode}' ===")
    
    # --- COMMON SETUP: Generate Random Worker Personalities ---
    print("... Generating Worker Profiles (Distortion Parameters) ...")
    
    params = generate_nested_distortion_parameters(
        n_L1s=20, n_L0s_per_L1=10,
        real_percent_stunting=36,
        mean_percent_under_reporting_stunting=30, 
        mean_time_lag_L1=30,  
        mean_time_lag_L2=60   
    )
    L0_pool, L1_pool, L2_params = params[0], params[1], params[2]

    common_real_params = {
        'girl_ratio': 0.5, 
        'min_age': 0, 
        'max_age': 1725,
        'num_timepoints': 1,          
        'time_lags': [],              
        'percent_stunting': 36,       
        'percent_underweight': 34     
    }

    # ==========================================================================
    # CHART 1: VALIDATION 
    # ==========================================================================
    if mode in ['all', 'validation']:
        print("\n--- Generating Chart 1: L0 Validation Scatter ---")
        
        val_data = generate_nested_measurements(
            real_params=common_real_params, 
            L0_params_list=L0_pool[:1], L1_params_list=L1_pool[:1], L2_params_dict=L2_params,
            n_L1s=1, n_L0s_per_L1=1, n_children_per_L0=500, 
            n_children_L1=0, n_children_L2=0,
            haz_params=haz_params, waz_params=waz_params, 
            whz_params_lying=whz_params_lying, whz_params_standing=whz_params_standing,
            make_plots=False
        )
        
        l1_key = list(val_data.keys())[0]
        l0_key = list(val_data[l1_key].keys())[0]
        real_df = val_data[l1_key][l0_key]['real']['data']
        rep_df = val_data[l1_key][l0_key]['L0']['data']
        
        plt.figure(figsize=(8, 6))
        plt.scatter(real_df['haz'], rep_df['haz'], alpha=0.3, color='purple', label='Child')
        plt.plot([-3, 3], [-3, 3], 'r--', label='Perfect Truth') 
        plt.axhline(y=-2, color='k', linestyle=':', label='Reporting Threshold')
        plt.xlabel('True Biological HAZ')
        plt.ylabel('Reported HAZ (L0)')
        plt.title('Validation: Are Workers lying about Stunting?')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    # ==========================================================================
    # CHART 2: L1 STRATEGY 
    # ==========================================================================
    if mode in ['all', 'L1_strategy']:
        print("\n--- Generating Chart 2: Supervisor Optimization (L1 Budget) ---")
        l1_results = []
        l1_budget = 500  
        center_counts = [2, 4, 5, 10, 20, 25] 
        
        for n_centers in center_counts:
            n_kids = int(l1_budget / n_centers)
            
            sim_data = generate_nested_measurements(
                real_params=common_real_params,
                L0_params_list=L0_pool[:n_centers], L1_params_list=L1_pool[:1], L2_params_dict=L2_params,
                n_L1s=1, n_L0s_per_L1=n_centers, n_children_per_L0=n_kids,
                n_children_L1=n_kids, 
                n_children_L2=0,
                haz_params=haz_params, waz_params=waz_params, 
                whz_params_lying=whz_params_lying, whz_params_standing=whz_params_standing,
                make_plots=False
            )
            
            l1_key = list(sim_data.keys())[0]
            diffs = []
            for k in sim_data[l1_key]:
                if k == 'L1_info': continue
                
                # --- MATH FIX: Strict 1-to-1 Index alignment & Mean Absolute Error ---
                l0_haz = sim_data[l1_key][k]['L0']['data']['haz']
                l1_haz = sim_data[l1_key][k]['L1']['data']['haz']
                
                # Compare only the kids L1 actually measured
                aligned_diff = np.abs(l1_haz - l0_haz.loc[l1_haz.index]).mean()
                diffs.append(aligned_diff)
            
            l1_results.append({'Centers': n_centers, 'Kids_per_Center': n_kids, 'Detection_Score': np.mean(diffs)})
            
        df = pd.DataFrame(l1_results)

        plot_dual_axis_tradeoff(
            results_df=df,
            x_col='Centers',          # Primary X (Breadth)
            depth_col='Kids_per_Center', # Secondary X (Depth)
            y_col='Detection_Score',  # Y-Axis
            budget=l1_budget,
            title="Supervisor Strategy Optimization",
            y_label="Avg. Discrepancy Detected (HAZ)",
            higher_is_better=True  # We want supervisors to detect MORE fraud!
        )
        
    # ==========================================================================
    # CHART 3: L2 STRATEGY 
    # ==========================================================================
    if mode in ['all', 'L2_strategy']:
        print("\n--- Generating Chart 3: Auditor Optimization (L2 Budget) ---")
        l2_results = []
        audit_budget = 50 
        sup_counts = [2, 5, 10]
        
        for n_sups in sup_counts:
            n_centers = int(audit_budget / n_sups)
            
            sim_data = generate_nested_measurements(
                real_params=common_real_params, 
                L0_params_list=L0_pool[:n_sups*n_centers], L1_params_list=L1_pool[:n_sups], L2_params_dict=L2_params,
                n_L1s=n_sups, n_L0s_per_L1=n_centers, n_children_per_L0=20, 
                n_children_L1=20, 
                n_children_L2=20, 
                haz_params=haz_params, waz_params=waz_params, 
                whz_params_lying=whz_params_lying, whz_params_standing=whz_params_standing,
                make_plots=False
            )
            
            total_disc = 0
            for l1_k in sim_data:
                if l1_k == 'metadata': continue
                for l0_k in sim_data[l1_k]:
                    if l0_k == 'L1_info': continue
                    
                    # --- MATH FIX: Strict 1-to-1 Index alignment & Mean Absolute Error ---
                    l1_haz = sim_data[l1_k][l0_k]['L1']['data']['haz']
                    l2_haz = sim_data[l1_k][l0_k]['L2']['data']['haz']
                    
                    diff = np.abs(l2_haz - l1_haz.loc[l2_haz.index]).mean()
                    total_disc += diff
            
            l2_results.append({'Supervisors': n_sups, 'Centers_per_Sup': n_centers, 'Total_Fraud_Found': total_disc})

        df2 = pd.DataFrame(l2_results)
        plot_dual_axis_tradeoff(
            results_df=df2,
            x_col='Supervisors',          # Primary X (Breadth)
            depth_col='Centers_per_Sup',  # Secondary X (Depth)
            y_col='Total_Fraud_Found',    # Y-Axis
            budget=audit_budget,
            title="Auditor Strategy Optimization",
            y_label="Total Collusion Detected",
            higher_is_better=True # We want the Auditor to catch MORE fraud!
        )
        
    # ==========================================================================
    # CHART 4: GRAND AUDIT 
    # ==========================================================================
    if mode in ['all', 'grand_audit']:
        print("\n--- Generating Chart 4: Grand System Audit (3D Heatmap) ---")
        grand_results = []
        grand_budget = 4000 
        sup_opts = [2, 4, 8]      
        center_opts = [2, 5, 10]  
        
        for n_sups in sup_opts:
            for n_centers in center_opts:
                total_centers = n_sups * n_centers
                if total_centers == 0: continue
                n_kids = int(grand_budget / total_centers)
                
                if n_kids < 5: continue 
                
                sim_data = generate_nested_measurements(
                    real_params=common_real_params,
                    L0_params_list=L0_pool[:total_centers], L1_params_list=L1_pool[:n_sups], L2_params_dict=L2_params,
                    n_L1s=n_sups, n_L0s_per_L1=n_centers, n_children_per_L0=n_kids,
                    n_children_L1=n_kids, n_children_L2=n_kids,
                    haz_params=haz_params, waz_params=waz_params, 
                    whz_params_lying=whz_params_lying, whz_params_standing=whz_params_standing,
                    make_plots=False
                )
                
                all_diffs = []
                for l1_k in sim_data:
                    if l1_k == 'metadata': continue
                    for l0_k in sim_data[l1_k]:
                        if l0_k == 'L1_info': continue
                        
                        # --- MATH FIX: Individual Absolute Differences ---
                        real_haz = sim_data[l1_k][l0_k]['real']['data']['haz']
                        l2_haz = sim_data[l1_k][l0_k]['L2']['data']['haz']
                        
                        aligned_diffs = np.abs(l2_haz - real_haz.loc[l2_haz.index])
                        
                        # --- MEMORY FIX: Strip Pandas index to prevent RAM bloat ---
                        all_diffs.extend(aligned_diffs.tolist())
                
                # The total error is the mean of all individual absolute differences
                error = np.mean(all_diffs)
                grand_results.append({'Supervisors': n_sups, 'Centers': n_centers, 'Error': error})

        hm_df = pd.DataFrame(grand_results).pivot(index="Centers", columns="Supervisors", values="Error")
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(hm_df, annot=True, cmap='RdYlGn_r', fmt='.3f')
        plt.title(f'System Error Rate (Budget: {grand_budget} Kids)\nX=Supervisors, Y=Centers per Sup')
        plt.show()
















































import numpy as np
import pandas as pd
import random # <--- NEW FIX: Import standard python random

def run_monte_carlo_experiment(
    L0_pool, L1_pool, L2_params, 
    common_real_params, 
    haz_params, waz_params, whz_params_lying, whz_params_standing,
    budget=2000, 
    min_breadth=1, 
    max_breadth=20, 
    n_trials=5
):
    results = []
    z_score_95 = 1.96
    
    print(f"--- Starting Monte Carlo (Budget={budget}, Trials={n_trials}) ---")
    
    for n_units in range(min_breadth, max_breadth + 1):
        n_samples = int(budget / n_units)
        if n_samples < 5: 
            break
            
        trial_scores_error = []
        trial_scores_found = []
        
        for i in range(n_trials):
            # --- FIX 2: Safely sample dictionaries using Python's random.choices (allows duplicates so it never crashes) ---
            current_L0s = random.choices(L0_pool, k=n_units)
            
            # 2. Run Simulation
            try:
                data = generate_nested_measurements(
                    real_params=common_real_params,
                    L0_params_list=current_L0s, 
                    L1_params_list=L1_pool[:1], 
                    L2_params_dict=L2_params,
                    n_L1s=1, 
                    n_L0s_per_L1=n_units, 
                    n_children_per_L0=n_samples,
                    n_children_L1=n_samples,
                    n_children_L2=0,
                    haz_params=haz_params, waz_params=waz_params, 
                    whz_params_lying=whz_params_lying, whz_params_standing=whz_params_standing,
                    make_plots=False
                )
                
                l1_key = list(data.keys())[0]
                diffs = []
                real_vals = []
                measured_vals = [] # Track this in the same loop to guarantee order!
                
                for k in data[l1_key]:
                    if k == 'L1_info': continue
                    
                    # --- FIX 1: Strict 1-to-1 Index alignment & Mean Absolute Error ---
                    l0_haz = data[l1_key][k]['L0']['data']['haz']
                    l1_haz = data[l1_key][k]['L1']['data']['haz']
                    
                    aligned_diff = np.abs(l1_haz - l0_haz.loc[l1_haz.index]).mean()
                    diffs.append(aligned_diff)
                    
                    # Collect center means for the Top 30% calculation
                    real_vals.append(data[l1_key][k]['real']['data']['haz'].mean())
                    measured_vals.append(l1_haz.mean())
                
                trial_scores_error.append(np.mean(diffs))
                
                # --- METRIC 2: BEST UNITS FOUND (Did we find the top 30%?) ---
                n_top = max(1, int(n_units * 0.3)) # Top 30%
                
                # Identify True Top 30%
                sorted_real = np.argsort(real_vals)[-n_top:]
                
                # Identify Measured Top 30% (Guaranteed alignment because we built the lists together)
                sorted_measured = np.argsort(measured_vals)[-n_top:]
                
                # Count intersection (How many did we get right?)
                correct_picks = len(set(sorted_real).intersection(set(sorted_measured)))
                trial_scores_found.append(correct_picks)
                
            except Exception as e:
                print(f"Error on trial {i}: {e}")
                continue
        
        # --- AGGREGATE STATS ---
        if len(trial_scores_error) > 0:
            # Stats for Error
            mean_err = np.mean(trial_scores_error)
            ci_err = z_score_95 * (np.std(trial_scores_error) / np.sqrt(len(trial_scores_error)))
            
            # Stats for Found
            mean_found = np.mean(trial_scores_found)
            ci_found = z_score_95 * (np.std(trial_scores_found) / np.sqrt(len(trial_scores_found)))
            
            print(f"Units {n_units}: Mean Found={mean_found:.1f}, Mean Error={mean_err:.3f}")
            
            results.append({
                'breadth': n_units,
                'depth': n_samples,
                'mean_error': mean_err,
                'lower_ci_error': mean_err - ci_err,
                'upper_ci_error': mean_err + ci_err,
                'mean_found': mean_found,
                'lower_ci_found': mean_found - ci_found,
                'upper_ci_found': mean_found + ci_found
            })

    return pd.DataFrame(results)





















import matplotlib.pyplot as plt
import numpy as np

def plot_monte_carlo_tradeoff(df, budget, title, metric='error'):
    """
    Plots the Monte Carlo results with correct dual-axis labels and confidence intervals.
    metric options: 'error' (HAZ Error) or 'found' (Best Units Found)
    """
    # --- CRITICAL FIX 1: Sort data to prevent zig-zags and label mismatch! ---
    df_sorted = df.sort_values(by='breadth').copy()
    
    fig, ax1 = plt.subplots(figsize=(12, 8), dpi=100)

    # Choose Metric Columns
    if metric == 'error':
        y_col, l_ci, u_ci = 'mean_error', 'lower_ci_error', 'upper_ci_error'
        y_label_text = 'Avg. Discrepancy Detected (HAZ Error)'
        color_line = '#e74c3c' # Red for Error
        color_fill = '#e74c3c'
        label_text = 'Mean HAZ Error'
    else:
        y_col, l_ci, u_ci = 'mean_found', 'lower_ci_found', 'upper_ci_found'
        y_label_text = 'No. of "Real" Best Units Correctly Identified'
        color_line = '#1a5276' # Blue for Success
        color_fill = '#3498db'
        label_text = 'Correctly Identified Units'

    # 1. Plot Mean Line & CI using SORTED data
    ax1.plot(df_sorted['breadth'], df_sorted[y_col], marker='o', markersize=5, 
             color=color_line, linewidth=2, label=label_text)
    
    ax1.fill_between(df_sorted['breadth'], df_sorted[l_ci], df_sorted[u_ci], 
                     color=color_fill, alpha=0.25, label='95% Confidence Interval')

    # 2. Setup Primary Axis (Breadth)
    ax1.set_xticks(df_sorted['breadth']) # Explicitly set ticks so ax2 can copy them safely
    ax1.set_xlim(min(df_sorted['breadth']) * 0.9, max(df_sorted['breadth']) * 1.05)
    
    ax1.set_xlabel('No. of L0s Tested per Unit (Breadth)', fontsize=11, fontweight='bold', labelpad=25)
    ax1.set_ylabel(y_label_text, fontsize=12, fontweight='bold')
    ax1.set_title(f'{title}\n(Fixed Budget: {budget} Samples)', fontsize=13, pad=25)
    ax1.grid(True, linestyle=':', alpha=0.7)
    ax1.legend(loc='upper left')

    # 3. Setup Secondary Axis (Depth)
    ax2 = ax1.twiny()
    ax2.xaxis.set_ticks_position('bottom')
    ax2.xaxis.set_label_position('bottom')
    ax2.spines['bottom'].set_position(('outward', 60))
    
    # Safely map the depth labels to the explicit breadth ticks
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(df_sorted['breadth'])
    ax2.set_xticklabels(df_sorted['depth'])
    ax2.set_xlabel('No. of Samples per L0 (Depth)', fontsize=11, fontweight='bold', color='#d35400')

    # --- CRITICAL FIX 2: Manually adjust bottom margin instead of tight_layout() ---
    plt.subplots_adjust(bottom=0.2)
    
    plt.show()
    return fig









    


















def run_simulation_suite1(
    mode='all',             
    total_budget=2000,      
    haz_params=None, 
    waz_params=None, 
    whz_params_lying=None, 
    whz_params_standing=None
):
    print(f"=== Running Simulation Suite: Mode = '{mode}' ===")
    
    # 1. Generate a LARGE pool of workers (so we can random sample from them)
    print("... Generating Worker Pool (n=100) ...")
    params = generate_nested_distortion_parameters(
        n_L1s=20, n_L0s_per_L1=100, # Generate 100 potential L0s
        real_percent_stunting=36,
        mean_percent_under_reporting_stunting=30, 
        mean_time_lag_L1=30,  
        mean_time_lag_L2=60   
    )
    # Flatten the L0 list so we can sample from it easily
    L0_pool = params[0] 
    L1_pool = params[1]
    L2_params = params[2]

    # 2. SAFE REAL PARAMS (Fixes the crashing bug)
    common_real_params = {
        'girl_ratio': 0.5, 
        'min_age': 0, 
        'max_age': 1700,          # <--- SAFE BUFFER (Crucial Fix)
        'num_timepoints': 1,          
        'time_lags': [],              
        'percent_stunting': 36,       
        'percent_underweight': 34     
    }

    # ==========================================================================
    # CHART 2: L1 STRATEGY (Monte Carlo Version)
    # ==========================================================================
    if mode in ['all', 'L1_strategy']:
        print("\n--- Running Monte Carlo: Supervisor Optimization ---")
        
        # 1. Run Simulation (Calculates both metrics)
        df_results = run_monte_carlo_experiment(
            L0_pool=L0_pool, L1_pool=L1_pool, L2_params=L2_params,
            common_real_params=common_real_params,
            haz_params=haz_params, waz_params=waz_params, 
            whz_params_lying=whz_params_lying, whz_params_standing=whz_params_standing,
            budget=500,       
            min_breadth=5,    # Start higher (e.g., 5) to have enough units to pick "Top 30%"
            max_breadth=20,   
            n_trials=5        
        )
        
        # 2. Plot Chart A: HAZ Error (Red)
        plot_monte_carlo_tradeoff(
            df_results, budget=500, 
            title="Optimization A: Minimizing Measurement Error", 
            metric='error'
        )
        
        # 3. Plot Chart B: Best Units Found (Blue)
        plot_monte_carlo_tradeoff(
            df_results, budget=500, 
            title="Optimization B: Identifying High-Performing Units", 
            metric='found'
        )