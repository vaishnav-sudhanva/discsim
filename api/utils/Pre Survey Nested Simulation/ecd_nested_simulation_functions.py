import numpy as np
import pandas as pd
from scipy.stats import norm
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm
from disc_score import discrepancy_score


def generate_real_measurements(
    num_children,
    girl_ratio,
    num_timepoints,
    time_lags,
    #height_growth_mean,
    #height_growth_std,
    #weight_growth_mean,
    #weight_growth_std,
    min_age,
    max_age,
    haz_params,
    waz_params,
    whz_params_lying,
    whz_params_standing,
    haz_mean_0 = None,
    haz_std_0 = None,
    waz_mean_0 = None,
    waz_std_0 = None,
    percent_stunting = None,
    percent_underweight = None,
    min_height=None,
    max_height=None,
    min_weight=None,
    max_weight=None,
    plot_distributions = False,
    figsize = [10, 8],
    verbose = False
    
):
    """
    Generate real measurements for children at multiple time points, with correlated growth rates.

    Args:
        

    Returns:
        pd.DataFrame: Table with columns ['child_id', 'gender', 'timepoint', 'age', 'height', 'weight']
    """
    # Assign genders
    num_girls = int(num_children * girl_ratio)
    num_boys = num_children - num_girls
    genders = [1] * num_boys + [2] * num_girls
    np.random.shuffle(genders)

    # Assign starting ages - uniform distribution between min and max age
    start_ages = np.random.uniform(low=min_age, high=max_age, size=num_children).astype(int)
    start_ages = pd.Series(start_ages)

    # Prepare time lags and timepoints
    time_lags = np.array(time_lags)
    assert len(time_lags) == num_timepoints - 1
    time_offsets = np.concatenate(([0], np.cumsum(time_lags)))

    # Initial heights
    # Check that at least one of haz_mean or percent_stunting is not None. If both are None, throw an error
    if (haz_mean_0 is None or haz_std_0 is None) and percent_stunting is None:
        raise ValueError("At least one of haz_mean_0 or percent_stunting must be provided.")

    # If haz_mean_0 and haz_std_0 are provided, use them to generate initial haz distribution
    if haz_mean_0 is not None and haz_std_0 is not None:
        haz_0 = np.random.normal(loc=haz_mean_0, scale=haz_std_0, size=num_children)
        percent_stunting = (haz_0 < -2).mean()*100
    elif percent_stunting is not None:
        # If percent_stunting is provided, use it to calculate haz_mean_0, assuming standard deviation of 1. 
        # Percent stunting is the percentile of the distribution that is less than two standard deviations from the mean, i.e. < -2.
        haz_mean_0 = -2 - norm.ppf(percent_stunting/100)
        haz_0 = np.random.normal(loc=haz_mean_0, scale=1, size=num_children)

    # Initial weights
    # Check that at least one of waz_mean or percent_underweight is not None. If both are None, throw an error
    if (waz_mean_0 is None or waz_std_0 is None) and percent_underweight is None:
        raise ValueError("At least one of waz_mean_0 or percent_underweight must be provided.")

    # If waz_mean_0 and waz_std_0 are provided, use them to generate initial waz distribution
    if waz_mean_0 is not None and waz_std_0 is not None:
        waz_0 = np.random.normal(loc=waz_mean_0, scale=waz_std_0, size=num_children)
        percent_underweight = (waz_0 < -2).mean()*100
    elif percent_underweight is not None:
        # If percent_underweight is provided, use it to calculate waz_mean_0, assuming standard deviation of 1. 
        # Percent underweight is the percentile of the distribution that is less than two standard deviations from the mean, i.e. < -2.
        waz_mean_0 = -2 - norm.ppf(percent_underweight/100)
        waz_0 = np.random.normal(loc=waz_mean_0, scale=1, size=num_children)

    # Generate height and weight from age, haz and waz, and haz_params and waz_params
    heights_0 = height_from_haz(haz_0, start_ages, genders, haz_params)
    weights_0 = weight_from_waz(waz_0, start_ages, genders, waz_params)

    if min_height is not None:
        heights_0 = np.maximum(heights_0, min_height)
    if max_height is not None:
        heights_0 = np.minimum(heights_0, max_height)
    if min_weight is not None:
        weights_0 = np.maximum(weights_0, min_weight)
    if max_weight is not None:
        weights_0 = np.minimum(weights_0, max_weight)

    # Determine length or height measurement based on age (loh = 1 for lying if age < 2 years, else loh = 2 for standing)
    loh = pd.Series(np.where(start_ages < 730, 1, 2))  # 730 days = 2 years

    # Generate WHZ from height and weight
    whz_0 = calculate_whz(pd.Series(heights_0), pd.Series(weights_0), pd.Series(genders), loh, whz_params_lying, whz_params_standing)
    percent_wasting = (whz_0 < -2).mean() * 100
    if verbose:
        print('{0}% wasting'.format(percent_wasting))

    records = []
    real_measurements = {
        'data': pd.DataFrame({
            'child_id': [f"child_{i}" for i in range(num_children)],
            'gender': genders,
            # 'timepoint': [tp for _ in range(num_children)], # Uncomment if needed
            'age': start_ages,
            'height': heights_0,
            'weight': weights_0,
            'loh': loh,
            'haz': haz_0,
            'waz': waz_0,
            'whz': whz_0,
        }),
        'metadata': {
            'percent_stunting': percent_stunting,
            'percent_underweight': percent_underweight,
            'percent_wasting': percent_wasting
        }

    }
    
    # Plot age, haz, waz, whz, height and weight distributions if plot_distributions is True
    if plot_distributions:
        plt.figure(figsize=figsize, constrained_layout = True)

        plt.subplot(2, 3, 1)
        sns.histplot(real_measurements['data']['age']/365, bins=30, kde=True, color = 'lightgray')
        plt.title('Age (years)')

        plt.subplot(2, 3, 2)
        sns.histplot(real_measurements['data']['haz'], bins=30, kde=True, color = 'paleturquoise')
        plt.title('HAZ Distribution')

        plt.subplot(2, 3, 3)
        sns.histplot(real_measurements['data']['waz'], bins=30, kde=True, color = 'lightsalmon')
        plt.title('WAZ Distribution')

        plt.subplot(2, 3, 4)
        sns.histplot(real_measurements['data']['height'], bins=30, kde=True, color = 'turquoise')
        plt.xlim([40, 140])
        plt.title('Height (cm)')

        plt.subplot(2, 3, 5)
        sns.histplot(real_measurements['data']['weight'], bins=30, kde=True, color = 'salmon')
        plt.xlim([0, 30])
        plt.title('Weight (kg)')

        plt.subplot(2, 3, 6)
        sns.histplot(real_measurements['data']['whz'], bins=30, kde=True, color = 'lightyellow')
        plt.title('WHZ Distribution')

        plt.tight_layout()
        plt.show()

    return real_measurements

def generate_L0_distorted_measurements(
        real_measurements, 
        haz_params,
        waz_params,
        whz_params_lying,
        whz_params_standing,
        percent_under_reporting_stunting = None,
        percent_under_reporting_underweight = None,
        percent_under_reporting_wasting = None,
        reporting_threshold = -2,
        bunch_factor_haz = 0.05,
        bunch_factor_waz = 0.05,
        bunch_factor_whz = 0.05,
        bin_size = 0.1,
        error_mean_height = 0,
        error_sd_height = 1,
        error_mean_weight = 0,
        error_sd_weight = 0.1,
        make_plots = False, figsize = [15, 8]
        ):

    """
    Apply incentive and capacity-based distortions, and measurement error, to real data to generate distorted L0 data.
    Args:
        real_measurements (pd.DataFrame): Table with columns
        percent_under_reporting_stunting (float): Percentage of stunted children who are reported as non-stunted.
        percent_under_reporting_underweight (float): Percentage of underweight children who are reported as non
        percent_under_reporting_wasting (float): Percentage of wasted children who are reported as non-wasted.
        reporting_threshold (float): Z-score threshold for reporting (default is -2).
        n_bins (int): Number of bins to use for under-reporting distribution (default is 10).
    Returns:
        pd.DataFrame: Table with distorted measurements and reported status.
    """
    # Check that percent under reporting are provided either for stunting and underweight or wasting, and if not throw an error.
    if (percent_under_reporting_stunting is None or percent_under_reporting_underweight is None) and percent_under_reporting_wasting is None:
        raise ValueError("Percent under-reporting must be provided for either stunting and underewight, or wasting.")

    distorted_measurements = {
        'data': real_measurements['data'].copy(deep=True),
        'metadata': real_measurements['metadata'].copy()
    }
    distorted_measurements['metadata']['percent_under_reporting_stunting'] = percent_under_reporting_stunting
    distorted_measurements['metadata']['percent_under_reporting_underweight'] = percent_under_reporting_underweight
    distorted_measurements['metadata']['percent_under_reporting_wasting'] = percent_under_reporting_wasting

    # Apply under-reporting for stunting and underweight if provided
    if percent_under_reporting_stunting is not None and percent_under_reporting_underweight is not None:
        
        # Throw an error if percent under-reporting for stunting or underweight exceeds the actual percent stunting or underweight
        if percent_under_reporting_stunting > real_measurements['metadata']['percent_stunting']:
            raise ValueError("Percent under-reporting for stunting exceeds actual percent stunting.")
        if percent_under_reporting_underweight > real_measurements['metadata']['percent_underweight']:
            raise ValueError("Percent under-reporting for underweight exceeds actual percent underweight.")

        # Under-reporting for stunting
        distorted_measurements['data']['haz'], 
        distorted_measurements['metadata']['bunching_warning_haz'] = generate_bunched_data(threshold=reporting_threshold,
                                                               original_data=real_measurements['data']['haz'],
                                                               percent_below_threshold_original=real_measurements['metadata']['percent_stunting'],
                                                               percent_below_threshold_bunched=real_measurements['metadata']['percent_stunting'] - percent_under_reporting_stunting,
                                                               bunch_factor=bunch_factor_haz, bin_size=bin_size)
        # Under-reporting for underweight
        distorted_measurements['data']['waz'], 
        distorted_measurements['metadata']['bunching_warning_waz'] = generate_bunched_data(threshold=reporting_threshold,
                                                               original_data=distorted_measurements['data']['waz'],
                                                               percent_below_threshold_original=real_measurements['metadata']['percent_underweight'],
                                                               percent_below_threshold_bunched=real_measurements['metadata']['percent_underweight'] - percent_under_reporting_underweight,
                                                               bunch_factor=bunch_factor_waz, bin_size=bin_size)

        # Calculate distorted height and weight based on distorted HAZ and WAZ
        distorted_measurements['data']['height'] = height_from_haz(distorted_measurements['data']['haz'], real_measurements['data']['age'],
                                                           real_measurements['data']['gender'], haz_params)
        distorted_measurements['data']['weight'] = weight_from_waz(distorted_measurements['data']['waz'], real_measurements['data']['age'],
                                                           real_measurements['data']['gender'], waz_params)

        # Calculate distorted WHZ from distorted height and weight
        distorted_measurements['data']['whz'] = calculate_whz(distorted_measurements['data']['height'], distorted_measurements['data']['weight'],
                                                      distorted_measurements['data']['gender'], real_measurements['data']['loh'],
                                                      whz_params_lying, whz_params_standing)
            
    # Apply under-reporting for wasting if provided
    elif percent_under_reporting_wasting is not None:
        # Throw an error if percent under-reporting for wasting exceeds the actual percent wasting
        if percent_under_reporting_wasting > real_measurements['metadata']['percent_wasting']:
            raise ValueError("Percent under-reporting for wasting exceeds actual percent wasting.")
        
        # Under-reporting for wasting
        distorted_measurements['data']['whz'], 
        distorted_measurements['metadata']['bunching_warning_whz'] = generate_bunched_data(threshold=reporting_threshold,
                                                               original_data=distorted_measurements['data']['whz'],
                                                               percent_below_threshold_original=real_measurements['metadata']['percent_wasting'],
                                                               percent_below_threshold_bunched=real_measurements['metadata']['percent_wasting'] - percent_under_reporting_wasting,
                                                               bunch_factor=bunch_factor_whz, bin_size=bin_size)

        # Calculate distorted weight from distorted WHZ, assuming height remains un-distorted
        distorted_measurements['data']['weight'] = weight_from_whz(distorted_measurements['data']['whz'], real_measurements['data']['age'],
                                                           real_measurements['data']['gender'], whz_params_lying, whz_params_standing)

        # Calculate distorted WAZ from distorted weight
        distorted_measurements['data']['waz'] = calculate_waz(distorted_measurements['data']['weight'], real_measurements['data']['age'],
                                                       real_measurements['data']['gender'], waz_params)
        
    # Add measurement error to height and weight
    distorted_measurements['data']['height'] = add_measurement_error(distorted_measurements['data']['height'], 
                                                                     error_mean=error_mean_height, error_sd=error_sd_height)
    distorted_measurements['data']['weight'] = add_measurement_error(distorted_measurements['data']['weight'], 
                                                                     error_mean=error_mean_weight, error_sd=error_sd_weight)

    # Re-calculate HAZ, WAZ and WHZ from distorted height and weight with measurement error
    distorted_measurements['data']['haz'] = calculate_haz(distorted_measurements['data']['height'], real_measurements['data']['age'],
                                                 real_measurements['data']['gender'], haz_params)
    distorted_measurements['data']['waz'] = calculate_waz(distorted_measurements['data']['weight'], real_measurements['data']['age'],
                                                 real_measurements['data']['gender'], waz_params)
    distorted_measurements['data']['whz'] = calculate_whz(distorted_measurements['data']['height'], distorted_measurements['data']['weight'],
                                                 distorted_measurements['data']['gender'], distorted_measurements['data']['loh'],
                                                 whz_params_lying, whz_params_standing)

    # If make_plots is true, plot distributions of height, weight, haz, waz and whz for real and distorted data, and the differences
    if make_plots:

        fig, axs = plt.subplots(nrows=2, ncols=5, figsize=figsize, constrained_layout=True)

        # Improved color palette and outlines for clarity
        real_color = "#94979a"        # blue
        real_edge = "#0d2c47"
        distorted_color = "#a226c1"   # orange
        distorted_edge = "#83028f"
        alpha_real = 0.8
        alpha_distorted = 0.3

        # Row 1: Overlapping histograms for real and distorted measurements

        # Height
        sns.histplot(real_measurements['data']['height'], bins=30, kde=False, color=real_color, 
                     label='Real', alpha=alpha_real, edgecolor=real_edge, linewidth=0.2, ax=axs[0, 0])
        sns.histplot(distorted_measurements['data']['height'], bins=30, kde=False, color=distorted_color, 
                     label='Distorted', alpha=alpha_distorted, edgecolor=distorted_edge, linewidth=0.2, ax=axs[0, 0])
        axs[0, 0].set_xlabel('Height (cm)')
        axs[0, 0].set_title('Height')
        axs[0, 0].legend()

        # Weight
        axs[0, 1]
        sns.histplot(real_measurements['data']['weight'], bins=30, kde=False, color=real_color, 
                     label='Real', alpha=alpha_real, edgecolor=real_edge, linewidth=0.2, ax=axs[0, 1])
        sns.histplot(distorted_measurements['data']['weight'], bins=30, kde=False, color=distorted_color, 
                     label='Distorted', alpha=alpha_distorted, edgecolor=distorted_edge, linewidth=0.2, ax=axs[0, 1])
        axs[0, 1].set_xlabel('Weight (kg)')
        axs[0, 1].set_title('Weight')
        axs[0, 1].legend()

        # HAZ
        axs[0, 2]
        min_haz = min(real_measurements['data']['haz'].min(), distorted_measurements['data']['haz'].min())
        max_haz = max(real_measurements['data']['haz'].max(), distorted_measurements['data']['haz'].max())
        bins_haz = np.arange(min_haz, max_haz, bin_size)
        #axs[0, 2].hist(real_measurements['data']['haz'], bins=bins_haz, color=real_color, alpha=alpha_real, edgecolor=real_edge, linewidth=0.2, label='Real')
        #axs[0, 2].hist(distorted_measurements['data']['haz'], bins=bins_haz, color=distorted_color, alpha=alpha_distorted, edgecolor=distorted_edge, linewidth=0.2, label='Distorted')
        sns.histplot(real_measurements['data']['haz'], bins=bins_haz, kde=False, color=real_color, 
                     label='Real', alpha=alpha_real, edgecolor=real_edge, linewidth=0.2, ax=axs[0, 2])
        sns.histplot(distorted_measurements['data']['haz'], bins=bins_haz, kde=False, color=distorted_color, 
                     label='Distorted', alpha=alpha_distorted, edgecolor=distorted_edge, linewidth=0.2, ax=axs[0, 2])
        axs[0, 2].axvline(x=reporting_threshold, color='red', linestyle='--')
        axs[0, 2].set_xlabel('HAZ')
        axs[0, 2].set_title('HAZ')
        axs[0, 2].legend()

        # WAZ
        axs[0, 3].sharey(axs[0, 2])
        min_waz = min(real_measurements['data']['waz'].min(), distorted_measurements['data']['waz'].min())
        max_waz = max(real_measurements['data']['waz'].max(), distorted_measurements['data']['waz'].max())
        bins_waz = np.arange(min_waz, max_waz, bin_size)
        #axs[0, 3].hist(real_measurements['data']['waz'], bins=bins_waz, color=real_color, alpha=alpha_real, edgecolor=real_edge, linewidth=0.2, label='Real')
        sns.histplot(real_measurements['data']['waz'], bins=bins_waz, kde=False, color=real_color, 
                     label='Real', alpha=alpha_real, edgecolor=real_edge, linewidth=0.2, ax=axs[0, 3])
        sns.histplot(distorted_measurements['data']['waz'], bins=bins_waz, kde=False, color=distorted_color, 
                     label='Distorted', alpha=alpha_distorted, edgecolor=distorted_edge, linewidth=0.2, ax=axs[0, 3])
        axs[0, 3].axvline(x=reporting_threshold, color='red', linestyle='--')
        axs[0, 3].set_xlabel('WAZ')
        axs[0, 3].set_title('WAZ')
        axs[0, 3].legend()

        # WHZ
        axs[0, 4].sharey(axs[0, 2])
        min_whz = min(real_measurements['data']['whz'].min(), distorted_measurements['data']['whz'].min())
        max_whz = max(real_measurements['data']['whz'].max(), distorted_measurements['data']['whz'].max())
        bins_whz = np.arange(min_whz, max_whz, bin_size)
        #axs[0, 4].hist(real_measurements['data']['whz'], bins=bins_whz, color=real_color, alpha=alpha_real, edgecolor=real_edge, linewidth=0.2, label='Real')
        sns.histplot(real_measurements['data']['whz'], bins=bins_whz, kde=False, color=real_color, 
                     label='Real', alpha=alpha_real, edgecolor=real_edge, linewidth=0.2, ax=axs[0, 4])
        sns.histplot(distorted_measurements['data']['whz'], bins=bins_whz, kde=False, color=distorted_color,
                     label='Distorted', alpha=alpha_distorted, edgecolor=distorted_edge, linewidth=0.2, ax=axs[0, 4])
        axs[0, 4].axvline(x=reporting_threshold, color='red', linestyle='--')
        axs[0, 4].set_xlabel('WHZ')
        axs[0, 4].set_title('WHZ')
        axs[0, 4].legend()

        # Make all y-axes in first row share the same scale
        y_min = min([ax.get_ylim()[0] for ax in axs[0, :]])
        y_max = max([ax.get_ylim()[1] for ax in axs[0, :]])
        for ax in axs[0, :]:
            ax.set_ylim(y_min, y_max)

        # Row 2: Differences

        # Height difference scatter plot
        axs[1, 0].sharex(axs[0, 0])
        axs[1, 0].scatter(x=real_measurements['data']['height'],
                          y=distorted_measurements['data']['height'] - real_measurements['data']['height'],
                        color='k', marker = '.', alpha = 0.1)
        axs[1, 0].set_xlabel('Real height (cm)')
        axs[1, 0].set_ylabel('Distorted - Real height (cm)')
        axs[1, 0].set_title('Height Diff')

        # Weight difference scatter plot
        axs[1, 1].sharex(axs[0, 1])
        axs[1, 1].scatter(x=real_measurements['data']['weight'],
                          y=distorted_measurements['data']['weight'] - real_measurements['data']['weight'],
                          color='k', marker='.', alpha=0.1)
        axs[1, 1].set_xlabel('Real weight (kg)')
        axs[1, 1].set_ylabel('Distorted - Real weight (kg)')
        axs[1, 1].set_title('Weight Diff')

        # HAZ difference histogram
        axs[1, 2].sharex(axs[0, 2])
        freq_real, _ = np.histogram(real_measurements['data']['haz'], bins=bins_haz, density=False)
        freq_distorted, _ = np.histogram(distorted_measurements['data']['haz'], bins=bins_haz, density=False)
        axs[1, 2].bar(x=bins_haz[:-1], height=freq_distorted - freq_real, width=np.diff(bins_haz), color='gray', alpha=0.7)
        axs[1, 2].set_xlabel('HAZ')
        axs[1, 2].set_ylabel('Count: Distorted - Real')
        axs[1, 2].axhline(0, color='black', linestyle='--')
        axs[1, 2].axvline(x=reporting_threshold, color='red', linestyle='--')
        axs[1, 2].set_title('HAZ Diff')

        # WAZ difference histogram
        axs[1, 3].sharex(axs[0, 3])
        axs[1, 3].sharey(axs[1, 2])
        freq_real, _ = np.histogram(real_measurements['data']['waz'], bins=bins_waz, density=False)
        freq_distorted, _ = np.histogram(distorted_measurements['data']['waz'], bins=bins_waz, density=False)
        axs[1, 3].bar(x=bins_waz[:-1], height=freq_distorted - freq_real, width=np.diff(bins_waz), color='gray', alpha=0.7)
        axs[1, 3].set_xlabel('WAZ')
        axs[1, 3].set_ylabel('Count: Distorted - Real')
        axs[1, 3].axhline(0, color='black', linestyle='--')
        axs[1, 3].axvline(x=reporting_threshold, color='red', linestyle='--')
        axs[1, 3].set_title('WAZ Diff')

        # WHZ difference histogram
        axs[1, 4].sharex(axs[0, 4])
        axs[1, 4].sharey(axs[1, 2])
        freq_real, _ = np.histogram(real_measurements['data']['whz'], bins=bins_whz, density=False)
        freq_distorted, _ = np.histogram(distorted_measurements['data']['whz'], bins=bins_whz, density=False)
        axs[1, 4].bar(x=bins_whz[:-1], height=freq_distorted - freq_real, width=np.diff(bins_whz), color='gray', alpha=0.7)
        axs[1, 4].set_xlabel('WHZ')
        axs[1, 4].set_ylabel('Count: Distorted - Real')
        axs[1, 4].axhline(0, color='black', linestyle='--')
        axs[1, 4].axvline(x=reporting_threshold, color='red', linestyle='--')
        axs[1, 4].set_title('WHZ Diff')

        # Make row 2 and 3 share y-axis scales for each metric
        for col in range(5):
            y_min = min(axs[1, col].get_ylim()[0], axs[2, col].get_ylim()[0])
            y_max = max(axs[1, col].get_ylim()[1], axs[2, col].get_ylim()[1])
            axs[1, col].set_ylim(y_min, y_max)
            axs[2, col].set_ylim(y_min, y_max)

        plt.tight_layout()
        plt.show()

    # Return distorted data
    return distorted_measurements

def generate_L1_distorted_measurements(
        real_measurements,
        L0_distorted_measurements,
        haz_params,
        waz_params,
        whz_params_lying,
        whz_params_standing,
        num_children_L1,
        percent_copy,
        collusion_index,
        error_mean_height = 0,
        error_sd_height = 1,
        error_mean_weight = 0,
        error_sd_weight = 0.1,
        bunch_factor_haz = 0.05,
        bunch_factor_waz = 0.05,
        bunch_factor_whz = 0.05,
        bin_size = 0.1,
        reporting_threshold = -2,
        make_plots = False, 
        figsize = [15, 8]
        ):
    """
    Apply collusion and copying distortions, and measurement error, to L0 distorted data to generate L1 distorted data.
    Args:
        real_measurements (pd.DataFrame): Table with columns
        L0_distorted_measurements (pd.DataFrame): Table with L0 distorted measurements.
        num_children_L1 (int): Number of children to be measured at L1.
        percent_copy (float): Percentage of children whose measurements are copied exactly from L0 to L
        collusion_index (float): Index representing the level of collusion in the population.
        error_mean_height (float): Mean of measurement error for height.
        error_sd_height (float): Standard deviation of measurement error for height.
        error_mean_weight (float): Mean of measurement error for weight.
        error_sd_weight (float): Standard deviation of measurement error for weight.
        Returns:
        pd.DataFrame: Table with L1 distorted measurements.
        """
    
    # Select random subset of children to be measured by L1
    all_indices = real_measurements['data'].index

    # Throw an error if num_children_L1 exceeds the total number of children in real_measurements
    if num_children_L1 > len(all_indices):
        raise ValueError("num_children_L1 exceeds the total number of children in real_measurements.")
    L1_indices = np.random.choice(all_indices, size=num_children_L1, replace=False)
    
    # Create measurements dictionary with only selected children
    real_subset_L1 = {
        'data': real_measurements['data'].loc[L1_indices].copy(),
        'metadata': real_measurements['metadata'].copy()
    }
    L0_subset_L1 = {
        'data': L0_distorted_measurements['data'].loc[L1_indices].copy(),
        'metadata': L0_distorted_measurements['metadata'].copy()
    }
    
    # Generate L1 measurements for subset using existing logic
    distorted_measurements = {
        'data': real_subset_L1['data'].copy(deep=True),
        'metadata': L0_subset_L1['metadata'].copy(),
        'L1_indices': L1_indices
    }
    
    # Copying: choose a subset of children for which values are taken exactly from L0. Retain the indices so that no measurement error or collusion is applied to these children.
    num_children = len(real_subset_L1['data'])
    num_copy = int(num_children * percent_copy / 100)
    copy_indices = np.random.choice(real_subset_L1['data'].index, size=num_copy, replace=False)
    distorted_measurements['data'].loc[copy_indices, 
                                       ['height', 'weight', 'haz', 'waz', 'whz']] = L0_subset_L1['data'].loc[copy_indices, 
                                                                                    ['height', 'weight', 'haz', 'waz', 'whz']]

    # Collusion: for the remaining children, apply collusion-based distortion
    collude_indices = [i for i in real_subset_L1['data'].index if i not in copy_indices]
    num_collude = len(collude_indices)

    if num_collude > 0:
        # Calculate percentage of children to be under-reported based on collusion index, only if corresponding L0 values exist
        percent_under_reporting_stunting = (collusion_index * L0_subset_L1['metadata']['percent_under_reporting_stunting'] 
                                          if L0_subset_L1['metadata']['percent_under_reporting_stunting'] is not None else None)
        percent_under_reporting_underweight = (collusion_index * L0_subset_L1['metadata']['percent_under_reporting_underweight']
                                             if L0_subset_L1['metadata']['percent_under_reporting_underweight'] is not None else None)
        percent_under_reporting_wasting = (collusion_index * L0_subset_L1['metadata']['percent_under_reporting_wasting']
                                         if L0_subset_L1['metadata']['percent_under_reporting_wasting'] is not None else None)

        # Apply under-reporting for stunting and underweight if both are provided
        if percent_under_reporting_stunting is not None and percent_under_reporting_underweight is not None:
            # Under-reporting for stunting
            distorted_measurements['data'].loc[collude_indices, 'haz'], 
            distorted_measurements['metadata']['bunching_warning_haz'] = generate_bunched_data(
                threshold=-2,
                original_data=real_subset_L1['data'].loc[collude_indices, 'haz'],
                percent_below_threshold_original=(real_subset_L1['data'].loc[collude_indices, 'haz'] < -2).mean() * 100,
                percent_below_threshold_bunched=(real_subset_L1['data'].loc[collude_indices, 'haz'] < -2).mean() * 100 - percent_under_reporting_stunting,
                bunch_factor=bunch_factor_haz,
                bin_size=bin_size
            )
            # Under-reporting for underweight
            distorted_measurements['data'].loc[collude_indices, 'waz'],
            distorted_measurements['metadata']['bunching_warning_waz'] = generate_bunched_data(
                threshold=-2,
                original_data=real_subset_L1['data'].loc[collude_indices, 'waz'],
                percent_below_threshold_original=(real_subset_L1['data'].loc[collude_indices, 'waz'] < -2).mean() * 100,
                percent_below_threshold_bunched=(real_subset_L1['data'].loc[collude_indices, 'waz'] < -2).mean() * 100 - percent_under_reporting_underweight,
                bunch_factor=bunch_factor_waz,
                bin_size=bin_size
            )

            # Calculate distorted height and weight based on distorted HAZ and WAZ
            distorted_measurements['data'].loc[collude_indices, 'height'] = height_from_haz(
                distorted_measurements['data'].loc[collude_indices, 'haz'],
                real_subset_L1['data'].loc[collude_indices, 'age'],
                real_subset_L1['data'].loc[collude_indices, 'gender'],  # Changed from 'sex'
                haz_params
            )
            distorted_measurements['data'].loc[collude_indices, 'weight'] = weight_from_waz(
                distorted_measurements['data'].loc[collude_indices, 'waz'],
                real_subset_L1['data'].loc[collude_indices, 'age'],
                real_subset_L1['data'].loc[collude_indices, 'gender'],  # Changed from 'sex'
                waz_params
            )   

            # Calculate distorted WHZ from distorted height and weight
            distorted_measurements['data'].loc[collude_indices, 'whz'] = calculate_whz(
                distorted_measurements['data'].loc[collude_indices, 'height'],
                distorted_measurements['data'].loc[collude_indices, 'weight'],
                distorted_measurements['data'].loc[collude_indices, 'gender'],  # Changed from 'sex'
                distorted_measurements['data'].loc[collude_indices, 'loh'],
                whz_params_lying, whz_params_standing
            )

        # Apply under-reporting for wasting if provided
        elif percent_under_reporting_wasting is not None:

            # Under-reporting for wasting
            distorted_measurements['data'].loc[collude_indices, 'whz'],
            distorted_measurements['metadata']['bunching_warning_whz'] = generate_bunched_data(
                threshold=-2,
                original_data=real_subset_L1['data'].loc[collude_indices, 'whz'],
                percent_below_threshold_original=(real_subset_L1['data'].loc[collude_indices, 'whz'] < -2).mean() * 100,
                percent_below_threshold_bunched=(real_subset_L1['data'].loc[collude_indices, 'whz'] < -2).mean() * 100 - percent_under_reporting_wasting,
                bunch_factor=bunch_factor_whz,
                bin_size=bin_size
            )

            # Calculate distorted weight from distorted WHZ, assuming height remains un-distorted
            distorted_measurements['data'].loc[collude_indices, 'weight'] = weight_from_whz(
                distorted_measurements['data'].loc[collude_indices, 'whz'],
                real_subset_L1['data'].loc[collude_indices, 'height'],
                real_subset_L1['data'].loc[collude_indices, 'gender'],  # Changed from 'sex'
                real_subset_L1['data'].loc[collude_indices, 'loh'],
                whz_params_lying, whz_params_standing
            )

            # Calculate distorted WAZ from distorted weight
            distorted_measurements['data'].loc[collude_indices, 'waz'] = calculate_waz(
                distorted_measurements['data'].loc[collude_indices, 'weight'],
                real_subset_L1['data'].loc[collude_indices, 'age'],
                real_subset_L1['data'].loc[collude_indices, 'gender'],  # Changed from 'sex'
                waz_params
            )

        # Add measurement error to height and weight for colluding children
        distorted_measurements['data'].loc[collude_indices, 'height'] = add_measurement_error(
            distorted_measurements['data'].loc[collude_indices, 'height'], 
            error_mean=error_mean_height, error_sd=error_sd_height
        )
        distorted_measurements['data'].loc[collude_indices, 'weight'] = add_measurement_error(
            distorted_measurements['data'].loc[collude_indices, 'weight'], 
            error_mean=error_mean_weight, error_sd=error_sd_weight
        )

        # Re-calculate HAZ, WAZ and WHZ from distorted height and weight with measurement error for colluding children
        distorted_measurements['data'].loc[collude_indices, 'haz'] = calculate_haz(
            distorted_measurements['data'].loc[collude_indices, 'height'],
            real_subset_L1['data'].loc[collude_indices, 'age'],
            real_subset_L1['data'].loc[collude_indices, 'gender'],  # Changed from 'sex'
            haz_params
        )
        distorted_measurements['data'].loc[collude_indices, 'waz'] = calculate_waz(
            distorted_measurements['data'].loc[collude_indices, 'weight'],
            real_subset_L1['data'].loc[collude_indices, 'age'],
            real_subset_L1['data'].loc[collude_indices, 'gender'],  # Changed from 'sex'
            waz_params
        )
        distorted_measurements['data'].loc[collude_indices, 'whz'] = calculate_whz(
            distorted_measurements['data'].loc[collude_indices, 'height'],
            distorted_measurements['data'].loc[collude_indices, 'weight'],
            distorted_measurements['data'].loc[collude_indices, 'gender'],  # Changed from 'sex'
            distorted_measurements['data'].loc[collude_indices, 'loh'],
            whz_params_lying, whz_params_standing
        )

    # If make_plots is true, plot distributions of height, weight, haz, waz and whz for real, L0 and L1 data
    if make_plots:
        fig, axs = plt.subplots(nrows=3, ncols=5, figsize=figsize, constrained_layout=True)

        # Enhanced color palette for better visibility of three overlapping distributions
        real_color = "#94979a"      # gray
        real_edge = "#0d2c47"
        L0_color = "#a226c1"        # purple
        L0_edge = "#83028f"
        L1_color = "#26a269"        # green
        L1_edge = "#1a6b44"
        alpha_real = 0.8
        alpha_L0 = 0.4
        alpha_L1 = 0.3

        # Row 1: Overlapping histograms for real, L0 and L1 measurements
        # Height
        sns.histplot(real_measurements['data']['height'], bins=30, kde=False, color=real_color, 
                    label='Real', alpha=alpha_real, edgecolor=real_edge, linewidth=0.2, ax=axs[0, 0])
        sns.histplot(L0_distorted_measurements['data']['height'], bins=30, kde=False, color=L0_color, 
                    label='L0', alpha=alpha_L0, edgecolor=L0_edge, linewidth=0.2, ax=axs[0, 0])
        sns.histplot(distorted_measurements['data']['height'], bins=30, kde=False, color=L1_color, 
                    label='L1', alpha=alpha_L1, edgecolor=L1_edge, linewidth=0.2, ax=axs[0, 0])
        axs[0, 0].set_xlabel('Height (cm)')
        axs[0, 0].set_title('Height')
        axs[0, 0].legend()

        # Weight
        sns.histplot(real_measurements['data']['weight'], bins=30, kde=False, color=real_color, 
                    label='Real', alpha=alpha_real, edgecolor=real_edge, linewidth=0.2, ax=axs[0, 1])
        sns.histplot(L0_distorted_measurements['data']['weight'], bins=30, kde=False, color=L0_color, 
                    label='L0', alpha=alpha_L0, edgecolor=L0_edge, linewidth=0.2, ax=axs[0, 1])
        sns.histplot(distorted_measurements['data']['weight'], bins=30, kde=False, color=L1_color, 
                    label='L1', alpha=alpha_L1, edgecolor=L1_edge, linewidth=0.2, ax=axs[0, 1])
        axs[0, 1].set_xlabel('Weight (kg)')
        axs[0, 1].set_title('Weight')
        axs[0, 1].legend()

        # HAZ
        min_haz = min(real_measurements['data']['haz'].min(), L0_distorted_measurements['data']['haz'].min(), 
                     distorted_measurements['data']['haz'].min())
        max_haz = max(real_measurements['data']['haz'].max(), L0_distorted_measurements['data']['haz'].max(), 
                     distorted_measurements['data']['haz'].max())
        bins_haz = np.arange(min_haz, max_haz, bin_size)
        sns.histplot(real_measurements['data']['haz'], bins=bins_haz, kde=False, color=real_color, 
                    label='Real', alpha=alpha_real, edgecolor=real_edge, linewidth=0.2, ax=axs[0, 2])
        sns.histplot(L0_distorted_measurements['data']['haz'], bins=bins_haz, kde=False, color=L0_color, 
                    label='L0', alpha=alpha_L0, edgecolor=L0_edge, linewidth=0.2, ax=axs[0, 2])
        sns.histplot(distorted_measurements['data']['haz'], bins=bins_haz, kde=False, color=L1_color, 
                    label='L1', alpha=alpha_L1, edgecolor=L1_edge, linewidth=0.2, ax=axs[0, 2])
        axs[0, 2].axvline(x=reporting_threshold, color='red', linestyle='--')
        axs[0, 2].set_xlabel('HAZ')
        axs[0, 2].set_title('HAZ')
        axs[0, 2].legend()

        # WAZ
        min_waz = min(real_measurements['data']['waz'].min(), L0_distorted_measurements['data']['waz'].min(), 
                     distorted_measurements['data']['waz'].min())
        max_waz = max(real_measurements['data']['waz'].max(), L0_distorted_measurements['data']['waz'].max(), 
                     distorted_measurements['data']['waz'].max())
        bins_waz = np.arange(min_waz, max_waz, bin_size)
        sns.histplot(real_measurements['data']['waz'], bins=bins_waz, kde=False, color=real_color, 
                    label='Real', alpha=alpha_real, edgecolor=real_edge, linewidth=0.2, ax=axs[0, 3])
        sns.histplot(L0_distorted_measurements['data']['waz'], bins=bins_waz, kde=False, color=L0_color, 
                    label='L0', alpha=alpha_L0, edgecolor=L0_edge, linewidth=0.2, ax=axs[0, 3])
        sns.histplot(distorted_measurements['data']['waz'], bins=bins_waz, kde=False, color=L1_color, 
                    label='L1', alpha=alpha_L1, edgecolor=L1_edge, linewidth=0.2, ax=axs[0, 3])
        axs[0, 3].axvline(x=reporting_threshold, color='red', linestyle='--')
        axs[0, 3].set_xlabel('WAZ')
        axs[0, 3].set_title('WAZ')
        axs[0, 3].legend()

        # WHZ
        min_whz = min(real_measurements['data']['whz'].min(), L0_distorted_measurements['data']['whz'].min(), 
                     distorted_measurements['data']['whz'].min())
        max_whz = max(real_measurements['data']['whz'].max(), L0_distorted_measurements['data']['whz'].max(), 
                     distorted_measurements['data']['whz'].max())
        bins_whz = np.arange(min_whz, max_whz, bin_size)
        sns.histplot(real_measurements['data']['whz'], bins=bins_whz, kde=False, color=real_color, 
                    label='Real', alpha=alpha_real, edgecolor=real_edge, linewidth=0.2, ax=axs[0, 4])
        sns.histplot(L0_distorted_measurements['data']['whz'], bins=bins_whz, kde=False, color=L0_color, 
                    label='L0', alpha=alpha_L0, edgecolor=L0_edge, linewidth=0.2, ax=axs[0, 4])
        sns.histplot(distorted_measurements['data']['whz'], bins=bins_whz, kde=False, color=L1_color, 
                    label='L1', alpha=alpha_L1, edgecolor=L1_edge, linewidth=0.2, ax=axs[0, 4])
        axs[0, 4].axvline(x=reporting_threshold, color='red', linestyle='--')
        axs[0, 4].set_xlabel('WHZ')
        axs[0, 4].set_title('WHZ')
        axs[0, 4].legend()

        # Make all y-axes in first row share the same scale
        y_min = min([ax.get_ylim()[0] for ax in axs[0, :]])
        y_max = max([ax.get_ylim()[1] for ax in axs[0, :]])
        for ax in axs[0, :]:
            ax.set_ylim(y_min, y_max)

        # Row 2: Differences

        # Height difference scatter plot
        axs[1, 0].sharex(axs[0, 0])
        axs[1, 0].scatter(x=real_measurements['data']['height'],
                          y=distorted_measurements['data']['height'] - real_measurements['data']['height'],
                        color='k', marker = '.', alpha = 0.1)
        axs[1, 0].set_xlabel('Real height (cm)')
        axs[1, 0].set_ylabel('Distorted - Real height (cm)')
        axs[1, 0].set_title('Height Diff')

        # Weight difference scatter plot
        axs[1, 1].sharex(axs[0, 1])
        axs[1, 1].scatter(x=real_measurements['data']['weight'],
                          y=distorted_measurements['data']['weight'] - real_measurements['data']['weight'],
                          color='k', marker='.', alpha=0.1)
        axs[1, 1].set_xlabel('Real weight (kg)')
        axs[1, 1].set_ylabel('Distorted - Real weight (kg)')
        axs[1, 1].set_title('Weight Diff')

        # HAZ difference histogram
        axs[1, 2].sharex(axs[0, 2])
        freq_real, _ = np.histogram(real_measurements['data']['haz'], bins=bins_haz, density=False)
        freq_distorted, _ = np.histogram(distorted_measurements['data']['haz'], bins=bins_haz, density=False)
        axs[1, 2].bar(x=bins_haz[:-1], height=freq_distorted - freq_real, width=np.diff(bins_haz), color='gray', alpha=0.7)
        axs[1, 2].set_xlabel('HAZ')
        axs[1, 2].set_ylabel('Count: Distorted - Real')
        axs[1, 2].axhline(0, color='black', linestyle='--')
        axs[1, 2].axvline(x=reporting_threshold, color='red', linestyle='--')
        axs[1, 2].set_title('HAZ Diff')

        # WAZ difference histogram
        axs[1, 3].sharex(axs[0, 3])
        axs[1, 3].sharey(axs[1, 2])
        freq_real, _ = np.histogram(real_measurements['data']['waz'], bins=bins_waz, density=False)
        freq_distorted, _ = np.histogram(distorted_measurements['data']['waz'], bins=bins_waz, density=False)
        axs[1, 3].bar(x=bins_waz[:-1], height=freq_distorted - freq_real, width=np.diff(bins_waz), color='gray', alpha=0.7)
        axs[1, 3].set_xlabel('WAZ')
        axs[1, 3].set_ylabel('Count: Distorted - Real')
        axs[1, 3].axhline(0, color='black', linestyle='--')
        axs[1, 3].axvline(x=reporting_threshold, color='red', linestyle='--')
        axs[1, 3].set_title('WAZ Diff')

        # WHZ difference histogram
        axs[1, 4].sharex(axs[0, 4])
        axs[1, 4].sharey(axs[1, 2])
        freq_real, _ = np.histogram(real_subset_L1['data']['whz'], bins=bins_whz, density=False)
        freq_distorted, _ = np.histogram(distorted_measurements['data']['whz'], bins=bins_whz, density=False)
        axs[1, 4].bar(x=bins_whz[:-1], height=freq_distorted - freq_real, width=np.diff(bins_whz), color='gray', alpha=0.7)
        axs[1, 4].set_xlabel('WHZ')
        axs[1, 4].set_ylabel('Count: Distorted - Real')
        axs[1, 4].axhline(0, color='black', linestyle='--')
        axs[1, 4].axvline(x=reporting_threshold, color='red', linestyle='--')
        axs[1, 4].set_title('WHZ Diff')

        # Row 3: L1 - L0 differences

        # Height difference scatter plot
        axs[2, 0].scatter(x=L0_subset_L1['data']['height'],
                         y=distorted_measurements['data']['height'] - L0_subset_L1['data']['height'],
                         color='k', marker='.', alpha=0.1)
        axs[2, 0].set_xlabel('L0 height (cm)')
        axs[2, 0].set_ylabel('L1 - L0 height (cm)')
        axs[2, 0].set_title('Height Diff (L1 - L0)')

        # Weight difference scatter plot
        axs[2, 1].scatter(x=L0_subset_L1['data']['weight'],
                         y=distorted_measurements['data']['weight'] - L0_subset_L1['data']['weight'],
                         color='k', marker='.', alpha=0.1)
        axs[2, 1].set_xlabel('L0 weight (kg)')
        axs[2, 1].set_ylabel('L1 - L0 weight (kg)')
        axs[2, 1].set_title('Weight Diff (L1 - L0)')

        # HAZ difference histogram
        freq_L0, _ = np.histogram(L0_subset_L1['data']['haz'], bins=bins_haz, density=False)
        freq_L1, _ = np.histogram(distorted_measurements['data']['haz'], bins=bins_haz, density=False)
        axs[2, 2].bar(x=bins_haz[:-1], height=freq_L1 - freq_L0, width=np.diff(bins_haz), color='gray', alpha=0.7)
        axs[2, 2].set_xlabel('HAZ')
        axs[2, 2].set_ylabel('Count: L1 - L0')
        axs[2, 2].axhline(0, color='black', linestyle='--')
        axs[2, 2].axvline(x=reporting_threshold, color='red', linestyle='--')
        axs[2, 2].set_title('HAZ Diff (L1 - L0)')

        # WAZ difference histogram
        freq_L0, _ = np.histogram(L0_subset_L1['data']['waz'], bins=bins_waz, density=False)
        freq_L1, _ = np.histogram(distorted_measurements['data']['waz'], bins=bins_waz, density=False)
        axs[2, 3].bar(x=bins_waz[:-1], height=freq_L1 - freq_L0, width=np.diff(bins_waz), color='gray', alpha=0.7)
        axs[2, 3].set_xlabel('WAZ')
        axs[2, 3].set_ylabel('Count: L1 - L0')
        axs[2, 3].axhline(0, color='black', linestyle='--')
        axs[2, 3].axvline(x=reporting_threshold, color='red', linestyle='--')
        axs[2, 3].set_title('WAZ Diff (L1 - L0)')

        # WHZ difference histogram
        freq_L0, _ = np.histogram(L0_subset_L1['data']['whz'], bins=bins_whz, density=False)
        freq_L1, _ = np.histogram(distorted_measurements['data']['whz'], bins=bins_whz, density=False)
        axs[2, 4].bar(x=bins_whz[:-1], height=freq_L1 - freq_L0, width=np.diff(bins_whz), color='gray', alpha=0.7)
        axs[2, 4].set_xlabel('WHZ')
        axs[2, 4].set_ylabel('Count: L1 - L0')
        axs[2, 4].axhline(0, color='black', linestyle='--')
        axs[2, 4].axvline(x=reporting_threshold, color='red', linestyle='--')
        axs[2, 4].set_title('WHZ Diff (L1 - L0)')

        # Make row 2 and 3 share y-axis scales for each metric
        for col in range(5):
            y_min = min(axs[1, col].get_ylim()[0], axs[2, col].get_ylim()[0])
            y_max = max(axs[1, col].get_ylim()[1], axs[2, col].get_ylim()[1])
            axs[1, col].set_ylim(y_min, y_max)
            axs[2, col].set_ylim(y_min, y_max)

        plt.tight_layout()
        plt.show()

    return distorted_measurements

def generate_bunched_data(threshold, original_data, percent_below_threshold_original, percent_below_threshold_bunched,
                          bunch_factor = 0.05, bin_size = 0.1):
    """
    Distort the original data by applying a bunching effect around the threshold.
    Args:
        threshold (float): The threshold value for bunching.
        original_data (pd.Series): The original data to be distorted.
        percent_below_threshold_original (float): The percentage of data below the threshold in the original data.
        percent_below_threshold_bunched (float): The desired percentage of data below the threshold in the bunched data.
        bunch_factor (float): Float between 0 and 1 (exclusive) which gives the intensity of bunching (closer to 1 means more bunching).
        bin_size (float): The size of the bins for bunching.
    Returns:
        tuple: (bunched_data, warning_flag) where warning_flag indicates if percent_shift was too high
    """
    bunched_data = original_data.copy(deep = True)
    warning_flag = False
    
    # Divide the range of the data into bins of size bin_size
    bins = np.arange(original_data.min(), original_data.max() + bin_size, bin_size)
    binned_data = pd.cut(original_data, bins=bins, include_lowest=True)

    # Calculate percentage of points to be shifted above threshold from each bin
    percent_shift = 100 * (1 - percent_below_threshold_bunched/percent_below_threshold_original)
    
    # In each bin below the threshold, choose a random subset of points (percent_shift % from each bin)
    shift_indices = []
    for bin in binned_data.cat.categories:
        if bin.right <= threshold:
            # Find the number of data points in this bin
            n_points = (binned_data == bin).sum()
            # Calculate the number of points to shift
            n_shift = int(n_points * percent_shift/100)
            
            # Check if n_shift is larger than available points
            if n_shift > len(bunched_data[binned_data == bin]):
                warning_flag = True
                shift_points = bunched_data[binned_data == bin]  # Take all points
                shift_indices.extend(shift_points.index)
            else:
                # Choose random points to shift and get their indices in bunched_data
                shift_points = bunched_data[binned_data == bin].sample(n=n_shift, replace=False)
                shift_indices.extend(shift_points.index)

    # Assign exponentially decreasing probabilities to each bin above the threshold based on the bunch factor
    probabilities = []
    bins_above_threshold = [b for b in binned_data.cat.categories if b.left > threshold]
    for i, bin in enumerate(bins_above_threshold):
        prob = (1 - bunch_factor) ** i
        probabilities.append(prob)

    # Normalize probabilities to sum to 1
    probabilities = np.array(probabilities)
    probabilities /= probabilities.sum()

    # For each point to be shifted, choose a bin above the threshold based on the probabilities
    for idx in shift_indices:
        chosen_bin = np.random.choice(bins_above_threshold, p=probabilities)
        # Choose a random point from the chosen bin and assign it to the bunched data
        bunched_data.loc[idx] = np.random.uniform(chosen_bin.left, chosen_bin.right)

    return bunched_data, warning_flag

def add_measurement_error(data, error_mean=0.5, error_sd=0.1):
    """
    Add random measurement error to height and weight measurements.

    Args:
        data (pd.DataFrame)
        error_mean (float): Mean of the normal distribution for measurement error.
        error_sd (float): Standard deviation of the normal distribution for measurement error.

    Returns:
        pd.DataFrame: DataFrame with added measurement error.
    """
    data_with_error = data.copy(deep=True)
    error = np.random.normal(loc=error_mean, scale=error_sd, size=len(data))
    data_with_error += error
    
    return data_with_error

def calculate_haz(height, age, sex, haz_params):
    """
    Calculate Height-for-Age Z-scores (HAZ) using the WHO growth standards.

    Args:
        height (pd.Series): Height measurements.
        age (pd.Series): Age measurements.
        sex (pd.Series): Sex (gender) of the children.
        loh (pd.Series): Whether height was measured standing or lying down.
        haz_params (pd.DataFrame): HAZ parameters from WHO growth standards.

    Returns:
        pd.Series: HAZ scores.
    """
    # Make sure height, age, sex and loh are the same length, else throw an error
    if not (len(height) == len(age) == len(sex)):
        raise ValueError("Input series must have the same length")

    # Get M, S, L values for the age and sex
    m, s, l = get_msl(age, sex, haz_params)
    haz = get_anthro_zscore(height.dropna(), m, s, l)
    return haz

def height_from_haz(haz, age, sex, haz_params):
    """
    Calculate height from Height-for-Age Z-scores (HAZ) using the WHO growth standards.

    Args:
        haz (pd.Series): HAZ scores.
        age (pd.Series): Age measurements.
        sex (pd.Series): Sex (gender) of the children.
        loh (pd.Series): Whether height was measured standing or lying down.
        haz_params (pd.DataFrame): HAZ parameters from WHO growth standards.

    Returns:
        pd.Series: Height measurements.
    """
    # Make sure haz, age, sex and loh are the same length, else throw an error
    if not (len(haz) == len(age) == len(sex)):
        raise ValueError("Input series must have the same length")
    
    # Get M, S, L values for the age and sex
    m, s, l = get_msl(age, sex, haz_params)
    height = invert_anthro_zscore(haz, m, s, l)
    return height

def calculate_waz(weight, age, sex, waz_params):
    """
    Calculate Weight-for-Age Z-scores (WAZ) using the WHO growth standards.

    Args:
        weight (pd.Series): Weight measurements.
        age (pd.Series): Age measurements.
        sex (pd.Series): Sex (gender) of the children.
        waz_params (pd.DataFrame): WAZ parameters from WHO growth standards.

    Returns:
        pd.Series: WAZ scores.
    """
    # Make sure weight, age, and sex are the same length, else throw an error
    if not (len(weight) == len(age) == len(sex)):
        raise ValueError("Input series must have the same length")
    
    # Get M, S, L values for the age and sex
    m, s, l = get_msl(age, sex, waz_params)
    waz = get_anthro_zscore(weight, m, s, l)
    return waz

def weight_from_waz(waz, age, sex, waz_params):
    """
    Calculate weight from Weight-for-Age Z-scores (WAZ) using the WHO growth standards.

    Args:
        waz (pd.Series): WAZ scores.
        age (pd.Series): Age measurements.
        sex (pd.Series): Sex (gender) of the children.
        waz_params (pd.DataFrame): WAZ parameters from WHO growth standards.

    Returns:
        pd.Series: Weight measurements.
    """
    # Make sure waz, age, and sex are the same length, else throw an error
    if not (len(waz) == len(age) == len(sex)):
        raise ValueError("Input series must have the same length")
    
    # Get M, S, L values for the age and sex
    m, s, l = get_msl(age, sex, waz_params)
    weight = invert_anthro_zscore(waz, m, s, l)
    return weight

def weight_from_whz(whz, height, sex, loh, whz_params_lying, whz_params_standing):
    """
    Calculate weight from Weight-for-Height Z-scores (WHZ) using the WHO growth standards.

    Args:
        whz (pd.Series): WHZ scores.
        height (pd.Series): Height measurements.
        sex (pd.Series): Sex (gender) of the children.
        whz_params (pd.DataFrame): WHZ parameters from WHO growth standards.

    Returns:
        pd.Series: Weight measurements.
    """
    # Make sure whz, height, and sex are the same length, else throw an error
    if not (len(whz) == len(height) == len(sex) == len(loh)):
        raise ValueError("Input series must have the same length")
    
    # Split height, whz and sex into two groups based on loh
    height_0 = height[loh == 1]  # Lying height
    whz_0 = whz[loh == 1]  # WHZ corresponding to lying height
    sex_0 = sex[loh == 1]  # Sex corresponding to lying height

    height_1 = height[loh == 2]  # Standing height
    whz_1 = whz[loh == 2]  # WHZ corresponding to standing height
    sex_1 = sex[loh == 2]  # Sex corresponding to standing height

    # Get M, S, L values for the height and sex for lying height
    # Create a variable called height_adjusted in which each value from height_0 is replaced with the closest value from whz_params_lying['__000002']
    height_array = np.array(whz_params_lying['__000002'])
    height_adjusted = pd.Series(height_array[np.abs(height_array[:, None] - height_0.values).argmin(axis=0)]) 
    m, s, l = get_msl(pd.Series(height_adjusted), sex_0, whz_params_lying, age_label = '__000002')
    weight_0 = invert_anthro_zscore(whz_0, m, s, l)

    # Get M, S, L values for the height and sex for standing height
    # Create a variable called height_adjusted in which each value from height_1 is replaced with the closest value from whz_params_standing['__000003']
    height_array = np.array(whz_params_standing['__000003'])
    height_adjusted = pd.Series(height_array[np.abs(height_array[:, None] - height_1.values).argmin(axis=0)]) 
    m, s, l = get_msl(pd.Series(height_adjusted), sex_1, whz_params_standing, age_label = '__000003')
    weight_1 = invert_anthro_zscore(whz_1, m, s, l)

    # Combine the results
    weight = pd.concat([weight_0, weight_1]).sort_index()
    return weight

def calculate_whz(height, weight, sex, loh, whz_params_lying, whz_params_standing):
    """
    Calculate Weight-for-Height Z-scores (WHZ) using the WHO growth standards.

    Args:
        height (pd.Series): Height measurements.
        weight (pd.Series): Weight measurements.
        age (pd.Series): Age measurements.
        loh (pd.Series): Whether height was measured standing or lying down.
        whz_params_lying (pd.DataFrame): WHZ parameters from WHO growth standards for heights measured lying down.
        whz_params_standing (pd.DataFrame): WHZ parameters from WHO growth standards for heights measured standing up.

    Returns:
        pd.Series: WHZ scores.
    """
    # Make sure height, weight, age, loh and sex are the same length, else throw an error
    if not (len(height) == len(weight) == len(loh) == len(sex)):
        raise ValueError("Input series must have the same length")
    
    # Split height, weight and sex into two groups based on loh
    height_0 = height[loh == 1]  # Lying height
    weight_0 = weight[loh == 1] # Weight corresponding to lying height
    sex_0 = sex[loh == 1] # Sex corresponding to lying height

    height_1 = height[loh == 2]  # Standing height
    weight_1 = weight[loh == 2] # Weight corresponding to standing height
    sex_1 = sex[loh == 2] # Sex corresponding to standing height

    # Get M, S, L values for the age and sex for lying height
    # Create a variable called height_adjusted in which each value from height_0 is replaced with the closest value from whz_params_lying['__000002']
    height_array = np.array(whz_params_lying['__000002'])
    height_adjusted = pd.Series(height_array[np.abs(height_array[:, None] - height_0.values).argmin(axis=0)]) 
    m, s, l = get_msl(pd.Series(height_adjusted), sex_0, whz_params_lying, age_label = '__000002')
    whz_0 = get_anthro_zscore(weight_0.dropna(), m, s, l)

    # Get M, S, L values for the age and sex for standing height
    # Create a variable called height_adjusted in which each value from height_1 is replaced with the closest value from whz_params_standing['__000003']
    height_array = np.array(whz_params_standing['__000003'])
    height_adjusted = pd.Series(height_array[np.abs(height_array[:, None] - height_1.values).argmin(axis=0)]) 
    m, s, l = get_msl(pd.Series(height_adjusted), sex_1, whz_params_standing, age_label = '__000003')
    whz_1 = get_anthro_zscore(weight_1.dropna(), m, s, l)
    
    # Return combined whz values
    whz = pd.concat([whz_0, whz_1]).sort_index()
    return whz

def get_anthro_zscore(y, m, s, l):
    """
    Calculate the anthropometric Z-score using the WHO growth standards.

    Args:
        y (pd.Series): The measurement (e.g., height, weight).
        m (pd.Series): The median value for the reference population.
        s (pd.Series): The standard deviation for the reference population.
        l (pd.Series): The lambda (Box-Cox) transformation parameter.

    Returns:
        pd.Series: The Z-scores.
    """
    z = (np.power(y/m, l) - 1 )/ (s * l)
    return z

def invert_anthro_zscore(z, m, s, l):
    """
    Invert the anthropometric Z-score to obtain the original measurement.

    Args:
        z (pd.Series): The Z-scores.
        m (pd.Series): The median value for the reference population.
        s (pd.Series): The standard deviation for the reference population.
        l (pd.Series): The lambda (Box-Cox) transformation parameter.

    Returns:
        pd.Series: The original measurements.
    """
    y = m * np.power((1 + z * s * l), 1/l)
    return y

def get_msl(age, sex, params, age_label = '_agedays', sex_label = '__000001', verbose=False):

    """
    Get M, S, L values for a specific age and sex from the WHO growth standards.

    Args:
        params (pd.DataFrame): The growth standards parameters.
        age_label (int): The age in days.
        sex_label (int): The sex (1 for male, 2 for female).

    Returns:
        tuple: A tuple containing the M, S, L values.
    """
    n_vals = len(age)
    m, s, l = (np.empty(n_vals), np.empty(n_vals), np.empty(n_vals))

    # Get unique ages from params for mapping
    unique_ages = np.array(params[age_label])

    # Split params by sex
    split_idx = len(unique_ages) // 2

    # Verify that  first half is sex==1, second half is sex==2
    if not (np.all(params[sex_label][:split_idx] == 1) and np.all(params[sex_label][split_idx:] == 2)):
        raise ValueError("Inconsistent sex labels in params")

    # Create lookup dicts for each sex
    lookup_m = {
        1: dict(zip(params[age_label][:split_idx], params['m'][:split_idx])),
        2: dict(zip(params[age_label][split_idx:], params['m'][split_idx:]))
    }
    lookup_s = {
        1: dict(zip(params[age_label][:split_idx], params['s'][:split_idx])),
        2: dict(zip(params[age_label][split_idx:], params['s'][split_idx:]))
    }
    lookup_l = {
        1: dict(zip(params[age_label][:split_idx], params['l'][:split_idx])),
        2: dict(zip(params[age_label][split_idx:], params['l'][split_idx:]))
    }

    if verbose:
        for idx, i in tqdm(enumerate(age.dropna().index)):
            sex_i = np.array(sex)[idx]
            age_i = np.array(age)[idx]
            m[idx] = lookup_m[sex_i].get(age_i, np.nan)
            s[idx] = lookup_s[sex_i].get(age_i, np.nan)
            l[idx] = lookup_l[sex_i].get(age_i, np.nan)
    else:
        for idx, i in enumerate(age.dropna().index):
            sex_i = np.array(sex)[idx]
            age_i = np.array(age)[idx]
            m[idx] = lookup_m[sex_i].get(age_i, np.nan)
            s[idx] = lookup_s[sex_i].get(age_i, np.nan)
            l[idx] = lookup_l[sex_i].get(age_i, np.nan)

    return m, s, l

def calculate_discrepancy_scores(measurements1, measurements2, variable, method, 
                               make_plot=False, plot_title=None,
                               measurements1_name=None, measurements2_name=None,
                               discrepancy_unit=None):
    """Calculate discrepancy scores between two sets of measurements for a given variable.
    
    Args:
        measurements1 (dict): First set of measurements
        measurements2 (dict): Second set of measurements 
        variable (str): Variable to compare ('height', 'weight', 'haz', 'waz' or 'whz')
        method (str): Method for calculating discrepancy:
            - 'percent_difference'
            - 'absolute_difference'
            - 'absolute_percent_difference'
            - 'simple_difference'
            - 'percent_non_match'
            - 'directional_percent_non_match'
            - 'directional_difference'
        make_plot (bool): Whether to create visualization plots
        plot_title (str): Title for the plots
        measurements1_name (str): Label for measurements1 on x-axis
        measurements2_name (str): Label for measurements2 on x-axis
        discrepancy_unit (str): Unit for discrepancy scores (e.g., 'kg' for weight)
    
    Returns:
        numpy.array: Array of discrepancy scores for each measurement
    """
    
    # Check if variable exists in both dictionaries
    if variable not in measurements1 or variable not in measurements2:
        raise KeyError(f"Variable '{variable}' must be present in both measurement dictionaries")
        
    # Get the measurements for the specified variable
    values1 = measurements1[variable]
    values2 = measurements2[variable]
    
    # Calculate individual discrepancy scores
    if method == "percent_difference":
        disc_scores = (values1 - values2) / values2 * 100
    elif method == "absolute_difference":
        disc_scores = np.abs(values1 - values2)
    elif method == "absolute_percent_difference":
        disc_scores = np.abs((values1 - values2) / values2 * 100)
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
        if discrepancy_unit is None:
            discrepancy_label = "Discrepancy"
        else:
            discrepancy_label = f"Discrepancy ({discrepancy_unit})"
            
        # Create figure with 1 row and 3 columns
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(plot_title, fontsize=14)
        
        # Histogram of discrepancy scores
        ax1.hist(disc_scores, bins=50, color='lightgray', edgecolor='black')
        ax1.set_xlabel(discrepancy_label, fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        
        # Scatter plot against measurements1
        ax2.scatter(values1, disc_scores, alpha=0.2, color='turquoise', s=20, linewidth=0)
        ax2.set_xlabel(f"{measurements1_name} {variable}", fontsize=12)
        ax2.set_ylabel('Unit-wise discrepancy', fontsize=12)
        
        # Scatter plot against measurements2
        ax3.scatter(values2, disc_scores, alpha=0.2, color='salmon', s=20, linewidth=0)
        ax3.set_xlabel(f"{measurements2_name} {variable}", fontsize=12)
        ax3.set_ylabel('Unit-wise discrepancy', fontsize=12)
        
        # Adjust layout
        plt.tight_layout()
        plt.show()
    
    return disc_scores

def generate_nested_measurements(
        real_params,
        L0_params_list,  # List of dicts, one per L0
        L1_params_list,  # List of dicts, one per L1
        n_L1s,
        n_L0s_per_L1,
        n_children_per_L0,
        n_children_L1,
        haz_params,
        waz_params,
        whz_params_lying,
        whz_params_standing,
        make_plots=False
    ):
    """Generate nested measurements for multiple L1s and L0s.
    
    Args:
        real_params (dict): Parameters for generating real measurements
        L0_params_list (list): List of parameter dictionaries for L0 distortion, one per L0
        L1_params_list (list): List of parameter dictionaries for L1 distortion, one per L1
        n_L1s (int): Number of L1 units
        n_L0s_per_L1 (int): Number of L0 units per L1
        n_children_per_L0 (int): Number of children per L0
        n_children_L1 (int): Number of children measured by L1 from each L0
        haz_params, waz_params, whz_params_lying, whz_params_standing: WHO parameters
        make_plots (bool): Whether to show diagnostic plots
        
    Returns:
        dict: Nested dictionary containing measurements for each L1 and L0
    """
    
    # Validate length of parameter lists
    if len(L0_params_list) != n_L1s * n_L0s_per_L1:
        raise ValueError(f"L0_params_list must contain {n_L1s * n_L0s_per_L1} parameter sets")
    if len(L1_params_list) != n_L1s:
        raise ValueError(f"L1_params_list must contain {n_L1s} parameter sets")
    
    nested_measurements = {}
    L0_param_idx = 0  # Index to track current L0 parameters
    
    for L1_id in range(n_L1s):
        nested_measurements[f'L1_{L1_id}'] = {}
        L1_measurements_dict = {}  # Store L1 measurements for each L0
        
        for L0_id in range(n_L0s_per_L1):
            # Generate real measurements for this L0
            real_measurements = generate_real_measurements(
                num_children=n_children_per_L0,
                **real_params,
                haz_params=haz_params,
                waz_params=waz_params, 
                whz_params_lying=whz_params_lying,
                whz_params_standing=whz_params_standing,
                plot_distributions=make_plots
            )
            
            # Generate L0 distorted measurements using L0-specific parameters
            L0_measurements = generate_L0_distorted_measurements(
                real_measurements=real_measurements,
                **L0_params_list[L0_param_idx],
                haz_params=haz_params,
                waz_params=waz_params,
                whz_params_lying=whz_params_lying,
                whz_params_standing=whz_params_standing,
                make_plots=make_plots
            )
            L0_param_idx += 1
            
            # Generate L1 measurements for this specific L0
            L1_measurements = generate_L1_distorted_measurements(
                real_measurements=real_measurements,
                L0_distorted_measurements=L0_measurements,
                num_children_L1=n_children_L1,
                **L1_params_list[L1_id],
                haz_params=haz_params,
                waz_params=waz_params,
                whz_params_lying=whz_params_lying,
                whz_params_standing=whz_params_standing,
                make_plots=make_plots
            )
            
            # Store all measurements for this L0
            nested_measurements[f'L1_{L1_id}'][f'L0_{L0_id}'] = {
                'real': real_measurements,
                'L0': L0_measurements,
                'L1': L1_measurements
            }
            
            # Store L1 measurements separately to track which children were measured
            L1_measurements_dict[f'L0_{L0_id}'] = L1_measurements
        
        # Store L1 measurements info at L1 level
        nested_measurements[f'L1_{L1_id}']['L1_info'] = L1_measurements_dict
        
    return nested_measurements

def generate_nested_distortion_parameters(
        n_L1s, 
        n_L0s_per_L1,
        # Real percentages
        real_percent_stunting=None,
        real_percent_underweight=None,
        real_percent_wasting=None,
        # L0 parameter means
        mean_percent_under_reporting_stunting=30,
        mean_percent_under_reporting_underweight=30,
        mean_percent_under_reporting_wasting=None,
        mean_bunch_factor_haz=0.1,
        mean_bunch_factor_waz=0.1,
        mean_bunch_factor_whz=0.1,
        # L0 parameter standard deviations across units
        sd_across_units_percent_under_reporting_stunting=5,
        sd_across_units_percent_under_reporting_underweight=5,
        sd_across_units_percent_under_reporting_wasting=5,
        sd_across_units_bunch_factor_haz=0.02,
        sd_across_units_bunch_factor_waz=0.02,
        sd_across_units_bunch_factor_whz=0.02,
        # L0 parameter standard deviations within units
        sd_within_units_percent_under_reporting_stunting=2,
        sd_within_units_percent_under_reporting_underweight=2,
        sd_within_units_percent_under_reporting_wasting=2,
        sd_within_units_bunch_factor_haz=0.01,
        sd_within_units_bunch_factor_waz=0.01,
        sd_within_units_bunch_factor_whz=0.01,
        # L1 parameters
        mean_percent_copy=10,
        mean_collusion_index=0.5,
        sd_percent_copy=2,
        sd_collusion_index=0.1,
        random_seed=None
    ):
    """Generate nested distortion parameters for L0s and L1s.
    
    Args:
        n_L1s (int): Number of L1 units
        n_L0s_per_L1 (int): Number of L0s per L1 unit
        real_percent_* (float): Actual percentages in real data
        mean_* (float): Mean values for parameters
        sd_across_units_* (float): Standard deviation across units
        sd_within_units_* (float): Standard deviation within units
        random_seed (int): Random seed for reproducibility
        
    Returns:
        tuple: (L0_params_list, L1_params_list) containing parameter dictionaries
    """
    
    if random_seed is not None:
        np.random.seed(random_seed)
        
    # Validate mean under-reporting doesn't exceed real percentages
    if real_percent_stunting is not None and mean_percent_under_reporting_stunting is not None:
        if mean_percent_under_reporting_stunting > real_percent_stunting:
            raise ValueError(f"Mean under-reporting for stunting ({mean_percent_under_reporting_stunting}) cannot exceed real percent stunting ({real_percent_stunting})")
    
    if real_percent_underweight is not None and mean_percent_under_reporting_underweight is not None:
        if mean_percent_under_reporting_underweight > real_percent_underweight:
            raise ValueError(f"Mean under-reporting for underweight ({mean_percent_under_reporting_underweight}) cannot exceed real percent underweight ({real_percent_underweight})")
    
    if real_percent_wasting is not None and mean_percent_under_reporting_wasting is not None:
        if mean_percent_under_reporting_wasting > real_percent_wasting:
            raise ValueError(f"Mean under-reporting for wasting ({mean_percent_under_reporting_wasting}) cannot exceed real percent wasting ({real_percent_wasting})")
            
    L0_params_list = []
    L1_params_list = []
    
    # Generate L1 parameters
    for _ in range(n_L1s):
        L1_params = {
            'percent_copy': max(0, min(100, np.random.normal(mean_percent_copy, sd_percent_copy))),
            'collusion_index': max(0, min(1, np.random.normal(mean_collusion_index, sd_collusion_index)))
        }
        L1_params_list.append(L1_params)
    
    # Generate L0 parameters unit by unit
    for L1_id in range(n_L1s):
        # Generate unit means
        unit_means = {
            'percent_under_reporting_stunting': (
                min(real_percent_stunting, max(0, np.random.normal(mean_percent_under_reporting_stunting, 
                    sd_across_units_percent_under_reporting_stunting)))
                if mean_percent_under_reporting_stunting is not None else None
            ),
            'percent_under_reporting_underweight': (
                min(real_percent_underweight, max(0, np.random.normal(mean_percent_under_reporting_underweight,
                    sd_across_units_percent_under_reporting_underweight)))
                if mean_percent_under_reporting_underweight is not None else None
            ),
            'percent_under_reporting_wasting': (
                min(real_percent_wasting, max(0, np.random.normal(mean_percent_under_reporting_wasting,
                    sd_across_units_percent_under_reporting_wasting)))
                if mean_percent_under_reporting_wasting is not None else None
            ),
            'bunch_factor_haz': np.random.normal(mean_bunch_factor_haz, sd_across_units_bunch_factor_haz),
            'bunch_factor_waz': np.random.normal(mean_bunch_factor_waz, sd_across_units_bunch_factor_waz),
            'bunch_factor_whz': np.random.normal(mean_bunch_factor_whz, sd_across_units_bunch_factor_whz)
        }
        
        # Generate parameters for each L0 in this unit
        for _ in range(n_L0s_per_L1):
            L0_params = {
                'percent_under_reporting_stunting': (
                    min(real_percent_stunting, max(0, np.random.normal(
                        unit_means['percent_under_reporting_stunting'], 
                        sd_within_units_percent_under_reporting_stunting)))
                    if unit_means['percent_under_reporting_stunting'] is not None else None
                ),
                'percent_under_reporting_underweight': (
                    min(real_percent_underweight, max(0, np.random.normal(
                        unit_means['percent_under_reporting_underweight'],
                        sd_within_units_percent_under_reporting_underweight)))
                    if unit_means['percent_under_reporting_underweight'] is not None else None
                ),
                'percent_under_reporting_wasting': (
                    min(real_percent_wasting, max(0, np.random.normal(
                        unit_means['percent_under_reporting_wasting'],
                        sd_within_units_percent_under_reporting_wasting)))
                    if unit_means['percent_under_reporting_wasting'] is not None else None
                ),
                'bunch_factor_haz': max(0, min(1, np.random.normal(unit_means['bunch_factor_haz'],
                                                                  sd_within_units_bunch_factor_haz))),
                'bunch_factor_waz': max(0, min(1, np.random.normal(unit_means['bunch_factor_waz'],
                                                                  sd_within_units_bunch_factor_waz))),
                'bunch_factor_whz': max(0, min(1, np.random.normal(unit_means['bunch_factor_whz'],
                                                                  sd_within_units_bunch_factor_whz)))
            }
            L0_params_list.append(L0_params)
    
    return L0_params_list, L1_params_list

def calculate_ranks_L0_units(nested_measurements, measurement_var, method='simple_difference'):
    """
    Calculate real and measured ranks for units of L0s based on average discrepancy scores.
    
    Args:
        nested_measurements (dict): Nested dictionary containing measurements structured as:
            L1_id -> L0_id -> measurement_type -> measurements
        measurement_var (str): Variable to calculate discrepancy scores for (e.g., 'weight', 'height')
        method (str): Method to calculate discrepancy scores. Default: 'simple_difference'
            
    Returns:
        tuple: (real_ranks, measured_ranks) where each is an array of L1 unit ranks (1-based)
        ordered by the L1 unit IDs
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
            real_discrepancies.append(abs(real_disc.mean()))
            
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
            measured_discrepancies.append(abs(measured_disc.mean()))
        
        # Calculate average discrepancy for this L1 unit
        unit_real_discrepancies[L1_id] = real_discrepancies
        unit_measured_discrepancies[L1_id] = measured_discrepancies

    # Convert to arrays ordered by L1_id
    L1_ids = sorted([L1_id for L1_id in unit_real_discrepancies.keys()])
    mean_real_discrepancies = np.array([np.mean(unit_real_discrepancies[L1_id]) for L1_id in L1_ids])
    mean_measured_discrepancies = np.array([np.mean(unit_measured_discrepancies[L1_id]) for L1_id in L1_ids])

    # Calculate ranks (1-based ranking, ascending order of discrepancy)
    real_ranks = mean_real_discrepancies.argsort().argsort() + 1
    measured_ranks = mean_measured_discrepancies.argsort().argsort() + 1

    return real_ranks, measured_ranks, unit_real_discrepancies, unit_measured_discrepancies

def plot_nested_measurements_and_ranks(nested_measurements, measurement_var, measurement_unit, 
                                     real_discrepancies, measured_discrepancies,
                                     real_ranks, measured_ranks, figsize=(15, 6)):
    """
    Visualize nested measurements and ranks using scatter plots and violin plots.
    
    Args:
        nested_measurements (dict): Nested measurements dictionary
        measurement_var (str): Variable being measured (e.g., 'weight', 'height')
        measurement_unit (str): Unit of measurement (e.g., 'kg', 'cm')
        real_discrepancies (array): Real discrepancy scores for each L1 unit
        measured_discrepancies (array): Measured discrepancy scores for each L1 unit
        real_ranks (array): Array of real ranks for each L1 unit
        measured_ranks (array): Array of measured ranks for each L1 unit
        figsize (tuple): Figure size (width, height)
    """
    # Get number of L1 units
    n_L1s = len(real_ranks)
    
    # Create figure
    fig, axs = plt.subplots(3, n_L1s, figsize=figsize, sharey='row', constrained_layout=True)
    
    # Font sizes
    TITLE_SIZE = 12
    LABEL_SIZE = 12
    TICK_SIZE = 10
    
    # Color scheme for L0s - create a distinct color for each L0 in a unit
    L0_colors = plt.cm.Dark2(np.linspace(0, 1, max(len(nested_measurements[L1_id]) - 1 
                                                   for L1_id in nested_measurements 
                                                   if L1_id != 'metadata')))
    
    # Get order of L1 units based on real ranks
    L1_order = np.argsort(real_ranks)
    
    # Plot for each L1 unit
    for i, L1_idx in enumerate(L1_order):
        L1_id = f'L1_{L1_idx}'
        
        # Get measurement ranges for this unit to set axis limits
        all_measurements = []
        
        # Plot scatter plots for each L0
        for j, L0_id in enumerate(nested_measurements[L1_id]):
            if L0_id == 'L1_info':
                continue
                
            # Get measurements
            real_meas = nested_measurements[L1_id][L0_id]['real']['data'][measurement_var]
            L0_meas = nested_measurements[L1_id][L0_id]['L0']['data'][measurement_var]
            L1_meas = nested_measurements[L1_id][L0_id]['L1']['data'][measurement_var]
            
            all_measurements.extend(real_meas)
            all_measurements.extend(L0_meas)
            all_measurements.extend(L1_meas)
            
            # First row: Real vs L0
            axs[0, i].scatter(real_meas, L0_meas, color=L0_colors[j], alpha=0.5, 
                            label=f'L0_{j}', s=20)
            
            # Second row: L0 vs L1 (only for children measured by L1)
            L1_indices = nested_measurements[L1_id][L0_id]['L1']['data'].index
            axs[1, i].scatter(L1_meas, L0_meas.loc[L1_indices],
                            color=L0_colors[j], alpha=0.5, label=f'L0_{j}', s=20)
        
        # Set axis limits and add X=Y line for scatter plots
        axis_min = min(all_measurements)
        axis_max = max(all_measurements)
        line = np.linspace(axis_min, axis_max, 100)
        
        for row in [0, 1]:
            axs[row, i].plot(line, line, 'k--', alpha=0.5)
            axs[row, i].set_xlim(axis_min, axis_max)
            axs[row, i].set_ylim(axis_min, axis_max)
            if i == 0:
                if row == 0:
                    axs[row, i].set_ylabel(f'L0 {measurement_var} ({measurement_unit})', fontsize=LABEL_SIZE)
                else:
                    axs[row, i].set_ylabel(f'L0 {measurement_var} ({measurement_unit})', fontsize=LABEL_SIZE)
            axs[row, i].tick_params(axis='both', labelsize=TICK_SIZE)
        
        # Set x-labels
        axs[0, i].set_xlabel(f'Real {measurement_var} ({measurement_unit})', fontsize=LABEL_SIZE)
        axs[1, i].set_xlabel(f'L1 {measurement_var} ({measurement_unit})', fontsize=LABEL_SIZE)
        
        # Third row: Violin plots of discrepancies
        parts = axs[2, i].violinplot([real_discrepancies[L1_id], measured_discrepancies[L1_id]],
                                   positions=[1, 2],
                                   showmeans=True)
        
        # Color the violin plots
        colors = {'real': '#2c7bb6', 'measured': '#d7191c'}
        for pc, color in zip(parts['bodies'], colors.values()):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        parts['cmeans'].set_color('black')
        
        # Set labels for violin plots
        axs[2, i].set_xticks([1, 2])
        axs[2, i].set_xticklabels(['Real', 'Measured'], fontsize=TICK_SIZE)
        axs[2, i].tick_params(axis='y', labelsize=TICK_SIZE)
        if i == 0:
            axs[2, i].set_ylabel(f'{measurement_var.capitalize()} discrepancy ({measurement_unit})', fontsize=LABEL_SIZE)
        
        # Add title showing ranks
        axs[0, i].set_title(f'Real rank: {real_ranks[L1_idx]}\nMeasured rank: {measured_ranks[L1_idx]}', 
                           fontsize=TITLE_SIZE)
    
    return fig

def L0_unit_classification_confidence(
    # Real parameters
    real_params,
    n_L1s,
    n_L0s_per_L1,
    n_children_per_L0,
    n_children_L1,
    # WHO parameters
    haz_params,
    waz_params,
    whz_params_lying,
    whz_params_standing,
    # Analysis parameters
    measurement_var,
    measurement_unit,
    n_L1_units_rewarded,
    # Distortion parameters for L0s
    real_percent_stunting=None,
    real_percent_underweight=None,
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
    random_seed=None,
    n_simulations=100,  # Add n_simulations parameter
    make_plots=False
):
    """
    Analyze classification confidence for different parameter combinations.
    
    Args:
        [Original args from generate_nested_measurements except L0_params_list, L1_params_list]
        [All args from generate_nested_distortion_parameters]
        measurement_var (str): Variable to analyze (e.g., 'weight', 'height')
        measurement_unit (str): Unit of measurement (e.g., 'kg', 'cm')
        n_L1_units_rewarded (int): Number of top L1 units to consider
        n_simulations (int): Number of simulations to run for each parameter combination
        make_plots (bool): Whether to show diagnostic plots
    
    Returns:
        tuple: (DataFrame with results, plots)
    """
    # Convert inputs to lists if they're not already
    n_L0s_list = [n_L0s_per_L1] if isinstance(n_L0s_per_L1, int) else n_L0s_per_L1
    n_children_L0_list = [n_children_per_L0] if isinstance(n_children_per_L0, int) else n_children_per_L0
    n_children_L1_list = [n_children_L1] if isinstance(n_children_L1, int) else n_children_L1
    
    # Check if n_L1_units_rewarded is valid
    if n_L1_units_rewarded >= n_L1s:
        raise ValueError(f"n_L1_units_rewarded ({n_L1_units_rewarded}) must be less than n_L1s ({n_L1s})")
    
    # Create empty lists to store results
    results = []
    
    # Iterate over all combinations
    for n_L0s in n_L0s_list:
        for n_children_L0 in n_children_L0_list:
            for n_children_L1 in n_children_L1_list:
                sim_real_ranks = []
                sim_overlaps = []
                warning_count = 0  # Track warnings for this parameter combination

                
                # Run multiple simulations
                for sim in range(n_simulations):
                    # Generate distortion parameters
                    L0_params_list, L1_params_list = generate_nested_distortion_parameters(
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
                        random_seed=random_seed+sim if random_seed else None
                    )
                    
                    # Generate nested measurements
                    nested_measurements = generate_nested_measurements(
                        real_params=real_params,
                        L0_params_list=L0_params_list,
                        L1_params_list=L1_params_list,
                        n_L1s=n_L1s,
                        n_L0s_per_L1=n_L0s,
                        n_children_per_L0=n_children_L0,
                        n_children_L1=n_children_L1,
                        haz_params=haz_params,
                        waz_params=waz_params,
                        whz_params_lying=whz_params_lying,
                        whz_params_standing=whz_params_standing,
                        make_plots=False
                    )
                    
                    # Check for warnings in the measurements
                    warning_found = False
                    for L1_id in nested_measurements:
                        if L1_id == 'metadata':
                            continue

                        # Check each L0 in this L1 unit
                        for L0_id in nested_measurements[L1_id]:
                            if L0_id == 'L1_info':
                                continue

                            # Check the L0 data first
                            if 'bunching_warning_haz' in nested_measurements[L1_id][L0_id]['L0']['metadata']:
                                if nested_measurements[L1_id][L0_id]['L0']['metadata']['bunching_warning_haz']:
                                    warning_found = True
                                    break
                            if 'bunching_warning_waz' in nested_measurements[L1_id][L0_id]['L0']['metadata']:
                                if nested_measurements[L1_id][L0_id]['L0']['metadata']['bunching_warning_waz']:
                                    warning_found = True
                                    break
                            if 'bunching_warning_whz' in nested_measurements[L1_id][L0_id]['L0']['metadata']:
                                if nested_measurements[L1_id][L0_id]['L0']['metadata']['bunching_warning_whz']:
                                    warning_found = True
                                    break
                            # Check the L1 data too
                            if 'bunching_warning_haz' in nested_measurements[L1_id][L0_id]['L1']['metadata']:
                                if nested_measurements[L1_id][L0_id]['L1']['metadata']['bunching_warning_haz']:
                                    warning_found = True
                                    break
                            if 'bunching_warning_waz' in nested_measurements[L1_id][L0_id]['L1']['metadata']:
                                if nested_measurements[L1_id][L0_id]['L1']['metadata']['bunching_warning_waz']:
                                    warning_found = True
                                    break
                            if 'bunching_warning_whz' in nested_measurements[L1_id][L0_id]['L1']['metadata']:
                                if nested_measurements[L1_id][L0_id]['L1']['metadata']['bunching_warning_whz']:
                                    warning_found = True
                                    break
                        if warning_found:
                            break
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
                    print(f"\nWarning: For parameter combination:")
                    print(f"n_L0s_per_L1={n_L0s}, n_children_per_L0={n_children_L0}, n_children_L1={n_children_L1}")
                    print(f"Percent shift was too high in {warning_count} out of {n_simulations} simulations")
                

                # Store results with mean and SEM across simulations
                results.append({
                    'n_L0s_per_L1': n_L0s,
                    'n_children_per_L0': n_children_L0,
                    'n_children_L1': n_children_L1,
                    'real_rank_mean': np.mean(sim_real_ranks),
                    'real_rank_sem': np.std(sim_real_ranks) / np.sqrt(n_simulations),
                    'n_overlap_mean': np.mean(sim_overlaps),
                    'n_overlap_sem': np.std(sim_overlaps) / np.sqrt(n_simulations)
                })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
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
    
    # Colors for different parameter values
    color_map = plt.cm.Dark2
    
    for var in plot_vars:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Find other varying parameters
        other_vars = [v for v in plot_vars if v != var]
        
        if other_vars:
            # Plot lines for each value of the other parameter
            other_var = other_vars[0]
            other_values = sorted(results_df[other_var].unique())
            colors = color_map(np.linspace(0, 1, len(other_values)))
            
            for i, val in enumerate(other_values):
                mask = results_df[other_var] == val
                x = results_df[mask][var]
                
                # Plot real rank
                y = results_df[mask]['real_rank_mean']
                yerr = results_df[mask]['real_rank_sem']
                ax1.errorbar(x, y, yerr=yerr, fmt='o-', color=colors[i], 
                           label=f'{other_var}={val}', linewidth=2, markersize=8)
                
                # Plot overlap
                y = results_df[mask]['n_overlap_mean']
                yerr = results_df[mask]['n_overlap_sem']
                ax2.errorbar(x, y, yerr=yerr, fmt='o-', color=colors[i], 
                           label=f'{other_var}={val}', linewidth=2, markersize=8)
        else:
            # Single line if no other parameters vary
            x = results_df[var]
            
            # Plot real rank
            y = results_df['real_rank_mean']
            yerr = results_df['real_rank_sem']
            ax1.errorbar(x, y, yerr=yerr, fmt='ko-', linewidth=2, markersize=8)
            
            # Plot overlap
            y = results_df['n_overlap_mean']
            yerr = results_df['n_overlap_sem']
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
    
    return results_df, fig_list