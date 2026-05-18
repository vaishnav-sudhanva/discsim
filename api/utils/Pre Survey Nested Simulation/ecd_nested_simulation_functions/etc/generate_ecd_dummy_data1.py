import numpy as np
import pandas as pd
from scipy.stats import norm
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm
from disc_score import discrepancy_score
from . import ecd_anthro_score_calc















# march 1 : 3 method data drift code:





import numpy as np
import pandas as pd

def apply_biological_drift(
    current_heights, current_weights, current_ages, 
    current_haz, current_waz, genders, 
    time_lag_days, method, haz_params, waz_params
):
    """
    Applies biological drift to children over a specified time lag using 3 different mathematical logics.
    Returns the newly drifted heights AND weights to prevent "starvation" bugs.
    """
    if time_lag_days <= 0:
        return current_heights.copy(), current_weights.copy()
        
    aged_days = current_ages + time_lag_days
    
    # =========================================================================
    # LOGIC 1: Exact Z-Score Tracking (The Scientific Standard)
    # =========================================================================
    if method == 1:
        new_heights = ecd_anthro_score_calc.height_from_haz(current_haz, aged_days, genders, haz_params)
        new_weights = ecd_anthro_score_calc.weight_from_waz(current_waz, aged_days, genders, waz_params)
        
        return new_heights, new_weights
    # =========================================================================
    # LOGIC 2: Midpoint Binning (The Quantization Trap)
    # =========================================================================
    elif method == 2:
        bins = np.arange(-6.0, 6.5, 0.5)
        
        # HAZ Binning (with np.clip to prevent Out-of-Bounds index -1 crashes)
        haz_indices = np.clip(np.digitize(current_haz, bins) - 1, 0, len(bins) - 2)
        haz_midpoints = bins[haz_indices] + 0.25 
        base_mid_h = ecd_anthro_score_calc.height_from_haz(haz_midpoints, current_ages, genders, haz_params)
        aged_mid_h = ecd_anthro_score_calc.height_from_haz(haz_midpoints, aged_days, genders, haz_params)
        new_heights = current_heights + (aged_mid_h - base_mid_h)
        
        # WAZ Binning
        waz_indices = np.clip(np.digitize(current_waz, bins) - 1, 0, len(bins) - 2)
        waz_midpoints = bins[waz_indices] + 0.25
        base_mid_w = ecd_anthro_score_calc.weight_from_waz(waz_midpoints, current_ages, genders, waz_params)
        aged_mid_w = ecd_anthro_score_calc.weight_from_waz(waz_midpoints, aged_days, genders, waz_params)
        new_weights = current_weights + (aged_mid_w - base_mid_w)

    # =========================================================================
    # LOGIC 3: Interval Random Bounded Growth (The Noise Injector)
    # =========================================================================
    elif method == 3:
        saved_rng_state = np.random.get_state()
        bins = np.arange(-6.0, 6.5, 0.5)
        
        # HAZ Bounds
        haz_indices = np.clip(np.digitize(current_haz, bins) - 1, 0, len(bins) - 2)
        
        # --- FIXED TYPO HERE: Changed 'for' to 'from' ---
        h_growth_lower = ecd_anthro_score_calc.height_from_haz(bins[haz_indices], aged_days, genders, haz_params) - \
                         ecd_anthro_score_calc.height_from_haz(bins[haz_indices], current_ages, genders, haz_params)
        
        h_growth_upper = ecd_anthro_score_calc.height_from_haz(bins[haz_indices] + 0.5, aged_days, genders, haz_params) - \
                         ecd_anthro_score_calc.height_from_haz(bins[haz_indices] + 0.5, current_ages, genders, haz_params)
        
        h_low = np.nan_to_num(np.minimum(h_growth_lower, h_growth_upper), nan=0.0, posinf=10.0, neginf=0.0)
        h_high = np.nan_to_num(np.maximum(h_growth_lower, h_growth_upper), nan=0.1, posinf=10.0, neginf=0.0)
        new_heights = current_heights + np.random.uniform(low=h_low, high=h_high)
        
        # WAZ Bounds
        waz_indices = np.clip(np.digitize(current_waz, bins) - 1, 0, len(bins) - 2)
        w_growth_lower = ecd_anthro_score_calc.weight_from_waz(bins[waz_indices], aged_days, genders, waz_params) - \
                         ecd_anthro_score_calc.weight_from_waz(bins[waz_indices], current_ages, genders, waz_params)
        w_growth_upper = ecd_anthro_score_calc.weight_from_waz(bins[waz_indices] + 0.5, aged_days, genders, waz_params) - \
                         ecd_anthro_score_calc.weight_from_waz(bins[waz_indices] + 0.5, current_ages, genders, waz_params)
        
        w_low = np.nan_to_num(np.minimum(w_growth_lower, w_growth_upper), nan=0.0, posinf=5.0, neginf=0.0)
        w_high = np.nan_to_num(np.maximum(w_growth_lower, w_growth_upper), nan=0.1, posinf=5.0, neginf=0.0)
        new_weights = current_weights + np.random.uniform(low=w_low, high=w_high)

        np.random.set_state(saved_rng_state)

    else:
        raise ValueError("drift_method must be 1, 2, or 3.")
        
    return new_heights, new_weights







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
    heights_0 = ecd_anthro_score_calc.height_from_haz(haz_0, start_ages, genders, haz_params)
    weights_0 = ecd_anthro_score_calc.weight_from_waz(waz_0, start_ages, genders, waz_params)

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
    whz_0 = ecd_anthro_score_calc.calculate_whz(pd.Series(heights_0), pd.Series(weights_0), pd.Series(genders), loh, whz_params_lying, whz_params_standing)
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



# VS code change - Start 






    # Check that percent under reporting are provided either for stunting and underweight or wasting, and if not throw an error.
    # if (percent_under_reporting_stunting is None or percent_under_reporting_underweight is None) and percent_under_reporting_wasting is None:
    #     raise ValueError("Percent under-reporting must be provided for either stunting and underewight, or wasting.")

    # distorted_measurements = {
    #     'data': real_measurements['data'].copy(deep=True),
    #     'metadata': real_measurements['metadata'].copy()
    # }
    # distorted_measurements['metadata']['percent_under_reporting_stunting'] = percent_under_reporting_stunting
    # distorted_measurements['metadata']['percent_under_reporting_underweight'] = percent_under_reporting_underweight
    # distorted_measurements['metadata']['percent_under_reporting_wasting'] = percent_under_reporting_wasting

    # # Apply under-reporting for stunting and underweight if provided
    # if percent_under_reporting_stunting is not None and percent_under_reporting_underweight is not None:
        
    #     # Throw an error if percent under-reporting for stunting or underweight exceeds the actual percent stunting or underweight
    #     if percent_under_reporting_stunting > real_measurements['metadata']['percent_stunting']:
    #         percent_under_reporting_stunting = real_measurements['metadata']['percent_stunting']
    #     if percent_under_reporting_underweight > real_measurements['metadata']['percent_underweight']:
    #         percent_under_reporting_underweight = real_measurements['metadata']['percent_underweight']

    #     # Under-reporting for stunting
    #     distorted_measurements['data']['haz'], distorted_measurements['metadata']['bunching_warning_haz'] = generate_bunched_data(threshold=reporting_threshold,
    #                                                            original_data=real_measurements['data']['haz'],
    #                                                            percent_below_threshold_original=real_measurements['metadata']['percent_stunting'],
    #                                                            percent_below_threshold_bunched=real_measurements['metadata']['percent_stunting'] - percent_under_reporting_stunting,
    #                                                            bunch_factor=bunch_factor_haz, bin_size=bin_size)
    #     # Under-reporting for underweight
    #     distorted_measurements['data']['waz'], distorted_measurements['metadata']['bunching_warning_waz'] = generate_bunched_data(threshold=reporting_threshold,
    #                                                            original_data=distorted_measurements['data']['waz'],
    #                                                            percent_below_threshold_original=real_measurements['metadata']['percent_underweight'],
    #                                                            percent_below_threshold_bunched=real_measurements['metadata']['percent_underweight'] - percent_under_reporting_underweight,
    #                                                            bunch_factor=bunch_factor_waz, bin_size=bin_size)

    #     # Calculate distorted height and weight based on distorted HAZ and WAZ
    #     distorted_measurements['data']['height'] = ecd_anthro_score_calc.height_from_haz(distorted_measurements['data']['haz'], real_measurements['data']['age'],
    #                                                        real_measurements['data']['gender'], haz_params)
    #     distorted_measurements['data']['weight'] = ecd_anthro_score_calc.weight_from_waz(distorted_measurements['data']['waz'], real_measurements['data']['age'],
    #                                                        real_measurements['data']['gender'], waz_params)

    #     # Calculate distorted WHZ from distorted height and weight
    #     distorted_measurements['data']['whz'] = ecd_anthro_score_calc.calculate_whz(distorted_measurements['data']['height'], distorted_measurements['data']['weight'],
    #                                                   distorted_measurements['data']['gender'], real_measurements['data']['loh'],
    #                                                   whz_params_lying, whz_params_standing)
            
    # # Apply under-reporting for wasting if provided
    # elif percent_under_reporting_wasting is not None:
    #     # Throw an error if percent under-reporting for wasting exceeds the actual percent wasting
    #     if percent_under_reporting_wasting > real_measurements['metadata']['percent_wasting']:
    #         raise ValueError("Percent under-reporting for wasting exceeds actual percent wasting.")
        
    #     # Under-reporting for wasting
    #     distorted_measurements['data']['whz'], distorted_measurements['metadata']['bunching_warning_whz'] = generate_bunched_data(threshold=reporting_threshold,
    #                                                            original_data=distorted_measurements['data']['whz'],
    #                                                            percent_below_threshold_original=real_measurements['metadata']['percent_wasting'],
    #                                                            percent_below_threshold_bunched=real_measurements['metadata']['percent_wasting'] - percent_under_reporting_wasting,
    #                                                            bunch_factor=bunch_factor_whz, bin_size=bin_size)

    #     # Calculate distorted weight from distorted WHZ, assuming height remains un-distorted
    #     distorted_measurements['data']['weight'] = ecd_anthro_score_calc.weight_from_whz(distorted_measurements['data']['whz'], distorted_measurements['data']['height'],
    #                                                        real_measurements['data']['gender'], real_measurements['data']['loh'],whz_params_lying, whz_params_standing)

    #     # Calculate distorted WAZ from distorted weight
    #     distorted_measurements['data']['waz'] = ecd_anthro_score_calc.calculate_waz(distorted_measurements['data']['weight'], real_measurements['data']['age'],
    #                                                    real_measurements['data']['gender'], waz_params)
        
    # # Add measurement error to height and weight
    # distorted_measurements['data']['height'] = add_measurement_error(distorted_measurements['data']['height'], 
    #                                                                  error_mean=error_mean_height, error_sd=error_sd_height)
    # distorted_measurements['data']['weight'] = add_measurement_error(distorted_measurements['data']['weight'], 
    #                                                                  error_mean=error_mean_weight, error_sd=error_sd_weight)

    # # Re-calculate HAZ, WAZ and WHZ from distorted height and weight with measurement error
    # distorted_measurements['data']['haz'] = ecd_anthro_score_calc.calculate_haz(distorted_measurements['data']['height'], real_measurements['data']['age'],
    #                                              real_measurements['data']['gender'], haz_params)
    # distorted_measurements['data']['waz'] = ecd_anthro_score_calc.calculate_waz(distorted_measurements['data']['weight'], real_measurements['data']['age'],
    #                                              real_measurements['data']['gender'], waz_params)
    # distorted_measurements['data']['whz'] = ecd_anthro_score_calc.calculate_whz(distorted_measurements['data']['height'], distorted_measurements['data']['weight'],
    #                                              distorted_measurements['data']['gender'], distorted_measurements['data']['loh'],
    #                                              whz_params_lying, whz_params_standing)


# # 1. Validation: Ensure at least one under-reporting parameter is provided
#     if all(p is None for p in [percent_under_reporting_stunting, percent_under_reporting_underweight, percent_under_reporting_wasting]):
#         raise ValueError("At least one percent under-reporting parameter must be provided.")

    distorted_measurements = {
        'data': real_measurements['data'].copy(deep=True),
        'metadata': real_measurements['metadata'].copy()
    }
    distorted_measurements['metadata']['percent_under_reporting_stunting'] = percent_under_reporting_stunting
    distorted_measurements['metadata']['percent_under_reporting_underweight'] = percent_under_reporting_underweight
    distorted_measurements['metadata']['percent_under_reporting_wasting'] = percent_under_reporting_wasting

    # ==============================================================================
    # STEP 1: APPLY MEASUREMENT ERROR (Honest Mistakes happen first)
    # ==============================================================================
    distorted_measurements['data']['height'] = add_measurement_error(
        distorted_measurements['data']['height'], error_mean=error_mean_height, error_sd=error_sd_height
    )
    distorted_measurements['data']['weight'] = add_measurement_error(
        distorted_measurements['data']['weight'], error_mean=error_mean_weight, error_sd=error_sd_weight
    )

    # Recalculate Z-scores based on these NOISY measurements
    distorted_measurements['data']['haz'] = ecd_anthro_score_calc.calculate_haz(
        distorted_measurements['data']['height'], real_measurements['data']['age'], real_measurements['data']['gender'], haz_params
    )
    distorted_measurements['data']['waz'] = ecd_anthro_score_calc.calculate_waz(
        distorted_measurements['data']['weight'], real_measurements['data']['age'], real_measurements['data']['gender'], waz_params
    )
    distorted_measurements['data']['whz'] = ecd_anthro_score_calc.calculate_whz(
        distorted_measurements['data']['height'], distorted_measurements['data']['weight'], 
        real_measurements['data']['gender'], real_measurements['data']['loh'], whz_params_lying, whz_params_standing
    )

    # Update metadata actual percentages based on noisy data before bunching
    noisy_stunting_rate = (distorted_measurements['data']['haz'] < -2).mean() * 100
    noisy_underweight_rate = (distorted_measurements['data']['waz'] < -2).mean() * 100
    noisy_wasting_rate = (distorted_measurements['data']['whz'] < -2).mean() * 100

    # ==============================================================================
    # STEP 2: APPLY BUNCHING/COLLUSION (Fixing the failing noisy scores)
    # ==============================================================================
    
    # A. Stunting (HAZ)
    if percent_under_reporting_stunting is not None:
        target_bunching = min(percent_under_reporting_stunting, noisy_stunting_rate)
        distorted_measurements['data']['haz'], distorted_measurements['metadata']['bunching_warning_haz'] = generate_bunched_data(
            threshold=reporting_threshold, original_data=distorted_measurements['data']['haz'],
            percent_below_threshold_original=noisy_stunting_rate,
            percent_below_threshold_bunched=noisy_stunting_rate - target_bunching,
            bunch_factor=bunch_factor_haz, bin_size=bin_size
        )
        distorted_measurements['data']['height'] = ecd_anthro_score_calc.height_from_haz(
            distorted_measurements['data']['haz'], real_measurements['data']['age'], real_measurements['data']['gender'], haz_params
        )

    # B. Wasting (WHZ)
    if percent_under_reporting_wasting is not None:
        target_bunching = min(percent_under_reporting_wasting, noisy_wasting_rate)
        distorted_measurements['data']['whz'], distorted_measurements['metadata']['bunching_warning_whz'] = generate_bunched_data(
            threshold=reporting_threshold, original_data=distorted_measurements['data']['whz'],
            percent_below_threshold_original=noisy_wasting_rate,
            percent_below_threshold_bunched=noisy_wasting_rate - target_bunching,
            bunch_factor=bunch_factor_whz, bin_size=bin_size
        )
        distorted_measurements['data']['weight'] = ecd_anthro_score_calc.weight_from_whz(
            distorted_measurements['data']['whz'], distorted_measurements['data']['height'], 
            real_measurements['data']['gender'], real_measurements['data']['loh'], whz_params_lying, whz_params_standing
        )
        # Update WAZ to reflect fake weight
        distorted_measurements['data']['waz'] = ecd_anthro_score_calc.calculate_waz(
            distorted_measurements['data']['weight'], real_measurements['data']['age'], real_measurements['data']['gender'], waz_params
        )

    # C. Underweight (WAZ)
    elif percent_under_reporting_underweight is not None:
        target_bunching = min(percent_under_reporting_underweight, noisy_underweight_rate)
        distorted_measurements['data']['waz'], distorted_measurements['metadata']['bunching_warning_waz'] = generate_bunched_data(
            threshold=reporting_threshold, original_data=distorted_measurements['data']['waz'],
            percent_below_threshold_original=noisy_underweight_rate,
            percent_below_threshold_bunched=noisy_underweight_rate - target_bunching,
            bunch_factor=bunch_factor_waz, bin_size=bin_size
        )
        distorted_measurements['data']['weight'] = ecd_anthro_score_calc.weight_from_waz(
            distorted_measurements['data']['waz'], real_measurements['data']['age'], real_measurements['data']['gender'], waz_params
        )
        # Update WHZ to reflect fake weight
        distorted_measurements['data']['whz'] = ecd_anthro_score_calc.calculate_whz(
            distorted_measurements['data']['height'], distorted_measurements['data']['weight'], 
            real_measurements['data']['gender'], real_measurements['data']['loh'], whz_params_lying, whz_params_standing
        )



# VS code change - END



























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

        plt.tight_layout()
        plt.show()

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
        figsize = [15, 8],
        drift_method=1,
        l1_time_lag_days=0 # added by VS
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
        l1_time_lag_days (int): Time lag in days between L0 and L1 measurements, used to adjust ages for recalculating anthropometric scores.
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
    



#     # code added by VS on data drift (start)
#     # ==============================================================================
#     # APPLY BIOLOGICAL DRIFT (Update Real Data to T1)
#     # ==============================================================================
#     # ==============================================================================
#     # APPLY BIOLOGICAL DRIFT (Update Real Data to T1)
#     # ==============================================================================
#     # ==============================================================================
#     # APPLY BIOLOGICAL DRIFT (Update Real Data to T1)
#     # ==============================================================================
#     if l1_time_lag_days > 0:
#         # 1. Update Age (Using the existing 'age' column which is in days)
#         # We also create '_agedays' because the WHO function MIGHT look for it later.
#         current_age_days = real_subset_L1['data']['age']
#         future_age_days = current_age_days + l1_time_lag_days
        
#         # 2. Calculate New Height (Bio-Physics)
#         # Note: We pass 'age=future_age_days' explicitly to the function
#         future_height = ecd_anthro_score_calc.height_from_haz(
#             haz=real_subset_L1['data']['haz'], 
#             age=future_age_days, 
#             sex=real_subset_L1['data']['gender'], 
#             haz_params=haz_params
#         )
        
#         # 3. Apply Height Drift
#         real_subset_L1['data']['height'] = future_height
        
#         # 4. Calculate New Weight (Bio-Physics)
#         future_weight = ecd_anthro_score_calc.weight_from_waz(
#             waz=real_subset_L1['data']['waz'], 
#             age=future_age_days, 
#             sex=real_subset_L1['data']['gender'], 
#             waz_params=waz_params
#         )
        
#         # 5. Apply Weight Drift
#         real_subset_L1['data']['weight'] = future_weight

#         # 6. Update Metadata (Keep 'age' updated, and add '_agedays' for safety)
#         real_subset_L1['data']['age'] = future_age_days
#         real_subset_L1['data']['_agedays'] = future_age_days

# # 7. ADD THIS: Recalculate Ground Truth WHZ 
#         real_subset_L1['data']['whz'] = ecd_anthro_score_calc.calculate_whz(
#             real_subset_L1['data']['height'],
#             real_subset_L1['data']['weight'],
#             real_subset_L1['data']['gender'],
#             real_subset_L1['data']['loh'],
#             whz_params_lying, whz_params_standing
#         )

        




#         # VS code on data drift ends



# new dd code - vs start
# ==============================================================================
    # APPLY BIOLOGICAL DRIFT (Update Truth to L1 Visit Day)
    # ==============================================================================
    if l1_time_lag_days > 0:
        drifted_h, drifted_w = apply_biological_drift(
            current_heights=real_subset_L1['data']['height'],
            current_weights=real_subset_L1['data']['weight'],
            current_ages=real_subset_L1['data']['age'],
            current_haz=real_subset_L1['data']['haz'],
            current_waz=real_subset_L1['data']['waz'],
            genders=real_subset_L1['data']['gender'],
            time_lag_days=l1_time_lag_days,
            method=drift_method,
            haz_params=haz_params,
            waz_params=waz_params
        )
        
        # Update the Ground Truth BEFORE Supervisor applies human error/cheating
        real_subset_L1['data']['height'] = drifted_h
        real_subset_L1['data']['weight'] = drifted_w
        real_subset_L1['data']['age'] = real_subset_L1['data']['age'] + l1_time_lag_days
        real_subset_L1['data']['_agedays'] = real_subset_L1['data']['age'] 
        
        # Ground Truth WHZ must be recalculated because the child's shape changed
        real_subset_L1['data']['whz'] = ecd_anthro_score_calc.calculate_whz(
            real_subset_L1['data']['height'], real_subset_L1['data']['weight'],
            real_subset_L1['data']['gender'], real_subset_L1['data']['loh'],
            whz_params_lying, whz_params_standing
        )

    # Base the Supervisor's distorted measurements on the NOW-AGED truth
    distorted_measurements = {
        'data': real_subset_L1['data'].copy(deep=True),
        'metadata': L0_subset_L1['metadata'].copy(),
        'L1_indices': L1_indices
    }
# new dd code vs - end




    # Generate L1 measurements for subset using existing logic
    distorted_measurements = {
        'data': real_subset_L1['data'].copy(deep=True),
        'metadata': L0_subset_L1['metadata'].copy(),
        'L1_indices': L1_indices
    }

 # Removing the below codes: reason: 
 # The Issue: You updated the real_subset_L1 (Ground Truth) to account for time. 
 # However, what if the Supervisor copies data from L0 (distorted_measurements['data'].loc[copy_indices, ...] = L0_subset_L1['data'].loc[copy_indices, ...])? 
 # If the Supervisor copies L0 data, they are copying data from the past. 
 # When you later compare L1 to the updated Real data, the copied data will look like an error, which is correct. 
 # However, make sure that L0_subset_L1 is not also updated with the time lag, or the "copying" simulation will be broken. 
 # (It looks okay in the current code, but it's a fragile dependency).

    # Copying: choose a subset of children for which values are taken exactly from L0. Retain the indices so that no measurement error or collusion is applied to these children.
#    num_children = len(real_subset_L1['data'])
#    num_copy = int(num_children * percent_copy / 100)
#    copy_indices = np.random.choice(real_subset_L1['data'].index, size=num_copy, replace=False)
#    distorted_measurements['data'].loc[copy_indices, 
#                                       ['height', 'weight', 'haz', 'waz', 'whz']] = L0_subset_L1['data'].loc[copy_indices, 
#                                                                                    ['height', 'weight', 'haz', 'waz', 'whz']]

# Code change by VS- START  

# Copying: choose a subset of children for which values are taken exactly from L0. Retain the indices so that no measurement error or collusion is applied to these children.
    num_children = len(real_subset_L1['data'])
    num_copy = int(num_children * percent_copy / 100)
    copy_indices = np.random.choice(real_subset_L1['data'].index, size=num_copy, replace=False)
    
    if num_copy > 0:
        # 1. ONLY copy the physical measurements (Height & Weight) from L0 (the past)
        distorted_measurements['data'].loc[copy_indices, ['height', 'weight']] = \
            L0_subset_L1['data'].loc[copy_indices, ['height', 'weight']]
            
        # 2. Recalculate Z-scores using the copied Height/Weight but the NEW Age (T1).
        # This creates the "Drift Error" because the child aged, but their size didn't change.
        distorted_measurements['data'].loc[copy_indices, 'haz'] = ecd_anthro_score_calc.calculate_haz(
            distorted_measurements['data'].loc[copy_indices, 'height'],
            distorted_measurements['data'].loc[copy_indices, 'age'],
            distorted_measurements['data'].loc[copy_indices, 'gender'],
            haz_params
        )
        distorted_measurements['data'].loc[copy_indices, 'waz'] = ecd_anthro_score_calc.calculate_waz(
            distorted_measurements['data'].loc[copy_indices, 'weight'],
            distorted_measurements['data'].loc[copy_indices, 'age'],
            distorted_measurements['data'].loc[copy_indices, 'gender'],
            waz_params
        )
        distorted_measurements['data'].loc[copy_indices, 'whz'] = ecd_anthro_score_calc.calculate_whz(
            distorted_measurements['data'].loc[copy_indices, 'height'],
            distorted_measurements['data'].loc[copy_indices, 'weight'],
            distorted_measurements['data'].loc[copy_indices, 'gender'],
            distorted_measurements['data'].loc[copy_indices, 'loh'],
            whz_params_lying, whz_params_standing
        )

# Code change by VS- END

    # Collusion: for the remaining children, apply collusion-based distortion
    collude_indices = [i for i in real_subset_L1['data'].index if i not in copy_indices]
    num_collude = len(collude_indices)


# collusion: code change by VS - Start 
# The Concept: In anthropometry, you track three different metrics: Stunting (HAZ), Underweight (WAZ), and Wasting (WHZ). A government or NGO might put pressure on a Supervisor to improve just one of these metrics, or maybe two, or maybe all three.

# What your old code did (The Flaw):
# Look at this specific line of code from your original script:
# if percent_under_reporting_stunting is not None and percent_under_reporting_underweight is not None:

# Because the code uses the word and, it links Stunting and Underweight together like a locked door requiring two keys.

# If you run the simulation and say: "I want to see what happens if Supervisors fake 10% of Stunting data, but leave Underweight alone"...

# The computer checks the rule: "Is there Stunting data? Yes. Is there Underweight data? No."

# The Result: The computer evaluates the if statement as False. It skips the entire collusion block entirely. Absolutely zero data gets faked.

# Furthermore, your old code used an elif (Else-If) for Wasting (WHZ). That meant if the Supervisor faked Stunting and Underweight, the computer would skip Wasting entirely. It was impossible to simulate a world where a Supervisor faked all three metrics at the same time.

# Why this breaks the simulation:
# It creates "silent failures." You as the researcher might set parameters to test Stunting fraud, but because you didn't also set Underweight parameters, the code spits out a perfectly clean dataset. You would look at the results and falsely conclude, "Wow, our audit system easily handles Stunting fraud!" when in reality, the fraud never actually happened in the code.

# The Fix:
# Break the giant if / and / elif block into three completely independent statements:

# if percent_under_reporting_stunting is not None: (Do the Stunting fraud)

# if percent_under_reporting_wasting is not None: (Do the Wasting fraud)

# elif percent_under_reporting_underweight is not None: (Do the Underweight fraud) (Note: We keep WAZ as an elif to WHZ here because adjusting weight for wasting also inherently adjusts WAZ, and we don't want them mathematically fighting each other over the same fake weight value).
# here we are calculating the collusion, then applying measurement error, which should be done beforehand.

    # if num_collude > 0:
    #     # Calculate percentage of children to be under-reported based on collusion index, only if corresponding L0 values exist
    #     percent_under_reporting_stunting = (collusion_index * L0_subset_L1['metadata']['percent_under_reporting_stunting'] 
    #                                       if L0_subset_L1['metadata']['percent_under_reporting_stunting'] is not None else None)
    #     percent_under_reporting_underweight = (collusion_index * L0_subset_L1['metadata']['percent_under_reporting_underweight']
    #                                          if L0_subset_L1['metadata']['percent_under_reporting_underweight'] is not None else None)
    #     percent_under_reporting_wasting = (collusion_index * L0_subset_L1['metadata']['percent_under_reporting_wasting']
    #                                      if L0_subset_L1['metadata']['percent_under_reporting_wasting'] is not None else None)

    #     # Apply under-reporting for stunting and underweight if both are provided
    #     if percent_under_reporting_stunting is not None and percent_under_reporting_underweight is not None:
    #         # Under-reporting for stunting
    #         distorted_measurements['data'].loc[collude_indices, 'haz'], distorted_measurements['metadata']['bunching_warning_haz'] = generate_bunched_data(
    #             threshold=-2,
    #             original_data=real_subset_L1['data'].loc[collude_indices, 'haz'],
    #             percent_below_threshold_original=(real_subset_L1['data'].loc[collude_indices, 'haz'] < -2).mean() * 100,
    #             percent_below_threshold_bunched=(real_subset_L1['data'].loc[collude_indices, 'haz'] < -2).mean() * 100 - percent_under_reporting_stunting,
    #             bunch_factor=bunch_factor_haz,
    #             bin_size=bin_size
    #         )
    #         # Under-reporting for underweight
    #         distorted_measurements['data'].loc[collude_indices, 'waz'],distorted_measurements['metadata']['bunching_warning_waz'] = generate_bunched_data(
    #             threshold=-2,
    #             original_data=real_subset_L1['data'].loc[collude_indices, 'waz'],
    #             percent_below_threshold_original=(real_subset_L1['data'].loc[collude_indices, 'waz'] < -2).mean() * 100,
    #             percent_below_threshold_bunched=(real_subset_L1['data'].loc[collude_indices, 'waz'] < -2).mean() * 100 - percent_under_reporting_underweight,
    #             bunch_factor=bunch_factor_waz,
    #             bin_size=bin_size
    #         )

    #         # Calculate distorted height and weight based on distorted HAZ and WAZ
    #         distorted_measurements['data'].loc[collude_indices, 'height'] = ecd_anthro_score_calc.height_from_haz(
    #             distorted_measurements['data'].loc[collude_indices, 'haz'],
    #             real_subset_L1['data'].loc[collude_indices, 'age'],
    #             real_subset_L1['data'].loc[collude_indices, 'gender'],
    #             haz_params
    #         )
    #         distorted_measurements['data'].loc[collude_indices, 'weight'] = ecd_anthro_score_calc.weight_from_waz(
    #             distorted_measurements['data'].loc[collude_indices, 'waz'],
    #             real_subset_L1['data'].loc[collude_indices, 'age'],
    #             real_subset_L1['data'].loc[collude_indices, 'gender'],
    #             waz_params
    #         )   

    #         # Calculate distorted WHZ from distorted height and weight
    #         distorted_measurements['data'].loc[collude_indices, 'whz'] = ecd_anthro_score_calc.calculate_whz(
    #             distorted_measurements['data'].loc[collude_indices, 'height'],
    #             distorted_measurements['data'].loc[collude_indices, 'weight'],
    #             distorted_measurements['data'].loc[collude_indices, 'gender'],
    #             distorted_measurements['data'].loc[collude_indices, 'loh'],
    #             whz_params_lying, whz_params_standing
    #         )




        # # Apply under-reporting for wasting if provided
        # elif percent_under_reporting_wasting is not None:


    if num_collude > 0:
        # Fetch the target collusion percentages
        percent_under_reporting_stunting = (collusion_index * L0_subset_L1['metadata']['percent_under_reporting_stunting'] 
                                          if L0_subset_L1['metadata']['percent_under_reporting_stunting'] is not None else None)
        percent_under_reporting_underweight = (collusion_index * L0_subset_L1['metadata']['percent_under_reporting_underweight']
                                             if L0_subset_L1['metadata']['percent_under_reporting_underweight'] is not None else None)
        percent_under_reporting_wasting = (collusion_index * L0_subset_L1['metadata']['percent_under_reporting_wasting']
                                         if L0_subset_L1['metadata']['percent_under_reporting_wasting'] is not None else None)

        # ==============================================================================
        # STEP A: APPLY MEASUREMENT ERROR FIRST (Honest Mistakes)
        # ==============================================================================
        distorted_measurements['data'].loc[collude_indices, 'height'] = add_measurement_error(
            distorted_measurements['data'].loc[collude_indices, 'height'], 
            error_mean=error_mean_height, error_sd=error_sd_height
        )
        distorted_measurements['data'].loc[collude_indices, 'weight'] = add_measurement_error(
            distorted_measurements['data'].loc[collude_indices, 'weight'], 
            error_mean=error_mean_weight, error_sd=error_sd_weight
        )

        # Recalculate Z-scores based on these NOISY measurements
        distorted_measurements['data'].loc[collude_indices, 'haz'] = ecd_anthro_score_calc.calculate_haz(
            distorted_measurements['data'].loc[collude_indices, 'height'],
            distorted_measurements['data'].loc[collude_indices, 'age'],
            distorted_measurements['data'].loc[collude_indices, 'gender'],
            haz_params
        )
        distorted_measurements['data'].loc[collude_indices, 'waz'] = ecd_anthro_score_calc.calculate_waz(
            distorted_measurements['data'].loc[collude_indices, 'weight'],
            distorted_measurements['data'].loc[collude_indices, 'age'],
            distorted_measurements['data'].loc[collude_indices, 'gender'],
            waz_params
        )
        distorted_measurements['data'].loc[collude_indices, 'whz'] = ecd_anthro_score_calc.calculate_whz(
            distorted_measurements['data'].loc[collude_indices, 'height'],
            distorted_measurements['data'].loc[collude_indices, 'weight'],
            distorted_measurements['data'].loc[collude_indices, 'gender'],
            distorted_measurements['data'].loc[collude_indices, 'loh'],
            whz_params_lying, whz_params_standing
        )

        # ==============================================================================
        # STEP B: APPLY COLLUSION SECOND (Fudging the failing noisy scores)
        # ==============================================================================
        
        # 1. Stunting (HAZ) Collusion
        if percent_under_reporting_stunting is not None:
            distorted_measurements['data'].loc[collude_indices, 'haz'], distorted_measurements['metadata']['bunching_warning_haz'] = generate_bunched_data(
                threshold=-2,
                original_data=distorted_measurements['data'].loc[collude_indices, 'haz'],
                percent_below_threshold_original=(distorted_measurements['data'].loc[collude_indices, 'haz'] < -2).mean() * 100,
                percent_below_threshold_bunched=(distorted_measurements['data'].loc[collude_indices, 'haz'] < -2).mean() * 100 - percent_under_reporting_stunting,
                bunch_factor=bunch_factor_haz,
                bin_size=bin_size
            )
            # Reverse-engineer the faked Height from the fudged HAZ
            distorted_measurements['data'].loc[collude_indices, 'height'] = ecd_anthro_score_calc.height_from_haz(
                distorted_measurements['data'].loc[collude_indices, 'haz'],
                distorted_measurements['data'].loc[collude_indices, 'age'], 
                distorted_measurements['data'].loc[collude_indices, 'gender'],
                haz_params
            )

        # 2. Wasting (WHZ) Collusion
        if percent_under_reporting_wasting is not None:
            distorted_measurements['data'].loc[collude_indices, 'whz'], distorted_measurements['metadata']['bunching_warning_whz'] = generate_bunched_data(
                threshold=-2,
                original_data=distorted_measurements['data'].loc[collude_indices, 'whz'],
                percent_below_threshold_original=(distorted_measurements['data'].loc[collude_indices, 'whz'] < -2).mean() * 100,
                percent_below_threshold_bunched=(distorted_measurements['data'].loc[collude_indices, 'whz'] < -2).mean() * 100 - percent_under_reporting_wasting,
                bunch_factor=bunch_factor_whz,
                bin_size=bin_size
            )
            # Reverse-engineer the faked Weight from the fudged WHZ (using current height)
            distorted_measurements['data'].loc[collude_indices, 'weight'] = ecd_anthro_score_calc.weight_from_whz(
                distorted_measurements['data'].loc[collude_indices, 'whz'],
                distorted_measurements['data'].loc[collude_indices, 'height'], 
                distorted_measurements['data'].loc[collude_indices, 'gender'],
                distorted_measurements['data'].loc[collude_indices, 'loh'],
                whz_params_lying, whz_params_standing
            )
            # Update WAZ to match the newly faked weight
            distorted_measurements['data'].loc[collude_indices, 'waz'] = ecd_anthro_score_calc.calculate_waz(
                distorted_measurements['data'].loc[collude_indices, 'weight'],
                distorted_measurements['data'].loc[collude_indices, 'age'],
                distorted_measurements['data'].loc[collude_indices, 'gender'],
                waz_params
            )



        # 3. Underweight (WAZ) Collusion
        # We use 'elif' here so if they already faked Weight to fix WHZ, we don't accidentally overwrite it to fix WAZ.
        elif percent_under_reporting_underweight is not None:
            distorted_measurements['data'].loc[collude_indices, 'waz'], distorted_measurements['metadata']['bunching_warning_waz'] = generate_bunched_data(
                threshold=-2,
                original_data=distorted_measurements['data'].loc[collude_indices, 'waz'],
                percent_below_threshold_original=(distorted_measurements['data'].loc[collude_indices, 'waz'] < -2).mean() * 100,
                percent_below_threshold_bunched=(distorted_measurements['data'].loc[collude_indices, 'waz'] < -2).mean() * 100 - percent_under_reporting_underweight,
                bunch_factor=bunch_factor_waz,
                bin_size=bin_size
            )
            # Reverse-engineer the faked Weight from the fudged WAZ
            distorted_measurements['data'].loc[collude_indices, 'weight'] = ecd_anthro_score_calc.weight_from_waz(
                distorted_measurements['data'].loc[collude_indices, 'waz'],
                distorted_measurements['data'].loc[collude_indices, 'age'], 
                distorted_measurements['data'].loc[collude_indices, 'gender'],
                waz_params
            )
            # Update WHZ to match the newly faked weight
            distorted_measurements['data'].loc[collude_indices, 'whz'] = ecd_anthro_score_calc.calculate_whz(
                distorted_measurements['data'].loc[collude_indices, 'height'],
                distorted_measurements['data'].loc[collude_indices, 'weight'],
                distorted_measurements['data'].loc[collude_indices, 'gender'],
                distorted_measurements['data'].loc[collude_indices, 'loh'],
                whz_params_lying, whz_params_standing
            )


# collusion: VS code change - END














        #     # Under-reporting for wasting
        #     distorted_measurements['data'].loc[collude_indices, 'whz'], distorted_measurements['metadata']['bunching_warning_whz'] = generate_bunched_data(
        #         threshold=-2,
        #         original_data=real_subset_L1['data'].loc[collude_indices, 'whz'],
        #         percent_below_threshold_original=(real_subset_L1['data'].loc[collude_indices, 'whz'] < -2).mean() * 100,
        #         percent_below_threshold_bunched=(real_subset_L1['data'].loc[collude_indices, 'whz'] < -2).mean() * 100 - percent_under_reporting_wasting,
        #         bunch_factor=bunch_factor_whz,
        #         bin_size=bin_size
        #     )

        #     # Calculate distorted weight from distorted WHZ, assuming height remains un-distorted
        #     distorted_measurements['data'].loc[collude_indices, 'weight'] = ecd_anthro_score_calc.weight_from_whz(
        #         distorted_measurements['data'].loc[collude_indices, 'whz'],
        #         real_subset_L1['data'].loc[collude_indices, 'height'],
        #         real_subset_L1['data'].loc[collude_indices, 'gender'],
        #         real_subset_L1['data'].loc[collude_indices, 'loh'],
        #         whz_params_lying, whz_params_standing
        #     )

        #     # Calculate distorted WAZ from distorted weight
        #     distorted_measurements['data'].loc[collude_indices, 'waz'] = ecd_anthro_score_calc.calculate_waz(
        #         distorted_measurements['data'].loc[collude_indices, 'weight'],
        #         real_subset_L1['data'].loc[collude_indices, 'age'],
        #         real_subset_L1['data'].loc[collude_indices, 'gender'],
        #         waz_params
        #     )

        # # Add measurement error to height and weight for colluding children
        # distorted_measurements['data'].loc[collude_indices, 'height'] = add_measurement_error(
        #     distorted_measurements['data'].loc[collude_indices, 'height'], 
        #     error_mean=error_mean_height, error_sd=error_sd_height
        # )
        # distorted_measurements['data'].loc[collude_indices, 'weight'] = add_measurement_error(
        #     distorted_measurements['data'].loc[collude_indices, 'weight'], 
        #     error_mean=error_mean_weight, error_sd=error_sd_weight
        # )

        # # Re-calculate HAZ, WAZ and WHZ from distorted height and weight with measurement error for colluding children
        # distorted_measurements['data'].loc[collude_indices, 'haz'] = ecd_anthro_score_calc.calculate_haz(
        #     distorted_measurements['data'].loc[collude_indices, 'height'],
        #     real_subset_L1['data'].loc[collude_indices, 'age'],
        #     real_subset_L1['data'].loc[collude_indices, 'gender'],
        #     haz_params
        # )
        # distorted_measurements['data'].loc[collude_indices, 'waz'] = ecd_anthro_score_calc.calculate_waz(
        #     distorted_measurements['data'].loc[collude_indices, 'weight'],
        #     real_subset_L1['data'].loc[collude_indices, 'age'],
        #     real_subset_L1['data'].loc[collude_indices, 'gender'],
        #     waz_params
        # )
        # distorted_measurements['data'].loc[collude_indices, 'whz'] = ecd_anthro_score_calc.calculate_whz(
        #     distorted_measurements['data'].loc[collude_indices, 'height'],
        #     distorted_measurements['data'].loc[collude_indices, 'weight'],
        #     distorted_measurements['data'].loc[collude_indices, 'gender'],
        #     distorted_measurements['data'].loc[collude_indices, 'loh'],
        #     whz_params_lying, whz_params_standing
        # )

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
        sns.histplot(real_subset_L1['data']['height'], bins=30, kde=False, color=real_color, 
                    label='Real', alpha=alpha_real, edgecolor=real_edge, linewidth=0.2, ax=axs[0, 0])
        sns.histplot(L0_subset_L1['data']['height'], bins=30, kde=False, color=L0_color, 
                    label='L0', alpha=alpha_L0, edgecolor=L0_edge, linewidth=0.2, ax=axs[0, 0])
        sns.histplot(distorted_measurements['data']['height'], bins=30, kde=False, color=L1_color, 
                    label='L1', alpha=alpha_L1, edgecolor=L1_edge, linewidth=0.2, ax=axs[0, 0])
        axs[0, 0].set_xlabel('Height (cm)')
        axs[0, 0].set_title('Height')
        axs[0, 0].legend()

        # Weight
        sns.histplot(real_subset_L1['data']['weight'], bins=30, kde=False, color=real_color, 
                    label='Real', alpha=alpha_real, edgecolor=real_edge, linewidth=0.2, ax=axs[0, 1])
        sns.histplot(L0_subset_L1['data']['weight'], bins=30, kde=False, color=L0_color, 
                    label='L0', alpha=alpha_L0, edgecolor=L0_edge, linewidth=0.2, ax=axs[0, 1])
        sns.histplot(distorted_measurements['data']['weight'], bins=30, kde=False, color=L1_color, 
                    label='L1', alpha=alpha_L1, edgecolor=L1_edge, linewidth=0.2, ax=axs[0, 1])
        axs[0, 1].set_xlabel('Weight (kg)')
        axs[0, 1].set_title('Weight')
        axs[0, 1].legend()

        # HAZ
        min_haz = min(real_subset_L1['data']['haz'].min(), L0_subset_L1['data']['haz'].min(), 
                     distorted_measurements['data']['haz'].min())
        max_haz = max(real_subset_L1['data']['haz'].max(), L0_subset_L1['data']['haz'].max(), 
                     distorted_measurements['data']['haz'].max())
        bins_haz = np.arange(min_haz, max_haz, bin_size)
        sns.histplot(real_subset_L1['data']['haz'], bins=bins_haz, kde=False, color=real_color, 
                    label='Real', alpha=alpha_real, edgecolor=real_edge, linewidth=0.2, ax=axs[0, 2])
        sns.histplot(L0_subset_L1['data']['haz'], bins=bins_haz, kde=False, color=L0_color, 
                    label='L0', alpha=alpha_L0, edgecolor=L0_edge, linewidth=0.2, ax=axs[0, 2])
        sns.histplot(distorted_measurements['data']['haz'], bins=bins_haz, kde=False, color=L1_color, 
                    label='L1', alpha=alpha_L1, edgecolor=L1_edge, linewidth=0.2, ax=axs[0, 2])
        axs[0, 2].axvline(x=reporting_threshold, color='red', linestyle='--')
        axs[0, 2].set_xlabel('HAZ')
        axs[0, 2].set_title('HAZ')
        axs[0, 2].legend()

        # WAZ
        min_waz = min(real_subset_L1['data']['waz'].min(), L0_subset_L1['data']['waz'].min(), 
                     distorted_measurements['data']['waz'].min())
        max_waz = max(real_subset_L1['data']['waz'].max(), L0_subset_L1['data']['waz'].max(), 
                     distorted_measurements['data']['waz'].max())
        bins_waz = np.arange(min_waz, max_waz, bin_size)
        sns.histplot(real_subset_L1['data']['waz'], bins=bins_waz, kde=False, color=real_color, 
                    label='Real', alpha=alpha_real, edgecolor=real_edge, linewidth=0.2, ax=axs[0, 3])
        sns.histplot(L0_subset_L1['data']['waz'], bins=bins_waz, kde=False, color=L0_color, 
                    label='L0', alpha=alpha_L0, edgecolor=L0_edge, linewidth=0.2, ax=axs[0, 3])
        sns.histplot(distorted_measurements['data']['waz'], bins=bins_waz, kde=False, color=L1_color, 
                    label='L1', alpha=alpha_L1, edgecolor=L1_edge, linewidth=0.2, ax=axs[0, 3])
        axs[0, 3].axvline(x=reporting_threshold, color='red', linestyle='--')
        axs[0, 3].set_xlabel('WAZ')
        axs[0, 3].set_title('WAZ')
        axs[0, 3].legend()

        # WHZ
        min_whz = min(real_subset_L1['data']['whz'].min(), L0_subset_L1['data']['whz'].min(), 
                     distorted_measurements['data']['whz'].min())
        max_whz = max(real_subset_L1['data']['whz'].max(), L0_subset_L1['data']['whz'].max(), 
                     distorted_measurements['data']['whz'].max())
        bins_whz = np.arange(min_whz, max_whz, bin_size)
        sns.histplot(real_subset_L1['data']['whz'], bins=bins_whz, kde=False, color=real_color, 
                     label='Real', alpha=alpha_real, edgecolor=real_edge, linewidth=0.2, ax=axs[0, 4])
        sns.histplot(L0_subset_L1['data']['whz'], bins=bins_whz, kde=False, color=L0_color, 
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
        axs[1, 0].scatter(x=real_subset_L1['data']['height'],
                          y=distorted_measurements['data']['height'] - real_subset_L1['data']['height'],
                        color='k', marker = '.', alpha = 0.1)
        axs[1, 0].set_xlabel('Real height (cm)')
        axs[1, 0].set_ylabel('Distorted - Real height (cm)')
        axs[1, 0].set_title('Height Diff')

        # Weight difference scatter plot
        axs[1, 1].sharex(axs[0, 1])
        axs[1, 1].scatter(x=real_subset_L1['data']['weight'],
                          y=distorted_measurements['data']['weight'] - real_subset_L1['data']['weight'],
                          color='k', marker='.', alpha=0.1)
        axs[1, 1].set_xlabel('Real weight (kg)')
        axs[1, 1].set_ylabel('Distorted - Real weight (kg)')
        axs[1, 1].set_title('Weight Diff')

        # HAZ difference histogram
        axs[1, 2].sharex(axs[0, 2])
        freq_real, _ = np.histogram(real_subset_L1['data']['haz'], bins=bins_haz, density=False)
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
        freq_real, _ = np.histogram(real_subset_L1['data']['waz'], bins=bins_waz, density=False)
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

# def generate_L2_distorted_measurements(
#     real_measurements,
#     L0_measurements,
#     L1_measurements,
#     haz_params,
#     waz_params,
#     whz_params_lying,
#     whz_params_standing,
#     num_children_L2,
#     error_mean_height=0,
#     error_sd_height=1,
#     error_mean_weight=0,
#     error_sd_weight=0.1,
#     drift_mean_height=0,
#     drift_sd_height=0.1,
#     drift_mean_weight=0,
#     drift_sd_weight=0.05,
#     reporting_threshold=-2,
#     figsize=[15, 8],
#     make_plots=False,
#     l2_time_lag_days=0 # <--- Added by VS - data drift variable input
# ):
#     """
#     Generate L2 distorted measurements by adding measurement error and data drift to real measurements.

#     Args:
#         real_measurements (dict): Real measurements dictionary.
#         L0_measurements (dict): L0 measurements dictionary (for plotting).
#         L1_measurements (dict): L1 measurements dictionary (must include 'L1_indices').
#         haz_params, waz_params, whz_params_lying, whz_params_standing: WHO parameters.
#         num_children_L2 (int): Number of children to be measured by L2.
#         error_mean_height, error_sd_height, error_mean_weight, error_sd_weight: Measurement error parameters.
#         drift_mean_height, drift_sd_height, drift_mean_weight, drift_sd_weight: Data drift parameters.
#         reporting_threshold (float): Z-score threshold for reporting.
#         figsize (list): Figure size for plots.
#         make_plots (bool): Whether to plot distributions.

#     Returns:
#         dict: Distorted L2 measurements dictionary.
#     """
 

# #     L1_indices = L1_measurements['L1_indices']
# #     if num_children_L2 > len(L1_indices):
# #         raise ValueError("num_children_L2 exceeds the number of children measured by L1.")

# # #     L2_indices = np.random.choice(L1_indices, size=num_children_L2, replace=False)
# #     distorted_measurements = {
# #         'data': real_measurements['data'].loc[L2_indices].copy(),
# #         'metadata': real_measurements['metadata'].copy(),
# #         'L2_indices': L2_indices
# #     }



# # # Data drift variable added by Vs - Start

# # # ==============================================================================

# # # ==============================================================================
# #   # ==============================================================================
# #     # APPLY BIOLOGICAL DRIFT (Update Truth to L2 Visit Day)
# #     # ==============================================================================


# # # ==============================================================================
# #     # APPLY BIOLOGICAL DRIFT (Update Truth to L2 Visit Day)
# #     # ==============================================================================
# #     if l2_time_lag_days > 0:
# #         # 1. Update Age (Using the existing 'age' column)
# #         current_age_days = distorted_measurements['data']['age']
# #         future_age_days = current_age_days + l2_time_lag_days
        
# #         # 2. Calculate New Height
# #         future_height = ecd_anthro_score_calc.height_from_haz(
# #             haz=distorted_measurements['data']['haz'], 
# #             age=future_age_days, 
# #             sex=distorted_measurements['data']['gender'], 
# #             haz_params=haz_params
# #         )
        
# #         # 3. Apply Height Drift
# #         distorted_measurements['data']['height'] = future_height
        
# #         # 4. Calculate New Weight
# #         future_weight = ecd_anthro_score_calc.weight_from_waz(
# #             waz=distorted_measurements['data']['waz'], 
# #             age=future_age_days, 
# #             sex=distorted_measurements['data']['gender'], 
# #             waz_params=waz_params
# #         )
        
# #         # 5. Apply Weight Drift
# #         distorted_measurements['data']['weight'] = future_weight

# #         # 6. Update Metadata
# #         distorted_measurements['data']['age'] = future_age_days
# #         distorted_measurements['data']['_agedays'] = future_age_days

# # # f a child's HAZ and WAZ remain exactly the same over 1 month, their WHZ will not remain exactly the same. 
# # # Assuming WHZ stays constant is a mathematical trap. Here is exactly why they are related, 
# # # but not perfectly linked, and why your simulation code needs to handle WHZ differently.

# # #How your code currently works (The Flaw):Code locks HAZ to what it was at Month 0.
# # # Code calculates new physical Height ($H_{t1}$) based on $Age_{t1}$.
# # # Code locks WAZ to what it was at Month 0.
# # # Code calculates new physical Weight ($W_{t1}$) based on $Age_{t1}$.
# # # Code does nothing to WHZ. 
# # # The dataset now contains $H_{t1}$, $W_{t1}$, but $WHZ_{t0}$.
# # # If an auditor later runs a standard anthropometry script over your simulated data, 
# # # they will flag your "Ground Truth" data as physically impossible, 
# # # because the stated WHZ will not match the stated Height and Weight!

# #         # 7. Calculate New WHZ (Bio-Physics)
# #         distorted_measurements['data']['whz'] = ecd_anthro_score_calc.calculate_whz(
# #             distorted_measurements['data']['height'],
# #             distorted_measurements['data']['weight'],
# #             distorted_measurements['data']['gender'],
# #             distorted_measurements['data']['loh'],
# #             whz_params_lying, 
# #             whz_params_standing
# #         )





# # # Data drift variable added by VS - End







# #     # Add measurement error to height and weight
# #     distorted_measurements['data']['height'] = add_measurement_error(
# #         distorted_measurements['data']['height'], error_mean=error_mean_height, error_sd=error_sd_height
# #     )
# #     distorted_measurements['data']['weight'] = add_measurement_error(
# #         distorted_measurements['data']['weight'], error_mean=error_mean_weight, error_sd=error_sd_weight
# #     )



#     L1_indices = L1_measurements['L1_indices']
#     if num_children_L2 > len(L1_indices):
#         raise ValueError("num_children_L2 exceeds the number of children measured by L1.")

#     L2_indices = np.random.choice(L1_indices, size=num_children_L2, replace=False)
    
#     # -------------------------------------------------------------------------
#     # NEW FIX: 1. Extract the baseline truth into a separate 'Real' variable first!
#     # -------------------------------------------------------------------------
#     real_subset_L2 = {
#         'data': real_measurements['data'].loc[L2_indices].copy(deep=True),
#         'metadata': real_measurements['metadata'].copy()
#     }

#     # ==============================================================================
#     # APPLY BIOLOGICAL DRIFT (Update Truth to L2 Visit Day)
#     # ==============================================================================
#     if l2_time_lag_days > 0:
#         # 1. Update Age
#         current_age_days = real_subset_L2['data']['age']
#         future_age_days = current_age_days + l2_time_lag_days
        
#         # 2. Calculate New Height (Bio-Physics based on maintaining HAZ percentile)
#         future_height = ecd_anthro_score_calc.height_from_haz(
#             haz=real_subset_L2['data']['haz'], 
#             age=future_age_days, 
#             sex=real_subset_L2['data']['gender'], 
#             haz_params=haz_params
#         )
        
#         # 3. Apply Height Drift ONLY to the 'Real' dataset
#         real_subset_L2['data']['height'] = future_height
        
#         # 4. Calculate New Weight (Bio-Physics based on maintaining WAZ percentile)
#         future_weight = ecd_anthro_score_calc.weight_from_waz(
#             waz=real_subset_L2['data']['waz'], 
#             age=future_age_days, 
#             sex=real_subset_L2['data']['gender'], 
#             waz_params=waz_params
#         )
        
#         # 5. Apply Weight Drift ONLY to the 'Real' dataset
#         real_subset_L2['data']['weight'] = future_weight

#         # 6. Update Metadata
#         real_subset_L2['data']['age'] = future_age_days
#         real_subset_L2['data']['_agedays'] = future_age_days

#         # 7. Calculate New WHZ (Because body shape changed, WHZ must be recalculated)
#         real_subset_L2['data']['whz'] = ecd_anthro_score_calc.calculate_whz(
#             real_subset_L2['data']['height'],
#             real_subset_L2['data']['weight'],
#             real_subset_L2['data']['gender'],
#             real_subset_L2['data']['loh'],
#             whz_params_lying, 
#             whz_params_standing
#         )

#     # -------------------------------------------------------------------------
#     # NEW FIX: 2. Create the Auditor's base data by copying the NOW-UPDATED truth
#     # This guarantees the Auditor's mistakes happen on top of the child's current size.
#     # -------------------------------------------------------------------------
#     distorted_measurements = {
#         'data': real_subset_L2['data'].copy(deep=True),
#         'metadata': real_subset_L2['metadata'].copy(),
#         'L2_indices': L2_indices
#     }


# # removing old data drift logic.

# #    # Add data drift to height and weight
# #    distorted_measurements['data']['height'] = add_data_drift(
# #        distorted_measurements['data']['height'], drift_mean=drift_mean_height, drift_sd=drift_sd_height
# #    )
# #    distorted_measurements['data']['weight'] = add_data_drift(
# #        distorted_measurements['data']['weight'], drift_mean=drift_mean_weight, drift_sd=drift_sd_weight
# #    )
# #
#     # Re-calculate HAZ, WAZ, WHZ
#     distorted_measurements['data']['haz'] = ecd_anthro_score_calc.calculate_haz(
#         distorted_measurements['data']['height'],
#         distorted_measurements['data']['age'],
#         distorted_measurements['data']['gender'],
#         haz_params
#     )
#     distorted_measurements['data']['waz'] = ecd_anthro_score_calc.calculate_waz(
#         distorted_measurements['data']['weight'],
#         distorted_measurements['data']['age'],
#         distorted_measurements['data']['gender'],
#         waz_params
#     )
#     distorted_measurements['data']['whz'] = ecd_anthro_score_calc.calculate_whz(
#         distorted_measurements['data']['height'],
#         distorted_measurements['data']['weight'],
#         distorted_measurements['data']['gender'],
#         distorted_measurements['data']['loh'],
#         whz_params_lying,
#         whz_params_standing
#     )

#     # Plot dummy data
#     if make_plots:
#         real_data = real_measurements['data'].loc[L2_indices]
#         L0_data = L0_measurements['data'].loc[L2_indices]
#         L1_data = L1_measurements['data'].loc[L2_indices]
#         L2_data = distorted_measurements['data']

#         font_size = 16
#         fig, axs = plt.subplots(3, 5, figsize=figsize, constrained_layout=True)

#         # Row 1: Overlapping histograms for real, L0, L1, L2
#         colors = ['#94979a', '#a226c1', '#26a269', '#e67e22']
#         labels = ['Real', 'L0', 'L1', 'L2']
#         datasets = [real_data, L0_data, L1_data, L2_data]
#         vars = ['height', 'weight', 'haz', 'waz', 'whz']
#         for col, var in enumerate(vars):
#             for i, data in enumerate(datasets):
#                 sns.histplot(data[var], bins=30, kde=False, color=colors[i], label=labels[i],
#                              alpha=0.5, ax=axs[0, col])
#             if col >= 2:  # For HAZ, WAZ, WHZ
#                 axs[0, col].axvline(x=reporting_threshold, color='red', linestyle='--')
#             axs[0, col].set_title(var.upper(), fontsize=font_size)
#             axs[0, col].set_xlabel(var.capitalize(), fontsize=font_size)
#             axs[0, col].tick_params(axis='both', labelsize=font_size-2)
#         axs[0, -1].legend(fontsize=font_size-2)

#         # Row 2: Real vs L2 scatter plots
#         axs[1, 0].scatter(real_data['height'], L2_data['height'],
#                           color='k', marker='.', alpha=0.1)
#         axs[1, 0].set_xlabel('Real height (cm)', fontsize=font_size)
#         axs[1, 0].set_ylabel('L2 height (cm)', fontsize=font_size)
#         axs[1, 0].set_title('Height: Real vs L2', fontsize=font_size)

#         axs[1, 1].scatter(real_data['weight'], L2_data['weight'],
#                           color='k', marker='.', alpha=0.1)
#         axs[1, 1].set_xlabel('Real weight (kg)', fontsize=font_size)
#         axs[1, 1].set_ylabel('L2 weight (kg)', fontsize=font_size)
#         axs[1, 1].set_title('Weight: Real vs L2', fontsize=font_size)

#         bins_haz = np.arange(min(real_data['haz'].min(), L2_data['haz'].min()),
#                              max(real_data['haz'].max(), L2_data['haz'].max()), 0.1)
#         freq_real, _ = np.histogram(real_data['haz'], bins=bins_haz, density=False)
#         freq_L2, _ = np.histogram(L2_data['haz'], bins=bins_haz, density=False)
#         axs[1, 2].bar(x=bins_haz[:-1], height=freq_L2 - freq_real, width=np.diff(bins_haz), color='gray', alpha=0.7)
#         axs[1, 2].set_xlabel('HAZ', fontsize=font_size)
#         axs[1, 2].set_ylabel('Count: L2 - Real', fontsize=font_size)
#         axs[1, 2].axhline(0, color='black', linestyle='--')
#         axs[1, 2].axvline(x=reporting_threshold, color='red', linestyle='--')
#         axs[1, 2].set_title('HAZ Diff', fontsize=font_size)

#         bins_waz = np.arange(min(real_data['waz'].min(), L2_data['waz'].min()),
#                              max(real_data['waz'].max(), L2_data['waz'].max()), 0.1)
#         freq_real, _ = np.histogram(real_data['waz'], bins=bins_waz, density=False)
#         freq_L2, _ = np.histogram(L2_data['waz'], bins=bins_waz, density=False)
#         axs[1, 3].bar(x=bins_waz[:-1], height=freq_L2 - freq_real, width=np.diff(bins_waz), color='gray', alpha=0.7)
#         axs[1, 3].set_xlabel('WAZ', fontsize=font_size)
#         axs[1, 3].set_ylabel('Count: L2 - Real', fontsize=font_size)
#         axs[1, 3].axhline(0, color='black', linestyle='--')
#         axs[1, 3].axvline(x=reporting_threshold, color='red', linestyle='--')
#         axs[1, 3].set_title('WAZ Diff', fontsize=font_size)

#         bins_whz = np.arange(min(real_data['whz'].min(), L2_data['whz'].min()),
#                              max(real_data['whz'].max(), L2_data['whz'].max()), 0.1)
#         freq_real, _ = np.histogram(real_data['whz'], bins=bins_whz, density=False)
#         freq_L2, _ = np.histogram(L2_data['whz'], bins=bins_whz, density=False)
#         axs[1, 4].bar(x=bins_whz[:-1], height=freq_L2 - freq_real, width=np.diff(bins_whz), color='gray', alpha=0.7)
#         axs[1, 4].set_xlabel('WHZ', fontsize=font_size)
#         axs[1, 4].set_ylabel('Count: L2 - Real', fontsize=font_size)
#         axs[1, 4].axhline(0, color='black', linestyle='--')
#         axs[1, 4].axvline(x=reporting_threshold, color='red', linestyle='--')
#         axs[1, 4].set_title('WHZ Diff', fontsize=font_size)

#         # Row 3: L1 vs L2 scatter plots
#         axs[2, 0].scatter(L1_data['height'], L2_data['height'],
#                           color='k', marker='.', alpha=0.1)
#         axs[2, 0].set_xlabel('L1 height (cm)', fontsize=font_size)
#         axs[2, 0].set_ylabel('L2 height (cm)', fontsize=font_size)
#         axs[2, 0].set_title('Height: L1 vs L2', fontsize=font_size)

#         axs[2, 1].scatter(L1_data['weight'], L2_data['weight'],
#                           color='k', marker='.', alpha=0.1)
#         axs[2, 1].set_xlabel('L1 weight (kg)', fontsize=font_size)
#         axs[2, 1].set_ylabel('L2 weight (kg)', fontsize=font_size)
#         axs[2, 1].set_title('Weight: L1 vs L2', fontsize=font_size)

#         freq_L1, _ = np.histogram(L1_data['haz'], bins=bins_haz, density=False)
#         freq_L2, _ = np.histogram(L2_data['haz'], bins=bins_haz, density=False)
#         axs[2, 2].bar(x=bins_haz[:-1], height=freq_L2 - freq_L1, width=np.diff(bins_haz), color='gray', alpha=0.7)
#         axs[2, 2].set_xlabel('HAZ', fontsize=font_size)
#         axs[2, 2].set_ylabel('Count: L2 - L1', fontsize=font_size)
#         axs[2, 2].axhline(0, color='black', linestyle='--')
#         axs[2, 2].axvline(x=reporting_threshold, color='red', linestyle='--')
#         axs[2, 2].set_title('HAZ Diff (L2 - L1)', fontsize=font_size)

#         freq_L1, _ = np.histogram(L1_data['waz'], bins=bins_waz, density=False)
#         freq_L2, _ = np.histogram(L2_data['waz'], bins=bins_waz, density=False)
#         axs[2, 3].bar(x=bins_waz[:-1], height=freq_L2 - freq_L1, width=np.diff(bins_waz), color='gray', alpha=0.7)
#         axs[2, 3].set_xlabel('WAZ', fontsize=font_size)
#         axs[2, 3].set_ylabel('Count: L2 - L1', fontsize=font_size)
#         axs[2, 3].axhline(0, color='black', linestyle='--')
#         axs[2, 3].axvline(x=reporting_threshold, color='red', linestyle='--')
#         axs[2, 3].set_title('WAZ Diff (L2 - L1)', fontsize=font_size)

#         freq_L1, _ = np.histogram(L1_data['whz'], bins=bins_whz, density=False)
#         freq_L2, _ = np.histogram(L2_data['whz'], bins=bins_whz, density=False)
#         axs[2, 4].bar(x=bins_whz[:-1], height=freq_L2 - freq_L1, width=np.diff(bins_whz), color='gray', alpha=0.7)
#         axs[2, 4].set_xlabel('WHZ', fontsize=font_size)
#         axs[2, 4].set_ylabel('Count: L2 - L1', fontsize=font_size)
#         axs[2, 4].axhline(0, color='black', linestyle='--')
#         axs[2, 4].axvline(x=reporting_threshold, color='red', linestyle='--')
#         axs[2, 4].set_title('WHZ Diff (L2 - L1)', fontsize=font_size)

#         plt.tight_layout()
#         plt.show()   

#     return distorted_measurements


def generate_L2_distorted_measurements(
    real_measurements,
    L0_measurements,
    L1_measurements,
    haz_params,
    waz_params,
    whz_params_lying,
    whz_params_standing,
    num_children_L2,
    error_mean_height=0,
    error_sd_height=1,
    error_mean_weight=0,
    error_sd_weight=0.1,
    drift_mean_height=0,
    drift_sd_height=0.1,
    drift_mean_weight=0,
    drift_sd_weight=0.05,
    reporting_threshold=-2,
    figsize=[15, 8],
    make_plots=False,
    drift_method=1, # <--- Added by VS - data drift variable input
    l2_time_lag_days=0 # <--- Added by VS - data drift variable input
):
    """
    Generate L2 distorted measurements by adding measurement error and data drift to real measurements.
    """
    
    L1_indices = L1_measurements['L1_indices']
    if num_children_L2 > len(L1_indices):
        raise ValueError("num_children_L2 exceeds the number of children measured by L1.")

    L2_indices = np.random.choice(L1_indices, size=num_children_L2, replace=False)
    
    # -------------------------------------------------------------------------
    # NEW FIX: 1. Extract the baseline truth into a separate 'Real' variable first!
    # -------------------------------------------------------------------------
    real_subset_L2 = {
        'data': real_measurements['data'].loc[L2_indices].copy(deep=True),
        'metadata': real_measurements['metadata'].copy()
    }

    # # ==============================================================================
    # # APPLY BIOLOGICAL DRIFT (Update Truth to L2 Visit Day)
    # # ==============================================================================
    # if l2_time_lag_days > 0:
    #     # 1. Update Age
    #     current_age_days = real_subset_L2['data']['age']
    #     future_age_days = current_age_days + l2_time_lag_days
        
    #     # 2. Calculate New Height (Bio-Physics based on maintaining HAZ percentile)
    #     future_height = ecd_anthro_score_calc.height_from_haz(
    #         haz=real_subset_L2['data']['haz'], 
    #         age=future_age_days, 
    #         sex=real_subset_L2['data']['gender'], 
    #         haz_params=haz_params
    #     )
        
    #     # 3. Apply Height Drift ONLY to the 'Real' dataset
    #     real_subset_L2['data']['height'] = future_height
        
    #     # 4. Calculate New Weight (Bio-Physics based on maintaining WAZ percentile)
    #     future_weight = ecd_anthro_score_calc.weight_from_waz(
    #         waz=real_subset_L2['data']['waz'], 
    #         age=future_age_days, 
    #         sex=real_subset_L2['data']['gender'], 
    #         waz_params=waz_params
    #     )
        
    #     # 5. Apply Weight Drift ONLY to the 'Real' dataset
    #     real_subset_L2['data']['weight'] = future_weight

    #     # 6. Update Metadata
    #     real_subset_L2['data']['age'] = future_age_days
    #     real_subset_L2['data']['_agedays'] = future_age_days

    #     # 7. Calculate New WHZ (Because body shape changed, WHZ must be recalculated)
    #     real_subset_L2['data']['whz'] = ecd_anthro_score_calc.calculate_whz(
    #         real_subset_L2['data']['height'],
    #         real_subset_L2['data']['weight'],
    #         real_subset_L2['data']['gender'],
    #         real_subset_L2['data']['loh'],
    #         whz_params_lying, 
    #         whz_params_standing
    #     )

# new dd y vs start

# ==============================================================================
    # APPLY BIOLOGICAL DRIFT (Update Truth to L2 Visit Day)
    # ==============================================================================
    if l2_time_lag_days > 0:
        drifted_h, drifted_w = apply_biological_drift(
            current_heights=real_subset_L2['data']['height'],
            current_weights=real_subset_L2['data']['weight'],
            current_ages=real_subset_L2['data']['age'],
            current_haz=real_subset_L2['data']['haz'],
            current_waz=real_subset_L2['data']['waz'],
            genders=real_subset_L2['data']['gender'],
            time_lag_days=l2_time_lag_days,
            method=drift_method,
            haz_params=haz_params,
            waz_params=waz_params
        )
        
        real_subset_L2['data']['height'] = drifted_h
        real_subset_L2['data']['weight'] = drifted_w
        real_subset_L2['data']['age'] = real_subset_L2['data']['age'] + l2_time_lag_days
        real_subset_L2['data']['_agedays'] = real_subset_L2['data']['age'] 
        
        real_subset_L2['data']['whz'] = ecd_anthro_score_calc.calculate_whz(
            real_subset_L2['data']['height'], real_subset_L2['data']['weight'],
            real_subset_L2['data']['gender'], real_subset_L2['data']['loh'],
            whz_params_lying, whz_params_standing
        )

    # Base the Auditor's distorted measurements on the NOW-AGED truth
    distorted_measurements = {
        'data': real_subset_L2['data'].copy(deep=True),
        'metadata': real_subset_L2['metadata'].copy(),
        'L2_indices': L2_indices
    }

# new dd by vs end



    # -------------------------------------------------------------------------
    # NEW FIX: 2. Create the Auditor's base data by copying the NOW-UPDATED truth
    # This guarantees the Auditor's mistakes happen on top of the child's current size.
    # -------------------------------------------------------------------------
    distorted_measurements = {
        'data': real_subset_L2['data'].copy(deep=True),
        'metadata': real_subset_L2['metadata'].copy(),
        'L2_indices': L2_indices
    }

    # Add measurement error to height and weight
    distorted_measurements['data']['height'] = add_measurement_error(
        distorted_measurements['data']['height'], error_mean=error_mean_height, error_sd=error_sd_height
    )
    distorted_measurements['data']['weight'] = add_measurement_error(
        distorted_measurements['data']['weight'], error_mean=error_mean_weight, error_sd=error_sd_weight
    )

    # Re-calculate HAZ, WAZ, WHZ
    distorted_measurements['data']['haz'] = ecd_anthro_score_calc.calculate_haz(
        distorted_measurements['data']['height'],
        distorted_measurements['data']['age'],
        distorted_measurements['data']['gender'],
        haz_params
    )
    distorted_measurements['data']['waz'] = ecd_anthro_score_calc.calculate_waz(
        distorted_measurements['data']['weight'],
        distorted_measurements['data']['age'],
        distorted_measurements['data']['gender'],
        waz_params
    )
    distorted_measurements['data']['whz'] = ecd_anthro_score_calc.calculate_whz(
        distorted_measurements['data']['height'],
        distorted_measurements['data']['weight'],
        distorted_measurements['data']['gender'],
        distorted_measurements['data']['loh'],
        whz_params_lying,
        whz_params_standing
    )

    # Plot dummy data
    if make_plots:
        # NEW FIX: Use real_subset_L2 so the plots compare the Auditor to the child's size TODAY
        real_data = real_subset_L2['data'] 
        L0_data = L0_measurements['data'].loc[L2_indices]
        L1_data = L1_measurements['data'].loc[L2_indices]
        L2_data = distorted_measurements['data']

        font_size = 16
        fig, axs = plt.subplots(3, 5, figsize=figsize, constrained_layout=True)

        # Row 1: Overlapping histograms for real, L0, L1, L2
        colors = ['#94979a', '#a226c1', '#26a269', '#e67e22']
        labels = ['Real', 'L0', 'L1', 'L2']
        datasets = [real_data, L0_data, L1_data, L2_data]
        vars = ['height', 'weight', 'haz', 'waz', 'whz']
        for col, var in enumerate(vars):
            for i, data in enumerate(datasets):
                sns.histplot(data[var], bins=30, kde=False, color=colors[i], label=labels[i],
                             alpha=0.5, ax=axs[0, col])
            if col >= 2:  # For HAZ, WAZ, WHZ
                axs[0, col].axvline(x=reporting_threshold, color='red', linestyle='--')
            axs[0, col].set_title(var.upper(), fontsize=font_size)
            axs[0, col].set_xlabel(var.capitalize(), fontsize=font_size)
            axs[0, col].tick_params(axis='both', labelsize=font_size-2)
        axs[0, -1].legend(fontsize=font_size-2)

        # Row 2: Real vs L2 scatter plots
        axs[1, 0].scatter(real_data['height'], L2_data['height'],
                          color='k', marker='.', alpha=0.1)
        axs[1, 0].set_xlabel('Real height (cm)', fontsize=font_size)
        axs[1, 0].set_ylabel('L2 height (cm)', fontsize=font_size)
        axs[1, 0].set_title('Height: Real vs L2', fontsize=font_size)

        axs[1, 1].scatter(real_data['weight'], L2_data['weight'],
                          color='k', marker='.', alpha=0.1)
        axs[1, 1].set_xlabel('Real weight (kg)', fontsize=font_size)
        axs[1, 1].set_ylabel('L2 weight (kg)', fontsize=font_size)
        axs[1, 1].set_title('Weight: Real vs L2', fontsize=font_size)

        bins_haz = np.arange(min(real_data['haz'].min(), L2_data['haz'].min()),
                             max(real_data['haz'].max(), L2_data['haz'].max()), 0.1)
        freq_real, _ = np.histogram(real_data['haz'], bins=bins_haz, density=False)
        freq_L2, _ = np.histogram(L2_data['haz'], bins=bins_haz, density=False)
        axs[1, 2].bar(x=bins_haz[:-1], height=freq_L2 - freq_real, width=np.diff(bins_haz), color='gray', alpha=0.7)
        axs[1, 2].set_xlabel('HAZ', fontsize=font_size)
        axs[1, 2].set_ylabel('Count: L2 - Real', fontsize=font_size)
        axs[1, 2].axhline(0, color='black', linestyle='--')
        axs[1, 2].axvline(x=reporting_threshold, color='red', linestyle='--')
        axs[1, 2].set_title('HAZ Diff', fontsize=font_size)

        bins_waz = np.arange(min(real_data['waz'].min(), L2_data['waz'].min()),
                             max(real_data['waz'].max(), L2_data['waz'].max()), 0.1)
        freq_real, _ = np.histogram(real_data['waz'], bins=bins_waz, density=False)
        freq_L2, _ = np.histogram(L2_data['waz'], bins=bins_waz, density=False)
        axs[1, 3].bar(x=bins_waz[:-1], height=freq_L2 - freq_real, width=np.diff(bins_waz), color='gray', alpha=0.7)
        axs[1, 3].set_xlabel('WAZ', fontsize=font_size)
        axs[1, 3].set_ylabel('Count: L2 - Real', fontsize=font_size)
        axs[1, 3].axhline(0, color='black', linestyle='--')
        axs[1, 3].axvline(x=reporting_threshold, color='red', linestyle='--')
        axs[1, 3].set_title('WAZ Diff', fontsize=font_size)

        bins_whz = np.arange(min(real_data['whz'].min(), L2_data['whz'].min()),
                             max(real_data['whz'].max(), L2_data['whz'].max()), 0.1)
        freq_real, _ = np.histogram(real_data['whz'], bins=bins_whz, density=False)
        freq_L2, _ = np.histogram(L2_data['whz'], bins=bins_whz, density=False)
        axs[1, 4].bar(x=bins_whz[:-1], height=freq_L2 - freq_real, width=np.diff(bins_whz), color='gray', alpha=0.7)
        axs[1, 4].set_xlabel('WHZ', fontsize=font_size)
        axs[1, 4].set_ylabel('Count: L2 - Real', fontsize=font_size)
        axs[1, 4].axhline(0, color='black', linestyle='--')
        axs[1, 4].axvline(x=reporting_threshold, color='red', linestyle='--')
        axs[1, 4].set_title('WHZ Diff', fontsize=font_size)

        # Row 3: L1 vs L2 scatter plots
        axs[2, 0].scatter(L1_data['height'], L2_data['height'],
                          color='k', marker='.', alpha=0.1)
        axs[2, 0].set_xlabel('L1 height (cm)', fontsize=font_size)
        axs[2, 0].set_ylabel('L2 height (cm)', fontsize=font_size)
        axs[2, 0].set_title('Height: L1 vs L2', fontsize=font_size)

        axs[2, 1].scatter(L1_data['weight'], L2_data['weight'],
                          color='k', marker='.', alpha=0.1)
        axs[2, 1].set_xlabel('L1 weight (kg)', fontsize=font_size)
        axs[2, 1].set_ylabel('L2 weight (kg)', fontsize=font_size)
        axs[2, 1].set_title('Weight: L1 vs L2', fontsize=font_size)

        freq_L1, _ = np.histogram(L1_data['haz'], bins=bins_haz, density=False)
        freq_L2, _ = np.histogram(L2_data['haz'], bins=bins_haz, density=False)
        axs[2, 2].bar(x=bins_haz[:-1], height=freq_L2 - freq_L1, width=np.diff(bins_haz), color='gray', alpha=0.7)
        axs[2, 2].set_xlabel('HAZ', fontsize=font_size)
        axs[2, 2].set_ylabel('Count: L2 - L1', fontsize=font_size)
        axs[2, 2].axhline(0, color='black', linestyle='--')
        axs[2, 2].axvline(x=reporting_threshold, color='red', linestyle='--')
        axs[2, 2].set_title('HAZ Diff (L2 - L1)', fontsize=font_size)

        freq_L1, _ = np.histogram(L1_data['waz'], bins=bins_waz, density=False)
        freq_L2, _ = np.histogram(L2_data['waz'], bins=bins_waz, density=False)
        axs[2, 3].bar(x=bins_waz[:-1], height=freq_L2 - freq_L1, width=np.diff(bins_waz), color='gray', alpha=0.7)
        axs[2, 3].set_xlabel('WAZ', fontsize=font_size)
        axs[2, 3].set_ylabel('Count: L2 - L1', fontsize=font_size)
        axs[2, 3].axhline(0, color='black', linestyle='--')
        axs[2, 3].axvline(x=reporting_threshold, color='red', linestyle='--')
        axs[2, 3].set_title('WAZ Diff (L2 - L1)', fontsize=font_size)

        freq_L1, _ = np.histogram(L1_data['whz'], bins=bins_whz, density=False)
        freq_L2, _ = np.histogram(L2_data['whz'], bins=bins_whz, density=False)
        axs[2, 4].bar(x=bins_whz[:-1], height=freq_L2 - freq_L1, width=np.diff(bins_whz), color='gray', alpha=0.7)
        axs[2, 4].set_xlabel('WHZ', fontsize=font_size)
        axs[2, 4].set_ylabel('Count: L2 - L1', fontsize=font_size)
        axs[2, 4].axhline(0, color='black', linestyle='--')
        axs[2, 4].axvline(x=reporting_threshold, color='red', linestyle='--')
        axs[2, 4].set_title('WHZ Diff (L2 - L1)', fontsize=font_size)

        plt.show()   

    return distorted_measurements


# VS CHANGED BUNCHED DATA CODE - START


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

    # --- NEW: Wrap original logic in Try block to handle edge cases safely ---
    try:
        # --- NEW: Proactively raise error if 0% original (prevents RuntimeWarning) ---
        if percent_below_threshold_original == 0:
            raise ZeroDivisionError
            
        # --- ORIGINAL CODE STARTS HERE (Indented) -----------------------------
        
        # Divide the range of the data into bins of size bin_size
        bins = np.arange(original_data.min(), original_data.max() + bin_size, bin_size).tolist()
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
        
        # --- NEW: Check if there are bins to shift into ---
        if not bins_above_threshold:
            raise ValueError

        for i, bin in enumerate(bins_above_threshold):
            prob = (1 - bunch_factor) ** i
            probabilities.append(prob)

        # Normalize probabilities to sum to 1
        probabilities = np.array(probabilities)
        
        # --- NEW: Check if probabilities sum to zero ---
        if probabilities.sum() == 0:
             raise ValueError

        probabilities /= probabilities.sum()

        # For each point to be shifted, choose a bin above the threshold based on the probabilities
        for idx in shift_indices:
            chosen_bin = np.random.choice(bins_above_threshold, p=probabilities)
            # Choose a random point from the chosen bin and assign it to the bunched data
            bunched_data.loc[idx] = np.random.uniform(chosen_bin.left, chosen_bin.right)

        return bunched_data, warning_flag
        # --- ORIGINAL CODE ENDS -----------------------------------------------

    # --- NEW: Fallback for any calculation errors (returns original data) ---
    except (ZeroDivisionError, ValueError, OverflowError):
        return bunched_data, warning_flag






# VS CHANGED BUNCHED DATA CODE - END




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



# Change by VS: Removing the code below
# def add_data_drift(data, drift_mean=0, drift_sd=0.1):
#     """
#     Add random data drift to measurements.

#     Args:
#         data (pd.Series or pd.DataFrame): Data to be drifted.
#         drift_mean (float): Mean of the normal distribution for drift.
#         drift_sd (float): Standard deviation of the normal distribution for drift.

#     Returns:
#         pd.Series or pd.DataFrame: Data with added drift.
#     """
#     data_with_drift = data.copy(deep=True)
#     drift = np.random.normal(loc=drift_mean, scale=drift_sd, size=len(data))
#     data_with_drift += drift

#     return data_with_drift


def generate_nested_measurements(
        real_params,
        L0_params_list,  # List of dicts, one per L0
        L1_params_list,  # List of dicts, one per L1
        L2_params_dict,  # Dict of params (assuming single L2)
        n_L1s,
        n_L0s_per_L1,
        n_children_per_L0,
        n_children_L1,
        n_children_L2,
        haz_params,
        waz_params,
        whz_params_lying,
        whz_params_standing,
        drift_method=1, # <--- Added by VS - data drift variable input
        make_plots=False
    ):
    """Generate nested measurements for multiple L1s and L0s.
    
    Args:
        real_params (dict): Parameters for generating real measurements
        L0_params_list (list): List of parameter dictionaries for L0 distortion, one per L0
        L1_params_list (list): List of parameter dictionaries for L1 distortion, one per L1
        L2_params_dict (dict): Parameter dictionary for L2 distortion (assuming single L2)
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
            # VS Code Change 1 Start
            # --- NEW FIX: Make Child IDs globally unique! ---
            unique_ids = [f"L1_{L1_id}_L0_{L0_id}_child_{i}" for i in range(n_children_per_L0)]
            real_measurements['data']['child_id'] = unique_ids
            # ------------------------------------------------
            # VS Code Change 1 END


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
            
            # =========================================================
            # L1 GENERATION (Safely popping kwargs to prevent duplication)
            # =========================================================
            l1_kwargs = L1_params_list[L1_id].copy()
            # Safely remove the time lag from the dict so **l1_kwargs doesn't pass it twice
            actual_l1_lag = l1_kwargs.pop('l1_time_lag_days', l1_kwargs.pop('time_lag_L1', 0))

            L1_measurements = generate_L1_distorted_measurements(
                real_measurements=real_measurements,
                L0_distorted_measurements=L0_measurements,
                num_children_L1=n_children_L1,
                haz_params=haz_params,
                waz_params=waz_params,
                whz_params_lying=whz_params_lying,
                whz_params_standing=whz_params_standing,
                make_plots=make_plots,
                drift_method=drift_method,          # <-- The missing link!
                l1_time_lag_days=actual_l1_lag,     # <-- Safely passed once
                **l1_kwargs                         # <-- The rest of the parameters
            )

            # =========================================================
            # L2 GENERATION (Safely popping kwargs to prevent duplication)
            # =========================================================
            l2_kwargs = L2_params_dict.copy()
            actual_l2_lag = l2_kwargs.pop('l2_time_lag_days', l2_kwargs.pop('time_lag_L2', 0))

            L2_measurements = generate_L2_distorted_measurements(
                real_measurements=real_measurements,
                L0_measurements=L0_measurements,
                L1_measurements=L1_measurements,
                haz_params=haz_params,
                waz_params=waz_params,
                whz_params_lying=whz_params_lying,
                whz_params_standing=whz_params_standing,
                num_children_L2=n_children_L2,
                make_plots=make_plots,
                drift_method=drift_method,          # <-- The missing link!
                l2_time_lag_days=actual_l2_lag,     # <-- Safely passed once
                **l2_kwargs                         # <-- The rest of the parameters
            )

            
            # Store all measurements for this L0
            nested_measurements[f'L1_{L1_id}'][f'L0_{L0_id}'] = {
                'real': real_measurements,
                'L0': L0_measurements,
                'L1': L1_measurements,
                'L2': L2_measurements
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
        real_percent_stunting=36,
        real_percent_underweight=34,
        real_percent_wasting=None,
        # L0 parameter means
        mean_percent_under_reporting_stunting=30,
        mean_percent_under_reporting_underweight=30,
        mean_percent_under_reporting_wasting=None,
        mean_bunch_factor_haz=0.1,
        mean_bunch_factor_waz=0.1,
        mean_bunch_factor_whz=0.1,
        error_mean_height_all_L0s = 0,
        error_sd_height_all_L0s = 1,
        error_mean_weight_all_L0s = 0,
        error_sd_weight_all_L0s = 0.1,
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
        error_mean_height_L1 = 0,
        error_sd_height_L1 = 1,
        error_mean_weight_L1 = 0,
        error_sd_weight_L1 = 0.1,
        bunch_factor_haz_L1 = 0.05,
        bunch_factor_waz_L1 = 0.05,
        bunch_factor_whz_L1 = 0.05,
        # L2 parameters
        error_mean_height_L2=0,
        error_sd_height_L2=1,
        error_mean_weight_L2=0,
        error_sd_weight_L2=0.1,
        drift_mean_height_L2=0,
        drift_sd_height_L2=0.1,
        drift_mean_weight_L2=0,
        drift_sd_weight_L2=0.05,
        random_seed=None,
        # VS code addition on data drift
        # --- NEW INPUTS ---# VS code addition on data drift
        mean_time_lag_L1=0,  # Average days for Supervisor to visit# VS code addition on data drift
        sd_time_lag_L1=0,    # Variation (e.g., +/- 5 days)# VS code addition on data drift
        mean_time_lag_L2=0  # Average days for Auditor to visit# VS code addition on data drift
        # ------------------
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
        tuple: (L0_params_list, L1_params_list, L2_params_dict) containing parameter dictionaries
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
    L2_params_dict = {
        # 'error_mean_height': error_mean_height_L2,
        # 'error_sd_height': error_sd_height_L2,
        # 'error_mean_weight': error_mean_weight_L2,
        # 'error_sd_weight': error_sd_weight_L2,
        # 'drift_mean_height': drift_mean_height_L2,
        # 'drift_sd_height': drift_sd_height_L2,
        # 'drift_mean_weight': drift_mean_weight_L2,
        # 'drift_sd_weight': drift_sd_weight_L2,
        # 'time_lag_days': int(max(0, mean_time_lag_L2)) # <--- Added by VS data drift input
        'error_mean_height': error_mean_height_L2,
        'error_sd_height': error_sd_height_L2,
        'error_mean_weight': error_mean_weight_L2,
        'error_sd_weight': error_sd_weight_L2,
        'l2_time_lag_days': int(max(0, mean_time_lag_L2))

    }
    
    # Generate L1 parameters
    for _ in range(n_L1s):
        # Calculate random lag for THIS specific supervisor (Normal Distribution) #Code added by VS for DD
        # e.g. Mean=15, SD=5 -> could get 12, 22, 14...#Code added by VS for DD
        random_lag = np.random.normal(mean_time_lag_L1, sd_time_lag_L1)#Code added by VS for DD
        lag_days = int(max(0, random_lag)) # Ensure days are positive integers#Code added by VS for DD
        L1_params = {
            'percent_copy': max(0, min(100, np.random.normal(mean_percent_copy, sd_percent_copy))),
            'collusion_index': max(0, min(1, np.random.normal(mean_collusion_index, sd_collusion_index))),
            'error_mean_height': error_mean_height_L1,
            'error_sd_height': error_sd_height_L1,
            'error_mean_weight': error_mean_weight_L1,
            'error_sd_weight': error_sd_weight_L1,
            'bunch_factor_haz': bunch_factor_haz_L1,
            'bunch_factor_waz': bunch_factor_waz_L1,
            'bunch_factor_whz': bunch_factor_whz_L1,
            'l1_time_lag_days': lag_days, # <--- ADD THIS #Code added by VS for DD

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
                                                                  sd_within_units_bunch_factor_whz))),
                'error_mean_height': error_mean_height_all_L0s,
                'error_sd_height': error_sd_height_all_L0s,
                'error_mean_weight': error_mean_weight_all_L0s,
                'error_sd_weight': error_sd_weight_all_L0s
            }
            L0_params_list.append(L0_params)
    
    return L0_params_list, L1_params_list, L2_params_dict








def get_L1_L2_pairwise_data(nested_measurements):
    """
    Extract pairwise L1 and L2 measurements for each L1 unit.
    
    Args:
        nested_measurements (dict): Output from generate_nested_measurements()
        
    Returns:
        dict: Dictionary with structure {L1_id: {'L1': array, 'L2': array}} containing
              pairwise measurements for children measured by both L1 and L2
    """
    
    pairwise_data = {}
    
    for L1_id in nested_measurements:
        if L1_id == 'metadata':
            continue
            
        # Collect L1 and L2 data from all L0s within this L1 unit
        all_L1_data = []
        all_L2_data = []
        
        for L0_id in nested_measurements[L1_id]:
            if L0_id == 'L1_info':
                continue
                
            # Get L1, L2 data and L2 indices for this L0
            L1_data_from_L0 = nested_measurements[L1_id][L0_id]['L1']['data']
            L2_data_from_L0 = nested_measurements[L1_id][L0_id]['L2']['data']
            L2_indices_from_L0 = nested_measurements[L1_id][L0_id]['L2']['L2_indices']
            
            # Filter L1 data to only include children measured by L2 in this L0
            L1_subset = L1_data_from_L0.loc[L2_indices_from_L0]
            
            # --- NEW FIX: Set the unique child_id as the Pandas Index! ---
            # This ensures no overlap when we concat multiple L0s together.
            L1_subset.set_index('child_id', inplace=True)
            L2_data_from_L0 = L2_data_from_L0.set_index('child_id')
            # -------------------------------------------------------------

            # Append filtered data
            all_L1_data.append(L1_subset)
            all_L2_data.append(L2_data_from_L0)
        
        # Combine all L1 and L2 data for this L1 unit
        combined_L1_data = pd.concat(all_L1_data, ignore_index=False)
        combined_L2_data = pd.concat(all_L2_data, ignore_index=False)
        
        # Sort by index to ensure same order
        L1_measurements = combined_L1_data.sort_index()
        L2_measurements = combined_L2_data.sort_index()
                
        # Verify they have the same length
        if len(L1_measurements) != len(L2_measurements):
            raise ValueError(f"Mismatch in number of measurements for {L1_id}: "
                           f"L1 has {len(L1_measurements)}, L2 has {len(L2_measurements)}")
        
        # Store pairwise data
        pairwise_data[L1_id] = {
            'L1': L1_measurements,
            'L2': L2_measurements
        }
    
    return pairwise_data



























