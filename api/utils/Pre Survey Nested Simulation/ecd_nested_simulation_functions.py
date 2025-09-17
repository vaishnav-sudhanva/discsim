import numpy as np
import pandas as pd
from scipy.stats import norm
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm

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
    figsize = [10, 8]
    
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

    # Apply under-reporting for stunting and underweight if provided
    if percent_under_reporting_stunting is not None and percent_under_reporting_underweight is not None:
        
        # Throw an error if percent under-reporting for stunting or underweight exceeds the actual percent stunting or underweight
        if percent_under_reporting_stunting > real_measurements['metadata']['percent_stunting']:
            raise ValueError("Percent under-reporting for stunting exceeds actual percent stunting.")
        if percent_under_reporting_underweight > real_measurements['metadata']['percent_underweight']:
            raise ValueError("Percent under-reporting for underweight exceeds actual percent underweight.")

        # Under-reporting for stunting
        distorted_measurements['data']['haz'] = generate_bunched_data(threshold=reporting_threshold,
                                                               original_data=real_measurements['data']['haz'],
                                                               percent_below_threshold_original=real_measurements['metadata']['percent_stunting'],
                                                               percent_below_threshold_bunched=real_measurements['metadata']['percent_stunting'] - percent_under_reporting_stunting,
                                                               bunch_factor=bunch_factor_haz, bin_size=bin_size)
        # Under-reporting for underweight
        distorted_measurements['data']['waz'] = generate_bunched_data(threshold=reporting_threshold,
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
        distorted_measurements['data']['whz'] = generate_bunched_data(threshold=reporting_threshold,
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

        plt.figure(figsize=figsize, constrained_layout=True, sharex = 'col', sharey = 'row')

        # Improved color palette and outlines for clarity
        real_color = "#94979a"        # blue
        real_edge = "#0d2c47"
        distorted_color = "#a226c1"   # orange
        distorted_edge = "#83028f"
        alpha_real = 0.8
        alpha_distorted = 0.3

        # Row 1: Overlapping histograms for real and distorted measurements
        plt.subplot(2, 5, 1)
        sns.histplot(real_measurements['data']['height'], bins=30, kde=False, color=real_color, 
                     label='Real', alpha=alpha_real, edgecolor=real_edge, linewidth=0.2)
        sns.histplot(distorted_measurements['data']['height'], bins=30, kde=False, color=distorted_color, 
                     label='Distorted', alpha=alpha_distorted, edgecolor=distorted_edge, linewidth=0.2)
        plt.xlabel('Height (cm)')
        plt.title('Height')
        plt.legend()

        plt.subplot(2, 5, 2)
        sns.histplot(real_measurements['data']['weight'], bins=30, kde=False, color=real_color, 
                     label='Real', alpha=alpha_real, edgecolor=real_edge, linewidth=0.2)
        sns.histplot(distorted_measurements['data']['weight'], bins=30, kde=False, color=distorted_color, 
                     label='Distorted', alpha=alpha_distorted, edgecolor=distorted_edge, linewidth=0.2)
        plt.xlabel('Weight (kg)')
        plt.title('Weight')
        plt.legend()

        min_haz = min(real_measurements['data']['haz'].min(), distorted_measurements['data']['haz'].min())
        max_haz = max(real_measurements['data']['haz'].max(), distorted_measurements['data']['haz'].max())
        bins_haz = np.arange(min_haz, max_haz, bin_size)
        plt.subplot(2, 5, 3)
        sns.histplot(real_measurements['data']['haz'], bins=bins_haz, kde=False, color=real_color, 
                     label='Real', alpha=alpha_real, edgecolor=real_edge, linewidth=0.2)
        sns.histplot(distorted_measurements['data']['haz'], bins=bins_haz, kde=False, color=distorted_color, 
                     label='Distorted', alpha=alpha_distorted, edgecolor=distorted_edge, linewidth=0.2)
        plt.axvline(x=reporting_threshold, color='red', linestyle='--')
        plt.xlabel('HAZ')
        plt.title('HAZ')
        plt.legend()

        min_waz = min(real_measurements['data']['waz'].min(), distorted_measurements['data']['waz'].min())
        max_waz = max(real_measurements['data']['waz'].max(), distorted_measurements['data']['waz'].max())
        bins_waz = np.arange(min_waz, max_waz, bin_size)
        plt.subplot(2, 5, 4)
        sns.histplot(real_measurements['data']['waz'], bins=bins_waz, kde=False, color=real_color, 
                     label='Real', alpha=alpha_real, edgecolor=real_edge, linewidth=0.2)
        sns.histplot(distorted_measurements['data']['waz'], bins=bins_waz, kde=False, color=distorted_color, 
                     label='Distorted', alpha=alpha_distorted, edgecolor=distorted_edge, linewidth=0.2)
        plt.axvline(x=reporting_threshold, color='red', linestyle='--')
        plt.xlabel('WAZ')
        plt.title('WAZ')
        plt.legend()

        min_whz = min(real_measurements['data']['whz'].min(), distorted_measurements['data']['whz'].min())
        max_whz = max(real_measurements['data']['whz'].max(), distorted_measurements['data']['whz'].max())
        bins_whz = np.arange(min_whz, max_whz, bin_size)
        plt.subplot(2, 5, 5)
        sns.histplot(real_measurements['data']['whz'], bins=bins_whz, kde=False, color=real_color, 
                     label='Real', alpha=alpha_real, edgecolor=real_edge, linewidth=0.2)
        sns.histplot(distorted_measurements['data']['whz'], bins=bins_whz, kde=False, color=distorted_color, 
                     label='Distorted', alpha=alpha_distorted, edgecolor=distorted_edge, linewidth=0.2)
        plt.axvline(x=reporting_threshold, color='red', linestyle='--')
        plt.xlabel('WHZ')
        plt.title('WHZ')
        plt.legend()

        # Row 2: Differences
        plt.subplot(2, 5, 6)
        sns.scatterplot(x=real_measurements['data']['height'],
                        y=distorted_measurements['data']['height'] - real_measurements['data']['height'],
                        color='k', marker = '.', alpha = 0.5)
        plt.xlabel('Real height (cm)')
        plt.ylabel('Distorted - Real')
        plt.title('Height Diff')

        plt.subplot(2, 5, 7)
        sns.scatterplot(x=real_measurements['data']['weight'],
                        y=distorted_measurements['data']['weight'] - real_measurements['data']['weight'],
                        color='k', marker = '.', alpha = 0.5)
        plt.xlabel('Real weight (kg)')
        plt.ylabel('Distorted - Real')
        plt.title('Weight Diff')

        plt.subplot(2, 5, 8)
        freq_real, _ = np.histogram(real_measurements['data']['haz'], bins=bins_haz, density=False)
        freq_distorted, _ = np.histogram(distorted_measurements['data']['haz'], bins=bins_haz, density=False)
        plt.bar(x=bins_haz[:-1], height=freq_distorted - freq_real, width=np.diff(bins_haz), color='gray', alpha=0.7)
        plt.xlabel('HAZ')
        plt.ylabel('Distorted - Real')
        plt.axhline(0, color='black', linestyle='--')
        plt.axvline(x=reporting_threshold, color='red', linestyle='--')
        plt.title('HAZ Diff')

        plt.subplot(2, 5, 9)
        freq_real, _ = np.histogram(real_measurements['data']['waz'], bins=bins_waz, density=False)
        freq_distorted, _ = np.histogram(distorted_measurements['data']['waz'], bins=bins_waz, density=False)
        plt.bar(x=bins_waz[:-1], height=freq_distorted - freq_real, width=np.diff(bins_waz), color='gray', alpha=0.7)
        plt.xlabel('WAZ')
        plt.ylabel('Distorted - Real')
        plt.axhline(0, color='black', linestyle='--')
        plt.axvline(x=reporting_threshold, color='red', linestyle='--')
        plt.title('WAZ Diff')

        plt.subplot(2, 5, 10)
        freq_real, _ = np.histogram(real_measurements['data']['whz'], bins=bins_whz, density=False)
        freq_distorted, _ = np.histogram(distorted_measurements['data']['whz'], bins=bins_whz, density=False)
        plt.bar(x=bins_whz[:-1], height=freq_distorted - freq_real, width=np.diff(bins_whz), color='gray', alpha=0.7)
        plt.xlabel('WHZ')
        plt.ylabel('Distorted - Real')
        plt.axhline(0, color='black', linestyle='--')
        plt.axvline(x=reporting_threshold, color='red', linestyle='--')
        plt.title('WHZ Diff')

        plt.tight_layout()
        plt.show()

    # Return distorted data
    return distorted_measurements,

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
    """
    bunched_data = original_data.copy(deep = True)
    
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

    return bunched_data

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

def get_msl(age, sex, params, age_label = '_agedays', sex_label = '__000001'):

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

    for idx, i in tqdm(enumerate(age.dropna().index)):
        sex_i = np.array(sex)[idx]
        age_i = np.array(age)[idx]
        m[idx] = lookup_m[sex_i].get(age_i, np.nan)
        s[idx] = lookup_s[sex_i].get(age_i, np.nan)
        l[idx] = lookup_l[sex_i].get(age_i, np.nan)

    return m, s, l