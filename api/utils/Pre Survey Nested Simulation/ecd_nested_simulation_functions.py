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
    genders = ['F'] * num_girls + ['M'] * num_boys
    np.random.shuffle(genders)

    # Assign starting ages - uniform distribution between min and max age
    start_ages = np.random.uniform(low=min_age, high=max_age, size=num_children)

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
    elif percent_stunting is not None:
        # If percent_stunting is provided, use it to calculate haz_mean_0, assuming standard deviation of 1. 
        # Percent stunting is the percentile of the distribution that is less than two standard deviations from the mean, i.e. < -2.
        percent_stunting = percent_stunting / 100  # Convert to fraction
        haz_mean_0 = -2 - norm.ppf(percent_stunting)
        haz_0 = np.random.normal(loc=haz_mean_0, scale=1, size=num_children)

    # Initial weights
    # Check that at least one of waz_mean or percent_underweight is not None. If both are None, throw an error
    if (waz_mean_0 is None or waz_std_0 is None) and percent_underweight is None:
        raise ValueError("At least one of waz_mean_0 or percent_underweight must be provided.")

    # If waz_mean_0 and waz_std_0 are provided, use them to generate initial waz distribution
    if waz_mean_0 is not None and waz_std_0 is not None:
        waz_0 = np.random.normal(loc=waz_mean_0, scale=waz_std_0, size=num_children)
    elif percent_underweight is not None:
        # If percent_underweight is provided, use it to calculate waz_mean_0, assuming standard deviation of 1. 
        # Percent underweight is the percentile of the distribution that is less than two standard deviations from the mean, i.e. < -2.
        percent_underweight = percent_underweight / 100  # Convert to fraction
        waz_mean_0 = -2 - norm.ppf(percent_underweight)
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
    
    records = []
    for i in range(num_children):
        child_id = f"child_{i}"
        gender = genders[i]
        age = start_ages[i]
        height = heights_0[i]
        weight = weights_0[i]
        records.append({
            'child_id': child_id,
            'gender': gender,
            'timepoint': tp,
            'age': curr_age,
            'height': height,
            'weight': weight
        })
        real_measurements = pd.DataFrame(records)

    # Plot age, haz, waz, height and weight distributions if plot_distributions is True
    if plot_distributions:
        plt.figure(figsize=(15, 10))

        plt.subplot(2, 3, 1)
        sns.histplot(real_measurements['age'], bins=10, kde=True)
        plt.title('Age Distribution')

        plt.subplot(2, 3, 2)
        sns.histplot(real_measurements['haz'], bins=10, kde=True)
        plt.title('HAZ Distribution')

        plt.subplot(2, 3, 3)
        sns.histplot(real_measurements['waz'], bins=10, kde=True)
        plt.title('WAZ Distribution')

        plt.subplot(2, 3, 4)
        sns.histplot(real_measurements['height'], bins=10, kde=True)
        plt.title('Height Distribution')

        plt.subplot(2, 3, 5)
        sns.histplot(real_measurements['weight'], bins=10, kde=True)
        plt.title('Weight Distribution')

        plt.tight_layout()
        plt.show()

    return real_measurements

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

def calculate_whz(height, weight, sex, whz_params):
    """
    Calculate Weight-for-Height Z-scores (WHZ) using the WHO growth standards.

    Args:
        height (pd.Series): Height measurements.
        weight (pd.Series): Weight measurements.
        age (pd.Series): Age measurements.

    Returns:
        pd.Series: WHZ scores.
    """
    # Make sure height, weight, age and sex are the same length, else throw an error
    if not (len(height) == len(weight) == len(sex)):
        raise ValueError("Input series must have the same length")
    
    # Get M, S, L values for the age and sex
    m, s, l = get_msl(height, sex, whz_params, age_label = '__000003')
    whz = get_anthro_zscore(weight.dropna(), m, s, l)
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
        sex_i = sex[i]
        age_i = age[i]
        m[idx] = lookup_m[sex_i].get(age_i, np.nan)
        s[idx] = lookup_s[sex_i].get(age_i, np.nan)
        l[idx] = lookup_l[sex_i].get(age_i, np.nan)

    return m, s, l