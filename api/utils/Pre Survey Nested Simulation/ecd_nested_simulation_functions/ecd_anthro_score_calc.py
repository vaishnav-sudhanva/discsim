import numpy as np
import pandas as pd
from tqdm import tqdm

"""
Documentation: Added a detailed docstring defining all input types as pd.Series and parameters as pd.DataFrame.
Data Integrity: Replaced the use of .dropna() with raw height to ensure the output Z-score series matches the original input index length.
Validation: Maintained the strict length-check for height, age, and sex to prevent broadcast errors.
Structural Impact: Safe; the function signature remains identical, ensuring full compatibility with existing application calls.
"""

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
    #haz = get_anthro_zscore(height.dropna(), m, s, l)
    haz = get_anthro_zscore(height, m, s, l)
    return haz



"""
The transition from the old height_from_haz function to the new version introduces significant biological and mathematical refinements.
Mathematical Engine Swap: Replaced the generic invert_anthro_zscore call with the explicit Box-Cox power transformation formula ($height = M \times ((HAZ \times L \times S) + 1)^{1/L}$).
Biological Position Adjustment: Added a mandatory -0.7cm correction for children $\ge$ 730 days (2 years). This accounts for the WHO standard shift from measuring "Lying Down" (Length) to "Standing Up" (Height), where the spine compresses.
Index Preservation: Wrapped the final output in a pd.Series that explicitly reattaches the original haz.index. This prevents the "Pandas Index Trap" where row labels are lost during calculation.
Documentation: Added a detailed docstring defining the input requirements for haz, age, sex, and haz_params.
Structural Impact: Safe. The function signature remains identical. It still accepts four positional arguments and returns a pd.Series, maintaining full compatibility with your existing application.
"""
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
    # Make sure haz, age, sex are the same length, else throw an error
    if not (len(haz) == len(age) == len(sex)):
        raise ValueError("Input series must have the same length")
    
    # Get M, S, L values for the age and sex
    m, s, l = get_msl(age, sex, haz_params)
    height = invert_anthro_zscore(haz, m, s, l)
    
    #removing this
    # BIOLOGICAL POSITION ADJUSTMENT (The -0.65 HAZ Fix)
    # Convert age to numpy array safely to avoid Pandas index alignment issues
    # mask_standing = (np.array(age) >= 730)
    
    # Convert height to numpy array to safely apply the mask
    height_arr = np.array(height)
    # height_arr[mask_standing] -= 0.7
    
    # CRITICAL FIX: Extract index safely so it doesn't crash on numpy arrays
    idx = haz.index if hasattr(haz, 'index') else None
    
    return pd.Series(height_arr, index=idx)




"""
Gemini said
Registered Changes for calculate_waz and weight_from_waz
calculate_waz Documentation: Added a detailed docstring defining input requirements for weight, age, and sex.
calculate_waz Validation: Maintained strict length-checks across all input series to ensure data alignment.
"""
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



"""
weight_from_waz Logic: Continued use of the invert_anthro_zscore engine to reverse-engineer absolute weight from Z-scores.
Structural Impact: Safe; both functions maintain identical signatures and return pd.Series to ensure application compatibility.
"""

def weight_from_waz(waz, age, sex, waz_params):
    """
    Calculate weight from Weight-for-Age Z-scores (WAZ) using the WHO growth standards.
    Index-Safe Version.
    """
    if not (len(waz) == len(age) == len(sex)):
        raise ValueError("Input series must have the same length")
    
    m, s, l = get_msl(age, sex, waz_params)
    weight = invert_anthro_zscore(waz, m, s, l)
    
    # CRITICAL FIX: Extract index safely so it doesn't crash on numpy arrays
    idx = waz.index if hasattr(waz, 'index') else None
    
    return pd.Series(np.array(weight), index=idx)





"""
Registered Changes for calculate_whz and weight_from_whz
Data Splitting: Explicitly routes children into two groups (Lying Down vs. Standing Up) based on the loh indicator.
Refined Parameters: Uses specific WHO reference columns (__000002 for length and __000003 for height) to retrieve accurate M, S, and L values.
Index Safety: Implemented sort_index() before returning the combined data to ensure the infants and toddlers are restored to their original row order.
Structural Impact: Safe; the functions preserve original signatures while significantly increasing calculation accuracy and memory efficiency.
"""

def weight_from_whz(whz, height, sex, loh, whz_params_lying, whz_params_standing):
    """
    Reverse-engineers a child's absolute weight (in kg) based on their Weight-for-Height Z-score (WHZ).
    
    THEORY:
    The WHO Growth Standards use a "Box-Cox power exponential" method to map biological curves.
    Because biological weight isn't perfectly symmetrical (you can be 10kg overweight, but you 
    can't be 10kg underweight if you only weigh 8kg), the WHO uses three parameters:
    M (Median), S (Coefficient of variation), and L (Skewness/Power). 
    This function takes the child's Z-score and works backward through the Box-Cox equation 
    to find exactly how many kilograms they weigh.

    FUNCTION VARIABLES (Inputs):
    ---------------------------
    whz                 : (pd.Series) The simulated or actual Weight-for-Height Z-score.
                          (e.g., -2.0 means the child is 2 standard deviations below the median weight for their height).
    height              : (pd.Series) The physical height or length of the child in centimeters (cm).
    sex                 : (pd.Series) WHO biological sex code (1 = Male, 2 = Female). 
                          Needed because boys and girls have different biological growth curves.
    loh                 : (pd.Series) "Length or Height" indicator. 
                          1 = Measured Lying Down (Supine Length - standard for children < 24 months).
                          2 = Measured Standing Up (Standing Height - standard for children >= 24 months).
    whz_params_lying    : (pd.DataFrame) The WHO reference table for Lying Down ('wflanthro.dta').
    whz_params_standing : (pd.DataFrame) The WHO reference table for Standing Up ('wfhanthro.dta').
    """
    
    # ==========================================
    # 1. DATA VALIDATION
    # ==========================================
    # If the Pandas Series are different lengths, the row-by-row math will completely break.
    # This prevents silent Pandas alignment bugs before they happen.
    if not (len(whz) == len(height) == len(sex) == len(loh)):
        raise ValueError("Input series must have the same length")
    
    # ==========================================
    # 2. SPLITTING BY MEASUREMENT POSITION
    # ==========================================
    # BIOLOGICAL CONTEXT: The WHO maintains two separate tables for Weight-for-Height.
    # Why? Because a child's spine decompresses when lying down, making them slightly "longer" 
    # than when they are standing. We must route the data to the correct biological table.
    
    # Filter the datasets for ONLY the children measured lying down (loh == 1)
    height_0, whz_0, sex_0 = height[loh == 1], whz[loh == 1], sex[loh == 1]
    
    # Filter the datasets for ONLY the children measured standing up (loh == 2)
    height_1, whz_1, sex_1 = height[loh == 2], whz[loh == 2], sex[loh == 2]

    # ==========================================
    # 3. PROCESS GROUP 0: LYING DOWN (LOH == 1)
    # ==========================================
    # Step A: Get the WHO parameters (M, S, L) for this specific group's length.
    # '__000002' is the specific column name inside the WHO 'wflanthro.dta' file representing Length.
    # We pass `height_0` directly to `get_msl`. It uses `np.interp` to find the exact decimal 
    # parameters without needing to round the child's height or break the Pandas index.
    m_0, s_0, l_0 = get_msl(height_0, sex_0, whz_params_lying, age_label='__000002')
    
    # Step B: Mathematical Inversion.
    # We feed the Z-score (whz_0) and the biological parameters (m_0, s_0, l_0) into the 
    # inverted Box-Cox formula. 
    # The output (weight_0) is the absolute weight in kilograms for these babies.
    weight_0 = invert_anthro_zscore(whz_0, m_0, s_0, l_0)

    # ==========================================
    # 4. PROCESS GROUP 1: STANDING UP (LOH == 2)
    # ==========================================
    # Step A: Get parameters from the Standing Up table.
    # '__000003' is the specific column name inside the WHO 'wfhanthro.dta' file representing Height.
    m_1, s_1, l_1 = get_msl(height_1, sex_1, whz_params_standing, age_label='__000003')
    
    # Step B: Mathematical Inversion for the toddlers/older children.
    weight_1 = invert_anthro_zscore(whz_1, m_1, s_1, l_1)

    # ==========================================
    # 5. RECOMBINE AND RESTORE
    # ==========================================
    # pd.concat glues the two calculated weight lists back into one single column.
    # .sort_index() is CRITICAL. Because we split the data by LOH, the rows are out of order.
    # sorting by the index puts Child #1, Child #2, Child #3 perfectly back in their original rows.
    return pd.concat([weight_0, weight_1]).sort_index()






def calculate_whz(height, weight, sex, loh, whz_params_lying, whz_params_standing):
    """
    Calculate Weight-for-Height Z-scores (WHZ) using the WHO growth standards.
    
    FUNCTION ARGUMENTS (Inputs):
    ---------------------------
    - height: (pd.Series) The measured height/length of the children in centimeters (cm).
    - weight: (pd.Series) The measured weight of the children in kilograms (kg).
    - sex: (pd.Series) Biological sex indicator based on WHO codes (1 = Male, 2 = Female).
    - loh: (pd.Series) "Length or Height" indicator. 
           1 = Measured Lying Down (Length, usually for kids < 2 years old).
           2 = Measured Standing Up (Height, usually for kids >= 2 years old).
    - whz_params_lying: (pd.DataFrame) The raw WHO reference table ('wflanthro.dta') used for Lying Down.
    - whz_params_standing: (pd.DataFrame) The raw WHO reference table ('wfhanthro.dta') used for Standing Up.
    """
    
    # 1. Validation to ensure no mismatched data lengths
    if not (len(height) == len(weight) == len(loh) == len(sex)):
        raise ValueError("Input series must have the same length")
    
    # 2. Split variables into two groups based on measurement position (loh)
    # INTERNAL VARIABLES (Subsets):
    # - height_0, weight_0, sex_0: Only contains the data for babies measured LYING DOWN (loh == 1).
    height_0, weight_0, sex_0 = height[loh == 1], weight[loh == 1], sex[loh == 1]
    
    # - height_1, weight_1, sex_1: Only contains the data for toddlers measured STANDING UP (loh == 2).
    height_1, weight_1, sex_1 = height[loh == 2], weight[loh == 2], sex[loh == 2]

    # ==========================================
    # PROCESS GROUP 0: LYING DOWN (LOH == 1)
    # ==========================================
    # INTERNAL VARIABLES (WHO Parameters):
    # - m_0: Median (The baseline "perfect" weight for this specific length).
    # - s_0: Coefficient of Variation (How much the data spreads out from the median).
    # - l_0: Box-Cox power (A mathematical trick to fix skewed/asymmetrical biological data).
    # NOTE: '__000002' is the specific column name inside the WHO .dta file representing 'Length'.
    m_0, s_0, l_0 = get_msl(height_0, sex_0, whz_params_lying, age_label='__000002')
    
    # - whz_0: The final calculated Z-score series for the lying-down group.
    whz_0 = get_anthro_zscore(weight_0, m_0, s_0, l_0)

    # ==========================================
    # PROCESS GROUP 1: STANDING UP (LOH == 2)
    # ==========================================
    # NOTE: '__000003' is the specific column name inside the WHO .dta file representing 'Height'.
    m_1, s_1, l_1 = get_msl(height_1, sex_1, whz_params_standing, age_label='__000003')
    
    # - whz_1: The final calculated Z-score series for the standing-up group.
    whz_1 = get_anthro_zscore(weight_1, m_1, s_1, l_1)
    
    # 3. Recombine the two groups
    # pd.concat glues the lying-down Z-scores and standing-up Z-scores back into one single column.
    # .sort_index() puts them back into the exact original row order they were passed in as.
    return pd.concat([whz_0, whz_1]).sort_index()



#get_anthro_zscore / invert_anthro_zscore: Implemented Numpy stripping and Index Restoration to prevent "Pandas Index Trap" errors.
def get_anthro_zscore(y, m, s, l):
    """
    Calculate the anthropometric Z-score using the WHO growth standards.
    Index-safe version that prevents broadcast shape errors.
    """
    # CRITICAL FIX: Safe index extraction (won't crash on numpy arrays)
    original_index = y.index if hasattr(y, 'index') else None
    
    # Strip everything down to raw arrays for perfectly aligned math
    y_vals = np.asarray(y)
    m_vals = np.asarray(m)
    s_vals = np.asarray(s)
    l_vals = np.asarray(l)
    
    # Calculate Z-score
    z_vals = (np.power(y_vals / m_vals, l_vals) - 1 ) / (s_vals * l_vals)
    
    # Return as a series mapped correctly to the original index
    return pd.Series(z_vals, index=original_index)

def invert_anthro_zscore(z, m, s, l):
    """
    Invert the anthropometric Z-score to obtain the original measurement.
    Index-safe version to prevent broadcast shape errors.
    """
    # CRITICAL FIX: Safe index extraction (won't crash on numpy arrays)
    original_index = z.index if hasattr(z, 'index') else None
    
    # Strip everything down to raw arrays for perfectly aligned math
    z_vals = np.asarray(z)
    m_vals = np.asarray(m)
    s_vals = np.asarray(s)
    l_vals = np.asarray(l)
    
    # Calculate the original measurement (y)
    y_vals = m_vals * np.power((1 + z_vals * s_vals * l_vals), 1/l_vals)
    
    # Return as a series mapped correctly to the original index
    return pd.Series(y_vals, index=original_index)


#get_msl: Vectorized the lookup engine using np.interp, replacing slow loops with high-speed linear interpolation for fractional ages.
def get_msl(age, sex, params, age_label='_agedays', sex_label='__000001', verbose=False):
    """
    Get M, S, L values for a specific age and sex from the WHO growth standards.
    Vectorized using numpy interpolation for index safety.
    """
    # 1. Convert to numpy arrays to ensure index-independent math
    age_arr = np.array(age)
    sex_arr = np.array(sex)
    
    m = np.zeros(len(age_arr))
    s = np.zeros(len(age_arr))
    l = np.zeros(len(age_arr))

    # 2. Process Boys (1) and Girls (2) separately using masks
    for s_val in [1, 2]:
        mask = (sex_arr == s_val)
        if not mask.any():
            continue
            
        # Filter WHO reference table for this sex
        s_params = params[params[sex_label] == s_val].sort_values(age_label)
        
        # 3. Vectorized Lookup: This maps any age to the correct WHO L, S, M values
        m[mask] = np.interp(age_arr[mask], s_params[age_label], s_params['m'])
        s[mask] = np.interp(age_arr[mask], s_params[age_label], s_params['s'])
        l[mask] = np.interp(age_arr[mask], s_params[age_label], s_params['l'])
        
    # CRITICAL FIX: Check if 'age' is a Pandas Series with an index, otherwise use None
    idx = age.index if hasattr(age, 'index') else None
        
    # Return as Series, safely mapped back to the child IDs
    return pd.Series(m, index=idx), pd.Series(s, index=idx), pd.Series(l, index=idx)



















def apply_biological_growth(age, haz, waz, sex, time_lag_days, haz_params, waz_params):
    """
    Simulates the physical growth of children over a specific time period (time_lag).
    
    THEORY (Canalization / Method 1):
    Children naturally follow specific growth curves (Z-scores). If a child is at 
    -1.5 HAZ today, we assume they will still be at -1.5 HAZ in 30 days. However, 
    because they are 30 days older, the WHO standards dictate that their absolute 
    physical height (cm) and weight (kg) MUST increase to stay on that curve.
    
    This function calculates exactly how many cm and kg they gained during that lag.

    FUNCTION VARIABLES (Inputs):
    ---------------------------
    - age             : (pd.Series) The current age of the children in days.
    - haz             : (pd.Series) The current Height-for-Age Z-score (Baseline).
    - waz             : (pd.Series) The current Weight-for-Age Z-score (Baseline).
    - sex             : (pd.Series) WHO biological sex code (1 = Male, 2 = Female).
    - time_lag_days   : (int or pd.Series) How many days have passed between the L1 measurement 
                        and the L2 audit (e.g., 30 days).
    - haz_params      : (pd.DataFrame) WHO reference table for HAZ ('lenanthro.dta').
    - waz_params      : (pd.DataFrame) WHO reference table for WAZ ('weianthro.dta').
    
    RETURNS:
    --------
    - new_height      : (pd.Series) The child's new physical height/length in cm.
    - new_weight      : (pd.Series) The child's new physical weight in kg.
    """
    
    # ==========================================
    # 1. AGE THE CHILDREN
    # ==========================================
    # We add the delay (time_lag_days) to the baseline age to find out exactly 
    # how old the child is when the L2 Auditor finally arrives.
    new_age = age + time_lag_days

    # ==========================================
    # 2. CALCULATE NEW HEIGHT
    # ==========================================
    # We pass the OLD Z-score (haz) but the NEW age into our updated WHO function.
    # The function looks at the WHO tables for the older age, finds the parameters, 
    # and calculates exactly how tall the child must be to maintain their original Z-score.
    new_height = height_from_haz(haz=haz, 
                                 age=new_age, 
                                 sex=sex, 
                                 haz_params=haz_params)

    # ==========================================
    # 3. CALCULATE NEW WEIGHT
    # ==========================================
    # Similarly, we hold the Weight-for-Age Z-score (waz) constant, but pass in the NEW age.
    # This automatically models the exact fractional kilograms the child gained 
    # while waiting for the L2 Auditor.
    new_weight = weight_from_waz(waz=waz, 
                                 age=new_age, 
                                 sex=sex, 
                                 waz_params=waz_params)

    # Return the simulated absolute measurements 
    # (These become the new "True" biological parameters for the L2 Audit)
    return new_height, new_weight