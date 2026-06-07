# ui_text_config.py

TOOLTIPS = {
    # --- UI Input Parameters ---
    "n_L1s": "The total number of L1 Supervisors (Region/Sector/District Level Government Officer) simulated in the physical universe. Recommended to keep this number < 100. At 100 L1, The data takes 30-45 mins to run",
    "n_L0s_per_L1": "The number of L0 Anganwadi Clinics/Centers assigned to each L1 Supervisor.",
    "n_children_per_L0": "The number of children measured and tracked at each L0 Anganwadi Clinic/Center.",
    
    # --- Exact Mapped Distortion Parameters ---
    "mean_collusion_index": "The degree of coordinated fraud between clinics (L0) and supervisors (L1). 0.00 means errors are random and independent; 0.5 means L1 hides 50% of the Corrupt Data; 1.00 means L1 is hiding all the data that L0 is trying to hide.",
    "mean_percent_copy": "The percentage of child measurements noted by L0 that are copy-pasted exactly by L1 Supervisors, without actually measuring the child.",
    "error_sd_height_all_L0s": "The baseline level of measurement error caused by faulty scales, miscalibrated stadiometers, or poorly trained staff. Applies to both Height and Weight.",
    "mean_percent_under_reporting_stunting": "The percentage of genuinely stunted children that the L0 AWW (Anganwadi Worker) purposefully under report and bumps up above the -2.0 HAZ threshold to make their clinic's health outcomes look better.",
    "mean_percent_under_reporting_underweight": "The percentage of genuinely underweight children that the L0 AWW (Anganwadi Worker) purposefully under report and bumps up above the -2.0 WAZ threshold to hide malnutrition rates.",
    
    # --- System Architecture ---
    "target_percentile": "The target of 'Worst Units' the audit/model is trying to accurately identify. Ex: If target = 30%. We catch the 30% of units that make the most fraud and see how accurate we are. Are we catching all 30% fraud units, or 20%, etc. If we catch 24% out of that target 30% units. Then we declare that Ranking Accuracy is 80 out of 100 (24 out of 30 is 80%).",
    "output_variable": "Select whether the dashboard calculates and displays metrics for Height (HAZ), Weight (WAZ), or Both sequentially.",
    "l1_budget": "The percentage of the total L0 Clinics that the L1 Supervisor has the budget/capacity to audit.",
    "l2_budget": "The percentage of L1-audited Clinics that the L2 independent auditor will re-audit.",
    "has_l2": "Toggle whether an independent L2 auditor exists in this scenario. Turning this to 'No' skips L2 matrix calculations.",

    # --- Advanced Parameters ---
    "real_percent_stunting": "The true biological percentage of children in the population who are actually stunted.",
    "real_percent_underweight": "The true biological percentage of children in the population who are actually underweight.",
    "sd_across": "Standard deviation representing how much under-reporting varies between different L1 Regions/Sectors/Districts.",
    "sd_within": "Standard deviation representing how much under-reporting varies across L0 Anganwadi clinic within a single L1 region.",
    "bunch_factor": "The variance used when L0 clinics artificially 'bunch' children's measurements right above the malnutrition cutoff line (Z Score -2). Ex: 1 - imples all the malnourished children are measured just < -2; 0.1 implies all the malnourished childred are measured at any point <-2. The closer this parameter goes to 1, the closer the measurements are to <-2 (Like -1.9)",
    "sd_copy": "The variance applied to the Copying Rate by L1.",
    "sd_collusion": "The variance applied to the L1/L0 collusion index.",
    "time_lag": "The average number of days between the original measurement taken by AWW (Anganwadi Worker) and the next measurement/visit by the higher up (L1/L2).",
    "n_simulations": "The number of Monte Carlo iterations run per strategy. Higher numbers yield highly accurate variance/error bars but take longer to calculate.",
    
    # --- Plot Descriptions ---
    "plot_6_heatmap": "### 📖 How to read this chart\nThis heatmap shows the interaction between L1 and L2 budgets. \n* **The Blue Matrix (Left):** Shows the L1's baseline accuracy at catching bad clinics based on their sampling strategy.\n* **The Red/Green Matrix (Right):** Shows how accurately L2 auditors execute their cross-checks based on how deep they audit. Greener cells indicate high accuracy in catching the true worst offenders.",
    "plot_3_breadth_depth": "### 📖 How to read this chart\nThis chart illustrates the fundamental audit trade-off under a **fixed budget**.\n* Does accuracy improve if we audit **many clinics** but measure **few kids** per clinic (Breadth)? \n* Or is it better to audit **fewer clinics** but measure **every kid** inside them (Depth)? \n* **The Error Bars (I):** Represent the standard deviation (variance) across hundreds of simulated Monte Carlo audits. Smaller error bars mean the strategy produces highly consistent, predictable results.",
    "plot_2_intra_regional": "### 📖 How to read this chart\nThis trendline shows how an L1 Supervisor's accuracy in ranking their own internal L0 clinics improves as their overall budget/sampling percentage increases. The red star highlights the exact budget you have selected in the filters above.\n* **The Shaded Region:** Represents the confidence band / standard deviation across all random simulation runs. The wider the shadow, the more the accuracy fluctuates depending on which specific children the auditor happens to sample."
}













# # ui_text_config.py

# TOOLTIPS = {
#     # --- UI Input Parameters ---
#     "n_L1s": "The total number of L1 Supervisors (Regions/Districts) simulated in the physical universe.",
#     "n_L0s_per_L1": "The number of L0 Clinics/Centers assigned to each L1 Supervisor.",
#     "n_children_per_L0": "The number of children measured and tracked at each L0 Clinic.",
#     "l1_corruption_pct": "The percentage of L1 Supervisors who actively manipulate data to hide underperformance.",
#     "l0_fraud_pct": "The percentage of L0 Clinics that commit fraud (e.g., bunching child measurements just above the stunting cutoff).",
#     "collusion_factor": "The degree to which L1 and L0 coordinate their fraudulent reporting (0.0 = completely independent, 1.0 = total collusion).",
#     "copy_paste_pct": "The percentage of child records that are simply copy-pasted from previous months rather than actually measured.",
#     "equipment_error": "The random margin of error (in cm/kg) added to measurements due to faulty weighing scales or measuring boards.",
#     "target_percentile": "The target threshold of 'Worst Offenders' the audit is trying to accurately identify (e.g., catching the bottom 30%).",
#     "output_variable": "Select whether the dashboard calculates and displays metrics for Height (HAZ), Weight (WAZ), or Both sequentially.",
#     "l1_budget": "The percentage of the total L0 Clinics that the L1 Supervisor has the budget/capacity to audit.",
#     "l2_budget": "The percentage of L1-audited Clinics that the L2 independent auditor will re-audit.",
#     "has_l2": "Toggle whether an independent L2 auditor exists in this scenario. Turning this to 'No' skips L2 matrix calculations.",
# # --- Advanced Parameters ---

#     "real_percent_stunting": "The true biological percentage of children in the population who are actually stunted.",
#     "real_percent_underweight": "The true biological percentage of children in the population who are actually underweight.",
#     "sd_across": "Standard deviation representing how much under-reporting varies between different L0 clinics.",
#     "sd_within": "Standard deviation representing how much under-reporting varies within a single L0 clinic across different children.",
#     "bunch_factor": "The variance used when L0 clinics artificially 'bunch' children's measurements right above the malnutrition cutoff line.",
#     "sd_copy": "The variance applied to the copy-paste rate.",
#     "sd_collusion": "The variance applied to the L1/L0 collusion index.",
#     "time_lag": "The average number of days between the original measurement and the audit. Longer lags naturally introduce more biological variance.",
#     # --- Plot Descriptions ---
#     "n_simulations": "The number of Monte Carlo iterations run per strategy. Higher numbers yield highly accurate variance/error bars but take longer to calculate.",
#     "plot_6_heatmap": "### 📖 How to read this chart\nThis heatmap shows the interaction between L1 and L2 budgets. \n* **The Blue Matrix (Left):** Shows the L1's baseline accuracy at catching bad clinics based on their sampling strategy.\n* **The Red/Green Matrix (Right):** Shows how accurately L2 auditors execute their cross-checks based on how deep they audit. Greener cells indicate high accuracy in catching the true worst offenders.",
#     "plot_3_breadth_depth": "### 📖 How to read this chart\nThis chart illustrates the fundamental audit trade-off under a **fixed budget**.\n* Does accuracy improve if we audit **many clinics** but measure **few kids** per clinic (Breadth)? \n* Or is it better to audit **fewer clinics** but measure **every kid** inside them (Depth)? \n* **The Error Bars (I):** Represent the standard deviation (variance) across hundreds of simulated Monte Carlo audits. Smaller error bars mean the strategy produces highly consistent, predictable results.",
#     "plot_2_intra_regional": "### 📖 How to read this chart\nThis trendline shows how an L1 Supervisor's accuracy in ranking their own internal L0 clinics improves as their overall budget/sampling percentage increases. The red star highlights the exact budget you have selected in the filters above.\n* **The Shaded Region:** Represents the confidence band / standard deviation across all random simulation runs. The wider the shadow, the more the accuracy fluctuates depending on which specific children the auditor happens to sample."
# }