#!/usr/bin/env python3
"""
Statistical significance testing for treatment vs control groups
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load the data
df = pd.read_csv('enhanced_guest_analysis_summary.csv')

# Filter out podcasts with no guests
df_with_guests = df[df['total_guests'] > 0].copy()

print("=== Statistical Significance Analysis ===\n")
print(f"Total podcasts analyzed: {len(df)}")
print(f"Podcasts with guests: {len(df_with_guests)}")
print(f"Treatment group: {len(df_with_guests[df_with_guests['treatment'] == 1])}")
print(f"Control group: {len(df_with_guests[df_with_guests['treatment'] == 0])}")

# Metrics to test
metrics = [
    ('overall_female_percentage', 'Overall Female Percentage'),
    ('overall_urm_percentage', 'Overall URM Percentage'),
    ('episode_averaged_female_percentage', 'Episode-Averaged Female Percentage'),
    ('episode_averaged_urm_percentage', 'Episode-Averaged URM Percentage')
]

print("\n=== Results ===\n")

for metric, label in metrics:
    print(f"\n{label}:")
    print("-" * 50)
    
    # Split by treatment
    treatment = df_with_guests[df_with_guests['treatment'] == 1][metric].dropna()
    control = df_with_guests[df_with_guests['treatment'] == 0][metric].dropna()
    
    # Calculate means and standard errors
    treatment_mean = treatment.mean()
    control_mean = control.mean()
    treatment_se = treatment.sem()
    control_se = control.sem()
    
    print(f"Treatment (n={len(treatment)}): {treatment_mean:.2f}% ± {treatment_se:.2f}")
    print(f"Control (n={len(control)}): {control_mean:.2f}% ± {control_se:.2f}")
    print(f"Difference: {treatment_mean - control_mean:.2f} percentage points")
    
    # Perform t-test
    if len(treatment) > 1 and len(control) > 1:
        # Two-sample t-test (assuming unequal variances)
        t_stat, p_value = stats.ttest_ind(treatment, control, equal_var=False)
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt((treatment.std()**2 + control.std()**2) / 2)
        cohens_d = (treatment_mean - control_mean) / pooled_std if pooled_std > 0 else 0
        
        print(f"t-statistic: {t_stat:.3f}")
        print(f"p-value: {p_value:.4f}")
        print(f"Cohen's d: {cohens_d:.3f}")
        
        # Significance interpretation
        if p_value < 0.001:
            sig = "***"
        elif p_value < 0.01:
            sig = "**"
        elif p_value < 0.05:
            sig = "*"
        else:
            sig = "ns"
        
        print(f"Significance: {sig} {'(p < 0.001)' if p_value < 0.001 else f'(p = {p_value:.4f})'}")
        
        # Mann-Whitney U test (non-parametric alternative)
        u_stat, u_p_value = stats.mannwhitneyu(treatment, control, alternative='two-sided')
        print(f"Mann-Whitney U p-value: {u_p_value:.4f}")
    else:
        print("Insufficient data for statistical test")

# Additional analysis: Check for outliers
print("\n\n=== Distribution Analysis ===")
for metric, label in metrics:
    print(f"\n{label} - Percentile Distribution:")
    treatment = df_with_guests[df_with_guests['treatment'] == 1][metric].dropna()
    control = df_with_guests[df_with_guests['treatment'] == 0][metric].dropna()
    
    for group, data in [("Treatment", treatment), ("Control", control)]:
        if len(data) > 0:
            print(f"{group}: 25th={data.quantile(0.25):.1f}, 50th={data.quantile(0.50):.1f}, 75th={data.quantile(0.75):.1f}, 95th={data.quantile(0.95):.1f}")

# Save detailed results
results_df = pd.DataFrame()
for metric, label in metrics:
    treatment = df_with_guests[df_with_guests['treatment'] == 1][metric].dropna()
    control = df_with_guests[df_with_guests['treatment'] == 0][metric].dropna()
    
    if len(treatment) > 1 and len(control) > 1:
        t_stat, p_value = stats.ttest_ind(treatment, control, equal_var=False)
        results_df = pd.concat([results_df, pd.DataFrame({
            'metric': [label],
            'treatment_mean': [treatment.mean()],
            'control_mean': [control.mean()],
            'difference': [treatment.mean() - control.mean()],
            'p_value': [p_value],
            'significant': [p_value < 0.05]
        })])

if len(results_df) > 0:
    results_df.to_csv('statistical_significance_results.csv', index=False)
    print("\n\nDetailed results saved to: statistical_significance_results.csv")