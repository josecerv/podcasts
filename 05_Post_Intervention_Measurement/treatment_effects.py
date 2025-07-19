import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS

# Load data
summary_df = pd.read_csv('guest_analysis_summary.csv')
episodes_df = pd.read_csv('guest_analysis_episodes.csv')

# Filter for podcasts that had at least one guest
podcast_level = summary_df[summary_df['total_guests'] > 0].copy()

print("=" * 80)
print("TREATMENT EFFECTS ON DIVERSITY: OLS REGRESSION ANALYSIS")
print("=" * 80)

# Calculate percentages for podcast-level analysis
podcast_level['female_pct'] = podcast_level['female_guests'] / podcast_level['total_guests'] * 100
podcast_level['urm_pct'] = podcast_level['urm_guests'] / podcast_level['total_guests'] * 100

# Prepare episode-level data with fractional counting
episode_groups = episodes_df.groupby(['podcast_id', 'treatment', 'episode_title', 'episode_date']).agg({
    'gender': ['count', lambda x: (x == 'Female').sum()],
    'is_urm': 'sum'
}).reset_index()

episode_groups.columns = ['podcast_id', 'treatment', 'episode_title', 'episode_date', 
                         'total_guests', 'female_guests', 'urm_guests']

episode_groups['female_frac'] = episode_groups['female_guests'] / episode_groups['total_guests'] * 100
episode_groups['urm_frac'] = episode_groups['urm_guests'] / episode_groups['total_guests'] * 100

# Analysis 1: Podcast-level OLS
print("\n1. PODCAST-LEVEL ANALYSIS")
print("   Outcome: Percentage of all guests who are women/URM")
print("-" * 60)

# Women - Podcast level
X = sm.add_constant(podcast_level['treatment'])
model_women_podcast = OLS(podcast_level['female_pct'], X).fit()

print("\nWOMEN GUESTS:")
print(f"  Control group mean: {podcast_level[podcast_level['treatment']==0]['female_pct'].mean():.2f}%")
print(f"  Treatment group mean: {podcast_level[podcast_level['treatment']==1]['female_pct'].mean():.2f}%")
print(f"  Treatment effect: {model_women_podcast.params['treatment']:.2f} percentage points")
print(f"  Standard error: {model_women_podcast.bse['treatment']:.2f}")
print(f"  p-value: {model_women_podcast.pvalues['treatment']:.3f}")
print(f"  95% CI: [{model_women_podcast.conf_int().loc['treatment', 0]:.2f}, {model_women_podcast.conf_int().loc['treatment', 1]:.2f}]")

# URM - Podcast level
model_urm_podcast = OLS(podcast_level['urm_pct'], X).fit()

print("\nURM GUESTS:")
print(f"  Control group mean: {podcast_level[podcast_level['treatment']==0]['urm_pct'].mean():.2f}%")
print(f"  Treatment group mean: {podcast_level[podcast_level['treatment']==1]['urm_pct'].mean():.2f}%")
print(f"  Treatment effect: {model_urm_podcast.params['treatment']:.2f} percentage points")
print(f"  Standard error: {model_urm_podcast.bse['treatment']:.2f}")
print(f"  p-value: {model_urm_podcast.pvalues['treatment']:.3f}")
print(f"  95% CI: [{model_urm_podcast.conf_int().loc['treatment', 0]:.2f}, {model_urm_podcast.conf_int().loc['treatment', 1]:.2f}]")

# Analysis 2: Episode-level OLS with clustering
print("\n\n2. EPISODE-LEVEL ANALYSIS")
print("   Outcome: Percentage of guests per episode who are women/URM (fractional counting)")
print("-" * 60)

# Women - Episode level
X_ep = sm.add_constant(episode_groups['treatment'])
model_women_episode = OLS(episode_groups['female_frac'], X_ep).fit()

# Calculate clustered standard errors manually
from statsmodels.stats.sandwich_covariance import cov_cluster
cluster_cov_women = cov_cluster(model_women_episode, episode_groups['podcast_id'])
cluster_se_women = np.sqrt(np.diag(cluster_cov_women))

print("\nWOMEN GUESTS (per episode):")
print(f"  Control group mean: {episode_groups[episode_groups['treatment']==0]['female_frac'].mean():.2f}%")
print(f"  Treatment group mean: {episode_groups[episode_groups['treatment']==1]['female_frac'].mean():.2f}%")
print(f"  Treatment effect: {model_women_episode.params['treatment']:.2f} percentage points")
print(f"  Standard error (clustered): {cluster_se_women[1]:.2f}")
print(f"  p-value: {model_women_episode.pvalues['treatment']:.3f}")

# URM - Episode level
model_urm_episode = OLS(episode_groups['urm_frac'], X_ep).fit()
cluster_cov_urm = cov_cluster(model_urm_episode, episode_groups['podcast_id'])
cluster_se_urm = np.sqrt(np.diag(cluster_cov_urm))

print("\nURM GUESTS (per episode):")
print(f"  Control group mean: {episode_groups[episode_groups['treatment']==0]['urm_frac'].mean():.2f}%")
print(f"  Treatment group mean: {episode_groups[episode_groups['treatment']==1]['urm_frac'].mean():.2f}%")
print(f"  Treatment effect: {model_urm_episode.params['treatment']:.2f} percentage points")
print(f"  Standard error (clustered): {cluster_se_urm[1]:.2f}")
print(f"  p-value: {model_urm_episode.pvalues['treatment']:.3f}")

# Summary
print("\n\n3. INTERPRETATION")
print("-" * 60)
print("- Positive coefficients indicate the treatment increased diversity")
print("- Negative coefficients indicate the treatment decreased diversity")
print("- Episode-level analysis uses clustered SEs to account for within-podcast correlation")

print("\n4. SAMPLE INFORMATION")
print("-" * 60)
print(f"Total podcasts analyzed: {len(summary_df):,}")
print(f"Podcasts with at least 1 guest: {len(podcast_level):,}")
print(f"  - Treatment group: {len(podcast_level[podcast_level['treatment']==1]):,}")
print(f"  - Control group: {len(podcast_level[podcast_level['treatment']==0]):,}")
print(f"Total episodes with guests: {len(episode_groups):,}")
print(f"  - Treatment group: {len(episode_groups[episode_groups['treatment']==1]):,}")
print(f"  - Control group: {len(episode_groups[episode_groups['treatment']==0]):,}")