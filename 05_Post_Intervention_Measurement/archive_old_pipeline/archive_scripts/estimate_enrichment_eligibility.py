#!/usr/bin/env python3
"""
Estimate how many guests would be eligible for SerpAPI enrichment
"""

import json
import pandas as pd
from collections import Counter
import re

def analyze_guest_eligibility():
    """Analyze existing data to estimate enrichment needs"""
    
    print("=== Estimating Guest Enrichment Eligibility ===\n")
    
    # 1. Analyze existing classification data
    print("1. Analyzing existing guest classification data...")
    
    total_episodes = 0
    episodes_with_guests = 0
    total_guests = 0
    unique_guest_names = set()
    guest_name_frequency = Counter()
    
    try:
        with open('post_intervention_guest_classification_full.jsonl', 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    total_episodes += 1
                    
                    num_guests = data.get('total_guests', 0)
                    if num_guests > 0:
                        episodes_with_guests += 1
                        total_guests += num_guests
                        
                        # Extract guest names from explain field (rough estimate)
                        explain = data.get('explain', '')
                        # Look for patterns like "1 guest mentioned: Name"
                        name_matches = re.findall(r'(?:guest[s]? mentioned:|named?|is) ([A-Z][a-z]+ [A-Z][a-z]+)', explain)
                        for name in name_matches:
                            unique_guest_names.add(name)
                            guest_name_frequency[name] += 1
                            
                except json.JSONDecodeError:
                    continue
        
        print(f"  Total episodes analyzed: {total_episodes:,}")
        print(f"  Episodes with guests: {episodes_with_guests:,}")
        print(f"  Total guest appearances: {total_guests:,}")
        print(f"  Unique guest names found (rough estimate): {len(unique_guest_names):,}")
        
    except FileNotFoundError:
        print("  ERROR: Could not find classification file")
        return
    
    # 2. Estimate based on guest frequency
    print("\n2. Guest frequency analysis:")
    
    # Count how many guests appear multiple times
    single_appearance = sum(1 for count in guest_name_frequency.values() if count == 1)
    multiple_appearances = sum(1 for count in guest_name_frequency.values() if count > 1)
    
    print(f"  Guests appearing once: {single_appearance:,}")
    print(f"  Guests appearing multiple times: {multiple_appearances:,}")
    
    # 3. Calculate enrichment estimates
    print("\n3. Enrichment eligibility estimates:")
    
    # Conservative estimate: assume we can extract ~70% of actual guest names
    estimated_unique_guests = int(total_guests * 0.7)
    
    print(f"\n  Conservative estimate of unique guests: {estimated_unique_guests:,}")
    
    # Further filtering for enrichment eligibility
    print("\n  Filtering criteria for enrichment:")
    print("  - Remove common first names only (10%)")
    print("  - Remove unclear/partial names (15%)")
    print("  - Remove duplicate variations (10%)")
    
    eligible_after_filtering = int(estimated_unique_guests * 0.65)
    
    print(f"\n  Estimated eligible for enrichment: {eligible_after_filtering:,}")
    
    # 4. API call estimates
    print("\n4. SerpAPI call estimates:")
    
    # Each guest might need:
    # - 1 text search
    # - 1 image search
    calls_per_guest = 2
    
    total_api_calls = eligible_after_filtering * calls_per_guest
    
    print(f"  API calls per guest: {calls_per_guest}")
    print(f"  Total API calls needed: {total_api_calls:,}")
    
    # Cost estimates (SerpAPI pricing as of 2024)
    # Free tier: 100 searches/month
    # Basic: $50/month for 5,000 searches
    # Professional: $130/month for 15,000 searches
    # Business: $250/month for 30,000 searches
    
    print("\n5. Cost estimates (SerpAPI pricing):")
    
    if total_api_calls <= 100:
        print(f"  ✓ Would fit in FREE tier (100 searches/month)")
    elif total_api_calls <= 5000:
        print(f"  Would need Basic plan: $50/month (5,000 searches)")
    elif total_api_calls <= 15000:
        print(f"  Would need Professional plan: $130/month (15,000 searches)")
    elif total_api_calls <= 30000:
        print(f"  Would need Business plan: $250/month (30,000 searches)")
    else:
        print(f"  Would need Enterprise plan (>{total_api_calls:,} searches)")
    
    # 6. Optimization strategies
    print("\n6. Optimization strategies to reduce API calls:")
    print("  a) Cache all results (already implemented)")
    print("  b) Prioritize high-confidence guests only")
    print("  c) Skip enrichment for very common names")
    print("  d) Batch process by podcast (deduplicate guests)")
    print("  e) Use only text search first, image if needed")
    
    # 7. Phased approach recommendation
    print("\n7. Recommended phased approach:")
    
    phase1_guests = int(eligible_after_filtering * 0.1)  # Top 10%
    phase1_calls = phase1_guests * calls_per_guest
    
    print(f"\n  Phase 1 (Test with 10% sample):")
    print(f"    Guests: {phase1_guests:,}")
    print(f"    API calls: {phase1_calls:,}")
    print(f"    Cost: {'FREE' if phase1_calls <= 100 else '$50 (Basic plan)'}")
    
    phase2_guests = int(eligible_after_filtering * 0.25)  # Top 25%
    phase2_calls = phase2_guests * calls_per_guest
    
    print(f"\n  Phase 2 (Expand to 25%):")
    print(f"    Guests: {phase2_guests:,}")
    print(f"    API calls: {phase2_calls:,}")
    
    # Alternative free options
    print("\n8. Alternative FREE enrichment options:")
    print("  - Use Google Custom Search API (100 queries/day free)")
    print("  - Implement basic web scraping for public profiles")
    print("  - Use Wikipedia API for notable guests")
    print("  - Leverage podcast website data when available")
    
    return {
        'total_guests': total_guests,
        'estimated_unique_guests': estimated_unique_guests,
        'eligible_for_enrichment': eligible_after_filtering,
        'total_api_calls': total_api_calls
    }

def analyze_by_podcast_sample():
    """Analyze a sample of podcasts to get better estimates"""
    print("\n\n=== Detailed Sample Analysis ===\n")
    
    try:
        # Load the aggregated summary
        df = pd.read_csv('post_intervention_guest_summary_full.csv')
        
        print(f"Total podcasts with results: {len(df):,}")
        
        # Analyze guest counts
        total_guests = df['post_total_guests'].sum()
        podcasts_with_guests = len(df[df['post_total_guests'] > 0])
        
        print(f"Total guests across all podcasts: {total_guests:,}")
        print(f"Podcasts with at least 1 guest: {podcasts_with_guests:,}")
        
        # Guest distribution
        print("\nGuest count distribution:")
        guest_dist = df['post_total_guests'].value_counts().sort_index()
        for guests, count in guest_dist.head(10).items():
            if guests > 0:
                print(f"  {int(guests)} guests: {count} podcasts")
        
        # Estimate unique guests per podcast
        avg_guests_per_podcast = df[df['post_total_guests'] > 0]['post_total_guests'].mean()
        
        # Rough uniqueness factor (guests appear in ~1.5 podcasts on average)
        uniqueness_factor = 0.67  # Conservative estimate
        
        estimated_unique = int(total_guests * uniqueness_factor)
        
        print(f"\nUniqueness analysis:")
        print(f"  Average guests per podcast (with guests): {avg_guests_per_podcast:.2f}")
        print(f"  Estimated unique guests (67% of total): {estimated_unique:,}")
        
        # High-volume podcasts
        high_volume = df[df['post_total_guests'] > 10].sort_values('post_total_guests', ascending=False)
        if len(high_volume) > 0:
            print(f"\nHigh-volume podcasts (>10 guests): {len(high_volume)}")
            print("Top 5:")
            for _, row in high_volume.head().iterrows():
                print(f"  {row['podcastID'][:8]}...: {int(row['post_total_guests'])} guests")
        
    except FileNotFoundError:
        print("Could not find aggregated summary file")
    except Exception as e:
        print(f"Error analyzing sample: {e}")

def main():
    # Run analysis
    estimates = analyze_guest_eligibility()
    analyze_by_podcast_sample()
    
    # Final summary
    print("\n" + "="*60)
    print("EXECUTIVE SUMMARY")
    print("="*60)
    
    if estimates:
        print(f"\nEstimated unique guests eligible for enrichment: {estimates['eligible_for_enrichment']:,}")
        print(f"Estimated total API calls needed: {estimates['total_api_calls']:,}")
        
        # Quick recommendation
        calls = estimates['total_api_calls']
        if calls <= 100:
            print("\n✓ Recommendation: Use SerpAPI FREE tier")
            print("  All enrichment can be done at no cost!")
        elif calls <= 5000:
            print("\n✓ Recommendation: Start with FREE tier for testing")
            print("  Then upgrade to Basic plan ($50/month) if results are valuable")
        else:
            print("\n✓ Recommendation: Start with strategic sample")
            print("  1. Test with 100 high-profile guests (FREE)")
            print("  2. Evaluate value of enrichment")
            print("  3. Consider paid plan only if significant value demonstrated")
            
        print("\n💡 Cost-saving tip: Start with name-based demographics only")
        print("   Add web enrichment selectively for ambiguous cases")

if __name__ == "__main__":
    analyze_guest_eligibility()