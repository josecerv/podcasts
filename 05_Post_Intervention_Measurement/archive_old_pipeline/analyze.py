#!/usr/bin/env python3
"""
SIMPLE PODCAST GUEST ANALYSIS
Analyzes podcast guests for gender and race (Black/Hispanic)
"""

print("Starting Podcast Guest Analysis...")
print("This will take several hours for all podcasts.")
print("")

# Just import and run the main analysis
from run_enhanced_analysis import main

try:
    main()
    print("\n✓ Analysis complete! Check:")
    print("  - enhanced_guest_analysis_summary.csv (main results)")
    print("  - enhanced_guest_analysis_episodes.csv (episode details)")
except KeyboardInterrupt:
    print("\n\nAnalysis interrupted by user.")
except Exception as e:
    print(f"\n✗ Error: {e}")
    print("Check enhanced_analysis.log for details")