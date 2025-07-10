# Quick Start Guide

## 1. Install Missing Package
```bash
pip install feedparser
```

## 2. Run the Analysis
```bash
./run.sh
```

That's it! The script will:
- Load your OpenAI key from `../.env`
- Use SerpAPI (10,000 calls available)
- Analyze all podcasts
- Calculate percentages both ways

## Output Files
- `enhanced_guest_analysis_summary.csv` - Main results with both percentage types
- `enhanced_guest_analysis_episodes.csv` - Episode-level details
- `enhanced_analysis.log` - Detailed log

## Key Columns in Summary
- `overall_female_percentage` - Women as % of all guests
- `overall_urm_percentage` - Black/Hispanic as % of all guests
- `episode_averaged_female_percentage` - Average of per-episode percentages
- `episode_averaged_urm_percentage` - Average of per-episode percentages