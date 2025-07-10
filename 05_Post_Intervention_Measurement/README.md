# Podcast Guest Analysis

## What This Does
- Analyzes podcast episodes for guests
- Identifies gender (Male/Female) and race (specifically Black/Hispanic)
- Uses AI + web search for better accuracy
- Calculates percentages two ways

## Setup (First Time)
```bash
pip install -r requirements.txt
```

## Run It
```bash
python analyze.py
```

## What Happens
1. Reads `podcast_rss_export.csv`
2. Fetches RSS episodes
3. Extracts guest names
4. Uses GPT-4 to classify gender/race
5. For low-confidence cases: searches web for more info (using your 10,000 SerpAPI calls)
6. Outputs results

## Results
`enhanced_guest_analysis_summary.csv` with columns:
- `overall_female_percentage` - % women of all guests
- `overall_urm_percentage` - % Black/Hispanic of all guests  
- `episode_averaged_female_percentage` - Average of per-episode %
- `episode_averaged_urm_percentage` - Average of per-episode %

## NO Face Analysis
This version does NOT analyze photos/faces. It uses:
- Names
- Context from descriptions
- Web search for biographical info

## Time Estimate
~2-4 hours for all podcasts depending on network speed