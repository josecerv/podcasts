# Enhanced Podcast Guest Analysis - Simple Instructions

## Setup (One Time)

1. **Install Dependencies**
   ```bash
   pip install pandas numpy feedparser openai requests tqdm
   ```

2. **API Keys**
   - OpenAI key is loaded from parent directory `.env` file
   - SerpAPI key is already set in `config.json`

## Run the Analysis

```bash
python run_enhanced_analysis.py
```

That's it! The script will:
- Load podcasts from `podcast_rss_export.csv`
- Fetch RSS episodes
- Extract and classify guests
- Use SerpAPI for enrichment when confidence is low
- Calculate percentages two ways

## Output Files

1. **enhanced_guest_analysis_summary.csv**
   - One row per podcast
   - Contains both percentage calculations:
     - `overall_female_percentage`: Total female guests / total guests
     - `overall_urm_percentage`: Total URM guests / total guests  
     - `episode_averaged_female_percentage`: Average of per-episode percentages
     - `episode_averaged_urm_percentage`: Average of per-episode URM percentages

2. **enhanced_guest_analysis_episodes.csv**
   - One row per episode
   - Raw guest counts

3. **enrichment_cache/**
   - Cached SerpAPI results (automatic)
   - Prevents duplicate searches

## Understanding the Percentages

Example with 2 episodes:
- Episode 1: 1 woman, 0 men
- Episode 2: 0 women, 3 men

**Overall percentage**: 1 woman / 4 total guests = 25%
**Episode-averaged**: (100% + 0%) / 2 = 50%

The first shows overall representation, the second shows consistency across episodes.

## Monitoring Progress

- Progress bar shows podcast processing
- Check `enhanced_analysis.log` for details
- API calls remaining shown at end

## Notes

- URM = Black or Hispanic/Latino only
- Enrichment only runs for low-confidence classifications
- Cache prevents repeat searches
- Limited to 10,000 SerpAPI calls