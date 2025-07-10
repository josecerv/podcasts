# Enhanced Podcast Guest Analysis Pipeline v2.0

This enhanced version addresses the three key concerns you raised about guest extraction accuracy, RSS feed reliability, and demographic determination methods.

## Key Improvements

### 1. Enhanced Guest Extraction (95%+ accuracy target)
- **Pattern Matching**: Uses regex patterns to identify common guest introduction phrases
  - "featuring", "with guest", "interview with", "joined by", etc.
  - Professional titles: "author", "professor", "CEO", "founder", etc.
- **Named Entity Recognition (NER)**: spaCy-based extraction of person names
- **Context Extraction**: Captures affiliations, titles, and book information
- **Multi-method Validation**: Combines pattern and NER results with confidence scores

### 2. Robust RSS Feed Fetching (95%+ success rate target)
- **Retry Logic**: Up to 3 attempts with exponential backoff
- **User-Agent Rotation**: Multiple user agents to avoid blocks
- **Enhanced Error Handling**: Specific handling for different error types
- **Success Tracking**: Logs success/failure for each feed in `rss_fetch_success_log.csv`
- **Detailed Metrics**: Reports overall success rate and failure patterns

### 3. Multi-Source Demographic Analysis
- **Web Search Integration**: 
  - Searches for guest information using context (affiliation, books, etc.)
  - Supports both SerpAPI and Google Custom Search API
- **Photo Analysis with DeepFace**:
  - Downloads and analyzes guest photos when available
  - Uses multiple detection backends for robustness
  - Returns confidence scores for gender and race predictions
- **Combined Approach**:
  - Name-based analysis (existing)
  - Photo-based analysis (new)
  - Context-based validation (new)
  - Aggregated confidence scoring

## Setup Instructions

### 1. Install Dependencies

Run the setup script:
```bash
python setup_enhanced_pipeline.py
```

Or manually with Poetry:
```bash
poetry install
```

### 2. Set Environment Variables

Required:
```bash
export OPENAI_API_KEY='your-openai-api-key'
```

Optional (for enhanced features):
```bash
export SERPAPI_KEY='your-serpapi-key'
export GOOGLE_SEARCH_API_KEY='your-google-api-key'
export GOOGLE_CSE_ID='your-google-cse-id'
```

### 3. Prepare Input Data

Ensure `podcast_rss_export.csv` is in the current directory with columns:
- `podcastID`: Unique podcast identifier
- `rss`: RSS feed URL
- `latest_response_time`: Completion timestamp
- `treatment`: 0 or 1 for control/treatment group

### 4. Run the Enhanced Pipeline

```bash
python post-intervention-guest-enhanced.py
```

## Output Files

1. **post_intervention_guest_classification_enhanced.jsonl**
   - Raw LLM classification results with validated guest lists
   - Includes extraction quality scores

2. **post_intervention_guest_summary_enhanced.csv**
   - Aggregated podcast-level statistics
   - Includes extraction quality metrics

3. **post_intervention_guests_enriched.csv**
   - Individual guest-level data
   - Names, demographics, confidence scores, extraction methods

4. **rss_fetch_success_log.csv**
   - RSS fetch success/failure for each podcast
   - Use to identify problematic feeds

5. **guest_photos/** (directory)
   - Downloaded guest photos for analysis

6. **enrichment_cache/** (directory)
   - Cached web search and analysis results

## Configuration Options

Edit `pipeline_config.json` to customize:

```json
{
  "rss_settings": {
    "timeout": 30,
    "retry_attempts": 3,
    "retry_delay": 5
  },
  "llm_settings": {
    "model": "gpt-4o-mini",
    "max_workers": 20
  },
  "enrichment_settings": {
    "enable_web_search": true,
    "enable_photo_analysis": true,
    "max_images_per_guest": 3
  }
}
```

## Performance Metrics

The enhanced pipeline provides detailed metrics:

### RSS Fetching
- Total attempts
- Success rate (target: 95%+)
- Failure analysis

### Guest Extraction
- Extraction quality scores (high/medium/low)
- Number of guests found by each method
- Confidence scores for each guest

### Demographic Analysis
- Source of determination (name/photo/context)
- Confidence scores for each prediction
- Aggregated accuracy metrics

## Troubleshooting

### Low RSS Success Rate
1. Check network connectivity
2. Increase timeout in config
3. Review failed feeds in success log
4. Some feeds may be permanently unavailable

### Poor Guest Extraction
1. Review extraction quality scores
2. Check episode descriptions for unusual formats
3. Add custom patterns for specific podcast styles

### Demographic Analysis Issues
1. Ensure API keys are set correctly
2. Check enrichment cache for errors
3. Review photo download failures
4. Some guests may have limited online presence

## Scientific Integrity

This enhanced pipeline prioritizes scientific accuracy through:

1. **Multiple Data Sources**: Not relying solely on names
2. **Confidence Scoring**: Transparent about uncertainty
3. **Validation Methods**: Cross-checking with multiple approaches
4. **Detailed Logging**: Full traceability of decisions
5. **Guest-Level Data**: Enables manual review and validation

## Future Enhancements

Consider adding:
1. Manual validation interface
2. Active learning from corrections
3. Additional demographic APIs
4. Podcast-specific extraction rules
5. Batch photo processing optimization