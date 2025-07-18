# Podcast Guest Demographics Analysis

This script analyzes podcast episodes to identify guest demographics (gender and URM status) using advanced LLM-based extraction and demographic classification techniques.

## Features

- **LLM-Based Guest Extraction**: Uses GPT-4 to accurately extract guest names with full context (title, company, pronouns, works)
- **Intelligent Web Search**: Leverages extracted context for targeted searches to find additional demographic information
- **Multi-Method Demographics**: Combines pronouns, name analysis, professional affiliations, and web search for classification
- **Knowledge Base**: Maintains a persistent database of guest information across podcasts
- **Podcast Theme Analysis**: Identifies demographic-focused podcasts to improve classification accuracy

## Requirements

```bash
pip install openai feedparser requests python-dotenv
```

## Setup

1. Create a `.env` file in the parent directory with your API keys:
```
OPENAI_API_KEY=your_openai_api_key
SERP_API_KEY=your_serp_api_key  # Optional but recommended for better accuracy
```

2. Ensure you have the `podcast_rss_export.csv` file with columns:
   - `podcastID`: Unique identifier
   - `rss`: RSS feed URL
   - `latest_response_time`: ISO format timestamp
   - `treatment`: Treatment group (optional)

## Usage

```bash
# Run the analysis
./run_analysis.sh

# Or directly with Python
python3 guest_analysis.py
```

## Output Files

- `guest_analysis_summary.csv`: Podcast-level summary statistics
- `guest_analysis_episodes.csv`: Detailed guest-level demographics with evidence
- `guest_knowledge_base_summary.csv`: Aggregated guest information across all podcasts
- `guest_analysis.log`: Detailed processing log

## How It Works

1. **Guest Extraction**: Each episode description is analyzed by GPT-4 to extract only confirmed guests (not hosts) with their professional context

2. **Demographic Analysis**:
   - Uses pronouns from episode descriptions for gender
   - Analyzes names using Census data and statistical patterns
   - Searches for professional affiliations (HBCUs, Hispanic organizations)
   - Performs targeted web searches using extracted context
   - Final validation through LLM review

3. **Confidence Scoring**: Each classification includes a confidence score based on the strength of evidence

## Analysis Window

Per the pre-registered analysis plan:
- 1 month delay after survey completion
- 3 month measurement period
- Example: Survey completed March 1 → Analysis period April 1 - June 30

## Accuracy Features

- Filters out hosts, co-hosts, and show team members
- Requires minimum 80% confidence for guest extraction
- Validates classifications through multiple independent methods
- Maintains knowledge base to improve accuracy over time

## Performance

- Processes ~100 podcasts per hour (with web search enabled)
- Caches results to avoid redundant API calls
- Supports resuming from checkpoints if interrupted

## Classification Methods

### Gender
- Pronouns (he/she/they) from descriptions
- Name pattern analysis
- Web search for pronouns in online profiles

### Race/Ethnicity (URM = Black or Hispanic only)
- HBCU affiliations (60+ institutions)
- HSI affiliations (20+ institutions)
- Professional associations (NABJ, NAHJ, NSBE, etc.)
- Census surname data for Hispanic names
- Explicit mentions ("first Black CEO", "Latina founder")
- Book titles and professional context

## Troubleshooting

**Low confidence scores**: Add SERP_API_KEY for web search enrichment

**No guests found**: Check that episodes fall within the analysis window

**API errors**: Verify API keys are valid and have sufficient credits