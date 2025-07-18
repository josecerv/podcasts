#!/bin/bash

# Optimized Podcast Guest Analysis Runner
# Features: 20 parallel threads, podcast-level caching, batch processing

echo "======================================"
echo "Starting OPTIMIZED Guest Analysis"
echo "======================================"

# Load environment variables from parent directory
if [ -f "../.env" ]; then
    export $(cat ../.env | xargs)
    echo "✓ Environment variables loaded"
else
    echo "⚠ Warning: ../.env file not found"
    echo "Please create it with your API keys:"
    echo "  OPENAI_API_KEY=your-key-here"
    echo "  SERP_API_KEY=your-key-here (optional)"
    exit 1
fi

# Check if OpenAI key exists
if ! grep -q "OPENAI_API_KEY" ../.env; then
    echo "❌ Error: OPENAI_API_KEY not found in ../.env"
    exit 1
fi

# Check for SerpAPI key
if grep -q "SERP_API_KEY" ../.env && [ ! -z "$SERP_API_KEY" ]; then
    echo "✓ SerpAPI key found - web enrichment enabled"
else
    echo "ℹ SerpAPI key not found - web enrichment disabled"
fi

# Check for required file
if [ ! -f "podcast_rss_export.csv" ]; then
    echo "❌ Error: podcast_rss_export.csv not found"
    exit 1
fi

echo "✓ Input file found"
echo ""

# Run the optimized analysis
echo "Starting OPTIMIZED parallel analysis..."
echo "Major improvements:"
echo "- 30 parallel workers (optimized for Ryzen 7 7900X)"
echo "- Podcast-level caching with batch episode processing"
echo "- SerpAPI web search with caching for better demographics"
echo "- Forced binary classification (Male/Female, 4 race categories)"
echo "- Institutional affiliation detection (HBCUs, HSIs)"
echo "- Correct time window: 1-4 months AFTER survey completion"
echo "Estimated time: 4-6 hours for ~6,000 podcasts"
echo ""

python3 guest_analysis.py

echo ""
echo "======================================"
echo "Analysis complete!"
echo "Check the output files:"
echo "- guest_analysis_summary.csv"
echo "- guest_analysis_episodes.csv"
echo "- guest_knowledge_base_summary.csv"
echo "- guest_analysis.log"
echo "======================================"