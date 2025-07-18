#!/bin/bash
# Script to run the enhanced podcast guest analysis pipeline

echo "=== Enhanced Podcast Guest Analysis Pipeline ==="
echo "Date: $(date)"
echo ""

# Check environment variables
echo "Checking environment variables..."
if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set"
    echo "Please run: export OPENAI_API_KEY='your-key-here'"
    exit 1
fi

echo "✓ OPENAI_API_KEY is set"

# Optional API keys
if [ -n "$SERPAPI_KEY" ]; then
    echo "✓ SERPAPI_KEY is set (web enrichment enabled)"
else
    echo "⚠ SERPAPI_KEY not set (web enrichment disabled)"
fi

if [ -n "$GOOGLE_SEARCH_API_KEY" ]; then
    echo "✓ GOOGLE_SEARCH_API_KEY is set (alternative web search enabled)"
else
    echo "⚠ GOOGLE_SEARCH_API_KEY not set"
fi

echo ""

# Check input file
if [ ! -f "podcast_rss_export.csv" ]; then
    echo "ERROR: podcast_rss_export.csv not found"
    echo "Please ensure the input file is in the current directory"
    exit 1
fi

echo "✓ Input file found: podcast_rss_export.csv"
echo "  Podcasts to process: $(wc -l < podcast_rss_export.csv)"
echo ""

# Create necessary directories
echo "Creating directories..."
mkdir -p guest_photos enrichment_cache logs output
echo "✓ Directories created"
echo ""

# Backup existing output files
if [ -f "post_intervention_guest_classification_enhanced.jsonl" ]; then
    echo "Backing up existing output files..."
    timestamp=$(date +%Y%m%d_%H%M%S)
    mkdir -p backups/$timestamp
    mv post_intervention_guest_*.* backups/$timestamp/ 2>/dev/null
    mv rss_fetch_success_log.csv backups/$timestamp/ 2>/dev/null
    echo "✓ Backed up to backups/$timestamp/"
fi

echo ""

# Run options
echo "Select run mode:"
echo "1) Test mode (first 100 podcasts)"
echo "2) Full run (all podcasts)"
echo -n "Enter choice (1 or 2): "
read choice

case $choice in
    1)
        echo ""
        echo "Running in TEST MODE (100 podcasts)..."
        # Create test subset
        head -101 podcast_rss_export.csv > podcast_rss_export_test.csv
        
        # Temporarily rename files
        mv podcast_rss_export.csv podcast_rss_export_full.csv
        mv podcast_rss_export_test.csv podcast_rss_export.csv
        
        # Run enhanced pipeline
        python3 post-intervention-guest-enhanced.py
        
        # Restore files
        mv podcast_rss_export.csv podcast_rss_export_test.csv
        mv podcast_rss_export_full.csv podcast_rss_export.csv
        
        echo ""
        echo "Test run complete. Check output files:"
        echo "- post_intervention_guest_classification_enhanced.jsonl"
        echo "- post_intervention_guest_summary_enhanced.csv"
        echo "- post_intervention_guests_enriched.csv"
        echo "- rss_fetch_success_log.csv"
        ;;
        
    2)
        echo ""
        echo "Running FULL PIPELINE..."
        echo "This may take several hours. Press Ctrl+C to cancel."
        echo "Starting in 5 seconds..."
        sleep 5
        
        # Run enhanced pipeline
        python3 post-intervention-guest-enhanced.py
        
        echo ""
        echo "Full run complete!"
        ;;
        
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "=== Post-Processing ==="

# Generate comparison if original results exist
if [ -f "post_intervention_guest_summary_full.csv" ]; then
    echo "Generating comparison with original results..."
    python3 -c "
import pandas as pd
try:
    orig = pd.read_csv('post_intervention_guest_summary_full.csv')
    enhanced = pd.read_csv('post_intervention_guest_summary_enhanced.csv')
    
    print(f'Original podcasts: {len(orig)}')
    print(f'Enhanced podcasts: {len(enhanced)}')
    
    if 'post_total_guests' in orig.columns and 'post_total_guests' in enhanced.columns:
        orig_guests = orig['post_total_guests'].sum()
        enh_guests = enhanced['post_total_guests'].sum()
        improvement = ((enh_guests - orig_guests) / orig_guests * 100) if orig_guests > 0 else 0
        print(f'Original total guests: {orig_guests}')
        print(f'Enhanced total guests: {enh_guests}')
        print(f'Improvement: {improvement:+.1f}%')
except Exception as e:
    print(f'Could not generate comparison: {e}')
"
fi

# Check RSS success rate
if [ -f "rss_fetch_success_log.csv" ]; then
    echo ""
    echo "RSS Fetching Statistics:"
    python3 -c "
import pandas as pd
try:
    df = pd.read_csv('rss_fetch_success_log.csv', names=['podcast_id', 'url', 'status', 'episodes'])
    total = len(df)
    success = len(df[df['status'] == 'success'])
    rate = (success / total * 100) if total > 0 else 0
    print(f'Total attempts: {total}')
    print(f'Successful: {success}')
    print(f'Success rate: {rate:.2f}%')
    if rate >= 95:
        print('✓ RSS success rate meets 95% target!')
    else:
        print('⚠ RSS success rate below 95% target')
except Exception as e:
    print(f'Could not analyze RSS log: {e}')
"
fi

echo ""
echo "=== Pipeline Complete ==="
echo "Check post_intervention_error_log_enhanced.txt for detailed logs"