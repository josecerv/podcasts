#!/usr/bin/env python3
"""
Validation script to test the enhanced guest extraction pipeline
"""

import sys
import os
import json
import pandas as pd
from datetime import datetime
import logging

# Import the enhanced modules
try:
    from post_intervention_guest_enhanced import (
        extract_guests_with_patterns,
        extract_guests_with_ner,
        extract_guest_context,
        fetch_rss_with_retry,
        clean_text
    )
    from guest_enrichment import GuestEnricher
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Error importing enhanced modules: {e}")
    MODULES_AVAILABLE = False

def test_guest_extraction():
    """Test the guest extraction methods with sample data"""
    print("\n=== Testing Guest Extraction ===")
    
    test_cases = [
        {
            "title": "Episode 123: Interview with Dr. Jane Smith from Harvard",
            "description": "Today we're joined by Dr. Jane Smith, professor at Harvard University and author of 'The Future of AI'. She discusses her latest research on machine learning."
        },
        {
            "title": "Special Guest: John Doe, CEO of TechCorp",
            "description": "Featuring John Doe, founder and CEO of TechCorp. John shares insights from his new book 'Startup Success' and talks about entrepreneurship."
        },
        {
            "title": "Marketing Strategies with Sarah Johnson",
            "description": "Sarah Johnson joins us to discuss modern marketing. Sarah is the director of marketing at Global Industries and has 20 years of experience."
        }
    ]
    
    for i, test in enumerate(test_cases):
        print(f"\nTest Case {i+1}:")
        print(f"Title: {test['title']}")
        
        # Test pattern extraction
        pattern_guests = extract_guests_with_patterns(test['title'], test['description'])
        print(f"Pattern extraction found: {len(pattern_guests)} guests")
        for guest in pattern_guests:
            print(f"  - {guest['name']} (confidence: {guest['confidence']:.2f})")
        
        # Test NER extraction
        import spacy
        nlp = spacy.load("en_core_web_sm")
        ner_guests = extract_guests_with_ner(test['title'] + " " + test['description'])
        print(f"NER extraction found: {len(ner_guests)} guests")
        for guest in ner_guests:
            print(f"  - {guest['name']} (confidence: {guest['confidence']:.2f})")
        
        # Test context extraction
        all_guests = pattern_guests + ner_guests
        unique_names = set(g['name'] for g in all_guests)
        
        print("Context extraction:")
        for name in unique_names:
            context = extract_guest_context(test['title'] + " " + test['description'], name)
            if any(context.values()):
                print(f"  {name}:")
                for key, value in context.items():
                    if value:
                        print(f"    - {key}: {value}")

def test_rss_fetching():
    """Test RSS fetching with a sample feed"""
    print("\n=== Testing RSS Fetching ===")
    
    # Test with a known reliable RSS feed
    test_feeds = [
        {
            "podcastID": "test_001",
            "rss": "https://feeds.simplecast.com/54nAGcIl",  # The Daily from NYT
            "completion_timestamp": pd.Timestamp("2024-01-01", tz='UTC')
        }
    ]
    
    for feed_info in test_feeds:
        print(f"\nTesting feed: {feed_info['rss']}")
        result = fetch_rss_with_retry(feed_info)
        
        if result and hasattr(result, 'entries'):
            print(f"✓ Successfully fetched feed")
            print(f"  Found {len(result.entries)} entries")
            if result.entries:
                print(f"  Latest episode: {result.entries[0].get('title', 'No title')}")
        else:
            print(f"✗ Failed to fetch feed")

def analyze_previous_run():
    """Analyze results from previous run if available"""
    print("\n=== Analyzing Previous Run Results ===")
    
    files_to_check = [
        ("RSS Success Log", "rss_fetch_success_log.csv"),
        ("Enhanced Classifications", "post_intervention_guest_classification_enhanced.jsonl"),
        ("Guest-Level Data", "post_intervention_guests_enriched.csv"),
        ("Aggregated Summary", "post_intervention_guest_summary_enhanced.csv")
    ]
    
    for name, filename in files_to_check:
        if os.path.exists(filename):
            print(f"\n{name} ({filename}):")
            
            if filename.endswith('.csv'):
                try:
                    df = pd.read_csv(filename)
                    print(f"  Rows: {len(df)}")
                    print(f"  Columns: {list(df.columns)}")
                    
                    # Special analysis for RSS success log
                    if "rss_fetch_success_log" in filename and 'status' in df.columns:
                        success_rate = (df['status'] == 'success').mean() * 100
                        print(f"  RSS Success Rate: {success_rate:.2f}%")
                    
                    # Special analysis for guest data
                    if "guests_enriched" in filename:
                        if 'gender' in df.columns:
                            print(f"  Gender distribution: {df['gender'].value_counts().to_dict()}")
                        if 'is_urm' in df.columns:
                            urm_rate = df['is_urm'].mean() * 100
                            print(f"  URM rate: {urm_rate:.2f}%")
                        
                except Exception as e:
                    print(f"  Error reading file: {e}")
                    
            elif filename.endswith('.jsonl'):
                try:
                    count = 0
                    high_quality = 0
                    with open(filename, 'r') as f:
                        for line in f:
                            count += 1
                            try:
                                data = json.loads(line)
                                if data.get('extraction_quality') == 'high':
                                    high_quality += 1
                            except:
                                pass
                    print(f"  Total episodes: {count}")
                    if count > 0:
                        print(f"  High quality extractions: {high_quality} ({high_quality/count*100:.1f}%)")
                except Exception as e:
                    print(f"  Error reading file: {e}")
        else:
            print(f"\n{name} not found - run the pipeline first")

def test_enrichment():
    """Test guest enrichment functionality"""
    print("\n=== Testing Guest Enrichment ===")
    
    if not MODULES_AVAILABLE:
        print("Enhanced modules not available for testing")
        return
    
    # Check if API keys are available
    api_keys = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "SERPAPI_KEY": os.getenv("SERPAPI_KEY"),
        "GOOGLE_SEARCH_API_KEY": os.getenv("GOOGLE_SEARCH_API_KEY")
    }
    
    available_apis = [k for k, v in api_keys.items() if v]
    print(f"Available APIs: {', '.join(available_apis) if available_apis else 'None'}")
    
    if not available_apis:
        print("No API keys found - enrichment features will be limited")
        return
    
    # Test with a well-known person
    test_guest = {
        "name": "Elon Musk",
        "context": {
            "affiliation": "Tesla",
            "title": "CEO",
            "book": None
        }
    }
    
    print(f"\nTesting enrichment for: {test_guest['name']}")
    print("(This is a test with a well-known figure to validate the system)")
    
    try:
        enricher = GuestEnricher()
        result = enricher.enrich_guest(test_guest['name'], test_guest['context'])
        
        print("\nEnrichment results:")
        print(f"  Web search performed: {result.get('web_search_performed', False)}")
        print(f"  Photo analysis performed: {result.get('photo_analysis_performed', False)}")
        
        if result.get('photo_gender'):
            print(f"  Photo-based gender: {result['photo_gender']} (confidence: {result['photo_gender_confidence']:.2f})")
        if result.get('photo_race'):
            print(f"  Photo-based race: {result['photo_race']} (confidence: {result['photo_race_confidence']:.2f})")
        if result.get('verified_affiliation'):
            print(f"  Verified affiliation: {result['verified_affiliation']}")
            
    except Exception as e:
        print(f"Error during enrichment test: {e}")

def generate_comparison_report():
    """Generate a comparison report between original and enhanced results"""
    print("\n=== Generating Comparison Report ===")
    
    # Check for both original and enhanced output files
    original_file = "post_intervention_guest_summary_full.csv"
    enhanced_file = "post_intervention_guest_summary_enhanced.csv"
    
    if os.path.exists(original_file) and os.path.exists(enhanced_file):
        try:
            df_original = pd.read_csv(original_file)
            df_enhanced = pd.read_csv(enhanced_file)
            
            print("\nComparison Summary:")
            print(f"Original podcasts: {len(df_original)}")
            print(f"Enhanced podcasts: {len(df_enhanced)}")
            
            # Compare guest counts
            if 'post_total_guests' in df_original.columns and 'post_total_guests' in df_enhanced.columns:
                orig_guests = df_original['post_total_guests'].sum()
                enh_guests = df_enhanced['post_total_guests'].sum()
                improvement = ((enh_guests - orig_guests) / orig_guests * 100) if orig_guests > 0 else 0
                
                print(f"\nTotal guests found:")
                print(f"  Original: {orig_guests}")
                print(f"  Enhanced: {enh_guests}")
                print(f"  Improvement: {improvement:+.1f}%")
            
            # Save detailed comparison
            comparison_file = "enhancement_comparison_report.csv"
            merged = pd.merge(
                df_original, 
                df_enhanced, 
                on='podcastID', 
                suffixes=('_original', '_enhanced'),
                how='outer'
            )
            
            merged.to_csv(comparison_file, index=False)
            print(f"\nDetailed comparison saved to: {comparison_file}")
            
        except Exception as e:
            print(f"Error generating comparison: {e}")
    else:
        print("Both original and enhanced output files needed for comparison")
        print(f"Original file exists: {os.path.exists(original_file)}")
        print(f"Enhanced file exists: {os.path.exists(enhanced_file)}")

def main():
    """Main validation function"""
    print("Enhanced Pipeline Validation Tool")
    print("=" * 60)
    
    if not MODULES_AVAILABLE:
        print("\nWarning: Enhanced modules not fully available.")
        print("Some tests will be skipped.")
    
    # Run tests
    test_guest_extraction()
    test_rss_fetching()
    test_enrichment()
    analyze_previous_run()
    generate_comparison_report()
    
    print("\n" + "=" * 60)
    print("Validation complete!")
    print("\nNext steps:")
    print("1. Review the test results above")
    print("2. If all tests pass, run the full pipeline:")
    print("   python post-intervention-guest-enhanced.py")
    print("3. Monitor the RSS success rate and extraction quality")
    print("4. Review guest-level data for accuracy")

if __name__ == "__main__":
    main()