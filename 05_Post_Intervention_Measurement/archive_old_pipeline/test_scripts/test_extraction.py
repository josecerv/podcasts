#!/usr/bin/env python3
"""
Test Guest Extraction on Sample Podcasts
Allows manual verification of extraction quality
"""

import os
import sys
import json
import csv
import re
import logging
from datetime import datetime, timedelta
import random
import time
import requests
import feedparser
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path(__file__).parent.parent / '.env')

# Configuration
SAMPLE_SIZE = 50
EPISODES_PER_PODCAST = 3  # Test a few episodes per podcast
RSS_TIMEOUT = 30

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Guest extraction patterns
GUEST_PATTERNS = [
    r"(?:featuring|feature|feat\.?|with guest[s]?|guest[s]?:|interview with|joined by|welcoming|special guest[s]?|our guest[s]?|today'?s guest[s]?)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})",
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s+(?:joins us|is here|stops by|visits|shares|discusses|talks about)",
    r"(?:author|professor|dr\.?|expert|ceo|founder|director|coach|consultant)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})",
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}),\s+(?:author of|professor at|ceo of|founder of|director of)",
]

# Test prompt - same as production
GUEST_ANALYSIS_PROMPT = """Analyze this podcast episode to identify and classify guests.

IMPORTANT DEFINITIONS:
- GUEST: Someone interviewed or featured in the episode who is NOT a regular host, co-host, or show member
- URM (Underrepresented Minority): Black/African American or Hispanic/Latino ONLY

Episode Title: {title}
Episode Description: {description}

Potential guests found: {potential_guests}

Task:
1. Validate which names are actual guests (not hosts)
2. Identify any missed guests
3. For each guest, determine:
   - Gender (Male/Female/Unknown)
   - Race/Ethnicity (focus on identifying Black or Hispanic/Latino for URM)

Output format (JSON):
{{
    "guests": [
        {{
            "name": "Full Name",
            "gender": "Male/Female/Unknown",
            "is_urm": true/false,
            "confidence": 0.0-1.0
        }}
    ],
    "total_guests": number,
    "female_guests": number,
    "urm_guests": number
}}"""


def extract_guest_patterns(text: str) -> list:
    """Extract potential guest names using regex patterns."""
    potential_guests = set()
    
    for pattern in GUEST_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            name = match.strip()
            if len(name.split()) >= 2 and len(name) < 50:
                potential_guests.add(name)
    
    return list(potential_guests)


def fetch_sample_episodes(rss_url: str, after_date: datetime, before_date: datetime, limit: int = 3):
    """Fetch a sample of episodes from RSS feed."""
    episodes = []
    
    try:
        response = requests.get(rss_url, timeout=RSS_TIMEOUT, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        feed = feedparser.parse(response.content)
        
        for entry in feed.entries[:20]:  # Check more entries to find ones in date range
            pub_date = None
            for date_field in ['published_parsed', 'updated_parsed', 'created_parsed']:
                if hasattr(entry, date_field) and getattr(entry, date_field):
                    pub_date = datetime(*getattr(entry, date_field)[:6])
                    break
            
            if not pub_date:
                continue
                
            if after_date <= pub_date <= before_date:
                title = entry.get('title', '')
                
                description = ''
                for field in ['description', 'summary', 'content', 'itunes_summary']:
                    if hasattr(entry, field):
                        value = getattr(entry, field)
                        if isinstance(value, list) and value:
                            description = value[0].get('value', '')
                        elif isinstance(value, str):
                            description = value
                        if description:
                            break
                
                episodes.append({
                    'title': title,
                    'description': description,
                    'published_date': pub_date,
                    'raw_entry': entry  # Keep raw data for inspection
                })
                
                if len(episodes) >= limit:
                    break
    
    except Exception as e:
        logger.error(f"Error fetching RSS: {e}")
    
    return episodes


def test_single_episode(episode_data: dict, podcast_id: str):
    """Test extraction on a single episode and show results."""
    print("\n" + "="*80)
    print(f"PODCAST: {podcast_id}")
    print(f"DATE: {episode_data['published_date'].strftime('%Y-%m-%d')}")
    print("="*80)
    
    print(f"\nTITLE: {episode_data['title'][:200]}")
    print(f"\nDESCRIPTION (first 500 chars):")
    print("-" * 40)
    print(episode_data['description'][:500])
    print("-" * 40)
    
    # Extract guests using patterns
    combined_text = f"{episode_data['title']} {episode_data['description']}"
    potential_guests = extract_guest_patterns(combined_text)
    
    print(f"\nPATTERN EXTRACTION FOUND: {potential_guests}")
    
    # Test with LLM
    try:
        prompt = GUEST_ANALYSIS_PROMPT.format(
            title=episode_data['title'],
            description=episode_data['description'][:1500],  # Limit description length
            potential_guests=", ".join(potential_guests) if potential_guests else "None found"
        )
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert at analyzing podcast content."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=500
        )
        
        result = json.loads(response.choices[0].message.content)
        
        print(f"\nLLM ANALYSIS:")
        print(f"Total guests: {result['total_guests']}")
        print(f"Female guests: {result['female_guests']}")
        print(f"URM guests: {result['urm_guests']}")
        
        if result['guests']:
            print(f"\nGuest details:")
            for guest in result['guests']:
                print(f"  - {guest['name']} (Gender: {guest['gender']}, URM: {guest['is_urm']}, Confidence: {guest['confidence']})")
        
    except Exception as e:
        print(f"\nLLM Error: {e}")
    
    return {
        'episode': episode_data,
        'pattern_guests': potential_guests,
        'llm_result': result if 'result' in locals() else None
    }


def main():
    """Run test extraction on sample podcasts."""
    print("PODCAST GUEST EXTRACTION TEST")
    print("=" * 80)
    
    # Check API key
    if not os.getenv('OPENAI_API_KEY'):
        print("ERROR: OpenAI API key not found in ../.env")
        sys.exit(1)
    
    # Load podcast data
    podcasts = []
    with open('podcast_rss_export.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['rss_url'] and row['rss_url'].startswith('http'):
                podcasts.append(row)
    
    # Sample podcasts
    sample_podcasts = random.sample(podcasts, min(SAMPLE_SIZE, len(podcasts)))
    print(f"\nTesting {len(sample_podcasts)} random podcasts...")
    
    # Test results storage
    test_results = []
    issues_found = {
        'no_episodes': 0,
        'no_guests_found': 0,
        'pattern_mismatch': 0,
        'potential_host_confusion': 0
    }
    
    # Process sample
    for i, podcast in enumerate(sample_podcasts):
        podcast_id = podcast['podcast_id']
        rss_url = podcast['rss_url']
        response_time = datetime.fromisoformat(podcast['latest_response_time'])
        
        # Calculate window
        window_start = response_time
        window_end = response_time + timedelta(days=90)
        
        print(f"\n\n[{i+1}/{len(sample_podcasts)}] Testing podcast {podcast_id}")
        print(f"Window: {window_start.date()} to {window_end.date()}")
        
        # Fetch episodes
        episodes = fetch_sample_episodes(rss_url, window_start, window_end, EPISODES_PER_PODCAST)
        
        if not episodes:
            print("No episodes found in window!")
            issues_found['no_episodes'] += 1
            continue
        
        # Test each episode
        for episode in episodes:
            result = test_single_episode(episode, podcast_id)
            test_results.append(result)
            
            # Check for issues
            if result['llm_result']:
                if result['llm_result']['total_guests'] == 0:
                    issues_found['no_guests_found'] += 1
                
                # Check if pattern extraction missed guests that LLM found
                llm_names = [g['name'] for g in result['llm_result']['guests']]
                pattern_names = result['pattern_guests']
                
                if llm_names and not any(pn in ' '.join(llm_names) for pn in pattern_names):
                    issues_found['pattern_mismatch'] += 1
                    print("\n⚠️  PATTERN MISMATCH: LLM found guests that patterns missed!")
        
        # Rate limit
        time.sleep(1)
        
        # User checkpoint every 10 podcasts
        if (i + 1) % 10 == 0:
            input(f"\nProcessed {i+1} podcasts. Press Enter to continue...")
    
    # Summary
    print("\n\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Total episodes tested: {len(test_results)}")
    print(f"\nIssues found:")
    for issue, count in issues_found.items():
        print(f"  - {issue}: {count}")
    
    # Save detailed results
    with open('test_extraction_results.json', 'w') as f:
        json.dump(test_results, f, indent=2, default=str)
    
    print(f"\nDetailed results saved to: test_extraction_results.json")
    print("\nReview the results above to determine if patterns and prompts need optimization.")


if __name__ == "__main__":
    main()