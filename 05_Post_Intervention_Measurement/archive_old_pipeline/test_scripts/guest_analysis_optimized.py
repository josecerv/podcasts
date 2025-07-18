#!/usr/bin/env python3
"""
Optimized Podcast Guest Analysis Pipeline
Based on testing and pattern validation
"""

import os
import sys
import json
import csv
import re
import logging
from datetime import datetime, timedelta
from collections import defaultdict
import time
from typing import Dict, List, Tuple, Optional
import requests
import feedparser
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path(__file__).parent.parent / '.env')

# Configuration
RSS_TIMEOUT = 30
API_RETRY_ATTEMPTS = 3
API_RETRY_DELAY = 2
ANALYSIS_WINDOW_DAYS = 90  # 3 months
BATCH_SIZE = 10

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('guest_analysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Improved guest extraction patterns with better boundaries
GUEST_PATTERNS = [
    # Direct guest mentions with word boundaries
    r"\b(?:featuring|feature|feat\.?|with guest[s]?|guest[s]?:|interview with|interviewed|joined by|welcoming|special guest[s]?|our guest[s]?|today'?s guest[s]?)\s+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?(?:[a-z]+)?(?:\s+[A-Z][a-z]+){1,2})\b",
    
    # Name + action patterns with boundaries
    r"\b([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?(?:[a-z]+)?(?:\s+[A-Z][a-z]+){1,2})\s+(?:joins us|is here|stops by|visits us|shares|discusses|talks about|tells us)\b",
    
    # Title + name patterns with better capture
    r"\b(?:author|professor|prof\.?|dr\.?|expert|ceo|founder|director|coach|consultant|journalist|reporter)\s+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?(?:[a-z]+)?(?:\s+[A-Z][a-z]+){1,2})(?:\s*,|\s+(?:joins|shares|discusses|talks))\b",
    
    # Name + title patterns
    r"\b([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?(?:[a-z]+)?(?:\s+[A-Z][a-z]+){1,2}),\s+(?:author of|professor at|ceo of|founder of|director of|host of)\b",
    
    # Interview/conversation patterns
    r"\b(?:conversation with|chat with|talk with|speaking with|interview with)\s+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?(?:[a-z]+)?(?:\s+[A-Z][a-z]+){1,2})\b",
]

# Common false positive filters
FALSE_POSITIVE_FILTERS = [
    r'\b(?:host|co-host|hosts|cohost)\b',
    r'\b(?:episode|series|show|podcast|part)\b',
    r'\b(?:the|and|with|from|about|this|that)\b',
    r'^(?:of|at|in|on|for|by)\s',  # Starting with prepositions
]

# Optimized LLM prompt based on testing
GUEST_ANALYSIS_PROMPT = """Analyze this podcast episode to identify and classify guests.

CRITICAL RULES:
1. GUEST = Someone interviewed/featured who is NOT a regular host, co-host, or show member
2. DO NOT count hosts, co-hosts, or show team members as guests
3. URM = ONLY Black/African American or Hispanic/Latino (not Asian, Native American, etc.)
4. Look for context clues: "joins us", "interview with", "special guest", "author of", etc.

Episode Title: {title}
Episode Description: {description}

Pattern extraction found these potential names: {potential_guests}

Carefully analyze:
1. Which names are actual GUESTS (not hosts)?
2. Are there any guests mentioned that weren't caught by patterns?
3. For each verified guest:
   - What is their gender based on name/context?
   - Are they likely Black or Hispanic based on name/context/affiliation?

IMPORTANT: Many podcasts have NO guests - just hosts talking. Be conservative.

Output JSON format:
{{
    "guests": [
        {{
            "name": "Full Name",
            "gender": "Male/Female/Unknown",
            "is_urm": true/false,
            "confidence": 0.0-1.0,
            "context": "Brief context about why they're a guest"
        }}
    ],
    "total_guests": number,
    "female_guests": number,
    "urm_guests": number,
    "hosts_mentioned": ["Names identified as hosts, not guests"]
}}"""


def clean_text(text: str) -> str:
    """Clean HTML and normalize text."""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s\-.,;:!?\'"()]', ' ', text)
    return text.strip()


def extract_guest_patterns(text: str) -> List[str]:
    """Extract potential guest names using improved regex patterns."""
    text = clean_text(text)
    potential_guests = set()
    
    for pattern in GUEST_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            # Clean up the match
            name = match.strip()
            
            # Apply filters
            is_valid = True
            
            # Check length and word count
            words = name.split()
            if len(words) < 2 or len(words) > 4 or len(name) > 50:
                is_valid = False
            
            # Check for false positives
            for filter_pattern in FALSE_POSITIVE_FILTERS:
                if re.search(filter_pattern, name, re.IGNORECASE):
                    is_valid = False
                    break
            
            # Check if it's mostly lowercase (likely not a name)
            if name.islower():
                is_valid = False
            
            if is_valid:
                potential_guests.add(name)
    
    return list(potential_guests)


def analyze_guests_with_llm(title: str, description: str, potential_guests: List[str]) -> Dict:
    """Use LLM to validate and classify guests with improved prompt."""
    try:
        # Limit description length to avoid token limits
        description = description[:2000]
        
        prompt = GUEST_ANALYSIS_PROMPT.format(
            title=title,
            description=description,
            potential_guests=", ".join(potential_guests) if potential_guests else "None found by patterns"
        )
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert at analyzing podcast content. Be conservative - only identify clear guests, not hosts."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=600
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # Validate the result
        if 'guests' not in result:
            result['guests'] = []
        if 'total_guests' not in result:
            result['total_guests'] = len(result['guests'])
        if 'female_guests' not in result:
            result['female_guests'] = sum(1 for g in result['guests'] if g.get('gender') == 'Female')
        if 'urm_guests' not in result:
            result['urm_guests'] = sum(1 for g in result['guests'] if g.get('is_urm', False))
        
        return result
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in LLM response: {e}")
        return {
            "guests": [],
            "total_guests": 0,
            "female_guests": 0,
            "urm_guests": 0,
            "error": "JSON decode error"
        }
    except Exception as e:
        logger.error(f"LLM analysis error: {e}")
        return {
            "guests": [],
            "total_guests": 0,
            "female_guests": 0,
            "urm_guests": 0,
            "error": str(e)
        }


def fetch_rss_episodes(rss_url: str, after_date: datetime, before_date: datetime) -> List[Dict]:
    """Fetch episodes from RSS feed within the specified date window."""
    episodes = []
    
    try:
        # Set timeout for RSS fetch
        response = requests.get(rss_url, timeout=RSS_TIMEOUT, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Parse the feed
        feed = feedparser.parse(response.content)
        
        if feed.bozo:
            logger.warning(f"Feed parsing issue for {rss_url}: {feed.bozo_exception}")
            
        for entry in feed.entries:
            # Try to get publication date
            pub_date = None
            for date_field in ['published_parsed', 'updated_parsed', 'created_parsed']:
                if hasattr(entry, date_field) and getattr(entry, date_field):
                    pub_date = datetime(*getattr(entry, date_field)[:6])
                    break
            
            if not pub_date:
                continue
                
            # Check if within analysis window
            if after_date <= pub_date <= before_date:
                # Extract episode data
                title = entry.get('title', '')
                
                # Get description from various possible fields
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
                    'published_date': pub_date
                })
        
        logger.info(f"Found {len(episodes)} episodes in analysis window for {rss_url}")
        
    except requests.exceptions.Timeout:
        logger.error(f"RSS fetch timeout for {rss_url}")
    except Exception as e:
        logger.error(f"RSS fetch error for {rss_url}: {e}")
    
    return episodes


def analyze_podcast(podcast_data: Dict) -> Dict:
    """Analyze a single podcast for guest demographics."""
    podcast_id = podcast_data['podcastID']
    rss_url = podcast_data['rss']
    response_time = datetime.fromisoformat(podcast_data['latest_response_time'])
    
    # Calculate analysis window (3 months after response)
    window_start = response_time
    window_end = response_time + timedelta(days=ANALYSIS_WINDOW_DAYS)
    
    logger.info(f"Analyzing podcast {podcast_id} - Window: {window_start.date()} to {window_end.date()}")
    
    # Fetch episodes
    episodes = fetch_rss_episodes(rss_url, window_start, window_end)
    
    if not episodes:
        logger.warning(f"No episodes found for podcast {podcast_id}")
        return {
            'podcast_id': podcast_id,
            'treatment': podcast_data.get('treatment', ''),
            'episodes_analyzed': 0,
            'total_guests': 0,
            'female_guests': 0,
            'urm_guests': 0,
            'female_percentage': 0.0,
            'urm_percentage': 0.0
        }
    
    # Analyze each episode
    total_guests = 0
    total_female = 0
    total_urm = 0
    episode_details = []
    
    for episode in episodes:
        # Extract potential guests
        combined_text = f"{episode['title']} {episode['description']}"
        potential_guests = extract_guest_patterns(combined_text)
        
        # Analyze with LLM
        analysis = analyze_guests_with_llm(
            episode['title'], 
            episode['description'], 
            potential_guests
        )
        
        # Track totals
        total_guests += analysis['total_guests']
        total_female += analysis['female_guests']
        total_urm += analysis['urm_guests']
        
        # Store episode details
        episode_details.append({
            'title': episode['title'],
            'date': episode['published_date'],
            'guests': analysis.get('guests', []),
            'total': analysis['total_guests'],
            'female': analysis['female_guests'],
            'urm': analysis['urm_guests'],
            'hosts_mentioned': analysis.get('hosts_mentioned', [])
        })
    
    # Calculate percentages
    female_pct = (total_female / total_guests * 100) if total_guests > 0 else 0.0
    urm_pct = (total_urm / total_guests * 100) if total_guests > 0 else 0.0
    
    return {
        'podcast_id': podcast_id,
        'treatment': podcast_data.get('treatment', ''),
        'episodes_analyzed': len(episodes),
        'total_guests': total_guests,
        'female_guests': total_female,
        'urm_guests': total_urm,
        'female_percentage': round(female_pct, 2),
        'urm_percentage': round(urm_pct, 2),
        'window_start': window_start.isoformat(),
        'window_end': window_end.isoformat(),
        'episode_details': episode_details
    }


def main():
    """Main analysis pipeline."""
    logger.info("Starting Optimized Podcast Guest Analysis Pipeline")
    
    # Check for API key
    if not os.getenv('OPENAI_API_KEY'):
        logger.error("OpenAI API key not found in ../.env file")
        sys.exit(1)
    
    # Load podcast data
    input_file = 'podcast_rss_export.csv'
    if not os.path.exists(input_file):
        logger.error(f"Input file {input_file} not found")
        sys.exit(1)
    
    # Read podcasts
    podcasts = []
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['rss'] and row['rss'].startswith('http'):
                podcasts.append(row)
    
    logger.info(f"Loaded {len(podcasts)} podcasts for analysis")
    
    # Process podcasts
    results = []
    episode_results = []
    
    for i, podcast in enumerate(podcasts):
        try:
            logger.info(f"Processing {i+1}/{len(podcasts)}: {podcast['podcastID']}")
            
            result = analyze_podcast(podcast)
            results.append(result)
            
            # Flatten episode details for detailed output
            for episode in result.get('episode_details', []):
                episode_results.append({
                    'podcast_id': result['podcast_id'],
                    'treatment': result['treatment'],
                    'episode_title': episode['title'],
                    'episode_date': episode['date'],
                    'total_guests': episode['total'],
                    'female_guests': episode['female'],
                    'urm_guests': episode['urm'],
                    'guest_names': ', '.join([g['name'] for g in episode['guests']]),
                    'hosts_identified': ', '.join(episode.get('hosts_mentioned', []))
                })
            
            # Progress update
            if (i + 1) % 10 == 0:
                logger.info(f"Progress: {i+1}/{len(podcasts)} podcasts processed")
                
        except Exception as e:
            logger.error(f"Error processing podcast {podcast['podcastID']}: {e}")
            continue
    
    # Write summary results
    summary_file = 'guest_analysis_summary.csv'
    with open(summary_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['podcast_id', 'treatment', 'episodes_analyzed', 'total_guests', 
                     'female_guests', 'urm_guests', 'female_percentage', 
                     'urm_percentage', 'window_start', 'window_end']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({k: result.get(k, '') for k in fieldnames})
    
    # Write episode details
    episodes_file = 'guest_analysis_episodes.csv'
    with open(episodes_file, 'w', newline='', encoding='utf-8') as f:
        if episode_results:
            fieldnames = list(episode_results[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(episode_results)
    
    # Calculate overall statistics
    total_podcasts = len(results)
    podcasts_with_guests = sum(1 for r in results if r['total_guests'] > 0)
    total_all_guests = sum(r['total_guests'] for r in results)
    total_all_female = sum(r['female_guests'] for r in results)
    total_all_urm = sum(r['urm_guests'] for r in results)
    
    # Calculate by treatment group
    treatment_stats = defaultdict(lambda: {'count': 0, 'guests': 0, 'female': 0, 'urm': 0})
    for result in results:
        treatment = result.get('treatment', 'unknown')
        treatment_stats[treatment]['count'] += 1
        treatment_stats[treatment]['guests'] += result['total_guests']
        treatment_stats[treatment]['female'] += result['female_guests']
        treatment_stats[treatment]['urm'] += result['urm_guests']
    
    logger.info("\n" + "="*50)
    logger.info("ANALYSIS COMPLETE")
    logger.info("="*50)
    logger.info(f"Podcasts analyzed: {total_podcasts}")
    logger.info(f"Podcasts with guests: {podcasts_with_guests}")
    logger.info(f"Total guests identified: {total_all_guests}")
    logger.info(f"Female guests: {total_all_female} ({total_all_female/total_all_guests*100:.1f}%)" if total_all_guests > 0 else "Female guests: 0")
    logger.info(f"URM guests: {total_all_urm} ({total_all_urm/total_all_guests*100:.1f}%)" if total_all_guests > 0 else "URM guests: 0")
    
    logger.info("\nBy Treatment Group:")
    for treatment, stats in treatment_stats.items():
        if stats['guests'] > 0:
            logger.info(f"  {treatment}: {stats['count']} podcasts, {stats['guests']} guests "
                       f"({stats['female']/stats['guests']*100:.1f}% female, "
                       f"{stats['urm']/stats['guests']*100:.1f}% URM)")
    
    logger.info(f"\nResults saved to:")
    logger.info(f"  - {summary_file}")
    logger.info(f"  - {episodes_file}")


if __name__ == "__main__":
    main()