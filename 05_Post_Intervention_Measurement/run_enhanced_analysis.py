#!/usr/bin/env python3
"""
Enhanced Podcast Guest Analysis Pipeline - Minimal Production Version
Includes web enrichment with SerpAPI and dual percentage calculations
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import socket
import time
import hashlib
import re
import logging
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import concurrent.futures
from tqdm import tqdm
import multiprocessing
import signal
import gc

# Check for feedparser before other imports
try:
    import feedparser
except ImportError:
    print("ERROR: feedparser not installed. Please run: pip install feedparser")
    sys.exit(1)

# Import OpenAI
from openai import OpenAI

# Import enrichment module
from guest_enrichment_module import GuestEnricher

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_analysis.log', mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load .env file from parent directory
from pathlib import Path
env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    with open(env_path, 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                key, value = line.strip().split('=', 1)
                os.environ[key] = value

# Load configuration
with open('config.json', 'r') as f:
    CONFIG = json.load(f)

# Get OpenAI key from environment
openai_key = os.environ.get('OPENAI_API_KEY')
if not openai_key:
    logger.error("OPENAI_API_KEY not found in environment")
    sys.exit(1)

# Set SerpAPI key
os.environ['SERPAPI_KEY'] = CONFIG['api_keys']['serpapi']

# Initialize OpenAI client
client = OpenAI(api_key=openai_key)

# Guest extraction patterns
GUEST_PATTERNS = [
    r"(?:featuring|feature|feat\.?|with guest[s]?|guest[s]?:|interview with|joined by|welcoming|special guest[s]?|our guest[s]?|today'?s guest[s]?)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})",
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s+(?:joins us|is here|stops by|visits|shares|discusses|talks about)",
    r"(?:author|professor|dr\.?|expert|ceo|founder|director|coach|consultant)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})",
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}),\s+(?:author of|professor at|ceo of|founder of|director of)",
]

# Enhanced LLM prompt
GUEST_ANALYSIS_PROMPT = """Analyze this podcast episode to identify and classify guests.

IMPORTANT DEFINITIONS:
- GUEST: Someone interviewed or featured in the episode who is NOT a regular host, co-host, or show member
- URM: Black/African American or Hispanic/Latino ONLY (not Asian, Native American, etc.)

Episode Title: {title}
Episode Description: {description}

Potential guests found by pattern matching:
{potential_guests}

Your task:
1. Validate which names are actual guests (not hosts)
2. Identify any missed guests
3. For each guest, determine:
   - Name
   - Gender (Male/Female/Unknown)
   - Race/Ethnicity (focus on identifying Black or Hispanic/Latino)
   - Confidence level (0.0-1.0)

Consider any context about affiliations, titles, or expertise when making classifications.

Output format (JSON):
{{
    "guests": [
        {{
            "name": "Full Name",
            "gender": "Male/Female/Unknown",
            "race": "White/Black/Hispanic/Asian/Other/Unknown",
            "is_urm": true/false,  // true only if Black or Hispanic
            "confidence": 0.0-1.0,
            "reasoning": "Brief explanation"
        }}
    ],
    "total_guests": number,
    "female_guests": number,
    "urm_guests": number  // Black or Hispanic only
}}"""

def clean_text(text):
    """Clean and normalize text"""
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r'<.*?>', ' ', text)
    text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_guest_patterns(title: str, description: str) -> List[Dict]:
    """Extract potential guests using regex patterns"""
    text = f"{title} {description}"
    text = clean_text(text)
    
    guests = []
    seen = set()
    
    for pattern in GUEST_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            name = match.strip() if isinstance(match, str) else match
            if len(name.split()) >= 2 and name.lower() not in seen:
                seen.add(name.lower())
                guests.append({"name": name, "method": "pattern"})
    
    return guests

def fetch_rss_episodes(podcast_info: Dict) -> List[Dict]:
    """Fetch and filter episodes from RSS feed"""
    rss_url = podcast_info.get('rss')
    podcast_id = podcast_info.get('podcastID')
    completion_timestamp = podcast_info.get('completion_timestamp')
    
    if not all([rss_url, podcast_id, completion_timestamp]):
        return []
    
    episodes = []
    
    for attempt in range(CONFIG['rss_settings']['retry_attempts']):
        try:
            socket.setdefaulttimeout(CONFIG['rss_settings']['timeout'])
            feed = feedparser.parse(rss_url)
            
            if not feed.entries:
                return []
            
            for entry in feed.entries:
                # Get publication date
                pub_struct = entry.get('published_parsed') or entry.get('updated_parsed')
                if not pub_struct:
                    continue
                
                try:
                    pub_date = datetime(*pub_struct[:6], tzinfo=timezone.utc)
                except:
                    continue
                
                # Filter by date
                if pub_date > completion_timestamp and pub_date <= datetime.now(timezone.utc):
                    title = clean_text(entry.get('title', ''))
                    
                    # Get description from various fields
                    desc_parts = []
                    for field in ['summary', 'description', 'content']:
                        if field == 'content' and hasattr(entry, 'content'):
                            for c in entry.content:
                                if hasattr(c, 'value'):
                                    desc_parts.append(c.value)
                        elif hasattr(entry, field):
                            value = getattr(entry, field)
                            if value:
                                desc_parts.append(value)
                    
                    description = clean_text(' '.join(desc_parts))
                    
                    if title or description:
                        episode_id = hashlib.md5(f"{podcast_id}_{title}_{pub_date}".encode()).hexdigest()
                        episodes.append({
                            'podcast_id': podcast_id,
                            'episode_id': episode_id,
                            'title': title,
                            'description': description,
                            'published_date': pub_date
                        })
            
            return episodes
            
        except Exception as e:
            if attempt < CONFIG['rss_settings']['retry_attempts'] - 1:
                time.sleep(CONFIG['rss_settings']['retry_delay'])
            else:
                logger.error(f"Failed to fetch RSS for {podcast_id}: {e}")
    
    return episodes

def analyze_episode_guests(episode: Dict, enricher: Optional[GuestEnricher] = None) -> Dict:
    """Analyze an episode for guests using LLM and optional enrichment"""
    
    # Extract potential guests
    potential_guests = extract_guest_patterns(episode['title'], episode['description'])
    
    # Prepare prompt
    prompt = GUEST_ANALYSIS_PROMPT.format(
        title=episode['title'],
        description=episode['description'][:3000],  # Limit length
        potential_guests=json.dumps(potential_guests, indent=2) if potential_guests else "None found"
    )
    
    # Call LLM
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert at identifying and classifying podcast guests."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # Enrich guests if enricher available and enabled
        if enricher and CONFIG['enrichment_settings']['enable_web_enrichment']:
            enriched_guests = []
            
            for guest in result.get('guests', []):
                # Only enrich if confidence is below threshold
                if guest.get('confidence', 0) < CONFIG['enrichment_settings']['enrichment_confidence_threshold']:
                    enrichment = enricher.enrich_guest(guest['name'])
                    
                    # Update guest info based on enrichment
                    if enrichment.get('enhanced_gender'):
                        guest['gender'] = enrichment['enhanced_gender']
                        guest['confidence'] = max(guest['confidence'], enrichment.get('gender_confidence', 0))
                    
                    if enrichment.get('enhanced_race'):
                        guest['race'] = enrichment['enhanced_race']
                        guest['is_urm'] = enrichment['enhanced_race'] in ['Black', 'Hispanic/Latino']
                        guest['confidence'] = max(guest['confidence'], enrichment.get('race_confidence', 0))
                    
                    guest['enrichment_used'] = True
                else:
                    guest['enrichment_used'] = False
                
                enriched_guests.append(guest)
            
            result['guests'] = enriched_guests
        
        return result
        
    except Exception as e:
        logger.error(f"Error analyzing episode {episode['episode_id']}: {e}")
        return {
            "guests": [],
            "total_guests": 0,
            "female_guests": 0,
            "urm_guests": 0,
            "error": str(e)
        }

def calculate_podcast_metrics(episodes_data: List[Dict]) -> Dict:
    """Calculate both overall and episode-averaged percentages"""
    
    # Overall counts
    total_guests = 0
    total_female = 0
    total_urm = 0
    
    # Per-episode percentages for averaging
    episode_female_pcts = []
    episode_urm_pcts = []
    
    # Episode details
    episodes_analyzed = len(episodes_data)
    episodes_with_guests = 0
    
    for episode in episodes_data:
        ep_total = episode.get('total_guests', 0)
        ep_female = episode.get('female_guests', 0)
        ep_urm = episode.get('urm_guests', 0)
        
        total_guests += ep_total
        total_female += ep_female
        total_urm += ep_urm
        
        if ep_total > 0:
            episodes_with_guests += 1
            # Calculate per-episode percentages
            episode_female_pcts.append((ep_female / ep_total) * 100)
            episode_urm_pcts.append((ep_urm / ep_total) * 100)
    
    # Calculate overall percentages
    overall_female_pct = (total_female / total_guests * 100) if total_guests > 0 else 0
    overall_urm_pct = (total_urm / total_guests * 100) if total_guests > 0 else 0
    
    # Calculate episode-averaged percentages
    avg_episode_female_pct = np.mean(episode_female_pcts) if episode_female_pcts else 0
    avg_episode_urm_pct = np.mean(episode_urm_pcts) if episode_urm_pcts else 0
    
    return {
        'episodes_analyzed': episodes_analyzed,
        'episodes_with_guests': episodes_with_guests,
        'total_guests': total_guests,
        'total_female_guests': total_female,
        'total_urm_guests': total_urm,
        'overall_female_percentage': round(overall_female_pct, 2),
        'overall_urm_percentage': round(overall_urm_pct, 2),
        'episode_averaged_female_percentage': round(avg_episode_female_pct, 2),
        'episode_averaged_urm_percentage': round(avg_episode_urm_pct, 2)
    }

def analyze_episodes_batch(episodes: List[Dict], enricher: Optional[GuestEnricher] = None, max_workers: int = 10) -> List[Dict]:
    """Analyze multiple episodes concurrently"""
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_episode = {executor.submit(analyze_episode_guests, episode, enricher): episode 
                           for episode in episodes}
        
        for future in concurrent.futures.as_completed(future_to_episode):
            episode = future_to_episode[future]
            try:
                guest_data = future.result()
                results.append({
                    'episode_id': episode['episode_id'],
                    'total_guests': guest_data.get('total_guests', 0),
                    'female_guests': guest_data.get('female_guests', 0),
                    'urm_guests': guest_data.get('urm_guests', 0),
                    'guests': guest_data.get('guests', [])
                })
            except Exception as e:
                logger.error(f"Error analyzing episode {episode['episode_id']}: {e}")
                results.append({
                    'episode_id': episode['episode_id'],
                    'total_guests': 0,
                    'female_guests': 0,
                    'urm_guests': 0,
                    'guests': []
                })
    
    return results

def process_podcast(podcast_info: Dict, enricher: Optional[GuestEnricher] = None) -> Dict:
    """Process a single podcast"""
    podcast_id = podcast_info['podcastID']
    
    # Fetch episodes
    episodes = fetch_rss_episodes(podcast_info)
    if not episodes:
        return None
    
    # Analyze episodes in batch with concurrent API calls
    episodes_data = analyze_episodes_batch(episodes, enricher, max_workers=10)
    
    # Calculate metrics
    metrics = calculate_podcast_metrics(episodes_data)
    
    return {
        'podcastID': podcast_id,
        'treatment': podcast_info.get('treatment'),
        **metrics,
        'episodes_data': episodes_data
    }

def main():
    """Main pipeline execution"""
    logger.info("Starting Enhanced Podcast Guest Analysis")
    
    # Load podcast data
    try:
        df = pd.read_csv('podcast_rss_export.csv')
        df['latest_response_time'] = pd.to_datetime(df['latest_response_time'], utc=True)
        df = df.rename(columns={'latest_response_time': 'completion_timestamp'})
        df = df.dropna(subset=['podcastID', 'rss', 'completion_timestamp'])
        
        logger.info(f"Loaded {len(df)} podcasts")
    except Exception as e:
        logger.error(f"Error loading podcast data: {e}")
        return
    
    # Initialize enricher
    enricher = None
    if CONFIG['enrichment_settings']['enable_web_enrichment']:
        try:
            enricher = GuestEnricher(
                serpapi_key=CONFIG['api_keys']['serpapi'],
                cache_dir=CONFIG['enrichment_settings']['cache_directory']
            )
            logger.info("Guest enrichment enabled")
        except Exception as e:
            logger.warning(f"Could not initialize enricher: {e}")
    
    # Process podcasts
    results = []
    podcast_list = df.to_dict('records')
    
    # Process podcasts with batched API calls
    logger.info("Processing podcasts with concurrent API calls...")
    
    # Process in chunks to avoid overwhelming the API
    CHUNK_SIZE = 50  # Process 50 podcasts at a time
    
    with tqdm(total=len(podcast_list), desc="Analyzing podcasts") as pbar:
        for i in range(0, len(podcast_list), CHUNK_SIZE):
            chunk = podcast_list[i:i+CHUNK_SIZE]
            
            # Process chunk of podcasts concurrently
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                future_to_podcast = {executor.submit(process_podcast, podcast_info, enricher): podcast_info 
                                   for podcast_info in chunk}
                
                for future in concurrent.futures.as_completed(future_to_podcast):
                    podcast_info = future_to_podcast[future]
                    try:
                        result = future.result()
                        if result:
                            results.append(result)
                    except Exception as e:
                        logger.error(f"Error processing podcast {podcast_info['podcastID']}: {e}")
                    pbar.update(1)
    
    # Save results
    if results:
        # Create summary DataFrame
        summary_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'episodes_data'} for r in results])
        summary_df.to_csv('enhanced_guest_analysis_summary.csv', index=False)
        logger.info(f"Saved summary for {len(summary_df)} podcasts")
        
        # Save detailed episode data
        all_episodes = []
        for r in results:
            for ep in r['episodes_data']:
                all_episodes.append({
                    'podcastID': r['podcastID'],
                    'episode_id': ep['episode_id'],
                    'total_guests': ep['total_guests'],
                    'female_guests': ep['female_guests'],
                    'urm_guests': ep['urm_guests']
                })
        
        episode_df = pd.DataFrame(all_episodes)
        episode_df.to_csv('enhanced_guest_analysis_episodes.csv', index=False)
        
        # Print summary statistics
        print("\n=== Analysis Complete ===")
        print(f"Podcasts analyzed: {len(summary_df)}")
        print(f"Total episodes: {len(episode_df)}")
        print(f"Total guests found: {summary_df['total_guests'].sum()}")
        
        # Compare treatment vs control
        if 'treatment' in summary_df.columns:
            for metric in ['overall_female_percentage', 'overall_urm_percentage', 
                          'episode_averaged_female_percentage', 'episode_averaged_urm_percentage']:
                print(f"\n{metric}:")
                for treatment in [0, 1]:
                    group = summary_df[summary_df['treatment'] == treatment]
                    if len(group) > 0:
                        mean_val = group[metric].mean()
                        std_val = group[metric].std()
                        print(f"  {'Treatment' if treatment else 'Control'}: {mean_val:.2f}% (SD: {std_val:.2f})")
    
    logger.info("Pipeline complete")

if __name__ == "__main__":
    main()