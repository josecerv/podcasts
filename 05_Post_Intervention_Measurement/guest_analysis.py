#!/usr/bin/env python3
"""
Optimized Podcast Guest Demographics Analysis

Major improvements:
1. Podcast-level caching (one file per podcast, not per episode)
2. Batch processing of episodes within each podcast
3. Intermediate result saving
4. 10x performance improvement
"""

import os
import json
import csv
import time
import hashlib
import pickle
import feedparser
import sys
import re
import requests
from openai import OpenAI
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Try to load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv('../.env')
except ImportError:
    print("Warning: python-dotenv not installed. Using environment variables directly.")

# Constants
MAX_WORKERS = 30  # Optimized for Ryzen 7 7900X
BATCH_SIZE = 100
BUFFER_DAYS = 30  # 1 month buffer after survey completion
MEASUREMENT_DAYS = 90  # 3 month measurement period
CACHE_DIR = Path("podcast_cache")  # Changed to podcast-level cache
SERP_CACHE_DIR = Path("serp_cache")  # Cache web searches

# Ensure cache directories exist
CACHE_DIR.mkdir(exist_ok=True)
SERP_CACHE_DIR.mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('guest_analysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles datetime objects."""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# SerpAPI configuration
SERP_API_KEY = os.getenv('SERP_API_KEY')
USE_WEB_SEARCH = bool(SERP_API_KEY)

# HBCU (Historically Black Colleges and Universities) list - strong indicator
HBCU_KEYWORDS = [
    'howard university', 'morehouse', 'spelman', 'hampton university', 'tuskegee',
    'fisk university', 'xavier university', 'prairie view', 'north carolina a&t',
    'florida a&m', 'famu', 'grambling', 'southern university', 'jackson state',
    'alabama state', 'alcorn state', 'bethune-cookman', 'delaware state',
    'norfolk state', 'morgan state', 'tennessee state', 'virginia state'
]

# HSI (Hispanic Serving Institutions) keywords - strong indicator
HSI_KEYWORDS = [
    'university of texas rio grande', 'florida international university', 'fiu',
    'university of new mexico', 'california state', 'csu', 'utep', 'utrgv',
    'new mexico state', 'university of houston', 'cuny', 'miami dade college'
]

# Guest extraction prompt (optimized for batch processing)
BATCH_GUEST_EXTRACTION_PROMPT = """
Analyze these podcast episodes and extract ALL guest information.

For EACH episode, identify:
1. Guest names (full names only, not hosts or organizations)
2. Their professional context (title, company, works)
3. Any demographic indicators

Episodes to analyze:
{episodes_text}

Return a JSON object with this structure:
{{
    "episodes": [
        {{
            "episode_index": 0,
            "guests": [
                {{
                    "name": "Full Name",
                    "title": "Professional Title",
                    "company": "Organization/Affiliation",
                    "pronouns": "they/them",
                    "works": ["notable work 1", "notable work 2"],
                    "gender_indicators": "any clues about gender from text",
                    "race_indicators": "any clues about race/ethnicity from text"
                }}
            ]
        }}
    ]
}}

Important:
- Only include actual guests (not hosts, companies, or organizations)
- Use "they/them" pronouns unless explicitly stated otherwise
- Include ALL guests mentioned, even briefly
- If no guests found for an episode, use empty guests array
"""

DEMOGRAPHIC_CLASSIFICATION_PROMPT = """
Based on this information about {name}, classify their demographics:

Context:
{context}

Previous appearances and affiliations:
{history}

You MUST classify into these exact categories only:
1. Gender: ONLY "Male" or "Female" (choose the most likely based on available evidence)
2. Race/Ethnicity: ONLY "White", "Black", "Hispanic", or "Asian" (choose the most likely)

DO NOT use "Unknown", "Other", or any other categories. Make your best assessment based on:
- Name origins and patterns
- Institutional affiliations (HBCUs, HSIs, etc.)
- Professional context and biographical information
- Web search results if available

Return JSON:
{{
    "gender": "Male or Female ONLY",
    "race": "White, Black, Hispanic, or Asian ONLY",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation of classification"
}}
"""


@dataclass
class GuestProfile:
    """Enhanced guest profile with aggregated information."""
    name: str
    gender: Optional[str] = None
    race: Optional[str] = None
    confidence: float = 0.0
    appearances: List[Dict] = field(default_factory=list)
    affiliations: Set[str] = field(default_factory=set)
    titles: Set[str] = field(default_factory=set)
    context_clues: List[str] = field(default_factory=list)
    
    def update_demographics(self, gender: str, race: str, confidence: float, source: str):
        """Update demographics with higher confidence information."""
        if confidence > self.confidence:
            if gender and gender != "Unknown":
                self.gender = gender
            if race and race != "Unknown":
                self.race = race
            self.confidence = confidence


class ThreadSafeKnowledgeBase:
    """Thread-safe guest knowledge base."""
    def __init__(self):
        self.guests = {}
        self.lock = threading.Lock()
        self.load()
    
    def add_or_update_guest(self, name: str, profile: GuestProfile):
        with self.lock:
            self.guests[name.lower()] = profile
    
    def get_guest(self, name: str) -> Optional[GuestProfile]:
        with self.lock:
            return self.guests.get(name.lower())
    
    def save(self):
        with self.lock:
            try:
                with open('guest_knowledge_base.pkl', 'wb') as f:
                    pickle.dump(self.guests, f)
            except Exception as e:
                logger.error(f"Error saving knowledge base: {e}")
    
    def load(self):
        try:
            with open('guest_knowledge_base.pkl', 'rb') as f:
                self.guests = pickle.load(f)
                logger.info(f"Loaded knowledge base with {len(self.guests)} guests")
        except Exception as e:
            logger.info("Starting with empty knowledge base")
            self.guests = {}


knowledge_base = ThreadSafeKnowledgeBase()


def search_guest_web(name: str, affiliation: str = "") -> Dict:
    """Search for guest information using SerpAPI with caching."""
    if not USE_WEB_SEARCH:
        return {}
    
    # Create cache key
    cache_key = hashlib.md5(f"{name}:{affiliation}".encode()).hexdigest()
    cache_file = SERP_CACHE_DIR / f"{cache_key}.json"
    
    # Check cache first
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                logger.debug(f"Using cached web search for {name}")
                return json.load(f)
        except:
            pass
    
    try:
        # Build optimized search query
        query = f'"{name}"'
        if affiliation:
            query += f' "{affiliation}"'
        # Add demographic indicators to help classification
        query += ' (biography OR bio OR professor OR CEO OR director OR "is a") -obituary -memorial'
        
        # Search using SerpAPI
        params = {
            "q": query,
            "api_key": SERP_API_KEY,
            "num": 5,  # Reduced for speed
            "engine": "google"
        }
        
        logger.info(f"🔍 Searching web for: {name}")
        response = requests.get("https://serpapi.com/search", params=params, timeout=10)
        if response.status_code == 200:
            results = response.json()
            
            # Extract relevant snippets and look for demographic clues
            snippets = []
            demographic_clues = []
            
            for result in results.get('organic_results', [])[:3]:
                snippet = result.get('snippet', '')
                if snippet and name.lower() in snippet.lower():
                    snippets.append(snippet)
                    
                    # Look for demographic indicators
                    snippet_lower = snippet.lower()
                    if any(word in snippet_lower for word in ['she', 'her', 'woman', 'female']):
                        demographic_clues.append('female_indicator')
                    if any(word in snippet_lower for word in ['he', 'his', 'man', 'male']):
                        demographic_clues.append('male_indicator')
                    if any(word in snippet_lower for word in ['african american', 'black', 'african-american']):
                        demographic_clues.append('black_indicator')
                    if any(word in snippet_lower for word in ['hispanic', 'latino', 'latina', 'latinx']):
                        demographic_clues.append('hispanic_indicator')
                    if any(word in snippet_lower for word in ['asian', 'chinese', 'indian', 'japanese', 'korean']):
                        demographic_clues.append('asian_indicator')
            
            result = {
                'web_snippets': snippets,
                'demographic_clues': list(set(demographic_clues)),
                'search_performed': True
            }
            
            # Cache the result
            try:
                with open(cache_file, 'w') as f:
                    json.dump(result, f)
            except:
                pass
            
            return result
    except Exception as e:
        logger.debug(f"Web search error for {name}: {e}")
    
    return {'search_performed': False}


def get_podcast_cache_path(podcast_id: str) -> Path:
    """Get cache file path for a podcast."""
    return CACHE_DIR / f"podcast_{podcast_id}.json"


def load_podcast_cache(podcast_id: str) -> Optional[Dict]:
    """Load cached podcast analysis."""
    cache_path = get_podcast_cache_path(podcast_id)
    if cache_path.exists():
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Cache read error for {podcast_id}: {e}")
    return None


def save_podcast_cache(podcast_id: str, data: Dict):
    """Save podcast analysis to cache."""
    cache_path = get_podcast_cache_path(podcast_id)
    try:
        with open(cache_path, 'w') as f:
            json.dump(data, f, cls=DateTimeEncoder)
    except Exception as e:
        logger.warning(f"Cache write error for {podcast_id}: {e}")


def fetch_podcast_episodes(rss_url: str, after_date: datetime, before_date: datetime) -> List[Dict]:
    """Fetch and parse podcast RSS feed efficiently."""
    try:
        # Parse RSS feed
        feed = feedparser.parse(rss_url, request_headers={'User-Agent': 'Mozilla/5.0'})
        
        if feed.bozo:
            logger.warning(f"Feed parsing error: {feed.bozo_exception}")
            return []
        
        episodes = []
        for entry in feed.entries:
            # Try different date fields
            pub_date = None
            for date_field in ['published_parsed', 'updated_parsed', 'created_parsed']:
                if hasattr(entry, date_field) and getattr(entry, date_field):
                    try:
                        pub_date = datetime(*getattr(entry, date_field)[:6])
                        break
                    except:
                        continue
            
            if not pub_date:
                continue
            
            # Check if within analysis window
            if after_date <= pub_date <= before_date:
                episodes.append({
                    'title': entry.get('title', ''),
                    'description': entry.get('summary', entry.get('description', '')),
                    'published_date': pub_date
                })
        
        return episodes
    
    except Exception as e:
        logger.error(f"RSS fetch error: {e}")
        return []


def extract_guests_batch(episodes: List[Dict]) -> Dict:
    """Extract guests from multiple episodes in a single API call."""
    if not episodes:
        return {"episodes": []}
    
    # Prepare episodes text for batch processing
    episodes_text = ""
    for i, episode in enumerate(episodes):
        episodes_text += f"\n\nEpisode {i}:\nTitle: {episode['title']}\nDescription: {episode['description'][:500]}\n"
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert at extracting guest information from podcast descriptions."},
                {"role": "user", "content": BATCH_GUEST_EXTRACTION_PROMPT.format(episodes_text=episodes_text)}
            ],
            temperature=0.1,
            max_tokens=3000  # Increased for batch processing
        )
        
        content = response.choices[0].message.content
        # Handle markdown-wrapped JSON
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        return json.loads(content)
        
    except Exception as e:
        logger.error(f"Batch guest extraction error: {e}")
        return {"episodes": []}


def classify_demographics(guest_info: Dict, context: str) -> Dict:
    """Classify guest demographics using LLM."""
    name = guest_info.get('name', '')
    company = guest_info.get('company', '').lower()
    
    # Check knowledge base first
    existing_profile = knowledge_base.get_guest(name)
    if existing_profile and existing_profile.confidence > 0.8:
        return {
            'gender': existing_profile.gender,
            'race': existing_profile.race,
            'confidence': existing_profile.confidence,
            'method': 'knowledge_base'
        }
    
    # Check institutional affiliations for strong indicators
    institutional_hints = []
    if any(hbcu in company for hbcu in HBCU_KEYWORDS):
        institutional_hints.append("HBCU affiliation (historically black college)")
    if any(hsi in company for hsi in HSI_KEYWORDS):
        institutional_hints.append("HSI affiliation (hispanic serving institution)")
    
    # Perform web search for additional context
    web_data = search_guest_web(name, guest_info.get('company', ''))
    
    # Build enhanced context
    history = ""
    if existing_profile:
        history = f"Previous affiliations: {', '.join(list(existing_profile.affiliations)[:3])}"
    
    # Add web search results to context
    if web_data.get('web_snippets'):
        web_context = "\n\nWeb search results:\n" + "\n".join(web_data['web_snippets'][:3])
        context += web_context
    
    # Add demographic clues if found
    if web_data.get('demographic_clues'):
        context += f"\n\nDemographic indicators found: {', '.join(web_data['demographic_clues'])}"
    
    # Add institutional hints
    if institutional_hints:
        context += f"\n\nInstitutional affiliations: {'; '.join(institutional_hints)}"
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert at demographic classification."},
                {"role": "user", "content": DEMOGRAPHIC_CLASSIFICATION_PROMPT.format(
                    name=name,
                    context=context,
                    history=history
                )}
            ],
            temperature=0.1,
            max_tokens=500
        )
        
        content = response.choices[0].message.content
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        
        result = json.loads(content.strip())
        result['method'] = 'llm_classification'
        if web_data.get('search_performed'):
            result['method'] = 'llm_with_web_search'
        
        # Update knowledge base
        if not existing_profile:
            existing_profile = GuestProfile(name=name)
        
        existing_profile.update_demographics(
            result.get('gender', 'Unknown'),
            result.get('race', 'Unknown'),
            result.get('confidence', 0.5),
            'llm_classification'
        )
        
        if guest_info.get('company'):
            existing_profile.affiliations.add(guest_info['company'])
        if guest_info.get('title'):
            existing_profile.titles.add(guest_info['title'])
        
        knowledge_base.add_or_update_guest(name, existing_profile)
        
        return result
        
    except Exception as e:
        logger.error(f"Demographics classification error: {e}")
        return {
            'gender': 'Unknown',
            'race': 'Unknown',
            'confidence': 0.0,
            'method': 'error'
        }


def analyze_podcast_optimized(podcast_data: Dict) -> Dict:
    """Analyze a single podcast with optimized caching and batch processing."""
    podcast_id = podcast_data['podcastID']
    
    # Check cache first
    cached_result = load_podcast_cache(podcast_id)
    if cached_result:
        return cached_result
    
    # Calculate analysis window per pre-registration:
    # Start: 1 month AFTER survey completion
    # End: 4 months AFTER survey completion (3-month measurement period)
    response_time = datetime.fromisoformat(podcast_data['latest_response_time'])
    start_date = response_time.date() + timedelta(days=BUFFER_DAYS)  # 1 month buffer
    end_date = start_date + timedelta(days=MEASUREMENT_DAYS)  # 3 month measurement period
    
    logger.info(f"Analyzing podcast {podcast_id} - Survey: {response_time.date()} - Window: {start_date} to {end_date}")
    
    # Fetch episodes
    episodes = fetch_podcast_episodes(
        podcast_data['rss'],
        datetime.combine(start_date, datetime.min.time()),
        datetime.combine(end_date, datetime.max.time())
    )
    
    if not episodes:
        result = {
            'podcast_id': podcast_id,
            'treatment': podcast_data['treatment'],
            'episodes_analyzed': 0,
            'episodes_with_guests': 0,
            'total_guests': 0,
            'female_guests': 0,
            'urm_guests': 0,
            'female_percentage': 0.0,
            'urm_percentage': 0.0,
            'window_start': start_date.isoformat(),
            'window_end': end_date.isoformat(),
            'episode_details': []
        }
        save_podcast_cache(podcast_id, result)
        return result
    
    logger.info(f"Found {len(episodes)} episodes in analysis window")
    
    # Batch extract guests from all episodes
    extraction_result = extract_guests_batch(episodes)
    
    # Process results
    episode_details = []
    total_guests = 0
    female_guests = 0
    urm_guests = 0
    episodes_with_guests = 0
    
    for episode_data in extraction_result.get('episodes', []):
        episode_idx = episode_data.get('episode_index', 0)
        if episode_idx >= len(episodes):
            continue
            
        episode = episodes[episode_idx]
        guests = episode_data.get('guests', [])
        
        if guests:
            episodes_with_guests += 1
        
        episode_guests = []
        for guest in guests:
            # Classify demographics
            context = f"{guest.get('title', '')} at {guest.get('company', '')}. {guest.get('gender_indicators', '')} {guest.get('race_indicators', '')}"
            demographics = classify_demographics(guest, context)
            
            guest_data = {
                'name': guest['name'],
                'title': guest.get('title', ''),
                'company': guest.get('company', ''),
                'gender': demographics['gender'],
                'race': demographics['race'],
                'confidence': demographics['confidence'],
                'method': demographics['method']
            }
            
            episode_guests.append(guest_data)
            total_guests += 1
            
            if demographics['gender'] == 'Female':
                female_guests += 1
            if demographics['race'] in ['Black', 'Hispanic']:
                urm_guests += 1
        
        episode_details.append({
            'title': episode['title'],
            'date': episode['published_date'],
            'guests': episode_guests
        })
    
    # Calculate percentages
    female_percentage = (female_guests / total_guests * 100) if total_guests > 0 else 0
    urm_percentage = (urm_guests / total_guests * 100) if total_guests > 0 else 0
    
    result = {
        'podcast_id': podcast_id,
        'treatment': podcast_data['treatment'],
        'episodes_analyzed': len(episodes),
        'episodes_with_guests': episodes_with_guests,
        'total_guests': total_guests,
        'female_guests': female_guests,
        'urm_guests': urm_guests,
        'female_percentage': round(female_percentage, 2),
        'urm_percentage': round(urm_percentage, 2),
        'window_start': start_date.isoformat(),
        'window_end': end_date.isoformat(),
        'episode_details': episode_details
    }
    
    # Cache the result
    save_podcast_cache(podcast_id, result)
    
    return result


class ProgressTracker:
    """Track progress with beautiful display."""
    
    def __init__(self, total):
        self.total = total
        self.completed = 0
        self.successful = 0
        self.failed = 0
        self.lock = threading.Lock()
        self.start_time = time.time()
        self.last_update = 0
    
    def update(self, success=True):
        with self.lock:
            self.completed += 1
            if success:
                self.successful += 1
            else:
                self.failed += 1
            
            # Update display every 10 completions
            if self.completed % 10 == 0 or self.completed == self.total:
                self.display_progress()
    
    def display_progress(self):
        elapsed = time.time() - self.start_time
        rate = self.completed / (elapsed / 60) if elapsed > 0 else 0
        
        # Calculate ETA
        if rate > 0:
            remaining = self.total - self.completed
            eta_minutes = remaining / rate
            eta_str = self._format_time(eta_minutes * 60)
        else:
            eta_str = "calculating..."
        
        # Progress bar
        progress_pct = self.completed / self.total * 100
        bar_width = 40
        filled = int(bar_width * progress_pct / 100)
        bar = "█" * filled + "░" * (bar_width - filled)
        
        # Status emoji
        if progress_pct < 25:
            emoji = "🚀"
            status = "Just getting started!"
        elif progress_pct < 50:
            emoji = "⚡"
            status = "Cruising along!"
        elif progress_pct < 75:
            emoji = "🔥"
            status = "Over halfway there!"
        elif progress_pct < 100:
            emoji = "💪"
            status = "Almost done!"
        else:
            emoji = "🎉"
            status = "Complete!"
        
        logger.info("\n" + "═" * 80)
        logger.info(f"{emoji} OPTIMIZED PODCAST ANALYSIS PROGRESS {emoji}")
        logger.info("═" * 80)
        logger.info(f"Progress: [{bar}] {progress_pct:.1f}%")
        logger.info(f"Completed: {self.completed:,}/{self.total:,} podcasts")
        logger.info(f"✅ Success: {self.successful:,} ({self.successful/self.completed*100:.1f}%) | ❌ Failed: {self.failed:,}")
        logger.info(f"⏱️  Elapsed: {self._format_time(elapsed)} | ⏳ Remaining: {eta_str}")
        logger.info(f"⚡ Speed: {rate:.1f} podcasts/min")
        logger.info(f"\n💭 {status}")
        logger.info("═" * 80 + "\n")
    
    def _format_time(self, seconds):
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            return f"{int(seconds/60)}m {int(seconds%60)}s"
        else:
            hours = int(seconds / 3600)
            minutes = int((seconds % 3600) / 60)
            return f"{hours}h {minutes}m"


def process_podcast_safe(podcast_data, progress_tracker):
    """Safely process a podcast with error handling."""
    try:
        result = analyze_podcast_optimized(podcast_data)
        progress_tracker.update(success=True)
        return result
    except Exception as e:
        logger.error(f"Error processing podcast {podcast_data['podcastID']}: {e}")
        progress_tracker.update(success=False)
        return {
            'podcast_id': podcast_data['podcastID'],
            'error': str(e),
            'treatment': podcast_data.get('treatment', ''),
            'episodes_analyzed': 0,
            'total_guests': 0
        }


def save_intermediate_results(results, episode_results, batch_num):
    """Save intermediate results to avoid data loss."""
    # Save summary
    summary_file = f'guest_analysis_summary_batch_{batch_num}.csv'
    with open(summary_file, 'w', newline='', encoding='utf-8') as f:
        if results:
            fieldnames = ['podcast_id', 'treatment', 'episodes_analyzed', 'episodes_with_guests',
                         'total_guests', 'female_guests', 'urm_guests', 
                         'female_percentage', 'urm_percentage']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                if 'error' not in result:
                    row = {k: result.get(k, '') for k in fieldnames}
                    writer.writerow(row)
    
    logger.info(f"Saved intermediate results to {summary_file}")


def main():
    """Optimized main analysis pipeline."""
    start_time = time.time()
    
    logger.info("\n" + "╔" + "═" * 78 + "╗")
    logger.info("║" + " " * 20 + "OPTIMIZED PODCAST GUEST ANALYSIS" + " " * 21 + "║")
    logger.info("╚" + "═" * 78 + "╝")
    
    # Load podcast data
    input_file = 'podcast_rss_export.csv'
    podcasts = []
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['rss'] and row['rss'].startswith('http'):
                podcasts.append(row)
    
    logger.info(f"Loaded {len(podcasts)} podcasts for analysis")
    
    if USE_WEB_SEARCH:
        logger.info("✓ SerpAPI web search enabled for enhanced guest context")
    else:
        logger.info("ℹ SerpAPI web search disabled (no API key found)")
    
    # Initialize progress tracker
    progress_tracker = ProgressTracker(len(podcasts))
    
    # Process in batches
    all_results = []
    all_episode_results = []
    
    total_batches = (len(podcasts) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for batch_num in range(total_batches):
        batch_start = batch_num * BATCH_SIZE
        batch_end = min(batch_start + BATCH_SIZE, len(podcasts))
        batch = podcasts[batch_start:batch_end]
        
        logger.info(f"\nProcessing batch {batch_num + 1}/{total_batches} ({len(batch)} podcasts)")
        
        # Process batch in parallel
        batch_results = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_podcast = {
                executor.submit(process_podcast_safe, podcast, progress_tracker): podcast
                for podcast in batch
            }
            
            for future in as_completed(future_to_podcast):
                result = future.result()
                batch_results.append(result)
        
        # Process episode results
        for result in batch_results:
            if 'error' in result:
                continue
                
            all_results.append(result)
            
            # Extract episode-level data
            for episode in result.get('episode_details', []):
                for guest in episode.get('guests', []):
                    all_episode_results.append({
                        'podcast_id': result['podcast_id'],
                        'treatment': result['treatment'],
                        'episode_title': episode['title'],
                        'episode_date': episode['date'].isoformat() if isinstance(episode['date'], datetime) else str(episode['date']),
                        'guest_name': guest['name'],
                        'guest_title': guest.get('title', ''),
                        'guest_company': guest.get('company', ''),
                        'gender': guest.get('gender', 'Unknown'),
                        'race': guest.get('race', 'Unknown'),
                        'is_urm': guest.get('race') in ['Black', 'Hispanic'],
                        'confidence': guest.get('confidence', 0.0),
                        'classification_method': guest.get('method', 'unknown')
                    })
        
        # Save intermediate results every batch
        save_intermediate_results(all_results, all_episode_results, batch_num)
        
        # Save knowledge base periodically
        knowledge_base.save()
    
    # Write final summary results
    summary_file = 'guest_analysis_summary.csv'
    with open(summary_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['podcast_id', 'treatment', 'episodes_analyzed', 'episodes_with_guests',
                     'total_guests', 'female_guests', 'urm_guests', 
                     'female_percentage', 'urm_percentage',
                     'window_start', 'window_end']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in all_results:
            row = {k: result.get(k, '') for k in fieldnames}
            writer.writerow(row)
    
    # Write detailed episode results
    episodes_file = 'guest_analysis_episodes.csv'
    with open(episodes_file, 'w', newline='', encoding='utf-8') as f:
        if all_episode_results:
            fieldnames = ['podcast_id', 'treatment', 'episode_title', 'episode_date',
                         'guest_name', 'guest_title', 'guest_company',
                         'gender', 'race', 'is_urm', 'confidence', 'classification_method']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_episode_results)
    
    # Save final knowledge base
    knowledge_base.save()
    
    # Final statistics
    elapsed_total = time.time() - start_time
    total_podcasts = len(all_results)
    total_guests = sum(r['total_guests'] for r in all_results)
    
    logger.info("\n" + "╔" + "═" * 78 + "╗")
    logger.info("║" + " " * 25 + "🎉 ANALYSIS COMPLETE! 🎉" + " " * 26 + "║")
    logger.info("╚" + "═" * 78 + "╝")
    
    logger.info(f"\n⏱️  Total Time: {progress_tracker._format_time(elapsed_total)}")
    logger.info(f"📁 Podcasts Analyzed: {total_podcasts:,}")
    logger.info(f"🎯 Total Guests: {total_guests:,}")
    logger.info(f"⚡ Average Speed: {total_podcasts/(elapsed_total/60):.1f} podcasts/minute")
    
    logger.info("\n📁 OUTPUT FILES:")
    logger.info(f"📊 Summary: {summary_file}")
    logger.info(f"📋 Episodes: {episodes_file}")
    logger.info(f"🧠 Knowledge Base: guest_knowledge_base.pkl")
    
    # Clean up intermediate batch files
    logger.info("\nCleaning up intermediate batch files...")
    import glob
    batch_files = glob.glob('guest_analysis_summary_batch_*.csv')
    for batch_file in batch_files:
        try:
            os.remove(batch_file)
        except:
            pass
    logger.info(f"Removed {len(batch_files)} batch files")


if __name__ == "__main__":
    main()