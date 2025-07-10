#!/usr/bin/env python3
"""
Enhanced Post-Intervention Podcast Guest Analysis Pipeline
Version 2.0 - With improved guest extraction, RSS reliability, and demographic analysis

Key improvements:
1. Pattern-based and NER guest extraction
2. Robust RSS fetching with retries and user-agent rotation
3. Multi-source demographic analysis (names, web search, photos)
4. Guest context enrichment (affiliations, books, etc.)
"""

import sys
import os
import feedparser
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timezone, timedelta
import hashlib
import time
import re
import multiprocessing
import gc
from typing import List, Dict, Optional, Tuple, Set
import concurrent.futures
import signal
import socket
import json
from tqdm.auto import tqdm
from openai import OpenAI
import string
from scipy import stats
import requests
from bs4 import BeautifulSoup
import spacy
from nameparser import HumanName
import joblib
from collections import Counter, defaultdict
import warnings
from urllib.parse import urlparse
import random

warnings.filterwarnings('ignore')

# --- Configuration ---
INPUT_CSV = 'podcast_rss_export.csv'
RAW_LLM_OUTPUT_JSONL = 'post_intervention_guest_classification_enhanced.jsonl'
AGGREGATED_OUTPUT_CSV = 'post_intervention_guest_summary_enhanced.csv'
ENRICHED_GUESTS_CSV = 'post_intervention_guests_enriched.csv'
ERROR_LOG_FILE = "post_intervention_error_log_enhanced.txt"
RSS_SUCCESS_LOG = "rss_fetch_success_log.csv"

# RSS Feed Processing Config
RSS_TIMEOUT = 30
RSS_RETRY_ATTEMPTS = 3
RSS_RETRY_DELAY = 5
MIN_REQUIRED_EPISODES = 1
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 PodcastScraper/2.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.101 Safari/537.36",
    "PodcastAddict/v5 (Linux; Android 11) PodcastScraper/2.0",
]

# LLM Config
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
GOOGLE_SEARCH_API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
LLM_MODEL = "gpt-4o-mini"
MAX_LLM_WORKERS = 20
LLM_RETRY_ATTEMPTS = 3
LLM_RETRY_DELAY = 5

# Guest Extraction Patterns
GUEST_PATTERNS = [
    r"(?:featuring|feature|feat\.?|with guest[s]?|guest[s]?:|interview with|joined by|welcoming|special guest[s]?|our guest[s]?|today'?s guest[s]?)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})",
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s+(?:joins us|is here|stops by|visits|shares|discusses|talks about)",
    r"(?:author|professor|dr\.?|expert|ceo|founder|director|coach|consultant)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})",
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}),\s+(?:author of|professor at|ceo of|founder of|director of)",
]

# Demographic Analysis Config
URM_CATEGORIES = ['Black', 'Hispanic/Latino', 'Native American']
DEEPFACE_BACKENDS = ['opencv', 'retinaface', 'mtcnn']
MAX_IMAGES_PER_GUEST = 3
IMAGE_SEARCH_TIMEOUT = 15

# --- Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(processName)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler(ERROR_LOG_FILE, mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load spaCy model for NER
try:
    nlp = spacy.load("en_core_web_sm")
except:
    logger.warning("spaCy model not found. Installing...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

printable = set(string.printable)

# --- Helper Functions ---
def clean_text(text):
    """Clean and normalize text"""
    try:
        if not isinstance(text, str): 
            text = str(text)
        # Remove HTML tags
        text = re.sub(r'<.*?>', ' ', text)
        # Decode HTML entities
        text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
        text = text.replace('&quot;', '"').replace('&apos;', "'").replace('&nbsp;', ' ')
        # Remove control characters
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except Exception as e:
        logger.debug(f"Error cleaning text: {e}")
        return ''

def extract_guests_with_patterns(title: str, description: str) -> List[Dict[str, str]]:
    """Extract guest names using regex patterns"""
    guests = []
    text = f"{title} {description}"
    text = clean_text(text)
    
    seen_names = set()
    
    for pattern in GUEST_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            name = match.strip()
            # Basic validation
            if len(name.split()) >= 2 and len(name) < 50 and name.lower() not in seen_names:
                seen_names.add(name.lower())
                guests.append({
                    "name": name,
                    "extraction_method": "pattern",
                    "confidence": 0.8
                })
    
    return guests

def extract_guests_with_ner(text: str) -> List[Dict[str, str]]:
    """Extract guest names using spaCy NER"""
    guests = []
    doc = nlp(text[:5000])  # Limit text length for performance
    
    seen_names = set()
    
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            name = ent.text.strip()
            # Filter out common false positives
            if (len(name.split()) >= 2 and 
                len(name) < 50 and 
                name.lower() not in seen_names and
                not any(word in name.lower() for word in ['podcast', 'show', 'episode', 'series'])):
                
                seen_names.add(name.lower())
                guests.append({
                    "name": name,
                    "extraction_method": "ner",
                    "confidence": 0.7
                })
    
    return guests

def extract_guest_context(text: str, guest_name: str) -> Dict[str, str]:
    """Extract contextual information about a guest"""
    context = {
        "affiliation": None,
        "title": None,
        "book": None,
        "expertise": None
    }
    
    # Create search windows around the guest name
    name_pattern = re.escape(guest_name)
    window_size = 100
    
    matches = list(re.finditer(name_pattern, text, re.IGNORECASE))
    
    for match in matches[:3]:  # Check first 3 occurrences
        start = max(0, match.start() - window_size)
        end = min(len(text), match.end() + window_size)
        window = text[start:end]
        
        # Extract affiliation
        affiliation_match = re.search(
            r"(?:from|at|with|of)\s+(?:the\s+)?([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3})\s*(?:University|Institute|Company|Corporation|Organization)",
            window, re.IGNORECASE
        )
        if affiliation_match and not context["affiliation"]:
            context["affiliation"] = affiliation_match.group(0)
        
        # Extract title
        title_match = re.search(
            r"(?:Dr\.?|Professor|Prof\.?|CEO|Founder|Director|Author|Expert|Coach|Consultant)",
            window, re.IGNORECASE
        )
        if title_match and not context["title"]:
            context["title"] = title_match.group(0)
        
        # Extract book
        book_match = re.search(
            r"(?:author of|wrote|new book|latest book|book titled)\s*[\"']?([^\"']{5,50})[\"']?",
            window, re.IGNORECASE
        )
        if book_match and not context["book"]:
            context["book"] = book_match.group(1)
    
    return context

def fetch_rss_with_retry(podcast_info: Dict) -> List[Dict]:
    """Fetch RSS feed with retry logic and better error handling"""
    rss_url = podcast_info.get('rss')
    podcast_id = podcast_info.get('podcastID')
    
    if not rss_url or not podcast_id:
        return []
    
    for attempt in range(RSS_RETRY_ATTEMPTS):
        try:
            # Random user agent
            user_agent = random.choice(USER_AGENTS)
            feedparser.USER_AGENT = user_agent
            
            # Set timeout
            original_timeout = socket.getdefaulttimeout()
            socket.setdefaulttimeout(RSS_TIMEOUT)
            
            logger.debug(f"Fetching RSS for {podcast_id} (attempt {attempt + 1}/{RSS_RETRY_ATTEMPTS})")
            
            parsed_feed = feedparser.parse(rss_url)
            
            # Check if feed is valid
            if hasattr(parsed_feed, 'bozo') and parsed_feed.bozo:
                if isinstance(parsed_feed.bozo_exception, feedparser.CharacterEncodingOverride):
                    # This is often benign
                    if parsed_feed.entries:
                        logger.debug(f"Feed {podcast_id} has CharacterEncodingOverride but has entries")
                else:
                    logger.warning(f"Feed {podcast_id} bozo: {type(parsed_feed.bozo_exception).__name__}")
                    if not parsed_feed.entries and attempt < RSS_RETRY_ATTEMPTS - 1:
                        time.sleep(RSS_RETRY_DELAY * (attempt + 1))
                        continue
            
            if parsed_feed.entries:
                # Log success
                with open(RSS_SUCCESS_LOG, 'a') as f:
                    f.write(f"{podcast_id},{rss_url},success,{len(parsed_feed.entries)}\n")
                return parsed_feed
            
        except socket.timeout:
            logger.warning(f"Timeout fetching {podcast_id} (attempt {attempt + 1})")
        except Exception as e:
            logger.error(f"Error fetching {podcast_id}: {type(e).__name__} - {e}")
        finally:
            socket.setdefaulttimeout(original_timeout)
        
        if attempt < RSS_RETRY_ATTEMPTS - 1:
            time.sleep(RSS_RETRY_DELAY * (attempt + 1))
    
    # Log failure
    with open(RSS_SUCCESS_LOG, 'a') as f:
        f.write(f"{podcast_id},{rss_url},failed,0\n")
    
    return None

def generate_episode_id(podcast_id, entry):
    """Generate unique episode ID"""
    guid = entry.get('guid', None)
    if guid and isinstance(guid, str) and len(guid) > 10:
        return hashlib.sha256(guid.encode('utf-8', 'ignore')).hexdigest()[:64]
    
    components = [
        str(podcast_id),
        clean_text(entry.get('title', '')),
        str(entry.get('published_parsed', '')),
        clean_text(entry.get('link', ''))[:100]
    ]
    unique_str = ''.join(str(c) for c in components if c is not None).encode('utf-8', 'ignore')
    return hashlib.sha256(unique_str).hexdigest()

def get_robust_description(entry):
    """Extract description from various RSS fields"""
    parts = []
    
    # Check content field
    if 'content' in entry and entry.content:
        for content_item in entry.content:
            if hasattr(content_item, 'value') and content_item.value:
                parts.append(content_item.value)
    
    # Check various description fields
    for key in ['itunes_summary', 'description', 'summary', 'subtitle', 'itunes_subtitle']:
        value = entry.get(key)
        if value and isinstance(value, str):
            parts.append(value)
    
    # Check summary_detail
    summary_detail = entry.get('summary_detail')
    if summary_detail and isinstance(summary_detail, dict):
        value = summary_detail.get('value')
        if value and isinstance(value, str):
            parts.append(value)
    
    # Check media_description
    media_description = entry.get('media_description')
    if media_description and isinstance(media_description, str):
        parts.append(media_description)
    
    # Remove duplicates while preserving order
    unique_parts = list(dict.fromkeys(parts))
    return clean_text(' '.join(unique_parts).strip())

def fetch_and_filter_episodes_enhanced(podcast_info):
    """Enhanced episode fetching with better reliability"""
    rss_url = podcast_info.get('rss')
    podcast_id = podcast_info.get('podcastID')
    
    try:
        completion_timestamp_val = podcast_info.get('completion_timestamp')
        current_timestamp = datetime.now(timezone.utc)
        
        if not rss_url or not podcast_id or completion_timestamp_val is None or pd.isna(completion_timestamp_val):
            logger.info(f"Skipping {podcast_id}: Missing essential info")
            return []
        
        episodes = []
        parsed_feed = fetch_rss_with_retry(podcast_info)
        
        if not parsed_feed or not parsed_feed.entries:
            return []
        
        logger.debug(f"Processing {len(parsed_feed.entries)} entries for {podcast_id}")
        
        for entry_idx, entry in enumerate(parsed_feed.entries):
            published_parsed_struct = entry.get('published_parsed') or entry.get('updated_parsed')
            if not published_parsed_struct:
                continue
            
            try:
                published_date = datetime(*published_parsed_struct[:6], tzinfo=timezone.utc)
            except (ValueError, TypeError) as e:
                logger.info(f"P:{podcast_id} E:{entry_idx+1} - Invalid date: {e}")
                continue
            
            if published_date > completion_timestamp_val and published_date <= current_timestamp:
                description = get_robust_description(entry)
                title = clean_text(entry.get('title', ''))
                
                if not description and not title:
                    continue
                
                episode_id_val = generate_episode_id(str(podcast_id), entry)
                
                # Extract guests using multiple methods
                pattern_guests = extract_guests_with_patterns(title, description)
                ner_guests = extract_guests_with_ner(f"{title} {description}")
                
                # Combine and deduplicate guests
                all_guests = pattern_guests + ner_guests
                unique_guests = {}
                for guest in all_guests:
                    name_key = guest['name'].lower()
                    if name_key not in unique_guests or guest['confidence'] > unique_guests[name_key]['confidence']:
                        unique_guests[name_key] = guest
                
                # Extract context for each guest
                for guest in unique_guests.values():
                    context = extract_guest_context(f"{title} {description}", guest['name'])
                    guest.update(context)
                
                episodes.append({
                    'podcast_id': str(podcast_id),
                    'episode_id': episode_id_val,
                    'unique_id': f"{str(podcast_id)}-{episode_id_val}",
                    'episode_title': title,
                    'episode_description': description,
                    'episode_published_utc': published_date,
                    'extracted_guests': list(unique_guests.values())
                })
        
        return episodes
        
    except Exception as e:
        logger.error(f"Error in fetch_and_filter_episodes_enhanced for {podcast_id}: {e}", exc_info=True)
        return []

# Enhanced LLM prompt with guest context
sys_prompt_llm_enhanced = """You are an expert at analyzing podcast episode descriptions to identify and classify guests.

IMPORTANT DEFINITIONS:
- GUEST: Someone interviewed or featured in the episode who is NOT a regular host, co-host, or show member
- DO NOT COUNT: Hosts, co-hosts, regular show members, moderators, or family members of guests

You will be provided with:
1. Episode title and description
2. A list of potential guests extracted using NER and pattern matching
3. Context about each potential guest (affiliation, title, book, etc.)

Your task is to:
1. Validate which extracted names are actual guests (not hosts)
2. Identify any missed guests from the description
3. Classify each guest's demographics based on their name and context

For demographics:
- URM (Underrepresented Minority): Hispanic/Latino, Black/African American, or Native American
- Use context clues (affiliation, expertise area) to inform your classification when names are ambiguous

Output format (JSON):
{
    "validated_guests": [
        {
            "name": "Full Name",
            "is_guest": true/false,
            "gender": "Male/Female/Unknown",
            "is_urm": true/false,
            "urm_category": "Hispanic/Latino|Black|Native American|None",
            "confidence": 0.0-1.0,
            "reasoning": "Brief explanation"
        }
    ],
    "total_guests": number,
    "urm_guests": number,
    "female_guests": number,
    "extraction_quality": "high/medium/low",
    "notes": "Any important observations"
}
"""

def classify_episode_guests_enhanced(episode_info):
    """Enhanced guest classification with context"""
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    episode_title = episode_info.get("episode_title", "")
    episode_description = episode_info.get("episode_description", "")
    extracted_guests = episode_info.get("extracted_guests", [])
    
    # Prepare the enhanced prompt
    clean_title = "".join(filter(lambda x: x in printable, episode_title))
    clean_description = "".join(filter(lambda x: x in printable, episode_description))
    
    # Truncate if too long
    max_desc_len = 6000
    if len(clean_description) > max_desc_len:
        clean_description = clean_description[:max_desc_len] + "..."
    
    user_prompt = json.dumps({
        "episode_title": clean_title,
        "episode_description": clean_description,
        "extracted_guests": extracted_guests
    })
    
    base_result = {
        "podcast_id": episode_info['podcast_id'],
        "episode_id": episode_info['episode_id'],
        "unique_id": episode_info['unique_id'],
        "episode_published_utc": episode_info['episode_published_utc'].isoformat() if episode_info['episode_published_utc'] else None,
        "total_guests": 0,
        "urm_guests": 0,
        "female_guests": 0,
        "validated_guests": [],
        "extraction_quality": "low",
        "llm_error": None
    }
    
    for attempt in range(LLM_RETRY_ATTEMPTS):
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": sys_prompt_llm_enhanced},
                    {"role": "user", "content": user_prompt}
                ],
                model=LLM_MODEL,
                response_format={"type": "json_object"},
                temperature=0.1
            )
            
            gpt_response = chat_completion.choices[0].message.content
            parsed_data = json.loads(gpt_response)
            
            # Process the response
            validated_guests = parsed_data.get('validated_guests', [])
            
            # Count demographics
            total_guests = sum(1 for g in validated_guests if g.get('is_guest', False))
            urm_guests = sum(1 for g in validated_guests if g.get('is_guest', False) and g.get('is_urm', False))
            female_guests = sum(1 for g in validated_guests if g.get('is_guest', False) and g.get('gender') == 'Female')
            
            base_result.update({
                "total_guests": total_guests,
                "urm_guests": urm_guests,
                "female_guests": female_guests,
                "validated_guests": validated_guests,
                "extraction_quality": parsed_data.get('extraction_quality', 'medium'),
                "notes": parsed_data.get('notes', ''),
                "llm_error": None
            })
            
            return base_result
            
        except Exception as e:
            error_msg = f"LLM Error attempt {attempt+1}/{LLM_RETRY_ATTEMPTS}: {type(e).__name__} - {e}"
            base_result["llm_error"] = error_msg
            logger.warning(error_msg)
            
            if attempt < LLM_RETRY_ATTEMPTS - 1:
                time.sleep(LLM_RETRY_DELAY * (attempt + 1))
    
    return base_result

# Import guest enrichment module
try:
    from guest_enrichment import enrich_guests_batch, GuestEnricher
    ENRICHMENT_AVAILABLE = True
except ImportError:
    logger.warning("Guest enrichment module not available. Will use basic analysis only.")
    ENRICHMENT_AVAILABLE = False

def process_podcast_enhanced(podcast_dict):
    """Enhanced podcast processing wrapper"""
    podcast_id = podcast_dict.get('podcastID', 'Unknown')
    try:
        episodes = fetch_and_filter_episodes_enhanced(podcast_dict)
        return episodes
    except Exception as e:
        logger.error(f"Error processing podcast {podcast_id}: {e}", exc_info=True)
        return []

def aggregate_results_enhanced(llm_output_file):
    """Enhanced aggregation with guest-level data"""
    logger.info(f"Starting enhanced aggregation from {llm_output_file}...")
    
    results = []
    all_guests = []
    
    try:
        with open(llm_output_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                try:
                    result = json.loads(line)
                    results.append(result)
                    
                    # Extract individual guest data
                    if 'validated_guests' in result:
                        for guest in result['validated_guests']:
                            if guest.get('is_guest', False):
                                guest_record = {
                                    'podcast_id': result['podcast_id'],
                                    'episode_id': result['episode_id'],
                                    'episode_date': result.get('episode_published_utc'),
                                    'guest_name': guest['name'],
                                    'gender': guest.get('gender', 'Unknown'),
                                    'is_urm': guest.get('is_urm', False),
                                    'urm_category': guest.get('urm_category', 'None'),
                                    'confidence': guest.get('confidence', 0.0),
                                    'extraction_method': guest.get('extraction_method', 'llm')
                                }
                                all_guests.append(guest_record)
                                
                except json.JSONDecodeError:
                    logger.warning(f"Skipping invalid JSON on line {line_num+1}")
                    continue
                    
    except FileNotFoundError:
        logger.error(f"LLM output file not found: {llm_output_file}")
        return pd.DataFrame(), pd.DataFrame()
    except Exception as e:
        logger.error(f"Error reading LLM output file: {e}", exc_info=True)
        return pd.DataFrame(), pd.DataFrame()
    
    if not results:
        logger.warning("No results loaded from LLM output file")
        return pd.DataFrame(), pd.DataFrame()
    
    # Create DataFrames
    df_episodes = pd.DataFrame(results)
    df_guests = pd.DataFrame(all_guests) if all_guests else pd.DataFrame()
    
    # Save guest-level data
    if not df_guests.empty:
        df_guests.to_csv(ENRICHED_GUESTS_CSV, index=False)
        logger.info(f"Saved {len(df_guests)} guest records to {ENRICHED_GUESTS_CSV}")
    
    # Process episode-level aggregation
    count_cols = ['total_guests', 'urm_guests', 'female_guests']
    for col in count_cols:
        df_episodes[col] = pd.to_numeric(df_episodes[col], errors='coerce')
    
    # Filter valid rows
    valid_rows_mask = df_episodes[count_cols].notna().all(axis=1) & df_episodes['llm_error'].isna()
    df_valid = df_episodes[valid_rows_mask].copy()
    
    if df_valid.empty:
        logger.warning("No valid LLM classifications found")
        return pd.DataFrame(), df_guests
    
    # Calculate percentages
    df_valid['ep_pct_female'] = np.where(
        df_valid['total_guests'] > 0,
        (df_valid['female_guests'] / df_valid['total_guests']) * 100,
        np.nan
    )
    
    df_valid['ep_pct_urm'] = np.where(
        df_valid['total_guests'] > 0,
        (df_valid['urm_guests'] / df_valid['total_guests']) * 100,
        np.nan
    )
    
    # Aggregate by podcast
    summary = df_valid.groupby('podcast_id').agg(
        post_total_episodes_analyzed=pd.NamedAgg(column='episode_id', aggfunc='nunique'),
        post_total_guests_sum=pd.NamedAgg(column='total_guests', aggfunc='sum'),
        post_urm_guests_sum=pd.NamedAgg(column='urm_guests', aggfunc='sum'),
        post_female_guests_sum=pd.NamedAgg(column='female_guests', aggfunc='sum'),
        post_avg_ep_pct_female=pd.NamedAgg(column='ep_pct_female', aggfunc=lambda x: np.nanmean(x) if not x.isnull().all() else 0.0),
        post_avg_ep_pct_urm=pd.NamedAgg(column='ep_pct_urm', aggfunc=lambda x: np.nanmean(x) if not x.isnull().all() else 0.0),
        extraction_quality_score=pd.NamedAgg(column='extraction_quality', aggfunc=lambda x: (x == 'high').sum() / len(x) if len(x) > 0 else 0.0)
    ).reset_index()
    
    # Calculate overall percentages
    summary['post_overall_pct_urm'] = np.where(
        summary['post_total_guests_sum'] > 0,
        (summary['post_urm_guests_sum'] / summary['post_total_guests_sum']) * 100,
        0.0
    )
    
    summary['post_overall_pct_female'] = np.where(
        summary['post_total_guests_sum'] > 0,
        (summary['post_female_guests_sum'] / summary['post_total_guests_sum']) * 100,
        0.0
    )
    
    # Rename columns
    summary.rename(columns={
        'post_total_guests_sum': 'post_total_guests',
        'post_urm_guests_sum': 'post_urm_guests',
        'post_female_guests_sum': 'post_female_guests',
        'podcast_id': 'podcastID'
    }, inplace=True)
    
    logger.info(f"Aggregation complete. Produced data for {len(summary)} podcasts")
    
    return summary, df_guests

def analyze_rss_success_rate():
    """Analyze RSS fetching success rate"""
    if not os.path.exists(RSS_SUCCESS_LOG):
        return
    
    try:
        df = pd.read_csv(RSS_SUCCESS_LOG, names=['podcast_id', 'url', 'status', 'episode_count'])
        
        total_attempts = len(df)
        successful = len(df[df['status'] == 'success'])
        success_rate = (successful / total_attempts * 100) if total_attempts > 0 else 0
        
        logger.info(f"\n=== RSS Fetching Success Rate ===")
        logger.info(f"Total attempts: {total_attempts}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {total_attempts - successful}")
        logger.info(f"Success rate: {success_rate:.2f}%")
        
        if success_rate < 95:
            logger.warning(f"Success rate below 95% target!")
            
            # Analyze failure patterns
            failed_df = df[df['status'] == 'failed']
            if not failed_df.empty:
                logger.info("\nAnalyzing failure patterns...")
                # Could add more detailed analysis here
                
    except Exception as e:
        logger.error(f"Error analyzing RSS success rate: {e}")

def summarize_by_condition_enhanced(aggregated_file, input_file, guest_file=None):
    """Enhanced summary with additional metrics"""
    podcasts_in_final_summary = 0
    
    try:
        # Load aggregated podcast data
        agg_df = pd.read_csv(aggregated_file)
        if 'podcastID' not in agg_df.columns:
            logger.error(f"Required column 'podcastID' not found in {aggregated_file}")
            return 0
            
    except Exception as e:
        logger.error(f"Error reading aggregated file: {e}")
        return 0
    
    try:
        # Load input data with treatment info
        input_df = pd.read_csv(input_file, usecols=['podcastID', 'treatment'])
        if 'treatment' not in input_df.columns:
            logger.error(f"Required column 'treatment' not found in {input_file}")
            return 0
            
    except Exception as e:
        logger.error(f"Error reading input file: {e}")
        return 0
    
    # Merge data
    summary_df = pd.merge(input_df, agg_df, on='podcastID', how='inner')
    podcasts_in_final_summary = summary_df['podcastID'].nunique()
    
    logger.info(f"Merged data successfully. {podcasts_in_final_summary} podcasts in final summary.")
    
    if summary_df.empty:
        logger.warning("No podcasts remaining after merge")
        return 0
    
    # Add extraction quality analysis
    quality_summary = summary_df.groupby('treatment')['extraction_quality_score'].agg(['mean', 'std']).reset_index()
    
    # Standard condition summary
    condition_summary = summary_df.groupby('treatment').agg(
        n_podcasts=pd.NamedAgg(column='podcastID', aggfunc='nunique'),
        mean_post_overall_pct_female=pd.NamedAgg(column='post_overall_pct_female', aggfunc='mean'),
        sd_post_overall_pct_female=pd.NamedAgg(column='post_overall_pct_female', aggfunc='std'),
        mean_post_overall_pct_urm=pd.NamedAgg(column='post_overall_pct_urm', aggfunc='mean'),
        sd_post_overall_pct_urm=pd.NamedAgg(column='post_overall_pct_urm', aggfunc='std'),
        mean_post_avg_ep_pct_female=pd.NamedAgg(column='post_avg_ep_pct_female', aggfunc='mean'),
        sd_post_avg_ep_pct_female=pd.NamedAgg(column='post_avg_ep_pct_female', aggfunc='std'),
        mean_post_avg_ep_pct_urm=pd.NamedAgg(column='post_avg_ep_pct_urm', aggfunc='mean'),
        sd_post_avg_ep_pct_urm=pd.NamedAgg(column='post_avg_ep_pct_urm', aggfunc='std'),
        total_episodes=pd.NamedAgg(column='post_total_episodes_analyzed', aggfunc='sum'),
        total_guests=pd.NamedAgg(column='post_total_guests', aggfunc='sum')
    ).reset_index()
    
    # Generate output
    output_lines = [
        "\n=== Enhanced Post-Intervention Guest Diversity Analysis ===",
        f"Analysis Date: {datetime.now().isoformat()}",
        f"Total Podcasts Analyzed: {podcasts_in_final_summary}",
        ""
    ]
    
    # Add extraction quality info
    output_lines.append("--- Extraction Quality Summary ---")
    for _, row in quality_summary.iterrows():
        group = "Treatment" if row['treatment'] == 1 else "Control"
        output_lines.append(f"{group}: {row['mean']:.2%} high-quality extractions (SD={row['std']:.2%})")
    
    output_lines.append("\n--- Guest Diversity by Condition ---")
    
    for _, row in condition_summary.iterrows():
        group = "Treatment" if row['treatment'] == 1 else "Control"
        output_lines.extend([
            f"\nGroup: {group}",
            f"  Podcasts: {row['n_podcasts']:.0f}",
            f"  Episodes: {row['total_episodes']:.0f}",
            f"  Total Guests: {row['total_guests']:.0f}",
            f"  Overall % Female: {row['mean_post_overall_pct_female']:.2f}% (SD={row['sd_post_overall_pct_female']:.2f})",
            f"  Overall % URM: {row['mean_post_overall_pct_urm']:.2f}% (SD={row['sd_post_overall_pct_urm']:.2f})",
            f"  Per-Episode % Female: {row['mean_post_avg_ep_pct_female']:.2f}% (SD={row['sd_post_avg_ep_pct_female']:.2f})",
            f"  Per-Episode % URM: {row['mean_post_avg_ep_pct_urm']:.2f}% (SD={row['sd_post_avg_ep_pct_urm']:.2f})"
        ])
    
    # Statistical tests
    output_lines.append("\n--- Statistical Tests (Treatment vs. Control) ---")
    control_group = summary_df[summary_df['treatment'] == 0]
    treatment_group = summary_df[summary_df['treatment'] == 1]
    
    if len(control_group) < 2 or len(treatment_group) < 2:
        output_lines.append("  Insufficient data for t-tests")
    else:
        metrics = {
            "Overall % Female": "post_overall_pct_female",
            "Overall % URM": "post_overall_pct_urm",
            "Per-Episode % Female": "post_avg_ep_pct_female",
            "Per-Episode % URM": "post_avg_ep_pct_urm"
        }
        
        for name, col in metrics.items():
            ctrl_vals = control_group[col].dropna()
            treat_vals = treatment_group[col].dropna()
            
            if len(ctrl_vals) >= 2 and len(treat_vals) >= 2:
                try:
                    t_stat, p_val = stats.ttest_ind(treat_vals, ctrl_vals, equal_var=False)
                    cohen_d = (treat_vals.mean() - ctrl_vals.mean()) / np.sqrt((treat_vals.var() + ctrl_vals.var()) / 2)
                    output_lines.append(f"  {name}: t={t_stat:.3f}, p={p_val:.4f}, d={cohen_d:.3f}")
                except Exception as e:
                    output_lines.append(f"  {name}: Error - {e}")
    
    # Guest-level analysis if available
    if guest_file and os.path.exists(guest_file):
        try:
            guest_df = pd.read_csv(guest_file)
            output_lines.extend([
                "\n--- Guest-Level Analysis ---",
                f"Total unique guests: {guest_df['guest_name'].nunique()}",
                f"Gender distribution: {guest_df['gender'].value_counts().to_dict()}",
                f"URM categories: {guest_df[guest_df['is_urm']]['urm_category'].value_counts().to_dict()}"
            ])
        except Exception as e:
            logger.error(f"Error loading guest file: {e}")
    
    output_lines.append("\n" + "="*60)
    
    final_output = '\n'.join(output_lines)
    print(final_output)
    
    # Save to log
    with open(ERROR_LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(final_output)
    
    return podcasts_in_final_summary

# Main pipeline
def main():
    """Enhanced main pipeline"""
    start_time = time.time()
    logger.info(f"=== Enhanced Pipeline Start: {datetime.now().isoformat()} ===")
    
    # Initialize RSS success log
    with open(RSS_SUCCESS_LOG, 'w') as f:
        f.write("")  # Clear previous log
    
    podcast_counts = {
        "initial_csv_rows": 0,
        "after_essential_dropna": 0,
        "with_any_fetched_episodes": 0,
        "total_relevant_episodes": 0,
        "meeting_min_episode_criteria": 0,
        "final_episodes_for_llm": 0,
        "with_aggregated_llm_data": 0,
        "in_final_condition_summary": 0
    }
    
    try:
        # Load input data
        podcasts_df_initial = pd.read_csv(INPUT_CSV)
        podcast_counts["initial_csv_rows"] = len(podcasts_df_initial)
        logger.info(f"Loaded {podcast_counts['initial_csv_rows']} rows from {INPUT_CSV}")
        
        if 'treatment' not in podcasts_df_initial.columns:
            logger.critical("'treatment' column missing!")
            return
        
        # Data preparation
        podcasts_df = podcasts_df_initial.copy()
        podcasts_df['latest_response_time'] = pd.to_datetime(
            podcasts_df['latest_response_time'], 
            errors='coerce', 
            utc=True
        )
        
        essential_cols = ['podcastID', 'rss', 'latest_response_time', 'treatment']
        initial_rows = len(podcasts_df)
        podcasts_df.dropna(subset=essential_cols, inplace=True)
        podcast_counts["after_essential_dropna"] = len(podcasts_df)
        
        logger.info(f"{podcast_counts['after_essential_dropna']} rows after dropping missing data")
        
        if podcasts_df.empty:
            logger.info("No podcasts to process")
            return
        
        # RSS fetching phase
        logger.info(f"Starting enhanced RSS feed fetching for {len(podcasts_df)} podcasts...")
        
        podcasts_df_renamed = podcasts_df.rename(columns={'latest_response_time': 'completion_timestamp'})
        podcast_info_list = podcasts_df_renamed.to_dict('records')
        
        # Process in batches
        BATCH_SIZE = 500
        all_relevant_episodes = []
        num_workers = min(8, multiprocessing.cpu_count())
        
        logger.info(f"Using {num_workers} workers for RSS fetching")
        
        with tqdm(total=len(podcast_info_list), desc="Fetching RSS feeds") as pbar:
            for i in range(0, len(podcast_info_list), BATCH_SIZE):
                batch = podcast_info_list[i:i + BATCH_SIZE]
                
                try:
                    with multiprocessing.Pool(processes=num_workers) as pool:
                        batch_results = list(pool.imap_unordered(process_podcast_enhanced, batch))
                        
                    for episodes in batch_results:
                        if episodes:
                            all_relevant_episodes.extend(episodes)
                    
                    pbar.update(len(batch))
                    
                except Exception as e:
                    logger.error(f"Error in batch {i//BATCH_SIZE + 1}: {e}")
                
                gc.collect()
        
        # Analyze RSS success rate
        analyze_rss_success_rate()
        
        podcast_counts["total_relevant_episodes"] = len(all_relevant_episodes)
        
        if all_relevant_episodes:
            # Count unique podcasts
            unique_podcasts = set(ep['podcast_id'] for ep in all_relevant_episodes)
            podcast_counts["with_any_fetched_episodes"] = len(unique_podcasts)
            
            logger.info(f"Fetched {podcast_counts['total_relevant_episodes']} episodes from {podcast_counts['with_any_fetched_episodes']} podcasts")
        else:
            logger.warning("No relevant episodes found")
            return
        
        # Filter by minimum episodes
        episodes_df = pd.DataFrame(all_relevant_episodes)
        episode_counts = episodes_df.groupby('podcast_id')['episode_id'].nunique()
        valid_podcasts = episode_counts[episode_counts >= MIN_REQUIRED_EPISODES].index
        
        episodes_to_process = episodes_df[episodes_df['podcast_id'].isin(valid_podcasts)]
        podcast_counts["meeting_min_episode_criteria"] = len(valid_podcasts)
        podcast_counts["final_episodes_for_llm"] = len(episodes_to_process)
        
        logger.info(f"{podcast_counts['meeting_min_episode_criteria']} podcasts meet minimum episode criteria")
        logger.info(f"{podcast_counts['final_episodes_for_llm']} episodes for LLM classification")
        
        # LLM classification phase
        if not episodes_to_process.empty:
            logger.info("Starting enhanced LLM classification...")
            
            episodes_list = episodes_to_process.to_dict('records')
            
            # Optional: Enrich guests before LLM classification
            if ENRICHMENT_AVAILABLE and (SERPAPI_KEY or GOOGLE_SEARCH_API_KEY):
                logger.info("Enriching guest information...")
                
                # Extract all unique guests for enrichment
                all_guests_to_enrich = []
                for episode in episodes_list:
                    for guest in episode.get('extracted_guests', []):
                        all_guests_to_enrich.append({
                            'name': guest['name'],
                            'context': {
                                'affiliation': guest.get('affiliation'),
                                'title': guest.get('title'),
                                'book': guest.get('book')
                            }
                        })
                
                # Deduplicate
                unique_guests = {}
                for guest in all_guests_to_enrich:
                    key = guest['name'].lower()
                    if key not in unique_guests:
                        unique_guests[key] = guest
                
                # Enrich in batches
                enriched_data = enrich_guests_batch(list(unique_guests.values()), max_workers=5)
                
                # Create enrichment lookup
                enrichment_lookup = {
                    data['guest_name'].lower(): data 
                    for data in enriched_data
                }
                
                # Add enrichment data back to episodes
                for episode in episodes_list:
                    for guest in episode.get('extracted_guests', []):
                        enrichment = enrichment_lookup.get(guest['name'].lower(), {})
                        guest['enrichment'] = enrichment
            
            # LLM classification
            with open(RAW_LLM_OUTPUT_JSONL, 'w', encoding='utf-8') as fout:
                with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_LLM_WORKERS) as executor:
                    futures = {
                        executor.submit(classify_episode_guests_enhanced, ep): ep['unique_id'] 
                        for ep in episodes_list
                    }
                    
                    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Classifying episodes"):
                        try:
                            result = future.result()
                            fout.write(json.dumps(result) + '\n')
                        except Exception as e:
                            uid = futures[future]
                            logger.error(f"Error classifying {uid}: {e}")
                            fout.write(json.dumps({"unique_id": uid, "llm_error": str(e)}) + '\n')
            
            logger.info("LLM classification complete")
        
        # Aggregation phase
        aggregated_df, guest_df = aggregate_results_enhanced(RAW_LLM_OUTPUT_JSONL)
        
        if not aggregated_df.empty:
            podcast_counts["with_aggregated_llm_data"] = aggregated_df['podcastID'].nunique()
            
            try:
                aggregated_df.to_csv(AGGREGATED_OUTPUT_CSV, index=False, encoding='utf-8')
                logger.info(f"Saved aggregated results for {podcast_counts['with_aggregated_llm_data']} podcasts")
            except Exception as e:
                logger.error(f"Failed to save aggregated results: {e}")
        
        # Final summary
        if os.path.exists(AGGREGATED_OUTPUT_CSV) and os.path.getsize(AGGREGATED_OUTPUT_CSV) > 0:
            podcasts_in_summary = summarize_by_condition_enhanced(
                AGGREGATED_OUTPUT_CSV, 
                INPUT_CSV,
                ENRICHED_GUESTS_CSV if os.path.exists(ENRICHED_GUESTS_CSV) else None
            )
            podcast_counts["in_final_condition_summary"] = podcasts_in_summary
        
    except Exception as e:
        logger.critical(f"Critical error in main pipeline: {e}", exc_info=True)
    
    # Final summary
    end_time = time.time()
    duration = end_time - start_time
    
    logger.info(f"\n=== Pipeline Summary ===")
    logger.info(f"Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    logger.info(f"Podcast counts: {json.dumps(podcast_counts, indent=2)}")
    logger.info(f"=== Pipeline End: {datetime.now().isoformat()} ===")

def signal_handler(signum, frame):
    """Handle interruption signals"""
    logger.warning(f"Signal {signal.Signals(signum).name} received. Shutting down...")
    sys.exit(0)

if __name__ == "__main__":
    print(f"Enhanced Post-Intervention Analysis v2.0")
    print(f"Python: {sys.version}")
    print(f"Working Directory: {os.getcwd()}")
    
    multiprocessing.freeze_support()
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except SystemExit:
        pass
    except Exception as e:
        logger.critical(f"Unhandled exception: {e}", exc_info=True)