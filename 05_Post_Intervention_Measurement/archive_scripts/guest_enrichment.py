#!/usr/bin/env python3
"""
Guest Enrichment Module - Web search and photo analysis for demographic determination
"""

import os
import json
import time
import requests
from typing import Dict, List, Optional, Tuple
import logging
from PIL import Image
from io import BytesIO
import cv2
import numpy as np
from deepface import DeepFace
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote_plus
import hashlib

logger = logging.getLogger(__name__)

# Configuration
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
GOOGLE_SEARCH_API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

# Directories
GUEST_PHOTOS_DIR = "guest_photos"
ENRICHMENT_CACHE_DIR = "enrichment_cache"
os.makedirs(GUEST_PHOTOS_DIR, exist_ok=True)
os.makedirs(ENRICHMENT_CACHE_DIR, exist_ok=True)

# DeepFace configuration
DEEPFACE_BACKENDS = ['opencv', 'retinaface', 'mtcnn']
DEEPFACE_MODELS = ['VGG-Face', 'Facenet', 'OpenFace', 'DeepFace']

# Preload DeepFace models
def preload_deepface_models():
    """Preload DeepFace models for faster analysis"""
    try:
        sample = np.zeros((100, 100, 3), dtype=np.uint8)
        _ = DeepFace.analyze(
            sample, 
            actions=['gender', 'race'], 
            detector_backend='opencv', 
            enforce_detection=False, 
            silent=True
        )
        logger.info("DeepFace models preloaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error preloading DeepFace models: {e}")
        return False

class GuestEnricher:
    """Enriches guest information with web data and photo analysis"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        preload_deepface_models()
    
    def get_cache_key(self, guest_name: str, context: Dict) -> str:
        """Generate cache key for guest"""
        cache_str = f"{guest_name}_{context.get('affiliation', '')}_{context.get('book', '')}"
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def load_from_cache(self, cache_key: str) -> Optional[Dict]:
        """Load enrichment data from cache"""
        cache_file = os.path.join(ENRICHMENT_CACHE_DIR, f"{cache_key}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return None
    
    def save_to_cache(self, cache_key: str, data: Dict):
        """Save enrichment data to cache"""
        cache_file = os.path.join(ENRICHMENT_CACHE_DIR, f"{cache_key}.json")
        try:
            with open(cache_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.error(f"Error saving to cache: {e}")
    
    def search_guest_info(self, guest_name: str, context: Dict) -> Dict:
        """Search for additional guest information"""
        search_queries = []
        
        # Build targeted search queries
        base_query = f'"{guest_name}"'
        
        if context.get('affiliation'):
            search_queries.append(f'{base_query} {context["affiliation"]}')
        
        if context.get('book'):
            search_queries.append(f'{base_query} author "{context["book"]}"')
        
        if context.get('title'):
            search_queries.append(f'{base_query} {context["title"]}')
        
        # Default query if no context
        if not search_queries:
            search_queries.append(f'{base_query} biography')
        
        results = {
            'additional_info': {},
            'image_urls': [],
            'web_sources': []
        }
        
        for query in search_queries[:2]:  # Limit queries
            try:
                # Try SerpAPI first
                if SERPAPI_KEY:
                    serp_results = self._search_serpapi(query)
                    results['image_urls'].extend(serp_results.get('images', []))
                    results['web_sources'].extend(serp_results.get('sources', []))
                
                # Fallback to Google CSE
                elif GOOGLE_SEARCH_API_KEY and GOOGLE_CSE_ID:
                    google_results = self._search_google_cse(query)
                    results['image_urls'].extend(google_results.get('images', []))
                    results['web_sources'].extend(google_results.get('sources', []))
                
            except Exception as e:
                logger.error(f"Error searching for {guest_name}: {e}")
        
        # Extract structured data from web sources
        if results['web_sources']:
            results['additional_info'] = self._extract_structured_data(
                results['web_sources'], guest_name
            )
        
        return results
    
    def _search_serpapi(self, query: str) -> Dict:
        """Search using SerpAPI"""
        params = {
            "q": query,
            "api_key": SERPAPI_KEY,
            "num": 10
        }
        
        results = {'images': [], 'sources': []}
        
        try:
            # Text search
            response = self.session.get(
                "https://serpapi.com/search", 
                params=params, 
                timeout=15
            )
            data = response.json()
            
            # Extract web sources
            if "organic_results" in data:
                for result in data["organic_results"][:5]:
                    results['sources'].append({
                        'url': result.get('link'),
                        'title': result.get('title'),
                        'snippet': result.get('snippet')
                    })
            
            # Image search
            params['tbm'] = 'isch'
            response = self.session.get(
                "https://serpapi.com/search", 
                params=params, 
                timeout=15
            )
            data = response.json()
            
            if "images_results" in data:
                for img in data["images_results"][:5]:
                    if img.get("original"):
                        results['images'].append(img["original"])
            
        except Exception as e:
            logger.error(f"SerpAPI error: {e}")
        
        return results
    
    def _search_google_cse(self, query: str) -> Dict:
        """Search using Google Custom Search API"""
        results = {'images': [], 'sources': []}
        
        try:
            # Text search
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "key": GOOGLE_SEARCH_API_KEY,
                "cx": GOOGLE_CSE_ID,
                "q": query,
                "num": 10
            }
            
            response = self.session.get(url, params=params, timeout=15)
            data = response.json()
            
            if "items" in data:
                for item in data["items"][:5]:
                    results['sources'].append({
                        'url': item.get('link'),
                        'title': item.get('title'),
                        'snippet': item.get('snippet')
                    })
            
            # Image search
            params['searchType'] = 'image'
            params['num'] = 5
            
            response = self.session.get(url, params=params, timeout=15)
            data = response.json()
            
            if "items" in data:
                for item in data["items"]:
                    if item.get('link'):
                        results['images'].append(item['link'])
            
        except Exception as e:
            logger.error(f"Google CSE error: {e}")
        
        return results
    
    def _extract_structured_data(self, sources: List[Dict], guest_name: str) -> Dict:
        """Extract structured information from web sources"""
        info = {
            'verified_affiliation': None,
            'verified_title': None,
            'additional_context': []
        }
        
        # Simple extraction from snippets
        for source in sources:
            snippet = source.get('snippet', '').lower()
            
            # Look for university affiliations
            if 'university' in snippet or 'professor' in snippet:
                # Extract university name
                import re
                uni_match = re.search(
                    r'(?:at|from)\s+(\w+\s+university|\w+\s+college)', 
                    snippet, 
                    re.IGNORECASE
                )
                if uni_match and not info['verified_affiliation']:
                    info['verified_affiliation'] = uni_match.group(1)
            
            # Look for titles
            title_keywords = ['professor', 'dr.', 'phd', 'author', 'ceo', 'founder', 'director']
            for keyword in title_keywords:
                if keyword in snippet and not info['verified_title']:
                    info['verified_title'] = keyword.title()
                    break
        
        return info
    
    def download_and_analyze_photo(self, image_url: str, guest_id: str) -> Optional[Dict]:
        """Download and analyze a photo using DeepFace"""
        try:
            # Download image
            response = self.session.get(image_url, timeout=15)
            image = Image.open(BytesIO(response.content))
            
            # Convert to RGB if necessary
            if image.mode in ('RGBA', 'LA', 'P'):
                image = image.convert('RGB')
            
            # Resize if too large
            max_size = (800, 800)
            image.thumbnail(max_size, Image.LANCZOS)
            
            # Save image
            image_path = os.path.join(GUEST_PHOTOS_DIR, f"{guest_id}.jpg")
            image.save(image_path, format='JPEG', quality=85)
            
            # Analyze with DeepFace
            results = []
            for backend in DEEPFACE_BACKENDS[:2]:  # Try first 2 backends
                try:
                    analysis = DeepFace.analyze(
                        img_path=image_path,
                        actions=['gender', 'race'],
                        detector_backend=backend,
                        enforce_detection=True,
                        align=True,
                        silent=True
                    )
                    
                    result = analysis[0] if isinstance(analysis, list) else analysis
                    
                    # Process gender
                    gender_dict = result.get('gender', {})
                    gender = max(gender_dict, key=gender_dict.get) if gender_dict else 'unknown'
                    gender_confidence = gender_dict.get(gender, 0.0) / 100.0
                    gender = {'Man': 'Male', 'Woman': 'Female'}.get(gender, gender)
                    
                    # Process race
                    race_dict = result.get('race', {})
                    race = result.get('dominant_race', 'unknown')
                    race_confidence = race_dict.get(race, 0.0) / 100.0
                    
                    # Map to our categories
                    race_map = {
                        'asian': 'Asian',
                        'black': 'Black',
                        'indian': 'South Asian',
                        'latino hispanic': 'Hispanic/Latino',
                        'middle eastern': 'Middle Eastern',
                        'white': 'White'
                    }
                    race = race_map.get(race.lower(), race)
                    
                    results.append({
                        'backend': backend,
                        'gender': gender,
                        'gender_confidence': gender_confidence,
                        'race': race,
                        'race_confidence': race_confidence,
                        'is_urm': race in ['Black', 'Hispanic/Latino', 'Native American']
                    })
                    
                    # If high confidence, return immediately
                    if gender_confidence > 0.8 and race_confidence > 0.8:
                        return results[0]
                    
                except Exception as e:
                    logger.debug(f"DeepFace {backend} failed: {e}")
                    continue
            
            # Return best result
            if results:
                return max(results, key=lambda x: x['gender_confidence'] + x['race_confidence'])
            
        except Exception as e:
            logger.error(f"Error analyzing photo: {e}")
        
        return None
    
    def enrich_guest(self, guest_name: str, context: Dict) -> Dict:
        """Main enrichment function"""
        cache_key = self.get_cache_key(guest_name, context)
        
        # Check cache
        cached_data = self.load_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        enrichment = {
            'guest_name': guest_name,
            'original_context': context,
            'web_search_performed': False,
            'photo_analysis_performed': False,
            'photo_gender': None,
            'photo_gender_confidence': 0.0,
            'photo_race': None,
            'photo_race_confidence': 0.0,
            'photo_is_urm': None,
            'verified_affiliation': None,
            'verified_title': None,
            'confidence_score': 0.0
        }
        
        # Search for additional information
        try:
            search_results = self.search_guest_info(guest_name, context)
            enrichment['web_search_performed'] = True
            
            # Update with verified information
            additional_info = search_results.get('additional_info', {})
            enrichment['verified_affiliation'] = additional_info.get('verified_affiliation')
            enrichment['verified_title'] = additional_info.get('verified_title')
            
            # Try photo analysis
            image_urls = search_results.get('image_urls', [])
            if image_urls:
                guest_id = hashlib.md5(guest_name.encode()).hexdigest()[:16]
                
                for url in image_urls[:3]:  # Try up to 3 images
                    photo_result = self.download_and_analyze_photo(url, guest_id)
                    if photo_result:
                        enrichment['photo_analysis_performed'] = True
                        enrichment['photo_gender'] = photo_result['gender']
                        enrichment['photo_gender_confidence'] = photo_result['gender_confidence']
                        enrichment['photo_race'] = photo_result['race']
                        enrichment['photo_race_confidence'] = photo_result['race_confidence']
                        enrichment['photo_is_urm'] = photo_result['is_urm']
                        break
            
            # Calculate overall confidence
            confidence_components = []
            if enrichment['photo_gender_confidence'] > 0:
                confidence_components.append(enrichment['photo_gender_confidence'])
            if enrichment['photo_race_confidence'] > 0:
                confidence_components.append(enrichment['photo_race_confidence'])
            if enrichment['verified_affiliation']:
                confidence_components.append(0.7)
            
            if confidence_components:
                enrichment['confidence_score'] = np.mean(confidence_components)
            
        except Exception as e:
            logger.error(f"Error enriching guest {guest_name}: {e}")
        
        # Save to cache
        self.save_to_cache(cache_key, enrichment)
        
        return enrichment

def enrich_guests_batch(guests_info: List[Dict], max_workers: int = 5) -> List[Dict]:
    """Enrich multiple guests in parallel"""
    enricher = GuestEnricher()
    enriched_guests = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_guest = {
            executor.submit(
                enricher.enrich_guest, 
                guest['name'], 
                guest.get('context', {})
            ): guest
            for guest in guests_info
        }
        
        for future in as_completed(future_to_guest):
            guest = future_to_guest[future]
            try:
                enrichment = future.result()
                enrichment['original_guest_info'] = guest
                enriched_guests.append(enrichment)
            except Exception as e:
                logger.error(f"Error enriching {guest['name']}: {e}")
                enriched_guests.append({
                    'guest_name': guest['name'],
                    'error': str(e),
                    'original_guest_info': guest
                })
    
    return enriched_guests