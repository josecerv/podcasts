#!/usr/bin/env python3
"""
Guest Enrichment Module with SerpAPI integration and caching
"""

import os
import json
import hashlib
import logging
import requests
from typing import Dict, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)

class GuestEnricher:
    """Enriches guest information using web search with caching"""
    
    def __init__(self, serpapi_key: str, cache_dir: str = "enrichment_cache"):
        self.serpapi_key = serpapi_key
        self.cache_dir = cache_dir
        self.api_calls_made = 0
        self.cache_hits = 0
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load existing cache index
        self.cache_index_file = os.path.join(cache_dir, "cache_index.json")
        self.cache_index = self._load_cache_index()
    
    def _load_cache_index(self) -> Dict:
        """Load cache index"""
        if os.path.exists(self.cache_index_file):
            try:
                with open(self.cache_index_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_cache_index(self):
        """Save cache index"""
        with open(self.cache_index_file, 'w') as f:
            json.dump(self.cache_index, f, indent=2)
    
    def _get_cache_key(self, guest_name: str) -> str:
        """Generate cache key for guest"""
        return hashlib.md5(guest_name.lower().encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[Dict]:
        """Load enrichment data from cache"""
        if cache_key in self.cache_index:
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'r') as f:
                        self.cache_hits += 1
                        return json.load(f)
                except:
                    pass
        return None
    
    def _save_to_cache(self, cache_key: str, data: Dict):
        """Save enrichment data to cache"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        with open(cache_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Update index
        self.cache_index[cache_key] = {
            'guest_name': data.get('guest_name'),
            'cached_at': datetime.now().isoformat(),
            'has_results': bool(data.get('search_results'))
        }
        self._save_cache_index()
    
    def enrich_guest(self, guest_name: str) -> Dict:
        """Enrich guest information using web search"""
        
        # Check cache first
        cache_key = self._get_cache_key(guest_name)
        cached_data = self._load_from_cache(cache_key)
        if cached_data:
            logger.debug(f"Cache hit for {guest_name}")
            return cached_data
        
        # Prepare enrichment result
        enrichment = {
            'guest_name': guest_name,
            'searched_at': datetime.now().isoformat(),
            'search_results': [],
            'enhanced_gender': None,
            'enhanced_race': None,
            'gender_confidence': 0.0,
            'race_confidence': 0.0,
            'additional_context': {}
        }
        
        # Only search if we haven't exceeded API limit
        if self.api_calls_made >= 10000:
            logger.warning("Reached API call limit")
            self._save_to_cache(cache_key, enrichment)
            return enrichment
        
        try:
            # Search for guest information
            search_query = f'"{guest_name}" biography OR about OR professor OR author OR speaker'
            
            params = {
                'q': search_query,
                'api_key': self.serpapi_key,
                'num': 5  # Get top 5 results
            }
            
            response = requests.get('https://serpapi.com/search', params=params, timeout=30)
            self.api_calls_made += 1
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract relevant information from search results
                if 'organic_results' in data:
                    for result in data['organic_results'][:3]:
                        enrichment['search_results'].append({
                            'title': result.get('title'),
                            'snippet': result.get('snippet'),
                            'link': result.get('link')
                        })
                
                # Extract knowledge graph if available
                if 'knowledge_graph' in data:
                    kg = data['knowledge_graph']
                    enrichment['additional_context']['knowledge_graph'] = {
                        'title': kg.get('title'),
                        'type': kg.get('type'),
                        'description': kg.get('description')
                    }
                
                # Analyze results for demographic clues
                all_text = ' '.join([r.get('snippet', '') for r in enrichment['search_results']])
                
                # Gender indicators (simple heuristic)
                if any(pronoun in all_text.lower() for pronoun in [' she ', ' her ', ' hers ', ' herself ']):
                    enrichment['enhanced_gender'] = 'Female'
                    enrichment['gender_confidence'] = 0.8
                elif any(pronoun in all_text.lower() for pronoun in [' he ', ' his ', ' him ', ' himself ']):
                    enrichment['enhanced_gender'] = 'Male'
                    enrichment['gender_confidence'] = 0.8
                
                # Race/ethnicity indicators (very basic - would need more sophisticated analysis)
                race_indicators = {
                    'Black': ['african american', 'black author', 'black professor', 'african descent'],
                    'Hispanic/Latino': ['hispanic', 'latino', 'latina', 'latinx', 'mexican american', 'puerto rican']
                }
                
                for race, indicators in race_indicators.items():
                    if any(indicator in all_text.lower() for indicator in indicators):
                        enrichment['enhanced_race'] = race
                        enrichment['race_confidence'] = 0.7
                        break
                
        except Exception as e:
            logger.error(f"Error enriching {guest_name}: {e}")
        
        # Save to cache
        self._save_to_cache(cache_key, enrichment)
        
        return enrichment
    
    def get_stats(self) -> Dict:
        """Get enrichment statistics"""
        return {
            'api_calls_made': self.api_calls_made,
            'cache_hits': self.cache_hits,
            'cache_size': len(self.cache_index),
            'api_calls_remaining': max(0, 10000 - self.api_calls_made)
        }